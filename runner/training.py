"""Experiment-runner training stage: manifest's frozen QLoRA settings -> one
merged, standalone-loadable checkpoint directory per epoch.

Phase 1 wires only a fake, deterministic trainer here (see `runner.fakes`); a
real HF/PEFT/CUDA trainer is a later, separately qualified integration.

An OOM (or other resource-limit violation) is not treated as an anecdote to
discard: it is preserved as its own first-class failure-evidence run bundle.
The manifest authorizes exactly one automatic OOM fallback
(`single_oom_sequence_length_2048_to_1536_then_full_restart`); it is applied at
most once per run. If the OOM recurs afterward, the run stops and finalizes as
a failed bundle rather than retrying again or being silently discarded.

Every other fallback is rejected outright: `request_fallback` is the runner's
only path for applying one, and it consults `allowed_technical_fallbacks` in
`protocol/manifest.json`, never a hardcoded copy of the list.

Optional issue-19 recovery journals only completed epochs. A reused epoch must
match its full signature and a content-digested merged checkpoint in an
immutable finalized bundle; training never resumes model/optimizer state inside
an epoch.
"""
from __future__ import annotations
import dataclasses, hashlib, json, platform, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path

from protocol.validate_manifest import load as load_manifest, sha256 as manifest_sha256
from runner.bundle import CHECKSUM_FILE, write_bundle, finalize_bundle, verify_bundle
from runner.fakes import OutOfMemoryError
from runner.recovery import AttemptLedger, RecoveryWorkspace, StageSignature
from runner.core import (
    _adapter_command, _adapter_environment, _adapter_events, _adapter_metadata, _adapter_notes,
)

APPROVED_OOM_FALLBACK = "single_oom_sequence_length_2048_to_1536_then_full_restart"
OOM_FALLBACK_SEQUENCE_LENGTH = 1536


def request_fallback(name: str, manifest: dict) -> None:
    """Reject any fallback not explicitly named in the manifest's frozen list.

    Per `fallback_policy` ("no_quality_trigger;any_other_change_is_new_protocol_version
    _and_requires_new_baseline"), this is the runner's single choke point for applying a
    technical fallback -- it never trusts a hardcoded copy of the allowed set.
    """
    allowed = set(manifest["allowed_technical_fallbacks"])
    if name not in allowed:
        raise ValueError(f"fallback not authorized by protocol/manifest.json: {name!r}")


@dataclasses.dataclass(frozen=True)
class TrainingResult:
    run_id: str
    stage: str
    outcome: str  # "success" or "failed"
    bundle_dir: str
    checksums: dict
    metrics: dict


@dataclasses.dataclass(frozen=True)
class TrainingRecovery:
    """Durable state for completed-epoch training recovery (issue #19)."""

    workspace: RecoveryWorkspace
    stage_key: str
    ledger: AttemptLedger | None = None

    def attempt_ledger(self) -> AttemptLedger:
        return self.ledger or AttemptLedger(self.workspace.root / "attempts.jsonl")


def _isoformat(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _directory_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    directory = Path(directory)
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _checkpoint_input_digest(manifest: dict) -> str:
    return "sha256:" + hashlib.sha256(
        manifest["model"]["revision"].encode("utf-8")
    ).hexdigest()


def _epoch_stage_key(recovery: TrainingRecovery, sequence_length: int, epoch: int) -> str:
    return f"{recovery.stage_key}-seq{sequence_length}-epoch{epoch}"


def _training_signature(manifest, manifest_path, *, seed, epoch, sequence_length) -> StageSignature:
    return StageSignature.create(
        manifest_digest="sha256:" + manifest_sha256(manifest_path),
        protocol_version=manifest["protocol_version"],
        upstream_commit=manifest["upstream"]["commit"],
        upstream_tree=manifest["upstream"]["tree"],
        model_revision=manifest["model"]["revision"],
        seed=seed,
        stage="training",
        epoch=epoch,
        checkpoint_digest=_checkpoint_input_digest(manifest),
        effective_evaluation_config={
            "training": manifest["training"], "sequence_length": sequence_length,
        },
        expected_example_ids=[],
    )


def _checkpoint_reference(checkpoint: dict) -> str:
    return json.dumps(checkpoint, sort_keys=True, separators=(",", ":"))


def _recovered_checkpoint(workspace: RecoveryWorkspace, stage_key: str) -> dict:
    bundle_dir = workspace.completed_bundle(stage_key)
    verify_bundle(bundle_dir)
    try:
        checkpoint = json.loads(workspace.recovery_reference(stage_key) or "")
        checkpoint_dir = Path(checkpoint["merged_dir"])
        if not checkpoint_dir.is_dir() or _directory_digest(checkpoint_dir) != checkpoint["integrity"]:
            raise ValueError("merged checkpoint integrity mismatch")
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"cannot reuse completed training epoch '{stage_key}': {error}"
        ) from error
    checkpoint.pop("integrity")
    return checkpoint


def _read_checksums(bundle_dir: Path) -> dict:
    checksums = {}
    for line in (bundle_dir / CHECKSUM_FILE).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    checksums[CHECKSUM_FILE] = hashlib.sha256((bundle_dir / CHECKSUM_FILE).read_bytes()).hexdigest()
    return checksums


def _result_from_completed_bundle(workspace: RecoveryWorkspace, stage_key: str) -> TrainingResult:
    bundle_dir = workspace.completed_bundle(stage_key)
    verify_bundle(bundle_dir)
    metrics = json.loads((bundle_dir / "metrics.json").read_text(encoding="utf-8"))
    run_id = json.loads((bundle_dir / "manifest.yaml").read_text(encoding="utf-8"))["run_id"]
    return TrainingResult(
        run_id=run_id, stage="training", outcome=metrics["outcome"],
        bundle_dir=str(bundle_dir), checksums=_read_checksums(bundle_dir), metrics=metrics,
    )


def run_training(manifest_path, *, trainer, telemetry, storage, seed: int,
                  run_id: str | None = None, clock=time.time,
                  recovery: TrainingRecovery | None = None) -> TrainingResult:
    manifest = load_manifest(manifest_path)
    training_cfg = manifest["training"]
    if seed not in training_cfg["seeds"]:
        raise ValueError(f"seed {seed} is not one of the manifest's frozen seeds {training_cfg['seeds']}")

    epochs = training_cfg["optimizer"]["epochs"]
    sequence_length = training_cfg["data"]["max_sequence_length"]
    checkpoints: dict[str, dict] = {}
    fallback_applied = False
    outcome = "success"
    failure_reason = None
    interruption_error = None
    epoch_signatures: dict[int, tuple[str, StageSignature]] = {}

    # A completed whole run can be returned directly. Its checkpoints stay in
    # that immutable finalized bundle; a rerun never copies or rewrites them.
    if recovery is not None:
        completed = []
        for candidate_epoch in range(1, epochs + 1):
            state_key = _epoch_stage_key(recovery, sequence_length, candidate_epoch)
            signature = _training_signature(
                manifest, manifest_path, seed=seed, epoch=candidate_epoch,
                sequence_length=sequence_length,
            )
            if not recovery.workspace.has_state(state_key):
                break
            inspection = recovery.workspace.inspect_stage(state_key, signature)
            if inspection.status == "incompatible":
                raise ValueError(
                    f"cannot resume training '{recovery.stage_key}': recovery state "
                    f"is incompatible on field '{inspection.differing_field}'"
                )
            if inspection.status != "completed":
                break
            completed.append(_recovered_checkpoint(recovery.workspace, state_key))
        if len(completed) == epochs:
            return _result_from_completed_bundle(recovery.workspace, _epoch_stage_key(recovery, sequence_length, epochs))

    run_id = run_id or f"training-seed{seed}-{int(clock())}"
    bundle_dir = storage.new_run_dir(run_id)
    started_ts = clock()
    attempt_id = f"{recovery.stage_key}:{uuid.uuid4().hex}" if recovery is not None else None
    telemetry.start()

    log_lines = [f"start training run {run_id} protocol_version={manifest['protocol_version']} seed={seed}"]

    epoch = 1
    while epoch <= epochs:
        state_key = None
        signature = None
        if recovery is not None:
            state_key = _epoch_stage_key(recovery, sequence_length, epoch)
            signature = _training_signature(
                manifest, manifest_path, seed=seed, epoch=epoch,
                sequence_length=sequence_length,
            )
            epoch_signatures[epoch] = (state_key, signature)
            if recovery.workspace.has_state(state_key):
                inspection = recovery.workspace.inspect_stage(state_key, signature)
                if inspection.status == "incompatible":
                    raise ValueError(
                        f"cannot resume training '{recovery.stage_key}': recovery state "
                        f"is incompatible on field '{inspection.differing_field}'"
                    )
                if inspection.status == "completed":
                    checkpoints[f"epoch-{epoch}"] = _recovered_checkpoint(recovery.workspace, state_key)
                    log_lines.append(f"epoch {epoch}: reused validated merged checkpoint")
                    epoch += 1
                    continue
            recovery.workspace.write_state(state_key, signature, status="running")
        try:
            fingerprint = trainer.train_epoch(seed=seed, epoch=epoch, sequence_length=sequence_length, config=training_cfg)
        except OutOfMemoryError as exc:
            log_lines.append(f"epoch {epoch}: OOM at sequence_length={sequence_length}: {exc}")
            if fallback_applied:
                outcome = "failed"
                failure_reason = (
                    f"OOM recurred at epoch {epoch} after the single approved fallback "
                    f"({APPROVED_OOM_FALLBACK}) was already applied; preserved as failure "
                    "evidence, not retried again"
                )
                log_lines.append(failure_reason)
                break
            request_fallback(APPROVED_OOM_FALLBACK, manifest)
            fallback_applied = True
            sequence_length = OOM_FALLBACK_SEQUENCE_LENGTH
            log_lines.append(f"applying approved fallback {APPROVED_OOM_FALLBACK}: sequence_length -> {sequence_length}; full restart")
            checkpoints = {}
            epoch_signatures = {}
            epoch = 1
            continue
        except BaseException as exc:
            outcome = "interrupted"
            failure_reason = f"epoch {epoch} interrupted: {exc}"
            log_lines.append(failure_reason)
            interruption_error = exc
            break
        checkpoint_dir = bundle_dir / "checkpoints" / f"epoch-{epoch}"
        try:
            trainer.merge_checkpoint(fingerprint, checkpoint_dir)
        except BaseException as exc:
            outcome = "interrupted"
            failure_reason = f"epoch {epoch} interrupted during checkpoint merge: {exc}"
            log_lines.append(failure_reason)
            interruption_error = exc
            break
        checkpoints[f"epoch-{epoch}"] = {
            "fingerprint": fingerprint,
            "sequence_length": sequence_length,
            "merged_dir": str(checkpoint_dir),
        }
        log_lines.append(f"epoch {epoch}: checkpoint merged fingerprint={fingerprint}")
        epoch += 1

    telemetry_rows = telemetry.stop()
    log_lines.extend(_adapter_events(trainer, telemetry))
    log_lines.append(f"finished training run {run_id} outcome={outcome}")

    command = f"{sys.executable} -m runner.training --manifest {manifest_path} --run-id {run_id} --seed {seed}"
    metrics = {
        "stage": "training",
        "seed": seed,
        "outcome": outcome,
        "fallback_applied": fallback_applied,
        "checkpoints": checkpoints,
    }
    if failure_reason:
        metrics["failure_reason"] = failure_reason

    notes = "# Training run notes\n\nFake adapters only: no GPU, no model weights.\n"
    if outcome != "success":
        notes += f"\n**Preserved failure evidence:** {failure_reason}\n"

    manifest_record = {
            "run_id": run_id, "stage": "training", "outcome": outcome,
            "protocol_version": manifest["protocol_version"],
            "upstream_commit": manifest["upstream"]["commit"],
            "model_revision": manifest["model"]["revision"],
            "seed": seed,
    }
    manifest_record.update(_adapter_metadata(trainer, telemetry))
    contents = {
        "manifest.yaml": json.dumps(manifest_record, indent=2, sort_keys=True),
        "command.sh": (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"{_adapter_command(command, trainer, telemetry)}\n"
        ),
        "config.yaml": json.dumps(training_cfg, indent=2, sort_keys=True),
        "environment.txt": _adapter_environment(telemetry, "\n".join([
            f"python={platform.python_version()}",
            f"platform={platform.platform()}",
            "gpu=none (fake adapters; no real GPU or model weights used)",
        ]) + "\n", trainer),
        "metrics.json": json.dumps(metrics, indent=2, sort_keys=True),
        "execution.log": "\n".join(log_lines) + "\n",
        "gpu.csv": "t,vram_mb,util_pct\n" + "\n".join(
            f"{row['t']},{row['vram_mb']},{row['util_pct']}" for row in telemetry_rows
        ) + "\n",
        "notes.md": _adapter_notes("training", notes, trainer, telemetry),
    }
    write_bundle(bundle_dir, contents)
    checksums = finalize_bundle(bundle_dir)
    verify_bundle(bundle_dir)
    if recovery is not None:
        for epoch_key, checkpoint in checkpoints.items():
            completed_epoch = int(epoch_key.removeprefix("epoch-"))
            state_key, signature = epoch_signatures[completed_epoch]
            recovery.workspace.write_state(
                state_key, signature, status="completed",
                recovery_reference=_checkpoint_reference({
                    **checkpoint, "integrity": _directory_digest(Path(checkpoint["merged_dir"])),
                }),
                completed_bundle=str(bundle_dir),
            )
        if outcome != "success" and epoch <= epochs and epoch in epoch_signatures:
            state_key, signature = epoch_signatures[epoch]
            recovery.workspace.write_state(
                state_key, signature, status="interrupted",
                recovery_reference="restart-epoch",
            )
        ledger_signature = epoch_signatures.get(epoch, epoch_signatures[max(epoch_signatures)])[1]
        recovery.attempt_ledger().append(
            attempt_id, ledger_signature,
            status="completed" if outcome == "success" else outcome,
            started_at=_isoformat(started_ts), ended_at=_isoformat(clock()),
            wall_seconds=clock() - started_ts, gpu_hours=None,
            state_reference=f"{recovery.stage_key}.json",
        )
    if interruption_error is not None:
        raise interruption_error
    return TrainingResult(run_id=run_id, stage="training", outcome=outcome, bundle_dir=str(bundle_dir), checksums=checksums, metrics=metrics)
