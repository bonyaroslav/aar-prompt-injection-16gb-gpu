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
"""
from __future__ import annotations
import dataclasses, json, platform, sys, time
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner.bundle import write_bundle, finalize_bundle
from runner.fakes import OutOfMemoryError

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


def run_training(manifest_path, *, trainer, telemetry, storage, seed: int,
                  run_id: str | None = None, clock=time.time) -> TrainingResult:
    manifest = load_manifest(manifest_path)
    training_cfg = manifest["training"]
    if seed not in training_cfg["seeds"]:
        raise ValueError(f"seed {seed} is not one of the manifest's frozen seeds {training_cfg['seeds']}")

    run_id = run_id or f"training-seed{seed}-{int(clock())}"
    bundle_dir = storage.new_run_dir(run_id)
    telemetry.start()

    log_lines = [f"start training run {run_id} protocol_version={manifest['protocol_version']} seed={seed}"]
    epochs = training_cfg["optimizer"]["epochs"]
    sequence_length = training_cfg["data"]["max_sequence_length"]
    checkpoints: dict[str, dict] = {}
    fallback_applied = False
    outcome = "success"
    failure_reason = None

    epoch = 1
    while epoch <= epochs:
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
            epoch = 1
            continue
        checkpoint_dir = bundle_dir / "checkpoints" / f"epoch-{epoch}"
        trainer.merge_checkpoint(fingerprint, checkpoint_dir)
        checkpoints[f"epoch-{epoch}"] = {
            "fingerprint": fingerprint,
            "sequence_length": sequence_length,
            "merged_dir": str(checkpoint_dir),
        }
        log_lines.append(f"epoch {epoch}: checkpoint merged fingerprint={fingerprint}")
        epoch += 1

    telemetry_rows = telemetry.stop()
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
    if outcome == "failed":
        notes += f"\n**Preserved failure evidence:** {failure_reason}\n"

    contents = {
        "manifest.yaml": json.dumps({
            "run_id": run_id, "stage": "training", "outcome": outcome,
            "protocol_version": manifest["protocol_version"],
            "upstream_commit": manifest["upstream"]["commit"],
            "model_revision": manifest["model"]["revision"],
            "seed": seed,
        }, indent=2, sort_keys=True),
        "command.sh": f"#!/usr/bin/env bash\nset -euo pipefail\n{command}\n",
        "config.yaml": json.dumps(training_cfg, indent=2, sort_keys=True),
        "environment.txt": "\n".join([
            f"python={platform.python_version()}",
            f"platform={platform.platform()}",
            "gpu=none (fake adapters; no real GPU or model weights used)",
        ]) + "\n",
        "metrics.json": json.dumps(metrics, indent=2, sort_keys=True),
        "execution.log": "\n".join(log_lines) + "\n",
        "gpu.csv": "t,vram_mb,util_pct\n" + "\n".join(
            f"{row['t']},{row['vram_mb']},{row['util_pct']}" for row in telemetry_rows
        ) + "\n",
        "notes.md": notes,
    }
    write_bundle(bundle_dir, contents)
    checksums = finalize_bundle(bundle_dir)
    return TrainingResult(run_id=run_id, stage="training", outcome=outcome, bundle_dir=str(bundle_dir), checksums=checksums, metrics=metrics)
