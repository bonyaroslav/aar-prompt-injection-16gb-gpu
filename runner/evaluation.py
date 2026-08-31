"""Trained-checkpoint evaluation stage: the same manifest-frozen evaluation
protocol as the baseline stage (`runner.core.run_baseline`), with the model
backend swapped for a merged training checkpoint.

Held-out InjecAgent is out of scope here. Per `protocol/manifest.json`'s
`selection.held_out_unavailable_until: selection_record_finalized` and
`protocol/heldout_sealing.md`, checkpoint selection is decided on visible safety
and capability alone; the sealed held-out reveal for the selected checkpoint is
a separate, later stage (see the ready-for-agent issue queue).

Public seam: `run_trained_evaluation`. Its `config.yaml` is built from
`runner.core.effective_eval_config`, the same helper the baseline stage uses --
the only difference between the two stages' effective config is the `checkpoint`
field, which is exactly what a contract test in `tests/test_evaluation.py`
asserts.

Optional resumability (issue #18): pass an `EvaluationRecovery` and an interrupted
evaluation restarts from validated completed work -- already-scored examples are
reused, never regenerated or rescored -- and produces exactly the same final
metrics and finalized checksummed bundle topology as an uninterrupted run. Per
the #16 decision the *finalized* bundle is still promoted only at the whole-
evaluation boundary, after all six benchmarks complete and `verify_bundle`
passes; the per-example journal (`runner.recovery.RecoveryWorkspace`) is a
resume optimisation, not a new protocol recovery guarantee.
"""
from __future__ import annotations
import dataclasses, hashlib, json, platform, sys, time, uuid
from datetime import datetime, timezone

from protocol.validate_manifest import load as load_manifest, sha256 as manifest_sha256
from runner.bundle import CHECKSUM_FILE, finalize_bundle, verify_bundle, write_bundle
from runner.core import (
    VISIBLE_SAFETY_BENCHMARKS, CAPABILITY_BENCHMARKS, resolve_sample_count, effective_eval_config,
    _adapter_command, _adapter_environment, _adapter_events, _adapter_metadata, _adapter_notes,
)
from runner.recovery import AttemptLedger, RecoveryWorkspace, StageSignature

# The manifest groups every scored benchmark under exactly one of these keys.
_BENCHMARK_GROUPS = (
    ("visible_safety", VISIBLE_SAFETY_BENCHMARKS),
    ("capability", CAPABILITY_BENCHMARKS),
)


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    run_id: str
    stage: str
    bundle_dir: str
    checksums: dict
    metrics: dict


@dataclasses.dataclass(frozen=True)
class EvaluationRecovery:
    """Where a resumable trained evaluation keeps its recovery state.

    `stage_key` is stable across restarts (it names the resumable evaluation --
    e.g. ``eval-seed17-epoch1``); the run bundle's `run_id` is per-attempt and
    independent. `ledger` defaults to an `attempts.jsonl` in the workspace root.
    """
    workspace: RecoveryWorkspace
    stage_key: str
    ledger: AttemptLedger | None = None

    def attempt_ledger(self) -> AttemptLedger:
        return self.ledger or AttemptLedger(self.workspace.root / "attempts.jsonl")


def _isoformat(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _load_all_items(manifest: dict, dataset) -> dict[str, list[dict]]:
    eval_cfg = manifest["evaluation"]
    items_by_benchmark: dict[str, list[dict]] = {}
    for group_key, names in _BENCHMARK_GROUPS:
        for name in names:
            count = resolve_sample_count(eval_cfg[group_key][name]["sample_ids"])
            items_by_benchmark[name] = list(dataset.load_items(name, count))
    return items_by_benchmark


def _ordered_example_ids(items_by_benchmark: dict[str, list[dict]]) -> list[str]:
    ordered = []
    for _, names in _BENCHMARK_GROUPS:
        for name in names:
            ordered.extend(f"{name}:{item['id']}" for item in items_by_benchmark[name])
    return ordered


def _evaluation_signature(manifest, manifest_path, *, seed, epoch, checkpoint_fingerprint,
                           ordered_example_ids) -> StageSignature:
    return StageSignature.create(
        manifest_digest="sha256:" + manifest_sha256(manifest_path),
        protocol_version=manifest["protocol_version"],
        upstream_commit=manifest["upstream"]["commit"],
        upstream_tree=manifest["upstream"]["tree"],
        model_revision=manifest["model"]["revision"],
        seed=seed,
        stage="trained_evaluation",
        epoch=epoch,
        checkpoint_digest=checkpoint_fingerprint,
        effective_evaluation_config=effective_eval_config(manifest, checkpoint_fingerprint),
        expected_example_ids=ordered_example_ids,
    )


def _score_benchmark(name, cfg, items, *, model, scorer, already, on_scored) -> tuple[dict, int]:
    """Score `items` in their fixed dataset order. An example already present in
    `already` (a prior attempt's journalled outcome) is reused verbatim -- never
    regenerated or rescored -- so the resumed aggregation walks the values in the
    identical order an uninterrupted run would.
    """
    item_scores = {}
    for item in items:
        key = (name, item["id"])
        if key in already:
            item_scores[item["id"]] = already[key]
            continue
        output = model.generate(name, item, cfg)
        outcome = scorer.score(name, item, output, cfg)
        item_scores[item["id"]] = outcome
        on_scored(name, item["id"], outcome)
    values = [entry["score"] for entry in item_scores.values()]
    aggregate = {"metric": cfg["metric"], "value": (sum(values) / len(values)) if values else None}
    return {"items": item_scores, "aggregate": aggregate}, len(items)


def _read_checksums(bundle_dir) -> dict:
    checksums = {}
    for line in (bundle_dir / CHECKSUM_FILE).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    checksums[CHECKSUM_FILE] = hashlib.sha256((bundle_dir / CHECKSUM_FILE).read_bytes()).hexdigest()
    return checksums


def _result_from_completed_bundle(workspace: RecoveryWorkspace, stage_key: str) -> EvaluationResult:
    bundle_dir = workspace.completed_bundle(stage_key)
    verify_bundle(bundle_dir)
    metrics = json.loads((bundle_dir / "metrics.json").read_text(encoding="utf-8"))
    run_id = json.loads((bundle_dir / "manifest.yaml").read_text(encoding="utf-8"))["run_id"]
    return EvaluationResult(
        run_id=run_id, stage="trained_evaluation", bundle_dir=str(bundle_dir),
        checksums=_read_checksums(bundle_dir), metrics=metrics,
    )


def run_trained_evaluation(manifest_path, *, model, dataset, scorer, telemetry, storage,
                            seed: int, epoch: int, checkpoint: dict,
                            run_id: str | None = None, clock=time.time,
                            recovery: EvaluationRecovery | None = None) -> EvaluationResult:
    """Evaluate one merged training checkpoint on the three visible safety
    benchmarks and the three capability gates. `checkpoint` is the per-epoch
    entry produced by `runner.training.run_training`'s `metrics["checkpoints"]`
    (must contain `fingerprint` and `merged_dir`).

    When `recovery` is supplied the evaluation is resumable: it validates the
    stage signature (seed, checkpoint identity, benchmark set, effective config,
    expected IDs, protocol/upstream provenance, model revision) against any
    durable state before skipping work, reuses journalled example outcomes, and
    records every attempt in the attempt ledger.
    """
    manifest = load_manifest(manifest_path)
    checkpoint_fingerprint = checkpoint["fingerprint"]

    items_by_benchmark = _load_all_items(manifest, dataset)
    ordered_example_ids = _ordered_example_ids(items_by_benchmark)

    already: dict = {}
    signature = None
    if recovery is not None:
        signature = _evaluation_signature(
            manifest, manifest_path, seed=seed, epoch=epoch,
            checkpoint_fingerprint=checkpoint_fingerprint,
            ordered_example_ids=ordered_example_ids,
        )
        if recovery.workspace.has_state(recovery.stage_key):
            inspection = recovery.workspace.inspect_stage(recovery.stage_key, signature)
            if inspection.status == "incompatible":
                raise ValueError(
                    f"cannot resume trained evaluation '{recovery.stage_key}': recovery "
                    f"state is incompatible on field '{inspection.differing_field}'"
                )
            if inspection.status == "completed":
                return _result_from_completed_bundle(recovery.workspace, recovery.stage_key)
            if inspection.status in ("running", "recoverable", "interrupted"):
                already = recovery.workspace.completed_progress(recovery.stage_key)
            else:
                raise ValueError(
                    f"cannot resume trained evaluation '{recovery.stage_key}': recovery "
                    f"state is unavailable after hard loss ({inspection.action})"
                )

    run_id = run_id or f"eval-seed{seed}-epoch{epoch}-{int(clock())}"
    bundle_dir = storage.new_run_dir(run_id)

    attempt_id = None
    started_ts = clock()
    if recovery is not None:
        recovery.workspace.write_state(recovery.stage_key, signature, status="running")
        attempt_id = f"{recovery.stage_key}:{uuid.uuid4().hex}"

    telemetry.start()
    log_lines = [
        f"start trained-checkpoint evaluation run {run_id} protocol_version={manifest['protocol_version']} "
        f"seed={seed} epoch={epoch} checkpoint={checkpoint_fingerprint}"
    ]
    if recovery is not None and already:
        log_lines.append(f"resuming: {len(already)} example(s) reused from prior attempt(s)")

    def on_scored(benchmark, example_id, outcome):
        if recovery is not None:
            recovery.workspace.record_progress(
                recovery.stage_key, benchmark=benchmark, example_id=example_id, outcome=outcome
            )

    eval_cfg = manifest["evaluation"]
    metrics = {
        "stage": "trained_evaluation", "seed": seed, "epoch": epoch,
        "checkpoint": checkpoint_fingerprint, "benchmarks": {},
    }
    try:
        for group_key, names in _BENCHMARK_GROUPS:
            for name in names:
                result, n = _score_benchmark(
                    name, eval_cfg[group_key][name], items_by_benchmark[name],
                    model=model, scorer=scorer, already=already, on_scored=on_scored,
                )
                metrics["benchmarks"][name] = result
                log_lines.append(f"scored {name}: n={n}")
    except BaseException:
        if recovery is not None:
            try:
                recovery.workspace.write_state(
                    recovery.stage_key, signature, status="interrupted",
                    recovery_reference="progress-journal",
                )
                recovery.attempt_ledger().append(
                    attempt_id, signature, status="interrupted",
                    started_at=_isoformat(started_ts), ended_at=_isoformat(clock()),
                    wall_seconds=clock() - started_ts, gpu_hours=None,
                    state_reference=f"{recovery.stage_key}.json",
                )
            except Exception:
                pass
        raise

    telemetry_rows = telemetry.stop()
    # `dataset` is deliberately excluded from every adapter-notification helper below:
    # `RealDatasetAdapter.manifest_metadata`/`environment_lines` exist only to surface the
    # InjecAgent source commit for held-out-touching stages, and evaluating that property
    # would read `heldout_dir` -- exactly what this stage must never do (see module docstring).
    log_lines.extend(_adapter_events(model, scorer, telemetry))
    log_lines.append(f"finished trained-checkpoint evaluation run {run_id}")

    command = (
        f"{sys.executable} -m runner.evaluation --manifest {manifest_path} --run-id {run_id} "
        f"--seed {seed} --epoch {epoch} --checkpoint-dir {checkpoint['merged_dir']}"
    )
    manifest_record = {
        "run_id": run_id, "stage": "trained_evaluation",
        "protocol_version": manifest["protocol_version"],
        "upstream_commit": manifest["upstream"]["commit"],
        "model_revision": manifest["model"]["revision"],
        "seed": seed, "epoch": epoch, "checkpoint": checkpoint_fingerprint,
    }
    manifest_record.update(_adapter_metadata(model, scorer, telemetry))
    contents = {
        "manifest.yaml": json.dumps(manifest_record, indent=2, sort_keys=True),
        "command.sh": (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"{_adapter_command(command, model, scorer, telemetry)}\n"
        ),
        "config.yaml": json.dumps(effective_eval_config(manifest, checkpoint_fingerprint), indent=2, sort_keys=True),
        "environment.txt": _adapter_environment(telemetry, "\n".join([
            f"python={platform.python_version()}",
            f"platform={platform.platform()}",
            "gpu=none (fake adapters; no real GPU or model weights used)",
        ]) + "\n", model, scorer),
        "metrics.json": json.dumps(metrics, indent=2, sort_keys=True),
        "execution.log": "\n".join(log_lines) + "\n",
        "gpu.csv": "t,vram_mb,util_pct\n" + "\n".join(
            f"{row['t']},{row['vram_mb']},{row['util_pct']}" for row in telemetry_rows
        ) + "\n",
        "notes.md": _adapter_notes(
            "trained_evaluation",
            "# Trained-checkpoint evaluation run notes\n\nFake adapters only: no GPU, no model weights. "
            "Held-out InjecAgent is not evaluated in this stage.\n",
            model, scorer, telemetry,
        ),
    }
    write_bundle(bundle_dir, contents)
    checksums = finalize_bundle(bundle_dir)
    # #18: promote the finalized bundle only after every benchmark completes and
    # its checksums validate clean.
    verify_bundle(bundle_dir)

    if recovery is not None:
        recovery.workspace.write_state(
            recovery.stage_key, signature, status="completed", completed_bundle=str(bundle_dir)
        )
        recovery.attempt_ledger().append(
            attempt_id, signature, status="completed",
            started_at=_isoformat(started_ts), ended_at=_isoformat(clock()),
            wall_seconds=clock() - started_ts, gpu_hours=None,
            state_reference=f"{recovery.stage_key}.json",
        )

    return EvaluationResult(run_id=run_id, stage="trained_evaluation", bundle_dir=str(bundle_dir), checksums=checksums, metrics=metrics)
