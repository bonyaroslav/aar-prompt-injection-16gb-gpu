"""Real seed training + evaluation + checkpoint selection for issue #12
(RESEARCH_PLAN.md Phase 5+6, run for real rather than against fake adapters).

Trains one seed's QLoRA adapter for all three frozen epochs against the
frozen training dataset (issue #11's `training_data` builder output),
evaluates every epoch checkpoint with the identical visible-safety +
capability protocol the frozen baseline used (`runner.evaluation
.run_trained_evaluation`, which never touches held-out InjecAgent), and
finalizes a checksummed checkpoint-selection record via the existing
`runner.selection` contract. Held-out InjecAgent is never read or revealed
by this module: no `HeldOutSealer` is constructed here, and the dataset
adapter's `heldout_dir` is never dereferenced (it exists only because
`RealDatasetAdapter` takes one positionally).

Like `runner.real_baseline`, the orchestration entrypoint (`run_real_seed`)
is exercised only by the real hardware run; the pure/testable pieces --
loading the frozen training examples and baseline benchmarks, shaping
selection candidates, and comparing measured resource use against the
manifest's per-seed and cumulative limits -- are unit-tested offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import time
import uuid
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner.bundle import verify_bundle
from runner.evaluation import EvaluationRecovery, run_trained_evaluation
from runner.gpu_smoke import _verify_upstream
from runner.real_adapters import RealDatasetAdapter, RealModelAdapter, RealScorerAdapter, RealTelemetryAdapter
from runner.real_baseline import _directory_bytes
from runner.real_training import RealQLoRATrainerAdapter
from runner.recovery import AttemptLedger, RecoveryWorkspace, finalized_inputs_only
from runner.reveal import HeldOutRevealRecovery, run_selection_and_reveal
from runner.selection import finalize_selection_record, select_checkpoint, verify_selection_record
from runner.storage import LocalStorageAdapter
from runner.training import TrainingRecovery, run_training


def load_training_examples(dataset_path: str | Path) -> list[dict]:
    """Read the issue #11 training-data builder's JSONL output. Every record
    already carries a `messages` field ending in an assistant turn -- exactly
    what `runner.real_training.encode_response_only` needs; the extra
    provenance fields (source/category/generation_rule/content_hash) simply
    ride along unused.
    """
    examples = []
    with Path(dataset_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "messages" not in record:
                raise ValueError(f"training example missing 'messages': {record!r}")
            examples.append(record)
    if not examples:
        raise ValueError(f"no training examples found in {dataset_path}")
    return examples


def load_baseline_benchmarks(baseline_metrics_path: str | Path) -> dict:
    """The frozen baseline's per-benchmark aggregates, in the shape
    `runner.selection.select_checkpoint` expects as `baseline_benchmarks`.
    """
    metrics = json.loads(Path(baseline_metrics_path).read_text(encoding="utf-8"))
    if metrics.get("stage") != "baseline":
        raise ValueError(f"not a baseline metrics.json (stage={metrics.get('stage')!r}): {baseline_metrics_path}")
    return metrics["benchmarks"]


def build_candidates(epoch_metrics: dict[int, dict]) -> list[dict]:
    """Shape each epoch's `run_trained_evaluation` metrics into the
    `{"epoch", "checkpoint_digest", "benchmarks"}` candidates
    `runner.selection.select_checkpoint` needs.
    """
    return [
        {"epoch": epoch, "checkpoint_digest": metrics["checkpoint"], "benchmarks": metrics["benchmarks"]}
        for epoch, metrics in sorted(epoch_metrics.items())
    ]


def compare_seed_resource_use(manifest: dict, *, wall_seconds: float, peak_vram_mb: float,
                              bundle_bytes: int, prior_cumulative_gpu_hours: float) -> dict:
    """Compare this seed's measured training+evaluation resource use against the
    manifest's per-seed wall-hour cap and the *cumulative* GPU-hour cap across
    every run so far. Unlike `runner.real_baseline.compare_against_projection`
    (the first run, with nothing to accumulate), a seed run always has a
    predecessor -- at minimum the baseline -- so `prior_cumulative_gpu_hours`
    must be carried forward explicitly rather than assumed zero.
    """
    limits = manifest["resources"]
    peak_vram_gb = peak_vram_mb / 1024.0
    wall_hours = wall_seconds / 3600.0
    gpu_hours = wall_hours  # single-GPU run: gpu-hours == wall-hours for this stage
    cumulative_gpu_hours = prior_cumulative_gpu_hours + gpu_hours
    bundle_gb = bundle_bytes / (1024.0 ** 3)
    findings = []
    if peak_vram_gb > limits["vram_allocated_gb_max"]:
        findings.append(
            f"measured peak_vram_gb {peak_vram_gb:.3f} exceeds vram_allocated_gb_max {limits['vram_allocated_gb_max']}"
        )
    if wall_hours > limits["wall_hours_per_seed_max"]:
        findings.append(
            f"measured wall_hours {wall_hours:.3f} exceeds wall_hours_per_seed_max {limits['wall_hours_per_seed_max']}"
        )
    if cumulative_gpu_hours > limits["gpu_hours_total_max"]:
        findings.append(
            f"cumulative gpu_hours {cumulative_gpu_hours:.3f} exceeds gpu_hours_total_max {limits['gpu_hours_total_max']}"
        )
    if bundle_gb > limits["storage_gb_max"]:
        findings.append(f"measured bundle_disk_gb {bundle_gb:.3f} exceeds storage_gb_max {limits['storage_gb_max']}")
    return {
        "measured": {
            "wall_seconds": wall_seconds, "peak_vram_gb": peak_vram_gb,
            "gpu_hours": gpu_hours, "bundle_disk_gb": bundle_gb,
        },
        "prior_cumulative_gpu_hours": prior_cumulative_gpu_hours,
        "cumulative_gpu_hours": cumulative_gpu_hours,
        "limits": dict(limits),
        "feasibility_findings": findings,
    }


def write_seed_comparison_artifact(root: Path, artifact_id: str, comparison: dict) -> Path:
    artifact = Path(root) / artifact_id
    if artifact.exists():
        raise FileExistsError(f"seed resource comparison artifact already exists: {artifact}")
    artifact.mkdir(parents=True)
    payload = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    target = artifact / "seed_resource_comparison.json"
    target.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (artifact / "checksums.sha256").write_text(f"{digest}  seed_resource_comparison.json\n", encoding="utf-8")
    return artifact


def write_or_verify_seed_comparison_artifact(root: Path, artifact_id: str, comparison: dict) -> Path:
    """Recovery-aware `write_seed_comparison_artifact`: a re-run of a completed
    seed recomputes byte-identical resource accounting (the attempt ledger and
    finalized bundles it reads are immutable), so re-finalizing to the same stable
    id is a checksum-verified no-op rather than a `FileExistsError`. Different
    content at the same id is still rejected -- the artifact stays immutable.
    """
    artifact = Path(root) / artifact_id
    payload = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    target = artifact / "seed_resource_comparison.json"
    if artifact.exists():
        existing = target.read_text(encoding="utf-8")
        if existing != payload:
            raise RuntimeError(f"seed resource comparison already finalized with different content: {artifact}")
        verify_seed_comparison_artifact(artifact)
        return artifact
    return write_seed_comparison_artifact(root, artifact_id, comparison)


def verify_seed_comparison_artifact(artifact: Path) -> None:
    artifact = Path(artifact)
    target = artifact / "seed_resource_comparison.json"
    recorded = (artifact / "checksums.sha256").read_text(encoding="utf-8").split("  ", 1)[0]
    if hashlib.sha256(target.read_bytes()).hexdigest() != recorded:
        raise ValueError(f"seed resource comparison checksum mismatch: {artifact}")


# --- Recovery-aware split-run seam (issue #22) ------------------------------
#
# `runner.training` / `runner.evaluation` / `runner.reveal` each already resume
# at their own safe boundary (issues #19 / #18 / #20). This seam is the stable
# per-seed identity above them: it routes each stage through its recovery
# contract, keeps operational state in a workspace outside the evidence root,
# aggregates the real (completed + interrupted + explicitly unavailable) resource
# cost, and promotes the finalized discovery topology only once every stage is
# done. A resumed seed and an uninterrupted seed expose the same final artifacts.

FALLBACK_SEQUENCE_LENGTH = 1536


def _training_epoch_key(seed: int, sequence_length: int, epoch: int) -> str:
    return f"training-seed{seed}-seq{sequence_length}-epoch{epoch}"


def _eval_stage_key(seed: int, epoch: int) -> str:
    return f"eval-seed{seed}-epoch{epoch}"


def _reveal_stage_key(seed: int) -> str:
    return f"reveal-seed{seed}"


def _seed_ledger_prefixes(seed: int, epochs: int) -> tuple[str, ...]:
    return (
        f"training-seed{seed}:",
        f"{_reveal_stage_key(seed)}:",
        *(f"{_eval_stage_key(seed, epoch)}:" for epoch in range(1, epochs + 1)),
    )


def _read_stage_state(recovery_root: Path, state_key: str) -> dict | None:
    path = Path(recovery_root) / f"{state_key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "unreadable"}


def _stage_status(recovery_root: Path, state_key: str) -> dict:
    record = _read_stage_state(recovery_root, state_key)
    if record is None:
        return {"state_key": state_key, "status": "absent"}
    status = record.get("status")
    out = {"state_key": state_key, "status": status}
    if record.get("recovery_reference"):
        out["recovery_reference"] = record["recovery_reference"]
    if status == "completed":
        bundle = record.get("completed_bundle")
        try:
            verify_bundle(Path(bundle))
            out["completed_bundle"] = bundle
        except (OSError, ValueError, TypeError):
            out["status"] = "completed-bundle-unverifiable"
    return out


def _completed_training_boundary(recovery_root: Path, seed: int, epochs: int,
                                  base_sequence_length: int = 2048) -> tuple[int, str] | None:
    """Return `(sequence_length, last-epoch state_key)` for a fully completed
    training run, honoring the one authorized 2048->1536 OOM restart namespace.
    """
    for sequence_length in (base_sequence_length, FALLBACK_SEQUENCE_LENGTH):
        keys = [_training_epoch_key(seed, sequence_length, e) for e in range(1, epochs + 1)]
        if all((_read_stage_state(recovery_root, k) or {}).get("status") == "completed" for k in keys):
            return sequence_length, keys[-1]
    return None


def seed_run_status(manifest_path, *, recovery_root: Path, output_root: Path, seed: int) -> dict:
    """Restart status: completed / interrupted / recoverable stages and the exact
    next continuation action for one seed. Reads durable state only; runs nothing.
    """
    frozen = load_manifest(manifest_path)
    epochs = frozen["training"]["optimizer"]["epochs"]
    base_seq = frozen["training"]["data"]["max_sequence_length"]

    stages: list[dict] = []
    training_seq = base_seq
    fallback = _completed_training_boundary(recovery_root, seed, epochs)
    if any(
        (_read_stage_state(recovery_root, _training_epoch_key(seed, FALLBACK_SEQUENCE_LENGTH, e)) or {})
        for e in range(1, epochs + 1)
    ):
        training_seq = FALLBACK_SEQUENCE_LENGTH
    for epoch in range(1, epochs + 1):
        stages.append({"stage": f"training-epoch{epoch}",
                       **_stage_status(recovery_root, _training_epoch_key(seed, training_seq, epoch))})
    for epoch in range(1, epochs + 1):
        stages.append({"stage": f"eval-epoch{epoch}",
                       **_stage_status(recovery_root, _eval_stage_key(seed, epoch))})

    selection_path = Path(output_root) / f"selection-seed{seed}" / "selection_record.json"
    selection: dict = {"finalized": selection_path.exists()}
    if selection_path.exists():
        record = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["selected_checkpoint_digest"] = record.get("selected_checkpoint_digest")
        selection["selected_epoch"] = record.get("selected_epoch")

    reveal = _stage_status(recovery_root, _reveal_stage_key(seed))
    comparison_finalized = (Path(output_root) / f"seed{seed}-resource-comparison").exists()

    training_done = fallback is not None or all(
        s["status"] == "completed" for s in stages if s["stage"].startswith("training-epoch")
    )
    evals_done = all(s["status"] == "completed" for s in stages if s["stage"].startswith("eval-epoch"))
    if not training_done:
        first = next(s for s in stages if s["stage"].startswith("training-epoch") and s["status"] != "completed")
        next_action = f"resume-training (from {first['stage']}: {first['status']})"
    elif not evals_done:
        first = next(s for s in stages if s["stage"].startswith("eval-epoch") and s["status"] != "completed")
        next_action = f"resume-evaluation ({first['stage']}: {first['status']})"
    elif not selection["finalized"]:
        next_action = "finalize-selection"
    elif selection.get("selected_checkpoint_digest") is not None and reveal["status"] != "REVEALED":
        next_action = "run-heldout-reveal"
    elif not comparison_finalized:
        next_action = "finalize-resource-comparison"
    else:
        next_action = "complete"

    return {
        "seed": seed, "stages": stages, "selection": selection, "reveal": reveal,
        "resource_comparison_finalized": comparison_finalized, "next_action": next_action,
    }


def aggregate_seed_resource_intervals(recovery_root: Path, *, seed: int, epochs: int,
                                       unavailable_intervals=()) -> dict:
    """Real per-seed cost from the attempt ledger: completed and interrupted GPU
    attempts are summed; explicitly declared power-loss gaps are recorded as
    `unavailable` rather than silently counted as zero or as GPU time.
    """
    ledger = AttemptLedger(Path(recovery_root) / "attempts.jsonl")
    prefixes = _seed_ledger_prefixes(seed, epochs)
    rows = [row for row in ledger.rows() if any(row["attempt_id"].startswith(p) for p in prefixes)]
    active_wall_seconds = sum(float(row["wall_seconds"]) for row in rows)
    interrupted = [row for row in rows if row["status"] == "interrupted"]
    unavailable = [
        {"seconds": float(item["seconds"]), "reason": item.get("reason", "unspecified")}
        for item in unavailable_intervals
    ]
    unavailable_seconds = sum(item["seconds"] for item in unavailable)
    return {
        "active_wall_seconds": active_wall_seconds,
        "attempt_count": len(rows),
        "interrupted_attempt_count": len(interrupted),
        "attempts": rows,
        "unavailable_intervals": unavailable,
        "unavailable_seconds": unavailable_seconds,
        "elapsed_including_unavailable_seconds": active_wall_seconds + unavailable_seconds,
    }


def _canonical_selection_digest(selection_path: Path) -> str:
    record = json.loads(selection_path.read_text(encoding="utf-8"))
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def discover_finalized_seed_evidence(manifest_path, *, recovery_root: Path, output_root: Path,
                                      seed: int) -> dict:
    """Validate the final discovery topology for a completed seed: one training
    result, three evaluations, one selection, a conditional reveal only when the
    selection is eligible, resource evidence, and clean checksums -- with every
    recovery-workspace path rejected as a finalized input.
    """
    frozen = load_manifest(manifest_path)
    epochs = frozen["training"]["optimizer"]["epochs"]
    workspace = RecoveryWorkspace(recovery_root, output_root)

    boundary = _completed_training_boundary(recovery_root, seed, epochs)
    if boundary is None:
        raise ValueError(f"seed {seed}: training is not finalized")
    training_bundle = workspace.completed_bundle(boundary[1])
    eval_bundles = [workspace.completed_bundle(_eval_stage_key(seed, e)) for e in range(1, epochs + 1)]
    finalized_inputs_only([training_bundle, *eval_bundles], recovery_root)

    selection_path = Path(output_root) / f"selection-seed{seed}" / "selection_record.json"
    verify_selection_record(selection_path, _canonical_selection_digest(selection_path))
    record = json.loads(selection_path.read_text(encoding="utf-8"))
    if not record.get("finalized"):
        raise ValueError(f"seed {seed}: selection record is not finalized")

    comparison_dir = Path(output_root) / f"seed{seed}-resource-comparison"
    verify_seed_comparison_artifact(comparison_dir)

    reveal_bundle = None
    reveal_state = _read_stage_state(recovery_root, _reveal_stage_key(seed))
    if record.get("selected_checkpoint_digest") is not None:
        if not reveal_state or reveal_state.get("status") != "REVEALED":
            raise ValueError(f"seed {seed}: eligible selection has no completed held-out reveal")
        reveal_bundle = Path(reveal_state["transaction"]["reveal_bundle"])
        verify_bundle(reveal_bundle)
        if RecoveryWorkspace(recovery_root, output_root).root.resolve() in reveal_bundle.resolve().parents:
            raise ValueError("reveal bundle must not live under the recovery workspace")
    elif reveal_state and reveal_state.get("status") == "REVEALED":
        raise ValueError(f"seed {seed}: null selection must not produce a reveal bundle")

    return {
        "seed": seed,
        "training_bundle": str(training_bundle),
        "evaluation_bundles": [str(path) for path in eval_bundles],
        "selection_record": str(selection_path),
        "selected_checkpoint_digest": record.get("selected_checkpoint_digest"),
        "reveal_bundle": str(reveal_bundle) if reveal_bundle is not None else None,
        "resource_comparison": str(comparison_dir),
        "recovery_state_excluded": True,
    }


def seed_notes_text(stage: str) -> str:
    return (
        f"# {stage.replace('_', ' ').title()} run notes\n\n"
        "Real Hugging Face/CUDA adapters; part of the frozen seed training/evaluation/"
        "selection protocol (RESEARCH_PLAN.md Phase 5+6). Held-out InjecAgent is not "
        "read or revealed by this run.\n"
    )


def _epoch_number(checkpoint_key: str) -> int:
    return int(checkpoint_key.rsplit("-", 1)[1])


def _peak_vram_mb_from_bundles(bundle_dirs) -> float:
    """Durable peak VRAM: the max `vram_mb` recorded in every stage bundle's
    `gpu.csv`. A pure resume run has no fresh telemetry, so the finalized bundles
    -- not this process's in-memory rows -- are the authoritative source.
    """
    peak = 0.0
    for bundle_dir in bundle_dirs:
        csv_path = Path(bundle_dir) / "gpu.csv"
        if not csv_path.exists():
            continue
        for line in csv_path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    peak = max(peak, float(parts[1]))
                except ValueError:
                    pass
    return peak


def _release(adapter) -> None:
    getattr(adapter, "release", lambda: None)()


def _orchestrate_seed(*, manifest_path, frozen, baseline_benchmarks, output_root: Path, seed: int,
                       prior_cumulative_gpu_hours: float, reproduction_command, upstream_provenance,
                       storage, trainer, make_stage_telemetry, make_eval_model, make_eval_dataset,
                       scorer, recovery_root: Path | None = None, heldout: dict | None = None,
                       unavailable_intervals=(), stamp: str | None = None) -> dict:
    """Recovery-aware seed orchestration shared by the real run and its tests.

    When `recovery_root` is set every stage is routed through its recovery
    contract (`runner.training` / `runner.evaluation` / `runner.reveal`), the
    per-seed identity is stable across restarts, the selection record and
    resource artifact live at stable ids, and an ordinary interruption
    propagates so the next invocation resumes from the last safe boundary.
    """
    epochs = frozen["training"]["optimizer"]["epochs"]
    base_seq = frozen["training"]["data"]["max_sequence_length"]
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    # Per-attempt run-bundle id: unique even if a session is restarted inside the
    # same second. Recovery keys on the stable stage id, never on this.
    run_stamp = f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    Path(output_root).mkdir(parents=True, exist_ok=True)
    recovery_aware = recovery_root is not None
    overall_start = time.monotonic()
    workspace = RecoveryWorkspace(recovery_root, output_root) if recovery_aware else None
    training_recovery = TrainingRecovery(workspace, f"training-seed{seed}") if workspace else None

    result = {"training": None, "evaluations": {}, "selection": None,
              "reveal": None, "upstream_provenance": upstream_provenance}

    train_telemetry = make_stage_telemetry()
    training = run_training(
        manifest_path, trainer=trainer, telemetry=train_telemetry, storage=storage,
        seed=seed, run_id=f"training-seed{seed}-{run_stamp}", recovery=training_recovery,
    )
    verify_bundle(Path(training.bundle_dir))
    result["training"] = training
    live_peak_rows = list(getattr(train_telemetry, "_rows", []))
    bundle_dirs = [Path(training.bundle_dir)]

    def _resource_comparison(selection_dirs) -> tuple[dict, Path]:
        dirs = bundle_dirs + list(selection_dirs)
        if recovery_aware:
            intervals = aggregate_seed_resource_intervals(
                recovery_root, seed=seed, epochs=epochs, unavailable_intervals=unavailable_intervals,
            )
            wall_seconds = intervals["active_wall_seconds"]
            peak_vram_mb = _peak_vram_mb_from_bundles(dirs)
        else:
            intervals = None
            wall_seconds = time.monotonic() - overall_start
            peak_vram_mb = max((row["vram_mb"] for row in live_peak_rows), default=0)
        comparison = compare_seed_resource_use(
            frozen, wall_seconds=wall_seconds, peak_vram_mb=peak_vram_mb,
            bundle_bytes=sum(_directory_bytes(d) for d in dirs),
            prior_cumulative_gpu_hours=prior_cumulative_gpu_hours,
        )
        comparison["upstream_provenance"] = upstream_provenance
        if intervals is not None:
            comparison["resource_intervals"] = intervals
        artifact_id = (f"seed{seed}-resource-comparison" if recovery_aware
                       else f"seed{seed}-resource-comparison-{stamp}")
        writer = write_or_verify_seed_comparison_artifact if recovery_aware else write_seed_comparison_artifact
        return comparison, writer(output_root, artifact_id, comparison)

    if training.outcome != "success":
        # Unrecoverable OOM (or other technical failure): `run_training` preserved
        # its own failure-evidence bundle. Nothing to evaluate or select.
        _release(trainer)
        comparison, comparison_dir = _resource_comparison([])
        result["resource_comparison"] = comparison
        result["resource_comparison_dir"] = str(comparison_dir)
        return result

    epoch_metrics: dict[int, dict] = {}
    for checkpoint_key, checkpoint in sorted(training.metrics["checkpoints"].items()):
        epoch = _epoch_number(checkpoint_key)
        model = make_eval_model(checkpoint)
        eval_telemetry = make_stage_telemetry()
        evaluation_recovery = (
            EvaluationRecovery(workspace, _eval_stage_key(seed, epoch)) if workspace else None
        )
        evaluation = run_trained_evaluation(
            manifest_path, model=model, dataset=make_eval_dataset(), scorer=scorer,
            telemetry=eval_telemetry, storage=storage, seed=seed, epoch=epoch, checkpoint=checkpoint,
            run_id=f"eval-seed{seed}-epoch{epoch}-{run_stamp}", recovery=evaluation_recovery,
        )
        verify_bundle(Path(evaluation.bundle_dir))
        _release(model)
        live_peak_rows.extend(getattr(eval_telemetry, "_rows", []))
        bundle_dirs.append(Path(evaluation.bundle_dir))
        epoch_metrics[epoch] = evaluation.metrics
        result["evaluations"][epoch] = evaluation

    _release(trainer)

    candidates = build_candidates(epoch_metrics)
    record = select_checkpoint(manifest_path, baseline_benchmarks=baseline_benchmarks, candidates=candidates)
    if recovery_aware:
        selection_path = Path(output_root) / f"selection-seed{seed}" / "selection_record.json"
    else:
        selection_path = Path(output_root) / f"selection-seed{seed}-{stamp}" / "selection_record.json"
    finalized = finalize_selection_record(record, selection_path)
    result["selection"] = {"record": record, **finalized}

    selected_digest = record.get("selected_checkpoint_digest")
    if selected_digest is not None:
        if not recovery_aware or heldout is None:
            raise RuntimeError(
                f"seed {seed} finalized an eligible checkpoint ({selected_digest}) but no sealed "
                "held-out material was supplied; halting for held-out authorization rather than "
                "revealing the baseline alone or promoting a capability-failing checkpoint"
            )
        selected_epoch = record["selected_epoch"]
        selected_checkpoint = training.metrics["checkpoints"][f"epoch-{selected_epoch}"]
        reveal_recovery = HeldOutRevealRecovery(workspace, _reveal_stage_key(seed))
        reveal_result = run_selection_and_reveal(
            manifest_path, selection_record=record, selection_path=selection_path,
            sealer=heldout["sealer"], model=heldout["make_model"](selected_checkpoint),
            dataset=heldout["dataset"], scorer=heldout["scorer"], storage=storage,
            telemetry=make_stage_telemetry(), recovery=reveal_recovery,
            authorization_identity=heldout.get("authorization_identity", "default"),
            checkpoint_digest=selected_digest,
        )
        result["reveal"] = reveal_result

    comparison, comparison_dir = _resource_comparison([selection_path.parent])
    result["resource_comparison"] = comparison
    result["resource_comparison_dir"] = str(comparison_dir)

    if recovery_aware:
        result["discovery"] = discover_finalized_seed_evidence(
            manifest_path, recovery_root=recovery_root, output_root=output_root, seed=seed,
        )
    return result


def run_real_seed(*, manifest_path: Path, upstream_root: Path, suite_dir: Path, heldout_dir: Path,
                  dataset_path: Path, baseline_metrics_path: Path, output_root: Path,
                  model_cache: Path, work_dir: Path, seed: int,
                  prior_cumulative_gpu_hours: float, reproduction_command: str | None = None,
                  smoke_max_steps: int | None = None,
                  max_items_per_benchmark: int | dict | None = None,
                  recovery_root: Path | None = None,
                  unavailable_intervals=()) -> dict:
    """`smoke_max_steps` and `max_items_per_benchmark` default to `None`, which is
    what the real evidence run must use: the frozen manifest's exact epoch/step
    counts and exact sample counts, with no reduction. They exist so this same
    orchestration can be smoke-validated end-to-end (a couple of steps, a couple
    of eval items per benchmark) before committing to the real, multi-hour run.
    Passing either non-None means the resulting run is a code-path smoke check,
    not scientific evidence.

    `recovery_root` (issue #22) points at an operational workspace **outside**
    `output_root`. When set, every stage resumes at its own safe boundary, the
    per-seed identity is stable across sessions, and a restarted seed exposes the
    same finalized artifacts as an uninterrupted one.
    """
    frozen = load_manifest(manifest_path)
    if seed not in frozen["training"]["seeds"]:
        raise ValueError(f"seed {seed} is not one of the manifest's frozen seeds {frozen['training']['seeds']}")
    upstream_provenance = _verify_upstream(
        upstream_root, frozen["upstream"]["commit"], frozen["upstream"]["tree"]
    )
    examples = load_training_examples(dataset_path)
    baseline_benchmarks = load_baseline_benchmarks(baseline_metrics_path)

    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(
        repo_id=frozen["model"]["id"], revision=frozen["model"]["revision"],
        cache_dir=model_cache, local_files_only=True,
    )

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output_root.mkdir(parents=True, exist_ok=True)
    storage = LocalStorageAdapter(output_root)
    trainer = RealQLoRATrainerAdapter(
        snapshot, None, examples, work_dir / f"adapters-seed{seed}-{stamp}",
        smoke_max_steps=smoke_max_steps,
    )

    def make_stage_telemetry():
        telemetry = RealTelemetryAdapter()
        telemetry.command_text = reproduction_command
        telemetry.notes_text = seed_notes_text
        return telemetry

    def make_eval_model(checkpoint):
        return RealModelAdapter(
            checkpoint["merged_dir"], None, upstream_root, decoding=frozen["evaluation"]["decoding"],
        )

    def make_eval_dataset():
        return RealDatasetAdapter(suite_dir, heldout_dir, max_items_per_benchmark=max_items_per_benchmark)

    return _orchestrate_seed(
        manifest_path=manifest_path, frozen=frozen, baseline_benchmarks=baseline_benchmarks,
        output_root=output_root, seed=seed, prior_cumulative_gpu_hours=prior_cumulative_gpu_hours,
        reproduction_command=reproduction_command, upstream_provenance=upstream_provenance,
        storage=storage, trainer=trainer, make_stage_telemetry=make_stage_telemetry,
        make_eval_model=make_eval_model, make_eval_dataset=make_eval_dataset,
        scorer=RealScorerAdapter(upstream_root), recovery_root=recovery_root,
        unavailable_intervals=unavailable_intervals, stamp=stamp,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("protocol/manifest.json"))
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prior-cumulative-gpu-hours", type=float, required=True)
    parser.add_argument(
        "--smoke-max-steps", type=int, default=None,
        help="code-path smoke check only: cap optimizer steps per epoch (omit for real Phase-5/6 evidence)",
    )
    parser.add_argument(
        "--smoke-max-items-per-benchmark", type=int, default=None,
        help="code-path smoke check only: cap eval items per benchmark (omit for real Phase-5/6 evidence)",
    )
    parser.add_argument(
        "--recovery-root", type=Path, default=None,
        help="issue #22: operational recovery workspace OUTSIDE --output-root; enables split-run resume",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="print the restart status for --seed under --recovery-root and exit without running anything",
    )
    parser.add_argument(
        "--unavailable-interval", action="append", default=[], metavar="SECONDS:REASON",
        help="declare an explicit hard-power-loss gap to record (not counted as GPU time); repeatable",
    )
    args = parser.parse_args(argv)

    if args.status:
        if args.recovery_root is None:
            parser.error("--status requires --recovery-root")
        print(json.dumps(
            seed_run_status(args.manifest, recovery_root=args.recovery_root,
                            output_root=args.output_root, seed=args.seed),
            indent=2, sort_keys=True, default=str,
        ))
        return 0

    unavailable_intervals = []
    for spec in args.unavailable_interval:
        seconds, _, reason = spec.partition(":")
        unavailable_intervals.append({"seconds": float(seconds), "reason": reason or "unspecified"})

    command_text = shlex.join([sys.executable, "-m", "runner.real_seed_run", *(argv or sys.argv[1:])])
    result = run_real_seed(
        manifest_path=args.manifest, upstream_root=args.upstream_root,
        suite_dir=args.suite_dir, heldout_dir=args.heldout_dir,
        dataset_path=args.dataset_path, baseline_metrics_path=args.baseline_metrics,
        output_root=args.output_root, model_cache=args.model_cache, work_dir=args.work_dir,
        seed=args.seed, prior_cumulative_gpu_hours=args.prior_cumulative_gpu_hours,
        reproduction_command=command_text,
        smoke_max_steps=args.smoke_max_steps,
        max_items_per_benchmark=args.smoke_max_items_per_benchmark,
        recovery_root=args.recovery_root,
        unavailable_intervals=unavailable_intervals,
    )
    print(json.dumps({
        "training_bundle": result["training"].bundle_dir,
        "training_outcome": result["training"].outcome,
        "evaluation_bundles": {epoch: evaluation.bundle_dir for epoch, evaluation in result["evaluations"].items()},
        "selection": result["selection"],
        "reveal": (result["reveal"].__dict__ if result.get("reveal") is not None else None),
        "resource_comparison": result.get("resource_comparison"),
        "discovery": result.get("discovery"),
    }, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
