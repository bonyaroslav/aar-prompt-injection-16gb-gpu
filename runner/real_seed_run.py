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
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner.bundle import verify_bundle
from runner.evaluation import run_trained_evaluation
from runner.gpu_smoke import _verify_upstream
from runner.real_adapters import RealDatasetAdapter, RealModelAdapter, RealScorerAdapter, RealTelemetryAdapter
from runner.real_baseline import _directory_bytes
from runner.real_training import RealQLoRATrainerAdapter
from runner.selection import finalize_selection_record, select_checkpoint
from runner.storage import LocalStorageAdapter
from runner.training import run_training


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


def seed_notes_text(stage: str) -> str:
    return (
        f"# {stage.replace('_', ' ').title()} run notes\n\n"
        "Real Hugging Face/CUDA adapters; part of the frozen seed training/evaluation/"
        "selection protocol (RESEARCH_PLAN.md Phase 5+6). Held-out InjecAgent is not "
        "read or revealed by this run.\n"
    )


def _epoch_number(checkpoint_key: str) -> int:
    return int(checkpoint_key.rsplit("-", 1)[1])


def run_real_seed(*, manifest_path: Path, upstream_root: Path, suite_dir: Path, heldout_dir: Path,
                  dataset_path: Path, baseline_metrics_path: Path, output_root: Path,
                  model_cache: Path, work_dir: Path, seed: int,
                  prior_cumulative_gpu_hours: float, reproduction_command: str | None = None,
                  smoke_max_steps: int | None = None,
                  max_items_per_benchmark: int | dict | None = None) -> dict:
    """`smoke_max_steps` and `max_items_per_benchmark` default to `None`, which is
    what the real issue-12 evidence run must use: the frozen manifest's exact
    epoch/step counts and exact sample counts, with no reduction. They exist so
    this same orchestration can be smoke-validated end-to-end (a couple of steps,
    a couple of eval items per benchmark) before committing to the real,
    multi-hour run -- mirroring `RealQLoRATrainerAdapter.smoke_max_steps` and
    `RealDatasetAdapter.max_items_per_benchmark`, which already exist for exactly
    this purpose (see `runner.gpu_smoke`). Passing either non-None here means the
    resulting run is a code-path smoke check, not Phase-5/6 evidence.
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
    train_telemetry = RealTelemetryAdapter()
    train_telemetry.command_text = reproduction_command
    train_telemetry.notes_text = seed_notes_text

    overall_start = time.monotonic()
    training = run_training(
        manifest_path, trainer=trainer, telemetry=train_telemetry, storage=storage,
        seed=seed, run_id=f"training-seed{seed}-{stamp}",
    )
    verify_bundle(Path(training.bundle_dir))
    peak_vram_rows = list(train_telemetry._rows)
    bundle_dirs = [Path(training.bundle_dir)]

    result = {"training": training, "evaluations": {}, "selection": None, "upstream_provenance": upstream_provenance}
    if training.outcome != "success":
        # An unrecoverable OOM (or other technical failure) is preserved as its own
        # failure-evidence bundle by `run_training` itself; there is nothing to
        # evaluate or select, so this stops here rather than fabricating candidates.
        trainer.release()
        wall_seconds = time.monotonic() - overall_start
        comparison = compare_seed_resource_use(
            frozen, wall_seconds=wall_seconds,
            peak_vram_mb=max((row["vram_mb"] for row in peak_vram_rows), default=0),
            bundle_bytes=sum(_directory_bytes(bundle_dir) for bundle_dir in bundle_dirs),
            prior_cumulative_gpu_hours=prior_cumulative_gpu_hours,
        )
        comparison["upstream_provenance"] = upstream_provenance
        comparison_dir = write_seed_comparison_artifact(
            output_root, f"seed{seed}-resource-comparison-{stamp}", comparison
        )
        result["resource_comparison"] = comparison
        result["resource_comparison_dir"] = str(comparison_dir)
        return result

    epoch_metrics: dict[int, dict] = {}
    for checkpoint_key, checkpoint in sorted(training.metrics["checkpoints"].items()):
        epoch = _epoch_number(checkpoint_key)
        model = RealModelAdapter(
            checkpoint["merged_dir"], None, upstream_root, decoding=frozen["evaluation"]["decoding"],
        )
        eval_telemetry = RealTelemetryAdapter()
        eval_telemetry.command_text = reproduction_command
        eval_telemetry.notes_text = seed_notes_text
        evaluation = run_trained_evaluation(
            manifest_path, model=model,
            dataset=RealDatasetAdapter(suite_dir, heldout_dir, max_items_per_benchmark=max_items_per_benchmark),
            scorer=RealScorerAdapter(upstream_root), telemetry=eval_telemetry, storage=storage,
            seed=seed, epoch=epoch, checkpoint=checkpoint,
            run_id=f"eval-seed{seed}-epoch{epoch}-{stamp}",
        )
        verify_bundle(Path(evaluation.bundle_dir))
        model.release()
        peak_vram_rows.extend(eval_telemetry._rows)
        bundle_dirs.append(Path(evaluation.bundle_dir))
        epoch_metrics[epoch] = evaluation.metrics
        result["evaluations"][epoch] = evaluation

    trainer.release()
    wall_seconds = time.monotonic() - overall_start

    candidates = build_candidates(epoch_metrics)
    record = select_checkpoint(manifest_path, baseline_benchmarks=baseline_benchmarks, candidates=candidates)
    selection_path = output_root / f"selection-seed{seed}-{stamp}" / "selection_record.json"
    finalized = finalize_selection_record(record, selection_path)
    result["selection"] = {"record": record, **finalized}
    bundle_dirs.append(selection_path.parent)

    comparison = compare_seed_resource_use(
        frozen, wall_seconds=wall_seconds,
        peak_vram_mb=max((row["vram_mb"] for row in peak_vram_rows), default=0),
        bundle_bytes=sum(_directory_bytes(bundle_dir) for bundle_dir in bundle_dirs),
        prior_cumulative_gpu_hours=prior_cumulative_gpu_hours,
    )
    comparison["upstream_provenance"] = upstream_provenance
    comparison_dir = write_seed_comparison_artifact(
        output_root, f"seed{seed}-resource-comparison-{stamp}", comparison
    )
    result["resource_comparison"] = comparison
    result["resource_comparison_dir"] = str(comparison_dir)
    return result


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
    args = parser.parse_args(argv)
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
    )
    print(json.dumps({
        "training_bundle": result["training"].bundle_dir,
        "training_outcome": result["training"].outcome,
        "evaluation_bundles": {epoch: evaluation.bundle_dir for epoch, evaluation in result["evaluations"].items()},
        "selection": result["selection"],
        "resource_comparison": result.get("resource_comparison"),
    }, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
