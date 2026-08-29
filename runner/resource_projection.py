"""Full-run resource projection, extrapolated from Phase 3 smoke-test measurements.

Per the real-GPU smoke-test ticket: "Projected full-run VRAM/wall-time/GPU-hours/disk,
extrapolated from the smoke-test measurements, are computed and compared against the
manifest's resource limits -- if projected use would exceed a limit, that is reported
as a feasibility finding, not silently absorbed." This module is pure Python (no torch
dependency) so its arithmetic is unit-testable without the real GPU/HF stack; the smoke
test itself supplies the measured inputs.

Item counts mirror `protocol/manifest.json`'s frozen `evaluation` block. Tensor Trust's
two benchmarks each score BOTH an attack arm and a defense-validity arm per published
row, so their generate-call count is 2x the row count (`600` = 2 x 300 published rows),
matching how `runner.core._run_benchmark` counts items via `resolve_sample_count`.
"""
from __future__ import annotations
import math

FULL_EVAL_ITEM_COUNTS = {
    "open_prompt_injection": 300,
    "tensor_trust_hijack": 600,
    "tensor_trust_extract": 600,
    "injecagent": 200,
    "mmlu": 300,
    "gsm8k": 200,
    "ifeval": 200,
}


def _full_training_steps_per_epoch(manifest: dict) -> int:
    training_cfg = manifest["training"]
    n_examples = training_cfg["data"]["count"]
    micro_batch = training_cfg["optimizer"]["micro_batch"]
    grad_accum = training_cfg["optimizer"]["gradient_accumulation"]
    return math.ceil(n_examples / (micro_batch * grad_accum))


def project_full_run_resources(
    manifest: dict, *,
    measured_seconds_per_item: dict,
    measured_peak_vram_mb: float,
    measured_train_seconds_per_step: float,
    measured_checkpoint_bytes: int,
    default_seconds_per_item: float,
) -> dict:
    """Extrapolate Phase 3 smoke-test measurements to the manifest-frozen full run.

    `measured_seconds_per_item`: {benchmark_name: seconds}, from timing the smoke
    test's own (tiny) real generate calls per benchmark. A benchmark missing from
    this dict (e.g. one the smoke test skipped) falls back to
    `default_seconds_per_item` -- explicitly recorded in the output so a reader
    can see which figures are measured vs. assumed.
    `measured_peak_vram_mb`: peak VRAM observed during the smoke test (eval + train).
    `measured_train_seconds_per_step`: wall time per optimizer step during the tiny
    real QLoRA training smoke test.
    `measured_checkpoint_bytes`: on-disk size of one real merged checkpoint dir.

    Returns a dict with the projected totals, the manifest's own limits, and a
    `feasibility_findings` list that is non-empty exactly when a projection
    exceeds its corresponding limit -- callers must not silently absorb this.
    """
    resources = manifest["resources"]
    training_cfg = manifest["training"]
    epochs = training_cfg["optimizer"]["epochs"]
    seeds = training_cfg["seeds"]
    n_seeds = len(seeds)

    per_benchmark_seconds = {}
    assumed_default_for = []
    for name, count in FULL_EVAL_ITEM_COUNTS.items():
        seconds_per_item = measured_seconds_per_item.get(name)
        if seconds_per_item is None:
            seconds_per_item = default_seconds_per_item
            assumed_default_for.append(name)
        per_benchmark_seconds[name] = count * seconds_per_item
    eval_pass_seconds = sum(per_benchmark_seconds.values())

    baseline_wall_seconds = eval_pass_seconds  # baseline runs the full suite exactly once
    trained_eval_wall_seconds = eval_pass_seconds * epochs * n_seeds

    steps_per_epoch = _full_training_steps_per_epoch(manifest)
    training_wall_seconds = measured_train_seconds_per_step * steps_per_epoch * epochs * n_seeds

    total_wall_seconds = baseline_wall_seconds + trained_eval_wall_seconds + training_wall_seconds
    total_gpu_hours = total_wall_seconds / 3600
    per_seed_wall_hours = (trained_eval_wall_seconds / n_seeds + training_wall_seconds / n_seeds) / 3600

    checkpoints_total = epochs * n_seeds
    projected_storage_bytes = measured_checkpoint_bytes * checkpoints_total
    projected_storage_gb = projected_storage_bytes / (1024 ** 3)

    projected_peak_vram_gb = measured_peak_vram_mb / 1024

    findings = []
    if projected_peak_vram_gb > resources["vram_allocated_gb_max"]:
        findings.append(
            f"projected peak VRAM {projected_peak_vram_gb:.2f} GB exceeds the "
            f"{resources['vram_allocated_gb_max']} GB manifest limit"
        )
    if per_seed_wall_hours > resources["wall_hours_per_seed_max"]:
        findings.append(
            f"projected per-seed wall time {per_seed_wall_hours:.2f}h exceeds the "
            f"{resources['wall_hours_per_seed_max']}h manifest limit"
        )
    if total_gpu_hours > resources["gpu_hours_total_max"]:
        findings.append(
            f"projected total GPU-hours {total_gpu_hours:.2f} exceeds the "
            f"{resources['gpu_hours_total_max']} GPU-hour manifest limit"
        )
    if projected_storage_gb > resources["storage_gb_max"]:
        findings.append(
            f"projected storage {projected_storage_gb:.2f} GB exceeds the "
            f"{resources['storage_gb_max']} GB manifest limit"
        )

    return {
        "per_benchmark_seconds": per_benchmark_seconds,
        "assumed_default_seconds_per_item_for": assumed_default_for,
        "baseline_wall_seconds": baseline_wall_seconds,
        "trained_eval_wall_seconds": trained_eval_wall_seconds,
        "training_wall_seconds": training_wall_seconds,
        "total_wall_seconds": total_wall_seconds,
        "total_gpu_hours": total_gpu_hours,
        "per_seed_wall_hours": per_seed_wall_hours,
        "projected_peak_vram_gb": projected_peak_vram_gb,
        "projected_storage_gb": projected_storage_gb,
        "checkpoints_total": checkpoints_total,
        "limits": {
            "vram_allocated_gb_max": resources["vram_allocated_gb_max"],
            "wall_hours_per_seed_max": resources["wall_hours_per_seed_max"],
            "gpu_hours_total_max": resources["gpu_hours_total_max"],
            "storage_gb_max": resources["storage_gb_max"],
        },
        "feasibility_findings": findings,
    }
