"""Seed-replication continuation rule (`protocol/manifest.json`'s `training.seeds`
and `resources.gpu_hours_total_max`): seed 17 always runs first, alone, as a
technical feasibility pilot. Seeds 42 and 2026 auto-continue only if seed 17
completed without an unrecoverable technical failure *and* seed 17's own
measured cost, extrapolated across all three seeds, stays within the total
GPU-hour budget.

Poor model quality is never a stopping condition here, and that has to be
provably true rather than merely documented: `decide_continuation` reads
exactly two fields off `seed1_result` -- `outcome` and `gpu_hours` -- so there
is no code path by which a visible-safety, held-out, or capability number could
change its answer (`tests/test_continuation.py` asserts this by inspecting the
function's own source, not just by example).
"""
from __future__ import annotations
import dataclasses

from protocol.validate_manifest import load as load_manifest


@dataclasses.dataclass(frozen=True)
class ContinuationDecision:
    continue_replication: bool
    reasons: list
    projected_gpu_hours: float
    budget_gpu_hours: float


def decide_continuation(manifest_path, seed1_result: dict) -> ContinuationDecision:
    """Decide whether to auto-continue to seeds 42 and 2026 after seed 17.

    `seed1_result` is expected to carry at least `outcome` (`runner.training
    .run_training`'s own `TrainingResult.outcome` vocabulary: `"success"` or
    `"failed"`) and `gpu_hours` (seed 17's own measured cost); any other keys
    (visible/held-out/capability scores, notes, checkpoints, ...) are accepted
    on the dict but never read.
    """
    manifest = load_manifest(manifest_path)
    seeds = manifest["training"]["seeds"]
    budget = manifest["resources"]["gpu_hours_total_max"]

    projected = seed1_result["gpu_hours"] * len(seeds)
    reasons = []
    if seed1_result["outcome"] != "success":
        reasons.append(f"seed 1 did not complete technically (outcome={seed1_result['outcome']!r})")
    if projected > budget:
        reasons.append(f"projected total GPU-hours {projected:.2f} would exceed the {budget} GPU-hour budget")

    return ContinuationDecision(
        continue_replication=not reasons,
        reasons=reasons,
        projected_gpu_hours=projected,
        budget_gpu_hours=budget,
    )


def summarize_seeds(manifest_path, values_by_seed: dict) -> dict:
    """Per-seed/mean/range/standard-deviation summary of the seeds actually
    executed (`manifest.json`'s `analysis.seed_summary` fields), computed once
    all executed seeds are in. This is a descriptive summary of the seeds this
    study ran, not an inferential, population-level confidence interval --
    callers must not report `range`/`standard_deviation` as one.
    """
    manifest = load_manifest(manifest_path)
    expected_fields = set(manifest["analysis"]["seed_summary"])

    values = list(values_by_seed.values())
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n

    summary = {
        "per_seed": dict(sorted(values_by_seed.items())),
        "mean": mean,
        "range": {"min": min(values), "max": max(values)},
        "standard_deviation": variance ** 0.5,
    }
    if not expected_fields <= summary.keys():
        raise ValueError(f"summary is missing manifest-declared seed_summary fields: {expected_fields - summary.keys()}")
    summary["framing"] = (
        "descriptive summary of the seeds actually executed in this study; "
        "not a population-level confidence interval"
    )
    return summary
