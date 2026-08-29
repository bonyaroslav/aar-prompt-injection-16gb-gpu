"""Bootstrap-analysis stage (`protocol/manifest.json`'s `analysis` block): paired
bootstrap resampling over fixed example IDs, producing 95% percentile intervals
for every baseline-to-trained difference and for the visible composite.

Operates purely on already-computed per-item scores -- the `metrics.json` shape
produced by `runner.core.run_baseline` / `runner.evaluation.run_trained_evaluation`
(`benchmarks[name]["items"][example_id]["score"]`) -- so it can be exercised with
plain fixtures, independent of any model/dataset/scorer/training adapter and of
whether the training/selection stages have produced anything real yet.

Determinism: every replicate draw comes from a single `random.Random(seed)`
seeded once at the top of the call and consumed in a fixed order (sorted example
IDs; benchmark order = `runner.core.VISIBLE_SAFETY_BENCHMARKS`), so the same
fixtures plus the same manifest-frozen `analysis.bootstrap_seed` always produce
byte-identical output across repeated runs.

For the three-seed summary (each seed's own value plus mean/range/standard
deviation, framed descriptively rather than as a population-level interval),
see `runner.continuation.summarize_seeds` -- re-exported here to complete this
stage's public API without duplicating that logic.
"""
from __future__ import annotations
import random

from protocol.validate_manifest import load as load_manifest
from runner.core import VISIBLE_SAFETY_BENCHMARKS
from runner.continuation import summarize_seeds  # noqa: F401 (re-exported)

INTERVAL_LABEL = "95_percentile_paired_by_fixed_example_id"


def _percentile(sorted_values: list, pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted list (matches
    numpy's default 'linear' method, without adding a numpy dependency)."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (pct / 100) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _item_scores(benchmark: dict) -> dict:
    return {example_id: entry["score"] for example_id, entry in benchmark["items"].items()}


def _paired_diffs(baseline_scores: dict, candidate_scores: dict, *, label: str) -> list:
    ids = sorted(baseline_scores)
    if set(ids) != set(candidate_scores):
        raise ValueError(f"{label}: baseline and candidate must share exactly the same fixed example IDs")
    if not ids:
        raise ValueError(f"{label}: no example IDs to bootstrap")
    return [candidate_scores[i] - baseline_scores[i] for i in ids]


def paired_bootstrap_ci(baseline_scores: dict, candidate_scores: dict, *, seed: int, replicates: int) -> dict:
    """95% percentile bootstrap CI for the paired mean difference
    (candidate - baseline) over the fixed example IDs common to both inputs.

    `baseline_scores` / `candidate_scores`: {example_id: numeric score}, keyed by
    the same fixed example IDs -- per-item scores are only comparable when paired
    by the same ID on both sides. Deterministic given `seed`.
    """
    diffs = _paired_diffs(baseline_scores, candidate_scores, label="paired_bootstrap_ci")
    n = len(diffs)
    observed = sum(diffs) / n

    rng = random.Random(seed)
    replicate_means = []
    for _ in range(replicates):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        replicate_means.append(total / n)
    replicate_means.sort()

    return {
        "observed_difference": observed,
        "ci_low": _percentile(replicate_means, 2.5),
        "ci_high": _percentile(replicate_means, 97.5),
        "n": n,
        "replicates": replicates,
        "seed": seed,
        "interval": INTERVAL_LABEL,
    }


def bootstrap_benchmark_difference(manifest_path, *, name: str, baseline_benchmark: dict, candidate_benchmark: dict) -> dict:
    """`paired_bootstrap_ci` for one named benchmark, using the manifest-frozen
    `analysis.bootstrap_seed` / `analysis.bootstrap_replicates` -- the seam
    production callers should use instead of hand-picking seed/replicates.
    """
    manifest = load_manifest(manifest_path)
    analysis_cfg = manifest["analysis"]
    result = paired_bootstrap_ci(
        _item_scores(baseline_benchmark), _item_scores(candidate_benchmark),
        seed=analysis_cfg["bootstrap_seed"], replicates=analysis_cfg["bootstrap_replicates"],
    )
    return {"benchmark": name, **result}


def bootstrap_visible_composite(manifest_path, *, baseline_benchmarks: dict, candidate_benchmarks: dict) -> dict:
    """95% percentile bootstrap CI for the visible composite -- the unweighted
    mean of the three visible-safety benchmarks' paired differences, the same
    definition `runner.selection.visible_composite` uses for the observed point
    estimate. Each replicate independently resamples each benchmark's own
    fixed-ID pool (paired within that benchmark), then averages the three
    resampled means, mirroring how the observed composite itself averages three
    independently measured per-benchmark differences. All resamples are drawn
    from one `random.Random(seed)` in fixed benchmark order for determinism.
    """
    manifest = load_manifest(manifest_path)
    analysis_cfg = manifest["analysis"]
    seed = analysis_cfg["bootstrap_seed"]
    replicates = analysis_cfg["bootstrap_replicates"]

    per_benchmark_diffs = {
        name: _paired_diffs(
            _item_scores(baseline_benchmarks[name]), _item_scores(candidate_benchmarks[name]), label=name,
        )
        for name in VISIBLE_SAFETY_BENCHMARKS
    }

    observed = sum(sum(d) / len(d) for d in per_benchmark_diffs.values()) / len(VISIBLE_SAFETY_BENCHMARKS)

    rng = random.Random(seed)
    replicate_composites = []
    for _ in range(replicates):
        benchmark_means = []
        for name in VISIBLE_SAFETY_BENCHMARKS:
            diffs = per_benchmark_diffs[name]
            n = len(diffs)
            total = sum(diffs[rng.randrange(n)] for _ in range(n))
            benchmark_means.append(total / n)
        replicate_composites.append(sum(benchmark_means) / len(benchmark_means))
    replicate_composites.sort()

    return {
        "benchmark": "visible_composite",
        "observed_difference": observed,
        "ci_low": _percentile(replicate_composites, 2.5),
        "ci_high": _percentile(replicate_composites, 97.5),
        "n": {name: len(d) for name, d in per_benchmark_diffs.items()},
        "replicates": replicates,
        "seed": seed,
        "interval": INTERVAL_LABEL,
    }
