"""Claim tables and statistics (issue #28).

The publication's central table plus the statistics that make it defensible,
built as one pure transform over already-loaded ``metrics.json`` documents.

Attempt 1 finished with every completed seed finalizing
``NO_ELIGIBLE_CHECKPOINT``, so "baseline versus selected trained checkpoint" is
undefined. The analysis unit here is **every prespecified epoch of every
completed run** -- never a post-hoc winner, and no epoch is styled as "best".

The primary table's organising axis is **evaluation modality**, not the usual
safety-versus-capability split: one group is scored by sampled free generation
plus string/parser matching, the other by first-token log-likelihood ranking
over fixed candidates with no tokens generated at all. Grouping this way makes
the finding visible in the table's structure. Each checkpoint row carries a
column answering whether a capability gate built only from the no-generation
benchmark, at the manifest's own tolerance, would have passed it.

Everything is a pure function over dicts: no model, dataset, scorer, trainer,
telemetry or storage dependency, and no I/O. The paired bootstrap is delegated
to :mod:`runner.analysis` (the frozen bootstrap stage) and the cross-run
summary to :func:`runner.continuation.summarize_seeds`; only exact McNemar,
which the bootstrap does not supply, is implemented here.
"""
from __future__ import annotations

import math

from protocol.validate_manifest import load as load_manifest
from runner.analysis import (
    bootstrap_benchmark_difference,
    bootstrap_visible_composite,
    summarize_seeds,
)
from runner.core import CAPABILITY_BENCHMARKS, VISIBLE_SAFETY_BENCHMARKS

BENCHMARKS = VISIBLE_SAFETY_BENCHMARKS + CAPABILITY_BENCHMARKS

MODALITY_FREE_GENERATION = "free_generation_sampled_string_scored"
MODALITY_LIKELIHOOD_RANKED = "likelihood_ranked_no_generation"

_POST_HOC_KEYS = frozenset({
    "selected_checkpoint_digest", "selected_epoch", "finalized", "selected",
    "best", "is_winner", "winner", "rank",
})

INTERVAL_CONDITIONALITY = (
    "Interval is conditional on the specific evaluated example IDs (the frozen "
    "sample_ids for this benchmark); it is not an inferential statement about "
    "the benchmark's full population."
)

PRIMARY_TABLE_CAPTION = (
    "Benchmarks are grouped by evaluation modality, not by the usual "
    "safety-versus-capability split. The two groups differ on four confounded "
    "axes at once: (1) the chat template applied to the prompt; (2) sampled "
    "free-generation decoding versus deterministic likelihood scoring; (3) the "
    "generation token budget; and (4) the scoring method -- string/parser "
    "matching over generated text versus first-token log-likelihood ranking "
    "over fixed candidates. These axes are confounded together, not cleanly "
    "separated, so this table is an existence proof about measurement modality, "
    "not a clean two-way contrast. Ticket #30 tests the largest single axis "
    "(chat-mode MMLU); the final wording of this caption depends on its "
    "outcome. The multiple-choice-only-gate column reports whether a capability "
    "gate built solely from the no-generation, likelihood-ranked benchmark "
    "would have passed each checkpoint at the manifest's own tolerance."
)

COMPOSITE_DECOMPOSITION_NOTE = (
    "The visible composite is an unweighted mean of absolute deltas across "
    "benchmarks with very different baseline headroom, so a single benchmark "
    "can dominate it arithmetically as well as empirically. It may never be "
    "reported without this per-benchmark decomposition beside it."
)


class CompositeWithoutDecompositionError(ValueError):
    """Raised on any attempt to render the visible composite value alone."""


# --- small helpers ------------------------------------------------------


def _item_scores(benchmark: dict) -> dict:
    return {example_id: entry["score"] for example_id, entry in benchmark["items"].items()}


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _is_binary(*score_maps: dict) -> bool:
    return all(
        value in (0, 1, 0.0, 1.0)
        for score_map in score_maps for value in score_map.values()
    )


def _paired_scores(baseline_metrics: dict, doc: dict, name: str) -> tuple[dict, dict]:
    base = _item_scores(baseline_metrics["benchmarks"][name])
    cand = _item_scores(doc["benchmarks"][name])
    if set(base) != set(cand):
        raise ValueError(
            f"{name}: baseline and trained epoch must be scored over identical example IDs"
        )
    if not base:
        raise ValueError(f"{name}: no example IDs to compare")
    return base, cand


def _benchmark_eval_config(evaluation: dict, name: str) -> dict:
    if name in evaluation["visible_safety"]:
        return evaluation["visible_safety"][name]
    return evaluation["capability"][name]


def _modality(config: dict) -> str:
    """Classify a benchmark by how it is scored, from its frozen manifest eval
    config -- never a hardcoded benchmark list."""
    if config.get("scorer") == "first_token_logit" or config.get("max_new_tokens") == 1:
        return MODALITY_LIKELIHOOD_RANKED
    return MODALITY_FREE_GENERATION


# --- analysis unit -----------------------------------------------------


def analysis_units(epoch_metrics, epochs: int) -> list[dict]:
    """Validated, ordered list of ``{"seed", "epoch", "doc"}`` -- one per
    prespecified epoch of every completed run.

    Rejects anything that looks like a post-hoc winner: a selection-record-shaped
    dict, an epoch doc carrying a ``selected``/``best``/``rank`` marker, or a
    run supplying only some of its prespecified epochs.
    """
    by_seed: dict = {}
    for doc in epoch_metrics:
        marker = _POST_HOC_KEYS.intersection(doc)
        if marker:
            raise ValueError(
                f"analysis input carries post-hoc selection keys {sorted(marker)}; "
                "the analysis unit is every prespecified epoch, never a selected winner"
            )
        for key in ("seed", "epoch", "benchmarks"):
            if key not in doc:
                raise ValueError(f"epoch metrics document is missing {key!r}")
        seed_bucket = by_seed.setdefault(doc["seed"], {})
        if doc["epoch"] in seed_bucket:
            raise ValueError(f"seed {doc['seed']}: epoch {doc['epoch']} supplied twice")
        seed_bucket[doc["epoch"]] = doc

    if not by_seed:
        raise ValueError("no epoch metrics supplied")

    wanted = set(range(1, epochs + 1))
    units: list[dict] = []
    for seed in sorted(by_seed):
        got = set(by_seed[seed])
        if got != wanted:
            raise ValueError(
                f"seed {seed}: the analysis unit is every prespecified epoch "
                f"{sorted(wanted)}, got {sorted(got)}; a partial set (for example only "
                "a post-hoc winner) is rejected"
            )
        for epoch in sorted(wanted):
            units.append({"seed": seed, "epoch": epoch, "doc": by_seed[seed][epoch]})
    return units


# --- exact McNemar ----------------------------------------------------


def mcnemar_exact(baseline_scores: dict, candidate_scores: dict) -> dict:
    """Two-sided exact McNemar test for paired binary outcomes over the fixed
    example IDs common to both inputs.

    Uses the exact binomial tail on the discordant pairs with p = 0.5 (the test
    a reviewer expects for paired binary outcomes; the paired bootstrap does not
    supply it). ``b`` counts IDs the baseline got right and the trained epoch
    got wrong; ``c`` counts the reverse.
    """
    ids = sorted(baseline_scores)
    if set(ids) != set(candidate_scores):
        raise ValueError("exact McNemar needs identical paired example IDs")
    if not ids:
        raise ValueError("exact McNemar needs at least one paired example")

    b = c = 0
    for example_id in ids:
        base = baseline_scores[example_id]
        cand = candidate_scores[example_id]
        for value in (base, cand):
            if value not in (0, 1, 0.0, 1.0):
                raise ValueError(
                    f"exact McNemar needs binary 0/1 scores, got {value!r} for {example_id}"
                )
        if base > cand:
            b += 1
        elif cand > base:
            c += 1

    n = b + c
    k = min(b, c)
    tail = sum(math.comb(n, j) for j in range(k + 1)) * (0.5 ** n)
    p_value = min(1.0, 2.0 * tail)
    return {
        "discordant_baseline_only": b,
        "discordant_trained_only": c,
        "n_discordant": n,
        "p_value": p_value,
        "test": "exact McNemar (two-sided binomial, p=0.5)",
    }


# --- visible composite ----------------------------------------------


def visible_composite_block(baseline_metrics: dict, doc: dict) -> dict:
    """The visible composite for one checkpoint, always packaged with its
    per-benchmark decomposition. There is deliberately no code path that returns
    the scalar alone -- see :func:`render_composite`.
    """
    per_benchmark_delta = {}
    per_benchmark_baseline = {}
    per_benchmark_trained = {}
    for name in VISIBLE_SAFETY_BENCHMARKS:
        base, cand = _paired_scores(baseline_metrics, doc, name)
        per_benchmark_baseline[name] = _mean(base.values())
        per_benchmark_trained[name] = _mean(cand.values())
        per_benchmark_delta[name] = per_benchmark_trained[name] - per_benchmark_baseline[name]

    composite = _mean(per_benchmark_delta.values())
    dominant = max(per_benchmark_delta, key=lambda name: abs(per_benchmark_delta[name]))
    return {
        "composite_absolute_delta": composite,
        "per_benchmark_delta": per_benchmark_delta,
        "per_benchmark_baseline": per_benchmark_baseline,
        "per_benchmark_trained": per_benchmark_trained,
        "dominant_benchmark": dominant,
        "definition": "unweighted mean of the three visible-safety absolute deltas",
        "decomposition_note": COMPOSITE_DECOMPOSITION_NOTE,
    }


def render_composite(block: dict) -> dict:
    """Return a rendered composite row. Raises if ``block`` does not carry a
    non-empty per-benchmark decomposition -- the composite is never shown alone.
    """
    decomposition = block.get("per_benchmark_delta")
    if not decomposition:
        raise CompositeWithoutDecompositionError(
            "the visible composite may not be rendered without its per-benchmark decomposition"
        )
    return {
        "composite_absolute_delta": block["composite_absolute_delta"],
        "per_benchmark_delta": dict(decomposition),
        "dominant_benchmark": block["dominant_benchmark"],
        "decomposition_note": block["decomposition_note"],
    }


# --- cross-run summary --------------------------------------------


def _cross_run_block(manifest_path, values_by_run: dict) -> dict:
    raw = summarize_seeds(manifest_path, values_by_run)
    n = len(values_by_run)
    return {
        "per_run": dict(sorted(raw["per_seed"].items())),
        "mean": raw["mean"],
        "range": raw["range"],
        "population_standard_deviation": raw["standard_deviation"],
        "n_runs": n,
        "framing": (
            f"Descriptive population statistic over the runs actually executed (N={n}). "
            "'population_standard_deviation' is the population standard deviation of "
            "those runs, not an inferential or population-level confidence interval. "
            "It measures run-to-run variability under a fixed nominal configuration -- "
            "adapter initialisation happens before the run seed is applied -- and must "
            "not be described as seed variance."
        ),
    }


# --- top-level report --------------------------------------------


def build_claim_report(manifest_path, *, baseline_metrics: dict, epoch_metrics) -> dict:
    """Build the full claim-table report from already-loaded metrics documents.

    ``baseline_metrics`` is the frozen baseline ``metrics.json`` dict;
    ``epoch_metrics`` is an iterable of per-epoch trained ``metrics.json`` dicts,
    each carrying ``seed``, ``epoch`` and ``benchmarks``. The completed-seed and
    trained-checkpoint counts are derived from the input, so the report is
    correct at two completed seeds and at three.
    """
    manifest = load_manifest(manifest_path)
    epochs = manifest["training"]["optimizer"]["epochs"]
    evaluation = manifest["evaluation"]
    gates = manifest["selection"]["capability_gates"]

    units = analysis_units(list(epoch_metrics), epochs)
    seeds = sorted({unit["seed"] for unit in units})

    modality = {
        name: _modality(_benchmark_eval_config(evaluation, name)) for name in BENCHMARKS
    }
    mc_only_benchmarks = [
        name for name in CAPABILITY_BENCHMARKS
        if modality[name] == MODALITY_LIKELIHOOD_RANKED
    ]

    rows: list[dict] = []
    bootstrap: list[dict] = []
    mcnemar: list[dict] = []
    composites: list[dict] = []

    for unit in units:
        seed, epoch, doc = unit["seed"], unit["epoch"], unit["doc"]

        mc_only_gate_passes = None
        if mc_only_benchmarks:
            mc_only_gate_passes = True
            for name in mc_only_benchmarks:
                base, cand = _paired_scores(baseline_metrics, doc, name)
                decline = _mean(base.values()) - _mean(cand.values())
                if decline > gates[f"{name}_max_decline"]:
                    mc_only_gate_passes = False

        for name in BENCHMARKS:
            base, cand = _paired_scores(baseline_metrics, doc, name)
            base_value, trained_value = _mean(base.values()), _mean(cand.values())
            rows.append({
                "run_seed": seed,
                "epoch": epoch,
                "benchmark": name,
                "modality": modality[name],
                "baseline": base_value,
                "trained": trained_value,
                "absolute_delta": trained_value - base_value,
                "n": len(cand),
                "multiple_choice_only_gate_passes": mc_only_gate_passes,
            })

            difference = bootstrap_benchmark_difference(
                manifest_path, name=name,
                baseline_benchmark=baseline_metrics["benchmarks"][name],
                candidate_benchmark=doc["benchmarks"][name],
            )
            bootstrap.append({
                "run_seed": seed, "epoch": epoch,
                **difference,
                "conditional_on": INTERVAL_CONDITIONALITY,
            })

            if _is_binary(base, cand):
                mcnemar.append({
                    "run_seed": seed, "epoch": epoch, "benchmark": name,
                    **mcnemar_exact(base, cand),
                })
            else:
                mcnemar.append({
                    "run_seed": seed, "epoch": epoch, "benchmark": name,
                    "applicable": False,
                    "reason": (
                        "exact McNemar needs binary per-item outcomes; this "
                        "benchmark's metric averages multiple arms per item"
                    ),
                })

        composites.append({
            "run_seed": seed, "epoch": epoch,
            **render_composite(visible_composite_block(baseline_metrics, doc)),
        })

        composite_difference = bootstrap_visible_composite(
            manifest_path,
            baseline_benchmarks={
                name: baseline_metrics["benchmarks"][name]
                for name in VISIBLE_SAFETY_BENCHMARKS
            },
            candidate_benchmarks={
                name: doc["benchmarks"][name] for name in VISIBLE_SAFETY_BENCHMARKS
            },
        )
        bootstrap.append({
            "run_seed": seed, "epoch": epoch,
            **composite_difference,
            "conditional_on": INTERVAL_CONDITIONALITY,
        })

    modality_groups = {}
    for group in sorted(set(modality.values())):
        modality_groups[group] = {
            "benchmarks": [name for name in BENCHMARKS if modality[name] == group],
            "rows": [row for row in rows if row["modality"] == group],
        }

    primary_table = {
        "caption": PRIMARY_TABLE_CAPTION,
        "analysis_unit": (
            "every prespecified epoch of every completed run; no single epoch is "
            "elevated over the others"
        ),
        "grouping_axis": "evaluation_modality",
        "modality_groups": modality_groups,
        "multiple_choice_only_gate": {
            "benchmarks": mc_only_benchmarks,
            "max_decline": {
                name: gates[f"{name}_max_decline"] for name in mc_only_benchmarks
            },
            "source": "protocol/manifest.json selection.capability_gates",
            "meaning": (
                "whether a capability gate built solely from the no-generation, "
                "likelihood-ranked benchmark(s) would have passed this checkpoint at "
                "the manifest's own tolerance"
            ),
        },
    }

    cross_run_summary: dict = {}
    for epoch in range(1, epochs + 1):
        per_epoch: dict = {}
        for name in BENCHMARKS:
            values = {
                unit["seed"]: _mean(_item_scores(unit["doc"]["benchmarks"][name]).values())
                for unit in units if unit["epoch"] == epoch
            }
            if len(values) >= 2:
                per_epoch[name] = _cross_run_block(manifest_path, values)
        composite_values = {
            row["run_seed"]: row["composite_absolute_delta"]
            for row in composites if row["epoch"] == epoch
        }
        if len(composite_values) >= 2:
            per_epoch["visible_composite_delta"] = _cross_run_block(
                manifest_path, composite_values
            )
        if per_epoch:
            cross_run_summary[f"epoch_{epoch}"] = per_epoch

    return {
        "protocol_version": manifest["protocol_version"],
        "analysis_unit": primary_table["analysis_unit"],
        "completed_seeds": seeds,
        "completed_seed_count": len(seeds),
        "epochs_per_seed": epochs,
        "trained_checkpoint_count": len(units),
        "primary_table": primary_table,
        "paired_bootstrap": bootstrap,
        "mcnemar_exact": mcnemar,
        "visible_composite": composites,
        "cross_run_summary": cross_run_summary,
    }


def render_report(report: dict) -> str:
    """Canonical text form -- byte-identical for byte-identical inputs."""
    import json

    return json.dumps(report, indent=2, sort_keys=True) + "\n"
