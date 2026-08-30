"""Orchestrates the full ADR 0001 Route A training-data build: fetch -> filter
-> template -> dedup -> hash -> report counts, unattended end-to-end.

`build_dataset` is the pure, offline, fully-testable core: it takes
already-fetched raw rows and exclusion-pool key sets and returns examples plus
a report. `run_real_build` is the thin, network-touching wiring that fetches
the real sources and the real ADR 0001 exclusion pool, then writes the dataset
and report to disk -- this is the only function `scripts/build_training_data.py`
calls, and the only place in this package where "real" and "fake" paths meet.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from training_data.dedup import Deduplicator, pool_keys
from training_data.templates import (
    generate_ambiguous_boundary_examples,
    generate_clean_control_examples,
    generate_prompt_injection_examples,
    generate_refusal_calibration_examples,
)

# Matches protocol/manifest.json's training.data.mix (0.4/0.3/0.2/0.1) at count 5000.
TARGET_COUNTS = {
    "prompt_injection": 2000,
    "clean_control": 1500,
    "ambiguous_boundary": 1000,
    "refusal_calibration": 500,
}
assert sum(TARGET_COUNTS.values()) == 5000

# Construction-order shuffle only (not a protocol value): keeps clean_control and
# ambiguous_boundary drawing from disjoint Dolly rows without depending on Dolly's
# on-disk category ordering. Changing this reshuffles which Dolly rows land in which
# category but never changes the frozen target counts or mix.
_DOLLY_SHUFFLE_SEED = 20260830
_DOLLY_OVERSAMPLE_FACTOR = 3


def build_dataset(*, injection_raw_rows, dolly_rows, exclusion_exact_keys, exclusion_near_keys,
                   token_cap: int, targets: dict[str, int] | None = None) -> dict:
    targets = dict(targets or TARGET_COUNTS)
    dedup = Deduplicator(exclude_exact=exclusion_exact_keys, exclude_near=exclusion_near_keys)

    shuffled_dolly = list(dolly_rows)
    random.Random(_DOLLY_SHUFFLE_SEED).shuffle(shuffled_dolly)
    clean_pool_size = min(len(shuffled_dolly), targets["clean_control"] * _DOLLY_OVERSAMPLE_FACTOR)
    clean_pool = shuffled_dolly[:clean_pool_size]
    # The rest of the shuffled pool, not a fixed multiple: ambiguous_boundary only accepts
    # rows that already have a Dolly `context` field (~30% of the corpus), and clean_control
    # draws from all categories regardless of context, so a same-sized slice would starve
    # ambiguous_boundary of enough context-bearing rows to hit its target.
    boundary_pool = shuffled_dolly[clean_pool_size:]

    by_category = {
        "prompt_injection": generate_prompt_injection_examples(
            injection_raw_rows, targets["prompt_injection"], dedup, token_cap
        ),
        "clean_control": generate_clean_control_examples(
            clean_pool, targets["clean_control"], dedup, token_cap
        ),
        "ambiguous_boundary": generate_ambiguous_boundary_examples(
            boundary_pool, targets["ambiguous_boundary"], dedup, token_cap
        ),
        "refusal_calibration": generate_refusal_calibration_examples(
            targets["refusal_calibration"], dedup, token_cap
        ),
    }
    all_examples = [example for examples in by_category.values() for example in examples]
    counts = {category: len(examples) for category, examples in by_category.items()}
    shortfalls = {
        category: targets[category] - counts[category]
        for category in targets if counts[category] < targets[category]
    }
    source_breakdown: dict[str, int] = {}
    for example in all_examples:
        source_breakdown[example.source] = source_breakdown.get(example.source, 0) + 1

    return {
        "examples": all_examples,
        "report": {
            "targets": targets,
            "counts": counts,
            "total": len(all_examples),
            "shortfalls": shortfalls,
            "by_source": source_breakdown,
        },
    }


def write_dataset(examples, dataset_path: str | Path) -> None:
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_record(), sort_keys=True, ensure_ascii=False) + "\n")


def write_report(report: dict, report_path: str | Path, **extra) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**report, **extra}
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_real_build(*, upstream_root: str | Path, work_dir: str | Path,
                    dataset_path: str | Path, report_path: str | Path,
                    token_cap: int) -> dict:
    """Fetch every real ADR 0001 source and exclusion-pool text, build the
    dataset, and write it plus its report to disk. No manual intervention
    step and no HF_TOKEN: every call below is either a local template or an
    unauthenticated Hugging Face / raw-GitHub fetch.
    """
    from training_data import sources
    from training_data.exclusion_pool import (
        collect_eval_texts, collect_full_pool_texts, fetch_full_pool_texts, fetch_published_eval_rows,
    )

    work_dir = Path(work_dir)
    published_rows = fetch_published_eval_rows(upstream_root, work_dir / "published_eval")
    full_pool_rows = fetch_full_pool_texts(upstream_root)
    exclusion_texts = collect_eval_texts(published_rows) + collect_full_pool_texts(full_pool_rows)
    exact_keys, near_keys = pool_keys(exclusion_texts)

    injection_raw_rows = (
        sources.load_deepset_prompt_injection_attacks() + sources.load_gandalf_ignore_instructions()
    )
    dolly_rows = sources.load_dolly_rows()

    result = build_dataset(
        injection_raw_rows=injection_raw_rows,
        dolly_rows=dolly_rows,
        exclusion_exact_keys=exact_keys,
        exclusion_near_keys=near_keys,
        token_cap=token_cap,
    )
    write_dataset(result["examples"], dataset_path)
    write_report(
        result["report"], report_path,
        exclusion_pool_size=len(exclusion_texts),
        injection_raw_pool_size=len(injection_raw_rows),
        dolly_pool_size=len(dolly_rows),
    )
    return result
