"""Maintainer-only export of visible scores; never reads held-out data files.

Readers use reproduce.py, which has no dependency on the source repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

BENCHMARKS = ("open_prompt_injection", "tensor_trust_hijack", "tensor_trust_extract", "gsm8k", "ifeval", "mmlu")


def export(root: Path, destination: Path) -> None:
    sources = []

    def read(relative):
        path = root / relative
        raw = path.read_bytes()
        sources.append({"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)

    def one(pattern):
        paths = list((root / "runs").glob(pattern))
        if len(paths) != 1:
            raise ValueError(f"Expected one input for {pattern}, found {len(paths)}")
        return paths[0].relative_to(root)

    manifest = read("protocol/manifest.json")
    claim = read("analysis/attempt1-claim-report.json")
    integrity = read("analysis/attempt1-integrity-report.json")
    baseline_path = one("real-baseline-*/metrics.json")
    documents = [("baseline", "", 0, read(baseline_path))]
    selections = []
    seeds = manifest["training"]["seeds"]
    epochs = manifest["training"]["optimizer"]["epochs"]
    for seed in seeds:
        record = read(one(f"selection-seed{seed}*/selection_record.json"))
        selections.append({
            "nominal_seed": seed,
            "selected_epoch": record["selected_epoch"],
            "selected_checkpoint_digest": record["selected_checkpoint_digest"],
            "candidates": [{
                "epoch": row["epoch"], "eligible": row["eligible"],
                "visible_composite": row["visible"]["composite"],
                "mean_normalized_retention": row["capability"]["mean_normalized_retention"],
            } for row in record["candidates"]],
        })
        for epoch in range(1, epochs + 1):
            doc = read(one(f"eval-seed{seed}-epoch{epoch}-*/metrics.json"))
            if (doc["seed"], doc["epoch"]) != (seed, epoch):
                raise ValueError("Run metadata does not match its directory")
            documents.append((f"run{seed}_epoch{epoch}", seed, epoch, doc))

    destination.mkdir(parents=True, exist_ok=True)
    score_path = destination / "visible-scores.csv"
    count = 0
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("state", "nominal_seed", "epoch", "benchmark", "example_id", "score"))
        for state, seed, epoch, doc in documents:
            for name in BENCHMARKS:
                for example_id, outcome in sorted(doc["benchmarks"][name]["items"].items()):
                    if outcome.get("valid") is not True:
                        raise ValueError("Unexpected invalid visible record; do not silently drop it")
                    writer.writerow((state, seed, epoch, name, example_id, outcome["score"]))
                    count += 1
    resources = integrity["integrity_records"]["resource_accounting"]
    reference_rows = [row for group in claim["primary_table"]["modality_groups"].values() for row in group["rows"]]
    intervals = [{key: row[key] for key in (
        "benchmark", "run_seed", "epoch", "n", "observed_difference", "ci_low", "ci_high"
    )} for row in claim["paired_bootstrap"] if row["benchmark"] in BENCHMARKS]
    if len(intervals) != len(seeds) * epochs * len(BENCHMARKS):
        raise ValueError("Expected one paired interval for each trained benchmark mean")
    context = {
        "schema": "direction-a-visible-evidence-v1",
        "prepared": "2026-09-06",
        "scope": "Visible per-item numeric scores only. No prompts, completions, weights, or held-out outcomes. Records are observations, not new model evaluations.",
        "protocol_version": manifest["protocol_version"],
        "model": manifest["model"], "upstream": manifest["upstream"],
        "training": manifest["training"],
        "decoding": manifest["evaluation"]["decoding"],
        "selection": manifest["selection"], "analysis": manifest["analysis"],
        "seeds": seeds, "epochs": epochs,
        "benchmarks": list(BENCHMARKS),
        "sample_counts": {name: len(documents[0][3]["benchmarks"][name]["items"]) for name in BENCHMARKS},
        "score_rows": count,
        "hardware": "NVIDIA RTX 4080, 16 GB",
        "scientific_gpu_accounted_hours": resources["scientific_totals"]["gpu_hours"],
        "gpu_accounted_hour_limit": resources["scientific_totals"]["gpu_hours_budget"],
        "peak_vram_gib": resources["peak_vram"]["value_gb"],
        "declared_vram_gib": resources["peak_vram"]["declared_allocation_gb"],
        "selected_records": selections,
        "recorded_scores": [{key: row[key] for key in (
            "run_seed", "epoch", "benchmark", "baseline", "trained", "n"
        )} for row in reference_rows],
        "recorded_paired_intervals": intervals,
        "source_files": sources,
        "visible_scores_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
    }
    (destination / "context.json").write_text(json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Exported {count} numeric score rows from {len(documents)} model states. No held-out files read.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    export(args.repo_root.resolve(), args.out.resolve())
