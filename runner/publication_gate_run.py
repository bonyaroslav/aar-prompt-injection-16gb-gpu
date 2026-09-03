"""Operator glue: run the issue #32 publication gates over the current outputs.

Assembles the real (gitignored) evidence tree into the #28 claim report and the
#29 integrity report, registers their publication sections
(:mod:`runner.publication_report_inputs`), builds the provenance manifest and
runs both gates (:mod:`runner.publication_gates`). CPU-only: it reads finalized
``metrics.json`` / ``execution.log`` / ``gpu.csv`` / resource-comparison JSON and
the built training corpus. No model, scorer, trainer, GPU or held-out access.

    python -m runner.publication_gate_run \
        --evidence-root runs --recovery-root recovery \
        --out analysis/publication-provenance-manifest.json

Exits non-zero (via the gate error) when a number is an orphan or a report uses
unsupported claim language.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from runner.claim_tables import build_claim_report
from runner.frozen_inputs import freeze_inputs
from runner.integrity_report import build_integrity_report
from runner.publication_gates import (
    build_baseline_resource_supplement,
    build_corpus_supplement,
    build_power_notes_supplement,
    build_provenance_manifest,
    run_gates,
)
from runner.publication_report_inputs import register_current_reports

REPO_ROOT = Path(__file__).resolve().parents[1]


class EvidenceAssemblyError(RuntimeError):
    """The real evidence tree is missing an artifact this run needs."""


def _one(root: Path, pattern: str, *, reject=()) -> Path:
    matches = [
        child for child in sorted(root.glob(pattern))
        if child.is_dir() and not any(token in child.name for token in reject)
    ]
    if not matches:
        raise EvidenceAssemblyError(f"no directory matches {pattern} under {root}")
    if len(matches) > 1:
        raise EvidenceAssemblyError(f"ambiguous {pattern} under {root}: {[m.name for m in matches]}")
    return matches[0]


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _peak_vram_gb(gpu_csv: Path) -> float:
    with Path(gpu_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return max(float(row["vram_mb"]) for row in rows) / 1024.0


def assemble_evidence(*, evidence_root: Path, recovery_root: Path, repo_root: Path,
                      corpus_root: Path) -> tuple[dict, dict, dict, Path]:
    """Return ``(claim_report, integrity_report, frozen_record, manifest_path)``."""
    manifest_path = repo_root / "protocol" / "manifest.json"
    frozen_record = freeze_inputs(
        manifest_path=manifest_path, evidence_root=evidence_root,
        repo_root=repo_root, recovery_root=recovery_root,
    )
    epochs = frozen_record["epochs_per_seed"]
    seeds = frozen_record["completed_seeds"]

    baseline_dir = _one(evidence_root, "real-baseline-*", reject=("comparison", "data"))
    baseline = {
        "metrics": _json(baseline_dir / "metrics.json"),
        "execution_log": _text(baseline_dir / "execution.log"),
    }

    checkpoints: list[dict] = []
    phase_vram_peaks_gb: dict[int, dict] = {}
    for seed in seeds:
        eval_peaks: list[float] = []
        for epoch in range(1, epochs + 1):
            eval_dir = _one(evidence_root, f"eval-seed{seed}-epoch{epoch}-*")
            checkpoints.append({
                "seed": seed,
                "epoch": epoch,
                "metrics": _json(eval_dir / "metrics.json"),
                "execution_log": _text(eval_dir / "execution.log"),
            })
            eval_peaks.append(_peak_vram_gb(eval_dir / "gpu.csv"))
        training_dir = _one(evidence_root, f"training-seed{seed}-*", reject=("epoch",))
        phase_vram_peaks_gb[seed] = {
            "training": _peak_vram_gb(training_dir / "gpu.csv"),
            "evaluation": max(eval_peaks),
        }

    selection_records = [
        _json(_one(evidence_root, f"selection-seed{seed}*") / "selection_record.json")
        for seed in seeds
    ]

    resource_comparisons: dict = {
        "baseline": _json(
            _one(evidence_root, "real-baseline-comparison-*") / "baseline_resource_comparison.json"
        ),
    }
    for seed in seeds:
        resource_comparisons[seed] = _json(
            _one(evidence_root, f"seed{seed}-resource-comparison*") / "seed_resource_comparison.json"
        )

    corpus = {
        "rows": [
            json.loads(line)
            for line in (corpus_root / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ],
        "report": _json(corpus_root / "report.json"),
    }

    claim_report = build_claim_report(
        manifest_path,
        baseline_metrics=baseline["metrics"],
        epoch_metrics=[cp["metrics"] for cp in checkpoints],
    )
    integrity_report = build_integrity_report(
        manifest_path,
        evidence={
            "frozen_input_record": frozen_record,
            "baseline": baseline,
            "checkpoints": checkpoints,
            "selection_records": selection_records,
            "resource_comparisons": resource_comparisons,
            "phase_vram_peaks_gb": phase_vram_peaks_gb,
            "non_scientific_runs": [],  # #33 carries the full smoke/recovery ledger
            "corpus": corpus,
        },
    )
    return claim_report, integrity_report, frozen_record, manifest_path


def run(*, evidence_root: Path, recovery_root: Path, repo_root: Path,
        corpus_root: Path, power_notes: Path) -> dict:
    claim_report, integrity_report, frozen_record, manifest_path = assemble_evidence(
        evidence_root=evidence_root, recovery_root=recovery_root,
        repo_root=repo_root, corpus_root=corpus_root,
    )
    reports = register_current_reports(
        claim_report=claim_report, integrity_report=integrity_report
    )
    supplemental_sources = [
        build_corpus_supplement(corpus_root / "dataset.jsonl", corpus_root / "report.json"),
        build_baseline_resource_supplement(
            _one(evidence_root, "real-baseline-comparison-*") / "baseline_resource_comparison.json"
        ),
        build_power_notes_supplement(power_notes),
    ]
    protocol_manifest = _json(manifest_path)
    provenance_manifest = build_provenance_manifest(
        frozen_input_record=frozen_record, reports=reports,
        supplemental_sources=supplemental_sources, protocol_manifest=protocol_manifest,
    )
    run_gates(
        provenance_manifest=provenance_manifest, frozen_input_record=frozen_record,
        reports=reports, supplemental_sources=supplemental_sources,
        protocol_manifest=protocol_manifest,
    )
    return provenance_manifest


def _summary(manifest: dict) -> str:
    sections = sum(len(report["sections"]) for report in manifest["reports"])
    numbers = sum(
        len(section["numbers"])
        for report in manifest["reports"] for section in report["sections"]
    )
    return (
        f"reports={len(manifest['reports'])} sections={sections} "
        f"receipted_numbers={numbers} orphans=0 claim_language_violations=0"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--evidence-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--recovery-root", type=Path, default=REPO_ROOT / "recovery")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--corpus-root", type=Path, default=REPO_ROOT / "data" / "training")
    parser.add_argument("--power-notes", type=Path, default=REPO_ROOT / "protocol" / "power_notes.md")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    try:
        manifest = run(
            evidence_root=args.evidence_root, recovery_root=args.recovery_root,
            repo_root=args.repo_root, corpus_root=args.corpus_root,
            power_notes=args.power_notes,
        )
    except (OSError, EvidenceAssemblyError) as error:
        print(f"publication gate run failed to assemble evidence: {error}", file=sys.stderr)
        return 2

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
    print(_summary(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
