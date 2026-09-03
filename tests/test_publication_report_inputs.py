"""Issue #32: the read-only registration adapter for the #28/#29 reports.

Offline: hand-shaped report dicts and a synthetic frozen input record. No model,
dataset, scorer, trainer, telemetry, storage, GPU or real evidence tree.
"""
from __future__ import annotations

import unittest

from runner.publication_gates import (
    ProvenanceGateError,
    build_provenance_manifest,
    run_gates,
)
from runner.publication_report_inputs import register_current_reports

SEEDS = [17, 42, 2026]
EPOCHS = 3


def _frozen_record():
    inputs = [
        {"role": "protocol_manifest", "digest": "sha256:" + "a" * 64},
        {"role": "baseline_bundle", "digest": "sha256:" + "b" * 64},
    ]
    for seed in SEEDS:
        for epoch in range(1, EPOCHS + 1):
            inputs.append({
                "role": f"seed{seed}_evaluation_bundle_epoch{epoch}",
                "digest": f"sha256:{seed:04d}{epoch}" + "c" * 55,
            })
        for suffix in ("training_bundle", "selection_record", "resource_comparison",
                       "outcomes_summary", "continuation_decision"):
            inputs.append({
                "role": f"seed{seed}_{suffix}",
                "digest": f"sha256:{seed:04d}" + suffix[:1] + "d" * 55,
            })
    return {
        "analysis_version": "phase1-analysis-2026-09",
        "protocol_manifest_digest": "a" * 64,
        "inputs": inputs,
    }


def _claim_report():
    return {
        "completed_seeds": SEEDS,
        "epochs_per_seed": EPOCHS,
        "primary_table": {"caption": "measured deltas", "rows": [{"delta": -0.23}]},
        "paired_bootstrap": [{"ci_low": -0.4, "ci_high": -0.1}],
        "mcnemar_exact": [{"p_value": 0.03}],
        "visible_composite": [{"composite_absolute_delta": 0.12}],
        "cross_run_summary": {"epoch_1": {"gsm8k": {"mean": 0.31}}},
    }


def _integrity_report():
    return {
        "completed_seeds": SEEDS,
        "epochs_per_seed": EPOCHS,
        "failure_mode_evidence": {
            "generation_failure_signature": {"recorded": True, "truncations": 34},
            "tensor_trust_degeneracy": {"verdict": "refusal_degeneracy", "moved": 12},
            "utility_control_arm_comparison": {"delta": -0.2},
            "corpus_nutrition_label": {"total_examples": 5000, "multi_step_share": 0.18},
        },
        "integrity_records": {
            "held_out_disposition": {"valid_only_mde80_pp": 10.8},
            "reproducibility_disclosure": {"published_as": "a first-class section"},
            "resource_accounting": {"scientific_totals": {"gpu_hours": 47.34}},
            "sample_count_convention": {"mmlu": 300},
        },
    }


class RegisterCurrentReportsTests(unittest.TestCase):
    def test_two_reports_with_expected_section_ids(self):
        reports = register_current_reports(
            claim_report=_claim_report(), integrity_report=_integrity_report()
        )

        by_id = {report["report_id"]: report for report in reports}
        self.assertEqual(set(by_id), {"claim_tables", "integrity_report"})
        self.assertEqual(
            [section["id"] for section in by_id["claim_tables"]["sections"]],
            ["primary_table", "paired_bootstrap", "mcnemar_exact",
             "visible_composite", "cross_run_summary"],
        )
        self.assertIn(
            "corpus_nutrition_label",
            [section["id"] for section in by_id["integrity_report"]["sections"]],
        )
        for report in reports:
            for section in report["sections"]:
                self.assertIn(section["kind"], {"table", "figure"})

    def test_every_cited_role_resolves_against_the_frozen_record(self):
        reports = register_current_reports(
            claim_report=_claim_report(), integrity_report=_integrity_report()
        )
        supplements = [
            {"role": "training_corpus_digest_only_supplement", "digest": "sha256:" + "1" * 64},
            {"role": "baseline_resource_digest_only_supplement", "digest": "sha256:" + "2" * 64},
            {"role": "power_notes_digest_only_supplement", "digest": "sha256:" + "3" * 64},
        ]

        manifest = build_provenance_manifest(
            frozen_input_record=_frozen_record(), reports=reports,
            supplemental_sources=supplements,
        )
        run_gates(
            provenance_manifest=manifest, frozen_input_record=_frozen_record(),
            reports=reports, supplemental_sources=supplements,
        )

        corpus_section = next(
            section
            for report in manifest["reports"] if report["report_id"] == "integrity_report"
            for section in report["sections"] if section["id"] == "corpus_nutrition_label"
        )
        self.assertEqual(len(corpus_section["input_digests"]), 2)

    def test_missing_supplement_makes_the_bound_section_fail_closed(self):
        reports = register_current_reports(
            claim_report=_claim_report(), integrity_report=_integrity_report()
        )
        with self.assertRaisesRegex(ProvenanceGateError, "power_notes_digest_only_supplement"):
            build_provenance_manifest(
                frozen_input_record=_frozen_record(), reports=reports,
                supplemental_sources=[
                    {"role": "training_corpus_digest_only_supplement",
                     "digest": "sha256:" + "1" * 64},
                    {"role": "baseline_resource_digest_only_supplement",
                     "digest": "sha256:" + "2" * 64},
                ],
            )


if __name__ == "__main__":
    unittest.main()
