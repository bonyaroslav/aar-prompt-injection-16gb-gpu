import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from runner.publication_gates import (
    ProvenanceGateError,
    ClaimLanguageError,
    _canonical_digest,
    build_baseline_resource_supplement,
    build_corpus_supplement,
    build_power_notes_supplement,
    build_provenance_manifest,
    check_claim_language,
    run_gates,
    verify_provenance,
)
from runner.publication_report_inputs import register_current_reports


def _protocol_manifest():
    return {
        "protocol_version": "phase1-2026-08-29",
        "analysis": {
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 271828,
            "interval": "95_percentile_paired_by_fixed_example_id",
        },
    }


def _frozen_record_with_protocol():
    record = _frozen_record()
    record["protocol_manifest_digest"] = _canonical_digest(_protocol_manifest())
    return record


def _frozen_record():
    return {
        "analysis_version": "phase1-analysis-2026-09",
        "protocol_manifest_digest": "a" * 64,
        "inputs": [
            {"role": "protocol_manifest", "digest": "sha256:" + "a" * 64},
            {"role": "baseline_bundle", "digest": "sha256:" + "b" * 64},
        ],
    }


def _report(value=0.25, text=None):
    return {
        "report_id": "claim_tables",
        "sections": [{
            "kind": "table",
            "id": "primary_table",
            "source_roles": ["protocol_manifest", "baseline_bundle"],
            "content": {"score": value, "summary": text or "Measured score."},
        }],
    }


def _corpus_report():
    return {
        "report_id": "integrity_report",
        "sections": [{
            "kind": "table",
            "id": "corpus_nutrition",
            "source_roles": ["training_corpus_digest_only_supplement"],
            "content": {"total_examples": 1},
        }],
    }


class ProvenanceReceiptTests(unittest.TestCase):
    def test_changed_number_is_an_orphan_with_its_location_named(self):
        manifest = build_provenance_manifest(
            frozen_input_record=_frozen_record(), reports=[_report(0.25)],
            supplemental_sources=[],
        )

        with self.assertRaisesRegex(
            ProvenanceGateError,
            r"orphan value 0.75.*claim_tables.*primary_table.*score",
        ):
            verify_provenance(
                provenance_manifest=manifest, frozen_input_record=_frozen_record(),
                reports=[_report(0.75)], supplemental_sources=[],
            )


class ManifestMetadataTests(unittest.TestCase):
    def test_manifest_records_bootstrap_parameters_and_digest_meaning(self):
        manifest = build_provenance_manifest(
            frozen_input_record=_frozen_record_with_protocol(),
            reports=[_report(0.25)], supplemental_sources=[],
            protocol_manifest=_protocol_manifest(),
        )

        self.assertEqual(manifest["bootstrap_parameters"]["replicates"], 10000)
        self.assertEqual(manifest["bootstrap_parameters"]["seed"], 271828)
        self.assertIn("canonical-JSON", manifest["protocol_manifest_digest_meaning"])
        self.assertEqual(manifest["protocol_manifest_digest_kind"], "canonical_json")

    def test_protocol_manifest_digest_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ProvenanceGateError, "canonical digest"):
            build_provenance_manifest(
                frozen_input_record=_frozen_record(),  # digest "a" * 64
                reports=[_report(0.25)], supplemental_sources=[],
                protocol_manifest=_protocol_manifest(),
            )

    def test_clean_reports_pass_both_gates(self):
        frozen = _frozen_record_with_protocol()
        reports = [_report(0.25)]
        manifest = build_provenance_manifest(
            frozen_input_record=frozen, reports=reports, supplemental_sources=[],
            protocol_manifest=_protocol_manifest(),
        )

        returned = run_gates(
            provenance_manifest=manifest, frozen_input_record=frozen,
            reports=reports, supplemental_sources=[],
            protocol_manifest=_protocol_manifest(),
        )

        self.assertEqual(returned, manifest)


class SupplementalSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_corpus(self, directory: Path, dataset_text: str):
        directory.mkdir()
        dataset = directory / "dataset.jsonl"
        report = directory / "report.json"
        dataset.write_text(dataset_text, encoding="utf-8", newline="")
        report.write_text('{"total": 1}', encoding="utf-8")
        return dataset, report

    def test_corpus_supplement_is_invariant_to_json_key_order_and_line_endings(self):
        first = self._write_corpus(self.root / "first", '{"b":2,"a":1}\n')
        second = self._write_corpus(self.root / "second", '{"a":1,"b":2}\r\n')

        self.assertEqual(
            build_corpus_supplement(*first)["digest"],
            build_corpus_supplement(*second)["digest"],
        )

    def test_resource_and_power_note_supplements_are_canonical_content_digests(self):
        resource = self.root / "baseline_resource_comparison.json"
        power_notes = self.root / "power_notes.md"
        resource.write_text('{"b":2,"a":1}', encoding="utf-8")
        power_notes.write_text("line one\r\nline two\r\n", encoding="utf-8", newline="")

        resource_source = build_baseline_resource_supplement(resource)
        power_source = build_power_notes_supplement(power_notes)

        self.assertEqual(resource_source["role"], "baseline_resource_digest_only_supplement")
        self.assertEqual(resource_source["digest_kind"], "canonical_json")
        self.assertEqual(power_source["role"], "power_notes_digest_only_supplement")
        self.assertEqual(power_source["digest_kind"], "lf_normalized_text")

    def test_corpus_nutrition_section_requires_the_named_supplement(self):
        with self.assertRaisesRegex(
            ProvenanceGateError, "training_corpus_digest_only_supplement"
        ):
            build_provenance_manifest(
                frozen_input_record=_frozen_record(), reports=[_corpus_report()],
                supplemental_sources=[],
            )


class ClaimLanguageTests(unittest.TestCase):
    def test_forbidden_terms_each_fail(self):
        for term in ("robust", "secure", "resistant", "mitigation", "defense that works"):
            with self.subTest(term=term):
                with self.assertRaisesRegex(ClaimLanguageError, term):
                    check_claim_language([_report(text=f"The intervention is {term}.")])

    def test_capability_claim_without_modality_fails(self):
        with self.assertRaisesRegex(ClaimLanguageError, "evaluation modality"):
            check_claim_language([_report(text="Capability preserved after training.")])

    def test_capability_claim_with_modality_passes(self):
        check_claim_language([
            _report(text="Capability preserved under likelihood_ranked_no_generation.")
        ])

    def test_capability_gate_reference_without_a_claim_action_is_allowed(self):
        check_claim_language([
            _report(text="Every checkpoint was scored against the frozen capability gates.")
        ])


class CommandLineTests(unittest.TestCase):
    def test_cli_returns_nonzero_for_orphan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen_path = root / "frozen.json"
            reports_path = root / "reports.json"
            manifest_path = root / "provenance.json"
            protocol_path = root / "manifest.json"
            frozen = _frozen_record_with_protocol()
            manifest = build_provenance_manifest(
                frozen_input_record=frozen, reports=[_report(0.25)],
                supplemental_sources=[], protocol_manifest=_protocol_manifest(),
            )
            frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
            reports_path.write_text(json.dumps([_report(0.75)]), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            protocol_path.write_text(json.dumps(_protocol_manifest()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable, "-m", "runner.publication_gates",
                    "--frozen-input", str(frozen_path),
                    "--provenance", str(manifest_path),
                    "--reports", str(reports_path),
                    "--protocol-manifest", str(protocol_path),
                ],
                capture_output=True, text=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("orphan value", completed.stderr)


class CurrentReportRegistrationTests(unittest.TestCase):
    def test_integrity_sections_bind_their_specific_supplemental_sources(self):
        claim = {
            "primary_table": {"rows": 1},
            "paired_bootstrap": [], "mcnemar_exact": [],
            "visible_composite": [], "cross_run_summary": {},
        }
        integrity = {
            "failure_mode_evidence": {
                "generation_failure_signature": {}, "tensor_trust_degeneracy": {},
                "utility_control_arm_comparison": {},
                "corpus_nutrition_label": {"total_examples": 1},
            },
            "integrity_records": {
                "held_out_disposition": {"valid_only_mde80_pp": 10.8},
                "reproducibility_disclosure": {},
                "resource_accounting": {"gpu_hours": 6.25},
                "sample_count_convention": {},
            },
        }

        reports = register_current_reports(claim_report=claim, integrity_report=integrity)
        sections = {
            (report["report_id"], section["id"]): section
            for report in reports for section in report["sections"]
        }

        self.assertIn(
            "training_corpus_digest_only_supplement",
            sections[("integrity_report", "corpus_nutrition_label")]["source_roles"],
        )
        self.assertIn(
            "baseline_resource_digest_only_supplement",
            sections[("integrity_report", "resource_accounting")]["source_roles"],
        )
        self.assertIn(
            "power_notes_digest_only_supplement",
            sections[("integrity_report", "held_out_disposition")]["source_roles"],
        )


if __name__ == "__main__":
    unittest.main()
