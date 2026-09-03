import unittest
from pathlib import Path

from runner.corpus_ablation import (
    CorpusValidationError,
    _repo_relative,
    append_attempt,
    resource_row,
    validate_corpus,
)


class CorpusValidationTests(unittest.TestCase):
    def test_accepts_complete_clean_only_report(self):
        report = {
            "targets": {"prompt_injection": 0, "clean_control": 3500,
                        "ambiguous_boundary": 1000, "refusal_calibration": 500},
            "counts": {"prompt_injection": 0, "clean_control": 3500,
                       "ambiguous_boundary": 1000, "refusal_calibration": 500},
            "total": 5000,
            "shortfalls": {},
        }
        examples = [{"category": "clean_control"}, {"category": "ambiguous_boundary"}]

        validate_corpus(report, examples)

    def test_rejects_shortfall_or_prompt_injection_record(self):
        report = {
            "targets": {"prompt_injection": 0}, "counts": {"prompt_injection": 0},
            "total": 4999, "shortfalls": {"clean_control": 1},
        }
        with self.assertRaisesRegex(CorpusValidationError, "shortfall"):
            validate_corpus(report, [])
        with self.assertRaisesRegex(CorpusValidationError, "prompt-injection"):
            validate_corpus({**report, "total": 5000, "shortfalls": {}}, [{"category": "prompt_injection"}])


class AblationEvidenceShapeTests(unittest.TestCase):
    def test_repo_relative_accepts_a_relative_ablation_output_path(self):
        self.assertEqual(
            _repo_relative(Path("ablation/issue-31/attempts.json")),
            "ablation/issue-31/attempts.json",
        )

    def test_attempt_row_preserves_mid_epoch_recovery_evidence(self):
        rows = append_attempt(
            rows=[], epoch=1, status="completed",
            recovery_evidence={"mid_epoch_resume_fired": True, "save_measurements": [{"step_index": 120}]},
        )

        self.assertEqual(rows[0]["epoch"], 1)
        self.assertTrue(rows[0]["recovery_evidence"]["mid_epoch_resume_fired"])
        self.assertEqual(rows[0]["recovery_evidence"]["save_measurements"][0]["step_index"], 120)

    def test_resource_row_is_separate_from_scientific_attempt_one_totals(self):
        row = resource_row(gpu_hours=12.74, wall_hours=12.74, source="ablation/example/attempts.jsonl")

        self.assertEqual(row["category"], "ablation")
        self.assertEqual(row["label"], "corpus-ablation")
        self.assertEqual(row["source"], "ablation/example/attempts.jsonl")


if __name__ == "__main__":
    unittest.main()
