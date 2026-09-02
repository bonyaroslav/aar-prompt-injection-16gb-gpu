import json
import tempfile
import unittest
from pathlib import Path

from runner.diagnostic_chatmode_mmlu import (
    CheckpointIntegrityError,
    build_bundle_contents,
    resource_row,
    verify_model_states,
)
from runner.bundle import BUNDLE_FILES
from runner.training import _directory_digest


def _write_merged_dir(path: Path, weight_bytes: bytes = b"weights") -> str:
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type": "qwen"}', encoding="utf-8")
    (path / "model.safetensors").write_bytes(weight_bytes)
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return _directory_digest(path)


class VerifyModelStatesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "recovery").mkdir()

    def _manifest(self, states):
        return {"model_states": states}

    def _checkpoint_state(self, name, seed, epoch, digest, *, source, recovery_ref=None):
        state = {
            "state": name, "seed": seed, "epoch": epoch,
            "merged_dir": f"runs/{name}/checkpoints/epoch-{epoch}",
            "expected_integrity_digest": digest, "integrity_source": source,
        }
        if recovery_ref:
            state["recovery_reference"] = recovery_ref
        return state

    def _write_recovery(self, ref, merged_dir, integrity):
        (self.root / ref).parent.mkdir(parents=True, exist_ok=True)
        (self.root / ref).write_text(json.dumps({
            "recovery_reference": json.dumps({"merged_dir": merged_dir, "integrity": integrity}),
        }), encoding="utf-8")

    def test_baseline_row_passes_through_and_checkpoints_are_digest_checked(self):
        merged = self.root / "runs/cp/checkpoints/epoch-1"
        digest = _write_merged_dir(merged)
        manifest = self._manifest([
            {"state": "baseline", "integrity_source": "huggingface_revision_pin"},
            self._checkpoint_state("cp", 17, 1, digest, source="no_runtime_digest_pre_issue_22"),
        ])

        verified = verify_model_states(manifest, repo_root=self.root)

        self.assertEqual(verified[0]["state"], "baseline")
        self.assertTrue(verified[0]["verified"])
        self.assertEqual(verified[1]["recomputed_integrity_digest"], digest)
        self.assertIsNone(verified[1]["recovery_cross_check"])

    def test_recovery_reference_is_cross_checked_for_seeds_42_and_2026(self):
        merged_rel = "runs/cp42/checkpoints/epoch-2"
        digest = _write_merged_dir(self.root / merged_rel)
        self._write_recovery("recovery/training-seed42-seq2048-epoch2.json", merged_rel, digest)
        manifest = self._manifest([
            self._checkpoint_state("cp42", 42, 2, digest, source="recovery_reference",
                                   recovery_ref="recovery/training-seed42-seq2048-epoch2.json"),
        ])

        verified = verify_model_states(manifest, repo_root=self.root)

        self.assertTrue(verified[0]["recovery_cross_check"])

    def test_digest_mismatch_stops_without_retraining(self):
        merged = self.root / "runs/cp/checkpoints/epoch-1"
        _write_merged_dir(merged)
        manifest = self._manifest([
            self._checkpoint_state("cp", 17, 1, "sha256:" + "0" * 64,
                                   source="no_runtime_digest_pre_issue_22"),
        ])

        with self.assertRaisesRegex(CheckpointIntegrityError, "digest mismatch"):
            verify_model_states(manifest, repo_root=self.root)

    def test_missing_directory_is_reported(self):
        manifest = self._manifest([
            self._checkpoint_state("cp", 17, 1, "sha256:x", source="no_runtime_digest_pre_issue_22"),
        ])
        with self.assertRaisesRegex(CheckpointIntegrityError, "missing"):
            verify_model_states(manifest, repo_root=self.root)

    def test_recovery_disagreement_is_reported(self):
        merged_rel = "runs/cp2026/checkpoints/epoch-1"
        digest = _write_merged_dir(self.root / merged_rel)
        self._write_recovery("recovery/training-seed2026-seq2048-epoch1.json", merged_rel,
                             "sha256:" + "9" * 64)
        manifest = self._manifest([
            self._checkpoint_state("cp2026", 2026, 1, digest, source="recovery_reference",
                                   recovery_ref="recovery/training-seed2026-seq2048-epoch1.json"),
        ])
        with self.assertRaisesRegex(CheckpointIntegrityError, "disagrees"):
            verify_model_states(manifest, repo_root=self.root)


class BundleContentTests(unittest.TestCase):
    DIAG = {
        "diagnostic_version": "diag-chatmode-mmlu-2026-09-02",
        "downstream_of": {
            "protocol_version": "phase1-2026-08-29",
            "canonical_manifest_digest": "399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20",
        },
        "model": {"revision": "15852e8c16360a2fea060d615a32b45270f8a8fc"},
        "change": {"scorer": "first_token_logit", "sample_ids": "capability_publisher_seed_42_first_300",
                   "metric": "exact_match_choice", "max_new_tokens": 1,
                   "candidate_strings": [" A", " B", " C", " D"]},
        "analysis": {"bootstrap_seed": 303030, "bootstrap_replicates": 10000},
        "boundaries": ["outputs under diagnostics/"],
    }

    def test_produces_all_bundle_files_with_diagnostic_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "m.json"
            manifest_path.write_text(json.dumps(self.DIAG), encoding="utf-8")
            states = [
                {"state": "baseline", "seed": None, "epoch": None},
                {"state": "seed17-epoch1", "seed": 17, "epoch": 1,
                 "integrity_source": "no_runtime_digest_pre_issue_22",
                 "recomputed_integrity_digest": "sha256:abc"},
            ]
            scores = {
                "baseline": {"items": {"mmlu-0": {"score": 1.0, "valid": True}},
                             "aggregate": {"metric": "exact_match_choice", "value": 1.0}},
                "seed17-epoch1": {"items": {"mmlu-0": {"score": 0.0, "valid": True}},
                                  "aggregate": {"metric": "exact_match_choice", "value": 0.0}},
            }
            contents = build_bundle_contents(
                diagnostic_manifest=self.DIAG, diagnostic_manifest_path=manifest_path,
                model_states=states, per_state_scores=scores,
                candidate_strings=[" A", " B", " C", " D"],
                telemetry_rows=[{"t": 0.0, "vram_mb": 100, "util_pct": 5}],
                environment_text="python=3.12", log_lines=["start", "done"],
                command_text="python -m runner.diagnostic_chatmode_mmlu",
            )

        self.assertEqual(set(contents), set(BUNDLE_FILES))
        self.assertIn("NOT Attempt-1 evidence", contents["notes.md"])
        metrics = json.loads(contents["metrics.json"])
        self.assertTrue(metrics["mmlu_use_chat_template"])
        self.assertEqual(set(metrics["model_states"]), {"baseline", "seed17-epoch1"})
        manifest_record = json.loads(contents["manifest.yaml"])
        self.assertEqual(manifest_record["run_stage"], "diagnostic_chatmode_mmlu")
        self.assertTrue(manifest_record["mmlu_use_chat_template"])
        self.assertEqual(contents["environment.txt"], "python=3.12\n")

    def test_resource_row_is_a_non_scientific_run_shape(self):
        row = resource_row(gpu_hours=1.7, wall_hours=1.7, source="chatmode-mmlu-x/execution.log")
        self.assertEqual(set(row), {"category", "label", "gpu_hours", "wall_hours", "source"})
        self.assertEqual(row["category"], "diagnostic")


if __name__ == "__main__":
    unittest.main()
