import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import runner.recovery as recovery
from runner.recovery import RecoveryWorkspace, StageSignature


class StageSignatureTests(unittest.TestCase):
    def _signature(self, **changes):
        values = {
            "manifest_digest": "sha256:manifest", "protocol_version": "phase1-2026-08-29",
            "upstream_commit": "a" * 40, "upstream_tree": "b" * 40,
            "model_revision": "c" * 40, "seed": 17, "stage": "evaluation",
            "epoch": 1, "checkpoint_digest": "sha256:checkpoint",
            "effective_evaluation_config": {"batch_size": 32},
            "expected_example_ids": ["visible:0001", "visible:0002"],
        }
        values.update(changes)
        return StageSignature.create(**values)

    def test_equal_inputs_produce_equal_canonical_digest(self):
        self.assertEqual(self._signature().digest, self._signature().digest)

    def test_first_difference_names_changed_signature_field(self):
        self.assertEqual(self._signature().first_difference(self._signature(seed=42)), "seed")

    def test_signature_snapshot_isolated_from_nested_input_and_payload_mutation(self):
        config = {"batch_size": 32, "scoring": {"threshold": 0.5}}
        example_ids = ["visible:0001", "visible:0002"]
        signature = self._signature(
            effective_evaluation_config=config,
            expected_example_ids=example_ids,
        )
        expected = self._signature(
            effective_evaluation_config={"batch_size": 32, "scoring": {"threshold": 0.5}},
            expected_example_ids=["visible:0001", "visible:0002"],
        )

        config["scoring"]["threshold"] = 0.9
        example_ids.append("visible:0003")
        payload = signature.payload
        payload["seed"] = 42
        payload["effective_evaluation_config"]["scoring"]["threshold"] = 0.1

        self.assertEqual(signature.digest, expected.digest)
        self.assertEqual(signature.payload, expected.payload)
        self.assertIsNone(signature.first_difference(expected))


class RecoveryWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        temporary_root = Path(self.temporary_directory.name)
        self.recovery_root = temporary_root / "recovery"
        self.evidence_root = temporary_root / "evidence"
        self.signature = self.signature_for()

    def signature_for(self, **changes):
        values = {
            "manifest_digest": "sha256:manifest", "protocol_version": "phase1-2026-08-29",
            "upstream_commit": "a" * 40, "upstream_tree": "b" * 40,
            "model_revision": "c" * 40, "seed": 17, "stage": "evaluation",
            "epoch": 1, "checkpoint_digest": "sha256:checkpoint",
            "effective_evaluation_config": {"batch_size": 32},
            "expected_example_ids": ["visible:0001", "visible:0002"],
        }
        values.update(changes)
        return StageSignature.create(**values)

    def test_compatible_safe_boundary_is_recoverable(self):
        workspace = RecoveryWorkspace(self.recovery_root, self.evidence_root)

        workspace.write_state(
            "attempt-1", self.signature, status="interrupted", recovery_reference="epoch-1"
        )
        inspection = workspace.inspect_stage("attempt-1", self.signature)

        self.assertEqual((inspection.status, inspection.action), ("recoverable", "resume-from:epoch-1"))

    def test_rejects_traversal_attempt_id_without_writing_evidence(self):
        workspace = RecoveryWorkspace(self.recovery_root, self.evidence_root)
        evidence_state = self.evidence_root / "attempt.json"

        with self.assertRaisesRegex(ValueError, "attempt ID"):
            workspace.write_state("../evidence/attempt", self.signature, status="running")

        self.assertFalse(evidence_state.exists())

    def test_replacing_an_attempt_state_exposes_a_complete_valid_document(self):
        workspace = RecoveryWorkspace(self.recovery_root, self.evidence_root)
        workspace.write_state(
            "attempt-1", self.signature, status="completed", completed_bundle="old-bundle"
        )

        replacement_path = self.recovery_root / "attempt-1.json"
        real_replace = recovery.os.replace
        replacements = []

        def capture_replace(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            self.assertTrue(source_path.exists())
            self.assertEqual(destination_path, replacement_path)
            self.assertEqual(source_path.parent, replacement_path.parent)
            self.assertTrue(source_path.name.startswith(".attempt-1.json."))
            self.assertTrue(source_path.name.endswith(".tmp"))
            replacements.append((source_path, destination_path))
            return real_replace(source, destination)

        with patch("runner.recovery.os.replace", side_effect=capture_replace):
            path = workspace.write_state(
                "attempt-1", self.signature, status="interrupted", recovery_reference="epoch-2"
            )
        record = json.loads(path.read_text(encoding="utf-8"))
        inspection = workspace.inspect_stage("attempt-1", self.signature)

        self.assertEqual(len(replacements), 1)
        self.assertEqual(
            record,
            {
                "completed_bundle": None,
                "recovery_reference": "epoch-2",
                "signature": self.signature.payload,
                "signature_digest": self.signature.digest,
                "status": "interrupted",
            },
        )
        self.assertEqual((inspection.status, inspection.action), ("recoverable", "resume-from:epoch-2"))

    def test_mismatched_signature_preserves_original_state(self):
        workspace = RecoveryWorkspace(self.recovery_root, self.evidence_root)
        path = workspace.write_state(
            "attempt-1", self.signature, status="interrupted", recovery_reference="epoch-1"
        )

        inspection = workspace.inspect_stage("attempt-1", self.signature_for(seed=42))

        self.assertEqual(inspection.status, "incompatible")
        self.assertEqual(inspection.differing_field, "seed")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["signature_digest"], self.signature.digest)

    def test_tampered_stored_digest_is_an_incompatible_diagnostic(self):
        workspace = RecoveryWorkspace(self.recovery_root, self.evidence_root)
        path = workspace.write_state(
            "attempt-1", self.signature, status="completed", completed_bundle="not-a-bundle"
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        record["signature_digest"] = "sha256:forged"
        path.write_text(json.dumps(record), encoding="utf-8")

        inspection = workspace.inspect_stage("attempt-1", self.signature)

        self.assertEqual(
            (inspection.status, inspection.action, inspection.differing_field),
            ("incompatible", "diagnose", "signature_digest"),
        )

    def test_malformed_state_is_a_hard_loss_diagnostic(self):
        workspace = RecoveryWorkspace(self.recovery_root, self.evidence_root)
        path = workspace.write_state("attempt-1", self.signature, status="running")
        record = json.loads(path.read_text(encoding="utf-8"))
        del record["status"]
        path.write_text(json.dumps(record), encoding="utf-8")

        inspection = workspace.inspect_stage("attempt-1", self.signature)

        self.assertEqual(
            (inspection.status, inspection.action),
            ("unavailable-after-hard-loss", "record-hard-loss"),
        )

    def test_rejects_recovery_root_inside_finalized_evidence_root(self):
        with self.assertRaisesRegex(ValueError, "outside evidence root"):
            RecoveryWorkspace(self.evidence_root / "recovery", self.evidence_root)


class AttemptLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        temporary_root = Path(self.temporary_directory.name)
        self.recovery_root = temporary_root / "recovery"
        self.signature = StageSignature.create(
            manifest_digest="sha256:manifest",
            protocol_version="phase1-2026-08-29",
            upstream_commit="a" * 40,
            upstream_tree="b" * 40,
            model_revision="c" * 40,
            seed=17,
            stage="evaluation",
            epoch=1,
            checkpoint_digest="sha256:checkpoint",
            effective_evaluation_config={"batch_size": 32},
            expected_example_ids=["visible:0001", "visible:0002"],
        )

    def test_ledger_preserves_unavailable_gpu_time_and_attempt_identity(self):
        ledger = recovery.AttemptLedger(self.recovery_root / "attempts.jsonl")

        ledger.append(
            "attempt-1",
            self.signature,
            status="interrupted",
            started_at="2026-08-31T10:00:00Z",
            ended_at="2026-08-31T10:05:00Z",
            wall_seconds=300.0,
            gpu_hours=None,
            state_reference="states/attempt-1.json",
        )

        row = json.loads((self.recovery_root / "attempts.jsonl").read_text().strip())
        self.assertEqual((row["attempt_id"], row["gpu_hours"]), ("attempt-1", "unavailable"))

    def test_ledger_rejects_duplicate_attempt_identity(self):
        ledger = recovery.AttemptLedger(self.recovery_root / "attempts.jsonl")
        kwargs = {
            "status": "running",
            "started_at": "2026-08-31T10:00:00Z",
            "ended_at": None,
            "wall_seconds": 0.0,
            "gpu_hours": None,
            "state_reference": "states/attempt-1.json",
        }
        ledger.append("attempt-1", self.signature, **kwargs)

        with self.assertRaisesRegex(ValueError, "attempt identity already recorded"):
            ledger.append("attempt-1", self.signature, **kwargs)


class CompletedInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        temporary_root = Path(self.temporary_directory.name)
        self.recovery_root = temporary_root / "recovery"
        self.evidence_root = temporary_root / "evidence"
        self.signature = StageSignature.create(
            manifest_digest="sha256:manifest",
            protocol_version="phase1-2026-08-29",
            upstream_commit="a" * 40,
            upstream_tree="b" * 40,
            model_revision="c" * 40,
            seed=17,
            stage="evaluation",
            epoch=1,
            checkpoint_digest="sha256:checkpoint",
            effective_evaluation_config={"batch_size": 32},
            expected_example_ids=["visible:0001", "visible:0002"],
        )

    def test_completed_state_with_invalid_bundle_is_not_completed(self):
        workspace = RecoveryWorkspace(self.recovery_root, self.evidence_root)
        workspace.write_state(
            "attempt-1",
            self.signature,
            status="completed",
            completed_bundle=self.evidence_root / "bad",
        )

        self.assertEqual(
            workspace.inspect_stage("attempt-1", self.signature).status,
            "unavailable-after-hard-loss",
        )


if __name__ == "__main__":
    unittest.main()
