import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import runner.recovery as recovery
from runner.bundle import finalize_bundle, write_bundle
from runner.recovery import RecoveryWorkspace, StageSignature, finalized_inputs_only


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

    def test_independent_ledgers_claim_one_concurrent_attempt_identity(self):
        ledger_path = self.recovery_root / "attempts.jsonl"
        ledgers = [recovery.AttemptLedger(ledger_path), recovery.AttemptLedger(ledger_path)]
        scan_barrier = threading.Barrier(2)
        start_barrier = threading.Barrier(2)
        outcomes = []
        outcome_lock = threading.Lock()
        original_rows = recovery.AttemptLedger.rows

        def synchronize_initial_scans(ledger):
            rows = original_rows(ledger)
            scan_barrier.wait(timeout=5)
            return rows

        def append_from(ledger):
            start_barrier.wait(timeout=5)
            try:
                ledger.append(
                    "attempt-1",
                    self.signature,
                    status="running",
                    started_at="2026-08-31T10:00:00Z",
                    ended_at=None,
                    wall_seconds=0.0,
                    gpu_hours=None,
                    state_reference="states/attempt-1.json",
                )
            except ValueError as error:
                outcome = error
            else:
                outcome = None
            with outcome_lock:
                outcomes.append(outcome)

        with patch.object(recovery.AttemptLedger, "rows", synchronize_initial_scans):
            threads = [threading.Thread(target=append_from, args=(ledger,)) for ledger in ledgers]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(sum(outcome is None for outcome in outcomes), 1)
        self.assertEqual(sum(isinstance(outcome, ValueError) for outcome in outcomes), 1)
        self.assertEqual(
            [row["attempt_id"] for row in recovery.AttemptLedger(ledger_path).rows()],
            ["attempt-1"],
        )

    def test_failed_append_releases_claim_for_retry(self):
        ledger = recovery.AttemptLedger(self.recovery_root / "attempts.jsonl")
        kwargs = {
            "status": "running",
            "started_at": "2026-08-31T10:00:00Z",
            "ended_at": None,
            "wall_seconds": 0.0,
            "gpu_hours": None,
            "state_reference": "states/attempt-1.json",
        }

        with patch("runner.recovery.Path.open", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(OSError, "disk unavailable"):
                ledger.append("attempt-1", self.signature, **kwargs)

        ledger.append("attempt-1", self.signature, **kwargs)
        self.assertEqual([row["attempt_id"] for row in ledger.rows()], ["attempt-1"])


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


class FinalizedInputTests(unittest.TestCase):
    STAGES = ("training", "eval-1", "eval-2", "eval-3", "selection", "reveal", "resources")

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.recovery_root = self.temporary_root / "recovery"
        self.evidence_root = self.temporary_root / "evidence"

    def _finalized_bundle(self, bundle_dir):
        write_bundle(
            bundle_dir,
            {
                "manifest.yaml": "manifest: fixture\n",
                "command.sh": "#!/usr/bin/env bash\n",
                "config.yaml": "seed: 17\n",
                "environment.txt": "python: fixture\n",
                "metrics.json": "{}\n",
                "execution.log": "completed\n",
                "gpu.csv": "timestamp,power\n",
                "notes.md": "fixture\n",
            },
        )
        finalize_bundle(bundle_dir)
        return bundle_dir

    def _complete_seed_fixture(self, name, interrupted_after=None):
        bundle_root = self.evidence_root / name
        for stage in self.STAGES:
            self._finalized_bundle(bundle_root / stage)
            if stage == interrupted_after:
                recovery_root = self.recovery_root / name
                recovery_root.mkdir(parents=True, exist_ok=True)
                (recovery_root / f"{stage}.json").write_text("resumed\n", encoding="utf-8")
        return bundle_root

    def _topology(self, bundle_root):
        return {
            path.name
            for path in finalized_inputs_only(bundle_root.iterdir(), self.recovery_root)
        }

    def test_finalized_inputs_reject_recovery_and_non_checksummed_paths(self):
        with self.assertRaisesRegex(ValueError, "recovery workspace"):
            finalized_inputs_only(
                [self.recovery_root / "states" / "attempt-1.json"], self.recovery_root
            )

        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            finalized_inputs_only([self.evidence_root / "not-finalized"], self.recovery_root)

    def test_uninterrupted_and_resumed_fixture_expose_same_finalized_topology(self):
        uninterrupted = self._complete_seed_fixture("uninterrupted")
        resumed = self._complete_seed_fixture("resumed", interrupted_after="eval-1")

        self.assertEqual(self._topology(uninterrupted), self._topology(resumed))
        self.assertEqual(
            self._topology(uninterrupted),
            {"training", "eval-1", "eval-2", "eval-3", "selection", "reveal", "resources"},
        )


class StageInspectionStatusTests(unittest.TestCase):
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

    def _inspect_fixture_statuses(self):
        workspace = RecoveryWorkspace(self.recovery_root, self.evidence_root)
        completed_bundle = self.evidence_root / "completed"
        write_bundle(
            completed_bundle,
            {
                "manifest.yaml": "manifest: fixture\n",
                "command.sh": "#!/usr/bin/env bash\n",
                "config.yaml": "seed: 17\n",
                "environment.txt": "python: fixture\n",
                "metrics.json": "{}\n",
                "execution.log": "completed\n",
                "gpu.csv": "timestamp,power\n",
                "notes.md": "fixture\n",
            },
        )
        finalize_bundle(completed_bundle)
        fixtures = {
            "completed": ("completed", {"completed_bundle": completed_bundle}),
            "running": ("running", {}),
            "interrupted": ("interrupted", {}),
            "recoverable": ("interrupted", {"recovery_reference": "epoch-1"}),
            "incompatible": ("running", {}),
        }
        inspections = {}
        for name, (status, kwargs) in fixtures.items():
            workspace.write_state(name, self.signature, status=status, **kwargs)
            requested_signature = (
                self.signature if name != "incompatible" else StageSignature.create(
                    manifest_digest="sha256:manifest",
                    protocol_version="phase1-2026-08-29",
                    upstream_commit="a" * 40,
                    upstream_tree="b" * 40,
                    model_revision="c" * 40,
                    seed=42,
                    stage="evaluation",
                    epoch=1,
                    checkpoint_digest="sha256:checkpoint",
                    effective_evaluation_config={"batch_size": 32},
                    expected_example_ids=["visible:0001", "visible:0002"],
                )
            )
            inspections[name] = workspace.inspect_stage(name, requested_signature)
        inspections["unavailable-after-hard-loss"] = workspace.inspect_stage(
            "missing", self.signature
        )
        return inspections

    def test_inspection_exposes_each_required_status(self):
        expected = {
            "completed",
            "running",
            "interrupted",
            "recoverable",
            "incompatible",
            "unavailable-after-hard-loss",
        }
        inspections = self._inspect_fixture_statuses()

        self.assertEqual({inspection.status for inspection in inspections.values()}, expected)
        self.assertTrue(all(inspection.action for inspection in inspections.values()))


if __name__ == "__main__":
    unittest.main()
