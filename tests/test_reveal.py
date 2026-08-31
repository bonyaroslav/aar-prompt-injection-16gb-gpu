import json, tempfile, unittest
from pathlib import Path

from protocol.heldout import HeldOutSealer
from runner.bundle import verify_bundle, BUNDLE_FILES, CHECKSUM_FILE
from runner.core import run_baseline
from runner.fakes import FakeModelAdapter, FakeDatasetAdapter, FakeScorerAdapter, FakeTelemetryAdapter
from runner.reveal import (
    run_trained_held_out_evaluation, finalize_and_authorize_selection,
    build_reveal_package, run_reveal, HeldOutRevealRecovery,
    run_selection_and_reveal,
)
from runner.recovery import AttemptLedger, RecoveryWorkspace
from runner.selection import select_checkpoint
from runner.storage import LocalStorageAdapter

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"


class ShiftedInjecAgentDatasetAdapter(FakeDatasetAdapter):
    """Same as the ordinary fake dataset adapter except InjecAgent candidate IDs
    are shifted, simulating a dataset-adapter drift between the baseline freeze
    and the trained-checkpoint evaluation."""

    def load_items(self, benchmark: str, sample_count: int):
        if benchmark == "injecagent":
            return [{"id": f"injecagent-shifted-{i:04d}"} for i in range(sample_count)]
        return super().load_items(benchmark, sample_count)


class CountingInjecAgentModel(FakeModelAdapter):
    """Records the private evaluation boundary without exposing its payload."""

    def __init__(self):
        self.injecagent_generations = 0

    def generate(self, benchmark: str, item: dict, config: dict) -> str:
        if benchmark == "injecagent":
            self.injecagent_generations += 1
        return super().generate(benchmark, item, config)


class CountingHeldOutSealer(HeldOutSealer):
    def __init__(self, root):
        super().__init__(root)
        self.reveal_calls = 0

    def reveal(self, selection_record: dict):
        self.reveal_calls += 1
        return super().reveal(selection_record)


class RevealTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.storage = LocalStorageAdapter(self.tmp.name)
        self.heldout_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.heldout_tmp.cleanup)
        self.selection_path = Path(self.tmp.name) / "selection.json"

    def _sealer(self):
        return HeldOutSealer(self.heldout_tmp.name)

    def _run_baseline(self, sealer, run_id="baseline-reveal-test"):
        return run_baseline(
            MANIFEST,
            model=FakeModelAdapter(), dataset=FakeDatasetAdapter(), scorer=FakeScorerAdapter(),
            telemetry=FakeTelemetryAdapter(), storage=self.storage, held_out_sealer=sealer, run_id=run_id,
        )

    def _selection_record(self, baseline_benchmarks):
        # A candidate identical to baseline passes every capability gate (zero decline)
        # and is therefore eligible/selected, even though its visible composite is zero.
        candidate = {"epoch": 1, "checkpoint_digest": "sha256:candidate", "benchmarks": baseline_benchmarks}
        record = select_checkpoint(MANIFEST, baseline_benchmarks=baseline_benchmarks, candidates=[candidate])
        self.assertEqual(record["selected_checkpoint_digest"], "sha256:candidate")
        return record

    def _happy_path(self, sealer):
        baseline_result = self._run_baseline(sealer)
        run_trained_held_out_evaluation(
            MANIFEST, model=FakeModelAdapter(), dataset=FakeDatasetAdapter(), scorer=FakeScorerAdapter(), sealer=sealer,
        )
        record = self._selection_record(baseline_result.metrics["benchmarks"])
        finalize_and_authorize_selection(record, self.selection_path, sealer)
        return record

    # -- happy path -----------------------------------------------------

    def test_full_state_machine_sealed_to_authorized_to_revealed(self):
        sealer = self._sealer()
        record = self._happy_path(sealer)
        package = build_reveal_package(sealer, record)
        self.assertEqual(set(package.keys()), {"baseline", "trained", "invalid_classification"})
        for label in ("baseline", "trained"):
            self.assertEqual(set(package[label].keys()), {"valid_only", "intent_to_evaluate"})
            self.assertIn("value", package[label]["valid_only"])
            self.assertIn("n", package[label]["valid_only"])
            self.assertIn("value", package[label]["intent_to_evaluate"])
            self.assertEqual(package[label]["intent_to_evaluate"]["n"], 200)
        self.assertEqual(set(package["invalid_classification"].keys()), {"baseline", "trained"})
        for label in ("baseline", "trained"):
            counts = package["invalid_classification"][label]
            self.assertEqual(counts["valid"] + counts["invalid"], 200)

    def test_run_reveal_produces_a_clean_checksummed_bundle(self):
        sealer = self._sealer()
        record = self._happy_path(sealer)
        result = run_reveal(MANIFEST, sealer=sealer, selection_record=record, storage=self.storage, telemetry=FakeTelemetryAdapter())
        self.assertEqual(result.stage, "reveal")
        bundle_dir = Path(result.bundle_dir)
        for name in (*BUNDLE_FILES, CHECKSUM_FILE):
            self.assertTrue((bundle_dir / name).exists(), f"missing {name}")
        verify_bundle(bundle_dir)  # must not raise
        self.assertIn("held_out", result.metrics)
        self.assertEqual(
            set(result.metrics["held_out"]["injecagent"].keys()), {"baseline", "trained", "invalid_classification"}
        )

    def test_reveal_bundle_never_contains_candidate_ids_or_raw_output(self):
        sealer = self._sealer()
        record = self._happy_path(sealer)
        result = run_reveal(MANIFEST, sealer=sealer, selection_record=record, storage=self.storage, telemetry=FakeTelemetryAdapter())
        bundle_dir = Path(result.bundle_dir)
        for path_name in (*BUNDLE_FILES, CHECKSUM_FILE):
            text = (bundle_dir / path_name).read_text()
            for candidate_id in (f"injecagent-{i:04d}" for i in range(200)):
                self.assertNotIn(candidate_id, text)
            self.assertNotIn("fake-output:injecagent", text)

    def test_finalizing_selection_automatically_authorizes_the_sealer(self):
        sealer = self._sealer()
        baseline_result = self._run_baseline(sealer)
        run_trained_held_out_evaluation(
            MANIFEST, model=FakeModelAdapter(), dataset=FakeDatasetAdapter(), scorer=FakeScorerAdapter(), sealer=sealer,
        )
        record = self._selection_record(baseline_result.metrics["benchmarks"])
        self.assertEqual(sealer.commitments()["state"], "SEALED")
        finalize_and_authorize_selection(record, self.selection_path, sealer)
        self.assertEqual(sealer.commitments()["state"], "AUTHORIZED")
        # reveal() never returns baseline or trained alone.
        package = build_reveal_package(sealer, record)
        self.assertIn("baseline", package)
        self.assertIn("trained", package)

    # -- rejection paths --------------------------------------------------

    def test_reveal_rejected_when_selection_was_never_authorized(self):
        sealer = self._sealer()
        self._run_baseline(sealer)
        run_trained_held_out_evaluation(
            MANIFEST, model=FakeModelAdapter(), dataset=FakeDatasetAdapter(), scorer=FakeScorerAdapter(), sealer=sealer,
        )
        with self.assertRaises(PermissionError):
            build_reveal_package(sealer, {"finalized": True, "selected_checkpoint_digest": "sha256:candidate"})

    def test_reveal_rejected_when_record_does_not_match_what_was_authorized(self):
        sealer = self._sealer()
        record = self._happy_path(sealer)
        mismatched = dict(record, selected_checkpoint_digest="sha256:someone-else")
        with self.assertRaises(PermissionError):
            build_reveal_package(sealer, mismatched)

    def test_reveal_rejected_when_state_is_not_authorized(self):
        sealer = self._sealer()
        self._run_baseline(sealer)
        run_trained_held_out_evaluation(
            MANIFEST, model=FakeModelAdapter(), dataset=FakeDatasetAdapter(), scorer=FakeScorerAdapter(), sealer=sealer,
        )
        record = {"finalized": True, "selected_checkpoint_digest": "sha256:candidate"}
        # Deliberately skip finalize_and_authorize_selection: state stays SEALED.
        with self.assertRaises(PermissionError):
            build_reveal_package(sealer, record)

    def test_trained_evaluation_rejected_when_candidate_commitment_changed_since_freezing(self):
        sealer = self._sealer()
        self._run_baseline(sealer)
        with self.assertRaisesRegex(PermissionError, "commitment changed since freezing"):
            run_trained_held_out_evaluation(
                MANIFEST, model=FakeModelAdapter(), dataset=ShiftedInjecAgentDatasetAdapter(),
                scorer=FakeScorerAdapter(), sealer=sealer,
            )


class HeldOutRevealTransactionTests(unittest.TestCase):
    """Issue #20: every durable boundary retries without a second private run."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.heldout_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.heldout_tmp.cleanup)
        self.storage = LocalStorageAdapter(self.tmp.name)
        self.workspace = RecoveryWorkspace(Path(self.heldout_tmp.name) / "recovery", Path(self.tmp.name))
        self.recovery = HeldOutRevealRecovery(self.workspace, "selected-heldout")
        self.selection_path = Path(self.tmp.name) / "selection.json"
        self.sealer = CountingHeldOutSealer(self.heldout_tmp.name)
        self.baseline = run_baseline(
            MANIFEST, model=FakeModelAdapter(), dataset=FakeDatasetAdapter(), scorer=FakeScorerAdapter(),
            telemetry=FakeTelemetryAdapter(), storage=self.storage, held_out_sealer=self.sealer,
            run_id="transaction-baseline",
        )
        candidate = {
            "epoch": 1,
            "checkpoint_digest": "sha256:selected-checkpoint",
            "benchmarks": self.baseline.metrics["benchmarks"],
        }
        self.record = select_checkpoint(
            MANIFEST, baseline_benchmarks=self.baseline.metrics["benchmarks"], candidates=[candidate],
        )

    def _run(self, model, *, record=None, checkpoint_digest=None, authorization_identity="evaluator-a", fail_after=None):
        return run_selection_and_reveal(
            MANIFEST,
            selection_record=self.record if record is None else record,
            selection_path=self.selection_path,
            sealer=self.sealer,
            model=model,
            dataset=FakeDatasetAdapter(),
            scorer=FakeScorerAdapter(),
            storage=self.storage,
            telemetry=FakeTelemetryAdapter(),
            recovery=self.recovery,
            checkpoint_digest=checkpoint_digest,
            authorization_identity=authorization_identity,
            fail_after=fail_after,
        )

    def test_failure_after_each_transition_retries_one_logical_evaluation_and_reveal(self):
        for transition in ("SEALED", "SELECTION_FINALIZED", "AUTHORIZED", "TRAINED_RESULT_SEALED", "REVEALED"):
            with self.subTest(transition=transition):
                self.setUp()
                model = CountingInjecAgentModel()
                with self.assertRaisesRegex(RuntimeError, f"injected failure after {transition}"):
                    self._run(model, fail_after=transition)
                result = self._run(model)
                self.assertEqual(result.state, "REVEALED")
                self.assertEqual(model.injecagent_generations, 200)
                self.assertEqual(result.reveal_count, 1)
                self.assertEqual(self.sealer.reveal_calls, 1)
                verify_bundle(Path(result.reveal.bundle_dir))
                self.assertEqual(
                    [row["status"] for row in AttemptLedger(self.workspace.root / "attempts.jsonl").rows()],
                    ["SEALED", "SELECTION_FINALIZED", "AUTHORIZED", "TRAINED_RESULT_SEALED", "REVEALED"],
                )

    def test_retry_rejects_changed_transaction_identity_inputs(self):
        model = CountingInjecAgentModel()
        with self.assertRaisesRegex(RuntimeError, "injected failure after AUTHORIZED"):
            self._run(model, fail_after="AUTHORIZED")
        changed_selection = dict(self.record, selected_checkpoint_digest="sha256:other")
        with self.assertRaisesRegex(ValueError, "selection digest"):
            self._run(model, record=changed_selection)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self._run(model, checkpoint_digest="sha256:other")
        with self.assertRaisesRegex(ValueError, "authorization identity"):
            self._run(model, authorization_identity="evaluator-b")
        state = json.loads(self.sealer.state_file.read_text(encoding="utf-8"))
        state["candidates"] = "sha256:changed-commitment"
        self.sealer._write(state)
        with self.assertRaisesRegex(ValueError, "candidate commitment"):
            self._run(model)


if __name__ == "__main__":
    unittest.main()
