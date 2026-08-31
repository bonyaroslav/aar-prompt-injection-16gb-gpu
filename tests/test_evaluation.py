import json, tempfile, unittest
from pathlib import Path

from runner.core import effective_eval_config
from runner.evaluation import EvaluationRecovery, run_trained_evaluation
from runner.fakes import FakeModelAdapter, FakeDatasetAdapter, FakeScorerAdapter, FakeTelemetryAdapter
from runner.recovery import AttemptLedger, RecoveryWorkspace
from runner.storage import LocalStorageAdapter
from runner.bundle import verify_bundle, BUNDLE_FILES, CHECKSUM_FILE
from protocol.validate_manifest import load as load_manifest

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"
CHECKPOINT = {"fingerprint": "sha256:fake-epoch-1-fingerprint", "sequence_length": 2048, "merged_dir": "/tmp/fake/epoch-1"}


class TrainedCheckpointEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.storage = LocalStorageAdapter(self.tmp.name)

    def _run(self, run_id="eval-seed17-epoch1", epoch=1, checkpoint=None):
        return run_trained_evaluation(
            MANIFEST,
            model=FakeModelAdapter(),
            dataset=FakeDatasetAdapter(),
            scorer=FakeScorerAdapter(),
            telemetry=FakeTelemetryAdapter(),
            storage=self.storage,
            seed=17,
            epoch=epoch,
            checkpoint=checkpoint or CHECKPOINT,
            run_id=run_id,
        )

    def test_effective_config_identical_to_baseline_apart_from_checkpoint_field(self):
        manifest = load_manifest(MANIFEST)
        baseline_cfg = effective_eval_config(manifest, checkpoint=None)
        trained_cfg = effective_eval_config(manifest, checkpoint=CHECKPOINT["fingerprint"])
        differing_keys = {k for k in baseline_cfg if baseline_cfg[k] != trained_cfg.get(k)}
        self.assertEqual(differing_keys, {"checkpoint"})
        self.assertIsNone(baseline_cfg["checkpoint"])
        self.assertEqual(trained_cfg["checkpoint"], CHECKPOINT["fingerprint"])

    def test_returns_structured_result_not_console_text(self):
        result = self._run()
        self.assertEqual(result.stage, "trained_evaluation")
        self.assertIsInstance(result.metrics, dict)
        self.assertIsInstance(result.checksums, dict)
        self.assertTrue(Path(result.bundle_dir).is_dir())

    def test_evaluates_exactly_the_three_visible_and_three_capability_benchmarks(self):
        result = self._run()
        self.assertEqual(
            set(result.metrics["benchmarks"].keys()),
            {"open_prompt_injection", "tensor_trust_hijack", "tensor_trust_extract", "mmlu", "gsm8k", "ifeval"},
        )

    def test_metrics_never_mention_held_out_injecagent(self):
        result = self._run()
        self.assertNotIn("held_out", result.metrics)
        blob = json.dumps(result.metrics)
        self.assertNotIn("injecagent", blob)

    def test_records_seed_epoch_and_checkpoint_fingerprint(self):
        result = self._run(epoch=1)
        self.assertEqual(result.metrics["seed"], 17)
        self.assertEqual(result.metrics["epoch"], 1)
        self.assertEqual(result.metrics["checkpoint"], CHECKPOINT["fingerprint"])

    def test_config_yaml_records_checkpoint_field(self):
        result = self._run()
        config = json.loads((Path(result.bundle_dir) / "config.yaml").read_text())
        self.assertEqual(config["checkpoint"], CHECKPOINT["fingerprint"])

    def test_bundle_contains_required_files_and_verifies_clean(self):
        result = self._run()
        bundle_dir = Path(result.bundle_dir)
        for name in (*BUNDLE_FILES, CHECKSUM_FILE):
            self.assertTrue((bundle_dir / name).exists(), f"missing {name}")
        verify_bundle(bundle_dir)  # must not raise

    def test_second_run_with_same_run_id_is_rejected(self):
        self._run(run_id="eval-dup")
        with self.assertRaises(FileExistsError):
            self._run(run_id="eval-dup")


class _OverridingTelemetryAdapter(FakeTelemetryAdapter):
    """Mimics a real adapter that overrides the evidence-bundle captions -- e.g.
    `RealTelemetryAdapter` setting `.command_text`/`.notes_text` after construction
    (see `runner.real_seed_run.run_real_seed`)."""

    def __init__(self):
        super().__init__()
        self.command_text = "the real reproduction command"
        self.notes_text = lambda stage: f"real notes for {stage}"

    def environment_text(self) -> str:
        return "real environment facts\n"


class _ManifestMetadataGuardDatasetAdapter(FakeDatasetAdapter):
    """A dataset adapter that would blow up if the eval stage ever touched its
    InjecAgent-provenance surface -- exactly the property `RealDatasetAdapter`
    exposes that reads `heldout_dir`, which this stage must never do."""

    @property
    def manifest_metadata(self):
        raise AssertionError("trained-checkpoint evaluation must never call dataset.manifest_metadata()")

    @property
    def environment_lines(self):
        raise AssertionError("trained-checkpoint evaluation must never touch dataset.environment_lines")


class AdapterOverrideTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.storage = LocalStorageAdapter(self.tmp.name)

    def test_real_style_telemetry_overrides_command_environment_and_notes(self):
        result = run_trained_evaluation(
            MANIFEST, model=FakeModelAdapter(), dataset=FakeDatasetAdapter(),
            scorer=FakeScorerAdapter(), telemetry=_OverridingTelemetryAdapter(),
            storage=self.storage, seed=17, epoch=1, checkpoint=CHECKPOINT, run_id="eval-override",
        )
        bundle_dir = Path(result.bundle_dir)
        self.assertIn("the real reproduction command", (bundle_dir / "command.sh").read_text())
        self.assertEqual((bundle_dir / "environment.txt").read_text(), "real environment facts\n")
        self.assertEqual((bundle_dir / "notes.md").read_text(), "real notes for trained_evaluation")

    def test_never_touches_dataset_manifest_metadata_or_environment_lines(self):
        # Must not raise: proves the eval stage never accesses either guarded property.
        result = run_trained_evaluation(
            MANIFEST, model=FakeModelAdapter(), dataset=_ManifestMetadataGuardDatasetAdapter(),
            scorer=FakeScorerAdapter(), telemetry=FakeTelemetryAdapter(),
            storage=self.storage, seed=17, epoch=1, checkpoint=CHECKPOINT, run_id="eval-guarded",
        )
        self.assertEqual(result.stage, "trained_evaluation")


class _InterruptingModelAdapter:
    """Wraps `FakeModelAdapter`, raising after `fail_after` generations to inject
    an interruption mid-evaluation."""

    def __init__(self, fail_after):
        self._inner = FakeModelAdapter()
        self.fail_after = fail_after
        self.generated = []

    def generate(self, benchmark, item, config):
        if len(self.generated) >= self.fail_after:
            raise RuntimeError("injected interruption")
        self.generated.append((benchmark, item["id"]))
        return self._inner.generate(benchmark, item, config)


class _CountingScorerAdapter(FakeScorerAdapter):
    def __init__(self):
        self.scored = []

    def score(self, benchmark, item, output, config):
        self.scored.append((benchmark, item["id"]))
        return super().score(benchmark, item, output, config)


class _CountingModelAdapter(FakeModelAdapter):
    def __init__(self):
        self.generated = []

    def generate(self, benchmark, item, config):
        self.generated.append((benchmark, item["id"]))
        return super().generate(benchmark, item, config)


EXPECTED_EXAMPLE_COUNT = 300 * 4 + 200 * 2  # OPI/TT-hijack/TT-extract/MMLU + GSM8K/IFEval


class ResumableEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.storage = LocalStorageAdapter(root / "runs")
        self.recovery_root = root / "recovery"
        self.evidence_root = root / "runs"

    def _workspace(self):
        return RecoveryWorkspace(self.recovery_root, self.evidence_root)

    def _run(self, *, run_id, recovery=None, model=None, scorer=None, dataset=None):
        return run_trained_evaluation(
            MANIFEST,
            model=model or FakeModelAdapter(),
            dataset=dataset or FakeDatasetAdapter(),
            scorer=scorer or FakeScorerAdapter(),
            telemetry=FakeTelemetryAdapter(),
            storage=self.storage,
            seed=17,
            epoch=1,
            checkpoint=CHECKPOINT,
            run_id=run_id,
            recovery=recovery,
        )

    def test_uninterrupted_recovery_run_matches_a_plain_run_exactly(self):
        plain = self._run(run_id="plain")
        recovered = self._run(
            run_id="recovered",
            recovery=EvaluationRecovery(self._workspace(), "eval-seed17-epoch1"),
        )

        self.assertEqual(recovered.metrics, plain.metrics)
        self.assertEqual(
            (Path(recovered.bundle_dir) / "metrics.json").read_text(),
            (Path(plain.bundle_dir) / "metrics.json").read_text(),
        )
        self.assertEqual(
            (Path(recovered.bundle_dir) / "config.yaml").read_text(),
            (Path(plain.bundle_dir) / "config.yaml").read_text(),
        )
        verify_bundle(Path(recovered.bundle_dir))

    def test_interrupted_then_resumed_matches_uninterrupted_and_scores_no_item_twice(self):
        reference = self._run(run_id="reference")

        workspace = self._workspace()
        recovery = EvaluationRecovery(workspace, "eval-seed17-epoch1")
        interrupting_model = _InterruptingModelAdapter(fail_after=400)
        with self.assertRaises(RuntimeError):
            self._run(run_id="attempt-1", recovery=recovery, model=interrupting_model)

        resume_scorer = _CountingScorerAdapter()
        resumed = self._run(
            run_id="attempt-2", recovery=recovery, scorer=resume_scorer,
        )

        # nothing the first attempt already scored is scored again
        first_attempt_scored = set(interrupting_model.generated)
        self.assertEqual(len(first_attempt_scored), 400)
        self.assertTrue(first_attempt_scored.isdisjoint(resume_scorer.scored))
        self.assertEqual(len(resume_scorer.scored), len(set(resume_scorer.scored)))
        self.assertEqual(
            len(first_attempt_scored) + len(resume_scorer.scored), EXPECTED_EXAMPLE_COUNT
        )

        # identical final metrics and finalized-artifact topology
        self.assertEqual(resumed.metrics, reference.metrics)
        self.assertEqual(
            (Path(resumed.bundle_dir) / "metrics.json").read_text(),
            (Path(reference.bundle_dir) / "metrics.json").read_text(),
        )
        self.assertEqual(
            {p.name for p in Path(resumed.bundle_dir).iterdir()},
            set(BUNDLE_FILES) | {CHECKSUM_FILE},
        )
        verify_bundle(Path(resumed.bundle_dir))

    def test_final_aggregation_counts_every_expected_example_exactly_once(self):
        resumed = self._run(
            run_id="agg",
            recovery=EvaluationRecovery(self._workspace(), "eval-seed17-epoch1"),
        )
        total = sum(
            len(benchmark["items"]) for benchmark in resumed.metrics["benchmarks"].values()
        )
        self.assertEqual(total, EXPECTED_EXAMPLE_COUNT)

    def test_no_model_generation_is_interrupted_to_write_progress(self):
        workspace = self._workspace()
        recovery = EvaluationRecovery(workspace, "eval-seed17-epoch1")
        model = _InterruptingModelAdapter(fail_after=5)
        with self.assertRaises(RuntimeError):
            self._run(run_id="attempt-1", recovery=recovery, model=model)

        # exactly the 5 fully-generated items are journalled; the 6th (which raised
        # before returning an output) is not
        journalled = workspace.completed_progress("eval-seed17-epoch1")
        self.assertEqual(len(journalled), 5)
        self.assertEqual(set(journalled), set(model.generated))

    def test_completed_stage_short_circuits_without_rescoring(self):
        workspace = self._workspace()
        recovery = EvaluationRecovery(workspace, "eval-seed17-epoch1")
        first = self._run(run_id="first", recovery=recovery)

        replay_model = _CountingModelAdapter()
        replay_scorer = _CountingScorerAdapter()
        again = run_trained_evaluation(
            MANIFEST, model=replay_model, dataset=FakeDatasetAdapter(),
            scorer=replay_scorer, telemetry=FakeTelemetryAdapter(), storage=self.storage,
            seed=17, epoch=1, checkpoint=CHECKPOINT, run_id="second", recovery=recovery,
        )

        self.assertEqual(replay_model.generated, [])
        self.assertEqual(replay_scorer.scored, [])
        self.assertEqual(again.metrics, first.metrics)
        self.assertEqual(again.bundle_dir, first.bundle_dir)
        verify_bundle(Path(again.bundle_dir))

    def test_resume_rejects_an_incompatible_signature_and_preserves_state(self):
        workspace = self._workspace()
        recovery = EvaluationRecovery(workspace, "eval-seed17-epoch1")
        interrupting_model = _InterruptingModelAdapter(fail_after=10)
        with self.assertRaises(RuntimeError):
            self._run(run_id="attempt-1", recovery=recovery, model=interrupting_model)

        with self.assertRaisesRegex(ValueError, "checkpoint_digest"):
            run_trained_evaluation(
                MANIFEST, model=FakeModelAdapter(), dataset=FakeDatasetAdapter(),
                scorer=FakeScorerAdapter(), telemetry=FakeTelemetryAdapter(), storage=self.storage,
                seed=17, epoch=1,
                checkpoint={**CHECKPOINT, "fingerprint": "sha256:a-different-checkpoint"},
                run_id="attempt-2", recovery=recovery,
            )

        # the interrupted partial state is still there for diagnosis / a correct resume
        self.assertEqual(len(workspace.completed_progress("eval-seed17-epoch1")), 10)

    def test_each_attempt_is_recorded_in_the_attempt_ledger(self):
        workspace = self._workspace()
        recovery = EvaluationRecovery(workspace, "eval-seed17-epoch1")
        with self.assertRaises(RuntimeError):
            self._run(
                run_id="attempt-1", recovery=recovery,
                model=_InterruptingModelAdapter(fail_after=50),
            )
        self._run(run_id="attempt-2", recovery=recovery)

        rows = AttemptLedger(self.recovery_root / "attempts.jsonl").rows()
        self.assertEqual([row["status"] for row in rows], ["interrupted", "completed"])
        self.assertTrue(all(row["gpu_hours"] == "unavailable" for row in rows))
        self.assertEqual(len({row["attempt_id"] for row in rows}), 2)


if __name__ == "__main__":
    unittest.main()
