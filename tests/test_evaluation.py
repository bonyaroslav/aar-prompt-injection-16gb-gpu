import json, tempfile, unittest
from pathlib import Path

from runner.core import effective_eval_config
from runner.evaluation import run_trained_evaluation
from runner.fakes import FakeModelAdapter, FakeDatasetAdapter, FakeScorerAdapter, FakeTelemetryAdapter
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


if __name__ == "__main__":
    unittest.main()
