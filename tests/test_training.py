import json, tempfile, unittest
from pathlib import Path

from runner.training import run_training, request_fallback, APPROVED_OOM_FALLBACK, OOM_FALLBACK_SEQUENCE_LENGTH
from runner.fakes import FakeTrainerAdapter, FakeTelemetryAdapter
from runner.storage import LocalStorageAdapter
from runner.bundle import verify_bundle, BUNDLE_FILES, CHECKSUM_FILE
from protocol.validate_manifest import load as load_manifest

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"
SEED = load_manifest(MANIFEST)["training"]["seeds"][0]
EPOCHS = load_manifest(MANIFEST)["training"]["optimizer"]["epochs"]
BASE_SEQUENCE_LENGTH = load_manifest(MANIFEST)["training"]["data"]["max_sequence_length"]


class TrainingStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.storage = LocalStorageAdapter(self.tmp.name)

    def _run(self, trainer, run_id, seed=SEED):
        return run_training(
            MANIFEST,
            trainer=trainer,
            telemetry=FakeTelemetryAdapter(),
            storage=self.storage,
            seed=seed,
            run_id=run_id,
        )

    def test_full_technical_success_produces_one_merged_checkpoint_per_epoch(self):
        result = self._run(FakeTrainerAdapter(), run_id="training-success")
        self.assertEqual(result.outcome, "success")
        self.assertEqual(result.metrics["fallback_applied"], False)
        checkpoints = result.metrics["checkpoints"]
        self.assertEqual(set(checkpoints.keys()), {f"epoch-{i}" for i in range(1, EPOCHS + 1)})
        for epoch_key, info in checkpoints.items():
            self.assertEqual(info["sequence_length"], BASE_SEQUENCE_LENGTH)
            merged_dir = Path(info["merged_dir"])
            self.assertTrue((merged_dir / "model.json").exists(), f"{epoch_key} missing merged model dir")
            merged = json.loads((merged_dir / "model.json").read_text())
            self.assertEqual(merged["fingerprint"], info["fingerprint"])

    def test_bundle_contains_required_files_and_verifies_clean(self):
        result = self._run(FakeTrainerAdapter(), run_id="training-bundle-check")
        bundle_dir = Path(result.bundle_dir)
        for name in (*BUNDLE_FILES, CHECKSUM_FILE):
            self.assertTrue((bundle_dir / name).exists(), f"missing {name}")
        verify_bundle(bundle_dir)  # must not raise

    def test_recoverable_oom_applies_approved_fallback_exactly_once(self):
        trainer = FakeTrainerAdapter(oom_at_epoch=2, oom_sequence_lengths=frozenset({BASE_SEQUENCE_LENGTH}))
        result = self._run(trainer, run_id="training-recoverable-oom")
        self.assertEqual(result.outcome, "success")
        self.assertTrue(result.metrics["fallback_applied"])
        checkpoints = result.metrics["checkpoints"]
        self.assertEqual(set(checkpoints.keys()), {f"epoch-{i}" for i in range(1, EPOCHS + 1)})
        for info in checkpoints.values():
            self.assertEqual(info["sequence_length"], OOM_FALLBACK_SEQUENCE_LENGTH)
        log_text = (Path(result.bundle_dir) / "execution.log").read_text()
        self.assertIn(APPROVED_OOM_FALLBACK, log_text)

    def test_unrecoverable_oom_is_preserved_as_a_failed_run_bundle(self):
        trainer = FakeTrainerAdapter(oom_at_epoch=2, oom_sequence_lengths=frozenset({BASE_SEQUENCE_LENGTH, OOM_FALLBACK_SEQUENCE_LENGTH}))
        result = self._run(trainer, run_id="training-unrecoverable-oom")
        self.assertEqual(result.outcome, "failed")
        self.assertTrue(result.metrics["fallback_applied"])
        self.assertIn("failure_reason", result.metrics)
        # Preserved, not deleted: the bundle is still complete and verifies clean.
        bundle_dir = Path(result.bundle_dir)
        for name in (*BUNDLE_FILES, CHECKSUM_FILE):
            self.assertTrue((bundle_dir / name).exists(), f"missing {name}")
        verify_bundle(bundle_dir)  # must not raise
        # Only epoch 1 (the last attempt before the second OOM at epoch 2) is recorded.
        self.assertEqual(set(result.metrics["checkpoints"].keys()), {"epoch-1"})

    def test_three_scenarios_produce_distinct_correctly_labeled_bundles(self):
        success = self._run(FakeTrainerAdapter(), run_id="training-label-success")
        recoverable = self._run(
            FakeTrainerAdapter(oom_at_epoch=2, oom_sequence_lengths=frozenset({BASE_SEQUENCE_LENGTH})),
            run_id="training-label-recoverable",
        )
        unrecoverable = self._run(
            FakeTrainerAdapter(oom_at_epoch=2, oom_sequence_lengths=frozenset({BASE_SEQUENCE_LENGTH, OOM_FALLBACK_SEQUENCE_LENGTH})),
            run_id="training-label-unrecoverable",
        )
        self.assertEqual(success.metrics["outcome"], "success")
        self.assertFalse(success.metrics["fallback_applied"])
        self.assertEqual(recoverable.metrics["outcome"], "success")
        self.assertTrue(recoverable.metrics["fallback_applied"])
        self.assertEqual(unrecoverable.metrics["outcome"], "failed")
        self.assertTrue(unrecoverable.metrics["fallback_applied"])
        self.assertEqual({success.bundle_dir, recoverable.bundle_dir, unrecoverable.bundle_dir}.__len__(), 3)

    def test_unauthorized_fallback_is_rejected_not_silently_applied(self):
        manifest = load_manifest(MANIFEST)
        with self.assertRaises(ValueError):
            request_fallback("increase_batch_size_beyond_manifest", manifest)
        for name in manifest["allowed_technical_fallbacks"]:
            request_fallback(name, manifest)  # must not raise

    def test_seed_propagates_deterministically_into_fake_checkpoint_fingerprint(self):
        trainer = FakeTrainerAdapter()
        training_cfg = load_manifest(MANIFEST)["training"]
        fp_a = trainer.train_epoch(seed=17, epoch=1, sequence_length=BASE_SEQUENCE_LENGTH, config=training_cfg)
        fp_b = trainer.train_epoch(seed=17, epoch=1, sequence_length=BASE_SEQUENCE_LENGTH, config=training_cfg)
        fp_c = trainer.train_epoch(seed=42, epoch=1, sequence_length=BASE_SEQUENCE_LENGTH, config=training_cfg)
        self.assertEqual(fp_a, fp_b)
        self.assertNotEqual(fp_a, fp_c)

    def test_rejects_seed_not_in_manifest_frozen_seeds(self):
        with self.assertRaises(ValueError):
            self._run(FakeTrainerAdapter(), run_id="training-bad-seed", seed=999999)

    def test_second_run_with_same_run_id_is_rejected(self):
        self._run(FakeTrainerAdapter(), run_id="training-dup")
        with self.assertRaises(FileExistsError):
            self._run(FakeTrainerAdapter(), run_id="training-dup")

    def test_no_gpu_model_or_network_dependency_in_fakes(self):
        import runner.fakes as fakes_mod
        src = Path(fakes_mod.__file__).read_text()
        for banned in ("torch", "transformers", "requests", "huggingface_hub", "socket"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main()
