import json, tempfile, unittest
from pathlib import Path

from runner.training import (
    TrainingRecovery, run_training, request_fallback, APPROVED_OOM_FALLBACK,
    OOM_FALLBACK_SEQUENCE_LENGTH,
)
from runner.fakes import FakeTrainerAdapter, FakeTelemetryAdapter
from runner.recovery import AttemptLedger, RecoveryWorkspace
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


class _InterruptingTrainer(FakeTrainerAdapter):
    """Fails once after epoch 1 has been completely merged."""

    def __init__(self, interrupt=True):
        super().__init__()
        self.train_calls = []
        self.merge_calls = []
        self._interrupt = interrupt
        self._interrupted = False

    def train_epoch(self, **kwargs):
        self.train_calls.append(kwargs["epoch"])
        if self._interrupt and kwargs["epoch"] == 2 and not self._interrupted:
            self._interrupted = True
            raise RuntimeError("injected interruption")
        return super().train_epoch(**kwargs)

    def merge_checkpoint(self, fingerprint, output_dir):
        self.merge_calls.append(Path(output_dir).name)
        return super().merge_checkpoint(fingerprint, output_dir)


class _MergeInterruptingTrainer(_InterruptingTrainer):
    def merge_checkpoint(self, fingerprint, output_dir):
        if Path(output_dir).name == "epoch-2":
            raise RuntimeError("injected merge interruption")
        return super().merge_checkpoint(fingerprint, output_dir)


class ResumableTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.storage = LocalStorageAdapter(root / "runs")
        self.recovery_root = root / "recovery"
        self.evidence_root = root / "runs"

    def _recovery(self):
        return TrainingRecovery(
            RecoveryWorkspace(self.recovery_root, self.evidence_root),
            "training-seed17",
        )

    def _run(self, trainer, run_id, recovery):
        return run_training(
            MANIFEST, trainer=trainer, telemetry=FakeTelemetryAdapter(),
            storage=self.storage, seed=17, run_id=run_id, recovery=recovery,
        )

    def test_interrupted_attempt_reuses_completed_epoch_from_its_finalized_bundle(self):
        recovery = self._recovery()
        interrupted = _InterruptingTrainer()
        with self.assertRaisesRegex(RuntimeError, "injected interruption"):
            self._run(interrupted, "attempt-1", recovery)
        self.assertEqual(interrupted.train_calls, [1, 2])

        resumed_trainer = _InterruptingTrainer(interrupt=False)
        resumed = self._run(resumed_trainer, "attempt-2", recovery)
        self.assertEqual(resumed.outcome, "success")
        self.assertEqual(resumed_trainer.train_calls, [2, 3])
        self.assertEqual(resumed_trainer.merge_calls, ["epoch-2", "epoch-3"])
        self.assertTrue(Path(resumed.metrics["checkpoints"]["epoch-1"]["merged_dir"]).is_dir())
        verify_bundle(Path(resumed.bundle_dir))

        replay_trainer = _InterruptingTrainer(interrupt=False)
        replayed = self._run(replay_trainer, "attempt-3", recovery)
        self.assertEqual(replay_trainer.train_calls, [])
        self.assertEqual(replayed.bundle_dir, resumed.bundle_dir)

        rows = AttemptLedger(self.recovery_root / "attempts.jsonl").rows()
        self.assertEqual([row["status"] for row in rows], ["interrupted", "completed"])

    def test_recovery_rejects_a_signature_with_a_different_seed(self):
        recovery = self._recovery()
        with self.assertRaisesRegex(RuntimeError, "injected interruption"):
            self._run(_InterruptingTrainer(), "attempt-1", recovery)

        with self.assertRaisesRegex(ValueError, "seed"):
            run_training(
                MANIFEST, trainer=_InterruptingTrainer(interrupt=False),
                telemetry=FakeTelemetryAdapter(), storage=self.storage, seed=42,
                run_id="attempt-2", recovery=recovery,
            )

    def test_recovery_rejects_a_tampered_completed_checkpoint(self):
        recovery = self._recovery()
        with self.assertRaisesRegex(RuntimeError, "injected interruption"):
            self._run(_InterruptingTrainer(), "attempt-1", recovery)

        state_key = "training-seed17-seq2048-epoch1"
        checkpoint = json.loads(recovery.workspace.recovery_reference(state_key))
        (Path(checkpoint["merged_dir"]) / "model.json").write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self._run(_InterruptingTrainer(interrupt=False), "attempt-2", recovery)

    def test_merge_interruption_is_accounted_and_restarts_that_epoch(self):
        recovery = self._recovery()
        with self.assertRaisesRegex(RuntimeError, "injected merge interruption"):
            self._run(_MergeInterruptingTrainer(interrupt=False), "attempt-1", recovery)

        rows = AttemptLedger(self.recovery_root / "attempts.jsonl").rows()
        self.assertEqual([row["status"] for row in rows], ["interrupted"])
        resumed_trainer = _InterruptingTrainer(interrupt=False)
        self._run(resumed_trainer, "attempt-2", recovery)
        self.assertEqual(resumed_trainer.train_calls, [2, 3])


if __name__ == "__main__":
    unittest.main()
