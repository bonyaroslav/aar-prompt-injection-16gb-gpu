import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import runner.ablation_training as ablation_training
from runner.ablation_training import MidEpochCheckpointStore, run_ablation_epoch
from runner.recovery import RecoveryWorkspace, StageSignature, finalized_inputs_only


def _signature():
    return StageSignature.create(
        manifest_digest="sha256:ablation-manifest",
        protocol_version="ablation-v1",
        upstream_commit="a" * 40,
        upstream_tree="b" * 40,
        model_revision="c" * 40,
        seed=99,
        stage="training",
        epoch=1,
        checkpoint_digest="sha256:base-model",
        effective_evaluation_config={"sequence_length": 32},
        expected_example_ids=[],
    )


def _state(step_index):
    return {
        "adapter_weights": {"lora.weight": b"adapter-weights"},
        "optimizer_state": {"exp_avg": [0.25, -0.5], "exp_avg_sq": [0.0625, 0.25]},
        "scheduler_state": {"last_epoch": step_index, "base_lrs": [0.001]},
        "cpu_rng_state": random.Random(1234 + step_index).getstate(),
        "cuda_rng_state": [b"cuda-state-0"],
        "step_index": step_index,
    }


class MidEpochCheckpointStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.workspace = RecoveryWorkspace(root / "recovery", root / "evidence")
        self.store = MidEpochCheckpointStore(
            self.workspace, "ablation-seed99-epoch1", _signature()
        )

    def test_load_round_trips_every_required_mutable_state_field(self):
        """Removing any persisted state component would make this fail."""
        expected = _state(step_index=3)

        measurement = self.store.save(expected)

        self.assertEqual(self.store.load(), expected)
        self.assertEqual(measurement.step_index, 3)
        self.assertGreater(measurement.byte_count, 0)
        self.assertGreaterEqual(measurement.save_seconds, 0.0)

    def test_interrupted_new_save_keeps_previous_checkpoint_loadable(self):
        """Promoting a pointer before its slot is durable would lose this checkpoint."""
        previous = _state(step_index=2)
        self.store.save(previous)

        with patch(
            "runner.ablation_training._write_json_atomically",
            side_effect=OSError("injected pointer-write fault"),
        ):
            with self.assertRaisesRegex(OSError, "injected pointer-write fault"):
                self.store.save(_state(step_index=3))

        self.assertEqual(self.store.load(), previous)

    def test_recovery_checkpoint_is_not_accepted_as_finalized_evidence(self):
        """Treating recovery bytes as finalized input would weaken evidence isolation."""
        self.store.save(_state(step_index=1))

        with self.assertRaisesRegex(ValueError, "recovery workspace"):
            finalized_inputs_only([self.store.root], self.workspace.root)

    def test_rejects_a_world_readable_posix_recovery_workspace(self):
        """Untrusted checkpoint replacement would make pickle deserialization unsafe."""
        if os.name != "posix":
            self.skipTest("POSIX mode bits are not authoritative on this platform")
        insecure_root = Path(self.temporary_directory.name) / "insecure-recovery"
        insecure_root.mkdir(mode=0o755)
        workspace = RecoveryWorkspace(insecure_root, Path(self.temporary_directory.name) / "evidence")

        with self.assertRaisesRegex(ValueError, "private"):
            MidEpochCheckpointStore(workspace, "insecure", _signature())

    def test_measurement_includes_pointer_promotion(self):
        """Measuring before pointer promotion understates a durable recovery save."""
        events = []
        times = iter([1.0, 3.5])
        original_writer = ablation_training._write_json_atomically

        def clock():
            events.append("clock")
            return next(times)

        def write_pointer(path, document):
            events.append("pointer")
            return original_writer(path, document)

        store = MidEpochCheckpointStore(
            self.workspace, "measured-promotion", _signature(), clock=clock,
        )
        with patch("runner.ablation_training._write_json_atomically", side_effect=write_pointer):
            measurement = store.save(_state(step_index=1))

        self.assertEqual(events, ["clock", "pointer", "clock"])
        self.assertEqual(measurement.save_seconds, 2.5)


class _ToyRuntime:
    """A deterministic CPU optimizer that exposes the production step seam."""

    def __init__(self, *, interrupt_at=None):
        self.weights = np.array([0.25, -0.75], dtype=np.float64)
        self.moment = np.zeros(2, dtype=np.float64)
        self.scheduler_last_epoch = 0
        self.cpu_random = random.Random(9182)
        self.interrupt_at = interrupt_at
        self.executed_step_indexes = []

    def optimizer_safe_step(self, step_index):
        if step_index == self.interrupt_at:
            raise RuntimeError(f"injected interruption at optimizer step {step_index}")
        self.executed_step_indexes.append(step_index)
        gradient = np.array(
            [step_index + 1.0, self.cpu_random.random()], dtype=np.float64
        )
        self.moment = 0.8 * self.moment + 0.2 * gradient
        learning_rate = 0.01 / (1 + self.scheduler_last_epoch)
        self.weights -= learning_rate * self.moment
        self.scheduler_last_epoch += 1

    def capture_mid_epoch_state(self, step_index):
        return {
            "adapter_weights": {"toy.weight": self.weights.tobytes()},
            "optimizer_state": {"moment": self.moment.tobytes()},
            "scheduler_state": {"last_epoch": self.scheduler_last_epoch},
            "cpu_rng_state": self.cpu_random.getstate(),
            "cuda_rng_state": [b"cpu-only-toy-has-no-cuda-device"],
            "step_index": step_index,
        }

    def restore_mid_epoch_state(self, state):
        self.weights = np.frombuffer(
            state["adapter_weights"]["toy.weight"], dtype=np.float64
        ).copy()
        self.moment = np.frombuffer(
            state["optimizer_state"]["moment"], dtype=np.float64
        ).copy()
        self.scheduler_last_epoch = state["scheduler_state"]["last_epoch"]
        self.cpu_random.setstate(state["cpu_rng_state"])

    @property
    def weights_bytes(self):
        return self.weights.tobytes()


class AblationEpochRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.workspace = RecoveryWorkspace(root / "recovery", root / "evidence")

    def _store(self, name):
        return MidEpochCheckpointStore(self.workspace, name, _signature())

    def test_interrupted_and_resumed_toy_run_executes_each_logical_step_once(self):
        """Restarting from the wrong step would repeat or skip a toy update."""
        uninterrupted_runtime = _ToyRuntime()
        uninterrupted = run_ablation_epoch(
            protocol_version="ablation-v1", runtime=uninterrupted_runtime,
            total_steps=8, checkpoint_store=self._store("uninterrupted"),
            checkpoint_interval=1,
        )

        for interrupt_at in range(1, 8):
            interrupted_runtime = _ToyRuntime(interrupt_at=interrupt_at)
            resumed_store = self._store(f"interrupted-{interrupt_at}")
            with self.assertRaisesRegex(RuntimeError, f"optimizer step {interrupt_at}"):
                run_ablation_epoch(
                    protocol_version="ablation-v1", runtime=interrupted_runtime,
                    total_steps=8, checkpoint_store=resumed_store, checkpoint_interval=1,
                )
            resumed_runtime = _ToyRuntime()
            resumed = run_ablation_epoch(
                protocol_version="ablation-v1", runtime=resumed_runtime,
                total_steps=8, checkpoint_store=resumed_store, checkpoint_interval=1,
            )

            self.assertEqual(
                interrupted_runtime.executed_step_indexes + resumed_runtime.executed_step_indexes,
                list(range(8)),
            )
            self.assertEqual(resumed_runtime.weights_bytes, uninterrupted_runtime.weights_bytes)
            self.assertTrue(resumed.mid_epoch_resume_fired)
            self.assertEqual(resumed.recovery_evidence["mid_epoch_resume_fired"], True)
            self.assertEqual(
                [entry["step_index"] for entry in resumed.recovery_evidence["save_measurements"]],
                list(range(interrupt_at + 1, 9)),
            )

        self.assertEqual(uninterrupted.checkpoint_steps, list(range(1, 9)))

    def test_checkpoint_interval_defaults_to_120_and_is_overridable(self):
        """Ignoring the interval would increase checkpoint I/O without authorization."""
        default = run_ablation_epoch(
            protocol_version="ablation-v1", runtime=_ToyRuntime(), total_steps=121,
            checkpoint_store=self._store("default-interval"),
        )
        override = run_ablation_epoch(
            protocol_version="ablation-v1", runtime=_ToyRuntime(), total_steps=5,
            checkpoint_store=self._store("overridden-interval"), checkpoint_interval=2,
        )

        self.assertEqual(default.checkpoint_steps, [120])
        self.assertEqual(override.checkpoint_steps, [2, 4])

    def test_checkpoint_callback_runs_only_after_durable_save(self):
        observed = []
        store = self._store("callback")

        run_ablation_epoch(
            protocol_version="ablation-v1", runtime=_ToyRuntime(), total_steps=3,
            checkpoint_store=store, checkpoint_interval=1,
            on_checkpoint=lambda measurement: observed.append((measurement.step_index, store.load()["step_index"])),
        )

        self.assertEqual(observed, [(1, 1), (2, 2), (3, 3)])

    def test_frozen_attempt_one_protocol_is_rejected(self):
        """Allowing Attempt-1 here would silently change its frozen recovery contract."""
        with self.assertRaisesRegex(ValueError, "ablation-only"):
            run_ablation_epoch(
                protocol_version="phase1-2026-08-29", runtime=_ToyRuntime(), total_steps=1,
                checkpoint_store=self._store("attempt-one"), checkpoint_interval=1,
            )

    def test_rejects_a_checkpoint_store_with_a_different_protocol_version(self):
        """A mismatched store signature could otherwise relabel Attempt-1 recovery state."""
        attempt_one_signature = StageSignature.create(
            manifest_digest="sha256:manifest", protocol_version="phase1-2026-08-29",
            upstream_commit="a" * 40, upstream_tree="b" * 40, model_revision="c" * 40,
            seed=17, stage="training", epoch=1, checkpoint_digest="sha256:base",
            effective_evaluation_config={}, expected_example_ids=[],
        )
        store = MidEpochCheckpointStore(
            self.workspace, "mismatched-protocol", attempt_one_signature,
        )

        with self.assertRaisesRegex(ValueError, "checkpoint signature protocol"):
            run_ablation_epoch(
                protocol_version="ablation-v1", runtime=_ToyRuntime(), total_steps=1,
                checkpoint_store=store, checkpoint_interval=1,
            )


if __name__ == "__main__":
    unittest.main()
