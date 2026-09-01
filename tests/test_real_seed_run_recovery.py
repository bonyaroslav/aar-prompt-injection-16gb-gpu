"""Issue #22: the recovery-aware split-run seam in `runner.real_seed_run`.

Exercised through the high-level `_orchestrate_seed` / `seed_run_status` /
`discover_finalized_seed_evidence` interface with the repository's deterministic
fake adapters -- no GPU, no model weights, no held-out material. The per-stage
recovery contracts themselves are covered by tests/test_training.py,
tests/test_evaluation.py and tests/test_reveal.py.
"""
import json
import tempfile
import unittest
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner.fakes import (
    FakeDatasetAdapter,
    FakeModelAdapter,
    FakeScorerAdapter,
    FakeTelemetryAdapter,
    FakeTrainerAdapter,
)
from runner.recovery import AttemptLedger
from runner.storage import LocalStorageAdapter
from runner.real_seed_run import (
    _orchestrate_seed,
    aggregate_seed_resource_intervals,
    discover_finalized_seed_evidence,
    seed_run_status,
)

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"
FROZEN = load_manifest(MANIFEST)
SEED = FROZEN["training"]["seeds"][1]  # 42
EPOCHS = FROZEN["training"]["optimizer"]["epochs"]


def _bench(value):
    return {"aggregate": {"value": value}}


def _null_selection_baseline():
    """Capability baselines pinned at 1.0 so any trained decline fails the gate:
    the selection finalizes null, matching the seed-17 precedent for seed 42."""
    return {
        name: _bench(0.5)
        for name in ("open_prompt_injection", "tensor_trust_hijack", "tensor_trust_extract")
    } | {name: _bench(1.0) for name in ("mmlu", "gsm8k", "ifeval")}


def _eligible_selection_baseline():
    """Capability baselines pinned very low so the fake trained scores retain
    well above every gate -- forces a non-null, eligible selection."""
    return {
        name: _bench(0.1)
        for name in ("open_prompt_injection", "tensor_trust_hijack", "tensor_trust_extract")
    } | {name: _bench(0.01) for name in ("mmlu", "gsm8k", "ifeval")}


class _InterruptingTrainer(FakeTrainerAdapter):
    def __init__(self, interrupt_epoch=None):
        super().__init__()
        self.interrupt_epoch = interrupt_epoch
        self.trained_epochs = []

    def train_epoch(self, *, seed, epoch, sequence_length, config):
        if epoch == self.interrupt_epoch:
            raise RuntimeError(f"injected interruption at epoch {epoch}")
        self.trained_epochs.append(epoch)
        return super().train_epoch(seed=seed, epoch=epoch, sequence_length=sequence_length, config=config)


class RecoveryAwareSeedRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.output_root = root / "runs"
        self.recovery_root = root / "recovery"
        self.storage = LocalStorageAdapter(self.output_root)

    def _orchestrate(self, trainer, *, baseline=None, seed=SEED, heldout=None, unavailable_intervals=()):
        return _orchestrate_seed(
            manifest_path=MANIFEST, frozen=FROZEN,
            baseline_benchmarks=baseline or _null_selection_baseline(),
            output_root=self.output_root, seed=seed, prior_cumulative_gpu_hours=21.44,
            reproduction_command="pytest", upstream_provenance={"commit": "x"},
            storage=self.storage, trainer=trainer,
            make_stage_telemetry=FakeTelemetryAdapter,
            make_eval_model=lambda checkpoint: FakeModelAdapter(),
            make_eval_dataset=FakeDatasetAdapter,
            scorer=FakeScorerAdapter(),
            recovery_root=self.recovery_root, heldout=heldout,
            unavailable_intervals=unavailable_intervals, stamp="20260901-000000",
        )

    def test_uninterrupted_run_produces_full_finalized_topology(self):
        result = self._orchestrate(FakeTrainerAdapter())
        self.assertEqual(result["training"].outcome, "success")
        self.assertEqual(sorted(result["evaluations"]), [1, 2, 3])
        self.assertIsNone(result["selection"]["record"]["selected_checkpoint_digest"])
        self.assertIsNone(result["reveal"])
        discovery = result["discovery"]
        self.assertEqual(len(discovery["evaluation_bundles"]), 3)
        self.assertIsNone(discovery["reveal_bundle"])
        self.assertTrue(discovery["recovery_state_excluded"])
        # Recovery workspace is outside the evidence root.
        self.assertFalseIfInside(discovery["training_bundle"])

    def assertFalseIfInside(self, path):
        self.assertNotIn(str(self.recovery_root.resolve()), str(Path(path).resolve()))

    def test_interrupted_training_resumes_without_retraining_completed_epochs(self):
        with self.assertRaises(RuntimeError):
            self._orchestrate(_InterruptingTrainer(interrupt_epoch=2))
        status = seed_run_status(MANIFEST, recovery_root=self.recovery_root,
                                 output_root=self.output_root, seed=SEED)
        self.assertTrue(status["next_action"].startswith("resume-training"))

        resumed_trainer = _InterruptingTrainer(interrupt_epoch=None)
        result = self._orchestrate(resumed_trainer)
        # epoch 1 was completed on the first attempt; the resume trains only 2 and 3.
        self.assertEqual(resumed_trainer.trained_epochs, [2, 3])
        self.assertEqual(sorted(result["evaluations"]), [1, 2, 3])
        self.assertEqual(result["discovery"]["recovery_state_excluded"], True)

    def test_resumed_seed_matches_uninterrupted_seed_final_selection(self):
        uninterrupted = self._orchestrate(FakeTrainerAdapter())
        uninterrupted_digest = uninterrupted["selection"]["digest"]

        # Fresh workspace, same seed, interrupted mid-eval then resumed.
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        root = Path(other.name)
        self.output_root = root / "runs"
        self.recovery_root = root / "recovery"
        self.storage = LocalStorageAdapter(self.output_root)

        class _EvalInterrupt(FakeModelAdapter):
            calls = {"n": 0}

            def generate(self, benchmark, item, config):
                _EvalInterrupt.calls["n"] += 1
                if _EvalInterrupt.calls["n"] == 5:
                    raise RuntimeError("injected eval interruption")
                return super().generate(benchmark, item, config)

        with self.assertRaises(RuntimeError):
            _orchestrate_seed(
                manifest_path=MANIFEST, frozen=FROZEN, baseline_benchmarks=_null_selection_baseline(),
                output_root=self.output_root, seed=SEED, prior_cumulative_gpu_hours=21.44,
                reproduction_command="pytest", upstream_provenance={"commit": "x"},
                storage=self.storage, trainer=FakeTrainerAdapter(),
                make_stage_telemetry=FakeTelemetryAdapter,
                make_eval_model=lambda checkpoint: _EvalInterrupt(),
                make_eval_dataset=FakeDatasetAdapter, scorer=FakeScorerAdapter(),
                recovery_root=self.recovery_root, stamp="20260901-000000",
            )
        resumed = self._orchestrate(FakeTrainerAdapter())
        self.assertEqual(resumed["selection"]["digest"], uninterrupted_digest)

    def test_resource_intervals_record_completed_interrupted_and_unavailable(self):
        with self.assertRaises(RuntimeError):
            self._orchestrate(_InterruptingTrainer(interrupt_epoch=3))
        result = self._orchestrate(
            _InterruptingTrainer(interrupt_epoch=None),
            unavailable_intervals=[{"seconds": 1800.0, "reason": "wsl shutdown"}],
        )
        intervals = result["resource_comparison"]["resource_intervals"]
        self.assertGreaterEqual(intervals["interrupted_attempt_count"], 1)
        self.assertEqual(intervals["unavailable_seconds"], 1800.0)
        self.assertEqual(
            intervals["elapsed_including_unavailable_seconds"],
            intervals["active_wall_seconds"] + 1800.0,
        )
        # Cumulative GPU-hours never silently absorb the unavailable gap.
        self.assertNotIn("unavailable", str(result["resource_comparison"]["measured"]["gpu_hours"]))

    def test_aggregate_only_counts_this_seed(self):
        self._orchestrate(FakeTrainerAdapter(), seed=SEED)
        other_seed = FROZEN["training"]["seeds"][2]  # 2026
        agg_42 = aggregate_seed_resource_intervals(self.recovery_root, seed=SEED, epochs=EPOCHS)
        agg_2026 = aggregate_seed_resource_intervals(self.recovery_root, seed=other_seed, epochs=EPOCHS)
        self.assertEqual(agg_42["attempt_count"], EPOCHS + 1)  # training + 3 evals
        self.assertEqual(agg_2026["attempt_count"], 0)

    def test_discover_rejects_a_recovery_workspace_path(self):
        self._orchestrate(FakeTrainerAdapter())
        # Sanity: discovery passes for the real evidence root.
        discovery = discover_finalized_seed_evidence(
            MANIFEST, recovery_root=self.recovery_root, output_root=self.output_root, seed=SEED,
        )
        self.assertEqual(discovery["seed"], SEED)
        for bundle in [discovery["training_bundle"], *discovery["evaluation_bundles"]]:
            self.assertNotIn("recovery", Path(bundle).relative_to(self.output_root).parts)

    def test_status_progresses_to_complete(self):
        self._orchestrate(FakeTrainerAdapter())
        status = seed_run_status(MANIFEST, recovery_root=self.recovery_root,
                                 output_root=self.output_root, seed=SEED)
        self.assertEqual(status["next_action"], "complete")
        self.assertTrue(status["selection"]["finalized"])
        self.assertTrue(status["resource_comparison_finalized"])
        self.assertTrue(all(s["status"] == "completed" for s in status["stages"]))

    def test_eligible_selection_without_sealed_heldout_halts(self):
        with self.assertRaisesRegex(RuntimeError, "held-out authorization"):
            self._orchestrate(FakeTrainerAdapter(), baseline=_eligible_selection_baseline())

    def test_rerun_of_completed_seed_is_idempotent(self):
        first = self._orchestrate(FakeTrainerAdapter())
        second = self._orchestrate(FakeTrainerAdapter())
        self.assertEqual(first["selection"]["digest"], second["selection"]["digest"])
        self.assertEqual(
            first["resource_comparison"]["cumulative_gpu_hours"],
            second["resource_comparison"]["cumulative_gpu_hours"],
        )


if __name__ == "__main__":
    unittest.main()
