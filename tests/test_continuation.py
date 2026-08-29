import inspect, re, tempfile, unittest
from pathlib import Path

from runner.bundle import verify_bundle
from runner.continuation import decide_continuation, summarize_seeds
from runner.fakes import FakeTelemetryAdapter, FakeTrainerAdapter
from runner.storage import LocalStorageAdapter
from runner.training import run_training

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"

# Budget math shared by the "over budget" tests: manifest declares 3 seeds and a
# 72 GPU-hour total budget, so 25 GPU-hours for seed 1 alone projects to 75 -- over.
OVER_BUDGET_GPU_HOURS = 25.0
UNDER_BUDGET_GPU_HOURS = 20.0


class DecideContinuationTests(unittest.TestCase):
    def test_technically_sound_but_arbitrarily_poor_quality_still_continues(self):
        for visible_composite, held_out in [(-0.9, 0.0), (0.0, 0.0), (-1.0, -1.0)]:
            seed1_result = {
                "outcome": "success", "gpu_hours": UNDER_BUDGET_GPU_HOURS,
                "visible_composite": visible_composite, "held_out": held_out,
            }
            decision = decide_continuation(MANIFEST, seed1_result)
            self.assertTrue(decision.continue_replication, f"quality={visible_composite},{held_out} should not block continuation")
            self.assertEqual(decision.reasons, [])

    def test_unrecoverable_technical_failure_stops_continuation(self):
        seed1_result = {"outcome": "failed", "gpu_hours": UNDER_BUDGET_GPU_HOURS}
        decision = decide_continuation(MANIFEST, seed1_result)
        self.assertFalse(decision.continue_replication)
        self.assertTrue(any("technically" in reason for reason in decision.reasons))

    def test_projected_cost_over_budget_stops_continuation_without_touching_evidence(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        storage = LocalStorageAdapter(tmp.name)
        result = run_training(
            MANIFEST, trainer=FakeTrainerAdapter(), telemetry=FakeTelemetryAdapter(),
            storage=storage, seed=17, run_id="seed17-continuation-test",
        )
        self.assertEqual(result.outcome, "success")
        bundle_dir = Path(result.bundle_dir)
        checksums_before = (bundle_dir / "checksums.sha256").read_text()

        seed1_result = {"outcome": result.outcome, "gpu_hours": OVER_BUDGET_GPU_HOURS}
        decision = decide_continuation(MANIFEST, seed1_result)
        self.assertFalse(decision.continue_replication)
        self.assertTrue(any("GPU-hour" in reason for reason in decision.reasons))
        self.assertAlmostEqual(decision.projected_gpu_hours, OVER_BUDGET_GPU_HOURS * 3)

        # The decision must not delete, rewrite, or hide the seed-1 evidence bundle.
        verify_bundle(bundle_dir)  # must not raise
        self.assertEqual((bundle_dir / "checksums.sha256").read_text(), checksums_before)

    def test_decision_never_reads_a_quality_field_off_seed1_result(self):
        source = inspect.getsource(decide_continuation)
        keys_read = set(re.findall(r"seed1_result\[[\"']([\w_]+)[\"']\]", source))
        self.assertEqual(keys_read, {"outcome", "gpu_hours"})


class SummarizeSeedsTests(unittest.TestCase):
    def test_summary_has_manifest_declared_fields_and_is_framed_descriptively(self):
        summary = summarize_seeds(MANIFEST, {17: 0.10, 42: 0.14, 2026: 0.06})
        self.assertEqual(summary["per_seed"], {17: 0.10, 42: 0.14, 2026: 0.06})
        self.assertAlmostEqual(summary["mean"], (0.10 + 0.14 + 0.06) / 3)
        self.assertEqual(summary["range"], {"min": 0.06, "max": 0.14})
        self.assertGreater(summary["standard_deviation"], 0)
        self.assertNotIn("confidence_interval", summary)
        self.assertNotIn("ci", summary)
        self.assertIn("not a population-level confidence interval", summary["framing"])

    def test_single_seed_summary_has_zero_spread(self):
        summary = summarize_seeds(MANIFEST, {17: 0.10})
        self.assertEqual(summary["mean"], 0.10)
        self.assertEqual(summary["range"], {"min": 0.10, "max": 0.10})
        self.assertEqual(summary["standard_deviation"], 0.0)


if __name__ == "__main__":
    unittest.main()
