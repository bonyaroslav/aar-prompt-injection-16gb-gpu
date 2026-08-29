import json, unittest
from pathlib import Path

from runner.analysis import (
    paired_bootstrap_ci, bootstrap_benchmark_difference, bootstrap_visible_composite, summarize_seeds,
)

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"


def items(scores: dict) -> dict:
    """Build a `metrics.json`-shaped benchmark dict from {id: score}."""
    return {"items": {example_id: {"score": score} for example_id, score in scores.items()}}


class PairedBootstrapCiTests(unittest.TestCase):
    def test_constant_diff_collapses_ci_to_the_exact_known_point(self):
        # Every candidate score exceeds baseline by exactly 0.2, for every ID.
        # Every possible bootstrap resample -- whatever IDs it draws -- is still
        # a mean of values all equal to 0.2, so the *entire* replicate
        # distribution is the single point 0.2: this is derivable by hand, not
        # just "it ran".
        baseline = {f"id-{i}": 0.5 for i in range(20)}
        candidate = {f"id-{i}": 0.7 for i in range(20)}
        result = paired_bootstrap_ci(baseline, candidate, seed=271828, replicates=500)
        self.assertAlmostEqual(result["observed_difference"], 0.2)
        self.assertAlmostEqual(result["ci_low"], 0.2)
        self.assertAlmostEqual(result["ci_high"], 0.2)
        self.assertEqual(result["n"], 20)
        self.assertEqual(result["interval"], "95_percentile_paired_by_fixed_example_id")

    def test_observed_difference_is_the_plain_paired_mean(self):
        baseline = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
        candidate = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}
        result = paired_bootstrap_ci(baseline, candidate, seed=1, replicates=200)
        self.assertAlmostEqual(result["observed_difference"], 0.25)

    def test_two_point_extreme_diffs_bound_the_ci_within_the_known_support(self):
        # diffs are {0, 1}; every bootstrap resample mean is one of 0, 0.5, 1 --
        # so the CI can never fall outside [0, 1], a hand-derivable bound.
        baseline = {"a": 0.0, "b": 0.0}
        candidate = {"a": 0.0, "b": 1.0}
        result = paired_bootstrap_ci(baseline, candidate, seed=271828, replicates=10000)
        self.assertGreaterEqual(result["ci_low"], 0.0)
        self.assertLessEqual(result["ci_high"], 1.0)
        self.assertIn(result["ci_low"], (0.0, 0.5))
        self.assertIn(result["ci_high"], (0.5, 1.0))

    def test_deterministic_same_seed_same_fixtures_byte_identical(self):
        baseline = {f"id-{i}": 0.4 + 0.01 * i for i in range(15)}
        candidate = {f"id-{i}": 0.5 + 0.02 * i for i in range(15)}
        first = paired_bootstrap_ci(baseline, candidate, seed=271828, replicates=1000)
        second = paired_bootstrap_ci(baseline, candidate, seed=271828, replicates=1000)
        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )

    def test_different_seed_can_change_the_ci_but_not_the_observed_value(self):
        baseline = {f"id-{i}": 0.4 + 0.01 * i for i in range(15)}
        candidate = {f"id-{i}": 0.5 + 0.02 * i for i in range(15)}
        a = paired_bootstrap_ci(baseline, candidate, seed=1, replicates=1000)
        b = paired_bootstrap_ci(baseline, candidate, seed=2, replicates=1000)
        self.assertAlmostEqual(a["observed_difference"], b["observed_difference"])

    def test_mismatched_ids_are_rejected(self):
        baseline = {"a": 0.5, "b": 0.5}
        candidate = {"a": 0.6, "c": 0.6}
        with self.assertRaisesRegex(ValueError, "same fixed example IDs"):
            paired_bootstrap_ci(baseline, candidate, seed=1, replicates=100)

    def test_empty_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no example IDs"):
            paired_bootstrap_ci({}, {}, seed=1, replicates=100)


class BootstrapBenchmarkDifferenceTests(unittest.TestCase):
    def test_reads_seed_and_replicates_from_the_frozen_manifest(self):
        from protocol.validate_manifest import load as load_manifest
        manifest = load_manifest(MANIFEST)
        baseline = items({f"id-{i}": 0.5 for i in range(10)})
        candidate = items({f"id-{i}": 0.6 for i in range(10)})
        result = bootstrap_benchmark_difference(
            MANIFEST, name="open_prompt_injection", baseline_benchmark=baseline, candidate_benchmark=candidate,
        )
        self.assertEqual(result["benchmark"], "open_prompt_injection")
        self.assertEqual(result["seed"], manifest["analysis"]["bootstrap_seed"])
        self.assertEqual(result["replicates"], manifest["analysis"]["bootstrap_replicates"])
        self.assertAlmostEqual(result["ci_low"], 0.1)
        self.assertAlmostEqual(result["ci_high"], 0.1)


class BootstrapVisibleCompositeTests(unittest.TestCase):
    def test_constant_per_benchmark_diffs_give_an_exact_known_composite(self):
        # Each benchmark's diff is a different constant (0.1 / 0.2 / 0.3), so
        # every replicate composite is exactly the mean of those three
        # constants regardless of which IDs get resampled -- a hand-derivable
        # point, same collapsing argument as the single-benchmark case.
        baseline_benchmarks = {
            "open_prompt_injection": items({f"id-{i}": 0.5 for i in range(10)}),
            "tensor_trust_hijack": items({f"id-{i}": 0.5 for i in range(10)}),
            "tensor_trust_extract": items({f"id-{i}": 0.5 for i in range(10)}),
        }
        candidate_benchmarks = {
            "open_prompt_injection": items({f"id-{i}": 0.6 for i in range(10)}),
            "tensor_trust_hijack": items({f"id-{i}": 0.7 for i in range(10)}),
            "tensor_trust_extract": items({f"id-{i}": 0.8 for i in range(10)}),
        }
        result = bootstrap_visible_composite(
            MANIFEST, baseline_benchmarks=baseline_benchmarks, candidate_benchmarks=candidate_benchmarks,
        )
        expected = (0.1 + 0.2 + 0.3) / 3
        self.assertAlmostEqual(result["observed_difference"], expected)
        self.assertAlmostEqual(result["ci_low"], expected)
        self.assertAlmostEqual(result["ci_high"], expected)
        self.assertEqual(result["n"], {"open_prompt_injection": 10, "tensor_trust_hijack": 10, "tensor_trust_extract": 10})

    def test_mismatched_ids_in_one_benchmark_are_rejected(self):
        baseline_benchmarks = {
            "open_prompt_injection": items({"a": 0.5, "b": 0.5}),
            "tensor_trust_hijack": items({"a": 0.5, "b": 0.5}),
            "tensor_trust_extract": items({"a": 0.5, "b": 0.5}),
        }
        candidate_benchmarks = {
            "open_prompt_injection": items({"a": 0.6, "b": 0.6}),
            "tensor_trust_hijack": items({"a": 0.6, "c": 0.6}),  # id mismatch
            "tensor_trust_extract": items({"a": 0.6, "b": 0.6}),
        }
        with self.assertRaisesRegex(ValueError, "tensor_trust_hijack"):
            bootstrap_visible_composite(
                MANIFEST, baseline_benchmarks=baseline_benchmarks, candidate_benchmarks=candidate_benchmarks,
            )


class SeedSummaryReExportTests(unittest.TestCase):
    def test_summarize_seeds_is_available_from_the_analysis_module(self):
        # Re-exported from runner.continuation, not reimplemented here.
        summary = summarize_seeds(MANIFEST, {17: 0.10, 42: 0.14, 2026: 0.06})
        self.assertIn("not a population-level confidence interval", summary["framing"])
        self.assertNotIn("ci", summary)
        self.assertNotIn("confidence_interval", summary)


if __name__ == "__main__":
    unittest.main()
