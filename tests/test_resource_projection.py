import unittest
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner.resource_projection import project_full_run_resources, FULL_EVAL_ITEM_COUNTS

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"


class ProjectFullRunResourcesTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(MANIFEST)

    def test_comfortably_within_all_limits_reports_no_findings(self):
        result = project_full_run_resources(
            self.manifest,
            measured_seconds_per_item={name: 0.01 for name in FULL_EVAL_ITEM_COUNTS},
            measured_peak_vram_mb=4096,
            measured_train_seconds_per_step=0.1,
            measured_checkpoint_bytes=10 * 1024 ** 2,
            default_seconds_per_item=0.01,
        )
        self.assertEqual(result["feasibility_findings"], [])
        self.assertEqual(result["assumed_default_seconds_per_item_for"], [])

    def test_missing_benchmark_measurement_falls_back_to_default_and_is_recorded(self):
        measured = {name: 0.01 for name in FULL_EVAL_ITEM_COUNTS if name != "gsm8k"}
        result = project_full_run_resources(
            self.manifest, measured_seconds_per_item=measured, measured_peak_vram_mb=4096,
            measured_train_seconds_per_step=0.1, measured_checkpoint_bytes=10 * 1024 ** 2,
            default_seconds_per_item=0.02,
        )
        self.assertEqual(result["assumed_default_seconds_per_item_for"], ["gsm8k"])
        self.assertAlmostEqual(result["per_benchmark_seconds"]["gsm8k"], 200 * 0.02)

    def test_excessive_vram_is_flagged_as_a_feasibility_finding(self):
        result = project_full_run_resources(
            self.manifest,
            measured_seconds_per_item={name: 0.01 for name in FULL_EVAL_ITEM_COUNTS},
            measured_peak_vram_mb=20000,  # 19.5 GB > the manifest's 15.5 GB max
            measured_train_seconds_per_step=0.1,
            measured_checkpoint_bytes=10 * 1024 ** 2,
            default_seconds_per_item=0.01,
        )
        self.assertTrue(any("VRAM" in f for f in result["feasibility_findings"]))

    def test_excessive_wall_time_is_flagged_and_arithmetic_is_correct(self):
        # Force a large per-item cost so projected per-seed wall time blows past 24h.
        result = project_full_run_resources(
            self.manifest,
            measured_seconds_per_item={name: 15.0 for name in FULL_EVAL_ITEM_COUNTS},
            measured_peak_vram_mb=4096,
            measured_train_seconds_per_step=0.1,
            measured_checkpoint_bytes=10 * 1024 ** 2,
            default_seconds_per_item=15.0,
        )
        total_items = sum(FULL_EVAL_ITEM_COUNTS.values())
        expected_eval_pass_seconds = total_items * 15.0
        self.assertAlmostEqual(result["baseline_wall_seconds"], expected_eval_pass_seconds)
        epochs = self.manifest["training"]["optimizer"]["epochs"]
        n_seeds = len(self.manifest["training"]["seeds"])
        self.assertAlmostEqual(result["trained_eval_wall_seconds"], expected_eval_pass_seconds * epochs * n_seeds)
        self.assertTrue(any("wall time" in f for f in result["feasibility_findings"]))

    def test_excessive_storage_is_flagged(self):
        result = project_full_run_resources(
            self.manifest,
            measured_seconds_per_item={name: 0.01 for name in FULL_EVAL_ITEM_COUNTS},
            measured_peak_vram_mb=4096,
            measured_train_seconds_per_step=0.1,
            measured_checkpoint_bytes=50 * 1024 ** 3,  # 50 GB per checkpoint -> way over budget
            default_seconds_per_item=0.01,
        )
        self.assertTrue(any("storage" in f for f in result["feasibility_findings"]))
        epochs = self.manifest["training"]["optimizer"]["epochs"]
        n_seeds = len(self.manifest["training"]["seeds"])
        self.assertEqual(result["checkpoints_total"], epochs * n_seeds)

    def test_limits_in_output_match_the_manifest(self):
        result = project_full_run_resources(
            self.manifest,
            measured_seconds_per_item={name: 0.01 for name in FULL_EVAL_ITEM_COUNTS},
            measured_peak_vram_mb=4096,
            measured_train_seconds_per_step=0.1,
            measured_checkpoint_bytes=10 * 1024 ** 2,
            default_seconds_per_item=0.01,
        )
        self.assertEqual(result["limits"]["vram_allocated_gb_max"], self.manifest["resources"]["vram_allocated_gb_max"])
        self.assertEqual(result["limits"]["gpu_hours_total_max"], self.manifest["resources"]["gpu_hours_total_max"])


if __name__ == "__main__":
    unittest.main()
