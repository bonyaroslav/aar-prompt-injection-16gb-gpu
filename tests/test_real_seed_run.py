import json
import tempfile
import unittest
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner.real_seed_run import (
    build_candidates,
    compare_seed_resource_use,
    load_baseline_benchmarks,
    load_training_examples,
    seed_notes_text,
    write_seed_comparison_artifact,
)

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"


def _bench(value):
    return {"aggregate": {"value": value}}


def _benchmarks(**values):
    return {name: _bench(value) for name, value in values.items()}


class LoadTrainingExamplesTests(unittest.TestCase):
    def test_reads_jsonl_records_with_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.jsonl"
            records = [
                {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], "category": "clean_control"},
                {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], "category": "refusal_calibration"},
            ]
            path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            examples = load_training_examples(path)
            self.assertEqual(len(examples), 2)
            self.assertEqual(examples[0]["messages"][-1]["content"], "hello")
            self.assertEqual(examples[0]["category"], "clean_control")

    def test_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.jsonl"
            record = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
            path.write_text(json.dumps(record) + "\n\n\n", encoding="utf-8")
            self.assertEqual(len(load_training_examples(path)), 1)

    def test_rejects_record_missing_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.jsonl"
            path.write_text(json.dumps({"category": "x"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing 'messages'"):
                load_training_examples(path)

    def test_rejects_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no training examples"):
                load_training_examples(path)


class LoadBaselineBenchmarksTests(unittest.TestCase):
    def test_extracts_benchmarks_from_a_baseline_metrics_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            benchmarks = _benchmarks(open_prompt_injection=0.5, tensor_trust_hijack=0.5, tensor_trust_extract=0.5,
                                     mmlu=0.6, gsm8k=0.6, ifeval=0.6)
            path.write_text(json.dumps({"stage": "baseline", "benchmarks": benchmarks}), encoding="utf-8")
            self.assertEqual(load_baseline_benchmarks(path), benchmarks)

    def test_rejects_metrics_file_that_is_not_a_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(json.dumps({"stage": "training", "benchmarks": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a baseline metrics.json"):
                load_baseline_benchmarks(path)


class BuildCandidatesTests(unittest.TestCase):
    def test_shapes_epoch_metrics_into_selection_candidates_sorted_by_epoch(self):
        epoch_metrics = {
            2: {"checkpoint": "sha256:epoch2", "benchmarks": {"a": 1}},
            1: {"checkpoint": "sha256:epoch1", "benchmarks": {"a": 0}},
        }
        candidates = build_candidates(epoch_metrics)
        self.assertEqual([c["epoch"] for c in candidates], [1, 2])
        self.assertEqual(candidates[0]["checkpoint_digest"], "sha256:epoch1")
        self.assertEqual(candidates[0]["benchmarks"], {"a": 0})


class CompareSeedResourceUseTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(MANIFEST)

    def test_within_limits_has_no_findings_and_accumulates_prior_gpu_hours(self):
        comparison = compare_seed_resource_use(
            self.manifest, wall_seconds=10 * 3600.0, peak_vram_mb=14000.0,
            bundle_bytes=1_000_000, prior_cumulative_gpu_hours=6.26,
        )
        self.assertEqual(comparison["feasibility_findings"], [])
        self.assertAlmostEqual(comparison["measured"]["gpu_hours"], 10.0)
        self.assertAlmostEqual(comparison["cumulative_gpu_hours"], 16.26)
        self.assertEqual(comparison["prior_cumulative_gpu_hours"], 6.26)

    def test_exceeding_per_seed_wall_hours_is_a_finding(self):
        comparison = compare_seed_resource_use(
            self.manifest, wall_seconds=25 * 3600.0, peak_vram_mb=1000.0,
            bundle_bytes=0, prior_cumulative_gpu_hours=0.0,
        )
        self.assertTrue(any("wall_hours" in finding for finding in comparison["feasibility_findings"]))

    def test_exceeding_cumulative_gpu_hours_is_a_finding_even_when_this_run_alone_is_small(self):
        comparison = compare_seed_resource_use(
            self.manifest, wall_seconds=1 * 3600.0, peak_vram_mb=1000.0,
            bundle_bytes=0, prior_cumulative_gpu_hours=71.5,
        )
        self.assertTrue(any("cumulative gpu_hours" in finding for finding in comparison["feasibility_findings"]))

    def test_exceeding_vram_limit_is_a_finding(self):
        comparison = compare_seed_resource_use(
            self.manifest, wall_seconds=3600.0, peak_vram_mb=16_000.0,
            bundle_bytes=0, prior_cumulative_gpu_hours=0.0,
        )
        self.assertTrue(any("peak_vram_gb" in finding for finding in comparison["feasibility_findings"]))


class WriteSeedComparisonArtifactTests(unittest.TestCase):
    def test_writes_checksummed_json_and_rejects_a_second_write(self):
        comparison = {"feasibility_findings": []}
        with tempfile.TemporaryDirectory() as tmp:
            artifact = write_seed_comparison_artifact(Path(tmp), "seed17-comparison", comparison)
            self.assertEqual(
                json.loads((artifact / "seed_resource_comparison.json").read_text()), comparison
            )
            self.assertRegex(
                (artifact / "checksums.sha256").read_text(), r"^[0-9a-f]{64}  seed_resource_comparison\.json\n$"
            )
            with self.assertRaises(FileExistsError):
                write_seed_comparison_artifact(Path(tmp), "seed17-comparison", comparison)


class SeedNotesTextTests(unittest.TestCase):
    def test_states_held_out_is_not_read_or_revealed(self):
        notes = seed_notes_text("training")
        self.assertIn("Held-out InjecAgent is not read or revealed by this run.", notes)
        self.assertIn("Training", notes.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
