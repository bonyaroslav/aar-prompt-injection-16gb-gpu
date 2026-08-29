import tempfile, unittest
from pathlib import Path

from runner.core import run_baseline, resolve_sample_count
from runner.fakes import (
    FakeModelAdapter, FakeDatasetAdapter, FakeScorerAdapter,
    FakeTelemetryAdapter, PUBLISHER_OPI_DEFAULT,
)
from runner.storage import LocalStorageAdapter
from runner.bundle import verify_bundle, BUNDLE_FILES, CHECKSUM_FILE

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"

class RunnerBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.storage = LocalStorageAdapter(self.tmp.name)

    def _run(self, run_id="baseline-test-001"):
        return run_baseline(
            MANIFEST,
            model=FakeModelAdapter(),
            dataset=FakeDatasetAdapter(),
            scorer=FakeScorerAdapter(),
            telemetry=FakeTelemetryAdapter(),
            storage=self.storage,
            run_id=run_id,
        )

    def test_resolve_sample_count_reads_manifest_declared_value(self):
        self.assertEqual(resolve_sample_count("publisher_seed_42_first_300;100_each_injected_task"), 300)
        self.assertEqual(resolve_sample_count("capability_publisher_seed_42_first_200"), 200)

    def test_returns_structured_result_not_console_text(self):
        result = self._run()
        self.assertEqual(result.stage, "baseline")
        self.assertIsInstance(result.metrics, dict)
        self.assertIsInstance(result.checksums, dict)
        self.assertTrue(Path(result.bundle_dir).is_dir())

    def test_bundle_contains_required_files(self):
        result = self._run()
        bundle_dir = Path(result.bundle_dir)
        for name in (*BUNDLE_FILES, CHECKSUM_FILE):
            self.assertTrue((bundle_dir / name).exists(), f"missing {name}")

    def test_metrics_keyed_by_fixed_example_id(self):
        result = self._run()
        opi_items = result.metrics["benchmarks"]["open_prompt_injection"]["items"]
        self.assertIn("open_prompt_injection-0000", opi_items)
        self.assertIn("score", opi_items["open_prompt_injection-0000"])

    def test_opi_sample_count_resolves_to_manifest_value_not_publisher_default(self):
        # The fake mirrors the real publisher's own buggy default (210)...
        self.assertEqual(len(FakeDatasetAdapter().load_open_prompt_injection()), PUBLISHER_OPI_DEFAULT)
        # ...but the runner must never rely on that default: manifest declares 300.
        result = self._run()
        self.assertEqual(len(result.metrics["benchmarks"]["open_prompt_injection"]["items"]), 300)

    def test_finalized_bundle_verifies_clean(self):
        result = self._run()
        verify_bundle(Path(result.bundle_dir))  # must not raise

    def test_mutation_after_finalize_is_detected_via_checksum_mismatch(self):
        result = self._run()
        target = Path(result.bundle_dir) / "metrics.json"
        target.chmod(0o644)
        target.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_bundle(Path(result.bundle_dir))

    def test_second_run_with_same_run_id_is_rejected(self):
        self._run(run_id="baseline-dup")
        with self.assertRaises(FileExistsError):
            self._run(run_id="baseline-dup")

    def test_no_gpu_model_or_network_dependency_in_fakes(self):
        import runner.fakes as fakes_mod
        src = Path(fakes_mod.__file__).read_text()
        for banned in ("torch", "transformers", "requests", "huggingface_hub", "socket"):
            self.assertNotIn(banned, src)

if __name__ == "__main__":
    unittest.main()
