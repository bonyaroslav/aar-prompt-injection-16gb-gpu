import json, tempfile, unittest
from pathlib import Path

from runner.core import run_baseline, resolve_sample_count, read_held_out_result
from runner.fakes import (
    FakeModelAdapter, FakeDatasetAdapter, FakeScorerAdapter,
    FakeTelemetryAdapter, PUBLISHER_OPI_DEFAULT,
)
from runner.storage import LocalStorageAdapter
from runner.bundle import verify_bundle, BUNDLE_FILES, CHECKSUM_FILE
from protocol.heldout import HeldOutSealer

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"

class RunnerBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.storage = LocalStorageAdapter(self.tmp.name)
        # Held-out root is deliberately a separate temp dir from run-bundle storage,
        # mirroring the requirement that it never live under runs/ or the bundle.
        self.heldout_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.heldout_tmp.cleanup)
        self.sealer = HeldOutSealer(self.heldout_tmp.name)

    def _run(self, run_id="baseline-test-001", sealer=None):
        return run_baseline(
            MANIFEST,
            model=FakeModelAdapter(),
            dataset=FakeDatasetAdapter(),
            scorer=FakeScorerAdapter(),
            telemetry=FakeTelemetryAdapter(),
            storage=self.storage,
            held_out_sealer=sealer or self.sealer,
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

    def test_injecagent_result_never_appears_as_plaintext_in_bundle(self):
        result = self._run()
        held_out = result.metrics["held_out"]["injecagent"]
        self.assertEqual(set(held_out.keys()), {"receipt", "commitments"})
        self.assertEqual(set(held_out["receipt"].keys()), {"label", "digest", "valid", "invalid"})
        self.assertEqual(held_out["receipt"]["label"], "baseline")
        self.assertEqual(held_out["receipt"]["valid"] + held_out["receipt"]["invalid"], 200)
        # commitments are digests, never the plaintext candidate list or validity text
        self.assertEqual(set(held_out["commitments"].keys()), {"state", "candidates", "validity"})
        for key in ("candidates", "validity"):
            self.assertRegex(held_out["commitments"][key], r"^[0-9a-f]{64}$")

        bundle_metrics = json.loads((Path(result.bundle_dir) / "metrics.json").read_text())
        bundle_blob = json.dumps(bundle_metrics)
        for candidate_id in (f"injecagent-{i:04d}" for i in range(200)):
            self.assertNotIn(candidate_id, bundle_blob)
        for path_name in BUNDLE_FILES:
            text = (Path(result.bundle_dir) / path_name).read_text()
            self.assertNotIn("fake-output:injecagent", text)

    def test_held_out_cannot_be_read_through_runner_before_authorization(self):
        # Mirrors tests/test_protocol.py::test_heldout_cannot_be_read_before_selection,
        # but exercised through the runner's own seam rather than the sealer directly.
        self._run()
        with self.assertRaises(PermissionError):
            read_held_out_result(self.sealer, {"finalized": False})
        with self.assertRaises(PermissionError):
            read_held_out_result(self.sealer, {"finalized": True, "checkpoint": "sha256:unfinalized-selection"})

    def test_reusing_a_sealer_for_a_second_baseline_run_is_rejected(self):
        self._run(run_id="baseline-a")
        with self.assertRaisesRegex(RuntimeError, "already frozen"):
            self._run(run_id="baseline-b")

if __name__ == "__main__":
    unittest.main()
