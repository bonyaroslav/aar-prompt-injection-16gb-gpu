import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from protocol.heldout import HeldOutSealer
from protocol.validate_manifest import load as load_manifest
from runner.bundle import verify_bundle
from runner.core import run_baseline
from runner.fakes import (
    FakeDatasetAdapter, FakeModelAdapter, FakeScorerAdapter, FakeTelemetryAdapter,
)
from runner.gpu_smoke import (
    build_smoke_manifest, smoke_training_examples, validate_smoke_paths,
    validate_merged_checkpoint, write_projection_artifact,
    write_or_verify_data_manifest, _verify_upstream,
)
from runner.storage import LocalStorageAdapter


MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"


class EvidenceTelemetry(FakeTelemetryAdapter):
    command_text = "python -m runner.gpu_smoke --model-cache /cache"
    def environment_text(self):
        return "os=Linux-WSL2\npython=3.12.14\ntorch_cuda=13.0\ngpu=RTX 4080\n"

    @property
    def events(self):
        return ["real telemetry samples=2", "peak vram=4096 MB"]

    def notes_text(self, stage):
        return f"# {stage} smoke notes\n\nReal HF/CUDA adapters; hardware qualification only.\n"


class EvidenceDataset(FakeDatasetAdapter):
    def manifest_metadata(self):
        return {"injecagent_source_commit": "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"}

    @property
    def environment_lines(self):
        return ["injecagent_source_commit=f19c9f2c79a41046eb13c03c51a24c567a8ffa07"]


class SmokeManifestTests(unittest.TestCase):
    def test_reduces_only_ephemeral_counts_without_mutating_frozen_manifest(self):
        frozen = load_manifest(MANIFEST)
        before = copy.deepcopy(frozen)

        smoke = build_smoke_manifest(
            frozen, samples_per_benchmark=1, training_count=2, epochs=1
        )

        self.assertEqual(frozen, before)
        for cfg in smoke["evaluation"]["visible_safety"].values():
            self.assertIn("first_1", cfg["sample_ids"])
        for cfg in smoke["evaluation"]["capability"].values():
            self.assertIn("first_1", cfg["sample_ids"])
        self.assertEqual(smoke["evaluation"]["held_out"]["injecagent"]["candidate_count"], 1)
        self.assertEqual(smoke["training"]["data"]["count"], 2)
        self.assertEqual(smoke["training"]["optimizer"]["epochs"], 1)
        self.assertEqual(smoke["model"]["revision"], frozen["model"]["revision"])
        self.assertEqual(smoke["evaluation"]["decoding"], frozen["evaluation"]["decoding"])
        self.assertNotIn("100_each_injected_task", smoke["evaluation"]["visible_safety"]["open_prompt_injection"]["sample_ids"])

    def test_training_fixture_is_long_enough_to_force_max_length_qualification(self):
        examples = smoke_training_examples()
        long_response = examples[0]["messages"][-1]["content"]
        self.assertGreater(len(long_response), 20_000)


class SmokePathPolicyTests(unittest.TestCase):
    def test_rejects_heldout_or_seal_roots_overlapping_public_or_repository_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            suite = root / "runs" / "data"
            output = root / "runs"
            external = root.parent / f"{root.name}-external"
            cases = [
                (suite, external),
                (output, external),
                (root, external),
                (external, external),
            ]
            for heldout, seal in cases:
                with self.subTest(heldout=heldout, seal=seal):
                    with self.assertRaisesRegex(ValueError, "restricted roots must not overlap"):
                        validate_smoke_paths(root, suite, heldout, output, seal)

    def test_accepts_pairwise_separate_external_restricted_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            validate_smoke_paths(
                root, root / "runs" / "data", root.parent / f"{root.name}-heldout",
                root / "runs", root.parent / f"{root.name}-seal",
            )


class RealEvidenceHookTests(unittest.TestCase):
    def test_same_baseline_runner_records_real_environment_and_events_when_adapter_supplies_them(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as heldout:
            result = run_baseline(
                MANIFEST,
                model=FakeModelAdapter(), dataset=EvidenceDataset(), scorer=FakeScorerAdapter(),
                telemetry=EvidenceTelemetry(), storage=LocalStorageAdapter(tmp),
                held_out_sealer=HeldOutSealer(heldout), run_id="real-evidence-hook",
            )
            bundle = Path(result.bundle_dir)

            self.assertTrue((bundle / "environment.txt").read_text().startswith(EvidenceTelemetry().environment_text()))
            log = (bundle / "execution.log").read_text()
            self.assertIn("real telemetry samples=2", log)
            self.assertIn("peak vram=4096 MB", log)
            self.assertNotIn("gpu=none", (bundle / "environment.txt").read_text())
            self.assertIn("injecagent_source_commit=f19c9f2c", (bundle / "environment.txt").read_text())
            manifest = json.loads((bundle / "manifest.yaml").read_text())
            self.assertEqual(manifest["injecagent_source_commit"], "f19c9f2c79a41046eb13c03c51a24c567a8ffa07")
            notes = (bundle / "notes.md").read_text()
            self.assertIn("Real HF/CUDA adapters", notes)
            self.assertNotIn("Fake adapters only", notes)
            self.assertIn(EvidenceTelemetry.command_text, (bundle / "command.sh").read_text())
            verify_bundle(bundle)


class ProjectionArtifactTests(unittest.TestCase):
    def test_merged_checkpoint_validation_hashes_every_file_and_records_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "model.safetensors").write_bytes(b"weights")

            result = validate_merged_checkpoint(
                checkpoint,
                model_factory=lambda path: FakeModelAdapter(),
                safetensors_validator=lambda path: 320,
            )

            self.assertEqual(result["generation_output"], "fake-output:open_prompt_injection:validation")
            self.assertEqual(result["safetensors_tensor_count"], 320)
            self.assertEqual(set(result["files_sha256"]), {"config.json", "model.safetensors"})
            self.assertRegex(result["files_sha256"]["model.safetensors"], r"^[0-9a-f]{64}$")

    def test_projection_artifact_is_new_checksummed_json_and_preserves_findings(self):
        projection = {
            "total_gpu_hours": 80.0,
            "feasibility_findings": ["projected total GPU-hours exceeds limit"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            artifact = write_projection_artifact(Path(tmp), "projection-smoke", projection)

            self.assertEqual(json.loads((artifact / "resource_projection.json").read_text()), projection)
            checksum = (artifact / "checksums.sha256").read_text()
            self.assertRegex(checksum, r"^[0-9a-f]{64}  resource_projection\.json\n$")
            with self.assertRaises(FileExistsError):
                write_projection_artifact(Path(tmp), "projection-smoke", projection)


class PublishedDataIntegrityTests(unittest.TestCase):
    def test_manifest_records_files_and_rejects_later_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "gsm8k.jsonl"
            target.write_text('{"prompt":"q","answer":"1"}\n', encoding="utf-8")
            manifest = write_or_verify_data_manifest(root, [target], expected_rows={"gsm8k.jsonl": 1})
            self.assertEqual(manifest["files"]["gsm8k.jsonl"]["rows"], 1)
            self.assertEqual(
                manifest["files"]["gsm8k.jsonl"]["sha256"],
                hashlib.sha256(target.read_bytes()).hexdigest(),
            )
            target.write_text('{"prompt":"changed","answer":"1"}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "published data integrity mismatch"):
                write_or_verify_data_manifest(root, [target], expected_rows={"gsm8k.jsonl": 1})


class UpstreamProvenanceTests(unittest.TestCase):
    def test_verification_rejects_dirty_behavior_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            behavior = root / "scripts" / "publish_suite.py"
            behavior.parent.mkdir()
            behavior.write_text("PINNED = True\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            tree = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            self.assertEqual(_verify_upstream(root, head, tree), {"commit": head, "tree": tree})
            behavior.write_text("PINNED = False\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "behavior files are dirty"):
                _verify_upstream(root, head, tree)


if __name__ == "__main__":
    unittest.main()
