import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner.real_baseline import (
    baseline_notes_text,
    compare_against_projection,
    load_canonical_projection,
    resolve_publish_counts,
    validate_baseline_paths,
    write_comparison_artifact,
)

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"


class ResolvePublishCountsTests(unittest.TestCase):
    def test_matches_frozen_manifest_declared_counts(self):
        frozen = load_manifest(MANIFEST)
        counts = resolve_publish_counts(frozen)

        self.assertEqual(counts["open_prompt_injection"], 300)  # publisher divides by 3 internally
        self.assertEqual(counts["tensor_trust_hijack"], 300)
        self.assertEqual(counts["tensor_trust_extract"], 300)
        self.assertEqual(counts["mmlu"], 300)
        self.assertEqual(counts["gsm8k"], 200)
        self.assertEqual(counts["ifeval"], 200)
        self.assertEqual(counts["injecagent"], 200)

    def test_rejects_opi_count_not_divisible_by_three_tasks(self):
        frozen = load_manifest(MANIFEST)
        broken = dict(frozen)
        broken["evaluation"] = dict(frozen["evaluation"])
        broken["evaluation"]["visible_safety"] = dict(frozen["evaluation"]["visible_safety"])
        broken["evaluation"]["visible_safety"]["open_prompt_injection"] = {
            **frozen["evaluation"]["visible_safety"]["open_prompt_injection"],
            "sample_ids": "publisher_seed_42_first_301;100_each_injected_task",
        }
        with self.assertRaisesRegex(ValueError, "divisible by 3"):
            resolve_publish_counts(broken)


class BaselinePathPolicyTests(unittest.TestCase):
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
                        validate_baseline_paths(root, suite, heldout, output, seal)

    def test_accepts_pairwise_separate_external_restricted_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            validate_baseline_paths(
                root, root / "runs" / "data", root.parent / f"{root.name}-heldout",
                root / "runs", root.parent / f"{root.name}-seal",
            )


class CompareAgainstProjectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(MANIFEST)
        self.projection = {
            "canonical_qualification": True,
            "baseline_wall_seconds": 19892.925042599956,
            "projected_peak_vram_gb": 14.9248046875,
        }

    def test_within_limits_has_no_feasibility_findings(self):
        comparison = compare_against_projection(
            manifest=self.manifest, projection=self.projection,
            wall_seconds=19000.0, peak_vram_mb=14000.0, bundle_bytes=1_000_000,
        )
        self.assertEqual(comparison["feasibility_findings"], [])
        self.assertAlmostEqual(comparison["measured"]["peak_vram_gb"], 14000.0 / 1024.0)
        self.assertAlmostEqual(comparison["measured"]["gpu_hours"], 19000.0 / 3600.0)
        self.assertAlmostEqual(comparison["delta"]["wall_seconds"], 19000.0 - 19892.925042599956)
        self.assertEqual(comparison["limits"], self.manifest["resources"])

    def test_exceeding_vram_limit_is_reported_as_a_finding_not_silently_absorbed(self):
        comparison = compare_against_projection(
            manifest=self.manifest, projection=self.projection,
            wall_seconds=19000.0, peak_vram_mb=16_000.0, bundle_bytes=0,
        )
        self.assertTrue(any("peak_vram_gb" in finding for finding in comparison["feasibility_findings"]))

    def test_exceeding_wall_hours_per_seed_limit_is_reported(self):
        comparison = compare_against_projection(
            manifest=self.manifest, projection=self.projection,
            wall_seconds=25 * 3600.0, peak_vram_mb=1000.0, bundle_bytes=0,
        )
        self.assertTrue(any("wall_hours" in finding for finding in comparison["feasibility_findings"]))

    def test_missing_projected_wall_seconds_yields_none_delta_not_a_crash(self):
        comparison = compare_against_projection(
            manifest=self.manifest, projection={"canonical_qualification": True},
            wall_seconds=100.0, peak_vram_mb=100.0, bundle_bytes=0,
        )
        self.assertIsNone(comparison["delta"]["wall_seconds"])


class CanonicalProjectionTests(unittest.TestCase):
    def test_rejects_non_canonical_projection_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resource_projection.json"
            path.write_text(json.dumps({"canonical_qualification": False}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical_qualification"):
                load_canonical_projection(path)

    def test_accepts_canonical_projection_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resource_projection.json"
            path.write_text(json.dumps({"canonical_qualification": True, "x": 1}), encoding="utf-8")
            self.assertEqual(load_canonical_projection(path), {"canonical_qualification": True, "x": 1})


class ComparisonArtifactTests(unittest.TestCase):
    def test_comparison_artifact_is_new_checksummed_json_and_preserves_findings(self):
        comparison = {
            "feasibility_findings": ["measured wall_hours 25.0 exceeds wall_hours_per_seed_max 24"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            artifact = write_comparison_artifact(Path(tmp), "comparison-test", comparison)

            self.assertEqual(
                json.loads((artifact / "baseline_resource_comparison.json").read_text()), comparison
            )
            checksum = (artifact / "checksums.sha256").read_text()
            self.assertRegex(checksum, r"^[0-9a-f]{64}  baseline_resource_comparison\.json\n$")
            digest = hashlib.sha256(
                (artifact / "baseline_resource_comparison.json").read_bytes()
            ).hexdigest()
            self.assertIn(digest, checksum)
            with self.assertRaises(FileExistsError):
                write_comparison_artifact(Path(tmp), "comparison-test", comparison)


class BaselineNotesTextTests(unittest.TestCase):
    def test_states_this_is_the_frozen_baseline_not_a_smoke_qualification(self):
        notes = baseline_notes_text("baseline")
        self.assertIn("frozen Phase-4 baseline", notes)
        self.assertNotIn("hardware qualification", notes.split("Not a ")[0])
        self.assertIn("Not a hardware qualification or smoke test.", notes)


if __name__ == "__main__":
    unittest.main()
