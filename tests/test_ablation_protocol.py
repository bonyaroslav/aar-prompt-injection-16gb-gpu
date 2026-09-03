import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from protocol.ablation import manifest as ablation


REPO_ROOT = Path(__file__).parents[1]
MANIFEST = REPO_ROOT / "protocol" / "ablation" / "corpus-ablation-2026-09-02.json"
DIGESTS = REPO_ROOT / "protocol" / "ablation" / "digests.md"


class AblationManifestTests(unittest.TestCase):
    def test_shipped_manifest_is_downstream_of_frozen_attempt_one(self):
        data = ablation.load(MANIFEST)

        self.assertEqual(data["ablation_version"], "ablation-corpus-2026-09-02")
        self.assertEqual(data["training"]["seed"], 42)
        self.assertEqual(data["corpus"]["targets"], {
            "prompt_injection": 0,
            "clean_control": 3500,
            "ambiguous_boundary": 1000,
            "refusal_calibration": 500,
        })
        self.assertEqual(data["corpus"]["dolly_oversample_factor"], 2)
        self.assertEqual(data["corpus"]["construction_token_cap"], 1536)
        self.assertNotEqual(data["analysis"]["bootstrap_seed"], 271828)

    def test_loader_fails_closed_when_frozen_manifest_digest_drifts(self):
        altered = json.loads(MANIFEST.read_text(encoding="utf-8"))
        altered["downstream_of"]["canonical_manifest_digest"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "altered.json"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ablation.AblationManifestError, "no longer matches"):
                ablation.load(path)

    def test_loader_rejects_missing_boundaries(self):
        altered = json.loads(MANIFEST.read_text(encoding="utf-8"))
        del altered["boundaries"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "altered.json"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ablation.AblationManifestError, "boundaries"):
                ablation.load(path)

    def test_digest_record_and_lf_attribute_cover_manifest(self):
        data = ablation.load(MANIFEST)
        digest_text = DIGESTS.read_text(encoding="utf-8")
        self.assertIn(ablation.canonical_digest(data), digest_text)
        self.assertIn(ablation.raw_sha256(MANIFEST), digest_text)

        result = subprocess.run(
            ["git", "check-attr", "eol", "--", "protocol/ablation/corpus-ablation-2026-09-02.json",
             "protocol/ablation/digests.md"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout.splitlines(), [
            "protocol/ablation/corpus-ablation-2026-09-02.json: eol: lf",
            "protocol/ablation/digests.md: eol: lf",
        ])


if __name__ == "__main__":
    unittest.main()
