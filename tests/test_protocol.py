import json, subprocess, tempfile, unittest
from pathlib import Path
from protocol.validate_manifest import load
from protocol.heldout import HeldOutSealer

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"
REPO_ROOT = MANIFEST.parents[1]

class ProtocolTests(unittest.TestCase):
    def test_manifest_checkout_policy_forces_lf(self):
        result = subprocess.run(
            [
                "git", "check-attr", "eol", "--",
                "protocol/manifest.json", "protocol/manifest.sha256",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.stdout.splitlines(),
            [
                "protocol/manifest.json: eol: lf",
                "protocol/manifest.sha256: eol: lf",
            ],
        )

    def test_frozen_manifest_validates_offline(self):
        self.assertEqual(load(MANIFEST)["protocol_version"], "phase1-2026-08-29")

    def test_missing_decoding_default_is_rejected(self):
        data = json.loads(MANIFEST.read_text())
        del data["evaluation"]["decoding"]["seed"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"; p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "implicit decoding:seed"): load(p)

    def test_unapproved_fallback_is_rejected(self):
        data = json.loads(MANIFEST.read_text())
        data["allowed_technical_fallbacks"].append("change_learning_rate")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"; p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "unauthorized fallback"): load(p)

    def test_heldout_cannot_be_read_before_selection(self):
        with tempfile.TemporaryDirectory() as d:
            s = HeldOutSealer(d); s.freeze(["b", "a"], "invalid=technical")
            s.store_receipt("baseline", b"private-baseline", 1, 0)
            s.store_receipt("trained", b"private-trained", 1, 0)
            with self.assertRaises(PermissionError): s.reveal({"finalized": False})
            record = {"finalized": True, "checkpoint": "sha256:abc"}; s.authorize(record)
            self.assertEqual(s.reveal(record)["baseline"], b"private-baseline")

if __name__ == "__main__": unittest.main()
