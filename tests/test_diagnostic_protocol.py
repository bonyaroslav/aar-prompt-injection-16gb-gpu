import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from protocol.diagnostic import manifest as diag

REPO_ROOT = Path(__file__).parents[1]
DIAG_MANIFEST = REPO_ROOT / "protocol" / "diagnostic" / "chatmode-mmlu-2026-09-02.json"
DIGESTS_MD = REPO_ROOT / "protocol" / "diagnostic" / "digests.md"


class DiagnosticManifestTests(unittest.TestCase):
    def test_shipped_manifest_loads_and_is_downstream_of_the_live_frozen_manifest(self):
        data = diag.load(DIAG_MANIFEST)
        self.assertEqual(data["diagnostic_version"], "diag-chatmode-mmlu-2026-09-02")
        self.assertEqual(data["change"]["parameter"], "use_chat_template")
        self.assertFalse(data["change"]["attempt_1_value"])
        self.assertTrue(data["change"]["diagnostic_value"])
        # Nine merged checkpoints plus the baseline model state.
        states = data["model_states"]
        self.assertEqual(len(states), 10)
        self.assertEqual(states[0]["state"], "baseline")
        self.assertEqual(sum(1 for s in states if s.get("merged_dir")), 9)
        self.assertNotEqual(
            data["analysis"]["bootstrap_seed"], 271828,
            "diagnostic must not reuse the frozen Attempt-1 bootstrap seed",
        )

    def test_canonical_digest_is_deterministic_and_key_order_invariant(self):
        data = diag.load(DIAG_MANIFEST)
        shuffled = {k: data[k] for k in reversed(list(data))}
        self.assertEqual(diag.canonical_digest(data), diag.canonical_digest(shuffled))

    def test_load_fails_closed_when_downstream_of_digest_drifts(self):
        data = json.loads(DIAG_MANIFEST.read_text(encoding="utf-8"))
        data["downstream_of"]["canonical_manifest_digest"] = "0" * 64
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(diag.DiagnosticManifestError, "no longer matches"):
                diag.load(p)

    def test_load_rejects_a_manifest_missing_a_required_block(self):
        data = json.loads(DIAG_MANIFEST.read_text(encoding="utf-8"))
        del data["boundaries"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(diag.DiagnosticManifestError, "boundaries"):
                diag.load(p)

    def test_digests_md_records_the_current_manifest_identities(self):
        text = DIGESTS_MD.read_text(encoding="utf-8")
        self.assertIn(diag.canonical_digest(diag.load(DIAG_MANIFEST)), text)
        self.assertIn(diag.raw_sha256(DIAG_MANIFEST), text)

    def test_diagnostic_protocol_files_are_pinned_to_lf(self):
        result = subprocess.run(
            ["git", "check-attr", "eol", "--",
             "protocol/diagnostic/chatmode-mmlu-2026-09-02.json",
             "protocol/diagnostic/digests.md"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            ["protocol/diagnostic/chatmode-mmlu-2026-09-02.json: eol: lf",
             "protocol/diagnostic/digests.md: eol: lf"],
        )


if __name__ == "__main__":
    unittest.main()
