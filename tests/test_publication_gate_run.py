"""Issue #32: offline unit tests for the publication-gate operator helpers.

The full ``run`` path reads the gitignored real evidence tree and is exercised by
the operator command recorded in ``docs/issue-32-provenance-source-decision.md``;
here only the pure discovery/telemetry helpers are checked, fully offline.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from runner.publication_gate_run import EvidenceAssemblyError, _one, _peak_vram_gb


class HelperTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_peak_vram_gb_is_the_max_row_in_mib_over_1024(self):
        csv_path = self.root / "gpu.csv"
        csv_path.write_text(
            "t,vram_mb,util_pct\n0.1,15896,0\n0.4,16320,2\n0.7,15902,5\n",
            encoding="utf-8",
        )
        self.assertAlmostEqual(_peak_vram_gb(csv_path), 16320 / 1024.0)

    def test_one_rejects_missing_and_ambiguous_matches(self):
        with self.assertRaisesRegex(EvidenceAssemblyError, "no directory matches"):
            _one(self.root, "eval-seed42-epoch1-*")

        (self.root / "eval-seed42-epoch1-aaaa").mkdir()
        (self.root / "eval-seed42-epoch1-bbbb").mkdir()
        with self.assertRaisesRegex(EvidenceAssemblyError, "ambiguous"):
            _one(self.root, "eval-seed42-epoch1-*")

    def test_one_honours_reject_tokens(self):
        (self.root / "real-baseline-20260829").mkdir()
        (self.root / "real-baseline-comparison-20260829").mkdir()
        self.assertEqual(
            _one(self.root, "real-baseline-*", reject=("comparison",)).name,
            "real-baseline-20260829",
        )


if __name__ == "__main__":
    unittest.main()
