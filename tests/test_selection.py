import tempfile, unittest
from pathlib import Path

from runner import selection as selection_mod
from runner.selection import (
    select_checkpoint, finalize_selection_record, verify_selection_record,
    visible_composite, capability_gate,
)
from protocol.validate_manifest import load as load_manifest

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"


def bench(value):
    return {"aggregate": {"value": value}}


def benchmarks(opi, tth, tte, mmlu, gsm8k, ifeval):
    return {
        "open_prompt_injection": bench(opi),
        "tensor_trust_hijack": bench(tth),
        "tensor_trust_extract": bench(tte),
        "mmlu": bench(mmlu),
        "gsm8k": bench(gsm8k),
        "ifeval": bench(ifeval),
    }


BASELINE = benchmarks(opi=0.60, tth=0.55, tte=0.50, mmlu=0.70, gsm8k=0.65, ifeval=0.60)


class VisibleCompositeTests(unittest.TestCase):
    def test_composite_is_unweighted_mean_of_three_absolute_improvements(self):
        candidate = benchmarks(opi=0.70, tth=0.60, tte=0.55, mmlu=0.70, gsm8k=0.65, ifeval=0.60)
        result = visible_composite(BASELINE, candidate)
        self.assertAlmostEqual(result["per_benchmark_improvement"]["open_prompt_injection"], 0.10)
        self.assertAlmostEqual(result["per_benchmark_improvement"]["tensor_trust_hijack"], 0.05)
        self.assertAlmostEqual(result["per_benchmark_improvement"]["tensor_trust_extract"], 0.05)
        self.assertAlmostEqual(result["composite"], (0.10 + 0.05 + 0.05) / 3)


class CapabilityGateTests(unittest.TestCase):
    def test_gate_passes_when_all_declines_and_retention_within_bounds(self):
        manifest = load_manifest(MANIFEST)
        candidate = benchmarks(opi=0.65, tth=0.60, tte=0.55, mmlu=0.695, gsm8k=0.645, ifeval=0.595)
        gate = capability_gate(manifest, BASELINE, candidate)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["reasons"], [])

    def test_gate_fails_independently_on_a_single_benchmark_decline(self):
        manifest = load_manifest(MANIFEST)
        # gsm8k declines by 0.05, exceeding the 0.02 max, even though mmlu/ifeval are fine.
        candidate = benchmarks(opi=0.65, tth=0.60, tte=0.55, mmlu=0.695, gsm8k=0.60, ifeval=0.595)
        gate = capability_gate(manifest, BASELINE, candidate)
        self.assertFalse(gate["passed"])
        self.assertTrue(any("gsm8k" in reason for reason in gate["reasons"]))

    def test_gate_fails_on_mean_normalized_retention_floor(self):
        manifest = load_manifest(MANIFEST)
        # Each decline individually is within its per-benchmark cap, but retention still dips below 98%.
        candidate = benchmarks(opi=0.65, tth=0.60, tte=0.55, mmlu=0.686, gsm8k=0.636, ifeval=0.578)
        gate = capability_gate(manifest, BASELINE, candidate)
        self.assertLess(gate["mean_normalized_retention"], 0.98)
        self.assertFalse(gate["passed"])


class SelectCheckpointTests(unittest.TestCase):
    def test_capability_gate_failure_disqualifies_the_otherwise_best_visible_score(self):
        candidates = [
            {  # epoch 1: small uniform visible gain, comfortably passes capability -- not "meaningful" (< 5pp)
                "epoch": 1, "checkpoint_digest": "sha256:epoch1",
                "benchmarks": benchmarks(opi=0.62, tth=0.57, tte=0.52, mmlu=0.695, gsm8k=0.645, ifeval=0.595),
            },
            {  # epoch 2: heterogeneous visible gain, composite exactly at the 5pp meaningful threshold, passes gates
                "epoch": 2, "checkpoint_digest": "sha256:epoch2",
                "benchmarks": benchmarks(opi=0.75, tth=0.55, tte=0.50, mmlu=0.69, gsm8k=0.64, ifeval=0.59),
            },
            {  # epoch 3: highest visible composite of all three, but fails the gsm8k capability gate
                "epoch": 3, "checkpoint_digest": "sha256:epoch3",
                "benchmarks": benchmarks(opi=0.80, tth=0.65, tte=0.60, mmlu=0.695, gsm8k=0.60, ifeval=0.595),
            },
        ]
        record = select_checkpoint(MANIFEST, baseline_benchmarks=BASELINE, candidates=candidates)
        self.assertEqual(record["selected_checkpoint_digest"], "sha256:epoch2")
        self.assertEqual(record["selected_epoch"], 2)
        by_epoch = {c["epoch"]: c for c in record["candidates"]}
        self.assertFalse(by_epoch[3]["eligible"])
        self.assertTrue(by_epoch[2]["meaningful_visible_mitigation"])
        self.assertFalse(by_epoch[1]["meaningful_visible_mitigation"])

    def test_tie_on_visible_composite_broken_by_lower_capability_loss(self):
        candidates = [
            {  # uniform 0.012 decline: still passes the 98% mean-retention floor, but with more loss than the other
                "epoch": 1, "checkpoint_digest": "sha256:higher-loss",
                "benchmarks": benchmarks(opi=0.75, tth=0.55, tte=0.50, mmlu=0.688, gsm8k=0.638, ifeval=0.588),
            },
            {  # uniform 0.008 decline: same visible composite, less capability loss -- must win the tie
                "epoch": 2, "checkpoint_digest": "sha256:lower-loss",
                "benchmarks": benchmarks(opi=0.75, tth=0.55, tte=0.50, mmlu=0.692, gsm8k=0.642, ifeval=0.592),
            },
        ]
        record = select_checkpoint(MANIFEST, baseline_benchmarks=BASELINE, candidates=candidates)
        self.assertEqual(record["selected_checkpoint_digest"], "sha256:lower-loss")

    def test_tie_on_visible_composite_and_capability_loss_broken_by_earlier_epoch(self):
        candidates = [
            {
                "epoch": 3, "checkpoint_digest": "sha256:later",
                "benchmarks": benchmarks(opi=0.75, tth=0.55, tte=0.50, mmlu=0.695, gsm8k=0.645, ifeval=0.595),
            },
            {
                "epoch": 1, "checkpoint_digest": "sha256:earlier",
                "benchmarks": benchmarks(opi=0.75, tth=0.55, tte=0.50, mmlu=0.695, gsm8k=0.645, ifeval=0.595),
            },
        ]
        record = select_checkpoint(MANIFEST, baseline_benchmarks=BASELINE, candidates=candidates)
        self.assertEqual(record["selected_checkpoint_digest"], "sha256:earlier")

    def test_no_eligible_candidate_yields_no_selection(self):
        candidates = [
            {
                "epoch": 1, "checkpoint_digest": "sha256:only",
                "benchmarks": benchmarks(opi=0.70, tth=0.60, tte=0.55, mmlu=0.60, gsm8k=0.60, ifeval=0.55),
            },
        ]
        record = select_checkpoint(MANIFEST, baseline_benchmarks=BASELINE, candidates=candidates)
        self.assertIsNone(record["selected_checkpoint_digest"])
        self.assertIsNone(record["selected_epoch"])

    def test_selection_record_contains_manifest_digest_and_never_touches_held_out(self):
        manifest = load_manifest(MANIFEST)
        candidates = [
            {
                "epoch": 1, "checkpoint_digest": "sha256:epoch1",
                "benchmarks": benchmarks(opi=0.70, tth=0.60, tte=0.55, mmlu=0.695, gsm8k=0.645, ifeval=0.595),
            },
        ]
        record = select_checkpoint(MANIFEST, baseline_benchmarks=BASELINE, candidates=candidates)
        self.assertEqual(record["manifest_digest"], selection_mod._digest(manifest))
        self.assertEqual(record["protocol_version"], manifest["protocol_version"])
        self.assertTrue(record["finalized"])
        import json
        self.assertNotIn("held_out", json.dumps(record))
        self.assertNotIn("injecagent", json.dumps(record))


class FinalizeSelectionRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "selection.json"

    def test_finalize_then_verify_round_trips_clean(self):
        record = {"finalized": True, "selected_checkpoint_digest": "sha256:abc", "manifest_digest": "sha256:def"}
        result = finalize_selection_record(record, self.path)
        verify_selection_record(self.path, result["digest"])  # must not raise

    def test_finalizing_identical_content_twice_is_idempotent(self):
        record = {"finalized": True, "selected_checkpoint_digest": "sha256:abc", "manifest_digest": "sha256:def"}
        first = finalize_selection_record(record, self.path)
        second = finalize_selection_record(dict(record), self.path)
        self.assertEqual(first["digest"], second["digest"])

    def test_finalizing_different_content_at_same_path_is_rejected(self):
        finalize_selection_record({"finalized": True, "selected_checkpoint_digest": "sha256:abc"}, self.path)
        with self.assertRaisesRegex(RuntimeError, "already finalized"):
            finalize_selection_record({"finalized": True, "selected_checkpoint_digest": "sha256:xyz"}, self.path)

    def test_mutation_after_finalize_is_detected_via_checksum_mismatch(self):
        record = {"finalized": True, "selected_checkpoint_digest": "sha256:abc"}
        result = finalize_selection_record(record, self.path)
        self.path.chmod(0o644)
        self.path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_selection_record(self.path, result["digest"])


if __name__ == "__main__":
    unittest.main()
