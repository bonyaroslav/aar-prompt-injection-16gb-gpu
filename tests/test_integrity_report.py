"""Issue #29: failure-mode evidence and integrity records.

Pure offline tests: hand-shaped execution-log text, ``metrics.json`` dicts,
resource-comparison dicts and a small corpus fixture. No model, dataset, scorer,
trainer, telemetry or storage; no GPU; the real evidence tree is never read
(the only committed files read are ``protocol/manifest.json``,
``protocol/deviations.md`` and ``RESEARCH_PLAN.md``).
"""
from __future__ import annotations

import unittest
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner import integrity_report as ir
from runner.integrity_report import (
    RevealBundlePresentError,
    assert_no_reveal_bundle,
    build_integrity_report,
    corpus_nutrition_label,
    generation_failure_signature,
    held_out_disposition,
    parse_generation_signature,
    render_report,
    reproducibility_disclosure,
    resource_accounting,
    sample_count_convention,
    tensor_trust_degeneracy,
    tensor_trust_distribution,
    utility_control_arm_comparison,
)

REPO = Path(__file__).parents[1]
REAL_MANIFEST = REPO / "protocol" / "manifest.json"
EPOCHS = load_manifest(REAL_MANIFEST)["training"]["optimizer"]["epochs"]

VISIBLE = ("open_prompt_injection", "tensor_trust_hijack", "tensor_trust_extract")
CAPABILITY = ("mmlu", "gsm8k", "ifeval")

REAL_TIMING_LOG = (
    "start trained-checkpoint evaluation run eval-seed42-epoch1 seed=42 epoch=1\n"
    "scored open_prompt_injection: n=300\n"
    "real model timing benchmark=gsm8k calls=200 mean_seconds=6.223944\n"
    "real model timing benchmark=ifeval calls=200 mean_seconds=7.783331\n"
    "real model timing benchmark=mmlu calls=300 mean_seconds=0.298250\n"
    "real model timing benchmark=open_prompt_injection calls=300 mean_seconds=0.424426\n"
    "real model timing benchmark=tensor_trust_extract calls=600 mean_seconds=1.774305\n"
    "real model timing benchmark=tensor_trust_hijack calls=600 mean_seconds=1.460016\n"
    "real model completions_truncated=34\n"
    "finished trained-checkpoint evaluation run eval-seed42-epoch1\n"
)
OLD_FORMAT_LOG = (
    "start trained-checkpoint evaluation run eval-seed17-epoch1 seed=17 epoch=1\n"
    "scored open_prompt_injection: n=300\n"
    "scored gsm8k: n=200\n"
    "finished trained-checkpoint evaluation run eval-seed17-epoch1\n"
)
BASELINE_LOG = (
    "start baseline run real-baseline\n"
    "real model timing benchmark=gsm8k calls=200 mean_seconds=45.957129\n"
    "real model timing benchmark=ifeval calls=200 mean_seconds=22.119045\n"
    "real model timing benchmark=mmlu calls=300 mean_seconds=0.297971\n"
    "real model timing benchmark=open_prompt_injection calls=300 mean_seconds=0.383799\n"
    "real model timing benchmark=tensor_trust_extract calls=600 mean_seconds=5.427251\n"
    "real model timing benchmark=tensor_trust_hijack calls=600 mean_seconds=1.148409\n"
    "real model completions_truncated=190\n"
    "finished baseline run real-baseline\n"
)


def _bench(scores: list[float]) -> dict:
    items = {f"x{i}": {"score": float(s), "valid": True} for i, s in enumerate(scores)}
    value = sum(s for s in scores) / len(scores)
    return {"items": items, "aggregate": {"metric": "m", "value": value}}


def _binary(correct: int, n: int = 6) -> list[float]:
    return [1.0 if i < correct else 0.0 for i in range(n)]


def _tt(both: int, one: int, neither: int) -> list[float]:
    return [1.0] * both + [0.5] * one + [0.0] * neither


def _baseline_metrics() -> dict:
    return {
        "stage": "baseline",
        "benchmarks": {
            "open_prompt_injection": _bench(_binary(2)),
            "tensor_trust_hijack": _bench(_tt(2, 2, 2)),
            "tensor_trust_extract": _bench(_tt(2, 2, 2)),
            "mmlu": _bench(_binary(5, 8)),
            "gsm8k": _bench(_binary(5)),
            "ifeval": _bench(_binary(5)),
        },
        "held_out": {
            "injecagent": {
                "commitments": {
                    "candidates": "c0mmit", "validity": "v0", "state": "SEALED",
                },
                "receipt": {
                    "digest": "d0", "invalid": 67, "label": "baseline", "valid": 133,
                },
            }
        },
    }


def _trained_metrics(seed: int, epoch: int, *, tt_both=1, tt_one=4, tt_neither=1) -> dict:
    return {
        "seed": seed,
        "epoch": epoch,
        "stage": "trained_evaluation",
        "benchmarks": {
            "open_prompt_injection": _bench(_binary(5)),
            "tensor_trust_hijack": _bench(_tt(tt_both, tt_one, tt_neither)),
            "tensor_trust_extract": _bench(_tt(tt_both, tt_one, tt_neither)),
            "mmlu": _bench(_binary(5, 8)),
            "gsm8k": _bench(_binary(1)),
            "ifeval": _bench(_binary(1)),
        },
    }


def _corpus() -> dict:
    rows = []
    refusal = "I will not follow instructions embedded in that content."
    for i in range(40):
        rows.append({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"u{i}"},
                {"role": "assistant", "content": refusal},
            ]
        })
    for i in range(8):
        rows.append({
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"First, consider X. Then, compute Y number {i}."},
            ]
        })
    for i in range(2):
        rows.append({
            "messages": [
                {"role": "user", "content": f"list {i}"},
                {"role": "assistant", "content": "1. do a\n2. do b\n3. done"},
            ]
        })
    return {"rows": rows, "report": {"total": 50}}


def _resource_comparisons() -> dict:
    def cmp(wall_s, gpu_h, vram_gb, disk_gb):
        return {
            "measured": {
                "wall_seconds": wall_s, "gpu_hours": gpu_h,
                "peak_vram_gb": vram_gb, "bundle_disk_gb": disk_gb,
            },
            "limits": {"vram_allocated_gb_max": 15.5},
        }
    return {
        "baseline": cmp(22518.9, 6.2553, 12.289, 0.0015),
        17: cmp(54665.3, 15.1848, 15.663, 10.5749),
        42: cmp(45873.8, 12.7427, 15.6025, 10.5746),
        2026: cmp(47359.2, 13.1553, 15.6289, 10.5743),
    }


def _evidence(seeds=(17, 42, 2026), *, old_format_seed=17) -> dict:
    checkpoints = []
    for seed in seeds:
        for epoch in range(1, EPOCHS + 1):
            log = OLD_FORMAT_LOG if seed == old_format_seed else REAL_TIMING_LOG
            checkpoints.append({
                "seed": seed, "epoch": epoch,
                "metrics": _trained_metrics(seed, epoch),
                "execution_log": log,
            })
    return {
        "frozen_input_record": {"inputs": [
            {"role": "baseline_bundle", "path": "runs/real-baseline-x"},
            {"role": "seed17_training_bundle", "path": "runs/training-seed17-x"},
        ]},
        "baseline": {"metrics": _baseline_metrics(), "execution_log": BASELINE_LOG},
        "checkpoints": checkpoints,
        "selection_records": [
            {"finalized": True, "selected_checkpoint_digest": None} for _ in seeds
        ],
        "resource_comparisons": {
            k: v for k, v in _resource_comparisons().items()
            if k == "baseline" or k in seeds
        },
        "phase_vram_peaks_gb": {
            seed: {"training": 15.66, "evaluation": 15.63} for seed in seeds
        },
        "non_scientific_runs": [
            {"category": "smoke", "label": "gpu-smoke-191237", "gpu_hours": 0.03,
             "wall_hours": 0.03, "source": "runs/gpu-smoke-*/gpu.csv"},
        ],
        "corpus": _corpus(),
    }


class ParserTests(unittest.TestCase):
    def test_real_format_log_yields_truncation_and_timing(self):
        sig = parse_generation_signature(REAL_TIMING_LOG)
        self.assertTrue(sig["recorded"])
        self.assertEqual(sig["truncated_completions"], 34)
        self.assertAlmostEqual(sig["seconds_per_item"]["gsm8k"], 6.223944)
        self.assertEqual(len(sig["seconds_per_item"]), 6)

    def test_old_format_log_is_marked_not_recorded(self):
        sig = parse_generation_signature(OLD_FORMAT_LOG)
        self.assertFalse(sig["recorded"])
        self.assertIsNone(sig["truncated_completions"])
        self.assertEqual(sig["seconds_per_item"], {})

    def test_tensor_trust_distribution_bins_three_values(self):
        dist = tensor_trust_distribution(_bench(_tt(3, 5, 2)))
        self.assertEqual(dist, {"both": 3, "one": 5, "neither": 2, "n": 10})

    def test_tensor_trust_distribution_rejects_off_scale_score(self):
        with self.assertRaises(ir.IntegrityReportError):
            tensor_trust_distribution(_bench([0.0, 0.25, 1.0]))


class GenerationFailureTests(unittest.TestCase):
    def test_reports_baseline_and_every_checkpoint_with_mechanisms_named(self):
        block = generation_failure_signature(_evidence(), load_manifest(REAL_MANIFEST))
        self.assertEqual(block["baseline"]["truncated_completions"], 190)
        self.assertIn("stop token", block["reading_guide"])
        self.assertIn("shorter", block["reading_guide"])
        self.assertIn("longer", block["reading_guide"])
        self.assertNotIn("mmlu", block["free_generation_benchmarks"])
        # seeds 42 and 2026 recorded, seed 17 not
        seeds_present = {row["seed"] for row in block["per_checkpoint"]}
        self.assertEqual(seeds_present, {42, 2026})
        for row in block["per_checkpoint"]:
            self.assertEqual(row["truncation_delta_vs_baseline"], 34 - 190)

    def test_seed_17_checkpoints_are_reported_unavailable_not_dropped_silently(self):
        block = generation_failure_signature(_evidence(), load_manifest(REAL_MANIFEST))
        unavailable = {(r["seed"], r["epoch"]) for r in block["unavailable"]}
        self.assertEqual(unavailable, {(17, 1), (17, 2), (17, 3)})
        self.assertIn("predates", block["unavailable"][0]["reason"])

    def test_correct_at_two_seeds(self):
        block = generation_failure_signature(
            _evidence(seeds=(42, 2026), old_format_seed=None),
            load_manifest(REAL_MANIFEST),
        )
        self.assertEqual(len(block["per_checkpoint"]), 6)
        self.assertEqual(block["unavailable"], [])


class TensorTrustDegeneracyTests(unittest.TestCase):
    def test_both_to_one_migration_is_a_named_finding(self):
        ev = _evidence(seeds=(42,), old_format_seed=None)
        for cp in ev["checkpoints"]:
            cp["metrics"] = _trained_metrics(42, cp["epoch"], tt_both=0, tt_one=5, tt_neither=1)
        block = tensor_trust_degeneracy(ev)
        self.assertIn("refusal-degeneracy signature present", block["verdict"])
        self.assertIn("'both' to 'one'", block["decision_rule"])
        for row in block["per_run"]:
            self.assertLess(row["migration"]["both"], 0)
            self.assertGreater(row["migration"]["one"], 0)
            self.assertIn("both -> one", row["verdict"])

    def test_neither_to_both_migration_refutes_degeneracy(self):
        ev = _evidence(seeds=(42,), old_format_seed=None)
        for cp in ev["checkpoints"]:
            cp["metrics"] = _trained_metrics(42, cp["epoch"], tt_both=5, tt_one=0, tt_neither=1)
        block = tensor_trust_degeneracy(ev)
        self.assertIn("no refusal-degeneracy signature", block["verdict"])
        for row in block["per_run"]:
            self.assertIn("refuted", row["verdict"])

    def test_distribution_reported_per_run_across_three_values(self):
        block = tensor_trust_degeneracy(_evidence())
        for row in block["per_run"]:
            for key in ("baseline_distribution", "trained_distribution"):
                self.assertEqual(set(row[key]), {"both", "one", "neither", "n"})


class UtilityControlArmTests(unittest.TestCase):
    def test_splits_benchmarks_by_presence_of_a_control_arm(self):
        block = utility_control_arm_comparison(_evidence())
        self.assertEqual(block["without_control_arm"], ["open_prompt_injection"])
        self.assertEqual(
            set(block["with_control_arm"]),
            {"tensor_trust_hijack", "tensor_trust_extract"},
        )
        self.assertIn("intended task", block["note"])
        for row in block["per_run"]:
            self.assertAlmostEqual(
                row["gap"],
                row["mean_gain_without_control_arm"] - row["mean_gain_with_control_arm"],
            )


class CorpusNutritionTests(unittest.TestCase):
    def test_reports_all_five_named_quantities(self):
        label = corpus_nutrition_label(_corpus())
        self.assertEqual(label["total_examples"], 50)
        self.assertEqual(label["distinct_assistant_responses"], 10)
        self.assertAlmostEqual(label["most_frequent_response_coverage"]["top_1"], 40 / 50)
        for unit in ("response_length_chars", "response_length_words"):
            self.assertEqual(
                set(label[unit]), {"min", "median", "p90", "p99", "max", "mean"}
            )
        # 8 "First, ... Then, ..." + 2 numbered lists = 10 of 50
        self.assertAlmostEqual(label["multi_step_reasoning_share"], 10 / 50)
        self.assertIn("proxy", label["multi_step_rule"])

    def test_rejects_empty_corpus(self):
        with self.assertRaises(ir.IntegrityReportError):
            corpus_nutrition_label({"rows": [], "report": {}})


class HeldOutDispositionTests(unittest.TestCase):
    def test_records_every_required_element(self):
        block = held_out_disposition(_evidence(), load_manifest(REAL_MANIFEST))
        self.assertEqual(block["terminal_state"], "NEVER_AUTHORIZED")
        self.assertIn("selection has no selected checkpoint", block["enforced_in_code"])
        self.assertEqual(block["sealed_baseline_candidates"]["valid"], 133)
        self.assertEqual(block["sealed_baseline_candidates"]["invalid"], 67)
        self.assertEqual(
            block["sealed_baseline_candidates"]["intent_to_evaluate_total"], 200
        )
        self.assertIn("does not make the population-level baseline secret", block["sealing_scope"])
        self.assertIn("least statistical power", block["pre_registered_minimum_detectable_effect"]["reading"])
        self.assertIn("future attempt", block["future_attempt_disposition"])
        self.assertTrue(block["no_reveal_bundle"])
        self.assertTrue(block["all_completed_selections_null"])


class NoRevealBundleTests(unittest.TestCase):
    def test_passes_when_only_a_sealed_receipt_is_present(self):
        assert_no_reveal_bundle(_evidence())  # no raise

    def test_raises_on_a_reveal_stage_metrics_document(self):
        ev = _evidence()
        ev["checkpoints"][0]["metrics"]["stage"] = "reveal"
        with self.assertRaises(RevealBundlePresentError):
            assert_no_reveal_bundle(ev)

    def test_raises_on_a_revealed_injecagent_aggregate(self):
        ev = _evidence()
        ev["baseline"]["metrics"]["held_out"]["injecagent"]["valid_only"] = {"value": 0.9}
        with self.assertRaises(RevealBundlePresentError):
            assert_no_reveal_bundle(ev)

    def test_raises_on_a_reveal_role_in_the_frozen_record(self):
        ev = _evidence()
        ev["frozen_input_record"]["inputs"].append(
            {"role": "reveal_bundle", "path": "runs/reveal-abc"}
        )
        with self.assertRaises(RevealBundlePresentError):
            assert_no_reveal_bundle(ev)


class ReproducibilityDisclosureTests(unittest.TestCase):
    def test_seven_items_three_also_in_deviations(self):
        block = reproducibility_disclosure()
        self.assertEqual(len(block["items"]), 7)
        self.assertEqual(len(block["also_in_deviations"]), 3)
        self.assertEqual(
            set(block["also_in_deviations"]),
            {
                "manifest_names_unused_mc_scorer",
                "freeform_treatment_read_by_no_code",
                "decoding_applied_globally_not_per_benchmark",
            },
        )

    def test_the_three_drifts_are_present_in_deviations_md(self):
        text = (REPO / "protocol" / "deviations.md").read_text(encoding="utf-8")
        self.assertIn("first_token_logit", text)
        self.assertIn("freeform_treatment", text)
        self.assertIn("per benchmark", text)
        self.assertIn("issue #29", text)


class ResourceAccountingTests(unittest.TestCase):
    def test_separates_scientific_totals_from_all_incurred_compute(self):
        block = resource_accounting(_evidence(), load_manifest(REAL_MANIFEST))
        sci = block["scientific_totals"]["gpu_hours"]
        allc = block["all_incurred_compute"]["gpu_hours"]
        self.assertAlmostEqual(sci, 6.2553 + 15.1848 + 12.7427 + 13.1553)
        self.assertAlmostEqual(allc, sci + 0.03)
        self.assertEqual(len(block["all_incurred_compute"]["non_scientific_runs"]), 1)

    def test_peak_vram_is_attributed_to_the_training_phase(self):
        block = resource_accounting(_evidence(), load_manifest(REAL_MANIFEST))
        self.assertAlmostEqual(block["peak_vram"]["value_gb"], 15.663)
        self.assertEqual(block["peak_vram"]["phase"], "training")
        self.assertGreater(block["peak_vram"]["overage_gb"], 0.16)
        self.assertIn("unbatched", block["unbatched_evaluation"].lower())

    def test_correct_at_two_seeds(self):
        block = resource_accounting(
            _evidence(seeds=(42, 2026), old_format_seed=None),
            load_manifest(REAL_MANIFEST),
        )
        self.assertAlmostEqual(
            block["scientific_totals"]["gpu_hours"], 6.2553 + 12.7427 + 13.1553
        )


class SampleCountConventionTests(unittest.TestCase):
    def test_states_the_reconciled_convention(self):
        block = sample_count_convention(load_manifest(REAL_MANIFEST))
        self.assertEqual(block["visible_safety_candidates"]["tensor_trust_hijack"], 300)
        self.assertEqual(block["tensor_trust_arm_evaluations"]["tensor_trust_hijack"], 600)
        self.assertEqual(block["held_out_sealed_candidates"], 200)
        self.assertIn("600 arm-evaluations", block["convention"])

    def test_research_plan_section_5_is_reconciled(self):
        text = (REPO / "RESEARCH_PLAN.md").read_text(encoding="utf-8")
        # the old wording is gone
        self.assertNotIn("secret extraction, 600 items", text)
        self.assertNotIn("instruction override, 600 items", text)
        self.assertIn("arm-evaluation", text)


class BuildReportTests(unittest.TestCase):
    def test_end_to_end_at_three_seeds(self):
        report = build_integrity_report(REAL_MANIFEST, evidence=_evidence())
        self.assertEqual(report["completed_seeds"], [17, 42, 2026])
        self.assertEqual(report["trained_checkpoint_count"], 9)
        self.assertIn("failure_mode_evidence", report)
        self.assertIn("integrity_records", report)
        self.assertTrue(report["integrity_records"]["no_reveal_bundle_verified"])

    def test_end_to_end_at_two_seeds(self):
        report = build_integrity_report(
            REAL_MANIFEST, evidence=_evidence(seeds=(42, 2026), old_format_seed=None)
        )
        self.assertEqual(report["completed_seeds"], [42, 2026])
        self.assertEqual(report["completed_seed_count"], 2)
        self.assertEqual(report["trained_checkpoint_count"], 6)

    def test_partial_epoch_set_is_rejected(self):
        ev = _evidence()
        ev["checkpoints"] = [cp for cp in ev["checkpoints"] if not (cp["seed"] == 42 and cp["epoch"] == 3)]
        with self.assertRaises(ir.IntegrityReportError):
            build_integrity_report(REAL_MANIFEST, evidence=ev)

    def test_output_is_byte_identical_across_runs(self):
        a = render_report(build_integrity_report(REAL_MANIFEST, evidence=_evidence()))
        b = render_report(build_integrity_report(REAL_MANIFEST, evidence=_evidence()))
        self.assertEqual(a, b)

    def test_no_output_field_names_an_epoch_best_or_winner(self):
        text = render_report(build_integrity_report(REAL_MANIFEST, evidence=_evidence()))
        self.assertNotIn('"best"', text)
        self.assertNotIn("winner", text)


if __name__ == "__main__":
    unittest.main()
