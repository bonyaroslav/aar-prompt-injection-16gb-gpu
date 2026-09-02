"""Issue #28: claim tables and statistics.

Pure offline tests: every case builds hand-shaped ``metrics.json`` dicts and a
manifest fixture (the frozen manifest with a smaller bootstrap replicate count
for speed). No model, dataset, scorer, trainer, telemetry or storage; no GPU;
the real evidence tree is never read.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner import claim_tables
from runner.claim_tables import (
    CompositeWithoutDecompositionError,
    MODALITY_FREE_GENERATION,
    MODALITY_LIKELIHOOD_RANKED,
    analysis_units,
    build_claim_report,
    mcnemar_exact,
    render_composite,
    render_report,
    visible_composite_block,
)

REAL_MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"
EPOCHS = load_manifest(REAL_MANIFEST)["training"]["optimizer"]["epochs"]

BENCH_N = {
    "open_prompt_injection": 6, "tensor_trust_hijack": 6, "tensor_trust_extract": 6,
    "mmlu": 8, "gsm8k": 6, "ifeval": 6,
}
BASELINE_CORRECT = {
    "open_prompt_injection": 2, "tensor_trust_hijack": 2, "tensor_trust_extract": 2,
    "mmlu": 5, "gsm8k": 5, "ifeval": 5,
}
TRAINED_CORRECT = {
    "open_prompt_injection": 5, "tensor_trust_hijack": 4, "tensor_trust_extract": 4,
    "mmlu": 5, "gsm8k": 1, "ifeval": 1,
}


def _bench(correct: int, n: int) -> dict:
    items = {f"x{i}": {"score": 1.0 if i < correct else 0.0, "valid": True} for i in range(n)}
    value = sum(v["score"] for v in items.values()) / n
    return {"items": items, "aggregate": {"metric": "m", "value": value}}


def _doc(correct: dict, *, seed=None, epoch=None) -> dict:
    doc = {"benchmarks": {name: _bench(correct[name], BENCH_N[name]) for name in BENCH_N}}
    if seed is not None:
        doc["seed"] = seed
    if epoch is not None:
        doc["epoch"] = epoch
    return doc


def _baseline() -> dict:
    return _doc(BASELINE_CORRECT)


def _run(seed: int, *, trained=None) -> list[dict]:
    trained = trained or TRAINED_CORRECT
    return [_doc(trained, seed=seed, epoch=epoch) for epoch in range(1, EPOCHS + 1)]


def _manifest(tmp: Path, **analysis_overrides) -> Path:
    data = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    data["analysis"]["bootstrap_replicates"] = 200
    for dotted, value in analysis_overrides.items():
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    path = tmp / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class AnalysisUnitTests(unittest.TestCase):
    def test_unit_is_every_epoch_of_every_run(self):
        units = analysis_units(_run(17) + _run(42), EPOCHS)
        self.assertEqual(
            [(u["seed"], u["epoch"]) for u in units],
            [(17, 1), (17, 2), (17, 3), (42, 1), (42, 2), (42, 3)],
        )

    def test_selection_record_shaped_input_is_rejected(self):
        winner = {"seed": 17, "epoch": 3, "benchmarks": {},
                  "selected_checkpoint_digest": None, "finalized": True}
        with self.assertRaisesRegex(ValueError, "post-hoc"):
            analysis_units([winner], EPOCHS)

    def test_only_a_winner_epoch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "every prespecified epoch"):
            analysis_units([_doc(TRAINED_CORRECT, seed=17, epoch=2)], EPOCHS)

    def test_best_marker_is_rejected(self):
        marked = _doc(TRAINED_CORRECT, seed=17, epoch=1)
        marked["best"] = True
        with self.assertRaisesRegex(ValueError, "post-hoc"):
            analysis_units([marked, _doc(TRAINED_CORRECT, seed=17, epoch=2),
                            _doc(TRAINED_CORRECT, seed=17, epoch=3)], EPOCHS)

    def test_no_output_labels_any_epoch_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_claim_report(
                _manifest(Path(tmp)), baseline_metrics=_baseline(),
                epoch_metrics=_run(17) + _run(42),
            )
        blob = render_report(report).lower()
        self.assertNotIn("best", blob)
        self.assertNotIn("winner", blob)


class McNemarExactTests(unittest.TestCase):
    def _counts(self, b: int, c: int) -> dict:
        baseline, candidate = {}, {}
        idx = 0
        for _ in range(b):  # baseline right, candidate wrong
            baseline[f"i{idx}"], candidate[f"i{idx}"] = 1.0, 0.0
            idx += 1
        for _ in range(c):  # baseline wrong, candidate right
            baseline[f"i{idx}"], candidate[f"i{idx}"] = 0.0, 1.0
            idx += 1
        baseline["agree"], candidate["agree"] = 1.0, 1.0
        return mcnemar_exact(baseline, candidate)

    def test_no_discordant_pairs_gives_p_one(self):
        result = self._counts(0, 0)
        self.assertEqual(result["n_discordant"], 0)
        self.assertEqual(result["p_value"], 1.0)

    def test_hand_checked_values(self):
        # two-sided exact binomial tail, p=0.5
        self.assertAlmostEqual(self._counts(2, 0)["p_value"], 0.5)
        self.assertAlmostEqual(self._counts(3, 0)["p_value"], 0.25)
        self.assertAlmostEqual(self._counts(5, 0)["p_value"], 0.0625)
        self.assertAlmostEqual(self._counts(8, 2)["p_value"], 112 / 1024)

    def test_symmetric_in_b_and_c(self):
        self.assertEqual(self._counts(7, 3)["p_value"], self._counts(3, 7)["p_value"])

    def test_non_binary_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            mcnemar_exact({"a": 0.5}, {"a": 1.0})


class PrimaryTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmpdir = Path(self.tmp.name)

    def _report(self, seeds=(17, 42), **manifest_overrides):
        epoch_metrics = []
        for seed in seeds:
            epoch_metrics += _run(seed)
        return build_claim_report(
            _manifest(self.tmpdir, **manifest_overrides),
            baseline_metrics=_baseline(), epoch_metrics=epoch_metrics,
        )

    def test_groups_by_evaluation_modality(self):
        table = self._report()["primary_table"]
        self.assertEqual(table["grouping_axis"], "evaluation_modality")
        self.assertEqual(
            table["modality_groups"][MODALITY_LIKELIHOOD_RANKED]["benchmarks"], ["mmlu"],
        )
        free = table["modality_groups"][MODALITY_FREE_GENERATION]["benchmarks"]
        self.assertIn("gsm8k", free)
        self.assertIn("open_prompt_injection", free)

    def test_modality_is_derived_from_the_manifest_not_hardcoded(self):
        # Rescore MMLU as generated free text -> it must move groups.
        table = self._report(**{
            "evaluation.capability.mmlu.scorer": "upstream_final_number_parser",
            "evaluation.capability.mmlu.max_new_tokens": 64,
        })["primary_table"]
        self.assertEqual(
            table["modality_groups"][MODALITY_FREE_GENERATION]["benchmarks"],
            list(claim_tables.BENCHMARKS),
        )
        self.assertNotIn(MODALITY_LIKELIHOOD_RANKED, table["modality_groups"])
        self.assertEqual(table["multiple_choice_only_gate"]["benchmarks"], [])

    def test_multiple_choice_only_gate_column_from_frozen_thresholds(self):
        rows = self._report()["primary_table"]["modality_groups"][
            MODALITY_LIKELIHOOD_RANKED]["rows"]
        # MMLU is unchanged in the fixture, so an MMLU-only gate passes.
        self.assertTrue(all(row["multiple_choice_only_gate_passes"] for row in rows))

    def test_gate_column_flips_when_the_frozen_tolerance_changes(self):
        declined = dict(TRAINED_CORRECT, mmlu=1)  # 5/8 -> 1/8, a 0.5 decline
        strict = build_claim_report(
            _manifest(self.tmpdir), baseline_metrics=_baseline(),
            epoch_metrics=_run(17, trained=declined) + _run(42, trained=declined),
        )
        self.assertFalse(
            strict["primary_table"]["modality_groups"][MODALITY_LIKELIHOOD_RANKED]
            ["rows"][0]["multiple_choice_only_gate_passes"]
        )
        lenient = build_claim_report(
            _manifest(self.tmpdir, **{"selection.capability_gates.mmlu_max_decline": 0.9}),
            baseline_metrics=_baseline(),
            epoch_metrics=_run(17, trained=declined) + _run(42, trained=declined),
        )
        self.assertTrue(
            lenient["primary_table"]["modality_groups"][MODALITY_LIKELIHOOD_RANKED]
            ["rows"][0]["multiple_choice_only_gate_passes"]
        )

    def test_caption_names_all_four_confounded_axes(self):
        caption = self._report()["primary_table"]["caption"].lower()
        self.assertIn("chat template", caption)
        self.assertIn("sampl", caption)          # sampled decoding vs deterministic
        self.assertIn("token budget", caption)
        self.assertIn("scoring method", caption)
        self.assertIn("#30", caption)

    def test_counts_are_derived_and_correct_at_two_and_three_seeds(self):
        two = self._report(seeds=(17, 42))
        self.assertEqual(two["completed_seed_count"], 2)
        self.assertEqual(two["trained_checkpoint_count"], 2 * EPOCHS)

        three = self._report(seeds=(17, 42, 2026))
        self.assertEqual(three["completed_seed_count"], 3)
        self.assertEqual(three["trained_checkpoint_count"], 3 * EPOCHS)
        self.assertEqual(three["completed_seeds"], [17, 42, 2026])


class CompositeTests(unittest.TestCase):
    def test_block_always_carries_the_decomposition(self):
        block = visible_composite_block(_baseline(), _doc(TRAINED_CORRECT))
        self.assertEqual(set(block["per_benchmark_delta"]),
                         set(claim_tables.VISIBLE_SAFETY_BENCHMARKS))
        self.assertIn("dominant_benchmark", block)

    def test_rendering_the_composite_alone_fails(self):
        with self.assertRaises(CompositeWithoutDecompositionError):
            render_composite({"composite_absolute_delta": 0.4})
        with self.assertRaises(CompositeWithoutDecompositionError):
            render_composite({"composite_absolute_delta": 0.4, "per_benchmark_delta": {}})

    def test_report_composites_carry_per_benchmark_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_claim_report(
                _manifest(Path(tmp)), baseline_metrics=_baseline(),
                epoch_metrics=_run(17) + _run(42),
            )
        for entry in report["visible_composite"]:
            self.assertEqual(set(entry["per_benchmark_delta"]),
                             set(claim_tables.VISIBLE_SAFETY_BENCHMARKS))
            self.assertIn("decomposition_note", entry)


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manifest = _manifest(Path(self.tmp.name))
        self.report = build_claim_report(
            self.manifest, baseline_metrics=_baseline(),
            epoch_metrics=_run(17) + _run(42) + _run(2026),
        )

    def test_every_contrast_has_a_bootstrap_interval_with_conditionality(self):
        contrasts = {(b["run_seed"], b["epoch"], b["benchmark"]) for b in self.report["paired_bootstrap"]}
        for seed in (17, 42, 2026):
            for epoch in range(1, EPOCHS + 1):
                for name in claim_tables.BENCHMARKS:
                    self.assertIn((seed, epoch, name), contrasts)
                self.assertIn((seed, epoch, "visible_composite"), contrasts)
        for entry in self.report["paired_bootstrap"]:
            self.assertIn("conditional on the specific evaluated example IDs",
                          entry["conditional_on"])
            self.assertEqual(entry["interval"], "95_percentile_paired_by_fixed_example_id")

    def test_bootstrap_reads_seed_and_replicates_from_the_given_manifest(self):
        analysis_cfg = load_manifest(self.manifest)["analysis"]
        for entry in self.report["paired_bootstrap"]:
            self.assertEqual(entry["seed"], analysis_cfg["bootstrap_seed"])
            self.assertEqual(entry["replicates"], analysis_cfg["bootstrap_replicates"])

    def test_mcnemar_marks_non_binary_benchmarks_not_applicable(self):
        # tensor_trust_hijack's real metric is (HRR+DV)/2, so per-item scores can
        # be 0.5 -- McNemar does not apply and must be flagged, not crash.
        def half_credit_doc(seed, epoch):
            doc = _doc(TRAINED_CORRECT, seed=seed, epoch=epoch)
            doc["benchmarks"]["tensor_trust_hijack"]["items"]["x0"]["score"] = 0.5
            return doc

        with tempfile.TemporaryDirectory() as tmp:
            report = build_claim_report(
                _manifest(Path(tmp)), baseline_metrics=_baseline(),
                epoch_metrics=[half_credit_doc(17, e) for e in range(1, EPOCHS + 1)]
                + _run(42),
            )
        entry = next(m for m in report["mcnemar_exact"]
                     if m["benchmark"] == "tensor_trust_hijack" and m["run_seed"] == 17)
        self.assertFalse(entry["applicable"])
        self.assertIn("binary", entry["reason"])
        # binary benchmarks in the same report still get a real test
        gsm8k = next(m for m in report["mcnemar_exact"]
                     if m["benchmark"] == "gsm8k" and m["run_seed"] == 17)
        self.assertIn("p_value", gsm8k)

    def test_mcnemar_reported_per_benchmark(self):
        keyed = {(m["run_seed"], m["epoch"], m["benchmark"]) for m in self.report["mcnemar_exact"]}
        for name in claim_tables.BENCHMARKS:
            self.assertIn((17, 1, name), keyed)
        gsm8k = next(m for m in self.report["mcnemar_exact"]
                     if m["benchmark"] == "gsm8k" and m["run_seed"] == 17 and m["epoch"] == 1)
        # baseline 5/6 right, trained 1/6 right, over identical IDs: 4 discordant,
        # all baseline-only -> two-sided exact p = 2 * 0.5**4 = 0.125.
        self.assertEqual(gsm8k["discordant_baseline_only"], 4)
        self.assertEqual(gsm8k["discordant_trained_only"], 0)
        self.assertAlmostEqual(gsm8k["p_value"], 0.125)

    def test_cross_run_summary_framing(self):
        block = self.report["cross_run_summary"]["epoch_1"]["gsm8k"]
        self.assertEqual(block["n_runs"], 3)
        framing = block["framing"].lower()
        self.assertIn("population", framing)
        self.assertIn("descriptive", framing)
        self.assertIn("run-to-run", framing)
        self.assertIn("must not be described as seed variance", framing)
        self.assertNotIn("per_seed", json.dumps(self.report["cross_run_summary"]))
        self.assertIn("population_standard_deviation", block)
        self.assertNotIn("confidence_interval", json.dumps(block))

    def test_cross_run_summary_correct_at_two_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_claim_report(
                _manifest(Path(tmp)), baseline_metrics=_baseline(),
                epoch_metrics=_run(17) + _run(42),
            )
        self.assertEqual(report["cross_run_summary"]["epoch_1"]["gsm8k"]["n_runs"], 2)


class DeterminismTests(unittest.TestCase):
    def test_same_frozen_inputs_reproduce_byte_identical_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest(Path(tmp))
            args = dict(baseline_metrics=_baseline(),
                        epoch_metrics=_run(17) + _run(42) + _run(2026))
            first = render_report(build_claim_report(manifest, **args))
            second = render_report(build_claim_report(manifest, **args))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
