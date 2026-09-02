import unittest

from runner.diagnostic_report import (
    ALL_STATES,
    CHECKPOINT_STATES,
    build_chatmode_report,
    render_report,
)

MANIFEST = {
    "diagnostic_version": "diag-chatmode-mmlu-2026-09-02",
    "downstream_of": {"protocol_version": "phase1-2026-08-29"},
    "analysis": {"bootstrap_seed": 303030, "bootstrap_replicates": 200},
    "ambiguity_rule": {
        "delta_threshold": 0.02,
        "chance_level": 0.25,
        "chance_band": 0.03,
        "robustness_rerun": {"rationale": "leading-space tokenization caveat"},
    },
    "reading_table": {
        "flat_or_up_in_chat_mode": "modality, not the chat wrapper",
        "collapses_in_chat_mode": "chat pathway specifically damaged",
    },
}

IDS = [f"mmlu-{i:04d}" for i in range(40)]


def bench(scores):
    return {"items": {ident: {"score": float(s), "valid": True} for ident, s in zip(IDS, scores)}}


def uniform(value):
    return bench([value] * len(IDS))


class BuildChatmodeReportTests(unittest.TestCase):
    def _states(self, chat_scores_by_state):
        return {state: bench(chat_scores_by_state[state]) for state in ALL_STATES}

    def test_flat_or_up_when_no_checkpoint_declines(self):
        raw_baseline = uniform(1.0)
        epochs = {state: uniform(1.0) for state in CHECKPOINT_STATES}
        chat = self._states({state: [1.0] * len(IDS) for state in ALL_STATES})

        report = build_chatmode_report(
            MANIFEST, attempt1_baseline_mmlu=raw_baseline,
            attempt1_epoch_mmlu=epochs, chatmode_mmlu=chat,
        )

        self.assertEqual(report["primary"]["reading"]["verdict"], "flat_or_up_in_chat_mode")
        self.assertIn("chat wrapper", report["primary"]["reading"]["reading"])
        self.assertFalse(report["robustness_rerun_required"])
        self.assertEqual(len(report["primary"]["per_state"]), 10)

    def test_collapse_when_every_checkpoint_drops_hard_in_chat_mode(self):
        raw_baseline = uniform(1.0)
        epochs = {state: uniform(1.0) for state in CHECKPOINT_STATES}
        # chat mode: baseline holds, every checkpoint halves.
        chat_scores = {"baseline": [1.0] * len(IDS)}
        for state in CHECKPOINT_STATES:
            chat_scores[state] = [1.0, 0.0] * (len(IDS) // 2)
        chat = self._states(chat_scores)

        report = build_chatmode_report(
            MANIFEST, attempt1_baseline_mmlu=raw_baseline,
            attempt1_epoch_mmlu=epochs, chatmode_mmlu=chat,
        )

        primary = report["primary"]
        self.assertEqual(primary["reading"]["verdict"], "collapses_in_chat_mode")
        self.assertEqual(primary["baseline_row"]["classification"], "flat_or_up")
        self.assertEqual(len(primary["reading"]["checkpoints_collapsed"]), 9)
        for row in primary["per_state"]:
            if row["state"] != "baseline":
                self.assertLess(row["absolute_delta"], -0.02)
                self.assertGreater(row["mcnemar_exact"]["n_discordant"], 0)

    def test_baseline_collapse_to_chance_reclassifies_as_confounded(self):
        # Attempt-1 raw: baseline 0.60, checkpoints 1.0. Chat mode: baseline falls
        # to chance (0.25, delta -0.35); each checkpoint declines only slightly
        # (0.95, delta -0.05) -- the baseline's decline dwarfs the checkpoints'.
        raw_baseline = bench([1.0, 1.0, 1.0, 0.0, 0.0] * 8)           # 0.60
        epochs = {state: uniform(1.0) for state in CHECKPOINT_STATES}  # 1.00
        chat = {}
        chat["baseline"] = bench([1.0, 0.0, 0.0, 0.0] * 10)           # 0.25 -> near chance
        for state in CHECKPOINT_STATES:
            chat[state] = bench([1.0] * 38 + [0.0] * 2)               # 0.95, delta -0.05

        report = build_chatmode_report(
            MANIFEST, attempt1_baseline_mmlu=raw_baseline,
            attempt1_epoch_mmlu=epochs, chatmode_mmlu=chat,
        )
        primary = report["primary"]
        self.assertEqual(primary["reading"]["verdict"], "confounded_by_baseline_modality_effect")
        self.assertIn("near chance", primary["reading"]["reading"])
        self.assertIn("baseline", primary["near_chance_states"])
        self.assertTrue(report["robustness_rerun_required"])

    def test_ambiguous_state_flags_the_robustness_rerun(self):
        raw_baseline = uniform(1.0)
        epochs = {state: uniform(1.0) for state in CHECKPOINT_STATES}
        # One example flips for one checkpoint -> |delta| = 1/40 = 0.025 -> not < 0.02.
        # Flip nothing for the rest; flip a single item back and forth to sit under threshold.
        chat_scores = {state: [1.0] * len(IDS) for state in ALL_STATES}
        chat_scores["seed17-epoch1"] = [0.0] + [1.0] * (len(IDS) - 1)  # delta = -0.025
        chat = self._states(chat_scores)
        report = build_chatmode_report(
            MANIFEST, attempt1_baseline_mmlu=raw_baseline,
            attempt1_epoch_mmlu=epochs, chatmode_mmlu=chat,
        )
        # -0.025 <= -0.02 -> classified as a collapse, not ambiguous.
        self.assertEqual(report["primary"]["reading"]["verdict"], "mixed")

    def test_true_ambiguity_triggers_and_second_table_is_reported(self):
        raw_baseline = uniform(1.0)
        epochs = {state: uniform(1.0) for state in CHECKPOINT_STATES}
        # Half the checkpoint's items flip down, half of *those* flip back up in raw:
        # construct a near-zero delta with a CI that crosses zero.
        chat_scores = {state: [1.0] * len(IDS) for state in ALL_STATES}
        # seed42-epoch2: delta exactly 0 (no change) -> |0| < 0.02 and CI is [0,0]
        # CI [0,0] does NOT cross zero (needs ci_low < 0 < ci_high). Use a mild wobble:
        wob = [1.0, 1.0, 0.0, 1.0] * (len(IDS) // 4)
        chat_scores["seed42-epoch2"] = wob
        epochs["seed42-epoch2"] = bench([1.0, 0.0, 1.0, 1.0] * (len(IDS) // 4))
        chat = self._states(chat_scores)

        report = build_chatmode_report(
            MANIFEST, attempt1_baseline_mmlu=raw_baseline,
            attempt1_epoch_mmlu=epochs, chatmode_mmlu=chat,
        )
        if report["robustness_rerun_required"]:
            self.assertIn("seed42-epoch2", report["primary"]["ambiguous_states"])
            self.assertFalse(report["robustness_rerun_performed"])
            with_rerun = build_chatmode_report(
                MANIFEST, attempt1_baseline_mmlu=raw_baseline,
                attempt1_epoch_mmlu=epochs, chatmode_mmlu=chat,
                chatmode_mmlu_no_leading_space=chat,
            )
            self.assertTrue(with_rerun["robustness_rerun_performed"])
            self.assertIn("no_leading_space", with_rerun)

    def test_rejects_mismatched_example_ids(self):
        raw_baseline = uniform(1.0)
        epochs = {state: uniform(1.0) for state in CHECKPOINT_STATES}
        chat = {state: uniform(1.0) for state in ALL_STATES}
        chat["baseline"] = {"items": {"mmlu-9999": {"score": 1.0, "valid": True}}}
        with self.assertRaisesRegex(ValueError, "identical example IDs"):
            build_chatmode_report(
                MANIFEST, attempt1_baseline_mmlu=raw_baseline,
                attempt1_epoch_mmlu=epochs, chatmode_mmlu=chat,
            )

    def test_rejects_missing_model_state(self):
        raw_baseline = uniform(1.0)
        epochs = {state: uniform(1.0) for state in CHECKPOINT_STATES}
        chat = {state: uniform(1.0) for state in ALL_STATES if state != "seed2026-epoch3"}
        with self.assertRaisesRegex(ValueError, "seed2026-epoch3"):
            build_chatmode_report(
                MANIFEST, attempt1_baseline_mmlu=raw_baseline,
                attempt1_epoch_mmlu=epochs, chatmode_mmlu=chat,
            )

    def test_render_report_is_byte_stable(self):
        raw_baseline = uniform(1.0)
        epochs = {state: uniform(1.0) for state in CHECKPOINT_STATES}
        chat = {state: uniform(1.0) for state in ALL_STATES}
        args = dict(attempt1_baseline_mmlu=raw_baseline, attempt1_epoch_mmlu=epochs,
                    chatmode_mmlu=chat)
        self.assertEqual(
            render_report(build_chatmode_report(MANIFEST, **args)),
            render_report(build_chatmode_report(MANIFEST, **args)),
        )


if __name__ == "__main__":
    unittest.main()
