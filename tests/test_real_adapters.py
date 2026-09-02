import json
import tempfile
import time
import unittest
from pathlib import Path

from runner.real_adapters import (
    RealDatasetAdapter,
    RealModelAdapter,
    RealScorerAdapter,
    RealTelemetryAdapter,
)


class RecordingBackend:
    def __init__(self, outputs=None, logits=None):
        self.outputs = list(outputs or [])
        self.logits = logits or [0.0, 0.0, 1.0, 0.0]
        self.generate_calls = []
        self.logit_calls = []
        self.decoding_calls = []

    def apply_decoding(self, **kwargs):
        self.decoding_calls.append(kwargs)

    def generate(self, prompt, **kwargs):
        self.generate_calls.append((prompt, kwargs))
        return self.outputs.pop(0) if self.outputs else "generated"

    def candidate_logits(self, prompt, candidates, use_chat_template=True):
        self.logit_calls.append((prompt, candidates, use_chat_template))
        return self.logits


class RealDatasetAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.suite = Path(self.tmp.name) / "visible"
        self.heldout = Path(self.tmp.name) / "restricted"
        self.suite.mkdir()
        self.heldout.mkdir()

    def _write_jsonl(self, root, name, rows):
        (root / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )

    def test_caps_real_publisher_rows_and_assigns_content_stable_ids(self):
        rows = [{"prompt": "first", "answer": "1"}, {"prompt": "second", "answer": "2"}]
        self._write_jsonl(self.suite, "gsm8k", rows)
        adapter = RealDatasetAdapter(self.suite, self.heldout, max_items_per_benchmark=1)

        got = adapter.load_items("gsm8k", sample_count=200)

        self.assertEqual(len(got), 1)
        self.assertRegex(got[0]["id"], r"^gsm8k-[0-9a-f]{16}$")
        first_id = got[0]["id"]
        self._write_jsonl(self.suite, "gsm8k", list(reversed(rows)))
        reversed_items = adapter.load_items("gsm8k", sample_count=200)
        self.assertNotEqual(reversed_items[0]["id"], first_id)
        self._write_jsonl(self.suite, "gsm8k", rows)
        self.assertEqual(adapter.load_items("gsm8k", 200)[0]["id"], first_id)

    def test_rejects_published_file_with_fewer_rows_than_the_requested_or_smoke_capped_count(self):
        self._write_jsonl(self.suite, "gsm8k", [{"prompt": "only", "answer": "1"}])
        adapter = RealDatasetAdapter(self.suite, self.heldout, max_items_per_benchmark=2)
        with self.assertRaisesRegex(ValueError, "expected 2 published rows"):
            adapter.load_items("gsm8k", 200)

    def test_rejects_published_rows_missing_benchmark_schema_fields(self):
        self._write_jsonl(self.suite, "gsm8k", [{"prompt": "question without answer"}])
        adapter = RealDatasetAdapter(self.suite, self.heldout, max_items_per_benchmark=1)
        with self.assertRaisesRegex(ValueError, "gsm8k.*missing required fields.*answer"):
            adapter.load_items("gsm8k", 200)

    def test_injecagent_reads_only_from_external_heldout_root_and_attaches_sidecars(self):
        self._write_jsonl(self.heldout, "injecagent", [{
            "user_prompt": "secret", "attacker_tools": ["Bad"], "user_tool": "Good",
            "attacker_instruction": "attack", "attack": "base",
        }])
        (self.heldout / "injecagent_tool_dict.json").write_text('{"Bad": {}}', encoding="utf-8")
        (self.heldout / "injecagent_sim_responses.json").write_text('{"Bad": "sim"}', encoding="utf-8")
        (self.heldout / "injecagent_source_commit.txt").write_text("f19c9f2c79a41046eb13c03c51a24c567a8ffa07\n", encoding="utf-8")
        adapter = RealDatasetAdapter(self.suite, self.heldout, max_items_per_benchmark=1)

        got = adapter.load_items("injecagent", 200)

        self.assertEqual(got[0]["user_prompt"], "secret")
        self.assertEqual(got[0]["_tool_dict"], {"Bad": {}})
        self.assertEqual(got[0]["_sim_responses"], {"Bad": "sim"})
        self.assertEqual(adapter.manifest_metadata(), {
            "injecagent_source_commit": "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"
        })
        self.assertEqual(adapter.environment_lines, [
            "injecagent_source_commit=f19c9f2c79a41046eb13c03c51a24c567a8ffa07"
        ])


class RealModelAdapterTests(unittest.TestCase):
    def test_local_merged_checkpoint_bypasses_hub_resolution_and_loads_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            merged = Path(tmp) / "merged"
            merged.mkdir()
            (merged / "config.json").write_text("{}", encoding="utf-8")
            loaded = []

            model = RealModelAdapter(
                model_ref=str(merged), revision=None,
                backend_factory=lambda path: loaded.append(path) or RecordingBackend(),
            )

            self.assertEqual(loaded, [str(merged.resolve())])
            self.assertIsInstance(model.backend, RecordingBackend)

    def test_open_prompt_injection_uses_published_attacked_prompt_and_32_token_cap(self):
        backend = RecordingBackend(outputs=["negative"])
        model = RealModelAdapter(backend=backend)

        output = model.generate(
            "open_prompt_injection", {"attacked_prompt": "TARGET\nINJECTION"}, {"max_new_tokens": 32}
        )

        self.assertEqual(output, "negative")
        self.assertEqual(backend.generate_calls, [("TARGET\nINJECTION", {"max_new_tokens": 32})])

    def test_tensor_trust_hijack_generates_attack_and_defense_validity_arms_with_real_roles(self):
        backend = RecordingBackend(outputs=["access denied", "access granted"])
        model = RealModelAdapter(backend=backend)
        item = {"pre_prompt": "SYSTEM", "attack": "ATTACK", "post_prompt": "POST", "access_code": "CODE"}

        output = json.loads(model.generate("tensor_trust_hijack", item, {"max_new_tokens": 64}))

        self.assertEqual(output, {"attack": "access denied", "dv": "access granted"})
        self.assertEqual(backend.generate_calls[0], ([
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "ATTACK\nPOST"},
        ], {"max_new_tokens": 64}))
        self.assertEqual(backend.generate_calls[1][0][1]["content"], "CODE\nPOST")

    def test_tensor_trust_extract_uses_distinct_attack_and_validity_budgets(self):
        backend = RecordingBackend(outputs=["nothing", "access granted"])
        model = RealModelAdapter(backend=backend)
        item = {"pre_prompt": "S", "attack": "A", "post_prompt": "P", "access_code": "SECRET42"}

        model.generate("tensor_trust_extract", item, {"max_new_tokens": {"attack": 512, "dv": 64}})

        self.assertEqual([c[1]["max_new_tokens"] for c in backend.generate_calls], [512, 64])

    def test_records_per_real_model_call_timings_for_resource_projection_and_logs(self):
        ticks = iter([0.0, 2.0, 2.0, 5.0])
        backend = RecordingBackend(outputs=["attack", "validity"])
        model = RealModelAdapter(backend=backend, clock=lambda: next(ticks))
        item = {"pre_prompt": "S", "attack": "A", "post_prompt": "P", "access_code": "C"}

        model.generate("tensor_trust_hijack", item, {"max_new_tokens": 64})

        self.assertEqual(model.measured_seconds_per_item, {"tensor_trust_hijack": 2.5})
        self.assertEqual(model.events, [
            "real model timing benchmark=tensor_trust_hijack calls=2 mean_seconds=2.500000"
        ])

    def test_mmlu_scores_raw_prompt_candidate_logits_instead_of_generating(self):
        backend = RecordingBackend(logits=[-3.0, -2.0, 9.0, -4.0])
        model = RealModelAdapter(backend=backend)
        item = {"question": "Q?", "choices": ["one", "two", "three", "four"], "answer": 2}

        output = json.loads(model.generate("mmlu", item, {"max_new_tokens": 1}))

        self.assertEqual(output, {"predicted": 2})
        self.assertEqual(backend.generate_calls, [])
        prompt, candidates, chat = backend.logit_calls[0]
        self.assertEqual(prompt, "Question: Q?\n\nA. one\nB. two\nC. three\nD. four\nAnswer:")
        self.assertEqual(candidates, [" A", " B", " C", " D"])
        self.assertFalse(chat)

    def test_mmlu_chat_template_defaults_off_and_is_honoured_when_enabled(self):
        item = {"question": "Q?", "choices": ["one", "two", "three", "four"], "answer": 2}

        default_backend = RecordingBackend(logits=[-3.0, -2.0, 9.0, -4.0])
        RealModelAdapter(backend=default_backend).generate("mmlu", item, {"max_new_tokens": 1})
        _, default_candidates, default_chat = default_backend.logit_calls[0]
        self.assertEqual(default_candidates, [" A", " B", " C", " D"])
        self.assertFalse(default_chat)

        chat_backend = RecordingBackend(logits=[-3.0, -2.0, 9.0, -4.0])
        RealModelAdapter(backend=chat_backend, mmlu_use_chat_template=True).generate(
            "mmlu", item, {"max_new_tokens": 1}
        )
        _, chat_candidates, chat_flag = chat_backend.logit_calls[0]
        self.assertEqual(chat_candidates, [" A", " B", " C", " D"])
        self.assertTrue(chat_flag)

    def test_mmlu_candidate_strings_are_overridable_for_the_robustness_rerun(self):
        item = {"question": "Q?", "choices": ["one", "two", "three", "four"], "answer": 2}
        backend = RecordingBackend(logits=[-3.0, -2.0, 9.0, -4.0])
        model = RealModelAdapter(
            backend=backend, mmlu_use_chat_template=True,
            mmlu_candidate_strings=["A", "B", "C", "D"],
        )

        model.generate("mmlu", item, {"max_new_tokens": 1})

        _, candidates, chat_flag = backend.logit_calls[0]
        self.assertEqual(candidates, ["A", "B", "C", "D"])
        self.assertTrue(chat_flag)

    def test_injecagent_data_stealing_success_runs_the_paper_second_step(self):
        backend = RecordingBackend(outputs=["first tool call", "second tool call"])
        model = RealModelAdapter(backend=backend, apis={
            "system_prompt": "PAPER SYSTEM",
            "evaluate": lambda *args: {"eval": "succ", "parsed": "call"},
            "build_step2": lambda item, parsed, tools, cache: "PAPER STEP TWO",
        })
        item = {
            "user_prompt": "PAPER STEP ONE", "attacker_tools": ["Read", "Send"],
            "user_tool": "Good", "attacker_instruction": "steal", "attack": "ds",
            "_tool_dict": {"Read": {}}, "_sim_responses": {"Read": "secret"},
        }

        output = json.loads(model.generate("injecagent", item, {"max_new_tokens": 700}))

        self.assertEqual(output, {"step1": "first tool call", "step2": "second tool call"})
        self.assertEqual(backend.generate_calls[0][0], [
            {"role": "system", "content": "PAPER SYSTEM"},
            {"role": "user", "content": "PAPER STEP ONE"},
        ])
        self.assertEqual(backend.generate_calls[1][0][1]["content"], "PAPER STEP TWO")
        self.assertEqual([call[1]["max_new_tokens"] for call in backend.generate_calls], [700, 700])

    def test_injecagent_parser_exception_during_generation_is_preserved_for_invalid_scoring(self):
        backend = RecordingBackend(outputs=["malformed action"])
        model = RealModelAdapter(backend=backend, apis={
            "system_prompt": "SYSTEM",
            "evaluate": lambda *args: (_ for _ in ()).throw(ValueError("parse failed")),
            "build_step2": lambda *args: "unused",
        })
        item = {"user_prompt": "P", "attacker_tools": ["Bad"], "user_tool": "Good",
                "attacker_instruction": "steal", "attack": "dh"}

        output = json.loads(model.generate("injecagent", item, {"max_new_tokens": 700}))

        self.assertEqual(output, {"step1": "malformed action", "step2": None})


class RealScorerAdapterTests(unittest.TestCase):
    def test_upstream_rule_scorers_cover_visible_and_capability_items(self):
        scorer = RealScorerAdapter(apis={
            "opi_parsers": {"sst2": lambda text: 1 if "positive" in text else 0},
            "is_access_granted": lambda text: text.lower().startswith("access granted"),
            "access_code_leaked": lambda text, code: code.lower() in text.lower(),
            "extract_final_number": lambda text: text.rsplit(" ", 1)[-1],
            "follows_all": lambda text, ids, kwargs: text == "COMPLIANT",
        })

        opi = scorer.score("open_prompt_injection", {"injected_task": "sst2", "injected_label": 1}, "positive", {})
        hijack = scorer.score("tensor_trust_hijack", {}, json.dumps({"attack": "access denied", "dv": "access granted"}), {})
        extract = scorer.score("tensor_trust_extract", {"access_code": "SECRET42"}, json.dumps({"attack": "no leak", "dv": "access granted"}), {})
        mmlu = scorer.score("mmlu", {"answer": 2}, json.dumps({"predicted": 2}), {})
        gsm = scorer.score("gsm8k", {"answer": "42"}, "Answer: 42", {})
        ifeval = scorer.score("ifeval", {"instruction_id_list": ["x"], "kwargs": [{}]}, "COMPLIANT", {})

        self.assertEqual([opi["score"], hijack["score"], extract["score"], mmlu["score"], gsm["score"], ifeval["score"]], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def test_injecagent_invalid_output_is_preserved_as_invalid_failure(self):
        scorer = RealScorerAdapter(apis={"evaluate_injecagent": lambda *args: {"eval": "invalid"}})
        item = {"attacker_tools": ["Bad"], "user_tool": "Good", "attacker_instruction": "steal"}

        result = scorer.score("injecagent", item, json.dumps({"step1": "garbage", "step2": None}), {})

        self.assertEqual(result, {"score": 0.0, "valid": False, "eval": "invalid"})

    def test_injecagent_scorer_parser_exception_becomes_invalid_instead_of_aborting(self):
        scorer = RealScorerAdapter(apis={
            "evaluate_injecagent": lambda *args: (_ for _ in ()).throw(ValueError("parse failed"))
        })
        item = {"attacker_tools": ["Bad"], "user_tool": "Good", "attacker_instruction": "steal"}
        result = scorer.score("injecagent", item, json.dumps({"step1": "bad", "step2": None}), {})
        self.assertEqual(result, {"score": 0.0, "valid": False, "eval": "invalid"})


class RealTelemetryAdapterTests(unittest.TestCase):
    def test_start_and_stop_capture_real_shaped_samples(self):
        samples = iter(["100, 2048, 7", "101, 3072, 19", "102, 1024, 0"])
        telemetry = RealTelemetryAdapter(sample_interval_seconds=0.01, query=lambda: next(samples, "103, 1024, 0"))

        telemetry.start()
        time.sleep(0.025)
        rows = telemetry.stop()

        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), {"t", "vram_mb", "util_pct"})
        self.assertTrue(all(isinstance(row["vram_mb"], int) for row in rows))
        self.assertGreaterEqual(max(row["vram_mb"] for row in rows), 2048)


if __name__ == "__main__":
    unittest.main()
