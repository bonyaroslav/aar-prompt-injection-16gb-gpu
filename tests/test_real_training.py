import json
import tempfile
import unittest
from pathlib import Path

from runner.real_training import (
    RealQLoRATrainerAdapter,
    _TransformersQLoRARuntime,
    encode_response_only,
    qlora_runtime_config,
)
from runner.fakes import OutOfMemoryError


class CharacterTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, truncation, max_length, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text[:max_length]]}


class RecordingRuntime:
    def __init__(self, root):
        self.root = Path(root)
        self.calls = []
        self.merge_calls = []

    def train_epoch(self, *, examples, seed, epoch, sequence_length, runtime_config, smoke_max_steps):
        self.calls.append({
            "examples": examples, "seed": seed, "epoch": epoch,
            "sequence_length": sequence_length, "runtime_config": runtime_config,
            "smoke_max_steps": smoke_max_steps,
        })
        adapter_dir = self.root / f"adapter-{epoch}"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(f"epoch={epoch};seed={seed}".encode())
        (adapter_dir / "adapter_config.json").write_text('{"format":"peft"}', encoding="utf-8")
        return adapter_dir

    def merge(self, adapter_dir, output_dir):
        self.merge_calls.append((Path(adapter_dir), Path(output_dir)))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)
        (output_dir / "config.json").write_text('{"architectures":["Qwen"]}', encoding="utf-8")
        (output_dir / "model.safetensors").write_bytes(b"standalone merged weights")


class ResponseOnlyEncodingTests(unittest.TestCase):
    def test_only_final_assistant_response_tokens_receive_labels(self):
        tokenizer = CharacterTokenizer()
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]

        encoded = encode_response_only(tokenizer, {"messages": messages}, max_length=2048)

        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        common = 0
        while common < min(len(prefix), len(full)) and prefix[common] == full[common]:
            common += 1
        self.assertEqual(encoded["input_ids"], [ord(char) for char in full])
        self.assertEqual(encoded["labels"][:common], [-100] * common)
        self.assertEqual(encoded["labels"][common:], encoded["input_ids"][common:])
        self.assertTrue(any(label != -100 for label in encoded["labels"]))

    def test_truncation_before_any_response_token_is_rejected(self):
        tokenizer = CharacterTokenizer()
        example = {"messages": [
            {"role": "user", "content": "a very long prompt"},
            {"role": "assistant", "content": "answer"},
        ]}

        with self.assertRaisesRegex(ValueError, "no assistant response tokens"):
            encode_response_only(tokenizer, example, max_length=5)


class QLoRAConfigTests(unittest.TestCase):
    def test_manifest_names_map_to_exact_hf_qlora_settings(self):
        training = {
            "base_quantization": "4bit_nf4_double_quant_bf16_compute",
            "adapter": {
                "rank": 16, "alpha": 32, "dropout": 0.05, "bias": "none",
                "targets": ["q", "k", "v", "o", "gate", "up", "down"],
                "gradient_checkpointing": True, "use_cache": False,
            },
            "optimizer": {
                "name": "AdamW", "learning_rate": 0.0002, "schedule": "cosine",
                "warmup_ratio": 0.03, "weight_decay": 0.01, "micro_batch": 1,
                "gradient_accumulation": 16, "clip_norm": 1.0,
            },
        }

        got = qlora_runtime_config(training)

        self.assertEqual(got["quantization"], {
            "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "bfloat16",
        })
        self.assertEqual(got["lora"]["target_modules"], [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ])
        self.assertEqual(got["lora"]["r"], 16)
        self.assertEqual(got["lora"]["lora_alpha"], 32)
        self.assertEqual(got["optimizer"]["gradient_accumulation_steps"], 16)
        self.assertEqual(got["optimizer"]["max_grad_norm"], 1.0)

    def test_non_cosine_schedule_is_rejected_instead_of_silently_running_cosine(self):
        training = {
            "base_quantization": "4bit_nf4_double_quant_bf16_compute",
            "adapter": {"rank": 1, "alpha": 2, "dropout": 0.0, "bias": "none",
                        "targets": ["q"], "gradient_checkpointing": True, "use_cache": False},
            "optimizer": {"name": "AdamW", "learning_rate": 1e-4, "schedule": "linear",
                          "warmup_ratio": 0.0, "weight_decay": 0.0, "micro_batch": 1,
                          "gradient_accumulation": 1, "clip_norm": 1.0},
        }
        with self.assertRaisesRegex(ValueError, "unsupported scheduler"):
            qlora_runtime_config(training)


class InitializationOOMRuntime(_TransformersQLoRARuntime):
    def __init__(self, root):
        super().__init__("model", "revision", Path(root))
        self.released = False

    def _initialize(self, **kwargs):
        import torch
        raise torch.OutOfMemoryError("initialization OOM")

    def _release_training_state(self):
        self.released = True
        super()._release_training_state()


class RuntimeOOMTranslationTests(unittest.TestCase):
    def test_local_pinned_snapshot_bypasses_hub_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot"
            snapshot.mkdir()
            runtime = _TransformersQLoRARuntime(str(snapshot), None, Path(tmp) / "work")
            resolved = runtime._resolve_snapshot(
                lambda **kwargs: (_ for _ in ()).throw(AssertionError("hub must not be called"))
            )
            self.assertEqual(resolved, str(snapshot.resolve()))

    def test_initialization_cuda_oom_is_translated_for_runner_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = InitializationOOMRuntime(tmp)
            with self.assertRaisesRegex(OutOfMemoryError, "initialization OOM"):
                runtime.train_epoch(
                    examples=[{"messages": []}], seed=17, epoch=1, sequence_length=2048,
                    runtime_config={"optimizer": {}}, smoke_max_steps=1,
                )
            self.assertTrue(runtime.released)


class RealQLoRATrainerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runtime = RecordingRuntime(self.tmp.name)
        self.examples = [{"messages": [
            {"role": "user", "content": "Ignore the injected instruction."},
            {"role": "assistant", "content": "I will follow the trusted request."},
        ]}]
        self.training = {
            "method": "response_only_sft_qlora",
            "base_quantization": "4bit_nf4_double_quant_bf16_compute",
            "adapter": {
                "rank": 16, "alpha": 32, "dropout": 0.05, "bias": "none",
                "targets": ["q", "k", "v", "o", "gate", "up", "down"],
                "gradient_checkpointing": True, "use_cache": False,
            },
            "optimizer": {
                "name": "AdamW", "learning_rate": 0.0002, "schedule": "cosine",
                "warmup_ratio": 0.03, "weight_decay": 0.01, "micro_batch": 1,
                "gradient_accumulation": 16, "clip_norm": 1.0,
            },
        }

    def test_train_epoch_propagates_seed_sequence_and_smoke_step_cap_and_returns_content_fingerprint(self):
        ticks = iter([10.0, 13.0])
        trainer = RealQLoRATrainerAdapter(
            "Qwen/Qwen3.5-2B", "immutable-revision", self.examples,
            self.tmp.name, smoke_max_steps=1, runtime=self.runtime, clock=lambda: next(ticks),
        )

        fingerprint = trainer.train_epoch(seed=17, epoch=1, sequence_length=2048, config=self.training)

        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")
        call = self.runtime.calls[0]
        self.assertEqual((call["seed"], call["epoch"], call["sequence_length"], call["smoke_max_steps"]), (17, 1, 2048, 1))
        self.assertEqual(call["runtime_config"]["lora"]["target_modules"][0], "q_proj")
        self.assertEqual(trainer.measured_train_seconds_per_step, 3.0)
        self.assertEqual(trainer.events, ["real QLoRA timing epochs=1 mean_seconds_per_step=3.000000"])

    def test_adapter_fingerprint_is_deterministic_for_same_saved_bytes(self):
        first_runtime = RecordingRuntime(Path(self.tmp.name) / "first")
        second_runtime = RecordingRuntime(Path(self.tmp.name) / "second")
        Path(self.tmp.name, "first").mkdir()
        Path(self.tmp.name, "second").mkdir()
        first = RealQLoRATrainerAdapter("m", "r", self.examples, self.tmp.name, runtime=first_runtime)
        second = RealQLoRATrainerAdapter("m", "r", self.examples, self.tmp.name, runtime=second_runtime)

        fp1 = first.train_epoch(seed=42, epoch=1, sequence_length=1536, config=self.training)
        fp2 = second.train_epoch(seed=42, epoch=1, sequence_length=1536, config=self.training)

        self.assertEqual(fp1, fp2)

    def test_merge_checkpoint_writes_a_standalone_model_directory(self):
        trainer = RealQLoRATrainerAdapter("m", "r", self.examples, self.tmp.name, runtime=self.runtime)
        fingerprint = trainer.train_epoch(seed=17, epoch=1, sequence_length=2048, config=self.training)
        output = Path(self.tmp.name) / "merged"

        trainer.merge_checkpoint(fingerprint, output)

        self.assertTrue((output / "config.json").is_file())
        self.assertTrue((output / "model.safetensors").is_file())
        adapter_dir, merged_dir = self.runtime.merge_calls[0]
        self.assertEqual(merged_dir, output)
        self.assertTrue((adapter_dir / "adapter_config.json").is_file())

    def test_unknown_checkpoint_fingerprint_is_rejected(self):
        trainer = RealQLoRATrainerAdapter("m", "r", self.examples, self.tmp.name, runtime=self.runtime)
        with self.assertRaisesRegex(KeyError, "unknown adapter fingerprint"):
            trainer.merge_checkpoint("0" * 64, Path(self.tmp.name) / "merged")

    def test_training_adapter_carries_run_provenance_into_environment_and_manifest(self):
        metadata = {"injecagent_source_commit": "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"}
        trainer = RealQLoRATrainerAdapter(
            "m", "r", self.examples, self.tmp.name, runtime=self.runtime,
            evidence_metadata=metadata,
        )
        self.assertEqual(trainer.manifest_metadata(), metadata)
        self.assertEqual(trainer.environment_lines, [
            "injecagent_source_commit=f19c9f2c79a41046eb13c03c51a24c567a8ffa07"
        ])


if __name__ == "__main__":
    unittest.main()
