import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runner.real_training import (
    RealQLoRATrainerAdapter,
    _TransformersQLoRARuntime,
    _deterministic_epoch_order,
    _step_example_indexes,
    encode_response_only,
    qlora_runtime_config,
)
from runner.fakes import OutOfMemoryError
from runner.ablation_training import MidEpochCheckpointStore
from runner.recovery import RecoveryWorkspace, StageSignature


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

    def begin_ablation_epoch(self, *, examples, seed, epoch, sequence_length, runtime_config, smoke_max_steps):
        self.ablation_begin = {
            "examples": examples, "seed": seed, "epoch": epoch,
            "sequence_length": sequence_length, "runtime_config": runtime_config,
            "smoke_max_steps": smoke_max_steps,
        }
        self.ablation_steps = []
        self.restored_state = None
        return self

    def optimizer_safe_step(self, step_index):
        self.ablation_steps.append(step_index)

    def capture_mid_epoch_state(self, step_index):
        return {
            "adapter_weights": {"adapter": b"recorded-adapter"},
            "optimizer_state": {"step": step_index},
            "scheduler_state": {"last_epoch": step_index},
            "cpu_rng_state": (3, (1, 2, 3), None),
            "cuda_rng_state": [b"recorded-cuda-rng"],
            "step_index": step_index,
        }

    def restore_mid_epoch_state(self, state):
        self.restored_state = state

    def save_ablation_adapter(self, *, seed, epoch, sequence_length):
        adapter_dir = self.root / f"ablation-adapter-{epoch}"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(
            f"ablation-epoch={epoch};seed={seed};seq={sequence_length}".encode()
        )
        (adapter_dir / "adapter_config.json").write_text('{"format":"peft"}', encoding="utf-8")
        return adapter_dir


class InterruptingRecordingRuntime(RecordingRuntime):
    def __init__(self, root, *, interrupt_at=None):
        super().__init__(root)
        self.interrupt_at = interrupt_at
        self.executed_steps = []

    def optimizer_safe_step(self, step_index):
        if step_index == self.interrupt_at:
            raise RuntimeError(f"injected interruption at optimizer step {step_index}")
        self.executed_steps.append(step_index)


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
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            class OutOfMemoryError(RuntimeError):
                pass

            cuda = FakeCuda()

        with tempfile.TemporaryDirectory() as tmp:
            runtime = InitializationOOMRuntime(tmp)
            with patch.dict(sys.modules, {"torch": FakeTorch}):
                with self.assertRaisesRegex(OutOfMemoryError, "initialization OOM"):
                    runtime.train_epoch(
                        examples=[{"messages": []}], seed=17, epoch=1, sequence_length=2048,
                        runtime_config={"optimizer": {}}, smoke_max_steps=1,
                    )
            self.assertTrue(runtime.released)

    def test_mid_epoch_state_captures_and_restores_adapter_optimizer_scheduler_and_rng(self):
        """Dropping any mutable field would make a restarted epoch diverge."""
        class Optimizer:
            def __init__(self):
                self.loaded = None
                self.zeroed = False

            def state_dict(self):
                return {"moment": b"optimizer-moment"}

            def load_state_dict(self, state):
                self.loaded = state

            def zero_grad(self, *, set_to_none):
                self.zeroed = set_to_none

        class Scheduler:
            def __init__(self):
                self.loaded = None

            def state_dict(self):
                return {"last_epoch": 7}

            def load_state_dict(self, state):
                self.loaded = state

        class Cuda:
            def __init__(self):
                self.restored = None

            def is_available(self):
                return True

            def get_rng_state_all(self):
                return [b"cuda-rng"]

            def set_rng_state_all(self, state):
                self.restored = state

        class Torch:
            def __init__(self):
                self.cuda = Cuda()
                self.cpu_rng_restored = None

            def get_rng_state(self):
                return b"cpu-rng"

            def set_rng_state(self, state):
                self.cpu_rng_restored = state

        torch = Torch()
        peft = type("Peft", (), {})()
        peft.get_peft_model_state_dict = lambda model: {"lora.weight": b"adapter"}
        restored_adapters = []
        peft.set_peft_model_state_dict = lambda model, state: restored_adapters.append(state)
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = _TransformersQLoRARuntime("model", "revision", Path(temporary_directory))
            runtime.model = object()
            runtime.optimizer = Optimizer()
            runtime.scheduler = Scheduler()

            with patch.dict(sys.modules, {"torch": torch, "peft": peft}):
                state = runtime.capture_mid_epoch_state(step_index=7)
                runtime.restore_mid_epoch_state(state)

            loaded_optimizer = runtime.optimizer.loaded
            loaded_scheduler = runtime.scheduler.loaded
            optimizer_zeroed = runtime.optimizer.zeroed

        self.assertEqual(state, {
            "adapter_weights": {"lora.weight": b"adapter"},
            "optimizer_state": {"moment": b"optimizer-moment"},
            "scheduler_state": {"last_epoch": 7},
            "cpu_rng_state": b"cpu-rng",
            "cuda_rng_state": [b"cuda-rng"],
            "step_index": 7,
        })
        self.assertEqual(restored_adapters, [{"lora.weight": b"adapter"}])
        self.assertEqual(loaded_optimizer, {"moment": b"optimizer-moment"})
        self.assertEqual(loaded_scheduler, {"last_epoch": 7})
        self.assertEqual(torch.cpu_rng_restored, b"cpu-rng")
        self.assertEqual(torch.cuda.restored, [b"cuda-rng"])
        self.assertTrue(optimizer_zeroed)

    def test_step_index_reconstructs_the_seeded_epoch_data_position(self):
        """Persisting a data cursor instead would add mutable recovery state."""
        order = _deterministic_epoch_order(seed=17, epoch=2, example_count=5)

        self.assertEqual(order, [1, 3, 2, 4, 0])
        self.assertEqual(
            _step_example_indexes(order, step_index=1, micro_batch_size=1,
                                  gradient_accumulation_steps=2),
            [2, 4],
        )


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

    def test_ablation_bridge_uses_injected_runtime_and_reports_save_measurements(self):
        """Bypassing the runtime seam would make the CPU recovery path untestable."""
        recovery_root = Path(self.tmp.name) / "recovery"
        workspace = RecoveryWorkspace(recovery_root, Path(self.tmp.name) / "evidence")
        signature = StageSignature.create(
            manifest_digest="sha256:ablation-manifest", protocol_version="ablation-v1",
            upstream_commit="a" * 40, upstream_tree="b" * 40, model_revision="c" * 40,
            seed=17, stage="training", epoch=1, checkpoint_digest="sha256:base",
            effective_evaluation_config={"sequence_length": 2048}, expected_example_ids=[],
        )
        store = MidEpochCheckpointStore(workspace, "adapter-bridge", signature)
        trainer = RealQLoRATrainerAdapter(
            "m", "r", self.examples, self.tmp.name, runtime=self.runtime,
        )

        result = trainer.run_ablation_epoch(
            protocol_version="ablation-v1", seed=17, epoch=1, sequence_length=2048,
            config=self.training, checkpoint_store=store, total_steps=2,
            checkpoint_interval=2,
        )

        self.assertEqual(result.checkpoint_steps, [2])
        self.assertEqual(self.runtime.ablation_begin["seed"], 17)
        self.assertEqual(self.runtime.ablation_steps, [0, 1])
        self.assertEqual(store.load()["step_index"], 2)
        self.assertGreater(result.checkpoints[0].byte_count, 0)

    def test_ablation_checkpoint_is_registered_for_existing_merge_path(self):
        workspace = RecoveryWorkspace(
            Path(self.tmp.name) / "recovery", Path(self.tmp.name) / "evidence"
        )
        signature = StageSignature.create(
            manifest_digest="sha256:ablation-manifest", protocol_version="ablation-v1",
            upstream_commit="a" * 40, upstream_tree="b" * 40, model_revision="c" * 40,
            seed=42, stage="training", epoch=1, checkpoint_digest="sha256:base",
            effective_evaluation_config={"sequence_length": 2048}, expected_example_ids=[],
        )
        trainer = RealQLoRATrainerAdapter("m", "r", self.examples, self.tmp.name, runtime=self.runtime)
        trainer.run_ablation_epoch(
            protocol_version="ablation-v1", seed=42, epoch=1, sequence_length=2048,
            config=self.training, checkpoint_store=MidEpochCheckpointStore(workspace, "save-bridge", signature),
            total_steps=1, checkpoint_interval=1,
        )

        fingerprint = trainer.save_ablation_checkpoint(seed=42, epoch=1, sequence_length=2048)
        merged = Path(self.tmp.name) / "merged-ablation"
        trainer.merge_checkpoint(fingerprint, merged)

        self.assertTrue((merged / "config.json").is_file())
        self.assertTrue(self.runtime.merge_calls)

    def test_ablation_bridge_requires_an_explicit_seed(self):
        """A default seed would silently reconstruct a different epoch order."""
        trainer = RealQLoRATrainerAdapter(
            "m", "r", self.examples, self.tmp.name, runtime=self.runtime,
        )
        workspace = RecoveryWorkspace(
            Path(self.tmp.name) / "recovery", Path(self.tmp.name) / "evidence"
        )
        store = MidEpochCheckpointStore(workspace, "explicit-seed", StageSignature.create(
            manifest_digest="sha256:ablation-manifest", protocol_version="ablation-v1",
            upstream_commit="a" * 40, upstream_tree="b" * 40, model_revision="c" * 40,
            seed=17, stage="training", epoch=1, checkpoint_digest="sha256:base",
            effective_evaluation_config={"sequence_length": 2048}, expected_example_ids=[],
        ))

        with self.assertRaises(TypeError):
            trainer.run_ablation_epoch(
                protocol_version="ablation-v1", epoch=1, sequence_length=2048,
                config=self.training, checkpoint_store=store, total_steps=1,
            )

    def test_ablation_bridge_rejects_attempt_one_before_runtime_initialization(self):
        """Initializing Attempt-1 here would itself change frozen computation state."""
        workspace = RecoveryWorkspace(
            Path(self.tmp.name) / "recovery", Path(self.tmp.name) / "evidence"
        )
        store = MidEpochCheckpointStore(workspace, "attempt-one-bridge", StageSignature.create(
            manifest_digest="sha256:manifest", protocol_version="phase1-2026-08-29",
            upstream_commit="a" * 40, upstream_tree="b" * 40, model_revision="c" * 40,
            seed=17, stage="training", epoch=1, checkpoint_digest="sha256:base",
            effective_evaluation_config={}, expected_example_ids=[],
        ))
        trainer = RealQLoRATrainerAdapter(
            "m", "r", self.examples, self.tmp.name, runtime=self.runtime,
        )

        with self.assertRaisesRegex(ValueError, "ablation-only"):
            trainer.run_ablation_epoch(
                protocol_version="phase1-2026-08-29", seed=17, epoch=1,
                sequence_length=2048, config=self.training, checkpoint_store=store,
                total_steps=1,
            )

        self.assertFalse(hasattr(self.runtime, "ablation_begin"))

    def test_ablation_bridge_resumes_each_step_once_through_injected_runtime(self):
        """Restarting at an off-by-one step would repeat or skip the trainer seam."""
        workspace = RecoveryWorkspace(
            Path(self.tmp.name) / "recovery", Path(self.tmp.name) / "evidence"
        )
        signature = StageSignature.create(
            manifest_digest="sha256:ablation-manifest", protocol_version="ablation-v1",
            upstream_commit="a" * 40, upstream_tree="b" * 40, model_revision="c" * 40,
            seed=17, stage="training", epoch=1, checkpoint_digest="sha256:base",
            effective_evaluation_config={"sequence_length": 2048}, expected_example_ids=[],
        )
        store = MidEpochCheckpointStore(workspace, "exact-adapter-resume", signature)
        interrupted_runtime = InterruptingRecordingRuntime(self.tmp.name, interrupt_at=2)
        interrupted = RealQLoRATrainerAdapter(
            "m", "r", self.examples, self.tmp.name, runtime=interrupted_runtime,
        )
        with self.assertRaisesRegex(RuntimeError, "optimizer step 2"):
            interrupted.run_ablation_epoch(
                protocol_version="ablation-v1", seed=17, epoch=1, sequence_length=2048,
                config=self.training, checkpoint_store=store, total_steps=4,
                checkpoint_interval=1,
            )

        resumed_runtime = InterruptingRecordingRuntime(self.tmp.name)
        resumed = RealQLoRATrainerAdapter(
            "m", "r", self.examples, self.tmp.name, runtime=resumed_runtime,
        ).run_ablation_epoch(
            protocol_version="ablation-v1", seed=17, epoch=1, sequence_length=2048,
            config=self.training, checkpoint_store=store, total_steps=4,
            checkpoint_interval=1,
        )

        self.assertEqual(interrupted_runtime.executed_steps + resumed_runtime.executed_steps, [0, 1, 2, 3])
        self.assertTrue(resumed.mid_epoch_resume_fired)
        self.assertEqual(resumed_runtime.restored_state["step_index"], 2)


if __name__ == "__main__":
    unittest.main()
