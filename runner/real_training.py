"""Real response-only QLoRA trainer for the runner's existing trainer seam."""
from __future__ import annotations

import gc
import hashlib
import math
import random
import time
from pathlib import Path

from runner.fakes import OutOfMemoryError


_TARGET_MODULES = {
    "q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj",
    "gate": "gate_proj", "up": "up_proj", "down": "down_proj",
}


def encode_response_only(tokenizer, example: dict, max_length: int) -> dict:
    """Tokenize a chat example and mask every token before the final response."""
    messages = example.get("messages")
    if not messages or len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("training example must end with an assistant response")
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(
        prompt_text, truncation=True, max_length=max_length, add_special_tokens=False
    )["input_ids"]
    full_ids = tokenizer(
        full_text, truncation=True, max_length=max_length, add_special_tokens=False
    )["input_ids"]
    common = 0
    while common < min(len(prompt_ids), len(full_ids)) and prompt_ids[common] == full_ids[common]:
        common += 1
    labels = [-100] * common + list(full_ids[common:])
    if not any(label != -100 for label in labels):
        raise ValueError("training example has no assistant response tokens after truncation")
    return {"input_ids": list(full_ids), "labels": labels, "attention_mask": [1] * len(full_ids)}


def qlora_runtime_config(training: dict) -> dict:
    if training["base_quantization"] != "4bit_nf4_double_quant_bf16_compute":
        raise ValueError(f"unsupported frozen quantization: {training['base_quantization']!r}")
    adapter = training["adapter"]
    optimizer = training["optimizer"]
    if optimizer["schedule"].lower() != "cosine":
        raise ValueError(f"unsupported scheduler: {optimizer['schedule']!r}")
    try:
        target_modules = [_TARGET_MODULES[name] for name in adapter["targets"]]
    except KeyError as exc:
        raise ValueError(f"unsupported LoRA target shorthand: {exc.args[0]!r}") from exc
    return {
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        "lora": {
            "r": int(adapter["rank"]),
            "lora_alpha": int(adapter["alpha"]),
            "lora_dropout": float(adapter["dropout"]),
            "bias": adapter["bias"],
            "target_modules": target_modules,
            "task_type": "CAUSAL_LM",
        },
        "gradient_checkpointing": bool(adapter["gradient_checkpointing"]),
        "use_cache": bool(adapter["use_cache"]),
        "optimizer": {
            "name": optimizer["name"],
            "learning_rate": float(optimizer["learning_rate"]),
            "schedule": optimizer["schedule"],
            "warmup_ratio": float(optimizer["warmup_ratio"]),
            "weight_decay": float(optimizer["weight_decay"]),
            "micro_batch_size": int(optimizer["micro_batch"]),
            "gradient_accumulation_steps": int(optimizer["gradient_accumulation"]),
            "max_grad_norm": float(optimizer["clip_norm"]),
            "epochs": int(optimizer.get("epochs", 1)),
        },
    }


def _directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class RealQLoRATrainerAdapter:
    """Train and merge real PEFT adapters while retaining the runner interface."""

    def __init__(self, model_ref, revision, training_examples, work_dir,
                 smoke_max_steps=None, runtime=None, clock=time.monotonic,
                 evidence_metadata=None):
        self.model_ref = model_ref
        self.revision = revision
        self.training_examples = list(training_examples)
        if not self.training_examples:
            raise ValueError("at least one training example is required")
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.smoke_max_steps = int(smoke_max_steps) if smoke_max_steps is not None else None
        self.clock = clock
        self.runtime = runtime or _TransformersQLoRARuntime(model_ref, revision, self.work_dir)
        self._adapters: dict[str, Path] = {}
        self._train_measurements: list[tuple[float, int]] = []
        self._evidence_metadata = dict(evidence_metadata or {})

    def manifest_metadata(self) -> dict:
        return dict(self._evidence_metadata)

    @property
    def environment_lines(self) -> list[str]:
        return [f"{key}={value}" for key, value in sorted(self._evidence_metadata.items())]

    @property
    def measured_train_seconds_per_step(self) -> float:
        seconds = sum(duration for duration, _ in self._train_measurements)
        steps = sum(steps for _, steps in self._train_measurements)
        return seconds / steps if steps else 0.0

    @property
    def events(self) -> list[str]:
        if not self._train_measurements:
            return []
        return [
            f"real QLoRA timing epochs={len(self._train_measurements)} "
            f"mean_seconds_per_step={self.measured_train_seconds_per_step:.6f}"
        ]

    def train_epoch(self, *, seed: int, epoch: int, sequence_length: int, config: dict) -> str:
        if config["method"] != "response_only_sft_qlora":
            raise ValueError(f"unsupported training method: {config['method']!r}")
        started = self.clock()
        adapter_dir = Path(self.runtime.train_epoch(
            examples=self.training_examples,
            seed=seed,
            epoch=epoch,
            sequence_length=sequence_length,
            runtime_config=qlora_runtime_config(config),
            smoke_max_steps=self.smoke_max_steps,
        ))
        duration = self.clock() - started
        optimizer = config["optimizer"]
        batches = math.ceil(len(self.training_examples) / int(optimizer["micro_batch"]))
        steps = max(1, math.ceil(batches / int(optimizer["gradient_accumulation"])))
        if self.smoke_max_steps is not None:
            steps = min(steps, self.smoke_max_steps)
        self._train_measurements.append((duration, steps))
        fingerprint = _directory_fingerprint(adapter_dir)
        self._adapters[fingerprint] = adapter_dir
        return fingerprint

    def merge_checkpoint(self, fingerprint: str, output_dir: Path) -> None:
        try:
            adapter_dir = self._adapters[fingerprint]
        except KeyError as exc:
            raise KeyError(f"unknown adapter fingerprint: {fingerprint}") from exc
        self.runtime.merge(adapter_dir, Path(output_dir))
        output = Path(output_dir)
        has_weights = any(output.glob("*.safetensors")) or any(output.glob("pytorch_model*.bin"))
        if not (output / "config.json").is_file() or not has_weights:
            raise RuntimeError(f"merged checkpoint is not a standalone HF model directory: {output}")

    def release(self) -> None:
        release = getattr(self.runtime, "_release_training_state", None)
        if release is not None:
            release()


class _TransformersQLoRARuntime:
    def __init__(self, model_ref: str, revision: str, work_dir: Path):
        self.model_ref = model_ref
        self.revision = revision
        self.work_dir = work_dir
        self.snapshot: str | None = None
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.state_key = None

    def _resolve_snapshot(self, snapshot_download=None) -> str:
        local_path = Path(self.model_ref).expanduser()
        if local_path.is_dir():
            return str(local_path.resolve())
        if not self.revision:
            raise ValueError("immutable revision is required for a Hugging Face model ID")
        if snapshot_download is None:
            from huggingface_hub import snapshot_download
        return snapshot_download(
            repo_id=self.model_ref, revision=self.revision, local_files_only=True
        )

    def _release_training_state(self):
        self.scheduler = None
        self.optimizer = None
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _initialize(self, *, examples, seed, sequence_length, runtime_config, smoke_max_steps):
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            get_cosine_schedule_with_warmup,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("real QLoRA requires CUDA")
        self.snapshot = self._resolve_snapshot()
        self.tokenizer = AutoTokenizer.from_pretrained(self.snapshot)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quant = dict(runtime_config["quantization"])
        quant["bnb_4bit_compute_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            self.snapshot,
            quantization_config=BitsAndBytesConfig(**quant),
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = runtime_config["use_cache"]
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=runtime_config["gradient_checkpointing"]
        )
        if runtime_config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        model = get_peft_model(model, LoraConfig(**runtime_config["lora"]))
        model.train()
        self.model = model

        opt = runtime_config["optimizer"]
        if opt["name"].lower() != "adamw":
            raise ValueError(f"unsupported optimizer: {opt['name']!r}")
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=opt["learning_rate"], weight_decay=opt["weight_decay"],
        )
        accumulation = opt["gradient_accumulation_steps"]
        batches = math.ceil(len(examples) / opt["micro_batch_size"])
        steps_per_epoch = max(1, math.ceil(batches / accumulation))
        if smoke_max_steps is not None:
            steps_per_epoch = min(steps_per_epoch, smoke_max_steps)
        total_steps = max(1, steps_per_epoch * opt["epochs"])
        warmup_steps = int(total_steps * opt["warmup_ratio"])
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        self.state_key = (seed, sequence_length)

    def _collate(self, encoded: list[dict]):
        import torch
        pad = self.tokenizer.pad_token_id
        max_length = max(len(row["input_ids"]) for row in encoded)
        input_ids = torch.full((len(encoded), max_length), pad, dtype=torch.long)
        labels = torch.full((len(encoded), max_length), -100, dtype=torch.long)
        attention = torch.zeros((len(encoded), max_length), dtype=torch.long)
        for index, row in enumerate(encoded):
            length = len(row["input_ids"])
            input_ids[index, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
            labels[index, :length] = torch.tensor(row["labels"], dtype=torch.long)
            attention[index, :length] = 1
        device = next(self.model.parameters()).device
        return input_ids.to(device), labels.to(device), attention.to(device)

    def train_epoch(self, *, examples, seed, epoch, sequence_length, runtime_config, smoke_max_steps):
        import torch

        try:
            if self.state_key != (seed, sequence_length) or epoch == 1 and self.model is None:
                self._release_training_state()
                self._initialize(
                    examples=examples, seed=seed, sequence_length=sequence_length,
                    runtime_config=runtime_config, smoke_max_steps=smoke_max_steps,
                )
        except torch.OutOfMemoryError as exc:
            self._release_training_state()
            raise OutOfMemoryError(str(exc)) from exc
        torch.manual_seed(seed + epoch)
        randomizer = random.Random(seed + epoch)
        order = list(range(len(examples)))
        randomizer.shuffle(order)
        encoded = [encode_response_only(self.tokenizer, examples[index], sequence_length) for index in order]
        opt = runtime_config["optimizer"]
        micro_batch = opt["micro_batch_size"]
        accumulation = opt["gradient_accumulation_steps"]
        optimizer_steps = 0
        self.optimizer.zero_grad(set_to_none=True)
        try:
            for batch_index in range(0, len(encoded), micro_batch):
                batch = encoded[batch_index:batch_index + micro_batch]
                input_ids, labels, attention = self._collate(batch)
                loss = self.model(input_ids=input_ids, attention_mask=attention, labels=labels).loss
                (loss / accumulation).backward()
                final_batch = batch_index + micro_batch >= len(encoded)
                accumulation_boundary = ((batch_index // micro_batch) + 1) % accumulation == 0
                if accumulation_boundary or final_batch:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt["max_grad_norm"])
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if smoke_max_steps is not None and optimizer_steps >= smoke_max_steps:
                        break
        except torch.OutOfMemoryError as exc:
            self._release_training_state()
            raise OutOfMemoryError(str(exc)) from exc

        adapter_dir = self.work_dir / "adapters" / f"seed-{seed}" / f"epoch-{epoch}-seq-{sequence_length}"
        if adapter_dir.exists():
            raise FileExistsError(f"adapter checkpoint already exists: {adapter_dir}")
        adapter_dir.mkdir(parents=True)
        self.model.save_pretrained(adapter_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(adapter_dir)
        return adapter_dir

    def merge(self, adapter_dir: Path, output_dir: Path):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if output_dir.exists():
            raise FileExistsError(f"merged checkpoint already exists: {output_dir}")
        if self.snapshot is None:
            self.snapshot = self._resolve_snapshot()
        base = AutoModelForCausalLM.from_pretrained(
            self.snapshot, torch_dtype=torch.bfloat16, device_map={"": "cpu"}, low_cpu_mem_usage=True
        )
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        output_dir.mkdir(parents=True)
        merged.save_pretrained(output_dir, safe_serialization=True)
        AutoTokenizer.from_pretrained(self.snapshot).save_pretrained(output_dir)
        del merged, base
        gc.collect()
