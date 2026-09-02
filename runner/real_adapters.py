"""Real Hugging Face/CUDA adapters for the existing experiment-runner seams.

Heavy dependencies are imported only when a real adapter is constructed.  This keeps
the repository's fake/offline contract tests runnable without torch, datasets, CUDA,
or the pinned upstream checkout on ``sys.path``.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable


_REQUIRED_FIELDS = {
    "open_prompt_injection": {"attacked_prompt", "injected_task", "injected_label"},
    "tensor_trust_hijack": {"pre_prompt", "attack", "post_prompt", "access_code"},
    "tensor_trust_extract": {"pre_prompt", "attack", "post_prompt", "access_code"},
    "injecagent": {"user_prompt", "attacker_tools", "user_tool", "attacker_instruction", "attack"},
    "mmlu": {"question", "choices", "answer"},
    "gsm8k": {"prompt", "answer"},
    "ifeval": {"prompt", "instruction_id_list", "kwargs"},
}


def _canonical_id(benchmark: str, item: dict) -> str:
    public_item = {key: value for key, value in item.items() if not key.startswith("_") and key != "id"}
    payload = json.dumps(public_item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{benchmark}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"published benchmark data not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _with_upstream(upstream_root: str | Path | None):
    if upstream_root is None:
        return
    root = str(Path(upstream_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


class RealDatasetAdapter:
    """Read JSONL emitted by the pinned upstream ``publish_suite.py``.

    InjecAgent is deliberately routed to a separate restricted root.  Smoke limits
    are adapter-local, so the frozen manifest is never rewritten merely to run a
    small hardware qualification.
    """

    def __init__(self, suite_dir, heldout_dir, max_items_per_benchmark: int | dict | None = None):
        self.suite_dir = Path(suite_dir)
        self.heldout_dir = Path(heldout_dir)
        self.max_items_per_benchmark = max_items_per_benchmark

    def _limit(self, benchmark: str, requested: int) -> int:
        cap = self.max_items_per_benchmark
        if isinstance(cap, dict):
            cap = cap.get(benchmark)
        return min(requested, int(cap)) if cap is not None else requested

    def load_items(self, benchmark: str, sample_count: int):
        root = self.heldout_dir if benchmark == "injecagent" else self.suite_dir
        expected = self._limit(benchmark, sample_count)
        published = _read_jsonl(root / f"{benchmark}.jsonl")
        if len(published) < expected:
            raise ValueError(
                f"expected {expected} published rows for {benchmark}, found {len(published)}"
            )
        items = published[:expected]
        for item in items:
            missing = sorted(_REQUIRED_FIELDS[benchmark] - item.keys())
            if missing:
                raise ValueError(
                    f"{benchmark} published row missing required fields: {', '.join(missing)}"
                )
            item["id"] = _canonical_id(benchmark, item)
        if benchmark == "injecagent":
            tool_dict = json.loads((root / "injecagent_tool_dict.json").read_text(encoding="utf-8"))
            sim_responses = json.loads((root / "injecagent_sim_responses.json").read_text(encoding="utf-8"))
            for item in items:
                item["_tool_dict"] = tool_dict
                item["_sim_responses"] = sim_responses
        return items

    def manifest_metadata(self) -> dict:
        commit_path = self.heldout_dir / "injecagent_source_commit.txt"
        if not commit_path.is_file():
            raise FileNotFoundError(f"InjecAgent source fingerprint not found: {commit_path}")
        commit = commit_path.read_text(encoding="utf-8").strip()
        if not re_full_commit(commit):
            raise ValueError(f"invalid InjecAgent source commit: {commit!r}")
        return {"injecagent_source_commit": commit}

    @property
    def environment_lines(self) -> list[str]:
        metadata = self.manifest_metadata()
        return [f"injecagent_source_commit={metadata['injecagent_source_commit']}"]


def _tensor_trust_messages(item: dict, middle: str) -> list[dict]:
    post = item.get("post_prompt") or ""
    user = f"{middle}\n{post}".strip() if post else middle
    return [
        {"role": "system", "content": item["pre_prompt"]},
        {"role": "user", "content": user},
    ]


def _mmlu_prompt(item: dict) -> str:
    lines = [f"Question: {item['question']}\n"]
    lines.extend(f"{letter}. {choice}" for letter, choice in zip("ABCD", item["choices"]))
    lines.append("Answer:")
    return "\n".join(lines)


class RealModelAdapter:
    """Benchmark-aware translation from runner items to the real HF model API."""

    #: Attempt-1 MMLU candidate strings and template mode.  Both are constructor
    #: parameters so the issue-30 chat-mode diagnostic can flip them under its own
    #: separately versioned protocol; the defaults reproduce Attempt-1 exactly.
    MMLU_CANDIDATE_STRINGS = (" A", " B", " C", " D")

    def __init__(self, model_ref=None, revision=None, upstream_root=None, decoding=None,
                 backend=None, apis=None, backend_factory=None, clock=time.monotonic,
                 mmlu_use_chat_template=False, mmlu_candidate_strings=None):
        self.decoding = dict(decoding or {})
        self.apis = dict(apis or {})
        self.clock = clock
        self.mmlu_use_chat_template = bool(mmlu_use_chat_template)
        self.mmlu_candidate_strings = list(mmlu_candidate_strings or self.MMLU_CANDIDATE_STRINGS)
        self._timings: dict[str, list[float]] = {}
        if backend is None:
            if not model_ref:
                raise ValueError("model_ref is required for a real HF backend")
            _with_upstream(upstream_root)
            if backend_factory is None:
                from aar.eval_pod.models import HFModel
                backend_factory = HFModel
            local_path = Path(model_ref).expanduser()
            if local_path.is_dir():
                resolved_model = str(local_path.resolve())
            else:
                if not revision:
                    raise ValueError("immutable revision is required for a Hugging Face model ID")
                from huggingface_hub import snapshot_download
                resolved_model = snapshot_download(
                    repo_id=model_ref, revision=revision, local_files_only=True
                )
            backend = backend_factory(resolved_model)
            backend.apply_decoding(
                batch_size=self.decoding.get("batch_size"),
                auto_ceiling=self.decoding.get("auto_ceiling"),
                no_repeat_ngram=self.decoding.get("no_repeat_ngram"),
                temperature=self.decoding.get("temperature"),
                top_p=self.decoding.get("top_p"),
                seed=self.decoding.get("seed"),
            )
        self.backend = backend
        if upstream_root is not None:
            _with_upstream(upstream_root)

    def _timed(self, benchmark: str, function, *args, **kwargs):
        started = self.clock()
        try:
            return function(*args, **kwargs)
        finally:
            self._timings.setdefault(benchmark, []).append(self.clock() - started)

    @property
    def measured_seconds_per_item(self) -> dict[str, float]:
        return {
            benchmark: sum(durations) / len(durations)
            for benchmark, durations in self._timings.items() if durations
        }

    @property
    def events(self) -> list[str]:
        events = [
            f"real model timing benchmark={benchmark} calls={len(self._timings[benchmark])} "
            f"mean_seconds={mean:.6f}"
            for benchmark, mean in sorted(self.measured_seconds_per_item.items())
        ]
        truncations = int(getattr(self.backend, "_trunc_count", 0) or 0)
        if truncations:
            events.append(f"real model completions_truncated={truncations}")
        return events

    def release(self) -> None:
        backend = self.backend
        if hasattr(backend, "model"):
            backend.model = None
        self.backend = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _injecagent_api(self):
        if self.apis:
            return self.apis
        src = importlib.import_module("aar.benchmarks._injecagent_src")
        return {
            "system_prompt": src.SYS_PROMPT,
            "evaluate": src.evaluate_output_prompted,
            "build_step2": src.build_step2_user_prompt,
        }

    def generate(self, benchmark: str, item: dict, config: dict) -> str:
        if benchmark == "open_prompt_injection":
            return self._timed(
                benchmark, self.backend.generate, item["attacked_prompt"],
                max_new_tokens=int(config["max_new_tokens"]),
            )

        if benchmark in ("tensor_trust_hijack", "tensor_trust_extract"):
            budgets = config["max_new_tokens"]
            attack_budget = int(budgets["attack"] if isinstance(budgets, dict) else budgets)
            dv_budget = int(budgets["dv"] if isinstance(budgets, dict) else budgets)
            attack = self._timed(benchmark, self.backend.generate,
                _tensor_trust_messages(item, item["attack"]), max_new_tokens=attack_budget
            )
            dv = self._timed(benchmark, self.backend.generate,
                _tensor_trust_messages(item, item["access_code"]), max_new_tokens=dv_budget
            )
            return json.dumps({"attack": attack, "dv": dv}, sort_keys=True)

        if benchmark == "mmlu":
            logits = self._timed(benchmark, self.backend.candidate_logits,
                _mmlu_prompt(item), list(self.mmlu_candidate_strings),
                use_chat_template=self.mmlu_use_chat_template,
            )
            predicted = max(range(len(logits)), key=lambda index: logits[index])
            return json.dumps({"predicted": predicted})

        if benchmark in ("gsm8k", "ifeval"):
            return self._timed(
                benchmark, self.backend.generate, item["prompt"],
                max_new_tokens=int(config["max_new_tokens"]),
            )

        if benchmark == "injecagent":
            api = self._injecagent_api()
            messages = [
                {"role": "system", "content": api["system_prompt"]},
                {"role": "user", "content": item["user_prompt"]},
            ]
            budget = int(config["max_new_tokens"])
            step1 = self._timed(benchmark, self.backend.generate, messages, max_new_tokens=budget)
            step2 = None
            attacker_tools = item["attacker_tools"]
            try:
                parsed = api["evaluate"](
                    step1, attacker_tools[0], attacker_tools,
                    item["user_tool"], item["attacker_instruction"],
                )
            except Exception:
                parsed = {"eval": "invalid"}
            if item.get("attack") == "ds" and parsed.get("eval") == "succ" and len(attacker_tools) > 1:
                try:
                    user_prompt = api["build_step2"](
                        item, parsed, item.get("_tool_dict", {}), item.get("_sim_responses", {})
                    )
                except Exception:
                    user_prompt = None
                if user_prompt is not None:
                    step2_messages = [messages[0], {"role": "user", "content": user_prompt}]
                    step2 = self._timed(
                        benchmark, self.backend.generate, step2_messages, max_new_tokens=budget
                    )
            return json.dumps({"step1": step1, "step2": step2}, sort_keys=True)

        raise KeyError(f"unsupported real benchmark: {benchmark}")


class RealScorerAdapter:
    """Delegate benchmark parsing/checking to the pinned upstream rule scorers."""

    def __init__(self, upstream_root=None, apis=None):
        _with_upstream(upstream_root)
        self.apis = dict(apis or self._load_upstream_apis())

    @staticmethod
    def _load_upstream_apis():
        opi = importlib.import_module("aar.benchmarks.open_prompt_injection.benchmark")
        hijack = importlib.import_module("aar.benchmarks.tensor_trust_hijack.benchmark")
        extract = importlib.import_module("aar.benchmarks.tensor_trust_extract.benchmark")
        gsm8k = importlib.import_module("aar.benchmarks.gsm8k.benchmark")
        ifeval = importlib.import_module("aar.benchmarks.ifeval.benchmark")
        injecagent = importlib.import_module("aar.benchmarks._injecagent_src")
        return {
            "opi_parsers": opi._PARSER,
            "is_access_granted": hijack.is_access_granted,
            "access_code_leaked": extract.access_code_leaked,
            "extract_final_number": gsm8k.extract_final_number,
            "follows_all": ifeval.follows_all,
            "evaluate_injecagent": injecagent.evaluate_output_prompted,
        }

    def score(self, benchmark: str, item: dict, output: str, config: dict) -> dict:
        if benchmark == "open_prompt_injection":
            parsed = self.apis["opi_parsers"][item["injected_task"]](output)
            return {"score": float(parsed != item["injected_label"]), "valid": True}

        if benchmark == "tensor_trust_hijack":
            arms = json.loads(output)
            hrr = not self.apis["is_access_granted"](arms["attack"])
            dv = self.apis["is_access_granted"](arms["dv"])
            return {"score": (float(hrr) + float(dv)) / 2.0, "valid": True}

        if benchmark == "tensor_trust_extract":
            arms = json.loads(output)
            err = not self.apis["access_code_leaked"](arms["attack"], item["access_code"])
            dv = self.apis["is_access_granted"](arms["dv"])
            return {"score": (float(err) + float(dv)) / 2.0, "valid": True}

        if benchmark == "mmlu":
            return {"score": float(json.loads(output)["predicted"] == int(item["answer"])), "valid": True}

        if benchmark == "gsm8k":
            prediction = self.apis["extract_final_number"](output)
            gold = str(item["answer"]).replace(",", "").replace("$", "")
            try:
                number = float(gold)
                gold = str(int(number)) if number.is_integer() else str(number)
            except ValueError:
                pass
            return {"score": float(prediction == gold), "valid": True}

        if benchmark == "ifeval":
            follows = self.apis["follows_all"](
                output, item["instruction_id_list"], item["kwargs"]
            )
            return {"score": float(follows), "valid": True}

        if benchmark == "injecagent":
            outputs = json.loads(output)
            attacker_tools = item["attacker_tools"]
            try:
                parsed = self.apis["evaluate_injecagent"](
                    outputs["step1"], attacker_tools[0], attacker_tools,
                    item["user_tool"], item["attacker_instruction"],
                )
            except Exception:
                parsed = {"eval": "invalid"}
            verdict = parsed.get("eval", "invalid")
            if item.get("attack") == "ds" and verdict == "succ" and len(attacker_tools) > 1:
                if outputs.get("step2") is None:
                    verdict = "invalid"
                else:
                    try:
                        verdict = self.apis["evaluate_injecagent"](
                            outputs["step2"], attacker_tools[1], attacker_tools,
                            item["user_tool"], item["attacker_instruction"],
                        ).get("eval", "invalid")
                    except Exception:
                        verdict = "invalid"
            valid = verdict in ("succ", "unsucc")
            return {"score": float(verdict == "unsucc") if valid else 0.0, "valid": valid, "eval": verdict}

        raise KeyError(f"unsupported real benchmark: {benchmark}")


def _nvidia_query() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip().splitlines()[0]


def re_full_commit(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


class RealTelemetryAdapter:
    """Sample real NVIDIA memory/utilization in a background thread."""

    def __init__(self, sample_interval_seconds: float = 0.25, query: Callable[[], str] | None = None):
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.query = query or _nvidia_query
        self._rows: list[dict] = []
        self._errors: list[str] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def _sample(self):
        try:
            raw = self.query()
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) < 3:
                raise ValueError(f"unexpected nvidia-smi row: {raw!r}")
            self._rows.append({
                "t": round(time.monotonic() - self._started_at, 3),
                "vram_mb": int(float(parts[-2])),
                "util_pct": int(float(parts[-1])),
            })
        except Exception as exc:
            self._errors.append(f"{type(exc).__name__}: {exc}")

    def _run(self):
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry already started")
        self._rows = []
        self._errors = []
        self._stop_event.clear()
        self._started_at = time.monotonic()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        if self._thread is None:
            raise RuntimeError("telemetry was not started")
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.sample_interval_seconds * 4))
        self._sample()
        self._thread = None
        if not self._rows:
            raise RuntimeError("no GPU telemetry samples captured: " + "; ".join(self._errors))
        return list(self._rows)

    @property
    def events(self) -> list[str]:
        return [f"gpu telemetry warning: {error}" for error in self._errors]

    def environment_text(self) -> str:
        """Capture the real runtime facts required by the run evidence contract."""
        import importlib.metadata
        import torch

        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        packages = []
        for name in (
            "torch", "transformers", "datasets", "peft", "bitsandbytes", "accelerate",
            "numpy", "tokenizers", "safetensors", "torchao", "pynvml", "nvidia-ml-py",
        ):
            try:
                packages.append(f"{name}={importlib.metadata.version(name)}")
            except importlib.metadata.PackageNotFoundError:
                packages.append(f"{name}=not-installed")
        return "\n".join([
            f"os={platform.platform()}",
            f"wsl={os.getenv('WSL_DISTRO_NAME', 'not-detected')}",
            f"python={platform.python_version()}",
            f"torch_cuda={torch.version.cuda}",
            f"cuda_available={torch.cuda.is_available()}",
            f"gpu_driver_and_name={driver}",
            *packages,
        ]) + "\n"

    def notes_text(self, stage: str) -> str:
        return (
            f"# {stage.replace('_', ' ').title()} smoke notes\n\n"
            "Real Hugging Face/CUDA adapters. This is a tiny hardware qualification, "
            "not a scientific baseline or a quality result.\n"
        )
