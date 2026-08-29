"""Deterministic fake adapters for the runner's no-GPU, no-download contract tests.

These never call a real model, dataset host, or scorer service. FakeDatasetAdapter
deliberately mirrors the upstream publisher's own default-argument bug (n=210 for
open_prompt_injection) so a contract test can prove the runner always resolves and
passes the manifest-declared sample count (300) instead of silently inheriting it.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

PUBLISHER_OPI_DEFAULT = 210

class OutOfMemoryError(RuntimeError):
    """Simulated CUDA OOM raised by FakeTrainerAdapter."""

class FakeTrainerAdapter:
    """Deterministic fake QLoRA trainer for the training stage's contract tests.

    `train_epoch` never touches a real model or GPU: the "checkpoint" is a sha256
    fingerprint over (seed, epoch, sequence_length, training config), so the same
    seed reproducibly yields the same fingerprint (needed for seed-replication
    testing) and different seeds diverge.

    OOM is simulated, not physical: it fires when `epoch == oom_at_epoch` and the
    caller is currently training at one of `oom_sequence_lengths`. Passing just
    `{2048}` models a *recoverable* OOM (the runner's 2048->1536 fallback then
    succeeds); passing `{2048, 1536}` models an OOM that recurs even after the
    single approved fallback, exercising the unrecoverable-failure path.
    """
    def __init__(self, oom_at_epoch: int | None = None, oom_sequence_lengths: frozenset[int] = frozenset()):
        self.oom_at_epoch = oom_at_epoch
        self.oom_sequence_lengths = oom_sequence_lengths

    def train_epoch(self, *, seed: int, epoch: int, sequence_length: int, config: dict) -> str:
        if epoch == self.oom_at_epoch and sequence_length in self.oom_sequence_lengths:
            raise OutOfMemoryError(f"simulated OOM at epoch={epoch} sequence_length={sequence_length}")
        fingerprint_src = (
            f"seed={seed}:epoch={epoch}:sequence_length={sequence_length}:"
            f"method={config['method']}:rank={config['adapter']['rank']}:alpha={config['adapter']['alpha']}"
        )
        return hashlib.sha256(fingerprint_src.encode()).hexdigest()

    def merge_checkpoint(self, fingerprint: str, output_dir: Path) -> None:
        """Simulate merging a PEFT adapter into a standalone, loadable model dir.

        `aar/eval_pod/models.py::load_model()` (upstream) only accepts a HF id or a
        local model directory, never a bare adapter, so every epoch's checkpoint
        must already be merged by the time the evaluation stage sees it.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "model.json").write_text(
            json.dumps({"fingerprint": fingerprint, "format": "fake-merged-standalone"}, sort_keys=True),
            encoding="utf-8",
        )

class FakeDatasetAdapter:
    def load_open_prompt_injection(self, sample_count: int = PUBLISHER_OPI_DEFAULT):
        return [{"id": f"open_prompt_injection-{i:04d}"} for i in range(sample_count)]

    def load_items(self, benchmark: str, sample_count: int):
        if benchmark == "open_prompt_injection":
            return self.load_open_prompt_injection(sample_count=sample_count)
        return [{"id": f"{benchmark}-{i:04d}"} for i in range(sample_count)]

class FakeModelAdapter:
    def generate(self, benchmark: str, item: dict, config: dict) -> str:
        return f"fake-output:{benchmark}:{item['id']}"

class FakeScorerAdapter:
    """Deterministic pass/fail derived only from benchmark+item id, never randomness."""
    def score(self, benchmark: str, item: dict, output: str, config: dict) -> dict:
        digest = hashlib.sha256(f"{benchmark}:{item['id']}:{output}".encode()).hexdigest()
        return {"score": int(digest[0], 16) % 2, "valid": int(digest[1], 16) % 4 != 0}

class FakeTelemetryAdapter:
    def __init__(self):
        self._rows: list[dict] = []

    def start(self) -> None:
        self._rows = [{"t": 0, "vram_mb": 1024, "util_pct": 5}]

    def stop(self) -> list[dict]:
        self._rows.append({"t": 1, "vram_mb": 1024, "util_pct": 0})
        return list(self._rows)
