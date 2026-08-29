"""Deterministic fake adapters for the runner's no-GPU, no-download contract tests.

These never call a real model, dataset host, or scorer service. FakeDatasetAdapter
deliberately mirrors the upstream publisher's own default-argument bug (n=210 for
open_prompt_injection) so a contract test can prove the runner always resolves and
passes the manifest-declared sample count (300) instead of silently inheriting it.
"""
from __future__ import annotations
import hashlib

PUBLISHER_OPI_DEFAULT = 210

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
