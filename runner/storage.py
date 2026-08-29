"""Local filesystem storage adapter: allocates one run directory under `root`.

Real (not a stand-in): the runner always writes actual run bundles to disk.
Tests point `root` at a temp directory for isolation; production use defaults
to the repository's `runs/` directory per RESEARCH_PLAN.md's evidence contract.
"""
from __future__ import annotations
from pathlib import Path

class LocalStorageAdapter:
    def __init__(self, root: str | Path = "runs"):
        self.root = Path(root)

    def new_run_dir(self, run_id: str) -> Path:
        run_dir = self.root / run_id
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        return run_dir
