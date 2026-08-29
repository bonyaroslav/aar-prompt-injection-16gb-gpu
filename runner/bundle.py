"""Immutable, checksummed run-bundle writer, per RESEARCH_PLAN.md's evidence contract."""
from __future__ import annotations
import hashlib, os
from pathlib import Path

BUNDLE_FILES = (
    "manifest.yaml", "command.sh", "config.yaml", "environment.txt",
    "metrics.json", "execution.log", "gpu.csv", "notes.md",
)
CHECKSUM_FILE = "checksums.sha256"

def write_bundle(bundle_dir: Path, contents: dict[str, str]) -> None:
    missing = [name for name in BUNDLE_FILES if name not in contents]
    if missing:
        raise ValueError(f"missing required bundle files: {missing}")
    bundle_dir.mkdir(parents=True, exist_ok=False)
    for name in BUNDLE_FILES:
        (bundle_dir / name).write_text(contents[name], encoding="utf-8")

def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def finalize_bundle(bundle_dir: Path) -> dict[str, str]:
    """Write checksums.sha256 for every bundle file and best-effort lock them read-only.

    Returns the file->digest mapping (including the checksum file itself).
    """
    checksums = {name: _digest(bundle_dir / name) for name in BUNDLE_FILES}
    lines = [f"{checksums[name]}  {name}" for name in BUNDLE_FILES]
    checksum_path = bundle_dir / CHECKSUM_FILE
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksums[CHECKSUM_FILE] = _digest(checksum_path)
    for name in (*BUNDLE_FILES, CHECKSUM_FILE):
        try: os.chmod(bundle_dir / name, 0o444)
        except OSError: pass
    try: os.chmod(bundle_dir, 0o555)
    except OSError: pass
    return checksums

def verify_bundle(bundle_dir: Path) -> None:
    """Recompute checksums and raise ValueError on any mismatch or missing/extra file."""
    checksum_path = bundle_dir / CHECKSUM_FILE
    if not checksum_path.exists():
        raise ValueError(f"checksum mismatch: {CHECKSUM_FILE} is missing")
    recorded = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    mismatches = []
    for name in BUNDLE_FILES:
        path = bundle_dir / name
        if not path.exists():
            mismatches.append(f"{name}: missing")
        elif name not in recorded:
            mismatches.append(f"{name}: not recorded in {CHECKSUM_FILE}")
        elif _digest(path) != recorded[name]:
            mismatches.append(f"{name}: content changed since finalization")
    if mismatches:
        raise ValueError("checksum mismatch: " + "; ".join(mismatches))
