"""Checkpoint selection: visible composite + capability gates -> one canonical,
checksummed selection record.

Selection never touches held-out InjecAgent -- per `protocol/manifest.json`'s
`selection.held_out_unavailable_until: selection_record_finalized` and
`protocol/heldout_sealing.md`, the sealed held-out reveal only happens after
this record exists, is finalized, and is separately authorized (a later stage).
`select_checkpoint` is a pure function over already-computed visible-safety and
capability aggregates -- no model, dataset, scorer, or sealer dependency -- so
it can be exercised with plain fixtures.
"""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner.core import VISIBLE_SAFETY_BENCHMARKS, CAPABILITY_BENCHMARKS

MEANINGFUL_VISIBLE_MITIGATION_LABEL = "meaningful visible mitigation"


def _digest(value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def visible_composite(baseline_benchmarks: dict, candidate_benchmarks: dict) -> dict:
    """Unweighted mean of the three per-benchmark absolute improvements over baseline."""
    per_benchmark = {
        name: candidate_benchmarks[name]["aggregate"]["value"] - baseline_benchmarks[name]["aggregate"]["value"]
        for name in VISIBLE_SAFETY_BENCHMARKS
    }
    composite = sum(per_benchmark.values()) / len(per_benchmark)
    return {"per_benchmark_improvement": per_benchmark, "composite": composite}


def capability_gate(manifest: dict, baseline_benchmarks: dict, candidate_benchmarks: dict) -> dict:
    """Each capability benchmark's gate is evaluated independently; failing any
    one -- or falling below the mean-normalized-retention floor -- disqualifies
    the candidate regardless of its visible composite.
    """
    gates = manifest["selection"]["capability_gates"]
    max_decline = {
        "mmlu": gates["mmlu_max_decline"],
        "gsm8k": gates["gsm8k_max_decline"],
        "ifeval": gates["ifeval_max_decline"],
    }
    declines, retentions, reasons = {}, [], []
    for name in CAPABILITY_BENCHMARKS:
        base = baseline_benchmarks[name]["aggregate"]["value"]
        cand = candidate_benchmarks[name]["aggregate"]["value"]
        decline = base - cand
        declines[name] = decline
        retentions.append((cand / base) if base else 1.0)
        if decline > max_decline[name]:
            reasons.append(f"{name} decline {decline:.4f} exceeds max {max_decline[name]}")
    mean_retention = sum(retentions) / len(retentions)
    if mean_retention < gates["mean_normalized_retention_min"]:
        reasons.append(
            f"mean normalized retention {mean_retention:.4f} below min {gates['mean_normalized_retention_min']}"
        )
    return {
        "declines": declines,
        "mean_capability_loss": sum(declines.values()) / len(declines),
        "mean_normalized_retention": mean_retention,
        "passed": not reasons,
        "reasons": reasons,
    }


def evaluate_candidate(manifest: dict, baseline_benchmarks: dict, candidate: dict) -> dict:
    """`candidate` = {"epoch": int, "checkpoint_digest": str, "benchmarks": {...}}."""
    visible = visible_composite(baseline_benchmarks, candidate["benchmarks"])
    capability = capability_gate(manifest, baseline_benchmarks, candidate["benchmarks"])
    meaningful_threshold = manifest["selection"]["meaningful_improvement_absolute"]
    return {
        "epoch": candidate["epoch"],
        "checkpoint_digest": candidate["checkpoint_digest"],
        "visible": visible,
        "capability": capability,
        "eligible": capability["passed"],
        "meaningful_visible_mitigation": capability["passed"] and visible["composite"] >= meaningful_threshold,
    }


def select_checkpoint(manifest_path, *, baseline_benchmarks: dict, candidates: list[dict]) -> dict:
    """Select the checkpoint with the highest visible composite among candidates
    that pass every capability gate. Ties are broken deterministically: lower
    mean capability loss first, then earlier epoch. Returns the canonical
    (unwritten, unchecksummed) selection record; pass it to
    `finalize_selection_record` to freeze it.
    """
    manifest = load_manifest(manifest_path)
    evaluated = [evaluate_candidate(manifest, baseline_benchmarks, c) for c in candidates]
    eligible = [e for e in evaluated if e["eligible"]]
    selected = None
    if eligible:
        selected = sorted(
            eligible,
            key=lambda e: (-e["visible"]["composite"], e["capability"]["mean_capability_loss"], e["epoch"]),
        )[0]
    return {
        "protocol_version": manifest["protocol_version"],
        "manifest_digest": _digest(manifest),
        "candidates": evaluated,
        "selected_checkpoint_digest": selected["checkpoint_digest"] if selected else None,
        "selected_epoch": selected["epoch"] if selected else None,
        "finalized": True,
    }


def finalize_selection_record(record: dict, path: str | Path) -> dict:
    """Write the canonical selection record exactly once. Finalizing again with
    byte-identical content is a no-op (idempotent); finalizing again with
    different content at the same path is rejected -- the record is immutable
    once written, mirroring `runner.bundle.finalize_bundle`.
    """
    path = Path(path)
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if path.exists():
        existing_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if existing_digest != digest:
            raise RuntimeError(f"selection record already finalized with different content: {path}")
        return {"digest": existing_digest, "path": str(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical, encoding="utf-8")
    try: os.chmod(path, 0o444)
    except OSError: pass
    return {"digest": digest, "path": str(path)}


def verify_selection_record(path: str | Path, expected_digest: str) -> None:
    """Recompute the on-disk digest and raise ValueError on any mismatch or
    missing file -- the read-side counterpart of `finalize_selection_record`."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"checksum mismatch: selection record missing at {path}")
    actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError(f"checksum mismatch: selection record changed since finalization at {path}")
