"""Held-out reveal + authorization gate: the runner's only path from a finalized
checkpoint selection to the one-time combined baseline+trained InjecAgent reveal.

Per `protocol/heldout_sealing.md`, checkpoint selection (`runner.selection`) never
touches held-out InjecAgent. Once a selection record is finalized, three things
must happen, in order, before held-out numbers can ever be read back out:

1. The selected checkpoint is evaluated on InjecAgent, reusing the candidate
   commitment the baseline already froze -- never re-freezing it
   (`run_trained_held_out_evaluation`).
2. Finalizing the selection record immediately authorizes the sealer with that
   same record (`finalize_and_authorize_selection`) -- there is no separate,
   forgettable "now call authorize()" step.
3. `build_reveal_package`/`run_reveal` produce baseline and selected-trained
   results together, never separately, as aggregates plus the invalid-
   classification table -- no candidate list, prompt, tool response, secret, or
   raw held-out text.
"""
from __future__ import annotations
import dataclasses, json, platform, sys, time
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner.bundle import write_bundle, finalize_bundle
from runner.core import _run_held_out_injecagent
from runner.selection import finalize_selection_record, _digest


def run_trained_held_out_evaluation(manifest_path, *, model, dataset, scorer, sealer, label: str = "trained") -> dict:
    """Evaluate the selected checkpoint on the frozen InjecAgent candidates and
    store its receipt under `label` (default `"trained"`). Reuses the baseline's
    candidate/validity commitment rather than re-freezing it; raises
    `PermissionError` if the dataset adapter would now produce a candidate list
    or validity-rule text that no longer matches what was frozen.
    """
    manifest = load_manifest(manifest_path)
    injecagent_cfg = manifest["evaluation"]["held_out"]["injecagent"]
    validity_rules = json.dumps(manifest["held_out_policy"], sort_keys=True, separators=(",", ":"))
    return _run_held_out_injecagent(
        injecagent_cfg, model=model, dataset=dataset, scorer=scorer,
        sealer=sealer, validity_rules=validity_rules, label=label, freeze=False,
    )


def finalize_and_authorize_selection(record: dict, path: str | Path, sealer) -> dict:
    """Finalize the selection record on disk, then immediately authorize the
    sealer with that same record. A finalized-but-not-authorized record would
    leave the reveal gate permanently shut, so the two happen together from the
    caller's point of view -- never as two steps a caller could do out of order.
    """
    result = finalize_selection_record(record, path)
    sealer.authorize(record)
    return result


def _aggregate_injecagent(blob: bytes) -> dict:
    """Turn one sealed InjecAgent blob (per-candidate outcomes) into the two
    denominators `protocol/heldout_sealing.md` requires, plus valid/invalid
    counts -- never the per-candidate outcomes or candidate IDs themselves.
    """
    per_candidate = json.loads(blob)["items"]
    total = len(per_candidate)
    valid_scores = [outcome["score"] for outcome in per_candidate.values() if outcome.get("valid", True)]
    intent_scores = [outcome["score"] if outcome.get("valid", True) else 0 for outcome in per_candidate.values()]
    return {
        "valid_only": {"value": (sum(valid_scores) / len(valid_scores)) if valid_scores else None, "n": len(valid_scores)},
        "intent_to_evaluate": {"value": (sum(intent_scores) / total) if total else None, "n": total},
        "valid_count": len(valid_scores),
        "invalid_count": total - len(valid_scores),
    }


def build_reveal_package(sealer, selection_record: dict) -> dict:
    """The runner's public seam for producing the combined baseline+trained
    InjecAgent reveal package. Delegates entirely to `HeldOutSealer.reveal`, so
    a missing/mismatched selection record or a state other than `AUTHORIZED`
    surfaces here as the same `PermissionError` `reveal()` raises -- there is no
    runner-side code path that returns baseline or trained alone.
    """
    blobs = sealer.reveal(selection_record)
    baseline = _aggregate_injecagent(blobs["baseline"])
    trained = _aggregate_injecagent(blobs["trained"])
    return {
        "baseline": {"valid_only": baseline["valid_only"], "intent_to_evaluate": baseline["intent_to_evaluate"]},
        "trained": {"valid_only": trained["valid_only"], "intent_to_evaluate": trained["intent_to_evaluate"]},
        "invalid_classification": {
            "baseline": {"valid": baseline["valid_count"], "invalid": baseline["invalid_count"]},
            "trained": {"valid": trained["valid_count"], "invalid": trained["invalid_count"]},
        },
    }


@dataclasses.dataclass(frozen=True)
class RevealResult:
    run_id: str
    stage: str
    bundle_dir: str
    checksums: dict
    metrics: dict


def run_reveal(manifest_path, *, sealer, selection_record: dict, storage, telemetry,
                run_id: str | None = None, clock=time.time) -> RevealResult:
    """The reveal stage's runner-interface seam: build the reveal package (which
    raises on every rejected path before any bundle is allocated), then archive
    it as the one repository-bound reveal artifact.
    """
    manifest = load_manifest(manifest_path)
    package = build_reveal_package(sealer, selection_record)

    run_id = run_id or f"reveal-{int(clock())}"
    bundle_dir = storage.new_run_dir(run_id)
    telemetry.start()

    selection_digest = _digest(selection_record)
    log_lines = [
        f"start reveal run {run_id} protocol_version={manifest['protocol_version']} selection_digest={selection_digest}",
        "held-out reveal authorized; baseline and selected-trained InjecAgent produced together",
    ]
    telemetry_rows = telemetry.stop()
    log_lines.append(f"finished reveal run {run_id}")

    command = f"{sys.executable} -m runner.reveal --manifest {manifest_path} --run-id {run_id}"
    metrics = {"stage": "reveal", "selection_digest": selection_digest, "held_out": {"injecagent": package}}
    contents = {
        "manifest.yaml": json.dumps({
            "run_id": run_id, "stage": "reveal",
            "protocol_version": manifest["protocol_version"],
            "upstream_commit": manifest["upstream"]["commit"],
            "model_revision": manifest["model"]["revision"],
            "selection_digest": selection_digest,
        }, indent=2, sort_keys=True),
        "command.sh": f"#!/usr/bin/env bash\nset -euo pipefail\n{command}\n",
        "config.yaml": json.dumps({"selection_digest": selection_digest}, indent=2, sort_keys=True),
        "environment.txt": "\n".join([
            f"python={platform.python_version()}",
            f"platform={platform.platform()}",
            "gpu=none (fake adapters; no real GPU or model weights used)",
        ]) + "\n",
        "metrics.json": json.dumps(metrics, indent=2, sort_keys=True),
        "execution.log": "\n".join(log_lines) + "\n",
        "gpu.csv": "t,vram_mb,util_pct\n" + "\n".join(
            f"{row['t']},{row['vram_mb']},{row['util_pct']}" for row in telemetry_rows
        ) + "\n",
        "notes.md": (
            "# Reveal run notes\n\n"
            "Baseline and selected-trained InjecAgent revealed together after authorization. "
            "Only `valid_only`/`intent_to_evaluate` aggregates and the invalid-classification "
            "table are recorded here -- no candidate list, prompt, tool response, secret, or "
            "raw held-out text.\n"
        ),
    }
    write_bundle(bundle_dir, contents)
    checksums = finalize_bundle(bundle_dir)
    return RevealResult(run_id=run_id, stage="reveal", bundle_dir=str(bundle_dir), checksums=checksums, metrics=metrics)
