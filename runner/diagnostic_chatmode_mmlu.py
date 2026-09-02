"""Chat-mode MMLU confound test -- execution harness (issue #30).

Re-scores MMLU on the frozen Attempt-1 baseline model and every finalized merged
checkpoint with the chat template **enabled**, over the same fixed 300 example
identifiers, same prompt text, same candidate strings, same scorer. No training,
no new checkpoints -- this is re-scoring only.

Like ``runner.real_baseline``: the real-hardware entrypoint (``run_diagnostic``)
needs the pinned upstream checkout, real model weights and CUDA and is exercised
only by the actual GPU run; the pure pieces below (checkpoint integrity
verification, bundle-content assembly, the resource-row shaper) are unit-tested
offline.

Boundaries (from the diagnostic manifest, enforced here):
  * outputs are written under ``diagnostics/`` -- never ``runs/``, never ``analysis/``;
  * the held-out InjecAgent root is never referenced (no ``HeldOutSealer``);
  * the diagnostic result never feeds checkpoint selection or the frozen bootstrap.
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
import time
from pathlib import Path

from protocol.diagnostic.manifest import canonical_digest, load as load_diagnostic_manifest, raw_sha256
from runner.bundle import finalize_bundle, verify_bundle, write_bundle
from runner.training import _directory_digest

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_STAGE = "diagnostic_chatmode_mmlu"
MMLU_SAMPLE_COUNT = 300


class CheckpointIntegrityError(RuntimeError):
    """A merged checkpoint is missing or does not match its recorded digest.

    Raising this stops the diagnostic before any GPU time is spent; the harness
    never retrains in response.
    """


# --- checkpoint integrity (pure) --------------------------------------


def _recovery_integrity(repo_root: Path, reference: str) -> str | None:
    path = repo_root / reference
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    return json.loads(state["recovery_reference"])["integrity"]


def verify_model_states(diagnostic_manifest: dict, *, repo_root: Path = REPO_ROOT) -> list[dict]:
    """Verify every non-baseline model state before any GPU time is spent.

    For each of the nine merged checkpoint directories: recompute
    ``runner.training._directory_digest`` and require it to equal the manifest's
    ``expected_integrity_digest``. For seeds 42 and 2026 the expected value is
    additionally cross-checked against the run-time ``recovery/`` record. Any
    mismatch, or a missing / non-standalone directory, raises
    :class:`CheckpointIntegrityError` -- the diagnostic stops and reports, it
    never retrains.

    Returns the verified, digest-pinned model-state rows (the baseline row passes
    through unchanged; its integrity is the Hugging Face revision pin).
    """
    verified: list[dict] = []
    for state in diagnostic_manifest["model_states"]:
        if not state.get("merged_dir"):
            verified.append({**state, "verified": True})
            continue

        merged_dir = repo_root / state["merged_dir"]
        if not merged_dir.is_dir():
            raise CheckpointIntegrityError(
                f"{state['state']}: merged checkpoint directory is missing: {merged_dir}"
            )
        has_config = (merged_dir / "config.json").is_file()
        has_weights = any(merged_dir.glob("*.safetensors")) or any(merged_dir.glob("pytorch_model*.bin"))
        if not (has_config and has_weights):
            raise CheckpointIntegrityError(
                f"{state['state']}: not a standalone Hugging Face model directory: {merged_dir}"
            )

        actual = _directory_digest(merged_dir)
        expected = state["expected_integrity_digest"]
        if actual != expected:
            raise CheckpointIntegrityError(
                f"{state['state']}: merged checkpoint digest mismatch "
                f"(expected {expected}, recomputed {actual}); stop and report, do not retrain"
            )

        recovery_match = None
        if state.get("integrity_source") == "recovery_reference":
            recorded = _recovery_integrity(repo_root, state["recovery_reference"])
            if recorded is None:
                raise CheckpointIntegrityError(
                    f"{state['state']}: recovery reference not found: {state['recovery_reference']}"
                )
            if recorded != expected:
                raise CheckpointIntegrityError(
                    f"{state['state']}: recovery-recorded integrity {recorded} disagrees "
                    f"with the manifest's expected digest {expected}"
                )
            recovery_match = True

        verified.append({
            **state,
            "verified": True,
            "recomputed_integrity_digest": actual,
            "recovery_cross_check": recovery_match,
        })
    return verified


# --- bundle content (pure) -------------------------------------------


def diagnostic_notes_text() -> str:
    return (
        "# Chat-mode MMLU confound test (issue #30)\n\n"
        "DIAGNOSTIC OUTPUT -- NOT Attempt-1 evidence.\n\n"
        "This bundle re-scores MMLU with the chat template enabled on the frozen\n"
        "Attempt-1 baseline model and every finalized merged checkpoint, paired\n"
        "item by item against the Attempt-1 raw-completion-mode result. It is\n"
        "produced under the separately versioned diagnostic protocol\n"
        "`protocol/diagnostic/chatmode-mmlu-2026-09-02.json`.\n\n"
        "It must never enter an Attempt-1 evidence bundle, the frozen "
        "10,000-replicate bootstrap (analysis.bootstrap_seed = 271828), or "
        "checkpoint selection. The held-out InjecAgent benchmark is untouched.\n"
    )


def build_bundle_contents(*, diagnostic_manifest: dict, diagnostic_manifest_path: Path,
                           model_states: list[dict], per_state_scores: dict,
                           candidate_strings: list[str], telemetry_rows: list[dict],
                           environment_text: str, log_lines: list[str],
                           command_text: str) -> dict:
    """Assemble the eight checksummed bundle files for one diagnostic pass."""
    manifest_record = {
        "run_stage": DIAGNOSTIC_STAGE,
        "diagnostic_version": diagnostic_manifest["diagnostic_version"],
        "diagnostic_manifest_canonical_digest": canonical_digest(diagnostic_manifest),
        "diagnostic_manifest_sha256": raw_sha256(diagnostic_manifest_path),
        "downstream_of_protocol_version": diagnostic_manifest["downstream_of"]["protocol_version"],
        "downstream_of_canonical_manifest_digest":
            diagnostic_manifest["downstream_of"]["canonical_manifest_digest"],
        "model_revision": diagnostic_manifest["model"]["revision"],
        "mmlu_use_chat_template": True,
        "mmlu_candidate_strings": candidate_strings,
    }
    metrics = {
        "stage": DIAGNOSTIC_STAGE,
        "diagnostic_version": diagnostic_manifest["diagnostic_version"],
        "mmlu_use_chat_template": True,
        "mmlu_candidate_strings": candidate_strings,
        "scorer": diagnostic_manifest["change"]["scorer"],
        "sample_ids": diagnostic_manifest["change"]["sample_ids"],
        "model_states": {
            row["state"]: {
                "seed": row.get("seed"),
                "epoch": row.get("epoch"),
                "integrity_source": row.get("integrity_source"),
                "recomputed_integrity_digest": row.get("recomputed_integrity_digest"),
                "benchmarks": {"mmlu": per_state_scores[row["state"]]},
            }
            for row in model_states
        },
    }
    config = {
        "change": diagnostic_manifest["change"],
        "candidate_strings": candidate_strings,
        "analysis": diagnostic_manifest["analysis"],
        "boundaries": diagnostic_manifest["boundaries"],
    }
    return {
        "manifest.yaml": json.dumps(manifest_record, indent=2, sort_keys=True),
        "command.sh": "#!/usr/bin/env bash\nset -euo pipefail\n" + command_text + "\n",
        "config.yaml": json.dumps(config, indent=2, sort_keys=True),
        "environment.txt": environment_text if environment_text.endswith("\n") else environment_text + "\n",
        "metrics.json": json.dumps(metrics, indent=2, sort_keys=True),
        "execution.log": "\n".join(log_lines) + "\n",
        "gpu.csv": "t,vram_mb,util_pct\n" + "\n".join(
            f"{row['t']},{row['vram_mb']},{row['util_pct']}" for row in telemetry_rows
        ) + "\n",
        "notes.md": diagnostic_notes_text(),
    }


def resource_row(*, gpu_hours: float, wall_hours: float, source: str) -> dict:
    """Shape this diagnostic's measured compute as a
    ``runner.integrity_report`` ``non_scientific_runs`` row so it folds into the
    combined all-incurred-compute figure without touching the scientific totals.
    """
    return {
        "category": "diagnostic",
        "label": "chat-mode MMLU confound re-score (issue #30)",
        "gpu_hours": gpu_hours,
        "wall_hours": wall_hours,
        "source": source,
    }


# --- real-hardware entrypoint ---------------------------------------


def _score_mmlu(model, scorer, items, config) -> dict:
    out = {}
    for item in items:
        output = model.generate("mmlu", item, config)
        outcome = scorer.score("mmlu", item, output, config)
        out[item["id"]] = {"score": outcome["score"], "valid": outcome["valid"]}
    return {
        "items": out,
        "aggregate": {
            "metric": config["metric"],
            "value": (sum(e["score"] for e in out.values()) / len(out)) if out else None,
        },
    }


def run_diagnostic(*, diagnostic_manifest_path: Path, upstream_root: Path, suite_dir: Path,
                    output_root: Path, model_cache: Path, candidate_strings: list[str] | None = None,
                    reproduction_command: str | None = None, max_items: int | None = None) -> dict:
    """Execute one chat-mode MMLU pass over the baseline model and nine merged
    checkpoints. ``max_items`` reduces the per-state sample count for an
    end-to-end code-path smoke check -- a run with ``max_items`` set is not
    scientific evidence.
    """
    from huggingface_hub import snapshot_download

    from protocol.validate_manifest import load as load_frozen_manifest
    from runner.gpu_smoke import _verify_upstream
    from runner.real_adapters import (
        RealDatasetAdapter, RealModelAdapter, RealScorerAdapter, RealTelemetryAdapter,
    )

    diagnostic = load_diagnostic_manifest(diagnostic_manifest_path)
    frozen = load_frozen_manifest(REPO_ROOT / "protocol" / "manifest.json")
    _verify_upstream(
        upstream_root,
        expected_commit=frozen["upstream"]["commit"],
        expected_tree=frozen["upstream"]["tree"],
    )
    verified_states = verify_model_states(diagnostic)
    # Decoding block unchanged from Attempt-1 (only the template flag differs);
    # likelihood-ranked MMLU consults only batch_size from it.
    decoding = dict(frozen["evaluation"]["decoding"])

    candidate_strings = list(candidate_strings or diagnostic["change"]["candidate_strings"])
    smoke = max_items is not None
    count = min(MMLU_SAMPLE_COUNT, max_items) if smoke else MMLU_SAMPLE_COUNT

    dataset = RealDatasetAdapter(suite_dir, suite_dir, max_items_per_benchmark={"mmlu": count})
    items = list(dataset.load_items("mmlu", MMLU_SAMPLE_COUNT))
    scorer = RealScorerAdapter(upstream_root)
    mmlu_config = {
        "metric": diagnostic["change"]["metric"],
        "max_new_tokens": diagnostic["change"]["max_new_tokens"],
    }

    snapshot = snapshot_download(
        repo_id=diagnostic["model"]["id"], revision=diagnostic["model"]["revision"],
        cache_dir=model_cache, local_files_only=True,
    )

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    variant_tag = "" if candidate_strings == diagnostic["change"]["candidate_strings"] else "-nolead"
    run_id = f"chatmode-mmlu-{stamp}{variant_tag}"
    bundle_dir = Path(output_root) / run_id
    if bundle_dir.exists():
        raise FileExistsError(f"diagnostic bundle already exists: {bundle_dir}")

    telemetry = RealTelemetryAdapter()
    telemetry.start()
    started = time.monotonic()
    log_lines = [
        f"start {DIAGNOSTIC_STAGE} run {run_id} diagnostic_version={diagnostic['diagnostic_version']} "
        f"chat_template=on candidates={candidate_strings} smoke={smoke} n_per_state={count}"
    ]

    per_state_scores: dict[str, dict] = {}
    for row in verified_states:
        state = row["state"]
        if state == "baseline":
            model_ref = snapshot
        else:
            model_ref = str(REPO_ROOT / row["merged_dir"])
        model = RealModelAdapter(
            model_ref, None, upstream_root,
            decoding=decoding, mmlu_use_chat_template=True,
            mmlu_candidate_strings=candidate_strings,
        )
        try:
            per_state_scores[state] = _score_mmlu(model, scorer, items, mmlu_config)
        finally:
            model.release()
        log_lines.append(
            f"scored mmlu chat-mode state={state} n={len(items)} "
            f"value={per_state_scores[state]['aggregate']['value']}"
        )

    telemetry_rows = telemetry.stop()
    wall_seconds = time.monotonic() - started
    peak_vram_mb = max((r["vram_mb"] for r in telemetry_rows), default=0)
    log_lines.extend(telemetry.events)
    log_lines.append(f"finished {DIAGNOSTIC_STAGE} run {run_id} wall_seconds={wall_seconds:.1f}")

    command_text = reproduction_command or shlex.join([
        sys.executable, "-m", "runner.diagnostic_chatmode_mmlu",
        "--diagnostic-manifest", str(diagnostic_manifest_path),
    ])
    contents = build_bundle_contents(
        diagnostic_manifest=diagnostic, diagnostic_manifest_path=Path(diagnostic_manifest_path),
        model_states=verified_states, per_state_scores=per_state_scores,
        candidate_strings=candidate_strings, telemetry_rows=telemetry_rows,
        environment_text=telemetry.environment_text(), log_lines=log_lines,
        command_text=command_text,
    )
    write_bundle(bundle_dir, contents)
    finalize_bundle(bundle_dir)
    verify_bundle(bundle_dir)

    wall_hours = wall_seconds / 3600.0
    resource = resource_row(gpu_hours=wall_hours, wall_hours=wall_hours,
                            source=f"{run_id}/execution.log")
    resource_path = Path(output_root) / f"{run_id}-resource.json"
    resource_path.write_text(json.dumps({
        "run_id": run_id,
        "smoke": smoke,
        "peak_vram_gb": peak_vram_mb / 1024.0,
        "non_scientific_run_row": resource,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id, "bundle_dir": str(bundle_dir), "smoke": smoke,
        "candidate_strings": candidate_strings, "wall_seconds": wall_seconds,
        "peak_vram_gb": peak_vram_mb / 1024.0, "resource_artifact": str(resource_path),
        "per_state_values": {
            state: scores["aggregate"]["value"] for state, scores in per_state_scores.items()
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--diagnostic-manifest", type=Path,
                        default=REPO_ROOT / "protocol" / "diagnostic" / "chatmode-mmlu-2026-09-02.json")
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "diagnostics")
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--no-leading-space", action="store_true",
                        help="robustness re-run: MMLU candidate strings ['A','B','C','D']")
    parser.add_argument("--max-items", type=int, default=None,
                        help="smoke only: reduce per-state MMLU sample count")
    args = parser.parse_args(argv)

    command_text = shlex.join([sys.executable, "-m", "runner.diagnostic_chatmode_mmlu", *(argv or sys.argv[1:])])
    result = run_diagnostic(
        diagnostic_manifest_path=args.diagnostic_manifest, upstream_root=args.upstream_root,
        suite_dir=args.suite_dir, output_root=args.output_root, model_cache=args.model_cache,
        candidate_strings=["A", "B", "C", "D"] if args.no_leading_space else None,
        reproduction_command=command_text, max_items=args.max_items,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
