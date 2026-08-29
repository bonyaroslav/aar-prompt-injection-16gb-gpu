"""Trained-checkpoint evaluation stage: the same manifest-frozen evaluation
protocol as the baseline stage (`runner.core.run_baseline`), with the model
backend swapped for a merged training checkpoint.

Held-out InjecAgent is out of scope here. Per `protocol/manifest.json`'s
`selection.held_out_unavailable_until: selection_record_finalized` and
`protocol/heldout_sealing.md`, checkpoint selection is decided on visible safety
and capability alone; the sealed held-out reveal for the selected checkpoint is
a separate, later stage (see the ready-for-agent issue queue).

Public seam: `run_trained_evaluation`. Its `config.yaml` is built from
`runner.core.effective_eval_config`, the same helper the baseline stage uses --
the only difference between the two stages' effective config is the `checkpoint`
field, which is exactly what a contract test in `tests/test_evaluation.py`
asserts.
"""
from __future__ import annotations
import dataclasses, json, platform, sys, time

from protocol.validate_manifest import load as load_manifest
from runner.bundle import write_bundle, finalize_bundle
from runner.core import VISIBLE_SAFETY_BENCHMARKS, CAPABILITY_BENCHMARKS, _run_benchmark, effective_eval_config


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    run_id: str
    stage: str
    bundle_dir: str
    checksums: dict
    metrics: dict


def run_trained_evaluation(manifest_path, *, model, dataset, scorer, telemetry, storage,
                            seed: int, epoch: int, checkpoint: dict,
                            run_id: str | None = None, clock=time.time) -> EvaluationResult:
    """Evaluate one merged training checkpoint on the three visible safety
    benchmarks and the three capability gates. `checkpoint` is the per-epoch
    entry produced by `runner.training.run_training`'s `metrics["checkpoints"]`
    (must contain `fingerprint` and `merged_dir`).
    """
    manifest = load_manifest(manifest_path)
    checkpoint_fingerprint = checkpoint["fingerprint"]
    run_id = run_id or f"eval-seed{seed}-epoch{epoch}-{int(clock())}"
    bundle_dir = storage.new_run_dir(run_id)
    telemetry.start()

    log_lines = [
        f"start trained-checkpoint evaluation run {run_id} protocol_version={manifest['protocol_version']} "
        f"seed={seed} epoch={epoch} checkpoint={checkpoint_fingerprint}"
    ]

    eval_cfg = manifest["evaluation"]
    metrics = {
        "stage": "trained_evaluation", "seed": seed, "epoch": epoch,
        "checkpoint": checkpoint_fingerprint, "benchmarks": {},
    }
    for name in VISIBLE_SAFETY_BENCHMARKS:
        result, n = _run_benchmark(name, eval_cfg["visible_safety"][name], model=model, dataset=dataset, scorer=scorer)
        metrics["benchmarks"][name] = result
        log_lines.append(f"scored {name}: n={n}")
    for name in CAPABILITY_BENCHMARKS:
        result, n = _run_benchmark(name, eval_cfg["capability"][name], model=model, dataset=dataset, scorer=scorer)
        metrics["benchmarks"][name] = result
        log_lines.append(f"scored {name}: n={n}")

    telemetry_rows = telemetry.stop()
    log_lines.append(f"finished trained-checkpoint evaluation run {run_id}")

    command = (
        f"{sys.executable} -m runner.evaluation --manifest {manifest_path} --run-id {run_id} "
        f"--seed {seed} --epoch {epoch} --checkpoint-dir {checkpoint['merged_dir']}"
    )
    contents = {
        "manifest.yaml": json.dumps({
            "run_id": run_id, "stage": "trained_evaluation",
            "protocol_version": manifest["protocol_version"],
            "upstream_commit": manifest["upstream"]["commit"],
            "model_revision": manifest["model"]["revision"],
            "seed": seed, "epoch": epoch, "checkpoint": checkpoint_fingerprint,
        }, indent=2, sort_keys=True),
        "command.sh": f"#!/usr/bin/env bash\nset -euo pipefail\n{command}\n",
        "config.yaml": json.dumps(effective_eval_config(manifest, checkpoint_fingerprint), indent=2, sort_keys=True),
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
        "notes.md": "# Trained-checkpoint evaluation run notes\n\nFake adapters only: no GPU, no model weights. Held-out InjecAgent is not evaluated in this stage.\n",
    }
    write_bundle(bundle_dir, contents)
    checksums = finalize_bundle(bundle_dir)
    return EvaluationResult(run_id=run_id, stage="trained_evaluation", bundle_dir=str(bundle_dir), checksums=checksums, metrics=metrics)
