"""Experiment-runner core: frozen manifest -> baseline stage -> checksummed run bundle.

Public seam: `run_baseline`. It accepts a manifest path and one adapter per
external dependency (model, dataset, scorer, telemetry, storage) and returns a
structured `RunResult`, never parsed console text. Phase 1 wires only fake,
deterministic adapters here; real GPU/HF adapters are a later, separately
qualified integration (see docs/adr and the ready-for-agent issue queue).
"""
from __future__ import annotations
import dataclasses, json, platform, re, sys, time
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner.bundle import write_bundle, finalize_bundle

VISIBLE_SAFETY_BENCHMARKS = ("open_prompt_injection", "tensor_trust_hijack", "tensor_trust_extract")
CAPABILITY_BENCHMARKS = ("mmlu", "gsm8k", "ifeval")

_SAMPLE_COUNT_RE = re.compile(r"first_(\d+)")

def resolve_sample_count(sample_ids: str) -> int:
    """Extract the frozen sample count from a manifest `sample_ids` string.

    e.g. "publisher_seed_42_first_300;100_each_injected_task" -> 300. This is the
    single choke point the runner uses instead of trusting any adapter's own
    default (the OPI publisher's own default is a smaller, wrong 210).
    """
    match = _SAMPLE_COUNT_RE.search(sample_ids)
    if not match:
        raise ValueError(f"cannot resolve sample count from sample_ids: {sample_ids!r}")
    return int(match.group(1))

@dataclasses.dataclass(frozen=True)
class RunResult:
    run_id: str
    stage: str
    bundle_dir: str
    checksums: dict
    metrics: dict

def _run_benchmark(name: str, cfg: dict, *, model, dataset, scorer) -> tuple[dict, int]:
    sample_count = resolve_sample_count(cfg["sample_ids"])
    items = dataset.load_items(name, sample_count)
    item_scores = {}
    for item in items:
        output = model.generate(name, item, cfg)
        item_scores[item["id"]] = scorer.score(name, item, output, cfg)
    values = [v["score"] for v in item_scores.values()]
    aggregate = {"metric": cfg["metric"], "value": (sum(values) / len(values)) if values else None}
    return {"items": item_scores, "aggregate": aggregate}, sample_count

def run_baseline(manifest_path, *, model, dataset, scorer, telemetry, storage,
                  run_id: str | None = None, clock=time.time) -> RunResult:
    manifest = load_manifest(manifest_path)
    run_id = run_id or f"baseline-{int(clock())}"
    bundle_dir = storage.new_run_dir(run_id)

    log_lines = [f"start baseline run {run_id} protocol_version={manifest['protocol_version']}"]
    telemetry.start()

    metrics = {"stage": "baseline", "benchmarks": {}}
    eval_cfg = manifest["evaluation"]
    for name in VISIBLE_SAFETY_BENCHMARKS:
        result, n = _run_benchmark(name, eval_cfg["visible_safety"][name], model=model, dataset=dataset, scorer=scorer)
        metrics["benchmarks"][name] = result
        log_lines.append(f"scored {name}: n={n}")
    for name in CAPABILITY_BENCHMARKS:
        result, n = _run_benchmark(name, eval_cfg["capability"][name], model=model, dataset=dataset, scorer=scorer)
        metrics["benchmarks"][name] = result
        log_lines.append(f"scored {name}: n={n}")

    telemetry_rows = telemetry.stop()
    log_lines.append(f"finished baseline run {run_id}")

    command = f"{sys.executable} -m runner.core --manifest {manifest_path} --stage baseline --run-id {run_id}"
    contents = {
        "manifest.yaml": json.dumps({
            "run_id": run_id, "stage": "baseline",
            "protocol_version": manifest["protocol_version"],
            "upstream_commit": manifest["upstream"]["commit"],
            "model_revision": manifest["model"]["revision"],
        }, indent=2, sort_keys=True),
        "command.sh": f"#!/usr/bin/env bash\nset -euo pipefail\n{command}\n",
        "config.yaml": json.dumps(eval_cfg, indent=2, sort_keys=True),
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
        "notes.md": "# Baseline run notes\n\nFake adapters only: no GPU, no model weights, no dataset downloads.\n",
    }
    write_bundle(bundle_dir, contents)
    checksums = finalize_bundle(bundle_dir)
    return RunResult(run_id=run_id, stage="baseline", bundle_dir=str(bundle_dir), checksums=checksums, metrics=metrics)
