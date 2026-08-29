"""Experiment-runner core: frozen manifest -> baseline stage -> checksummed run bundle.

Public seam: `run_baseline`. It accepts a manifest path and one adapter per
external dependency (model, dataset, scorer, telemetry, storage, held_out_sealer)
and returns a structured `RunResult`, never parsed console text. Phase 1 wires
only fake, deterministic adapters here; real GPU/HF adapters are a later,
separately qualified integration (see docs/adr and the ready-for-agent issue
queue).

Held-out InjecAgent never travels the plain visible-benchmark path: it is
frozen and stored through `held_out_sealer` (a `protocol.heldout.HeldOutSealer`),
so the public run bundle only ever sees digests, an opaque receipt, and
valid/invalid counts. `read_held_out_result` is the runner's only seam for
reading held-out results back out, and it delegates entirely to the sealer's
own SEALED/AUTHORIZED state machine.
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

def _run_held_out_injecagent(cfg: dict, *, model, dataset, scorer, sealer, validity_rules: str, label: str) -> dict:
    """Evaluate InjecAgent entirely behind the sealer: freeze commitments, then
    store an append-only receipt. Returns only the opaque receipt (label,
    digest, valid/invalid counts) -- per-candidate outcomes and plaintext
    outputs never leave this function.
    """
    sample_count = cfg["candidate_count"]
    items = dataset.load_items("injecagent", sample_count)
    candidate_ids = [item["id"] for item in items]
    sealer.freeze(candidate_ids, validity_rules)
    per_candidate = {}
    valid_count = invalid_count = 0
    for item in items:
        output = model.generate("injecagent", item, cfg)
        outcome = scorer.score("injecagent", item, output, cfg)
        per_candidate[item["id"]] = outcome
        if outcome.get("valid", True):
            valid_count += 1
        else:
            invalid_count += 1
    payload = json.dumps({"items": per_candidate}, sort_keys=True, separators=(",", ":")).encode()
    return sealer.store_receipt(label, payload, valid_count, invalid_count)

def read_held_out_result(sealer, selection_record: dict) -> dict:
    """The runner's only seam for reading held-out results back out.

    Delegates entirely to the sealer's SEALED/AUTHORIZED state machine, so
    there is no runner-side code path that can return InjecAgent plaintext,
    per-candidate outcomes, or aggregates while sealing is in effect: this
    raises PermissionError under the same conditions `HeldOutSealer.reveal`
    does.
    """
    return sealer.reveal(selection_record)

def run_baseline(manifest_path, *, model, dataset, scorer, telemetry, storage, held_out_sealer,
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

    injecagent_cfg = eval_cfg["held_out"]["injecagent"]
    validity_rules = json.dumps(manifest["held_out_policy"], sort_keys=True, separators=(",", ":"))
    receipt = _run_held_out_injecagent(
        injecagent_cfg, model=model, dataset=dataset, scorer=scorer,
        sealer=held_out_sealer, validity_rules=validity_rules, label="baseline",
    )
    metrics["held_out"] = {"injecagent": {"receipt": receipt, "commitments": held_out_sealer.commitments()}}
    log_lines.append(f"sealed injecagent: candidates={injecagent_cfg['candidate_count']} label=baseline digest={receipt['digest']}")

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
