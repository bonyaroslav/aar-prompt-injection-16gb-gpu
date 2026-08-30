"""Real, full-scale, sealed baseline run for issue #10 (RESEARCH_PLAN.md Phase 4).

Unlike `runner.gpu_smoke`, this module never builds a reduced or modified manifest
copy: it runs the exact frozen `protocol/manifest.json` sample counts declared for
Phase 4. Any change to those counts after this run requires a new protocol version
and a fresh baseline (RESEARCH_PLAN.md Sec. 7). Following the same convention as
`run_gpu_smoke`, `run_real_baseline` itself is exercised only by the real hardware
run (it needs the pinned upstream checkout, real model weights, and CUDA); the
pure/testable pieces below (sample-count resolution, path policy, resource
comparison, artifact writing) are unit-tested offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from protocol.heldout import HeldOutSealer
from protocol.validate_manifest import load as load_manifest
from runner.bundle import verify_bundle
from runner.core import resolve_sample_count, run_baseline
from runner.gpu_smoke import (
    _import_publisher, _paths_overlap, _verify_upstream, write_or_verify_data_manifest,
)
from runner.real_adapters import (
    RealDatasetAdapter, RealModelAdapter, RealScorerAdapter, RealTelemetryAdapter,
)
from runner.storage import LocalStorageAdapter


def validate_baseline_paths(repository_root: Path, suite_dir: Path, heldout_dir: Path,
                            output_root: Path, restricted_state_root: Path) -> None:
    public = [repository_root.resolve(), suite_dir.resolve(), output_root.resolve()]
    restricted = [heldout_dir.resolve(), restricted_state_root.resolve()]
    unsafe = _paths_overlap(restricted[0], restricted[1]) or any(
        _paths_overlap(secret, visible) for secret in restricted for visible in public
    )
    if unsafe:
        raise ValueError(
            "restricted roots must not overlap each other or repository/suite/output roots"
        )


def resolve_publish_counts(manifest: dict) -> dict:
    """The exact publisher `n` per benchmark implied by the frozen manifest.

    `_publish_open_prompt_injection`'s own `n` is already the desired *total* row
    count -- it divides by 3 internally (`per = n // 3`) to get "100 each injected
    task" from a declared total of 300. Passing the manifest's declared total
    straight through (not pre-dividing it) is what actually reproduces that split;
    every other benchmark's `n` is its declared sample/candidate count verbatim.
    """
    eval_cfg = manifest["evaluation"]
    opi_total = resolve_sample_count(
        eval_cfg["visible_safety"]["open_prompt_injection"]["sample_ids"]
    )
    if opi_total % 3:
        raise ValueError(
            f"open_prompt_injection sample count must be divisible by 3 injected tasks: {opi_total}"
        )
    counts = {"open_prompt_injection": opi_total}
    for name in ("tensor_trust_hijack", "tensor_trust_extract"):
        counts[name] = resolve_sample_count(eval_cfg["visible_safety"][name]["sample_ids"])
    for name in ("mmlu", "gsm8k", "ifeval"):
        counts[name] = resolve_sample_count(eval_cfg["capability"][name]["sample_ids"])
    counts["injecagent"] = eval_cfg["held_out"]["injecagent"]["candidate_count"]
    return counts


def publish_full_baseline_data(upstream_root: Path, suite_dir: Path, heldout_dir: Path,
                               manifest: dict) -> None:
    """Publish the frozen manifest's exact declared sample counts (no smoke reduction)."""
    suite_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)
    publisher = _import_publisher(upstream_root)
    counts = resolve_publish_counts(manifest)
    visible = {
        "open_prompt_injection": publisher._publish_open_prompt_injection,
        "tensor_trust_hijack": publisher._publish_tensor_trust_hijack,
        "tensor_trust_extract": publisher._publish_tensor_trust_extract,
        "mmlu": publisher._publish_mmlu,
        "gsm8k": publisher._publish_gsm8k,
        "ifeval": publisher._publish_ifeval,
    }
    visible_targets = [suite_dir / f"{name}.jsonl" for name in visible]
    visible_manifest = suite_dir / "smoke_data_manifest.json"
    heldout_targets = [
        heldout_dir / "injecagent.jsonl",
        heldout_dir / "injecagent_tool_dict.json",
        heldout_dir / "injecagent_sim_responses.json",
        heldout_dir / "injecagent_source_commit.txt",
    ]
    heldout_manifest = heldout_dir / "smoke_data_manifest.json"
    if not visible_manifest.exists() and any(path.exists() for path in visible_targets):
        raise RuntimeError("existing published visible data has no integrity manifest")
    if not heldout_manifest.exists() and any(path.exists() for path in heldout_targets):
        raise RuntimeError("existing held-out data has no integrity manifest")
    for name, publish in visible.items():
        target = suite_dir / f"{name}.jsonl"
        if not target.exists():
            publish(suite_dir, n=counts[name], seed=42)
    if not heldout_targets[0].exists():
        publisher._publish_injecagent(heldout_dir, n=counts["injecagent"], seed=42)
    cache = Path(publisher.config.HARNESS_RUNS_DIR) / "_injecagent_cache"
    commit = subprocess.run(
        ["git", "-c", f"safe.directory={cache}", "-C", str(cache), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    heldout_targets[3].write_text(commit + "\n", encoding="utf-8")
    write_or_verify_data_manifest(
        suite_dir, visible_targets,
        expected_rows={f"{name}.jsonl": counts[name] for name in visible},
    )
    write_or_verify_data_manifest(
        heldout_dir, heldout_targets, expected_rows={"injecagent.jsonl": counts["injecagent"]},
    )


def load_canonical_projection(path: Path) -> dict:
    projection = json.loads(Path(path).read_text(encoding="utf-8"))
    if not projection.get("canonical_qualification"):
        raise ValueError(f"projection artifact is not marked canonical_qualification: {path}")
    return projection


def compare_against_projection(*, manifest: dict, projection: dict, wall_seconds: float,
                               peak_vram_mb: float, bundle_bytes: int) -> dict:
    """Compare this run's measured telemetry against the smoke test's Phase-3 projection.

    Any exceeded resource limit is returned as a feasibility finding rather than
    silently absorbed -- the same policy `resource_projection.py` uses for the
    smoke test's own projection.
    """
    limits = manifest["resources"]
    peak_vram_gb = peak_vram_mb / 1024.0
    wall_hours = wall_seconds / 3600.0
    gpu_hours = wall_hours  # single-GPU run: gpu-hours == wall-hours for this stage
    bundle_gb = bundle_bytes / (1024.0 ** 3)
    findings = []
    if peak_vram_gb > limits["vram_allocated_gb_max"]:
        findings.append(
            f"measured peak_vram_gb {peak_vram_gb:.3f} exceeds "
            f"vram_allocated_gb_max {limits['vram_allocated_gb_max']}"
        )
    if wall_hours > limits["wall_hours_per_seed_max"]:
        findings.append(
            f"measured wall_hours {wall_hours:.3f} exceeds "
            f"wall_hours_per_seed_max {limits['wall_hours_per_seed_max']}"
        )
    if gpu_hours > limits["gpu_hours_total_max"]:
        findings.append(
            f"measured gpu_hours {gpu_hours:.3f} exceeds "
            f"gpu_hours_total_max {limits['gpu_hours_total_max']}"
        )
    projected_baseline_seconds = projection.get("baseline_wall_seconds")
    return {
        "measured": {
            "wall_seconds": wall_seconds, "peak_vram_gb": peak_vram_gb,
            "gpu_hours": gpu_hours, "bundle_disk_gb": bundle_gb,
        },
        "projected": {
            "baseline_wall_seconds": projected_baseline_seconds,
            "peak_vram_gb": projection.get("projected_peak_vram_gb"),
        },
        "delta": {
            "wall_seconds": (
                None if projected_baseline_seconds is None
                else wall_seconds - projected_baseline_seconds
            ),
        },
        "limits": dict(limits),
        "feasibility_findings": findings,
    }


def write_comparison_artifact(root: Path, artifact_id: str, comparison: dict) -> Path:
    artifact = Path(root) / artifact_id
    if artifact.exists():
        raise FileExistsError(f"comparison artifact already exists: {artifact}")
    artifact.mkdir(parents=True)
    payload = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    target = artifact / "baseline_resource_comparison.json"
    target.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (artifact / "checksums.sha256").write_text(
        f"{digest}  baseline_resource_comparison.json\n", encoding="utf-8"
    )
    return artifact


def _directory_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())


def baseline_notes_text(stage: str) -> str:
    """The real baseline's own evidence caption.

    Deliberately distinct from `RealTelemetryAdapter`'s smoke-qualification default
    text: this run is the frozen Phase-4 scientific baseline (RESEARCH_PLAN.md
    Sec. 7), not a hardware qualification, and its notes.md must say so.
    """
    return (
        f"# {stage.replace('_', ' ').title()} run notes\n\n"
        "Real Hugging Face/CUDA adapters; full-scale frozen Phase-4 baseline "
        "(RESEARCH_PLAN.md Sec. 7) at the manifest's exact declared sample counts. "
        "Not a hardware qualification or smoke test.\n"
    )


def run_real_baseline(*, manifest_path: Path, upstream_root: Path, suite_dir: Path,
                      heldout_dir: Path, output_root: Path, restricted_state_root: Path,
                      model_cache: Path, projection_artifact: Path,
                      reproduction_command: str | None = None) -> dict:
    frozen = load_manifest(manifest_path)
    validate_baseline_paths(
        manifest_path.resolve().parent.parent, suite_dir, heldout_dir,
        output_root, restricted_state_root,
    )
    upstream_provenance = _verify_upstream(
        upstream_root, frozen["upstream"]["commit"], frozen["upstream"]["tree"]
    )
    publish_full_baseline_data(upstream_root, suite_dir, heldout_dir, frozen)

    dataset = RealDatasetAdapter(suite_dir, heldout_dir)
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(
        repo_id=frozen["model"]["id"], revision=frozen["model"]["revision"],
        cache_dir=model_cache, local_files_only=True,
    )
    model = RealModelAdapter(
        snapshot, None, upstream_root, decoding=frozen["evaluation"]["decoding"],
    )
    scorer = RealScorerAdapter(upstream_root)
    telemetry = RealTelemetryAdapter()
    telemetry.command_text = reproduction_command
    telemetry.notes_text = baseline_notes_text
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output_root.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    baseline = run_baseline(
        manifest_path, model=model, dataset=dataset, scorer=scorer, telemetry=telemetry,
        storage=LocalStorageAdapter(output_root),
        held_out_sealer=HeldOutSealer(restricted_state_root / f"seal-{stamp}"),
        run_id=f"real-baseline-{stamp}",
    )
    wall_seconds = time.monotonic() - start
    peak_vram_mb = max((row["vram_mb"] for row in telemetry._rows), default=0)
    model.release()

    bundle_dir = Path(baseline.bundle_dir)
    verify_bundle(bundle_dir)

    projection = load_canonical_projection(projection_artifact)
    comparison = compare_against_projection(
        manifest=frozen, projection=projection, wall_seconds=wall_seconds,
        peak_vram_mb=peak_vram_mb, bundle_bytes=_directory_bytes(bundle_dir),
    )
    comparison["upstream_provenance"] = upstream_provenance
    comparison_dir = write_comparison_artifact(
        output_root, f"real-baseline-comparison-{stamp}", comparison
    )
    return {
        "baseline": baseline, "comparison": comparison, "comparison_dir": str(comparison_dir),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("protocol/manifest.json"))
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--restricted-state-root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--projection-artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    command_text = shlex.join([sys.executable, "-m", "runner.real_baseline", *(argv or sys.argv[1:])])
    result = run_real_baseline(
        manifest_path=args.manifest, upstream_root=args.upstream_root,
        suite_dir=args.suite_dir, heldout_dir=args.heldout_dir,
        output_root=args.output_root, restricted_state_root=args.restricted_state_root,
        model_cache=args.model_cache, projection_artifact=args.projection_artifact,
        reproduction_command=command_text,
    )
    print(json.dumps({
        "baseline_bundle": result["baseline"].bundle_dir,
        "comparison_dir": result["comparison_dir"],
        "feasibility_findings": result["comparison"]["feasibility_findings"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
