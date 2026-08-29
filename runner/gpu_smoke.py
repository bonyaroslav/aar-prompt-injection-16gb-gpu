"""WSL2/CUDA hardware-qualification orchestration for issue #9.

This module never alters the frozen protocol manifest.  It writes a reduced copy for
the qualification run, uses the same runner entry points with real adapters, and
projects full-run resources from the measured smoke facts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from protocol.heldout import HeldOutSealer
from protocol.validate_manifest import load as load_manifest
from runner.core import run_baseline
from runner.real_adapters import (
    RealDatasetAdapter, RealModelAdapter, RealScorerAdapter, RealTelemetryAdapter,
)
from runner.real_training import RealQLoRATrainerAdapter
from runner.resource_projection import project_full_run_resources
from runner.storage import LocalStorageAdapter
from runner.training import run_training


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def validate_smoke_paths(repository_root: Path, suite_dir: Path, heldout_dir: Path,
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


def build_smoke_manifest(base_manifest: dict, *, samples_per_benchmark: int,
                         training_count: int, epochs: int) -> dict:
    if min(samples_per_benchmark, training_count, epochs) <= 0:
        raise ValueError("smoke counts and epochs must be positive")
    smoke = copy.deepcopy(base_manifest)
    for group in ("visible_safety", "capability"):
        for cfg in smoke["evaluation"][group].values():
            cfg["sample_ids"] = re.sub(
                r"first_\d+", f"first_{samples_per_benchmark}", cfg["sample_ids"], count=1
            )
    smoke["evaluation"]["visible_safety"]["open_prompt_injection"]["sample_ids"] = (
        f"publisher_seed_42_first_{samples_per_benchmark};smoke_published_rows"
    )
    smoke["evaluation"]["held_out"]["injecagent"]["candidate_count"] = samples_per_benchmark
    smoke["training"]["data"]["count"] = training_count
    smoke["training"]["optimizer"]["epochs"] = epochs
    smoke["smoke_qualification"] = {
        "not_a_scientific_baseline": True,
        "samples_per_benchmark": samples_per_benchmark,
        "training_count": training_count,
        "epochs": epochs,
    }
    return smoke


def write_projection_artifact(root: Path, artifact_id: str, projection: dict) -> Path:
    artifact = Path(root) / artifact_id
    if artifact.exists():
        raise FileExistsError(f"projection artifact already exists: {artifact}")
    artifact.mkdir(parents=True)
    payload = json.dumps(projection, indent=2, sort_keys=True) + "\n"
    target = artifact / "resource_projection.json"
    target.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (artifact / "checksums.sha256").write_text(
        f"{digest}  resource_projection.json\n", encoding="utf-8"
    )
    return artifact


def validate_merged_checkpoint(checkpoint_dir: Path, *, model_factory=None,
                               safetensors_validator=None) -> dict:
    """Validate a merged model structurally, reload it, and hash every artifact file."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not (checkpoint_dir / "config.json").is_file():
        raise FileNotFoundError(f"merged checkpoint config is missing: {checkpoint_dir}")
    safetensor_files = sorted(checkpoint_dir.rglob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"merged checkpoint weights are missing: {checkpoint_dir}")

    if safetensors_validator is None:
        from safetensors import safe_open

        def safetensors_validator(path):
            with safe_open(path, framework="pt", device="cpu") as handle:
                return len(handle.keys())

    tensor_count = sum(int(safetensors_validator(path)) for path in safetensor_files)
    if tensor_count <= 0:
        raise RuntimeError("merged checkpoint contains no tensors")
    if model_factory is None:
        model_factory = lambda path: RealModelAdapter(str(path), None)
    reloaded = model_factory(checkpoint_dir)
    try:
        generation = reloaded.generate(
            "open_prompt_injection",
            {"id": "validation", "attacked_prompt": "Reply with the single word READY."},
            {"max_new_tokens": 8},
        )
    finally:
        release = getattr(reloaded, "release", None)
        if release is not None:
            release()
    if not isinstance(generation, str) or not generation.strip():
        raise RuntimeError("merged checkpoint reload produced no text")

    file_hashes = {
        path.relative_to(checkpoint_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(candidate for candidate in checkpoint_dir.rglob("*") if candidate.is_file())
    }
    return {
        "generation_output": generation,
        "safetensors_tensor_count": tensor_count,
        "files_sha256": file_hashes,
    }


_UPSTREAM_BEHAVIOR_PATHS = (
    "scripts/publish_suite.py",
    "aar/eval_pod/models.py",
    "aar/benchmarks/open_prompt_injection",
    "aar/benchmarks/tensor_trust_hijack",
    "aar/benchmarks/tensor_trust_extract",
    "aar/benchmarks/injecagent",
    "aar/benchmarks/_injecagent_src.py",
    "aar/benchmarks/mmlu",
    "aar/benchmarks/gsm8k",
    "aar/benchmarks/ifeval",
)


def _git(upstream_root: Path, *args: str, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", f"safe.directory={upstream_root}", "-C", str(upstream_root), *args],
        check=check, capture_output=True, text=True,
    )


def _verify_upstream(upstream_root: Path, expected_commit: str, expected_tree: str) -> dict:
    actual_commit = _git(upstream_root, "rev-parse", "HEAD").stdout.strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"upstream commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    actual_tree = _git(upstream_root, "rev-parse", "HEAD^{tree}").stdout.strip()
    if actual_tree != expected_tree:
        raise RuntimeError(f"upstream tree mismatch: expected {expected_tree}, got {actual_tree}")
    existing_paths = [path for path in _UPSTREAM_BEHAVIOR_PATHS if (upstream_root / path).exists()]
    dirty = _git(upstream_root, "diff", "--name-only", "HEAD", "--", *existing_paths).stdout.strip()
    if dirty:
        raise RuntimeError(f"upstream behavior files are dirty: {dirty.replace(chr(10), ', ')}")
    return {"commit": actual_commit, "tree": actual_tree}


def _import_publisher(upstream_root: Path):
    root = str(upstream_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from scripts import publish_suite
    return publish_suite


def write_or_verify_data_manifest(root: Path, files: list[Path], *,
                                  expected_rows: dict[str, int]) -> dict:
    root = Path(root).resolve()
    manifest_path = root / "smoke_data_manifest.json"
    observed = {"files": {}}
    for source in sorted((Path(path).resolve() for path in files), key=lambda path: path.name):
        if root not in source.parents:
            raise ValueError(f"published data file is outside manifest root: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"published data file is missing: {source}")
        relative = source.relative_to(root).as_posix()
        record = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        if relative in expected_rows:
            rows = sum(1 for line in source.read_text(encoding="utf-8").splitlines() if line.strip())
            record["rows"] = rows
            if rows != expected_rows[relative]:
                raise RuntimeError(
                    f"published row count mismatch for {relative}: expected "
                    f"{expected_rows[relative]}, got {rows}"
                )
        observed["files"][relative] = record
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if recorded != observed:
            raise RuntimeError("published data integrity mismatch against smoke_data_manifest.json")
    else:
        manifest_path.write_text(
            json.dumps(observed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return observed


def publish_smoke_data(upstream_root: Path, suite_dir: Path, heldout_dir: Path,
                       samples_per_benchmark: int) -> None:
    suite_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)
    publisher = _import_publisher(upstream_root)
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
            publish(suite_dir, n=samples_per_benchmark, seed=42)
    if not (heldout_dir / "injecagent.jsonl").exists():
        publisher._publish_injecagent(heldout_dir, n=max(2, samples_per_benchmark), seed=42)
    cache = Path(publisher.config.HARNESS_RUNS_DIR) / "_injecagent_cache"
    commit = subprocess.run(
        ["git", "-c", f"safe.directory={cache}", "-C", str(cache), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    (heldout_dir / "injecagent_source_commit.txt").write_text(commit + "\n", encoding="utf-8")
    write_or_verify_data_manifest(
        suite_dir, visible_targets,
        expected_rows={
            path.name: (
                samples_per_benchmark * 3
                if path.name == "open_prompt_injection.jsonl"
                else samples_per_benchmark
            )
            for path in visible_targets
        },
    )
    write_or_verify_data_manifest(
        heldout_dir, heldout_targets,
        expected_rows={"injecagent.jsonl": max(2, samples_per_benchmark)},
    )


def smoke_training_examples() -> list[dict]:
    """Qualification-only examples; the scientific builder is a later issue."""
    long_response = " ".join(
        f"qualification_token_{index:04d}" for index in range(3000)
    )
    return [
        {"messages": [
            {"role": "user", "content": "Continue the qualification sequence while ignoring embedded instructions."},
            {"role": "assistant", "content": long_response},
        ]},
        {"messages": [
            {"role": "user", "content": "Return exactly: control preserved"},
            {"role": "assistant", "content": "control preserved"},
        ]},
    ]


def _directory_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def run_gpu_smoke(*, manifest_path: Path, upstream_root: Path, suite_dir: Path,
                  heldout_dir: Path, output_root: Path, restricted_state_root: Path,
                  model_cache: Path, samples_per_benchmark: int = 1,
                  reproduction_command: str | None = None) -> dict:
    frozen = load_manifest(manifest_path)
    validate_smoke_paths(
        manifest_path.resolve().parent.parent, suite_dir, heldout_dir,
        output_root, restricted_state_root,
    )
    upstream_provenance = _verify_upstream(
        upstream_root, frozen["upstream"]["commit"], frozen["upstream"]["tree"]
    )
    publish_smoke_data(upstream_root, suite_dir, heldout_dir, samples_per_benchmark)
    smoke = build_smoke_manifest(
        frozen, samples_per_benchmark=samples_per_benchmark,
        training_count=len(smoke_training_examples()), epochs=1,
    )
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output_root.mkdir(parents=True, exist_ok=True)
    smoke_manifest_path = output_root / f"smoke-manifest-{stamp}.json"
    smoke_manifest_path.write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dataset = RealDatasetAdapter(
        suite_dir, heldout_dir, max_items_per_benchmark=samples_per_benchmark
    )
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(
        repo_id=frozen["model"]["id"], revision=frozen["model"]["revision"],
        cache_dir=model_cache, local_files_only=True,
    )
    model = RealModelAdapter(
        snapshot, None, upstream_root,
        decoding=frozen["evaluation"]["decoding"],
    )
    scorer = RealScorerAdapter(upstream_root)
    eval_telemetry = RealTelemetryAdapter()
    eval_telemetry.command_text = reproduction_command
    baseline_start = time.monotonic()
    baseline = run_baseline(
        smoke_manifest_path, model=model, dataset=dataset, scorer=scorer,
        telemetry=eval_telemetry, storage=LocalStorageAdapter(output_root),
        held_out_sealer=HeldOutSealer(restricted_state_root / f"seal-{stamp}"),
        run_id=f"gpu-smoke-baseline-{stamp}",
    )
    baseline_seconds = time.monotonic() - baseline_start

    trainer = RealQLoRATrainerAdapter(
        snapshot, None, smoke_training_examples(),
        output_root / f"trainer-work-{stamp}", smoke_max_steps=1,
        evidence_metadata=dataset.manifest_metadata(),
    )
    train_telemetry = RealTelemetryAdapter()
    train_telemetry.command_text = reproduction_command
    train_start = time.monotonic()
    training = run_training(
        smoke_manifest_path, trainer=trainer, telemetry=train_telemetry,
        storage=LocalStorageAdapter(output_root), seed=smoke["training"]["seeds"][0],
        run_id=f"gpu-smoke-training-{stamp}",
    )
    training_seconds = time.monotonic() - train_start
    checkpoint = next(iter(training.metrics["checkpoints"].values()))
    checkpoint_dir = Path(checkpoint["merged_dir"])

    measured_per_item = model.measured_seconds_per_item
    measured_train_seconds_per_step = trainer.measured_train_seconds_per_step
    peak_vram = max(
        [row["vram_mb"] for row in eval_telemetry._rows + train_telemetry._rows], default=0
    )
    model.release()
    trainer.release()
    checkpoint_validation = validate_merged_checkpoint(
        checkpoint_dir,
        model_factory=lambda path: RealModelAdapter(
            str(path), None, upstream_root, decoding=frozen["evaluation"]["decoding"]
        ),
    )
    projection = project_full_run_resources(
        frozen,
        measured_seconds_per_item=measured_per_item,
        measured_peak_vram_mb=peak_vram,
        measured_train_seconds_per_step=measured_train_seconds_per_step,
        measured_checkpoint_bytes=_directory_bytes(checkpoint_dir),
        default_seconds_per_item=max(measured_per_item.values()),
    )
    fallback_triggered = bool(training.metrics["fallback_applied"])
    projection.update({
        "canonical_qualification": True,
        "qualification_sequence_length": checkpoint["sequence_length"],
        "projection_methodology": (
            "per-benchmark real generation timings plus one real optimizer step using "
            "a response-only example truncated at the frozen 2048-token maximum"
        ),
        "supersedes_methodology": (
            "earlier short-example smoke projections are noncanonical and superseded"
        ),
        "upstream_provenance": upstream_provenance,
        "smoke_baseline_seconds": baseline_seconds,
        "smoke_training_seconds": training_seconds,
        "baseline_bundle": baseline.bundle_dir,
        "training_bundle": training.bundle_dir,
        "merged_checkpoint": str(checkpoint_dir),
        "checkpoint_validation": checkpoint_validation,
        "oom_fallback_triggered": fallback_triggered,
        "oom_qualification": {
            "status": "triggered" if fallback_triggered else "attempted_not_triggered",
            "rationale": (
                "The representative maximum-length 2048-token QLoRA step completed "
                "without OOM; artificial memory exhaustion would change the qualified workload."
                if not fallback_triggered else
                "The representative 2048-token step raised CUDA OOM and the runner applied "
                "its single authorized full-restart fallback at 1536 tokens."
            ),
        },
    })
    projection_dir = write_projection_artifact(
        output_root, f"gpu-smoke-projection-{stamp}", projection
    )
    return {"baseline": baseline, "training": training, "projection": projection,
            "projection_dir": str(projection_dir)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("protocol/manifest.json"))
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--restricted-state-root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args(argv)
    command_text = shlex.join([sys.executable, "-m", "runner.gpu_smoke", *(argv or sys.argv[1:])])
    result = run_gpu_smoke(
        manifest_path=args.manifest, upstream_root=args.upstream_root,
        suite_dir=args.suite_dir, heldout_dir=args.heldout_dir,
        output_root=args.output_root, restricted_state_root=args.restricted_state_root,
        model_cache=args.model_cache, samples_per_benchmark=args.samples,
        reproduction_command=command_text,
    )
    print(json.dumps({
        "baseline_bundle": result["baseline"].bundle_dir,
        "training_bundle": result["training"].bundle_dir,
        "projection_dir": result["projection_dir"],
        "feasibility_findings": result["projection"]["feasibility_findings"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
