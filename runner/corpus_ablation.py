"""Issue #31 corpus-ablation orchestration.

This module owns ablation-only corpus validation and evidence shapes.  The real
GPU entrypoint is intentionally separate from Attempt-1 training/selection: its
outputs belong beneath the gitignored ``ablation/`` root and are never eligible
for selection, the frozen bootstrap, or held-out evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import signal
import sys
import time
from pathlib import Path

from protocol.ablation.manifest import canonical_digest, load as load_ablation_manifest
from protocol.validate_manifest import load as load_frozen_manifest
from runner.ablation_training import MidEpochCheckpointStore
from runner.evaluation import EvaluationRecovery, run_trained_evaluation
from runner.gpu_smoke import _verify_upstream
from runner.real_adapters import RealModelAdapter, RealScorerAdapter, RealTelemetryAdapter
from runner.real_seed_run import load_training_examples
from runner.real_training import RealQLoRATrainerAdapter
from runner.recovery import RecoveryWorkspace, StageSignature
from runner.storage import LocalStorageAdapter
from runner.training import _directory_digest
from training_data.build import run_real_build


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST_PATH = REPO_ROOT / "protocol" / "manifest.json"


class DeliberateInterruption(RuntimeError):
    """Raised after a durable test checkpoint to exercise mid-epoch recovery."""


class CorpusValidationError(ValueError):
    """The ablation corpus is not the authorized complete clean-only corpus."""


def validate_corpus(report: dict, examples) -> None:
    """Fail before GPU work unless the builder produced the authorized corpus."""
    if report.get("shortfalls"):
        raise CorpusValidationError(f"ablation corpus has category shortfall: {report['shortfalls']}")
    if report.get("total") != 5000:
        raise CorpusValidationError(f"ablation corpus must contain exactly 5000 examples, got {report.get('total')}")
    if any(example.get("category") == "prompt_injection" for example in examples):
        raise CorpusValidationError("ablation corpus contains a prompt-injection example")


def append_attempt(*, rows: list[dict], epoch: int, status: str, recovery_evidence: dict) -> list[dict]:
    """Return a ledger row retaining the recovery facts needed by the decision record."""
    return [
        *rows,
        {
            "epoch": int(epoch),
            "status": status,
            "recovery_evidence": dict(recovery_evidence),
        },
    ]


def resource_row(*, gpu_hours: float, wall_hours: float, source: str) -> dict:
    """Shape ablation compute separately from frozen scientific Attempt-1 totals."""
    return {
        "category": "ablation",
        "label": "corpus-ablation",
        "gpu_hours": float(gpu_hours),
        "wall_hours": float(wall_hours),
        "source": str(source),
    }


class VisibleOnlyDataset:
    """Read only the six published visible benchmarks; no sealed-root argument exists."""

    def __init__(self, suite_dir: Path, max_items_per_benchmark: int | None = None):
        self.suite_dir = Path(suite_dir)
        self.max_items_per_benchmark = max_items_per_benchmark

    def load_items(self, benchmark: str, sample_count: int):
        from runner.real_adapters import _REQUIRED_FIELDS, _canonical_id, _read_jsonl

        if benchmark == "injecagent":
            raise ValueError("the ablation never loads the held-out benchmark")
        expected = min(sample_count, self.max_items_per_benchmark) if self.max_items_per_benchmark else sample_count
        rows = _read_jsonl(self.suite_dir / f"{benchmark}.jsonl")
        if len(rows) < expected:
            raise ValueError(f"expected {expected} published rows for {benchmark}, found {len(rows)}")
        items = rows[:expected]
        for item in items:
            missing = sorted(_REQUIRED_FIELDS[benchmark] - item.keys())
            if missing:
                raise ValueError(f"{benchmark} published row missing required fields: {', '.join(missing)}")
            item["id"] = _canonical_id(benchmark, item)
        return items


def _require_ablation_output_root(output_root: Path) -> None:
    allowed = (REPO_ROOT / "ablation").resolve()
    try:
        output_root.resolve().relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"ablation output root must stay below {allowed}") from error


def _repo_relative(path: Path) -> str:
    """Return a stable repository-relative evidence path."""
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _epoch_signature(*, ablation: dict, frozen: dict, epoch: int, corpus_digest: str) -> StageSignature:
    return StageSignature.create(
        manifest_digest="sha256:" + canonical_digest(ablation),
        protocol_version=ablation["ablation_version"],
        upstream_commit=frozen["upstream"]["commit"],
        upstream_tree=frozen["upstream"]["tree"],
        model_revision=frozen["model"]["revision"],
        seed=ablation["training"]["seed"],
        stage="corpus_ablation_training",
        epoch=epoch,
        checkpoint_digest="sha256:" + corpus_digest,
        effective_evaluation_config={"sequence_length": frozen["training"]["data"]["max_sequence_length"]},
        expected_example_ids=[],
    )


def _completed_epoch(workspace: RecoveryWorkspace, stage: str, signature: StageSignature) -> dict | None:
    if not workspace.has_state(stage):
        return None
    record = workspace.transaction_state(stage)
    stored = StageSignature.create(**record["signature"])
    if stored.digest != record.get("signature_digest") or stored.first_difference(signature):
        raise ValueError(f"ablation recovery state is incompatible for {stage}")
    transaction = record["transaction"]
    if record.get("status") != "completed":
        return None
    checkpoint = Path(transaction["checkpoint"]["merged_dir"])
    if not checkpoint.is_dir() or _directory_digest(checkpoint) != transaction["checkpoint"]["integrity"]:
        raise ValueError(f"ablation completed checkpoint is missing or changed: {checkpoint}")
    return transaction


def run_ablation(*, ablation_manifest_path: Path, upstream_root: Path, suite_dir: Path,
                 output_root: Path, recovery_root: Path, model_cache: Path, work_dir: Path,
                 smoke_max_steps: int | None = None, max_items_per_benchmark: int | None = None,
                 deliberate_interrupt_once_after_checkpoint: int | None = None) -> dict:
    """Build, train, evaluate, and record the authorized separate corpus ablation."""
    output_root = Path(output_root)
    _require_ablation_output_root(output_root)
    ablation = load_ablation_manifest(ablation_manifest_path)
    frozen = load_frozen_manifest(FROZEN_MANIFEST_PATH)
    if ablation["training"]["seed"] != 42:
        raise ValueError("issue #31 is authorized only for seed 42")
    if ablation["resources"]["prior_scientific_gpu_hours"] + ablation["resources"]["projected_gpu_hours"] > ablation["resources"]["cumulative_gpu_hour_cap"]:
        raise ValueError("projected ablation would exceed the cumulative GPU-hour cap")
    _verify_upstream(upstream_root, frozen["upstream"]["commit"], frozen["upstream"]["tree"])

    corpus_dir = output_root / "corpus"
    result = run_real_build(
        upstream_root=upstream_root, work_dir=corpus_dir / "_exclusion_pool_scratch",
        dataset_path=corpus_dir / "dataset.jsonl", report_path=corpus_dir / "report.json",
        token_cap=int(ablation["corpus"]["construction_token_cap"]),
        targets=ablation["corpus"]["targets"],
        dolly_oversample_factor=ablation["corpus"]["dolly_oversample_factor"],
    )
    validate_corpus(result["report"], [example.to_record() for example in result["examples"]])
    report = json.loads((corpus_dir / "report.json").read_text(encoding="utf-8"))
    corpus_digest = hashlib.sha256((corpus_dir / "dataset.jsonl").read_bytes()).hexdigest()
    report["dataset_sha256"] = corpus_digest
    _write_json(corpus_dir / "report.json", report)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("real corpus ablation requires CUDA")
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(
        repo_id=frozen["model"]["id"], revision=frozen["model"]["revision"],
        cache_dir=model_cache, local_files_only=True,
    )
    examples = load_training_examples(corpus_dir / "dataset.jsonl")
    optimizer = frozen["training"]["optimizer"]
    total_steps = math.ceil(math.ceil(len(examples) / int(optimizer["micro_batch"])) / int(optimizer["gradient_accumulation"]))
    if smoke_max_steps is not None:
        total_steps = min(total_steps, smoke_max_steps)
    workspace = RecoveryWorkspace(Path(recovery_root), output_root)
    trainer = RealQLoRATrainerAdapter(snapshot, None, examples, work_dir, smoke_max_steps=smoke_max_steps)
    started = time.monotonic()
    attempts = []
    checkpoints = {}
    try:
        for epoch in range(1, int(ablation["training"]["epochs"]) + 1):
            stage = f"corpus-ablation-seed42-epoch{epoch}"
            signature = _epoch_signature(ablation=ablation, frozen=frozen, epoch=epoch, corpus_digest=corpus_digest)
            completed = _completed_epoch(workspace, stage, signature)
            if completed is not None:
                checkpoints[epoch] = completed["checkpoint"]
                attempts = append_attempt(rows=attempts, epoch=epoch, status="reused", recovery_evidence=completed["recovery_evidence"])
                continue
            workspace.write_transaction_state(stage, signature, state="running", transaction={"corpus_digest": corpus_digest})
            store = MidEpochCheckpointStore(workspace, stage, signature)
            sentinel = workspace.root / f"{stage}.deliberate-interruption.json"

            def interrupt_after_durable_save(measurement):
                if deliberate_interrupt_once_after_checkpoint is None or sentinel.exists():
                    return
                if measurement.step_index >= deliberate_interrupt_once_after_checkpoint:
                    _write_json(sentinel, {"step_index": measurement.step_index, "reason": "issue-31 deliberate recovery test"})
                    raise DeliberateInterruption(f"deliberate interruption after durable step {measurement.step_index}")

            try:
                recovery = trainer.run_ablation_epoch(
                    protocol_version=ablation["ablation_version"], seed=42, epoch=epoch,
                    sequence_length=int(frozen["training"]["data"]["max_sequence_length"]),
                    config=frozen["training"], checkpoint_store=store, total_steps=total_steps,
                    checkpoint_interval=int(ablation["recovery"]["checkpoint_interval_steps"]),
                    on_checkpoint=interrupt_after_durable_save,
                )
            except DeliberateInterruption:
                workspace.write_transaction_state(stage, signature, state="interrupted", transaction={"corpus_digest": corpus_digest, "deliberate": True})
                raise
            if any(item["save_seconds"] > 30 for item in recovery.recovery_evidence["save_measurements"]):
                raise RuntimeError("ablation recovery checkpoint save exceeded 30 seconds")
            fingerprint = trainer.save_ablation_checkpoint(
                seed=42, epoch=epoch, sequence_length=int(frozen["training"]["data"]["max_sequence_length"]),
            )
            merged_dir = output_root / "checkpoints" / f"epoch-{epoch}"
            trainer.merge_checkpoint(fingerprint, merged_dir)
            checkpoint = {"fingerprint": fingerprint, "merged_dir": str(merged_dir),
                          "sequence_length": int(frozen["training"]["data"]["max_sequence_length"]),
                          "integrity": _directory_digest(merged_dir)}
            transaction = {"checkpoint": checkpoint, "recovery_evidence": recovery.recovery_evidence, "corpus_digest": corpus_digest}
            workspace.write_transaction_state(stage, signature, state="completed", transaction=transaction)
            checkpoints[epoch] = checkpoint
            attempts = append_attempt(rows=attempts, epoch=epoch, status="completed", recovery_evidence=recovery.recovery_evidence)

        storage = LocalStorageAdapter(output_root / "evaluations")
        evaluations = {}
        for epoch, checkpoint in sorted(checkpoints.items()):
            model = RealModelAdapter(checkpoint["merged_dir"], None, upstream_root, decoding=frozen["evaluation"]["decoding"])
            telemetry = RealTelemetryAdapter()
            try:
                evaluation = run_trained_evaluation(
                    FROZEN_MANIFEST_PATH, model=model,
                    dataset=VisibleOnlyDataset(suite_dir, max_items_per_benchmark),
                    scorer=RealScorerAdapter(upstream_root), telemetry=telemetry, storage=storage,
                    seed=42, epoch=epoch, checkpoint=checkpoint,
                    run_id=f"ablation-seed42-epoch{epoch}",
                    recovery=EvaluationRecovery(workspace, f"corpus-ablation-eval{epoch}"),
                )
                evaluations[epoch] = {"bundle_dir": evaluation.bundle_dir, "metrics": evaluation.metrics}
            finally:
                model.release()
        wall_hours = (time.monotonic() - started) / 3600.0
        attempts_path = output_root / "attempts.json"
        _write_json(attempts_path, {"attempts": attempts})
        resources = resource_row(gpu_hours=wall_hours, wall_hours=wall_hours, source=_repo_relative(attempts_path))
        _write_json(output_root / "resource.json", resources)
        return {"corpus_report": report, "checkpoints": checkpoints, "evaluations": evaluations,
                "attempts": attempts, "resource": resources}
    finally:
        trainer.release()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--ablation-manifest", type=Path, default=REPO_ROOT / "protocol" / "ablation" / "corpus-ablation-2026-09-02.json")
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--smoke-max-steps", type=int)
    parser.add_argument("--smoke-max-items-per-benchmark", type=int)
    parser.add_argument("--deliberate-interrupt-once-after-checkpoint", type=int)
    args = parser.parse_args(argv)
    try:
        result = run_ablation(
            ablation_manifest_path=args.ablation_manifest, upstream_root=args.upstream_root, suite_dir=args.suite_dir,
            output_root=args.output_root, recovery_root=args.recovery_root, model_cache=args.model_cache,
            work_dir=args.work_dir, smoke_max_steps=args.smoke_max_steps,
            max_items_per_benchmark=args.smoke_max_items_per_benchmark,
            deliberate_interrupt_once_after_checkpoint=args.deliberate_interrupt_once_after_checkpoint,
        )
    except DeliberateInterruption as error:
        print(str(error), file=sys.stderr)
        return 75
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
