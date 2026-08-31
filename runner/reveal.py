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
import dataclasses, hashlib, json, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path

from protocol.validate_manifest import load as load_manifest, sha256 as manifest_sha256
from runner.bundle import CHECKSUM_FILE, write_bundle, finalize_bundle, verify_bundle
from runner.core import _run_held_out_injecagent
from runner.selection import finalize_selection_record, _digest
from runner.recovery import AttemptLedger, RecoveryWorkspace, StageSignature


TRANSACTION_STATES = (
    "SEALED", "SELECTION_FINALIZED", "AUTHORIZED", "TRAINED_RESULT_SEALED", "REVEALED",
)


@dataclasses.dataclass(frozen=True)
class HeldOutRevealRecovery:
    """External durable state for one selected held-out evaluation/reveal."""

    workspace: RecoveryWorkspace
    stage_key: str
    ledger: AttemptLedger | None = None

    def attempt_ledger(self) -> AttemptLedger:
        return self.ledger or AttemptLedger(self.workspace.root / "attempts.jsonl")


@dataclasses.dataclass(frozen=True)
class HeldOutRevealTransactionResult:
    state: str
    selection_digest: str
    reveal: "RevealResult"
    reveal_count: int


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


def _isoformat(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _transaction_signature(manifest, manifest_path, *, selection_digest: str,
                           checkpoint_digest: str, candidate_commitment: str,
                           authorization_identity: str) -> StageSignature:
    """A public-metadata-only signature for the held-out transaction.

    Candidate IDs, prompts, and sealed outcomes are intentionally absent.  The
    candidate commitment binds the frozen private population without revealing
    it, while the selected checkpoint and selection-record digest bind the
    visible-only choice that authorized this one transaction.
    """
    return StageSignature.create(
        manifest_digest="sha256:" + manifest_sha256(manifest_path),
        protocol_version=manifest["protocol_version"],
        upstream_commit=manifest["upstream"]["commit"],
        upstream_tree=manifest["upstream"]["tree"],
        model_revision=manifest["model"]["revision"],
        seed=None,
        stage="heldout_reveal_transaction",
        epoch=selection_digest,
        checkpoint_digest=checkpoint_digest,
        effective_evaluation_config={
            "held_out": manifest["evaluation"]["held_out"],
            "selection_digest": selection_digest,
            "candidate_commitment": candidate_commitment,
            "authorization_identity": authorization_identity,
        },
        expected_example_ids=[],
    )


def _transaction_identity(selection_record: dict, sealer, authorization_identity: str) -> dict:
    commitments = sealer.commitments()
    candidate_commitment = commitments.get("candidates")
    if not candidate_commitment:
        raise ValueError("candidate commitment is not sealed")
    checkpoint_digest = selection_record.get("selected_checkpoint_digest")
    if not checkpoint_digest:
        raise ValueError("selection has no selected checkpoint")
    return {
        "selection_digest": _digest(selection_record),
        "checkpoint_digest": checkpoint_digest,
        "candidate_commitment": candidate_commitment,
        "authorization_identity": authorization_identity,
    }


def _validate_transaction_identity(record: dict, requested: dict) -> None:
    stored = record.get("transaction", {}).get("identity", {})
    for field, display in (
        ("selection_digest", "selection digest"),
        ("checkpoint_digest", "checkpoint"),
        ("candidate_commitment", "candidate commitment"),
        ("authorization_identity", "authorization identity"),
    ):
        if stored.get(field) != requested[field]:
            raise ValueError(f"held-out transaction {display} changed")


def _read_checksums(bundle_dir: Path) -> dict:
    checksums = {}
    for line in (bundle_dir / CHECKSUM_FILE).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    checksums[CHECKSUM_FILE] = hashlib.sha256((bundle_dir / CHECKSUM_FILE).read_bytes()).hexdigest()
    return checksums


def _result_from_bundle(bundle_dir: Path, selection_digest: str) -> RevealResult:
    verify_bundle(bundle_dir)
    metrics = json.loads((bundle_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest_record = json.loads((bundle_dir / "manifest.yaml").read_text(encoding="utf-8"))
    if metrics.get("selection_digest") != selection_digest or manifest_record.get("selection_digest") != selection_digest:
        raise ValueError("finalized reveal bundle selection digest changed")
    return RevealResult(
        run_id=manifest_record["run_id"], stage="reveal", bundle_dir=str(bundle_dir),
        checksums=_read_checksums(bundle_dir), metrics=metrics,
    )


def run_selection_and_reveal(
    manifest_path, *, selection_record: dict, selection_path: str | Path, sealer,
    model, dataset, scorer, storage, telemetry, recovery: HeldOutRevealRecovery,
    authorization_identity: str, checkpoint_digest: str | None = None,
    fail_after: str | None = None,
) -> HeldOutRevealTransactionResult:
    """Durably perform the one permitted selected-checkpoint held-out reveal.

    This is the only transaction entry point for the post-selection path.  It
    treats all recovery payloads as opaque external state and archives only the
    normal combined, checksummed reveal bundle.  ``fail_after`` exists solely
    for deterministic fault-injection tests; callers must not use it in a real
    run.
    """
    if fail_after is not None and fail_after not in TRANSACTION_STATES:
        raise ValueError(f"unknown transaction transition: {fail_after}")
    selected_checkpoint = selection_record.get("selected_checkpoint_digest")
    if checkpoint_digest is not None and checkpoint_digest != selected_checkpoint:
        raise ValueError("held-out transaction checkpoint changed from finalized selection")
    manifest = load_manifest(manifest_path)
    identity = _transaction_identity(selection_record, sealer, authorization_identity)
    signature = _transaction_signature(manifest, manifest_path, **identity)
    workspace = recovery.workspace
    stage_key = recovery.stage_key

    if workspace.has_state(stage_key):
        stored = workspace.transaction_state(stage_key)
        _validate_transaction_identity(stored, identity)
        stored_signature = StageSignature.create(**stored["signature"])
        differing = stored_signature.first_difference(signature)
        if differing:
            raise ValueError(f"held-out transaction signature changed on {differing}")
        state = stored["status"]
        if state not in TRANSACTION_STATES:
            raise ValueError(f"unknown durable held-out transaction state: {state}")
        transaction = dict(stored["transaction"])
    else:
        if sealer.commitments().get("state") != "SEALED":
            raise ValueError("held-out transaction must begin from SEALED")
        state = None
        transaction = {"identity": identity}

    def transition(next_state: str, **values) -> None:
        nonlocal state, transaction
        transaction = {**transaction, **values}
        workspace.write_transaction_state(
            stage_key, signature, state=next_state, transaction=transaction,
        )
        ledger = recovery.attempt_ledger()
        attempt_id = f"{stage_key}:{identity['selection_digest']}:{next_state}"
        if not any(row["attempt_id"] == attempt_id for row in ledger.rows()):
            ledger.append(
                attempt_id, signature, status=next_state,
                started_at=_isoformat(time.time()), ended_at=_isoformat(time.time()),
                wall_seconds=0.0, gpu_hours=None, state_reference=f"{stage_key}.json",
            )
        state = next_state
        if fail_after == next_state:
            raise RuntimeError(f"injected failure after {next_state}")

    if state is None:
        transition("SEALED")

    # This verifies/reuses the immutable record even after a crash between its
    # file finalization and the matching durable transition.
    selection = finalize_selection_record(selection_record, selection_path)
    if selection["digest"] != identity["selection_digest"]:
        raise ValueError("finalized selection digest changed")
    if state == "SEALED":
        transition("SELECTION_FINALIZED", selection_path=str(selection_path))

    if state == "SELECTION_FINALIZED":
        sealer.authorize(selection_record, authorization_identity)
        transition("AUTHORIZED")

    if state == "AUTHORIZED":
        receipt = sealer.receipt("trained")
        if receipt is None:
            receipt = run_trained_held_out_evaluation(
                manifest_path, model=model, dataset=dataset, scorer=scorer, sealer=sealer,
            )
        transition("TRAINED_RESULT_SEALED", trained_receipt=receipt)

    if state == "TRAINED_RESULT_SEALED":
        run_id = f"reveal-{identity['selection_digest']}"
        bundle_dir = Path(storage.root) / run_id
        if bundle_dir.exists():
            reveal = _result_from_bundle(bundle_dir, identity["selection_digest"])
        else:
            reveal = run_reveal(
                manifest_path, sealer=sealer, selection_record=selection_record,
                storage=storage, telemetry=telemetry, run_id=run_id,
            )
            verify_bundle(Path(reveal.bundle_dir))
        transition("REVEALED", reveal_bundle=reveal.bundle_dir, reveal_checksums=reveal.checksums)

    if state != "REVEALED":
        raise ValueError(f"held-out transaction stopped in unexpected state: {state}")
    reveal = _result_from_bundle(Path(transaction["reveal_bundle"]), identity["selection_digest"])
    return HeldOutRevealTransactionResult(
        state=state, selection_digest=identity["selection_digest"], reveal=reveal, reveal_count=1,
    )
