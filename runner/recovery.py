"""Shared recovery contract primitives."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import uuid

from runner.bundle import verify_bundle


SIGNATURE_FIELDS = (
    "manifest_digest",
    "protocol_version",
    "upstream_commit",
    "upstream_tree",
    "model_revision",
    "seed",
    "stage",
    "epoch",
    "checkpoint_digest",
    "effective_evaluation_config",
    "expected_example_ids",
)


@dataclass(frozen=True)
class StageSignature:
    _canonical_payload: str
    digest: str

    def __init__(self, payload: dict, digest: str):
        canonical_payload = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        object.__setattr__(self, "_canonical_payload", canonical_payload)
        object.__setattr__(self, "digest", digest)

    @property
    def payload(self) -> dict:
        return json.loads(self._canonical_payload)

    @classmethod
    def create(
        cls,
        *,
        manifest_digest,
        protocol_version,
        upstream_commit,
        upstream_tree,
        model_revision,
        seed,
        stage,
        epoch=None,
        checkpoint_digest=None,
        effective_evaluation_config=None,
        expected_example_ids=None,
    ):
        values = locals()
        payload = {key: values[key] for key in SIGNATURE_FIELDS}
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        return cls(payload=payload, digest="sha256:" + hashlib.sha256(encoded).hexdigest())

    def first_difference(self, other):
        payload = self.payload
        other_payload = other.payload
        return next(
            (key for key in SIGNATURE_FIELDS if payload[key] != other_payload[key]),
            None,
        )


@dataclass(frozen=True)
class StageInspection:
    status: str
    action: str
    differing_field: str | None = None


def _write_json_atomically(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class AttemptLedger:
    def __init__(self, path: Path):
        self.path = Path(path)

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def _claim_path(self, attempt_id: str) -> Path:
        digest = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()
        return self.path.parent / ".attempt-claims" / f"{digest}.claim"

    def _claim(self, attempt_id: str) -> Path:
        claim_path = self._claim_path(attempt_id)
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                claim_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as error:
            raise ValueError(f"attempt identity already recorded: {attempt_id}") from error
        try:
            try:
                os.write(descriptor, attempt_id.encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            claim_path.unlink(missing_ok=True)
            raise
        return claim_path

    def append(
        self,
        attempt_id: str,
        signature: StageSignature,
        *,
        status: str,
        started_at: str,
        ended_at: str | None,
        wall_seconds: float,
        gpu_hours: float | None,
        state_reference: str,
    ) -> None:
        row = {
            "attempt_id": attempt_id,
            "signature_digest": signature.digest,
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "wall_seconds": wall_seconds,
            "gpu_hours": "unavailable" if gpu_hours is None else gpu_hours,
            "state_reference": state_reference,
        }
        if any(existing["attempt_id"] == attempt_id for existing in self.rows()):
            raise ValueError(f"attempt identity already recorded: {attempt_id}")
        claim_path = self._claim(attempt_id)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            claim_path.unlink(missing_ok=True)
            raise


class RecoveryWorkspace:
    def __init__(self, root: Path, evidence_root: Path):
        self.root = Path(root)
        self.evidence_root = Path(evidence_root)
        if self.root.resolve().is_relative_to(self.evidence_root.resolve()):
            raise ValueError("recovery workspace must be outside evidence root")

    def _state_path(self, attempt_id: str) -> Path:
        if not isinstance(attempt_id, str) or attempt_id in {"", ".", ".."}:
            raise ValueError("attempt ID must identify a state file")
        path = self.root / f"{attempt_id}.json"
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError as error:
            raise ValueError("attempt ID must stay inside the recovery workspace") from error
        return path

    def _read_state(self, attempt_id: str) -> dict:
        return json.loads(self._state_path(attempt_id).read_text(encoding="utf-8"))

    def write_state(
        self,
        attempt_id: str,
        signature: StageSignature,
        *,
        status: str,
        recovery_reference: str | None = None,
        completed_bundle: str | None = None,
    ) -> Path:
        path = self._state_path(attempt_id)
        _write_json_atomically(
            path,
            {
                "signature": signature.payload,
                "signature_digest": signature.digest,
                "status": status,
                "recovery_reference": recovery_reference,
                "completed_bundle": (
                    str(completed_bundle) if completed_bundle is not None else None
                ),
            },
        )
        return path

    def inspect_stage(
        self, attempt_id: str, requested_signature: StageSignature
    ) -> StageInspection:
        try:
            record = self._read_state(attempt_id)
            stored = StageSignature.create(**record["signature"])
        except (OSError, ValueError, KeyError, TypeError):
            return StageInspection("unavailable-after-hard-loss", "record-hard-loss")

        if stored.digest != record.get("signature_digest"):
            return StageInspection("incompatible", "diagnose", "signature_digest")

        differing = stored.first_difference(requested_signature)
        if differing:
            return StageInspection("incompatible", "diagnose", differing)
        status = record.get("status")
        if status == "completed":
            try:
                verify_bundle(Path(record.get("completed_bundle")))
            except (OSError, ValueError, TypeError):
                return StageInspection("unavailable-after-hard-loss", "record-hard-loss")
            return StageInspection("completed", "use-finalized-artifact")
        if status == "running":
            return StageInspection("running", "wait-for-safe-boundary")
        if status == "interrupted" and record.get("recovery_reference"):
            return StageInspection(
                "recoverable", f"resume-from:{record['recovery_reference']}"
            )
        if status == "interrupted":
            return StageInspection("interrupted", "restart-stage")
        return StageInspection("unavailable-after-hard-loss", "record-hard-loss")
