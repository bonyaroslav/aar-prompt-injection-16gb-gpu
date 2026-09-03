"""Ablation-only optimizer-step recovery primitives.

The frozen Attempt-1 training runner deliberately remains in ``runner.training``
with completed-epoch recovery. This module stores mutable ablation state only in
the externally selected recovery workspace.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import stat
import time
import uuid

from runner.recovery import RecoveryWorkspace, StageSignature


_STATE_FIELDS = frozenset({
    "adapter_weights",
    "optimizer_state",
    "scheduler_state",
    "cpu_rng_state",
    "cuda_rng_state",
    "step_index",
})


def _fsync_directory(path: Path) -> None:
    """Persist a rename's directory entry where the platform exposes it."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_atomically(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _write_json_atomically(path: Path, document: dict) -> None:
    _write_bytes_atomically(
        path, (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
    )


@dataclass(frozen=True)
class CheckpointMeasurement:
    step_index: int
    byte_count: int
    save_seconds: float


@dataclass(frozen=True)
class AblationEpochResult:
    mid_epoch_resume_fired: bool
    checkpoints: tuple[CheckpointMeasurement, ...]

    @property
    def checkpoint_steps(self) -> list[int]:
        return [measurement.step_index for measurement in self.checkpoints]

    @property
    def recovery_evidence(self) -> dict:
        """Serializable per-epoch facts for the ablation decision record."""
        return {
            "mid_epoch_resume_fired": self.mid_epoch_resume_fired,
            "save_measurements": [
                {
                    "step_index": measurement.step_index,
                    "byte_count": measurement.byte_count,
                    "save_seconds": measurement.save_seconds,
                }
                for measurement in self.checkpoints
            ],
        }


class MidEpochCheckpointStore:
    """Persist one ablation epoch's mutable state in two atomic slots."""

    def __init__(
        self,
        workspace: RecoveryWorkspace,
        checkpoint_id: str,
        signature: StageSignature,
        *,
        clock=time.perf_counter,
    ):
        if not isinstance(checkpoint_id, str) or checkpoint_id in {"", ".", ".."}:
            raise ValueError("checkpoint ID must identify one recovery directory")
        self.workspace = workspace
        self.signature = signature
        self.clock = clock
        workspace.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix" and stat.S_IMODE(workspace.root.stat().st_mode) & 0o077:
            raise ValueError("recovery workspace must be private to use checkpoint state")
        root = workspace.root / f"{checkpoint_id}.mid-epoch"
        try:
            root.resolve().relative_to(workspace.root.resolve())
        except ValueError as error:
            raise ValueError("checkpoint ID must stay inside the recovery workspace") from error
        self.root = root

    def _pointer_path(self) -> Path:
        return self.root / "current.json"

    def _slot_path(self, slot: str) -> Path:
        if slot not in {"a", "b"}:
            raise ValueError("checkpoint slot must be 'a' or 'b'")
        return self.root / f"slot-{slot}.pickle"

    def _pointer(self) -> dict | None:
        path = self._pointer_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("mid-epoch checkpoint pointer is unreadable") from error

    def _validate_state(self, state: dict) -> None:
        if not isinstance(state, dict) or set(state) != _STATE_FIELDS:
            raise ValueError("checkpoint state must contain exactly the required mutable fields")
        if not isinstance(state["step_index"], int) or state["step_index"] < 0:
            raise ValueError("checkpoint step_index must be a non-negative integer")

    def _validate_pointer(self, pointer: dict) -> None:
        try:
            stored = StageSignature.create(**pointer["signature"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("mid-epoch checkpoint signature is invalid") from error
        if stored.digest != pointer.get("signature_digest"):
            raise ValueError("mid-epoch checkpoint signature digest is invalid")
        difference = stored.first_difference(self.signature)
        if difference:
            raise ValueError(f"mid-epoch checkpoint is incompatible on field '{difference}'")

    def save(self, state: dict) -> CheckpointMeasurement:
        """Write the inactive slot, then atomically promote it with the pointer."""
        self._validate_state(state)
        started = self.clock()
        previous = self._pointer()
        active_slot = previous.get("slot") if previous else None
        next_slot = "b" if active_slot == "a" else "a"
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        _write_bytes_atomically(self._slot_path(next_slot), payload)
        _write_json_atomically(
            self._pointer_path(),
            {
                "slot": next_slot,
                "payload_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "signature": self.signature.payload,
                "signature_digest": self.signature.digest,
                "step_index": state["step_index"],
                "byte_count": len(payload),
            },
        )
        return CheckpointMeasurement(
            step_index=state["step_index"],
            byte_count=len(payload),
            save_seconds=self.clock() - started,
        )

    def load(self) -> dict | None:
        pointer = self._pointer()
        if pointer is None:
            return None
        self._validate_pointer(pointer)
        try:
            payload = self._slot_path(pointer["slot"]).read_bytes()
        except (KeyError, OSError, ValueError) as error:
            raise ValueError("current mid-epoch checkpoint slot is unavailable") from error
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if digest != pointer.get("payload_digest"):
            raise ValueError("current mid-epoch checkpoint digest is invalid")
        try:
            state = pickle.loads(payload)
        except (pickle.UnpicklingError, EOFError) as error:
            raise ValueError("current mid-epoch checkpoint payload is invalid") from error
        self._validate_state(state)
        if state["step_index"] != pointer.get("step_index"):
            raise ValueError("current mid-epoch checkpoint step index is invalid")
        return state


def run_ablation_epoch(
    *,
    protocol_version: str,
    runtime,
    total_steps: int,
    checkpoint_store: MidEpochCheckpointStore,
    checkpoint_interval: int = 120,
    on_checkpoint=None,
) -> AblationEpochResult:
    """Run an ablation epoch, checkpointing only after optimizer-safe steps.

    ``runtime.optimizer_safe_step`` must complete its optimizer step, scheduler
    step, and gradient reset before returning. The state carries no data cursor:
    the zero-based step index is the only position supplied back to the
    deterministic runtime when work resumes.
    """
    if protocol_version == "phase1-2026-08-29":
        raise ValueError("mid-epoch recovery is ablation-only")
    if checkpoint_store.signature.payload["protocol_version"] != protocol_version:
        raise ValueError("checkpoint signature protocol does not match the requested ablation protocol")
    if not isinstance(total_steps, int) or total_steps < 0:
        raise ValueError("total_steps must be a non-negative integer")
    if not isinstance(checkpoint_interval, int) or checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be a positive integer")

    restored_state = checkpoint_store.load()
    start_step = 0
    if restored_state is not None:
        start_step = restored_state["step_index"]
        if start_step > total_steps:
            raise ValueError("mid-epoch checkpoint is beyond the requested epoch")
        runtime.restore_mid_epoch_state(restored_state)

    checkpoints = []
    for step_index in range(start_step, total_steps):
        runtime.optimizer_safe_step(step_index)
        completed_steps = step_index + 1
        if completed_steps % checkpoint_interval == 0:
            state = runtime.capture_mid_epoch_state(completed_steps)
            if state.get("step_index") != completed_steps:
                raise ValueError("runtime snapshot must identify the completed optimizer step")
            measurement = checkpoint_store.save(state)
            checkpoints.append(measurement)
            if on_checkpoint is not None:
                on_checkpoint(measurement)

    return AblationEpochResult(
        mid_epoch_resume_fired=restored_state is not None,
        checkpoints=tuple(checkpoints),
    )
