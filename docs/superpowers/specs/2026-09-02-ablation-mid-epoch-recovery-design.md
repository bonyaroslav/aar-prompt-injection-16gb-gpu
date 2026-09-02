# Ablation Mid-Epoch Recovery Design

**Issue:** #26 — Mid-epoch training recovery for the ablation protocol

## Goal

Allow only a new ablation protocol to resume a training epoch from a durable
optimizer-step boundary. The result of an interrupted-and-resumed CPU toy run
must match the uninterrupted run byte-for-byte, while the frozen Attempt-1
training path and every finalized bundle under `runs/` stay unchanged.

## Scope and compatibility

The implementation is an opt-in module, `runner.ablation_training`; it rejects
the frozen `phase1-2026-08-29` protocol version. `runner.training.run_training`
and its completed-epoch `TrainingRecovery` remain the Attempt-1 implementation
and are not modified. A later ablation manifest or CLI invokes the new module
with its own protocol version and recovery workspace.

## Components and data flow

`MidEpochCheckpointStore` owns a caller-selected directory in a
`RecoveryWorkspace`. Its constructor rejects a root below the evidence root.
It alternates between `slot-a` and `slot-b`: write the complete checkpoint to a
temporary file, flush and fsync it, atomically replace the inactive slot, then
atomically replace a compact pointer document. The pointer includes the stage
signature, completed optimizer-step index, checkpoint digest, and byte count;
the returned measurement is taken after that pointer promotion. Thus a fault
before pointer promotion leaves the prior pointed slot loadable.

The checkpoint payload contains only mutable training state: adapter weights,
optimizer state, scheduler state, CPU RNG state, CUDA RNG state, and the
completed optimizer-step index. Data position is deliberately absent. The
runner calls the injected runtime with a deterministic step index; the runtime
reconstructs epoch order from the seeded shuffle and derives the micro-batch
range from that index.

`run_ablation_epoch` runs one step at a time. It resumes from the durable step
index, calls the runtime's optimizer-safe step method, and saves only after
that method has completed optimizer step, scheduler step, and gradient reset.
The configurable checkpoint interval defaults to 120 steps. Its result reports
whether resume fired plus every save's measured byte count and elapsed time,
which is rendered into the issue #26 decision record by the caller. The runtime
seam uses `capture_mid_epoch_state` and `restore_mid_epoch_state`, avoiding the
Transformers runtime's existing `snapshot` path attribute.

`RealQLoRATrainerAdapter` gains an ablation-only injected-runtime bridge. The
production Transformers runtime uses PEFT adapter state, optimizer/scheduler
state, `torch.get_rng_state()`, and `torch.cuda.get_rng_state_all()`; CUDA is
not required by the recovery core or its tests.

## Failure handling and evidence isolation

A load validates the stored stage signature and payload digest before returning
state. A corrupted or incomplete inactive slot is never considered current.
Recovery payloads are opaque recovery state and are rejected by the existing
finalized-input guard; they are neither copied into nor used to rewrite a
finalized bundle. The decision record documents CPU-measured save latency and
bytes as implementation evidence, and the real ablation runner exposes the
same measurements for its eventual GPU decision record.

## Verification

CPU-only tests use a deterministic NumPy toy adapter through the same
step-runtime seam. They prove exact logical step coverage after an interruption
at each optimizer boundary, full payload round trips, byte-identical final
weights, two-slot crash safety, default/configured interval behavior,
Attempt-1 rejection, resume reporting, and recovery-root isolation. Existing
`tests/test_training.py` and `tests/test_real_seed_run_recovery.py` remain
unchanged and continue to pass.
