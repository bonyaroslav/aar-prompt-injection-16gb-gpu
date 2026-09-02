# Issue #26 mid-epoch training recovery decision

## Scope

This implementation is for a new ablation protocol only. The entrypoint rejects
the frozen Attempt-1 version, `phase1-2026-08-29`; `runner.training.run_training`
continues to recover only completed epochs. No finalized evidence under `runs/`
was read, changed, re-checksummed, or promoted by this work.

## Recovery boundary

The ablation runner checkpoints only after `optimizer_safe_step` has completed
the optimizer update, scheduler update, and gradient reset. The state contains
adapter weights, optimizer state, scheduler state, CPU RNG, CUDA RNG, and the
completed optimizer-step index. It intentionally does not contain a data
cursor: the runtime recreates its seeded epoch shuffle and derives the relevant
micro-batches from that step index.

The state store is in the external recovery workspace. It alternates two slots,
writes and fsyncs the new inactive slot before atomically promoting its pointer,
and leaves the old pointed slot intact if promotion fails. Recovery bytes stay
outside the evidence root and remain ineligible as finalized inputs.

## Save measurement

The CPU fixture ran on 2026-09-02 against the real two-slot file store with the
complete required-state envelope at optimizer step 3. It recorded **4,057 bytes**
and **0.012087700 seconds** for one save, including atomic pointer promotion.
This is measured implementation evidence,
not a prediction and not a GPU timing. It does not establish target-model GPU
overhead.

Each ablation epoch returns `recovery_evidence`, including
`mid_epoch_resume_fired` and a record for every save with `step_index`,
`byte_count`, and `save_seconds`. The actual ablation execution must preserve
those values in its final decision record. If a real save exceeds about 30
seconds, increase the configured interval instead of accepting the overhead.

## Acceptance evidence

| Requirement | Evidence |
| --- | --- |
| Exact no-skip/no-repeat CPU resume | `AblationEpochRunnerTests.test_interrupted_and_resumed_toy_run_executes_each_logical_step_once` |
| Byte-identical toy weights | Same test compares uninterrupted/resumed NumPy weight bytes |
| Full state round trip | `MidEpochCheckpointStoreTests.test_load_round_trips_every_required_mutable_state_field` |
| Torn-save safety | `MidEpochCheckpointStoreTests.test_interrupted_new_save_keeps_previous_checkpoint_loadable` |
| Default and configurable interval | `AblationEpochRunnerTests.test_checkpoint_interval_defaults_to_120_and_is_overridable` |
| Attempt-1 isolation | `AblationEpochRunnerTests.test_frozen_attempt_one_protocol_is_rejected`; untouched Attempt-1 suites |
| Recovery/evidence isolation | `MidEpochCheckpointStoreTests.test_recovery_checkpoint_is_not_accepted_as_finalized_evidence` |
| Real adapter state and injected seam | `RuntimeOOMTranslationTests.test_mid_epoch_state_captures_and_restores_adapter_optimizer_scheduler_and_rng`; `RealQLoRATrainerAdapterTests.test_ablation_bridge_uses_injected_runtime_and_reports_save_measurements` |
