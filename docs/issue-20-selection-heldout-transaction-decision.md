# Issue #20 selection and held-out reveal transaction decision

Makes post-selection held-out evaluation and reveal a single durable,
idempotent transaction without changing `protocol/manifest.json`, modifying a
finalized bundle, or touching a GPU path.

## Decision

`runner.reveal.run_selection_and_reveal` is the sole recovery-aware entry
point.  Its external `HeldOutRevealRecovery` record is bound to the canonical
digest of the already-finalized visible-safety/capability selection record.  It
also binds the selected checkpoint digest, frozen candidate commitment, and
authorization identity.  A retry compares every one of those inputs before it
can reuse state; any change is rejected.

The durable states are, in order:

1. `SEALED` — candidate commitment is present in the restricted sealer.
2. `SELECTION_FINALIZED` — the canonical selection record is immutable.
3. `AUTHORIZED` — the sealer has the same selection digest and authorization
   identity.
4. `TRAINED_RESULT_SEALED` — the existing sealed trained receipt is reused, or
   one trained held-out evaluation has produced it.
5. `REVEALED` — a normal finalized, checksummed combined reveal bundle exists.

The state is atomically written in `RecoveryWorkspace` and each transition has
a stable `AttemptLedger` row.  Recovery after a completed trained receipt never
calls the candidate evaluator again.  Recovery after a completed reveal verifies
and returns the deterministic finalized bundle rather than requesting another
reveal.

## Isolation and evidence boundary

Selection continues to be `runner.selection.select_checkpoint`, the existing
pure function over visible-safety and capability aggregates.  It has no sealer,
dataset, model, or held-out dependency.  The transaction stores only public
metadata in its recovery state: digests, checkpoint digest, authorization
identity, receipt metadata, and bundle checksum metadata.  Candidate IDs,
prompts, responses, secrets, and per-candidate outcomes remain in the
restricted sealer blobs.

`run_reveal` remains the only repository-bound output path.  Its ordinary
bundle topology is unchanged and contains the combined baseline/trained
`valid_only` and `intent_to_evaluate` aggregates plus invalid-classification
counts.  The issue-20 transaction verifies the finalized bundle before marking
`REVEALED` and on every revealed retry.

## Acceptance evidence

| #20 criterion | Verified evidence |
| --- | --- |
| Durable five-state transaction | `HeldOutRevealRecovery`, `RecoveryWorkspace.write_transaction_state`, and `run_selection_and_reveal` use the exact `SEALED` → `REVEALED` sequence. |
| Matching retries / identity rejection | `HeldOutRevealTransactionTests.test_retry_rejects_changed_transaction_identity_inputs` covers selection, checkpoint, candidate commitment, and authorization identity. |
| No second trained evaluation or reveal | `test_failure_after_each_transition_retries_one_logical_evaluation_and_reveal` injects a failure after each state and asserts 200 total trained generations, one reveal call, and one row per transition. |
| Selection / held-out isolation | Existing `SelectCheckpointTests.test_selection_record_contains_manifest_digest_and_never_touches_held_out`; the transaction only receives the finalized record and sealed commitment. |
| Public-artifact contents and topology | Existing `RevealTests.test_reveal_bundle_never_contains_candidate_ids_or_raw_output`; issue-20 retry test runs `verify_bundle` on the final reveal artifact. |
| Regression coverage | Focused selection, reveal, and protocol tests pass; the full test command is recorded in the issue closure result. |

## Scope boundary

This is CPU-only recovery-contract work.  It neither evaluates a real
checkpoint nor starts issue #21.  Partial recovery state stays outside evidence
roots; a pre-existing incomplete or checksum-invalid reveal directory stops the
transaction for review rather than being overwritten.
