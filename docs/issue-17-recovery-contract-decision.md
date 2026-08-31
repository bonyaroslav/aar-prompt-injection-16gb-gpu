# Issue #17 recovery artifact contract decision

The #16 decision fixes recovery boundaries at completed epochs, whole evaluations,
and whole selections. This implementation makes no manifest or protocol change, and
finalized evidence bundles were not modified.

| #17 criterion | Verified evidence |
| --- | --- |
| Required signature fields and mismatch rejection | `StageSignatureTests`; `RecoveryWorkspaceTests.test_mismatched_signature_preserves_original_state` |
| Atomic external recovery state and status actions | `RecoveryWorkspaceTests`; `StageInspectionStatusTests` |
| Unique append-only attempt accounting | `AttemptLedgerTests` |
| Checksummed finalized artifact gate | `CompletedInspectionTests`; `FinalizedInputTests` |
| Same uninterrupted/resumed finalized topology | `FinalizedInputTests.test_uninterrupted_and_resumed_fixture_expose_same_finalized_topology` |
