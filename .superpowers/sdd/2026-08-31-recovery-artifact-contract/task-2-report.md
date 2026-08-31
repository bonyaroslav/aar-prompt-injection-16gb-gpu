# Task 2 report: atomic recovery records and status inspection

## Delivered scope

- Added `RecoveryWorkspace` with a recovery root guard, one JSON state file per
  attempt, and temporary-file-plus-`os.replace` persistence.
- Added `StageInspection` and recovery inspection outcomes for compatible safe
  boundaries, running work, interrupted work, incompatible signatures, completed
  bundles, and hard-loss diagnostics.
- Stored signatures are reconstructed with `StageSignature.create(**record["signature"])`.
  Inspection compares that recomputed digest to the recorded digest before any
  status decision; a mismatch returns `incompatible` / `diagnose` with
  `signature_digest` as the differing field.
- Added focused tests for recovery/resume behavior, signature mismatch state
  preservation, digest tampering, malformed records, and the evidence-root guard.

## TDD evidence

1. The new recovery tests initially failed because `RecoveryWorkspace` did not
   exist.
2. The malformed-record regression then failed with `KeyError: 'status'` before
   the missing-status path was changed to return a hard-loss diagnostic.
3. The focused recovery suite passed after each minimal implementation change.

## Verification

Ran in Ubuntu WSL using the required interpreter:

```text
/mnt/c/Projects/automated_alignment_researcher/.venv/bin/python -m unittest discover -s tests -p test_recovery.py -v
Ran 8 tests in 0.003s
OK
```

`git diff --check` reported no whitespace errors. The brief's exact
`python -m unittest tests.test_recovery.RecoveryWorkspaceTests -v` command
cannot import this repository's `tests` directory because it has no
`tests/__init__.py`; unittest discovery runs the same focused file without
altering unrelated test-package structure.

## Scope confirmation

Only `runner/recovery.py`, `tests/test_recovery.py`, and this Task-2 report were
changed. No protocol, evidence, held-out, or plan artifacts were modified.

## Fix round 1: review findings

- `_state_path` now resolves each candidate state path and requires it to remain
  beneath the resolved recovery root. Empty, `.` and `..` attempt identifiers
  are also rejected. This prevents a traversal identifier such as
  `../evidence/attempt` from creating or replacing a state file under the
  finalized evidence root.
- Added a traversal regression that asserts the identifier is rejected and the
  target evidence path was not written.
- Added a replacement regression that writes an existing attempt again, reads
  the resulting JSON as a complete document, verifies no prior bundle value
  remains, and confirms the replacement is recoverable from its new boundary.

### TDD and verification evidence

The traversal regression first failed with `AssertionError: ValueError not
raised`, demonstrating the existing write escaped the recovery workspace. The
replacement regression was already green against the existing
temporary-file-plus-`os.replace` implementation. After the containment change,
the focused Ubuntu WSL suite passed:

```text
/mnt/c/Projects/automated_alignment_researcher/.venv/bin/python -m unittest discover -s tests -p test_recovery.py -v
Ran 10 tests in 0.005s
OK
```

Self-review confirmed the resolved-path guard executes before parent-directory
creation and therefore before any evidence-root write. `git diff --check`
reported no whitespace errors.
