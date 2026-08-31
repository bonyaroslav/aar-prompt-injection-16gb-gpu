# Task 3 report: attempt ledger and checksum-gated completion

## Scope

Implemented only Task 3's recovery-core and recovery-test changes:

- `runner/recovery.py`
- `tests/test_recovery.py`

No protocol, evidence, held-out, plan, or Task 4 files were modified.

## Implementation

- Added `AttemptLedger`, an append-only JSONL ledger that records the attempt ID,
  stage-signature digest, lifecycle timing/status values, and state reference.
- Converts unavailable GPU time (`None`) to the explicit durable value
  `"unavailable"`.
- Rejects an attempt ID that already exists in the ledger before appending a second
  record; writes flush and fsync the appended row.
- Preserved checksum-gated completion through `verify_bundle`; a completed state
  whose artifact does not verify is returned as `unavailable-after-hard-loss`.
- Normalized `completed_bundle` to text when recovery state is written so callers
  can supply a `Path` without creating a non-serializable state document.

## TDD evidence

1. Added the ledger and invalid-completed-bundle tests before production changes.
2. Ran the recovery discovery suite with the required Ubuntu WSL virtualenv. The
   red run produced the expected missing-ledger failures:
   `AttributeError: module 'runner.recovery' has no attribute 'AttemptLedger'`.
   It also exposed the required bundle-path serialization gap.
3. Added the smallest ledger implementation and path normalization, then reran the
   recovery discovery suite successfully.

## Verification

Commands run from the Task 3 worktree using
`/mnt/c/Projects/automated_alignment_researcher/.venv/bin/python`:

```text
python -m unittest discover -s tests -p test_recovery.py -v
Ran 13 tests ... OK

python -m unittest discover -s tests -v
Ran 186 tests ... OK
```

The full run emitted one pre-existing PyTorch `pynvml` deprecation warning; it
had no test failures or errors.

## Concern

`AttemptLedger` supplies in-process duplicate exclusion via its lock plus a
durable on-disk duplicate scan. Cross-process locking is outside this Task 3
contract and was not added.

## Round 1 remediation: shared duplicate claim

The original concern was confirmed: two independent `AttemptLedger` instances
could each scan an empty JSONL file and append the same attempt ID.

- Added an exclusive `O_CREAT | O_EXCL` claim file under
  `.attempt-claims/`, keyed by the SHA-256 digest of the attempt ID. The claim
  is therefore shared by all ledger instances targeting the same recovery
  workspace and never uses the raw attempt ID as a path component.
- Existing ledger rows continue to reject a duplicate before a claim is made;
  a new exclusive claim closes the scan-to-append race for concurrently empty
  scans.
- If a claim write, claim close, or ledger append fails, the claim file is
  removed so the failed attempt does not leave an untracked durable claim.
- Added a deterministic concurrent regression: two independent ledger
  instances are synchronized after their initial empty scans, then race to
  append the same attempt. Exactly one succeeds and exactly one JSONL row is
  present. A second test makes the append fail and proves the claim is released
  for a retry.

### Round 1 verification

```text
python -m unittest discover -s tests -p test_recovery.py -v
Ran 15 tests ... OK
```
