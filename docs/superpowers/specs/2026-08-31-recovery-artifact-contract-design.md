# Recovery and Finalized-Artifact Contract Design

**Issue:** #17 — Define the recovery and finalized-artifact contract

## Goal

Provide a CPU-only, reusable recovery core that can identify compatible partial
work, preserve incompatible state, account for every attempt, and expose only
checksummed completed artifacts to later stages.

## Scope and boundaries

This change adds the shared contract only. It does not alter the frozen
manifest, rerun evidence, access held-out data, change existing stage
orchestration, or implement the stage-specific training, evaluation, and
held-out transactions assigned to #18–#20.

Recovery records live beneath a caller-selected workspace outside the evidence
root. Existing finalized bundles remain immutable and are validated with
`runner.bundle.verify_bundle` rather than copied or rewritten.

## Components

### `StageSignature`

An immutable canonical JSON value whose SHA-256 digest identifies a recoverable
stage. Its required fields are: manifest/protocol digest, upstream commit and
tree, model revision, seed, stage type, epoch/checkpoint identity, effective
evaluation configuration, and expected example IDs. Canonical serialization
uses sorted keys and compact JSON so equivalent inputs have the same digest.

### `RecoveryWorkspace`

Owns one durable state file per attempt. State writes use a temporary file in
the destination directory followed by `os.replace`, so readers see either the
previous complete document or the new complete document. Every state includes
the signature payload and digest, status, recovery reference, and an optional
completed-bundle path.

Existing state with a different signature is never overwritten: inspection
returns `incompatible`, reports the first differing signature field, and leaves
the original record available for diagnosis.

### `AttemptLedger`

Records one JSON object per unique attempt identity in an append-only ledger
under the recovery workspace. Each row records its signature digest, start/end
timestamps, status, wall seconds, GPU hours (or an explicit `unavailable`
value), and state-file reference. A process-local lock plus append-and-flush
write makes an attempt record durable before its state can be promoted.

### `inspect_stage`

Returns a compact status and continuation action:

- `completed` only when its completed bundle passes `verify_bundle`.
- `running` for a durable active state.
- `interrupted` for a stopped attempt with no resumable boundary.
- `recoverable` for a compatible state at a recorded safe boundary.
- `incompatible` for a signature mismatch.
- `unavailable-after-hard-loss` when the durable record says partial state was
  lost and no recovery reference exists.

No status implicitly launches or resumes work.

### Finalized-input guard

A narrow helper accepts only bundle paths that verify cleanly and rejects any
path under the recovery workspace or any incomplete/non-checksummed path. It
is intentionally not a replacement for #14's final analysis discovery.

## Data flow

1. A later stage constructs and stores a `StageSignature` before work begins.
2. It creates an attempt-ledger entry and atomically records its safe-boundary
   state outside the evidence root.
3. On restart, `inspect_stage` compares the requested signature to the durable
   record and exposes the permitted continuation action.
4. When a stage has a normal finalized bundle, the guard verifies its checksum
   before exposing it as an analysis input; recovery files remain invisible.

## Failure handling

Malformed recovery JSON, a missing completed bundle, a checksum failure, or a
signature mismatch does not trigger a retry or overwrite. Inspection reports
the state as incompatible or unavailable, preserves the record, and leaves the
caller to decide its next authorized action.

## Verification

`tests/test_recovery.py` will use temporary recovery and evidence roots. It
will prove canonical signatures, mismatch preservation, atomic state replacement,
attempt accounting, checksum-gated completion, and equal finalized logical
topology for a synthetic uninterrupted and resumed seed. The fixture includes
training, three evaluations, selection, reveal, resources, and checksums while
proving recovery records are excluded.
