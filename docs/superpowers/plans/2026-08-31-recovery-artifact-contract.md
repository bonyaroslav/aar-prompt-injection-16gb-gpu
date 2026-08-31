# Recovery and Finalized-Artifact Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the issue #17 CPU-only recovery contract while keeping recovery state outside immutable, finalized evidence.

**Architecture:** A new `runner.recovery` module owns canonical stage signatures, atomic state documents, a per-workspace attempt ledger, status inspection, and a narrow finalized-input guard. Existing training, evaluation, selection, and reveal entrypoints do not change; later issues integrate this core at their stage boundaries.

**Tech Stack:** Python standard library (`dataclasses`, `hashlib`, `json`, `os`, `threading`, `uuid`), existing `runner.bundle.verify_bundle`, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-recovery-artifact-contract-design.md`

## Global Constraints

- Do not modify `protocol/manifest.json`, `RESEARCH_PLAN.md`, finalized evidence, or held-out isolation.
- Recovery workspace paths must be outside the final evidence root; recovery files must never be treated as analysis inputs.
- No GPU work, model loading, network calls, or protocol changes.
- Write production code only after its associated test has failed for the intended missing behavior.

---

### Task 1: Canonical stage signatures

**Files:**
- Create: `runner/recovery.py`
- Create: `tests/test_recovery.py`

**Interfaces:**
- Produces: `StageSignature.create(*, manifest_digest, protocol_version, upstream_commit, upstream_tree, model_revision, seed, stage, epoch=None, checkpoint_digest=None, effective_evaluation_config=None, expected_example_ids=None) -> StageSignature`
- Produces: `StageSignature.digest: str`, `StageSignature.payload: dict`, `StageSignature.first_difference(other: StageSignature) -> str | None`

- [x] **Step 1: Write the failing signature tests**

```python
class StageSignatureTests(unittest.TestCase):
    def _signature(self, **changes):
        values = {
            "manifest_digest": "sha256:manifest", "protocol_version": "phase1-2026-08-29",
            "upstream_commit": "a" * 40, "upstream_tree": "b" * 40,
            "model_revision": "c" * 40, "seed": 17, "stage": "evaluation",
            "epoch": 1, "checkpoint_digest": "sha256:checkpoint",
            "effective_evaluation_config": {"batch_size": 32},
            "expected_example_ids": ["visible:0001", "visible:0002"],
        }
        values.update(changes)
        return StageSignature.create(**values)

    def test_equal_inputs_produce_equal_canonical_digest(self):
        self.assertEqual(self._signature().digest, self._signature().digest)

    def test_first_difference_names_changed_signature_field(self):
        self.assertEqual(self._signature().first_difference(self._signature(seed=42)), "seed")
```

- [x] **Step 2: Run the signature tests and verify they fail because `StageSignature` is absent**

Run: `python -m unittest tests.test_recovery.StageSignatureTests -v`

Expected: Import failure naming `runner.recovery` or `StageSignature`.

- [x] **Step 3: Implement the immutable canonical signature**

```python
@dataclass(frozen=True)
class StageSignature:
    payload: dict
    digest: str

    @classmethod
    def create(cls, **values):
        payload = {key: values[key] for key in SIGNATURE_FIELDS}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return cls(payload=payload, digest="sha256:" + hashlib.sha256(encoded).hexdigest())

    def first_difference(self, other):
        return next((key for key in SIGNATURE_FIELDS if self.payload[key] != other.payload[key]), None)
```

- [x] **Step 4: Run the signature tests and verify they pass**

Run: `python -m unittest tests.test_recovery.StageSignatureTests -v`

Expected: 2 tests pass.

### Task 2: Atomic recovery records and status inspection

**Files:**
- Modify: `runner/recovery.py`
- Modify: `tests/test_recovery.py`

**Interfaces:**
- Consumes: `StageSignature`
- Produces: `RecoveryWorkspace(root: Path, evidence_root: Path)`
- Produces: `RecoveryWorkspace.write_state(attempt_id, signature, *, status, recovery_reference=None, completed_bundle=None) -> Path`
- Produces: `RecoveryWorkspace.inspect_stage(attempt_id, requested_signature) -> StageInspection`

- [x] **Step 1: Write failing atomic-state and mismatch tests**

```python
def test_compatible_safe_boundary_is_recoverable(self):
    workspace = RecoveryWorkspace(self.recovery_root, self.evidence_root)
    workspace.write_state("attempt-1", self.signature, status="interrupted", recovery_reference="epoch-1")
    inspection = workspace.inspect_stage("attempt-1", self.signature)
    self.assertEqual((inspection.status, inspection.action), ("recoverable", "resume-from:epoch-1"))

def test_mismatched_signature_preserves_original_state(self):
    workspace = RecoveryWorkspace(self.recovery_root, self.evidence_root)
    path = workspace.write_state("attempt-1", self.signature, status="interrupted", recovery_reference="epoch-1")
    inspection = workspace.inspect_stage("attempt-1", self.signature_for(seed=42))
    self.assertEqual(inspection.status, "incompatible")
    self.assertEqual(inspection.differing_field, "seed")
    self.assertEqual(json.loads(path.read_text())["signature_digest"], self.signature.digest)

def test_rejects_recovery_root_inside_finalized_evidence_root(self):
    with self.assertRaisesRegex(ValueError, "outside evidence root"):
        RecoveryWorkspace(self.evidence_root / "recovery", self.evidence_root)
```

- [x] **Step 2: Run the recovery-state tests and verify they fail because workspace methods are absent**

Run: `python -m unittest tests.test_recovery.RecoveryWorkspaceTests -v`

Expected: failure naming `RecoveryWorkspace` or its missing method.

- [x] **Step 3: Implement temp-file-plus-replace state persistence and inspection**

```python
def _write_json_atomically(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)

def inspect_stage(self, attempt_id, requested_signature):
    record = self._read_state(attempt_id)
    stored = StageSignature(payload=record["signature"], digest=record["signature_digest"])
    differing = stored.first_difference(requested_signature)
    if differing:
        return StageInspection("incompatible", "diagnose", differing)
    if record["status"] == "completed":
        try:
            verify_bundle(Path(record["completed_bundle"]))
        except (OSError, ValueError):
            return StageInspection("unavailable-after-hard-loss", "record-hard-loss")
        return StageInspection("completed", "use-finalized-artifact")
    if record["status"] == "running":
        return StageInspection("running", "wait-for-safe-boundary")
    if record["status"] == "interrupted" and record.get("recovery_reference"):
        return StageInspection("recoverable", f"resume-from:{record['recovery_reference']}")
    if record["status"] == "interrupted":
        return StageInspection("interrupted", "restart-stage")
    return StageInspection("unavailable-after-hard-loss", "record-hard-loss")
```

- [x] **Step 4: Run the recovery-state tests and verify they pass**

Run: `python -m unittest tests.test_recovery.RecoveryWorkspaceTests -v`

Expected: all `RecoveryWorkspaceTests` pass.

### Task 3: Attempt ledger and checksum-gated completion

**Files:**
- Modify: `runner/recovery.py`
- Modify: `tests/test_recovery.py`

**Interfaces:**
- Consumes: `RecoveryWorkspace`, existing `runner.bundle.verify_bundle`
- Produces: `AttemptLedger.append(attempt_id, signature, *, status, started_at, ended_at, wall_seconds, gpu_hours, state_reference) -> None`
- Produces: `RecoveryWorkspace.inspect_stage(...) -> StageInspection` with `completed` only for a verified bundle.

- [x] **Step 1: Write failing ledger and checksum-gating tests**

```python
def test_ledger_preserves_unavailable_gpu_time_and_attempt_identity(self):
    ledger = AttemptLedger(self.recovery_root / "attempts.jsonl")
    ledger.append("attempt-1", self.signature, status="interrupted", started_at="2026-08-31T10:00:00Z",
                  ended_at="2026-08-31T10:05:00Z", wall_seconds=300.0, gpu_hours=None,
                  state_reference="states/attempt-1.json")
    row = json.loads((self.recovery_root / "attempts.jsonl").read_text().strip())
    self.assertEqual((row["attempt_id"], row["gpu_hours"]), ("attempt-1", "unavailable"))

def test_ledger_rejects_duplicate_attempt_identity(self):
    ledger = AttemptLedger(self.recovery_root / "attempts.jsonl")
    kwargs = dict(status="running", started_at="2026-08-31T10:00:00Z", ended_at=None,
                  wall_seconds=0.0, gpu_hours=None, state_reference="states/attempt-1.json")
    ledger.append("attempt-1", self.signature, **kwargs)
    with self.assertRaisesRegex(ValueError, "attempt identity already recorded"):
        ledger.append("attempt-1", self.signature, **kwargs)

def test_completed_state_with_invalid_bundle_is_not_completed(self):
    workspace = RecoveryWorkspace(self.recovery_root, self.evidence_root)
    workspace.write_state("attempt-1", self.signature, status="completed", completed_bundle=self.evidence_root / "bad")
    self.assertEqual(workspace.inspect_stage("attempt-1", self.signature).status, "unavailable-after-hard-loss")
```

- [x] **Step 2: Run the ledger tests and verify they fail because ledger/checksum behavior is absent**

Run: `python -m unittest tests.test_recovery.AttemptLedgerTests tests.test_recovery.CompletedInspectionTests -v`

Expected: failure naming `AttemptLedger` or returning a non-compliant completed status.

- [x] **Step 3: Implement append-and-flush ledger writes and checksum validation**

```python
def append(self, attempt_id, signature, *, status, started_at, ended_at, wall_seconds, gpu_hours, state_reference):
    if any(row["attempt_id"] == attempt_id for row in self.rows()):
        raise ValueError(f"attempt identity already recorded: {attempt_id}")
    row = {"attempt_id": attempt_id, "signature_digest": signature.digest, "status": status,
           "started_at": started_at, "ended_at": ended_at, "wall_seconds": wall_seconds,
           "gpu_hours": "unavailable" if gpu_hours is None else gpu_hours,
           "state_reference": state_reference}
    with self._lock, self.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
```

- [x] **Step 4: Run the ledger tests and verify they pass**

Run: `python -m unittest tests.test_recovery.AttemptLedgerTests tests.test_recovery.CompletedInspectionTests -v`

Expected: all targeted tests pass.

### Task 4: Finalized-input guard and topology fixture

**Files:**
- Modify: `runner/recovery.py`
- Modify: `tests/test_recovery.py`
- Create: `docs/issue-17-recovery-contract-decision.md`

**Interfaces:**
- Produces: `finalized_inputs_only(paths: Iterable[Path], recovery_root: Path) -> list[Path]`
- Consumes: finalized bundles with `checksums.sha256`; the fixture uses the existing bundle writer/finalizer.

- [x] **Step 1: Write failing guard and uninterrupted/resumed-topology tests**

```python
def test_finalized_inputs_reject_recovery_and_non_checksummed_paths(self):
    with self.assertRaisesRegex(ValueError, "recovery workspace"):
        finalized_inputs_only([self.recovery_root / "states" / "attempt-1.json"], self.recovery_root)

def test_uninterrupted_and_resumed_fixture_expose_same_finalized_topology(self):
    uninterrupted = self._complete_seed_fixture("uninterrupted")
    resumed = self._complete_seed_fixture("resumed", interrupted_after="eval-1")
    self.assertEqual(self._topology(uninterrupted), self._topology(resumed))
    self.assertEqual(self._topology(uninterrupted), {"training", "eval-1", "eval-2", "eval-3", "selection", "reveal", "resources"})
```

- [x] **Step 2: Run the guard/topology tests and verify they fail because the guard is absent**

Run: `python -m unittest tests.test_recovery.FinalizedInputTests -v`

Expected: import or assertion failure naming `finalized_inputs_only`.

- [x] **Step 3: Implement the narrow guard and document the acceptance evidence**

```python
def finalized_inputs_only(paths, recovery_root):
    recovery_root = Path(recovery_root).resolve()
    accepted = []
    for candidate in map(Path, paths):
        resolved = candidate.resolve()
        if resolved == recovery_root or recovery_root in resolved.parents:
            raise ValueError(f"recovery workspace is not a finalized input: {candidate}")
        verify_bundle(resolved)
        accepted.append(resolved)
    return accepted
```

Write `docs/issue-17-recovery-contract-decision.md` with this evidence table:

```markdown
| #17 criterion | Verified evidence |
| --- | --- |
| Required signature fields and mismatch rejection | `StageSignatureTests`; `RecoveryWorkspaceTests.test_mismatched_signature_preserves_original_state` |
| Atomic external recovery state and status actions | `RecoveryWorkspaceTests`; `StageInspectionStatusTests` |
| Unique append-only attempt accounting | `AttemptLedgerTests` |
| Checksummed finalized artifact gate | `CompletedInspectionTests`; `FinalizedInputTests` |
| Same uninterrupted/resumed finalized topology | `FinalizedInputTests.test_uninterrupted_and_resumed_fixture_expose_same_finalized_topology` |
```

State that the #16 decision fixes completed-epoch, whole-evaluation, and whole-selection recovery boundaries; that this implementation makes no manifest or protocol change; and that finalized evidence bundles were not modified.

- [x] **Step 4: Run targeted recovery tests and verify they pass**

Run: `python -m unittest tests.test_recovery -v`

Expected: all recovery tests pass.

- [x] **Step 5: Add explicit status coverage**

```python
def test_inspection_exposes_each_required_status(self):
    expected = {"completed", "running", "interrupted", "recoverable", "incompatible", "unavailable-after-hard-loss"}
    self.assertEqual(set(self._inspect_fixture_statuses()), expected)
```

Run: `python -m unittest tests.test_recovery.StageInspectionStatusTests -v`

Expected: every required status is exercised with a defined continuation action.

### Task 5: Final verification and issue-scoped commit

**Files:**
- Modify: `runner/recovery.py`
- Modify: `tests/test_recovery.py`
- Create: `docs/issue-17-recovery-contract-decision.md`

- [x] **Step 1: Run the full test suite**

Run: `python -m unittest discover -s tests -v`

Expected: exit code 0 with no failures or errors.

- [x] **Step 2: Validate the issue-only diff and canonical document**

Run: `git diff --check HEAD; git diff --name-only HEAD`

Expected: no whitespace errors and only issue #17 paths beyond the committed design document.

- [x] **Step 3: Commit the implementation with explicit paths**

Run: `git add runner/recovery.py tests/test_recovery.py docs/issue-17-recovery-contract-decision.md && git commit -m "feat: add issue 17 recovery contract"`

Expected: one issue-scoped commit; do not push.
