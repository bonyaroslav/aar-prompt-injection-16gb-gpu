# Ablation Mid-Epoch Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact, double-buffered optimizer-step recovery for the new ablation-only training API.

**Architecture:** `runner.ablation_training` owns the opt-in step runner and a two-slot recovery store outside evidence. It treats a checkpoint as immutable mutable-state bytes plus an atomically promoted pointer. `runner.real_training` supplies an injected-runtime bridge; the frozen Attempt-1 runner remains untouched.

**Tech Stack:** Python standard library (`pickle`, `json`, `os`, `time`, `uuid`), NumPy CPU fixtures, optional PyTorch/PEFT imports confined to real runtime execution, existing `RecoveryWorkspace` and `StageSignature`.

**Spec:** `docs/superpowers/specs/2026-09-02-ablation-mid-epoch-recovery-design.md`

## Global Constraints

- Reject `phase1-2026-08-29`; do not modify `runner.training.run_training`, `tests/test_training.py`, `tests/test_real_seed_run_recovery.py`, frozen manifests, or finalized evidence under `runs/`.
- Checkpoint only after the runtime reports optimizer step, scheduler step, and gradient reset complete; reconstruct data location from the persisted step index.
- Persist adapter weights, optimizer state, scheduler state, CPU/CUDA RNG state, step index, and stage-signature digest outside the evidence root.
- Use temporary-file plus atomic replace and retain the formerly current slot until the new pointer is promoted.
- Each behavioral change begins with a test that fails for the missing contract, then receives the minimum implementation.

---

### Task 1: Double-buffered checkpoint store

**Files:**
- Create: `runner/ablation_training.py`
- Create: `tests/test_ablation_training.py`

**Interfaces:**
- Produces: `MidEpochCheckpointStore(workspace: RecoveryWorkspace, checkpoint_id: str, signature: StageSignature, clock: Callable[[], float] = time.perf_counter)`
- Produces: `save(state: dict) -> CheckpointMeasurement` and `load() -> dict | None`
- Produces: `CheckpointMeasurement(step_index: int, byte_count: int, save_seconds: float)`

- [x] **Step 1: Write failing tests for a complete state round trip and two-slot interrupted save**

```python
def test_load_round_trips_every_required_state_field(self):
    state = fixture_state(step_index=3)
    self.store.save(state)
    self.assertEqual(self.store.load(), state)

def test_failed_new_save_keeps_previous_checkpoint_loadable(self):
    original = fixture_state(step_index=2)
    self.store.save(original)
    with self.assertRaisesRegex(OSError, "injected write fault"):
        self.store.save(fixture_state(step_index=3), fail_before_pointer=True)
    self.assertEqual(self.store.load(), original)
```

- [x] **Step 2: Run the tests and verify they fail because the store is absent**

Run: `python -m unittest tests.test_ablation_training.MidEpochCheckpointStoreTests -v`

Expected: import failure naming `runner.ablation_training` or `MidEpochCheckpointStore`.

- [x] **Step 3: Implement the minimal pickled-state slots and atomic pointer**

```python
def save(self, state):
    self._validate_state(state)
    slot = self._inactive_slot()
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    _write_bytes_atomically(self._slot_path(slot), payload)
    measurement = CheckpointMeasurement(state["step_index"], len(payload), self.clock() - started)
    _write_json_atomically(self._pointer_path(), self._pointer(slot, payload, measurement))
    return measurement
```

- [x] **Step 4: Run the store tests and verify they pass**

Run: `python -m unittest tests.test_ablation_training.MidEpochCheckpointStoreTests -v`

Expected: PASS.

### Task 2: Exact step-resume engine

**Files:**
- Modify: `runner/ablation_training.py`
- Modify: `tests/test_ablation_training.py`

**Interfaces:**
- Produces: `run_ablation_epoch(*, protocol_version: str, runtime, total_steps: int, checkpoint_store: MidEpochCheckpointStore, checkpoint_interval: int = 120) -> AblationEpochResult`
- Consumes runtime methods `restore_mid_epoch_state(state)`, `optimizer_safe_step(step_index)`, and `capture_mid_epoch_state(step_index)`.

- [x] **Step 1: Write failing exact-resume/default-interval/resume-report tests with a CPU NumPy toy runtime**

```python
def test_interrupted_and_resumed_toy_run_has_each_logical_step_once_and_byte_identical_weights(self):
    uninterrupted = run_toy(total_steps=8, interval=1)
    interrupted, store = run_toy(total_steps=8, interval=1, interrupt_after=5)
    resumed = run_toy(total_steps=8, interval=1, store=store)
    self.assertEqual(resumed.executed_step_indexes, list(range(8)))
    self.assertEqual(resumed.weights_bytes, uninterrupted.weights_bytes)
    self.assertTrue(resumed.mid_epoch_resume_fired)

def test_checkpoint_interval_defaults_to_120_and_is_overridable(self):
    self.assertEqual(run_toy(total_steps=121).checkpoint_steps, [120])
    self.assertEqual(run_toy(total_steps=5, interval=2).checkpoint_steps, [2, 4])
```

- [x] **Step 2: Run the engine tests and verify they fail because `run_ablation_epoch` is absent**

Run: `python -m unittest tests.test_ablation_training.AblationEpochRunnerTests -v`

Expected: import failure naming `run_ablation_epoch`.

- [x] **Step 3: Implement the opt-in, post-reset step loop**

```python
if protocol_version == "phase1-2026-08-29":
    raise ValueError("mid-epoch recovery is ablation-only")
state = checkpoint_store.load()
if state is not None:
    runtime.restore(state)
    start_step = state["step_index"]
for step_index in range(start_step, total_steps):
    runtime.optimizer_safe_step(step_index)
    completed_steps = step_index + 1
    if completed_steps % checkpoint_interval == 0:
        checkpoints.append(checkpoint_store.save(runtime.capture_mid_epoch_state(completed_steps)))
```

- [x] **Step 4: Run the engine tests and verify they pass**

Run: `python -m unittest tests.test_ablation_training.AblationEpochRunnerTests -v`

Expected: PASS.

### Task 3: Real injected-runtime bridge and decision record

**Files:**
- Modify: `runner/real_training.py`
- Modify: `tests/test_real_training.py`
- Create: `docs/issue-26-mid-epoch-training-recovery-decision.md`

**Interfaces:**
- Produces: `RealQLoRATrainerAdapter.run_ablation_epoch(...) -> AblationEpochResult`
- Consumes the new runtime seam and exposes save measurements for evidence.

- [x] **Step 1: Write a failing adapter test using the existing `RecordingRuntime` seam**

```python
def test_ablation_bridge_uses_the_injected_runtime_and_reports_save_measurements(self):
    result = self.trainer.run_ablation_epoch(protocol_version="ablation-v1", epoch=1,
        sequence_length=2048, config=self.training, checkpoint_store=self.store, total_steps=2)
    self.assertEqual(result.checkpoint_steps, [2])
    self.assertGreater(result.checkpoints[0].byte_count, 0)
```

- [x] **Step 2: Run the adapter test and verify it fails because the bridge is absent**

Run: `python -m unittest tests.test_real_training.RealQLoRATrainerAdapterTests.test_ablation_bridge_uses_the_injected_runtime_and_reports_save_measurements -v`

Expected: `AttributeError` naming `run_ablation_epoch`.

- [x] **Step 3: Implement the bridge and document measured evidence**

The real runtime captures and restores PEFT adapter state, optimizer/scheduler
state, CPU RNG, CUDA RNG, and step index. Add the decision record describing
the CPU test’s measured save bytes/latency and the emitted per-save measurement
fields that a real ablation run records; do not claim a GPU timing was measured
without a GPU run.

- [x] **Step 4: Run targeted tests and verify they pass**

Run: `python -m unittest tests.test_ablation_training tests.test_real_training -v`

Expected: PASS.

### Task 4: Issue acceptance verification and delivery

**Files:**
- Modify: `runner/ablation_training.py`
- Modify: `runner/real_training.py`
- Modify: `tests/test_ablation_training.py`
- Modify: `tests/test_real_training.py`
- Create: `docs/issue-26-mid-epoch-training-recovery-decision.md`

- [ ] **Step 1: Verify untouched Attempt-1 recovery tests**

Run: `python -m unittest tests.test_training tests.test_real_seed_run_recovery -v`

Expected: PASS without edits to either file.

- [ ] **Step 2: Verify the complete suite and every acceptance criterion**

Run: `python -m unittest discover -s tests -v`

Expected: exit code 0. Check the acceptance checklist against the tests,
decision record, `git diff --name-only`, and `git diff --check`.

- [ ] **Step 3: Commit and push only after fast-forward safety is proven**

Run: `git fetch origin; git merge-base --is-ancestor HEAD origin/master; git status --short; git add runner/ablation_training.py runner/real_training.py tests/test_ablation_training.py tests/test_real_training.py docs/issue-26-mid-epoch-training-recovery-decision.md docs/superpowers/specs/2026-09-02-ablation-mid-epoch-recovery-design.md docs/superpowers/plans/2026-09-02-ablation-mid-epoch-recovery.md; git commit -m "feat: add ablation mid-epoch recovery"; git push origin master`

Expected: the local branch is based on `origin/master`, only issue #26 paths are staged, and push fast-forwards `master`.
