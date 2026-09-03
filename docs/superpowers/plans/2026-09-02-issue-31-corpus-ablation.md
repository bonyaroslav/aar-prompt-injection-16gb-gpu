# Issue #31 Corpus Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a separately versioned, recovery-tested corpus ablation that replaces the 2,000 prompt-injection examples with 2,000 clean-control examples while preserving the frozen Attempt-1 train/evaluation configuration.

**Architecture:** `training_data.build` gains an explicit, default-preserving Dolly reservation parameter and real-build provenance reporting.  A new `protocol/ablation` manifest binds the ablation to the frozen manifest digest, and `runner.corpus_ablation` owns outputs under `ablation/`, the external mid-epoch recovery workspace, non-selection evaluation, resource accounting, and the appendix report.

**Tech Stack:** Python 3.12, unittest, PyTorch/CUDA under WSL2, Hugging Face datasets/Transformers/PEFT, existing `runner` bundles and recovery primitives.

**Spec:** GitHub issue #31 and `docs/handover/issue-31-handover.md`

## Global Constraints

- Use seed `42`; do not alter adapter-initialization ordering or a frozen Attempt-1 value.
- Use the re-cached public ADR 0001 source datasets offline; do not use a token, model-generated data, paid API, or InjecAgent.
- Outputs live only under gitignored `ablation/`; recovery state lives outside that evidence root.
- Preserve Attempt-1 bundles, selection, frozen bootstrap seed `271828`, and held-out data unchanged.
- Run the full test suite before the real launch; smoke first; launch detached; never push.
- Count all ablation GPU time against the combined 72-hour ledger and stop before exceeding it.

---

### Task 1: Default-preserving corpus construction and provenance

**Files:**
- Modify: `training_data/build.py`
- Modify: `tests/test_training_data.py`

**Interfaces:**
- Consumes: `build_dataset` with fetched-or-fixture source-row inputs and optional per-category targets.
- Produces: `build_dataset` and `run_real_build` with `dolly_oversample_factor: int = 3`, plus a deterministic Dolly-row digest in the report.

- [ ] **Step 1: Write the failing tests**

```python
def test_default_dolly_oversample_is_byte_identical_to_the_legacy_build(self):
    targets = {"prompt_injection": 2, "clean_control": 3, "ambiguous_boundary": 2, "refusal_calibration": 2}
    kwargs = {"injection_raw_rows": self._injection_rows(4), "dolly_rows": self._dolly_rows(30),
              "exclusion_exact_keys": set(), "exclusion_near_keys": set(), "token_cap": TOKEN_CAP, "targets": targets}
    legacy = build_dataset(**kwargs)
    explicit = build_dataset(**kwargs, dolly_oversample_factor=3)
    self.assertEqual([x.to_record() for x in explicit["examples"]], [x.to_record() for x in legacy["examples"]])
    self.assertEqual(explicit["report"], legacy["report"])

def test_clean_only_ablation_fixture_has_5000_examples_and_no_injection(self):
    result = build_dataset(injection_raw_rows=[], dolly_rows=self._dolly_rows(12000),
                           exclusion_exact_keys=set(), exclusion_near_keys=set(), token_cap=TOKEN_CAP,
                           targets={"prompt_injection": 0, "clean_control": 3500,
                                    "ambiguous_boundary": 1000, "refusal_calibration": 500},
                           dolly_oversample_factor=1)
    self.assertEqual(result["report"]["total"], 5000)
    self.assertEqual(result["report"]["shortfalls"], {})
    self.assertFalse(any(x.category == "prompt_injection" for x in result["examples"]))
```

- [ ] **Step 2: Run the focused test module to verify the new tests fail**

Run: `.venv/Scripts/python.exe -m unittest tests.test_training_data -q`

Expected: failure because `dolly_oversample_factor` is not an accepted keyword.

- [ ] **Step 3: Implement the smallest default-preserving parameterization**

```python
def build_dataset(*, injection_raw_rows, dolly_rows, exclusion_exact_keys, exclusion_near_keys,
                  token_cap: int, targets: dict[str, int] | None = None,
                  dolly_oversample_factor: int = _DOLLY_OVERSAMPLE_FACTOR):
    if not isinstance(dolly_oversample_factor, int) or dolly_oversample_factor <= 0:
        raise ValueError("dolly_oversample_factor must be a positive integer")
    clean_pool_size = min(len(shuffled_dolly), targets["clean_control"] * dolly_oversample_factor)
```

Pass the parameter through `run_real_build`, add the sorted canonical Dolly-row digest to `report.json`, and retain the existing default call path unchanged.

- [ ] **Step 4: Re-run the focused module to verify green**

Run: `.venv/Scripts/python.exe -m unittest tests.test_training_data -q`

Expected: all tests pass.

### Task 2: Fails-closed ablation protocol identity

**Files:**
- Create: `protocol/ablation/corpus-ablation-2026-09-02.json`
- Create: `protocol/ablation/manifest.py`
- Create: `protocol/ablation/digests.md`
- Modify: `.gitignore`
- Modify: `.gitattributes`
- Create: `tests/test_ablation_protocol.py`

**Interfaces:**
- Consumes: frozen `protocol/manifest.json` and its canonical digest.
- Produces: `protocol.ablation.manifest.load(path) -> dict`, which rejects a missing required block or a changed frozen-manifest digest.

- [ ] **Step 1: Write the failing manifest tests**

```python
def test_ablation_manifest_is_downstream_of_the_frozen_manifest(self):
    data = ablation.load(MANIFEST)
    self.assertEqual(data["ablation_version"], "ablation-corpus-2026-09-02")
    self.assertEqual(data["training"]["seed"], 42)
    self.assertEqual(data["corpus"]["targets"]["prompt_injection"], 0)
    self.assertEqual(data["corpus"]["targets"]["clean_control"], 3500)

def test_load_rejects_changed_frozen_manifest_identity(self):
    altered = json.loads(MANIFEST.read_text())
    altered["downstream_of"]["canonical_manifest_digest"] = "0" * 64
    with self.assertRaisesRegex(ablation.AblationManifestError, "no longer matches"):
        ablation.load(write_temp(altered))
```

- [ ] **Step 2: Run the focused test to verify red**

Run: `.venv/Scripts/python.exe -m unittest tests.test_ablation_protocol -q`

Expected: import failure because `protocol.ablation` does not exist.

- [ ] **Step 3: Implement the manifest loader and committed manifest**

```python
def load(path: str | Path, *, frozen_manifest_path: str | Path = FROZEN_MANIFEST_PATH) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [key for key in _REQUIRED if key not in data]
    if missing:
        raise AblationManifestError(f"ablation manifest is missing required keys: {missing}")
    if data["downstream_of"]["canonical_manifest_digest"] != frozen_canonical_digest(frozen_manifest_path):
        raise AblationManifestError("frozen Attempt-1 manifest no longer matches ablation provenance")
    return data
```

Declare the corpus targets, lower oversample factor, frozen baseline-reuse justification, boundaries, recovery interval, distinct bootstrap seed, and resource budget. Pin `protocol/ablation/**` to LF and ignore top-level `ablation/` output.

- [ ] **Step 4: Re-run the focused test to verify green**

Run: `.venv/Scripts/python.exe -m unittest tests.test_ablation_protocol -q`

Expected: all tests pass and `digests.md` contains both manifest identities.

### Task 3: Separate recovery-aware ablation harness and report transform

**Files:**
- Create: `runner/corpus_ablation.py`
- Create: `runner/ablation_report.py`
- Create: `tests/test_corpus_ablation.py`
- Create: `tests/test_ablation_report.py`

**Interfaces:**
- Consumes: loaded ablation manifest, corpus paths, frozen baseline metrics, `RealQLoRATrainerAdapter`, `run_ablation_epoch`, and the visible-only benchmark adapters.
- Produces: `run_ablation` returning a corpus report, three epoch `recovery_evidence` rows, three visible evaluation bundle paths, one `non_scientific_runs` resource row, and no selection/reveal result; `build_ablation_report` returning paired seed-42 benchmark comparisons.

- [ ] **Step 1: Write failing boundary and recovery-assembly tests**

```python
def test_bundle_layout_never_uses_runs_or_selection_or_heldout(self):
    contents = build_ablation_bundle_contents(
        ablation_manifest={"ablation_version": "ablation-corpus-2026-09-02"},
        corpus_report={"total": 5000, "shortfalls": {}}, epoch=1, recovery_evidence={"mid_epoch_resume_fired": False, "save_measurements": []},
    )
    self.assertIn("NOT Attempt-1 evidence", contents["notes.md"])
    self.assertNotIn("selection", json.dumps(contents).lower())
    self.assertNotIn("injecagent", inspect.getsource(corpus_ablation).lower())

def test_attempt_ledger_records_a_resumed_epoch(self):
    ledger = append_attempt(ledger=[], epoch=1, recovery_evidence={"mid_epoch_resume_fired": True,
                                                                      "save_measurements": []})
    self.assertTrue(ledger[0]["recovery_evidence"]["mid_epoch_resume_fired"])
```

- [ ] **Step 2: Run focused tests to verify red**

Run: `.venv/Scripts/python.exe -m unittest tests.test_corpus_ablation tests.test_ablation_report -q`

Expected: import failure because the ablation harness and report modules do not exist.

- [ ] **Step 3: Implement the minimal separate harness**

```python
def run_ablation(*, ablation_manifest_path: Path, output_root: Path,
                 recovery_root: Path, smoke_max_steps: int | None = None,
                 max_items_per_benchmark: int | None = None) -> dict:
    """Build the approved corpus, train/evaluate epochs 1..3, and write ablation-only outputs."""
```

Use `RecoveryWorkspace` outside `output_root` and `run_ablation_epoch` for each epoch. Evaluate only the six visible frozen benchmarks on their fixed IDs; never instantiate selection or held-out adapters. Fail closed on corpus shortfall, any prompt-injection record, changed frozen digest, unavailable CUDA, recovery save above 30 seconds, or projected ledger over 72 GPU-hours. Provide `--smoke-max-steps`, `--smoke-max-items-per-benchmark`, `--kill-after-step`, and deterministic reproduction-command handling for the smoke and deliberate-kill workflow.

- [ ] **Step 4: Run focused tests to verify green**

Run: `.venv/Scripts/python.exe -m unittest tests.test_corpus_ablation tests.test_ablation_report -q`

Expected: all tests pass without network, CUDA, or finalized-bundle reads.

### Task 4: Verification, smoke, real recovery launch, and decision record

**Files:**
- Create: `docs/issue-31-corpus-ablation-decision.md`
- Modify: `RESEARCH_PLAN.md`
- Modify: issue-scoped tests and protocol files from Tasks 1-3

**Interfaces:**
- Consumes: validated code, cached source data, `ablation/` artifacts, external recovery state, and the issue #31 ledger.
- Produces: a complete decision record with corpus provenance/digest, recovery evidence, paired appendix, all-incurred compute, caveats, and a `Closes #31` commit.

- [ ] **Step 1: Run the complete offline suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -q`

Expected: all existing and new tests pass before GPU work.

- [ ] **Step 2: Run the detached smoke harness**

Run: documented WSL command with `--smoke-max-steps 2 --smoke-max-items-per-benchmark 2` and output/recovery roots outside Attempt-1 evidence.

Expected: corpus report has no shortfall or injection records; the train/evaluate path emits only `ablation/` artifacts.

- [ ] **Step 3: Exercise recovery with one deliberate epoch-1 termination**

Run: start the exact real command with `--kill-after-step <mid-epoch-safe test step>`, let it exit after a durable checkpoint, then rerun the exact command without that test-only switch.

Expected: the epoch-1 attempt ledger includes `mid_epoch_resume_fired: true`; resumed work completes from the recorded optimizer step without entering an Attempt-1 output.

- [ ] **Step 4: Launch and monitor the real detached run**

Run: the exact no-smoke, no-kill WSL command via `setsid nohup`, writing log and durable outputs under the documented ablation and external-recovery roots.

Expected: three completed epochs and three visible-only evaluation bundles. Poll every 20 minutes using only elapsed/stage/`nvidia-smi`/`free -h`/new-directory signals unless the handover's escalation thresholds are crossed.

- [ ] **Step 5: Finalize evidence and close without pushing**

Run: full suite; integrity checks; append the decision record and a short `RESEARCH_PLAN.md` status; stage only issue #31 paths; commit with `Closes #31` and `Co-Authored-By: Claude Sonnet 5`; close GitHub issue #31 with the evidence comment.

Expected: no push; the ablation stays outside Attempt-1 selection/bootstrap/held-out evidence.
