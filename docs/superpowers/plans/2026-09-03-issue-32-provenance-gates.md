# Issue #32 Provenance and Claim-Language Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline command that verifies every registered #28/#29 report number against a provenance manifest and rejects unsupported claim language.

**Architecture:** `runner.publication_gates` is one deep module. It canonicalises the optional corpus supplement, builds/verifies number receipts against the #27 frozen-input record, scans text for claim violations, and exposes one CLI; #28 and #29 remain pure report transforms.

**Tech Stack:** Python 3.13, standard library (`argparse`, `hashlib`, `json`, `re`, `pathlib`), `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-03-issue-32-provenance-gates-design.md`; `docs/issue-32-provenance-source-decision.md`

## Global Constraints

- Use the canonical-JSON content digest for publication provenance; never use a raw-file digest for the corpus supplement.
- Keep #27’s frozen-input allowlist, `protocol/manifest.json`, finalized bundles, held-out material, data contents, and GPU state unchanged.
- The supplemental-source registry records only canonical digests of parsed
  corpus rows/build report, parsed baseline resource comparison, and
  LF-normalised power notes.
- The checker must fail closed, name an orphan’s report/section/location/value, and return a non-zero CLI status.
- Tests are offline fixtures only; run the existing full suite unchanged before finalization.

---

### Task 1: Provenance receipt contract

**Files:**
- Create: `runner/publication_gates.py`
- Create: `tests/test_publication_gates.py`

**Interfaces:**
- Consumes: #27-shaped frozen input record and JSON-compatible report section descriptors.
- Produces: `build_provenance_manifest`, `verify_provenance`, and `ProvenanceGateError`.

- [x] **Step 1: Write failing receipt tests**

```python
def test_changed_number_is_an_orphan_with_its_location_named(self):
    reports = [_report(value=0.25)]
    manifest = build_provenance_manifest(
        frozen_input_record=_frozen_record(), reports=reports,
        corpus_supplement=None,
    )
    altered = [_report(value=0.75)]
    with self.assertRaisesRegex(ProvenanceGateError, r"orphan value 0.75.*claim_tables.*primary_table.*score"):
        verify_provenance(
            provenance_manifest=manifest, frozen_input_record=_frozen_record(),
            reports=altered, corpus_supplement=None,
        )
```

- [x] **Step 2: Run the focused test and confirm RED**

Run: `$env:PYTHONPATH='.'; .venv\Scripts\python.exe -m unittest tests.test_publication_gates -q`

Expected: import failure because `runner.publication_gates` does not exist.

- [x] **Step 3: Implement canonical receipts and verification**

```python
def _receipt(*, report_id, section_kind, section_id, location, value, input_digests):
    payload = {
        "report_id": report_id, "section_kind": section_kind,
        "section_id": section_id, "location": location,
        "value": value, "input_digests": sorted(input_digests),
    }
    return "sha256:" + _canonical_digest(payload)

def verify_provenance(*, provenance_manifest, frozen_input_record, reports, corpus_supplement):
    expected = build_provenance_manifest(
        frozen_input_record=frozen_input_record, reports=reports,
        corpus_supplement=corpus_supplement,
    )
    _raise_first_manifest_difference(provenance_manifest, expected)
```

The difference helper must compare section identity and numeric receipts so an
inserted or altered numeric leaf raises `ProvenanceGateError` with the report,
section, JSON location, and current value.

- [x] **Step 4: Run the focused test and confirm GREEN**

Run: `$env:PYTHONPATH='.'; .venv\Scripts\python.exe -m unittest tests.test_publication_gates -q`

Expected: the changed-value test passes by observing the expected exception.

### Task 2: Digest-only corpus source and manifest metadata

**Files:**
- Modify: `runner/publication_gates.py`
- Modify: `tests/test_publication_gates.py`

**Interfaces:**
- Consumes: a JSONL corpus path and a JSON build-report path.
- Produces: `build_corpus_supplement(dataset_path, report_path) -> dict` and a manifest with protocol, analysis-version, bootstrap, and resolved-digest metadata.

- [x] **Step 1: Write failing corpus-digest tests**

```python
def test_corpus_supplement_is_invariant_to_json_key_order_and_line_endings(self):
    first = _write_corpus(self.root / "first", '{"b":2,"a":1}\n')
    second = _write_corpus(self.root / "second", '{"a":1,"b":2}\r\n')
    self.assertEqual(
        build_corpus_supplement(*first)["digest"],
        build_corpus_supplement(*second)["digest"],
    )

def test_corpus_nutrition_section_requires_the_named_supplement(self):
    with self.assertRaisesRegex(ProvenanceGateError, "training_corpus_digest_only_supplement"):
        build_provenance_manifest(
            frozen_input_record=_frozen_record(), reports=[_corpus_report()],
            corpus_supplement=None,
        )
```

- [x] **Step 2: Run the focused test and confirm RED**

Run: `$env:PYTHONPATH='.'; .venv\Scripts\python.exe -m unittest tests.test_publication_gates -q`

Expected: missing `build_corpus_supplement` or absent supplemental-source validation.

- [x] **Step 3: Implement the canonical supplement and source resolution**

```python
def build_corpus_supplement(dataset_path: Path, report_path: Path) -> dict:
    rows = [json.loads(line) for line in Path(dataset_path).read_text(encoding="utf-8").splitlines() if line]
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return {
        "role": "training_corpus_digest_only_supplement",
        "digest_kind": "canonical_json",
        "digest": "sha256:" + _canonical_digest({"rows": rows, "report": report}),
    }
```

Resolve ordinary source roles only from the frozen input record and the one
supplemental role only from this return value. Include the frozen-record digest,
canonical protocol digest and its meaning, analysis version, and the frozen
manifest’s bootstrap configuration in each generated provenance manifest.

- [x] **Step 4: Run the focused test and confirm GREEN**

Run: `$env:PYTHONPATH='.'; .venv\Scripts\python.exe -m unittest tests.test_publication_gates -q`

Expected: canonical-equivalent corpus files have one digest and an unbound
corpus-nutrition section fails closed.

### Task 3: Claim-language and one-command gate

**Files:**
- Modify: `runner/publication_gates.py`
- Modify: `tests/test_publication_gates.py`

**Interfaces:**
- Consumes: registered report text and paths to frozen record, provenance manifest, report registry, and optional corpus files.
- Produces: `check_claim_language`, `run_gates`, and `python -m runner.publication_gates` status 0/2.

- [x] **Step 1: Write failing language and CLI tests**

```python
def test_forbidden_terms_each_fail(self):
    for term in ("robust", "secure", "resistant", "mitigation", "defense that works"):
        with self.subTest(term=term):
            with self.assertRaisesRegex(ClaimLanguageError, term):
                check_claim_language([_report(text=f"The intervention is {term}." )])

def test_capability_claim_without_modality_fails(self):
    with self.assertRaisesRegex(ClaimLanguageError, "evaluation modality"):
        check_claim_language([_report(text="Capability preserved after training.")])

def test_cli_returns_nonzero_for_orphan(self):
    completed = subprocess.run(_cli_args_with_altered_report(), capture_output=True, text=True)
    self.assertEqual(completed.returncode, 2)
    self.assertIn("orphan value", completed.stderr)
```

- [x] **Step 2: Run the focused test and confirm RED**

Run: `$env:PYTHONPATH='.'; .venv\Scripts\python.exe -m unittest tests.test_publication_gates -q`

Expected: missing language checker and command-line entry point.

- [x] **Step 3: Implement text scanning and CLI**

```python
def run_gates(**kwargs) -> dict:
    verify_provenance(**kwargs)
    check_claim_language(kwargs["reports"])
    return kwargs["provenance_manifest"]

if __name__ == "__main__":
    raise SystemExit(main())
```

Walk all string leaves with locations. Match forbidden language
case-insensitively and match only capability sentences that also contain a
claim action. Accept a qualified capability statement only when its sentence
also names a defined modality. The CLI loads the JSON record/manifest/report
registry, prints `publication gate failed: <reason>` to stderr, and exits 2 on
either gate error.

- [x] **Step 4: Run targeted tests and then the full suite**

Run: `$env:PYTHONPATH='.'; .venv\Scripts\python.exe -m unittest tests.test_publication_gates -q`

Expected: all new gate tests pass offline.

Run: `$env:PYTHONPATH='.'; .venv\Scripts\python.exe -m unittest discover -s tests -q`

Expected: all existing tests pass with their established pre-existing skip.

### Task 4: Evidence record, issue closure, and no-push handoff

**Files:**
- Modify: `docs/issue-32-provenance-source-decision.md`
- Modify: `RESEARCH_PLAN.md`
- Modify: issue-scoped test and gate files from Tasks 1–3

**Interfaces:**
- Consumes: green offline tests and a successful gate run over the current #28/#29 reports.
- Produces: decision record with actual digests/command/evidence, one issue-scoped commit, and a closed GitHub issue.

- [x] **Step 1: Run the gate over the registered current reports**

Run: `python -m runner.publication_gates --frozen-input <record> --provenance <manifest> --reports <registry> --corpus-dataset data/training/dataset.jsonl --corpus-report data/training/report.json`

Expected: status 0; #28/#29 sections have zero orphans and no claim-language violation.

- [x] **Step 2: Record verified evidence and update status**

Add the command, canonical protocol digest, corpus supplemental digest, report
section count, and test outcome to the issue #32 decision record. Update only
the current-status paragraph in `RESEARCH_PLAN.md`; do not amend earlier
experimental results.

- [x] **Step 3: Review and commit only issue-scoped files**

Run: `git diff --check` and `git status --short`.

Stage the explicit #32 paths and commit with a concise subject containing
`Closes #32` and the required `Co-Authored-By: Claude Sonnet 5` trailer. Do not
push.

- [x] **Step 4: Verify commit and close the live issue**

Run: `git show --check --stat --oneline HEAD` and confirm issue #32 is still
open. Close it with an evidence comment naming the gate command, offline suite,
and commit ID. Do not begin #33.
