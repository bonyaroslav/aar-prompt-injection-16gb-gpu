# Issue #32 — provenance and claim-language gates design

**Status:** approved for implementation, 2026-09-03
**Issue:** GitHub #32, "Provenance manifest and claim-language gate"
**Decision record:** `docs/issue-32-provenance-source-decision.md`

## Purpose

Build one offline-only publication gate that validates the analysis outputs
available before issue #33: the structured claim-table report from #28 and the
structured integrity report from #29. It must make an altered reported number
or unsupported efficacy/capability language fail mechanically, rather than
relying on a reader to notice it.

The gate does not modify the frozen protocol, any finalized bundle, data,
selection, held-out material, or GPU state. It is a pure CPU/fixture workflow.

## Terms and source boundary

* **Frozen input record** is the #27 JSON output from
  `runner.frozen_inputs.freeze_inputs`. Its `inputs` list is the authority for
  normal report-source digests; its `protocol_manifest_digest` and
  `analysis_version` are publication metadata.
* **Supplemental sources** are the three digest-only commitments approved in
  the issue-#32 decision record: parsed `dataset.jsonl` plus `report.json`,
  parsed baseline resource comparison, and LF-normalised power notes. They do
  not expand #27's allowlist.
* **Report** is a named structured artifact split into named publication
  sections. Each section has a kind (`table` or `figure`), a JSON-compatible
  content value, and declared source roles. The current reports are the #28
  claim tables and #29 integrity report.
* **Orphan** is a numeric value whose report/section/location/value receipt is
  absent from the already-built provenance manifest. This covers inserted and
  altered values, including derived values such as bootstrap intervals and
  deltas: they have a receipt tied to their source digests even when their
  exact decimal representation does not occur verbatim in a source file.

## Module and interface

Add `runner/publication_gates.py`. Its public interface remains intentionally
small:

```python
def build_corpus_supplement(dataset_path: Path, report_path: Path) -> dict: ...
def build_provenance_manifest(*, frozen_input_record: dict,
                              reports: list[dict],
                              corpus_supplement: dict | None) -> dict: ...
def verify_provenance(*, provenance_manifest: dict,
                      frozen_input_record: dict, reports: list[dict],
                      corpus_supplement: dict | None) -> None: ...
def check_claim_language(reports: list[dict]) -> None: ...
def run_gates(**kwargs) -> dict: ...
```

`run_gates` runs provenance verification and language verification together,
then returns the verified provenance manifest. The module’s command line loads
the same JSON inputs and exits with status 2, naming the failed report location,
on either gate failure.

### Provenance manifest

The manifest has a schema version; the #27 frozen-record digest; the frozen
protocol’s canonical content digest with an explicit `canonical_json` meaning;
analysis version; bootstrap parameters extracted from the frozen protocol; and
the optional supplemental-source registry. It maps every table/figure section
to resolved source digests. Every numeric leaf receives a receipt calculated from its
report ID, section ID, JSON location, canonical numeric value, and resolved
source digests.

Verification recomputes those receipts from the submitted reports. A changed
or inserted number has no matching receipt and fails as an orphan with its
value and precise location. A missing section, unknown source role, changed
frozen-record identity, changed corpus supplement, or source digest not
present in the frozen record also fails closed.

The corpus supplement is the canonical JSON SHA-256 of a two-member object:
the JSONL rows in recorded order and the parsed build report. The resource
supplement is a canonical JSON SHA-256 of the parsed baseline comparison. The
power-notes supplement is an LF-normalised UTF-8 SHA-256. The manifest stores
only their roles and digests, never their contents.

### Claim-language gate

The gate walks each report’s human-readable string leaves and rejects the
following case-insensitive expressions as whole words or an exact phrase:
`robust`, `secure`, `resistant`, `mitigation`, and `defense that works`.

A capability statement is text containing `capability` plus a claim action
such as preserved, retained, maintained, improved, declined, collapsed, or
failed. It must name an evaluation modality in the same sentence or line:
`free_generation_sampled_string_scored`,
`likelihood_ranked_no_generation`, free-generation, generation-scored,
likelihood-ranked, or log-likelihood. Mere references to a capability *gate*
without a claim action are not capability statements.

## Current report registration

The registration adapter keeps #28 and #29 unchanged. It declares their
publication sections and source roles instead of duplicating their analysis
logic:

* Claim tables: `primary_table`, `paired_bootstrap`, `mcnemar_exact`,
  `visible_composite`, and `cross_run_summary` cite the protocol, baseline and
  per-epoch evaluation inputs; bootstrap sections also cite the protocol
  because it fixes the parameters.
* Integrity report: failure-mode sections cite the relevant baseline/evaluation
  inputs; resource accounting additionally cites
  `baseline_resource_digest_only_supplement`; corpus nutrition additionally
  cites `training_corpus_digest_only_supplement`; held-out MDE figures
  additionally cite `power_notes_digest_only_supplement`; other policy and
  disclosure sections cite the frozen protocol and registered governance inputs.

The adapter accepts already-rendered JSON reports. It performs no model,
dataset, scorer, trainer, telemetry, storage, or held-out operation. Issue #33
can register its results document and diagnostic/ablation appendices through
the same report/section format, while keeping those appendices separately
versioned rather than adding them to Attempt-1 inputs.

## Test and error behavior

Tests use temporary JSON reports and tiny corpus fixtures only. They cover a
successful manifest, a fabricated numeric value with report/section/location in
the error, canonical content-digest invariance, every forbidden term separately,
a modality-less capability statement, a qualified capability statement, and a
CLI invocation that exits non-zero for either gate. The full repository suite
remains the final regression check.

## Non-goals

This issue does not assess the experiment again, choose a checkpoint, change
the capability gates, rewrite existing decision records, publish a final
results document, package artifacts, or run GPU work. Those are either frozen
or belong to #33.
