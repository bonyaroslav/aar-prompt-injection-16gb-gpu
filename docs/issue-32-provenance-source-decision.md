# Issue #32 — corpus-nutrition provenance-source decision

**Status:** accepted
**Decision date:** 2026-09-03
**Issue:** [#32](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/32)

## Decision

Issue #32 will preserve issue #27's frozen-input allowlist and add three
**digest-only supplemental provenance sources** for the #29 integrity report:

1. the training-corpus pair used for corpus-nutrition figures;
2. the baseline resource-comparison record used for baseline GPU/time figures;
3. the power-notes text used for the held-out MDE figures.

The corpus supplement is the canonical content digest of the pair:

- `data/training/dataset.jsonl`, parsed as JSON Lines in recorded row order;
- `data/training/report.json`, parsed as JSON.

Its content digest is computed by serialising the parsed pair as canonical JSON
(sorted object keys, compact separators, UTF-8) and hashing that representation
with SHA-256. The baseline resource comparison is parsed then canonicalised as
JSON before hashing. The power notes are normalised to LF line endings before
their UTF-8 SHA-256 is calculated. The provenance manifest records only each
digest, its canonicalisation rule, and its source role. It does not publish the
corpus contents or add any of these sources to the frozen analysis-input
allowlist.

Every #29 section cites the applicable supplemental digest in addition to its
frozen-input digests: corpus nutrition cites
`training_corpus_digest_only_supplement`; resource accounting cites
`baseline_resource_digest_only_supplement`; and held-out MDE figures cite
`power_notes_digest_only_supplement`. All #28 numbers continue to trace only to
the canonical frozen-input record. These source roles are not finalized
Attempt-1 bundles or newly admitted general analysis inputs.

## Why this decision is necessary

Issue #32 requires every reported number to resolve to a canonical provenance
digest and requires both gates to pass over all analysis outputs produced so
far. Issue #29's `runner.integrity_report.corpus_nutrition_label` computes
several reported quantities from the complete built corpus: total examples,
distinct assistant responses, response-frequency coverage, response-length
distribution, and multi-step share.

The existing frozen-input record from issue #27 deliberately excludes
`data/training` and the training source data. Its finalized training-bundle
entries attest to each bundle through `checksums.sha256`; the inspected
seed-17 finalized training bundle records the dataset *path* in `command.sh`,
but its `manifest.yaml` and `metrics.json` contain no corpus-content digest.
The corpus-nutrition quantities therefore cannot be traced honestly to an
existing frozen-input digest. A subsequent read-only source inventory also
found that #29's `resource_accounting` consumes the separately finalized
`baseline_resource_comparison.json`, which #27 does not record, and that
`held_out_disposition` reports MDE figures sourced from `protocol/power_notes.md`.
Neither may be silently attributed to an unrelated bundle digest.

Each new digest is content-based rather than a raw-file SHA-256 so it is
invariant to checkout line-ending settings, matching the publication-provenance
principle established for the frozen protocol in `protocol/digests.md`.

## Alternatives considered

1. **Add the three sources to issue #27's frozen-input allowlist.** Rejected:
   that would change the carefully scoped admissible-input boundary after it was
   finalized, even though issue #32 only needs attestations for specific
   integrity-report sources.
2. **Exclude corpus-nutrition figures from the #32 gate.** Rejected: issue #32
   requires the gates to cover every analysis output produced so far, including
   issue #29's integrity report.
3. **Record raw file digests.** Rejected: raw bytes can vary with checkout
   configuration; a canonical parsed-content digest supports the required
   checkout-invariant publication provenance.

## Boundaries and consequences

- The frozen Phase-1 protocol, its canonical digest, and all finalized bundles
  remain unchanged.
- The sources are one-way commitments only. The #32 provenance manifest
  contains no corpus rows, prompts, responses, resource telemetry, or power-note
  text.
- A changed source produces a different supplemental digest, so the provenance
  check must fail rather than silently reuse a prior attribution.
- Future packaging in issue #33 may carry the digest and its explanation, but
  must not treat the corpus itself as an Attempt-1 evaluation input or include
  it in the secrets-excluded evidence package merely because this commitment
  exists.

## Evidence consulted

- Live GitHub issue #32, including its requirements that all reported numbers
  trace to canonical provenance and that the gates cover the existing analysis
  outputs.
- `runner/frozen_inputs.py` and
  `docs/issue-27-frozen-input-manifest-decision.md`, which define and explain
  the frozen-input allowlist and its exclusion of `data/training`.
- `runner/integrity_report.py` and
  `docs/issue-29-failure-mode-integrity-decision.md`, which define the
  corpus-nutrition report and its source data.
- The locally finalized seed-17 training bundle, whose manifest and metrics
  demonstrate that it has no corpus-content digest.
- The locally finalized baseline resource-comparison artifact, which supplies
  #29's baseline resource values but is absent from #27's record.
- `protocol/power_notes.md`, named by #29 as the source of its held-out MDE
  figures.
- `protocol/digests.md`, which establishes canonical content digest semantics
  for publication provenance.

## Verified execution (2026-09-03)

The gates shipped as `runner/publication_gates.py` (the pure receipt +
claim-language mechanism), `runner/publication_report_inputs.py` (the read-only
adapter that declares the #28/#29 publication sections and their source roles,
touching neither transform), and `runner/publication_gate_run.py` (operator glue
that assembles the gitignored real evidence into the two reports and runs both
gates). Offline fixture tests: `tests/test_publication_gates.py` (13),
`tests/test_publication_report_inputs.py` (4), `tests/test_publication_gate_run.py` (3).

One command over the current analysis outputs:

```
python -m runner.publication_gate_run --evidence-root runs --recovery-root recovery \
    --out analysis/publication-provenance-manifest.json
```

Result: `reports=2 sections=13 receipted_numbers=1763 orphans=0
claim_language_violations=0`, exit 0. The committed
`analysis/publication-provenance-manifest.json` (`schema_version:
publication-provenance-2`) records digests and numeric receipts only — no corpus
row, prompt, response, resource telemetry or power-note text.

Identities bound by that run:

| item | digest |
|---|---|
| protocol manifest (canonical JSON content SHA-256) | `399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20` |
| `training_corpus_digest_only_supplement` | `sha256:5703b350849b859cea57985d8b3e81212431251e9c108781ca3070b856223868` |
| `baseline_resource_digest_only_supplement` | `sha256:486117730949fad68f20b7c8c6564e390bbae69cf92609363eb2685caaed5e36` |
| `power_notes_digest_only_supplement` | `sha256:e7fb11602ba542d11614cae56295509a3add1946212b9bbc2832f0ae8e95ce5c` |

Bootstrap parameters recorded in the manifest: `replicates: 10000`, `seed:
271828`, `interval: 95_percentile_paired_by_fixed_example_id`, from
`protocol/manifest.json analysis`.

`non_scientific_runs` is passed empty here: the full smoke/recovery compute
ledger is assembled and re-verified in #33's complete run. `phase_vram_peaks_gb`
is computed from each bundle's `gpu.csv` so peak VRAM keeps its training-phase
attribution.
