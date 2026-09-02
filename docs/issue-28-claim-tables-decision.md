# Issue #28 claim tables and statistics

**Issue:** [#28](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/28)
**Decision date:** 2026-09-02
**Protocol:** `phase1-2026-08-29` (`protocol/manifest.json`)
**Scope:** adds `runner/claim_tables.py` (one pure module) and its tests. No
protocol change, no finalized evidence touched, no held-out access.

## What shipped

`runner.claim_tables.build_claim_report(manifest_path, *, baseline_metrics,
epoch_metrics)` -- a pure transform over already-loaded `metrics.json` dicts
that returns the publication's central table plus the statistics that make it
defensible. No model / dataset / scorer / trainer / telemetry / storage
dependency and no I/O, mirroring `runner.analysis` (the frozen bootstrap
stage), which it reuses. This is the only new analysis seam the #28-#33 set
introduces.

## Design decisions

### Analysis unit: every prespecified epoch, never a winner

Attempt 1 finished with every completed seed finalizing
`NO_ELIGIBLE_CHECKPOINT`, so "baseline versus selected trained checkpoint" is
undefined. `analysis_units` takes the per-epoch metrics documents and requires,
per seed, exactly the manifest's prespecified epoch set (`{1..epochs}`). It
rejects: a selection-record-shaped dict, any epoch doc carrying a
`selected` / `best` / `winner` / `rank` marker, and a run that supplies only
some of its epochs (e.g. only a post-hoc winner). No field in the output names
any epoch "best" or "winner".

### Primary table organised by evaluation modality

`_modality` classifies each benchmark from its **frozen manifest eval config**,
never a hardcoded list: `scorer == "first_token_logit"` or
`max_new_tokens == 1` -> `likelihood_ranked_no_generation`, else
`free_generation_sampled_string_scored`. On the frozen manifest MMLU is the
sole no-generation benchmark. The table's rows are grouped by that axis so the
finding is visible in the table's structure.

Each checkpoint row carries `multiple_choice_only_gate_passes`: whether a
capability gate built only from the no-generation benchmark(s), at
`selection.capability_gates.<name>_max_decline`, would have passed that
checkpoint. On the real evidence every one of the nine checkpoints passes an
MMLU-only gate while failing the real multi-benchmark gate -- the modality
confound in one column.

### Caption names all four confounded axes

`PRIMARY_TABLE_CAPTION` states that the two groups differ simultaneously on
(1) chat template, (2) sampled decoding vs deterministic likelihood scoring,
(3) generation token budget, (4) scoring method, that these are confounded not
separated, that the table is an existence proof about measurement modality, and
that ticket #30 tests the largest axis so the final wording depends on its
outcome.

### Visible composite never rendered alone

`visible_composite_block` always returns the composite **with** its
per-benchmark decomposition, dominant benchmark, and a note that it averages
absolute deltas across benchmarks of very different headroom.
`render_composite` raises `CompositeWithoutDecompositionError` if handed a block
without a non-empty `per_benchmark_delta`. There is no code path yielding the
scalar by itself.

### Statistics attached to the table

- **Paired bootstrap** for every contrast -- each (run seed, epoch, benchmark)
  baseline-vs-trained pair and each (run seed, epoch) visible composite --
  delegated to `runner.analysis.bootstrap_benchmark_difference` /
  `bootstrap_visible_composite` (manifest-pinned seed and replicate count), with
  an explicit `conditional_on` note that the interval is conditional on the
  evaluated example IDs, not a population statement.
- **Exact McNemar** (`mcnemar_exact`): two-sided exact binomial tail on the
  discordant pairs, p = 0.5. Hand-checked values are asserted (b=2 -> 0.5,
  b=3 -> 0.25, b=5 -> 0.0625, b=8/c=2 -> 112/1024, no discordant -> 1.0).
  Reported per benchmark; benchmarks whose metric averages multiple arms per
  item (Tensor Trust hijack/extract -> non-binary per-item scores) are flagged
  `applicable: false` rather than forced.
- **Cross-run summary** reuses `runner.continuation.summarize_seeds`. Its
  framing is rewritten to state it is a **descriptive population statistic over
  the runs actually executed** (population SD, never an inferential or
  population-level interval) and that it is **run-to-run variability under a
  fixed nominal configuration -- adapter initialisation precedes the run seed --
  and must not be described as seed variance**. Included only where >= 2 runs
  exist, so it is correct at two completed seeds and at three.

## Validation

- `tests/test_claim_tables.py` (25 tests), fully offline: hand-shaped
  `metrics.json` dicts and a manifest fixture (frozen manifest with
  `bootstrap_replicates` reduced for speed). No GPU, no adapters, real evidence
  tree never read.
- Acceptance criteria covered one-to-one (see test names).
- End-to-end against the real evidence tree: 3 seeds / 9 checkpoints, MMLU alone
  in the no-generation group, all nine checkpoints pass an MMLU-only gate,
  gsm8k seed-17 epoch-1 exact McNemar p ~ 5e-24 (100 discordant baseline-only
  vs 5 trained-only), output byte-identical across two runs.
- Full repository suite: 283 pass, 1 skip (pre-existing missing-`torch` skip).
