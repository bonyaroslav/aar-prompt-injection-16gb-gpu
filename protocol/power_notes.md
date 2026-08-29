# Pre-registered power notes: minimum detectable effect per benchmark

Computed 2026-08-29, before any baseline or trained evaluation runs, from the published Qwen3.5-2B
rows in `benchmark_docs/prompt_injection/baseline.json` (already fingerprinted in
`protocol/provenance.json`). This document exists so the manifest's fixed thresholds
(`selection.meaningful_improvement_absolute = 0.05`, `selection.capability_gates.*`) can be judged
against statistical power *before* any quality result exists, rather than after.

## Method

Each visible/held-out metric is a rule-scored proportion, so per-benchmark standard error is
approximated with the binomial formula `SE = sqrt(p(1-p)/n)` using the published baseline `mean`
and `n`. This matches the published 95% CIs closely (recomputing `1.96 * SE` reproduces the
published `ci_low`/`ci_high` half-width to within rounding on all four benchmarks below), so the
binomial approximation is treated as adequate for pre-registration purposes.

A baseline-vs-trained **difference** is treated as an independent two-sample comparison —
`SE_diff = sqrt(2) * SE_baseline` — which assumes the trained run has similar variance to baseline
and treats the two runs as independent draws over the same fixed item IDs. This is a conservative
(wider) approximation: the manifest's actual analysis uses paired bootstrap over fixed example IDs,
which will likely be tighter than this if per-item outcomes are positively correlated across
baseline/trained generations (same item, same underlying difficulty). Treat every MDE below as an
upper bound on the true minimum detectable effect, not an exact figure.

Two power levels are reported:
- **MDE50** — `1.96 * SE_diff`: the smallest observed difference that would fall outside a 95% CI
  around zero (bare two-sided significance at α=0.05, ~50% power to detect a true effect of this size).
- **MDE80** — `2.80 * SE_diff` (`z_{0.975} + z_{0.80} ≈ 1.96 + 0.84`): the true effect size needed for
  ~80% power to distinguish it from zero at α=0.05, two-sided.

## Per-benchmark results

| Benchmark | Baseline mean | n | SE (binomial) | SE_diff (√2×) | MDE50 | MDE80 |
|---|---:|---:|---:|---:|---:|---:|
| `open_prompt_injection` | 0.3133 | 300 | 0.0268 | 0.0379 | **7.4 pp** | **10.6 pp** |
| `tensor_trust_hijack` | 0.5050 | 600 | 0.0204 | 0.0289 | **5.7 pp** | **8.1 pp** |
| `tensor_trust_extract` | 0.5383 | 600 | 0.0204 | 0.0288 | **5.6 pp** | **8.1 pp** |
| `injecagent` (`valid_only`, n=134) | 0.8881 | 134 | 0.0272 | 0.0385 | **7.5 pp** | **10.8 pp** |
| `injecagent` (`intent_to_evaluate`, n=200, ~119/200 successes) | ≈0.595 | 200 | 0.0347 | 0.0491 | **9.6 pp** | **13.8 pp** |

## Composite (unweighted mean of the three visible absolute improvements)

Treating the three visible benchmarks as independent (different item sets, same assumption as
above), `Var(composite_diff) = (1/9) * Σ SE_diff²`:

`SE_diff(composite) = sqrt[(0.0379² + 0.0289² + 0.0288²) / 9] ≈ 0.0186` → **MDE50 ≈ 3.6 pp, MDE80 ≈ 5.2 pp**.

## Reading this before Phase 4

- **The manifest's own +5 percentage-point meaningful-improvement threshold sits almost exactly at
  the composite's MDE80 (5.2 pp).** This is worth stating plainly in the eventual report: the
  threshold is not a conservative floor with power to spare, and it is not so lax that noise alone
  clears it either — it is calibrated close to the 80%-power boundary for the aggregated composite
  under this (conservative, independent-samples) approximation. A true composite effect smaller than
  ~5 pp has a real chance of clearing the observed-value threshold by chance in either direction less
  than half the time; a true effect at exactly 5 pp has roughly even odds of being reported as
  "meaningful" under a bare significance framing, though the manifest's decision rule is a fixed
  threshold on the point estimate rather than a significance test, so this is context, not a formal
  power guarantee for the actual selection rule.
- **No single visible benchmark in isolation is well-powered at the composite's 5 pp bar.** Each
  individual leg needs 8–11 pp (MDE80) to be distinguishable from a null result on its own. This is
  exactly why the manifest requires the *combined* composite rather than any one benchmark — consistent
  with `protocol/deviations.md`'s point that upstream's own OPI-only optimization need not transfer.
- **`injecagent` has the least power of any declared metric**, and the gap between `valid_only`
  (10.8 pp) and `intent_to_evaluate` (13.8 pp) is itself informative: a training run that changes the
  *valid-rate* (currently 67%, a known ≤7B-parameter characteristic per upstream's own baseline notes)
  will move `intent_to_evaluate` for reasons unrelated to attack resistance. Both denominators must be
  reported together, exactly as `protocol/heldout_sealing.md` already requires — this power gap is why.
- These figures use the *published* baseline `n`/`mean`, not a fresh baseline run. Phase 4 must
  recompute realized SEs from this study's own baseline evaluation (same items, same scorer, but a
  fresh generation with `decoding.seed=1234`) and may find them somewhat different from this
  pre-registration estimate; this document is a planning artifact, not a substitute for that
  measurement.
