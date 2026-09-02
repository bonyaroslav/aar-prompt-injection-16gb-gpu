# Finalization handover: three completed seeds

**Status date:** 2026-09-02 (originally 2026-09-01, two seeds; updated when seed 2026 finalized)
**Purpose:** Decision-neutral handover for designing a replacement finalization issue after GitHub issue #14 was closed as not planned. It records verified evidence and open questions; it does not choose an audience, venue, scientific claim, or next protocol.

## Scope and boundary

This handover covers the frozen Phase-1 manifest, the frozen real baseline, and finalized visible/capability evidence for seeds 17, 42, and 2026. It does not use raw held-out InjecAgent data, recovery state, secrets, credentials, model caches, or smoke outputs as scientific evidence.

`protocol/manifest.json` has canonical-JSON content digest `399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20`, the checkout-invariant identity used for publication and selection provenance. Its separate raw-file SHA-256 is `296e093bb1a6fc72f6e4cdf6ed3de5cde77a9e3da90df73db4538a2a98e6f4ac`, matching `protocol/manifest.sha256` and used for byte-integrity and stage signatures. It specifies seeds 17, 42, and 2026; a 10,000-replicate seed-271828 paired bootstrap over fixed visible-example IDs; capability-gated checkpoint selection; and held-out reveal only for a finalized selected checkpoint.

## Work completed

| Work | Outcome | Durable record |
| --- | --- | --- |
| Frozen baseline | Completed with the manifest-exact visible/capability evaluation topology | `runs/real-baseline-20260829-205020/` |
| Seed 17 | Three training epochs and three evaluations completed; finalized `NO_ELIGIBLE_CHECKPOINT` | `analysis/seed17-outcomes-summary.md`; `docs/issue-21-null-selection-continuation-decision.md` |
| Seed 42 | Recovery-aware training and three evaluations completed; finalized `NO_ELIGIBLE_CHECKPOINT` | `analysis/seed42-outcomes-summary.md`; `docs/issue-22-seed-42-execution-decision.md` |
| Seed 2026 | Recovery-aware training and three evaluations completed; finalized `NO_ELIGIBLE_CHECKPOINT`; frozen replication set `[17, 42, 2026]` complete | `analysis/seed2026-outcomes-summary.md`; `docs/issue-23-seed-2026-execution-decision.md` |
| Recovery workflow | Finalized-only discovery, interruption accounting, and null-selection handling were decided and implemented | `docs/issue-16-` through `issue-23-` decision records |
| Original finalization issue | GitHub issue #14 closed as not planned so finalization requirements can be redesigned | GitHub issue #14 |

## Results from the three completed seeds

### Seed 17

Seed 17 completed three frozen epochs without OOM fallback and evaluated each epoch. Every epoch improved visible safety but failed the frozen capability gates; its finalized selection record has `selected_epoch: null` and `selected_checkpoint_digest: null`.

| Measure | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | ---: | ---: | ---: |
| Visible composite improvement | +0.1739 | +0.2050 | +0.2061 |
| Open Prompt Injection improvement | +0.3267 | +0.4467 | +0.4967 |
| GSM8K decline | 0.4750 | 0.2850 | 0.1800 |
| IFEval decline | 0.2050 | 0.2200 | 0.1800 |
| Mean normalized capability retention | 0.7107 | 0.7750 | 0.8365 |

Resource measurement: 15.1848 GPU-hours; observed peak VRAM 15.663 GiB, above the manifest's 15.5 GiB allocation but within the physical 16 GiB card.

### Seed 42

Seed 42 completed through the recovery-aware workflow, also without OOM fallback. Every epoch failed the capability gates and its finalized selection is null.

| Measure | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | ---: | ---: | ---: |
| Visible composite improvement | +0.2733 | +0.2294 | +0.1794 |
| Open Prompt Injection improvement | +0.5733 | +0.5600 | +0.4867 |
| GSM8K decline | 0.4550 | 0.2250 | 0.2350 |
| IFEval decline | 0.2300 | 0.1300 | 0.1300 |
| Mean normalized capability retention | 0.7141 | 0.8549 | 0.8426 |

Resource measurement: 12.7427 GPU-hours; observed peak VRAM 15.6025 GiB, again above the declared allocation.

### Seed 2026

Seed 2026 completed through the identical recovery-aware workflow, also without OOM fallback. Every epoch failed the capability gates and its finalized selection is null.

| Measure | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | ---: | ---: | ---: |
| Visible composite improvement | +0.2056 | +0.1994 | +0.1939 |
| Open Prompt Injection improvement | +0.3100 | +0.3700 | +0.4267 |
| GSM8K decline | 0.4450 | 0.2900 | 0.2000 |
| IFEval decline | 0.1800 | 0.2000 | 0.1450 |
| Mean normalized capability retention | 0.7398 | 0.7856 | 0.8464 |

Resource measurement: 13.1553 GPU-hours; observed peak VRAM 15.6289 GiB, again above the declared allocation.

### Current interpretation boundary

The three completed seeds support a concise negative finding: under this exact frozen QLoRA intervention and protocol, all three seeds showed visible gains but produced no capability-eligible checkpoint (nine trained checkpoints, nine capability-gate failures). The visible composite is strongly driven by Open Prompt Injection; it is not evidence of uniform safety improvement.

The evidence does **not** support a capability-preserving mitigation claim, a held-out InjecAgent generalization claim, a causal explanation for capability loss, or a population-level inference from three seeds. Fixed-example bootstrap intervals are conditional on the evaluated examples; cross-seed mean/range/standard deviation are descriptive only.

## Current outcome of Issue #23 / seed 2026

GitHub issue #23 ("Conditionally execute and verify seed 2026") is **finalized and closed** (2026-09-02). Seed 2026's real evidence run finalized the full training-plus-three-evaluation-plus-selection-plus-resource topology; the continuation and finalization decision is recorded in `docs/issue-23-seed-2026-execution-decision.md`. The frozen replication set `[17, 42, 2026]` is complete at three seeds, all null selections.

`runs/seed2026-smoke/` (and the pre-run adapter directories `runs/adapters-seed2026-20260901-103119` and `-103738`) are workflow scaffolding only: gitignored, no ledger attempt, must never enter effect estimates, bootstrap intervals, or cross-seed summaries. The evidence run is `runs/*-seed2026-20260901-112915-bf0809d1` (no `--smoke-*` flag). Cumulative GPU use after seed 2026 is 47.3381 of 72 hours, 24.6619 remaining.

The seed-2026 null selection does not authorize held-out reveal or post-hoc checkpoint selection.

## Verified finalized artifacts

Checksums below were verified during the handover assessment. Training bundles are provenance inputs; evaluation `metrics.json` files are the visible/capability analysis source; selection records establish terminal state; resource artifacts support feasibility reporting.

| Evidence | Path | Integrity and role |
| --- | --- | --- |
| Frozen baseline | `runs/real-baseline-20260829-205020/` | `checksums.sha256`: 8/8 files passed; baseline visible/capability metrics |
| Seed-17 training | `runs/training-seed17-20260830-071553/` | 8/8 passed; training provenance |
| Seed-17 evaluations | `runs/eval-seed17-epoch{1,2,3}-20260830-071553/` | 8/8 passed for each; per-epoch metrics |
| Seed-17 selection | `runs/selection-seed17-20260830-071553/selection_record.json` | SHA-256 `46dfe6eab879994e8b248a7a5f5c80d35681a7e610aa2ae70c501b1f72e2e5f8`; finalized null selection |
| Seed-17 resources | `runs/seed17-resource-comparison-20260830-071553/` | 1/1 passed |
| Seed-42 training | `runs/training-seed42-20260831-201248-1b487000/` | 8/8 passed; training provenance |
| Seed-42 evaluations | `runs/eval-seed42-epoch{1,2,3}-20260831-201248-1b487000/` | 8/8 passed for each; per-epoch metrics |
| Seed-42 selection | `runs/selection-seed42/selection_record.json` | SHA-256 `a2cdd7c2e1c1b8989f9b7c254cae56f2b812605436ba04861a1d667a79b2cdce`; finalized null selection |
| Seed-42 resources | `runs/seed42-resource-comparison/` | 1/1 passed |
| Seed-2026 training | `runs/training-seed2026-20260901-112915-bf0809d1/` | 8/8 passed; training provenance |
| Seed-2026 evaluations | `runs/eval-seed2026-epoch{1,2,3}-20260901-112915-bf0809d1/` | 8/8 passed for each; per-epoch metrics |
| Seed-2026 selection | `runs/selection-seed2026/selection_record.json` | SHA-256 `8df462a4548fe652660409ef76b2b987a7794a0904f9b400cf8bdf1ba10a0d23`; finalized null selection |
| Seed-2026 resources | `runs/seed2026-resource-comparison/` | 1/1 passed |

Known metadata caveats: the frozen baseline `notes.md` has a stale smoke caption; seed-17 evaluation captions incorrectly described fake adapters despite real manifest-exact metrics. These finalized files must not be rewritten; disclose the caveats when describing provenance.

## Excluded material

The following are not scientific analysis inputs: `recovery/` and `recovery-smoke-seed2026/` operational state; `runs/seed2026-smoke/`; model caches and private work directories; `data/` and training source data; any restricted external held-out root; and raw InjecAgent candidates, prompts, tool outputs, receipts, secrets, or credentials.

For seeds 17 and 42, the absence of a held-out reveal bundle is expected: the checksum-valid null selections prohibit the reveal transaction. It is not a missing result.

## Original intention of Issue #14

The closed issue was intended as the final Phase-8 packaging step: run the manifest-pinned bootstrap; make per-benchmark, interval, and composite tables; report descriptive cross-seed summaries; aggregate resource use and manual interventions; and produce a checksummed, secrets-excluded package whose publication numbers each trace to evidence artifacts.

That intent remains useful, but its requirements did not fully cover the evidence state now observed.

## Problems found in the old finalization requirements

1. **Null-selection ambiguity:** "baseline-vs-trained" does not define the analysis unit when three trained epochs exist but no checkpoint is selected. A replacement must report all prespecified epoch results or another prespecified unit, never a post-hoc winner.
2. **Held-out ambiguity:** "every benchmark" could include InjecAgent, whose public contract offers aggregates only after authorization and no result at all for null selection. It is not a visible-style paired-bootstrap input.
3. **Two-seed gap:** the old issue anticipated one or three completed seeds. The study completed all three (`[17, 42, 2026]`), so the three-seed path applies; the analysis should still not hardcode the seed count, in case a later integrity finding invalidates a seed.
4. **Resource ambiguity:** it did not specify treatment of smoke, interruption, unavailable intervals, cumulative artifacts, or double counting. GPU/wall time is additive by attempt; peak VRAM is a maximum; disk needs a defined snapshot/unique-artifact policy.
5. **Insufficient package provenance:** the package needs a manifest mapping every table/figure number to exact input hashes, the frozen manifest digest, analysis version, and bootstrap parameters.
6. **Weak exclusion criterion:** filename filtering alone is insufficient. An automated allowlist must reject recovery paths, smoke paths, symlinks, archives, caches, raw held-out material, and credential-like content.
7. **Publication-risk wording:** visible gains cannot be framed as successful mitigation while every candidate fails the frozen capability gates.

## Questions for the next finalization-design session

- Is the intended output a negative-results technical report, reproducibility/feasibility report, intervention observation, or something else?
- Which claims are useful and defensible for the intended audience and relevant prior literature?
- Should reporting focus on every epoch, descriptive seed summaries, or both?
- Seed 2026 completed with a third null selection; how should the three-seed descriptive summary integrate it without implying a population-level inference?
- Should resources report only scientific evidence runs, all incurred GPU attempts, or both as separately labeled totals?
- Which artifacts can be public, which remain local, and which automated checksum/provenance/allowlist check enforces the boundary?

## Minimum requirements for a replacement finalization issue

Seed 2026 has completed (2026-09-02), so this is now unblocked. It should explicitly define: allowed finalized input hashes; null selection as a complete terminal topology with no held-out reveal; the per-seed analysis unit; bootstrap scope and interpretation limits; permitted seed counts including two; no-double-count resource rules; automated exclusion and provenance checks; and claim language that distinguishes a capability-gated negative result from efficacy or held-out-generalization claims.

## Source records

- `protocol/manifest.json` and `protocol/manifest.sha256`
- `analysis/seed17-outcomes-summary.md`
- `analysis/seed42-outcomes-summary.md`
- `analysis/seed2026-outcomes-summary.md`
- `docs/issue-16-recovery-boundaries-decision.md` through `docs/issue-23-seed-2026-execution-decision.md`
- Closed GitHub issues #14 and #23
