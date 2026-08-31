# Seed 17 outcomes summary (issue #12)

**Run date:** 2026-08-30/31 (wall clock 07:15:53 UTC 2026-08-30 start, ~15.18h duration)
**Protocol version:** `phase1-2026-08-29` (`protocol/manifest.json`)
**Upstream commit:** `1899ad64fbfbc65790d259471cc4bf4de9437aa9`

This is a plain data dump of issue #12's real evidence, for downstream analysis
(issue #14 and the #16-#23 resumable-workflow tickets) — not an interpretation.
Every number below traces to a checksummed run bundle under `runs/` (gitignored
evidence; re-derivable from the commits that produced it: `06cc210`, `1b96be1`).

## 1. Per-benchmark scores by epoch, vs. frozen baseline

| Benchmark | Baseline | Epoch 1 | Epoch 2 | Epoch 3 | Metric |
|---|---:|---:|---:|---:|---|
| `open_prompt_injection` | 0.1800 | 0.5067 | 0.6267 | 0.6767 | `1-ASV_combine_attack_only` |
| `tensor_trust_hijack` | 0.4917 | 0.5433 | 0.5533 | 0.5417 | `(HRR+DV)/2` |
| `tensor_trust_extract` | 0.5983 | 0.7417 | 0.7050 | 0.6700 | `(ERR+DV)/2` |
| `mmlu` | 0.5667 | 0.6300 | 0.6067 | 0.5933 | `exact_match_choice` |
| `gsm8k` | 0.7350 | 0.2600 | 0.4500 | 0.5550 | `exact_match_final_number` |
| `ifeval` | 0.6150 | 0.4100 | 0.3950 | 0.4350 | `instruction_compliance` |

Source: `runs/real-baseline-20260829-205020/metrics.json`,
`runs/eval-seed17-epoch{1,2,3}-20260830-071553/metrics.json`.

## 2. Selection-stage derived values (visible composite, capability gate)

| Epoch | Visible composite (mean of 3 abs. improvements) | OPI Δ | TT-hijack Δ | TT-extract Δ | GSM8K decline | IFEval decline | MMLU decline | Mean norm. retention | Capability gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | +0.1739 | +0.3267 | +0.0083 | +0.1433 | +0.4750 | +0.2050 | -0.0633 | 0.7107 | **FAIL** (all 3 reasons) |
| 2 | +0.2050 | +0.4467 | +0.0617 | +0.1067 | +0.2850 | +0.2200 | -0.0400 | 0.7750 | **FAIL** (all 3 reasons) |
| 3 | +0.2061 | +0.4967 | +0.0500 | +0.0717 | +0.1800 | +0.1800 | -0.0267 | 0.8365 | **FAIL** (all 3 reasons) |

Gate thresholds (`protocol/manifest.json` `selection.capability_gates`): GSM8K
max decline 0.02, IFEval max decline 0.03, mean normalized retention min 0.98.
Every epoch missed all three by a wide margin. `meaningful_improvement_absolute`
is 0.05 — every epoch's visible composite clears it, but `eligible` (capability
gate passed) is `false` for all three, so `meaningful_visible_mitigation` is
`false` for all three too.

**Selection outcome:** `selected_checkpoint_digest: null`, `selected_epoch: null`
(`runs/selection-seed17-20260830-071553/selection_record.json`,
digest `46dfe6eab879994e8b248a7a5f5c80d35681a7e610aa2ae70c501b1f72e2e5f8`).

## 3. Pattern worth flagging for later analysis

Visible safety (`open_prompt_injection` especially) improves monotonically
epoch-over-epoch. Capability *degrades most severely after epoch 1* and then
*partially recovers* by epoch 3 (GSM8K 0.26 → 0.45 → 0.555; IFEval
0.41 → 0.395 → 0.435) without fully returning to baseline. `tensor_trust_extract`
moves the opposite way (0.742 → 0.705 → 0.670, still above baseline).
`mmlu` is the only capability benchmark that stayed close to or above baseline
throughout. This U-shaped-then-partial-recovery capability curve, alongside
monotonically climbing visible safety, is a naturally testable hypothesis for
a 4th+ epoch or a lower/warmed-up learning rate — not something this ticket's
scope authorizes changing (that would be a new protocol version).

## 4. Resource use vs. limits

| Metric | Measured | Limit | Within limit? |
|---|---:|---:|---|
| Wall time (this seed) | 54,665.26 s = 15.18 h | 24 h/seed max | Yes |
| GPU-hours (this seed) | 15.18 h | — | — |
| Cumulative GPU-hours (baseline 6.26h + this seed) | 21.44 h | 72 h total max | Yes |
| Peak VRAM | 15.663 GiB | 15.5 GiB *declared* allocation | **No** — exceeded by 0.163 GiB (still within the physical 16 GiB card; not a crash) |
| Evidence bundle disk | 10.57 GiB | 250 GB total max | Yes |

Source: `runs/seed17-resource-comparison-20260830-071553/seed_resource_comparison.json`.
The VRAM overage is the run's only feasibility finding — recorded, not fatal.
This measurement is exactly what issue #16 ("measure Issue 12 and lock recovery
boundaries") needs as its real-cost input, rather than the Phase-3 projection
(~20.1 h/seed, ~14.9 GiB peak) it was originally sized against.

## 5. Training mechanics

- Outcome: `success`, all 3 epochs, `fallback_applied: false` (no OOM, so the
  single approved 2048→1536-token fallback was never needed).
- Sequence length stayed at the frozen 2048 for every epoch.
- Checkpoint fingerprints (sha256, first 16 hex chars): epoch-1 `a350123dae1be4d7`,
  epoch-2 `8308ffdbc50880f4`, epoch-3 `e1eb32ccbd706976`.

## 6. Held-out isolation

InjecAgent was never read or revealed by this run — `runner.evaluation
.run_trained_evaluation` doesn't evaluate it, and `runner.real_seed_run`
never constructs a `HeldOutSealer`. Confirmed by grepping every produced
bundle: the only `injecagent` mentions are the frozen manifest's own
declared config text (`config.yaml`) and this repo's own explanatory prose
(`notes.md`) — no held-out content, scores, or candidate IDs.

## 7. Known evidence-bundle defect (non-blocking, already fixed for future runs)

The three real eval bundles' `command.sh`/`environment.txt`/`notes.md` incorrectly
read "fake adapters ... no real GPU or model weights used" — a pre-existing gap
in `runner/evaluation.py` (never wired real-adapter caption overrides through,
unlike `runner/core.py`/`runner/training.py`) that this run's real invocation
was the first to expose. `metrics.json` in all three bundles is correct and
manifest-exact; the finalized bundles are immutable per the evidence contract,
so this is documented rather than patched in place. Fixed in commit `1b96be1`
for every subsequent run.

## Evidence index

| Artifact | Path |
|---|---|
| Training bundle | `runs/training-seed17-20260830-071553/` |
| Epoch 1 eval bundle | `runs/eval-seed17-epoch1-20260830-071553/` |
| Epoch 2 eval bundle | `runs/eval-seed17-epoch2-20260830-071553/` |
| Epoch 3 eval bundle | `runs/eval-seed17-epoch3-20260830-071553/` |
| Selection record | `runs/selection-seed17-20260830-071553/selection_record.json` |
| Resource comparison | `runs/seed17-resource-comparison-20260830-071553/seed_resource_comparison.json` |
| Baseline (for comparison) | `runs/real-baseline-20260829-205020/` |

All `runs/` evidence is gitignored (per RESEARCH_PLAN.md's evidence contract);
this summary is the durable, committed record of what it says.
