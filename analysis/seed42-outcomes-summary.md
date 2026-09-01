# Seed 42 outcomes summary (issue #22)

**Run date:** 2026-08-31 to 2026-09-01
**Protocol:** `phase1-2026-08-29` (`protocol/manifest.json`)
**Seed:** `42`
**Upstream:** commit `1899ad64fbfbc65790d259471cc4bf4de9437aa9`, tree
`00f1edb9193487e7e306177709b1760be180d7ac`

This is the durable analysis record for issue #22's real seed-42 evidence. It
reports the finalized, checksummed artifacts and gives only protocol-bounded
interpretation. It does not use held-out InjecAgent data: seed 42 finalized a
null selection, so no held-out reveal was authorized or performed.

## 1. Finalized evidence and recovery status

Issue #22 ran the second frozen replication seed through the recovery-aware
orchestration. The real run completed in one session; recovery nevertheless
recorded durable stage state and an append-only ledger outside `runs/`.

| Check | Result |
| --- | --- |
| Training | Success; all three epochs completed |
| OOM fallback | Not applied; every epoch used the frozen 2,048-token setting |
| Evaluations | Three completed, one for each epoch |
| Finalized selection | `selected_epoch: null`; `selected_checkpoint_digest: null` |
| Resource artifact | Finalized at the stable `seed42-resource-comparison` id |
| Attempts | 4 completed, 0 interrupted, 0 declared unavailable intervals |
| Held-out reveal | Not created; null selection prohibits it |
| Artifact integrity | Training, all three evaluations, and the resource artifact passed their recorded SHA-256 checksums |

The selection-record SHA-256 is
`a2cdd7c2e1c1b8989f9b7c254cae56f2b812605436ba04861a1d667a79b2cdce`.
Recovery state is operational state under `recovery/`, not a finalized analysis
input.

## 2. Per-benchmark scores versus the frozen baseline

| Benchmark | Baseline | Epoch 1 | Epoch 2 | Epoch 3 | Metric |
| --- | ---: | ---: | ---: | ---: | --- |
| `open_prompt_injection` | 0.1800 | 0.7533 | 0.7400 | 0.6667 | `1-ASV_combine_attack_only` |
| `tensor_trust_hijack` | 0.4917 | 0.6300 | 0.5633 | 0.5767 | `(HRR+DV)/2` |
| `tensor_trust_extract` | 0.5983 | 0.7067 | 0.6550 | 0.5650 | `(ERR+DV)/2` |
| `mmlu` | 0.5667 | 0.6433 | 0.6133 | 0.6000 | `exact_match_choice` |
| `gsm8k` | 0.7350 | 0.2800 | 0.5100 | 0.5000 | `exact_match_final_number` |
| `ifeval` | 0.6150 | 0.3850 | 0.4850 | 0.4850 | `instruction_compliance` |

Each trained evaluation contains the frozen fixed-ID sample topology: 300 items
for each of the four 300-item benchmarks and 200 items for GSM8K and IFEval.
The per-item results remain available for the manifest-frozen paired bootstrap
analysis in issue #14; this seed summary does not substitute an item-level
interval for the eventual three-seed result.

## 3. Selection and capability gates

The frozen selection rule requires a meaningful visible composite *and* passage
of every capability gate. All three epochs improved the visible composite, but
none was eligible.

| Epoch | Visible composite | OPI delta | TT-hijack delta | TT-extract delta | GSM8K decline | IFEval decline | MMLU decline | Mean normalized retention | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | +0.2733 | +0.5733 | +0.1383 | +0.1083 | 0.4550 | 0.2300 | -0.0767 | 0.7141 | **FAIL** |
| 2 | +0.2294 | +0.5600 | +0.0717 | +0.0567 | 0.2250 | 0.1300 | -0.0467 | 0.8549 | **FAIL** |
| 3 | +0.1794 | +0.4867 | +0.0850 | -0.0333 | 0.2350 | 0.1300 | -0.0333 | 0.8426 | **FAIL** |

The applicable limits are GSM8K decline <= 0.02, IFEval decline <= 0.03, and
mean normalized retention >= 0.98. Every epoch failed GSM8K, IFEval, and mean
retention by a wide margin. The finalized outcome is therefore
`NO_ELIGIBLE_CHECKPOINT`, not a successful mitigation and not a reason to select
the least-bad epoch after the fact.

## 4. Relationship to seed 17

Seed 42 reinforces, but does not by itself complete, the replication picture.
Both completed seeds show strong visible improvement dominated by Open Prompt
Injection and capability regressions that make every checkpoint ineligible.

| Seed | GSM8K decline, epochs 1 / 2 / 3 | IFEval decline, epochs 1 / 2 / 3 | Retention, epochs 1 / 2 / 3 |
| --- | --- | --- | --- |
| 17 | 0.475 / 0.285 / 0.180 | 0.205 / 0.220 / 0.180 | 0.711 / 0.775 / 0.836 |
| 42 | 0.455 / 0.225 / 0.235 | 0.230 / 0.130 / 0.130 | 0.714 / 0.855 / 0.843 |

The repeat is meaningful operational and descriptive evidence: the observed
trade-off is not unique to seed 17. It is not yet a final stability claim; seed
2026 and issue #14's frozen multi-seed analysis remain necessary.

## 5. Resource and feasibility result

| Metric | Measured | Limit / interpretation |
| --- | ---: | --- |
| Seed-42 active wall / GPU-accounted time | 45,873.82 s = 12.7427 h | Within the 24 h per-seed wall limit |
| Training time | 7.942 h | Three completed epochs |
| Evaluation time, epochs 1 / 2 / 3 | 1.383 h / 1.558 h / 1.861 h | Three completed evaluations |
| Cumulative GPU-hours | 34.1828 h | Baseline 6.2553 + seed 17 15.1848 + seed 42 12.7427; within 72 h |
| Remaining GPU budget | 37.8172 h | Available before seed 2026 |
| Peak VRAM | 15.6025 GiB | Above the 15.5 GiB declared allocation, below the physical 16 GiB card: feasibility finding, not an OOM |
| Finalized bundle disk | 10.5746 GiB | Within the 250 GiB limit |

All four ledger rows are completed stages. Their per-row GPU-time field is
unavailable, so the resource artifact honestly derives the active run duration
from their recorded wall seconds; there are no interruption or power-loss gaps
to impute.

## 6. Bounded interpretation and next use

The current evidence supports three claims:

- The recovery-aware seed path completed with the required finalized evidence
  topology and did not contaminate scientific inputs with recovery state.
- This QLoRA intervention substantially improves the visible composite for a
  second seed, overwhelmingly through Open Prompt Injection rather than a
  uniform gain across the prompt-injection axis.
- The same intervention incurs capability losses far beyond the frozen gates,
  producing a valid null selection. It therefore does **not** demonstrate
  capability-preserving prompt-injection mitigation.

It does not support a held-out-generalization claim, because held-out data was
correctly left sealed. It also does not authorize a protocol change, a selected
failed checkpoint, or a final three-seed conclusion. Issue #23 should consume
the updated cumulative value (`34.1827786627532` GPU-hours) and use the same
recovery-aware path for seed 2026. Issue #14 can then compute the frozen paired
bootstrap and descriptive three-seed summaries from finalized evidence only.

## Evidence index

| Artifact | Path |
| --- | --- |
| Training bundle | `runs/training-seed42-20260831-201248-1b487000/` |
| Epoch 1 evaluation | `runs/eval-seed42-epoch1-20260831-201248-1b487000/` |
| Epoch 2 evaluation | `runs/eval-seed42-epoch2-20260831-201248-1b487000/` |
| Epoch 3 evaluation | `runs/eval-seed42-epoch3-20260831-201248-1b487000/` |
| Selection record | `runs/selection-seed42/selection_record.json` |
| Resource comparison | `runs/seed42-resource-comparison/seed_resource_comparison.json` |
| Frozen baseline | `runs/real-baseline-20260829-205020/` |
| Recovery workspace (excluded from evidence) | `recovery/` |

All `runs/` evidence is gitignored under the project evidence contract. This
committed summary indexes the immutable result without copying raw held-out
material or mutable recovery records into the analysis layer.
