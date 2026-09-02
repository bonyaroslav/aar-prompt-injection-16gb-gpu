# Seed 2026 outcomes summary (issue #23)

**Run date:** 2026-09-01 to 2026-09-02  
**Protocol:** `phase1-2026-08-29` (`protocol/manifest.json`)  
**Seed:** `2026` (the third frozen replication seed)  
**Upstream:** commit `1899ad64fbfbc65790d259471cc4bf4de9437aa9`, tree
`00f1edb9193487e7e306177709b1760be180d7ac`

This is the durable analysis record for issue #23's real seed-2026 evidence.
It reports the finalized, checksummed artifacts and gives only
protocol-bounded interpretation. The frozen selection rule finalized a null
selection, so held-out InjecAgent data remained sealed and no held-out reveal
was authorized or performed.

## 1. Finalized evidence and recovery status

Issue #23 completed the third frozen replication seed through the
recovery-aware orchestration. All required training, evaluation, selection,
and resource artifacts are present and finalized.

| Check | Result |
| --- | --- |
| Training | Success; all three epochs completed |
| OOM fallback | Not applied; every epoch used the frozen 2,048-token setting |
| Evaluations | Three completed, one for each epoch |
| Finalized selection | `selected_epoch: null`; `selected_checkpoint_digest: null` |
| Resource artifact | Finalized at the stable `seed2026-resource-comparison` id |
| Attempts | 4 completed, 0 interrupted, 0 declared unavailable intervals |
| Held-out reveal | Not created; null selection prohibits it |
| Artifact integrity | Training, all three evaluations, and the resource artifact passed their recorded SHA-256 checksums |

The selection-record SHA-256 is
`8df462a4548fe652660409ef76b2b987a7794a0904f9b400cf8bdf1ba10a0d23`.
Recovery state is operational state under `recovery/`, not a finalized analysis
input.

## 2. Per-benchmark scores versus the frozen baseline

| Benchmark | Baseline | Epoch 1 | Epoch 2 | Epoch 3 | Metric |
| --- | ---: | ---: | ---: | ---: | --- |
| `open_prompt_injection` | 0.1800 | 0.4900 | 0.5500 | 0.6067 | `1-ASV_combine_attack_only` |
| `tensor_trust_hijack` | 0.4917 | 0.6133 | 0.6117 | 0.5650 | `(HRR+DV)/2` |
| `tensor_trust_extract` | 0.5983 | 0.7833 | 0.7067 | 0.6800 | `(ERR+DV)/2` |
| `mmlu` | 0.5667 | 0.6333 | 0.6100 | 0.5933 | `exact_match_choice` |
| `gsm8k` | 0.7350 | 0.2900 | 0.4450 | 0.5350 | `exact_match_final_number` |
| `ifeval` | 0.6150 | 0.4350 | 0.4150 | 0.4700 | `instruction_compliance` |

Each trained evaluation contains the frozen fixed-ID sample topology: 300 items
for each of the four 300-item benchmarks and 200 items for GSM8K and IFEval.
The per-item results remain available for the manifest-frozen paired bootstrap
analysis; this seed summary does not substitute an item-level interval for the
eventual three-seed result.

## 3. Selection and capability gates

The frozen selection rule requires a meaningful visible composite *and* passage
of every capability gate. All three epochs improved the visible composite, but
none was eligible.

| Epoch | Visible composite | OPI delta | TT-hijack delta | TT-extract delta | GSM8K decline | IFEval decline | MMLU decline | Mean normalized retention | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | +0.2056 | +0.3100 | +0.1217 | +0.1850 | 0.4450 | 0.1800 | -0.0667 | 0.7398 | **FAIL** |
| 2 | +0.1994 | +0.3700 | +0.1200 | +0.1083 | 0.2900 | 0.2000 | -0.0433 | 0.7856 | **FAIL** |
| 3 | +0.1939 | +0.4267 | +0.0733 | +0.0817 | 0.2000 | 0.1450 | -0.0267 | 0.8464 | **FAIL** |

The applicable limits are GSM8K decline <= 0.02, IFEval decline <= 0.03, and
mean normalized retention >= 0.98. Every epoch failed GSM8K, IFEval, and mean
retention by a wide margin; MMLU was above baseline at every epoch. The
finalized outcome is therefore `NO_ELIGIBLE_CHECKPOINT`, not a successful
mitigation and not a reason to select the least-bad epoch after the fact.

## 4. Relationship to seeds 17 and 42

The third completed seed strengthens the descriptive replication picture: all
three seeds produced visible-composite improvements alongside capability losses
that made every evaluated checkpoint ineligible. In seed 2026, Open Prompt
Injection improved monotonically, while Tensor Trust gains were largest at epoch
1 and declined thereafter. GSM8K and IFEval partially recovered by epoch 3 but
remained far below their respective gates.

| Seed | Visible composite, epochs 1 / 2 / 3 | GSM8K decline, epochs 1 / 2 / 3 | IFEval decline, epochs 1 / 2 / 3 | Retention, epochs 1 / 2 / 3 |
| --- | --- | --- | --- | --- |
| 17 | +0.1739 / +0.2050 / +0.2061 | 0.475 / 0.285 / 0.180 | 0.205 / 0.220 / 0.180 | 0.711 / 0.775 / 0.836 |
| 42 | +0.2733 / +0.2294 / +0.1794 | 0.455 / 0.225 / 0.235 | 0.230 / 0.130 / 0.130 | 0.714 / 0.855 / 0.843 |
| 2026 | +0.2056 / +0.1994 / +0.1939 | 0.445 / 0.290 / 0.200 | 0.180 / 0.200 / 0.145 | 0.740 / 0.786 / 0.846 |

At epoch 3, all three seeds retain a positive visible composite (+0.1794 to
+0.2061) and fail every capability gate. Their similar endpoint retention
(0.8365, 0.8426, and 0.8464) is descriptive evidence of a repeatable observed
trade-off under this protocol, not a final stability or population-level claim.
The protocol's fixed-example paired bootstrap and the planned three-seed
descriptive analysis remain necessary for the specified final summary.

## 5. Resource and feasibility result

| Metric | Measured | Limit / interpretation |
| --- | ---: | --- |
| Seed-2026 active wall / GPU-accounted time | 47,359.22 s = 13.1553 h | Within the 24 h per-seed wall limit |
| Training time | 8.0100 h | Three completed epochs |
| Evaluation time, epochs 1 / 2 / 3 | 1.4443 h / 1.7770 h / 1.9230 h | Three completed evaluations |
| Cumulative GPU-hours | 47.3381 h | Baseline 6.2553 + seed 17 15.1848 + seed 42 12.7427 + seed 2026 13.1553; within 72 h |
| Remaining GPU budget | 24.6619 h | Remaining after all three frozen seeds |
| Peak VRAM | 15.6289 GiB | Above the 15.5 GiB declared allocation, below the physical 16 GiB card: feasibility finding, not an OOM |
| Finalized bundle disk | 10.5743 GiB | Within the 250 GiB limit |

All four ledger rows are completed stages. Their per-row GPU-time field is
unavailable, so the resource artifact honestly derives active run duration from
their recorded wall seconds; there are no interruption or power-loss gaps to
impute.

## 6. Bounded interpretation and next use

The completed evidence supports four claims:

- The recovery-aware third-seed path completed with the required finalized
  evidence topology and did not contaminate scientific inputs with recovery
  state.
- The frozen QLoRA intervention improved the visible composite at every epoch
  in all three completed seeds, with Open Prompt Injection a substantial
  contributor to each seed's gains.
- All nine trained checkpoints failed the prespecified GSM8K, IFEval, and
  mean-retention gates, producing three valid null selections.
- The intervention therefore does **not** demonstrate capability-preserving
  prompt-injection mitigation under this frozen protocol.

The evidence does not support a held-out-generalization claim, because
InjecAgent was correctly left sealed. It does not authorize protocol changes,
post-hoc checkpoint selection, or causal conclusions about the capability
losses. A final analysis may now use only finalized baseline and three-seed
visible/capability artifacts to run the manifest-specified paired bootstrap and
descriptive cross-seed summaries.

## Evidence index

| Artifact | Path |
| --- | --- |
| Training bundle | `runs/training-seed2026-20260901-112915-bf0809d1/` |
| Epoch 1 evaluation | `runs/eval-seed2026-epoch1-20260901-112915-bf0809d1/` |
| Epoch 2 evaluation | `runs/eval-seed2026-epoch2-20260901-112915-bf0809d1/` |
| Epoch 3 evaluation | `runs/eval-seed2026-epoch3-20260901-112915-bf0809d1/` |
| Selection record | `runs/selection-seed2026/selection_record.json` |
| Resource comparison | `runs/seed2026-resource-comparison/seed_resource_comparison.json` |
| Frozen baseline | `runs/real-baseline-20260829-205020/` |
| Recovery workspace (excluded from evidence) | `recovery/` |

All `runs/` evidence is gitignored under the project evidence contract. This
committed summary indexes the immutable result without copying raw held-out
material or mutable recovery records into the analysis layer.
