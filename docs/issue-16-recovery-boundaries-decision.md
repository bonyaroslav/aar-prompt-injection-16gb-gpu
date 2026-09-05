# Issue 16 decision record: seed-17 recovery boundaries

**Issue:** [#16](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/16)
**Decision date:** 2026-08-31
**Protocol:** `phase1-2026-08-29` (`protocol/manifest.json`)
**Scope:** records the completed Issue #12 seed-17 evidence and selects recovery
boundaries. It does not change a frozen protocol value, rerun evidence, or read
held-out InjecAgent material.

## Decision

- **Training:** recover only after a **completed epoch**, including its
  successfully merged checkpoint. Do not claim mid-epoch recovery.
- **Checkpoint merge:** treat each epoch's merge as a **whole stage**; the
  resulting merged checkpoint is the durable boundary consumed by evaluation.
- **Epoch evaluations:** recover at the **whole-checkpoint evaluation** boundary
  for each of epochs 1, 2, and 3. No checkpoint evaluation exceeded seven
  hours, so benchmark-item or small-batch recovery is not required by the
  specification.
- **Individual benchmarks:** no separately materialized benchmark stage has
  timing, GPU-time, peak-VRAM, or output-size telemetry. They remain inside the
  whole-checkpoint evaluation boundary; no finer boundary is adopted.
- **Selection:** recover at the **whole selection stage** boundary, represented
  by the finalized `selection_record.json`.
- **Reveal:** no reveal was produced. The finalized selection has no eligible
  checkpoint, so no reveal boundary is selected here.

These are the coarsest boundaries supported by the completed evidence and fit a
roughly 6–7-hour operating session without changing the frozen protocol.

## Measured evidence and availability

`gpu.csv` records elapsed telemetry span and observed device memory, not a
stage-specific GPU-time accounting. Accordingly, its duration values are
reported as wall-time spans and every unavailable GPU-time field remains
explicitly unavailable.

| Unit | Wall-time evidence | GPU time | Observed peak VRAM | Output size | Session / frozen-limit comparison |
| --- | ---: | --- | ---: | ---: | --- |
| Seed-17 training (all epochs) | 33,702.843 s = 9.362 h | unavailable per stage | 16,039 MiB = 15.663 GiB | 11,353,067,916 B | Exceeds 6–7 h as one stage; seed total remains under the 24 h cap. Peak exceeds the 15.5 GiB declared allocation. |
| Training epoch 1 | about 2.8–3.3 h (committed summary range) | unavailable | unavailable separately | included in training output | Fits 6–7 h; completed epoch is the recovery boundary. |
| Training epoch 2 | about 2.8–3.3 h (committed summary range) | unavailable | unavailable separately | included in training output | Fits 6–7 h; completed epoch is the recovery boundary. |
| Training epoch 3 | about 2.8–3.3 h (committed summary range) | unavailable | unavailable separately | included in training output | Fits 6–7 h; completed epoch is the recovery boundary. |
| Epoch-1 checkpoint merge | unavailable | unavailable | unavailable | 3,783,692,158 B | Whole merge is retained with its completed epoch; no measured duration supports a finer decision. |
| Epoch-2 checkpoint merge | unavailable | unavailable | unavailable | 3,783,692,158 B | Whole merge is retained with its completed epoch; no measured duration supports a finer decision. |
| Epoch-3 checkpoint merge | unavailable | unavailable | unavailable | 3,783,692,158 B | Whole merge is retained with its completed epoch; no measured duration supports a finer decision. |
| Epoch-1 evaluation | 6,651.762 s = 1.848 h | unavailable per stage | 16,038 MiB = 15.662 GiB | 531,645 B | Fits 6–7 h; whole-checkpoint recovery is adequate. Peak exceeds declared allocation. |
| Epoch-2 evaluation | 6,983.926 s = 1.940 h | unavailable per stage | 13,546 MiB = 13.229 GiB | 552,760 B | Fits 6–7 h; whole-checkpoint recovery is adequate. |
| Epoch-3 evaluation | 7,251.643 s = 2.014 h | unavailable per stage | 13,838 MiB = 13.514 GiB | 558,851 B | Fits 6–7 h; whole-checkpoint recovery is adequate. |
| Each individual benchmark within each evaluation | unavailable separately; logs record only completion and sample counts | unavailable | unavailable | unavailable separately | No independent artifact or telemetry supports a smaller recovery boundary. |
| Selection | unavailable | unavailable | unavailable | 2,307 B (`selection_record.json`) | Whole finalized record is the boundary. |
| Reveal | not produced | not produced | not produced | not produced | No selected checkpoint exists; no reveal-stage decision is made. |

The evaluation logs confirm all six benchmarks completed in each epoch:
`open_prompt_injection` (300), `tensor_trust_hijack` (300),
`tensor_trust_extract` (300), `mmlu` (300), `gsm8k` (200), and `ifeval`
(200). They do not record benchmark-level resource measurements; none are
inferred here.

## Cumulative resource decision

| Measure | Recorded value | Frozen limit / target | Result |
| --- | ---: | ---: | --- |
| Baseline GPU-hours | 6.2553 h (reported as 6.26 h) | included in 72 h cumulative cap | consumed before seed 17 |
| Completed seed-17 GPU-hours | 15.1848 h | 24 h per-seed wall cap; 72 h cumulative GPU cap | within both time caps |
| Cumulative GPU-hours through seed 17 | 21.4401 h | 72 h | within cap; 50.5599 h remains before later measured attempts |
| Seed-17 wall time | 54,665.264 s = 15.185 h | 24 h | within cap, but not a 6–7 h session |
| Finalized seed evidence bundle | 10.5749 GiB | 250 GB | within cap |
| Highest observed device VRAM | 15.6631 GiB | 15.5 GiB declared allocation | feasibility finding: exceeds declared allocation by 0.1631 GiB |

The only #12 attempt with durable aggregate resource accounting is the completed
seed-17 run above. No separately recorded interrupted or failed #12 attempt is
available to add; no zero or estimate is substituted.

## Evidence basis

- `runs/seed17-resource-comparison-20260830-071553/seed_resource_comparison.json`:
  final seed, baseline, cumulative, bundle-size, and peak-VRAM accounting.
- `runs/training-seed17-20260830-071553/gpu.csv` and its three merged checkpoint
  directories: training telemetry span, observed peak, and output bytes.
- `runs/eval-seed17-epoch{1,2,3}-20260830-071553/gpu.csv`, `execution.log`, and
  bundle sizes: evaluation telemetry spans, observed peaks, completed benchmark
  counts, and output bytes.
- `runs/selection-seed17-20260830-071553/selection_record.json`: finalized
  null selection and final-record size.
- `analysis/seed17-outcomes-summary.md` and `docs/issue-12-analysis-for-further-work.md`:
  committed epoch-duration range and the prior evidence interpretation.
  (`docs/issue-12-analysis-summary.md`, a byte-identical duplicate of the latter,
  was deleted 2026-09-05; recover from git history if needed.)

## Protocol compatibility

This decision preserves the manifest's training, evaluation, selection,
resource, and held-out rules. It introduces no new fallback and no finer
recovery guarantee. The Issue #15 condition for item/small-batch evaluation
recovery is not triggered because the longest recorded checkpoint evaluation is
2.014 hours, below seven hours.
