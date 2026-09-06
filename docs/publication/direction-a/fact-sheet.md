# Checked fact sheet — Direction A

Prepared 6 September 2026 from the saved per-item scores. This is the factual reference for drafting, not a new experiment or a certification.

**Main finding:** all nine checkpoints were ineligible under the recorded acceptance rules. Their generation-scored GSM8K and IFEval losses exceeded the allowed limits, despite higher OPI scores. No checkpoint was selected; no held-out comparison was revealed.

**In ordinary language:** the update improved one test but made other required work worse. The preset rule therefore rejected every candidate. The results do not establish why the model's behavior changed or whether a different training recipe would qualify.

## What was tested

- Model: `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Recipe: response-only QLoRA; NF4 training with bf16 compute; LoRA rank 16, alpha 32; 5,000 examples; learning rate 0.0002; three epochs. Merge/evaluation used bf16 base weights plus the trained adapter.
- Runs: nominal seeds 17, 42, 2026. Three related checkpoints per run, with one shared baseline. Initialization precedes the recorded run seed, so these are not fully reproducible seed-controlled trials.
- Evaluation: 300 fixed items each for OPI, each Tensor Trust benchmark, and MMLU; 200 each for GSM8K and IFEval. Tensor Trust has two arms per item. MMLU ranks first-token logits; the other benchmarks score sampled text (temperature 1, top-p 1).

## The acceptance rule

A checkpoint is eligible only if all three task-loss limits and the mean-retention floor pass. The code selects the highest visible composite among eligible candidates. A separate +5-point threshold labels meaningful visible improvement; it does not override failed gates.

| Check | Maximum allowed loss / required retention | Minimum score from this baseline | Observed trained range |
|---|---:|---:|---:|
| GSM8K (generation-scored) | 2 percentage points | 71.50% | 26.00–55.50% |
| IFEval (generation-scored) | 3 percentage points | 58.50% | 38.50–48.50% |
| MMLU (likelihood-ranked) | 2 percentage points | 54.67% | 59.33–64.33% |
| Mean normalized retention | At least 98% | — | 71.07–85.49% |

Retention means the average of each task's trained score divided by its baseline score; it is not an average of the three raw accuracies.

## All scores, with no selected winner

Values are percentages. The table and figure show every recorded epoch. Each cell is a mean over the fixed items, not a measure of deployment performance.

| Run / epoch | OPI | TT hijack | TT extraction | GSM8K | IFEval | MMLU | Eligible? |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | 18.00 | 49.17 | 59.83 | 73.50 | 61.50 | 56.67 | Reference |
| 17 / 1 | 50.67 | 54.33 | 74.17 | 26.00 | 41.00 | 63.00 | No |
| 17 / 2 | 62.67 | 55.33 | 70.50 | 45.00 | 39.50 | 60.67 | No |
| 17 / 3 | 67.67 | 54.17 | 67.00 | 55.50 | 43.50 | 59.33 | No |
| 42 / 1 | 75.33 | 63.00 | 70.67 | 28.00 | 38.50 | 64.33 | No |
| 42 / 2 | 74.00 | 56.33 | 65.50 | 51.00 | 48.50 | 61.33 | No |
| 42 / 3 | 66.67 | 57.67 | 56.50 | 50.00 | 48.50 | 60.00 | No |
| 2026 / 1 | 49.00 | 61.33 | 78.33 | 29.00 | 43.50 | 63.33 | No |
| 2026 / 2 | 55.00 | 61.17 | 70.67 | 44.50 | 41.50 | 61.00 | No |
| 2026 / 3 | 60.67 | 56.50 | 68.00 | 53.50 | 47.00 | 59.33 | No |

**One calculation you can check:** run 17, epoch 3 has GSM8K 55.50% versus baseline 73.50%. The loss is 18 percentage points, exceeding the 2-point limit. This is one worked example, not a selected checkpoint.

![Six benchmark panels across epochs and three runs. OPI scores rise above baseline; all GSM8K and IFEval points are below their required floors. MMLU stays above its floor.](figure.png)

**Figure caption:** observed scores for the shared baseline and all nine checkpoints. Each line describes a training run, not independent trials at every epoch. The shaded regions mark scores below the declared individual task floors. No confidence bands are drawn; the recorded paired intervals are supplied separately. A score is a benchmark's own metric, not a guarantee of successful authorized work.

## Costs, uncertainty, and limits

- Main baseline plus three runs: **47.3381 GPU-accounted hours**, against a 72-hour budget. Some accounting uses active wall time; evaluation was unbatched. This is this implementation's cost, not a hardware minimum.
- Hardware: NVIDIA RTX 4080, 16 GB. Peak memory **15.663 GiB**, above the declared 15.5 GiB allocation but within the card.
- The approximately 59.87-hour figure additionally includes later diagnostic and ablation work. It is not the original three-run total and omits at least a separately recorded diagnostic smoke.
- The 95% paired intervals in `paired-intervals.csv` are the historical 10,000-resample estimates over fixed item pairs. They do not estimate training-population variability or repeated-decoding noise. Individual intervals are not a simultaneous guarantee across all comparisons.
- Historical outputs were not retained. Higher OPI scores do not identify refusal, suppression, or successful completion of the legitimate task. Generalization was not established because no held-out comparison was revealed.
- MMLU uses first-token logits in both the pinned upstream and the local default. Older prose claiming a scoring-modality deviation is incorrect. Task and evaluation-format differences remain confounded.

## What each source proves

- [visible-scores.csv](visible-scores.csv): the exported observations used to recalculate every mean and gate.
- [context.json](context.json): configuration, original file hashes, recorded selections, reference means, intervals, and resource source values. Original paths are provenance labels; reproduction does not open them.
- [checkpoint-decisions.csv](checkpoint-decisions.csv): recalculated losses, retention, and eligibility for all nine candidates.
- [paired-intervals.csv](paired-intervals.csv): uncertainty for the paired changes; `--verify-bootstrap` independently recalculates all 54 intervals from the exported observations.
- [README](README.md): a small-step explanation and runnable reproduction commands.

This supports a bounded engineering case study. It does not establish a general failure of fine-tuning, a successful deployed defense, or the cause of the observed changes. The figure and arithmetic can be reproduced from this folder; retraining cannot be reproduced from the numeric-score export.
