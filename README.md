# Prompt-injection fine-tuning on one 16 GB GPU — nine ineligible checkpoints

[![Results](https://img.shields.io/badge/Results-fact_sheet-2563eb)](docs/publication/direction-a/fact-sheet.md)
[![Reproduce analysis](https://img.shields.io/badge/Reproduce_analysis-CPU_only-0f766e)](docs/publication/direction-a/README.md)
[![Protocol](https://img.shields.io/badge/Protocol-recorded-64748b)](protocol/manifest.json)
[![License: MIT](https://img.shields.io/badge/License-MIT-64748b)](LICENSE)

A single-practitioner study of one response-only QLoRA recipe on a 2B model,
using an evaluation setup adapted from Anthropic's **Automated Alignment
Researcher (AAR)** study. The [scientific foundations](#scientific-foundations)
below explain that connection and the role of SecFid in interpreting the results.

**All nine checkpoints increased the visible OPI score but failed the
generation-scored acceptance gates. No checkpoint was selected.**

The acceptance rules were fixed before the main runs. No held-out comparison
was revealed. The result applies to this recipe and evaluation setup.

## Start here

- **Understand the study:** [reader's guide](docs/START_HERE.md).
- **Inspect the result:** [checked fact sheet](docs/publication/direction-a/fact-sheet.md).
- **Check what the scores mean:** [scoring audit](docs/publication/direction-a/scorer-fact-sheet.md).
- **Recalculate the tables without a GPU:** [small reproduction package](docs/publication/direction-a/README.md).

These checked materials supersede conflicting interpretations in older summaries.

## Scientific foundations

Two publications provide the starting point and the interpretive framework for
this independent study:

1. **Original inspiration and evaluation setting — Chen Yueh-Han, Jiaxin Wen,
   and Jan Hendrik Kirchner (2026),
   [Automated Researchers Can Mitigate Well-Characterized Alignment Failures](https://alignment.anthropic.com/2026/automated-alignment-researchers/).**
   This study inspired the project and supplied its evaluation setting through
   the [pinned upstream harness](https://github.com/YuehHanChen/automated_alignment_researcher/tree/1899ad64fbfbc65790d259471cc4bf4de9437aa9).
   Our experiment uses the same prompt-injection target, Qwen3.5-2B, on a
   consumer GPU. It tests one recipe, not the paper's iterative method search.

2. **Framework for interpreting the scores — Mitchell Hermon, Rahul Gupta,
   Weitong Ruan, Ekraam Sabir, and Haohan Wang (2026),
   [Security–Fidelity Tradeoffs: The Hidden Cost of Prompt Injection Defense (SecFid)](https://arxiv.org/abs/2606.30783).**
   SecFid distinguishes executing an injected instruction, processing it as task
   data, and ignoring it. It informed our [scoring audit](docs/publication/direction-a/scorer-fact-sheet.md).
   It cannot explain these checkpoints' behavior: their generated outputs were
   not retained.

---

## Result in one table

| | |
|---|---|
| Model | `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Intervention | response-only SFT via QLoRA (NF4 4-bit, bf16 compute), LoRA r=16 α=32 on all seven projections, 3 epochs over 5,000 examples |
| Runs | nominal seeds `17`, `42`, `2026` — 3 runs × 3 epochs = **9 checkpoints**, sharing one baseline |
| Selection outcome | **`NO_ELIGIBLE_CHECKPOINT` in all three runs** (`selected_checkpoint_digest: null`) |
| Held-out (InjecAgent) | Reveal remained `NEVER_AUTHORIZED`; no baseline-versus-trained comparison was revealed |
| Hardware | one RTX 4080, 16 GiB physical |
| Compute | **47.3381 GPU-accounted hours** for the main baseline plus three runs; the ≈59.87 figure also includes later diagnostic/ablation work |
| Peak VRAM | 15.663 GiB against a declared 15.5 GiB allocation — **over budget, inside the card**, reported as a feasibility finding |

Epoch-3 change against the frozen baseline, in score units (0.01 = 1 percentage point):

| Benchmark | Evaluation modality | seed 17 | seed 42 | seed 2026 |
|---|---|---:|---:|---:|
| `open_prompt_injection` | free-generation, sampled, string-scored | +0.497 | +0.487 | +0.427 |
| `tensor_trust_hijack` | free-generation, sampled, string-scored | +0.050 | +0.085 | +0.073 |
| `tensor_trust_extract` | free-generation, sampled, string-scored | +0.072 | −0.033 | +0.082 |
| `gsm8k` | free-generation, sampled, string-scored | **−0.180** | **−0.235** | **−0.200** |
| `ifeval` | free-generation, sampled, string-scored | **−0.180** | **−0.130** | **−0.145** |
| `mmlu` | likelihood-ranked, no generation | +0.027 | +0.033 | +0.027 |

Frozen gates: GSM8K decline ≤ 0.02, IFEval decline ≤ 0.03, MMLU decline ≤ 0.02,
mean normalized retention ≥ 0.98. **9 of 9 checkpoints fail.**

---

## What is interesting here

1. **The acceptance decision is explicit.** Generation-scored GSM8K and IFEval
   losses exceed the recorded tolerances at every checkpoint, despite higher OPI
   scores. The package shows every run and epoch, without selecting a winner.
2. **The scoring rules leave behavior unresolved.** OPI tests mismatch with the
   injected target, not successful authorized work. Tensor Trust averages an
   attack arm and an authorized-access control; a half-score does not identify
   which arm passed. More items pass both arms in 17 of 18 trained
   checkpoint/benchmark comparisons, but this does not identify every output's
   behavior. Historical outputs were not retained. Timing cannot establish
   answer length, and score categories cannot establish refusal.
3. **Likelihood-ranked MMLU and the generation-scored tests differ.** MMLU alone
   would pass all nine checkpoints, but task content and evaluation format are
   confounded. Run 42 epoch 3 changes correctness on 64/300 MMLU items (21.3%):
   27 become incorrect and 37 become correct, for a net +3.33 percentage points.
   These are correctness transitions, not all answer changes or evidence of a
   defective benchmark.

Detailed evidence and scientific positioning are in the
[Directions A/B knowledge base](docs/publication-directions-a-b-knowledge-base.md).

---

## Scope

This is a case study of one recipe, with three related training trajectories
and one shared baseline. Results are descriptive, not population estimates.
They establish neither an effective defense nor a general failure of QLoRA.
No held-out comparison or adaptive-attack evaluation is reported.
**No model is released.**

---

## Reproducing the analysis

The compact publication package includes visible numeric results and a script
that needs neither model weights nor the private run directories. From a checkout:

```bash
python docs/publication/direction-a/reproduce.py
```

This recalculates the fact sheet and score/decision tables and checks them against
the historical records. Figure and optional paired-interval reproduction commands
are in the [package README](docs/publication/direction-a/README.md).

Full historical report assembly requires local run files, logs, and corpus data;
see the [validation guide](docs/issue-33-validation-guide.md). The public package
reproduces the numeric reanalysis, not training or missing outputs.

Tests:

```bash
python -m pytest tests/ -q
```

Execution bundles (`runs/`, `diagnostics/`, `ablation/`) are not in version control.

---

## Execution limits and corrected disclosures

These disclosures correct earlier summaries:

1. **Runs are not reproducible from the recorded seed.** Adapter initialisation happens
   before the run seed is applied, so that seed does not determine the initial adapter weights.
2. **Training and evaluation used different weight representations.** Training used
   4-bit NF4; merge/evaluation used a bf16 base plus the adapter. The effect of
   this change was not isolated; it does not by itself invalidate evaluation.
3. **No training loss was recorded.** It is computed and discarded — there is no loss curve.
4. **There was no separate training validation split.** The visible benchmark suite
   supplies development/selection measurements; a separate final comparison
   remained sealed. Visible results are not an independent generalization test.
5. **An earlier MMLU disclosure was incorrect.** The pinned upstream also uses
   first-token candidate logits without a chat template. The local default did
   not introduce the claimed scoring-modality deviation.
6. **A declared free-form decoding treatment is read by no code.**
7. **Decoding is applied once globally** rather than per benchmark as upstream documents.

Further environment and protocol deviations: [`protocol/deviations.md`](protocol/deviations.md).
That historical file also contains the superseded MMLU statement. Frozen evidence
was not rewritten to erase these mistakes; the checked fact sheet and knowledge
base record their corrected interpretation. Raw outputs were not retained, and
the integrity report's Tensor Trust category labels cannot diagnose refusal.

Two additional caveats a reader should carry:

- Evaluation samples at `temperature=1.0, top_p=1.0`, one sample per item. The
  single-evaluation decoding noise floor was never measured.
- The GPU-hour figure measures an **unbatched** generation loop (`batch_size: 32` is
  declared in the manifest and consumed by nothing on this path). Treat it as this
  study's cost, not as a hardware floor.

---

## Layout

| Path | Contents |
|---|---|
| `protocol/` | The frozen manifest, its digests, held-out sealing policy, deviations, and separately versioned diagnostic/ablation protocols |
| `runner/` | Stage runners and analysis transforms |
| `training_data/` | Corpus builder — deduplicated against every visible eval set; InjecAgent never touched by construction |
| `analysis/` | Committed fact base: claim report, integrity report, frozen input record, provenance manifest, per-seed summaries |
| `docs/` | One decision record per closed ticket, ADRs, handovers, and the issue-33 framing set |
| `tests/` | Offline tests |
| `RESEARCH_PLAN.md` | The running project log |
| `CITATION.cff` | Citation metadata for this research software and its scientific references |
| `LICENSE` / `LICENSE_SCOPE.md` | MIT license for original contributions and third-party scope notes |

---

## Citation and license

If you use this code or its analysis, cite the repository using
[`CITATION.cff`](CITATION.cff), and record the commit used. Cite the related papers
above when discussing their methods or findings. This repository is a research
software artifact; it does not claim a separate peer-reviewed publication or DOI.

Original project code and documentation are available under the
[MIT License](LICENSE). Third-party code, datasets, model weights, and imported
assistant instructions retain their own terms; see [license scope](LICENSE_SCOPE.md).

---

## Status

The three-run experiment is complete. The final publication scope is this
README, the checked fact sheets, and the numeric reanalysis package: the
checkpoint rejection result and scoring audit. Behavioral explanations remain
unresolved; new experiments and a separate website are outside this scope.

The [scope decision](docs/adr/0002-issue-33-claim-framing.md) supersedes the earlier
publication plan. It does not certify the old issue #33 checklist as completed.
