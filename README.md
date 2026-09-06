# Prompt-injection fine-tuning on one 16 GB GPU — nine ineligible checkpoints

A single-practitioner study of one response-only QLoRA recipe on a 2B model,
using an evaluation setup adapted from Anthropic's AAR study.

**All nine checkpoints increased the visible OPI score but failed the
generation-scored acceptance gates. No checkpoint was selected.**

All nine trained checkpoints failed a frozen capability contract whose binding gates were generation-scored.
The contract was fixed before the main runs. No held-out comparison was revealed.
This is a negative selection result for one recipe, not a finding that training
had no effect or that fine-tuning fails in general.

Start with the [checked fact sheet](docs/publication/direction-a/fact-sheet.md)
and [small reproduction package](docs/publication/direction-a/README.md).
They supersede conflicting interpretations in older narrative summaries.

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

Epoch-3 change against the frozen baseline, per run:

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

## What this study does **not** claim

- No efficacy claim. This is not a working defence, and no checkpoint is recommended
  for any use.
- No held-out generalization claim. No baseline-versus-trained InjecAgent
  comparison was revealed, and none may be inferred from the visible scores.
- No adaptive-attack claim. No attacker who knows about this intervention was ever run.
- No claim that response-only QLoRA fails in general, and no claim about any other
  model, scale, corpus, or recipe.
- No population-level inference from three runs.

**No model is released.** A checkpoint the protocol rejected is not a shippable artifact.

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

The older `runner.publication_gate_run` path assembles the full reports from
gitignored run metrics, logs, telemetry, recovery records, and corpus files.
It cannot reproduce those reports from committed aggregates alone. Historical
report identities remain recorded in the
[ADR](docs/adr/0002-issue-33-claim-framing.md); the
[validation guide](docs/issue-33-validation-guide.md) describes the local-input path.

Tests:

```bash
python -m pytest tests/ -q
```

Execution evidence (`runs/`, `diagnostics/`, `ablation/`) is deliberately not in
version control — the main bundle snapshot is ~31.7 GiB. The new publication
folder provides the small visible-score subset needed for its own reanalysis.
It does not recreate missing outputs or retrain the original models.

---

## Execution limits and corrected disclosures

The frozen integrity report contains historical disclosures and interpretations.
The corrected distinctions for publication are:

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
| `runner/` | Stage runners and pure analysis transforms (~8.4k lines) |
| `training_data/` | Corpus builder — deduplicated against every visible eval set; InjecAgent never touched by construction |
| `analysis/` | Committed fact base: claim report, integrity report, frozen input record, provenance manifest, per-seed summaries |
| `docs/` | One decision record per closed ticket, ADRs, handovers, and the issue-33 framing set |
| `tests/` | 30 offline test modules |
| `RESEARCH_PLAN.md` | The running project log |

Start here: [checked fact sheet](docs/publication/direction-a/fact-sheet.md).

---

## Status

The frozen replication set `[17, 42, 2026]` is complete. The analysis chain
(issues #27–#32) is closed. Issue #33 — the results document and evidence package —
remains open. The Direction A fact sheet and reader reanalysis package now exist.
The broader publication decisions remain in
[`docs/adr/0002-issue-33-claim-framing.md`](docs/adr/0002-issue-33-claim-framing.md).
New model measurements are needed for a behavioral explanation, not for the
bounded account of the existing rejection decision.
