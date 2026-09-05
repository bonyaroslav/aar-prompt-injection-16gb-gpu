# Prompt-injection fine-tuning on one 16 GB GPU — a preregistered null result

A single-practitioner replication-style study: can a transparent response-only
QLoRA fine-tune reduce a 2B model's prompt-injection susceptibility on one consumer
GPU, without paid judges, and without breaking the model?

**The answer, across three independent runs, is no — and the protocol said so
before anyone looked.**

All nine trained checkpoints failed a frozen capability contract whose binding gates were generation-scored.
The contract was fixed before the first GPU-hour was spent. No checkpoint was
selected. The held-out benchmark was never unsealed.

---

## Result in one table

| | |
|---|---|
| Model | `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Intervention | response-only SFT via QLoRA (NF4 4-bit, bf16 compute), LoRA r=16 α=32 on all seven projections, 3 epochs over 5,000 examples |
| Runs | seeds `17`, `42`, `2026` — 3 seeds × 3 epochs = **9 checkpoints** |
| Selection outcome | **`NO_ELIGIBLE_CHECKPOINT` in all three runs** (`selected_checkpoint_digest: null`) |
| Held-out (InjecAgent) | `NEVER_AUTHORIZED` — 200 candidates sealed before execution, never opened |
| Hardware | one RTX 4080, 16 GiB physical |
| Compute | ≈59.87 of a declared 72 GPU-hour cap |
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

Three observations, in decreasing order of how well the evidence supports them.
Full argument and prior-art positioning:
[`docs/handover/publication-analysis-final.md`](docs/handover/publication-analysis-final.md).

1. **The safety gain tracks whether the benchmark can detect degeneracy.** Open
   Prompt Injection awards a point for *any* output that is not the attacker's
   target label — an empty string scores full marks. Tensor Trust additionally
   checks the model still does its job. The benchmark without that control arm gains
   +31 to +57 pp; the ones with it gain +2.6 to +15.3 pp, and **that gain declines
   monotonically with every training epoch in all three runs.** Meanwhile generation
   times collapsed from 45.96 s/item to 6.2–8.7 s/item — the model stopped writing
   long answers.
2. **MMLU did not hold steady, it churned.** Up to 21.3 % of MMLU items changed
   answer by epoch 3 while the aggregate reads +2.7 pp and is not statistically
   distinguishable from zero. An aggregate tolerance of ±2 pp cannot see a model
   that rewrote a fifth of its answers.
3. **A gate built only from the likelihood-ranked benchmark passes all nine
   checkpoints** that the generation-scored gates rejected. Stated as an existence
   proof only: there is exactly one likelihood-ranked benchmark in this suite and it
   is also the only pure-recall one, so evaluation modality and task type are
   confounded here and cannot be separated from this data.

---

## What this study does **not** claim

- No efficacy claim. This is not a working defence, and no checkpoint is recommended
  for any use.
- No held-out generalization claim. InjecAgent was never scored; no such number
  exists and none may be inferred.
- No adaptive-attack claim. No attacker who knows about this intervention was ever run.
- No claim that response-only QLoRA fails in general, and no claim about any other
  model, scale, corpus, or recipe.
- No population-level inference from three runs.

**No model is released.** A checkpoint the protocol rejected is not a shippable artifact.

---

## Reproducing the analysis

The full analysis chain is offline and deterministic. From a checkout:

```bash
python -m runner.publication_gate_run --dump-reports analysis --out analysis/publication-provenance-manifest.json
```

This regenerates every committed report byte-identically. Expected SHA-256 values for
the four fact-base artifacts are recorded in
[`docs/adr/0002-issue-33-claim-framing.md`](docs/adr/0002-issue-33-claim-framing.md).
Hand-verification procedure:
[`docs/issue-33-validation-guide.md`](docs/issue-33-validation-guide.md).

Tests:

```bash
python -m pytest tests/ -q
```

Execution evidence (`runs/`, `diagnostics/`, `ablation/`) is deliberately not in
version control — it is ~31.7 GiB of checksummed bundles. Everything needed to check
the reported numbers is committed under `analysis/`.

---

## Known defects, disclosed up front

Published as a first-class section rather than a footnote
(`analysis/attempt1-integrity-report.json` → `integrity_records.reproducibility_disclosure`):

1. **Runs are not reproducible from the recorded seed.** Adapter initialisation happens
   before the run seed is applied, so the seed controls data shuffling and dropout only.
2. **The evaluated model is not the trained model.** Training used 4-bit NF4; merge and
   evaluation used bf16 weights.
3. **No training loss was recorded.** It is computed and discarded — there is no loss curve.
4. **There is no validation split.** The only model-selection signal is the visible
   benchmark suite the study reports; the design is selection-on-test.
5. **The frozen manifest names a multiple-choice scorer the pinned upstream does not use.**
   This study honours the manifest value, so MMLU alone is evaluated in a different
   modality from every other benchmark.
6. **A declared free-form decoding treatment is read by no code.**
7. **Decoding is applied once globally** rather than per benchmark as upstream documents.

Further environment and protocol deviations: [`protocol/deviations.md`](protocol/deviations.md).

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

Start here: [`docs/handover/publication-analysis-final.md`](docs/handover/publication-analysis-final.md).

---

## Status

The frozen replication set `[17, 42, 2026]` is complete. The analysis chain
(issues #27–#32) is closed. Issue #33 — the results document and evidence package —
is open pending two cheap follow-up measurements and five wording decisions recorded
in [`docs/adr/0002-issue-33-claim-framing.md`](docs/adr/0002-issue-33-claim-framing.md).
