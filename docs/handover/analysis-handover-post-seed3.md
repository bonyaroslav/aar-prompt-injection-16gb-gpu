# Handover — Seed 3 onward, for the publication/demand analysis assistant

**Written:** 2026-09-03. Repo `origin/master` @ `d5e4605` (pushed).

**Audience:** an assistant that already holds (a) demand signals from scientific
sites, Reddit and Hugging Face, (b) which publication styles are currently in
demand, (c) accumulated tips — and that already knows **Seed 1 (17)** and
**Seed 2 (42)** results, but nothing after.

**What this document is:** the delta. Seed 3, everything built afterwards, every
artifact with a repo-relative path, the observations that matter, and the one
decision still open. It carries no demand or venue analysis — that is your side
of the table.

**What you are being asked for:** help decide how this result should be framed
and where it should go, using the option space in
`docs/adr/0002-issue-33-claim-framing.md`.

---

## 1. Start here — the five-line delta

1. **Seed 3 (2026) completed.** Same outcome as seeds 17 and 42:
   `NO_ELIGIBLE_CHECKPOINT`. The frozen replication set `[17, 42, 2026]` is
   **complete** — 3 seeds, 9 checkpoints, 9 capability-gate failures, 3 null
   selections, held-out benchmark never unsealed.
2. **A full analysis chain was then built and closed** (#27–#32): frozen input
   manifest, claim tables + statistics, failure-mode/integrity reports, a
   chat-mode confound diagnostic, a clean-corpus ablation, and publication
   provenance + claim-language gates.
3. **Two follow-up experiments changed the story.** #30 (diagnostic) and #31
   (ablation) both landed and both bear directly on how the finding may be
   worded. See §4.
4. **One issue remains open — #33**, the results document and evidence package.
   It is **deliberately paused** on a framing decision the maintainer wants to
   take slowly.
5. **Your input is wanted at exactly that pause point.** The evidence is frozen
   and independently checkable; only the wording is undecided.

---

## 2. Seed 3 (seed 2026) — the result you are missing

Closed as issue #23, commit `88bf966`. Decision record:
`docs/issue-23-seed-2026-execution-decision.md`. Durable summary:
`analysis/seed2026-outcomes-summary.md`.

Ran through the same recovery-aware split-run workflow as seed 42, in one
session, no OOM fallback, all three frozen epochs plus three evaluations plus a
finalized selection.

| Measure | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | ---: | ---: | ---: |
| Visible composite improvement | +0.2056 | +0.1994 | +0.1939 |
| Open Prompt Injection improvement | +0.3100 | +0.3700 | +0.4267 |
| GSM8K decline | 0.4450 | 0.2900 | 0.2000 |
| IFEval decline | 0.1800 | 0.2000 | 0.1450 |
| Mean normalized capability retention | 0.7398 | 0.7856 | 0.8464 |

Every epoch **failed** the frozen gates (GSM8K decline ≤ 0.02, IFEval ≤ 0.03,
retention ≥ 0.98). Finalized `selected_checkpoint_digest: null`.
13.1553 GPU-hours; peak VRAM 15.6289 GiB. Selection digest
`8df462a4548fe652660409ef76b2b987a7794a0904f9b400cf8bdf1ba10a0d23`.

**Why it matters to you:** the third seed removes "it was a fluke" as a reading.
Three independent runs, same shape, no cherry-picking possible — the selection
rule was frozen in advance and rejected everything.

---

## 3. What was built after Seed 3 — the analysis chain

All closed. Each ticket has a decision record under `docs/`.

| # | What it produced | Decision record |
| --- | --- | --- |
| 24–26 | agent-workflow repair; LF pinning + canonical-digest note; ablation mid-epoch recovery | `docs/issue-26-mid-epoch-training-recovery-decision.md` |
| 27 | frozen input manifest + exclusion allowlist — the checksum-verified input set every later ticket reads from | `docs/issue-27-frozen-input-manifest-decision.md` |
| 28 | claim tables + statistics — the central table, grouped by **evaluation modality** | `docs/issue-28-claim-tables-decision.md` |
| 29 | failure-mode evidence + integrity records — *what broke* and *what a reader must be told* | `docs/issue-29-failure-mode-integrity-decision.md` |
| 30 | chat-mode MMLU confound test (real GPU diagnostic) | `docs/issue-30-chatmode-mmlu-diagnostic-decision.md` |
| 31 | clean-corpus ablation (real GPU, 11.88 h) | `docs/issue-31-corpus-ablation-decision.md` |
| 32 | provenance manifest + claim-language gate | `docs/issue-32-provenance-source-decision.md` |
| **33** | **OPEN** — results document + evidence package | `docs/adr/0002-issue-33-claim-framing.md` |

---

## 4. The two results that change the framing

### 4a. #30 — chat-mode MMLU diagnostic

The headline pattern is that generation-scored benchmarks collapsed while the
one multiple-choice benchmark (MMLU) held. The obvious objection: MMLU is scored
*without* the chat template — outside the interface that was fine-tuned. #30
tested exactly that.

| Model state | MMLU raw mode | MMLU chat mode | Delta |
| --- | ---: | ---: | ---: |
| baseline (untrained) | 0.567 | **0.250** (chance) | −0.317 |
| 9 checkpoints (mean) | ~0.61 | ~0.556 | −0.055 |

**Reading:** the untrained model cannot do MMLU through the chat interface at
all; the fine-tune *repairs* it. Measured fairly, checkpoints lose ~5 pp —
nothing like GSM8K's −18 to −48 pp in the same runs. **The chat-template
confound does not overturn the finding.** But the deeper *scoring-modality*
axis (likelihood ranking vs sampled generation) is still untested.

### 4b. #31 — clean-corpus ablation

Retrained on a corpus with **zero** prompt-injection rows.

| State | OPI | MMLU | GSM8K | IFEval |
| --- | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.180 | 0.567 | 0.735 | 0.615 |
| Clean ablation, epoch 3 | 0.400 | 0.613 | **0.490** | **0.430** |
| Attempt 1 seed 42, epoch 3 | 0.667 | 0.600 | 0.500 | 0.485 |

**The capability collapse survived removing the injection data.** So the attack
examples are not the sufficient cause. Caveat that limits how strongly this can
be stated: single seed, and the corpus was *rebuilt* (different length filter,
different category mix), not simply stripped.

---

## 5. Artifact map — every file, with repo-relative paths

### 5a. Committed evidence you can quote from directly

| Path | Content | Value to you |
| --- | --- | --- |
| `analysis/attempt1-claim-report.json` | per-benchmark baseline/trained means, deltas, 10k-replicate paired bootstrap, exact McNemar, visible composite, cross-run summary | every headline number, machine-readable |
| `analysis/attempt1-integrity-report.json` | failure-mode evidence + integrity records | the *mechanism* evidence and the disclosure list |
| `analysis/attempt1-frozen-input-record.json` | every finalized input bound to a digest | provenance backbone |
| `analysis/publication-provenance-manifest.json` | a receipt for all 1,763 reported numbers | proof no number is orphaned |
| `analysis/seed17-outcomes-summary.md` | Seed 1 durable summary | you have this already |
| `analysis/seed42-outcomes-summary.md` | Seed 2 durable summary | you have this already |
| `analysis/seed2026-outcomes-summary.md` | **Seed 3 durable summary** | **new to you** |
| `analysis/analysis-config.json` | analysis version + per-seed governance records | version pin |

Regenerate all of the above byte-identically with:

```
python -m runner.publication_gate_run --dump-reports analysis \
    --out analysis/publication-provenance-manifest.json
```

### 5b. The four #33 framing documents — read these before advising

| Path | What it is |
| --- | --- |
| `docs/issue-33-claim-framing-dossier.md` | prose fact base; every fact with provenance; the D1–D5 decision list |
| `docs/issue-33-interpretations.md` | **six** internally-consistent readings of the same evidence, each written as if it were the thesis, each with its own critic's rebuttal and implied decisions |
| `docs/adr/0002-issue-33-claim-framing.md` | frozen option space, arguments for/against, validation checklist, **empty sign-off table** |
| `docs/issue-33-validation-guide.md` | how to regenerate and hand-verify every number against raw evidence |

### 5c. Code produced (context only — you do not need to read it)

`runner/frozen_inputs.py`, `runner/claim_tables.py`, `runner/integrity_report.py`,
`runner/diagnostic_chatmode_mmlu.py`, `runner/diagnostic_report.py`,
`runner/corpus_ablation.py`, `runner/ablation_training.py`,
`runner/publication_gates.py`, `runner/publication_report_inputs.py`,
`runner/publication_gate_run.py` — with matching offline tests under `tests/`.
Separately versioned protocols: `protocol/diagnostic/`, `protocol/ablation/`.
Full suite: **369 pass / 1 skip**.

### 5d. Prior handover notes (house style reference)

`docs/issue-14-finalization-handover.md`, `docs/issue-27-handover.md`,
`docs/handover/issue-30-handover.md`, `docs/handover/issue-31-handover.md`.

---

## 6. Observations that should drive the framing conversation

1. **The strongest single number.** A capability gate built *only* from the
   likelihood-ranked benchmark, at the protocol's own tolerance, **passes all 9
   checkpoints** that the generation-scored gates rejected. Someone who built
   their eval suite that way would have shipped a broken model.
2. **The response distribution visibly collapsed.** GSM8K generation time fell
   from 46 s/item to 6–9 s/item; truncations *fell*. The model stopped producing
   long reasoning and emitted something short. This is direct behavioural
   evidence, not just a score drop. Caveat: available for 6 of 9 checkpoints
   (seed 17's bundles predate the timing log lines).
3. **The training corpus is measurably low-diversity.** 5,000 examples but only
   2,505 distinct assistant responses; the top 10 responses cover half the
   corpus; median response 46 words; 2.9 % multi-step.
4. **The safety gain is narrow.** The visible composite is dominated by one
   benchmark (Open Prompt Injection, +43 to +50 pp); the others moved +5 to +9 pp.
5. **The live objection.** There is exactly one likelihood-ranked benchmark and
   it is also the only pure-recall benchmark. "Modality" and "task type" are
   perfectly confounded. This is the first thing a sharp reviewer will say.
6. **Feasibility is itself a result.** Full protocol — frozen baseline, three
   replicated seeds, gated selection, held-out sealing, provenance gates — on one
   16 GB consumer GPU, unattended, in about 59.9 of 72 GPU-hours.

---

## 7. The open decision (this is where you come in)

Issue #33 cannot close until five wording decisions are made. Full text in
`docs/adr/0002-issue-33-claim-framing.md`.

| | Decision | Options |
| --- | --- | --- |
| **D1** | how far to generalize the headline away from prompt injection | (a) keep injection as subject · (b) fully generalize to corpus diversity · (c) lead with the measurement-modality lesson |
| **D2** | how strongly to state the mechanism | claimed mechanism · best-supported hypothesis |
| **D3** | wording of the ablation result | "injection data is not the cause" · "removing it did not restore the gates" |
| **D4** | the untested sampling / token-budget axes | ship with them open · spend ~3–4 GPU-h closing one first |
| **D5** | MMLU wording | "did not detect the collapse" · "was unaffected" · "improved" |

**D4 is the only one with operational cost.** The other four are prose.

---

## 8. Hard constraints — any framing you propose must satisfy these

These are enforced in code by `runner/publication_gates.py`; wording that
violates them is rejected automatically.

- **Banned words:** *robust*, *secure*, *resistant*, *mitigation*,
  *"defense that works"*.
- **Every capability sentence must name its evaluation modality.**
- **Forbidden claims:** any efficacy claim; any held-out generalization claim;
  any adaptive-attack claim; any claim that response-only QLoRA generally fails;
  any claim about other models or scales; any population inference from 2–3 runs.
- **Held-out benchmark is `NEVER_AUTHORIZED`.** Sealed candidate counts
  (133 valid / 67 invalid) are publishable metadata; no InjecAgent score exists
  and none may be implied.
- **Out of scope for #33:** the arXiv/journal write-up itself, venue choice, and
  any model release. A gate-rejected checkpoint is not a shippable artifact.

---

## 9. What I recommend we brainstorm next

Ordered by how much they unblock.

1. **Match the six interpretations against real demand.** You hold the demand
   signal; `docs/issue-33-interpretations.md` holds six framings. Which of the
   six has an actual audience? The "evaluation blind spot" framing (I1) and the
   "feasibility on consumer hardware" framing (I6) look like different readerships
   entirely — negative-results / eval-methodology venues vs practitioner
   communities. Tell us which is real.
2. **Venue-first, then wording.** Venue is formally out of scope for #33, but it
   silently determines D1 and D2. If the target is an eval-methodology audience,
   D1 → (c). If it is a practitioner / Hugging Face audience, D1 → (a) with
   feasibility leading. Decide the reader before the sentence.
3. **Decide whether to buy the extra experiment (D4).** About 12 GPU-hours
   remain; 3–4 of them close the sampling axis. Worth it only if the modality
   claim is the headline. If we lead with feasibility or the null result, skip it.
4. **Stress-test the confound (observation 5) against reviewer expectations.**
   If the venues in demand would reject on the modality / task-type confound,
   that pushes toward I4 or I6 and away from I1/I2.
5. **Consider splitting the output.** The evidence may support two artifacts with
   different audiences — a short methodology note on the eval blind spot, and a
   feasibility / negative-result report. Your demand data should say whether two
   narrow pieces beat one broad one.
6. **Check format demand, not just topic demand.** Repo, model card, blog post,
   preprint, dataset card — the artifacts already exist in a form that suits
   several. Which format is actually being consumed?

---

## 10. Explicit non-goals of this handover

- No demand, venue, audience or publication-style analysis is included — that is
  yours to supply.
- No claim has been chosen. `analysis/results.md` does not exist.
- Nothing in `docs/issue-33-*` or `docs/adr/0002-*` should be read as a decision;
  the sign-off table is intentionally empty.
