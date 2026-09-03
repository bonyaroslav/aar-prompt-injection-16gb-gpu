# Issue #33 — claim-framing dossier

**Status:** facts gathered, framing decision deferred to the maintainer
**Date:** 2026-09-03
**Purpose:** Issue #33's results document must state a scientific claim whose
wording the issue makes conditional on the outcomes of #30 and #31. Both have now
landed. This file assembles every fact that bears on the wording so the claim can
be decided after deeper analysis. It recommends nothing and closes nothing.

The packaging half of #33 is unblocked and independent of this decision: the
maintainer chose a **manifest-only** package (no physical staging). That work can
proceed in parallel.

---

## 1. The claim as issue #33 prescribes it

Verbatim from the issue body:

> A low-diversity safety SFT corpus collapses the model's response distribution.
> Every generation-scored benchmark detects it; the log-likelihood-scored
> multiple-choice benchmark does not. A capability gate built only from
> log-likelihood MC benchmarks cannot see response-distribution collapse, because
> it never asks the model to produce a response.

Scoped as **"an existence proof with an identified mechanism."** Must state up
front what was **not** varied: multiple choice inside the chat interface,
generation scored deterministically, any corpus containing chain-of-thought, any
other model or scale.

### The two conditional triggers, verbatim

> If #30 shows the multiple-choice benchmark also collapses in chat mode, the
> claim above is retired and replaced by a claim about harness/interface
> mismatch.

> If #31 shows the reasoning collapse survives removing the injection examples,
> the prompt-injection framing is dropped entirely.

### Forbidden claims (issue body)

Any efficacy claim; any held-out generalization claim; any adaptive-attack claim;
any claim that response-only QLoRA generally fails; any claim about other models
or scales; any population-level inference from two or three runs.

---

## 2. Trigger #30 — chat-mode MMLU confound test (closed, `6fc8b61`)

**Decision record:** `docs/issue-30-chatmode-mmlu-diagnostic-decision.md`.
Diagnostic protocol `protocol/diagnostic/chatmode-mmlu-2026-09-02.json` (canonical
digest `d21e34a834bcb26965e009b7baa0b34158007e6ddc6ae272e608e64111927731`).

### What it resolved

The largest single confounded axis is the **chat template**: Attempt-1 scores
MMLU in raw-completion mode, outside the interface that was fine-tuned. #30
re-scored MMLU with the chat template **on**, same 300 fixed IDs, same candidates,
same scorer, paired item-by-item against Attempt-1.

| Model state | MMLU raw (Attempt-1) | MMLU chat mode | Δ |
| --- | ---: | ---: | ---: |
| baseline (untrained) | 0.567 | **0.250** (chance) | **−0.317** |
| checkpoints (9, mean) | ~0.61 | ~0.556 | **−0.055** (range −0.020 to −0.100) |

Only 4 of 9 checkpoint deltas are individually significant by exact McNemar at
p < 0.05; all nine are negative. A no-leading-space robustness re-run agreed to
within ~1 pp on every state.

**Reading (report verdict `confounded_by_baseline_modality_effect`):**

1. The untrained base model cannot do MMLU through the chat interface (collapses
   to chance). The fine-tune **repairs** chat-mode MMLU (checkpoints 0.53–0.58 vs
   base 0.25) — the opposite of "the fine-tune damaged the chat pathway."
2. Measured *through* the fine-tuned chat interface, MMLU on the checkpoints
   declines ~5 pp — nothing like the generation benchmarks' collapse in the same
   runs (GSM8K −18 to −48 pp, IFEval −18 to −22 pp).
3. **The chat-template confound does not overturn the Attempt-1 MMLU result.**

### What it left open

The **scoring-modality axis** — likelihood ranking over fixed candidates versus
sampled free generation. MMLU is likelihood-ranked in *both* raw and chat mode, so
#30 cannot separate it. Issue #30's own conclusion: "It remains the open
confound, and issue #28's primary-table caption should keep saying so."

### Consequence for the claim

The "harness/interface mismatch" replacement trigger did **not** fire — MMLU did
not collapse in chat mode. The claim's second sentence ("the log-likelihood-scored
multiple-choice benchmark does not [detect it]") is **supported as an
association**: the one likelihood-ranked benchmark declines ~5 pp (and *improves*
in raw mode) while every generation-scored benchmark collapses. It is **not**
supported as a clean single-axis mechanism, because the generation-vs-likelihood
contrast is still confounded with sampling and token budget (Section 5).

---

## 3. Trigger #31 — clean-corpus ablation (closed, `c85f326`)

**Decision record:** `docs/issue-31-corpus-ablation-decision.md`.
Ablation protocol `protocol/ablation/corpus-ablation-2026-09-02.json` (canonical
digest `c6e36b48d5de4ec151b6a0a23bfe493474cedeb2fd9db9e42094804e4005b7b3`).

### What it resolved

Attempt 1's 5,000-row corpus had 2,000 prompt-injection rows. The ablation corpus
has **0 prompt-injection** (3,500 clean-control, 1,000 ambiguous-boundary, 500
refusal-calibration). One seed (42), three epochs, six visible benchmarks.

| State | OPI | TT extract | TT hijack | MMLU | GSM8K | IFEval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.180 | 0.598 | 0.492 | 0.567 | 0.735 | 0.615 |
| Clean ablation, epoch 3 | 0.400 | 0.715 | 0.532 | 0.613 | **0.490** | **0.430** |
| Attempt 1 seed 42, epoch 3 | 0.667 | 0.565 | 0.577 | 0.600 | 0.500 | 0.485 |

Epoch-3 capability change vs the same baseline: **GSM8K −0.245, IFEval −0.185**,
while MMLU **improved** +0.046. The OPI improvement shrank to +0.220 (Attempt-1
seed-42 epoch 3 was +0.487).

**The capability collapse survived removing the injection category.**

### What it left open — confounds that weaken a strong "injection is not the cause" claim

- **One seed**, no pre-authorized multi-seed or bootstrap analysis. Descriptive
  only, by design.
- **The ablation corpus is not "Attempt-1 minus the injection rows."** It was
  built through a separate path: Dolly oversample factor 2 and a **1,536-token
  construction filter** (the frozen trainer uses a 2,048-token sequence length),
  to keep assistant tokens under the tokenizer. Clean-control also grew
  1,500 → 3,500. Composition changed on more than one axis.
- Six visible benchmarks only; no held-out.

Issue #31's own framing: it "does not support the simple claim that the
prompt-injection rows alone caused the capability collapse … compatible with a
broader effect of this response-only SFT setup and corpus composition; isolating
which remaining component causes it needs a separately authorized follow-up." And:
"It does establish the narrower, useful negative result: deleting the explicit
prompt-injection category did not restore the frozen capability gates."

### Consequence for the claim

The literal instruction ("the prompt-injection framing is dropped entirely")
**fired**. The mechanism in the claim is a corpus property ("low-diversity …
collapses the response distribution"), not injection semantics, and the ablation
is consistent with that. The open question is how far to generalize the headline
and how strongly to state causation (Section 6, D1–D3).

---

## 4. Direct evidence for "collapses the model's response distribution"

From the #29 integrity report (`runner.integrity_report`), computed over the
finalized Attempt-1 bundles.

### 4a. Generation-failure signature (the mechanism section)

Seed-17 bundles predate the machine-readable log lines and are timing-unavailable.
Seeds 42 and 2026 (six checkpoints):

| | baseline | s42 e1 | s42 e3 | s2026 e1 | s2026 e3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| truncated completions | 190 | 34 | 103 | 59 | 185 |
| GSM8K seconds/item | 45.96 | 6.22 | 8.67 | 6.55 | 8.44 |
| IFEval seconds/item | 22.12 | 7.78 | 10.07 | 8.47 | 10.88 |

GSM8K generation time drops by **−37 to −40 s/item** against the baseline; IFEval
by **−11 to −14 s/item**. Truncations *fall* (fewer runaway completions). This is
the report's "early collapse to a short answer/refusal" mechanism, not "unbounded
rambling" — the model stops producing long reasoning chains on the
generation-scored benchmarks. Direct evidence that the *output distribution*, not
just the score, changed.

### 4b. Training-corpus nutrition label

- 5,000 examples, **2,505 distinct assistant responses** (50.1 % distinct).
- Most-frequent-response coverage: top-1 = **10.0 %**, top-5 = 42.0 %,
  top-10 = **50.0 %** of the whole corpus.
- Response length: median **46 words** / 246 chars; p90 77 words.
- Multi-step-reasoning share (coarse lexical proxy): **2.9 %**.

Report note (already promoted out of "limitations"): "a corpus whose assistant
responses are few, near-duplicated, short, and rarely multi-step is a plausible
cause of capability collapse under SFT."

### 4c. Tensor Trust degeneracy check

Refusal-degeneracy signature (both-defended → one-defended migration) present on
exactly **one** of 18 checkpoint×benchmark cells (seed42-epoch3-tensor_trust_extract).
Not a broad degeneracy signal.

---

## 5. The modality split (#28 primary table)

Grouped by evaluation modality. Epoch-3 absolute deltas vs the frozen baseline:

| Benchmark | Modality | s17 | s42 | s2026 |
| --- | --- | ---: | ---: | ---: |
| open_prompt_injection | free-generation, sampled, string-scored | +0.497 | +0.487 | +0.427 |
| tensor_trust_hijack | free-generation … | +0.050 | +0.085 | +0.073 |
| tensor_trust_extract | free-generation … | +0.072 | −0.033 | +0.082 |
| gsm8k | free-generation … | **−0.180** | **−0.235** | **−0.200** |
| ifeval | free-generation … | **−0.180** | **−0.130** | **−0.145** |
| **mmlu** | **likelihood-ranked, no generation** | **+0.027** | **+0.033** | **+0.027** |

**Multiple-choice-only-gate column:** a capability gate built solely from MMLU at
the manifest's own tolerance (`mmlu_max_decline = 0.02`) **passes all 9
checkpoints** — every checkpoint that destroyed GSM8K and IFEval. This is the
concrete number behind the claim's third sentence.

Capability gates that actually applied (all 9 checkpoints **FAIL**): GSM8K decline
≤ 0.02, IFEval decline ≤ 0.03, mean normalized retention ≥ 0.98.

Primary-table caption (current, from #28): the two groups "differ on four
confounded axes at once: (1) the chat template applied to the prompt; (2) sampled
free-generation decoding versus deterministic likelihood scoring; (3) the
generation token budget; and (4) the scoring method … These axes are confounded
together, not cleanly separated, so this table is an existence proof about
measurement modality, not a clean two-way contrast." #30 retired axis (1); axes
(2)–(4) remain.

Cross-run summary is explicitly **run-to-run variability under a fixed nominal
configuration** (adapter init precedes the run seed), N = 3, descriptive — not
seed variance, not a confidence interval. Epoch-3 population SD: GSM8K 0.023,
IFEval 0.021, MMLU 0.003.

---

## 6. Decisions for the maintainer

### D1 — how far to generalize the headline away from prompt injection

| Option | For | Against / risk |
| --- | --- | --- |
| **(a)** Keep prompt injection as the study's subject and the triggering corpus; state the ablation shows the injection rows are not the *sufficient* cause | Honest to what was actually run (a prompt-injection SFT study); the whole protocol, corpus and visible suite are injection-oriented | Issue says "dropped entirely"; a reviewer may read residual injection framing as hedging |
| **(b)** Fully generalize: "a low-diversity, response-only safety SFT corpus collapses the response distribution"; injection appears only in method notes/appendix | Matches the issue's literal instruction and the corpus-property mechanism; the ablation supports a non-injection framing | Over-reaches from one corpus family and one 2B model; risks sounding like a general claim about response-only SFT (a forbidden claim) unless carefully bounded |
| **(c)** Headline is the **measurement-modality** finding (Section 5), with the low-diversity corpus as the leading, evidence-backed mechanism hypothesis | The modality split + the MC-only-gate result are the most robust, least-confounded facts in the study | "Existence proof with an identified mechanism" (issue's words) is softened to "identified candidate mechanism" |

### D2 — how strongly to state the mechanism

Issue wants "an identified mechanism." Evidence *for* the corpus-diversity
mechanism: 4a (outputs got short), 4b (corpus is low-diversity), #31 (collapse
survives injection removal → points at a corpus-wide property). Evidence *against*
treating it as isolated: no ablation arm varies response diversity while holding
everything else fixed; #31 changed corpus construction on several axes at once.
Choice: **"finding with identified mechanism"** vs **"finding = the modality
split; mechanism = the best-supported hypothesis, not isolated."**

### D3 — how to word the #31 result

- Strong: "the prompt-injection training data is not the cause of the capability
  collapse."
- Supported: "removing the explicit prompt-injection category did not restore the
  capability gates; the effect is not attributable to that category alone."
- The second is what the single-seed, differently-constructed ablation actually
  licenses.

### D4 — the untested sampling / token-budget axes

Issue body: countering them "would need the reasoning benchmark re-run with
deterministic decoding, roughly three to four GPU-hours, which was considered and
not taken." ~12.1 GPU-h remain under the 72-h cap. Options: (i) ship with the axis
explicitly open (matches issue text); (ii) authorize a new diagnostic protocol and
run the deterministic GSM8K re-score first. This is a scope decision.

### D5 — MMLU wording

MMLU *improved* in raw mode (+0.027 to +0.033 at epoch 3) and declined only ~5 pp
measured through the fine-tuned chat interface (#30). "Did not detect the
collapse" is accurate; "was unaffected" is not (small chat-mode decline);
"improved" is true only in the Attempt-1 raw-completion modality.

---

## 7. Resource facts for the "under 72 h" criterion

| Bucket | GPU-hours | Source |
| --- | ---: | --- |
| Scientific (baseline + 3 seeds) | 47.338 | #29 `resource_accounting.scientific_totals` |
| #31 clean-corpus ablation | 11.885 | `docs/issue-31-corpus-ablation-decision.md` |
| #30 chat-mode MMLU diagnostic | 0.652 | `docs/issue-30-chatmode-mmlu-diagnostic-decision.md` |
| **All-incurred total** | **≈ 59.87** | **< 72 h cap** |

Peak VRAM 15.663 GiB (training phase), 0.163 GiB over the 15.5 GiB declared
allocation, under the 16 GiB card. Finalized-bundle disk 31.7 GiB (< 250 GiB).
The current provenance-gate run records `non_scientific_runs: []`; #33's package
run must fold in the #30 and #31 resource lines.

---

## 8. Raw evidence pointers

- `analysis/seed{17,42,2026}-outcomes-summary.md` — per-seed finalized results.
- `docs/issue-14-finalization-handover.md` — verified finalized-artifact table.
- `analysis/publication-provenance-manifest.json` — #32 provenance manifest
  (`reports=2 sections=13 receipted_numbers=1763 orphans=0`).
- `runner/claim_tables.py` / `runner/integrity_report.py` — the pure transforms
  the numbers above were rendered from.
- `docs/issue-30-chatmode-mmlu-diagnostic-decision.md`,
  `docs/issue-31-corpus-ablation-decision.md`,
  `docs/issue-32-provenance-source-decision.md`.
- `runs/`, `diagnostics/`, `ablation/` — gitignored execution evidence, present
  locally.
