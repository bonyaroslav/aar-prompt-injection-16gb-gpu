# Issue #33 — six interpretations of the same evidence

**Status:** no decision. This file exists so the maintainer can look at the
result from several internally-consistent angles before choosing the claim
wording. Every interpretation below is built from the **same frozen fact base**
(`docs/issue-33-validation-guide.md`); they differ only in what they claim that
evidence *means*.

Read each one as if it were the thesis of the results document. Then use
`docs/adr/0002-issue-33-claim-framing.md` to record which parts of which
interpretation survive scrutiny.

Shorthand used below:
- **generation-scored** benchmarks: OPI, Tensor Trust hijack/extract, GSM8K,
  IFEval — the model writes text, then a parser/rule scores it.
- **likelihood-ranked** benchmark: MMLU — no text is generated; the model ranks
  four fixed options by first-token log-probability.
- **the gates**: GSM8K decline ≤ 0.02, IFEval decline ≤ 0.03, mean normalized
  retention ≥ 0.98 (`protocol/manifest.json selection.capability_gates`).

---

## I1 — The measurement-modality lesson (narrowest, most defensible)

**Thesis.** Across three seeds and nine checkpoints, a transparent response-only
QLoRA fine-tune improved the visible safety composite (overwhelmingly one
benchmark, Open Prompt Injection) while failing every capability gate. The loss
was visible in **every generation-scored benchmark** (GSM8K −18 to −24 pp, IFEval
−13 to −23 pp) and **not** in the single likelihood-ranked benchmark (MMLU, which
moved +3 pp). A capability gate constructed only from the likelihood-ranked
benchmark, at the protocol's own tolerance, would have passed all nine checkpoints
that the generation-scored gates rejected. The practical lesson is about
**evaluation construction**: a capability suite that never asks the model to
generate a response can miss a large regression that generation-based evaluation
detects immediately.

**Supporting evidence.** The primary table (`attempt1-claim-report.json`
`primary_table`); the `multiple_choice_only_gate` column showing
`passes = true` for 9/9; the generation-failure signature (GSM8K generation time
46 s/item → 6–9 s/item on seeds 42/2026 — outputs became far shorter).

**Concessions.** It does not explain *why* the generation benchmarks fell. It does
not claim the corpus, the injection data, or SFT-in-general is the cause. "MMLU
survived" is an observation about one benchmark, not a proven property of
likelihood scoring.

**Critic's rebuttal.** "You have one likelihood-ranked benchmark and it is also
your only pure-recall benchmark. This is a single data point dressed as a
category." (See I4.)

**What would firm it up.** A second likelihood-ranked benchmark that requires
reasoning (e.g. a multiple-choice GSM8K variant scored by log-likelihood); or a
generation-scored recall benchmark. Either would separate modality from task type.

**Implied decisions.** D1 → (c). D2 → association, mechanism as hypothesis. D3 →
careful wording. D5 → "did not detect the collapse."

**Forbidden-claim check.** Clean — no efficacy, generalization, adaptive-attack,
"QLoRA generally fails", cross-model, or population claim.

---

## I2 — The low-diversity-corpus mechanism (issue #33's literal claim)

**Thesis.** A low-diversity safety SFT corpus collapses the model's response
distribution. The training corpus has 2,505 distinct assistant responses across
5,000 examples; its ten most frequent responses account for half the corpus;
responses are short (median 46 words) and rarely multi-step (2.9 %). Fine-tuning
on it drives the model toward short, templated outputs: on the generation-scored
benchmarks, generation time and output length collapse and the scores fall with
them. The likelihood-ranked benchmark does not ask for a response and is
unaffected. Removing the prompt-injection examples entirely (#31) did not prevent
the collapse, so the mechanism is a property of the corpus as a whole, not of the
attack examples.

**Supporting evidence.** Corpus nutrition label; generation-failure signature
(shorter outputs); #31 (collapse survives injection removal); the modality split.

**Concessions.** No experiment varied response diversity while holding everything
else fixed. #31's clean corpus was rebuilt through a different path (1,536-token
construction filter, Dolly oversampled ×2, clean-control 1,500 → 3,500), so it is
not a clean "same corpus minus injection." Single seed for #31.

**Critic's rebuttal.** "'Low-diversity' is doing a lot of work with no controlled
test behind it. Your clean-corpus arm changed four things at once and still
collapsed — that is at least as consistent with 'this SFT setup collapses
capability regardless of corpus' (I5) as with a diversity mechanism."

**What would firm it up.** A third training arm: same size and construction, but
with deliberately diversified / lengthened assistant responses (e.g. distinct
paraphrases, some multi-step). If capability is preserved there, the mechanism
holds.

**Implied decisions.** D1 → (b). D2 → "finding with identified mechanism". D3 →
"the effect is not attributable to the injection category alone". D5 → "did not
detect the collapse".

**Forbidden-claim check.** Risk point: must not phrase as "response-only QLoRA
fails" — keep it scoped to *this corpus construction* on *this model*.

---

## I3 — The prompt-injection SFT capability tradeoff (keeps the study's subject)

**Thesis.** This study asked whether one transparent QLoRA intervention can reduce
prompt-injection susceptibility on a 2B model under consumer compute without
paid judges. The answer across three seeds is a consistent **negative**: the
intervention produced a large, benchmark-narrow safety gain (Open Prompt
Injection +43 to +50 pp; other safety benchmarks +5 to +9 pp) at a capability
cost that failed every gate on every checkpoint, so no checkpoint was eligible and
the held-out benchmark was never unsealed. A follow-up ablation removing the
explicit prompt-injection training examples still failed the capability gates,
which tells us the injection examples are not the *sole* cause but does not move
the headline: under this protocol, this intervention does not deliver
capability-preserving prompt-injection robustness.

**Supporting evidence.** Per-seed outcomes summaries; the null selections;
visible-composite decomposition (OPI-dominated); #31.

**Concessions.** "Negative result for this intervention" is not "negative result
for the approach". Two or three runs are not a population. The safety gain is
real on the visible benchmarks even if narrow.

**Critic's rebuttal.** "Issue #33 explicitly says: if the collapse survives
removing the injection examples, drop the prompt-injection framing entirely. This
interpretation keeps it. You are describing a corpus/SFT effect as if it were a
prompt-injection finding."

**What would firm it up.** Nothing further is needed for the negative claim
itself; it is the best-evidenced statement in the study. Firming up *why* pushes
toward I2 or I5.

**Implied decisions.** D1 → (a). D2 → mechanism stays a secondary, hypothesis-level
section. D3 → "did not restore the gates". D4 → ship with the axis open.

**Forbidden-claim check.** Clean, provided the safety gain is never called a
"mitigation" and the null is never softened.

---

## I4 — The task-type confound (the skeptic's minimal read)

**Thesis.** The benchmarks that fell (GSM8K, IFEval) require multi-step reasoning
and instruction-following. The benchmark that held (MMLU) requires factual recall
and is the only one scored by log-likelihood. In this study, **evaluation
modality and task type are perfectly confounded**: there is exactly one
likelihood-ranked benchmark and it is exactly the one recall benchmark. The
defensible statement is therefore: *this intervention preserved multiple-choice
factual recall while degrading generated multi-step reasoning and instruction
compliance; whether that split is driven by scoring modality, task type, or both
cannot be determined from this data.*

**Supporting evidence.** Benchmark composition of the frozen suite; the #28
caption already naming four confounded axes; #30 explicitly leaving the
scoring-modality axis untested.

**Concessions.** The generation-failure signature (outputs got shorter) is a real
mechanism signal that a pure "task type" story does not fully account for — it
points at *output behaviour* changing, which is modality-adjacent.

**Critic's rebuttal.** "This is so hedged it barely says anything. The MC-only-gate
result is still striking regardless of *why* MMLU held: a practitioner who built
their gate that way would have shipped a broken model."

**What would firm it up.** Same as I1 — a benchmark that breaks the confound.
Absent that, this interpretation is the floor below I1.

**Implied decisions.** D1 → (c) or (a). D2 → explicitly *not* a claimed mechanism.
D3 → "did not restore the gates". D5 → "did not detect the collapse" with the
confound stated in the same breath.

**Forbidden-claim check.** Clean by construction — it is the conservative read.

---

## I5 — This SFT setup is simply underpowered (the capacity read)

**Thesis.** A 2B model, 4-bit QLoRA adapters, a response-only objective, three
epochs, and a 5,000-example corpus may not have the capacity to absorb a safety
behaviour change without spending capability. Every training arm attempted —
three seeds on the injection corpus, one seed on a clean corpus — collapsed the
generation-scored capability benchmarks by a similar magnitude. The most
parsimonious reading is that *this configuration* trades capability for any
strong behavioural shift, and neither the injection data nor corpus diversity
specifically needs to be invoked.

**Supporting evidence.** #31 (clean corpus also collapsed, similar magnitude);
consistency of the collapse across seeds and corpora; small model + aggressive
adapter setup.

**Concessions.** No arm tested a larger model, a longer/CoT corpus, full
fine-tuning, or fewer epochs, so "underpowered configuration" is itself
untested — it is a hypothesis of the same standing as I2's diversity mechanism.
MMLU *improving* slightly is not obviously consistent with a pure capacity-crunch
story and needs addressing.

**Critic's rebuttal.** "Then why did MMLU go up? And why is OPI improvement so
much larger on the injection corpus than the clean one? A flat 'capacity' story
does not predict either."

**What would firm it up.** A single arm at fewer epochs or a smaller learning
rate; or one arm on a 7–8B model. Out of scope for this attempt.

**Implied decisions.** D1 → (b) but reframed around configuration, not corpus. D2
→ hypothesis only. D3 → "did not restore the gates; consistent with a broader
effect of this configuration". D4 → the deterministic re-run would not resolve
this.

**Forbidden-claim check.** Must avoid "response-only QLoRA fails" — this is
exactly the forbidden generalization if stated carelessly. Keep it to "this
configuration in this study".

---

## I6 — Feasibility and negative selection result only (maximally conservative)

**Thesis.** Two things are established beyond reasonable dispute. **(1) Feasibility:**
one practitioner ran the full protocol — frozen baseline, three replicated seeds,
per-epoch evaluation, capability-gated selection, held-out sealing, provenance
gates — on a single 16 GB consumer GPU, unattended, in ≈ 59.9 GPU-hours of the
72-hour budget, with every finalized bundle checksum-verified. **(2) Negative
selection result:** under the frozen selection rule, none of the nine checkpoints
was capability-eligible, across three seeds, so the study selected nothing and
never unsealed the held-out benchmark. Everything about the *cause* of the
capability loss — modality, corpus diversity, task type, configuration capacity —
is exploratory and belongs in a clearly-labelled "exploratory observations"
section, not the headline.

**Supporting evidence.** The entire finalized evidence tree; resource accounting;
`held_out_disposition: NEVER_AUTHORIZED`; the #32 provenance manifest.

**Concessions.** This under-claims relative to what issue #33 asks for ("an
existence proof with an identified mechanism"). It leaves the most interesting
observation (the modality split) as a footnote.

**Critic's rebuttal.** "You did the analysis in #28–#31. Refusing to state any of
it in the headline wastes the work and the reader's time."

**What would firm it up.** Nothing — it is deliberately the minimal claim.

**Implied decisions.** D1 → injection stays as the study's subject, no mechanism
headline. D2 → no mechanism claim at all. D3 → "did not restore the gates". D4 →
ship with everything open.

**Forbidden-claim check.** Clean by design.

---

## Cross-map: which interpretation implies which decision

| | D1 headline | D2 mechanism | D3 #31 wording | D4 extra run |
|---|---|---|---|---|
| I1 modality lesson | (c) modality | hypothesis | careful | ship open |
| I2 corpus diversity | (b) generalize | claimed | "not injection alone" | ship open |
| I3 injection tradeoff | (a) keep injection | secondary | "did not restore" | ship open |
| I4 task-type confound | (c) / (a) | explicitly none | "did not restore" | ship open |
| I5 configuration capacity | (b) reframed | hypothesis | "broader effect" | run would not help |
| I6 feasibility + null | (a) keep subject | none | "did not restore" | ship open |

**Common ground across all six** (candidate for text that needs no verdict):
the negative selection result, the feasibility result, the OPI-dominated visible
gain, the MC-only-gate observation, and that the sampling / token-budget axes are
untested.

**Where they genuinely diverge:** how strongly to name a mechanism (I2 vs
everyone), whether to keep or drop the prompt-injection framing (I3/I6 vs
I1/I2/I4), and whether the modality split is a *finding* or a *confounded
observation* (I1 vs I4).
