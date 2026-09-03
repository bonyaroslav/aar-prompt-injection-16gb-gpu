# ADR 0002: Issue #33 results-document claim framing

**Status:** OPEN — fact base frozen, verdict deferred to an independent validation
pass. This ADR records the option space and the evidence; it does **not** choose.
Issue #33 stays open until the sign-off block below is filled.

**Decides (eventually):** how issue #33's committed results document
(`analysis/results.md`, not yet written) states its scientific claim, given that
both wording triggers in the issue body have now resolved.

## Context

Issue #33 requires one committed document that states what the study found, its
scope, an explicit "what was not varied" list, five named limitations, and no
forbidden claim; plus a manifest-only checksummed evidence package. The issue
body makes the claim's wording conditional on #30 and #31:

- *"If #30 shows the multiple-choice benchmark also collapses in chat mode, the
  claim … is retired and replaced by a claim about harness/interface mismatch."*
  — **did not fire.** #30 (closed `6fc8b61`) found MMLU does not collapse in chat
  mode; the fine-tune repairs chat-mode MMLU. The scoring-modality axis
  (likelihood vs sampled generation) remains untested.
- *"If #31 shows the reasoning collapse survives removing the injection examples,
  the prompt-injection framing is dropped entirely."* — **fired.** #31 (closed
  `c85f326`) found the capability collapse persisted with zero injection rows,
  but on a single seed and a separately-constructed corpus.

The maintainer has chosen to take the claim decision slowly, after an independent
re-check, rather than under time pressure. This ADR and its siblings prepare the
ground for that.

## Frozen fact base

All numbers in the dossier and interpretations trace to these committed,
regenerable artifacts. Regeneration command and hand-check procedure:
`docs/issue-33-validation-guide.md`.

| Artifact | SHA-256 (LF-normalized committed file) |
|---|---|
| `analysis/attempt1-claim-report.json` | `09e3e534fc4c81d61be14d50dc9909921275b84a63831cd706a5dbb268e85b0e` |
| `analysis/attempt1-integrity-report.json` | `d3781d87df16d1410b0c2e027639236f772d6f87ec6a4b50bf83e6df6e324f2d` |
| `analysis/attempt1-frozen-input-record.json` | `d7e298da95fbc49a7619089c0ef4aa207720d3513bc717a4b65336442d61ad2e` |
| `analysis/publication-provenance-manifest.json` | `5a49c508a8098e79a724f740ede74b690ab332d754104603c90737342cab94ba` |

Protocol anchor: canonical-JSON content digest of `protocol/manifest.json` =
`399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20`
(checkout-invariant, per ticket #25 / `protocol/digests.md`). Analysis version
`phase1-analysis-2026-09`. Completed seeds `[17, 42, 2026]`, 9 checkpoints.

Prose fact base and per-fact provenance: `docs/issue-33-claim-framing-dossier.md`.
Six worked interpretations: `docs/issue-33-interpretations.md`.

## Decision points (frozen; not resolved here)

### D1 — how far to generalize the headline away from prompt injection
- **(a)** keep prompt injection as the study's subject and triggering corpus;
  state the ablation shows the injection rows are not the sufficient cause.
- **(b)** fully generalize to "a low-diversity, response-only safety SFT corpus";
  injection only in method notes / appendix.
- **(c)** headline is the measurement-modality result; low-diversity corpus is
  the leading, evidence-backed mechanism *hypothesis*.

Arguments — *for (b)/(c)*: issue text says "dropped entirely"; #31 supports a
non-injection framing; the mechanism is a corpus property. *For (a)*: the whole
protocol, corpus and visible suite are injection-oriented; #31 is single-seed and
its corpus was rebuilt on several axes; (b) risks the forbidden "response-only
QLoRA fails" generalization.

### D2 — how strongly to state the mechanism
- **claimed** ("finding with an identified mechanism", the issue's language), or
- **hypothesis** ("the finding is the modality split / null result; low-diversity
  corpus is the best-supported explanation, not isolated").

Arguments — *for claimed*: three converging signals (short outputs, low-diversity
corpus, collapse survives injection removal). *For hypothesis*: no arm varied
response diversity with everything else fixed; task-type confound (I4) is
unresolved; #31 changed several corpus properties at once.

### D3 — wording of the #31 result
- **strong**: "the prompt-injection training data is not the cause".
- **supported**: "removing the explicit prompt-injection category did not restore
  the capability gates; the effect is not attributable to that category alone".

The single-seed, differently-constructed ablation licenses only the supported
form. (Recorded as the likely verdict, not the decision.)

### D4 — the untested sampling / token-budget axes
- **(i)** ship with the axis explicitly open (matches the issue text: "considered
  and not taken").
- **(ii)** authorize a new diagnostic protocol and run the ~3–4 GPU-hour
  deterministic GSM8K re-score first (≈ 12 GPU-hours remain under the cap).

Note: (ii) closes only the sampling axis, not the token-budget axis, and would
not resolve I4 or I5. It is the only decision point with operational
consequences; the rest is prose.

### D5 — MMLU wording
"did not detect the collapse" (accurate) vs "was unaffected" (wrong — small
chat-mode decline) vs "improved" (true only in Attempt-1 raw-completion mode).

## Validation checklist for the independent pass

Before filling the sign-off block, the reviewing process must:

1. **Regenerate** the fact base per the validation guide and confirm byte-identical
   output and the four SHA-256 values above.
2. **Hand-verify** at least five headline numbers against the raw
   `runs/**/metrics.json` they derive from (the guide lists them).
3. **Weigh I1 vs I4 explicitly** — decide on the record whether the
   generation-vs-likelihood split is stated as a finding or a confounded
   observation, given there is one likelihood-ranked benchmark and it is the one
   recall benchmark.
4. **Assess the mechanism evidence coverage** — the generation-failure signature
   exists for 6 of 9 checkpoints (seed 17's bundles predate the timing lines).
   Decide whether that supports a mechanism *claim* (D2).
5. **Re-examine D4 on current facts** — the "not taken" call was made when the GPU
   budget was tighter; ~12 hours now remain. Confirm or overturn.
6. **Pre-check the chosen wording against the #32 claim-language gate**
   (`runner.publication_gates.check_claim_language`): no
   robust/secure/resistant/mitigation/"defense that works", every capability
   sentence names its evaluation modality.
7. **Tag venue dependence** — note which of D1–D5 would re-open if a specific
   publication venue is later chosen (venue is out of scope for #33).
8. **Consider proportionality** — this is a solo negative-result report on a 2B
   model; confirm the validation effort matches the stakes.

## Trigger conditions (what would move an answer)

- A second likelihood-ranked *reasoning* benchmark, or a generation-scored
  *recall* benchmark → resolves I1 vs I4, moves D1/D2.
- A third training arm varying response diversity with construction held fixed →
  promotes I2's mechanism from hypothesis to claim (D2).
- Authorizing and running the deterministic GSM8K re-score (D4 → ii) → partially
  closes the modality confound, strengthens I1.
- Any integrity finding invalidating a seed → the analysis already does not
  hardcode the seed count; re-run the fact base, re-check.

## Out of scope for this ADR and issue #33

The arXiv / journal write-up (separate work, may cite only these artifacts); any
model release; choice of publication venue; a new protocol version.

## Common ground (needs no verdict)

Text the results document can state regardless of D1–D5, per all six
interpretations: the negative selection result (9/9 checkpoints capability-
ineligible, 3 null selections, held-out never unsealed); the feasibility result
(full protocol on one 16 GB GPU, ≈ 59.9 of 72 GPU-hours, all bundles
checksum-verified); the OPI-dominated visible gain; the MC-only-gate observation
(a gate built only from MMLU passes all 9 rejected checkpoints); and that the
sampling and token-budget axes are untested.

## Sign-off

| Decision | Verdict | Rationale | Decided by | Date |
|---|---|---|---|---|
| D1 | _pending_ | | | |
| D2 | _pending_ | | | |
| D3 | _pending_ | | | |
| D4 | _pending_ | | | |
| D5 | _pending_ | | | |

When all five are filled: write `analysis/results.md`, run the full gate pass
over the complete set including that document, build
`analysis/publication-package-manifest.json`, update `RESEARCH_PLAN.md`, commit
`Closes #33`, close the issue.
