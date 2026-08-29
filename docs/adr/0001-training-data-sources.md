# ADR 0001: Training-data construction route for Attempt 1

**Status:** accepted, 2026-08-29
**Decides:** the open item in `RESEARCH_SPEC.md`'s implementation decisions — "Build training data
only from upstream-authorized training splits or newly generated examples produced without viewing
evaluation content" left the actual source unresolved. This ADR resolves it.

## Context

`RESEARCH_SPEC.md` requires 5,000 response-only SFT examples (40% prompt-injection attacks / 30%
clean instruction-following controls / 20% ambiguous trust-boundary cases / 10% refusal-calibration),
built without paid LLM APIs (Attempt 1 scope) and without any overlap with the visible evaluation
sets. Three routes were considered:

- **(A) Public-dataset + template synthesis** — assemble from existing public corpora and
  deterministic templates. Fully auditable (every example hashes back to a public source or a fixed
  template), zero API cost, no new model dependency.
- **(B) Local-model generation** — use a larger local instruct model (e.g. a 4-bit 7-8B model) to
  generate examples. Higher diversity, but adds a second model's provenance to the study and GPU
  hours.
- **(C) Paid-API generation** — best quality, but breaks the declared "no paid APIs in Attempt 1"
  scope; would require a new ADR reopening that scope decision.

## Decision

**Route A**, restricted to non-gated public sources so Attempt 1 needs no `HF_TOKEN` and no
per-source access request:

| Category | Target n | Source(s) | License | Notes |
|---|---:|---|---|---|
| Prompt-injection attacks | 2,000 | `deepset/prompt-injections` (546 train / 116 test) + `Lakera/gandalf_ignore_instructions` (777/111/112) | apache-2.0 / MIT | ~1,400 raw rows; remainder made by deterministic templated variation (paraphrase/reorder/wrap), not by a generator model |
| Clean instruction-following controls | 1,500 | `databricks/databricks-dolly-15k` (15,011 rows) | cc-by-sa-3.0 | already has instruction + context + response fields, so no answer needs generating |
| Ambiguous trust-boundary cases | 1,000 | templated (own construction) | n/a | doubly-embeds a Dolly-style instruction/context pair with a benign-but-boundary-adjacent aside |
| Refusal/calibration cases | 500 | templated (own construction) | n/a | one fixed policy-refusal response per template family |

**`hackaprompt/hackaprompt-dataset` is explicitly excluded from Attempt 1.** It is gated
(`gated: auto` on the Hub) and would require an `HF_TOKEN` and per-user access grant before any
automated pipeline could touch it — a manual, human-gated step this study's "no manual intervention
after launch" resource rule (`RESEARCH_SPEC.md` story 16) is designed to avoid. It may be added in a
future protocol version if the seed-1 pilot shows the smaller corpus is insufficient; that is a new
protocol version per the fallback policy, not a Phase 5 default.

## Exclusion rule (binding)

None of the following may be read during training-data construction, generation-template design, or
deduplication — they are evaluation-only, per `protocol/manifest.json` and `scripts/publish_suite.py`:

- `HumanCompatibleAI/tensor-trust-data` (feeds `tensor_trust_hijack` / `tensor_trust_extract`) —
  including the ~475 of ~775 rows *not* selected into the published 300-item eval subset. Unused
  rows from the same pool are still the same distribution; taking them would not violate the
  literal string-level dedup check but would defeat its purpose.
- `stanfordnlp/sst2`, `sms_spam`, `hsol` (feed `open_prompt_injection`)
- `uiuc-kang-lab/InjecAgent` (the held-out benchmark; already excluded by `RESEARCH_SPEC.md`)

## Consequences

- Attempt 1 data construction needs no `HF_TOKEN` and can run unattended, consistent with the
  resource rules.
- The likely weakness is template-surface overfitting: a corpus built mostly from two attack
  datasets plus deterministic templates may teach pattern-matching refusal rather than general
  trust-boundary reasoning, which is exactly the failure mode the held-out InjecAgent leg exists to
  detect (`protocol/heldout_sealing.md`, `protocol/power_notes.md`).
- If the seed-1 pilot (Phase 5) shows no visible-composite movement at all, the documented fallback
  is Route B (local-model generation), not silently enriching Route A with gated or eval-adjacent
  data. Route B requires a new protocol version and a fresh baseline per the fallback policy.
- Route C (paid-API generation) remains out of scope for Attempt 1 and is not reconsidered by this
  ADR.
