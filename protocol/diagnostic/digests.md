# Diagnostic protocol digest identities

## `chatmode-mmlu-2026-09-02.json` (issue #30 — chat-mode MMLU confound test)

A separately versioned **diagnostic** protocol, explicitly downstream of the
finalized Attempt-1 evidence. It re-scores MMLU with the chat template enabled on
the frozen baseline and every finalized merged checkpoint, paired item by item
against the Attempt-1 raw-completion-mode result. It changes exactly one value
from `phase1-2026-08-29` — `evaluation.capability.mmlu.use_chat_template`
(`false` → `true`) — and is authorized in writing in the issue #30 body.

- **Canonical-JSON content digest** (checkout-invariant; the publication /
  provenance identity, same recipe as the frozen manifest's canonical digest):
  `d21e34a834bcb26965e009b7baa0b34158007e6ddc6ae272e608e64111927731`
- **Raw-file SHA-256** (byte integrity; recorded in each diagnostic bundle's
  `manifest.yaml` as `diagnostic_manifest_sha256`):
  `33d23c1747fc75840fa28d5cc3e89df377fb13ddb08c9273cc907251c06e3820`

`protocol/diagnostic/manifest.py` recomputes the canonical digest on load and
**fails closed** if the live `protocol/manifest.json` no longer produces the
`downstream_of.canonical_manifest_digest`
(`399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20`) this
diagnostic was pinned against. `.gitattributes` pins `protocol/**` to `text
eol=lf`, so both identities reproduce on a fresh clone.

Diagnostic outputs live under `diagnostics/` (gitignored, outside `runs/` and
`analysis/`) and never enter an Attempt-1 evidence bundle, the frozen
10,000-replicate bootstrap (`analysis.bootstrap_seed = 271828`), or checkpoint
selection. The `ambiguity_rule` block also triggers the no-leading-space
robustness re-run when any model state lands near chance in chat mode, which is
what happened here (the untrained baseline). Outcome and both result tables:
`docs/issue-30-chatmode-mmlu-diagnostic-decision.md`.
