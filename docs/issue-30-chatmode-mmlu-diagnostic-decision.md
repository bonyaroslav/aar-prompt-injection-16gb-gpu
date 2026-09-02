# Issue #30 — Chat-mode MMLU confound test (diagnostic)

**Issue:** [#30](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/30)
**Decision date:** 2026-09-02
**Downstream of:** frozen protocol `phase1-2026-08-29`
(`protocol/manifest.json`, canonical digest
`399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20`)
**Diagnostic protocol:** `protocol/diagnostic/chatmode-mmlu-2026-09-02.json`
(canonical digest `d21e34a834bcb26965e009b7baa0b34158007e6ddc6ae272e608e64111927731`,
raw SHA-256 `33d23c1747fc75840fa28d5cc3e89df377fb13ddb08c9273cc907251c06e3820`)

## Why this diagnostic exists

The Attempt-1 headline finding contrasts one multiple-choice benchmark that held
up (MMLU) against generation benchmarks that collapsed. Those benchmarks differ
on four confounded axes at once; the largest is that MMLU is scored **without the
chat template**, in raw-completion mode, generating no tokens — while the
intervention was response-only SFT **on chat-templated data**. The one benchmark
that survived is the one evaluated outside the interface that was fine-tuned. A
reviewer reaches this in the first paragraph. Without this ticket it is only
disclosed (`protocol/deviations.md`, issue #28's primary-table caption), never
tested.

The test re-scores MMLU on the frozen baseline and every finalized checkpoint
with the chat template **enabled** — same 300 fixed example IDs, same prompt
text, same candidate strings (`" A" " B" " C" " D"`), same `first_token_logit`
scorer, same `max_new_tokens: 1` — paired item by item against the Attempt-1
result. No training, no new checkpoints.

## Authorization and boundaries

The diagnostic protocol version is **authorized in writing in the issue #30
body** ("This authorization is granted here in writing"). This is an authorized
deviation being executed, not a deviation being decided. It declares the frozen
Attempt-1 baseline as its baseline; the reuse is justified because model
identity, revision, evaluation suite, sample identifiers, decoding block and
scorer are all unchanged — only `evaluation.capability.mmlu.use_chat_template`
differs (`false` → `true`).

Enforced boundaries (`protocol/diagnostic/chatmode-mmlu-2026-09-02.json`
`boundaries`, and in code):

- Outputs are written under `diagnostics/` — outside `runs/` and `analysis/` —
  and never enter an Attempt-1 evidence bundle. `runner.frozen_inputs` does not
  discover them.
- The result never feeds checkpoint selection. Both completed selection records
  keep `selected_checkpoint_digest: null`.
- The diagnostic bootstrap uses its own seed (303030), never the frozen
  Attempt-1 `analysis.bootstrap_seed` (271828); it does not read, extend or
  contribute to the frozen 10,000-replicate bootstrap.
- The held-out InjecAgent benchmark is untouched — no `HeldOutSealer`, the
  restricted root is never read.
- GPU-hours are recorded on the diagnostic's own resource line
  (`diagnostics/chatmode-mmlu-<stamp>-resource.json`, shaped as a
  `runner.integrity_report` `non_scientific_runs` row) and folded into the
  combined all-incurred-compute figure, not the scientific Attempt-1 totals.

## Checkpoint integrity — verified before any GPU time

Acceptance criterion: the merged checkpoints must be present and digest-matching
before GPU time is spent; if absent the ticket stops and reports rather than
retraining.

All nine merged checkpoints (three seeds × three epochs) are present and are
structurally valid standalone Hugging Face model directories (`config.json` +
`model.safetensors`, 320 tensors, 3.76 GB each).

The `fingerprint` / `checkpoint_digest` token carried through the training
bundles, evaluation bundles and selection records is a hash of the *adapter*
(LoRA) directory and is **not reproducible** from any on-disk artifact for any of
the nine (verified with `runner.real_training._directory_fingerprint`). The
re-verifiable identity is the *merged-directory* digest
(`runner.training._directory_digest`):

| Seeds | Checkpoints | Anchor | Result |
| --- | --- | --- | --- |
| 42, 2026 | 6 | `integrity` value recorded at run time in `recovery/training-seed{42,2026}-seq2048-epoch{1,2,3}.json` | **6/6 recomputed digests match exactly** |
| 17 | 3 | none — seed 17's run predates the issue #22 resumable split-run seam and ran without `--recovery-root`, so no run-time merged-checkpoint digest was ever recorded | digests first computed during this diagnostic (2026-09-02) and pinned in the diagnostic manifest as `expected_integrity_digest` with `integrity_source: no_runtime_digest_pre_issue_22` |

Seed-17 limitation (disclosed, non-blocking): the three seed-17 merged
directories are present, structurally valid, and their file modification times
are unchanged since the 2026-08-30 training run, but their bit-identity to what
the frozen Attempt-1 seed-17 evaluations scored cannot be independently verified.
This is handled the same way issue #29 handled seed 17's missing machine-readable
timing lines — disclosed and worked around, not treated as tampering. Decision
(recorded in the ticket): include all three seeds and document the gap.

`runner.diagnostic_chatmode_mmlu.verify_model_states` recomputes
`_directory_digest` for all nine directories and asserts equality with the
manifest's `expected_integrity_digest` (cross-checking seeds 42/2026 against
`recovery/`) before any model is loaded; any mismatch raises
`CheckpointIntegrityError` and the harness never retrains.

## What shipped

- `runner/real_adapters.py` — `RealModelAdapter` gains two constructor
  parameters, both defaulting to Attempt-1 behaviour: `mmlu_use_chat_template`
  (`False`) and `mmlu_candidate_strings` (`(" A", " B", " C", " D")`). The MMLU
  branch previously hard-coded both. Regression tests in
  `tests/test_real_adapters.py` assert the default forwards
  `use_chat_template=False` with the unchanged candidates and that the flag is
  honoured when enabled.
- `protocol/diagnostic/` — the separately versioned diagnostic manifest, a
  `manifest.py` loader that fails closed if the frozen manifest's canonical
  digest drifts, and `digests.md`. `.gitattributes` already pins `protocol/**`
  to LF.
- `runner/diagnostic_chatmode_mmlu.py` — the execution harness:
  `verify_model_states` (pure, above) plus the real-hardware `run_diagnostic`,
  which loads the baseline snapshot and each merged checkpoint with
  `mmlu_use_chat_template=True`, re-scores the 300 MMLU items, and writes one
  checksummed bundle under `diagnostics/chatmode-mmlu-<stamp>/` plus a resource
  artifact. `--max-items` gives an end-to-end smoke path; `--no-leading-space`
  runs the robustness variant.
- `runner/diagnostic_report.py` — `build_chatmode_report`, a pure transform that
  pairs the chat-mode per-item scores against the Attempt-1 raw-mode per-item
  scores over the identical IDs, computes the paired bootstrap CI (diagnostic
  seed) and exact McNemar per state, and reads the outcome against issue #30's
  two-row table.
- Offline tests: `tests/test_diagnostic_protocol.py`,
  `tests/test_diagnostic_chatmode_mmlu.py`, `tests/test_diagnostic_report.py`.

## Tokenization caveat

After a chat template's assistant-turn opener, a candidate string's leading
space (`" A"`) may tokenize differently than in raw-completion mode. The primary
run keeps the identical Attempt-1 candidate strings for comparability. The
diagnostic manifest triggers the no-leading-space robustness re-run
(`["A","B","C","D"]`) when any model state is *ambiguous*
(`|Δ| < 0.02` and the 95% paired-bootstrap CI crosses zero) **or** *near-chance*
(chat-mode accuracy within 0.03 of 0.25).

**The re-run fired** — the untrained baseline landed at chance (see below). It
was performed and both tables are reported. The two variants agree to within
~1 pp on every model state, so the leading-space tokenization concern is **not**
the explanation for the baseline's chat-mode collapse.

## Result

**The re-score data** (`diagnostics/chatmode-mmlu-report-final.json`; primary
run `diagnostics/chatmode-mmlu-20260902-125442/`, robustness re-run
`diagnostics/chatmode-mmlu-20260902-132049-nolead/`):

| Model state | MMLU raw mode (Attempt-1) | MMLU chat mode | Δ | 95% paired CI | exact McNemar p |
| --- | ---: | ---: | ---: | :---: | ---: |
| **baseline (untrained)** | 0.567 | **0.250** | **−0.317** | [−0.387, −0.247] | 7.5e-16 |
| seed 17 · epoch 1 | 0.630 | 0.537 | −0.093 | [−0.140, −0.047] | 2.3e-04 |
| seed 17 · epoch 2 | 0.607 | 0.563 | −0.043 | [−0.087, −0.003] | 6.0e-02 |
| seed 17 · epoch 3 | 0.593 | 0.573 | −0.020 | [−0.063, +0.020] | 4.3e-01 |
| seed 42 · epoch 1 | 0.643 | 0.543 | −0.100 | [−0.150, −0.050] | 1.3e-04 |
| seed 42 · epoch 2 | 0.613 | 0.563 | −0.050 | [−0.097, −0.007] | 4.4e-02 |
| seed 42 · epoch 3 | 0.600 | 0.550 | −0.050 | [−0.100, +0.000] | 6.3e-02 |
| seed 2026 · epoch 1 | 0.633 | 0.577 | −0.057 | [−0.097, −0.017] | 9.5e-03 |
| seed 2026 · epoch 2 | 0.610 | 0.570 | −0.040 | [−0.090, +0.007] | 1.4e-01 |
| seed 2026 · epoch 3 | 0.593 | 0.533 | −0.060 | [−0.110, −0.010] | 2.5e-02 |

The no-leading-space re-run gives the same picture: baseline 0.257, checkpoints
0.537–0.570, every checkpoint Δ within ~1 pp of the primary run.

**Reading** (report `verdict: confounded_by_baseline_modality_effect`):

1. **The untrained base model cannot do MMLU through the chat interface.** Its
   score falls from 0.567 in raw-completion mode to **chance** (0.250 / 0.257,
   4-way) with the chat template on — robust to both candidate-string forms. The
   `first_token_logit` ranking degenerates after the chat template's
   assistant-turn opener for this model.
2. **Every fine-tuned checkpoint declines only modestly** — mean ≈ −5.5 pp
   (range −2.0 to −10.0 pp), and only 4 of 9 are individually significant by
   exact McNemar at p < 0.05, though the direction is consistent (all nine
   negative).
3. **The fine-tune repairs chat-interface MMLU**, opposite to the
   "damaged the chat pathway" worry: checkpoints score 0.53–0.58 in chat mode
   versus the base model's 0.25. Response-only SFT on chat-templated data taught
   the model to emit a letter answer after the assistant-turn opener; the base
   model does not.
4. **The chat wrapper is not why MMLU survived Attempt-1.** Measured *through*
   the fine-tuned chat interface, MMLU on the checkpoints declines ~5 pp — nothing
   like the generation benchmarks' collapse in the same runs (GSM8K −18 to
   −48 pp, IFEval −18 to −22 pp). The largest single confounded axis (the chat
   template) does not overturn the finding.
5. **What this does not resolve:** the *scoring-modality* axis — likelihood
   ranking versus sampled free generation. MMLU is likelihood-ranked in both raw
   and chat mode, so this test does not separate that axis. It remains the open
   confound, and issue #28's primary-table caption should keep saying so.

Net: the diagnostic **strengthens** the Attempt-1 MMLU result against the
chat-template confound and adds a finding (the intervention improves chat-mode
MMLU relative to the base model), while leaving the scoring-modality confound
explicitly untested.

## Resource use

| Pass | Wall / GPU-hours | Peak VRAM |
| --- | ---: | ---: |
| Code-path smoke (`--max-items 3`, not evidence) | 0.089 | 6.60 GiB |
| Primary re-score (300 items × 10 model states) | 0.326 | 12.28 GiB |
| No-leading-space robustness re-run | 0.327 | 12.32 GiB |
| **Diagnostic total (two scientific passes)** | **0.652** | 12.32 GiB |

Budgeted at ~2 GPU-hours in the issue-30 handover; **0.652 GPU-hours** used for
the two full passes (0.74 including the smoke). Recorded on the diagnostic's own
resource lines (`diagnostics/chatmode-mmlu-<stamp>-resource.json`, each a
`runner.integrity_report` `non_scientific_runs` row) and folded into the combined
all-incurred-compute figure only: cumulative scientific GPU use is unchanged at
47.34 of 72 hours; all-incurred compute rises to ≈ 48.1 hours. Peak VRAM
12.32 GiB is well within the 15.5 GiB soft cap (this is inference-only, no
training-phase memory).
