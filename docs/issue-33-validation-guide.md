# Issue #33 — how to independently validate the fact base

**Purpose:** everything in `docs/issue-33-claim-framing-dossier.md`,
`docs/issue-33-interpretations.md` and `docs/adr/0002-issue-33-claim-framing.md`
rests on a small set of committed, regenerable artifacts. This guide lets an
independent pass (a fresh session, or a human) confirm those artifacts are
faithful to the raw finalized evidence before the claim verdict is made.

Nothing here requires a GPU, a model, a network, or the held-out benchmark.

## 1. The committed artifacts

| File | What it is | Contains held-out data? |
|---|---|---|
| `analysis/attempt1-claim-report.json` | rendered `runner.claim_tables.build_claim_report` output — per-benchmark baseline/trained means, deltas, paired bootstrap, exact McNemar, visible composite, cross-run summary | No (`assert_no_reveal_bundle` enforced) |
| `analysis/attempt1-integrity-report.json` | rendered `runner.integrity_report.build_integrity_report` output — failure-mode evidence + integrity records | No — sealed counts (133/67) and MDE figures only, same values already receipted in the provenance manifest |
| `analysis/attempt1-frozen-input-record.json` | rendered `runner.frozen_inputs.freeze_inputs` output — every finalized input path bound to a digest | No |
| `analysis/publication-provenance-manifest.json` | #32 provenance manifest — a receipt for every numeric leaf of the two reports | No |

Raw-file SHA-256 (of the LF-normalized committed bytes), recorded in
`docs/adr/0002-issue-33-claim-framing.md`. These are convenience anchors; the
authoritative identity is the canonical protocol digest
`399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20` plus the
regeneration check below.

## 2. Regenerate and confirm determinism

From the repo root, with the `runs/` and `recovery/` and `data/training/` trees
present locally (they are gitignored execution evidence):

```
python -m runner.publication_gate_run \
    --evidence-root runs --recovery-root recovery \
    --dump-reports analysis \
    --out analysis/publication-provenance-manifest.json
```

Expected stdout: `reports=2 sections=13 receipted_numbers=1763 orphans=0
claim_language_violations=0`, exit 0.

Then `git diff --stat analysis/` must show **no change** to any of the four files.
The run is byte-deterministic: it re-computes the same 10,000-replicate paired
bootstrap (`analysis.bootstrap_seed = 271828`) and canonical-JSON serialization
every time. A non-empty diff means either the evidence tree changed or a
transform changed — investigate before trusting any number.

## 3. Hand-verify headline numbers against raw evidence

Each row: open the raw file, read `benchmarks.<name>.aggregate.value`, compare.
All six visible benchmarks sample 300 or 200 fixed items; the aggregate is the
mean per-item score.

| Claim in the dossier | Raw file | Field | Expected |
|---|---|---|---|
| Baseline GSM8K 0.735 | `runs/real-baseline-20260829-205020/metrics.json` | `benchmarks.gsm8k.aggregate.value` | `0.735` |
| Baseline IFEval 0.615 | same | `benchmarks.ifeval.aggregate.value` | `0.615` |
| Baseline MMLU 0.567 | same | `benchmarks.mmlu.aggregate.value` | `0.5666666666666667` |
| Baseline OPI 0.180 | same | `benchmarks.open_prompt_injection.aggregate.value` | `0.18` |
| Seed 42 epoch 3 GSM8K 0.500 (−0.235) | `runs/eval-seed42-epoch3-20260831-201248-1b487000/metrics.json` | `benchmarks.gsm8k.aggregate.value` | `0.5` |
| Seed 42 epoch 3 IFEval 0.485 (−0.130) | same | `benchmarks.ifeval.aggregate.value` | `0.485` |
| Seed 42 epoch 3 MMLU 0.600 (+0.033) | same | `benchmarks.mmlu.aggregate.value` | `0.6` |
| Seed 42 epoch 3 OPI 0.667 (+0.487) | same | `benchmarks.open_prompt_injection.aggregate.value` | `0.6666666666666666` |
| Seed 17 epoch 1 GSM8K 0.260 (−0.475) | `runs/eval-seed17-epoch1-20260830-071553/metrics.json` | `benchmarks.gsm8k.aggregate.value` | `0.26` |
| Seed 17 epoch 1 OPI 0.507 (+0.327) | same | `benchmarks.open_prompt_injection.aggregate.value` | `0.5066666666666667` |

Cross-check that the same values appear in
`analysis/attempt1-claim-report.json` under `primary_table.modality_groups.*.rows`
(fields `baseline`, `trained`, `absolute_delta`).

## 4. Verify the "would a MMLU-only gate have passed?" number

In `analysis/attempt1-claim-report.json`, every row of
`primary_table.modality_groups.*.rows` has
`multiple_choice_only_gate_passes`. Confirm it is `true` for all 45 rows
(9 checkpoints × 5 generation benchmarks; the value is per-checkpoint, repeated).
The rule: MMLU decline ≤ `selection.capability_gates.mmlu_max_decline` (= 0.02).
Since every checkpoint's MMLU *rose*, the decline is negative and the gate passes.
Contrast with `visible_composite` / the per-seed outcomes summaries, where every
checkpoint FAILS the gates that actually applied (GSM8K ≤ 0.02, IFEval ≤ 0.03,
retention ≥ 0.98).

## 5. Verify the generation-failure signature

`analysis/attempt1-integrity-report.json`
`failure_mode_evidence.generation_failure_signature`:

- `baseline.seconds_per_item.gsm8k` ≈ `45.96`; `baseline.truncated_completions`
  = `190`.
- `per_checkpoint` has **6** rows (seeds 42 and 2026). `unavailable` has **3**
  rows (seed 17 — its bundles predate the machine-readable
  `completions_truncated` / timing log lines). **The "outputs got shorter"
  mechanism evidence therefore covers 6 of 9 checkpoints.** This is a stated
  limitation, not a defect; weigh it when deciding D2.
- Each seed-42/2026 row: `seconds_per_item_delta_vs_baseline.gsm8k` between
  `-37` and `-40`; `seconds_per_item_delta_vs_baseline.ifeval` between `-11` and
  `-15`; `truncation_delta_vs_baseline` negative (fewer runaway completions).

## 6. Verify the corpus nutrition label

`failure_mode_evidence.corpus_nutrition_label`, computed from
`data/training/dataset.jsonl` + `report.json` (gitignored; digest-only supplement
`training_corpus_digest_only_supplement` in the provenance manifest):

- `total_examples` = `5000`, `distinct_assistant_responses` = `2505`.
- `most_frequent_response_coverage.top_1` = `0.1`, `top_10` ≈ `0.5004`.
- `response_length_words.median` = `46`; `multi_step_reasoning_share` = `0.029`.

The label contains **counts and distributions only — no response text**. To
re-derive independently: parse `dataset.jsonl`, take the last assistant message of
each row, count frequencies.

## 7. Verify resource totals (the "under 72 h" criterion)

`integrity_records.resource_accounting`:

- `scientific_totals.gpu_hours` ≈ `47.338` (baseline 6.255 + seed 17 15.185 +
  seed 42 12.743 + seed 2026 13.155; additive by seed).
- `all_incurred_compute.non_scientific_runs` is currently `[]`. **Issue #33's
  package run must fold in** #30 (0.652 GPU-h, from
  `docs/issue-30-chatmode-mmlu-diagnostic-decision.md`) and #31 (11.885 GPU-h,
  from `docs/issue-31-corpus-ablation-decision.md`) → all-incurred ≈ **59.87 h**
  < 72.
- `peak_vram.value_gb` ≈ `15.663`, `phase` = `training`, `overage_gb` ≈ `0.163`
  over the 15.5 GiB declared allocation, under the 16 GiB card.

## 8. Known caveats a validator will meet

Documented already; not defects, but they bound what the claim can say:

1. **Seed 17 timing gap** — §5 above; 6/9 checkpoints for the generation
   signature.
2. **Seed 17 post-hoc checkpoint digests** — seed 17's merged-checkpoint digests
   were first computed during #30 (it predates the #22 recovery seam); seeds
   42/2026 digest-match their run-time records. Bit-identity of seed 17's merged
   dirs to what Attempt-1 scored cannot be independently reconfirmed.
3. **Stale bundle captions** — the frozen baseline `notes.md` has a leftover smoke
   caption; seed-17 eval bundles' `command.sh`/`notes.md` wrongly say "fake
   adapters". `metrics.json` (the scored values) is correct and manifest-exact in
   every case. Finalized bundles are never hand-edited (evidence contract);
   disclose when describing provenance.
4. **#31 is single-seed** and its corpus was rebuilt through a different
   construction path — see `docs/issue-31-corpus-ablation-decision.md`.
5. **Cross-run summary is descriptive** — N = 3 runs, run-to-run variability under
   a fixed nominal config (adapter init precedes the run seed), not seed variance,
   not an inferential interval.

## 9. What the #32 gates do and do not verify

`runner.publication_gates` (run in step 2) verifies: every numeric leaf of the two
reports has a provenance receipt tying it to specific input digests (zero
orphans); the protocol canonical digest matches the frozen record; the bootstrap
parameters match; no forbidden efficacy word appears; every capability sentence
names its modality.

It does **not** verify: that the transform *logic* is correct (steps 3–7 do that
by hand); that the raw bundles were correct at finalization (only that they are
unmodified since); or anything about the not-yet-written `analysis/results.md`.
