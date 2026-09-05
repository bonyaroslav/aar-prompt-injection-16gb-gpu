# Independent review — pre-flight

**For:** the fresh session executing `docs/independent-review-prompt.md`.
**Written by:** a session that had already read `docs/handover/publication-analysis-final.md`
and is therefore disqualified from the review itself.

## What this document is, and what it deliberately is not

It exists to save you search time: verified environment facts, a map of the code, the
shape of the raw evidence, and the hard guardrails. It contains **no findings, no
conclusions, and no steer**. The author knows what the handover concluded and has
withheld all of it, including anything that would betray it by emphasis or omission.

The rule applied throughout: **narrow your search cost, never narrow your attention.**
Every table below is a full cross-product — all six benchmarks, all ten model states,
every scorer, every gate — rather than a selection. If something here looks like it is
pointing you at one number rather than another, treat that as a defect in this document
and ignore it.

Verify anything here that matters to a conclusion you draw. It was checked by command on
2026-09-04 against `master` @ `e820fa3`, but it is not evidence — the repo is.

---

## 1. Environment

| Fact | Value |
|---|---|
| Repo root | `C:\Projects\aar-prompt-injection-16gb-gpu`, branch `master` @ `e820fa3` |
| Python | `.venv/Scripts/python.exe` — Python 3.13.15. Use this, not system Python |
| Upstream `aar` repo | `C:\Projects\automated_alignment_researcher` |
| Upstream `HEAD` | `1899ad64fbfbc65790d259471cc4bf4de9437aa9`, tree `00f1edb9193487e7e306177709b1760be180d7ac` — **matches the `protocol/manifest.json` pin exactly** |
| Upstream working tree | dirty: `uv.lock` modified; `_holdout_medium/`, `_holdout_opi_full/`, `configs/medium.yaml`, `generate_medium_config.py` untracked |
| Gitignored but present locally | `runs/`, `recovery/`, `data/`, `diagnostics/`, `ablation/` — raw evidence is available to you |
| `runs/**/metrics.json` | 23 files. Ten are the model states under review; the rest are smoke and training runs |
| Tests | `tests/`, 29 files. `unittest`, not `pytest` — pytest is not installed. Run: `.venv/Scripts/python.exe -m unittest discover -s tests`. Verified 2026-09-04: `Ran 369 tests … OK (skipped=1)` in 166 s. The skip is a torch gap in the Windows venv; it is a clean skip, **not** a failure, so any error here means something regressed |
| `README.md` | exists — 163 lines, currently untracked. It carries the project's own defect list and non-claims |
| Network | required. 37 distinct URLs in the prompt, plus fresh searches |

**On the upstream working tree.** `protocol/manifest.json` sets
`working_tree_policy: "committed_head_only; dirty_untracked_checkout_is_not_evidence"`.
Read upstream code through the pin, not the checkout:

```bash
git -C /c/Projects/automated_alignment_researcher show 1899ad64fbfbc65790d259471cc4bf4de9437aa9:<path>
```

**On the knowledge cutoff.** This model's training data ends May 2026. A large share of
the prompt's citation list post-dates that. You cannot recall those papers; fetch every
one or mark it unverified. Do not reconstruct an abstract from memory and call it
verified.

---

## 2. What the raw evidence contains

Per-file shape of `runs/**/metrics.json`:

```
{"stage", "seed", "epoch", "checkpoint",
 "benchmarks": {<name>: {"aggregate": {"metric", "value"},
                         "items": {<item_id>: {"score", "valid"}}}}}
```

**The ten model states**, all with identical benchmark coverage and item counts:

| State | Path |
|---|---|
| frozen baseline | `runs/real-baseline-20260829-205020/metrics.json` |
| 9 checkpoints | `runs/eval-seed17-epoch{1,2,3}-20260830-071553/metrics.json` |
| | `runs/eval-seed42-epoch{1,2,3}-20260831-201248-1b487000/metrics.json` |
| | `runs/eval-seed2026-epoch{1,2,3}-20260901-112915-bf0809d1/metrics.json` |

| Benchmark | n per state |
|---|---:|
| `gsm8k` | 200 |
| `ifeval` | 200 |
| `mmlu` | 300 |
| `open_prompt_injection` | 300 |
| `tensor_trust_extract` | 300 |
| `tensor_trust_hijack` | 300 |

Verified: all six present in all ten states, counts identical across states, item IDs
shared — so every state is paired with every other state on a common ID set.

**Recomputable from `runs/` alone:** every aggregate, every per-item score, every delta,
and every paired per-item contrast between any two of the ten states, for all six
benchmarks.

**Not recomputable — you must mark these "could not verify":** anything depending on the
*content*, *length*, or *timing* of a model output. Per-item records carry `score` and
`valid` only; no generated text is retained anywhere in the repository. Claims of that
kind rest on aggregate signatures in `analysis/attempt1-integrity-report.json`, which you
can read and reason about but cannot independently re-derive without GPU time. Say so
plainly wherever it applies, rather than treating that report as raw evidence.

The held-out benchmark contributes no per-item data at all — see §4.

---

## 3. Code map

A grep-saver. Every scorer, every loop, every seed call. Line numbers verified against
`e820fa3`; re-check before citing, since they drift.

| What | Location |
|---|---|
| Per-benchmark scoring, all six | `runner/real_adapters.py:313` — `RealScorerAdapter.score`, dispatched by benchmark name |
| Upstream scorer bindings | `runner/real_adapters.py:296-311` — `_load_upstream_apis` |
| Model adapter and decoding config | `runner/real_adapters.py:135-175` |
| Evaluation loop | `runner/evaluation.py:109-126` — `_score_benchmark`; `model.generate` at `:121` |
| Training, main path | `runner/real_training.py:459-478` — `_initialize` at `:465`, `torch.manual_seed` at `:472` |
| Training, ablation path | `runner/real_training.py:292-322` — `_initialize` at `:301`, `torch.manual_seed` at `:308` |
| `_initialize` body | `runner/real_training.py:393-458` |
| Selection and capability gates | `runner/selection.py` |
| Claim tables, paired bootstrap, exact McNemar | `runner/claim_tables.py` |
| Failure-mode evidence | `runner/integrity_report.py` |
| Provenance and claim-language gates | `runner/publication_gates.py` — `verify_provenance`, `check_claim_language:282-304` |
| Gate CLI | `runner/publication_gate_run.py` |
| Upstream benchmark modules | `<upstream>/aar/benchmarks/{gsm8k,ifeval,mmlu,open_prompt_injection,tensor_trust_extract,tensor_trust_hijack,injecagent}/` |

Prompt Rule 3 asks you to read the code rather than the prose about it. The reports in
`analysis/` make behavioural claims about all six benchmarks; each is checkable against
the modules above and against the pinned upstream. Check them, not a subset.

**One API detail.** `check_claim_language` takes a list of structured report dicts and
walks `report["sections"][…]["content"]`. It does not read Markdown. Running it over a
document means wrapping the text in that shape yourself.

---

## 4. Guardrails

Non-negotiable. Breaking any of these invalidates the review.

1. **Do not unseal the held-out benchmark.** `protocol/heldout_sealing.md` governs.
   InjecAgent is `SEALED` / `NEVER_AUTHORIZED`. The only held-out data in the repository
   is an opaque receipt under `held_out` in the baseline `metrics.json` — digests and
   valid/invalid counts. Do not seek per-candidate outcomes, aggregates, comparisons, or
   the restricted blob. Read `protocol/heldout_sealing.md` in full, including its section
   on what sealing does and does not mean, before writing anything about the held-out
   design.
2. **Do not read `_holdout_medium/` or `_holdout_opi_full/`** in the upstream checkout.
   Untracked, outside the pinned tree, outside the evidence contract.
3. **Read-only on evidence.** Do not edit anything under `runs/`, `recovery/`,
   `analysis/`, or `protocol/`. The single permitted write is the regeneration in §6,
   whose entire purpose is to produce an empty diff.
4. **No GPU, no new measurement.** The review scopes and costs future experiments; it
   does not run them.
5. **Invent no numbers.** Cite nothing this project did not measure. Provenance is the
   project's load-bearing asset and one fabricated figure discards it.
6. **Do not open `docs/handover/`** — four files: `analysis-handover-post-seed3.md`,
   `issue-30-handover.md`, `issue-31-handover.md`, `publication-analysis-final.md` —
   until sections 1–5 and 7 of your review are written to disk. This is the whole point
   of commissioning you.

---

## 5. Recomputation

Write your own script. Derive first, compare second — do not read a number out of
`analysis/*.json` and then go looking for it in `runs/`. Compute over **all 6 benchmarks
× 10 model states**, with no benchmark or checkpoint singled out in advance.

1. **Aggregates.** Re-derive each `benchmarks.<name>.aggregate.value` as the mean of
   `items.*.score`. Check all 60 cells against the stored aggregate, and check the item
   counts against the table in §2.
2. **Deltas.** Every checkpoint minus baseline, per benchmark — a 6 × 9 table. Examine
   each row for the same standard properties: sign, size, trend across the three epochs,
   and spread across the three seeds.
3. **Paired structure.** For every benchmark × checkpoint, build the 2×2 table against
   baseline over the shared item IDs — `n00, n01, n10, n11` — with the discordant count
   `n01 + n10` as a fraction of n, and exact two-sided McNemar. Compare against
   `analysis/attempt1-claim-report.json` → `mcnemar_exact`.
4. **Paired bootstrap.** 10,000 replicates, seed `271828`, 95th-percentile paired by
   fixed example ID — all of it declared in `protocol/manifest.json` → `analysis`.
   Compare intervals against the claim report.
5. **Gates.** `protocol/manifest.json` → `selection.capability_gates` declares four:
   `mmlu_max_decline` 0.02, `gsm8k_max_decline` 0.02, `ifeval_max_decline` 0.03,
   `mean_normalized_retention_min` 0.98. Re-apply each independently to all nine
   checkpoints and record, per checkpoint, which of the four it passes and which it
   fails. Also check `visible_composite` and `meaningful_improvement_absolute` (0.05).
6. **Resources.** Re-add the per-seed GPU-hour figures plus the follow-up runs recorded
   in `docs/issue-30-chatmode-mmlu-diagnostic-decision.md` and
   `docs/issue-31-corpus-ablation-decision.md`. Check the total against the declared cap
   and against `integrity_records.resource_accounting`.

`docs/issue-33-validation-guide.md` §3–§7 lists ten spot values with exact field paths,
and §8 lists caveats the project already knows about. Use it as a cross-check after you
have derived your own numbers, and treat §8 as claims to test rather than facts to accept.

---

## 6. Regeneration

```bash
.venv/Scripts/python.exe -m runner.publication_gate_run --dump-reports analysis --out analysis/publication-provenance-manifest.json
```

`--evidence-root runs` and `--recovery-root recovery` are the defaults, so the prompt's
short form and the validation guide's long form are equivalent.

Operational note: this runs fine under the Windows venv — it needs no torch and no GPU,
and completes in a few seconds. It was executed on 2026-09-04 and did produce the
expected stdout and an empty diff. Run it yourself anyway; that it passed once for
someone else is not your evidence.

Expected stdout, per `docs/issue-33-validation-guide.md:38`:
`reports=2 sections=13 receipted_numbers=1763 orphans=0 claim_language_violations=0`,
exit 0.

Then `git diff --stat analysis/` **must be empty**. A non-empty diff is itself a finding
— investigate before trusting anything downstream. Also confirm the four artifact
SHA-256 values at `docs/adr/0002-issue-33-claim-framing.md:38-43` and the protocol
canonical digest `399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20`.

Note what these gates check and what they do not:
`docs/issue-33-validation-guide.md:147-157` states the boundary in the project's own
words. Decide for yourself whether that boundary is drawn honestly.

---

## 7. Documents in the repo

Inventory, not a reading list. No ranking is implied and read order is yours.

| Path | What it holds |
|---|---|
| `README.md` | the repo's front door: result table, non-claims, disclosed defects |
| `RESEARCH_PLAN.md` | the study plan |
| `protocol/manifest.json` | the frozen protocol |
| `protocol/deviations.md` | recorded drifts between manifest and code |
| `protocol/heldout_sealing.md` | the sealing and reveal procedure |
| `protocol/power_notes.md` | minimum-detectable-effect figures |
| `protocol/digests.md` | digest conventions |
| `analysis/attempt1-claim-report.json` | per-benchmark deltas, paired bootstrap, exact McNemar, composites |
| `analysis/attempt1-integrity-report.json` | failure-mode evidence and the disclosure list |
| `analysis/attempt1-frozen-input-record.json` | finalized input paths bound to digests |
| `analysis/publication-provenance-manifest.json` | a receipt for every reported number |
| `analysis/seed{17,42,2026}-outcomes-summary.md` | per-seed results |
| `docs/issue-33-claim-framing-dossier.md` | the fact base with provenance |
| `docs/issue-33-interpretations.md` | six candidate readings of the same evidence |
| `docs/issue-33-validation-guide.md` | the project's own validation recipe |
| `docs/adr/0002-issue-33-claim-framing.md` | five open wording decisions, unsigned |
| `docs/adr/0001-training-data-sources.md` | corpus source decisions |
| `docs/issue-*-decision.md` | per-issue decision records, #12 through #33 |
| `docs/handover/` | **blocked until section 6** |
| `runner/`, `training_data/`, `scripts/` | the code that produced every number |
| `tests/` | 29 test modules |
| `runs/**/metrics.json` | raw per-item scores (gitignored, present locally) |
