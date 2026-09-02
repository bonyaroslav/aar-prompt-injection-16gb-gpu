# Issue #30 handover — Chat-mode MMLU confound test

**Written:** 2026-09-02, after #27/#28/#29 closed. Decision-neutral: records
state and pointers, does not pre-decide the diagnostic protocol's contents.

## Start here

1. **This is a real-GPU ticket.** It re-scores MMLU (no retraining) on the
   RTX 4080 under WSL2/CUDA. If that environment is not reachable, the
   `work-next-issue` rule is **stop and report** — do not simulate it, do not
   skip to #31. First real command: confirm `nvidia-smi` works and the venv
   imports torch+CUDA.
2. **Follow `/work-next-issue` + TDD**, exactly as #27/#28/#29 did: orient →
   read the full issue → implement with offline fixture tests → run full suite →
   write `docs/issue-30-*-decision.md` → update `RESEARCH_PLAN.md` Status
   paragraph → commit `Closes #30` with the `Co-Authored-By: Claude Sonnet 5`
   trailer → `gh issue close 30` with an evidence comment → **do not push**.
3. **Read first:** the memory index `MEMORY.md`, then
   `[[seed-run-split-recovery]]` and `[[test-env-torch-gap]]`; the full issue
   body (`gh issue view 30`); `docs/issue-14-finalization-handover.md`;
   `docs/issue-29-...-decision.md`.

## Repo / evidence state

- `master` is **3 commits ahead of `origin/master`, not pushed** (`9a93e06`
  #27, `c797289` #28, `5ec29a0` #29). Working tree clean.
- Analysis chain: #27, #28, #29 closed. Open: **#30**, #31, #32, #33. #30 is
  blocked only by #27 (closed) — it is the next eligible ticket. #30 does *not*
  depend on #29.
- Frozen protocol `phase1-2026-08-29` (`protocol/manifest.json`). Canonical-JSON
  manifest digest `399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20`
  (publication identity); raw-file digest `296e093b...` (byte integrity). See
  `protocol/digests.md`.
- Replication set `[17, 42, 2026]` complete, all `NO_ELIGIBLE_CHECKPOINT` null
  selections. Cumulative scientific GPU use 47.34 of 72 h (~24.7 h remain; #30
  is budgeted at ~2 h).

## Test env (unchanged)

- Interpreter: `.venv/Scripts/python.exe` (CPython 3.13, **no torch**).
- Run scripts / tests with `PYTHONPATH=.` prefixed.
- Full suite: `.venv/Scripts/python.exe -m unittest discover -s tests -q`
  — currently **313 pass / 1 skip** (~155 s). The one skip is the known
  missing-`torch` case in `test_real_training`; that is the baseline, not a
  regression. "Existing suite passes unchanged" = still 313/1 plus whatever
  #30 adds.

## #30-specific facts already checked

- **All nine merged checkpoints are on disk**:
  `runs/training-seed{17,42,2026}-<stamp>/checkpoints/epoch-{1,2,3}/`. The
  acceptance criterion still requires verifying they **digest-match** before
  spending GPU time — recorded digests are in each training bundle's
  `execution.log` (`epoch N: checkpoint merged fingerprint=<sha256>`) and each
  eval bundle's `manifest.yaml` (`"checkpoint": "<sha256>"`). Seed stamps:
  17 → `20260830-071553`, 42 → `20260831-201248-1b487000`,
  2026 → `20260901-112915-bf0809d1`.
- **The chat-template flag lives at `runner/real_adapters.py` ~line 235**: the
  eval path's `if benchmark == "mmlu":` branch calls
  `self.backend.candidate_logits(_mmlu_prompt(item), [" A"," B"," C"," D"],
  use_chat_template=False)`. There is a second `if benchmark == "mmlu":` at
  ~line 321. Make the flag a constructor parameter **defaulting to `False`**
  (so Attempt-1 behaviour is provably unchanged — add a regression test), and
  enable it only under the diagnostic protocol version. The issue says: no new
  injection seam, the existing injected-backend seam covers it.
- **MMLU config** (`evaluation.capability.mmlu`): `scorer: first_token_logit`,
  `max_new_tokens: 1`, 300 items, `sample_ids:
  capability_publisher_seed_42_first_300`. MMLU is the one benchmark that
  survived Attempt-1 (baseline 0.5667; seed-17 epochs 0.63 / 0.61 / 0.59 — flat
  to slightly up while GSM8K/IFEval collapsed). `protocol/deviations.md`
  already records (issue #29) that this `first_token_logit` scorer is a
  manifest/upstream drift — #30 is the test of what that drift costs.

## Authorization and boundaries (from the issue, binding)

- The diagnostic protocol version is **authorized in writing in the issue
  body** — you are executing an authorized deviation, not deciding one. It
  needs its own manifest + digest, declaring the frozen Attempt-1 baseline as
  its baseline with the reuse justification written down (model identity,
  revision, suite, sample IDs, decoding, scorers all unchanged — only the
  template flag differs).
- Diagnostic outputs are stored **separately**; they must never enter an
  Attempt-1 evidence bundle, the frozen 10 000-replicate bootstrap
  (`analysis.bootstrap_seed = 271828`), or checkpoint selection. **The held-out
  InjecAgent benchmark is untouched** — do not construct a `HeldOutSealer`, do
  not read the restricted root.
- GPU-hours consumed go on the diagnostic protocol's own resource line **and**
  into the combined all-incurred-compute figure. `runner.integrity_report`
  `resource_accounting` already has an `all_incurred_compute.non_scientific_runs`
  slot shaped `{category, label, gpu_hours, wall_hours, source}` for exactly
  this.
- **Tokenization caveat** to record: after a chat template's assistant-turn
  opener, a candidate string's leading space (`" A"`) may tokenize differently
  than in raw-completion mode. Primary run keeps the identical candidate
  strings; if the result is ambiguous, re-run once without the leading space
  and report both.

## Reusable primitives

- `runner.frozen_inputs.freeze_inputs` (#27) — the checksum-verified input set.
- `runner.claim_tables.build_claim_report` (#28) — modality-grouped tables +
  paired bootstrap + exact McNemar (pure, over `metrics.json` dicts).
- `runner.integrity_report.build_integrity_report` (#29) — failure-mode +
  integrity reports (pure). `parse_generation_signature`,
  `tensor_trust_distribution` exposed separately.
- Bundle / selection / analysis helpers: `runner.bundle`,
  `runner.real_adapters`, `runner.real_seed_run`, `runner.analysis`,
  `runner.continuation`, `runner.recovery` (recovery-boundary + checksum).
- `.gitattributes` pins `protocol/**`, `docs/**`, `analysis/**`,
  `RESEARCH_PLAN.md` to `text eol=lf` so digested files reproduce on a fresh
  clone — put any new digested diagnostic manifest under a path that this
  covers (or extend it).

## Stop conditions specific to #30

- Merged checkpoints missing or digest-mismatched → stop, report, **do not
  retrain**.
- CUDA/WSL not reachable → stop, report.
- Anything that would require the diagnostic result to feed selection, the
  bootstrap, or an Attempt-1 bundle → that is not this ticket.
