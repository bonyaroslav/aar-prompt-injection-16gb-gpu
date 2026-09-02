# Issue #31 handover — Corpus ablation (why capability collapsed)

**Written:** 2026-09-02, after #30 closed and pushed (`origin/master` @ `6fc8b61`).
Decision-neutral where the issue is; records the checks already done and the two
questions already answered.

## Start here

1. **Real-GPU ticket.** One seed, three epochs, ~12.7 GPU-hours, one variable
   changed (the training corpus). If `nvidia-smi` fails or the WSL venv can't
   import torch+CUDA, the `work-next-issue` rule is **stop and report** — do not
   simulate, do not skip to #32. First command: confirm `nvidia-smi` and
   `wsl -d Ubuntu -e bash -lc '<venv>/python -c "import torch;print(torch.cuda.is_available())"'`.
2. **Follow `/work-next-issue` + TDD**, exactly as #27–#30 did: orient → read the
   full issue (`gh issue view 31`) → implement with offline fixture tests → run
   full suite → write `docs/issue-31-corpus-ablation-decision.md` → update
   `RESEARCH_PLAN.md` Status → commit `Closes #31` with the
   `Co-Authored-By: Claude Sonnet 5` trailer → `gh issue close 31` with an
   evidence comment → **do not push** (the maintainer pushes).
3. **Read first:** `MEMORY.md`, then `[[seed-run-split-recovery]]` and
   `[[test-env-torch-gap]]`; the full issue body; `docs/issue-26-mid-epoch-training-recovery-decision.md`;
   `docs/issue-30-chatmode-mmlu-diagnostic-decision.md` (the diagnostic-protocol
   pattern this ticket mirrors); `docs/superpowers/specs/2026-09-02-ablation-mid-epoch-recovery-design.md`.

## Repo / evidence state

- `origin/master` @ `6fc8b61` (#30). Working tree clean. #27–#30 closed.
  Open analysis chain: **#31**, #32, #33.
- #31 blocked only by #26 and #29 — both closed. It is the next eligible ticket.
- Frozen protocol `phase1-2026-08-29`. Frozen replication set `[17, 42, 2026]`
  complete, all `NO_ELIGIBLE_CHECKPOINT`. Cumulative scientific GPU 47.34 of 72 h
  (~24.7 h remain; #31 budgeted at ~12.7 h → leaves ~12 h; #32/#33 are
  analysis-only, no GPU). #30's 0.65 GPU-h is all-incurred only.

## The two questions already answered (do not re-litigate)

### 1. Which seed → **seed 42**

The issue never states it, but its budget is copied verbatim from seed 42's
`runs/seed42-resource-comparison/`: training phase 28,590 s = **7.94 h**
("about 7.9 hours training"), evals 1.38 / 1.56 / 1.86 h (avg ~1.6 h,
"three evaluations at about 1.6 hours each"), total **12.74 h**
("about 12.7 GPU-hours"). Seed 17 = 15.18 h, seed 2026 = 13.16 h — neither fits.
Seed 42 is also the cleaner reference: full machine-readable timing lines
(seed 17's lack them, #29), checkpoint digests verified against run-time
`recovery/` state (#30 finding — seed 17 has none), correct eval-bundle captions
(seed 17's say "fake adapters", #12 defect), 0 interruptions.

**Run the ablation with seed value 42.** Read the appendix comparison against
`runs/eval-seed42-epoch{1,2,3}-20260831-201248-1b487000/` and
`analysis/seed42-outcomes-summary.md`. Seed-42 collapse to reproduce/contrast:
GSM8K decline −0.455 / −0.225 / −0.235, IFEval −0.23 / −0.13 / −0.13, MMLU flat.
Run-to-run SD across the 3 seeds on GSM8K/IFEval is ~0.02–0.03; the collapse is
~0.2–0.45, i.e. ~10× larger, so the single-run-vs-seed-42 comparison is
interpretable despite adapter-init-before-seed (the issue says this; **do not fix
the init ordering** — keep the trainer byte-identical).

### 2. Training-corpus source datasets → **resolved, no blocker**

`databricks/databricks-dolly-15k`, `deepset/prompt-injections`,
`Lakera/gandalf_ignore_instructions` had fallen out of the HF cache. Re-fetched
2026-09-02 (online, no token) and **verified they match Attempt-1 exactly**:
dolly 15,011 rows (= `report.json` `dolly_pool_size`), deepset label==1 = 263 +
gandalf = 1,000 → injection raw pool 1,263 (= `injection_raw_pool_size`), dolly
with `context` field 29.8%. **Not drifted.** They are now re-cached and load
offline (`HF_HUB_OFFLINE=1` verified). The Attempt-1 corpus
(`data/training/dataset.jsonl`, 5,000 rows) is still on disk *and* reproducible.

Residual gap to close inside #31 (not a blocker): `training_data/sources.py` pins
no `revision`; `report.json` records only pool sizes, no content digest. The
ablation protocol manifest should record the fetched `dolly_pool_size` /
`injection_raw_pool_size` and a content digest of the dolly rows (or pin
`revision=` on the three `load_dataset` calls). Exclusion-pool datasets
(sst2 / sms_spam / hate_speech_offensive) are cached and load offline.

## #31-specific facts already checked

- **Builder:** `training_data/build.py`. `_DOLLY_OVERSAMPLE_FACTOR = 3` and
  `_DOLLY_SHUFFLE_SEED = 20260830` (line ~38–39) are module constants, not
  protocol values. `build_dataset(..., targets=...)` already takes a per-category
  target map. `run_real_build` calls deepset+gandalf+dolly **unconditionally** —
  with `prompt_injection: 0` the injection rows are fetched but unused
  (`generate_prompt_injection_examples(rows, 0, ...)` → []); either leave it or
  add a skip. The oversample factor must become a parameter **defaulting to 3**,
  with an offline test (fake rows) asserting the default path is byte-identical
  to the pre-change output.
- **Corpus capacity trap (from the issue):** clean_control → ~3,500 at
  oversample 3 reserves 10,500 of ~15,000 Dolly rows, starving ambiguous_boundary
  (only ~30% context-bearing → ~1,350 candidates vs 1,000 target before dedup).
  Pass a lower oversample factor for this build.
- **Mid-epoch recovery:** `runner/ablation_training.py` (#26) —
  `run_ablation_epoch(protocol_version=..., ...)` rejects `phase1-2026-08-29`
  (raises "mid-epoch recovery is ablation-only"), so the ablation protocol needs
  its **own** `protocol_version`. State store is two-slot atomic in the external
  recovery workspace, outside `runs/`. Each epoch returns `recovery_evidence`
  with `mid_epoch_resume_fired` + per-save `{step_index, byte_count, save_seconds}`
  — the decision record must preserve these. Measured save (CPU fixture): 4,057
  bytes / 0.012 s at step 3. If a real save exceeds ~30 s, raise the interval.
- **Deliberate-kill test (acceptance criterion):** kill partway through epoch 1,
  re-run the identical command, run completes and the resume is in the attempt
  ledger.
- **Diagnostic-protocol pattern to copy from #30:** `protocol/diagnostic/`
  (`manifest.py` loads + fails closed if the frozen manifest's canonical digest
  drifts; `digests.md`), outputs under a gitignored top-level dir (`.gitignore`
  already has `diagnostics/`; add e.g. `ablation/` the same way), a
  `runner.integrity_report` `non_scientific_runs` resource row, and never
  entering an Attempt-1 bundle / the frozen bootstrap (`analysis.bootstrap_seed
  = 271828`) / selection / held-out. `.gitattributes` `protocol/**` and `docs/**`
  are LF-pinned; put the new digested manifest under one of those.
- **Peak VRAM** will again be ~15.6 GiB (identical training config) — over the
  15.5 GiB soft cap, under 16 GiB physical. All three Attempt-1 seeds did this;
  record the same feasibility finding, it is not a stop.
- **Disk:** 1.1 TB free, ~40 GB needed (3 merged checkpoints + adapters + 3 eval
  bundles). Fine.

## Reusable primitives

- `runner.ablation_training` (#26) — optimizer-step recovery.
- `runner.real_seed_run` / `runner.real_training` / `runner.evaluation` — the
  frozen train+eval path; the ablation reuses the eval half unchanged.
- `runner.real_adapters` — `RealModelAdapter` (eval), `RealQLoRATrainerAdapter`.
- `runner.claim_tables` / `runner.analysis` — paired bootstrap + McNemar for the
  appendix comparison (use the ablation manifest's own bootstrap seed, not
  271828), same as #30's `runner/diagnostic_report.py`.
- `training_data.build` / `.sources` / `.templates` / `.dedup` / `.exclusion_pool`.

## Execution — light status poll every 20 minutes

The ~12.7 h run must be launched detached and left alone. **Poll every 20 min**
with a *light* report only — do NOT read metrics.json / gpu.csv / execution.log
unless something looks wrong. Each report is ~3 lines:

- **elapsed** since the run started and since the last epoch/eval boundary you
  can see;
- **stage**: which epoch or which eval (infer from the bundle/adapter dirs that
  have appeared under the ablation output dir + recovery state file mtimes);
- **health**: one `nvidia-smi` line (GPU mem + util) and one WSL `free -h` line
  (RAM), plus "new dir: <name>" if one appeared since last poll;
- **ahead**: what's left (e.g. "epoch 2 of 3, then 3 evals").

A cheap one-liner is enough, e.g.:
`nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; wsl -d Ubuntu -e bash -lc 'free -h | sed -n 2p; ls -t <ablation-dir> | head -3'`

Only escalate to reading files if: the process died, GPU util sits at 0% for
two consecutive polls while the process is alive, RAM is exhausted, a save
exceeds ~30 s (check `recovery_evidence`), or peak VRAM crosses 16 GiB.
Mid-epoch recovery (#26) means a crash is recoverable by re-running the identical
command — that is the designed response, not a stop.

Launch pattern (from `[[seed-run-split-recovery]]`): detach so it survives the
session — `setsid nohup … </dev/null > <log> 2>&1 &` from inside a WSL shell, or
run it as a background task and poll. Smoke the harness end-to-end first
(a few steps / few eval items), same as #30 did, before the real launch.

## Stop conditions specific to #31

- CUDA/WSL not reachable → stop, report.
- Corpus build shortfall in any category, or any prompt-injection example present
  in the ablation corpus → stop, report (do not hand-fix the corpus).
- A save consistently exceeding ~30 s and raising the interval doesn't help →
  stop, report.
- Anything requiring the ablation result to feed selection, the frozen bootstrap,
  or an Attempt-1 bundle → that is not this ticket.
- Cumulative GPU-hours from the ledger would exceed 72 → stop, report.
