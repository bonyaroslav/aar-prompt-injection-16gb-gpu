# Issue #22 seed-42 execution and recovery-aware split-run decision

**Issue:** [#22](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/22)
**Decision date:** 2026-09-01
**Protocol:** `phase1-2026-08-29` (`protocol/manifest.json`, manifest digest
`399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20`)
**Scope:** wires the resumable split-run seam (issue #15) into the real seed
orchestration and executes seed 42 under it. No frozen protocol, dataset,
hyperparameter, seed, decoding, selection-rule, capability-gate, held-out, or
resource-limit value is changed. Held-out InjecAgent is not read: the finalized
selection is null.

## Continuation authority

Issue #21 recorded a passing continuation decision: seed 17 completed
technically in 15.1848 measured GPU-hours, cumulative use was 21.4401
GPU-hours, and the frozen rule projected 45.5544 GPU-hours for all three seeds
— under the 72-hour cap with 50.5599 hours remaining. Seed 42 was therefore
authorized under its own scope and gates. Seed 17's `NO_ELIGIBLE_CHECKPOINT`
interpretation did not pre-decide seed 42.

## Execution-path finding (acceptance criterion: recovery-aware path)

Before this issue, `runner.real_seed_run.run_real_seed` was a **one-shot
launcher**: a fresh timestamp per invocation, no `recovery=` wiring into
`run_training` / `run_trained_evaluation`, and `write_seed_comparison_artifact`
raising `FileExistsError` on any re-run. The issue #18 decision explicitly
deferred "wiring an actual multi-session resume into `runner.real_seed_run`
(smoke-limited exact command path + resume detection)" to #22. The old launcher
was **not** sufficient and was not assumed to be.

## Decision — the split-run seam

`runner.real_seed_run` now owns a stable per-seed identity above the existing
stage recovery contracts (#17 signatures / workspace / ledger, #19 completed
epochs, #18 whole-checkpoint evaluations, #20 held-out transaction):

- `_orchestrate_seed` routes every stage through its own recovery contract with
  a **stable stage key** (`training-seed42`, `eval-seed42-epoch{1,2,3}`,
  `reveal-seed42`). The per-attempt run-bundle id stays timestamped and unique;
  recovery keys never depend on it.
- Recovery state — signatures, per-epoch/per-example journals, the append-only
  attempt ledger — lives in an operational workspace (`recovery/`,
  `--recovery-root`) **outside** the `runs/` evidence root. `RecoveryWorkspace`
  rejects a root inside the evidence root; `runner.recovery.finalized_inputs_only`
  rejects any recovery path presented as a finalized input.
- An ordinary interruption (WSL shutdown, killed process, CUDA fault)
  propagates; the next invocation of the identical command
  (`runs/seed42-run.sh`) resumes from the last completed epoch / whole
  evaluation / finalized selection. A completed seed re-run is a
  checksum-verified no-op.
- The selection record (`runs/selection-seed42/selection_record.json`) and the
  resource artifact (`runs/seed42-resource-comparison/`) are written at **stable
  ids**; re-finalizing byte-identical content is idempotent, divergent content
  is rejected.
- `aggregate_seed_resource_intervals` sums every completed and interrupted GPU
  attempt from the ledger; explicitly declared hard-power-loss gaps
  (`--unavailable-interval SECONDS:REASON`) are recorded as `unavailable`,
  never silently counted as zero and never counted as GPU time.
- `seed_run_status` (`--status`) reports completed / interrupted / recoverable
  stages and the exact next continuation action, reading durable state only.
- `discover_finalized_seed_evidence` validates the final topology: one training
  result, three evaluations, one finalized selection, resource evidence, clean
  checksums, and a reveal bundle **iff** the selection is eligible — with every
  recovery-workspace path excluded.

Mid-epoch training recovery is **not** adopted (issue #19 completed-epoch
fallback, unchanged). The one authorized 2048→1536 OOM restart keeps its
distinct signature namespace; it was not triggered (no OOM).

## Smoke validation (exact command path)

`runs/seed42-run.sh` with `--smoke-max-steps 1 --smoke-max-items-per-benchmark 2`
and a throwaway `--output-root` / `--recovery-root` exercised the entire path on
the RTX 4080: model-cache resolution, storage, a real QLoRA step + merged
checkpoint reload per epoch, telemetry (`gpu.csv`), the six visible/capability
benchmarks, selection, `--status`, `discover_finalized_seed_evidence`, and the
idempotent resource artifact. A second identical invocation was a no-op: no
epoch retrained, no example rescored, no new attempt-ledger row, same null
selection. Every smoke limit was removed for the evidence run below.

## Seed-42 evidence run

_Command:_ `runs/seed42-run.sh` (frozen manifest, no reductions), a single
uninterrupted session, resumable by re-invocation.

| Measure | Value | Limit | Result |
| --- | ---: | ---: | --- |
| Training outcome | `success`, `fallback_applied: false`, sequence_length 2048 | success / preserved failure | PASS |
| Epoch-1 capability gate | gsm8k −0.4550, ifeval −0.2300, mmlu +0.0767; retention 0.7141 | 0.02 / 0.02 / 0.03 decline; 0.98 retention | FAIL (ineligible) |
| Epoch-2 capability gate | gsm8k −0.2250, ifeval −0.1300, mmlu +0.0467; retention 0.8549 | " | FAIL (ineligible) |
| Epoch-3 capability gate | gsm8k −0.2350, ifeval −0.1300, mmlu +0.0333; retention 0.8426 | " | FAIL (ineligible) |
| Visible composite (epoch 1 / 2 / 3) | 0.2733 / 0.2294 / 0.1794 (OPI +0.49–0.57 driven) | descriptive | improves, but capability-gated out |
| Finalized selection | `selected_checkpoint_digest: null`, `selected_epoch: null`, `finalized: true` | eligible checkpoint or null | `NO_ELIGIBLE_CHECKPOINT` |
| Selection-record digest | `a2cdd7c2e1c1b8989f9b7c254cae56f2b812605436ba04861a1d667a79b2cdce` | — | `verify_selection_record` PASS |
| Seed-42 active GPU-hours | 12.7427 h (wall 45,873.82 s) | 24 h per-seed wall / 72 h cumulative | PASS |
| Cumulative GPU-hours | 21.4401 + 12.7427 = **34.1828 h** | 72 h | PASS |
| Remaining budget | 72 − 34.1828 = **37.8172 h** | — | available |
| Peak VRAM | 15.6025 GiB | 15.5 GiB declared allocation (16 GiB card) | **feasibility finding** — exceeds declared allocation by 0.1025 GiB (recurs from seed 17's 15.663 GiB; not a failure) |
| Finalized bundle disk | 10.5746 GiB | 250 GB | PASS |
| Attempts (ledger) | 4 completed (1 training + 3 evaluations), 0 interrupted | — | recorded in cumulative totals |
| Declared unavailable intervals | 0 s | — | none this run |

_Attempt ledger (`recovery/attempts.jsonl`):_ training 28,590.1 s
(2026-08-31T20:12:48Z → 2026-09-01T04:09:18Z); eval epoch 1 4,977.7 s; eval
epoch 2 5,606.8 s; eval epoch 3 6,699.2 s (→ 2026-09-01T08:58:43Z). GPU-hours
per attempt are recorded `unavailable` at the stage level (single-GPU run:
seed GPU-hours == summed attempt wall-hours), matching the seed-17 /
issue-16 accounting.

_Finalized bundles (checksum-verified — `runner.bundle.verify_bundle`,
`runner.selection.verify_selection_record`, `discover_finalized_seed_evidence`):_

- Training: `runs/training-seed42-20260831-201248-1b487000`
- Evaluations: `runs/eval-seed42-epoch{1,2,3}-20260831-201248-1b487000`
  (each: open_prompt_injection 300, tensor_trust_hijack 300,
  tensor_trust_extract 300, mmlu 300, gsm8k 200, ifeval 200 — manifest-exact
  fixed example IDs)
- Selection: `runs/selection-seed42/selection_record.json`
- Resource comparison: `runs/seed42-resource-comparison/`
- Held-out reveal: none (null selection — see below)

## Held-out non-access

The finalized visible/capability selection has no eligible, non-null checkpoint:
epochs 1–3 each fail the frozen capability gates (gsm8k and ifeval declines far
exceed the 0.02 / 0.03 caps; mean normalized retention 0.71–0.85 is below the
0.98 floor). Per the frozen `held_out_unavailable_until` rule and the seed-17
precedent, the Issue #20 held-out transaction is **not** invoked for seed 42.
No sealed InjecAgent candidate, receipt, trained held-out evaluation, aggregate,
or reveal is constructed, read, or released. `_orchestrate_seed` would raise and
halt rather than reveal the baseline alone or promote a capability-failing
checkpoint. This is a terminal interpretation of the finalized null selection,
not permission to alter it.

Seed 42 is a `NO_ELIGIBLE_CHECKPOINT` result — a complete, publishable negative
outcome, consistent with seed 17.

## Closure-gate evidence

| #22 criterion | Status | Evidence |
| --- | --- | --- |
| Don't start without #21 pass + budget | PASS | #21 decision record; 50.5599 h remaining at start |
| Smoke the exact command path (storage/caches/telemetry/status/resume); drop limits for evidence | PASS | smoke section; evidence run used no `--smoke-*` flag |
| Train/reuse 3 epochs per #19, no frozen change | PASS | `runs/training-seed42-…/metrics.json` (`success`, no fallback); `verify_bundle` |
| 3 visible/capability evaluations per #18, checksums + fixed IDs | PASS | 3 eval bundles verify clean; 300/300/300/300/200/200 fixed IDs per epoch |
| Finalize one visible/capability-only selection; conditional #20 reveal | PASS | `runs/selection-seed42/selection_record.json` finalized null; #20 correctly not invoked |
| Record completed / interrupted / unavailable intervals cumulatively | PASS | `runs/seed42-resource-comparison/seed_resource_comparison.json` (`resource_intervals`: 4 attempts, 0 interrupted, 0 unavailable, 12.7427 h active) |
| Final discovery topology excludes recovery state | PASS | `discover_finalized_seed_evidence`: 1 training + 3 evals + 1 selection + resources + clean checksums; `recovery_state_excluded: true`; no reveal (null) |
| No unresolved protocol deviation / duplicate score / missing bundle / checksum failure | PASS | validation section; only open finding is the recurring peak-VRAM feasibility note |

## Validation

- `python -m unittest discover -s tests` under the real-GPU venv
  (`/mnt/c/Projects/automated_alignment_researcher/.venv`, torch 2.8.0+cu128):
  **218 pass, 0 error** (`tests/test_real_seed_run_recovery.py` adds 9 split-run
  recovery tests to the prior 209). Under the no-torch Windows venv the same 218
  run with exactly one error — the known unrelated `test_real_training` `torch`
  import — and it is the sole error.
- `discover_finalized_seed_evidence`, `verify_bundle` (×4), and
  `verify_selection_record` run against the finalized seed-42 evidence: all
  pass; selection-record digest
  `a2cdd7c2e1c1b8989f9b7c254cae56f2b812605436ba04861a1d667a79b2cdce`.
- `seed_run_status` for seed 42: `next_action: complete`, every stage
  `completed`, selection finalized, reveal absent.
- Recovery workspace (`recovery/`) and attempt ledger are preserved outside the
  evidence root and excluded from finalized discovery.

## Open finding (non-blocking)

Peak VRAM 15.6025 GiB exceeds the manifest's 15.5 GiB declared allocation by
0.1025 GiB (within the physical 16 GiB card), recorded as a feasibility finding
in `runs/seed42-resource-comparison/`. This recurs from seed 17 (15.663 GiB,
issue #16) and the baseline evaluations; it is a standing measurement note, not
a protocol violation or a stop condition, and no frozen value is changed in
response.
