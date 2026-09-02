# Issue #21 null-selection and continuation decision

**Issue:** [#21](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/21)
**Decision date:** 2026-08-31
**Protocol:** `phase1-2026-08-29` (`protocol/manifest.json`)
**Scope:** closes the seed-17 decision path after its finalized null selection.
It does not change the frozen protocol, modify finalized evidence, access
held-out material, or execute seed 42, seed 2026, or any later issue.

## Decision

Seed 17 is a `NO_ELIGIBLE_CHECKPOINT` terminal outcome.  Its finalized,
checksum-verified selection record has `selected_checkpoint_digest: null` and
`selected_epoch: null`; epochs 1, 2, and 3 each failed the frozen capability
gates.  A capability-failing epoch must not be substituted, and a baseline-only
result must not be revealed.

Consequently, the Issue #20 held-out transaction is intentionally **not
invoked** for seed 17.  No held-out candidate, receipt, evaluation, aggregate,
or reveal is read, produced, or released by this issue.  Held-out sealing
remains unchanged.

The frozen continuation rule nevertheless passes.  It depends on seed 17's
technical outcome and measured resource use, not model quality: the seed
completed successfully in 15.1848 measured GPU-hours, and the rule projects
45.5544 GPU-hours for all three frozen seeds, below the 72-hour cap.  Issue #22
is therefore authorized to **consider** seed 42 under its own scope and gates;
this issue does not start it.

## Selection identity and integrity

| Check | Result | Evidence |
| --- | --- | --- |
| Finalized selection identity | PASS — protocol `phase1-2026-08-29`, canonical-JSON content digest `399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20`, seed-17 candidates for epochs 1–3, `finalized: true`, selected checkpoint and epoch both `null` | `runs/selection-seed17-20260830-071553/selection_record.json` |
| Selection-record checksum | PASS — SHA-256 `46dfe6eab879994e8b248a7a5f5c80d35681a7e610aa2ae70c501b1f72e2e5f8` | `runner.selection.verify_selection_record` against the finalized record; prior durable index: `analysis/seed17-outcomes-summary.md` |
| Training bundle checksum topology | PASS — every required bundle file matches `checksums.sha256` | `runs/training-seed17-20260830-071553/`; `runner.bundle.verify_bundle` |
| Epoch-1 evaluation checksum topology | PASS — every required bundle file matches `checksums.sha256` | `runs/eval-seed17-epoch1-20260830-071553/`; `runner.bundle.verify_bundle` |
| Epoch-2 evaluation checksum topology | PASS — every required bundle file matches `checksums.sha256` | `runs/eval-seed17-epoch2-20260830-071553/`; `runner.bundle.verify_bundle` |
| Epoch-3 evaluation checksum topology | PASS — every required bundle file matches `checksums.sha256` | `runs/eval-seed17-epoch3-20260830-071553/`; `runner.bundle.verify_bundle` |

The selection record marks every candidate `eligible: false`: epoch 1 records
GSM8K decline 0.4750, IFEval decline 0.2050, and normalized retention 0.7107;
epoch 2 records 0.2850, 0.2200, and 0.7750; epoch 3 records 0.1800, 0.1800,
and 0.8365.  Each misses the frozen 0.02 / 0.03 / 0.98 gates.  The null
selection is thus the only valid final selection identity.

## Held-out non-access

The Issue #20 transaction requires a finalized selected checkpoint as part of
its transaction identity.  Seed 17 supplies `null`, so this issue does not
construct, authorize, or call that transaction.  In particular, it does not
read sealed InjecAgent candidates or receipts; run trained held-out evaluation;
read a held-out result; invoke a reveal; or publish any held-out aggregate.
There is no final reveal bundle for seed 17, by design.

This is a terminal interpretation of the finalized selection, not permission to
alter selection, select a failed candidate, or reveal the baseline alone.

## Cumulative resource and continuation evidence

| Measure | Actual recorded value | Frozen limit | Result |
| --- | ---: | ---: | --- |
| Baseline GPU-hours | 6.2553 h | included in 72 h cap | consumed |
| Seed-17 GPU-hours | 15.1848 h | 24 h per-seed wall cap / 72 h cumulative GPU cap | technical success; within applicable cap |
| Cumulative GPU-hours | 21.4401 h | 72 h | PASS |
| Remaining budget | 50.5599 h | 72 h less actual cumulative use | available before later measured attempts |
| Continuation-rule projection | 45.5544 h | 72 h | PASS |

`runs/seed17-resource-comparison-20260830-071553/seed_resource_comparison.json`
records the precise values: baseline 6.255256043516112 h, seed 17
15.184795426530556 h, and cumulative 21.44005147004667 h.  Running
`runner.continuation.decide_continuation` with the actual successful seed-17
outcome returns `continue_replication=True`, an empty reason list, projected
45.554386279592 GPU-hours, and budget 72 GPU-hours.  Capability results are not
inputs to this decision.

## Closure-gate evidence

| #21 criterion | Status | Evidence / terminal interpretation |
| --- | --- | --- |
| Verify finalized selection identity and checksums | PASS | Selection identity and all required visible-evidence checksum checks above. |
| Evaluate selected checkpoint and reveal once through #20 | TERMINAL — not applicable | No eligible selected checkpoint exists; invoking #20 with `null` is prohibited. |
| Publish only frozen held-out aggregates and invalid counts | TERMINAL — no output | No transaction exists, so no aggregate, count, raw material, or post-reveal tuning is produced. |
| Final reveal topology and checksums | TERMINAL — no reveal bundle | The null selection intentionally produces no reveal artifact. |
| Apply continuation rule using technical/resource evidence | PASS | Successful seed 17 and actual 15.1848 GPU-hours yield a passing rule decision. |
| Record decision, references, and remaining budget | PASS | This record, integrity table, and actual 50.5599-hour remaining budget. |
| Preserve evidence / stop later seeds only if continuation fails | PASS | Evidence is unchanged; continuation passes and authorizes #22 to consider seed 42 only. |

## Validation

- `runner.selection.verify_selection_record` passed for the finalized selection
  record and its expected SHA-256 digest.
- `runner.bundle.verify_bundle` passed for the seed-17 training bundle and all
  three visible/capability evaluation bundles.
- `runner.continuation.decide_continuation` passed with the actual seed-17
  technical outcome and measured GPU-hours.
- The repository test suite is run separately for this documentation-only
  decision.  The known unrelated missing-`torch` import error in
  `test_real_training` remains non-blocking if it is the sole error.
