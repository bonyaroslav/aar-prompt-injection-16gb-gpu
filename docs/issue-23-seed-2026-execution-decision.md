# Issue #23 seed-2026 execution, continuation, and finalization decision

**Issue:** [#23](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/23)
**Decision date:** 2026-09-02
**Protocol:** `phase1-2026-08-29` (`protocol/manifest.json`, canonical-JSON content digest
`399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20`)
**Scope:** executes and finalizes the third and last frozen replication seed
(`2026`) through the identical recovery-aware split-run workflow used for seed
42 (issue #22). No frozen protocol, dataset, hyperparameter, seed, decoding,
selection-rule, capability-gate, held-out, or resource-limit value is changed.
Held-out InjecAgent is not read: the finalized selection is null.

## Continuation authority

The frozen continuation rule (`runner.continuation.decide_continuation`) is
driven only by the seed-17 pilot's technical `outcome` and measured
`gpu_hours` — never by model quality (`tests/test_continuation.py` asserts
this against the function source). Issue #21 already recorded a passing
decision covering **both** later seeds: seed 17 completed successfully in
15.1848 GPU-hours, and the rule projects 45.5544 GPU-hours across all three
manifest seeds `[17, 42, 2026]`, under the 72-hour cap. Re-running the rule
today with the same seed-17 pilot result reproduces
`continue_replication=True`, empty `reasons`, `projected_gpu_hours ≈ 45.5544`,
`budget_gpu_hours = 72`.

Cumulative budget after this seed remains within the frozen limit:

| Component | GPU-hours |
| --- | ---: |
| Baseline | 6.2553 |
| Seed 17 | 15.1848 |
| Seed 42 | 12.7427 |
| Seed 2026 (this run) | 13.1553 |
| **Cumulative** | **47.3381** |
| Frozen cap | 72 |
| **Remaining** | **24.6619** |

Seed 2026 is therefore authorized and finalized under its own scope and gates.
Its `NO_ELIGIBLE_CHECKPOINT` outcome, like seed 17's and seed 42's, does not
feed back into the continuation rule and does not authorize any protocol change.

## Execution path

`runs/training-seed2026-20260901-112915-bf0809d1/command.sh` records the exact
evidence-run command — `python -m runner.real_seed_run … --seed 2026
--prior-cumulative-gpu-hours 34.1827786627532 --recovery-root recovery`, the
frozen manifest with **no `--smoke-*` reduction**. It is the identical
orchestration path finalized for seed 42: every stage routed through its
recovery contract on a stable stage key (`training-seed2026`,
`eval-seed2026-epoch{1,2,3}`), the per-attempt run-bundle id timestamped and
unique, recovery state confined to `recovery/` outside the `runs/` evidence
root.

`runs/seed2026-smoke/` (stamp `-101746`, `--smoke-max-steps 1
--smoke-max-items-per-benchmark 2`) and the pre-run adapter directories
`runs/adapters-seed2026-20260901-103119` and `-103738` are workflow scaffolding
only. They are gitignored, carry no ledger attempt, and must never enter effect
estimates, bootstrap intervals, or cross-seed summaries. The append-only
attempt ledger holds exactly four completed seed-2026 attempts, zero
interrupted, zero declared-unavailable intervals.

## Seed-2026 evidence run

_Command:_ `runs/training-seed2026-20260901-112915-bf0809d1/command.sh`
(frozen manifest, no reductions), resumable by re-invocation of the identical
command.

| Measure | Value | Limit | Result |
| --- | ---: | ---: | --- |
| Training outcome | `success`, `fallback_applied: false`, sequence_length 2048 | success / preserved failure | PASS |
| Epoch-1 capability gate | gsm8k −0.4450, ifeval −0.1800, mmlu +0.0667; retention 0.7398 | 0.02 / 0.03 decline; 0.98 retention | FAIL (ineligible) |
| Epoch-2 capability gate | gsm8k −0.2900, ifeval −0.2000, mmlu +0.0433; retention 0.7856 | " | FAIL (ineligible) |
| Epoch-3 capability gate | gsm8k −0.2000, ifeval −0.1450, mmlu +0.0267; retention 0.8464 | " | FAIL (ineligible) |
| Visible composite (epoch 1 / 2 / 3) | +0.2056 / +0.1994 / +0.1939 (OPI +0.31 → +0.43 driven) | descriptive | improves, but capability-gated out |
| Finalized selection | `selected_checkpoint_digest: null`, `selected_epoch: null`, `finalized: true` | eligible checkpoint or null | `NO_ELIGIBLE_CHECKPOINT` |
| Selection-record digest | `8df462a4548fe652660409ef76b2b987a7794a0904f9b400cf8bdf1ba10a0d23` | — | `verify_selection_record` PASS |
| Manifest digest in record | `399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20` | canonical content digest | matches `protocol/digests.md` |
| Seed-2026 active wall / GPU-accounted time | 13.1553 h (47,359.22 s) | 24 h per-seed wall / 72 h cumulative | PASS |
| Cumulative GPU-hours | 34.1828 + 13.1553 = **47.3381 h** | 72 h | PASS |
| Remaining budget | 72 − 47.3381 = **24.6619 h** | — | available |
| Peak VRAM | 15.6289 GiB | 15.5 GiB declared allocation (16 GiB card) | **feasibility finding** — exceeds declared allocation by 0.1289 GiB; recurs from seeds 17 (15.663) and 42 (15.6025); not a failure |
| Finalized bundle disk | 10.5743 GiB | 250 GB | PASS |
| Attempts (ledger) | 4 completed (1 training + 3 evaluations), 0 interrupted, 0 unavailable | — | recorded in cumulative totals |

_Attempt ledger (`recovery/attempts.jsonl`):_ training 28,835.97 s
(2026-09-01T11:29:15Z → 19:29:51Z); eval epoch 1 5,199.62 s; eval epoch 2
6,397.12 s; eval epoch 3 6,926.51 s (→ 2026-09-02T00:39:56Z). Per-attempt
GPU-hours are recorded `unavailable` at stage level (single-GPU run: seed
GPU-hours == summed attempt wall-hours), matching seed 17 / seed 42 / issue #16
accounting. There are no interruption or power-loss gaps to impute.

_Finalized bundles (checksum-verified — `runner.bundle.verify_bundle`,
`runner.selection.verify_selection_record`,
`runner.real_seed_run.discover_finalized_seed_evidence`):_

- Training: `runs/training-seed2026-20260901-112915-bf0809d1`
- Evaluations: `runs/eval-seed2026-epoch{1,2,3}-20260901-112915-bf0809d1`
  (each: open_prompt_injection 300, tensor_trust_hijack 300,
  tensor_trust_extract 300, mmlu 300, gsm8k 200, ifeval 200 — manifest-exact
  fixed example IDs)
- Selection: `runs/selection-seed2026/selection_record.json`
- Resource comparison: `runs/seed2026-resource-comparison/`
- Held-out reveal: none (null selection — see below)
- Durable analysis record: `analysis/seed2026-outcomes-summary.md`

## Held-out non-access

The finalized visible/capability selection has no eligible, non-null
checkpoint: epochs 1–3 each fail the frozen capability gates (gsm8k and ifeval
declines far exceed the 0.02 / 0.03 caps; mean normalized retention 0.74–0.85
is below the 0.98 floor). Per the frozen `held_out_unavailable_until` rule and
the seed-17 / seed-42 precedent, the issue #20 held-out transaction is **not**
invoked for seed 2026. No sealed InjecAgent candidate, receipt, trained
held-out evaluation, aggregate, or reveal is constructed, read, or released.
`discover_finalized_seed_evidence` confirms `reveal_bundle: null` and would
raise if a null selection had produced one. This is a terminal interpretation
of the finalized null selection, not permission to alter it.

Seed 2026 is a `NO_ELIGIBLE_CHECKPOINT` result — a complete, publishable
negative outcome, consistent with seeds 17 and 42.

## Completed-seed set

The frozen replication set `[17, 42, 2026]` is now complete: three seeds, nine
trained checkpoints, nine capability-gate failures, three finalized null
selections, held-out InjecAgent sealed throughout. `discover_finalized_seed_evidence`
returns a clean topology for seeds 42 and 2026 (recovery-aware seam); seed 17
(pre-seam one-shot run) verifies through direct `verify_bundle` /
`verify_selection_record` on its four bundles and selection record. Every
recovery-workspace path is rejected as a finalized input by
`runner.recovery.finalized_inputs_only`.

The analysis chain (issues #27–#33) may now freeze a three-seed input set. It
must also remain correct at two seeds; this decision does not stop any seed, so
the analysis unit is nine checkpoints (`3 seeds × 3 epochs`), gate contrast
nine-of-nine versus zero-of-nine, with the seed and checkpoint counts derived
rather than hardcoded.

## Closure-gate evidence

| #23 criterion | Status | Evidence |
| --- | --- | --- |
| Run only after #21 authorizes continuation and #22 released the GPU / recorded the updated budget | PASS | #21 decision (covers seeds 42 and 2026); #22 recorded 34.1828 h cumulative / 37.8172 h remaining; this run's `--prior-cumulative-gpu-hours 34.1827786627532` |
| Same exact-command smoke validation; remove smoke limits for the evidence run | PASS | `runs/seed2026-smoke/` (`--smoke-max-steps 1`); evidence-run `command.sh` carries no `--smoke-*` flag |
| Identical frozen training, evaluation, selection, held-out, recovery, fallback contracts as seed 42 | PASS | `_orchestrate_seed` recovery-aware path; no frozen value changed; `fallback_applied: false` |
| Produce and checksum training + three evaluations + selection + reveal + resource topology | PASS | `discover_finalized_seed_evidence(seed=2026)` clean; `verify_bundle` ×4 PASS; selection digest `8df462a4…` PASS; reveal correctly absent (null) |
| Record every recovery event or integrity failure; never change protocol in response to scores | PASS | ledger: 4 completed, 0 interrupted, 0 unavailable; only standing finding is the recurring peak-VRAM feasibility note |
| Re-run finalized-input discovery over seeds 17, 42, 2026; prove recovery files excluded | PASS | seeds 42 & 2026 via `discover_finalized_seed_evidence` (`recovery_state_excluded: true`); seed 17 via direct bundle verification; `finalized_inputs_only` rejects a recovery path |
| Cumulative resource total traceable and within the frozen limit, or explicit feasibility finding | PASS | 47.3381 / 72 GPU-hours; peak VRAM 15.6289 GiB recorded as a feasibility finding in `runs/seed2026-resource-comparison/` |
| Completed-seed set consumable by final analysis without a recovery-specific code path | PASS | discovery excludes `recovery/`; `analysis/seed2026-outcomes-summary.md` indexes the immutable result |

## Validation

- `python -m unittest discover -s tests -q` under the no-torch Windows venv
  (`.venv/Scripts/python.exe`): **234 pass, 0 error, 1 skipped**. The previously
  standing missing-`torch` error in `test_real_training` is now a skip.
- `discover_finalized_seed_evidence(seed=2026)`, `verify_bundle` (×4), and
  `verify_selection_record` run against the finalized seed-2026 evidence: all
  pass; selection-record digest
  `8df462a4548fe652660409ef76b2b987a7794a0904f9b400cf8bdf1ba10a0d23`.
- `decide_continuation` with the actual seed-17 pilot result:
  `continue_replication=True`, empty reasons, projected 45.5544 GPU-hours,
  budget 72.
- `runner.recovery.finalized_inputs_only` rejects
  `recovery/training-seed2026-seq2048-epoch1.json` presented as a finalized
  input.
- Recovery workspace (`recovery/`) and attempt ledger are preserved outside the
  evidence root and excluded from finalized discovery.

## Open finding (non-blocking)

Peak VRAM 15.6289 GiB exceeds the manifest's 15.5 GiB declared allocation by
0.1289 GiB (within the physical 16 GiB card), recorded as a feasibility finding
in `runs/seed2026-resource-comparison/`. This recurs from seed 17 (15.663 GiB)
and seed 42 (15.6025 GiB); it is a standing measurement note, not a protocol
violation or a stop condition, and no frozen value is changed in response.
