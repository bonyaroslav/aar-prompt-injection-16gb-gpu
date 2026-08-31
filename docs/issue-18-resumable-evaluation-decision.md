# Issue #18 resumable visible/capability evaluation decision

Makes `runner.evaluation.run_trained_evaluation` restart-safe on top of the #17
recovery contract, with no manifest or protocol change and no modification to any
finalized evidence bundle. Held-out InjecAgent is still never touched by this
stage.

## Boundary interpretation (no protocol change)

The #16 decision fixes the **finalized** recovery boundary at the whole-checkpoint
evaluation (no epoch evaluation exceeded seven hours, so finer boundaries are
"not required"). This implementation keeps that: the finalized, checksummed
bundle is promoted only after all three visible-safety and all three capability
benchmarks complete and `verify_bundle` passes.

The per-example progress journal (`RecoveryWorkspace.record_progress` /
`completed_progress`) is a resume optimisation *inside* that boundary, not a new
protocol recovery guarantee. It is written only after a full generate+score for
an example returns, so no active model generation is ever interrupted to meet a
write interval. A resumed run reuses journalled outcomes verbatim and walks the
benchmark values in the identical fixed dataset order an uninterrupted run uses,
so the final `metrics.json` and `config.yaml` are byte-identical. `gpu.csv`,
`run_id`, and timestamps legitimately differ across sessions; "the same ordinary
checksummed artifact" is therefore read as identical metrics / effective config /
fixed example IDs / bundle topology plus clean checksums.

## Acceptance evidence

| #18 criterion | Verified evidence |
| --- | --- |
| Persist completed example IDs, outcomes, ordering, signature, attempt telemetry | `RecoveryProgressJournalTests`; `ResumableEvaluationTests.test_each_attempt_is_recorded_in_the_attempt_ledger`; ordering + expected IDs carried in the stage signature payload (`_evaluation_signature`) |
| Resume validates seed, checkpoint identity, benchmark, effective config, expected IDs, protocol/upstream provenance, model revision before skipping | `ResumableEvaluationTests.test_resume_rejects_an_incompatible_signature_and_preserves_state` (via `StageSignature` over all of `manifest_digest`, `protocol_version`, `upstream_commit`, `upstream_tree`, `model_revision`, `seed`, `stage`, `epoch`, `checkpoint_digest`, `effective_evaluation_config`, `expected_example_ids`) |
| No active model generation interrupted for a bookkeeping interval | `ResumableEvaluationTests.test_no_model_generation_is_interrupted_to_write_progress` |
| Final aggregation includes every expected example exactly once, frozen metrics shape | `ResumableEvaluationTests.test_final_aggregation_counts_every_expected_example_exactly_once`; `test_interrupted_then_resumed_matches_uninterrupted_and_scores_no_item_twice` |
| Ordinary final bundle promoted only after all six benchmarks complete and pass checksum validation | `verify_bundle` gate in `run_trained_evaluation` after `finalize_bundle`; `test_interrupted_then_resumed...` asserts bundle topology `= BUNDLE_FILES ∪ {checksums.sha256}` and verifies clean; an interrupted attempt writes no bundle and stays `recoverable` |
| Injected interruption/resume matches uninterrupted metrics, fixed IDs, topology; no item scored twice | `ResumableEvaluationTests.test_interrupted_then_resumed_matches_uninterrupted_and_scores_no_item_twice` (first-attempt scored set disjoint from resume scored set; union equals all 1,600 expected examples) |
| Normal suite and focused recovery tests pass | `python -m unittest discover -s tests` = 202 pass, 1 pre-existing unrelated `torch` import error (`test_real_training.RuntimeOOMTranslationTests`, documented) |

## Scope boundary

Wiring an actual multi-session resume into `runner.real_seed_run` (smoke-limited
"exact command path" + resume detection) belongs to #22; training-epoch reuse to
#19; selection/held-out-reveal idempotency to #20. This issue delivers the
resumable evaluation mechanism and its contract tests only.
