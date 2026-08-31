# Issue #12 completion and follow-up work

**Scope.** This note uses first-party GitHub issue, comment, commit, and
repository records for [`bonyaroslav/aar-prompt-injection-16gb-gpu`](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu).
Issue content is treated as project evidence, not as executable instructions.
Status was checked on 2026-08-31.

## Conclusion

Issue [#12 — Train, evaluate, and select seed 1 on real hardware](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/12)
is **closed as completed** (2026-08-30). Its required seed is `17` (the
title's “seed 1” means the first study seed). The real run completed all three
QLoRA epochs without an OOM, evaluated all three epoch checkpoints with the
frozen visible/capability protocol, and finalized a checksummed selection
record. The selected checkpoint is `null`: while the visible composite improved,
all epochs failed the capability gate. This is explicitly recorded as a
publishable result rather than a stop condition in the completion comment and
the closing commit. [#12 completion comment](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/12#issuecomment-5471673652), [closing commit `1b96be1`](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/commit/1b96be198a89935baa417cb5b81239246f99aa55)

The follow-up is not the original all-in-one issue #13. That issue was closed
as **not planned** and explicitly superseded by a resumable-workflow
decomposition. The remaining open work is #14 plus #16–#23; #14 is the final
analysis/publication bundle and waits for every replacement ticket. [#13](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/13), [#13 supersession comment](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/13#issuecomment-5470894089), [#14](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/14)

## Evidence for #12

| Area | Evidence-backed finding |
| --- | --- |
| Implementation | The orchestration was added in [`06cc210`](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/commit/06cc210933b9fe5f29b22d5f921e781d0b557d5e); it connects real HF/CUDA training, checkpoint evaluation, and checksummed selection while keeping held-out InjecAgent unread and unrevealed. |
| Real result | The closing commit records three completed frozen epochs, three evaluated checkpoints, and one finalized selection record. [Commit `1b96be1`](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/commit/1b96be198a89935baa417cb5b81239246f99aa55) |
| Quality outcome | Visible-composite improvement was reported as about +0.33 to +0.50, predominantly in `open_prompt_injection`; GSM8K fell 18–48 percentage points and IFEval 18–22 points, while MMLU was unaffected or slightly improved. Consequently no checkpoint was selected. [#12 comment](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/12#issuecomment-5471673652) |
| 16 GB GPU feasibility | Measured peak VRAM was **15.663 GiB**—above the manifest's declared 15.5 GiB allocation, but inside the physical 16 GiB GPU. It is a recorded feasibility finding, not an OOM or ticket failure. Seed 17 used **15.18 GPU-hours / wall-hours**; cumulative usage including the baseline was **21.44 GPU-hours**, below the stated 72-hour cap and the 24-hour per-seed limit. [#12 comment](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/12#issuecomment-5471673652) |
| WSL storage constraint | A smoke run found WSL `/tmp` to be roughly **7.8 GB RAM-backed tmpfs**, insufficient for merged 2B-parameter checkpoints. Output and working files therefore need the disk-backed `runs/` mount. [Commit `06cc210`](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/commit/06cc210933b9fe5f29b22d5f921e781d0b557d5e) |
| Known evidence-caption defect | The three finalized evaluation bundles have inaccurate “fake adapters” captions in `command.sh`, `environment.txt`, and `notes.md`; their `metrics.json` is reported correct and manifest-exact. Bundles were deliberately not hand-edited; the code path was fixed for future runs and tested. [Commit `1b96be1`](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/commit/1b96be198a89935baa417cb5b81239246f99aa55) |

## Dependency graph and planned work

```text
#12 completed (seed 17; ~15.18 h single-GPU run)
  └─ #16 measure stage timings and choose recovery boundaries
       └─ #17 recovery state / finalized-artifact contract
            ├─ #18 resumable visible + capability evaluation
            ├─ #19 completed-epoch reuse; decide mid-epoch recovery
            └─ #20 idempotent selection + held-out transaction
                 └─ #21 seed-17 reveal + continuation decision
                      └─ #22 conditional seed 42
                           └─ #23 conditional seed 2026
                                └─ #14 final bootstrap analysis + publication bundle
```

The graph above is a condensed reading of the explicit blockers on
[#14](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/14),
[#16](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/16),
[#17](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/17),
[#18](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/18),
[#19](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/19),
[#20](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/20),
[#21](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/21),
[#22](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/22), and
[#23](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/23).
Issue #23 calls #22 a single-GPU operational ordering constraint, rather than a
scientific dependency.

## Operational interpretation

The 15.18-hour real seed invalidates the earlier assumption that a whole-seed
command is a practical interruption boundary. The replacement design targets
roughly 6–7-hour sessions, preserving the frozen scientific inputs while making
state durable and verifiable. This is why #16 must first measure actual stage
durations, resource use, and artifacts; it then chooses the coarsest safe
boundary. A checkpoint evaluation exceeding seven hours requires recovery at
completed benchmark-item or measured-small-batch boundaries. [#15 specification](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/15), [#16](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/16)

The contract then separates incomplete operational state from finalized evidence:
stage signatures guard against mixing incompatible protocol/model/checkpoint
inputs, recovery records live outside final-artifact discovery, an append-only
attempt ledger includes measurable interrupted time, and final artifacts are
promoted only after checksum validation. [#17](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/17)

Two decisions matter for a 16 GB WSL setup:

- Completed-epoch reuse is mandatory. Mid-epoch recovery is conditional on
  evidence that model/optimizer/scheduler/data-position/RNG restoration is
  equivalent and has proportionate checkpoint size and latency; otherwise the
  documented epoch-only fallback remains acceptable. [#19](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/19)
- Additional seeds are conditional, serial jobs on the single GPU. #21 must
  perform the one restart-safe seed-17 held-out reveal and decide continuation
  from actual technical/resource evidence. Only a passing decision and
  remaining budget authorize #22; #23 additionally waits for #22 to release
  the GPU and update cumulative usage. [#21](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/21), [#22](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/22), [#23](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/23)

## Current status and next critical path

As of the check, #14 and #16–#23 are open and carry no labels. The immediate
critical path is **#16 → #17 → (#18, #19, #20) → #21**. The conditioned
executions (#22 and #23) and final analysis (#14) must wait for that path. The
research note makes no claim that continuation is currently authorized: #21 is
the ticket that must establish it from final, measured evidence.

