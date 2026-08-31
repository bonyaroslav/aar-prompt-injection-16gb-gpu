# Issue 12 analysis and next-work summary

**Status checked:** 2026-08-31  
**Purpose:** durable handoff for further analysis after the real seed-17 run.

## Bottom line

Issue [#12](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/12)
completed its execution contract: seed 17 trained for all three frozen epochs
without OOM, every checkpoint was evaluated, and an immutable selection record
was finalized. The scientific result is negative, not incomplete. Visible
prompt-injection scores improved, but every checkpoint failed the frozen
capability gates, so `selected_checkpoint_digest` is `null`.

The run demonstrates that this experiment can execute on the physical 16 GiB
RTX 4080. It does not demonstrate reliable compliance with the manifest's
15.5 GiB allocation: measured device-level peak use was 15.663 GiB.

## Result summary

| Measure | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | ---: | ---: | ---: |
| Visible composite improvement | +0.1739 | +0.2050 | +0.2061 |
| Open Prompt Injection change | +0.3267 | +0.4467 | +0.4967 |
| Tensor Trust hijack change | +0.0517 | +0.0617 | +0.0500 |
| Tensor Trust extract change | +0.1433 | +0.1067 | +0.0717 |
| GSM8K decline | 0.4750 | 0.2850 | 0.1800 |
| IFEval decline | 0.2050 | 0.2200 | 0.1800 |
| Mean normalized capability retention | 0.7107 | 0.7750 | 0.8365 |
| Capability gate | Fail | Fail | Fail |

MMLU stayed near or above baseline. The contrast between stable choice-logit
performance and severe degradation on generated-answer benchmarks suggests a
testable hypothesis: the intervention may have damaged response behavior,
format compliance, or refusal calibration more than factual knowledge. The
current finalized bundles do not preserve enough visible raw generations to
distinguish over-refusal, parser/format failure, truncation, and incorrect
reasoning. This is a limitation, not evidence for any one mechanism.

The route-A training data makes over-refusal or template-surface learning
plausible: 40% of examples are prompt-injection cases with a relatively uniform
refusal response, while the strongest gain is concentrated in Open Prompt
Injection. No change to data, learning rate, epochs, or selection rules should
be made inside frozen Attempt 1. Such changes belong to a new protocol or
Attempt 2.

## Resource and recovery measurements

| Unit | Measured result | Decision implication |
| --- | ---: | --- |
| Baseline GPU time | 6.26 h | Already consumed by the study. |
| Seed-17 training, all epochs | about 9.36 h | Too long for one 6–7 h session. |
| Individual training epochs | about 2.8–3.3 h | Completed-epoch reuse is an adequate safe boundary. |
| Epoch-1 evaluation | about 1.85 h | Whole-checkpoint evaluation is a practical unit. |
| Epoch-2 evaluation | about 1.94 h | Whole-checkpoint evaluation is a practical unit. |
| Epoch-3 evaluation | about 2.01 h | Whole-checkpoint evaluation is a practical unit. |
| Seed-17 total | 15.18 h | Within the 24 h per-seed cap. |
| Cumulative through seed 17 | 21.44 GPU-h | Within the 72 GPU-h total cap. |
| Peak device VRAM | 15.663 GiB | Exceeds the declared 15.5 GiB allocation by 0.163 GiB. |
| Finalized seed bundle size | about 10.57 GiB | Does not include every operational cache/work artifact. |

The two remaining seeds at the same measured cost would produce a simple
projection of `6.26 + 3 × 15.18 = 51.81` GPU-hours, leaving about 20.19 hours
for held-out evaluation, interrupted attempts, and recovery overhead. This is
only a provisional budget result. The final continuation decision must use the
attempt ledger and actual cumulative cost.

Training telemetry contained intermittent `nvidia-smi` query failures and gaps,
so 15.663 GiB is the highest observed device value, not a guaranteed exact
application maximum. Future runs should retain device telemetry and also record
PyTorch peak allocated/reserved memory to distinguish application demand from
other Windows/WSL GPU clients.

## WSL and GPU-memory consequence

WSL's `memory=` setting controls the Linux VM's maximum system RAM. It does not
increase dedicated GPU VRAM or Windows' dynamic shared-GPU budget. Raising it
can help Linux OOM or heavy swapping, but it cannot eliminate this CUDA/VRAM
margin. See [WSL RAM and GPU-memory limits](wsl-memory-gpu-limits.md).

Within the frozen protocol, the proportionate mitigations are operational:

- keep checkpoint and temporary work on the disk-backed `runs/` mount rather
  than WSL's small RAM-backed `/tmp`;
- close unrelated GPU workloads and record free VRAM immediately before launch;
- retain the one approved 2048-to-1536 full-restart fallback for an actual OOM;
- preserve every OOM, retry, and telemetry gap as feasibility evidence.

Guaranteeing a peak below 15.5 GiB requires a measured memory-reducing protocol
change or a larger GPU. It cannot be achieved by reinterpreting shared system
memory as additional VRAM.

## Blocking specification mismatch

The most important problem in the upcoming work is logical rather than
computational:

- issue #12 finalized no eligible checkpoint;
- [#21](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/21)
  requires evaluation of "the selected seed-17 checkpoint";
- the current reveal path expects both baseline and trained sealed receipts.

Those requirements cannot be satisfied with a null checkpoint. Before #20/#21
implementation, define a durable `NO_ELIGIBLE_CHECKPOINT` terminal state. For
seed 17 it should preserve the null selection, perform no trained held-out
evaluation, leave the held-out baseline sealed, and proceed to the continuation
decision using only technical success and cumulative resources.

Revealing the baseline alone would spend held-out secrecy without enabling a
comparison. Selecting epoch 3 after seeing the results would violate the frozen
capability gate and turn the held-out analysis into a post-hoc protocol.

The existing continuation helper also projects `seed_cost × seed_count` and
does not include the baseline or interrupted attempts. The recovery-aware
decision must instead consume cumulative ledger cost plus projected remaining
work.

## Recommended issue order

1. [#16](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/16):
   record the measurements above and choose completed epoch and whole-checkpoint
   evaluation as the default recovery boundaries.
2. Amend [#20](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/20)
   and #21 to support `NO_ELIGIBLE_CHECKPOINT` before implementing their state
   machines.
3. [#17](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/17):
   implement signatures, atomic state, an append-only attempt ledger, and strict
   separation of partial state from finalized evidence.
4. [#19](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/19):
   implement completed-epoch reuse. Keep mid-epoch optimizer recovery
   conditional; current epoch duration does not justify its complexity.
5. Narrow [#18](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/18)
   to the coarsest measured safe boundary. Per-item recovery is unnecessary
   unless #16 finds a long individual benchmark or real interruption evidence.
6. [#21](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/21):
   finalize the null-selection handling and cumulative continuation decision.
7. If authorized, execute seeds 42 and 2026 serially through
   [#22](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/22)
   and [#23](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/23).
8. Run [#14](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/14)
   last, after the final one-or-three-seed artifact set is closed.

## Work that does not make sense now

- rerunning seed 17 unchanged;
- raising WSL system RAM as a supposed VRAM fix;
- forcing an artificial OOM to exercise the fallback;
- choosing a capability-failing checkpoint post hoc;
- revealing the held-out baseline without a trained comparison;
- implementing fine-grained mid-epoch or per-item recovery before #16 justifies it;
- tuning the method inside frozen Attempt 1;
- starting final publication analysis before the null-selection path and final
  evidence topology are resolved.

## Evidence references

- [Committed seed-17 outcome summary](../analysis/seed17-outcomes-summary.md)
- [Frozen protocol manifest](../protocol/manifest.json)
- [Research plan](../RESEARCH_PLAN.md)
- [Detailed GitHub issue follow-up](research/issue-12-follow-up.md)
- [WSL RAM and GPU-memory limits](wsl-memory-gpu-limits.md)
