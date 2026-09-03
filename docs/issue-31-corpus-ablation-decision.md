# Issue #31 — Clean-corpus ablation decision

**Issue:** [#31](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/31)
**Decision date:** 2026-09-03
**Downstream of:** frozen protocol `phase1-2026-08-29`
(`protocol/manifest.json`, canonical SHA-256
`399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20`)
**Ablation protocol:** `protocol/ablation/corpus-ablation-2026-09-02.json`
(canonical SHA-256
`c6e36b48d5de4ec151b6a0a23bfe493474cedeb2fd9db9e42094804e4005b7b3`,
raw SHA-256
`3a1f4c2a36b2ad09cb548fdb89f8f3ab96919bd781cade900b2de7e620a022d2`)

## Question and boundary

Attempt 1's 5,000-row training corpus contained 2,000 prompt-injection rows.
This authorized ablation asks whether removing that category changes the
capability-collapse pattern, while keeping the target model, upstream revision,
seed (42), optimizer, response-only objective, three epochs, decoding, visible
suite, and scorers fixed.

The ablation corpus is exactly 5,000 rows: **0 prompt-injection**, 3,500
clean-control, 1,000 ambiguous-boundary, and 500 refusal-calibration. Its
report records no shortfall, its dataset SHA-256 is
`bdc76c0242b7669abae306154459f1ed79efe58d5376a781e2ab4c595278ab4b`, and
its Dolly-source canonical SHA-256 is
`b7ec34819df7aef50e9d5696952cb9535f90270ddb9c9f1f060646c109b55a01`.

The source-dataset cache was the already verified/re-cached offline source
specified in the handover; it was not revisited or substituted. The separate
construction path uses Dolly oversample factor 2 and a 1,536-token construction
filter so every selected row retains assistant tokens under the real tokenizer.
Those settings are isolated to the ablation corpus build; the frozen trainer's
2,048-token sequence length, optimizer, and evaluation settings are unchanged.

This is an ablation-only result. It writes beneath gitignored `ablation/`, is
not discovered as Attempt-1 evidence, never enters selection or the frozen
bootstrap, and has no held-out-root argument. `VisibleOnlyDataset` refuses
`injecagent`; all three evaluations use only the six published visible
benchmarks.

## Recovery and execution evidence

The detached WSL run completed all three epochs and all three visible-only
evaluations in `ablation/issue-31-corpus-ablation-20260902/`. Its external,
private recovery workspace records the epoch transactions and two-slot
mid-epoch state.

The required deliberate-interruption acceptance test was performed after epoch
1's durable step-120 checkpoint: the process was terminated, then the same
command was rerun. The final attempt ledger records
`mid_epoch_resume_fired: true` for epoch 1. All recovery saves stayed far below
the 30-second escalation threshold: the ledger's recorded saves are 0.70--0.79
seconds for approximately 131.2 MB mutable-state snapshots. Epochs 2 and 3
also completed with durable step-120 and step-240 saves.

No VRAM overage or RAM exhaustion occurred. The run's separate resource ledger
records 11.8845 wall/GPU-hours. Added to Attempt 1's frozen scientific total of
47.34 hours, all incurred compute is 59.22 hours, below the 72-hour cap; the
frozen Attempt-1 scientific total is not altered.

## Result

The table compares the real untrained baseline, the clean-corpus ablation, and
the corresponding Attempt-1 seed-42 checkpoint. Values are the unchanged
per-benchmark aggregate metrics; this single-seed ablation is descriptive, not
a new selection or replication analysis.

| State | OPI | TT extract | TT hijack | MMLU | GSM8K | IFEval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.180 | 0.598 | 0.492 | 0.567 | 0.735 | 0.615 |
| Clean ablation, epoch 1 | 0.283 | 0.795 | 0.517 | 0.613 | 0.345 | 0.525 |
| Clean ablation, epoch 2 | 0.367 | 0.733 | 0.557 | 0.617 | 0.370 | 0.475 |
| Clean ablation, epoch 3 | 0.400 | 0.715 | 0.532 | 0.613 | 0.490 | 0.430 |
| Attempt 1 seed 42, epoch 3 | 0.667 | 0.565 | 0.577 | 0.600 | 0.500 | 0.485 |

Removing prompt-injection examples makes the OPI improvement much smaller
(epoch 3: +0.220 over baseline rather than Attempt 1 seed 42's +0.487), while
Tensor Trust extract remains elevated. Crucially, the capability failure does
**not** disappear: at epoch 3 GSM8K is -0.245 and IFEval is -0.185 versus the
same baseline, despite MMLU improving by +0.046. The clean-only corpus therefore
does not support the simple claim that the prompt-injection rows alone caused
the capability collapse. It is compatible with a broader effect of this
response-only SFT setup and corpus composition; isolating which remaining
component causes it needs a separately authorized follow-up.

The ablation is deliberately not compared as a formal causal estimate: it has
one seed, a purpose-built corpus construction filter, and no pre-authorized
multi-seed/bootstrap analysis. It does establish the narrower, useful negative
result: deleting the explicit prompt-injection category did not restore the
frozen capability gates.

## Reproducibility pointers

- Corpus and resource records: `ablation/issue-31-corpus-ablation-20260902/`
  (gitignored execution evidence).
- Recovery evidence: private WSL recovery workspace, with the final public
  attempt ledger linked by `resource.json`.
- Harness: `runner/corpus_ablation.py`; protocol and fail-closed loader:
  `protocol/ablation/`; mid-epoch primitives: `runner/ablation_training.py`.
- Offline test coverage includes corpus completeness/no-injection validation,
  protocol-digest drift, durable checkpoint callbacks, adapter registration,
  and path-safe resource finalization.
