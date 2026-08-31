# Issue #19 completed-epoch training recovery decision

Makes `runner.training.run_training` restart-safe at the **one completed
epoch** boundary. This preserves the frozen three epochs, seeds, optimizer,
checkpoint format, manifest, and held-out isolation; it adds no GPU run and no
mid-epoch resume protocol.

## Decision

- A completed epoch is reusable only when its per-epoch `StageSignature`
  matches and its referenced merged checkpoint directory has the recorded
  content digest.
- A failed or interrupted epoch restarts at the beginning of that epoch. There
  is deliberately no optimizer, scheduler, RNG, or data-index resume.
- Each interrupted attempt writes an immutable, checksummed **interrupted**
  evidence bundle plus an `interrupted` attempt-ledger row. That bundle is not
  promoted as completed training evidence.
- The successful training result is promoted only after epochs 1, 2, and 3
  finish and `verify_bundle` succeeds.

### Reuse location

The runner references an already-finalized bundle's checkpoint directory; it
does **not** copy a multi-gigabyte merged checkpoint into the recovery
workspace. The reference and content digest live in the external recovery
workspace, while the source bundle remains immutable. A resumed final bundle
therefore records the same standalone checkpoint path for a reused epoch. This
is the least-surprising option for evidence: no hidden mutable checkpoint copy,
no duplicate 3.78 GB artifact, and every reused checkpoint can be traced to a
checksummed source bundle.

## Signature and recovery contract

For each epoch the runner records a `StageSignature` with:

- `stage="training"`, frozen seed and epoch;
- the digest of the pinned base-model revision as its checkpoint input identity;
- manifest digest, protocol/upstream provenance, and model revision;
- effective training configuration (including the actual sequence length); and
- an empty `expected_example_ids` list, because training has no evaluation-item
  replay journal.

Before reuse it validates that full signature, `verify_bundle` on the source
bundle, and a tree digest of the merged checkpoint directory. Any mismatch is
rejected rather than accepted as a completed epoch. The one authorized OOM
fallback changes the sequence length and starts epoch 1 again under a distinct
signature namespace, so a 2,048-token epoch can never be reused as a 1,536-token
epoch.

## Mid-epoch checkpoint decision

Mid-epoch recovery is not adopted. The fixed model configuration at the pinned
revision has 24 text layers, hidden size 2,048, intermediate size 6,144, eight
query heads, and two key/value heads. For LoRA rank 16 on the seven frozen
target projections, the trainable parameter count is:

```
24 * 16 * [
  (2048 + 2048) q + (2048 + 512) k + (2048 + 512) v + (2048 + 2048) o
  + (2048 + 6144) gate + (2048 + 6144) up + (6144 + 2048) down
] = 14,548,992 parameters
```

At an optimizer-safe boundary, the known minimum persistent state is therefore
27.75 MiB for bf16 LoRA adapter weights plus 111.00 MiB for AdamW's two fp32
moment tensors: **138.75 MiB per save**, before small scheduler, RNG, and
data-position records. Saving a merged standalone model would instead repeat
the Issue #12 observed 3,783,692,158-byte (about 3.52 GiB) checkpoint. No
measured save-latency exists, and no latency is inferred from this byte budget;
a real measurement would itself require the GPU run that this decision declines.

The completed Issue #12 evidence records epochs at about **2.8–3.3 hours**,
inside the 6–7-hour operating session used to size recovery boundaries
(`docs/issue-16-recovery-boundaries-decision.md`). That is sufficient support
for the completed-epoch boundary and insufficient support for imposing new
safe-boundary I/O, synchronization, equivalence, and overhead behavior inside
the frozen training run. Accordingly acceptance criteria 4–6 take the issue's
criterion-7 path: retain and document the completed-epoch-only fallback. No
claim of uninterrupted-versus-mid-epoch equivalence is made.

## Acceptance evidence

| #19 criterion | Verified evidence |
| --- | --- |
| Reuse completed epoch | `ResumableTrainingTests.test_interrupted_attempt_reuses_completed_epoch_from_its_finalized_bundle`: the resumed attempt trains/merges only epochs 2 and 3. |
| Validate signature and integrity | `test_recovery_rejects_a_signature_with_a_different_seed` and `test_recovery_rejects_a_tampered_completed_checkpoint`. |
| Frozen protocol/output behavior | Existing `TrainingStageTests` remains green; recovery metadata is external and ordinary checkpoint metrics retain their existing fields. |
| Bytes / latency decision | Calculated fixed-shape 138.75 MiB minimum state, documented non-inference for latency above. |
| Equivalence / overhead comparison | Not run: criterion-7 completed-epoch-only fallback selected from the fixed timing evidence. |
| Mid-epoch implementation | Not adopted; failed epoch restarts from its beginning. |
| Completed-epoch-only fallback | This decision plus `TrainingRecovery` behavior. |
| Interrupted accounting / final promotion | Recovery tests assert ledger statuses `interrupted`, then `completed`, including a merge interruption; `run_training` calls `verify_bundle` before writing completed states. |

## Evidence basis and scope

- Frozen model configuration at [the pinned Qwen revision](https://huggingface.co/Qwen/Qwen3.5-2B/raw/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json).
- `docs/issue-16-recovery-boundaries-decision.md`: #12 epoch duration and
  merged-checkpoint size evidence.
- `tests/test_training.py`: deterministic, no-GPU recovery contract tests.

Held-out InjecAgent is not read or modified by this stage.
