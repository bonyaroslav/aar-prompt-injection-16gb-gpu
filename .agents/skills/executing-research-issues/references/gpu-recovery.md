# GPU Launch and Recovery

Apply this reference only to real GPU work or recovery from it.

## Launch gate

Start a real GPU stage only when the request explicitly authorizes GPU use, preferably with `GPU authorized`, and all of these are verified:

- the exact issue authorizes the stage and its blockers are resolved;
- manifest, provenance, model/checkpoint, dataset, and existing-state identities match;
- output is disk-backed rather than stored in WSL `/tmp`;
- no unrelated GPU workload is active;
- device telemetry and PyTorch allocated/reserved peak telemetry are enabled;
- checksums are validated before artifact promotion;
- the cumulative ledger includes baseline and interrupted intervals;
- projected execution remains within the frozen 24-hour per-seed and 72 GPU-hour cumulative limits.

Any failed gate is `BLOCKED` or `DEVIATION_REQUIRES_DECISION`.

## Boundaries and retries

Use completed epochs for training and whole-checkpoint evaluations as the default recovery boundaries until an accepted decision record establishes a finer boundary. Run one real seed or GPU stage at a time.

After an interruption, inspect durable state, telemetry, artifact completeness, and checksums before resuming. Record the interrupted interval and resource cost. Never automatically repeat a GPU command.

Apply the frozen 2048-to-1536 full-restart fallback only after a real OOM and only once. Preserve the OOM as evidence. Do not force an OOM or introduce finer recovery merely to exercise it.

Held-out data stays sealed without an eligible checkpoint. Never reveal the baseline alone. A null selection is a complete negative result, not permission to promote a capability-failing checkpoint.
