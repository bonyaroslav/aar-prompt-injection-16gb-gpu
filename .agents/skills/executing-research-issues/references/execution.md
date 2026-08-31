# Execution and Recovery

## Preflight

Fetch the live issue and turn every acceptance criterion into an evidence row. Briefly state what the preceding work produced and the consequence for this issue. Confirm blockers, protocol compatibility, relevant repository state, and whether existing user changes overlap the issue.

Estimate active time separately from elapsed and GPU time. Use a range and name the measurements or comparable work behind it. Identify the coarsest durable recovery boundary before starting anything expected to exceed 30 minutes.

- `READY`: all prerequisites exist and the issue can be completed without changing the protocol.
- `BLOCKED`: a dependency, permission, resource, or required evidence is missing. Stop without implementation.
- `DEVIATION_REQUIRES_DECISION`: completion would change frozen behavior, reveal policy, resource limits, or issue scope. Stop and explain the smallest decision needed.

When ready, an explicit `execute #N` authorizes in-scope local edits, proportionate validation, one issue-scoped commit, and closing that issue. It does not authorize pushing, starting another issue, protocol changes, or GPU work without the additional authorization in `gpu-recovery.md`.

## Work

Satisfy only the named issue's acceptance criteria. Prefer committed summaries and targeted artifact reads. Preserve finalized bundles; document their defects externally. Keep partial state separate from final evidence.

Retry a transient read-only or ordinary CPU operation at most once after identifying why a retry is safe. Stop before retrying GPU work, finalization, a checksum failure, or an external write.

Validate in proportion to the change: documentation checks for documentation-only work, targeted tests for code, and integrity checks for evidence paths. A failed check changes the result to `PARTIAL`, `BLOCKED`, or `DEVIATION_REQUIRES_DECISION`; it never becomes a completion footnote.

## Interruption and recovery

For a short task, the diff and task transcript are sufficient recovery state. Before a long task, confirm that its output is disk-backed and its completed boundary is observable from existing durable state. If no reliable boundary exists, stop as `BLOCKED`.

In `recover #N` mode, inspect the working tree, durable state or ledger, artifact completeness, checksums, telemetry, and resource usage before running anything. Resume from the coarsest verified boundary. Unknown or conflicting state stops for review; it is not overwritten or blindly rerun.

## Commit and closure gate

Complete all of these before closing the issue:

1. Every acceptance row cites verified evidence.
2. Relevant checks pass with no unresolved integrity or protocol finding.
3. The diff contains only issue-scoped changes and preserves unrelated user work.
4. A concise human-written commit is created by staging explicit issue paths.
5. The commit exists and the live issue is still open and unblocked.

Do not push unless the user separately requests it. Close the issue with a short result and commit identifier. If closure fails, report `COMPLETE_LOCALLY - ISSUE STILL OPEN` and do not claim full completion.

Finish with a phone-sized summary: result, commit, issue state, material verification, deviations, and one recommended next action. The recommendation is advisory only.

## Common mistakes

| Mistake | Required response |
| --- | --- |
| `execute next` | Assess and name one issue; do not execute it. |
| Dirty unrelated files | Exclude them; stop if changes overlap. |
| Commit exists but one criterion is unclear | Keep the issue open. |
| GitHub unavailable after a valid commit | Report local completion and the open issue. |
| A later issue looks easy | Recommend it without starting it. |
