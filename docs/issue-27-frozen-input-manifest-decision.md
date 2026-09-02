# Issue #27 frozen input manifest and exclusion allowlist

**Issue:** [#27](https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/27)
**Decision date:** 2026-09-02
**Protocol:** `phase1-2026-08-29` (`protocol/manifest.json`)
**Scope:** adds `runner/frozen_inputs.py` and `analysis/analysis-config.json`.
It does not change the frozen protocol, modify finalized evidence, access
held-out material, or run any seed.

## What shipped

`python -m runner.frozen_inputs` discovers the finalized inputs the analysis
chain reads from, verifies every one against its recorded checksums, rejects
everything that is not finalized evidence, and emits a JSON record binding each
accepted path to a digest. It **fails closed** (exit 2, offending artifact
named) on any missing, ambiguous, excluded, or checksum-failing input.

## Design decisions

### Inputs are discovered, seeds are derived

The frozen seed list (`training.seeds` = `[17, 42, 2026]`) and epoch count
(`training.optimizer.epochs` = 3) come from the manifest. A seed is *completed*
only when its full evidence set is discovered and verified under the evidence
root:

- baseline bundle `real-baseline-<stamp>` (once, not per seed);
- `training-seed<N>-<stamp>` + `eval-seed<N>-epoch{1..epochs}-<stamp>`;
- `selection-seed<N>*/selection_record.json`, `finalized: true`;
- `seed<N>-resource-comparison*/seed_resource_comparison.json`.

`completed_seed_count` and `trained_checkpoint_count` (= seeds x epochs) are
computed from that discovery, so the record is correct at two completed seeds as
well as three. A seed with **no** artifacts is simply absent; a seed with
**some but not all** artifacts aborts the run -- the incomplete-run case this
ticket exists to catch. This discovery path is uniform across all three seeds
and does not use the recovery workspace, so seed 17 (which predates the
recovery-aware split-run seam) is handled the same way as seeds 42 and 2026.

### Governance artifacts are required per completed seed

Each completed seed must also have its committed outcomes summary
(`analysis/seed<N>-outcomes-summary.md`, fixed convention) and its recorded
continuation decision. The decision records are not name-discoverable (the
seed-17 record is `docs/issue-21-null-selection-continuation-decision.md`), so
`analysis/analysis-config.json` registers them explicitly alongside the
`analysis_version`. A completed seed with evidence but no summary or no
registered/existing decision record aborts, naming the artifact. Evidence for a
seed the manifest never declared also aborts -- the evidence tree must not
contain an unauthorised run.

### Exclusion allowlist (filename filtering is not sufficient)

`classify_exclusion` inspects link status, on-disk location, and (for bundles)
`command.sh`, returning one of: `recovery` (delegated to
`runner.recovery.finalized_inputs_only`, the reused boundary primitive),
`smoke` (name token **or** a `--smoke` flag in `command.sh`), `symlink`,
`archive`, `cache` (incl. HF `models--*` / `blobs` / `snapshots`), `held-out`
(name token or under a supplied held-out root), `credential-like`. Each case has
its own offline test.

### Digest identities

| Input | Digest | Reproducible because |
| --- | --- | --- |
| `protocol/manifest.json` | canonical-JSON SHA-256 `399cf157...` (from #25) | parsed then re-serialised; invariant to checkout EOL |
| finalized bundle | SHA-256 of its `checksums.sha256` | that file already pins every bundle file; bundles are LF by construction |
| selection record | canonical-JSON SHA-256 (matches `runner.selection`) | canonical serialisation |
| resource comparison / outcomes summary / decision record | raw SHA-256 | `protocol/**`, `docs/**`, and now `analysis/**` pinned `text eol=lf` in `.gitattributes` |

## Validation

- New suite `tests/test_frozen_inputs.py` (24 tests) is fully offline: every
  case builds a synthetic evidence tree with the repository's real
  bundle/selection/resource primitives in a temp dir. No GPU, no model or
  dataset adapter, no held-out material, and the real `runs/` tree is never
  read; the only committed file read is `protocol/manifest.json`.
- Acceptance criteria are covered one-to-one (see the test names).
- `python -m runner.frozen_inputs` run against the real local evidence tree
  emits a 3-seed / 9-checkpoint record with manifest digest `399cf157...` and
  the seed-17/42/2026 selection digests `46dfe6ea...` / `a2cdd7c2...` /
  `8df462a4...`.
- Full repository suite: 258 tests pass, 1 skip (the pre-existing missing-`torch`
  skip in `test_real_training`).
- No finalized bundle is modified (asserted by `test_no_finalized_bundle_is_modified`).
