# Handover: issue #27 — Frozen input manifest and exclusion allowlist

**Prepared:** 2026-09-02, after seed 2026 finalized (commit `88bf966`).
**Status:** #27 is unblocked and is the next `ready-for-agent` issue in dependency order.

## Where things stand

- Frozen replication set `[17, 42, 2026]` is **complete**: 3 seeds, 9 trained
  checkpoints, 9 capability-gate failures, 3 finalized null selections,
  InjecAgent sealed throughout. Cumulative GPU 47.34 / 72 h.
- Blockers cleared: #25 (digest determinism), #26 (ablation mid-epoch recovery),
  #23 (seed 2026) are all closed. #24 (workflow repair) closed.
- Remaining open `ready-for-agent`: #27 → #28 → #29 → #30 → #31 → #32 → #33
  (the analysis-and-publication chain that replaced the closed #14).
- Full test suite: `234 pass, 1 skip` via `.venv/Scripts/python.exe -m unittest
  discover -s tests -q` (see [[test-env-torch-gap]]).

## What #27 builds

One command emits a checksum-verified, digest-bearing frozen-input record that
every later analysis ticket reads from, and **fails closed** when a required
input is missing. It is the tracer bullet: discover finalized bundles → verify
their recorded checksums → reject everything that is not finalized evidence →
emit a record binding each accepted input to its digest.

## Reuse, do not reimplement

- `runner.recovery.finalized_inputs_only(paths, recovery_root)` — already
  rejects any recovery-workspace path and runs `verify_bundle` on the rest.
  Build the wider allowlist (symlink / archive / cache / held-out /
  credential-like) around this; do not replace its boundary check.
- `runner.bundle.verify_bundle` — per-bundle `checksums.sha256` verification.
- `runner.selection.verify_selection_record(path, digest)` and
  `runner.real_seed_run._canonical_selection_digest(path)` — selection records.
- `runner.real_seed_run.verify_seed_comparison_artifact(dir)` — resource
  artifacts.
- `runner.real_seed_run.discover_finalized_seed_evidence(manifest, recovery_root=,
  output_root=, seed=)` — returns the clean per-seed topology for seeds 42 and
  2026 (recovery-aware seam). **Seed 17 predates the seam** and has no recovery
  boundary markers, so it raises `training is not finalized` there — verify seed
  17 with direct `verify_bundle` / `verify_selection_record` on its four bundles
  instead. #27 should present one discovery path that works for all three.

## Frozen values #27 must bind to

- Protocol manifest digest to record: the **canonical-JSON content digest**
  `399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20`
  (`protocol/digests.md` — invariant to checkout settings). Not the raw-file
  digest `296e093b…`.
- Analysis version: introduce one; later tickets cite it.

## Inputs that count (per completed seed + baseline)

| Input | Path (gitignored evidence) | Integrity |
| --- | --- | --- |
| Frozen baseline | `runs/real-baseline-20260829-205020/` | `verify_bundle` (8/8) |
| Seed 17 training | `runs/training-seed17-20260830-071553/` | `verify_bundle` |
| Seed 17 evals ×3 | `runs/eval-seed17-epoch{1,2,3}-20260830-071553/` | `verify_bundle` each |
| Seed 17 selection | `runs/selection-seed17-20260830-071553/selection_record.json` | digest `46dfe6ea…` |
| Seed 17 resources | `runs/seed17-resource-comparison-20260830-071553/` | `verify_seed_comparison_artifact` |
| Seed 42 training | `runs/training-seed42-20260831-201248-1b487000/` | `verify_bundle` |
| Seed 42 evals ×3 | `runs/eval-seed42-epoch{1,2,3}-20260831-201248-1b487000/` | `verify_bundle` each |
| Seed 42 selection | `runs/selection-seed42/selection_record.json` | digest `a2cdd7c2…` |
| Seed 42 resources | `runs/seed42-resource-comparison/` | `verify_seed_comparison_artifact` |
| Seed 2026 training | `runs/training-seed2026-20260901-112915-bf0809d1/` | `verify_bundle` |
| Seed 2026 evals ×3 | `runs/eval-seed2026-epoch{1,2,3}-20260901-112915-bf0809d1/` | `verify_bundle` each |
| Seed 2026 selection | `runs/selection-seed2026/selection_record.json` | digest `8df462a4…` |
| Seed 2026 resources | `runs/seed2026-resource-comparison/` | `verify_seed_comparison_artifact` |

## What must be rejected (each with its own test — filename filtering is not enough)

Recovery workspace paths (`recovery/`), smoke-run paths
(`runs/seed2026-smoke/`, `runs/gpu-smoke-*`, any bundle whose `command.sh`
carries a `--smoke-*` flag), the pre-run adapter dirs
`runs/adapters-seed2026-20260901-103119` / `-103738`, symlinks, archives
(`.tar`, `.zip`, …), caches / model caches, the restricted held-out root and any
raw InjecAgent material, private work dirs, anything credential-like.

## Seed-count must be derived, not hardcoded

Every count scales with completed seeds: at 3 seeds the analysis unit is 9
checkpoints and the gate contrast is 9-of-9 vs 0-of-9; the record must still be
correct at 2 seeds (in case a later integrity finding invalidates one). Derive
seed count and checkpoint count from the discovered evidence.

## Definition of done

- Offline tests against fixtures only — no GPU, no reading the real evidence
  tree.
- Existing suite (`234 pass, 1 skip`) unchanged.
- New behaviour TDD'd against the acceptance criteria in issue #27.
- No finalized bundle modified.
- On success: update `RESEARCH_PLAN.md` **Status**, commit `Closes #27` with the
  `Co-Authored-By: Claude Sonnet 5` trailer, `gh issue close 27`.

## Reference records

- `docs/issue-14-finalization-handover.md` — the finalization design constraints
  (null-selection topology, exclusion criterion, provenance manifest, claim
  language). #27–#33 implement these.
- `docs/issue-23-seed-2026-execution-decision.md`,
  `docs/issue-22-seed-42-execution-decision.md` — the pattern #27's record
  follows.
- `analysis/seed{17,42,2026}-outcomes-summary.md` — durable per-seed results.
- `protocol/digests.md`, `protocol/deviations.md` — digest identities and the
  line-ending hazard note.
