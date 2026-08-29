# InjecAgent sealing and reveal procedure

The held-out root is an external WSL path owned by the evaluation identity with mode `0700`; it is never under `runs/`, the repository, or a publishable bundle. Before execution, the sealing job writes a canonical sorted list of the 200 candidate identifiers, the validity-rule document, and their SHA-256 commitments. Only the commitments and counts enter the public manifest.

The evaluator writes baseline and each trained result as append-only blobs under the restricted root. The research runner receives only an opaque receipt (run id, candidate-count commitment, blob digest, and invalid/valid counts); it cannot request plaintext, per-candidate outcomes, aggregates, or comparisons while selection is open. All invalid turns and raw output remain in the restricted blob.

Checkpoint selection is finalized by a canonical JSON record containing the selected checkpoint digest, visible/capability metrics, and the manifest digest. The evaluator verifies that record and its digest, changes state from `SEALED` to `AUTHORIZED`, then atomically produces one reveal package containing baseline and selected-trained results together. The package is read-only and checksummed. A reveal request with a missing/mismatched selection record, changed candidate commitment, or non-final state is rejected.

The reveal package includes `valid_only` (denominator valid turns) and `intent_to_evaluate` (all 200 frozen candidates; invalid technical cases count as failures), plus the invalid classification table. No candidate list, prompt, tool response, secret, credential, or raw held-out text is copied into repository-bound artifacts.

## What "sealed" does and does not mean

This mechanism seals *this study's own measurement* of baseline and trained InjecAgent performance from *this study's own researcher*, so checkpoint selection cannot be tuned toward it. It does not — and cannot — make the population-level baseline number secret in any broader sense: `benchmark_docs/prompt_injection/baseline.json`, an upstream file already fingerprinted in this repository's own `protocol/provenance.json`, publishes Qwen3.5-2B's InjecAgent result in plaintext (`mean 0.8881, n 134 of 200, ci [0.828, 0.940]`). `scripts/publish_suite.py` reads that same file to populate the suite YAML's `baseline`/`optimum` fields, so the published number is load-bearing for `run_eval`'s closed-fraction arithmetic, not something the runner could avoid touching.

Two consequences follow, and should be stated plainly in any publication rather than left for a reviewer to notice:

1. **The blindness claim is about this run's held-out *evaluation*, not about the researcher's prior knowledge of the population baseline.** Anyone who reads this repository's own `provenance.json` before Phase 4 already knows the untrained model's approximate InjecAgent rate. Sealing prevents that number (and the *trained* checkpoint's number) from being read back out of the sealer during selection; it does not erase what was public going in.
2. **Ceiling and power are already visible before the study runs.** 0.8881 on a valid-rate of 67% (134/200) leaves roughly 11 points of headroom on `valid_only`, measured on a denominator that can itself shift under training. This is a known constraint to disclose up front (see the minimum-detectable-effect note in `protocol/power_notes.md`), not a discovery to make after the reveal.
