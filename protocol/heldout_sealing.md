# InjecAgent sealing and reveal procedure

The held-out root is an external WSL path owned by the evaluation identity with mode `0700`; it is never under `runs/`, the repository, or a publishable bundle. Before execution, the sealing job writes a canonical sorted list of the 200 candidate identifiers, the validity-rule document, and their SHA-256 commitments. Only the commitments and counts enter the public manifest.

The evaluator writes baseline and each trained result as append-only blobs under the restricted root. The research runner receives only an opaque receipt (run id, candidate-count commitment, blob digest, and invalid/valid counts); it cannot request plaintext, per-candidate outcomes, aggregates, or comparisons while selection is open. All invalid turns and raw output remain in the restricted blob.

Checkpoint selection is finalized by a canonical JSON record containing the selected checkpoint digest, visible/capability metrics, and the manifest digest. The evaluator verifies that record and its digest, changes state from `SEALED` to `AUTHORIZED`, then atomically produces one reveal package containing baseline and selected-trained results together. The package is read-only and checksummed. A reveal request with a missing/mismatched selection record, changed candidate commitment, or non-final state is rejected.

The reveal package includes `valid_only` (denominator valid turns) and `intent_to_evaluate` (all 200 frozen candidates; invalid technical cases count as failures), plus the invalid classification table. No candidate list, prompt, tool response, secret, credential, or raw held-out text is copied into repository-bound artifacts.
