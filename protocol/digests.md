# Manifest digest identities

The canonical digest for publication and provenance claims is the canonical-JSON
content SHA-256 of `protocol/manifest.json`:
`399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20`.
It is invariant to checkout settings and is the `manifest_digest` recorded in
selection records by `runner.selection`.

The raw-file SHA-256 is
`296e093bb1a6fc72f6e4cdf6ed3de5cde77a9e3da90df73db4538a2a98e6f4ac`.
It is recorded in `protocol/manifest.sha256`, checked by
`protocol.validate_manifest.sha256`, and embedded in training, evaluation, and
reveal `StageSignature` values.  `.gitattributes` requires LF for the protocol
files so this byte-integrity and recovery identity is reproducible across
checkouts; it is not the digest to cite for publication claims.
