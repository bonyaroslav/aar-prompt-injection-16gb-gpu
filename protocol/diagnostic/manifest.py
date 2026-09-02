"""Load and digest a diagnostic-protocol manifest (issue #30).

A diagnostic manifest is *not* the frozen protocol: it declares a small,
authorized deviation and names the frozen manifest it is downstream of. This
module gives it the same two digest identities the frozen manifest has (a
checkout-invariant canonical-JSON digest for provenance, a raw-file SHA-256 for
byte integrity) and refuses to load one whose recorded ``downstream_of`` digest
no longer matches the live ``protocol/manifest.json`` -- fail closed if the
frozen baseline has drifted underneath the diagnostic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from protocol.validate_manifest import load as load_frozen_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_MANIFEST_PATH = REPO_ROOT / "protocol" / "manifest.json"

_REQUIRED = (
    "schema", "diagnostic_version", "downstream_of", "authorization", "baseline",
    "model", "change", "model_states", "checkpoint_integrity_policy", "analysis",
    "ambiguity_rule", "boundaries",
)


class DiagnosticManifestError(ValueError):
    """The diagnostic manifest is malformed or its frozen baseline has drifted."""


def canonical_digest(manifest: dict) -> str:
    """Checkout-invariant content digest -- same canonical-JSON SHA-256 recipe as
    ``runner.frozen_inputs._canonical_manifest_digest`` uses for the frozen
    manifest, so the two digest identities are directly comparable."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def raw_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_canonical_digest(frozen_manifest_path: str | Path = FROZEN_MANIFEST_PATH) -> str:
    """The live ``protocol/manifest.json`` canonical digest, via the frozen
    manifest's own validating loader."""
    frozen = load_frozen_manifest(frozen_manifest_path)
    return hashlib.sha256(
        json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load(path: str | Path, *, frozen_manifest_path: str | Path = FROZEN_MANIFEST_PATH) -> dict:
    """Parse, validate and return the diagnostic manifest.

    Raises :class:`DiagnosticManifestError` when a required key is missing or when
    ``downstream_of.canonical_manifest_digest`` no longer equals the live frozen
    manifest's canonical digest.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    missing = [key for key in _REQUIRED if key not in data or data[key] in (None, "", [], {})]
    if missing:
        raise DiagnosticManifestError(f"missing diagnostic manifest keys: {', '.join(missing)}")

    recorded = data["downstream_of"].get("canonical_manifest_digest")
    live = frozen_canonical_digest(frozen_manifest_path)
    if recorded != live:
        raise DiagnosticManifestError(
            "diagnostic manifest is downstream of a frozen protocol digest that no "
            f"longer matches protocol/manifest.json (recorded {recorded}, live {live})"
        )
    return data


if __name__ == "__main__":  # pragma: no cover - operator convenience
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "manifest", nargs="?",
        default=Path(__file__).with_name("chatmode-mmlu-2026-09-02.json"),
    )
    args = parser.parse_args()
    data = load(args.manifest)
    print(f"valid diagnostic manifest: {args.manifest}")
    print(f"  diagnostic_version = {data['diagnostic_version']}")
    print(f"  canonical_digest   = {canonical_digest(data)}")
    print(f"  raw_sha256         = {raw_sha256(args.manifest)}")
