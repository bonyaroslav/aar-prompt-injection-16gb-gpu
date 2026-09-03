"""Load and digest the issue #31 corpus-ablation protocol manifest.

The ablation is downstream of immutable Attempt-1 evidence.  Its loader verifies
that relationship against the live frozen manifest before any corpus build or
GPU work can begin.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from protocol.validate_manifest import load as load_frozen_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_MANIFEST_PATH = REPO_ROOT / "protocol" / "manifest.json"

_REQUIRED = (
    "schema", "ablation_version", "purpose", "downstream_of", "authorization",
    "baseline", "training", "corpus", "recovery", "analysis", "resources", "boundaries",
)


class AblationManifestError(ValueError):
    """The ablation manifest is incomplete or no longer matches Attempt-1."""


def canonical_digest(manifest: dict) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def raw_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_canonical_digest(frozen_manifest_path: str | Path = FROZEN_MANIFEST_PATH) -> str:
    frozen = load_frozen_manifest(frozen_manifest_path)
    return canonical_digest(frozen)


def load(path: str | Path, *, frozen_manifest_path: str | Path = FROZEN_MANIFEST_PATH) -> dict:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in _REQUIRED if key not in data or data[key] in (None, "", [], {})]
    if missing:
        raise AblationManifestError(f"missing ablation manifest keys: {', '.join(missing)}")

    recorded = data["downstream_of"].get("canonical_manifest_digest")
    live = frozen_canonical_digest(frozen_manifest_path)
    if recorded != live:
        raise AblationManifestError(
            "ablation manifest is downstream of a frozen protocol digest that no longer matches "
            f"protocol/manifest.json (recorded {recorded}, live {live})"
        )
    return data


if __name__ == "__main__":  # pragma: no cover - operator convenience
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("manifest", nargs="?", default=Path(__file__).with_name("corpus-ablation-2026-09-02.json"))
    args = parser.parse_args()
    data = load(args.manifest)
    print(f"valid ablation manifest: {args.manifest}")
    print(f"  ablation_version = {data['ablation_version']}")
    print(f"  canonical_digest = {canonical_digest(data)}")
    print(f"  raw_sha256       = {raw_sha256(args.manifest)}")
