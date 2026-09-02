"""Frozen input manifest and exclusion allowlist (issue #27).

One command produces the checksum-verified, digest-bearing set of finalized
inputs that every later analysis ticket reads from, and **fails closed** when a
required input is missing rather than silently analysing an incomplete or
unauthorised run.

What counts as an input: the frozen baseline bundle, and for every completed
seed the training bundle, the three per-epoch evaluation bundles, the finalized
selection record, the resource comparison artifact, the committed per-seed
outcomes summary, and the recorded continuation decision authorising that seed.

The recovery module already rejects a recovery-workspace path presented as a
finalized input and verifies a bundle's ``checksums.sha256``
(:func:`runner.recovery.finalized_inputs_only`); this module reuses that
boundary and layers the wider allowlist (symlink / archive / cache / held-out /
smoke / credential-like) and the missing-governance-artifact checks around it.

Everything here is a pure function over a filesystem tree plus the frozen
manifest: no GPU, no model, no dataset, no held-out access. The seed count and
trained-checkpoint count are derived from the discovered evidence and the frozen
manifest, never hardcoded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner.bundle import CHECKSUM_FILE
from runner.real_seed_run import (
    _canonical_selection_digest,
    verify_seed_comparison_artifact,
)
from runner.recovery import finalized_inputs_only
from runner.selection import verify_selection_record

DEFAULT_CONFIG_PATH = Path("analysis/analysis-config.json")

ARCHIVE_SUFFIXES = (
    ".tar", ".tgz", ".gz", ".zip", ".7z", ".rar", ".bz2", ".xz", ".zst", ".lz4",
)
CREDENTIAL_NAME_TOKENS = (
    "credential", "secret", "password", "id_rsa", "id_ed25519", ".netrc",
    "authorized_keys", "token",
)
CREDENTIAL_SUFFIXES = (".pem", ".key", ".pfx", ".p12", ".keystore", ".env")
CACHE_PATH_TOKENS = (
    "__pycache__", ".cache", "cache", "hf_home", "huggingface", "hub",
    "blobs", "snapshots", "models--", ".pytest_cache",
)
HELDOUT_PATH_TOKENS = (
    "injecagent", "inject-agent", "held-out", "held_out", "heldout", "sealed",
)
SMOKE_PATH_TOKENS = ("smoke",)


class FrozenInputError(RuntimeError):
    """A required input is missing, ambiguous, or fails its integrity check."""

    def __init__(self, message: str, *, path: Path | str | None = None):
        super().__init__(message)
        self.path = None if path is None else str(path)


class ExcludedInputError(FrozenInputError):
    """A path that is not finalized evidence was presented as an input."""

    def __init__(self, reason: str, path: Path | str):
        super().__init__(f"excluded input ({reason}): {path}", path=path)
        self.reason = reason


# --- exclusion allowlist ---------------------------------------------------


def _relative_parts(path: Path, root: Path) -> list[str]:
    try:
        return list(path.resolve().relative_to(root.resolve()).parts)
    except ValueError:
        return list(path.parts)


def _has_symlink_component(path: Path, root: Path) -> bool:
    current = path
    root = root.resolve()
    seen = set()
    while True:
        if current.is_symlink():
            return True
        resolved = current.resolve()
        if resolved == root or resolved in seen:
            return False
        seen.add(resolved)
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _command_requests_smoke(path: Path) -> bool:
    command = path / "command.sh"
    if not command.is_file():
        return False
    text = command.read_text(encoding="utf-8", errors="replace").lower()
    return "--smoke" in text or "smoke" in text.split("#", 1)[0].split()


def classify_exclusion(path: Path, *, evidence_root: Path, recovery_root: Path,
                        heldout_root: Path | None = None) -> str | None:
    """Return the exclusion reason for ``path``, or ``None`` if it is admissible.

    Filename filtering alone is not sufficient, so this also inspects link
    status, the on-disk location relative to the recovery workspace and the
    restricted held-out root, and (for bundles) whether ``command.sh`` carries a
    smoke flag.
    """
    path = Path(path)
    lowered_parts = [part.lower() for part in _relative_parts(path, evidence_root)]
    name = path.name.lower()

    if _has_symlink_component(path, evidence_root):
        return "symlink"
    if path.suffix.lower() in ARCHIVE_SUFFIXES or any(
        part.endswith(ARCHIVE_SUFFIXES) for part in lowered_parts
    ):
        return "archive"
    if path.suffix.lower() in CREDENTIAL_SUFFIXES or any(
        token in name for token in CREDENTIAL_NAME_TOKENS
    ):
        return "credential-like"
    if any(
        token in part for part in lowered_parts for token in CACHE_PATH_TOKENS
    ):
        return "cache"
    if heldout_root is not None:
        try:
            path.resolve().relative_to(Path(heldout_root).resolve())
            return "held-out"
        except ValueError:
            pass
    if any(
        token in part for part in lowered_parts for token in HELDOUT_PATH_TOKENS
    ):
        return "held-out"
    if any(
        token in part for part in lowered_parts for token in SMOKE_PATH_TOKENS
    ):
        return "smoke"
    if path.is_dir() and _command_requests_smoke(path):
        return "smoke"

    try:
        finalized_inputs_only([path], recovery_root)
    except ValueError as error:
        if "recovery workspace" in str(error):
            return "recovery"
        # A checksum failure is a real integrity problem, not an exclusion --
        # let the caller surface it with the offending path named.
    return None


def assert_admissible_input(path: Path, *, evidence_root: Path, recovery_root: Path,
                             heldout_root: Path | None = None) -> None:
    reason = classify_exclusion(
        path, evidence_root=evidence_root, recovery_root=recovery_root,
        heldout_root=heldout_root,
    )
    if reason is not None:
        raise ExcludedInputError(reason, path)


# --- digests -------------------------------------------------------------


def _canonical_manifest_digest(manifest: dict) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bundle_digest(bundle_dir: Path) -> str:
    """A finalized bundle's identity is the digest of its ``checksums.sha256``
    -- which itself pins every bundle file -- so one value fingerprints the
    whole bundle."""
    return _sha256_file(Path(bundle_dir) / CHECKSUM_FILE)


# --- discovery ----------------------------------------------------------


def _unique_child(root: Path, pattern: str, *, kind: str,
                   reject_substrings: tuple[str, ...] = ()) -> list[Path]:
    matches = sorted(
        child for child in root.glob(pattern)
        if not any(bad in child.name for bad in reject_substrings)
    )
    return matches


def _verify_bundle_input(bundle_dir: Path, *, evidence_root: Path,
                          recovery_root: Path, repo_root: Path) -> dict:
    assert_admissible_input(
        bundle_dir, evidence_root=evidence_root, recovery_root=recovery_root,
    )
    try:
        finalized_inputs_only([bundle_dir], recovery_root)
    except ValueError as error:
        raise FrozenInputError(
            f"bundle failed its recorded checksums: {bundle_dir} ({error})",
            path=bundle_dir,
        ) from error
    return {
        "path": _repo_relative(bundle_dir, repo_root),
        "digest": "sha256:" + _bundle_digest(bundle_dir),
        "digest_kind": "bundle_checksums",
    }


def _repo_relative(path: Path, repo_root: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _discover_seed_evidence(*, evidence_root: Path, recovery_root: Path,
                             repo_root: Path, seed: int, epochs: int) -> list[dict] | None:
    """Return the ordered, verified input rows for one seed's evidence, or
    ``None`` when the seed has not been started at all. A seed that is *partly*
    present aborts -- the incomplete-run failure this ticket exists to catch."""
    slots: dict[str, list[Path]] = {
        "training_bundle": _unique_child(
            evidence_root, f"training-seed{seed}-*", kind="training bundle",
            reject_substrings=("epoch",),
        ),
        "selection_record": [
            child for child in sorted(evidence_root.glob(f"selection-seed{seed}*"))
            if (child / "selection_record.json").is_file()
        ],
        "resource_comparison": _unique_child(
            evidence_root, f"seed{seed}-resource-comparison*",
            kind="resource comparison",
        ),
    }
    for epoch in range(1, epochs + 1):
        slots[f"evaluation_bundle_epoch{epoch}"] = _unique_child(
            evidence_root, f"eval-seed{seed}-epoch{epoch}-*", kind="evaluation bundle",
        )

    present = {role: found for role, found in slots.items() if found}
    if not present:
        return None
    missing = [role for role in slots if role not in present]
    if missing:
        raise FrozenInputError(
            f"seed {seed}: evidence is present but incomplete -- missing {missing[0]}"
        )
    ambiguous = {role: found for role, found in present.items() if len(found) > 1}
    if ambiguous:
        role, found = next(iter(ambiguous.items()))
        raise FrozenInputError(
            f"seed {seed}: ambiguous {role}: {[p.name for p in found]}"
        )

    rows: list[dict] = []
    order = [
        "training_bundle",
        *(f"evaluation_bundle_epoch{e}" for e in range(1, epochs + 1)),
    ]
    for role in order:
        row = _verify_bundle_input(
            present[role][0], evidence_root=evidence_root,
            recovery_root=recovery_root, repo_root=repo_root,
        )
        rows.append({"role": f"seed{seed}_{role}", **row})

    selection_dir = present["selection_record"][0]
    selection_path = selection_dir / "selection_record.json"
    assert_admissible_input(
        selection_path, evidence_root=evidence_root, recovery_root=recovery_root,
    )
    try:
        verify_selection_record(selection_path, _canonical_selection_digest(selection_path))
    except ValueError as error:
        raise FrozenInputError(
            f"seed {seed}: selection record failed its checksum: {selection_path} ({error})",
            path=selection_path,
        ) from error
    record = json.loads(selection_path.read_text(encoding="utf-8"))
    if not record.get("finalized"):
        raise FrozenInputError(f"seed {seed}: selection record is not finalized")
    rows.append({
        "role": f"seed{seed}_selection_record",
        "path": _repo_relative(selection_path, repo_root),
        "digest": "sha256:" + _canonical_selection_digest(selection_path),
        "digest_kind": "canonical_json",
    })

    comparison_dir = present["resource_comparison"][0]
    assert_admissible_input(
        comparison_dir, evidence_root=evidence_root, recovery_root=recovery_root,
    )
    try:
        verify_seed_comparison_artifact(comparison_dir)
    except ValueError as error:
        raise FrozenInputError(
            f"seed {seed}: resource comparison failed its checksum: {comparison_dir} ({error})",
            path=comparison_dir,
        ) from error
    rows.append({
        "role": f"seed{seed}_resource_comparison",
        "path": _repo_relative(comparison_dir / "seed_resource_comparison.json", repo_root),
        "digest": "sha256:" + _sha256_file(comparison_dir / "seed_resource_comparison.json"),
        "digest_kind": "sha256",
    })
    return rows


def _governance_rows(*, repo_root: Path, seed: int, governance_records: dict) -> list[dict]:
    summary = repo_root / f"analysis/seed{seed}-outcomes-summary.md"
    if not summary.is_file():
        raise FrozenInputError(
            f"seed {seed}: completed evidence but no outcomes summary at "
            f"analysis/seed{seed}-outcomes-summary.md",
            path=summary,
        )
    registered = governance_records.get(str(seed))
    if not registered:
        raise FrozenInputError(
            f"seed {seed}: completed evidence but no continuation decision record "
            f"registered in the analysis config"
        )
    decision = repo_root / registered
    if not decision.is_file():
        raise FrozenInputError(
            f"seed {seed}: registered continuation decision record is missing: {registered}",
            path=decision,
        )
    return [
        {
            "role": f"seed{seed}_outcomes_summary",
            "path": _repo_relative(summary, repo_root),
            "digest": "sha256:" + _sha256_file(summary),
            "digest_kind": "sha256",
        },
        {
            "role": f"seed{seed}_continuation_decision",
            "path": _repo_relative(decision, repo_root),
            "digest": "sha256:" + _sha256_file(decision),
            "digest_kind": "sha256",
        },
    ]


def freeze_inputs(*, manifest_path: Path, evidence_root: Path, repo_root: Path,
                   recovery_root: Path, config_path: Path | None = None,
                   heldout_root: Path | None = None) -> dict:
    """Discover, verify and bind every finalized input, returning the frozen
    input record. Raises :class:`FrozenInputError` (fail closed) on any missing,
    ambiguous, excluded, or checksum-failing input."""
    manifest_path = Path(manifest_path)
    evidence_root = Path(evidence_root)
    repo_root = Path(repo_root)
    recovery_root = Path(recovery_root)

    manifest = load_manifest(manifest_path)
    epochs = manifest["training"]["optimizer"]["epochs"]
    frozen_seeds = list(manifest["training"]["seeds"])

    config_path = Path(config_path) if config_path is not None else repo_root / DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FrozenInputError(f"analysis config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    analysis_version = config["analysis_version"]
    governance_records = config.get("governance_records", {})

    if not evidence_root.is_dir():
        raise FrozenInputError(f"evidence root not found: {evidence_root}")

    # A selection directory for a seed the frozen manifest never declared means
    # the evidence tree holds an unauthorised run -- refuse rather than analyse it.
    declared = {str(seed) for seed in frozen_seeds}
    for child in sorted(evidence_root.glob("selection-seed*")):
        tail = child.name[len("selection-seed"):].split("-", 1)[0]
        if tail and tail not in declared:
            raise FrozenInputError(
                f"evidence tree holds selection records for undeclared seed {tail}: {child}"
            )

    inputs: list[dict] = [{
        "role": "protocol_manifest",
        "path": _repo_relative(manifest_path, repo_root),
        "digest": "sha256:" + _canonical_manifest_digest(manifest),
        "digest_kind": "canonical_json",
    }]

    baseline_matches = _unique_child(
        evidence_root, "real-baseline-*", kind="baseline bundle",
        reject_substrings=("comparison", "data"),
    )
    if not baseline_matches:
        raise FrozenInputError("frozen baseline bundle not found under the evidence root")
    if len(baseline_matches) > 1:
        raise FrozenInputError(
            f"ambiguous baseline bundle: {[p.name for p in baseline_matches]}"
        )
    inputs.append({
        "role": "baseline_bundle",
        **_verify_bundle_input(
            baseline_matches[0], evidence_root=evidence_root,
            recovery_root=recovery_root, repo_root=repo_root,
        ),
    })

    completed_seeds: list[int] = []
    for seed in frozen_seeds:
        seed_rows = _discover_seed_evidence(
            evidence_root=evidence_root, recovery_root=recovery_root,
            repo_root=repo_root, seed=seed, epochs=epochs,
        )
        if seed_rows is None:
            continue
        seed_rows += _governance_rows(
            repo_root=repo_root, seed=seed, governance_records=governance_records,
        )
        completed_seeds.append(seed)
        inputs.extend(seed_rows)

    if not completed_seeds:
        raise FrozenInputError("no completed seed has a full finalized evidence set")

    return {
        "analysis_version": analysis_version,
        "protocol_version": manifest["protocol_version"],
        "protocol_manifest_digest": _canonical_manifest_digest(manifest),
        "epochs_per_seed": epochs,
        "frozen_seeds": frozen_seeds,
        "completed_seeds": completed_seeds,
        "completed_seed_count": len(completed_seeds),
        "trained_checkpoint_count": len(completed_seeds) * epochs,
        "inputs": inputs,
    }


def _render(record: dict) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--manifest", type=Path, default=repo_root / "protocol" / "manifest.json")
    parser.add_argument("--evidence-root", type=Path, default=repo_root / "runs")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--recovery-root", type=Path, default=repo_root / "recovery")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--heldout-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="write the frozen input record here (default: stdout)")
    args = parser.parse_args(argv)

    try:
        record = freeze_inputs(
            manifest_path=args.manifest, evidence_root=args.evidence_root,
            repo_root=args.repo_root, recovery_root=args.recovery_root,
            config_path=args.config, heldout_root=args.heldout_root,
        )
    except FrozenInputError as error:
        parser.exit(status=2, message=f"frozen-input check failed: {error}\n")

    rendered = _render(record)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
