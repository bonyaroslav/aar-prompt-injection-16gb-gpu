"""Issue #27: frozen input manifest and exclusion allowlist.

Every test runs offline against a synthetic evidence tree built with the
repository's real bundle/selection/resource primitives -- no GPU, no model or
dataset adapter, no held-out material, and without reading the real ``runs/``
evidence tree. The frozen protocol manifest is the only committed file read.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from protocol.validate_manifest import load as load_manifest
from runner.bundle import BUNDLE_FILES, finalize_bundle, write_bundle
from runner.real_seed_run import write_seed_comparison_artifact
from runner.selection import finalize_selection_record
from runner import frozen_inputs
from runner.frozen_inputs import (
    ExcludedInputError,
    FrozenInputError,
    classify_exclusion,
    freeze_inputs,
)

MANIFEST = Path(__file__).parents[1] / "protocol" / "manifest.json"
FROZEN = load_manifest(MANIFEST)
EPOCHS = FROZEN["training"]["optimizer"]["epochs"]
SEEDS = FROZEN["training"]["seeds"]
CANONICAL_MANIFEST_DIGEST = (
    "399cf1572ccf580f4741429230cb556840a7e1c9fdb514ba053c9a6f16ce7f20"
)


def _finalized_bundle(path: Path, *, command: str = "python -m runner.real_seed_run\n") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    contents = {name: f"{name} for {path.name}\n" for name in BUNDLE_FILES}
    contents["command.sh"] = command
    write_bundle(path, contents)
    finalize_bundle(path)
    return path


def _finalized_selection(dir_path: Path, *, finalized: bool = True) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    record = {
        "protocol_version": FROZEN["protocol_version"],
        "manifest_digest": "sha256:" + CANONICAL_MANIFEST_DIGEST,
        "candidates": [],
        "selected_checkpoint_digest": None,
        "selected_epoch": None,
        "finalized": finalized,
    }
    finalize_selection_record(record, dir_path / "selection_record.json")
    return dir_path / "selection_record.json"


def _resource_artifact(evidence_root: Path, seed: int) -> Path:
    return write_seed_comparison_artifact(
        evidence_root, f"seed{seed}-resource-comparison",
        {"seed": seed, "measured": {"gpu_hours": 12.7}, "cumulative_gpu_hours": 34.2},
    )


def _make_writable(path: Path) -> None:
    os.chmod(path, 0o644)


class _Repo:
    """A synthetic repo + evidence tree for one test."""

    def __init__(self, root: Path, *, seeds=(17, 42)):
        self.root = root
        self.evidence_root = root / "runs"
        self.recovery_root = root / "recovery"
        self.config_path = root / "analysis" / "analysis-config.json"
        self.evidence_root.mkdir(parents=True)
        self.recovery_root.mkdir(parents=True)
        (root / "analysis").mkdir(exist_ok=True)
        (root / "docs").mkdir()
        self.seeds = list(seeds)

        _finalized_bundle(self.evidence_root / "real-baseline-20260829-205020")
        # decoys that must never be picked up as the baseline bundle
        (self.evidence_root / "real-baseline-data").mkdir()
        _finalized_bundle(self.evidence_root / "real-baseline-comparison-20260829-205020")

        for seed in seeds:
            self._add_seed_evidence(seed)
            self._add_seed_governance(seed)

        self._write_config()

    def _add_seed_evidence(self, seed: int, *, stamp: str = "20260901-112915-bf0809d1") -> None:
        _finalized_bundle(self.evidence_root / f"training-seed{seed}-{stamp}")
        for epoch in range(1, EPOCHS + 1):
            _finalized_bundle(self.evidence_root / f"eval-seed{seed}-epoch{epoch}-{stamp}")
        _finalized_selection(self.evidence_root / f"selection-seed{seed}")
        _resource_artifact(self.evidence_root, seed)

    def _add_seed_governance(self, seed: int) -> None:
        (self.root / "analysis" / f"seed{seed}-outcomes-summary.md").write_text(
            f"# Seed {seed} outcomes summary\n", encoding="utf-8",
        )
        (self.root / "docs" / f"seed{seed}-decision.md").write_text(
            f"# Seed {seed} continuation decision\n\ncumulative GPU-hour ledger cited.\n",
            encoding="utf-8",
        )

    def _write_config(self) -> None:
        self.config_path.write_text(
            json.dumps({
                "analysis_version": "test-analysis-v1",
                "governance_records": {
                    str(seed): f"docs/seed{seed}-decision.md" for seed in SEEDS
                },
            }, indent=2) + "\n",
            encoding="utf-8",
        )

    def freeze(self, **overrides):
        kwargs = dict(
            manifest_path=MANIFEST, evidence_root=self.evidence_root,
            repo_root=self.root, recovery_root=self.recovery_root,
            config_path=self.config_path,
        )
        kwargs.update(overrides)
        return freeze_inputs(**kwargs)


class FrozenInputRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _repo(self, **kwargs) -> _Repo:
        return _Repo(self.root, **kwargs)

    def test_emits_record_binding_every_input_to_a_digest(self):
        record = self._repo(seeds=(17, 42, 2026)).freeze()

        self.assertEqual(record["analysis_version"], "test-analysis-v1")
        self.assertEqual(record["protocol_manifest_digest"], CANONICAL_MANIFEST_DIGEST)
        self.assertEqual(record["protocol_version"], "phase1-2026-08-29")
        for row in record["inputs"]:
            self.assertIn("path", row)
            self.assertRegex(row["digest"], r"^sha256:[0-9a-f]{64}$")
        roles = {row["role"] for row in record["inputs"]}
        self.assertIn("protocol_manifest", roles)
        self.assertIn("baseline_bundle", roles)
        for seed in (17, 42, 2026):
            self.assertIn(f"seed{seed}_training_bundle", roles)
            self.assertIn(f"seed{seed}_evaluation_bundle_epoch{EPOCHS}", roles)
            self.assertIn(f"seed{seed}_selection_record", roles)
            self.assertIn(f"seed{seed}_resource_comparison", roles)
            self.assertIn(f"seed{seed}_outcomes_summary", roles)
            self.assertIn(f"seed{seed}_continuation_decision", roles)

    def test_manifest_digest_is_the_canonical_one_from_ticket_25(self):
        record = self._repo().freeze()
        manifest_row = next(r for r in record["inputs"] if r["role"] == "protocol_manifest")
        self.assertEqual(manifest_row["digest"], "sha256:" + CANONICAL_MANIFEST_DIGEST)
        self.assertEqual(manifest_row["digest_kind"], "canonical_json")
        # Reproducible from the parsed manifest, independent of the file's bytes.
        reparsed = json.dumps(load_manifest(MANIFEST), sort_keys=True, separators=(",", ":"))
        self.assertEqual(
            manifest_row["digest"],
            "sha256:" + hashlib.sha256(reparsed.encode()).hexdigest(),
        )

    def test_seed_and_checkpoint_counts_are_derived_not_hardcoded(self):
        two = self._repo(seeds=(17, 42)).freeze()
        self.assertEqual(two["completed_seed_count"], 2)
        self.assertEqual(two["trained_checkpoint_count"], 2 * EPOCHS)
        self.assertEqual(two["completed_seeds"], [17, 42])

        self.setUp()  # fresh tmp dir
        three = self._repo(seeds=(17, 42, 2026)).freeze()
        self.assertEqual(three["completed_seed_count"], 3)
        self.assertEqual(three["trained_checkpoint_count"], 3 * EPOCHS)

    def test_a_seed_that_never_started_simply_lowers_the_count(self):
        record = self._repo(seeds=(17, 42)).freeze()
        self.assertNotIn(2026, record["completed_seeds"])

    def test_checksum_failure_aborts_naming_the_offending_path(self):
        repo = self._repo()
        victim = repo.evidence_root / "eval-seed42-epoch2-20260901-112915-bf0809d1" / "metrics.json"
        _make_writable(victim)
        victim.write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(FrozenInputError) as caught:
            repo.freeze()
        self.assertIn("eval-seed42-epoch2", str(caught.exception))

    def test_missing_outcomes_summary_aborts_naming_the_artifact(self):
        repo = self._repo()
        (repo.root / "analysis" / "seed42-outcomes-summary.md").unlink()
        with self.assertRaises(FrozenInputError) as caught:
            repo.freeze()
        message = str(caught.exception)
        self.assertIn("seed 42", message)
        self.assertIn("outcomes summary", message)

    def test_missing_continuation_decision_aborts_naming_the_artifact(self):
        repo = self._repo()
        (repo.root / "docs" / "seed42-decision.md").unlink()
        with self.assertRaises(FrozenInputError) as caught:
            repo.freeze()
        message = str(caught.exception)
        self.assertIn("seed 42", message)
        self.assertIn("seed42-decision.md", message)

    def test_partly_present_seed_evidence_fails_closed(self):
        repo = self._repo(seeds=(17, 42))
        # seed 2026 got a training bundle but nothing else -- an interrupted run.
        _finalized_bundle(repo.evidence_root / "training-seed2026-20260901-112915")
        with self.assertRaises(FrozenInputError) as caught:
            repo.freeze()
        self.assertIn("seed 2026", str(caught.exception))
        self.assertIn("incomplete", str(caught.exception))

    def test_unfinalized_selection_record_is_rejected(self):
        repo = self._repo()
        selection_dir = repo.evidence_root / "selection-seed42"
        _make_writable(selection_dir / "selection_record.json")
        (selection_dir / "selection_record.json").unlink()
        _finalized_selection(selection_dir, finalized=False)
        with self.assertRaises(FrozenInputError) as caught:
            repo.freeze()
        self.assertIn("not finalized", str(caught.exception))

    def test_evidence_for_an_undeclared_seed_is_refused(self):
        repo = self._repo(seeds=(17, 42))
        _finalized_selection(repo.evidence_root / "selection-seed999")
        with self.assertRaises(FrozenInputError) as caught:
            repo.freeze()
        self.assertIn("999", str(caught.exception))

    def test_no_finalized_bundle_is_modified(self):
        repo = self._repo(seeds=(17, 42, 2026))
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in repo.evidence_root.rglob("*") if path.is_file()
        }
        repo.freeze()
        after = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in repo.evidence_root.rglob("*") if path.is_file()
        }
        self.assertEqual(before, after)

    def test_record_is_deterministic(self):
        repo = self._repo(seeds=(17, 42, 2026))
        self.assertEqual(repo.freeze(), repo.freeze())

    def test_cli_writes_record_and_fails_closed_with_exit_2(self):
        repo = self._repo(seeds=(17, 42))
        out = self.root / "out" / "frozen-inputs.json"
        rc = frozen_inputs.main([
            "--manifest", str(MANIFEST), "--evidence-root", str(repo.evidence_root),
            "--repo-root", str(self.root), "--recovery-root", str(repo.recovery_root),
            "--config", str(repo.config_path), "--out", str(out),
        ])
        self.assertEqual(rc, 0)
        written = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(written["completed_seed_count"], 2)

        (repo.root / "docs" / "seed42-decision.md").unlink()
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(stderr):
            frozen_inputs.main([
                "--manifest", str(MANIFEST), "--evidence-root", str(repo.evidence_root),
                "--repo-root", str(self.root), "--recovery-root", str(repo.recovery_root),
                "--config", str(repo.config_path),
            ])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("seed42-decision.md", stderr.getvalue())


class ExclusionAllowlistTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.evidence_root = self.root / "runs"
        self.recovery_root = self.root / "recovery"
        self.evidence_root.mkdir(parents=True)
        self.recovery_root.mkdir(parents=True)

    def _classify(self, path: Path, **kwargs):
        return classify_exclusion(
            path, evidence_root=self.evidence_root, recovery_root=self.recovery_root,
            **kwargs,
        )

    def test_admissible_bundle_is_not_excluded(self):
        bundle = _finalized_bundle(self.evidence_root / "training-seed17-20260830-071553")
        self.assertIsNone(self._classify(bundle))

    def test_recovery_workspace_path_is_rejected(self):
        inside = self.recovery_root / "training-seed17-seq2048-epoch1.json"
        inside.write_text("{}", encoding="utf-8")
        self.assertEqual(self._classify(inside), "recovery")

    def test_smoke_run_path_is_rejected_by_name(self):
        smoke = _finalized_bundle(self.evidence_root / "seed2026-smoke")
        self.assertEqual(self._classify(smoke), "smoke")

    def test_smoke_run_path_is_rejected_by_command_flag(self):
        sneaky = _finalized_bundle(
            self.evidence_root / "training-seed2026-20260901-103119",
            command="python -m runner.real_seed_run --smoke-eval\n",
        )
        self.assertEqual(self._classify(sneaky), "smoke")

    def test_symlink_is_rejected(self):
        target = _finalized_bundle(self.evidence_root / "training-seed17-real")
        link = self.evidence_root / "training-seed17-20260830-071553"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation not permitted in this environment")
        self.assertEqual(self._classify(link), "symlink")

    def test_archive_is_rejected(self):
        archive = self.evidence_root / "training-seed17-20260830-071553.tar.gz"
        archive.write_bytes(b"\x1f\x8b")
        self.assertEqual(self._classify(archive), "archive")

    def test_cache_path_is_rejected(self):
        cache = self.evidence_root / "model-cache" / "models--meta-llama" / "blobs"
        cache.mkdir(parents=True)
        self.assertEqual(self._classify(cache), "cache")

    def test_held_out_material_is_rejected_by_token(self):
        heldout = self.evidence_root / "injecagent-candidates"
        heldout.mkdir()
        self.assertEqual(self._classify(heldout), "held-out")

    def test_held_out_material_is_rejected_by_root(self):
        heldout_root = self.root / "sealed-heldout"
        (heldout_root / "candidates").mkdir(parents=True)
        self.assertEqual(
            self._classify(heldout_root / "candidates", heldout_root=heldout_root),
            "held-out",
        )

    def test_credential_like_path_is_rejected(self):
        for name in ("id_rsa", "hf_token.txt", "aws-credentials.json", "server.pem"):
            with self.subTest(name=name):
                cred = self.evidence_root / name
                cred.write_text("x", encoding="utf-8")
                self.assertEqual(self._classify(cred), "credential-like")

    def test_assert_admissible_raises_excluded_input_error(self):
        with self.assertRaises(ExcludedInputError) as caught:
            frozen_inputs.assert_admissible_input(
                self.evidence_root / "gpu-smoke-baseline-20260829-183311",
                evidence_root=self.evidence_root, recovery_root=self.recovery_root,
            )
        self.assertEqual(caught.exception.reason, "smoke")


if __name__ == "__main__":
    unittest.main()
