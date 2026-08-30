"""Unattended CLI entrypoint for the ADR 0001 Route A training-data build:
download -> filter -> template -> dedup -> hash -> report counts. No manual
intervention step, no `HF_TOKEN`.

    python scripts/build_training_data.py --upstream-root C:\\Projects\\automated_alignment_researcher

Writes `data/training/dataset.jsonl` (the constructed examples, each carrying
its source/generation_rule/category/content_hash) and `data/training/report.json`
(per-category counts, source breakdown, and any shortfall against target).
Both default under the repository's gitignored `data/` directory -- mirroring
`runs/`: reproducible from scratch, never committed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # allow `python scripts/build_training_data.py` without -m

from protocol.validate_manifest import load as load_manifest  # noqa: E402
from training_data.build import run_real_build  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "protocol" / "manifest.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root", required=True,
        help="path to the pinned automated_alignment_researcher checkout",
    )
    parser.add_argument("--work-dir", default=str(REPO_ROOT / "data" / "training" / "_exclusion_pool_scratch"))
    parser.add_argument("--dataset-path", default=str(REPO_ROOT / "data" / "training" / "dataset.jsonl"))
    parser.add_argument("--report-path", default=str(REPO_ROOT / "data" / "training" / "report.json"))
    args = parser.parse_args(argv)

    manifest = load_manifest(MANIFEST_PATH)
    token_cap = int(manifest["training"]["data"]["max_sequence_length"])

    result = run_real_build(
        upstream_root=args.upstream_root, work_dir=args.work_dir,
        dataset_path=args.dataset_path, report_path=args.report_path,
        token_cap=token_cap,
    )
    print(json.dumps(result["report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
