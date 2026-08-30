"""ADR 0001's binding exclusion pool: everything a constructed training example
must not exactly-or-near-duplicate.

Two families of source, per the ADR:

1. The published visible-eval subsets (`open_prompt_injection`,
   `tensor_trust_hijack`, `tensor_trust_extract`, `mmlu`, `gsm8k`, `ifeval`) --
   materialized with the exact same pinned-upstream publisher functions and
   `protocol/manifest.json` sample counts/seed the real baseline used, via
   `runner.gpu_smoke._import_publisher`.
2. The full upstream pools those eval subsets were sampled from --
   `HumanCompatibleAI/tensor-trust-data` (both hijacking and extraction, not
   just the ~300 selected rows), and the sst2/sms_spam/hsol splits
   `open_prompt_injection` draws from -- so unselected rows from the same
   distribution aren't quietly reused either.

InjecAgent is never touched here: it has no bearing on training-data
construction or dedup per the ADR, and this module never imports the
`heldout_dir` path.

`pool_keys_from_texts` (in `training_data.dedup`) is what turns any of this
into exact/near-duplicate key sets; the two `fetch_*` functions below only
gather the raw text, so they're the only functions in this module that touch
the network -- callers that want an offline/deterministic test seam should
call `collect_eval_texts` / `collect_full_pool_texts` directly with fake rows.
"""
from __future__ import annotations

from pathlib import Path

# benchmark -> function extracting every dedup-relevant string from one published row
_EVAL_TEXT_FIELDS = {
    "open_prompt_injection": lambda row: [row["attacked_prompt"], row["pnai_prompt"]],
    "tensor_trust_hijack": lambda row: [row["pre_prompt"], row["attack"], row["post_prompt"], row["access_code"]],
    "tensor_trust_extract": lambda row: [row["pre_prompt"], row["attack"], row["post_prompt"], row["access_code"]],
    "mmlu": lambda row: [row["question"], *row["choices"]],
    "gsm8k": lambda row: [row["prompt"]],
    "ifeval": lambda row: [row["prompt"]],
}

# protocol/manifest.json's frozen sample counts + seed for the visible-eval subsets.
_EVAL_SAMPLE_COUNTS = {
    "open_prompt_injection": 300,
    "tensor_trust_hijack": 300,
    "tensor_trust_extract": 300,
    "mmlu": 300,
    "gsm8k": 200,
    "ifeval": 200,
}
_EVAL_SEED = 42


def collect_eval_texts(published_rows: dict[str, list[dict]]) -> list[str]:
    """Flatten dedup-relevant text out of published eval rows keyed by benchmark name."""
    texts: list[str] = []
    for benchmark, rows in published_rows.items():
        extract = _EVAL_TEXT_FIELDS[benchmark]
        for row in rows:
            texts.extend(str(value) for value in extract(row))
    return texts


def collect_full_pool_texts(full_pool_rows: dict[str, list[str]]) -> list[str]:
    """Flatten the full-pool exclusion text lists (already extracted to plain strings)."""
    texts: list[str] = []
    for values in full_pool_rows.values():
        texts.extend(str(value) for value in values if value)
    return texts


def fetch_published_eval_rows(upstream_root: str | Path, work_dir: str | Path) -> dict[str, list[dict]]:
    """Materialize the six published visible-eval subsets via the pinned upstream
    publisher, at the manifest's frozen counts/seed, and read them back as rows.

    Reuses `runner.gpu_smoke`'s upstream-import and JSONL-reading helpers rather
    than re-implementing eval-set construction, so this can never silently drift
    from what the real baseline actually evaluated against.
    """
    from runner.gpu_smoke import _import_publisher
    from runner.real_adapters import _read_jsonl

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    publisher = _import_publisher(Path(upstream_root))
    publish_fn = {
        "open_prompt_injection": publisher._publish_open_prompt_injection,
        "tensor_trust_hijack": publisher._publish_tensor_trust_hijack,
        "tensor_trust_extract": publisher._publish_tensor_trust_extract,
        "mmlu": publisher._publish_mmlu,
        "gsm8k": publisher._publish_gsm8k,
        "ifeval": publisher._publish_ifeval,
    }
    rows: dict[str, list[dict]] = {}
    for benchmark, publish in publish_fn.items():
        target = work_dir / f"{benchmark}.jsonl"
        if not target.exists():
            publish(work_dir, n=_EVAL_SAMPLE_COUNTS[benchmark], seed=_EVAL_SEED)
        rows[benchmark] = _read_jsonl(target)
    return rows


def fetch_full_pool_texts(upstream_root: str | Path) -> dict[str, list[str]]:
    """Fetch the full upstream pools the visible-eval subsets were sampled from,
    unsliced -- so unselected rows from the same distribution are excluded too.
    """
    from runner.gpu_smoke import _import_publisher
    from datasets import load_dataset

    publisher = _import_publisher(Path(upstream_root))
    hijack_rows = publisher._fetch_tensor_trust("hijacking")
    extract_rows = publisher._fetch_tensor_trust("extraction")
    tt_fields = ("pre_prompt", "attack", "post_prompt", "access_code")
    sst2 = load_dataset("stanfordnlp/sst2", split="validation")
    sms_spam = load_dataset("ucirvine/sms_spam", split="train")
    hsol = load_dataset("tdavidson/hate_speech_offensive", split="train")
    return {
        "tensor_trust_data_hijacking_full": [
            str(row.get(field, "")) for row in hijack_rows for field in tt_fields
        ],
        "tensor_trust_data_extraction_full": [
            str(row.get(field, "")) for row in extract_rows for field in tt_fields
        ],
        "sst2_validation_full": [row["sentence"] for row in sst2],
        "sms_spam_train_full": [row["sms"] for row in sms_spam],
        "hsol_train_full": [row["tweet"] for row in hsol],
    }
