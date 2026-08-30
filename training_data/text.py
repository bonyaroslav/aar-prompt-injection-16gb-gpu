"""Text normalization, hashing, and coarse token-count helpers shared by the
training-data builder (`training_data.build`) and its deduplication pass
(`training_data.dedup`).
"""
from __future__ import annotations

import hashlib
import json
import re

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]")

# Coarse, tokenizer-free proxy for the manifest's `max_sequence_length: 2048`
# construction-time cap. The real trainer re-tokenizes and truncates with the
# model's own tokenizer (`runner.real_training.encode_response_only`); this
# heuristic only keeps obviously oversized examples out of the corpus.
_CHARS_PER_TOKEN = 4


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for near-duplicate matching."""
    collapsed = _WHITESPACE.sub(" ", text.strip().lower())
    return _PUNCTUATION.sub("", collapsed).strip()


def exact_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def near_duplicate_key(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def content_hash(messages: list[dict]) -> str:
    payload = json.dumps(list(messages), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def approximate_token_count(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)
