"""The one record shape every category builder in `training_data.templates`
produces, and the one shape `training_data.build` writes to the output JSONL.

Every example is traceable per the issue's acceptance criteria: `source` names
the upstream dataset (or "template" for own-construction categories),
`generation_rule` names the deterministic rule that produced it, `category` is
one of the manifest's four training-data categories, and `content_hash` is a
stable hash of the rendered chat messages.
"""
from __future__ import annotations

import dataclasses

from training_data.text import approximate_token_count, content_hash

CATEGORIES = ("prompt_injection", "clean_control", "ambiguous_boundary", "refusal_calibration")


@dataclasses.dataclass
class TrainingExample:
    messages: list[dict]
    category: str
    source: str
    generation_rule: str

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown training-data category: {self.category!r}")
        if len(self.messages) < 2 or self.messages[-1].get("role") != "assistant":
            raise ValueError("training example must end with an assistant response")

    def dedup_text(self) -> str:
        """The text checked against the deduplicator -- every message concatenated,
        so a duplicate user turn under a different assistant response still counts."""
        return "\n".join(f"{message['role']}:{message['content']}" for message in self.messages)

    def approx_tokens(self) -> int:
        return approximate_token_count(self.dedup_text())

    def content_hash(self) -> str:
        return content_hash(self.messages)

    def to_record(self) -> dict:
        return {
            "messages": self.messages,
            "category": self.category,
            "source": self.source,
            "generation_rule": self.generation_rule,
            "content_hash": self.content_hash(),
        }
