"""Exact + normalized-near-duplicate checking.

Two kinds of collisions matter here, per ADR 0001 and the issue's acceptance
criteria:

- Against ADR 0001's binding exclusion pools (`training_data.exclusion_pool`):
  the published visible-eval subsets plus the full upstream pools those subsets
  were drawn from (e.g. all of Tensor Trust hijacking/extraction, not just the
  300 selected rows).
- Within the constructed corpus itself, so the final 5,000 examples don't
  quietly repeat the same underlying text under two categories or two
  generation rules.
"""
from __future__ import annotations

from training_data.text import exact_key, near_duplicate_key


class Deduplicator:
    def __init__(self, exclude_exact=(), exclude_near=()):
        self._excluded_exact = set(exclude_exact)
        self._excluded_near = set(exclude_near)
        self._seen_exact: set[str] = set()
        self._seen_near: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        exact, near = exact_key(text), near_duplicate_key(text)
        return (
            exact in self._excluded_exact
            or near in self._excluded_near
            or exact in self._seen_exact
            or near in self._seen_near
        )

    def add(self, text: str) -> None:
        self._seen_exact.add(exact_key(text))
        self._seen_near.add(near_duplicate_key(text))

    def accept(self, text: str) -> bool:
        """Check-and-add in one step; returns True iff `text` was accepted (not a duplicate)."""
        if self.is_duplicate(text):
            return False
        self.add(text)
        return True


def pool_keys(texts) -> tuple[set[str], set[str]]:
    """Build the (exact, near-duplicate) key sets for a flat list of exclusion-pool texts."""
    exact = {exact_key(text) for text in texts if text}
    near = {near_duplicate_key(text) for text in texts if text}
    return exact, near
