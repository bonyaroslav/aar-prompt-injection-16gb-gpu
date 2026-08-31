"""Shared recovery contract primitives."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


SIGNATURE_FIELDS = (
    "manifest_digest",
    "protocol_version",
    "upstream_commit",
    "upstream_tree",
    "model_revision",
    "seed",
    "stage",
    "epoch",
    "checkpoint_digest",
    "effective_evaluation_config",
    "expected_example_ids",
)


@dataclass(frozen=True)
class StageSignature:
    _canonical_payload: str
    digest: str

    def __init__(self, payload: dict, digest: str):
        canonical_payload = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        object.__setattr__(self, "_canonical_payload", canonical_payload)
        object.__setattr__(self, "digest", digest)

    @property
    def payload(self) -> dict:
        return json.loads(self._canonical_payload)

    @classmethod
    def create(
        cls,
        *,
        manifest_digest,
        protocol_version,
        upstream_commit,
        upstream_tree,
        model_revision,
        seed,
        stage,
        epoch=None,
        checkpoint_digest=None,
        effective_evaluation_config=None,
        expected_example_ids=None,
    ):
        values = locals()
        payload = {key: values[key] for key in SIGNATURE_FIELDS}
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        return cls(payload=payload, digest="sha256:" + hashlib.sha256(encoded).hexdigest())

    def first_difference(self, other):
        payload = self.payload
        other_payload = other.payload
        return next(
            (key for key in SIGNATURE_FIELDS if payload[key] != other_payload[key]),
            None,
        )
