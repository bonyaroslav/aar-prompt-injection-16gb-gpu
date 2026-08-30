"""Construction-source fetchers for ADR 0001 Route A.

Every function here touches only non-gated Hugging Face datasets and needs no
`HF_TOKEN` -- confirmed by loading each with an unauthenticated client. Network
access happens only inside these functions, so `training_data.templates` and
`training_data.build` can be exercised offline against fake rows shaped like
these functions' return values.
"""
from __future__ import annotations

DEEPSET_SOURCE = "deepset/prompt-injections"
GANDALF_SOURCE = "Lakera/gandalf_ignore_instructions"
DOLLY_SOURCE = "databricks/databricks-dolly-15k"


def load_deepset_prompt_injection_attacks() -> list[dict]:
    """The label==1 (injection) rows of deepset/prompt-injections, across both splits."""
    from datasets import load_dataset

    dataset = load_dataset(DEEPSET_SOURCE)
    rows = []
    for split in ("train", "test"):
        for row in dataset[split]:
            if row["label"] == 1:
                rows.append({"text": row["text"], "source": DEEPSET_SOURCE, "split": split})
    return rows


def load_gandalf_ignore_instructions() -> list[dict]:
    """Every row of Lakera/gandalf_ignore_instructions (all rows are injection attempts)."""
    from datasets import load_dataset

    dataset = load_dataset(GANDALF_SOURCE)
    rows = []
    for split in ("train", "validation", "test"):
        for row in dataset[split]:
            rows.append({"text": row["text"], "source": GANDALF_SOURCE, "split": split})
    return rows


def load_dolly_rows() -> list[dict]:
    """All 15,011 rows of databricks-dolly-15k (single `train` split)."""
    from datasets import load_dataset

    dataset = load_dataset(DOLLY_SOURCE)["train"]
    return [
        {
            "instruction": row["instruction"],
            "context": row["context"],
            "response": row["response"],
            "category": row["category"],
            "source": DOLLY_SOURCE,
        }
        for row in dataset
    ]
