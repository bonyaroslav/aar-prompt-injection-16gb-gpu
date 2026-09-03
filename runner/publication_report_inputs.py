"""Read-only registration of the current analysis outputs for the #32 gates.

`runner.claim_tables.build_claim_report` (#28) and
`runner.integrity_report.build_integrity_report` (#29) are pure transforms that
already exist and are not touched here. This module only *declares* their
publication sections and the provenance source roles each section rests on, in
the ``{"report_id", "sections": [{"kind", "id", "source_roles", "content"}]}``
shape that :mod:`runner.publication_gates` consumes.

It performs no model, dataset, scorer, trainer, telemetry, storage or held-out
operation: it accepts the two already-rendered report dicts and returns a list.
"""
from __future__ import annotations

CLAIM_REPORT_ID = "claim_tables"
INTEGRITY_REPORT_ID = "integrity_report"

# The three digest-only supplements approved in
# docs/issue-32-provenance-source-decision.md. Ordinary roles come from the #27
# frozen input record; these three do not expand that allowlist.
TRAINING_CORPUS_SUPPLEMENT = "training_corpus_digest_only_supplement"
BASELINE_RESOURCE_SUPPLEMENT = "baseline_resource_digest_only_supplement"
POWER_NOTES_SUPPLEMENT = "power_notes_digest_only_supplement"

# Section id -> the claim-table sub-report key it wraps. Every claim-table
# section rests on the protocol manifest (it fixes the analysis unit, the
# capability gates and the bootstrap parameters), the frozen baseline bundle and
# every per-epoch evaluation bundle.
_CLAIM_SECTIONS = (
    "primary_table",
    "paired_bootstrap",
    "mcnemar_exact",
    "visible_composite",
    "cross_run_summary",
)


def _eval_roles(seeds, epochs):
    return [
        f"seed{seed}_evaluation_bundle_epoch{epoch}"
        for seed in seeds
        for epoch in range(1, epochs + 1)
    ]


def _seed_roles(seeds, suffix):
    return [f"seed{seed}_{suffix}" for seed in seeds]


def _section(kind, section_id, source_roles, content):
    return {
        "kind": kind,
        "id": section_id,
        "source_roles": list(dict.fromkeys(source_roles)),
        "content": content,
    }


def _completed_seeds(*reports):
    for report in reports:
        seeds = report.get("completed_seeds")
        if seeds:
            return list(seeds)
    return []


def _epochs_per_seed(*reports):
    for report in reports:
        epochs = report.get("epochs_per_seed")
        if epochs:
            return int(epochs)
    return 0


def register_current_reports(*, claim_report: dict, integrity_report: dict) -> list[dict]:
    """Return the publication-section descriptors for the #28 and #29 reports."""
    seeds = _completed_seeds(claim_report, integrity_report)
    epochs = _epochs_per_seed(claim_report, integrity_report)

    measurement_roles = (
        ["protocol_manifest", "baseline_bundle"]
        + _eval_roles(seeds, epochs)
    )
    governance_roles = (
        ["protocol_manifest"]
        + _seed_roles(seeds, "outcomes_summary")
        + _seed_roles(seeds, "continuation_decision")
    )

    claim_sections = [
        _section("table", section_id, measurement_roles, claim_report.get(section_id, {}))
        for section_id in _CLAIM_SECTIONS
    ]

    failure_modes = integrity_report.get("failure_mode_evidence", {})
    records = integrity_report.get("integrity_records", {})

    integrity_sections = [
        _section(
            "table", "generation_failure_signature", measurement_roles,
            failure_modes.get("generation_failure_signature", {}),
        ),
        _section(
            "table", "tensor_trust_degeneracy", measurement_roles,
            failure_modes.get("tensor_trust_degeneracy", {}),
        ),
        _section(
            "table", "utility_control_arm_comparison", measurement_roles,
            failure_modes.get("utility_control_arm_comparison", {}),
        ),
        _section(
            "table", "corpus_nutrition_label",
            ["protocol_manifest", TRAINING_CORPUS_SUPPLEMENT],
            failure_modes.get("corpus_nutrition_label", {}),
        ),
        _section(
            "table", "held_out_disposition",
            ["protocol_manifest", "baseline_bundle", POWER_NOTES_SUPPLEMENT]
            + _seed_roles(seeds, "selection_record"),
            records.get("held_out_disposition", {}),
        ),
        _section(
            "table", "reproducibility_disclosure", governance_roles,
            records.get("reproducibility_disclosure", {}),
        ),
        _section(
            "table", "resource_accounting",
            ["protocol_manifest", BASELINE_RESOURCE_SUPPLEMENT]
            + _seed_roles(seeds, "resource_comparison"),
            records.get("resource_accounting", {}),
        ),
        _section(
            "table", "sample_count_convention", ["protocol_manifest"],
            records.get("sample_count_convention", {}),
        ),
    ]

    return [
        {"report_id": CLAIM_REPORT_ID, "sections": claim_sections},
        {"report_id": INTEGRITY_REPORT_ID, "sections": integrity_sections},
    ]
