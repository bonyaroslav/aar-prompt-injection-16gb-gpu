"""Offline provenance receipts and claim-language gates for publication reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path


class PublicationGateError(ValueError):
    """Base class for a publication gate failure."""


class ProvenanceGateError(PublicationGateError):
    """A report number or source cannot be traced to approved provenance."""


class ClaimLanguageError(PublicationGateError):
    """A report uses forbidden or insufficiently qualified claim language."""


_FORBIDDEN_LANGUAGE = (
    ("robust", re.compile(r"\brobust\b", re.IGNORECASE)),
    ("secure", re.compile(r"\bsecure\b", re.IGNORECASE)),
    ("resistant", re.compile(r"\bresistant\b", re.IGNORECASE)),
    ("mitigation", re.compile(r"\bmitigation\b", re.IGNORECASE)),
    ("defense that works", re.compile(r"\bdefense\s+that\s+works\b", re.IGNORECASE)),
)
_CAPABILITY_ACTION = re.compile(
    r"\b(?:preserv(?:ed|es|ing)?|retain(?:ed|s|ing)?|maintain(?:ed|s|ing)?|"
    r"improv(?:ed|es|ing)?|declin(?:ed|es|ing)?|collaps(?:ed|es|ing)?|"
    r"fail(?:ed|s|ing)?)\b",
    re.IGNORECASE,
)
_MODALITY = re.compile(
    r"\b(?:free_generation_sampled_string_scored|likelihood_ranked_no_generation|"
    r"free-generation|generation-scored|likelihood-ranked|log-likelihood)\b",
    re.IGNORECASE,
)


_PROTOCOL_DIGEST_MEANING = (
    "canonical-JSON content SHA-256 of protocol/manifest.json (sorted keys, "
    "compact separators, UTF-8); invariant to git checkout settings, per "
    "protocol/digests.md and ticket #25. Not the raw-file SHA-256."
)


def _canonical_digest(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _bootstrap_parameters(protocol_manifest: dict | None) -> dict | None:
    if protocol_manifest is None:
        return None
    analysis = protocol_manifest.get("analysis", {})
    return {
        "replicates": analysis.get("bootstrap_replicates"),
        "seed": analysis.get("bootstrap_seed"),
        "interval": analysis.get("interval"),
        "source": "protocol/manifest.json analysis",
    }


def _text_digest(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_corpus_supplement(dataset_path: Path, report_path: Path) -> dict:
    """Return the content-only corpus provenance source approved for issue #32."""
    rows = [
        json.loads(line)
        for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return {
        "role": "training_corpus_digest_only_supplement",
        "digest_kind": "canonical_json",
        "digest": "sha256:" + _canonical_digest({"rows": rows, "report": report}),
    }


def build_baseline_resource_supplement(path: Path) -> dict:
    """Return the canonical digest-only source for baseline resource figures."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "role": "baseline_resource_digest_only_supplement",
        "digest_kind": "canonical_json",
        "digest": "sha256:" + _canonical_digest(value),
    }


def build_power_notes_supplement(path: Path) -> dict:
    """Return the LF-normalized digest-only source for published MDE figures."""
    text = Path(path).read_text(encoding="utf-8")
    return {
        "role": "power_notes_digest_only_supplement",
        "digest_kind": "lf_normalized_text",
        "digest": "sha256:" + _text_digest(text),
    }


def _numeric_leaves(value: object, location: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        yield location or "/", repr(value)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from _numeric_leaves(child, f"{location}/{index}")
        return
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _numeric_leaves(value[key], f"{location}/{key}")


def _string_leaves(value: object, location: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield location or "/", value
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from _string_leaves(child, f"{location}/{index}")
        return
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _string_leaves(value[key], f"{location}/{key}")


def _input_digests(frozen_input_record: dict, supplemental_sources: list[dict]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for row in [*frozen_input_record.get("inputs", []), *supplemental_sources]:
        role, digest = row.get("role"), row.get("digest")
        if not isinstance(role, str) or not isinstance(digest, str):
            raise ProvenanceGateError("provenance source needs string role and digest")
        if role in resolved:
            raise ProvenanceGateError(f"duplicate provenance source role: {role}")
        resolved[role] = digest
    return resolved


def _receipt(*, report_id: str, section_kind: str, section_id: str,
             location: str, value: str, input_digests: list[str]) -> str:
    payload = {
        "report_id": report_id,
        "section_kind": section_kind,
        "section_id": section_id,
        "location": location,
        "value": value,
        "input_digests": sorted(input_digests),
    }
    return "sha256:" + _canonical_digest(payload)


def _manifest_reports(*, reports: list[dict], sources: dict[str, str]) -> list[dict]:
    output: list[dict] = []
    for report in reports:
        report_id = report.get("report_id")
        if not isinstance(report_id, str):
            raise ProvenanceGateError("report needs a string report_id")
        sections = report.get("sections")
        if not isinstance(sections, list):
            raise ProvenanceGateError(f"report {report_id} needs a sections list")

        rendered_sections: list[dict] = []
        for section in sections:
            kind, section_id = section.get("kind"), section.get("id")
            if kind not in {"table", "figure"} or not isinstance(section_id, str):
                raise ProvenanceGateError(
                    f"report {report_id} section needs table/figure kind and string id"
                )
            roles = section.get("source_roles")
            if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
                raise ProvenanceGateError(f"report {report_id}/{section_id} needs source_roles")
            missing = [role for role in roles if role not in sources]
            if missing:
                raise ProvenanceGateError(
                    f"report {report_id}/{section_id} names unknown source role: {missing[0]}"
                )
            digests = sorted({sources[role] for role in roles})
            numbers = [
                {
                    "location": location,
                    "value": value,
                    "receipt": _receipt(
                        report_id=report_id, section_kind=kind, section_id=section_id,
                        location=location, value=value, input_digests=digests,
                    ),
                }
                for location, value in _numeric_leaves(section.get("content"))
            ]
            rendered_sections.append({
                "kind": kind,
                "id": section_id,
                "input_digests": digests,
                "numbers": numbers,
            })
        output.append({"report_id": report_id, "sections": rendered_sections})
    return output


def build_provenance_manifest(*, frozen_input_record: dict, reports: list[dict],
                              supplemental_sources: list[dict],
                              protocol_manifest: dict | None = None) -> dict:
    """Build an immutable receipt manifest for structured report numbers.

    When ``protocol_manifest`` is supplied its canonical digest is cross-checked
    against the frozen input record and its bootstrap parameters are recorded, so
    the manifest maps every table to the protocol digest, the analysis version
    and the bootstrap parameters (issue #32 acceptance criteria).
    """
    sources = _input_digests(frozen_input_record, supplemental_sources)
    protocol_digest = frozen_input_record.get("protocol_manifest_digest")
    analysis_version = frozen_input_record.get("analysis_version")
    if not isinstance(protocol_digest, str) or not isinstance(analysis_version, str):
        raise ProvenanceGateError("frozen input record lacks protocol digest or analysis version")
    if protocol_manifest is not None:
        recomputed = _canonical_digest(protocol_manifest)
        if recomputed != protocol_digest:
            raise ProvenanceGateError(
                "protocol manifest canonical digest does not match the frozen input record"
            )
    return {
        "schema_version": "publication-provenance-2",
        "frozen_input_record_digest": "sha256:" + _canonical_digest(frozen_input_record),
        "protocol_manifest_digest": protocol_digest,
        "protocol_manifest_digest_kind": "canonical_json",
        "protocol_manifest_digest_meaning": _PROTOCOL_DIGEST_MEANING,
        "analysis_version": analysis_version,
        "bootstrap_parameters": _bootstrap_parameters(protocol_manifest),
        "supplemental_sources": sorted(supplemental_sources, key=lambda row: row["role"]),
        "reports": _manifest_reports(reports=reports, sources=sources),
    }


def _number_index(manifest: dict) -> dict[tuple[str, str, str, str], dict]:
    indexed: dict[tuple[str, str, str, str], dict] = {}
    for report in manifest.get("reports", []):
        for section in report.get("sections", []):
            for number in section.get("numbers", []):
                key = (report["report_id"], section["kind"], section["id"], number["location"])
                indexed[key] = number
    return indexed


def verify_provenance(*, provenance_manifest: dict, frozen_input_record: dict,
                      reports: list[dict], supplemental_sources: list[dict],
                      protocol_manifest: dict | None = None) -> None:
    """Fail closed when a submitted report differs from its provenance receipts."""
    expected_record_digest = "sha256:" + _canonical_digest(frozen_input_record)
    if provenance_manifest.get("frozen_input_record_digest") != expected_record_digest:
        raise ProvenanceGateError("frozen input record digest no longer matches provenance manifest")

    current = build_provenance_manifest(
        frozen_input_record=frozen_input_record, reports=reports,
        supplemental_sources=supplemental_sources, protocol_manifest=protocol_manifest,
    )
    for field in ("protocol_manifest_digest", "analysis_version", "bootstrap_parameters"):
        if provenance_manifest.get(field) != current[field]:
            raise ProvenanceGateError(
                f"provenance manifest {field} no longer matches the frozen inputs"
            )
    recorded = _number_index(provenance_manifest)
    for key, number in _number_index(current).items():
        prior = recorded.get(key)
        if prior is None or prior.get("receipt") != number["receipt"]:
            report_id, _kind, section_id, location = key
            raise ProvenanceGateError(
                f"orphan value {number['value']} at {report_id}/{section_id}{location}"
            )
    if set(recorded) != set(_number_index(current)):
        raise ProvenanceGateError("provenance manifest has a number absent from the submitted reports")


def check_claim_language(reports: list[dict]) -> None:
    """Reject forbidden efficacy wording and unqualified capability claims."""
    for report in reports:
        report_id = report.get("report_id", "<unknown-report>")
        for section in report.get("sections", []):
            section_id = section.get("id", "<unknown-section>")
            for location, text in _string_leaves(section.get("content")):
                for label, pattern in _FORBIDDEN_LANGUAGE:
                    if pattern.search(text):
                        raise ClaimLanguageError(
                            f"forbidden efficacy language '{label}' at "
                            f"{report_id}/{section_id}{location}"
                        )
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                    if (
                        re.search(r"\bcapability\b", sentence, re.IGNORECASE)
                        and _CAPABILITY_ACTION.search(sentence)
                        and not _MODALITY.search(sentence)
                    ):
                        raise ClaimLanguageError(
                            "capability claim must name its evaluation modality at "
                            f"{report_id}/{section_id}{location}"
                        )


def run_gates(*, provenance_manifest: dict, frozen_input_record: dict,
              reports: list[dict], supplemental_sources: list[dict],
              protocol_manifest: dict | None = None) -> dict:
    """Run both publication gates and return the verified provenance manifest."""
    verify_provenance(
        provenance_manifest=provenance_manifest,
        frozen_input_record=frozen_input_record,
        reports=reports,
        supplemental_sources=supplemental_sources,
        protocol_manifest=protocol_manifest,
    )
    check_claim_language(reports)
    return provenance_manifest


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _supplements_from_args(args) -> list[dict]:
    pairs = (
        (args.corpus_dataset, args.corpus_report, build_corpus_supplement),
    )
    supplements: list[dict] = []
    for first, second, builder in pairs:
        if (first is None) != (second is None):
            raise ProvenanceGateError("--corpus-dataset and --corpus-report must be supplied together")
        if first is not None:
            supplements.append(builder(first, second))
    if args.baseline_resource is not None:
        supplements.append(build_baseline_resource_supplement(args.baseline_resource))
    if args.power_notes is not None:
        supplements.append(build_power_notes_supplement(args.power_notes))
    return supplements


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--frozen-input", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument(
        "--protocol-manifest", type=Path, default=Path("protocol/manifest.json")
    )
    parser.add_argument("--corpus-dataset", type=Path)
    parser.add_argument("--corpus-report", type=Path)
    parser.add_argument("--baseline-resource", type=Path)
    parser.add_argument("--power-notes", type=Path)
    args = parser.parse_args(argv)

    try:
        protocol_manifest = (
            _load_json(args.protocol_manifest)
            if args.protocol_manifest is not None and Path(args.protocol_manifest).is_file()
            else None
        )
        run_gates(
            provenance_manifest=_load_json(args.provenance),
            frozen_input_record=_load_json(args.frozen_input),
            reports=_load_json(args.reports),
            supplemental_sources=_supplements_from_args(args),
            protocol_manifest=protocol_manifest,
        )
    except (OSError, json.JSONDecodeError, PublicationGateError) as error:
        parser.exit(status=2, message=f"publication gate failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
