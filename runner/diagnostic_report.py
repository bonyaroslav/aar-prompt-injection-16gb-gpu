"""Chat-mode MMLU confound test: paired comparison and outcome reading (issue #30).

One pure transform over already-loaded ``metrics.json``-shaped dicts. It pairs
the diagnostic's chat-mode MMLU per-item scores against the frozen Attempt-1
raw-mode per-item scores over the identical 300 example IDs -- baseline against
baseline, each trained epoch against the same epoch -- and reads the result
against issue #30's two-row table.

No model / dataset / scorer / trainer / telemetry / storage dependency and no
I/O: the caller loads the frozen baseline metrics, the nine frozen trained-eval
metrics and the diagnostic run's own metrics, then hands the ``benchmarks["mmlu"]``
dicts here. The paired bootstrap reuses :func:`runner.analysis.paired_bootstrap_ci`
with the *diagnostic* manifest's own seed (never the frozen Attempt-1 seed);
exact McNemar reuses :func:`runner.claim_tables.mcnemar_exact`.
"""
from __future__ import annotations

from runner.analysis import paired_bootstrap_ci
from runner.claim_tables import mcnemar_exact

CHECKPOINT_STATES = tuple(
    f"seed{seed}-epoch{epoch}"
    for seed in (17, 42, 2026)
    for epoch in (1, 2, 3)
)
ALL_STATES = ("baseline",) + CHECKPOINT_STATES


def _item_scores(mmlu_benchmark: dict) -> dict:
    return {example_id: entry["score"] for example_id, entry in mmlu_benchmark["items"].items()}


def _paired(raw_benchmark: dict, chat_benchmark: dict, *, state: str) -> tuple[dict, dict]:
    raw = _item_scores(raw_benchmark)
    chat = _item_scores(chat_benchmark)
    if set(raw) != set(chat):
        raise ValueError(
            f"{state}: chat-mode and Attempt-1 raw-mode MMLU must be scored over "
            "identical example IDs"
        )
    if not raw:
        raise ValueError(f"{state}: no MMLU example IDs to compare")
    return raw, chat


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _compare_state(state: str, raw_benchmark: dict, chat_benchmark: dict, *,
                    seed: int, replicates: int, delta_threshold: float,
                    chance_level: float, chance_band: float) -> dict:
    raw, chat = _paired(raw_benchmark, chat_benchmark, state=state)
    mean_raw, mean_chat = _mean(raw.values()), _mean(chat.values())
    delta = mean_chat - mean_raw

    ci = paired_bootstrap_ci(raw, chat, seed=seed, replicates=replicates)
    ci_crosses_zero = ci["ci_low"] < 0.0 < ci["ci_high"]
    near_chance = abs(mean_chat - chance_level) <= chance_band
    ambiguous = (abs(delta) < delta_threshold and ci_crosses_zero) or near_chance
    classification = "collapse" if delta <= -delta_threshold else "flat_or_up"

    return {
        "state": state,
        "mmlu_attempt1_raw_mode": mean_raw,
        "mmlu_chat_mode": mean_chat,
        "absolute_delta": delta,
        "paired_bootstrap": {
            "observed_difference": ci["observed_difference"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "n": ci["n"],
            "replicates": ci["replicates"],
            "seed": ci["seed"],
            "interval": ci["interval"],
        },
        "mcnemar_exact": mcnemar_exact(raw, chat),
        "classification": classification,
        "ambiguous": ambiguous,
        "near_chance": near_chance,
    }


def _reading(checkpoint_rows: list[dict], baseline_row: dict, reading_table: dict) -> dict:
    collapses = [row["state"] for row in checkpoint_rows if row["classification"] == "collapse"]
    flat_or_up = [row["state"] for row in checkpoint_rows if row["classification"] == "flat_or_up"]
    if not collapses:
        verdict = "flat_or_up_in_chat_mode"
    elif not flat_or_up:
        verdict = "collapses_in_chat_mode"
    else:
        verdict = "mixed"

    baseline_delta = baseline_row["absolute_delta"]
    baseline_collapses = baseline_row["classification"] == "collapse"
    worst_checkpoint_delta = min((row["absolute_delta"] for row in checkpoint_rows), default=0.0)
    # "dwarfed" = the baseline's own chat-mode decline is more than 2x the worst
    # checkpoint's, i.e. the checkpoint declines mostly track a scoring-modality
    # property of this model, not fine-tune-specific damage to the chat pathway.
    baseline_dwarfs_checkpoints = (
        baseline_collapses and abs(baseline_delta) > 2 * abs(worst_checkpoint_delta)
    )

    if baseline_dwarfs_checkpoints and verdict == "collapses_in_chat_mode":
        verdict = "confounded_by_baseline_modality_effect"
        reading = (
            "Every checkpoint declines in chat mode, but the UNTRAINED baseline "
            f"declines far more (delta {baseline_delta:+.4f}, to near chance) than "
            f"the worst checkpoint ({worst_checkpoint_delta:+.4f}). The chat-mode "
            "decline is therefore a property of first-token-logit scoring under the "
            "chat template for this model, not fine-tune-specific damage: the "
            "fine-tune in fact leaves the checkpoints far better at chat-mode MMLU "
            "than the base model. The Attempt-1 finding is not explained away by "
            "the chat wrapper -- the checkpoints keep most of their MMLU ability "
            "in chat mode too, unlike the generation benchmarks."
        )
    else:
        reading = reading_table.get(verdict, (
            "Mixed across checkpoints: some collapse in chat mode and some do not. "
            "Report per-checkpoint; the seed/epoch pattern is the finding."
        ))

    if baseline_collapses:
        control_note = (
            "CONTROL: the untrained baseline model also declines in chat mode "
            f"(delta {baseline_delta:+.4f}). A checkpoint decline that merely tracks "
            "the baseline's is a property of chat-mode likelihood scoring for this "
            "model, not fine-tune-specific damage. Compare each checkpoint's delta "
            "against the baseline's before reading it as interface damage."
        )
    elif verdict not in ("flat_or_up_in_chat_mode",):
        control_note = (
            "CONTROL: the untrained baseline model holds in chat mode "
            f"(delta {baseline_delta:+.4f}), so a checkpoint decline is "
            "fine-tune-specific and points at the chat pathway."
        )
    else:
        control_note = f"CONTROL: baseline chat-mode delta {baseline_delta:+.4f}."

    return {
        "verdict": verdict,
        "reading": reading,
        "baseline_control": control_note,
        "checkpoints_flat_or_up": flat_or_up,
        "checkpoints_collapsed": collapses,
    }


def _build_variant(label, *, diagnostic_manifest, attempt1_baseline_mmlu,
                    attempt1_epoch_mmlu, chatmode_mmlu) -> dict:
    analysis = diagnostic_manifest["analysis"]
    seed = analysis["bootstrap_seed"]
    replicates = analysis["bootstrap_replicates"]
    rule = diagnostic_manifest["ambiguity_rule"]
    threshold = rule["delta_threshold"]
    chance_level = rule.get("chance_level", 0.25)
    chance_band = rule.get("chance_band", 0.03)

    missing = [state for state in ALL_STATES if state not in chatmode_mmlu]
    if missing:
        raise ValueError(f"{label}: chat-mode MMLU missing model states: {', '.join(missing)}")
    missing_epochs = [state for state in CHECKPOINT_STATES if state not in attempt1_epoch_mmlu]
    if missing_epochs:
        raise ValueError(
            f"Attempt-1 trained-eval MMLU missing model states: {', '.join(missing_epochs)}"
        )

    raw_by_state = {"baseline": attempt1_baseline_mmlu, **attempt1_epoch_mmlu}
    rows = [
        _compare_state(
            state, raw_by_state[state], chatmode_mmlu[state],
            seed=seed, replicates=replicates, delta_threshold=threshold,
            chance_level=chance_level, chance_band=chance_band,
        )
        for state in ALL_STATES
    ]
    by_state = {row["state"]: row for row in rows}
    checkpoint_rows = [by_state[state] for state in CHECKPOINT_STATES]

    return {
        "candidate_strings": label,
        "per_state": rows,
        "baseline_row": by_state["baseline"],
        "reading": _reading(checkpoint_rows, by_state["baseline"], diagnostic_manifest["reading_table"]),
        "ambiguous_states": [row["state"] for row in rows if row["ambiguous"]],
        "near_chance_states": [row["state"] for row in rows if row["near_chance"]],
    }


def build_chatmode_report(diagnostic_manifest: dict, *, attempt1_baseline_mmlu: dict,
                           attempt1_epoch_mmlu: dict, chatmode_mmlu: dict,
                           chatmode_mmlu_no_leading_space: dict | None = None) -> dict:
    """Build the full chat-mode MMLU confound-test report.

    ``attempt1_baseline_mmlu`` and each value of ``attempt1_epoch_mmlu`` (keyed by
    ``"seed<seed>-epoch<epoch>"``) are frozen Attempt-1 ``benchmarks["mmlu"]``
    dicts, scored in raw-completion mode. ``chatmode_mmlu`` is the diagnostic run's
    per-state MMLU output in the same shape, keyed by the ten model states
    (``"baseline"`` plus the nine checkpoints), scored with the chat template on.

    When the primary run leaves one or more states ambiguous the caller performs
    the no-leading-space robustness re-run and passes ``chatmode_mmlu_no_leading_space``;
    both tables are then reported side by side.
    """
    primary = _build_variant(
        "primary ([' A',' B',' C',' D'])", diagnostic_manifest=diagnostic_manifest,
        attempt1_baseline_mmlu=attempt1_baseline_mmlu,
        attempt1_epoch_mmlu=attempt1_epoch_mmlu, chatmode_mmlu=chatmode_mmlu,
    )

    robustness_required = bool(primary["ambiguous_states"])
    report = {
        "diagnostic_version": diagnostic_manifest["diagnostic_version"],
        "downstream_of": diagnostic_manifest["downstream_of"]["protocol_version"],
        "analysis_unit": (
            "the frozen baseline and every prespecified epoch of every completed "
            "run (three seeds x three epochs); no post-hoc winner"
        ),
        "tokenization_caveat": diagnostic_manifest["ambiguity_rule"]["robustness_rerun"]["rationale"],
        "robustness_rerun_required": robustness_required,
        "primary": primary,
    }

    if chatmode_mmlu_no_leading_space is not None:
        report["no_leading_space"] = _build_variant(
            "no-leading-space (['A','B','C','D'])",
            diagnostic_manifest=diagnostic_manifest,
            attempt1_baseline_mmlu=attempt1_baseline_mmlu,
            attempt1_epoch_mmlu=attempt1_epoch_mmlu,
            chatmode_mmlu=chatmode_mmlu_no_leading_space,
        )
        report["robustness_rerun_performed"] = True
    else:
        report["robustness_rerun_performed"] = False
        if robustness_required:
            report["robustness_rerun_note"] = (
                "Primary run left states ambiguous: "
                f"{', '.join(primary['ambiguous_states'])}. The no-leading-space "
                "re-run must be performed and passed as chatmode_mmlu_no_leading_space."
            )

    return report


def render_report(report: dict) -> str:
    """Canonical text form -- byte-identical for byte-identical inputs."""
    import json

    return json.dumps(report, indent=2, sort_keys=True) + "\n"


# --- operator glue (file loading; the transform above stays pure) ------


def _load_mmlu(metrics_path) -> dict:
    import json
    from pathlib import Path

    return json.loads(Path(metrics_path).read_text(encoding="utf-8"))["benchmarks"]["mmlu"]


def _load_chatmode_states(metrics_path) -> dict:
    import json
    from pathlib import Path

    doc = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    return {
        state: payload["benchmarks"]["mmlu"]
        for state, payload in doc["model_states"].items()
    }


def main(argv=None) -> int:
    import argparse
    import json
    from pathlib import Path

    from protocol.diagnostic.manifest import load as load_diagnostic_manifest

    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--diagnostic-manifest", type=Path,
                        default=repo_root / "protocol" / "diagnostic" / "chatmode-mmlu-2026-09-02.json")
    parser.add_argument("--baseline-metrics", type=Path,
                        default=repo_root / "runs" / "real-baseline-20260829-205020" / "metrics.json")
    parser.add_argument("--eval-glob-root", type=Path, default=repo_root / "runs")
    parser.add_argument("--chatmode-metrics", type=Path, required=True,
                        help="diagnostics/chatmode-mmlu-<stamp>/metrics.json (primary run)")
    parser.add_argument("--chatmode-metrics-no-leading-space", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    diagnostic = load_diagnostic_manifest(args.diagnostic_manifest)
    epoch_mmlu = {}
    for seed in (17, 42, 2026):
        for epoch in (1, 2, 3):
            matches = sorted(args.eval_glob_root.glob(f"eval-seed{seed}-epoch{epoch}-*"))
            if not matches:
                raise SystemExit(f"missing Attempt-1 eval bundle for seed{seed}-epoch{epoch}")
            epoch_mmlu[f"seed{seed}-epoch{epoch}"] = _load_mmlu(matches[0] / "metrics.json")

    report = build_chatmode_report(
        diagnostic,
        attempt1_baseline_mmlu=_load_mmlu(args.baseline_metrics),
        attempt1_epoch_mmlu=epoch_mmlu,
        chatmode_mmlu=_load_chatmode_states(args.chatmode_metrics),
        chatmode_mmlu_no_leading_space=(
            _load_chatmode_states(args.chatmode_metrics_no_leading_space)
            if args.chatmode_metrics_no_leading_space else None
        ),
    )
    rendered = render_report(report)
    if args.out is not None:
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
