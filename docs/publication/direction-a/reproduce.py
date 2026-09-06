"""Recalculate Direction A from the two supplied inputs, without the research repo.

Python standard library suffices for tables. --plot additionally uses Matplotlib.
--verify-bootstrap repeats the 54 historical paired intervals (CPU only).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

LABELS = {
    "open_prompt_injection": "OPI",
    "tensor_trust_hijack": "Tensor Trust: hijack",
    "tensor_trust_extract": "Tensor Trust: extraction",
    "gsm8k": "GSM8K", "ifeval": "IFEval", "mmlu": "MMLU",
}
CAPABILITY = ("gsm8k", "ifeval", "mmlu")
VISIBLE = ("open_prompt_injection", "tensor_trust_hijack", "tensor_trust_extract")


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Mismatch for {label}: {actual} != {expected}")


def load_inputs(folder):
    context = json.loads((folder / "context.json").read_text(encoding="utf-8"))
    raw = (folder / "visible-scores.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != context["visible_scores_sha256"]:
        raise ValueError("visible-scores.csv no longer matches its recorded hash")
    scores = defaultdict(dict)
    states = {}
    count = 0
    with (folder / "visible-scores.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            state, name, example = row["state"], row["benchmark"], row["example_id"]
            seed = int(row["nominal_seed"]) if row["nominal_seed"] else None
            metadata = (seed, int(row["epoch"]))
            if state in states and states[state] != metadata:
                raise ValueError("Inconsistent state metadata")
            states[state] = metadata
            key = (state, name)
            if example in scores[key]:
                raise ValueError("Duplicate example ID in a state/benchmark")
            value = float(row["score"])
            allowed = (0.0, .5, 1.0) if name.startswith("tensor_trust") else (0.0, 1.0)
            if name not in context["benchmarks"] or value not in allowed:
                raise ValueError("Unexpected benchmark or score")
            scores[key][example] = value
            count += 1
    expected_states = {"baseline": (None, 0)} | {
        f"run{seed}_epoch{epoch}": (seed, epoch)
        for seed in context["seeds"] for epoch in range(1, context["epochs"] + 1)
    }
    if states != expected_states or count != context["score_rows"]:
        raise ValueError("Incomplete state topology")
    for state in states:
        for name in context["benchmarks"]:
            base = scores["baseline", name]
            current = scores[state, name]
            if len(current) != context["sample_counts"][name] or current.keys() != base.keys():
                raise ValueError("Every state must contain the same fixed example IDs")
    return context, scores, states


def summarize(context, scores, states):
    means = {key: sum(values.values()) / len(values) for key, values in scores.items()}
    for expected in context["recorded_scores"]:
        state = f"run{expected['run_seed']}_epoch{expected['epoch']}"
        name = expected["benchmark"]
        close(means[state, name], expected["trained"], f"{state}/{name}")
        close(means["baseline", name], expected["baseline"], f"baseline/{name}")
    rules = context["selection"]["capability_gates"]
    rows = []
    for state, (seed, epoch) in states.items():
        if state == "baseline":
            continue
        losses = {name: means["baseline", name] - means[state, name] for name in CAPABILITY}
        retention = sum(means[state, name] / means["baseline", name] for name in CAPABILITY) / 3
        composite = sum(means[state, name] - means["baseline", name] for name in VISIBLE) / 3
        reasons = [name for name in CAPABILITY if losses[name] > rules[f"{name}_max_decline"]]
        if retention < rules["mean_normalized_retention_min"]:
            reasons.append("mean_normalized_retention")
        rows.append({"state": state, "nominal_seed": seed, "epoch": epoch,
                     **{f"{name}_loss_pp": losses[name] * 100 for name in CAPABILITY},
                     "mean_normalized_retention": retention, "visible_composite_delta_pp": composite * 100,
                     "eligible": not reasons,
                     "meaningful_visible_gain_with_gates": not reasons and composite >= context["selection"]["meaningful_improvement_absolute"],
                     "failed_gates": ";".join(reasons)})
    rows.sort(key=lambda row: (row["nominal_seed"], row["epoch"]))
    for record in context["selected_records"]:
        candidates = [row for row in rows if row["nominal_seed"] == record["nominal_seed"]]
        eligible = [row for row in candidates if row["eligible"]]
        if eligible:
            raise ValueError("This package expects the recorded no-eligible-checkpoint result")
        if record["selected_epoch"] is not None or record["selected_checkpoint_digest"] is not None:
            raise ValueError("Recorded selection contradicts the recalculated outcome")
        for expected in record["candidates"]:
            row = next(row for row in candidates if row["epoch"] == expected["epoch"])
            if row["eligible"] != expected["eligible"]:
                raise ValueError("Eligibility mismatch")
            close(row["mean_normalized_retention"], expected["mean_normalized_retention"], "retention")
            close(row["visible_composite_delta_pp"] / 100, expected["visible_composite"], "visible composite")
    return means, rows


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tensor_trust_audit(out, scores, states):
    """Recover only what the retained average identifies, without naming behavior."""
    levels = (0.0, 0.5, 1.0)
    rows = []
    for state in states:
        if state == "baseline":
            continue
        for name in ("tensor_trust_hijack", "tensor_trust_extract"):
            base, trained = scores["baseline", name], scores[state, name]
            base_counts, trained_counts = Counter(base.values()), Counter(trained.values())
            transitions = Counter((base[item], trained[item]) for item in base)
            def categories(counts):
                n = sum(counts.values())
                return {"neither": counts[0.0], "exactly_one": counts[0.5], "both": counts[1.0],
                        "authorized_access_bounds": [counts[1.0] / n, (counts[1.0] + counts[0.5]) / n]}
            rows.append({"state": state, "benchmark": name, "n": len(base),
                         "baseline": categories(base_counts), "trained": categories(trained_counts),
                         "paired_matrix": [[transitions[a, b] for b in levels] for a in levels]})
    result = {"scope": "Recorded score categories and logical bounds; no refusal or other behavioral labels.",
              "matrix_order": list(levels),
              "comparisons": len(rows),
              "comparisons_with_more_both_arm_passes": sum(row["trained"]["both"] > row["baseline"]["both"] for row in rows),
              "rows": rows}
    (out / "tensor-trust-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return len(rows)


def write_fact_sheet(out, context, means, decisions):
    rules = context["selection"]["capability_gates"]
    lines = [
        "# Checked fact sheet — Direction A", "",
        "Prepared 6 September 2026 from the saved per-item scores. This is the factual reference for drafting, not a new experiment or a certification.", "",
        "**Main finding:** all nine checkpoints were ineligible under the recorded acceptance rules. Their generation-scored GSM8K and IFEval losses exceeded the allowed limits, despite higher OPI scores. No checkpoint was selected; no held-out comparison was revealed.", "",
        "**In ordinary language:** the update improved one test but made other required work worse. The preset rule therefore rejected every candidate. The results do not establish why the model's behavior changed or whether a different training recipe would qualify.", "",
        "## What was tested", "",
        f"- Model: `{context['model']['id']}` at `{context['model']['revision']}`.",
        "- Recipe: response-only QLoRA; NF4 training with bf16 compute; LoRA rank 16, alpha 32; 5,000 examples; learning rate 0.0002; three epochs. Merge/evaluation used bf16 base weights plus the trained adapter.",
        "- Runs: nominal seeds 17, 42, 2026. Three related checkpoints per run, with one shared baseline. Initialization precedes the recorded run seed, so these are not fully reproducible seed-controlled trials.",
        "- Evaluation: 300 fixed items each for OPI, each Tensor Trust benchmark, and MMLU; 200 each for GSM8K and IFEval. Tensor Trust has two arms per item. MMLU ranks first-token logits; the other benchmarks score sampled text (temperature 1, top-p 1).", "",
        "## The acceptance rule", "",
        "A checkpoint is eligible only if all three task-loss limits and the mean-retention floor pass. The code selects the highest visible composite among eligible candidates. A separate +5-point threshold labels meaningful visible improvement; it does not override failed gates.", "",
        "| Check | Maximum allowed loss / required retention | Minimum score from this baseline | Observed trained range |",
        "|---|---:|---:|---:|",
    ]
    for name in CAPABILITY:
        values = [means[row["state"], name] * 100 for row in decisions]
        loss = rules[f"{name}_max_decline"] * 100
        lines.append(f"| {LABELS[name]} ({'likelihood-ranked' if name == 'mmlu' else 'generation-scored'}) | {loss:.0f} percentage points | {means['baseline', name]*100-loss:.2f}% | {min(values):.2f}–{max(values):.2f}% |")
    retentions = [row["mean_normalized_retention"] * 100 for row in decisions]
    lines.extend([
        f"| Mean normalized retention | At least {rules['mean_normalized_retention_min']*100:.0f}% | — | {min(retentions):.2f}–{max(retentions):.2f}% |", "",
        "Retention means the average of each task's trained score divided by its baseline score; it is not an average of the three raw accuracies.", "",
        "## All scores, with no selected winner", "",
        "Values are percentages. The table and figure show every recorded epoch. Each cell is a mean over the fixed items, not a measure of deployment performance.", "",
        "| Run / epoch | OPI | TT hijack | TT extraction | GSM8K | IFEval | MMLU | Eligible? |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        "| Baseline | " + " | ".join(f"{means['baseline', name]*100:.2f}" for name in context["benchmarks"]) + " | Reference |",
    ])
    for row in decisions:
        lines.append(f"| {row['nominal_seed']} / {row['epoch']} | " + " | ".join(f"{means[row['state'], name]*100:.2f}" for name in context["benchmarks"]) + f" | {'Yes' if row['eligible'] else 'No'} |")
    lines.extend([
        "", "**One calculation you can check:** run 17, epoch 3 has GSM8K 55.50% versus baseline 73.50%. The loss is 18 percentage points, exceeding the 2-point limit. This is one worked example, not a selected checkpoint.", "",
        "![Six benchmark panels across epochs and three runs. OPI scores rise above baseline; all GSM8K and IFEval points are below their required floors. MMLU stays above its floor.](figure.png)", "",
        "**Figure caption:** observed scores for the shared baseline and all nine checkpoints. Each line describes a training run, not independent trials at every epoch. The shaded regions mark scores below the declared individual task floors. No confidence bands are drawn; the recorded paired intervals are supplied separately. A score is a benchmark's own metric, not a guarantee of successful authorized work.", "",
        "## Costs, uncertainty, and limits", "",
        f"- Main baseline plus three runs: **{context['scientific_gpu_accounted_hours']:.4f} GPU-accounted hours**, against a {context['gpu_accounted_hour_limit']}-hour budget. Some accounting uses active wall time; evaluation was unbatched. This is this implementation's cost, not a hardware minimum.",
        f"- Hardware: {context['hardware']}. Peak memory **{context['peak_vram_gib']:.3f} GiB**, above the declared {context['declared_vram_gib']:.1f} GiB allocation but within the card.",
        "- The approximately 59.87-hour figure additionally includes later diagnostic and ablation work. It is not the original three-run total and omits at least a separately recorded diagnostic smoke.",
        "- The 95% paired intervals in `paired-intervals.csv` are the historical 10,000-resample estimates over fixed item pairs. They do not estimate training-population variability or repeated-decoding noise. Individual intervals are not a simultaneous guarantee across all comparisons.",
        "- Historical outputs were not retained. Higher OPI scores do not identify refusal, suppression, or successful completion of the legitimate task. Generalization was not established because no held-out comparison was revealed.",
        "- MMLU uses first-token logits in both the pinned upstream and the local default. Older prose claiming a scoring-modality deviation is incorrect. Task and evaluation-format differences remain confounded.", "",
        "## What each source proves", "",
        "- [visible-scores.csv](visible-scores.csv): the exported observations used to recalculate every mean and gate.",
        "- [context.json](context.json): configuration, original file hashes, recorded selections, reference means, intervals, and resource source values. Original paths are provenance labels; reproduction does not open them.",
        "- [checkpoint-decisions.csv](checkpoint-decisions.csv): recalculated losses, retention, and eligibility for all nine candidates.",
        "- [paired-intervals.csv](paired-intervals.csv): uncertainty for the paired changes; `--verify-bootstrap` independently recalculates all 54 intervals from the exported observations.",
        "- [README](README.md): a small-step explanation and runnable reproduction commands.", "",
        "This supports a bounded engineering case study. It does not establish a general failure of fine-tuning, a successful deployed defense, or the cause of the observed changes. The figure and arithmetic can be reproduced from this folder; retraining cannot be reproduced from the numeric-score export.", "",
    ])
    (out / "fact-sheet.md").write_text("\n".join(lines), encoding="utf-8")


def plot(out, context, means):
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="direction-a-mpl-"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "direction-a-v1"})
    colors = ("#1673a5", "#d47b13", "#227b62")
    markers = ("o", "s", "^")
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.5), sharey=True)
    fig.patch.set_facecolor("#ffffff")
    rules = context["selection"]["capability_gates"]
    for ax, name in zip(axes.flat, context["benchmarks"]):
        baseline = means["baseline", name] * 100
        if name in CAPABILITY:
            floor = baseline - rules[f"{name}_max_decline"] * 100
            ax.axhspan(0, floor, color="#f8e7e5", zorder=0)
            ax.axhline(floor, color="#b33d35", linewidth=1.4, linestyle=":", zorder=2)
            ax.text(2.99, floor - 3, f"Required ≥ {floor:.2f}%", color="#9b342e", ha="right", va="top", fontsize=9,
                    bbox={"facecolor": "#f8e7e5", "edgecolor": "none", "pad": 1})
        ax.axhline(baseline, color="#626d76", linewidth=1.1, linestyle="--", zorder=2)
        for seed, color, marker in zip(context["seeds"], colors, markers):
            xs = [0] + list(range(1, context["epochs"] + 1))
            ys = [baseline] + [means[f"run{seed}_epoch{epoch}", name] * 100 for epoch in xs[1:]]
            ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=5, markevery=[1, 2, 3], zorder=3)
        ax.scatter([0], [baseline], color="#303b45", s=36, zorder=5)
        modality = "Likelihood-ranked" if name == "mmlu" else ("Generation-scored" if name in CAPABILITY else ("Injected-target mismatch" if name == "open_prompt_injection" else "Attack/control average"))
        ax.set_title(f"{LABELS[name]}\n{modality}", loc="left", fontsize=11, pad=11)
        ax.set_ylim(0, 100)
        ax.set_xlim(-.14, 3.15)
        ax.set_xticks([0, 1, 2, 3], ["Baseline", "1", "2", "3"])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="y", color="#d7dee4", linewidth=.65, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#b3bdc6")
        ax.tick_params(length=0, pad=5)
        ax.set_xlabel("Training epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Score (%)")
    fig.suptitle("Higher injection scores; no checkpoint met the acceptance rules", x=.065, y=.982, ha="left", fontsize=17, fontweight="bold", color="#152e40")
    fig.text(.065, .937, "One recipe · three runs · nine checkpoints · one shared baseline · all individual GSM8K and IFEval gates failed", fontsize=10.5, color="#435866")
    handles = [Line2D([0], [0], color=color, marker=marker, label=f"Run {seed}", linewidth=2) for seed, color, marker in zip(context["seeds"], colors, markers)]
    handles += [Line2D([0], [0], color="#626d76", linestyle="--", label="Baseline score"), Line2D([0], [0], color="#b33d35", linestyle=":", label="Acceptance floor")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .054), ncol=5, frameon=False)
    fig.text(.065, .029, "Shading: below a declared task floor. Lines describe the observed runs; no confidence bands or deployment claims.", fontsize=9, color="#435866")
    fig.subplots_adjust(left=.065, right=.98, top=.86, bottom=.17, hspace=.48, wspace=.18)
    fig.savefig(out / "figure.png", dpi=160, metadata={"Software": "Matplotlib; Direction A reproduction"})
    fig.savefig(out / "figure.svg", metadata={"Date": None, "Creator": "Direction A reproduction"})
    plt.close(fig)
    return matplotlib.__version__


def verify_bootstrap(context, scores):
    def percentile(values, percentage):
        rank = percentage / 100 * (len(values) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (rank - lower)

    checked = 0
    for expected in context["recorded_paired_intervals"]:
        name = expected["benchmark"]
        if name not in context["benchmarks"]:
            raise ValueError("Only benchmark-level intervals belong in this package")
        state = f"run{expected['run_seed']}_epoch{expected['epoch']}"
        diffs = [scores[state, name][item] - scores["baseline", name][item] for item in sorted(scores["baseline", name])]
        count = len(diffs)
        rng = random.Random(context["analysis"]["bootstrap_seed"])
        resampled = []
        for _ in range(context["analysis"]["bootstrap_replicates"]):
            total = 0.0
            for _ in range(count):
                total += diffs[rng.randrange(count)]
            resampled.append(total / count)
        resampled.sort()
        close(sum(diffs) / count, expected["observed_difference"], f"{state}/{name} paired change")
        close(percentile(resampled, 2.5), expected["ci_low"], f"{state}/{name} lower CI")
        close(percentile(resampled, 97.5), expected["ci_high"], f"{state}/{name} upper CI")
        checked += 1
        if checked % 6 == 0:
            print(f"Verified {checked}/54 paired intervals", flush=True)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verify-bootstrap", action="store_true")
    args = parser.parse_args()
    folder = args.input_dir.resolve()
    out = (args.out or folder).resolve()
    context, scores, states = load_inputs(folder)
    means, decisions = summarize(context, scores, states)
    out.mkdir(parents=True, exist_ok=True)
    table = [{"state": state, "nominal_seed": seed if seed is not None else "", "epoch": epoch,
              "benchmark": name, "n": len(scores[state, name]), "score": means[state, name],
              "change_pp": (means[state, name] - means["baseline", name]) * 100}
             for state, (seed, epoch) in states.items() for name in context["benchmarks"]]
    write_csv(out / "all-scores.csv", list(table[0]), table)
    write_csv(out / "checkpoint-decisions.csv", list(decisions[0]), decisions)
    intervals = context["recorded_paired_intervals"]
    write_csv(out / "paired-intervals.csv", list(intervals[0]), intervals)
    audit_count = write_tensor_trust_audit(out, scores, states)
    write_fact_sheet(out, context, means, decisions)
    matplotlib_version = plot(out, context, means) if args.plot else None
    bootstrap_count = verify_bootstrap(context, scores) if args.verify_bootstrap else 0
    result = {
        "input_score_rows": context["score_rows"], "model_states": len(states),
        "matched_historical_trained_means": len(context["recorded_scores"]),
        "matched_selection_records": len(context["selected_records"]),
        "eligible_checkpoints": sum(row["eligible"] for row in decisions),
        "tensor_trust_comparisons_recalculated": audit_count,
        "independently_recomputed_paired_intervals": bootstrap_count,
        "matplotlib_version": matplotlib_version,
        "scope": "Reanalysis of supplied numeric observations; no model inference, original repository imports, or held-out access.",
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
