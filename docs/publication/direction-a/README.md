# Start here: the fact sheet and figure are prepared

**Open [fact-sheet.md](fact-sheet.md) first.** It contains the checked result, the complete score table, the acceptance limits, and one figure. You do not need to construct another fact sheet yourself.

The companion [Direction B fact sheet](scorer-fact-sheet.md) explains the scoring audit with a four-case table, actual counts, and simple arithmetic. Both directions now have a concrete factual reference.

A fact sheet is simply the document we use to keep later drafts consistent. Here, “checked” means its numbers were calculated from the saved observations and compared with the original reports and selection records. It does not mean every explanation in an older report was correct, or that a journal has reviewed the work.

## What was done, in small steps, and why

| Step completed | What it means in plain language | Where you can check it |
|---|---|---|
| 1. Identify the exact experiment | Use one recipe and its original three runs, rather than mix in later investigations. | Model, training, and protocol fields in [context.json](context.json). |
| 2. Read the individual results | Start with each item's saved score, rather than copy a paragraph from a handover. | [visible-scores.csv](visible-scores.csv): 16,000 records, including the shared baseline. |
| 3. Match the same items across model states | Ensure a change compares the same questions before and after training. | The script checks matching item IDs and counts for all ten states. |
| 4. Recalculate every average | Add the item scores and divide by the item count. Compare the result with the original report. | [all-scores.csv](all-scores.csv): 60 rows, comprising six baseline means and 54 trained means. |
| 5. Reapply the acceptance rules | Compare each loss with its allowed limit; also calculate mean retention. | [checkpoint-decisions.csv](checkpoint-decisions.csv), checked against all three original selection records. |
| 6. Put the result into one figure | Show all runs and epochs, using the same 0–100 vertical scale. Red shading marks unacceptable task scores. | [figure.png](figure.png), with [SVG](figure.svg) for resizing in an article. These are two formats of the same figure. |
| 7. Keep the unknowns visible | Separate what was measured from why it may have happened. | The limits section of [fact-sheet.md](fact-sheet.md). |
| 8. Make the arithmetic runnable elsewhere | Package the numeric observations, settings, and script together, without model files or private directory dependencies. | This folder is the reproduction package. |

The pairing matters: there is one sampled baseline, not a new baseline for each run. The original outputs are missing, so these steps check recorded scores, not whether every original parser judgment was semantically correct.

## A five-minute way to understand the result

These are optional reading steps, not another implementation checklist.

1. Open [the fact sheet](fact-sheet.md). Read the first two paragraphs; they state the result and its limit.
2. Find the baseline GSM8K score: **73.50%**. This is performance before the update.
3. Find run 17, epoch 3: **55.50%**. Subtract: `73.50 − 55.50 = 18.00` percentage points lost.
4. Find the allowed GSM8K loss: **2 points**. Eighteen exceeds two, so this candidate fails that requirement. The other candidates are checked the same way.
5. Look at the bottom-left and bottom-middle figure panels. Every trained point is below the required line. Then look at the top-left panel: OPI increased. That contrast is the core of Direction A.

The article can therefore say: **“The injection score increased, but every candidate failed the requirements for keeping other measured tasks usable, so we selected none.”** It cannot yet say the score increased because the model refused or suppressed the input. That is Direction B's unresolved behavioral question.

## Why these artifacts are enough for this stage

The **fact sheet** prevents contradictory numbers and overbroad conclusions. The **figure** lets a reader inspect all nine checkpoints at once. The **small data package** lets another person check the arithmetic. Each has a specific job. No additional figure or training sweep is needed to start drafting the bounded Direction A article.

The figure uses each benchmark's raw percentage scale. OPI, Tensor Trust, and the task tests are not interchangeable measures. We do not average them into one visual success score, use the unstable OPI/TT ratio as a headline, or draw confidence bands from only three run lines.

## Reproduce the tables

Requirements: Python 3.10 or newer. No GPU, model download, original repository, or third-party Python package is needed for the tables.

Open a terminal in this folder and run:

```text
python reproduce.py
```

This verifies the input hash and item pairing, recalculates all means and gate decisions, compares them with the recorded values, and writes the fact sheet and CSV tables. It also writes `verification.json`, which describes exactly what that invocation checked.

## Reproduce the figure

In the same folder, install the plotting dependency and run:

```text
python -m pip install -r requirements.txt
python reproduce.py --plot
```

The script writes `figure.png` and `figure.svg`. Matplotlib 3.11.1 was used for the supplied files. Layout and file bytes can vary with plotting dependencies/platforms; the numeric data and decisions are the reproducibility claim.

For the slower, optional uncertainty check:

```text
python reproduce.py --verify-bootstrap
```

This independently repeats all **54** historical benchmark-level paired intervals, with the frozen 10,000 resamples and seed 271828. It does not repeat model generation or the separate visible-composite bootstrap. The supplied intervals describe resampling of the evaluated item pairs, not uncertainty over all possible training runs or decoding samples.

Use `--out <new-folder>` to write generated outputs elsewhere. To reproduce the complete reader-facing folder including working relative links, copy this entire folder first and run there. The script uses its own directory to locate its inputs; it does not infer the original repository from your working directory.

## Files and boundaries

| File | Purpose |
|---|---|
| `visible-scores.csv` | Original visible numeric outcomes and hashed item identifiers. No prompts or completions. |
| `context.json` | Frozen setup, source-file hashes, recorded comparison values, and selection/resource metadata. |
| `reproduce.py` | Standalone calculation and plotting script. |
| `fact-sheet.md`, `all-scores.csv`, `checkpoint-decisions.csv` | Recalculated narrative and tables. |
| `paired-intervals.csv` | Historical paired-change intervals, optionally recalculated with the flag above. |
| `scorer-fact-sheet.md`, `tensor-trust-audit.json` | Direction B explanation and recalculated category counts, paired matrices, and logical bounds for all 18 comparisons. |
| `figure.png`, `figure.svg` | Two exports of one figure. |
| `verification.json` | Results of the most recent invocation. A value of zero recomputed intervals means that optional check was not run in that invocation. |
| `validation-notes.md` | The recorded checks performed when preparing this package, including the isolated-folder check. |
| `validation-receipt.json` | Dated verification record with the checked script/data hashes. |
| `export_from_repository.py` | Maintainer-only extraction tool. Unlike `reproduce.py`, it needs the original visible run bundles. Readers do not need to run it. |

The input hashes establish which files were used. They cannot make an incorrect measurement correct. This package enables reanalysis of recorded visible results; it does not recreate missing historical outputs, reproduce training from nominal seeds, or reveal the held-out comparison.

For writing examples and the reasoning behind the presentation, see [the short reading guide](../publication-examples.md). For the fuller A/B argument, see [the knowledge base](../../publication-directions-a-b-knowledge-base.md).
