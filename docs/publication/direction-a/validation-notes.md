# Checks performed on this package

Checked on 6 September 2026 using Python **3.12.14** on Windows 11 and Matplotlib **3.11.1**. The machine-readable record is [validation-receipt.json](validation-receipt.json); [verification.json](verification.json) records the supplied full reproduction run.

## What passed

| Check performed | Result | Why it matters |
|---|---|---|
| Compare original input files with the export's recorded SHA-256 hashes | All 16 matched. | Identifies the protocol, reports, run metrics, and selection records used. |
| Read visible per-item outcomes and check IDs, counts, allowed values, and pairing | 16,000 score records across ten model states. | Averages and paired changes use the intended observations. |
| Recalculate every mean | Six baseline means and 54 trained means; all recorded comparisons matched within `1e-12`. | The score table does not depend on copying prose. |
| Reapply the acceptance rules | Zero eligible checkpoints; all three selection records matched. | Checks the headline decision directly. |
| Independently repeat benchmark-level paired bootstraps | All 54 intervals matched using 10,000 resamples each and seed 271828. | Checks the supplied historical intervals from the exported item pairs. |
| Recalculate Tensor Trust categories and paired matrices | All 18 category comparisons matched the original count records; the worked matrix also matched. | Checks the B1 example and the 17/18 counterevidence from individual scores. |
| Run the package from a copy outside the repository | Figure, tables, audit, and interval verification completed. | The script does not need private run directories or original repository imports. |
| Run the copied script with Python isolated mode and no third-party dependencies | Numeric outputs were byte-identical to the full run. | Table reproduction does not require Matplotlib, model libraries, or a GPU. |
| Change the CSV in a separate temporary test input | Script rejected the changed file because its hash differed. | Tests that the input-integrity check actually detects a changed input. |
| Render and inspect the figure | PNG inspected; labels, thresholds, colors, and all six panels readable. | Confirms that the supplied figure can communicate the numerical result. |

The separate-folder full command was:

```text
python -B reproduce.py --plot --verify-bootstrap
```

A second run used `python -I -B reproduce.py --out <temporary-output-folder>` to check the standard-library path without overwriting the full-run verification record. Generated numeric tables, the fact sheet, and the Tensor Trust audit were compared byte for byte. The checked figure and full-run verification record were copied back into this package.

## What these checks do not establish

They verify reanalysis of the supplied observations. They do not recreate model training or missing historical outputs, validate the meaning of every original parser judgment, repeat model sampling, inspect the held-out comparison, or rerun the older full provenance pipeline. The separate visible-composite bootstrap was not repeated. These boundaries are why the articles should report a measured rejection decision and a scoring audit, without claiming a behavioral cause or deployment improvement.

`validation-receipt.json` binds this preparation check to the script and CSV hashes. If the script or data changes later, revalidate before relying on this receipt. Running the default command later can replace `verification.json` with a record of that shorter invocation; it does not erase this dated validation record.
