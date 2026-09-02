# Prompt-Injection Research Plan for a Single 16 GB GPU

**Status:** accepted initial plan; Phase 1 issues #2 (runner core: manifest -> baseline stage -> checksummed bundle, fake adapters), #3 (held-out sealing wired into the runner), #4 (training stage (QLoRA) + preserved failure/OOM evidence, fake trainer), #5 (trained-checkpoint evaluation + checkpoint selection, fake adapters: `runner/evaluation.py`, `runner/selection.py`), #6 (held-out reveal + authorization gate, fake adapters: `runner/reveal.py`), #7 (seed-replication continuation rule + seed summary: `runner/continuation.py`), and #8 (bootstrap analysis stage: `runner/analysis.py`, operating purely on `metrics.json` fixtures) closed 2026-08-29. Phase 3 issue #9 (real-GPU smoke test: real HF/CUDA adapters in `runner/real_adapters.py`, real QLoRA trainer in `runner/real_training.py`, smoke orchestration in `runner/gpu_smoke.py`, resource projection in `runner/resource_projection.py`) also closed 2026-08-29 -- the canonical qualification run (`runs/gpu-smoke-*-20260829-191237`, evidence gitignored) completed every declared benchmark path plus a real QLoRA step and merged-checkpoint reload on the RTX 4080 under WSL2 (venv at `C:\Projects\automated_alignment_researcher\.venv`), found no feasibility findings (projected 14.9 GiB peak VRAM, 20.1 h/seed, 65.8 total GPU-hours, 31.7 GiB storage, all within manifest limits), and recorded its environment deviations in `protocol/deviations.md`. Phase 4 issue #10 (real baseline run: `runner/real_baseline.py`) also closed 2026-08-30 -- the frozen baseline (`runs/real-baseline-20260829-205020`, evidence gitignored) ran the manifest's exact declared counts (300 OPI, 300 TT-hijack, 300 TT-extract, 300 MMLU, 200 GSM8K, 200 IFEval, 200 sealed InjecAgent candidates: 133 valid/67 invalid) for real on the RTX 4080, measured 6.26 wall-hours / 6.26 GPU-hours / 12.3 GiB peak VRAM (all within limits, no feasibility findings; `runs/real-baseline-comparison-20260829-205020` compares it against the Phase-3 smoke projection), and is checksum-verified immutable. One known, non-blocking defect: that bundle's `notes.md` was generated before `runner/real_baseline.py` had its own notes-text override and incorrectly reads "tiny hardware qualification, not a scientific baseline" (copied from `gpu_smoke`'s default) -- the run's actual sample counts, scores, and sealed held-out state are all correct and manifest-exact; only that one caption is wrong. Per the evidence contract the finalized, checksummed bundle is never hand-edited, so this is recorded here rather than patched in place; `runner/real_baseline.py` now writes a correct baseline-specific caption (`baseline_notes_text`) for every subsequent run. Phase 5 issue #11 (training-data builder, ADR 0001 route A: `training_data/` package + `scripts/build_training_data.py`) also closed 2026-08-30 -- an unattended, no-`HF_TOKEN` build (`python scripts/build_training_data.py --upstream-root <automated_alignment_researcher checkout>`) assembled the full 5,000-example manifest-exact mix (2,000 prompt-injection / 1,500 clean-control / 1,000 ambiguous-boundary / 500 refusal-calibration, `data/training/dataset.jsonl` + `data/training/report.json`, both gitignored like `runs/`) from `deepset/prompt-injections` + `Lakera/gandalf_ignore_instructions` (raw + deterministic templated variation) and `databricks/databricks-dolly-15k`, deduplicated (exact + normalized-near) against the six published visible-eval subsets plus the full ADR 0001 exclusion pools (all of Tensor Trust hijacking/extraction, not just the sampled rows, and the sst2/sms_spam/hsol splits `open_prompt_injection` draws from) -- zero shortfall, every example carries a recorded source/generation_rule/category/content_hash, and InjecAgent is never touched by construction or dedup. Phase 6 issue #12 (train/evaluate/select seed 1 on real hardware: `runner/real_seed_run.py`) also closed 2026-08-31 -- seed `17`'s QLoRA adapter trained for all three frozen epochs on the real RTX 4080 against the issue #11 dataset (`runs/training-seed17-20260830-071553`, outcome `success`, no OOM), each epoch checkpoint was evaluated against the exact same published eval data the baseline used (`runs/eval-seed17-epoch{1,2,3}-20260830-071553`), and a checksummed selection record was finalized (`runs/selection-seed17-20260830-071553`). The visible composite improved meaningfully every epoch (mostly driven by `open_prompt_injection`, +0.33 to +0.50), but every epoch also badly failed the capability gate (GSM8K declined 18-48pp, IFEval declined 18-22pp; MMLU was unaffected or slightly improved) -- per RESEARCH_PLAN.md Sec. 8 this is a capability-failing result and no checkpoint was selected (`selected_checkpoint_digest: null`), which is itself a complete, publishable, non-blocking outcome, not a stop condition. Resource use: 15.18 measured GPU-hours for this seed (21.44 cumulative with the baseline, both well within the 72-hour total cap), 15.18 wall-hours (within the 24-hour per-seed cap), but measured peak VRAM 15.663 GiB slightly exceeded the manifest's 15.5 GiB declared allocation (still within the physical 16 GiB card) -- recorded as a feasibility finding in `runs/seed17-resource-comparison-20260830-071553`, not a failure. One known, non-blocking defect discovered and fixed in the process: `runner/evaluation.py`'s `run_trained_evaluation` never wired real adapters' `command_text`/`environment_text`/`notes_text`/`manifest_metadata` overrides through (`runner/core.py` and `runner/training.py` already did) -- so this run's three real eval bundles have a `command.sh`/`environment.txt`/`notes.md` that incorrectly say "fake adapters ... no real GPU or model weights used", mirroring the issue #10 baseline caption defect. `metrics.json` (the actual scores) is correct and manifest-exact in all three bundles; per the evidence contract the finalized bundles are not hand-edited, so this is recorded here. `runner/evaluation.py` now routes through `_adapter_command`/`_adapter_environment`/`_adapter_notes`/`_adapter_metadata` (deliberately excluding the dataset adapter from all four, since `RealDatasetAdapter.manifest_metadata`/`environment_lines` read `heldout_dir` for InjecAgent provenance -- exactly what this held-out-sealed stage must never touch) for every subsequent run. While #12 was running, the maintainer restructured the remaining work: #13 (reveal held-out + conditional seeds 42/2026) was closed 2026-08-30 as superseded by an approved resumable-workflow decomposition (#15, a now-closed spec issue) -- #14 (analysis + publication bundle) now runs last, blocked by 8 new tickets (#16-#23) that split seed training/evaluation/selection/reveal into resumable, checksum-gated stages sized for ~6-7-hour sessions, motivated directly by #12's real ~15-hour single-shot run. None of #14 or #16-#23 carry the `ready-for-agent` label yet, so per this workflow's own rule none are eligible for autonomous pickup; #16 in particular ("measure Issue 12 and lock recovery boundaries") explicitly needs #12's real per-stage measurements, which are recorded above and in `runs/seed17-resource-comparison-20260830-071553`. `RESEARCH_SPEC.md` (detailed Implementation/Testing Decisions) was removed from the working tree on 2026-08-29 at the maintainer's request; the frozen protocol values it described live on in `protocol/manifest.json`, and its prior text is still recoverable via `git show 2f8359d:RESEARCH_SPEC.md`. **Update 2026-09-02:** the recovery/replication tickets #16-#23 are all closed. Seed 42 (`docs/issue-22-...`) and seed 2026 (`docs/issue-23-seed-2026-execution-decision.md`) both completed on real hardware through the recovery-aware split-run workflow, each a finalized `NO_ELIGIBLE_CHECKPOINT` null selection, so the frozen replication set `[17, 42, 2026]` is now complete: three seeds, nine trained checkpoints, nine capability-gate failures, InjecAgent sealed throughout; cumulative GPU use 47.34 of 72 hours. Prefactor tickets #24 (agent-workflow repair), #25 (`.gitattributes` LF pinning + `protocol/digests.md` canonical-digest note), and #26 (ablation mid-epoch training recovery, `runner/ablation_training.py`) are also closed. Remaining open work is the analysis-and-publication chain #27-#33 (replacing the old #14), starting with #27 (frozen input manifest + exclusion allowlist); the finalization design constraints are in `docs/issue-14-finalization-handover.md`. **Update 2026-09-02 (later):** #27 is closed -- `runner/frozen_inputs.py` (one command, `python -m runner.frozen_inputs`) discovers and checksum-verifies the finalized input set (baseline bundle plus, per completed seed, the training/eval/selection/resource artifacts, the committed outcomes summary, and the registered continuation decision), binds each accepted path to a digest alongside the canonical-JSON manifest digest from #25 and the analysis version declared in `analysis/analysis-config.json`, and fails closed on any missing, ambiguous, or excluded (recovery / smoke / symlink / archive / cache / held-out / credential-like) input. Seed count (3) and trained-checkpoint count (9) are derived from `protocol/manifest.json` and the discovered evidence, correct at two seeds as well as three. **Update 2026-09-02 (later still):** #28 is closed -- `runner/claim_tables.py` (`build_claim_report`, one pure transform over already-loaded `metrics.json` dicts, no model/dataset/scorer/trainer/telemetry/storage dependency) produces the publication's central table grouped by *evaluation modality* (free-generation string scoring vs no-generation likelihood ranking, derived from the frozen eval config), with a multiple-choice-only-gate column, a caption naming all four confounded axes, the visible composite always bound to its per-benchmark decomposition, per-contrast paired bootstrap intervals (reusing `runner.analysis`), exact McNemar per binary benchmark, and a cross-run summary framed as a descriptive population statistic over executed runs / run-to-run (not seed) variance. Byte-deterministic; correct at two and three seeds. **Update 2026-09-02 (later still):** #29 is closed -- `runner/integrity_report.py` (`build_integrity_report`, one pure transform over already-parsed bundle contents) produces the two issue-#29 reports: *what broke* (generation-failure signature with truncation counts + per-benchmark seconds-per-item and the two opposing mechanisms named; Tensor Trust three-value degeneracy check with decision rule and verdict; utility-control-arm comparison; training-corpus nutrition label -- 5,000 examples, 2,505 distinct assistant responses, top-10 responses covering ~50% of the corpus, 2.9% multi-step) and *what a reader must be told* (held-out disposition `NEVER_AUTHORIZED` with the enforcing code path and sealed 133/67 candidate counts; seven-item reproducibility disclosure, three items also appended to `protocol/deviations.md`; resource accounting separating scientific totals from all-incurred compute, peak VRAM attributed to the training phase; the reconciled sample-count convention). Seed 17's eval bundles predate the machine-readable timing lines, so its checkpoints are reported as timing-unavailable, not dropped. `RESEARCH_PLAN.md` Section 5 is reconciled: Tensor Trust is "300 items (600 arm-evaluations)", held-out is "200 sealed candidates". **Update 2026-09-02 (later still):** #30 is closed -- the chat-mode MMLU confound test. A separately versioned, authorized **diagnostic** protocol (`protocol/diagnostic/chatmode-mmlu-2026-09-02.json`, downstream of the frozen manifest, outputs under `diagnostics/` only, never touching selection / the frozen bootstrap / held-out) re-scored MMLU with the chat template **enabled** on the frozen baseline and all nine merged checkpoints, paired item-by-item against the Attempt-1 raw-mode result (`runner/diagnostic_chatmode_mmlu.py` + `runner/diagnostic_report.py`; `RealModelAdapter` gained `mmlu_use_chat_template`/`mmlu_candidate_strings`, both defaulting to Attempt-1 behaviour with regression tests). Checkpoint integrity was verified before any GPU time: seeds 42/2026's six merged directories digest-match the `_directory_digest` recorded at run time in `recovery/`; seed 17's three (pre-#22, no recovery state) are present and structurally valid but their digests were first computed now, a disclosed limitation. Result: the **untrained baseline collapses to chance** in chat mode (0.567 -> 0.25, robust to the leading-space tokenization re-run), while every checkpoint declines only ~2-10 pp (mean -5.5 pp, 4/9 significant) -- the fine-tune *repairs* chat-interface MMLU rather than damaging it, and MMLU's small chat-mode decline is nothing like the generation benchmarks' collapse in the same runs. The chat-template confound does not overturn the Attempt-1 MMLU finding; the deeper likelihood-vs-generation scoring-modality confound remains untested and #28's caption should keep saying so. Cost 0.65 GPU-hours (budgeted ~2); scientific cumulative unchanged at 47.34/72 h, all-incurred ~48.1 h. Decision record: `docs/issue-30-chatmode-mmlu-diagnostic-decision.md`. Next in the chain is #31 (corpus ablation: isolate the cause of capability collapse).
**Date:** 2026-08-29  
**Repository:** `aar-prompt-injection-16gb-gpu`

## 1. Personal goal

Determine whether one independent practitioner can run, audit, and publish a complete prompt-injection post-training experiment on a Windows PC with one 16 GB consumer GPU, without paid LLM judges.

The result may be positive, null, or negative. The useful contribution is an evidence-backed account of what fits, what improves, what generalizes, what breaks, and what it costs in local compute.

## 2. Source context

### Original study

Chen Yueh-Han, Jiaxin Wen, and Jan Hendrik Kirchner, [“Automated Researchers Can Reliably Mitigate Alignment Failures”](https://alignment.anthropic.com/2026/automated-alignment-researchers/), 28 August 2026.

The study tested whether automated researchers could propose and hill-climb post-training methods across ten alignment failures. Each axis combined multiple visible benchmarks, a hidden held-out benchmark, and capability gates.

This project does **not** reproduce the paper’s ten-axis result or its autonomous research scale. It studies one narrower question under a consumer-compute constraint.

### Upstream implementation

- Repository: [YuehHanChen/automated_alignment_researcher](https://github.com/YuehHanChen/automated_alignment_researcher)
- Pinned starting commit: `1899ad64fbfbc65790d259471cc4bf4de9437aa9`
- Local checkout: `C:\Projects\automated_alignment_researcher`
- Relevant guide: [`REPRODUCE.md`](https://github.com/YuehHanChen/automated_alignment_researcher/blob/main/REPRODUCE.md)

The upstream README calls this the paper’s official code. It is kept as a separate checkout; this repository records our protocol, wrappers, evidence, and analysis.

## 3. Experiment scope

| Item | Decision |
|---|---|
| Alignment failure | Prompt injection only |
| Target model | `Qwen/Qwen3.5-2B` |
| Hardware | One NVIDIA GPU with 16 GB VRAM; actual device is GeForce RTX 4080 |
| Platform | Windows 11 with WSL2 Linux and CUDA |
| Evaluation | Full upstream prompt-injection axis |
| Training | One transparent LoRA/QLoRA-compatible intervention, then three seeds if feasible |
| Paid judges | None; use only rule/logprob scorers |
| Held-out policy | Do not inspect held-out results until the method and checkpoint-selection rule are frozen |

Attempt 1 excludes sycophancy, paid APIs, 4B/7B target models, native Windows execution, Mac comparison, Petri audits, and a full autonomous AAR search. Attempt 2 will be chosen only after the Attempt 1 postmortem and will use a separate repository.

## 4. Questions to answer

1. **Feasibility:** Can the full evaluation and lightweight post-training workflow complete reliably within 16 GB VRAM?
2. **Visible improvement:** Does training improve the combined visible prompt-injection benchmarks, rather than only Open Prompt Injection?
3. **Generalization:** Does the frozen method improve held-out InjecAgent, which changes domain and format?
4. **Capability:** Are MMLU, GSM8K, and IFEval preserved under the upstream capability gate?
5. **Practical cost:** How much wall time, GPU time, peak VRAM, disk space, and human intervention are required?
6. **Reproducibility:** Are the direction and size of the result stable across training seeds?

These questions are useful to other practitioners because they separate “the code runs” from “training improves one benchmark” and from “the improvement transfers without damaging the model.”

## 5. Benchmarks

### Visible safety benchmarks

Sample-count convention (stated once here): **item = candidate**. Each
visible-safety benchmark samples 300 candidates (manifest `sample_ids:
publisher_seed_42_first_300`). Open Prompt Injection scores one output per
candidate; Tensor Trust hijack and extract each score two arms per candidate
(HRR/ERR arm + DV arm), i.e. 600 arm-evaluations over 300 items — this is what
`protocol/power_notes.md`'s `n = 600` counts.

- `open_prompt_injection`: indirect task redirection, 300 items.
- `tensor_trust_hijack`: direct instruction override, 300 items (600 arm-evaluations).
- `tensor_trust_extract`: secret extraction, 300 items (600 arm-evaluations).

### Held-out benchmark

- `injecagent`: agentic tool-output injection, 200 sealed candidates; only valid agent turns are scored.

### Capability gates

- MMLU: 300 items.
- GSM8K: 200 items.
- IFEval: 200 items.

All are evaluated locally. A result from OPI alone is not evidence of general prompt-injection robustness.

## 6. Feasible configuration

The experiment is designed for this configuration:

| Component | Configuration |
|---|---|
| Host | Windows 11 |
| Linux environment | WSL2 |
| GPU class | One NVIDIA consumer GPU with 16 GB VRAM |
| Target model | `Qwen/Qwen3.5-2B` |
| Inference | Hugging Face Transformers with CUDA |
| Training | Memory-conscious LoRA or QLoRA |
| Evaluation | Local rule/logprob scoring; no paid judge API |

This configuration should support model inference, the complete prompt-injection evaluation, and lightweight adapter training. It is not intended for large local judge models, full-parameter training, or the paper's original multi-agent/H200-scale search. Actual peak memory, runtime, and stable batch sizes are experimental results and must be measured rather than assumed.

## 7. Protocol

1. **Freeze provenance.** Record the upstream commit, model fingerprint/revision, environment, dataset construction command, and all local deviations.
2. **Run the full baseline.** Evaluate the untrained model on all three visible benchmarks, held-out InjecAgent, and the capability basket. Archive the result before training.
3. **Freeze the intervention.** Document data sources, generation rules, training objective, LoRA/QLoRA settings, checkpoint-selection rule, and three seeds before reading new results.
4. **Train locally.** Begin with one seed. Record failures and OOMs as runs, not as discarded anecdotes.
5. **Evaluate identically.** Use the same upstream commit, suite, model decoding, and scorer semantics as the baseline.
6. **Select without held-out access.** Choose the candidate using visible benchmarks plus capability gates only; then reveal held-out InjecAgent once.
7. **Repeat.** If the first end-to-end run is technically sound, run the two remaining training seeds and report their distribution.
8. **Analyze honestly.** Report per-benchmark values and confidence intervals, not only the aggregate headline.

## 8. Interpretation rules

Report these outcomes separately:

- **Technical success:** the workflow completes within the hardware limit.
- **Visible safety improvement:** the visible composite improves and the capability gate passes.
- **Generalization:** held-out InjecAgent improves after selection without held-out access.
- **Replication stability:** the effect has the same direction across seeds.

Do not describe a gain on OPI alone as prompt-injection mitigation. Do not call a capability-failing checkpoint successful. A null result, regression, OOM boundary, or lack of held-out transfer remains publishable when supported by complete artifacts.

## 9. Evidence contract

Every run receives an immutable directory under `runs/<run-id>/` containing:

```text
manifest.yaml       purpose, parent run, commits, model, data fingerprints, seeds
command.sh          exact WSL command
config.yaml         effective evaluation or training configuration
environment.txt     OS, WSL, Python, CUDA, driver, GPU, package versions
metrics.json        machine-readable scores or training metrics
execution.log       complete stdout/stderr and exit status
gpu.csv             timestamped VRAM/utilization telemetry
notes.md            warnings, deviations, and interpretation boundaries
checksums.sha256    integrity checks for the directory
```

Run files are never overwritten. API keys, `.env`, model caches, raw held-out data, and private credentials are never committed. Large adapters may be published on Hugging Face; the repository stores their URL and checksum.

## 10. Minimal repository structure

```text
README.md                 short public overview and reproduction entrypoint
RESEARCH_PLAN.md          stable scope, questions, and protocol
CONTEXT.md                concise glossary of project-specific terms
upstream.lock             upstream URL, commit, model and dataset revisions
configs/                  versioned evaluation and training configs
scripts/                  small build, run, capture, and analysis wrappers
runs/                     immutable evidence, one directory per run
analysis/                 derived tables, plots, and interpretation
publication/              concise article draft and publication assets
docs/adr/                 only decisions that materially change this plan
```

Do not copy the upstream source tree into this repository. Pin it and keep our changes in small wrappers or explicit patch files.

## 11. Accepted decisions

| Decision | Reason |
|---|---|
| Use a new companion repository, not a fork | Separates upstream code from experimental evidence and prose |
| Name it `aar-prompt-injection-16gb-gpu` | Describes the durable compute constraint without tying the study title to one GPU generation |
| Study only prompt injection in Attempt 1 | Full local rule-scored axis; no paid judge; target model fits |
| Keep `Qwen/Qwen3.5-2B` | Matches the paper’s target for this axis and is appropriate for the 16 GB constraint |
| Run the complete axis | The paper shows that OPI-only optimization need not transfer |
| Preserve held-out isolation | Prevents checkpoint selection from overfitting InjecAgent |
| Publish negative findings | Feasibility boundaries and failed transfer are useful results |
| Put Attempt 2 in another repository | Prevents axes, decisions, and artifacts from becoming mixed |
| Build training data via public-dataset + template synthesis, not paid API or local-model generation ([ADR 0001](docs/adr/0001-training-data-sources.md)) | Keeps Attempt 1 fully auditable and unattended (no `HF_TOKEN`, no gated datasets, no paid API); local-model generation is the documented fallback if the seed-1 pilot shows no movement |

If one of these decisions changes, add a short ADR explaining what changed and why; do not silently rewrite history.

## 12. Publication

Publish one concise, modest article with the same evidence in three places:

- **GitHub:** canonical code, protocol, raw run artifacts, analysis, and reproduction instructions.
- **Hugging Face:** readable technical article, adapter if distributable, and links to GitHub evidence.
- **r/LocalLLaMA:** short summary stating the hardware class, one main chart, limitations, and a request for methodological criticism.

Article structure: **question → setup → protocol → results → failures → limitations → reproduction**.

The headline claim must match the evidence. A suitable neutral working title is:

> **Can Prompt-Injection Post-Training Be Evaluated on a Single 16 GB Consumer GPU?**

## 13. Completion criteria

Attempt 1 is complete when the baseline and trained runs are archived, held-out isolation is documented, capability results are available, resource use is summarized, seed stability is addressed, and every public number traces to an immutable artifact.

Only then select the alignment failure for Attempt 2 using the lessons about cost, memory, runtime, scorer reliability, and scientific value.
