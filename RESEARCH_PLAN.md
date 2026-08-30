# Prompt-Injection Research Plan for a Single 16 GB GPU

**Status:** accepted initial plan; Phase 1 issues #2 (runner core: manifest -> baseline stage -> checksummed bundle, fake adapters), #3 (held-out sealing wired into the runner), #4 (training stage (QLoRA) + preserved failure/OOM evidence, fake trainer), #5 (trained-checkpoint evaluation + checkpoint selection, fake adapters: `runner/evaluation.py`, `runner/selection.py`), #6 (held-out reveal + authorization gate, fake adapters: `runner/reveal.py`), #7 (seed-replication continuation rule + seed summary: `runner/continuation.py`), and #8 (bootstrap analysis stage: `runner/analysis.py`, operating purely on `metrics.json` fixtures) closed 2026-08-29. Phase 3 issue #9 (real-GPU smoke test: real HF/CUDA adapters in `runner/real_adapters.py`, real QLoRA trainer in `runner/real_training.py`, smoke orchestration in `runner/gpu_smoke.py`, resource projection in `runner/resource_projection.py`) also closed 2026-08-29 -- the canonical qualification run (`runs/gpu-smoke-*-20260829-191237`, evidence gitignored) completed every declared benchmark path plus a real QLoRA step and merged-checkpoint reload on the RTX 4080 under WSL2 (venv at `C:\Projects\automated_alignment_researcher\.venv`), found no feasibility findings (projected 14.9 GiB peak VRAM, 20.1 h/seed, 65.8 total GPU-hours, 31.7 GiB storage, all within manifest limits), and recorded its environment deviations in `protocol/deviations.md`. Phase 4 issue #10 (real baseline run: `runner/real_baseline.py`) also closed 2026-08-30 -- the frozen baseline (`runs/real-baseline-20260829-205020`, evidence gitignored) ran the manifest's exact declared counts (300 OPI, 300 TT-hijack, 300 TT-extract, 300 MMLU, 200 GSM8K, 200 IFEval, 200 sealed InjecAgent candidates: 133 valid/67 invalid) for real on the RTX 4080, measured 6.26 wall-hours / 6.26 GPU-hours / 12.3 GiB peak VRAM (all within limits, no feasibility findings; `runs/real-baseline-comparison-20260829-205020` compares it against the Phase-3 smoke projection), and is checksum-verified immutable. One known, non-blocking defect: that bundle's `notes.md` was generated before `runner/real_baseline.py` had its own notes-text override and incorrectly reads "tiny hardware qualification, not a scientific baseline" (copied from `gpu_smoke`'s default) -- the run's actual sample counts, scores, and sealed held-out state are all correct and manifest-exact; only that one caption is wrong. Per the evidence contract the finalized, checksummed bundle is never hand-edited, so this is recorded here rather than patched in place; `runner/real_baseline.py` now writes a correct baseline-specific caption (`baseline_notes_text`) for every subsequent run. Next open, unblocked issue in dependency order: #11 (training-data builder, ADR 0001 route A) has no blockers and can proceed without real hardware; #12 (train/evaluate/select seed 1 on real hardware) is blocked by #11 and, once unblocked, is itself a large real-GPU undertaking that should not be auto-started without explicit confirmation given its cost and duration. `RESEARCH_SPEC.md` (detailed Implementation/Testing Decisions) was removed from the working tree on 2026-08-29 at the maintainer's request; the frozen protocol values it described live on in `protocol/manifest.json`, and its prior text is still recoverable via `git show 2f8359d:RESEARCH_SPEC.md`.
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

- `open_prompt_injection`: indirect task redirection, 300 items.
- `tensor_trust_hijack`: direct instruction override, 600 items.
- `tensor_trust_extract`: secret extraction, 600 items.

### Held-out benchmark

- `injecagent`: agentic tool-output injection, up to 300 candidates; only valid agent turns are scored.

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
