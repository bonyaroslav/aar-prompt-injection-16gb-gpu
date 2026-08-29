# Research Spec and Execution Plan

> **Purpose:** Run and publish an auditable prompt-injection post-training experiment using one RTX 4080-class GPU with 16 GB VRAM.
> **Companion:** [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) records the original high-level plan and source context (upstream paper, upstream repository, local checkout path). This document is the accepted, detailed spec plus the phase-by-phase execution plan; both are public artifacts meant to accompany the eventual results write-up.

## Problem Statement

A solo researcher needs to determine whether a complete, auditable prompt-injection post-training experiment can be run and published on one RTX 4080-class GPU with 16 GB of VRAM. The study must avoid conflating four different outcomes: hardware feasibility, mitigation effectiveness on visible safety benchmarks, transfer to held-out InjecAgent, and stability across training seeds. It also needs explicit operational definitions for feasibility and held-out blindness, so that implementation details cannot drift between baseline and trained runs, benchmark leakage goes unnoticed, or post-result choices weaken the credibility of the report.

## Solution

Build a reproducible experiment runner around a frozen protocol manifest. Its interface accepts the declared protocol and executes the baseline, lightweight post-training, visible evaluation, capability gates, checkpoint selection, held-out reveal, seed replication, resource measurement, and evidence packaging. Every run produces an immutable, checksummed bundle containing the effective configuration, provenance, commands, environment, metrics, logs, GPU telemetry, and notes.

The primary research claim is hardware feasibility: whether one practitioner can complete the declared end-to-end protocol on one 16 GB GPU. Visible-benchmark improvement, held-out generalization, and seed stability are secondary outcomes reported independently. A null or negative mitigation result remains publishable when the protocol completes and the evidence is valid.

The study is a constrained replication-plus-extension. It preserves the pinned upstream datasets, scorers, prompt semantics, and metric definitions while allowing a predeclared memory-conscious intervention and resource-aware execution changes. Every deviation from upstream behavior is disclosed.

## User Stories

1. As a solo researcher, I want to run the entire declared experiment on one 16 GB NVIDIA GPU, so that I can evaluate whether this research workflow is practical without institutional hardware.
2. As a solo researcher, I want a single experiment-runner interface, so that I do not need to coordinate baseline, training, evaluation, and reporting through undocumented manual steps.
3. As a researcher, I want the primary claim to concern end-to-end hardware feasibility, so that mitigation quality does not redefine whether the systems result succeeded.
4. As a reader, I want mitigation effectiveness, held-out transfer, and seed stability reported separately, so that distinct findings are not collapsed into one success label.
5. As a researcher, I want null and negative mitigation results to remain publishable, so that outcome-dependent reporting does not bias the study.
6. As a replicator, I want the upstream repository and exact commit pinned, so that I can recover the implementation used by the study.
7. As a replicator, I want the model and tokenizer revisions fingerprinted, so that mutable external model identifiers cannot silently change the experiment.
8. As a replicator, I want dataset revisions, split definitions, sample identifiers, and filtering rules frozen, so that I evaluate the same examples.
9. As a reviewer, I want upstream behavior distinguished from local deviations, so that I can assess fidelity without interpreting the study as an exact replication.
10. As a reviewer, I want decoding settings and scorer semantics identical between baseline and trained evaluations, so that changes in scores are attributable to the intervention.
11. As a researcher, I want the final prescribed run to complete without an out-of-memory error, so that the 16 GB feasibility claim is operationally meaningful.
12. As a researcher, I want peak allocated VRAM limited to 15.5 GB, so that the result leaves a small, explicit operating margin below the nominal device capacity.
13. As a researcher, I want each training seed limited to 24 hours of wall time, so that the experiment remains feasible for an individual practitioner.
14. As a researcher, I want the complete study limited to 72 GPU-hours, so that replication cost is bounded before results are known.
15. As a researcher, I want experiment storage limited to 250 GB, so that local disk requirements remain practical.
16. As a researcher, I want each stage to require no manual intervention after launch, so that automation and labor requirements are measurable rather than anecdotal.
17. As a reviewer, I want exploratory OOMs and failed runs retained as evidence, so that feasibility reporting does not hide unsuccessful attempts.
18. As a researcher, I want visible evaluation to cover OPI, Tensor Trust hijacking, and Tensor Trust extraction, so that improvement is not inferred from OPI alone.
19. As a reviewer, I want the visible safety result based on the combined declared suite, so that checkpoint selection cannot cherry-pick one favorable benchmark.
20. As a researcher, I want MMLU, GSM8K, and IFEval evaluated as separate capability gates, so that a safety improvement does not conceal unacceptable capability loss.
21. As a reviewer, I want a checkpoint that fails any capability gate excluded from a successful mitigation claim, so that capability preservation is a binding constraint.
22. As a researcher, I want held-out InjecAgent excluded from method development and checkpoint selection, so that it measures transfer rather than tuning.
23. As a reviewer, I want baseline and trained held-out outputs sealed until selection is frozen, so that the baseline held-out result cannot indirectly influence training decisions.
24. As a reviewer, I want the held-out candidate identifiers frozen before execution, so that the evaluated population cannot drift after failures are observed.
25. As a reviewer, I want validity determined mechanically and all invalid turns preserved, so that denominator choices are auditable.
26. As a reviewer, I want both valid-only and intent-to-evaluate held-out results, so that technical failures cannot disappear from the headline outcome.
27. As a researcher, I want three seed identifiers declared before any quality result is read, so that seed choice cannot be outcome-dependent.
28. As a researcher, I want seed one used as a feasibility pilot, so that avoidable technical failures do not consume the full compute budget.
29. As a reviewer, I want poor model quality forbidden as a reason to skip later seeds, so that replication decisions are not biased by the first result.
30. As a researcher, I want later seeds run automatically when seed one completes technically and projected compute remains within budget, so that the conditional replication rule is objective.
31. As a reviewer, I want the visible mitigation claim to require at least a five-percentage-point absolute improvement, so that trivial positive movement is not described as meaningful success.
32. As a reviewer, I want uncertainty intervals and per-benchmark scores reported alongside the visible composite, so that aggregation does not obscure heterogeneous behavior.
33. As a maintainer, I want one frozen protocol manifest to supply every stage, so that configuration knowledge remains local rather than duplicated across commands.
34. As a maintainer, I want the experiment runner to return a machine-readable run result, so that verification and reporting can avoid parsing console text.
35. As a maintainer, I want dependencies accepted through internal adapters where behavior truly varies, so that tests can replace expensive model, dataset, GPU, and sealing implementations without widening the public interface.
36. As a reviewer, I want every run bundle to include exact commands, effective configuration, environment capture, metrics, logs, telemetry, notes, and checksums, so that the evidence can be audited independently.
37. As a reviewer, I want run bundles immutable after finalization, so that reported evidence cannot be silently altered.
38. As a maintainer, I want secrets, credentials, caches, raw held-out data, and restricted artifacts excluded from version control, so that reproducibility does not create a security or licensing failure.
39. As a reader, I want technical success, visible improvement, held-out generalization, capability preservation, and seed stability reported as distinct judgments, so that limitations remain clear.
40. As a solo researcher, I want wall time, GPU time, peak VRAM, disk consumption, and human intervention reported, so that another practitioner can estimate the real replication burden.

## Implementation Decisions

- Treat hardware feasibility as the primary claim. Treat visible mitigation effectiveness, held-out generalization, capability preservation, and seed stability as separate secondary outcomes.
- Define a successful final prescribed run as having no OOM, peak allocated VRAM no greater than 15.5 GB, no more than 24 hours of wall time per training seed, no more than 72 total GPU-hours, no more than 250 GB of experiment storage, and no manual intervention after each stage is launched.
- Preserve exploratory failures and resource-limit violations as first-class run evidence; they are not final prescribed runs.
- Describe the work as a constrained replication-plus-extension. Preserve upstream datasets, scorers, prompt semantics, and metrics while disclosing every resource-driven or intervention-specific deviation.
- Expose one deep experiment-runner module. Its interface accepts a frozen protocol manifest and returns a machine-readable result that identifies the finalized run bundle and stage outcomes.
- Keep model loading, dataset access, scorer execution, GPU telemetry, artifact storage, and held-out sealing behind internal seams only when multiple adapters are required. Do not expose those internal choices through separate top-level workflows.
- Require the protocol manifest to resolve all effective model, tokenizer, upstream, dataset, decoding, scorer, intervention, seed, resource, selection, and analysis settings before the baseline begins.
- Require every finalized run bundle to be immutable and checksummed, with provenance, the effective manifest, exact commands, captured environment, metrics, logs, GPU telemetry, human-intervention notes, and failure records.
- Run the same declared evaluation semantics for baseline and trained checkpoints.
- Compute the visible mitigation decision over OPI, Tensor Trust hijacking, and Tensor Trust extraction, while retaining every benchmark result separately.
- Require at least a five-percentage-point absolute improvement on the declared visible composite for a meaningful visible mitigation result.
- Apply MMLU, GSM8K, and IFEval as binding capability gates. A checkpoint that fails any gate cannot count as a successful mitigation.
- Seal baseline InjecAgent outputs and metrics without revealing them. Freeze the protocol and selected checkpoint before revealing baseline and trained held-out results together.
- Freeze InjecAgent candidate identifiers and mechanical validity rules before execution. Preserve all invalid cases and report both valid-only and intent-to-evaluate results, counting invalid technical failures as failures in the latter.
- Predeclare all three seed identifiers. Execute seed one as a technical feasibility pilot, then execute seeds two and three when the full visible pipeline completes without an unrecoverable technical failure and projected total resource use remains within the frozen budget. Poor quality is not a stopping condition.
- Publish a modest report even when mitigation results are null, negative, capability-failing, or unstable, provided the evidence remains valid.
- Use QLoRA with a frozen 4-bit NF4 base model, double quantization, BF16 compute, and adapter-only training. Permit plain LoRA only as a documented compatibility fallback when the pinned software stack cannot execute QLoRA correctly; model quality cannot trigger the fallback.
- Use response-only supervised fine-tuning. Training examples pair trusted benign instructions and adversarial injected content with target responses that follow trusted instructions while ignoring untrusted instructions. Exclude DPO and custom losses from Attempt 1.
- Build training data only from upstream-authorized training splits or newly generated examples produced without viewing evaluation content. Freeze source revisions, generation templates, and hashes. Remove exact and normalized near-duplicates against every visible evaluation set. Exclude InjecAgent from construction and deduplication.
- Target 5,000 training examples: 40% prompt-injection attacks, 30% clean instruction-following controls, 20% ambiguous trust-boundary cases, and 10% refusal or calibration cases. Cap Attempt 1 sequences at 2,048 tokens and report final category counts after filtering.
- Configure adapters with rank 16, alpha 32, dropout 0.05, and no bias training. Target the attention `q`, `k`, `v`, and `o` projections plus the MLP `gate`, `up`, and `down` projections. Enable gradient checkpointing and disable the model cache during training.
- Use AdamW with a learning rate of `2e-4`, cosine decay, 3% warmup, weight decay `0.01`, three epochs, micro-batch size 1, gradient accumulation 16, and gradient clipping at 1.0. If an OOM occurs, reduce sequence length once from 2,048 to 1,536 tokens and restart; do not tune quality-related parameters after viewing scores.
- Evaluate the checkpoint at the end of each epoch. For each visible safety benchmark, compute the absolute improvement over baseline and define the visible composite as the unweighted mean of the three improvements. Disqualify checkpoints that fail any capability gate. Select the eligible checkpoint with the highest composite, breaking ties by lower capability loss and then earlier epoch.
- Gate capability separately against the frozen baseline: MMLU and GSM8K may each decline by no more than two percentage points absolute, IFEval may decline by no more than three points, and mean normalized capability retention across all three must be at least 98%. Confidence-interval overlap does not substitute for passing a gate.
- Compute uncertainty with paired bootstrap resampling over fixed example identifiers using 10,000 replicates and analysis seed `271828`. Report 95% percentile intervals for every baseline-to-trained difference and for the visible composite. For three training seeds, report each seed plus their mean, range, and standard deviation without claiming a reliable population-level interval.
- Use training seeds `17`, `42`, and `2026`, evaluation seed `314159`, and bootstrap-analysis seed `271828`. Propagate training seeds through Python, NumPy, PyTorch, data shuffling, and adapter initialization. Enable deterministic algorithms where supported and record all nondeterministic CUDA warnings.
- Permit only predefined technical fallbacks after a failed first run: the single sequence-length reduction, disabling optional fused kernels, or pinning a compatible package version. Any other change creates a new protocol version and requires rerunning the baseline. Data, learning rate, epochs, selection rules, capability gates, and scoring cannot change after quality results are viewed.
- Treat the pinned upstream implementation as authoritative for prompt templates, decoding, and scorer behavior. Export every effective value into the protocol manifest before baseline execution. Resolve and freeze implicit or nondeterministic upstream settings, and label and justify every deliberate deviation before generating results.

## Testing Decisions

- Test external behavior through the experiment-runner interface rather than testing implementation details inside individual libraries.
- The highest test seam is a complete protocol execution: given a frozen manifest and controlled adapters, the experiment runner must produce the expected stage outcomes and finalized evidence bundle.
- Prefer one end-to-end contract suite at this seam over separate public test interfaces for baseline, training, evaluation, selection, and packaging.
- Use lightweight deterministic adapters for model execution, datasets, scorers, telemetry, artifact storage, and held-out sealing in the contract suite. Exercise real GPU integrations separately as environment-qualified integration tests.
- Verify that baseline and trained evaluations resolve identical prompt, decoding, sample, and scorer settings except for the declared checkpoint or adapter.
- Verify that resource-limit violations, OOMs, partial stages, and external-process failures produce preserved failure evidence and cannot be finalized as successful prescribed runs.
- Verify that a finalized run bundle rejects or detects mutation through its checksums.
- Verify that restricted held-out results cannot be read through the experiment-runner interface before checkpoint selection is finalized.
- Verify that candidate IDs and validity decisions remain stable across baseline and trained InjecAgent evaluation.
- Verify both held-out denominators: valid-only and intent-to-evaluate with invalid technical cases treated as failures.
- Verify the visible composite using fixed benchmark fixtures, including heterogeneous per-benchmark results and the five-point meaningful-improvement threshold.
- Verify every capability gate independently and ensure that one failing gate disqualifies a checkpoint regardless of its visible safety score.
- Verify deterministic checkpoint ranking and tie-breaking once those rules are frozen.
- Verify that poor seed-one quality does not suppress seeds two and three when the technical and resource continuation conditions pass.
- Verify that projected resource-budget failure prevents additional seeds without rewriting or deleting the seed-one evidence.
- Verify that secrets, credentials, caches, and raw held-out material are excluded from publishable bundles and repository-bound artifacts.
- The protocol-manifest contract tests (`tests/test_protocol.py`) are the prior art for subsequent modules; the experiment runner adds contract tests at its own seam rather than duplicating them elsewhere.

## Out of Scope

- Native Windows execution; the declared runtime is WSL2 Linux with CUDA on a Windows 11 host.
- Multi-GPU execution, cloud training, paid inference, or paid LLM judges.
- Models larger than the fixed 2B target in Attempt 1, including 4B and 7B variants.
- Mac hardware comparisons.
- Petri audits, a sycophancy research axis, or a full autonomous AAR search.
- Using InjecAgent during data construction, method development, hyperparameter tuning, checkpoint selection, or capability-gate definition.
- Committing credentials, caches, raw held-out data, or artifacts whose license forbids redistribution.
- Treating a capability-failing checkpoint as a mitigation success.
- Attempt 2, which must live in a separate repository.

## Execution Plan

### What success means

The main question is simple: **Can one person complete this entire experiment on one 16 GB GPU?**

The experiment is technically feasible only when the final prescribed run:

- finishes without an out-of-memory error;
- uses no more than 15.5 GB of allocated GPU memory;
- takes no more than 24 hours per training run;
- stays within 72 total GPU-hours and 250 GB of storage; and
- needs no manual help after each stage starts.

Model quality is reported separately. A null or negative result is still useful when the experiment is complete and the evidence is trustworthy.

### Phase 1: Freeze the rules — **done**

- Record the exact upstream code commit, model and tokenizer revisions, dataset versions, sample IDs, prompts, decoding settings, scorers, seeds, training settings, resource limits, selection rule, and analysis method in one protocol manifest. See [`protocol/manifest.json`](protocol/manifest.json), checksummed by [`protocol/manifest.sha256`](protocol/manifest.sha256).
- Record every deliberate difference from the upstream project. See [`protocol/deviations.md`](protocol/deviations.md).
- Define how held-out InjecAgent results will remain sealed until checkpoint selection is final. See [`protocol/heldout_sealing.md`](protocol/heldout_sealing.md) and [`protocol/heldout.py`](protocol/heldout.py).
- Confirm that raw held-out data, credentials, and restricted files cannot enter publishable artifacts.

**Complete when:** a reviewer can identify exactly what will run without asking for an unstated value. The manifest is frozen before generating baseline quality results.

**Publication value:** Supports *"What does a trustworthy consumer-GPU experiment require?"*

### Phase 2: Build and test the experiment workflow — not started

- Build one experiment runner that accepts the frozen manifest.
- Make it run baseline evaluation, training, checkpoint evaluation, selection, held-out sealing and reveal, resource monitoring, and evidence packaging.
- Make each run produce an immutable, checksummed bundle.
- Test the complete workflow with small, deterministic fake inputs before using the real model or GPU.

**Complete when:** the small end-to-end test passes, failure cases are preserved, and the runner can complete a short real-GPU smoke test.

**Publication value:** Supports *"How can one person keep an AI experiment reproducible?"*

### Phase 3: Run small smoke tests

- Load the pinned model in 4-bit mode.
- Run a tiny sample from every visible benchmark and capability test.
- Train briefly on a tiny data sample.
- Confirm that GPU memory, runtime, disk use, logs, and checksums are captured.
- Test the approved fallbacks without looking at full benchmark quality.

**Complete when:** every stage works on a small sample and projected full-run resource use fits the declared limits. If it does not fit, stop and publish the feasibility failure rather than quietly changing the study.

**Publication value:** Supports *"What usually breaks before consumer-GPU training starts?"*

### Phase 4: Run and seal the baseline

- Evaluate the untrained model on OPI, Tensor Trust hijacking, Tensor Trust extraction, MMLU, GSM8K, and IFEval.
- Run baseline InjecAgent evaluation through the sealing process without viewing its results.
- Use the same prompts, decoding, sample IDs, and scorers that will be used for trained checkpoints.

**Complete when:** all baseline artifacts pass checksum and completeness checks. Any protocol change after this point creates a new protocol version and requires a new baseline.

**Publication value:** Provides the "before" evidence for every result article.

### Phase 5: Prepare data and train seed 1

- Build 5,000 examples using the accepted 40/30/20/10 category mix.
- Record source, generation rule, category, and hash for every example.
- Remove exact and normalized near-duplicates against visible evaluation data.
- Keep InjecAgent completely outside data creation and checking.
- Train QLoRA seed `17` using the frozen settings and save a checkpoint after each epoch.

**Complete when:** seed 1 finishes all three epochs, or an unrecoverable technical failure is fully recorded. One OOM permits only the declared reduction from 2,048 to 1,536 tokens and a full restart.

**Publication value:** Supports *"Can safety training fit on one gaming GPU, and what does it cost?"*

### Phase 6: Evaluate and select the checkpoint

- Evaluate every epoch checkpoint on the three visible safety benchmarks and all three capability tests.
- Calculate each safety improvement over baseline; average the three without weighting them.
- Reject checkpoints that exceed any capability-loss limit.
- Select the eligible checkpoint with the highest safety average. Break ties by lower capability loss, then earlier epoch.
- Keep held-out InjecAgent sealed.

**Complete when:** the selection record is finalized and checksummed. A meaningful visible improvement requires at least five percentage points while passing every capability gate.

**Publication value:** Supports *"Did training make the model harder to trick?"* and *"Did safety training make it less useful?"*

### Phase 7: Reveal held-out results and test repeatability

- After selection is final, reveal baseline and selected-checkpoint InjecAgent results together.
- Report valid-only results and results where technical failures count as failures.
- Classify invalid or failed agent turns using the frozen rules.
- Run seeds `42` and `2026` when seed 1 was technically sound and projected total use remains within the 72 GPU-hour budget.
- Evaluate later seeds using the same frozen protocol.

**Complete when:** held-out results are reported once without post-reveal tuning. Later seeds are completed or omitted only because the frozen technical or resource rule failed — not because seed 1 performed poorly.

**Publication value:** Supports *"Did improvement work on attacks the training never saw?"* and *"Would another run produce a similar result?"*

### Phase 8: Analyze and publish

- Separate conclusions about feasibility, visible safety improvement, capability preservation, held-out transfer, and repeatability.
- Report failures and null results with the same care as positive results.
- Publish a short main report and the evidence-led article series below.
- Make publishable manifests, code, summaries, and checksums easy to find.

**Complete when:** every public claim points to concrete evidence, every limitation is visible, and another practitioner can estimate whether they can repeat the work.

## Planned Article Series

Each article uses: **question → why it matters → what we did → what we observed → evidence → meaning → what it does not prove → practical advice**.

1. **Can you do serious AI safety research with one gaming GPU?** Evidence: completion status, memory, time, storage, failures, and intervention.
2. **What does this kind of experiment really cost?** Evidence: GPU-hours, wall time, disk use, setup effort, failed attempts, and repeat-run cost.
3. **Did the extra training make the model harder to trick?** Evidence: before-and-after safety scores, uncertainty, and representative behavior changes.
4. **Did making the model safer also make it less useful?** Evidence: knowledge, mathematics, and instruction-following changes; rejected checkpoints.
5. **Did the improvement work on attacks the training never saw?** Evidence: sealed held-out comparison, validity counts, and failure categories.
6. **Would we get a similar result if we ran the training again?** Evidence: three seed results, spread, resource variation, and nondeterminism.
7. **What failed, and what can the next researcher learn from it?** Evidence: OOMs, software failures, wasted work, useful fallbacks, and a replication checklist.
8. **Is this experiment worth repeating or extending?** Evidence: supported conclusions, remaining uncertainty, and the smallest useful follow-up study.

## Status and Guardrails

- Repository role: this repo holds the protocol, wrappers, evidence, and analysis meant to be published alongside the results write-up. The upstream paper's code lives in a separate local checkout at `C:\Projects\automated_alignment_researcher` and is not vendored here — only its pinned commit and file hashes are recorded, in [`protocol/provenance.json`](protocol/provenance.json).
- Phase 1 is complete: manifest frozen, provenance fingerprinted, deviations disclosed, held-out sealing implemented and tested.
- Phase 2 (the experiment-runner module) has not been started.
- Treat this document as authoritative. Freeze protocol values before the baseline; do not inspect InjecAgent content or results during data creation, tuning, or checkpoint selection; do not change quality-related settings after viewing results.
- WSL2 Ubuntu with CUDA is configured and working (`wsl -d Ubuntu`, GPU visible as an RTX 4080 with 16376 MiB). The system Python is 3.14.4, but the upstream checkout's project venv (`C:\Projects\automated_alignment_researcher\.venv`, Python 3.12.14) already has the full stack installed and CUDA-functional: `torch==2.8.0+cu128` (`torch.cuda.is_available() == True`), `transformers==5.16.1`, `peft==0.18.0`, `trl==0.21.0`, `bitsandbytes==0.46.1`, `datasets==3.6.0`, `accelerate==1.12.0`, `flash-attn==2.8.1`, `vllm==0.11.0`. Phase 3 does not need to confirm wheel availability; it needs to confirm this venv runs the frozen manifest's real workload. `unsloth` fails to import standalone in this venv (`NameError` at import time) — treat Unsloth as unavailable and rely on plain PEFT/TRL QLoRA (already the manifest's declared method) rather than the upstream `pyproject.toml`'s Unsloth-accelerated path.
