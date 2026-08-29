# Research Execution Plan

> **Status:** Draft — First Iteration  
> **Purpose:** Run and publish an auditable prompt-injection experiment using one RTX 4080-class GPU with 16 GB VRAM.  
> **Source of truth:** `RESEARCH_SPEC.local.md`

## What success means

The main question is simple: **Can one person complete this entire experiment on one 16 GB GPU?**

The experiment is technically feasible only when the final prescribed run:

- finishes without an out-of-memory error;
- uses no more than 15.5 GB of allocated GPU memory;
- takes no more than 24 hours per training run;
- stays within 72 total GPU-hours and 250 GB of storage; and
- needs no manual help after each stage starts.

Model quality is reported separately. A null or negative result is still useful when the experiment is complete and the evidence is trustworthy.

## Phase 1: Freeze the rules

### What to do

- Record the exact upstream code commit, model and tokenizer revisions, dataset versions, sample IDs, prompts, decoding settings, scorers, seeds, training settings, resource limits, selection rule, and analysis method in one protocol manifest.
- Record every deliberate difference from the upstream project.
- Define how held-out InjecAgent results will remain sealed until checkpoint selection is final.
- Confirm that raw held-out data, credentials, and restricted files cannot enter publishable artifacts.

### Expected result

One complete manifest controls every later stage. No important setting is left implicit.

### Evidence to save

- Protocol manifest and its checksum
- Upstream, model, tokenizer, and dataset fingerprints
- Environment and dependency versions
- Written held-out sealing and reveal procedure
- Written list of allowed technical fallbacks

### Complete when

A reviewer can identify exactly what will run without asking for an unstated value. Freeze the manifest before generating baseline quality results.

### Publication value

Supports: **“What does a trustworthy consumer-GPU experiment require?”**

## Phase 2: Build and test the experiment workflow

### What to do

- Build one experiment runner that accepts the frozen manifest.
- Make it run baseline evaluation, training, checkpoint evaluation, selection, held-out sealing and reveal, resource monitoring, and evidence packaging.
- Make each run produce an immutable, checksummed bundle.
- Test the complete workflow with small, deterministic fake inputs before using the real model or GPU.

### Expected result

One command can run a declared stage and return a clear machine-readable result. A failed stage preserves its evidence instead of disappearing.

### Evidence to save

- Test results for successful and failed runs
- Example run bundle
- Proof that changing a finalized bundle breaks checksum verification
- Proof that held-out results cannot be read early
- Proof that secrets and raw held-out data are excluded

### Complete when

The small end-to-end test passes, failure cases are preserved, and the runner can complete a short real-GPU smoke test.

### Publication value

Supports: **“How can one person keep an AI experiment reproducible?”**

## Phase 3: Run small smoke tests

### What to do

- Load the pinned model in 4-bit mode.
- Run a tiny sample from every visible benchmark and capability test.
- Train briefly on a tiny data sample.
- Confirm that GPU memory, runtime, disk use, logs, and checksums are captured.
- Test the approved fallbacks without looking at full benchmark quality.

### Expected result

The full path works at small scale before expensive runs begin.

### Evidence to save

- Peak GPU memory and runtime for each smoke stage
- Logs for any OOM, package, CUDA, or scorer failure
- The exact fallback used, if any
- A clear estimate of full-run time and storage

### Complete when

Every stage works on a small sample and projected full-run resource use fits the declared limits. If it does not fit, stop and publish the feasibility failure rather than quietly changing the study.

### Publication value

Supports: **“What usually breaks before consumer-GPU training starts?”**

## Phase 4: Run and seal the baseline

### What to do

- Evaluate the untrained model on OPI, Tensor Trust hijacking, Tensor Trust extraction, MMLU, GSM8K, and IFEval.
- Run baseline InjecAgent evaluation through the sealing process without viewing its results.
- Use the same prompts, decoding, sample IDs, and scorers that will be used for trained checkpoints.

### Expected result

A complete baseline exists for every later comparison, while held-out results remain hidden.

### Evidence to save

- Per-example visible and capability results
- Aggregate scores and error counts
- Runtime, peak VRAM, disk use, and intervention records
- Sealed held-out artifact and checksum
- Verification that the held-out result has not been revealed

### Complete when

All baseline artifacts pass checksum and completeness checks. Any protocol change after this point creates a new protocol version and requires a new baseline.

### Publication value

Provides the “before” evidence for every result article.

## Phase 5: Prepare data and train seed 1

### What to do

- Build 5,000 examples using the accepted 40/30/20/10 category mix.
- Record source, generation rule, category, and hash for every example.
- Remove exact and normalized near-duplicates against visible evaluation data.
- Keep InjecAgent completely outside data creation and checking.
- Train QLoRA seed `17` using the frozen settings and save a checkpoint after each epoch.

### Expected result

Three candidate checkpoints are produced by a training run that fits the resource limits.

### Evidence to save

- Final data counts and exclusion reasons
- Contamination-check report
- Training loss and learning-rate history
- GPU memory, GPU-hours, wall time, storage, warnings, and interventions
- Checkpoint hashes and full run bundle

### Complete when

Seed 1 finishes all three epochs, or an unrecoverable technical failure is fully recorded. One OOM permits only the declared reduction from 2,048 to 1,536 tokens and a full restart.

### Publication value

Supports: **“Can safety training fit on one gaming GPU, and what does it cost?”**

## Phase 6: Evaluate and select the checkpoint

### What to do

- Evaluate every epoch checkpoint on the three visible safety benchmarks and all three capability tests.
- Calculate each safety improvement over baseline.
- Average the three safety improvements without weighting them.
- Reject checkpoints that exceed any capability-loss limit.
- Select the eligible checkpoint with the highest safety average. Break ties by lower capability loss, then earlier epoch.
- Keep held-out InjecAgent sealed.

### Expected result

One checkpoint is selected using the frozen rule, or the study clearly reports that no checkpoint qualified.

### Evidence to save

- Per-example and aggregate results for every checkpoint
- Paired 95% bootstrap intervals using 10,000 resamples
- Capability-gate pass or fail for each checkpoint
- Selection calculation and selected checkpoint hash
- Examples of changed model behavior, chosen by a declared rule rather than by convenience

### Complete when

The selection record is finalized and checksummed. A meaningful visible improvement requires at least five percentage points while passing every capability gate.

### Publication value

Supports: **“Did training make the model harder to trick?”** and **“Did safety training make it less useful?”**

## Phase 7: Reveal held-out results and test repeatability

### What to do

- After selection is final, reveal baseline and selected-checkpoint InjecAgent results together.
- Report valid-only results and results where technical failures count as failures.
- Classify invalid or failed agent turns using the frozen rules.
- Run seeds `42` and `2026` when seed 1 was technically sound and projected total use remains within the 72 GPU-hour budget.
- Evaluate later seeds using the same frozen protocol.

### Expected result

The study shows whether improvement transfers to unseen agentic attacks and whether training results are repeatable.

### Evidence to save

- Held-out reveal record and checksum verification
- Fixed candidate IDs and validity decisions
- Valid-only and all-attempt results
- Each seed’s visible, capability, and resource results
- Mean, range, and standard deviation across seeds
- Every nondeterministic CUDA warning

### Complete when

Held-out results are reported once without post-reveal tuning. Later seeds are completed or omitted only because the frozen technical or resource rule failed—not because seed 1 performed poorly.

### Publication value

Supports: **“Did improvement work on attacks the training never saw?”** and **“Would another run produce a similar result?”**

## Phase 8: Analyze and publish

### What to do

- Separate conclusions about feasibility, visible safety improvement, capability preservation, held-out transfer, and repeatability.
- Report failures and null results with the same care as positive results.
- Publish a short main report and the evidence-led article series below.
- Make publishable manifests, code, summaries, and checksums easy to find.

### Expected result

Readers can understand the main finding quickly and inspect the supporting evidence when they want more detail.

### Evidence to save

- Final tables and figures linked to source result files
- Article claim-to-evidence checklist
- Limitations and “does not prove” statements
- Replication instructions and resource estimate
- Archive checksums and license notes

### Complete when

Every public claim points to concrete evidence, every limitation is visible, and another practitioner can estimate whether they can repeat the work.

## Planned article series

Each article uses: **question → why it matters → what we did → what we observed → evidence → meaning → what it does not prove → practical advice**.

1. **Can you do serious AI safety research with one gaming GPU?**  
   Evidence: completion status, memory, time, storage, failures, and intervention.

2. **What does this kind of experiment really cost?**  
   Evidence: GPU-hours, wall time, disk use, setup effort, failed attempts, and repeat-run cost.

3. **Did the extra training make the model harder to trick?**  
   Evidence: before-and-after safety scores, uncertainty, and representative behavior changes.

4. **Did making the model safer also make it less useful?**  
   Evidence: knowledge, mathematics, and instruction-following changes; rejected checkpoints.

5. **Did the improvement work on attacks the training never saw?**  
   Evidence: sealed held-out comparison, validity counts, and failure categories.

6. **Would we get a similar result if we ran the training again?**  
   Evidence: three seed results, spread, resource variation, and nondeterminism.

7. **What failed, and what can the next researcher learn from it?**  
   Evidence: OOMs, software failures, wasted work, useful fallbacks, and a replication checklist.

8. **Is this experiment worth repeating or extending?**  
   Evidence: supported conclusions, remaining uncertainty, and the smallest useful follow-up study.

## Immediate next action

Start Phase 1. Create the frozen protocol manifest and the held-out sealing procedure before running any full benchmark or training job.
