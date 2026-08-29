# Real-GPU Smoke Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify the repository's existing runner interfaces with real Hugging Face/CUDA dataset, model, scorer, telemetry, and QLoRA trainer adapters on the RTX 4080 under WSL2, then report projected full-run resource feasibility.

**Architecture:** Keep `run_baseline` and `run_training` dependency-injection signatures unchanged. Add lazy-importing real adapters which translate the runner's per-item protocol into the pinned upstream publishers, prompts, and rule scorers; add optional evidence hooks so the same runners write real environment/log metadata without breaking fake adapters. A smoke entry point creates a temporary, explicitly reduced manifest and external held-out/data directories, executes one or two real items per benchmark and a tiny QLoRA epoch, then writes a checksummed projection artifact.

**Tech Stack:** Python 3.12, unittest, Hugging Face Transformers/Datasets, PyTorch CUDA, bitsandbytes, PEFT, NVIDIA `nvidia-smi`, WSL2.

**Spec:** GitHub issue #9, `https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu/issues/9`; frozen values in `protocol/manifest.json`.

## Global Constraints

- Upstream checkout commit must be `1899ad64fbfbc65790d259471cc4bf4de9437aa9` and model/tokenizer revision must be `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Real adapters must plug into the existing `run_baseline` and `run_training` call signatures; fake adapter behavior and offline tests must remain intact.
- Held-out InjecAgent plaintext and per-candidate outcomes must remain outside publishable run bundles and unreadable before selection authorization.
- Smoke reductions affect only an ephemeral smoke manifest and must never rewrite `protocol/manifest.json`.
- Only the manifest's three technical fallbacks are permitted; resource overruns are findings, not implicit configuration changes.
- No baseline-quality interpretation or checkpoint selection is performed by this hardware qualification.

---

### Task 1: Resource projection

**Files:**
- Create: `runner/resource_projection.py`
- Create: `tests/test_resource_projection.py`

**Interfaces:**
- Produces: `project_full_run_resources(manifest, *, measured_seconds_per_item, measured_peak_vram_mb, measured_train_seconds_per_step, measured_checkpoint_bytes, default_seconds_per_item) -> dict`.

- [x] Run `python -m unittest tests.test_resource_projection -v` and confirm the previously written tests fail if `project_full_run_resources` is removed or returns empty projections.
- [x] Preserve the existing implementation that extrapolates all seven benchmark item counts, three epochs, three seeds, and merged-checkpoint storage.
- [x] Run `python -m unittest tests.test_resource_projection -v` and confirm all projection and limit-finding tests pass.

### Task 2: Real evaluation adapters

**Files:**
- Create: `runner/real_adapters.py`
- Create: `tests/test_real_adapters.py`

**Interfaces:**
- Produces: `RealDatasetAdapter(suite_dir, heldout_dir, max_items_per_benchmark=None).load_items(benchmark, sample_count) -> list[dict]`.
- Produces: `RealModelAdapter(model_ref, revision, upstream_root).generate(benchmark, item, config) -> str`.
- Produces: `RealScorerAdapter(upstream_root).score(benchmark, item, output, config) -> dict`.
- Produces: `RealTelemetryAdapter(sample_interval_seconds=0.25)` with the existing `start() -> None` and `stop() -> list[dict]` methods plus evidence text/events consumed opportunistically by the runner.

- [x] Write fixture-driven tests that name the observable breaks: stable IDs and smoke caps; upstream-verbatim OPI/Tensor Trust/InjecAgent prompt construction; MMLU logits rather than generated text; GSM8K/IFEval rule scoring; Tensor Trust dual-arm outputs; InjecAgent invalid preservation; telemetry CSV row shape.
- [x] Run `python -m unittest tests.test_real_adapters -v`; verify failures are caused by the missing module/behaviors.
- [x] Implement lazy imports and benchmark-specific translation while delegating parsing/checking to the pinned upstream modules.
- [x] Run `python -m unittest tests.test_real_adapters -v` and the complete offline suite.

### Task 3: Real QLoRA trainer adapter

**Files:**
- Create: `runner/real_training.py`
- Create: `tests/test_real_training.py`

**Interfaces:**
- Produces: `RealQLoRATrainerAdapter(model_ref, revision, training_examples, work_dir, smoke_max_steps=None)` implementing `train_epoch(*, seed, epoch, sequence_length, config) -> str` and `merge_checkpoint(fingerprint, output_dir) -> None`.

- [x] Write dependency-injected tests proving response-token-only labels, exact manifest LoRA target mapping (`q_proj` through `down_proj`), optimizer/accumulation propagation, deterministic adapter fingerprints, and standalone merged-directory output.
- [x] Run `python -m unittest tests.test_real_training -v` and verify the expected missing-behavior failures.
- [x] Implement 4-bit NF4/double-quant/bfloat16 PEFT training with gradient checkpointing, one persistent optimizer/scheduler, a smoke step cap, and per-epoch adapter snapshots merged into a freshly loaded non-quantized base model.
- [x] Run the focused and complete offline suites.

### Task 4: Runner evidence hooks and smoke orchestration

**Files:**
- Modify: `runner/core.py`
- Modify: `runner/training.py`
- Create: `runner/gpu_smoke.py`
- Create: `tests/test_gpu_smoke.py`

**Interfaces:**
- Consumes: the adapters and projection function from Tasks 1-3.
- Produces: `build_smoke_manifest(base_manifest, *, samples_per_benchmark, training_count, epochs) -> dict` and a CLI that accepts explicit upstream, suite, held-out, output, and model-cache paths.

- [x] Write tests proving fake bundles retain fake evidence, real evidence hooks populate `environment.txt`/`execution.log`, smoke manifest reductions do not mutate the frozen input, held-out paths stay external, and projection findings are serialized.
- [x] Run `python -m unittest tests.test_gpu_smoke -v` and verify expected failures.
- [x] Add backward-compatible optional evidence hooks and implement the smoke CLI without changing runner dependency-injection signatures.
- [x] Run focused tests and `python -m unittest discover -s tests -v`.

### Task 5: WSL2/CUDA qualification run and deviation reconciliation

**Files:**
- Modify if needed: `protocol/deviations.md`
- Create under ignored evidence root: `runs/gpu-smoke-*/...`

**Interfaces:**
- Consumes: `python -m runner.gpu_smoke` from Task 4.
- Produces: checksummed baseline/training/projection evidence, a loadable merged checkpoint, and exact environment/deviation findings.

- [x] Run WSL `nvidia-smi`, CUDA/PyTorch imports, pinned upstream commit verification, and model-cache resolution; stop on any undeclared deviation.
- [x] Run one real item from every benchmark with held-out sealing and capture GPU/environment/log evidence.
- [x] Run the tiny real QLoRA sample and load the merged checkpoint for one generation.
- [x] Attempt the approved 2048-to-1536 OOM path only if safely triggerable; otherwise record that it was not triggerable at smoke scale.
- [x] Compute the full-run projection from measured durations, peak VRAM, and checkpoint bytes; preserve every exceeded limit as a feasibility finding.
- [x] Compare observed package/hardware facts to `protocol/deviations.md` and add only newly observed deviations.
- [x] Run `python -m unittest discover -s tests -v`, verify all evidence bundles, and inspect `git diff --check` plus `git status --short`.
