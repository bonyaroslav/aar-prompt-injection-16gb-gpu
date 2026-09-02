# Phase 1 upstream reconciliation

The pinned upstream checkout is authoritative for prompt text, roles, decoding and per-benchmark scorer behavior. Its committed `HEAD` is `1899ad64fbfbc65790d259471cc4bf4de9437aa9`; the checkout also has unrelated dirty/untracked prior artifacts, which are excluded from provenance.

The accepted local protocol does not silently inherit three upstream choices:

1. Upstream reports a geometric mean of closed fractions; this study selects on the unweighted mean of absolute improvements across the three visible safety benchmarks and requires +5 percentage points.
2. Upstream capability filtering uses CI-overlap. This study uses the accepted absolute per-gate declines and 98% mean normalized retention.
3. Upstream rule scoring skips failed/empty generations. Held-out sealing preserves every invalid turn and reports both valid-only and intent-to-evaluate, where technical invalids are failures.

These are predeclared local analysis/protocol deviations, not post-result changes. The OPI combine-only attack, 300 published candidates per visible benchmark, 200 InjecAgent candidates, real system/user roles, and all decoding values are resolved from the pinned committed docs/code and exported in `manifest.json`.

## Environment deviations from `pyproject.toml`

The upstream `pyproject.toml` pins a much larger stack (Unsloth-accelerated training, vLLM/sglang serving, VERL/Ray RL) than this study's declared response-only QLoRA workload requires. The project venv actually installed and CUDA-functional at protocol-freeze time (`C:\Projects\automated_alignment_researcher\.venv`, Python 3.12.14) differs from that pin in the following recorded ways:

1. **`numpy==2.5.2` installed vs. upstream's pinned `numpy<2.0.0`.** No numpy-2 incompatibility has been observed in the benchmarks this study exercises (`open_prompt_injection`, `tensor_trust_hijack`, `tensor_trust_extract`, `injecagent`, `mmlu`, `gsm8k`, `ifeval`). If a numpy-2-specific failure appears during Phase 2/3, pinning `numpy<2.0.0` is an allowed technical fallback (`pin_compatible_package_version`) and does not require a new protocol version.
2. **Unsloth is unavailable.** `import unsloth` raises `NameError` standalone in this venv. This study's manifest already declares plain QLoRA via PEFT/TRL (`training.method = response_only_sft_qlora`), not an Unsloth-accelerated path, so this is a non-event for the frozen protocol — recorded so a reader does not assume Unsloth's 2x speedup applies to this study's reported training wall time.
3. **`tokenizers==0.23.1` and `safetensors==0.8.0` are installed vs. upstream's strict `0.22.1` and `0.7.0` pins.** The real-GPU smoke loaded the pinned tokenizer/model, ran every declared benchmark path, saved the PEFT adapter, wrote a 3.78 GB safetensors merged checkpoint, reloaded it, and generated successfully. No compatibility fallback was needed.
4. **Optional `torchao==0.15.0` C++ extensions are incompatible with the installed upstream-pinned `torch==2.8.0+cu128`.** Transformers reported that it skipped those extensions during both base and merged-model loading. This is the manifest-authorized `disable_optional_fused_kernels` path: the declared plain PEFT/bitsandbytes QLoRA step and HF inference completed without torchao, so no model, data, scorer, decoding, or training hyperparameter changed.
5. **`pynvml==13.0.1` emits a deprecation warning even though `nvidia-ml-py==13.590.44` is installed.** The runner's telemetry adapter calls the `nvidia-smi` CLI directly and captured 286 evaluation samples plus 112 training samples in the canonical qualification; it does not import or depend on pynvml. The warning therefore has no effect on the evidence.

Transformers resolved to `5.16.1`, which satisfies upstream's open-ended `>=4.51.0` constraint but warns that the upstream `torch_dtype` keyword is deprecated in favor of `dtype`. Both the base and merged model loaded and generated correctly; this is recorded as a compatibility observation, not a protocol deviation.

None of these observations changes any resolved model, dataset, prompt, decoding, scorer, or training-hyperparameter value in `manifest.json`.

## Manifest line endings and digest portability

The frozen manifest has two deliberately distinct digest identities.  The
canonical-JSON content digest is the checkout-invariant publication identity;
the raw-file digest in `protocol/manifest.sha256` is the byte-integrity value
used by `protocol.validate_manifest.sha256` and by recovery `StageSignature`
values.  Before `.gitattributes` pinned `protocol/**` to LF, a default Windows
checkout rewrote the raw bytes to CRLF and produced a false integrity mismatch.
That also made a recovery state written under one line-ending convention appear
incompatible under another.  The recorded raw digest and all finalized evidence
bundles remain unchanged; LF checkout policy now keeps future raw-file and
stage-signature identities portable.

## Canonical real-GPU qualification

The canonical issue-9 qualification is the run stamped `20260829-191237`. Its public
baseline, training, and projection artifacts are under `runs/` and are intentionally
excluded from version control. The projection artifact supersedes the two earlier
short-example exploratory projections: it measures one real optimizer step with a
response-only example that reaches the frozen 2048-token sequence ceiling.

The RTX 4080 run completed every declared benchmark path, a real bitsandbytes/PEFT
QLoRA step, PEFT merge, safetensors structural validation (320 tensors), and a fresh
HF reload whose recorded validation output was `READY`. The runner recursively
recorded SHA-256 digests for all six files in the merged checkpoint. The 2048-token
step completed without OOM, so the authorized 1536-token restart fallback was not
triggered; deliberately exhausting unrelated GPU memory would
have changed the qualified workload and was not used.

Projection from the canonical per-benchmark generation timings and maximum-length
training step is 14.925 GiB peak VRAM, 20.09 hours per seed, 65.79 total GPU-hours,
and 31.71 GiB for nine merged checkpoints. All are within the manifest limits, so
the canonical projection contains no feasibility findings. The visible publisher
integrity manifest also records upstream OPI's benchmark-specific behavior: `n=1`
publishes one row for each of its three injected tasks (three rows total), while the
qualification adapter deliberately consumes one row.

## Manifest/implementation drifts (issue #29)

Three gaps between the frozen `manifest.json` and what the code actually does.
They are recorded here (and in the issue #29 reproducibility disclosure) rather
than silently reconciled, because the manifest is frozen and the evidence was
produced against the code as it ran:

1. **The manifest names a multiple-choice scorer the pinned upstream does not
   use.** `evaluation.capability.mmlu.scorer = first_token_logit` with
   `max_new_tokens = 1`; the pinned upstream (`aar/benchmarks/mmlu/benchmark.py`,
   fingerprinted in `protocol/provenance.json`) scores MMLU by generated text,
   not a first-token logit ranking. This study's runner honours the manifest
   value, so MMLU alone is evaluated in a different modality from every other
   benchmark (see issue #28's modality-grouped primary table).
2. **A declared free-form decoding treatment is read by no code.**
   `decoding.freeform_treatment` (`no_repeat_ngram = 4`, `auto_ceiling = 1024`,
   `scope = "free-form judge-scored only"`) presupposes a judge-scored free-form
   path. This study uses only rule/logprob scorers (no paid or local judge), so
   nothing consults `freeform_treatment`; every benchmark uses the top-level
   `decoding` block.
3. **Decoding is applied once globally, not per benchmark as upstream
   documents.** Upstream resolves decoding parameters per benchmark; this
   study's runner applies the single top-level `decoding` block to every
   benchmark path. The per-benchmark `max_new_tokens` values in the manifest are
   still honoured, but the sampling parameters (`strategy`, `temperature`,
   `top_p`, `seed`, `no_repeat_ngram`) are global.

None of these changed a resolved model, dataset, prompt, or training
hyperparameter; they are disclosed so a reader does not assume the manifest text
and the executed code agree on scoring modality and decoding scope.

## InjecAgent source pin

`scripts/publish_suite.py` (`_publish_injecagent`) clones `https://github.com/uiuc-kang-lab/InjecAgent` with `git clone --depth 1` and no commit pin — every publish run takes whatever is at `HEAD` on the day it runs. Upstream does not fingerprint this dependency, so it is fingerprinted here instead: at protocol-freeze time (2026-08-29), `git ls-remote https://github.com/uiuc-kang-lab/InjecAgent HEAD` resolves to `f19c9f2c79a41046eb13c03c51a24c567a8ffa07`. The Phase 2 runner must record the actual cloned commit of `_injecagent_cache` inside every run's `manifest.yaml`/`environment.txt` (not assume this value), since InjecAgent's tool/task files feed the held-out benchmark's frozen candidate identifiers and any drift in that repository between baseline and trained runs would silently change what "the same held-out set" means.
