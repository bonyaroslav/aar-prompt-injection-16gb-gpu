# Phase 1 upstream reconciliation

The pinned upstream checkout is authoritative for prompt text, roles, decoding and per-benchmark scorer behavior. Its committed `HEAD` is `1899ad64fbfbc65790d259471cc4bf4de9437aa9`; the checkout also has unrelated dirty/untracked prior artifacts, which are excluded from provenance.

The accepted local protocol does not silently inherit three upstream choices:

1. Upstream reports a geometric mean of closed fractions; this study selects on the unweighted mean of absolute improvements across the three visible safety benchmarks and requires +5 percentage points.
2. Upstream capability filtering uses CI-overlap. This study uses the accepted absolute per-gate declines and 98% mean normalized retention.
3. Upstream rule scoring skips failed/empty generations. Held-out sealing preserves every invalid turn and reports both valid-only and intent-to-evaluate, where technical invalids are failures.

These are predeclared local analysis/protocol deviations, not post-result changes. The OPI combine-only attack, 300 published candidates per visible benchmark, 200 InjecAgent candidates, real system/user roles, and all decoding values are resolved from the pinned committed docs/code and exported in `manifest.json`.

## Environment deviations from `pyproject.toml`

The upstream `pyproject.toml` pins a much larger stack (Unsloth-accelerated training, vLLM/sglang serving, VERL/Ray RL) than this study's declared response-only QLoRA workload requires. The project venv actually installed and CUDA-functional at protocol-freeze time (`C:\Projects\automated_alignment_researcher\.venv`, Python 3.12.14) differs from that pin in two ways, recorded here rather than left implicit:

1. **`numpy==2.5.2` installed vs. upstream's pinned `numpy<2.0.0`.** No numpy-2 incompatibility has been observed in the benchmarks this study exercises (`open_prompt_injection`, `tensor_trust_hijack`, `tensor_trust_extract`, `injecagent`, `mmlu`, `gsm8k`, `ifeval`). If a numpy-2-specific failure appears during Phase 2/3, pinning `numpy<2.0.0` is an allowed technical fallback (`pin_compatible_package_version`) and does not require a new protocol version.
2. **Unsloth is unavailable.** `import unsloth` raises `NameError` standalone in this venv. This study's manifest already declares plain QLoRA via PEFT/TRL (`training.method = response_only_sft_qlora`), not an Unsloth-accelerated path, so this is a non-event for the frozen protocol — recorded so a reader does not assume Unsloth's 2x speedup applies to this study's reported training wall time.

Neither point changes any resolved model, dataset, prompt, decoding, scorer, or training-hyperparameter value in `manifest.json`.

## InjecAgent source pin

`scripts/publish_suite.py` (`_publish_injecagent`) clones `https://github.com/uiuc-kang-lab/InjecAgent` with `git clone --depth 1` and no commit pin — every publish run takes whatever is at `HEAD` on the day it runs. Upstream does not fingerprint this dependency, so it is fingerprinted here instead: at protocol-freeze time (2026-08-29), `git ls-remote https://github.com/uiuc-kang-lab/InjecAgent HEAD` resolves to `f19c9f2c79a41046eb13c03c51a24c567a8ffa07`. The Phase 2 runner must record the actual cloned commit of `_injecagent_cache` inside every run's `manifest.yaml`/`environment.txt` (not assume this value), since InjecAgent's tool/task files feed the held-out benchmark's frozen candidate identifiers and any drift in that repository between baseline and trained runs would silently change what "the same held-out set" means.
