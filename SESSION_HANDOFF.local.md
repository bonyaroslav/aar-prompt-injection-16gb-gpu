# Research Session Handoff

> **Status:** Ready for a fresh session  
> **Repository state:** Planning only; local memory files are intentionally untracked.

## Read first

1. `RESEARCH_SPEC.local.md` — complete accepted decisions Q1–Q19.
2. `RESEARCH_EXECUTION_PLAN.local.md` — Draft, First Iteration; plain-language phases, evidence, completion rules, and article plan.
3. `RESEARCH_PLAN.md` — original research description.

## Current goal

Determine whether one person can complete and publish an auditable prompt-injection post-training experiment using `Qwen/Qwen3.5-2B` on one RTX 4080-class 16 GB GPU.

The main outcome is hardware and workflow feasibility. Safety improvement, capability preservation, held-out transfer, and repeatability are separate outcomes. Null and negative findings remain publishable.

## Settled decisions

- Q1–Q19 in the local spec are accepted.
- Use the deep experiment-runner seam: one frozen manifest in, one immutable checksummed evidence bundle out.
- Use QLoRA response-only supervised fine-tuning with the frozen data mix, adapter settings, optimizer, checkpoint-selection rule, capability gates, bootstrap method, and seeds in the spec.
- Keep InjecAgent sealed until checkpoint selection is final.
- Preserve failed attempts as evidence.
- Write for solo ML practitioners, applied safety researchers, reproducibility-minded open-source engineers, and technically curious readers who may not know the specialist terminology.
- Plan a series of eight short, evidence-led articles using plain language.

## Guardrails

- Treat `RESEARCH_SPEC.local.md` as authoritative.
- Freeze protocol values before the baseline.
- Do not inspect InjecAgent content or results during data creation, tuning, or checkpoint selection.
- Do not change quality-related settings after viewing results.
- Keep these local memory files untracked unless the user explicitly changes that instruction.

## Next task

Execute Phase 1 only: design and write the frozen protocol manifest and held-out sealing procedure. Before implementation, inspect the pinned upstream checkout to resolve exact prompt, decoding, dataset, and scorer behavior. Report any conflict between upstream facts and the accepted spec before changing the protocol.

## Completion criterion for the next task

Phase 1 is complete only when every required protocol value is explicit, provenance is fingerprinted, allowed fallbacks are listed, held-out access is mechanically controlled, and the manifest can be validated without running a full benchmark.
