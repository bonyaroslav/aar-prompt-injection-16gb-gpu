# Independent review prompt

Paste everything below the line into a fresh agent session opened at the repo root.

---

You are an independent reviewer. Be adversarial and specific. I want to know where
this project is wrong, where it is weak, and how it should be published. Do not be
polite about it.

## The original goal

The study asked one question: **can a single transparent response-only QLoRA
fine-tune reduce a 2B model's susceptibility to prompt injection, on one 16 GB
consumer GPU, without paid judges, and without breaking the model?**

It was designed to follow a preregistered evidence contract rather than to produce a
good score:

- A protocol frozen before any GPU time (`protocol/manifest.json`) — model, revision,
  benchmarks, sample IDs, decoding, training hyperparameters, capability gates,
  selection rule, and resource caps all fixed in advance.
- A frozen untrained baseline, then three replicated runs (seeds 17, 42, 2026),
  three epochs each, evaluated on the same fixed example IDs.
- Capability gates that could reject a checkpoint no matter how good its safety
  score looked.
- A held-out benchmark (InjecAgent, 200 candidates) sealed before execution and
  openable only after a checkpoint was selected.
- A hard budget: 72 GPU-hours, one GPU, no manual intervention after launch.

## What was actually run

- Baseline plus 3 seeds x 3 epochs = 9 trained checkpoints.
- Six visible benchmarks: Open Prompt Injection, Tensor Trust hijack, Tensor Trust
  extract, MMLU, GSM8K, IFEval.
- Outcome: all nine checkpoints failed the frozen capability gates. Three null
  selections. The held-out benchmark was never unsealed.
- Two follow-ups: a chat-mode MMLU diagnostic (issue #30) and a clean-corpus
  ablation with zero injection rows (issue #31).
- About 59.9 of the 72 GPU-hours used.

## Where the evidence is

**Read `docs/independent-review-preflight.md` first.** It carries verified environment
facts, a map of the code, the exact shape of the raw evidence, and hard guardrails. It
was written by a session that had already read the handover, so it deliberately contains
no findings and no steer — only things that save you search time.

Then read these. Everything is committed and regenerable.

| Path | What it holds |
|---|---|
| `README.md` | the repo's front door: result table, non-claims, disclosed defects |
| `RESEARCH_PLAN.md` | the study plan |
| `protocol/manifest.json` | the frozen protocol |
| `protocol/deviations.md` | known drifts between the manifest and the code |
| `protocol/heldout_sealing.md` | the sealing and reveal procedure |
| `analysis/attempt1-claim-report.json` | per-benchmark deltas, paired bootstrap, exact McNemar, composites |
| `analysis/attempt1-integrity-report.json` | failure-mode evidence and the disclosure list |
| `analysis/publication-provenance-manifest.json` | a receipt for every reported number |
| `analysis/seed{17,42,2026}-outcomes-summary.md` | per-seed results |
| `docs/issue-33-claim-framing-dossier.md` | the fact base with provenance |
| `docs/issue-33-interpretations.md` | six candidate readings of the same evidence |
| `docs/issue-33-validation-guide.md` | the project's own validation recipe |
| `docs/adr/0002-issue-33-claim-framing.md` | five open wording decisions, unsigned |
| `runner/`, `training_data/`, `scripts/` | the code that produced every number |
| `tests/` | 29 test modules |
| `runs/**/metrics.json` | raw per-item scores (gitignored, present locally) |
| `C:\Projects\automated_alignment_researcher` | the pinned upstream repo — benchmarks, scorers, parsers |

Regenerate the fact base and confirm it is byte-identical:

```bash
python -m runner.publication_gate_run --dump-reports analysis --out analysis/publication-provenance-manifest.json
```

## Rules

1. **Form your own view before reading anyone else's.** Do not open anything under
   `docs/handover/` — four files: `analysis-handover-post-seed3.md`,
   `issue-30-handover.md`, `issue-31-handover.md`, `publication-analysis-final.md` —
   until sections 1–5 and 7 of your review are written to disk. Then read them and add
   the section listing where you agree, where you disagree, and what they missed.
   The whole point of this exercise is that you were not anchored.
2. **Check numbers against raw evidence,** not against the prose that quotes them.
   Recompute from `runs/**/metrics.json` where you can. The pre-flight's §5 says what
   the full cross-product is; derive first, compare second.
3. **Read the code, not just the reports.** Several claims depend on how a scorer or
   a training loop actually behaves. The upstream repo is on disk at the pinned commit
   — read it through `git show <pin>:<path>`, not through its dirty working tree.
4. **Say "I could not verify this"** rather than guessing. Mark every external claim
   as verified, unverified, or contradicted. Note that per-item records carry `score`
   and `valid` only — no generated text is retained anywhere, so any claim about what
   the model actually wrote is not re-derivable offline. Say so where it applies.
5. **Do not invent scores.** Cite no number this project did not measure.
6. **Do not unseal the held-out benchmark,** and do not read `_holdout_medium/` or
   `_holdout_opi_full/` in the upstream checkout. `protocol/heldout_sealing.md` governs;
   read it in full before writing about the held-out design. Stay read-only on `runs/`,
   `recovery/`, `analysis/` and `protocol/` — the one permitted write is the
   regeneration command above, whose purpose is to produce an empty diff. No GPU: this
   review scopes and costs future experiments, it does not run them.
7. **Write in your own voice, not the project's.** This review is a critique, not a
   published claim, so it is exempt from the project's claim-language gate
   (`runner.publication_gates.check_claim_language`, which bans "robust", "secure",
   "resistant", "mitigation", "defense that works"). Quote attacks in the words they
   would really be made in. Instead, add a short subsection listing any wording in your
   own recommendations that would fail that gate if carried into `analysis/results.md`.

## What I want from you

### 1. Gaps and criticism

What is missing, overstated, under-controlled, or simply wrong? Cover at minimum:
the training setup, the corpus, the benchmark choices, the scoring rules, the
statistics and sample sizes, the confounds, the reproducibility story, and the
resource claims. Rank by how much each one actually matters.

### 2. How this gets attacked in public

Assume this is posted to Reddit, Hacker News and Hugging Face, and read by people
who enjoy finding holes. Write the attacks in the voice they would actually be
made — "your data is garbage", "you measured the wrong thing", "your baseline is
implausible", "this is just X rediscovered", "you can't claim that from n=3".

For **each** attack, give me three things:

- **Where they are right.** State it plainly. Do not defend the indefensible.
- **Where they are wrong or overreaching**, with the evidence that answers them.
- **The gap** — what would have to be measured or changed to close it, and roughly
  what that costs.

Include attacks about data quality, test/benchmark validity, statistical claims,
and prior art specifically. I would rather hear the worst version now.

### 3. Prior art

Has this already been published? Search properly, including 2026 work. If someone
has already made the same argument, say so directly and tell me what, if anything,
is left. Give URLs.

### 4. How to publish it for a non-specialist audience

This is the part I care most about. The reader is an **enthusiast**, not a
researcher: they run local models, they have fine-tuned something once, they know
what a benchmark is, they do **not** know what IFEval is, they have never heard of
Tensor Trust, and they will leave in fifteen seconds if the opening is a methods
section.

Give me:

- **Five title options**, in plain language. No jargon in the title. Say which venue
  each one suits and why.
- **A section-by-section structure** with a target word count per section and one
  sentence describing what goes in it. Total 900–1,300 words.
- **The one image** the piece needs, described precisely enough to build.
- **One analogy** that explains the core result to someone who does not know how
  benchmarks are scored — and say honestly where the analogy breaks down.
- **The opening two sentences**, written out in full.
- **A "what I am not saying" list** in plain language.
- **Which venues to use, in what order**, and what to expect from each.

Avoid leading with "safety", "alignment", "negative result", or "preregistered" —
those words lose this audience. Lead with something concrete.

## Links to validate against

Verify each URL before citing it; some are preprints whose status may have changed.

**The upstream work this study replicates in miniature**

- https://alignment.anthropic.com/2026/automated-alignment-researchers/
- https://github.com/YuehHanChen/automated_alignment_researcher

**Prompt-injection defences**

- Meta SecAlign — https://arxiv.org/abs/2507.02735
- SecAlign (CCS 2025) — https://dl.acm.org/doi/10.1145/3719027.3744836
- StruQ (USENIX Sec 2025) — https://www.usenix.org/conference/usenixsecurity25/presentation/chen-sizhe
- DefensiveTokens (AISec 2025) — https://doi.org/10.1145/3733799.3762982
- Instruction Hierarchy — https://arxiv.org/abs/2404.13208
- SecFid, security–fidelity tradeoffs — https://arxiv.org/html/2606.30783v1
- ARGUS — https://arxiv.org/html/2605.03378v1

**Attacks and evaluation critiques**

- Checkpoint-GCG — https://arxiv.org/abs/2505.15738
- Breaking Fine-Tuning based Prompt Injection Defenses — https://arxiv.org/abs/2507.07417
- The Attacker Moves Second (USENIX Sec 2026) — https://www.usenix.org/conference/usenixsecurity26/presentation/nasr
- Adaptive Attacks Break Defenses (NAACL Findings 2025) — https://aclanthology.org/2025.findings-naacl.395/
- PIEval — https://arxiv.org/abs/2505.18333
- PISmith — https://arxiv.org/pdf/2603.13026
- LivePI — https://arxiv.org/pdf/2605.17986
- AI Agents May Always Fall for Prompt Injections — https://arxiv.org/pdf/2605.17634
- Defenses Learn Surface Heuristics (ACL 2026) — https://aclanthology.org/2026.acl-long.502/
- InjecGuard / over-defense — https://arxiv.org/pdf/2410.22770

**The benchmarks used here**

- Open Prompt Injection (USENIX Sec 2024) — https://www.usenix.org/conference/usenixsecurity24/presentation/liu-yupei
- InjecAgent (ACL Findings 2024) — https://aclanthology.org/2024.findings-acl.624/
- ExploitBench, graded scoring — https://arxiv.org/html/2605.14153v1

**Evaluation methodology**

- Answer Matching Outperforms Multiple Choice — https://arxiv.org/abs/2507.02856
- Right Answer, Wrong Score — https://arxiv.org/abs/2503.14996
- Multiple Choice Normalization (EleutherAI) — https://blog.eleuther.ai/multiple-choice-normalization/
- Artifacts or Abduction — https://arxiv.org/abs/2402.12483
- Multiple-Choice Questions are Efficient and Robust Evaluators (the opposing view) — https://arxiv.org/abs/2405.11966
- Lessons from the Trenches on Reproducible Evaluation — https://arxiv.org/abs/2405.14782
- When Benchmarks are Targets — https://arxiv.org/abs/2402.01781

**Fine-tuning damage and run-to-run variance**

- LoRA Learns Less and Forgets Less (TMLR) — https://arxiv.org/abs/2405.09673
- Slimming Down LLMs Without Losing Their Minds — https://arxiv.org/abs/2506.10885
- Mitigating the Alignment Tax of RLHF — https://arxiv.org/abs/2309.06256
- Macro/Micro Effects of Random Seeds — https://arxiv.org/abs/2503.07329

**Standards and venues**

- NeurIPS paper checklist — https://neurips.cc/Conferences/2021/PaperInformation/PaperChecklist
- ACL reproducibility checklist — https://2021.aclweb.org/calls/reproducibility-checklist/
- TMLR acceptance criteria — https://jmlr.org/tmlr/acceptance-criteria.html
- Hugging Face blog-explorers — https://huggingface.co/blog-explorers

## Output

Write one Markdown file, `docs/independent-review.md`, with these sections in order:

1. **Verdict** — ten lines. What is sound, what is not, what it is worth.
2. **Gaps and criticism**, ranked.
3. **Attack table** — one row per attack: the attack, where they are right, where
   they are wrong, the gap, and the cost to close it.
4. **Prior art**, with URLs and a verified / unverified / contradicted mark on each.
5. **The non-specialist publication plan**, per the shape requested above.
6. **Where I disagree with `docs/handover/publication-analysis-final.md`** — written
   last, after you have committed to everything above.
7. **ADR 0002 disposition.**

Keep it concrete. Every claim gets a file path, a line number, or a URL.

### On section 7

`docs/adr/0002-issue-33-claim-framing.md` is frozen at `Status: OPEN — verdict deferred
to an independent validation pass`. You are that pass; issue #33 cannot close without it.
So:

- Work its 8-item validation checklist (`docs/adr/0002-issue-33-claim-framing.md:101-125`)
  and report the evidence for each item, pass or fail.
- Give a recommended verdict and a one-line rationale for each of **D1 through D5**, laid
  out in the ADR's own sign-off table shape (`:155-161`). Recommend; the maintainer signs.
- Take the option space from the ADR itself and from the evidence you verified. Take no
  cue from anywhere else.

### Order of writing

Write sections 1–5 and 7 to disk **first**. Only then open `docs/handover/` and write
section 6. Section 6 is worth nothing if the sections above it were written after you
knew what the handover said — and it will be obvious to a reader if they were.
