# Project reading map

**Updated: 13 September 2026.** An index of the experiment, evidence, publication scope, and earlier decisions. The final scope is the existing README, checked fact sheets, and numeric reanalysis package for A and B1. Separate articles, a website, and new experiments are outside this scope.

**Repository:** https://github.com/bonyaroslav/aar-prompt-injection-16gb-gpu

All local file links below contain **absolute paths on this PC**. An agent reading the Markdown source can use the path inside the link parentheses with its filesystem tools; forward slashes work on Windows. Paths containing spaces are enclosed in angle brackets. Uploading this map alone does not upload the linked files: an agent elsewhere can answer from the orientation, but needs the relevant files supplied before checking their contents. The `G:` archive may require the mounted drive and filesystem permission.

## Contents

1. [Start here](#start-here)
2. [Choose a reading route](#choose-a-reading-route)
3. [Current conclusions and prepared materials](#current-conclusions-and-prepared-materials)
4. [Evidence and implementation](#evidence-and-implementation)
5. [Scientific work and useful links](#scientific-work-and-useful-links)
6. [How the publication thinking developed](#how-the-publication-thinking-developed)
7. [Technical decisions and execution history](#technical-decisions-and-execution-history)
8. [Corrections to carry into every answer](#corrections-to-carry-into-every-answer)
9. [Scope and maintenance](#scope-and-maintenance)

## Start here

The project tested **one response-only QLoRA training recipe on Qwen3.5-2B using one RTX 4080**. It inherited a prompt-injection evaluation setting from Anthropic's Automated Alignment Researcher work; it did not reproduce that study's iterative method search.

Three nominal runs produced three checkpoints each, compared with one shared baseline. **All nine checkpoints increased the visible Open Prompt Injection (OPI) score, but failed the generation-scored GSM8K and IFEval acceptance limits. None was selected, and no held-out comparison was revealed.** This is a negative selection result for this recipe, not evidence that fine-tuning generally fails.

The current publication directions are:

| Name used in current documents | Plain-language question | Present evidence boundary |
|---|---|---|
| **Direction A — checkpoint acceptance** | Did the update meet all the requirements, even when one score improved? | Reported in the README and checked fact sheet, with numeric reanalysis. |
| **Direction B1 — scoring audit** | What does a favorable injection score actually tell us? | Code and retained scores support an audit of what is identifiable and what information was lost. |
| **Direction B2 — behavioral study** | Did the checkpoints resist the attack while completing the legitimate job? | Requires new output-retaining measurements. Historical outputs were not retained. |
| **Direction C — reproducibility/engineering** | How can a small local experiment be audited and recovered? | An alternative proposed in the earlier decision memo; not a third completed article. |

**Naming trap:** “Angle A” in the September 1 V4 document meant a different, evaluation-format-centered argument. Call it **V4 Angle A** when discussing history. It is not today's Direction A. Likewise, “F1/F2” in the September 4 handover are historical interpretations, not automatically established findings.

For a newcomer, a checkpoint is a saved version of the trained model; an acceptance gate is a requirement that version must pass. OPI tests attack-target mismatch. Tensor Trust (TT) combines attack and authorized-control checks. GSM8K, IFEval, and MMLU are different task tests, with different scoring methods. Their percentages are not interchangeable measures of general usefulness.

### How an answering agent should use the sources

Start with the current fact sheet for the question. Follow its evidence links when the answer depends on a precise number, scoring rule, or contested explanation. Cite the specific file or scientific source used. Keep observed results, calculations, explanations, and proposals distinct.

Use **recorded observations and executed code** to resolve factual disagreements. The frozen protocol establishes what was declared; the implementation establishes what ran. Current fact sheets provide corrected synthesis. Historical handovers explain how thinking developed. A newer date, a “verified” label, or a checksum does not make an interpretation correct.

Historical prompts and implementation plans are reference material, not fresh instructions to run experiments or publish anything. Answering a question does not require opening sealed data. When a linked source is unavailable, state that access limit instead of claiming to have checked it.

## Choose a reading route

| Question or audience | Read first | Follow only if needed |
|---|---|---|
| Newcomer: “What happened?” | [A fact sheet](publication/direction-a/fact-sheet.md) | [Five-minute walkthrough](publication/direction-a/README.md) |
| “Why reject every checkpoint?” | [Calculated decisions](publication/direction-a/checkpoint-decisions.csv) | [Protocol thresholds](../protocol/manifest.json), [selection code](../runner/selection.py) |
| “Did the model refuse or ignore the document?” | [B scoring fact sheet](publication/direction-a/scorer-fact-sheet.md) | [Scorer](../runner/real_adapters.py), [TT audit](publication/direction-a/tensor-trust-audit.json) |
| Expert: “Can I verify the numbers?” | [Package validation](publication/direction-a/validation-notes.md) | [Standalone script](publication/direction-a/reproduce.py), then the evidence section below |
| “What can A and B offer, and to whom?” | [A/B knowledge base](publication-directions-a-b-knowledge-base.md) | Its claim boundaries, audiences, article presentation, and anticipated criticism sections |
| “What should the article look like?” | [Publication examples](publication/publication-examples.md) | Current fact sheets; older titles are inspiration only |
| “How does this relate to SecFid or Anthropic?” | [Scientific work below](#scientific-work-and-useful-links) | Original papers and the knowledge base's scientific positioning |
| “What did the later diagnostic/ablation establish?” | [Issue 30 diagnostic decision](issue-30-chatmode-mmlu-diagnostic-decision.md), [issue 31 ablation decision](issue-31-corpus-ablation-decision.md) | Their separate protocols; apply the correction notes below before drawing causal conclusions |
| “Why did earlier advice change?” | [Publication history below](#how-the-publication-thinking-developed) | [Correction ledger below](#corrections-to-carry-into-every-answer) |
| Engineer: “How did recovery or provenance work?” | [Technical history below](#technical-decisions-and-execution-history) | Corresponding implementation and tests; old task lists are not current execution requests |

## Current conclusions and prepared materials

These are the starting documents for present-day answers. The B fact sheet lives in the `direction-a` package because it reuses the same visible numeric export.

| File | Status and purpose | Relationship / when to open |
|---|---|---|
| [Repository README](../README.md) | Current project overview with corrected framing. | Quick project introduction; details live in the fact sheets. |
| [Public reader's guide](START_HERE.md) | Short entry point for GitHub visitors. | Routes readers to findings, scoring limits, and reproduction instructions. |
| [Repository presentation](publication/repository-presentation.md) | About description, topics, and public entry point. | Metadata; no new scientific results. |
| [Citation metadata](../CITATION.cff) / [MIT license](../LICENSE) / [license scope](../LICENSE_SCOPE.md) | Repository attribution and reuse terms for original contributions. | Includes the two scientific foundations and preserves third-party licensing boundaries. |
| [Directions A/B knowledge base](publication-directions-a-b-knowledge-base.md) | Current broad decision support, including plain-language explanations. | Central source for audiences, claim limits, article logic, B1/B2 distinction, and proposed next measurements. |
| [Direction A fact sheet](publication/direction-a/fact-sheet.md) | Checked quantitative drafting reference. | All ten model states, rules, costs, limits, and one worked rejection. Generated by the standalone script. |
| [Direction B scoring fact sheet](publication/direction-a/scorer-fact-sheet.md) | Checked code/score interpretation with a simple example. | Explains half-score ambiguity, actual counts, logical bounds, and what cannot be reconstructed. |
| [Walkthrough and package README](publication/direction-a/README.md) | Prepared small steps, reasons, commands, and file descriptions. | Answers “what is a fact sheet?” and “how was this checked?” without another preparation checklist. |
| [One figure: PNG](publication/direction-a/figure.png) / [SVG](publication/direction-a/figure.svg) | Same six-panel figure in two formats. | Shows all runs/epochs and individual task floors; SVG is useful for article layout. |
| [Publication examples](publication/publication-examples.md) | Current reading guide with concrete sections to examine. | Presentation inspiration, with explanations of what transfers and what does not. |

**Current boundary:** the existing A and B1 materials are the chosen publication scope. B2 is deferred. The [issue-33 ADR](adr/0002-issue-33-claim-framing.md) records this decision and preserves the superseded checklist without marking it completed.

## Evidence and implementation

### Small, self-contained reanalysis

Use this package before loading large raw run directories. It checks saved observations; it does not reproduce training or recover missing output text.

| File | What it contains / what to verify |
|---|---|
| [Visible scores](publication/direction-a/visible-scores.csv) | 16,000 numeric item outcomes across the shared baseline and nine checkpoints. No prompts, completions, or held-out outcomes. |
| [Context and source identities](publication/direction-a/context.json) | Model/configuration, original source paths and hashes, reference means, intervals, and selection records. |
| [All scores](publication/direction-a/all-scores.csv) | Six baseline means plus 54 trained means. |
| [Checkpoint decisions](publication/direction-a/checkpoint-decisions.csv) | Losses, retention, failed gates, and eligibility for every checkpoint. |
| [Paired intervals](publication/direction-a/paired-intervals.csv) | 54 benchmark-level change intervals; uncertainty over sampled item pairs, not a training-population claim. |
| [Tensor Trust audit](publication/direction-a/tensor-trust-audit.json) | All 18 category comparisons, paired matrices, and component identification bounds. No refusal labels. |
| [Standalone reproduction script](publication/direction-a/reproduce.py) / [plotting dependency](publication/direction-a/requirements.txt) | Python-only numeric reanalysis; optional plotting and bootstrap flags. See the package README for commands. |
| [Maintainer export script](publication/direction-a/export_from_repository.py) | Extracts the visible subset from original local bundles. Unlike the reader script, requires those bundles. |
| [Validation notes](publication/direction-a/validation-notes.md) | What was actually checked, including the separate-folder and standard-library runs. |
| [Dated validation receipt](publication/direction-a/validation-receipt.json) / [latest invocation record](publication/direction-a/verification.json) | Preparation receipt binds script/data hashes. The latest invocation record may change when someone reruns a shorter command. |

### Original declarations and frozen analysis

| File | Role / caution |
|---|---|
| [Research plan](../RESEARCH_PLAN.md) | Original goals and accumulated project log. Separate planned work from completed records. |
| [Protocol manifest](../protocol/manifest.json) | Declared recipe, sample sizes, decoding, thresholds, model/upstream pins, and budgets. |
| [Protocol digests](../protocol/digests.md) / [provenance](../protocol/provenance.json) | Identity and origin of declared inputs. A digest proves identity, not correctness. |
| [Deviations](../protocol/deviations.md) | Historical disclosure record. Its upstream-MMLU scoring statement is superseded by the correction below. |
| [Held-out sealing policy](../protocol/heldout_sealing.md) | Declared separation and reveal procedure. Consult policy rather than reading sealed outcomes. |
| [Power notes](../protocol/power_notes.md) | Planning approximation; does not establish achieved paired power or prove the test could never be informative. |
| [Claim report](../analysis/attempt1-claim-report.json) | Historical means, differences, paired analyses, and summaries. Use for exact report comparisons. |
| [Integrity report](../analysis/attempt1-integrity-report.json) | Counts, resource accounting, and disclosures mixed with interpretations. Read with the B fact sheet: category-based causal verdicts are too strong. |
| [Frozen input record](../analysis/attempt1-frozen-input-record.json) / [publication provenance](../analysis/publication-provenance-manifest.json) | Original evidence identities and numeric receipts. |
| [Analysis configuration](../analysis/analysis-config.json) | Declared statistical analysis settings. |
| [Run 17 summary](../analysis/seed17-outcomes-summary.md), [run 42 summary](../analysis/seed42-outcomes-summary.md), [run 2026 summary](../analysis/seed2026-outcomes-summary.md) | Per-run selection records in readable form; three trajectories share one baseline. |

### Code to consult for expert questions

| Question | File |
|---|---|
| How are outputs scored? | [Real adapters/scorers](../runner/real_adapters.py) |
| What is retained from evaluation? | [Evaluation](../runner/evaluation.py) |
| How are eligibility and selection calculated? | [Selection](../runner/selection.py) |
| How did training and adapter merging run? | [Real training](../runner/real_training.py) |
| How were data sources and exclusions constructed? | [Data-source ADR](adr/0001-training-data-sources.md), [corpus builder](../training_data/build.py), [exclusion pool](../training_data/exclusion_pool.py) |
| How were the original statistics/reports assembled? | [Analysis functions](../runner/analysis.py), [claim tables](../runner/claim_tables.py), [integrity report builder](../runner/integrity_report.py) |
| Why does full historical regeneration need private inputs? | [Report assembly](../runner/publication_gate_run.py), [input loading](../runner/publication_report_inputs.py) |
| How are provenance and claim wording checked? | [Publication gates](../runner/publication_gates.py) |
| What tests cover scoring, selection, and reporting? | [Scorer tests](../tests/test_real_adapters.py), [selection tests](../tests/test_selection.py), [publication tests](../tests/test_publication_gate_run.py) |

For a raw-data investigation, `context.json` and the frozen input record identify the needed files under the repository's local `runs/` and related evidence directories. This map intentionally does not enumerate model weights or sealed benchmark items. When comparing implementations, use the upstream harness at its **pinned commit** [`1899ad6`](https://github.com/YuehHanChen/automated_alignment_researcher/tree/1899ad64fbfbc65790d259471cc4bf4de9437aa9); a working checkout may contain unrelated changes.

## Scientific work and useful links

### Central references and their relationship to the project

**Anthropic AAR** supplies the inherited model/benchmark setting and selection context for A. **SecFid** supplies the closest behavioral distinctions for B. Neither publication establishes what caused these local checkpoints' results. The [current knowledge base](publication-directions-a-b-knowledge-base.md) explains those boundaries; the [publication reading guide](publication/publication-examples.md) points to useful sections.

For the exact inherited implementation, consult the [pinned upstream repository](https://github.com/YuehHanChen/automated_alignment_researcher/tree/1899ad64fbfbc65790d259471cc4bf4de9437aa9), [benchmark explanation](https://github.com/YuehHanChen/automated_alignment_researcher/blob/1899ad64fbfbc65790d259471cc4bf4de9437aa9/benchmark_docs/prompt_injection/bench_explanation.md), and [MMLU scorer](https://github.com/YuehHanChen/automated_alignment_researcher/blob/1899ad64fbfbc65790d259471cc4bf4de9437aa9/aar/benchmarks/mmlu/benchmark.py).

### Useful links

Primary-source links checked on **6 September 2026**. Dates below are initial arXiv submission dates unless stated otherwise, not necessarily conference dates. This is a focused reading list, not an exhaustive novelty review or a comparison of published performance.

| Scientific work | Date | Why useful / how it relates |
|---|---|---|
| [Security–Fidelity Tradeoffs: The Hidden Cost of Prompt Injection Defense (SecFid)](https://arxiv.org/abs/2606.30783) | 29 June 2026 | Closest B reference: distinguish executing instructions from processing or suppressing their text. Cannot label this project's missing historical outputs. |
| [Automated Researchers Can Mitigate Well-Characterized Alignment Failures](https://alignment.anthropic.com/2026/automated-alignment-researchers/) | 2026; exact day not verified in this link check | Original AAR research setting. Its iterative method search and acceptance rules differ from this one-recipe study. |
| [StruQ: Defending Against Prompt Injection with Structured Queries](https://arxiv.org/abs/2402.06363) | 9 February 2024 | Instruction/data separation through structure and training; methodological background, not a locally tested baseline. |
| [SecAlign: Defending Against Prompt Injection with Preference Optimization](https://arxiv.org/abs/2410.05451) | 7 October 2024 | Preference-based training prior work; helps distinguish proposed training ideas from established methods. |
| [Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks](https://arxiv.org/abs/2507.02735) | 3 July 2025 | Training and security/utility evaluation context. Different model/task settings prevent direct comparison with the local scores. |
| [Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game](https://arxiv.org/abs/2311.01011) | 2 November 2023 | Original extraction/hijacking benchmark family. Use the pinned local/AAR code for the exact saved-score semantics here. |
| [Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) | 23 February 2023 | Foundational threat-model explanation for instructions embedded in external data; useful for newcomers. |

### Presentation examples and local source notes

| Source | Role / boundary |
|---|---|
| [What's going on with the Open LLM Leaderboard?](https://huggingface.co/blog/open-llm-leaderboard-mmlu) | Technical explanatory example for B: concrete scoring/prompt comparisons. Not evidence of a cause in this experiment. |
| [Some negative steganography results](https://www.lesswrong.com/posts/EEvsL9cpgDAxAhTzt/some-negative-steganography-results) | Negative-results writing example for A; a different research problem, not a peer-reviewed validation of this work. |
| Scientific source verification notes *(local only)* | September 6 research notes behind the revised decision. Follow the primary sources for exact claims. |
| Audience/source verification notes *(local only)* | Qualitative audience/platform observations, not measured demand or promised reach. Recheck platform rules before acting. |
| Local AAR PDF *(local only)* | Archived paper copy. File existence checked; this indexing pass did not compare its contents with the live version. |

## How the publication thinking developed

Order below follows documented dates and logical dependencies. Same-day ordering reflects the relationship between documents, not an inferred creation timestamp. Current framing is in the September 6 fact sheets and knowledge base; older documents remain useful for understanding questions and abandoned proposals.

### Earlier planning archive on G:

These files were inventoried and their introductory status notes read. Existing September 5 “still valid” banners are historical assessments, not a fresh endorsement of every claim or venue rule.

| Stage | File | Role and present relationship |
|---|---|---|
| Aug 30: original/local boundary | AAR original vs local RTX 4080 facts *(local only)* | Early, seed-17-era comparison; partly stale. Current A fact sheet covers all runs. |
| Aug 30 source cut: publication norms | How to publish local LLM research *(local only)* | Ukrainian-language platform/advice reference. Rules and audience claims require fresh checking before publication. |
| Aug 31: evidence requirements | Minimum paper requirements *(local only)* | Ukrainian-language methodology guidance; proposed standards, not evidence that this project satisfies every requirement. |
| Aug 31: literature search | Closest prompt-injection research *(local only)* | Incomplete sweep that missed SecFid. Useful as a discovery list; superseded for positioning by the current scientific section. |
| Aug 31: early synthesis | Publication research handover *(local only)* | Two-run context and early suggestions. Completed three-run evidence and current claim limits take precedence. |
| Sep 1: V4 proposal | Publication-angle decision V4 *(local only)* | Partly retracted evaluation-format-centered proposal. Its “Angle A” is different from today's A. |
| Sep 1: critique of V4 | Prepublication falsification audit *(local only)* | Read alongside V4 to understand challenges. Mechanistic interpretations still require the current corrections. |
| Sep 1: packaging V4 | Titles, structures, reading list *(local only)* | Companion to V4; old titles can carry retired claims. Use current publication examples for drafting. |
| Historical checking tool | Prepublication checks script *(local only)* | Archive tool, not the current standalone reproduction path. Consult only for historical methodology; not rerun for this map. |

### Consolidation, review, and the current A/B package

| Stage | File | What changed / where it leads |
|---|---|---|
| Sep 3: completed-run handover | [Analysis handover after seed 3](handover/analysis-handover-post-seed3.md) | Consolidates completed runs and follow-ups; defers framing. Later summaries and current corrections qualify its interpretations. |
| Sep 3: framing evidence | [Issue-33 dossier](issue-33-claim-framing-dossier.md) | Historical assembled facts and decision questions; contains prose requiring the correction ledger. |
| Same decision set: alternatives | [Six interpretations](issue-33-interpretations.md) | Candidate theses, not six established results. Read to understand the options considered. |
| Same decision set: formal record | [Framing ADR](adr/0002-issue-33-claim-framing.md) | September 13 scope decision, with historical options, hashes, and uncompleted checklist preserved. |
| Same decision set: validation | [Original validation guide](issue-33-validation-guide.md) | Full local-input report regeneration. Some explanatory prose is superseded; current small package is independently runnable. |
| Sep 4: stronger publication recommendation | [Publication analysis — final handover](handover/publication-analysis-final.md) | Historically superseded the Sep 3 framing handover. Its “final” title is no longer a source-priority rule: F1/F2, mechanism, cost, and cheap-experiment claims need Sep 6 corrections. |
| Sep 4 review setup | [Independent-review preflight](independent-review-preflight.md) / [review prompt](independent-review-prompt.md) | Historical review instructions and environment snapshot. Not a completed review, current test result, or instruction to launch another review. |
| Sep 5: consolidated context | [September 5 knowledge base](handover/knowledge-base-2026-09-05.md) | Broad context dump and ranked proposals. Current A/B knowledge base supersedes conflicting conclusions; preserve for history. |
| This conversation's starting material | Pasted initial request/response *(local only)* | User-supplied motivation and earlier advice, not scientific evidence or a new instruction template. |
| Sep 6: corrected direction choice | Revised publication decision *(local only)* | Reframes A/B/C, corrects older claims, and explains audience value. Its remaining-work language predates the completed package and sharper B1/B2 split. |
| Sep 6: early verification | Verification script *(local only)* / result *(local only)* | First arithmetic/hash check supporting the memo. The current package adds portable inputs, bootstrap checks, and the TT audit. |
| Sep 6: developed argument | [Current A/B knowledge base](publication-directions-a-b-knowledge-base.md) | Develops audiences, defenses, B1/B2, and real TT examples; now links the prepared artifacts. |
| Sep 6: preparation completed | [A fact sheet](publication/direction-a/fact-sheet.md), [B fact sheet](publication/direction-a/scorer-fact-sheet.md), [walkthrough](publication/direction-a/README.md) | Concrete materials replace earlier instructions to construct a fact sheet, chart, or numeric package. |

Relationship in one line: **early proposals → V4 and its critique → completed-run consolidation → September 4/5 interpretations → September 6 corrections → A/B knowledge base → checked fact sheets and reanalysis package → this reading map**. This describes the document chain, not a claim that every later sentence is automatically more reliable.

## Technical decisions and execution history

Consult this section for engineering or historical execution questions. Issue numbers below give dependency order; they are not guaranteed calendar order. Decision records take precedence over earlier handovers for what was completed.

### Main execution and recovery

| Files | Topic / relationship |
|---|---|
| [Issue 12 analysis](issue-12-analysis-for-further-work.md) / [follow-up research](research/issue-12-follow-up.md) | Early result interpretation and proposed follow-up work; consult current conclusions before reusing old advice. |
| [Issue 16 recovery boundaries](issue-16-recovery-boundaries-decision.md) | Seed-17 recovery scope, Aug 31. |
| [Issue 17 artifact contract](issue-17-recovery-contract-decision.md) | Identity/finalization contract underlying later recovery. |
| [Issue 18 resumable evaluation](issue-18-resumable-evaluation-decision.md) | Evaluation recovery decisions. |
| [Issue 19 training recovery](issue-19-training-recovery-decision.md) | Completed-epoch recovery. |
| [Issue 20 selection/reveal transaction](issue-20-selection-heldout-transaction-decision.md) | Selection and reveal boundaries. |
| [Issue 21 null-selection continuation](issue-21-null-selection-continuation-decision.md) | Continuing after a run selects no checkpoint, Aug 31. |
| [Issue 22 run 42 execution](issue-22-seed-42-execution-decision.md) | Recovery-aware run record, Sep 1. |
| [Issue 23 run 2026 execution](issue-23-seed-2026-execution-decision.md) | Third-run completion, Sep 2. |
| [Issue 14 finalization handover](issue-14-finalization-handover.md) | Summary of all three completed runs; logically follows execution despite its smaller issue number. |
| [Issue 26 mid-epoch recovery](issue-26-mid-epoch-training-recovery-decision.md) | Additional recovery design used for the later ablation. |
| [WSL memory/GPU limits](wsl-memory-gpu-limits.md) | Environment guidance; verify current machine configuration if operating it. |

### Analysis, diagnostics, and ablation

| Files | Topic / relationship |
|---|---|
| [Issue 27 handover](issue-27-handover.md) → [frozen-input decision](issue-27-frozen-input-manifest-decision.md) | Planned then completed evidence discovery/identity work, Sep 2 decision. |
| [Issue 28 claim tables](issue-28-claim-tables-decision.md) | Statistical report construction, Sep 2. |
| [Issue 29 integrity reporting](issue-29-failure-mode-integrity-decision.md) | Report construction, Sep 2; successful implementation does not validate every behavioral interpretation it emitted. |
| [Issue 30 handover](handover/issue-30-handover.md) → [diagnostic decision](issue-30-chatmode-mmlu-diagnostic-decision.md) | Chat-template MMLU diagnostic, Sep 2. A separate format check, not a causal explanation of all task differences. |
| [Diagnostic protocol](../protocol/diagnostic/chatmode-mmlu-2026-09-02.json) / [digests](../protocol/diagnostic/digests.md) | Declared identity and scope for issue 30. |
| [Issue 31 handover](handover/issue-31-handover.md) → [ablation decision](issue-31-corpus-ablation-decision.md) | Sep 2 planning to Sep 3 decision. One nominal seed and a separately constructed corpus; several factors changed. |
| [Ablation protocol](../protocol/ablation/corpus-ablation-2026-09-02.json) / [digests](../protocol/ablation/digests.md) | Separate from the original three-run evidence and acceptance decision. |
| [Issue 32 provenance-source decision](issue-32-provenance-source-decision.md) | Provenance source choice supporting the frozen reports. |

### Implementation plans and design specifications

These are development history. Read the corresponding decision/code before assuming an unchecked task is still outstanding.

| Date / topic | Files |
|---|---|
| Aug 29: real GPU smoke | [Plan](superpowers/plans/2026-08-29-real-gpu-smoke-test.md) |
| Aug 31: recovery artifact contract | [Design](superpowers/specs/2026-08-31-recovery-artifact-contract-design.md) / [plan](superpowers/plans/2026-08-31-recovery-artifact-contract.md) |
| Sep 2: ablation mid-epoch recovery | [Design](superpowers/specs/2026-09-02-ablation-mid-epoch-recovery-design.md) / [plan](superpowers/plans/2026-09-02-ablation-mid-epoch-recovery.md) |
| Sep 2: corpus ablation | [Plan](superpowers/plans/2026-09-02-issue-31-corpus-ablation.md) |
| Sep 3: publication provenance gates | [Design](superpowers/specs/2026-09-03-issue-32-provenance-gates-design.md) / [plan](superpowers/plans/2026-09-03-issue-32-provenance-gates.md) |

## Corrections to carry into every answer

Historical files may contain useful numbers alongside superseded explanations. This table identifies the main disagreements and where to resolve them.

| Older wording or implication | Current interpretation | Check here |
|---|---|---|
| “Higher injection scores establish refusal/suppression” or TT category changes diagnose a mechanism | Original output text is missing. Half-scores hide which arm passed; 17/18 comparisons also show increased both-arm success. | [B fact sheet](publication/direction-a/scorer-fact-sheet.md) |
| “OPI/TT divergence isolates the effect of a utility control” | Tasks, prompts, parsers, and other details differ. Divergence does not isolate that cause. | [Current knowledge base](publication-directions-a-b-knowledge-base.md) |
| “Upstream MMLU generated answers; the local implementation changed its scoring modality” | Pinned upstream and the local default both rank first-token candidate logits. | [Pinned scorer](https://github.com/YuehHanChen/automated_alignment_researcher/blob/1899ad64fbfbc65790d259471cc4bf4de9437aa9/aar/benchmarks/mmlu/benchmark.py), [A fact sheet](publication/direction-a/fact-sheet.md) |
| “MMLU is pure recall,” or its contrast identifies a general evaluation-format cause | Task content and format remain confounded. The diagnostic tests one axis, not all causes. | [Current knowledge base](publication-directions-a-b-knowledge-base.md) |
| “21.3% changed answers, with +2.7 points net” for run 42 epoch 3 | 64/300 changed **correctness**: 27 became incorrect and 37 correct, net **+3.33 points**. | [Corrected README](../README.md) |
| “Shorter runtime proves shorter answers” | Timing is not a retained output-token count or behavioral label. | [Current knowledge base](publication-directions-a-b-knowledge-base.md) |
| “59.87 hours is the main experiment cost” | Main baseline plus three runs: **47.3381 GPU-accounted hours**. The larger number includes later work and is not a complete all-incurred ledger. | [A fact sheet](publication/direction-a/fact-sheet.md) |
| “Nine independent trials; recorded seeds reproduce initialization” | Three related trajectories, one shared sampled baseline, and initialization preceding the run seed. | [A fact sheet](publication/direction-a/fact-sheet.md) |
| “An approximate power calculation proves held-out testing was useless” | It is a planning approximation. No held-out comparison was revealed; achieved paired power was not established. | [Power notes](../protocol/power_notes.md), [current knowledge base](publication-directions-a-b-knowledge-base.md) |
| “Checksums imply clean-checkout reproduction of the whole study” | Original report assembly needs local inputs. The new small package reproduces its numeric reanalysis, not training or missing outputs. | [Validation notes](publication/direction-a/validation-notes.md) |
| “A cheap PNA-I re-score will establish legitimate-task fidelity” | PNA-I competence does not provide the missing legitimate-task reference/outcome. Runtime and behavioral labels require a separately specified diagnostic. | [Current knowledge base, B2 requirements](publication-directions-a-b-knowledge-base.md) |
| “Next, prepare the fact sheet, numeric subset, and chart” | Those materials now exist. Use them for A/B1 drafting; additional model measurements belong to B2. | [Prepared package](publication/direction-a/README.md) |

## Scope and maintenance

This map covers the current publication package; repository Markdown documentation under `docs/`, `analysis/`, and `protocol/`; selected implementation/evidence files; the earlier `G:` ArticleArtifacts archive; and this conversation's persistent attachment/research notes. It is a reading index, not an inventory of every code file, run bundle, or external reference mentioned anywhere in the project.

Local link targets were checked when this map was prepared. Archive file introductions were reviewed to classify them; this indexing pass did not repeat every old literature search or experiment. Current factual conclusions refer to the dated checks in the package validation notes. Article drafts and a final publication sign-off are separate from preparing this map.

When a new result or draft appears, add it under its purpose, state whether it is measured, proposed, or historical, and identify which earlier document it replaces or supplements. Update the reading route and correction table only when their answer changes. Preserve original evidence and historical context rather than silently rewriting the record.
