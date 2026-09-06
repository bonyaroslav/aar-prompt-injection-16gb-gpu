# Publications to learn from, with specific things to look at

Links checked on 6 September 2026. These are examples of presentation or scientific context, not predictions of how much attention this project will receive. You do not need to read all four before writing.

## 1. Best first example for explaining an evaluation problem

[What's going on with the Open LLM Leaderboard? — Hugging Face, 23 June 2023](https://huggingface.co/blog/open-llm-leaderboard-mmlu)

**Why it is relevant:** the authors investigate differing MMLU results by comparing implementations. They explain model evaluation in ordinary terms, show concrete prompt/scoring differences, and connect those differences to the reported results.

**Look at:** “MMLU comes in all shapes and sizes” and “Now how do we evaluate the model from these prompts?” The side-by-side examples do explanatory work; they are not decorative illustrations.

**Borrow for B:** start with one visible discrepancy, show the actual scoring rule, demonstrate the ambiguity with a small example, and state its practical consequence. Your Tensor Trust truth table and retained-score example can fill that role.

**Do not copy its conclusion:** its implementation comparison is a different experiment. Your MMLU results do not themselves establish a format-based cause of the other task losses.

## 2. Best example for writing up a failed attempt honestly

[Some negative steganography results — Fabien Roger, 9 December 2023](https://www.lesswrong.com/posts/EEvsL9cpgDAxAhTzt/some-negative-steganography-results)

**Why it is relevant:** it reports unsuccessful attempts with methods and uncertainty rather than presenting failure as proof that the broader approach cannot work. A substantive comment asks whether the setup was adequate and requests a successful positive control.

**Look at:** how the attempted setups are described, how the conclusion is limited, and the methodological challenge in the comments.

**Borrow for A:** explain what you attempted, the condition for success, what happened, and what remains open. Give readers enough detail to challenge one specific inference.

**Do not copy its scope or reception:** steganography is a different problem, and the author's audience does not predict yours. A failed local recipe is not a general impossibility result.

## 3. Closest scientific reference for Direction B

[SecFid: Security–Fidelity Tradeoffs — 29 June 2026](https://arxiv.org/html/2606.30783v1)

**Why it is relevant:** it distinguishes following an injected instruction, processing the suspicious text as data, and suppressing it. It also includes fidelity-aware training, so the broad observation and training idea are already prior work.

**Look at:** §3.1 for examples that make different behaviors observable; §§3.4–3.5 for the distinction between output labels and execution; Appendix E before making any training-novelty claim.

**Borrow for B:** define the legitimate task first, then specify outputs that would count as completing it, following the attacker, or failing in another way. Use its full behavioral decomposition rather than treating one favorable score as task success.

**Do not transfer its measurements to your model:** the original outputs from your experiment were not retained. The paper suggests how to measure behavior next; it cannot label your past outputs.

## 4. Original research setting for Direction A

[Automated Researchers Can Mitigate Well-Characterized Alignment Failures — Anthropic Alignment Science](https://alignment.anthropic.com/2026/automated-alignment-researchers/)

**Why it is relevant:** it supplies the source model/benchmark setting and describes how methods are evaluated and filtered. Its iterative method search differs from your one-recipe experiment. It explicitly discusses evaluation limitations.

**Look at:** §2.3 for evaluation and selection, and Appendix A.4 for capability-test details and filter limitations.

**Borrow for A:** make the route from measurement to selection explicit. Explain which data are used to develop/select a method and which results would support a broader conclusion.

**Do not frame your result as a refutation:** different methods, acceptance rules, and search budgets answer different questions. Your contribution is a local worked decision and an inspectable evidence package.

## How these examples support this project's presentation

| Choice already made | Reason | Most relevant example |
|---|---|---|
| State the rejection result early | Readers need to know what decision the evidence supports. | Negative-results report; AAR's evaluation description. |
| Use one figure containing all six benchmarks | Readers can see the intended improvement and the measured regressions together. | Our design choice, informed by the need to show competing measurements; not a rule prescribed by a paper. |
| Include a simple worked calculation | Readers can verify a conclusion without understanding the full training stack. | Hugging Face's concrete evaluation examples. |
| Keep the missing behavioral explanation explicit | A favorable attack score is not sufficient to establish useful task completion. | SecFid's behavioral distinctions. |
| Supply numeric data and a script | Another reader can reproduce the headline instead of taking the prose on trust. | The implementation and evidence transparency illustrated by the technical examples. |

If reading only one example now, start with the Hugging Face article. It is closest to the explanatory style needed here: a concrete puzzle, a small demonstration, evidence, and a bounded conclusion.
