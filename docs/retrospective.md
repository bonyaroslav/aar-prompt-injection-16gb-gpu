# One GPU, nine rejected checkpoints: my 2026 prompt-injection experiment

*Yaroslav Bondar, September 2026*

## What I got wrong

Twice I had a conclusion I liked, and twice the evidence took it away.

First I reported that my MMLU scoring differed from the upstream harness — a tidy
explanation for why one benchmark behaved unlike the others. It was wrong. The pinned
upstream uses first-token candidate logits without a chat template, exactly as my
default did. The correction is still in the README, next to the claim it replaced.

Then I had the comfortable story that the model had simply learned to refuse
everything: refusing an attack looks like resistance, and refusing arithmetic looks
like a capability collapse. Instead of asserting it, I read the scorer. The Tensor
Trust score averages two binary arms, so a half-score cannot tell you which arm
passed. When I counted the paired items, 17 of 18 trained comparisons had *more*
items passing both arms than the baseline. That is not what blanket refusal looks
like. I could not prove refusal and I could not rule it out, so I said so.

## What I read, and what I ran

Anthropic's Automated Alignment Researcher study supplied the evaluation setting and
the same prompt-injection target, Qwen3.5-2B. I adapted that setting to one RTX 4080
with 16 GB and tested a single response-only QLoRA recipe — 5,000 examples, three
epochs, three runs. SecFid's separation of executing an injected instruction,
processing it as data, and ignoring it gave me the vocabulary for what my own scores
could not distinguish.

47.3 GPU-hours later: injection scores rose on all nine checkpoints, and all nine
failed the GSM8K and IFEval limits I had fixed before the runs started. I selected
nothing. The held-out set was never unsealed.

## What I would do differently

Retain the generated answers. I kept the scores and discarded the text, so no
behavioural question about these checkpoints can be answered now — not by me, not by
anyone. That one omission cost more than the GPU hours did.

What I am carrying to the next experiment: acceptance rules written down before the
runs, a held-out set that stays sealed, and a recorded hash for every input. They are
cheap, and they are what lets a null result stand as a result instead of an anecdote.

## About tools

I wrote the code and documentation with AI coding assistants. The protocol, the
acceptance thresholds, the decision to keep those thresholds when they rejected every
candidate, and the two corrections above were mine.

---

- The numbers, thresholds and rejection decisions: [checked fact sheet](publication/direction-a/fact-sheet.md)
- What the injection scores do and do not establish: [scoring audit](publication/direction-a/scorer-fact-sheet.md)
- Recalculate every table yourself, no GPU required: [reproduction package](publication/direction-a/README.md)
- Project overview: [README](../README.md)
