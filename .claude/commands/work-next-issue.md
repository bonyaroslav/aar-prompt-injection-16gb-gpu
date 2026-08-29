---
description: Pick up the next ready-for-agent GitHub issue in dependency order, implement it fully, and close it out.
---

# Work next issue

Repo: `bonyaroslav/aar-prompt-injection-16gb-gpu`. Work only against issues labeled `ready-for-agent`.

## 1. Orient

- `git status`; if dirty, stop and report — don't touch in-progress work from another session.
- `gh issue list --repo bonyaroslav/aar-prompt-injection-16gb-gpu --state closed --label ready-for-agent` and the same with `--state open`, both sorted by number.
- Read each open issue's **Blocked by** line. The next issue is the lowest-numbered open one whose every blocker is closed.
- If none qualify (all remaining are blocked, or none remain), stop and report exactly why — don't guess or skip ahead.

## 2. Sanity-check before touching anything

If any closed issue's own blockers aren't all closed, or two issues' states conflict, stop — that's a critical inconsistency in the ticket graph, not something to paper over.

## 3. Implement

- Read the full issue (What to build / Acceptance criteria / Blocked by).
- `RESEARCH_SPEC.md` and `protocol/manifest.json` are authoritative. Never silently deviate from a frozen protocol value. Check `docs/adr/*.md` before re-deciding anything already decided there.
- Use TDD where the acceptance criteria describe testable behavior. Before considering the ticket done, run the existing suite (`python -m unittest discover -s tests -q`) plus whatever tests the ticket added.

## 4. Stop conditions — do NOT close the issue if any apply

- An acceptance criterion needs a decision only the user can make: new scope, a genuine protocol deviation, a missing credential/token, a destructive or irreversible action, real money or a paid API.
- Finishing would violate a capability gate, resource limit, or held-out-sealing rule from `RESEARCH_SPEC.md`.
- Required hardware/environment isn't reachable (e.g. a real-GPU ticket but WSL/CUDA isn't up).
- Tests fail and the fix isn't obvious and safe.

If any apply: commit only safe, clearly-marked-WIP partial progress if there is any; post a comment on the issue explaining what's blocking and what decision is needed (established-so-far / what's-needed-from-you, same shape as a triage needs-info note); leave the issue **open**; say so plainly in your final reply instead of reporting the run as done.

## 5. On success

- Run the full test suite once more.
- Update `RESEARCH_SPEC.md`'s "Status and Guardrails" section (a line or two, not a rewrite) so a fresh session sees current phase/issue state without reading this conversation.
- Commit with a message ending `Closes #<N>` plus the repo's usual `Co-Authored-By` trailer. Do not push unless asked.
- `gh issue close <N> --comment "<what shipped, and the commit it shipped in>"`.
- Report: which issue closed, what changed, and what's next in the queue.

## 6. Never

- Close an issue whose acceptance criteria aren't all met.
- Skip ahead to a later issue because an earlier one is blocked.
- Push, force-push, or touch any other issue/PR.
