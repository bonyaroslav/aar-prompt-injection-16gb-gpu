# Phase 1 upstream reconciliation

The pinned upstream checkout is authoritative for prompt text, roles, decoding and per-benchmark scorer behavior. Its committed `HEAD` is `1899ad64fbfbc65790d259471cc4bf4de9437aa9`; the checkout also has unrelated dirty/untracked prior artifacts, which are excluded from provenance.

The accepted local protocol does not silently inherit three upstream choices:

1. Upstream reports a geometric mean of closed fractions; this study selects on the unweighted mean of absolute improvements across the three visible safety benchmarks and requires +5 percentage points.
2. Upstream capability filtering uses CI-overlap. This study uses the accepted absolute per-gate declines and 98% mean normalized retention.
3. Upstream rule scoring skips failed/empty generations. Held-out sealing preserves every invalid turn and reports both valid-only and intent-to-evaluate, where technical invalids are failures.

These are predeclared local analysis/protocol deviations, not post-result changes. The OPI combine-only attack, 300 published candidates per visible benchmark, 200 InjecAgent candidates, real system/user roles, and all decoding values are resolved from the pinned committed docs/code and exported in `manifest.json`.
