# License scope and upstream attribution

The root [MIT License](LICENSE) applies to Yaroslav Bondar's original code and
documentation in this repository. It does not relicense third-party material
or grant rights to external dependencies, model weights, or benchmark content.
Preserve any applicable upstream copyright and license notices.

- **Evaluation harness:** the experiment uses
  [YuehHanChen/automated_alignment_researcher](https://github.com/YuehHanChen/automated_alignment_researcher)
  at commit `1899ad64fbfbc65790d259471cc4bf4de9437aa9`, recorded in
  [the protocol](protocol/manifest.json). Its code is loaded from a separate
  checkout. That pinned checkout has no top-level license file; this project's
  MIT license does not supply a license for it or for material derived from it.
- **Models and datasets:** model weights, source benchmark examples, and source
  training datasets are not relicensed here. Consult the relevant publishers'
  terms before obtaining or redistributing them. The standalone publication
  package contains numeric observations and provenance, not the original
  generated responses or a release of model weights.
- **Imported assistant configuration:** third-party skill or command text in
  `.agents/` and `.claude/` is excluded from this project's MIT grant. Consult
  its original sources for applicable terms.
- **Scientific references:** attribution and links identify the work this study
  builds on. The linked publications remain subject to their publishers' and
  authors' terms. Citation does not imply endorsement.

For the experiment's relationship to the two main papers, see the README's
[scientific foundations](README.md#scientific-foundations).
