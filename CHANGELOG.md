# Changelog

## 1.0.0

First tagged release, built on `main` (after the `seperating_complexity` merge and
the HTTP Archive subset-pull work). Scores servedness only on the signals it can
measure reliably, and reproduces its headline result from committed data.

### Script Servedness Score (SSS)
- **Scored on the two trustworthy signals — font support + font diversity:**
  `SSS = support_norm − similarity_norm`. No tuned weights, no epsilon.
- **Replaces the gap-ratio** `(support·(1−similarity))/(exposure+0.1)`, which was
  epsilon-dependent and rewarded low-demand scripts for the wrong reason.
- **Web exposure (demand) kept out of the score**, shipped as a labelled CONTEXT
  column. The HTTP Archive `subset=` pull (`exposure_research/bigquery_pull.py`) is
  the right fix, but the committed numbers aren't yet regenerated from it and the
  CJK Han/Katakana split is unreliable. See `exposure_research/DEMAND_PROVENANCE.md`.
- **Complexity kept out** as a separate prioritization (creation-difficulty) signal.
- Tiers: **Underserved** {Tamil, Bengali, Devanagari, Telugu},
  **Moderately served** {Arabic, Han}, **Well served** {Cyrillic, Katakana, Latin}.
  Canonical output: `data/final/script_servedness.csv`.

### Added
- `exposure_research/demand_audit.py` — quantifies the coverage-attribution
  confound (97–100% of non-Latin demand comes from multi-script bundler fonts;
  Inter/Poppins/FontAwesome drive it) from source data.
- `exposure_research/DEMAND_PROVENANCE.md` — provenance, the `subset=` fix, and the
  roadmap to promote demand back into the score.
- `data/final/script_servedness.csv` — canonical servedness table with tiers.
- `VERSION`, `CHANGELOG.md`.

### Changed
- `data_viz/generate_heatmap.py` — new SSS; Web Exposure relabelled as a context
  column; emits the servedness table.
- README rewritten to the v1.0 model with an explicit **Limitations** section;
  robustness narrowed to the ViT-vs-ResNet ablation (classical-CV claim dropped).
  The team's Monthly-deployment / BigQuery docs are preserved.

### Superseded
- The K-Means two-tier clustering in `final_model/` is superseded by the SSS.

### Known issues / next
- **Promote demand into the score** once a fresh `subset=` pull is committed and
  validated (DEMAND_PROVENANCE.md §4).
- Scope is 8 non-Latin scripts; extending the set is future work.
