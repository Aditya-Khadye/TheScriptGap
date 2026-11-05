# Changelog

## Unreleased

### Changed
- Diversity v1.1. `script_diversity_vit_100.py` and `script_diversity_cnn_100.py`
  now pin Pillow's layout engine to libraqm (HarfBuzz shaping) and refuse to run
  without it. v1.0 never set an engine, so its committed outputs were rendered
  unshaped: Indic conjuncts came out as unjoined parts and some Arabic glyphs
  came out blank. Latin is added to the same protocol (100 fonts, 99 reference
  glyphs, 140 px) instead of the separate 10-capital run. New flags:
  `--fonts-dir`, `--output-dir`, `--scripts`; `metadata.json` records the layout
  engine, Pillow and libraqm versions and the google/fonts commit.
- All nine scripts re-run in one pass on google/fonts `07d64d5a` (2026-04-09),
  which holds every font file in the v1.0 samples. Outputs in
  `similarity_research/diversity_research/vit_outputs_v1_1/` and
  `cnn_outputs_v1_1/`; `data/similarity/similarity_results.csv`,
  `data/final/script_servedness.csv`, the heatmap and the site copy are
  regenerated from them. v1.0 values are kept in
  `data/similarity/similarity_results_v1_0.csv`.
- Result: Latin's diversity falls from 0.85 to 0.37 once it is measured like the
  other scripts. Tiers: Tamil, Bengali, Devanagari and Telugu Underserved; Arabic
  (0.333) and Han (0.335) Moderately served, just above the 0.3 cut-off;
  Cyrillic, Latin and Katakana Well served. Non-Latin diversity order is
  unchanged by shaping (Spearman 1.00 vs v1.0).
- `analysis/family_diagnostic.py`, `analysis/population_check.py` and
  `analysis/robustness.py` read the v1.1 runs; Latin is included in the
  family and subset-tag checks.

### Added
- `analysis/shaping_check.py`: re-embeds the committed v1.0 fonts with and
  without shaping and recomputes diversity, SSS and tiers
  (`data/analysis/shaping_check*.csv`).
- `analysis/v11_checks.py`: v1.0 vs v1.1, exposure removed, +/-10% and +/-20%
  input shocks, ViT vs ResNet-50, and Latin scored on Latin-primary fonts only
  (`data/analysis/v11_*.csv`, `tests/test_v11_checks.py`). The four Indic
  scripts stay the bottom four under every shock; Arabic and Han are the
  scripts that cross the 0.3 boundary.

### Added
- `analysis/family_diagnostic.py`: family-bundling diagnostic for the diversity
  term. Recomputes mean cosine distance at file level, between-family only, and
  one representative per Google Fonts family, with a family-level cluster
  bootstrap (CIs and rank intervals). Diversity ranking holds (Spearman 0.98 vs
  file level; tiers 9/9 unchanged), but the order within the four Indic scripts
  is not identified.
- `analysis/population_check.py` + `analysis/gf_index.py`: aligns the diversity
  population with the support term by using Google Fonts subset tags instead of
  the >=10-codepoint threshold. Ranking holds (Spearman 0.93; tiers 9/9); 79% of
  families in the Han sample are not Chinese-tagged.
- `data/diagnostics/` outputs and README; `tests/test_family_diagnostic.py`.
- `complexity/closure.py` + `complexity/aggregate_closure.py`: Infrastructure
  Complexity Index v1.1. Expansion is now a measured GSUB closure per script
  (default-on shaping features, the script's OpenType tags or DFLT) instead of
  the v1.0 heuristic, which added 0.15 x base glyphs per matched GSUB
  FeatureRecord across the whole font and so gave every script in a file the
  same value. Friction is script-specific. One representative per subset-tagged
  family. Arabic Presentation Forms and CJK Compatibility Ideographs are
  excluded from the denominator. Telugu moves from 4th to most complex;
  composite Spearman vs v1.0 is 0.90. Outputs in `data/complexity_v11/`, tests
  in `tests/test_complexity_closure.py`. v1.0 outputs are unchanged.

### Fixed
- Latin diversity was embedded from 10 capitals at 160 px vs roughly 90 to 100
  glyphs at 140 px for the other scripts, so it was not on the same scale
  (fixed by diversity v1.1).

## 1.0.0

First tagged release. Ships the project's original Script Servedness Score —
*real font choice per unit of web demand* — hardened: the arbitrary epsilon
removed, made reproducible end-to-end, and backed by a test suite. Built on
current `main` (after the `seperating_complexity` merge and the HTTP Archive
`subset=` work).

### Script Servedness Score (SSS)
- **The original gap-ratio**, with the denominator divided by **log-scaled
  exposure** (the methodology's stated intent) instead of `(exposure_norm + 0.1)`:

  ```
  effective_choice = support_norm × (1 − similarity_norm)
  SSS = effective_choice / log10(exposure)
  ```

  This removes the arbitrary `0.1` floor with **no change to the ranking**
  (Spearman 0.95–0.98 vs the original at sensible ε).
- **Three inputs in the score:** support, diversity, and demand. Demand is the
  **weakest** input — the committed exposure numbers carry a coverage confound
  (font requests attributed to every script a font covers); the HTTP Archive
  `subset=` pull (`exposure_research/bigquery_pull.py`) is the upgrade path. The
  underserved ordering does **not** depend on demand (ρ = 1.00 with/without).
- **Complexity** ("engineering cost") is reported separately as a prioritization
  signal — not in the SSS.
- Tiers: **Underserved** {Bengali, Tamil, Devanagari, Telugu, Arabic, Han},
  **Moderately served** {Katakana, Cyrillic}, **Well served** {Latin}.
  Canonical output: `data/final/script_servedness.csv`.

### Added
- `analysis/scoring.py` — the SSS formula as a single, importable source of truth.
- `analysis/robustness.py` + `data/final/robustness.md` — robustness & sensitivity
  from committed data: ViT-B/16 vs ResNet-50 diversity **ρ = 0.95**; the log-demand
  denominator reproduces the original gap-ratio; demand does not reorder the
  non-Latin scripts (ρ = 1.00); tiers **7/8 identical** under ResNet.
- `tests/` — pytest suite (8 tests) pinning the committed result to the formula and
  asserting the robustness + demand-confound claims; wired into `pyproject.toml`.
- `exposure_research/demand_audit.py` + `DEMAND_PROVENANCE.md` — quantify and document
  the demand confound (97–100% of non-Latin "demand" comes from multi-script bundler
  fonts: Inter / Poppins / FontAwesome) and the `subset=` upgrade roadmap.
- `data/final/script_servedness.csv` (+ `data/final/README.md`) — canonical result
  with tiers and output provenance.
- `VERSION`, `CHANGELOG.md`; project version bumped to 1.0.0.

### Changed
- `data_viz/generate_heatmap.py` — emits the canonical servedness table; heatmap
  shows the three score inputs + the SSS.
- README rewritten to the v1.0 model with an explicit **Limitations** section.
- `docs/index.html` methodology updated: the `log10(exposure)` denominator, tiers
  from the SSS, complexity reported separately, and an honest exposure caption.

### Superseded
- The K-Means two-tier clustering in `final_model/` is superseded by the SSS.

### Known issues / next
- **Upgrade demand** to the per-page `subset=` signal (run + commit the BigQuery
  pull; validate the `subset=`-vs-fallback share). Steps in `DEMAND_PROVENANCE.md §4`.
- Scope is 8 non-Latin scripts; extending the set is future work.
