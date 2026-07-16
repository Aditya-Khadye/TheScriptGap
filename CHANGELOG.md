# Changelog

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
