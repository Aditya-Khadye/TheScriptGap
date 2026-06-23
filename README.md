# The Script Gap — v1.0

The Script Gap is a research framework that identifies writing systems
**under-served by open-source digital typography** — widely usable in the real
world but offered few, visually similar fonts. It combines font availability and
visual diversity into a single **Script Servedness Score (SSS)**, and reports web
exposure and engineering complexity alongside as context, to give a data-driven
view of where non-Latin type investment is most needed.

> **What's new in v1.0:** the servedness score is the project's original
> **gap-ratio**, hardened — *real font choice (support × diversity) per unit of web
> demand* — with the arbitrary `+0.1` removed, made reproducible end-to-end, and
> backed by a test suite. Demand is the **weakest** input (see *Limitations*), and
> engineering complexity is reported separately as a **prioritization** signal, not
> a dimension of servedness.

## Methodology

We study **8 major non-Latin scripts** (Devanagari, Arabic, Bengali, Tamil,
Telugu, Han, Katakana, Cyrillic), with **Latin as a well-served reference**, on
four indices:

| Index | What it measures | Role in v1.0 |
|---|---|---|
| **Support** | Distinct open-source font families per script (Google Fonts). | **In the SSS** |
| **Diversity** | Visual variety of available fonts, via Vision Transformer (ViT-B/16) glyph-image embeddings (ResNet-50 and classical-CV ablations). | **In the SSS** |
| **Exposure** | Per-script web font-request volume (HTTP Archive via BigQuery, Google Fonts CSS `subset=`). | **In the SSS** — weakest input (see *Limitations*) |
| **Complexity** | Per-script engineering difficulty from font binaries (glyph-expansion ratio, vertical footprint, OpenType-feature friction), via fontTools. | Separate **prioritization** lens |

**The score.** Support, diversity, and demand feed a single ratio:

```
effective_choice = support_norm × (1 − similarity_norm)     # quantity × variety
SSS = effective_choice / log₁₀(exposure)                    # choice per (log) demand
```

A script is well served when it has lots of *genuinely different* fonts relative to
how much it's read; a **high-demand script with little real choice scores lowest =
most underserved**, which is what a prioritization score should surface. This is the
project's original gap-ratio `(support·(1−similarity))/(exposure+0.1)` with one fix:
it divides by **log-scaled** exposure (always the stated intent) instead of an
`(exposure_norm + 0.1)` denominator that needed an arbitrary `0.1` floor. The ranking
is unchanged (Spearman 0.95–0.98 vs the original at sensible ε).

**Demand is the weakest input.** The committed exposure numbers carry a coverage
confound — a font request is mapped to every script the font covers, so Latin/UI
fonts (Inter, Roboto) inflate the scripts they merely bundle; `bigquery_pull.py`
is upgrading this to the per-page CSS `subset=` signal. Crucially, **the underserved
ordering does not depend on demand** — over the 8 non-Latin scripts the score with
and without the demand term ranks them identically (ρ = 1.00); demand mainly
separates the high-demand well-served scripts. Detail + roadmap:
[`exposure_research/DEMAND_PROVENANCE.md`](exposure_research/DEMAND_PROVENANCE.md).

**Why complexity is separate.** Complexity measures *creation difficulty* — a
**cause** of under-service, not a measure of how well readers are currently served.
It feeds the prioritization question, not the SSS.

## Key findings

| Tier | Scripts |
|---|---|
| **Underserved** | **Bengali, Tamil, Devanagari, Telugu, Arabic, Han** |
| **Moderately served** | Katakana, Cyrillic |
| **Well served** | Latin |

Only **Latin** is clearly well served; **Cyrillic and Katakana** trail it; the six
other non-Latin reading scripts are underserved — too few genuinely-different fonts
for their readership. The **Indic scripts {Bengali, Tamil, Devanagari, Telugu}** are
the most underserved cluster, consistent with the type-design literature (e.g. SIL /
Hossain et al. on "disproportionately few Indic fonts"). Han lands in the underserved
group because it has very few open-source families (26) despite high diversity. The
exact #1 (Bengali vs Tamil) is sensitive to the diversity normalization and shouldn't
be over-read. Canonical output:
[`data/final/script_servedness.csv`](data/final/script_servedness.csv).

## Robustness

All figures below are computed from committed data by `analysis/robustness.py`
and asserted in `tests/test_servedness.py` (full report: `data/final/robustness.md`).

- **Diversity vs. model choice:** ViT-B/16 vs. ResNet-50 Spearman **ρ = 0.95** over
  the 8 non-Latin scripts — the diversity ranking survives the deep-model swap.
  Classical pixel-wise CV diverges (ρ ≈ 0), so it is **not** used as robustness evidence.
- **The `+0.1` fix is ranking-neutral:** dividing by log₁₀(exposure) reproduces the
  original gap-ratio (Spearman 0.95–0.98 vs `(exposure_norm + ε)` at ε = 0.5–1.0).
- **The underserved ordering doesn't depend on demand:** with and without the demand
  term, the 8 non-Latin scripts rank identically (ρ = 1.00) — the weakest axis doesn't
  drive the headline.
- **Tiers vs. diversity model:** feeding the SSS ResNet-50 diversity instead of ViT
  leaves the tiers **7/8 identical**.

## Limitations

- **Supply = Google Fonts only** — an open-source-stylistic-choice proxy, not total
  font supply; under-counts commercial (Monotype/Adobe), system, and SIL fonts.
- **Demand is the weakest input** — a coverage-confounded proxy in the score (see
  `DEMAND_PROVENANCE.md`); HTTP Archive / CrUX is Chrome-biased and under-counts CJK /
  low-Chrome regions. The underserved ordering doesn't depend on it (ρ = 1.00 with/without).
- **n = 8 scripts** — a coarse tiering, not fine statistics; does not generalize to
  scripts outside the set (Hangul, Thai, Hebrew, … are not studied).
- **Min–max normalization is relative** to this 8-script set; index endpoints
  (e.g. Tamil = 0 diversity) are not absolute statements.
- **Complexity weights** (0.50 / 0.30 / 0.20) are a documented prior, not validated.

## Reproducing the result

```bash
uv run python pipeline.py            # support → exposure → viz → the SSS, heatmap, tiers
```

`python pipeline.py` reproduces the servedness score from the indices committed in
this repo. The heavy index stages (`diversity` = ViT/GPU, `complexity` = fontTools
over a Google Fonts clone) and a fresh `exposure`/`support` pull draw on external
data; their outputs are committed.

Validate and analyze:

```bash
uv run --with pytest --with pandas --with numpy pytest -q   # regression + robustness tests
uv run python analysis/robustness.py                        # robustness / sensitivity report
uv run python exposure_research/demand_audit.py             # demand-axis confound audit
```

## Repository structure

```
TheScriptGap/
├── pipeline.py                    # orchestrator (support → exposure → diversity → complexity → viz)
├── support_research/              # Support Index (Google Fonts families)
├── exposure_research/             # Demand: bigquery_pull.py (HTTP Archive subset= pull),
│                                  #   demand_audit.py, DEMAND_PROVENANCE.md
├── similarity_research/           # Diversity (ViT/CNN/classical) pipelines
├── complexity/                    # Complexity Index (fontTools) — prioritization lens
├── data/
│   ├── support/ exposure/ similarity/ complexity/   # committed index inputs
│   ├── viz/                       # heatmap.html / .png / heatmap_data.csv
│   └── final/script_servedness.csv                  # canonical result
├── data_viz/generate_heatmap.py   # SSS scoring + heatmap + servedness table
└── final_model/                   # legacy K-Means clustering (superseded by the SSS)
```

## Monthly deployment

The project includes a scheduled GitHub Actions workflow and local scripts for recurring data refresh.

### Local monthly run

```bash
# Light refresh: Google Fonts support + exposure + heatmap (needs GCP for fresh exposure)
./scripts/run_monthly.sh --force

# Full refresh: adds ViT diversity + fontTools complexity (needs GOOGLE_FONTS_DIR)
export GOOGLE_FONTS_DIR="$HOME/google/fonts"
./scripts/run_monthly.sh --full --force
```

### BigQuery preflight (exposure)

Pull HTTP Archive font-request counts before the exposure stage:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcp/thescriptgap.json"
export GOOGLE_CLOUD_PROJECT="your-gcp-project"
export GOOGLE_FONTS_API="your-api-key"

python -m exposure_research.bigquery_pull --force
```

Or run the full pipeline — the exposure stage calls this automatically:

```bash
python pipeline.py --stages exposure --force
```

Set `SKIP_BIGQUERY=1` to reuse existing `data/bigquery/big_query_data.csv`.

### GitHub Actions

Workflow: `.github/workflows/monthly-pipeline.yml`

Runs on the 5th of each month (after HTTP Archive crawls) and via **Actions → Monthly pipeline → Run workflow**.

**Repository secrets:**

| Secret | Purpose |
|--------|---------|
| `GOOGLE_FONTS_API` | Google Fonts Developer API key |
| `GOOGLE_CLOUD_PROJECT` | GCP project billed for BigQuery |
| `GCP_SA_KEY` | Service account JSON with BigQuery User role |

Scheduled runs execute **support → exposure → viz**. Use **Run workflow** with **Run heavy stages** for diversity + complexity (clones Google Fonts; CPU-only, slow).

Updated CSVs and `docs/viz/heatmap.html` are committed automatically when data changes.

## Website
https://aditya-khadye.github.io/TheScriptGap/
## Capstone Video
https://www.youtube.com/watch?v=wNpgtw6_ukI
## Partners
Commissioned by The Readability Consortium, addressed to Monotype, Google Fonts, and Adobe.
