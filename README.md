# The Script Gap — v1.0

The Script Gap is a research framework that identifies writing systems
**under-served by open-source digital typography** — widely usable in the real
world but offered few, visually similar fonts. It combines font availability and
visual diversity into a single **Script Servedness Score (SSS)**, and reports web
exposure and engineering complexity alongside as context, to give a data-driven
view of where non-Latin type investment is most needed.

> **What's new in v1.0:** the servedness score is built on the two signals we can
> measure reproducibly and defend — **font support** and **font diversity**. Web
> exposure (demand) is published as **context only** (the HTTP Archive `subset=`
> pull is the right fix, but the committed numbers aren't yet trusted to drive a
> headline — see below), and engineering complexity is a separate **prioritization**
> signal, not a dimension of servedness.

## Methodology

We study **8 major non-Latin scripts** (Devanagari, Arabic, Bengali, Tamil,
Telugu, Han, Katakana, Cyrillic), with **Latin as a well-served reference**, on
four indices:

| Index | What it measures | Role in v1.0 |
|---|---|---|
| **Support** | Distinct open-source font families per script (Google Fonts). | **Drives the SSS** |
| **Diversity** | Visual variety of available fonts, via Vision Transformer (ViT-B/16) glyph-image embeddings (ResNet-50 and classical-CV ablations). | **Drives the SSS** |
| **Exposure** | Per-script web font-request volume (HTTP Archive via BigQuery, using the Google Fonts CSS `subset=` parameter). | **Context only** (see *Limitations*) |
| **Complexity** | Per-script engineering difficulty from font binaries (glyph-expansion ratio, vertical footprint, OpenType-feature friction), via fontTools. | Separate **prioritization** lens |

**The score.** Support and diversity are normalized to [0, 1] and combined:

```
SSS = support_norm − similarity_norm        (similarity = 1 − diversity)
```

i.e. a script is well served when it has **both** plenty of fonts **and** real
visual choice. The score is then min–max normalized and split into tiers. (This
replaces the earlier gap-ratio `(support·(1−sim))/(exposure+0.1)`, which was
epsilon-dependent and rewarded low-demand scripts for the wrong reason.)

**Why demand is not in the score.** Exposure is a demand signal in principle, and
`exposure_research/bigquery_pull.py` attributes HTTP Archive requests to scripts
via the CSS `subset=` parameter — the right approach. But the committed per-script
numbers aren't yet regenerated from that pull, and the CJK Han/Katakana split is
unreliable, so v1.0 ships exposure as a clearly-labelled **context proxy** and
scores servedness on the two trustworthy axes. Detail + roadmap:
[`exposure_research/DEMAND_PROVENANCE.md`](exposure_research/DEMAND_PROVENANCE.md).

**Why complexity is separate.** Complexity measures *creation difficulty* — a
**cause** of under-service, not a measure of how well readers are currently served
(Han is hard to build yet well served). It feeds the prioritization question, not
the SSS.

## Key findings

| Tier | Scripts |
|---|---|
| **Underserved** | **Tamil, Bengali, Devanagari, Telugu** |
| **Moderately served** | Arabic, Han |
| **Well served** | Cyrillic, Katakana, Latin |

The headline is robust: the **Indic scripts cluster at the bottom** — few
open-source font families *and* low visual diversity — consistent with the
type-design literature (e.g. SIL / Hossain et al. on "disproportionately few Indic
fonts"). Arabic and Han sit in the middle (Han has few open-source families but
high diversity). The exact #1 (Tamil vs Bengali) is sensitive to the diversity
normalization and shouldn't be over-read; the **underserved cluster** is the
durable result. Canonical output:
[`data/final/script_servedness.csv`](data/final/script_servedness.csv).

## Robustness

The diversity rankings hold across deep-model choice: a **ViT-B/16 vs. ResNet-50**
ablation gives a high Spearman rank correlation, so the signal isn't an artifact of
one architecture. (Classical pixel-wise features are **not** a robust substitute —
their ranking diverges — so we do not cite them as diversity-robustness evidence.)

## Limitations

- **Supply = Google Fonts only** — an open-source-stylistic-choice proxy, not total
  font supply; under-counts commercial (Monotype/Adobe), system, and SIL fonts.
- **Demand is a proxy, not in the score** (see `DEMAND_PROVENANCE.md`); HTTP Archive /
  CrUX is Chrome-biased and under-counts CJK / low-Chrome regions.
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
this repo. Audit the demand axis with
`uv run python exposure_research/demand_audit.py`. The heavy index stages
(`diversity` = ViT/GPU, `complexity` = fontTools over a Google Fonts clone) and a
fresh `exposure`/`support` pull draw on external data; their outputs are committed.

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
