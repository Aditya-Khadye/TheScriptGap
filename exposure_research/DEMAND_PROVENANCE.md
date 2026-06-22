# Demand Axis (Exposure Index) — Provenance & Validity

> **v1.0 status: demand is a CONTEXT signal, not a score driver.** The team has
> built the right infrastructure to fix it (an HTTP Archive `subset=` pull), but
> the *committed* per-script numbers can't yet be trusted to carry a headline.
> This note records why, and how to promote demand back into the score.
> Reproduce the audit with
> `uv run --with pandas python exposure_research/demand_audit.py`.

## 1. What the metric is, and the good news

The Exposure Index is **web font-request counts attributed to scripts**. Two
attribution paths exist in the repo:

- **`exposure_research/bigquery_pull.py` + `sql/httparchive_font_requests.sql`
  (the right approach).** Pulls Google Fonts requests from HTTP Archive and reads
  the **CSS `subset=` parameter** (e.g. `…/css?family=Roboto&subset=cyrillic`).
  That is a genuine *per-page* script signal — a page that requests the Cyrillic
  subset actually renders Cyrillic — and it largely sidesteps the bundling
  confound below. It writes `data/exposure/exposure_filtered_results.csv` and
  needs GCP credentials + a crawl date to run.
- **Font-coverage fallback** (used when the `subset=` parameter is absent, and the
  source of `data/bigquery/big_query_data.csv`). Attributes a font's request count
  to *every script its glyph set covers* — which is badly confounded.

## 2. The confound in the coverage fallback (quantified)

For the coverage path, a script inherits the popularity of any font that merely
*bundles* its glyphs. Share of each script's coverage-based demand coming from
fonts covering ≥3 scripts, with the top contributor:

| Script | % from multi-script bundlers | top contributor |
|---|--:|---|
| Telugu | 99.5% | inter (Latin UI font) |
| Tamil | 98% | inter |
| Bengali | 99% | inter |
| Devanagari | 97% | poppins |
| Han | 99.97% | noto sans jp |
| Cyrillic | 85% | roboto |
| Arabic | 56% | fontawesome (icon font) |

So in the fallback, Tamil/Telugu/Bengali "demand" is mostly **Inter**, Devanagari
**Poppins**, and Arabic largely **FontAwesome** (an icon font). The `subset=`
path is what fixes this; the fallback should be avoided for load-bearing numbers.

## 3. Why demand is still CONTEXT-only in v1.0

- **The committed `exposure_filtered_results.csv` still carries the legacy
  numbers** (Latin 20.2M, Cyrillic 2.6M, …) — i.e. not yet regenerated from a
  fresh `subset=` pull, so its provenance is not yet verified against the good
  path. Until a `bigquery_pull.py` run is committed and checked, the numbers are
  treated as context.
- **The CJK Han/Katakana split is intrinsically unreliable** even with `subset=`
  (Japanese pages use both), and CrUX/HTTP-Archive is Chrome-biased and
  under-counts CJK / low-Chrome regions.
- The robust servedness conclusion does not need demand: it rests on **support +
  diversity** (see the SSS in `data_viz/generate_heatmap.py`).

## 4. Promoting demand back into the score (roadmap)

1. **Run and commit a fresh `subset=` pull** (`python -m exposure_research.bigquery_pull --force`
   with GCP creds) and confirm it replaces the legacy numbers.
2. **Validate** the result: re-run `demand_audit.py`; confirm the Inter/Poppins/
   FontAwesome inflation is gone and per-script totals track the `subset=` field,
   not coverage.
3. **Decide the CJK handling** (merge Han+Katakana demand, or keep split with a
   documented caveat).
4. Only then reintroduce demand into the SSS (e.g. as a provision term), with a
   sensitivity check against the support+diversity-only score.

Until 1–3 are done, keep the Exposure Index labelled as a **context proxy**.
