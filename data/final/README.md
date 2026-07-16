# `data/final/` — canonical outputs

Generated artifacts. Do not edit by hand; regenerate with the pipeline.

## `script_servedness.csv`

The canonical Script Servedness Score result (v1.0).

| column | meaning |
|---|---|
| `script` | writing system |
| `tier` | Underserved (<0.3) · Moderately served (<0.6) · Well served (≥0.6) |
| `servedness_score` | SSS, min–max normalized to [0, 1] (higher = better served) |
| `support_gf_families` | distinct open-source font families (Google Fonts) |
| `diversity_index` | visual variety of those fonts (ViT-B/16), 0–1 |
| `exposure_demand` | web font-request volume (HTTP Archive) — the demand term |

**Formula** (single source of truth: `analysis/scoring.py`):

```
effective_choice = support_norm × (1 − similarity_norm)     # similarity = 1 − diversity
SSS = effective_choice / log10(exposure)                    # choice per (log) demand
```

Real font choice (support × diversity) relative to web demand. Complexity
("engineering cost") is reported separately as a prioritization signal, not in the
SSS. Demand is the weakest input — see `exposure_research/DEMAND_PROVENANCE.md`.

**Regenerate:** `uv run python pipeline.py` (the viz stage writes this file).
**Verify:** `uv run --with pytest --with pandas --with numpy pytest -q` asserts this
file equals the formula recomputed from the raw inputs.

## `robustness.md`

Robustness & sensitivity report — diversity-model rank correlations (ViT/ResNet/
classical), the log-demand denominator vs the original epsilon, demand sensitivity,
and tier stability across diversity models. Regenerate with
`uv run python analysis/robustness.py`.

## Source data snapshots

For citation and reproducibility, the committed index inputs derive from these
pinned sources (record new values here whenever an index is regenerated):

| Input | Source | Pinned snapshot |
|---|---|---|
| Exposure (demand) | HTTP Archive `crawl.requests` via BigQuery | **2026-03-01 monthly crawl**, desktop client |
| Support (font families) | Google Fonts Developer API | API pull, June 2026 (date not recorded — record on next refresh) |
| Diversity + Complexity | `github.com/google/fonts` clone | commit `TBD` — maintainers: run `git -C "$GOOGLE_FONTS_DIR" rev-parse HEAD` and record here |

The diversity/complexity indices are computed from font binaries in the Google
Fonts clone, so the clone's commit SHA is the definitive version pin for the
paper. The demand pull is reproducible given the crawl date + client above.
