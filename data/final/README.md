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
| `exposure_context_proxy` | web font-request volume — **context only, not in the score** |

**Formula** (single source of truth: `analysis/scoring.py`):

```
SSS = log_support_norm − similarity_norm        (similarity = 1 − diversity_index)
```

Demand and complexity are deliberately excluded (see
`exposure_research/DEMAND_PROVENANCE.md` and the project README).

**Regenerate:** `uv run python pipeline.py` (the viz stage writes this file).
**Verify:** `uv run --with pytest --with pandas --with numpy pytest -q` asserts this
file equals the formula recomputed from the raw inputs.

## `robustness.md`

Robustness & sensitivity report (ViT-vs-ResNet/classical rank correlations, tier
stability across diversity models, signal-weighting sensitivity). Regenerate with
`uv run python analysis/robustness.py`.
