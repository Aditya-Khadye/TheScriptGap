# `data/diagnostics/`: family-bundling and population checks

Generated from committed v1.0 data; no GPU or font files needed.

```bash
uv run python -m analysis.family_diagnostic   # file vs family level diversity, bootstrap CIs
uv run python -m analysis.population_check    # subset-tag population vs >=10-codepoint threshold
```

`gf_index.csv` maps every Google Fonts file stem to its family, subsets and
primary script. It was built with `analysis/gf_index.py` from google/fonts
commit `b7c8331cf620346575487ef8db0e2c729f014675` (2026-10-15), METADATA.pb only.
The diversity corpus is a March 2026 clone, so a few families removed or renamed
since then are unmatched and treated as off-population (listed with a `?` prefix
in `population_members.csv`).

| file | contents |
|---|---|
| `family_diagnostic_by_script.csv` | file-level, between-family, one-rep-per-family and design-level mean cosine distance; within-family distance; bootstrap CI and rank interval |
| `family_representatives.csv` | family key and representative file per script |
| `sss_file_vs_family.csv` | SSS and tiers, committed vs family-level diversity |
| `population_check_by_script.csv` | on/off-population families, aligned distance, CI, rank interval, sample coverage |
| `population_members.csv` | every sampled family with its on/off-population flag |
| `sss_file_vs_aligned.csv` | SSS and tiers, committed vs population-aligned diversity |
| `*_summary.csv` | Spearman correlations, tier agreement, Latin comparability figures |

`bootstrap_family_rep.csv` (5,000 replicates) is regenerated on each run and not committed.

## Findings (v1.0 data)

- Ranking is stable: Spearman 0.98 (file vs family level), 0.93 (file vs population aligned); all 9 tiers unchanged.
- The four Indic scripts remain the least diverse group, but every one of their bootstrap rank intervals includes rank 1. Report the cluster, not an order.
- 79% of families in the Han sample are not tagged for Chinese (mostly Japanese and Korean fonts).
- Latin was embedded from 10 capitals at 160 px; the other scripts from roughly 90 to 100 glyphs at 140 px. Fewer glyphs inflate distances (median 1.44x in the 10- vs 100-glyph runs), so Latin's committed diversity (0.85) is not on the same scale as the others (about 0.58 on the 10-glyph scale).
