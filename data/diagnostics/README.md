# `data/diagnostics/`: family-bundling and population checks

Generated from the committed v1.1 diversity run (`vit_outputs_v1_1/`); no GPU or font files needed.

```bash
uv run python -m analysis.family_diagnostic   # file vs family level diversity, bootstrap CIs
uv run python -m analysis.population_check    # subset-tag population vs >=10-codepoint threshold
```

`gf_index.csv` maps every Google Fonts file stem to its family, subsets and
primary script. It was built with `analysis/gf_index.py` from google/fonts
commit `b7c8331cf620346575487ef8db0e2c729f014675` (2026-10-15), METADATA.pb only.
The v1.1 diversity run used google/fonts commit `07d64d5a` (2026-04-09), so a few
families removed or renamed since then are unmatched and treated as off-population (listed with a `?` prefix
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

## Findings (v1.1 data)

- Ranking is stable: Spearman 0.95 (file vs family level), 0.95 (file vs population aligned); all 9 tiers unchanged in both.
- The four Indic scripts remain the least diverse group. In the family-level bootstrap every one of their rank intervals includes rank 1; in the subset-aligned bootstrap their intervals all overlap. Report the cluster, not an order.
- 79% of families in the Han sample are not tagged for Chinese (mostly Japanese and Korean fonts).
- Latin is now embedded on the same protocol as the other scripts (v1.1). In v1.0 it was embedded from 10 capitals at 160 px, which put its diversity (0.85) on a different scale.
