# `data/complexity_v11/`: Infrastructure Complexity Index, measured

v1.1 replaces the v1.0 Expansion Ratio heuristic with a measured GSUB closure,
moves to one representative font per Google Fonts family, and assigns scripts by
Google Fonts subset tags (the support term's population). v1.0 outputs in
`data/complexity/` are kept unchanged for comparison.

```bash
uv run python -m complexity.closure <google/fonts checkout> 2   # ~8 min on 2 CPUs
uv run python -m complexity.aggregate_closure
```

Font source: google/fonts commit `b7c8331cf620346575487ef8db0e2c729f014675`
(2026-10-15), 1,894 families, one file each (`representatives.csv`).

## What changed and why

| | v0 (original) | v1.0 (committed) | v1.1 (this) |
|---|---|---|---|
| Expansion numerator | `maxp.numGlyphs` (whole font) | cmap glyphs + 0.15 × base per matched GSUB FeatureRecord (font-wide) | glyphs reachable from the script's cmap glyphs through GSUB lookups registered for the script's OpenType tags (or DFLT), default-on shaping features only |
| Problem | multi-script fonts: Kosugi Katakana = 84× | one value per font file regardless of script (99% of multi-script files within 10%); uncalibrated 0.15 | none known; composite components excluded by design |
| Denominator | script codepoints | script codepoints | script codepoints excluding Arabic Presentation Forms and CJK Compatibility Ideographs (compatibility encodings of shaped forms) |
| Friction | n/a | font-wide features; RTL if the font maps any RTL codepoint | features registered for the script's OT tags (else DFLT); RTL only for Arabic |
| Unit | n/a | every file with >= 10 codepoints | one representative per subset-tagged family |

## Results (medians over families)

| Script | Families | E v0 | E v1.0 | E v1.1 [95% CI] | V | F | Composite rank v1.0 → v1.1 |
|---|---:|---:|---:|---|---:|---:|---|
| Telugu | 29 | 12.95 | 1.89 | 7.77 [6.59, 11.53] | 1.76 | 2 | 4 → 1 |
| Devanagari | 67 | 9.73 | 4.90 | 4.79 [3.88, 6.48] | 1.55 | 3 | 1 → 2 |
| Bengali | 13 | 9.31 | 3.12 | 4.56 [4.46, 5.46] | 1.59 | 3 | 2 → 3 |
| Arabic | 60 | 8.01 | 2.46 | 2.74 [2.56, 3.09] | 1.50 | 4 | 3 → 4 |
| Tamil | 19 | 7.22 | 1.69 | 2.31 [2.00, 2.65] | 1.44 | 2 | 5 → 5 |
| Cyrillic | 310 | 8.19 | 1.59 | 1.00 | 1.20 | 2 | 6 → 6 |
| Latin | 1,884 | 2.10 | 1.45 | 1.01 | 1.25 | 1 | 7 → 7 |
| Han | 29 | 1.19 | 1.45 | 1.00 | 1.00 | 2 | 8 → 8 |
| Katakana | 67 | 84.18 | 1.44 | 1.00 | 1.00 | 1 | 9 → 9 |

- Composite (0.50 E + 0.30 V + 0.20 F, min-max): Spearman 0.90 vs v1.0. Across 231 weight settings (simplex, step 0.05) the median Spearman vs default weights is 0.97 (minimum 0.66); see `by_script.csv` for each script's rank range.
- Expansion and Friction now overlap less (Spearman 0.56 across scripts vs 0.84 in v1.0).
- The weights remain an unvalidated prior. Report the three components separately; treat the composite as secondary.
