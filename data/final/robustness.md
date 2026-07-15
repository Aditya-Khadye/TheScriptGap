# Robustness & sensitivity — Script Servedness Score

## 1. Diversity signal vs. model choice (Spearman rank corr.)

Pairwise Spearman ρ over the 8 non-Latin scripts:

| | ViT-B/16 (100-glyph) | ResNet-50 (100-glyph) | Classical CV | ViT-B/16 (10-glyph) |
|---|---|---|---|---|
| ViT-B/16 (100-glyph) | 1.00 | 0.95 | 0.00 | 0.93 |
| ResNet-50 (100-glyph) | 0.95 | 1.00 | 0.12 | 0.98 |
| Classical CV | 0.00 | 0.12 | 1.00 | 0.02 |
| ViT-B/16 (10-glyph) | 0.93 | 0.98 | 0.02 | 1.00 |

**Headline:** ViT-B/16 vs ResNet-50 ρ = **0.952** (strong — the diversity ranking survives the deep-model swap). Classical CV diverges (ViT vs classical ρ = 0.00), so it is NOT cited as diversity-robustness evidence.

## 2a. Demand axis — epsilon-independence & sensitivity

The shipped SSS divides effective choice by log10(exposure). Spearman ρ of that ranking vs the original `(exposure_norm + eps)` denominator:

- vs eps = 0.1: ρ = 0.904
- vs eps = 0.5: ρ = 0.976
- vs eps = 1.0: ρ = 0.976

Removing the arbitrary 0.1 floor does **not** change the ranking (it reproduces the original gap-ratio at sensible eps). Demand's influence: over the 8 non-Latin scripts, effective-choice-only (no demand) and the full SSS give the **same ranking (ρ = 1.00)** — so the underserved ordering does not depend on the (weakest) demand axis; demand mainly separates the high-demand well-served scripts (Latin, Cyrillic).

Most-underserved four: **Bengali, Tamil, Devanagari, Telugu**.

## 2b. Servedness tiers vs. diversity model

Tier agreement when the SSS is fed ResNet / classical diversity instead of ViT:

- vs ResNet-50: 7/8 tiers identical
- vs Classical CV: 4/8 tiers identical
