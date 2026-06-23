# Robustness & sensitivity — Script Servedness Score

## 1. Diversity signal vs. model choice (Spearman rank corr.)

Pairwise Spearman ρ over the 8 non-Latin scripts:

| | ViT-B/16 | ResNet-50 | Classical CV | ViT-B/16 |
|---|---|---|---|---|
| ViT-B/16 | 1.00 | 0.95 | 0.00 | 0.93 |
| ResNet-50 | 0.95 | 1.00 | 0.12 | 0.98 |
| Classical CV | 0.00 | 0.12 | 1.00 | 0.02 |
| ViT-B/16 | 0.93 | 0.98 | 0.02 | 1.00 |

**Headline:** ViT-B/16 vs ResNet-50 ρ = **0.952** (strong — the diversity ranking survives the deep-model swap). Classical CV diverges (ViT vs classical ρ = 0.00), so it is NOT cited as diversity-robustness evidence.

## 2a. Servedness vs. signal weighting

Most-underserved four under each ranking:

- support only:   Bengali, Tamil, Han, Telugu
- diversity only: Tamil, Bengali, Devanagari, Telugu
- combined SSS:   Tamil, Bengali, Telugu, Devanagari

Underserved cluster {Tamil, Bengali, Devanagari, Telugu} is the combined bottom-4: **True**. Support-only and diversity-only each recover 3/4 of it, so the cluster is not an artifact of the weighting.

## 2b. Servedness tiers vs. diversity model

Tier agreement when the SSS is fed ResNet / classical diversity instead of ViT (8 non-Latin scripts):

- vs ResNet-50: 8/8 tiers identical
- vs Classical CV: 4/8 tiers identical
