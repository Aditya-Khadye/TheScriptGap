# The Script Gap

The Script Gap is a research framework that identifies writing systems widely used in the real world but under-supported in digital typography. By combining web exposure data, font availability, engineering complexity, and visual diversity into a single Script Servedness Score, it surfaces the highest-impact gaps and provides a data-driven roadmap for prioritizing script support.

## Methodology

We quantify underservedness through four complementary indices:

- **Exposure Index** — real web reading demand per script, derived from Google's Chrome User Experience Report (CrUX) via BigQuery.
- **Support Index** — distinct font family count per script, pulled from Google Fonts.
- **Complexity Index** — engineering difficulty per script, computed from font binaries using fontTools (glyph expansion ratio, vertical footprint, infrastructure friction).
- **Diversity Index** — visual variety of available fonts, measured by Vision Transformer (ViT-B/16) embeddings with ResNet-50 and classical pixel-wise ablations.

These four indices are standardized and fed into K-Means clustering to produce the final underservedness tiers.

## Repository Structure

```
TheScriptGap/
├── exposure_research/          # CrUX BigQuery pipeline (Exposure Index)
│   └── output/                 # Exposure CSVs
├── support_research/           # Google Fonts analysis (Support Index)
│   └── output/                 # Font count and script relation CSVs
├── similarity_research/        # Complexity + Diversity pipelines
│   ├── script_similarity_pipeline.py    # fontTools Complexity Index
│   ├── similarity_index_*.csv
│   └── diversity_research/     # ViT, CNN, and classical CV pipelines
│       ├── script_diversity_vit_pipeline.py
│       ├── script_diversity_cnn_pipeline.py
│       ├── script_diversity_classical_cv.py
│       ├── vit_outputs_100/    # 100-glyph ViT results
│       ├── vit_outputs/        # 10-glyph ViT results
│       └── classical_cv_outputs/
├── final_model/                # K-Means clustering and tier assignment
│   └── kmeans_clustering.py
├── data_viz/                   # Heatmaps and visualizations
│   └── generate_heatmap.py
└── README.md
```

## Key Findings

K-Means clustering across all four indices produces two tiers:

- **Underserved** — Devanagari, Arabic, Bengali, Tamil, Telugu
- **Well served** — Cyrillic, Katakana, Han

Devanagari ranks most underserved: highest engineering complexity, lowest font diversity among high-demand scripts, and roughly 3,100 web page appearances per available font family. The four Indic scripts cluster together in embedding space, suggesting font development investments in one likely transfer to the others.

## Robustness

The diversity rankings hold across methodologies. A ViT-B/16 vs. ResNet-50 ablation produced a Spearman rank correlation of ρ = 0.881 (p = 0.004), and classical pixel-wise features preserved the same tier assignments — indicating the underservedness signal is robust to model choice rather than an artifact of any single approach.

## Partners

Commissioned by The Readability Consortium, collaborated with Monotype, Google Fonts, and Adobe.
