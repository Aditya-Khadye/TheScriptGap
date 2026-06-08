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
## Website Live
https://aditya-khadye.github.io/TheScriptGap/
## Capstone Video
https://www.youtube.com/watch?v=wNpgtw6_ukI
## Partners

Commissioned by The Readability Consortium, addressed to Monotype, Google Fonts, and Adobe.

## Running the Complexity Pipeline

The complexity pipeline reads font binaries from a local clone of the Google Fonts repository and computes per-script metrics.

1. Clone Google Fonts (recommended location):

```bash
git clone --depth 1 https://github.com/google/fonts "$HOME/google/fonts"
```

2. Point the pipeline to your clone. Either set the environment variable:

```bash
export GOOGLE_FONTS_DIR="$HOME/google/fonts"
```

Or place the clone under the repository default path (`similarity_research/diversity_research/fonts`).

3. Run the pipeline using the pipeline script:
```bash
python pipeline.py --stages exposure --force

```

4. Run scripts individually:

From the project root:

```bash
# in TheScriptGap project root
python -m complexity.main
python complexity/main.py
```

Alternatively run from any of the local folders:

```bash
# From any of the local script folders / works with IDE's run button
python main.py
```

If the pipeline reports `Google Fonts directory not found`, double-check `GOOGLE_FONTS_DIR` points to your clone.
