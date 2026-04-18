"""
===============================================================================
Script Diversity Index — Classical Computer Vision Pipeline (Pixel-Wise)
===============================================================================
Project:  TRC / Monotype — Identifying Underserved Scripts
Author:   Aditya (UCF MIT2 Lab)
Purpose:  Measure font diversity using classical computer vision features —
          no neural networks, no pretrained models, pure pixel analysis.

Why classical CV?
    ViT and CNN both use deep learning with ImageNet pretraining. Classical
    CV gives us a third, fully independent comparison that uses zero learned
    weights. If all three approaches (ViT, CNN, classical) agree on the
    diversity rankings, we have extremely strong evidence the signal is real.

    This is the most interpretable approach — every feature has a direct
    visual meaning that anyone can understand without knowing ML.

Feature Vector (per font):
    1.  Pixel Density          — What fraction of the canvas is ink vs white
                                  Captures stroke weight (bold vs light fonts)
    2.  Horizontal Projection  — Row-by-row pixel sums (normalized)
                                  Captures vertical stroke distribution
    3.  Vertical Projection    — Column-by-column pixel sums (normalized)
                                  Captures horizontal stroke distribution
    4.  HOG Features           — Histogram of Oriented Gradients
                                  Captures stroke direction and edge structure
    5.  Zone Density (3x3)     — Canvas divided into 9 zones, ink per zone
                                  Captures where ink sits in the character space
    6.  Contour Count          — Number of distinct stroke regions
                                  Captures structural complexity
    7.  Aspect Ratio of Ink    — Width/height of the bounding box of ink
                                  Captures character proportions

All features are concatenated into a single descriptor vector per font,
then L2-normalized. Diversity = mean pairwise cosine distance between fonts.

Ablation value:
    Compare rankings with ViT (Spearman ρ) to assess whether deep learning
    captures something classical CV misses, or whether the signal is so
    strong that even simple pixel statistics recover it.

Data Source: Google Fonts (github.com/google/fonts)
Usage:
    python script_diversity_classical_cv.py
    Output: classical_cv_outputs/
===============================================================================
"""

import sys
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

# Classical CV imports
try:
    from skimage.feature import hog
    from skimage import exposure as sk_exposure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("WARNING: scikit-image not found. HOG features will be skipped.")
    print("Install with: pip install scikit-image")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOOGLE_FONTS_DIR = Path("./fonts")
OUTPUT_DIR = Path("./classical_cv_outputs")
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"

CANVAS_SIZE = 128    # Smaller than ViT — classical CV doesn't need 224px
FONT_RENDER_SIZE = 100
MIN_CODEPOINT_COVERAGE = 10
MAX_FONTS_PER_SCRIPT = 100
N_ZONES = 3          # 3×3 = 9 spatial zones

# Same reference characters as ViT (10 per script for speed)
# Using 10 here since classical features are cheaper to compute
REFERENCE_CHARS = {
    "Cyrillic": [
        "\u0410", "\u0411", "\u0412", "\u0414", "\u0416",
        "\u041A", "\u041C", "\u0424", "\u0429", "\u042F",
    ],
    "Katakana": [
        "\u30A2", "\u30AB", "\u30B5", "\u30BF", "\u30CA",
        "\u30CF", "\u30DE", "\u30E4", "\u30E9", "\u30EF",
    ],
    "Devanagari": [
        "\u0915", "\u0916", "\u0917", "\u0920", "\u0924",
        "\u0928", "\u092A", "\u092E", "\u0930", "\u0938",
    ],
    "Arabic": [
        "\u0627", "\u0628", "\u062A", "\u062C", "\u062F",
        "\u0631", "\u0633", "\u0639", "\u0641", "\u0645",
    ],
    "Han": [
        "\u4E00", "\u4EBA", "\u5927", "\u5B57", "\u6587",
        "\u7528", "\u8AAD", "\u66F8", "\u9053", "\u98A8",
    ],
    "Bengali": [
        "\u0995", "\u0996", "\u0997", "\u09A4", "\u09A8",
        "\u09AA", "\u09AE", "\u09B0", "\u09B2", "\u09B8",
    ],
    "Tamil": [
        "\u0B85", "\u0B86", "\u0B87", "\u0B95", "\u0B9A",
        "\u0BA4", "\u0BA8", "\u0BAA", "\u0BAE", "\u0BB5",
    ],
    "Telugu": [
        "\u0C05", "\u0C06", "\u0C15", "\u0C17", "\u0C1A",
        "\u0C24", "\u0C28", "\u0C2A", "\u0C2E", "\u0C35",
    ],
}

TARGET_SCRIPTS = {
    "Cyrillic":    [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "Katakana":    [(0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    "Devanagari":  [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Arabic":      [(0x0600, 0x06FF), (0x0750, 0x077F),
                    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Han":         [(0x4E00, 0x9FFF), (0x3400, 0x4DBF),
                    (0x20000, 0x2A6DF), (0xF900, 0xFAFF)],
    "Bengali":     [(0x0980, 0x09FF)],
    "Tamil":       [(0x0B80, 0x0BFF)],
    "Telugu":      [(0x0C00, 0x0C7F)],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# STEP 1: Glyph Rendering
# ===========================================================================

def render_glyph(font_path: str, character: str,
                 canvas_size: int = CANVAS_SIZE,
                 font_size: int = FONT_RENDER_SIZE) -> Optional[np.ndarray]:
    """
    Render a glyph and return as a numpy array (0=ink, 255=white).
    Returns None if the glyph is empty or can't be rendered.
    """
    try:
        pil_font = ImageFont.truetype(font_path, size=font_size)
        img = Image.new("L", (canvas_size, canvas_size), color=255)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), character, font=pil_font)
        if bbox is None:
            return None

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w < 3 or text_h < 3:
            return None

        # Scale if too large
        if text_w > canvas_size - 8 or text_h > canvas_size - 8:
            scale = min((canvas_size - 8) / text_w, (canvas_size - 8) / text_h)
            new_size = max(10, int(font_size * scale))
            pil_font = ImageFont.truetype(font_path, size=new_size)
            img = Image.new("L", (canvas_size, canvas_size), color=255)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), character, font=pil_font)
            if bbox is None:
                return None
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

        x = (canvas_size - text_w) // 2 - bbox[0]
        y = (canvas_size - text_h) // 2 - bbox[1]
        draw.text((x, y), character, font=pil_font, fill=0)

        arr = np.array(img, dtype=np.float32)
        return arr

    except Exception:
        return None


# ===========================================================================
# STEP 2: Classical CV Feature Extraction
# ===========================================================================

def extract_classical_features(glyph: np.ndarray) -> np.ndarray:
    """
    Extract a classical CV feature vector from a rendered glyph image.

    The glyph is a 2D numpy array: 0 = ink (black), 255 = white background.
    We invert to a binary mask where 1 = ink, 0 = background.

    Features explained:

    1. PIXEL DENSITY (1 feature)
       Fraction of pixels that are ink. A thin script like Latin sans-serif
       has low density, a bold display font has high density. Captures
       stroke weight across the entire glyph.

    2. HORIZONTAL PROJECTION (16 features)
       Divide the canvas into 16 horizontal bands, compute the fraction
       of ink pixels in each band. This captures WHERE vertically the
       ink sits. A script with tall vowel marks above the baseline (like
       Devanagari) will have high values in the top bands. A script with
       deep descenders will have high values in the bottom bands. Fonts
       within a script that handle vowel marks differently will show
       different profiles here.

    3. VERTICAL PROJECTION (16 features)
       Same idea but column-wise. Captures horizontal stroke distribution.
       Useful for detecting whether a font has wide or narrow characters,
       and whether strokes cluster on the left vs right of the character.

    4. HOG FEATURES (variable, ~36 features)
       Histogram of Oriented Gradients — counts the direction of edges
       at different locations in the image. A font with strong horizontal
       strokes (like a serif font's cross-strokes) will have different HOG
       values than a font with diagonal calligraphic strokes. This is the
       most information-rich classical feature.

    5. ZONE DENSITY (9 features)
       Divide the canvas into a 3×3 grid of zones. Compute ink density
       in each zone. This captures spatial composition — does the character
       have more ink in the center, top-right, bottom-left, etc.? Useful
       for detecting structural differences between font families.

    6. CONTOUR COUNT (1 feature)
       Number of connected ink regions. A simple character like "一" (one)
       has 1 contour. A character with enclosed spaces like "目" (eye) has
       more. Captures structural complexity. Fonts that use different design
       approaches (open vs closed counters) will differ here.

    7. INK BOUNDING BOX ASPECT RATIO (1 feature)
       Width divided by height of the tightest bounding box around the ink.
       Captures character proportions. A condensed font will have lower
       aspect ratios than a wide font.

    All features are concatenated and L2-normalized so cosine distance
    works correctly for comparison.
    """
    # Binary mask: 1 = ink, 0 = white
    ink = (glyph < 128).astype(np.float32)
    total_pixels = ink.size

    features = []

    # --- 1. Pixel Density ---
    density = ink.sum() / total_pixels
    features.append(density)

    # --- 2. Horizontal Projection (16 bands) ---
    n_bands = 16
    band_h = CANVAS_SIZE // n_bands
    h_proj = []
    for i in range(n_bands):
        band = ink[i * band_h:(i + 1) * band_h, :]
        h_proj.append(band.sum() / (band_h * CANVAS_SIZE))
    features.extend(h_proj)

    # --- 3. Vertical Projection (16 bands) ---
    v_proj = []
    band_w = CANVAS_SIZE // n_bands
    for i in range(n_bands):
        band = ink[:, i * band_w:(i + 1) * band_w]
        v_proj.append(band.sum() / (CANVAS_SIZE * band_w))
    features.extend(v_proj)

    # --- 4. HOG Features ---
    if HAS_SKIMAGE:
        hog_features, _ = hog(
            ink,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            visualize=True,
            channel_axis=None,
        )
        features.extend(hog_features.tolist())
    else:
        # Fallback: simple gradient magnitude histogram
        gy, gx = np.gradient(ink)
        grad_mag = np.sqrt(gx**2 + gy**2)
        hist, _ = np.histogram(grad_mag, bins=36, range=(0, 1))
        features.extend((hist / (hist.sum() + 1e-8)).tolist())

    # --- 5. Zone Density (3×3 grid) ---
    zone_h = CANVAS_SIZE // N_ZONES
    zone_w = CANVAS_SIZE // N_ZONES
    for zi in range(N_ZONES):
        for zj in range(N_ZONES):
            zone = ink[zi * zone_h:(zi + 1) * zone_h, zj * zone_w:(zj + 1) * zone_w]
            features.append(zone.sum() / (zone_h * zone_w))

    # --- 6. Contour Count ---
    # Count connected components via simple flood-fill approximation
    # We use column transitions as a proxy (faster than full connected components)
    transitions = 0
    for row in ink:
        diff = np.diff(row)
        transitions += (diff > 0.5).sum()  # white to ink transitions
    # Normalize by canvas size
    features.append(transitions / CANVAS_SIZE)

    # --- 7. Bounding Box Aspect Ratio ---
    rows_with_ink = np.any(ink > 0, axis=1)
    cols_with_ink = np.any(ink > 0, axis=0)
    if rows_with_ink.any() and cols_with_ink.any():
        ink_height = rows_with_ink.sum()
        ink_width = cols_with_ink.sum()
        aspect = ink_width / (ink_height + 1e-8)
    else:
        aspect = 1.0
    features.append(aspect)

    # Concatenate and L2-normalize
    vec = np.array(features, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


# ===========================================================================
# STEP 3: Font Discovery
# ===========================================================================

def discover_fonts_per_script(fonts_dir: Path) -> Dict[str, List[str]]:
    """Walk Google Fonts and group font files by script."""
    font_extensions = {".ttf", ".otf"}
    script_fonts = defaultdict(list)
    font_files = [f for f in fonts_dir.rglob("*") if f.suffix.lower() in font_extensions]

    logger.info(f"Scanning {len(font_files)} font files...")

    for filepath in font_files:
        try:
            font = TTFont(filepath, fontNumber=0)
            cmap = font.getBestCmap()
            font.close()
            if not cmap:
                continue
            codepoints = set(cmap.keys())
            for script_name, ranges in TARGET_SCRIPTS.items():
                count = sum(1 for cp in codepoints for s, e in ranges if s <= cp <= e)
                if count >= MIN_CODEPOINT_COVERAGE:
                    script_fonts[script_name].append(str(filepath))
        except Exception:
            continue

    for script, fonts in script_fonts.items():
        logger.info(f"  {script}: {len(fonts)} fonts")
    return dict(script_fonts)


def check_font_has_chars(font_path: str, characters: List[str]) -> List[str]:
    """Return which characters the font supports."""
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        font.close()
        if not cmap:
            return []
        return [c for c in characters if ord(c[0]) in cmap]
    except Exception:
        return []


# ===========================================================================
# STEP 4: Diversity Metrics
# ===========================================================================

def compute_diversity_metrics(embeddings: np.ndarray) -> Dict:
    """Pairwise cosine diversity — identical to ViT pipeline."""
    n = embeddings.shape[0]
    if n < 2:
        return {"mean_cosine_distance": 0.0, "std_cosine_distance": 0.0,
                "embedding_spread": 0.0, "effective_dimensions": 0, "n_embeddings": n}

    sim = embeddings @ embeddings.T
    idx = np.triu_indices(n, k=1)
    dists = 1.0 - sim[idx]

    try:
        centered = embeddings - embeddings.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        explained = S ** 2 / np.sum(S ** 2)
        eff_dims = int(np.searchsorted(np.cumsum(explained), 0.90) + 1)
    except Exception:
        eff_dims = 0

    return {
        "mean_cosine_distance": float(np.mean(dists)),
        "std_cosine_distance": float(np.std(dists)),
        "embedding_spread": float(np.mean(np.std(embeddings, axis=0))),
        "effective_dimensions": eff_dims,
        "n_embeddings": n,
    }


# ===========================================================================
# STEP 5: Main Pipeline
# ===========================================================================

def run_pipeline(fonts_dir: Path) -> pd.DataFrame:
    """Full classical CV diversity pipeline."""
    script_fonts = discover_fonts_per_script(fonts_dir)
    results = []

    for script_name, font_paths in script_fonts.items():
        ref_chars = REFERENCE_CHARS.get(script_name, [])
        if not ref_chars:
            continue

        logger.info(f"\nProcessing {script_name} ({len(font_paths)} fonts)...")

        # Deterministic sampling
        font_paths = sorted(font_paths)
        np.random.seed(42)
        if len(font_paths) > MAX_FONTS_PER_SCRIPT:
            sampled = list(np.random.choice(font_paths, MAX_FONTS_PER_SCRIPT, replace=False))
        else:
            sampled = font_paths

        font_avg_features = []
        font_names = []
        total_glyphs = 0

        for font_path in sampled:
            supported = check_font_has_chars(font_path, ref_chars)
            if len(supported) < 3:
                continue

            glyph_features = []
            for char in supported:
                arr = render_glyph(font_path, char)
                if arr is not None:
                    feat = extract_classical_features(arr)
                    glyph_features.append(feat)

            if len(glyph_features) < 3:
                continue

            total_glyphs += len(glyph_features)

            # Average features across all glyphs for this font
            font_avg = np.mean(glyph_features, axis=0)
            norm = np.linalg.norm(font_avg)
            if norm > 0:
                font_avg = font_avg / norm

            font_avg_features.append(font_avg)
            font_names.append(Path(font_path).stem)

        if len(font_avg_features) < 2:
            logger.warning(f"  Insufficient fonts, skipping")
            continue

        matrix = np.vstack(font_avg_features)
        logger.info(f"  {len(font_avg_features)} fonts, {total_glyphs} glyphs, "
                    f"feature_dim={matrix.shape[1]}")

        metrics = compute_diversity_metrics(matrix)

        # Save
        np.save(EMBEDDINGS_DIR / f"{script_name}_classical_features.npy", matrix)
        pd.DataFrame({"font_name": font_names, "script": script_name}).to_csv(
            EMBEDDINGS_DIR / f"{script_name}_font_names.csv", index=False
        )

        # Per-font pairwise similarity
        sim_matrix = matrix @ matrix.T
        pairs = []
        for i in range(len(font_names)):
            for j in range(i + 1, len(font_names)):
                pairs.append({
                    "script": script_name,
                    "font_1": font_names[i],
                    "font_2": font_names[j],
                    "cosine_similarity": round(float(sim_matrix[i, j]), 4),
                    "cosine_distance": round(float(1.0 - sim_matrix[i, j]), 4),
                })
        if pairs:
            pd.DataFrame(pairs).sort_values(
                "cosine_similarity", ascending=False
            ).to_csv(EMBEDDINGS_DIR / f"{script_name}_font_pairwise.csv", index=False)

        results.append({
            "script": script_name,
            "model": "Classical CV",
            "feature_dim": matrix.shape[1],
            "fonts_analyzed": len(font_avg_features),
            "glyphs_rendered": total_glyphs,
            **metrics,
        })

        logger.info(f"  Diversity: cosine_dist={metrics['mean_cosine_distance']:.4f}, "
                    f"eff_dims={metrics['effective_dimensions']}")

    return pd.DataFrame(results)


def compute_diversity_index(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    col = "mean_cosine_distance"
    cmin, cmax = result[col].min(), result[col].max()
    if cmax - cmin == 0:
        result["diversity_index"] = 0.5
    else:
        result["diversity_index"] = (result[col] - cmin) / (cmax - cmin)
    return result.sort_values("diversity_index")


def print_report(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("SCRIPT DIVERSITY INDEX — CLASSICAL CV (Pixel-Wise)")
    print("=" * 80)
    print(f"\nFeature vector dimension: {df['feature_dim'].iloc[0]}")
    print("\n📊 Results:\n")
    cols = ["script", "fonts_analyzed", "glyphs_rendered",
            "mean_cosine_distance", "effective_dimensions"]
    print(df[cols].to_string(index=False, float_format="%.4f"))
    print("\n\n📐 Diversity Index:\n")
    print(df[["script", "diversity_index", "mean_cosine_distance"]].to_string(
        index=False, float_format="%.4f"
    ))
    top = df.loc[df["diversity_index"].idxmax()]
    bot = df.loc[df["diversity_index"].idxmin()]
    print(f"\n  Most diverse:  {top['script']} (D = {top['diversity_index']:.3f})")
    print(f"  Least diverse: {bot['script']} (D = {bot['diversity_index']:.3f})")
    print("\n" + "=" * 80)


def main():
    if not GOOGLE_FONTS_DIR.exists():
        logger.error(f"Google Fonts not found at {GOOGLE_FONTS_DIR}")
        logger.error("Clone it: git clone --depth 1 https://github.com/google/fonts")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    logger.info("Starting Classical CV Diversity Pipeline...")
    logger.info("No neural networks — pure pixel-level feature extraction")

    raw = run_pipeline(GOOGLE_FONTS_DIR)

    if raw.empty:
        logger.error("No results!")
        sys.exit(1)

    result = compute_diversity_index(raw)
    print_report(result)

    result.to_csv(OUTPUT_DIR / "diversity_classical_results.csv", index=False)
    result[["script", "diversity_index", "mean_cosine_distance",
            "fonts_analyzed", "feature_dim"]].to_csv(
        OUTPUT_DIR / "diversity_classical_summary.csv", index=False
    )

    # Merge all font pairwise files
    all_pairs = []
    for f in sorted(EMBEDDINGS_DIR.glob("*_font_pairwise.csv")):
        all_pairs.append(pd.read_csv(f))
    if all_pairs:
        pd.concat(all_pairs).to_csv(OUTPUT_DIR / "all_font_pairwise.csv", index=False)

    logger.info(f"Saved → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
