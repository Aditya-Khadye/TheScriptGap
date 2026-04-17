"""
===============================================================================
Script Diversity Index — Vision Transformer (ViT) Embedding Pipeline
===============================================================================
Project:  TRC / Monotype — Identifying Underserved Scripts
Author:   Aditya (UCF MIT2 Lab)
Purpose:  Measure the visual diversity of fonts within each writing system
          using a pretrained Vision Transformer as a feature extractor.

Methodology:
    For each script, we render a standardized set of reference characters
    across all available fonts. Each rendered glyph image is passed through
    a pretrained ViT-B/16 (frozen, no fine-tuning) to extract a 768-dim
    embedding vector. The diversity of a script's font ecosystem is then
    quantified as the mean pairwise cosine distance between all glyph
    embeddings across fonts.

    High diversity → fonts look visually distinct → healthy ecosystem
    Low diversity  → fonts all look the same → underserved / limited choices

Why ViT over CNN (ResNet)?
    - ViT's self-attention mechanism captures global glyph structure
      (the relationship between strokes across the entire character),
      while CNNs focus on local features (edges, curves).
    - For typography, global structure matters: the difference between
      a serif and sans-serif Devanagari font is in how strokes connect
      across the whole glyph, not just local edge patterns.
    - Pretrained ViT-B/16 (ImageNet) transfers well to glyph images
      because it has learned robust shape representations.

Why not fine-tune?
    - We use the ViT as a FEATURE EXTRACTOR, not a classifier.
    - We don't have labeled data for "font similarity" — and we don't
      need it. We're measuring embedding variance, not training a model.
    - Freezing weights ensures reproducibility: same model = same
      embeddings = same diversity scores.

Data Source:
    Google Fonts GitHub repository (github.com/google/fonts)

Usage:
    1. Ensure Google Fonts is cloned in ./fonts/
    2. Run: python script_diversity_vit_pipeline.py
    3. Output: diversity_index_results.csv, embeddings/ directory
===============================================================================
"""

import os
import sys
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from itertools import combinations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOOGLE_FONTS_DIR = Path("./fonts")
OUTPUT_DIR = Path("./vit_outputs")
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
FONT_SIMILARITY_DIR = OUTPUT_DIR / "font_similarity_pairs"

# Image rendering parameters — standardized so the ONLY variation
# the model sees is the actual glyph shape
CANVAS_SIZE = 224       # ViT-B/16 expects 224x224 input
FONT_RENDER_SIZE = 160  # font size in points for rendering
PADDING = 32            # pixels of padding around glyph

# Reference characters for each script
# We selected 10 high-frequency characters per script that:
#   1. Are structurally representative (cover different stroke patterns)
#   2. Are present in virtually every font for that script
#   3. Include a mix of simple and complex forms
#
# These are our "standardized test glyphs" — like picking a fixed set
# of benchmarks to evaluate every font against the same criteria.
REFERENCE_CHARS = {
    "Cyrillic": [
        "\u0410",  # А
        "\u0411",  # Б
        "\u0412",  # В
        "\u0414",  # Д
        "\u0416",  # Ж
        "\u041A",  # К
        "\u041C",  # М
        "\u0424",  # Ф
        "\u0429",  # Щ
        "\u042F",  # Я
    ],
    "Katakana": [
        "\u30A2",  # ア
        "\u30AB",  # カ
        "\u30B5",  # サ
        "\u30BF",  # タ
        "\u30CA",  # ナ
        "\u30CF",  # ハ
        "\u30DE",  # マ
        "\u30E4",  # ヤ
        "\u30E9",  # ラ
        "\u30EF",  # ワ
    ],
    "Devanagari": [
        "\u0915",  # क (ka)
        "\u0916",  # ख (kha)
        "\u0917",  # ग (ga)
        "\u0920",  # ठ (ttha)
        "\u0924",  # त (ta)
        "\u0928",  # न (na)
        "\u092A",  # प (pa)
        "\u092E",  # म (ma)
        "\u0930",  # र (ra)
        "\u0938",  # स (sa)
    ],
    "Arabic": [
        "\u0627",  # ا (alef)
        "\u0628",  # ب (ba)
        "\u062A",  # ت (ta)
        "\u062C",  # ج (jeem)
        "\u062F",  # د (dal)
        "\u0631",  # ر (ra)
        "\u0633",  # س (seen)
        "\u0639",  # ع (ain)
        "\u0641",  # ف (fa)
        "\u0645",  # م (meem)
    ],
    "Han": [
        "\u4E00",  # 一 (one)
        "\u4EBA",  # 人 (person)
        "\u5927",  # 大 (big)
        "\u5B57",  # 字 (character)
        "\u6587",  # 文 (writing)
        "\u7528",  # 用 (use)
        "\u8AAD",  # 読 (read)
        "\u66F8",  # 書 (write)
        "\u9053",  # 道 (way)
        "\u98A8",  # 風 (wind)
    ],
    "Bengali": [
        "\u0995",  # ক (ka)
        "\u0996",  # খ (kha)
        "\u0997",  # গ (ga)
        "\u09A4",  # ত (ta)
        "\u09A8",  # ন (na)
        "\u09AA",  # প (pa)
        "\u09AE",  # ম (ma)
        "\u09B0",  # র (ra)
        "\u09B2",  # ল (la)
        "\u09B8",  # স (sa)
    ],
    "Tamil": [
        "\u0B85",  # அ (a)
        "\u0B86",  # ஆ (aa)
        "\u0B87",  # இ (i)
        "\u0B95",  # க (ka)
        "\u0B9A",  # ச (ca)
        "\u0BA4",  # த (ta)
        "\u0BA8",  # ந (na)
        "\u0BAA",  # ப (pa)
        "\u0BAE",  # ம (ma)
        "\u0BB5",  # வ (va)
    ],
    "Telugu": [
        "\u0C05",  # అ (a)
        "\u0C06",  # ఆ (aa)
        "\u0C15",  # క (ka)
        "\u0C17",  # గ (ga)
        "\u0C1A",  # చ (ca)
        "\u0C24",  # త (ta)
        "\u0C28",  # న (na)
        "\u0C2A",  # ప (pa)
        "\u0C2E",  # మ (ma)
        "\u0C35",  # వ (va)
    ],
}

# Unicode ranges for script detection (same as similarity pipeline)
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

MIN_CODEPOINT_COVERAGE = 10

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
    # filename=OUTPUT_DIR / "vit_pipeline.log"
)
logger = logging.getLogger(__name__)


def save_pairwise_similarity(script_name: str, font_names: List[str], embeddings: np.ndarray, out_dir: Path):
    rows = []
    sim = embeddings @ embeddings.T
    for i in range(len(font_names)):
        for j in range(i + 1, len(font_names)):
            rows.append({
                "font_name1": font_names[i],
                "font_name2": font_names[j],
                "similarity": float(sim[i, j]),
            })
    pd.DataFrame(rows).to_csv(out_dir / f"font_similarity_pairs_{script_name}.csv", index=False)


# ===========================================================================
# STEP 1: Load the pretrained ViT as a feature extractor
# ===========================================================================

class ViTFeatureExtractor:
    """
    Wraps a pretrained ViT-B/16 as a frozen feature extractor.

    Architecture: ViT-B/16 (Vision Transformer, Base variant, 16x16 patches)
      - Input:  224×224 RGB image
      - Output: 768-dimensional embedding vector

    How it works:
      1. The image is split into 14×14 = 196 patches of 16×16 pixels each.
      2. Each patch is linearly projected to a 768-dim token.
      3. A learnable [CLS] token is prepended.
      4. 12 Transformer encoder layers process all 197 tokens with
         self-attention, allowing each patch to "see" every other patch.
      5. The [CLS] token's final representation is our embedding —
         it aggregates information about the entire glyph's structure.

    Why this works for glyphs:
      - Self-attention captures GLOBAL relationships between strokes.
        A serif font adds small strokes at stroke terminals — attention
        lets the model relate the terminal decoration to the main stroke
        body, even if they're in different patches.
      - The 768-dim embedding space is rich enough to distinguish subtle
        typographic differences (weight, contrast, x-height, terminals).
      - ImageNet pretraining gives the model robust low-level feature
        detectors (edges, curves, junctions) that transfer directly to
        glyph shapes.

    We freeze all weights because:
      - We have no labeled "font similarity" data to fine-tune on.
      - We don't NEED to fine-tune — we're measuring embedding VARIANCE,
        not classification accuracy. As long as different-looking glyphs
        map to different embeddings (which pretrained ViT does well),
        the diversity computation works.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        # Load pretrained ViT-B/16
        logger.info("Loading pretrained ViT-B/16...")
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # Remove the classification head — we want the embedding, not logits
        # The model structure is: encoder → heads.head (Linear 768→1000)
        # We replace heads.head with Identity to get raw 768-dim embeddings
        self.model.heads.head = nn.Identity()

        # Freeze all parameters — no gradient computation needed
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)

        # Standard ImageNet normalization — required because the model
        # was trained with these statistics
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # ViT expects 3-channel RGB, our glyphs are grayscale
            # We repeat the single channel 3 times
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        logger.info("ViT-B/16 feature extractor ready (768-dim embeddings)")

    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Convert a PIL glyph image to a 768-dim embedding vector.

        Args:
            image: Grayscale PIL image of a rendered glyph

        Returns:
            768-dimensional numpy array (L2-normalized)
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor).cpu().numpy().flatten()

        # L2 normalize so cosine distance = 1 - dot product
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """
        Batch extraction for efficiency — process multiple glyphs at once.

        Returns: (N, 768) numpy array of L2-normalized embeddings
        """
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            tensors = torch.stack([self.transform(img) for img in batch_imgs])
            tensors = tensors.to(self.device)

            embeddings = self.model(tensors).cpu().numpy()

            # L2 normalize each embedding
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # avoid division by zero
            embeddings = embeddings / norms

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)


# ===========================================================================
# STEP 2: Render standardized glyph images
# ===========================================================================

def render_glyph(
    font_path: str,
    character: str,
    canvas_size: int = CANVAS_SIZE,
    font_size: int = FONT_RENDER_SIZE,
) -> Optional[Image.Image]:
    """
    Render a single character using the specified font file.

    Rendering is standardized to ensure the ONLY variation the model
    sees is the glyph's visual design:
      - Fixed canvas size (224×224 pixels, matching ViT input)
      - White background, black glyph
      - Grayscale (no color information — we care about shape only)
      - Character centered on canvas using bounding box calculation
      - Anti-aliased rendering for smooth edges

    Why center the glyph?
      ViT's patch-based processing is somewhat position-sensitive.
      If we always put the glyph at (0,0), patches in the lower-right
      would always be empty, wasting model capacity. Centering ensures
      the glyph information is distributed across patches evenly.

    Returns None if the character cannot be rendered (missing glyph).
    """
    try:
        pil_font = ImageFont.truetype(font_path, size=font_size)

        # Create a grayscale canvas
        img = Image.new("L", (canvas_size, canvas_size), color=255)
        draw = ImageDraw.Draw(img)

        # Get the bounding box of the rendered character
        bbox = draw.textbbox((0, 0), character, font=pil_font)

        if bbox is None or (bbox[2] - bbox[0]) == 0 or (bbox[3] - bbox[1]) == 0:
            return None  # glyph is empty (like Adobe Blank)

        # Calculate position to center the glyph
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Skip if glyph is too small (probably a .notdef or tofu)
        if text_width < 5 or text_height < 5:
            return None

        x = (canvas_size - text_width) // 2 - bbox[0]
        y = (canvas_size - text_height) // 2 - bbox[1]

        # Render the glyph centered
        draw.text((x, y), character, font=pil_font, fill=0)

        return img

    except Exception as e:
        logger.debug(f"Could not render '{character}' from {font_path}: {e}")
        return None


def check_font_has_chars(font_path: str, characters: List[str]) -> List[str]:
    """
    Check which of the reference characters a font actually supports.
    Returns only the characters that have real glyphs (not .notdef).
    """
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        font.close()

        if not cmap:
            return []

        supported = []
        for char in characters:
            cp = ord(char)
            if cp in cmap:
                supported.append(char)

        return supported

    except Exception:
        return []


# ===========================================================================
# STEP 3: Discover fonts per script
# ===========================================================================

def discover_fonts_per_script(fonts_dir: Path) -> Dict[str, List[str]]:
    """
    Walk the Google Fonts directory and group font files by script.

    Returns: {script_name: [list of font file paths]}
    """
    font_extensions = {".ttf", ".otf"}
    script_fonts = defaultdict(list)
    font_files = [f for f in fonts_dir.rglob("*") if f.suffix.lower() in font_extensions]

    logger.info(f"Scanning {len(font_files)} font files for script coverage...")

    for filepath in font_files:
        try:
            font = TTFont(filepath, fontNumber=0)
            cmap = font.getBestCmap()
            font.close()

            if not cmap:
                continue

            codepoints = set(cmap.keys())

            for script_name, ranges in TARGET_SCRIPTS.items():
                count = sum(
                    1 for cp in codepoints
                    for start, end in ranges
                    if start <= cp <= end
                )
                if count >= MIN_CODEPOINT_COVERAGE:
                    script_fonts[script_name].append(str(filepath))

        except Exception:
            continue

    for script, fonts in script_fonts.items():
        logger.info(f"  {script}: {len(fonts)} fonts")

    return dict(script_fonts)


# ===========================================================================
# STEP 4: Compute diversity from embeddings
# ===========================================================================

def compute_diversity_metrics(
    embeddings: np.ndarray,
    font_names: List[str]
) -> Dict:
    """
    Given a matrix of embeddings (one per font-glyph pair), compute
    diversity metrics for the script.

    Metrics computed:

    1. Mean Pairwise Cosine Distance (primary metric)
       - For each pair of fonts, compute cosine distance between their
         average glyph embeddings.
       - Cosine distance = 1 - cosine_similarity
       - Range: [0, 2], but typically [0, 1] for normalized vectors
       - Higher = more diverse font ecosystem

    2. Embedding Spread (standard deviation of embeddings)
       - How spread out the embeddings are in the 768-dim space
       - Complementary to pairwise distance

    3. Effective Dimensionality (via PCA)
       - How many principal components capture 90% of variance
       - Higher = more varied visual features across fonts
       - A script where all fonts look the same would have low
         effective dimensionality (most variance in 1-2 components)

    Why cosine distance over Euclidean?
       Cosine distance measures angular separation in embedding space,
       which captures "direction" of visual features rather than
       magnitude. Two embeddings pointing in different directions are
       visually different even if they have similar norms. This is
       standard practice in representation learning.
    """
    n_embeddings = embeddings.shape[0]

    if n_embeddings < 2:
        return {
            "mean_cosine_distance": 0.0,
            "std_cosine_distance": 0.0,
            "embedding_spread": 0.0,
            "effective_dimensions": 0,
            "n_embeddings": n_embeddings,
        }

    # --- Mean Pairwise Cosine Distance ---
    # Since embeddings are L2-normalized, cosine_sim = dot product
    similarity_matrix = embeddings @ embeddings.T

    # Extract upper triangle (exclude diagonal = self-similarity of 1.0)
    upper_tri_indices = np.triu_indices(n_embeddings, k=1)
    pairwise_similarities = similarity_matrix[upper_tri_indices]
    pairwise_distances = 1.0 - pairwise_similarities

    mean_distance = float(np.mean(pairwise_distances))
    std_distance = float(np.std(pairwise_distances))

    # --- Embedding Spread ---
    embedding_spread = float(np.mean(np.std(embeddings, axis=0)))

    # --- Effective Dimensionality (PCA) ---
    # Center the embeddings
    centered = embeddings - embeddings.mean(axis=0)
    # Compute SVD (more numerically stable than eigendecomposition)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        explained_variance = S ** 2 / np.sum(S ** 2)
        cumulative_variance = np.cumsum(explained_variance)
        # Number of components for 90% variance
        effective_dims = int(np.searchsorted(cumulative_variance, 0.90) + 1)
    except:
        effective_dims = 0

    return {
        "mean_cosine_distance": mean_distance,
        "std_cosine_distance": std_distance,
        "embedding_spread": embedding_spread,
        "effective_dimensions": effective_dims,
        "n_embeddings": n_embeddings,
    }


# ===========================================================================
# STEP 5: Full pipeline — render, embed, compute diversity
# ===========================================================================

def run_diversity_pipeline(
    fonts_dir: Path,
    max_fonts_per_script: int = 100,  # cap for compute efficiency
) -> pd.DataFrame:
    """
    Full pipeline:
      1. Discover fonts per script
      2. Initialize ViT feature extractor
      3. For each script:
         a. Render reference glyphs across all fonts
         b. Extract ViT embeddings
         c. Compute per-font average embedding
         d. Calculate diversity metrics
      4. Return results DataFrame

    Why cap at 100 fonts?
      Pairwise computation scales as O(n²). With 771 Cyrillic fonts ×
      10 glyphs each, that's 7,710 embeddings and ~30M pairwise
      comparisons. Capping at 100 fonts (randomly sampled) keeps compute
      tractable while still capturing the distribution. We use a fixed
      random seed for reproducibility.
    """

    # --- Step 1: Discover fonts ---
    script_fonts = discover_fonts_per_script(fonts_dir)

    # --- Step 2: Initialize ViT ---
    extractor = ViTFeatureExtractor(device="cpu")

    # --- Step 3-4: Process each script ---
    results = []

    for script_name, font_paths in script_fonts.items():
        logger.info(f"\nProcessing {script_name} ({len(font_paths)} fonts)...")

        ref_chars = REFERENCE_CHARS.get(script_name, [])
        if not ref_chars:
            logger.warning(f"  No reference characters defined for {script_name}")
            continue

        # Sample fonts if too many (reproducible)
        np.random.seed(42)
        if len(font_paths) > max_fonts_per_script:
            sampled_paths = list(np.random.choice(
                font_paths, max_fonts_per_script, replace=False
            ))
            logger.info(f"  Sampled {max_fonts_per_script} of {len(font_paths)} fonts")
        else:
            sampled_paths = font_paths

        # Collect per-font average embeddings
        font_avg_embeddings = []
        font_names_used = []
        all_glyph_embeddings = []  # for detailed analysis

        for font_path in sampled_paths:
            # Check which reference chars this font supports
            supported_chars = check_font_has_chars(font_path, ref_chars)
            if len(supported_chars) < 3:  # need at least 3 glyphs
                continue

            # Render all supported reference glyphs
            glyph_images = []
            for char in supported_chars:
                img = render_glyph(font_path, char)
                if img is not None:
                    glyph_images.append(img)

            if len(glyph_images) < 3:
                continue

            # Extract embeddings for all glyphs in this font
            glyph_embeddings = extractor.extract_batch(glyph_images)
            all_glyph_embeddings.append(glyph_embeddings)

            # Average embedding for this font (represents the font's "style")
            font_avg = np.mean(glyph_embeddings, axis=0)
            font_avg = font_avg / np.linalg.norm(font_avg)  # re-normalize
            font_avg_embeddings.append(font_avg)
            font_names_used.append(Path(font_path).stem)

        if len(font_avg_embeddings) < 2:
            logger.warning(f"  Only {len(font_avg_embeddings)} fonts produced embeddings, skipping")
            continue

        save_pairwise_similarity(script_name, font_names_used, font_avg_matrix, FONT_SIMILARITY_DIR)

        font_avg_matrix = np.vstack(font_avg_embeddings)

        logger.info(
            f"  Extracted embeddings from {len(font_avg_embeddings)} fonts "
            f"({sum(e.shape[0] for e in all_glyph_embeddings)} total glyphs)"
        )

        # --- Compute diversity metrics ---
        metrics = compute_diversity_metrics(font_avg_matrix, font_names_used)

        # Also save embeddings for later visualization
        embedding_path = EMBEDDINGS_DIR / f"{script_name}_embeddings.npy"
        np.save(embedding_path, font_avg_matrix)

        results.append({
            "script": script_name,
            "total_fonts_available": len(font_paths),
            "fonts_analyzed": len(font_avg_embeddings),
            "glyphs_rendered": sum(e.shape[0] for e in all_glyph_embeddings),
            "reference_chars_used": len(ref_chars),
            **metrics,
        })

        logger.info(
            f"  Diversity: mean_cosine_dist={metrics['mean_cosine_distance']:.4f}, "
            f"spread={metrics['embedding_spread']:.4f}, "
            f"eff_dims={metrics['effective_dimensions']}"
        )

    return pd.DataFrame(results)


# ===========================================================================
# STEP 6: Normalize to a 0-1 Diversity Index
# ===========================================================================

def compute_diversity_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the mean cosine distance to a [0, 1] Diversity Index.

    D = 1.0 → Highest diversity (fonts look very different from each other)
    D = 0.0 → Lowest diversity (all fonts look the same)

    We use min-max scaling on mean_cosine_distance because:
      - It's our primary diversity signal
      - It's directly interpretable (distance between font embeddings)
      - The other metrics (spread, effective_dims) are complementary
        validation signals, not primary indicators
    """
    result = df.copy()

    col = "mean_cosine_distance"
    col_min = result[col].min()
    col_max = result[col].max()

    if col_max - col_min == 0:
        result["diversity_index"] = 0.5
    else:
        result["diversity_index"] = (result[col] - col_min) / (col_max - col_min)

    return result.sort_values("diversity_index", ascending=True)


# ===========================================================================
# STEP 7: Reporting
# ===========================================================================

def print_report(result_df: pd.DataFrame):
    """Pretty-print results for presentation."""

    print("\n" + "=" * 80)
    print("SCRIPT DIVERSITY INDEX — ViT EMBEDDING RESULTS")
    print("=" * 80)

    print("\n📊 Embedding Analysis Summary:\n")
    summary_cols = [
        "script", "fonts_analyzed", "glyphs_rendered",
        "mean_cosine_distance", "effective_dimensions"
    ]
    print(result_df[summary_cols].to_string(index=False, float_format="%.4f"))

    print("\n\n📐 Diversity Index (normalized 0-1):\n")
    div_cols = ["script", "diversity_index", "mean_cosine_distance",
                "embedding_spread"]
    print(result_df[div_cols].sort_values("diversity_index").to_string(
        index=False, float_format="%.4f"
    ))

    print("\n\n🔍 Interpretation:")
    print("  D → 1.0 = High visual diversity (fonts look very different)")
    print("  D → 0.0 = Low visual diversity (fonts all look similar)")

    if "diversity_index" in result_df.columns:
        most_diverse = result_df.loc[result_df["diversity_index"].idxmax()]
        least_diverse = result_df.loc[result_df["diversity_index"].idxmin()]
        print(f"\n  Most diverse:  {most_diverse['script']} "
              f"(D = {most_diverse['diversity_index']:.3f})")
        print(f"  Least diverse: {least_diverse['script']} "
              f"(D = {least_diverse['diversity_index']:.3f})")

    print("\n" + "=" * 80)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    """
    Full pipeline:
      1. Discover fonts per script from Google Fonts
      2. Render standardized reference glyphs
      3. Extract ViT-B/16 embeddings (768-dim)
      4. Compute pairwise cosine diversity per script
      5. Normalize to Diversity Index [0, 1]
      6. Export results
    """

    # --- Validate input ---
    if not GOOGLE_FONTS_DIR.exists():
        logger.error(
            f"Google Fonts directory not found at {GOOGLE_FONTS_DIR}\n"
            f"Please clone it: git clone --depth 1 https://github.com/google/fonts"
        )
        sys.exit(1)

    # --- Create output directories ---
    OUTPUT_DIR.mkdir(exist_ok=True)
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # --- Run pipeline ---
    logger.info("Starting ViT Diversity Pipeline...")
    raw_df = run_diversity_pipeline(GOOGLE_FONTS_DIR)

    if raw_df.empty:
        logger.error("No results produced!")
        sys.exit(1)

    # --- Compute normalized Diversity Index ---
    result_df = compute_diversity_index(raw_df)

    # --- Report ---
    print_report(result_df)

    # --- Save outputs ---
    result_df.to_csv(OUTPUT_DIR / "diversity_index_results.csv", index=False)
    logger.info(f"Saved results → {OUTPUT_DIR / 'diversity_index_results.csv'}")

    # Save a clean summary for merging with other indices
    summary = result_df[["script", "diversity_index", "mean_cosine_distance",
                          "fonts_analyzed"]].copy()
    summary.to_csv(OUTPUT_DIR / "diversity_index_summary.csv", index=False)
    logger.info(f"Saved summary → {OUTPUT_DIR / 'diversity_index_summary.csv'}")

    return result_df


if __name__ == "__main__":
    main()
