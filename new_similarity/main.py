"""
===============================================================================
Font Similarity & Diversity Pipeline — ViT Embedding Scores
===============================================================================
Project:  TRC / Monotype — Identifying Underserved Scripts
Purpose:  Compute pairwise font similarity scores and diversity metrics
          for all scripts using pretrained Vision Transformer embeddings.

Output:
    1. output/similarity_results.csv
       - Summary table with diversity metrics per script
    2. output/similarity_pairs/{script}.csv
       - Font-to-font similarity pairs within each script

Usage:
    1. Ensure Google Fonts is cloned in ./fonts/
    2. Run: python main.py
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

from .reference_chars import REFERENCE_CHARS
from sklearn.preprocessing import QuantileTransformer

try:
    from paths import SIMILARITY_DATA_DIR, GOOGLE_FONTS_DIR

except ModuleNotFoundError:
    # When running the module from different working directories, ensure
    # the project root (one level up from this package) is on sys.path.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from paths import SIMILARITY_DATA_DIR, GOOGLE_FONTS_DIR

SIMILARITY_PAIRS_DIR =  SIMILARITY_DATA_DIR / "similarity_pairs"
SIMILARITY_RESULTS_FILE =  SIMILARITY_DATA_DIR / "similarity_results.csv"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CANVAS_SIZE = 224
FONT_RENDER_SIZE = 160
MIN_CODEPOINT_COVERAGE = 10
MAX_FONTS_PER_SCRIPT = 3

TARGET_SCRIPTS = {
    # "Cyrillic":    [(0x0400, 0x04FF), (0x0500, 0x052F)],
    # "Katakana":    [(0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    # "Devanagari":  [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    # "Arabic":      [(0x0600, 0x06FF), (0x0750, 0x077F),
    #                 (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    # "Han":         [(0x4E00, 0x9FFF), (0x3400, 0x4DBF),
    #                 (0x20000, 0x2A6DF), (0xF900, 0xFAFF)],
    # "Bengali":     [(0x0980, 0x09FF)],
    "Tamil":       [(0x0B80, 0x0BFF)],
    "Telugu":      [(0x0C00, 0x0C7F)],
    # "Latin":       [(0x0000, 0x007F)],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ViTFeatureExtractor:
    def __init__(self, device: Optional[str] = None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading pretrained ViT-B/16 on device: {self.device}")
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads.head = nn.Identity()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((CANVAS_SIZE, CANVAS_SIZE)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        logger.info("ViT-B/16 feature extractor ready")

    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor).cpu().numpy().flatten()
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            tensors = torch.stack([self.transform(img) for img in batch_imgs]).to(self.device)
            embeddings = self.model(tensors).cpu().numpy()
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            all_embeddings.append(embeddings / norms)
        return np.vstack(all_embeddings)


def render_glyph(
    font_path: str,
    character: str,
    canvas_size: int = CANVAS_SIZE,
    font_size: int = FONT_RENDER_SIZE,
) -> Optional[Image.Image]:
    try:
        pil_font = ImageFont.truetype(font_path, size=font_size)
        img = Image.new("L", (canvas_size, canvas_size), color=255)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), character, font=pil_font)
        if bbox is None or (bbox[2] - bbox[0]) == 0 or (bbox[3] - bbox[1]) == 0:
            return None
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width < 5 or text_height < 5:
            return None
        x = (canvas_size - text_width) // 2 - bbox[0]
        y = (canvas_size - text_height) // 2 - bbox[1]
        draw.text((x, y), character, font=pil_font, fill=0)
        return img
    except Exception as e:
        logger.debug(f"Could not render '{character}' from {font_path}: {e}")
        return None

def get_flat_reference_chars(script_name: str) -> List[str]:
    """Flatten the tiered reference chars into a single list.
    
    Handles two formats:
    - Dict of lists: {"tier_name": [char, char, ...], ...}  (e.g. Devanagari)
    - Set of range tuples: {(start, end), ...}              (e.g. Latin)
    """
    entry = REFERENCE_CHARS.get(script_name, {})
    flat = []

    if isinstance(entry, set):
        # Codepoint Range: set of (start, end) codepoint range tuples
        for start, end in sorted(entry):
            flat.extend(chr(cp) for cp in range(start, end + 1))
    elif isinstance(entry, dict):
        # Explicit Tiered Reference: dict mapping tier names to lists of chars
        for chars in entry.values():
            flat.extend(chars)

    return flat

def check_font_has_chars(font_path: str, characters: List[str]) -> List[str]:
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        font.close()
        if not cmap:
            return []
        supported = []
        for char in characters:
            # Handle both single chars and multi-char sequences
            if len(char) == 1:
                if ord(char) in cmap:
                    supported.append(char)
            else:
                # For multi-char sequences, check if all chars are in the font
                if all(ord(c) in cmap for c in char):
                    supported.append(char)
        return supported
    except Exception:
        return []


def discover_fonts_for_script(fonts_dir: Path, script_name: str) -> List[str]:
    font_extensions = {".ttf", ".otf"}
    script_ranges = TARGET_SCRIPTS.get(script_name)
    if not script_ranges:
        raise ValueError(f"Unknown script: {script_name}")
    font_paths = []
    all_files = [f for f in fonts_dir.rglob("*") if f.suffix.lower() in font_extensions]
    logger.info(f"Scanning {len(all_files)} font files for {script_name} coverage...")
    for filepath in all_files:
        try:
            font = TTFont(filepath, fontNumber=0)
            cmap = font.getBestCmap()
            font.close()
            if not cmap:
                continue
            codepoints = set(cmap.keys())
            coverage = sum(1 for cp in codepoints for start, end in script_ranges if start <= cp <= end)
            if coverage >= MIN_CODEPOINT_COVERAGE:
                font_paths.append(str(filepath))
        except Exception:
            continue
    logger.info(f"Found {len(font_paths)} fonts covering {script_name}")
    return font_paths


def compute_font_average_embeddings(
    font_paths: List[str],
    script_name: str,
    extractor: ViTFeatureExtractor,
    max_fonts: int = MAX_FONTS_PER_SCRIPT,
    min_glyphs: int = 3,
) -> Tuple[List[str], np.ndarray]:
    ref_chars = get_flat_reference_chars(script_name)
    if not ref_chars:
        raise ValueError(f"No reference characters defined for {script_name}")
    if len(font_paths) > max_fonts:
        np.random.seed(42)
        font_paths = list(np.random.choice(font_paths, max_fonts, replace=False))
        logger.info(f"Sampled {max_fonts} fonts for {script_name}")
    
    logger.info(f"Processing {len(font_paths)} fonts for {script_name}...")
    logger.info(f"Reference characters available: {len(ref_chars)}")
    
    font_names = []
    avg_embeddings = []
    glyph_counts = []
    skipped_chars_support = 0
    skipped_render = 0
    
    for idx, font_path in enumerate(font_paths):
        font_stem = Path(font_path).stem
        supported = check_font_has_chars(font_path, ref_chars)
        
        if len(supported) < min_glyphs:
            logger.info(f"[{idx+1}/{len(font_paths)}] {font_stem}: SKIP (only {len(supported)}/{len(ref_chars)} chars supported)")
            skipped_chars_support += 1
            continue
        
        glyph_images = [render_glyph(font_path, char) for char in supported]
        glyph_images = [img for img in glyph_images if img is not None]
        
        if len(glyph_images) < min_glyphs:
            logger.info(f"[{idx+1}/{len(font_paths)}] {font_stem}: SKIP (rendered {len(glyph_images)}/{len(supported)} glyphs)")
            skipped_render += 1
            continue
        
        embeddings = extractor.extract_batch(glyph_images)
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding /= np.linalg.norm(avg_embedding) if np.linalg.norm(avg_embedding) > 0 else 1.0
        avg_embeddings.append(avg_embedding)
        font_names.append(font_stem.split("[", 1)[0])
        glyph_counts.append(len(glyph_images))
        logger.info(f"[{idx+1}/{len(font_paths)}] {font_stem}: ✓ ({len(glyph_images)} glyphs)")
    
    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Total fonts: {len(font_paths)}")
    logger.info(f"Skipped (insufficient char support): {skipped_chars_support}")
    logger.info(f"Skipped (render failures): {skipped_render}")
    logger.info(f"Successfully embedded: {len(font_names)}")
    
    if not avg_embeddings:
        logger.error(f"No fonts produced embeddings for {script_name}")
        return [], np.empty((0, 768), dtype=np.float32)
    
    return font_names, np.vstack(avg_embeddings)


def compute_pairwise_similarity(
    font_names: List[str],
    embeddings: np.ndarray,
) -> pd.DataFrame:
    if embeddings.shape[0] < 2:
        return pd.DataFrame(columns=["source", "target", "similarity"])
    similarity_matrix = embeddings @ embeddings.T
    rows = []
    n = len(font_names)
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "source": font_names[i],
                "target": font_names[j],
                "similarity": float(similarity_matrix[i, j]),
            })
    return pd.DataFrame(rows)


def compute_diversity_metrics(embeddings: np.ndarray) -> Dict:
    """Compute pairwise cosine diversity and PCA dimensionality."""
    n = embeddings.shape[0]
    if n < 2:
        return {
            "mean_cosine_distance": 0.0,
            "std_cosine_distance": 0.0,
            "embedding_spread": 0.0,
            "effective_dimensions": 0,
            "n_embeddings": n,
        }

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


def compute_diversity_index(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate diversity index from mean_cosine_distance values."""
    result = df.copy()
    col = "mean_cosine_distance"
    cmin, cmax = result[col].min(), result[col].max()
    if cmax - cmin == 0:
        result["diversity_index"] = 0.5
    else:
        result["diversity_index"] = (result[col] - cmin) / (cmax - cmin)
    return result


def cleanup_font_name(font_name: str) -> str:
    """Remove font variation tags like [wdth,wght] or [wght] from font names."""
    return re.sub(r'\[[\w,]+\]$', '', font_name).strip()


def standardize_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normal distribution transformation, then normalize to 0-1 range."""
    col_name = None
    for col in df.columns:
        if 'similarity' in col.lower():
            col_name = col
            break

    if not col_name:
        raise ValueError("No similarity column found")

    # Step 1: Apply QuantileTransformer to map to normal distribution
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    df['similarity_normal'] = qt.fit_transform(df[[col_name]])

    # Step 2: Normalize the normal-distributed values to 0-1 range
    df['similarity_normalized'] = (df['similarity_normal'] - df['similarity_normal'].min()) / \
                                   (df['similarity_normal'].max() - df['similarity_normal'].min())

    # Drop intermediate normal column
    df = df.drop(columns=['similarity_normal'])

    return df


def run_font_similarity_pipeline(
    fonts_dir: Path,
    script_name: str,
    max_fonts_per_script: int = MAX_FONTS_PER_SCRIPT,
    device: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], np.ndarray, int]:
    """
    Compute pairwise font similarity for a script.
    
    Returns:
        Tuple of (similarity_df, font_names, embeddings, total_fonts_found)
    """
    font_paths = discover_fonts_for_script(fonts_dir, script_name)
    if not font_paths:
        raise RuntimeError(f"No fonts found for script {script_name}")
    
    total_fonts_found = len(font_paths)
    
    extractor = ViTFeatureExtractor(device=device)
    font_names, embeddings = compute_font_average_embeddings(
        font_paths,
        script_name,
        extractor,
        max_fonts=max_fonts_per_script,
    )
    
    if len(font_names) < 2:
        raise RuntimeError(f"Not enough fonts produced embeddings for {script_name}")
    
    result_df = compute_pairwise_similarity(font_names, embeddings)
    return result_df, font_names, embeddings, total_fonts_found


def run_similarity_pipeline(force: bool = False) -> None:
    """Run complete pipeline for all scripts and generate output files."""
    
    if not GOOGLE_FONTS_DIR.exists():
        logger.error(f"Google Fonts directory not found at {GOOGLE_FONTS_DIR}")
        sys.exit(1)

    SIMILARITY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIMILARITY_PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    all_results = []

    for script_name in TARGET_SCRIPTS.keys():
        logger.info(f"\n=== Processing script: {script_name} ===")
        try:
            # Run similarity pipeline once per script - returns all needed data
            result_df, font_names, embeddings, total_fonts_found = run_font_similarity_pipeline(
                GOOGLE_FONTS_DIR,
                script_name=script_name,
                device=device
            )

            # Clean and standardize similarity scores
            result_df['source'] = result_df['source'].apply(cleanup_font_name)
            result_df['target'] = result_df['target'].apply(cleanup_font_name)
            result_df = standardize_similarity(result_df)

            # Save per-script similarity pairs
            output_file = SIMILARITY_PAIRS_DIR / f"{script_name}.csv"
            result_df.to_csv(output_file, index=False)
            logger.info(f"✓ Saved {script_name} pairs → {output_file}")

            # Use same embeddings to compute diversity metrics
            ref_chars = get_flat_reference_chars(script_name)
            metrics = compute_diversity_metrics(embeddings)

            # Build summary row
            summary_row = {
                "script": script_name,
                "total_fonts": total_fonts_found,
                "fonts_analyzed": metrics["n_embeddings"],
                "reference_chars": len(ref_chars),
                "glyphs_rendered": metrics["n_embeddings"] * len(ref_chars),  # Approximate
                "mean_cosine_distance": metrics["mean_cosine_distance"],
                "std_cosine_distance": metrics["std_cosine_distance"],
                "embedding_spread": metrics["embedding_spread"],
                "effective_dimensions": metrics["effective_dimensions"],
                "n_embeddings": metrics["n_embeddings"],
            }
            all_results.append(summary_row)

        except Exception as e:
            logger.error(f"Error processing {script_name}: {e}")

    # Create summary DataFrame and compute diversity index
    summary_df = pd.DataFrame(all_results)
    summary_df = compute_diversity_index(summary_df)

    # Save diversity summary
    summary_df.to_csv(SIMILARITY_RESULTS_FILE, index=False)
    logger.info(f"\n✓ Saved summary → {SIMILARITY_RESULTS_FILE}")

    print(f"\nOutput saved to: {SIMILARITY_DATA_DIR}")
    print(f"  - Summary: {SIMILARITY_RESULTS_FILE}")
    print(f"  - Pairs: {SIMILARITY_PAIRS_DIR}/")


def main() -> None:
    """Main entry point."""
    run_similarity_pipeline(force=True)


if __name__ == "__main__":
    main()

