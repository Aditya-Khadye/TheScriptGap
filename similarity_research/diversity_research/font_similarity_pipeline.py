"""
===============================================================================
Font Similarity Pipeline — ViT Embedding Pairwise Scores
===============================================================================
Project:  TRC / Monotype — Identifying Underserved Scripts
Author:   Generated from existing ViT diversity pipeline
Purpose:  Compute pairwise font similarity scores for a chosen script
          using pretrained Vision Transformer embeddings.

Usage:
    1. Ensure Google Fonts is cloned in ./fonts/
    2. Run: python font_similarity_pipeline.py --script Tamil
    3. Output: font_similarity_pairs_<script>.csv
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

from script_diversity_vit_100 import REFERENCE_CHARS, get_flat_reference_chars

from sklearn.preprocessing import QuantileTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOOGLE_FONTS_DIR = Path("./fonts")
OUTPUT_DIR = Path("./font_similarity_outputs/full_font_similarity_pairs")
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"

CANVAS_SIZE = 224
FONT_RENDER_SIZE = 160
MIN_CODEPOINT_COVERAGE = 10
MAX_FONTS_PER_SCRIPT = 99999

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


def run_font_similarity_pipeline(
    fonts_dir: Path,
    script_name: str = "Tamil",
    max_fonts_per_script: int = MAX_FONTS_PER_SCRIPT,
    device: Optional[str] = None,
) -> pd.DataFrame:
    font_paths = discover_fonts_for_script(fonts_dir, script_name)
    if not font_paths:
        raise RuntimeError(f"No fonts found for script {script_name}")
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
    return result_df

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
    
    # Optional: drop the intermediate normal column
    df = df.drop(columns=['similarity_normal'])
    
    return df

def main() -> pd.DataFrame:
    import argparse
    parser = argparse.ArgumentParser(description="Compute pairwise font similarity for a script")
    parser.add_argument("--script", default="Tamil", help="Script name to analyze")
    parser.add_argument("--max-fonts", type=int, default=MAX_FONTS_PER_SCRIPT,
                        help="Maximum number of fonts to analyze")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device for PyTorch (cuda or cpu). Defaults to cuda if available."
    )
    args = parser.parse_args()

    if not GOOGLE_FONTS_DIR.exists():
        logger.error(f"Google Fonts directory not found at {GOOGLE_FONTS_DIR}")
        sys.exit(1)
    OUTPUT_DIR.mkdir(exist_ok=True)
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    logger.info(f"Running font similarity pipeline for script: {args.script}")
    results = run_font_similarity_pipeline(GOOGLE_FONTS_DIR, args.script, args.max_fonts, args.device)

    results['source'] = results['source'].apply(cleanup_font_name)
    results['target'] = results['target'].apply(cleanup_font_name)

    results = standardize_similarity(results)

    print(results.head())

    output_file = OUTPUT_DIR / f"font_similarity_pairs_{args.script}.csv"
    results.to_csv(output_file, index=False)
    logger.info(f"Saved pairwise similarity CSV → {output_file}")
    return results


if __name__ == "__main__":
    main()
