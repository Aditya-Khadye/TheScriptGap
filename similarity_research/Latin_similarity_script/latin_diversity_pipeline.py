"""
===============================================================================
Latin Diversity Baseline - Vision Transformer (ViT) Embedding Pipeline
===============================================================================
Project:  TRC / Monotype - Identifying Underserved Scripts
Author:   Aditya (UCF MIT2 Lab)
Purpose:  Establish the LATIN reference baseline for the Script Diversity
          Index. The non-Latin pipeline measures each script's visual font
          diversity, but Latin itself was never measured. This script runs the
          identical ViT-B/16 embedding method on Latin fonts so Latin's
          diversity sits on the same scale as everything else.

What "diversity" means here:
    For Latin, render a fixed set of reference letters across many fonts, embed
    each glyph with a frozen ViT-B/16 (768-dim), average per font, and take the
    mean pairwise cosine distance between fonts. High = fonts look visually
    distinct (a rich ecosystem with real design choices). Latin is expected to
    land high: it has the deepest, most varied font ecosystem in Google Fonts.
    Combined with its low complexity, Latin anchors the "well-served" corner
    that underserved scripts (high complexity, low diversity) are measured
    against.

Differences from the non-Latin pipeline (deliberate):
    1. NO Latin-dominant filter. Diversity embeds only the rendered Latin
       glyphs, so a multi-script font (CJK, Arabic, etc.) contributes its real
       Latin type design, which is legitimate. This matches how the non-Latin
       run treated every script: any font with coverage counts.
    2. NO min-max normalization. It is degenerate on a single script. This
       outputs Latin's RAW diversity metrics. To get Latin's diversity_index on
       the [0, 1] scale, normalize mean_cosine_distance against the min/max from
       your diversity_index_results.csv, the same move used for complexity.

Fixes folded in from the non-Latin pipeline:
    - save_pairwise_similarity is now called AFTER font_avg_matrix is built
      (the original referenced it one line early -> UnboundLocalError).
    - FONT_SIMILARITY_DIR is created in main() (the original never made it).
    - Device auto-selects MPS (your M2 Max GPU) > CUDA > CPU.

Usage:
    1. Ensure Google Fonts is cloned in ./fonts/
    2. python3 latin_diversity_pipeline.py
       (if you hit an MPS op error: PYTORCH_ENABLE_MPS_FALLBACK=1 python3 ...)
    3. Output: latin_diversity_summary.csv, latin_diversity.json, embeddings/
===============================================================================
"""

import sys
import json
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOOGLE_FONTS_DIR    = Path("./fonts")              # set/anchor as needed
OUTPUT_DIR          = Path("./latin_vit_outputs")
EMBEDDINGS_DIR      = OUTPUT_DIR / "embeddings"
FONT_SIMILARITY_DIR = OUTPUT_DIR / "font_similarity_pairs"

# Rendering parameters (standardized so the only variation the model sees is
# the glyph's design)
CANVAS_SIZE      = 224     # ViT-B/16 input
FONT_RENDER_SIZE = 160

# Sampling. Latin has thousands of fonts; pairwise cost is O(n^2). Cap at the
# same 100 the non-Latin run used so Latin's diversity is computed on a
# comparable sample. Fixed seed for reproducibility.
MAX_FONTS   = 100
RANDOM_SEED = 42

# Latin reference letters: 10 capitals chosen for varied geometry, present in
# essentially every Latin font, and directly comparable to how Cyrillic (the
# closest bicameral analog) was handled with 10 capitals.
#   diagonal: A K   round+straight: B R   straight: E   round/complex: G O Q
#   zigzag: M   double curve: S
# Add lowercase 'a' and 'g' if you want to capture single vs double-story
# variation, but that makes Latin less comparable to the capital-based scripts.
REFERENCE_CHARS = {
    "Latin": ["A", "B", "E", "G", "K", "M", "O", "Q", "R", "S"],
}

# Latin detection ranges (letter-bearing blocks), same set as the E/V/F baseline
TARGET_SCRIPTS = {
    "Latin": [
        (0x0041, 0x005A),   # A-Z
        (0x0061, 0x007A),   # a-z
        (0x00C0, 0x00FF),   # Latin-1 Supplement (accented letters)
        (0x0100, 0x017F),   # Latin Extended-A
        (0x0180, 0x024F),   # Latin Extended-B
        (0x1E00, 0x1EFF),   # Latin Extended Additional
    ],
}

MIN_CODEPOINT_COVERAGE = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def pick_device() -> str:
    """MPS (Apple Silicon GPU) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def write_metadata(output_dir: Path, model: dict, rendering: dict, payload_extra: dict) -> None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": commit,
        "model": model,
        "rendering": rendering,
        **payload_extra,
    }
    (output_dir / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")


def save_pairwise_similarity(script_name: str, font_names: List[str],
                             embeddings: np.ndarray, out_dir: Path) -> None:
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
# STEP 1: Frozen ViT-B/16 feature extractor
# ===========================================================================

class ViTFeatureExtractor:
    """
    Pretrained ViT-B/16 (ImageNet, frozen) as a 768-dim feature extractor.
    Input 224x224 RGB, output the [CLS] embedding. Frozen because we measure
    embedding variance, not classification, so no fine-tuning is needed and
    freezing keeps the embeddings reproducible.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        logger.info(f"Loading pretrained ViT-B/16 on device: {device}")
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads.head = nn.Identity()   # drop classifier -> raw 768-dim
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        logger.info("ViT-B/16 feature extractor ready (768-dim embeddings)")

    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([self.transform(im) for im in batch]).to(self.device)
            emb = self.model(tensors).cpu().numpy()
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            all_embeddings.append(emb / norms)
        return np.vstack(all_embeddings)


# ===========================================================================
# STEP 2: Render standardized glyph images
# ===========================================================================

def render_glyph(font_path: str, character: str,
                 canvas_size: int = CANVAS_SIZE,
                 font_size: int = FONT_RENDER_SIZE) -> Optional[Image.Image]:
    """Render one character centered on a white 224x224 grayscale canvas."""
    try:
        pil_font = ImageFont.truetype(font_path, size=font_size)
        img = Image.new("L", (canvas_size, canvas_size), color=255)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), character, font=pil_font)
        if bbox is None or (bbox[2] - bbox[0]) == 0 or (bbox[3] - bbox[1]) == 0:
            return None
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw < 5 or th < 5:
            return None
        x = (canvas_size - tw) // 2 - bbox[0]
        y = (canvas_size - th) // 2 - bbox[1]
        draw.text((x, y), character, font=pil_font, fill=0)
        return img
    except Exception as e:
        logger.debug(f"Could not render '{character}' from {font_path}: {e}")
        return None


def check_font_has_chars(font_path: str, characters: List[str]) -> List[str]:
    """Return the subset of reference chars the font actually maps."""
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        font.close()
        if not cmap:
            return []
        return [c for c in characters if ord(c) in cmap]
    except Exception:
        return []


# ===========================================================================
# STEP 3: Discover Latin fonts
# ===========================================================================

def discover_fonts_per_script(fonts_dir: Path) -> Dict[str, List[str]]:
    """Group font files by target script (here: Latin only)."""
    from collections import defaultdict
    font_extensions = {".ttf", ".otf"}
    script_fonts = defaultdict(list)
    font_files = [f for f in fonts_dir.rglob("*") if f.suffix.lower() in font_extensions]
    logger.info(f"Scanning {len(font_files)} font files for Latin coverage...")

    for filepath in font_files:
        try:
            font = TTFont(filepath, fontNumber=0)
            cmap = font.getBestCmap()
            font.close()
            if not cmap:
                continue
            codepoints = set(cmap.keys())
            for script_name, ranges in TARGET_SCRIPTS.items():
                count = sum(1 for cp in codepoints
                            for start, end in ranges if start <= cp <= end)
                if count >= MIN_CODEPOINT_COVERAGE:
                    script_fonts[script_name].append(str(filepath))
        except Exception:
            continue

    for script, fonts in script_fonts.items():
        logger.info(f"  {script}: {len(fonts)} fonts")
    return dict(script_fonts)


# ===========================================================================
# STEP 4: Diversity metrics
# ===========================================================================

def compute_diversity_metrics(embeddings: np.ndarray, font_names: List[str]) -> Dict:
    """Mean/std pairwise cosine distance, embedding spread, effective dims."""
    n = embeddings.shape[0]
    if n < 2:
        return {"mean_cosine_distance": 0.0, "std_cosine_distance": 0.0,
                "embedding_spread": 0.0, "effective_dimensions": 0, "n_embeddings": n}

    sim = embeddings @ embeddings.T
    iu = np.triu_indices(n, k=1)
    distances = 1.0 - sim[iu]
    mean_distance = float(np.mean(distances))
    std_distance = float(np.std(distances))
    embedding_spread = float(np.mean(np.std(embeddings, axis=0)))

    centered = embeddings - embeddings.mean(axis=0)
    try:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        ev = S ** 2 / np.sum(S ** 2)
        effective_dims = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
    except Exception:
        effective_dims = 0

    return {"mean_cosine_distance": mean_distance, "std_cosine_distance": std_distance,
            "embedding_spread": embedding_spread, "effective_dimensions": effective_dims,
            "n_embeddings": n}


# ===========================================================================
# STEP 5: Run the Latin diversity pipeline
# ===========================================================================

def run_latin_diversity(fonts_dir: Path, max_fonts: int = MAX_FONTS) -> Optional[Dict]:
    script_fonts = discover_fonts_per_script(fonts_dir)
    latin_paths = script_fonts.get("Latin", [])
    if not latin_paths:
        logger.error("No Latin fonts found.")
        return None

    extractor = ViTFeatureExtractor(device=pick_device())
    ref_chars = REFERENCE_CHARS["Latin"]

    np.random.seed(RANDOM_SEED)
    if len(latin_paths) > max_fonts:
        sampled = list(np.random.choice(latin_paths, max_fonts, replace=False))
        logger.info(f"Sampled {max_fonts} of {len(latin_paths)} Latin fonts (seed={RANDOM_SEED})")
    else:
        sampled = latin_paths

    font_avg_embeddings, font_names_used, all_glyph_embeddings = [], [], []

    for font_path in sampled:
        supported = check_font_has_chars(font_path, ref_chars)
        if len(supported) < 3:
            continue
        glyph_images = [g for g in (render_glyph(font_path, c) for c in supported) if g is not None]
        if len(glyph_images) < 3:
            continue
        glyph_embeddings = extractor.extract_batch(glyph_images)
        all_glyph_embeddings.append(glyph_embeddings)
        font_avg = glyph_embeddings.mean(axis=0)
        norm = np.linalg.norm(font_avg)
        if norm == 0:
            continue
        font_avg_embeddings.append(font_avg / norm)
        font_names_used.append(Path(font_path).stem)

    if len(font_avg_embeddings) < 2:
        logger.error(f"Only {len(font_avg_embeddings)} fonts produced embeddings.")
        return None

    # Build the matrix FIRST, then save pairwise (fixes the original ordering bug)
    font_avg_matrix = np.vstack(font_avg_embeddings)
    save_pairwise_similarity("Latin", font_names_used, font_avg_matrix, FONT_SIMILARITY_DIR)

    metrics = compute_diversity_metrics(font_avg_matrix, font_names_used)

    np.save(EMBEDDINGS_DIR / "Latin_embeddings.npy", font_avg_matrix)
    pd.DataFrame({
        "font_name": font_names_used,
        "script": "Latin",
        "row_index": range(len(font_names_used)),
    }).to_csv(EMBEDDINGS_DIR / "Latin_font_names.csv", index=False)

    logger.info(
        f"Latin: {len(font_avg_embeddings)} fonts, "
        f"{sum(e.shape[0] for e in all_glyph_embeddings)} glyphs, "
        f"mean_cosine_dist={metrics['mean_cosine_distance']:.4f}, "
        f"eff_dims={metrics['effective_dimensions']}"
    )

    return {
        "script": "Latin",
        "total_fonts_available": len(latin_paths),
        "fonts_analyzed": len(font_avg_embeddings),
        "glyphs_rendered": sum(e.shape[0] for e in all_glyph_embeddings),
        "reference_chars_used": len(ref_chars),
        **metrics,
    }


# ===========================================================================
# STEP 6: Reporting
# ===========================================================================

def print_report(result: Dict) -> None:
    print("\n" + "=" * 80)
    print("LATIN DIVERSITY BASELINE - ViT-B/16")
    print("=" * 80)
    print(f"\nFonts available:  {result['total_fonts_available']}")
    print(f"Fonts analyzed:   {result['fonts_analyzed']}  (sample cap {MAX_FONTS})")
    print(f"Glyphs rendered:  {result['glyphs_rendered']}")
    print(f"Reference letters: {''.join(REFERENCE_CHARS['Latin'])}")
    print("\nDiversity metrics:")
    print(f"  mean_cosine_distance = {result['mean_cosine_distance']:.4f}   <- the baseline value")
    print(f"  std_cosine_distance  = {result['std_cosine_distance']:.4f}")
    print(f"  embedding_spread     = {result['embedding_spread']:.4f}")
    print(f"  effective_dimensions = {result['effective_dimensions']}")
    print("\n" + "-" * 80)
    print("To place Latin on the [0,1] Diversity Index, min-max its")
    print("mean_cosine_distance against the range in diversity_index_results.csv")
    print("(same step used to score Latin's complexity). Latin is expected high.")
    print("=" * 80)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    if not GOOGLE_FONTS_DIR.exists():
        logger.error(
            f"Google Fonts directory not found at {GOOGLE_FONTS_DIR}\n"
            f"Clone it: git clone --depth 1 https://github.com/google/fonts"
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    FONT_SIMILARITY_DIR.mkdir(parents=True, exist_ok=True)   # created (original did not)

    logger.info("Starting Latin Diversity Pipeline...")
    result = run_latin_diversity(GOOGLE_FONTS_DIR)
    if result is None:
        sys.exit(1)

    print_report(result)

    # Raw summary (mirrors diversity_index_summary.csv minus the degenerate index)
    pd.DataFrame([{
        "script": "Latin",
        "mean_cosine_distance": result["mean_cosine_distance"],
        "std_cosine_distance": result["std_cosine_distance"],
        "embedding_spread": result["embedding_spread"],
        "effective_dimensions": result["effective_dimensions"],
        "fonts_analyzed": result["fonts_analyzed"],
        "glyphs_rendered": result["glyphs_rendered"],
    }]).to_csv(OUTPUT_DIR / "latin_diversity_summary.csv", index=False)
    logger.info(f"Saved summary -> {OUTPUT_DIR / 'latin_diversity_summary.csv'}")

    with open(OUTPUT_DIR / "latin_diversity.json", "w") as f:
        json.dump({"device": pick_device(), "max_fonts": MAX_FONTS,
                   "random_seed": RANDOM_SEED, **result}, f, indent=2)
    logger.info(f"Saved full result -> {OUTPUT_DIR / 'latin_diversity.json'}")

    write_metadata(
        OUTPUT_DIR,
        model={"name": "ViT-B/16", "weights": "IMAGENET1K_V1", "feature_dim": 768},
        rendering={"canvas_size": CANVAS_SIZE, "font_render_size": FONT_RENDER_SIZE},
        payload_extra={"script": "Latin", "fonts_analyzed": result["fonts_analyzed"]},
    )
    return result


if __name__ == "__main__":
    main()
