from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
import logging

"""
Adapted from an existing ViT diversity and script similarity pipeline
Added ResNet50 feature extraction and script-to-script similarity, with an additional script-to-Latin summary

We tested ResNet50, ResNet18(value range: 0.83-0.91) and ConvNeXt-Tiny(value range: 0.81-0.93)
ResNet50 gave the most stable similarity scores, so we used it as the final backbone

"""


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GOOGLE_FONTS_DIR = Path("./fonts")
OUTPUT_DIR = Path("./cnn_similarity_outputs")
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"

CANVAS_SIZE = 224
FONT_RENDER_SIZE = 140

MAX_FONTS_PER_SCRIPT = 20

MIN_CODEPOINT_COVERAGE = 10

MIN_SUPPORTED_REFERENCE_CHARS = 3

REFERENCE_SCRIPT = "Latin"

TARGET_SCRIPTS = {
    "Latin":       [(0x0041, 0x005A), (0x0061, 0x007A)],
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

REFERENCE_CHARS = {
    # added more structurally varied Latin chars
    "Latin": ["A", "M", "W", "O", "Q", "a", "g", "m", "n", "y"],
  
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

def get_flat_reference_chars(script_name: str) -> List[str]:
    return REFERENCE_CHARS.get(script_name, [])

# discover fonts for each script
def discover_fonts_per_script(fonts_dir: Path) -> Dict[str, List[str]]:
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


#check whether this font supports the script's reference chars, and filters out fonts that don't support enough chars
#if supported, render those characters into glyph images
def check_font_has_chars(font_path: str, characters: List[str]) -> List[str]:
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        font.close()

        if not cmap:
            return []

        supported = []
        for char in characters:
            cp = ord(char[0])
            if cp in cmap:
                supported.append(char)
        return supported

    except Exception:
        return []

def render_glyph(font_path: str, character: str,
                 canvas_size: int = CANVAS_SIZE,
                 font_size: int = FONT_RENDER_SIZE) -> Optional[Image.Image]:
    try:
        pil_font = ImageFont.truetype(font_path, size=font_size)
        img = Image.new("L", (canvas_size, canvas_size), color=255)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), character, font=pil_font)
        if bbox is None or (bbox[2] - bbox[0]) < 3 or (bbox[3] - bbox[1]) < 3:
            return None

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w > canvas_size - 10 or text_h > canvas_size - 10:
            scale = min((canvas_size - 10) / text_w, (canvas_size - 10) / text_h)
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
        return img
    except Exception:
        return None

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


#extract glyph embeddings with ResNet50
#then L2 normalization, so we can compute cosine similarity later on
#each font is represented by the average of all its glyph embeddings
class CNNFeatureExtractor:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = nn.Sequential(*list(backbone.children())[:-1])

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        outputs = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            tensors = torch.stack([self.transform(img) for img in batch_imgs]).to(self.device)

            embeddings = self.model(tensors).cpu().numpy()
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
            outputs.append(embeddings)

        return np.vstack(outputs)

#calculate the similarity between script and script
def compute_similarity(script_a: np.ndarray, script_b: np.ndarray) -> float:
   
    sims = script_a @ script_b.T
    return float(np.mean(sims))


#build the script-to-script cosine similarity matrix
def compute_matrix(script_font_embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
    scripts = list(script_font_embeddings.keys())
    matrix = np.zeros((len(scripts), len(scripts)))

    for i, s1 in enumerate(scripts):
        for j, s2 in enumerate(scripts):
            matrix[i, j] = compute_similarity(
                script_font_embeddings[s1],
                script_font_embeddings[s2]
            )

    return pd.DataFrame(matrix, index=scripts, columns=scripts)

#run the full pipeline and save the outputs
def run_similarity(fonts_dir: Path):
    script_fonts = discover_fonts_per_script(fonts_dir)
    extractor = CNNFeatureExtractor(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    script_font_embeddings = {}
    summary_rows = []

    for script_name, font_paths in script_fonts.items():
        logger.info(f"Processing {script_name} with {len(font_paths)} candidate fonts...")

        ref_chars = get_flat_reference_chars(script_name)
        if not ref_chars:
            continue

        font_paths = sorted(font_paths)
        np.random.seed(42)
        if len(font_paths) > MAX_FONTS_PER_SCRIPT:
            font_paths = list(np.random.choice(font_paths, MAX_FONTS_PER_SCRIPT, replace=False))

        font_avg_embeddings = []
        glyph_count_total = 0

        for idx, font_path in enumerate(font_paths):
            if idx % 10 == 0:
                logger.info(f"{script_name}: processed {idx}/{len(font_paths)} fonts")

            supported = check_font_has_chars(font_path, ref_chars)
            if len(supported) < MIN_SUPPORTED_REFERENCE_CHARS:
                continue

            images = []
            for char in supported:
                img = render_glyph(font_path, char)
                if img is not None:
                    images.append(img)

            if len(images) < MIN_SUPPORTED_REFERENCE_CHARS:
                continue

            glyph_embeddings = extractor.extract_batch(images)
            glyph_count_total += len(images)

            font_avg = np.mean(glyph_embeddings, axis=0)
            font_avg = normalize(font_avg)
            font_avg_embeddings.append(font_avg)

        if len(font_avg_embeddings) < 2:
            logger.warning(f"Skipping {script_name}")
            continue

        font_matrix = np.vstack(font_avg_embeddings)
        script_font_embeddings[script_name] = font_matrix
        np.save(EMBEDDINGS_DIR / f"{script_name}_font_embeddings.npy", font_matrix)

        summary_rows.append({
            "script": script_name,
            "fonts_analyzed": len(font_avg_embeddings),
            "glyphs_rendered": glyph_count_total,
            "reference_chars_used": len(ref_chars),
        })


    if not script_font_embeddings:
        raise RuntimeError("No script font embeddings were produced.")

    similarity = compute_matrix(script_font_embeddings)
    summary = pd.DataFrame(summary_rows)

    if REFERENCE_SCRIPT in similarity.columns:
        summary["reference_script"] = REFERENCE_SCRIPT
        summary["similarity_to_reference"] = summary["script"].map(
            similarity[REFERENCE_SCRIPT].to_dict()
        )

    return similarity, summary

def main():

    """
    1. defines which writing systems to analyze
    2. font discovery
    3. glyph rendering
    4. CNN embedding
    5. similarity computation
    6. save output

    """
    if not GOOGLE_FONTS_DIR.exists():
        raise FileNotFoundError(f"Google Fonts directory not found: {GOOGLE_FONTS_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    similarity, summary = run_similarity(GOOGLE_FONTS_DIR)

    similarity.to_csv(OUTPUT_DIR / "cnn_script_similarity_matrix.csv")
    summary.to_csv(OUTPUT_DIR / "cnn_script_similarity_summary.csv", index=False)

    print("Done.")
    print(summary)


if __name__ == "__main__":
    main()
