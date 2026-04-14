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

# Want to seperate the pipeline out of the ViT model code and
#  the other script similarity code later in order to clean things up
from diversity_research import script_diversity_vit_pipeline as script_pipeline


def vit_photo_visualization(
    fonts_dir: Path,
    script: str = "Katakana", # Default to katakana just for testing
    glyph: str = "\u30BF",  # タ - a common katakana character
    max_fonts_per_script: int = 20,  # cap for compute efficiency
) -> pd.DataFrame:
    """
    This is a modified version of the run diversity pipeline from ViT research 
    focused on generating visualizations for presentation.

    Creates a gif with the specified glyph rendered from each font that
    actually contains that glyph.

    Since this is just for visualization, it uses a small subset of the
    fonts (20) and only considers fonts discovered for the given script.

    This lets us inspect the actual glyph shape that ViT sees, while
    ensuring only fonts with the requested glyph are rendered.
    """

    script_fonts = script_pipeline.discover_fonts_per_script(fonts_dir)
    script_fonts = script_fonts.get(script, [])
    if not script_fonts:
        return pd.DataFrame([])

    np.random.seed(42)
    sampled_paths = list(np.random.choice(script_fonts, min(len(script_fonts), max_fonts_per_script), replace=False))

    results = []
    gif_frames = []

    for font_path in sampled_paths:
        supported_chars = script_pipeline.check_font_has_chars(font_path, [glyph])
        if not supported_chars:
            continue

        glyph_img = script_pipeline.render_glyph(font_path, glyph)
        if glyph_img is None:
            continue

        gif_frames.append(glyph_img)
        results.append({
            "font": Path(font_path).stem
        })

    if gif_frames:
        output_gif = Path("similarity_research/viz/vit_visualization.gif")
        gif_frames[0].save(output_gif, save_all=True, append_images=gif_frames[1:], duration=500, loop=0)

    return pd.DataFrame(results)



def main():
    results = vit_photo_visualization(Path("C:/Users/User/Desktop/TheScriptGap/similarity_research/diversity_research/fonts"))
    print("Fonts Rendered in GIF:")
    print(results)

if __name__ == "__main__":
    main()
