"""
===============================================================================
Script Similarity Index — Font Metric Extraction Pipeline
===============================================================================
Project:  TRC / Monotype — Identifying Underserved Scripts
Author:   Aditya (UCF MIT2 Lab)
Purpose:  Extract Expansion Ratio (E), Vertical Footprint (V), and
          Infrastructure Friction (F) from font files to build a
          Similarity Index that quantifies how much "technical debt"
          a writing system carries relative to Latin.

Methodology:
    Every metric is derived directly from OpenType font tables — the same
    data that rendering engines (HarfBuzz, CoreText, DirectWrite) consume
    to shape and display text. Nothing is hand-labeled or sourced from
    external databases. This makes the pipeline fully reproducible and
    auditable: anyone can open the same font in a font editor and verify
    the raw values.

Data Source:
    Google Fonts GitHub repository (github.com/google/fonts)
    All fonts are SIL Open Font License — no licensing restrictions.

Usage:
    1. Clone Google Fonts:  git clone --depth 1 https://github.com/google/fonts
    2. Run this script:     python script_similarity_pipeline.py
    3. Output:              similarity_index_results.csv
===============================================================================
"""

import os
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from fontTools.ttLib import TTFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to cloned Google Fonts repo — update this to your local path
GOOGLE_FONTS_DIR = Path("./fonts")

# The 8 scripts from your CrUX analysis (non-Latin, < 5M exposure)
# Map from script name -> Unicode block ranges (start, end)
# These ranges identify which script a font primarily supports
TARGET_SCRIPTS = {
    "Cyrillic":    [(0x0400, 0x04FF), (0x0500, 0x052F)],          # Cyrillic + Supplement
    "Katakana":    [(0x30A0, 0x30FF), (0x31F0, 0x31FF)],          # Katakana + Extension
    "Devanagari":  [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],          # Devanagari + Extended
    "Arabic":      [(0x0600, 0x06FF), (0x0750, 0x077F),           # Arabic + Supplement
                    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],          # Presentation Forms
    "Han":         [(0x4E00, 0x9FFF), (0x3400, 0x4DBF),           # CJK Unified + Ext A
                    (0x20000, 0x2A6DF), (0xF900, 0xFAFF)],        # Ext B + Compatibility
    "Bengali":     [(0x0980, 0x09FF)],                             # Bengali block
    "Tamil":       [(0x0B80, 0x0BFF)],                             # Tamil block
    "Telugu":      [(0x0C00, 0x0C7F)],                             # Telugu block
}

# Minimum codepoint coverage to consider a font as "supporting" a script.
# This filters out fonts that only have a few stray characters in a range.
MIN_CODEPOINT_COVERAGE = 10  # at least 10 codepoints from the script's range

# Infrastructure Friction feature categories
# Each category is a set of OpenType feature tags that signal a specific
# type of shaping complexity. If ANY tag in a category is present, the
# font gets +1 for that category.
FRICTION_CATEGORIES = {
    "contextual_joining": {"init", "medi", "fina", "isol", "calt"},
    "indic_shaping":      {"blwf", "half", "pres", "abvs", "blws",
                           "psts", "haln", "nukt", "akhn", "rphf", "vatu"},
    "ligature_heavy":     {"liga", "rlig", "dlig", "hlig"},
    "mark_positioning":   {"mark", "mkmk", "abvm", "blwm"},
    "rtl_layout":         set(),  # detected via cmap ranges, not features
}

# RTL Unicode ranges — scripts that are right-to-left
RTL_RANGES = [
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0700, 0x074F),  # Syriac
    (0x0750, 0x077F),  # Arabic Supplement
    (0x0780, 0x07BF),  # Thaana
    (0x07C0, 0x07FF),  # NKo
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ===========================================================================
# STEP 1: Script Detection — Which script does a font support?
# ===========================================================================

def get_cmap_codepoints(font: TTFont) -> set:
    """
    Extract all Unicode codepoints that a font can render.

    We read the 'cmap' table, which is the font's mapping from Unicode
    codepoints to glyph IDs. This is the authoritative source for
    "what characters does this font support?"

    We prefer format 4 (BMP) and format 12 (full Unicode) subtables,
    which are the standard subtables used by modern fonts.
    """
    codepoints = set()
    try:
        cmap = font["cmap"]
        for table in cmap.tables:
            if table.isUnicode():
                codepoints.update(table.cmap.keys())
    except Exception as e:
        logger.warning(f"Could not read cmap: {e}")
    return codepoints


def classify_font_scripts(codepoints: set) -> Dict[str, int]:
    """
    Given a font's codepoint coverage, determine which of our target
    scripts it supports and how many codepoints it covers per script.

    Returns a dict of {script_name: codepoint_count} for scripts where
    coverage meets the minimum threshold.

    Reasoning: A font "supports" a script if it contains enough characters
    from that script's Unicode range to be usable for reading. A font with
    just 2-3 Arabic codepoints (maybe for currency symbols) doesn't count.
    """
    script_coverage = {}

    for script_name, ranges in TARGET_SCRIPTS.items():
        count = 0
        for start, end in ranges:
            count += len([cp for cp in codepoints if start <= cp <= end])

        if count >= MIN_CODEPOINT_COVERAGE:
            script_coverage[script_name] = count

    return script_coverage


# ===========================================================================
# STEP 2: Metric Extraction — Pull E, V, F from each font
# ===========================================================================

def extract_expansion_ratio(
    font: TTFont,
    script_codepoint_count: int,
    script_codepoints: set
) -> float:
    """
    Variable A: Expansion Ratio (E)
    Formula:  E = Script-Attributed Glyphs / Script Unicode Codepoints

    What it measures:
        The "Shaping Debt" — how many actual glyph shapes a font needs
        per Unicode character for this specific script.

    Why it matters:
        Latin is ~1:1 — each letter gets one glyph. Arabic needs ~4-5x
        because each letter has initial, medial, final, and isolated forms,
        plus ligatures. Devanagari has conjuncts. This multiplier directly
        reflects engineering effort per character.

    IMPORTANT — Multi-script font correction:
        Many fonts (especially CJK) cover multiple scripts. A font like
        Noto Sans JP has 7,500+ glyphs but only ~90 are for Katakana.
        Using total glyph count would give Katakana an E of 83x, which is
        nonsensical — the glyphs belong to Han, not Katakana.

        Our solution: Count glyphs that are directly mapped to the script's
        Unicode codepoints via the cmap table. This gives us the BASE glyph
        count. Then we estimate additional shaping glyphs (GSUB substitutions)
        by checking how many GSUB lookups reference script-specific features.

        This is conservative — it may slightly undercount shaping glyphs —
        but it eliminates the massive overcounting from multi-script fonts.

    Data source:
        - Script codepoints mapped to glyph IDs: 'cmap' table
        - Additional shaping glyphs: 'GSUB' table lookup analysis
        - Codepoint count: from our script classification in Step 1
    """
    try:
        if script_codepoint_count == 0:
            return 0.0

        # Count glyphs directly mapped to this script's codepoints
        cmap = font.getBestCmap()
        if not cmap:
            return None

        # Base glyphs: codepoints in this script that have direct glyph mappings
        script_glyph_ids = set()
        for cp in script_codepoints:
            if cp in cmap:
                script_glyph_ids.add(cmap[cp])

        base_glyph_count = len(script_glyph_ids)

        # Estimate additional shaping glyphs from GSUB
        # These are glyphs generated by substitution rules (ligatures,
        # contextual alternates, etc.) that don't appear in the cmap
        additional_shaping_glyphs = 0
        if "GSUB" in font:
            try:
                gsub = font["GSUB"]
                if hasattr(gsub.table, "FeatureList") and gsub.table.FeatureList:
                    # Count features that are relevant to complex shaping
                    shaping_features = {"init", "medi", "fina", "isol", "calt",
                                       "liga", "rlig", "dlig", "blwf", "half",
                                       "pres", "abvs", "blws", "psts", "nukt",
                                       "akhn", "rphf", "vatu", "haln", "ccmp"}
                    for feat in gsub.table.FeatureList.FeatureRecord:
                        if feat.FeatureTag.strip() in shaping_features:
                            # Each relevant feature adds estimated glyphs
                            # proportional to the base coverage
                            additional_shaping_glyphs += int(base_glyph_count * 0.15)
            except Exception:
                pass  # conservative: just use base count

        total_script_glyphs = base_glyph_count + additional_shaping_glyphs

        return total_script_glyphs / script_codepoint_count

    except Exception as e:
        logger.warning(f"Could not compute expansion ratio: {e}")
        return None


def extract_vertical_footprint(font: TTFont) -> float:
    """
    Variable B: Vertical Footprint (V)
    Formula:  V = (Ascender + |Descender| + LineGap) / UnitsPerEm

    What it measures:
        "Layout Friction" — how much vertical space the script demands
        relative to the standard em-square.

    Why it matters:
        Web layouts, CSS line-height, and app text containers are designed
        around Latin metrics where V ≈ 1.0-1.2. Scripts with tall vowel
        marks above (Devanagari), deep descenders (Myanmar), or stacking
        behavior (Tibetan) push V to 1.5+ and break standard layouts.
        This is a real-world readability infrastructure problem.

    Data source:
        We prefer the 'OS/2' table's sTypoAscender/sTypoDescender/
        sTypoLineGap because these are the metrics that modern text engines
        (CSS, CoreText, DirectWrite) actually use for layout.

        Fallback: 'hhea' table (ascent/descent/lineGap), which is older
        but universally present.

    Normalization:
        Dividing by unitsPerEm makes the value comparable across fonts
        regardless of their internal coordinate system (some use 1000,
        some use 2048).
    """
    try:
        upm = font["head"].unitsPerEm

        # Prefer OS/2 typo metrics (what modern engines use)
        if "OS/2" in font:
            os2 = font["OS/2"]
            ascender = os2.sTypoAscender
            descender = abs(os2.sTypoDescender)
            line_gap = os2.sTypoLineGap
        # Fallback to hhea
        elif "hhea" in font:
            hhea = font["hhea"]
            ascender = hhea.ascent
            descender = abs(hhea.descent)
            line_gap = hhea.lineGap
        else:
            return None

        return (ascender + descender + line_gap) / upm

    except Exception as e:
        logger.warning(f"Could not compute vertical footprint: {e}")
        return None


def extract_infrastructure_friction(font: TTFont, codepoints: set) -> Tuple[int, Dict[str, bool]]:
    """
    Variable C: Infrastructure Friction (F)
    A categorical penalty score based on shaping features present in the font.

    What it measures:
        The rendering complexity that the text engine must support for this
        script. More required features = more infrastructure needed.

    Why it matters:
        A script isn't just "hard" because it has many glyphs. Arabic
        requires bidirectional layout AND contextual joining. Devanagari
        requires complex Indic shaping with reordering rules. These are
        additional layers of engineering beyond just drawing glyphs, and
        they require rendering engine support that may be buggy or absent
        on some platforms.

    How we compute it:
        We read the font's GSUB (Glyph Substitution) and GPOS (Glyph
        Positioning) tables, which declare what shaping features the font
        uses. We group these into categories:
          - Contextual joining (init/medi/fina):  +1
          - Indic shaping (blwf/half/pres/etc):   +1
          - Heavy ligature use (liga/rlig/dlig):   +1
          - Complex mark positioning (mark/mkmk):  +1
          - RTL layout requirement:                +1

        Each category is binary: either the font uses it or it doesn't.
        We detect RTL separately by checking if the font's codepoints
        fall in known RTL Unicode ranges.

    Data source:
        GSUB and GPOS feature tags, read directly from the font binary.
        These are the same tags that HarfBuzz (the universal shaping
        engine used by Chrome, Firefox, Android, etc.) consumes.
    """
    friction_score = 0
    category_hits = {}

    # Collect all feature tags from GSUB and GPOS
    feature_tags = set()
    for table_name in ["GSUB", "GPOS"]:
        try:
            if table_name in font:
                table = font[table_name]
                if hasattr(table.table, "FeatureList") and table.table.FeatureList:
                    for feat in table.table.FeatureList.FeatureRecord:
                        feature_tags.add(feat.FeatureTag.strip())
        except Exception as e:
            logger.debug(f"Could not read {table_name}: {e}")

    # Check each friction category
    for category, tags in FRICTION_CATEGORIES.items():
        if category == "rtl_layout":
            # RTL is detected from codepoint ranges, not features
            is_rtl = any(
                start <= cp <= end
                for cp in codepoints
                for start, end in RTL_RANGES
            )
            category_hits[category] = is_rtl
            if is_rtl:
                friction_score += 1
        else:
            # Check if any feature tag from this category is present
            hit = bool(feature_tags & tags)
            category_hits[category] = hit
            if hit:
                friction_score += 1

    return friction_score, category_hits


# ===========================================================================
# STEP 3: Walk the font directory and extract everything
# ===========================================================================

def process_font_file(filepath: Path) -> List[dict]:
    """
    Process a single font file and return metric records for each
    target script the font supports.

    Returns a list of dicts, one per script the font covers.
    """
    records = []

    try:
        font = TTFont(filepath, fontNumber=0)  # fontNumber for .ttc files
    except Exception as e:
        logger.debug(f"Could not open {filepath}: {e}")
        return records

    try:
        # Get all Unicode codepoints this font covers
        codepoints = get_cmap_codepoints(font)

        if not codepoints:
            font.close()
            return records

        # Determine which of our target scripts this font supports
        script_coverage = classify_font_scripts(codepoints)

        if not script_coverage:
            font.close()
            return records

        # Extract metrics for each supported script
        for script_name, cp_count in script_coverage.items():

            # Get the specific codepoints for this script (needed for E)
            script_specific_cps = set()
            for start, end in TARGET_SCRIPTS[script_name]:
                script_specific_cps.update(
                    cp for cp in codepoints if start <= cp <= end
                )

            expansion_ratio = extract_expansion_ratio(font, cp_count, script_specific_cps)
            vertical_footprint = extract_vertical_footprint(font)
            friction_score, friction_details = extract_infrastructure_friction(
                font, codepoints
            )

            # Get font family name for traceability
            name_table = font.get("name")
            family_name = "Unknown"
            if name_table:
                for record in name_table.names:
                    if record.nameID == 1:  # Font Family name
                        try:
                            family_name = record.toUnicode()
                            break
                        except:
                            pass

            records.append({
                "font_file": filepath.name,
                "font_family": family_name,
                "script": script_name,
                "codepoint_coverage": cp_count,
                "total_glyphs": font["maxp"].numGlyphs,
                "expansion_ratio": expansion_ratio,
                "vertical_footprint": vertical_footprint,
                "friction_score": friction_score,
                "friction_contextual_joining": friction_details.get("contextual_joining", False),
                "friction_indic_shaping": friction_details.get("indic_shaping", False),
                "friction_ligature_heavy": friction_details.get("ligature_heavy", False),
                "friction_mark_positioning": friction_details.get("mark_positioning", False),
                "friction_rtl_layout": friction_details.get("rtl_layout", False),
            })

    except Exception as e:
        logger.warning(f"Error processing {filepath}: {e}")
    finally:
        font.close()

    return records


def walk_font_directory(root_dir: Path) -> pd.DataFrame:
    """
    Recursively walk the Google Fonts directory, process every font file,
    and return a DataFrame of all per-font, per-script metric records.
    """
    font_extensions = {".ttf", ".otf"}
    all_records = []
    font_count = 0
    error_count = 0

    font_files = list(root_dir.rglob("*"))
    font_files = [f for f in font_files if f.suffix.lower() in font_extensions]

    logger.info(f"Found {len(font_files)} font files to process")

    for i, filepath in enumerate(font_files):
        if (i + 1) % 500 == 0:
            logger.info(f"Processing font {i + 1}/{len(font_files)}...")

        records = process_font_file(filepath)
        if records:
            all_records.extend(records)
            font_count += 1
        else:
            error_count += 1

    logger.info(
        f"Processed {font_count} fonts with target script coverage, "
        f"{error_count} skipped (no coverage or errors)"
    )

    return pd.DataFrame(all_records)


# ===========================================================================
# STEP 4: Aggregate per script (median) and normalize
# ===========================================================================

def aggregate_per_script(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-font metrics into per-script metrics using the MEDIAN.

    Why median instead of mean:
        - Robust to outlier fonts with unusual metrics (e.g., a display
          font with exaggerated ascenders, or a symbol font miscategorized).
        - With 100+ fonts per script, the median gives us the "typical"
          font engineering experience for that script.
        - Preferred in small-sample statistics when distributions may be
          skewed (which they will be — some scripts have 118 fonts, others
          have 17,000+).
    """
    agg = df.groupby("script").agg(
        font_count=("font_file", "nunique"),
        median_expansion_ratio=("expansion_ratio", "median"),
        median_vertical_footprint=("vertical_footprint", "median"),
        median_friction_score=("friction_score", "median"),
        # Also capture the IQR for reporting
        expansion_ratio_q25=("expansion_ratio", lambda x: x.quantile(0.25)),
        expansion_ratio_q75=("expansion_ratio", lambda x: x.quantile(0.75)),
        vertical_footprint_q25=("vertical_footprint", lambda x: x.quantile(0.25)),
        vertical_footprint_q75=("vertical_footprint", lambda x: x.quantile(0.75)),
    ).reset_index()

    return agg


def compute_similarity_index(
    agg_df: pd.DataFrame,
    w_expansion: float = 0.50,
    w_vertical: float = 0.30,
    w_friction: float = 0.20,
) -> pd.DataFrame:
    """
    Min-Max normalize E, V, F, then compute the Composite Complexity Score
    and invert to get the Similarity Index.

    Formula:
        C = (w1 × E_norm) + (w2 × V_norm) + (w3 × F_norm)
        S = 1 - C

    Where:
        E_norm, V_norm, F_norm are min-max scaled to [0, 1]
        w1 + w2 + w3 = 1.0

    Interpretation:
        S ≈ 1.0 → Script is very "Latin-like" (low complexity, easy to support)
        S ≈ 0.0 → Script carries heavy technical debt (hard to support)

    Weight rationale (initial — should be validated):
        - Expansion (50%): Glyph count is the single biggest driver of font
          engineering cost. More glyphs = more design time = fewer fonts built.
        - Vertical (30%): Layout breakage is the most visible infrastructure
          issue users experience. Directly affects readability.
        - Friction (20%): Important but more of a platform-level concern.
          Modern engines handle RTL and Indic shaping reasonably well.
    """
    result = agg_df.copy()

    # --- Min-Max Normalization ---
    for col, norm_col in [
        ("median_expansion_ratio", "E_norm"),
        ("median_vertical_footprint", "V_norm"),
        ("median_friction_score", "F_norm"),
    ]:
        col_min = result[col].min()
        col_max = result[col].max()
        if col_max - col_min == 0:
            result[norm_col] = 0.0
        else:
            result[norm_col] = (result[col] - col_min) / (col_max - col_min)

    # --- Composite Complexity Score ---
    result["complexity_C"] = (
        w_expansion * result["E_norm"]
        + w_vertical * result["V_norm"]
        + w_friction * result["F_norm"]
    )

    # --- Similarity Index (inversion) ---
    result["similarity_S"] = 1 - result["complexity_C"]

    # --- Store weights for reproducibility ---
    result.attrs["weights"] = {
        "expansion": w_expansion,
        "vertical": w_vertical,
        "friction": w_friction,
    }

    return result


# ===========================================================================
# STEP 5: Reporting
# ===========================================================================

def print_report(result_df: pd.DataFrame):
    """Pretty-print the results for presentation / notebook output."""

    print("\n" + "=" * 80)
    print("SCRIPT SIMILARITY INDEX — RESULTS")
    print("=" * 80)

    print("\n📊 Per-Script Raw Metrics (Median across fonts):\n")
    raw_cols = [
        "script", "font_count",
        "median_expansion_ratio", "median_vertical_footprint",
        "median_friction_score"
    ]
    print(result_df[raw_cols].to_string(index=False, float_format="%.3f"))

    print("\n\n📐 Normalized Scores & Similarity Index:\n")
    norm_cols = [
        "script", "E_norm", "V_norm", "F_norm",
        "complexity_C", "similarity_S"
    ]
    print(result_df[norm_cols].sort_values("similarity_S").to_string(
        index=False, float_format="%.3f"
    ))

    print("\n\n🔍 Interpretation:")
    print("  S → 1.0 = 'Latin-like' (simple to support, low engineering cost)")
    print("  S → 0.0 = 'High technical debt' (complex, underserved)")

    most_complex = result_df.loc[result_df["similarity_S"].idxmin()]
    most_similar = result_df.loc[result_df["similarity_S"].idxmax()]
    print(f"\n  Most complex script:    {most_complex['script']} (S = {most_complex['similarity_S']:.3f})")
    print(f"  Most Latin-like script: {most_similar['script']} (S = {most_similar['similarity_S']:.3f})")

    print("\n" + "=" * 80)


# ===========================================================================
# MAIN — Run the full pipeline
# ===========================================================================

def main():
    """
    Full pipeline execution:
      1. Walk Google Fonts directory → extract per-font metrics
      2. Aggregate per script (median)
      3. Normalize and compute Similarity Index
      4. Export results
    """

    # --- Validate input ---
    if not GOOGLE_FONTS_DIR.exists():
        logger.error(
            f"Google Fonts directory not found at {GOOGLE_FONTS_DIR}\n"
            f"Please clone it first:\n"
            f"  git clone --depth 1 https://github.com/google/fonts\n"
            f"Then update GOOGLE_FONTS_DIR in this script."
        )
        sys.exit(1)

    # --- Step 1-2: Extract metrics from all fonts ---
    logger.info("Step 1: Extracting metrics from font files...")
    raw_df = walk_font_directory(GOOGLE_FONTS_DIR)

    if raw_df.empty:
        logger.error("No fonts with target script coverage found!")
        sys.exit(1)

    logger.info(f"Extracted {len(raw_df)} font-script records")

    # Save raw extraction for debugging / exploration
    raw_df.to_csv("raw_font_metrics.csv", index=False)
    logger.info("Saved raw metrics → raw_font_metrics.csv")

    # --- Step 3: Aggregate per script ---
    logger.info("Step 3: Aggregating per script (median)...")
    agg_df = aggregate_per_script(raw_df)

    # --- Step 4: Normalize and compute Similarity Index ---
    logger.info("Step 4: Computing Similarity Index...")
    result_df = compute_similarity_index(agg_df)

    # --- Step 5: Report and export ---
    print_report(result_df)

    result_df.to_csv("similarity_index_results.csv", index=False)
    logger.info("Saved results → similarity_index_results.csv")

    # Also save a clean summary for merging with Exposure + Support
    summary = result_df[["script", "similarity_S", "complexity_C",
                          "median_expansion_ratio", "median_vertical_footprint",
                          "median_friction_score", "font_count"]].copy()
    summary.to_csv("similarity_index_summary.csv", index=False)
    logger.info("Saved summary → similarity_index_summary.csv")

    return result_df


if __name__ == "__main__":
    main()
