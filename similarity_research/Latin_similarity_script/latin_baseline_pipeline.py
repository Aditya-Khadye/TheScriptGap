"""
===============================================================================
Latin Baseline - Font Metric Extraction Pipeline
===============================================================================
Project:  TRC / Monotype - Identifying Underserved Scripts
Author:   Aditya (UCF MIT2 Lab)
Purpose:  Establish the LATIN reference baseline for the Script Similarity
          Index. The non-Latin pipeline measures how much "technical debt"
          each writing system carries RELATIVE TO LATIN, but Latin itself was
          never measured. This script extracts the same three metrics,
          Expansion Ratio (E), Vertical Footprint (V), and Infrastructure
          Friction (F), from Latin fonts so the baseline vector
          (E_latin, V_latin, F_latin) is empirical instead of assumed.

Key differences from the non-Latin pipeline (both deliberate):

    1. LATIN-DOMINANT FILTERING.
       V and F are computed at the FONT level, not per-script. Latin appears
       in thousands of multi-script fonts (CJK fonts, Arabic fonts with a
       Latin fallback, etc.). A naive run would attach those scripts'
       RTL/joining friction and atypical vertical metrics to the "Latin"
       record and inflate the baseline. So a font only counts toward the
       Latin baseline if it is Latin-dominant: Latin plus simple LTR scripts
       (Greek, Cyrillic) is fine, but any real coverage of RTL / Indic-shaping
       / CJK scripts disqualifies it. Toggle with LATIN_DOMINANT_ONLY.
       Note: E is already protected from this by the glyph-attribution logic
       inherited from the original pipeline, so the filter mainly cleans V/F.

    2. NO MIN-MAX / SIMILARITY INVERSION.
       Min-max normalization across a single script is degenerate (min == max,
       everything collapses to 0 or 1) and Latin's similarity to itself is
       trivially 1.0. Instead this script reports proper baseline statistics:
       median, mean, std, IQR, p10/p90 for E and V, plus per-category friction
       prevalence so you can see WHY the median friction score lands where it
       does.

Data Source:
    Google Fonts GitHub repository (github.com/google/fonts)
    All fonts are SIL Open Font License, no licensing restrictions.

Usage:
    1. Clone Google Fonts:  git clone --depth 1 https://github.com/google/fonts
    2. Run this script:     python latin_baseline_pipeline.py
    3. Output:              latin_baseline_summary.csv (+ raw + json)

Runtime note:
    The non-Latin run filters to ~8 scripts. This one touches essentially
    every Latin-supporting font in the repo (the large majority of it), so a
    full pass is heavier. Set SAMPLE_SIZE for a quick reproducible sanity run
    before committing to the full extraction.
===============================================================================
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import pandas as pd
from fontTools.ttLib import TTFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to cloned Google Fonts repo, update this to your local path
GOOGLE_FONTS_DIR = Path("./fonts")

# The Latin script, defined as its letter-bearing Unicode ranges.
# Kept in the same dict structure as the non-Latin pipeline (TARGET_SCRIPTS)
# so the downstream functions stay byte-for-byte parallel and easy to diff.
#
# Methodology note: the non-Latin pipeline counts ENTIRE Unicode blocks per
# script. "Basic Latin" the block (0x00-0x7F) is mostly non-letters (control
# codes, digits, punctuation), so counting the whole block would mean
# something different for Latin than it does for, say, Devanagari. We count
# Latin LETTERS only. The effect on E is small either way (digits/punctuation
# map ~1:1) but this keeps the codepoint denominator honest.
TARGET_SCRIPTS = {
    "Latin": [
        (0x0041, 0x005A),   # Basic Latin: A-Z
        (0x0061, 0x007A),   # Basic Latin: a-z
        (0x00C0, 0x00FF),   # Latin-1 Supplement (accented letters; incl. the
                            #   two stray math signs x and div, negligible)
        (0x0100, 0x017F),   # Latin Extended-A
        (0x0180, 0x024F),   # Latin Extended-B
        (0x1E00, 0x1EFF),   # Latin Extended Additional
        # Optional specialist Latin letters (phonetic / historic / minority
        # orthographies). Off by default. Uncommenting barely moves the
        # aggregate baseline; turn on only to characterize the specialist tail.
         (0x0250, 0x02AF), # IPA Extensions (phonetic alphabet: schwa, esh...)
         (0x2C60, 0x2C7F), # Latin Extended-C (minority/orthographic letters)
         (0xA720, 0xA7FF), # Latin Extended-D (medievalist + African orthographies)
         (0xAB30, 0xAB6F), # Latin Extended-E (German dialectology / Teuthonista)
    ],
}

# Minimum codepoint coverage to consider a font as "supporting" Latin.
# A-Z alone is 26, so any real Latin font clears this easily. Kept at 10 for
# consistency with the non-Latin pipeline. Raise to 26 if you want to require
# the full basic alphabet.
MIN_CODEPOINT_COVERAGE = 26

# If True, restrict the baseline to Latin-DOMINANT fonts (see header note 1).
# If False, include any font with >= MIN_CODEPOINT_COVERAGE Latin letters
# (the naive population, useful only for comparison).
LATIN_DOMINANT_ONLY = True

# Scripts whose presence in a font would contaminate the Latin V/F baseline.
# A font is excluded from the Latin-dominant baseline if it has
# >= MIN_CODEPOINT_COVERAGE coverage of ANY of these. Greek and Cyrillic are
# intentionally NOT listed: they are simple LTR alphabets with Latin-like
# metrics and friction, so pan-European Latin+Greek+Cyrillic fonts are kept.
CONTAMINATING_RANGES = {
    # RTL scripts (add bidi + contextual-joining friction)
    "Arabic":     [(0x0600, 0x06FF), (0x0750, 0x077F),
                   (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Hebrew":     [(0x0590, 0x05FF)],
    "Syriac":     [(0x0700, 0x074F)],
    "Thaana":     [(0x0780, 0x07BF)],
    "NKo":        [(0x07C0, 0x07FF)],
    # Indic / Brahmic-shaping scripts (add reordering + conjunct friction)
    "Devanagari": [(0x0900, 0x097F)],
    "Bengali":    [(0x0980, 0x09FF)],
    "Gurmukhi":   [(0x0A00, 0x0A7F)],
    "Gujarati":   [(0x0A80, 0x0AFF)],
    "Oriya":      [(0x0B00, 0x0B7F)],
    "Tamil":      [(0x0B80, 0x0BFF)],
    "Telugu":     [(0x0C00, 0x0C7F)],
    "Kannada":    [(0x0C80, 0x0CFF)],
    "Malayalam":  [(0x0D00, 0x0D7F)],
    "Sinhala":    [(0x0D80, 0x0DFF)],
    "Thai":       [(0x0E00, 0x0E7F)],
    "Lao":        [(0x0E80, 0x0EFF)],
    "Tibetan":    [(0x0F00, 0x0FFF)],
    "Myanmar":    [(0x1000, 0x109F)],
    "Khmer":      [(0x1780, 0x17FF)],
    # CJK (large mark/alternate friction + atypical vertical metrics)
    "Han":        [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "Hiragana":   [(0x3040, 0x309F)],
    "Katakana":   [(0x30A0, 0x30FF)],
    "Hangul":     [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
}

# Quick-run controls. SAMPLE_SIZE samples raw font FILES (pre-filter) with a
# fixed seed for reproducibility. Set to None for the full repo.
SAMPLE_SIZE: Optional[int] = None
RANDOM_SEED = 42

# Parallelism. None = auto (min(cpu_count, 8), which targets the M2 Max's
# performance cores). Set to 1 to force the serial path for easier debugging.
MAX_WORKERS: Optional[int] = None

# Infrastructure Friction feature categories (identical to the non-Latin
# pipeline so the F values are on the same scale).
#
# Latin expectations worth keeping in mind when you read the prevalence:
#   - contextual_joining: Latin does NOT join like Arabic, but many Latin
#     fonts declare `calt` for stylistic contextual alternates, which trips
#     this category. That is expected and is left in for apples-to-apples
#     comparability, not a bug.
#   - ligature_heavy: Latin fonts routinely ship liga/dlig (fi, fl, ...).
#   - mark_positioning: Latin fonts with diacritics use mark/mkmk.
#   - indic_shaping / rtl_layout: should be ~0% for a clean Latin population.
FRICTION_CATEGORIES = {
    "contextual_joining": {"init", "medi", "fina", "isol", "calt"},
    "indic_shaping":      {"blwf", "half", "pres", "abvs", "blws",
                           "psts", "haln", "nukt", "akhn", "rphf", "vatu"},
    "ligature_heavy":     {"liga", "rlig", "dlig", "hlig"},
    "mark_positioning":   {"mark", "mkmk", "abvm", "blwm"},
    "rtl_layout":         set(),  # detected via cmap ranges, not features
}

# RTL Unicode ranges, scripts that are right-to-left
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
# STEP 1: Script Detection
# ===========================================================================

def get_cmap_codepoints(font: TTFont) -> set:
    """
    Extract all Unicode codepoints a font can render, read from the 'cmap'
    table (the authoritative Unicode-to-glyph map). Identical to the
    non-Latin pipeline.
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
    Determine which target scripts (here: just Latin) a font supports and how
    many codepoints it covers. Returns {script: count} for scripts meeting the
    minimum threshold. Identical logic to the non-Latin pipeline.
    """
    script_coverage = {}
    for script_name, ranges in TARGET_SCRIPTS.items():
        count = 0
        for start, end in ranges:
            count += len([cp for cp in codepoints if start <= cp <= end])
        if count >= MIN_CODEPOINT_COVERAGE:
            script_coverage[script_name] = count
    return script_coverage


def find_contaminating_script(codepoints: set) -> Optional[str]:
    """
    Return the name of the first contaminating (RTL / Indic / CJK) script with
    >= MIN_CODEPOINT_COVERAGE coverage, else None.

    Used to decide whether a font is Latin-dominant. Short-circuits on the
    first hit, which also keeps this cheap on huge CJK codepoint sets.
    """
    for script_name, ranges in CONTAMINATING_RANGES.items():
        count = 0
        for start, end in ranges:
            for cp in codepoints:
                if start <= cp <= end:
                    count += 1
                    if count >= MIN_CODEPOINT_COVERAGE:
                        return script_name
    return None


# ===========================================================================
# STEP 2: Metric Extraction (E, V, F)
# ===========================================================================

def extract_expansion_ratio(
    font: TTFont,
    script_codepoint_count: int,
    script_codepoints: set
) -> Optional[float]:
    """
    Variable A: Expansion Ratio (E)
    Formula:  E = Script-Attributed Glyphs / Script Unicode Codepoints

    What it measures:
        The "Shaping Debt", how many actual glyph shapes a font needs per
        Unicode character for this script.

    For Latin specifically:
        Latin is close to 1:1, each letter gets one glyph. The value lands
        a bit above 1.0 because the additional-shaping estimate below adds a
        small amount for declared features (ligatures, contextual alternates,
        composition). Typical Latin E is roughly 1.0 to 1.7. That low
        multiplier IS the baseline that Arabic (~4-5x) and the Indic scripts
        (conjuncts) are heavy relative to.

    Multi-script font correction (inherited, and the reason E is safe even
    without the Latin-dominant filter):
        We count only glyphs DIRECTLY mapped to Latin's codepoints via cmap,
        so a Noto Sans CJK font contributes its ~100 Latin glyphs, not its
        thousands of Han glyphs. Then we add a conservative estimate of GSUB
        shaping glyphs.

    Data source:
        - Base glyphs: 'cmap' table
        - Additional shaping glyphs: 'GSUB' feature analysis (crude 0.15x per
          relevant feature, same heuristic and same caveats as the original)
        - Codepoint count: from Step 1 classification
    """
    try:
        if script_codepoint_count == 0:
            return 0.0

        cmap = font.getBestCmap()
        if not cmap:
            return None

        # Base glyphs: Latin codepoints that have a direct glyph mapping
        script_glyph_ids = set()
        for cp in script_codepoints:
            if cp in cmap:
                script_glyph_ids.add(cmap[cp])

        base_glyph_count = len(script_glyph_ids)

        # Estimate additional shaping glyphs declared in GSUB (ligatures,
        # contextual alternates, composition, etc.)
        additional_shaping_glyphs = 0
        if "GSUB" in font:
            try:
                gsub = font["GSUB"]
                if hasattr(gsub.table, "FeatureList") and gsub.table.FeatureList:
                    shaping_features = {"init", "medi", "fina", "isol", "calt",
                                        "liga", "rlig", "dlig", "blwf", "half",
                                        "pres", "abvs", "blws", "psts", "nukt",
                                        "akhn", "rphf", "vatu", "haln", "ccmp"}
                    for feat in gsub.table.FeatureList.FeatureRecord:
                        if feat.FeatureTag.strip() in shaping_features:
                            additional_shaping_glyphs += int(base_glyph_count * 0.15)
            except Exception:
                pass  # conservative: fall back to base count

        total_script_glyphs = base_glyph_count + additional_shaping_glyphs
        return total_script_glyphs / script_codepoint_count

    except Exception as e:
        logger.warning(f"Could not compute expansion ratio: {e}")
        return None


def extract_vertical_footprint(font: TTFont) -> Optional[float]:
    """
    Variable B: Vertical Footprint (V)
    Formula:  V = (Ascender + |Descender| + LineGap) / UnitsPerEm

    What it measures:
        "Layout Friction", how much vertical space the font demands relative
        to the em-square. Web layouts and CSS line-height are tuned around
        Latin where V is roughly 1.0 to 1.2, which is exactly what this
        baseline should confirm.

    Font-global metric, identical extraction to the non-Latin pipeline.
    Prefers OS/2 typo metrics (what CSS/CoreText/DirectWrite use), falls back
    to hhea, normalized by unitsPerEm for cross-font comparability.
    """
    try:
        upm = font["head"].unitsPerEm
        if "OS/2" in font:
            os2 = font["OS/2"]
            ascender = os2.sTypoAscender
            descender = abs(os2.sTypoDescender)
            line_gap = os2.sTypoLineGap
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
    Categorical penalty (0-5) from shaping features present in the font.
    Identical logic to the non-Latin pipeline (so values are comparable).

    Reads GSUB/GPOS feature tags and buckets them into:
        contextual_joining, indic_shaping, ligature_heavy, mark_positioning,
        rtl_layout (the last detected from cmap ranges).

    For a clean Latin-dominant population you should expect indic_shaping and
    rtl_layout near 0%, with the score driven by ligatures, mark positioning
    for diacritics, and calt.
    """
    friction_score = 0
    category_hits = {}

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

    for category, tags in FRICTION_CATEGORIES.items():
        if category == "rtl_layout":
            is_rtl = any(
                start <= cp <= end
                for cp in codepoints
                for start, end in RTL_RANGES
            )
            category_hits[category] = is_rtl
            if is_rtl:
                friction_score += 1
        else:
            hit = bool(feature_tags & tags)
            category_hits[category] = hit
            if hit:
                friction_score += 1

    return friction_score, category_hits


# ===========================================================================
# STEP 3: Walk the font directory
# ===========================================================================

def process_font_file(filepath: Path) -> Tuple[List[dict], Optional[str]]:
    """
    Process a single font file.

    Returns (records, skip_reason). When records are produced, skip_reason is
    None. When the font is skipped, records is empty and skip_reason explains
    why ("no_codepoints", "no_latin", or "contaminated:<script>"). The skip
    reasons feed the diagnostic summary in walk_font_directory.
    """
    records: List[dict] = []

    try:
        font = TTFont(filepath, fontNumber=0)  # fontNumber handles .ttc
    except Exception as e:
        logger.debug(f"Could not open {filepath}: {e}")
        return records, "open_error"

    try:
        codepoints = get_cmap_codepoints(font)
        if not codepoints:
            return records, "no_codepoints"

        script_coverage = classify_font_scripts(codepoints)
        if "Latin" not in script_coverage:
            return records, "no_latin"

        # Latin-dominant gate (cleans V and F, see header note 1)
        if LATIN_DOMINANT_ONLY:
            contaminant = find_contaminating_script(codepoints)
            if contaminant is not None:
                return records, f"contaminated:{contaminant}"

        cp_count = script_coverage["Latin"]

        # Latin-specific codepoints actually present (needed for E)
        script_specific_cps = set()
        for start, end in TARGET_SCRIPTS["Latin"]:
            script_specific_cps.update(
                cp for cp in codepoints if start <= cp <= end
            )

        expansion_ratio = extract_expansion_ratio(font, cp_count, script_specific_cps)
        vertical_footprint = extract_vertical_footprint(font)
        friction_score, friction_details = extract_infrastructure_friction(font, codepoints)

        # Font family name for traceability
        name_table = font.get("name")
        family_name = "Unknown"
        if name_table:
            for record in name_table.names:
                if record.nameID == 1:  # Font Family name
                    try:
                        family_name = record.toUnicode()
                        break
                    except Exception:
                        pass

        records.append({
            "font_file": filepath.name,
            "font_family": family_name,
            "script": "Latin",
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
        return records, None

    except Exception as e:
        logger.warning(f"Error processing {filepath}: {e}")
        return records, "process_error"
    finally:
        font.close()


def walk_font_directory(root_dir: Path) -> pd.DataFrame:
    """
    Recursively walk the Google Fonts directory, process every font file in
    parallel across CPU cores, and return a DataFrame of per-font Latin metric
    records. Honors SAMPLE_SIZE (reproducible random subset of raw files) and
    MAX_WORKERS, and reports skip diagnostics.

    Parallelism: every font is parsed independently, so the work fans out over
    a process pool. Processes, not threads, because the parsing is CPU-bound
    pure Python and the GIL would serialize threads. On macOS the default
    'spawn' start method re-imports this module in each worker, which is why
    process_font_file and all config live at module level and main() is guarded
    by `if __name__ == "__main__"`. Set MAX_WORKERS=1 to force the serial path.
    """
    font_extensions = {".ttf", ".otf"}
    all_records: List[dict] = []
    kept = 0
    skip_reasons: Counter = Counter()

    font_files = [f for f in root_dir.rglob("*") if f.suffix.lower() in font_extensions]
    logger.info(f"Found {len(font_files)} font files")

    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(font_files):
        rng = random.Random(RANDOM_SEED)
        font_files = font_files[:]  # copy before shuffling
        rng.shuffle(font_files)
        font_files = font_files[:SAMPLE_SIZE]
        logger.info(f"SAMPLE_SIZE set: processing a random {SAMPLE_SIZE}-file subset (seed={RANDOM_SEED})")

    total = len(font_files)
    workers = MAX_WORKERS if MAX_WORKERS is not None else min(os.cpu_count() or 4, 8)
    workers = max(1, workers)

    def _tally(records, skip_reason):
        nonlocal kept
        if records:
            all_records.extend(records)
            kept += 1
        else:
            skip_reasons[skip_reason or "unknown"] += 1
            if skip_reason and skip_reason.startswith("contaminated:"):
                skip_reasons["contaminated_total"] += 1

    if workers == 1:
        # Serial path: slower, but tracebacks are readable when debugging
        logger.info("Processing serially (workers=1)")
        for i, filepath in enumerate(font_files):
            if (i + 1) % 500 == 0:
                logger.info(f"Processing font {i + 1}/{total}...")
            records, skip_reason = process_font_file(filepath)
            _tally(records, skip_reason)
    else:
        logger.info(f"Processing in parallel across {workers} worker processes")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_font_file, fp) for fp in font_files]
            for i, fut in enumerate(as_completed(futures)):
                if (i + 1) % 500 == 0:
                    logger.info(f"Processed {i + 1}/{total}...")
                try:
                    records, skip_reason = fut.result()
                except Exception as e:
                    logger.warning(f"Worker failed on a font: {e}")
                    skip_reasons["worker_exception"] += 1
                    continue
                _tally(records, skip_reason)

    logger.info(f"Kept {kept} Latin-dominant fonts" if LATIN_DOMINANT_ONLY
                else f"Kept {kept} Latin-supporting fonts")
    if skip_reasons:
        excluded_ms = skip_reasons.get("contaminated_total", 0)
        if LATIN_DOMINANT_ONLY and excluded_ms:
            logger.info(f"Excluded {excluded_ms} multi-script fonts (RTL/Indic/CJK coverage)")
        no_latin = skip_reasons.get("no_latin", 0) + skip_reasons.get("no_codepoints", 0)
        logger.info(f"Skipped {no_latin} fonts with no/low Latin coverage")

    df = pd.DataFrame(all_records)
    df.attrs["skip_reasons"] = dict(skip_reasons)
    df.attrs["kept"] = kept
    return df


# ===========================================================================
# STEP 4: Compute the Latin baseline
# ===========================================================================

def _safe_stats(series: pd.Series) -> Dict[str, Optional[float]]:
    """Median/mean/std/IQR/p10/p90 with NaNs dropped. None-safe on empty."""
    s = series.dropna()
    if s.empty:
        return {k: None for k in ["median", "mean", "std", "q25", "q75", "p10", "p90", "n"]}
    return {
        "median": round(float(s.median()), 4),
        "mean":   round(float(s.mean()), 4),
        "std":    round(float(s.std()), 4),
        "q25":    round(float(s.quantile(0.25)), 4),
        "q75":    round(float(s.quantile(0.75)), 4),
        "p10":    round(float(s.quantile(0.10)), 4),
        "p90":    round(float(s.quantile(0.90)), 4),
        "n":      int(s.shape[0]),
    }


def compute_latin_baseline(df: pd.DataFrame) -> dict:
    """
    Collapse per-font Latin metrics into the baseline reference.

    Uses the MEDIAN as the headline statistic (robust to display fonts with
    exaggerated metrics, same rationale as the non-Latin pipeline) and reports
    spread alongside it. Friction is summarized two ways: the median score AND
    the per-category prevalence (what fraction of Latin fonts trip each
    category), which is what actually explains the median.
    """
    friction_cols = [
        "friction_contextual_joining",
        "friction_indic_shaping",
        "friction_ligature_heavy",
        "friction_mark_positioning",
        "friction_rtl_layout",
    ]

    baseline = {
        "script": "Latin",
        "font_count": int(df["font_file"].nunique()),
        "latin_dominant_only": LATIN_DOMINANT_ONLY,
        "expansion_ratio": _safe_stats(df["expansion_ratio"]),
        "vertical_footprint": _safe_stats(df["vertical_footprint"]),
        "friction_score": _safe_stats(df["friction_score"]),
        "total_glyphs_median": round(float(df["total_glyphs"].median()), 1),
        "friction_category_prevalence_pct": {
            c.replace("friction_", ""): round(float(df[c].mean()) * 100, 1)
            for c in friction_cols
        },
    }

    # The single line you plug back into the main pipeline as the anchor:
    baseline["baseline_reference_vector"] = {
        "E_latin": baseline["expansion_ratio"]["median"],
        "V_latin": baseline["vertical_footprint"]["median"],
        "F_latin": baseline["friction_score"]["median"],
    }
    return baseline


def baseline_to_flat_df(baseline: dict) -> pd.DataFrame:
    """Flatten the nested baseline dict into a one-row DataFrame for CSV."""
    e = baseline["expansion_ratio"]
    v = baseline["vertical_footprint"]
    f = baseline["friction_score"]
    p = baseline["friction_category_prevalence_pct"]
    row = {
        "script": "Latin",
        "font_count": baseline["font_count"],
        "latin_dominant_only": baseline["latin_dominant_only"],
        "E_latin_median": e["median"], "E_mean": e["mean"], "E_std": e["std"],
        "E_q25": e["q25"], "E_q75": e["q75"], "E_p10": e["p10"], "E_p90": e["p90"],
        "V_latin_median": v["median"], "V_mean": v["mean"], "V_std": v["std"],
        "V_q25": v["q25"], "V_q75": v["q75"], "V_p10": v["p10"], "V_p90": v["p90"],
        "F_latin_median": f["median"], "F_mean": f["mean"],
        "total_glyphs_median": baseline["total_glyphs_median"],
        "prev_contextual_joining_pct": p["contextual_joining"],
        "prev_indic_shaping_pct": p["indic_shaping"],
        "prev_ligature_heavy_pct": p["ligature_heavy"],
        "prev_mark_positioning_pct": p["mark_positioning"],
        "prev_rtl_layout_pct": p["rtl_layout"],
    }
    return pd.DataFrame([row])


# ===========================================================================
# STEP 5: Reporting
# ===========================================================================

def print_baseline_report(baseline: dict):
    """Pretty-print the Latin baseline for a notebook / writeup."""
    e = baseline["expansion_ratio"]
    v = baseline["vertical_footprint"]
    f = baseline["friction_score"]
    p = baseline["friction_category_prevalence_pct"]
    ref = baseline["baseline_reference_vector"]

    print("\n" + "=" * 80)
    print("LATIN BASELINE - RESULTS")
    print("=" * 80)
    pop = "Latin-dominant" if baseline["latin_dominant_only"] else "all Latin-supporting"
    print(f"\nPopulation: {pop} fonts")
    print(f"Font count: {baseline['font_count']}")
    print(f"Median total glyphs/font: {baseline['total_glyphs_median']:.0f}")

    print("\nExpansion Ratio (E)   glyphs per Latin codepoint")
    print(f"  median {e['median']}   mean {e['mean']}   std {e['std']}")
    print(f"  IQR [{e['q25']}, {e['q75']}]   p10/p90 [{e['p10']}, {e['p90']}]")

    print("\nVertical Footprint (V)   (asc + |desc| + linegap) / upm")
    print(f"  median {v['median']}   mean {v['mean']}   std {v['std']}")
    print(f"  IQR [{v['q25']}, {v['q75']}]   p10/p90 [{v['p10']}, {v['p90']}]")

    print("\nInfrastructure Friction (F)   0-5 categorical")
    print(f"  median {f['median']}   mean {f['mean']}")
    print("  category prevalence (% of Latin fonts):")
    for cat, pct in p.items():
        print(f"    {cat:<22} {pct:>5.1f}%")

    print("\n" + "-" * 80)
    print("BASELINE REFERENCE VECTOR (plug into the Similarity Index):")
    print(f"  E_latin = {ref['E_latin']}")
    print(f"  V_latin = {ref['V_latin']}")
    print(f"  F_latin = {ref['F_latin']}")
    print("-" * 80)
    print("\nInterpretation:")
    print("  This is the anchor every non-Latin script is scored against.")
    print("  Expect indic_shaping and rtl_layout near 0% for a clean Latin run.")
    print("  A nonzero contextual_joining prevalence is `calt` (stylistic")
    print("  alternates), not Arabic-style joining.")
    print("=" * 80)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    if not GOOGLE_FONTS_DIR.exists():
        logger.error(
            f"Google Fonts directory not found at {GOOGLE_FONTS_DIR}\n"
            f"Clone it first:\n"
            f"  git clone --depth 1 https://github.com/google/fonts\n"
            f"Then update GOOGLE_FONTS_DIR in this script."
        )
        sys.exit(1)

    logger.info("Step 1-2: Extracting Latin metrics from font files...")
    raw_df = walk_font_directory(GOOGLE_FONTS_DIR)

    if raw_df.empty:
        logger.error("No Latin fonts found! Check GOOGLE_FONTS_DIR / LATIN_DOMINANT_ONLY.")
        sys.exit(1)

    logger.info(f"Extracted {len(raw_df)} Latin font records")
    raw_df.to_csv("latin_raw_metrics.csv", index=False)
    logger.info("Saved raw metrics -> latin_raw_metrics.csv")

    logger.info("Step 4: Computing Latin baseline...")
    baseline = compute_latin_baseline(raw_df)
    baseline["sample_size"] = SAMPLE_SIZE
    baseline["random_seed"] = RANDOM_SEED if SAMPLE_SIZE is not None else None
    baseline["skip_reasons"] = raw_df.attrs.get("skip_reasons", {})

    print_baseline_report(baseline)

    # Flat one-row CSV for merging with the Exposure + Support + non-Latin work
    baseline_to_flat_df(baseline).to_csv("latin_baseline_summary.csv", index=False)
    logger.info("Saved summary -> latin_baseline_summary.csv")

    # Full nested baseline as JSON (keeps stats + prevalence + provenance)
    with open("latin_baseline.json", "w") as fh:
        json.dump(baseline, fh, indent=2)
    logger.info("Saved full baseline -> latin_baseline.json")

    return baseline


if __name__ == "__main__":
    main()
