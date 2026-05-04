"""Support research data pipeline entrypoint.

This module prepares the Google Fonts support dataset for analysis and
visualization.

Inputs:
    - support_research/output/combined_google_bigquery.csv
      (merged Google Fonts + BigQuery script support data)

Outputs:
    - support_research/output/script_font_counts.csv
      (distinct font count for each supported script)

Responsibilities:
    - load the combined support dataset
    - normalize script names and font labels
    - explode script support rows for per-script counts
    - write a cleaned script font count summary CSV

Scope:
    - This file focuses on support data preparation only
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import utils.utils_formatting as utils

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
COMBINED_SUPPORT_PATH = OUTPUT_DIR / "combined_google_bigquery.csv"
SCRIPT_FONT_COUNTS_PATH = OUTPUT_DIR / "script_font_counts.csv"


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_support_data() -> pd.DataFrame:
    """Load the combined Google support dataset created in an earlier stage."""
    if not COMBINED_SUPPORT_PATH.exists():
        raise FileNotFoundError(
            f"Missing combined support file: {COMBINED_SUPPORT_PATH}."
            " Generate it from raw Google/BigQuery outputs before running this stage."
        )

    df = pd.read_csv(COMBINED_SUPPORT_PATH)
    df["supported_scripts"] = df["supported_scripts"].apply(utils.safe_literal_eval)
    return df


def build_script_font_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count distinct fonts per script after standardizing combined support data."""
    if "font_clean" not in df.columns:
        df["font_clean"] = df["font_name"]

    exploded_result = df.explode("supported_scripts")
    exploded_result = exploded_result.rename(columns={"supported_scripts": "script"})
    exploded_result = exploded_result[exploded_result["script"].notna()]

    dedupe_col = "font_clean" if "font_clean" in exploded_result.columns else "font_name"
    script_font_counts_df = (
        exploded_result.groupby("script", as_index=False)[dedupe_col]
        .nunique()
        .rename(columns={dedupe_col: "distinct_font_count"})
        .sort_values("distinct_font_count", ascending=False)
        .reset_index(drop=True)
    )
    return script_font_counts_df


def run_support_pipeline(force: bool = False) -> Path:
    """Prepare a support summary file that can be used by exposure and visualization stages."""
    ensure_output_dir()

    if SCRIPT_FONT_COUNTS_PATH.exists() and not force:
        print(f"Support output already exists: {SCRIPT_FONT_COUNTS_PATH} (use --force to rerun)")
        return SCRIPT_FONT_COUNTS_PATH

    print(f"Loading support data from: {COMBINED_SUPPORT_PATH}")
    df = load_support_data()
    script_font_counts_df = build_script_font_counts(df)
    script_font_counts_df.to_csv(SCRIPT_FONT_COUNTS_PATH, index=False)

    print(f"Wrote support summary to: {SCRIPT_FONT_COUNTS_PATH}")
    return SCRIPT_FONT_COUNTS_PATH


# NOTE: The helper below is retained for future cleanup and deduplication
# work, but it is not required by the current pipeline stage.

def fuzzy_match_fonts(df, column, threshold=95):
    """Map noisy font names to a canonical font name using fuzzy matching."""
    font_names = [f for f in df[column].unique().tolist() if pd.notna(f)]
    mapping = {}

    for name in font_names:
        if name in mapping:
            continue

        for other in font_names:
            if other in mapping:
                continue

            if name[0].lower() != other[0].lower():
                continue
            if abs(len(name) - len(other)) > 3:
                continue

            from rapidfuzz import fuzz

            score = fuzz.token_sort_ratio(name, other)
            if score >= threshold:
                print(f"Mapping '{other}' to '{name}' with score {score}")
                mapping[other] = name

    df["font_clean"] = df[column].map(mapping)
    return df


def main() -> None:
    print("Running support_research pipeline stage.")
    run_support_pipeline(force=False)


if __name__ == "__main__":
    main()
