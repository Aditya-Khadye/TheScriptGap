"""Exposure research pipeline entrypoint.

This module prepares the exposure dataset used by the Dash dashboard in
`exposure_research/dashboard.py`.

Inputs:
    - support_research/output/big_query_data.csv

Outputs:
    - exposure_research/output/exposure_treemap_data.csv (derived visualization data)

Responsibilities:
    - load and normalize BigQuery script support data
    - aggregate font count by script
    - export cleaned exposure data for visualization

Scope:
    - This file focuses on exposure data preparation and stage orchestration.
    - Dashboard UI code is separated into `exposure_research/dashboard.py`.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from plotly.express.colors import qualitative
import utils.utils_formatting as utils

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
SUPPORT_BIGQUERY_PATH = REPO_ROOT / "support_research" / "output" / "big_query_data.csv"
EXPOSURE_DATA_PATH = OUTPUT_DIR / "exposure_treemap_data.csv"

scripts_list = [
    "devanagari",
    "arabic",
    "bengali",
    "cyrillic",
    "katakana",
    "telugu",
    "tamil",
    "latin"
]

# Assign colors to scripts using a palette
color_palette = qualitative.Pastel
script_colors = {script: color_palette[i % len(color_palette)] for i, script in enumerate(scripts_list)}


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_big_query_data() -> pd.DataFrame:
    if not SUPPORT_BIGQUERY_PATH.exists():
        raise FileNotFoundError(
            f"Expected support output at {SUPPORT_BIGQUERY_PATH}."
            " Run the support stage before loading exposure data."
        )

    big_query_df = pd.read_csv(SUPPORT_BIGQUERY_PATH)
    big_query_df = big_query_df.rename(columns={"scripts": "supported_scripts"})
    return big_query_df


def prepare_exposure_data(save_path: Path | None = None) -> pd.DataFrame:
    """Prepare the exposure dataset for visualization and optional export."""
    big_query_df = load_big_query_data()
    big_query_df = utils.standardize_font_names(big_query_df)
    big_query_df["supported_scripts"] = big_query_df["supported_scripts"].apply(utils.safe_literal_eval)

    exploded_result = big_query_df.explode("supported_scripts")
    exploded_result = exploded_result.rename(columns={"supported_scripts": "script"})

    font_script_df = (
        exploded_result[["font_name", "script", "font_count"]]
        .groupby(["script", "font_name"], as_index=False)["font_count"].sum()
        .sort_values("font_count", ascending=False)
        .reset_index(drop=True)
    )

    font_script_df["font_name"] = font_script_df["font_name"].where(font_script_df["font_count"] >= 5000, "other")
    font_script_df = utils.filter_null_scripts(font_script_df)
    font_script_df = font_script_df[font_script_df["script"].isin(scripts_list)]

    script_totals = font_script_df.groupby("script")["font_count"].sum().sort_values(ascending=False)

    global script_colors
    scripts_list[:] = script_totals.index.tolist()
    script_colors = {script: color_palette[i % len(color_palette)] for i, script in enumerate(scripts_list)}

    if save_path is not None:
        ensure_output_dir()
        font_script_df.to_csv(save_path, index=False)

    return font_script_df


def run_exposure_pipeline(force: bool = False) -> Path:
    ensure_output_dir()
    if EXPOSURE_DATA_PATH.exists() and not force:
        print(f"Exposure output already exists: {EXPOSURE_DATA_PATH} (use --force to rerun)")
        return EXPOSURE_DATA_PATH

    font_script_df = prepare_exposure_data(save_path=EXPOSURE_DATA_PATH)
    print(f"Wrote exposure data to: {EXPOSURE_DATA_PATH}")
    return EXPOSURE_DATA_PATH


def main() -> None:
    run_exposure_pipeline(force=True)


if __name__ == "__main__":
    main()
