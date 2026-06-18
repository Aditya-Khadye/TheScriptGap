"""Exposure research pipeline entrypoint.

This module prepares the exposure dataset used by the Dash dashboard in
`exposure_research/dashboard.py` and writes summary CSVs for visualization.

Inputs:
    - data/bigquery/big_query_data.csv (from BigQuery preflight or manual export)

Outputs:
    - data/exposure/exposure_treemap_data.csv (treemap visualization data)
    - data/exposure/exposure_filtered_results.csv (per-script totals for heatmap)

Responsibilities:
    - optionally pull HTTP Archive data via BigQuery
    - load and normalize BigQuery script support data
    - aggregate font count by script
    - export cleaned exposure data for visualization

Scope:
    - This file focuses on exposure data preparation and stage orchestration.
    - Dashboard UI code is separated into `exposure_research/dashboard.py`.
"""

from pathlib import Path
import sys

import pandas as pd
from plotly.express.colors import qualitative

# Access paths to resources outside of this folder
try:
    from paths import EXPOSURE_DATA_DIR, BIGQUERY_DATA_DIR
    from utils import filter_null_scripts, safe_literal_eval, standardize_font_names
    from exposure_research.script_names import TREEMAP_SCRIPTS, TREEMAP_DISPLAY_SCRIPT, to_canonical_script
    from exposure_research.bigquery_pull import pull_exposure_data

except ModuleNotFoundError:
    # When running the module from different working directories, ensure
    # the project root (one level up from this package) is on sys.path.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from paths import EXPOSURE_DATA_DIR, BIGQUERY_DATA_DIR
    from utils import filter_null_scripts, safe_literal_eval, standardize_font_names
    from exposure_research.script_names import TREEMAP_SCRIPTS, TREEMAP_DISPLAY_SCRIPT, to_canonical_script
    from exposure_research.bigquery_pull import pull_exposure_data


OUTPUT_DIR = EXPOSURE_DATA_DIR
BIGQUERY_PATH = BIGQUERY_DATA_DIR / "big_query_data.csv"
EXPOSURE_DATA_PATH = OUTPUT_DIR / "exposure_treemap_data.csv"
EXPOSURE_FILTERED_PATH = OUTPUT_DIR / "exposure_filtered_results.csv"

scripts_list = list(TREEMAP_SCRIPTS)

# Assign colors to scripts using a palette
color_palette = qualitative.Pastel
script_colors = {script: color_palette[i % len(color_palette)] for i, script in enumerate(scripts_list)}


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_big_query_data() -> pd.DataFrame:
    if not BIGQUERY_PATH.exists():
        raise FileNotFoundError(
            f"Expected BigQuery output at {BIGQUERY_PATH}."
            " Run exposure_research/bigquery_pull.py or set SKIP_BIGQUERY=1 with existing data."
        )

    big_query_df = pd.read_csv(BIGQUERY_PATH)
    if "scripts" in big_query_df.columns and "supported_scripts" not in big_query_df.columns:
        big_query_df = big_query_df.rename(columns={"scripts": "supported_scripts"})
    return big_query_df


def build_exposure_filtered(font_script_df: pd.DataFrame) -> pd.DataFrame:
    """Build canonical per-script exposure totals for the heatmap stage."""
    totals = (
        font_script_df.groupby("script", as_index=False)["font_count"]
        .sum()
        .rename(columns={"font_count": "count"})
    )
    totals["canonical"] = totals["script"].map(to_canonical_script)
    totals = totals[totals["canonical"].notna()]
    summary = (
        totals.groupby("canonical", as_index=False)["count"]
        .sum()
        .rename(columns={"canonical": "script"})
        .sort_values("count", ascending=False)
    )
    return summary


def prepare_exposure_data(save_path: Path | None = None) -> pd.DataFrame:
    """Prepare the exposure dataset for visualization and optional export."""
    big_query_df = load_big_query_data()
    big_query_df = standardize_font_names(big_query_df)
    big_query_df["supported_scripts"] = big_query_df["supported_scripts"].apply(safe_literal_eval)

    exploded_result = big_query_df.explode("supported_scripts")
    exploded_result = exploded_result.rename(columns={"supported_scripts": "script"})
    exploded_result["script"] = exploded_result["script"].replace(TREEMAP_DISPLAY_SCRIPT)

    font_script_df = (
        exploded_result[["font_name", "script", "font_count"]]
        .groupby(["script", "font_name"], as_index=False)["font_count"].sum()
        .sort_values("font_count", ascending=False)
        .reset_index(drop=True)
    )

    font_script_df["font_name"] = font_script_df["font_name"].where(font_script_df["font_count"] >= 5000, "other")
    font_script_df = filter_null_scripts(font_script_df)
    font_script_df = font_script_df[font_script_df["script"].isin(scripts_list)]

    # Combine counts for "other" fonts
    font_script_df = (
        font_script_df.groupby(["script", "font_name"], as_index=False)["font_count"].sum()
        .sort_values("font_count", ascending=False)
        .reset_index(drop=True)
    )

    script_totals = font_script_df.groupby("script")["font_count"].sum().sort_values(ascending=False)

    global script_colors
    scripts_list[:] = script_totals.index.tolist()
    script_colors = {script: color_palette[i % len(color_palette)] for i, script in enumerate(scripts_list)}

    if save_path is not None:
        ensure_output_dir()
        font_script_df.to_csv(save_path, index=False)
        filtered = build_exposure_filtered(font_script_df)
        filtered.to_csv(EXPOSURE_FILTERED_PATH, index=False)

    return font_script_df


def run_exposure_pipeline(force: bool = False) -> Path:
    ensure_output_dir()
    if EXPOSURE_DATA_PATH.exists() and EXPOSURE_FILTERED_PATH.exists() and not force:
        print(f"Exposure output already exists: {EXPOSURE_DATA_PATH} (use --force to rerun)")
        return EXPOSURE_DATA_PATH

    try:
        pull_exposure_data(force=force)
    except Exception as exc:
        if BIGQUERY_PATH.exists():
            print(f"BigQuery pull skipped or failed ({exc}); using existing {BIGQUERY_PATH}")
        else:
            raise

    font_script_df = prepare_exposure_data(save_path=EXPOSURE_DATA_PATH)
    print(f"Wrote exposure data to: {EXPOSURE_DATA_PATH}")
    print(f"Wrote exposure summary to: {EXPOSURE_FILTERED_PATH}")
    return EXPOSURE_DATA_PATH


def main() -> None:
    run_exposure_pipeline(force=True)


if __name__ == "__main__":
    main()
