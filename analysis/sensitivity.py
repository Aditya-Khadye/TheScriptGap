import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path

import argparse


try:
    from paths import REPO_ROOT, DATA_ROOT
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import REPO_ROOT, DATA_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = DATA_ROOT / "analysis"

SUPPORT_CSV = DATA_ROOT / "support/script_font_counts.csv"
EXPOSURE_CSV = DATA_ROOT / "exposure/exposure_filtered_results.csv"
COMPLEXITY_CSV = DATA_ROOT / "complexity/complexity_index_summary.csv"

# NOTE: MIGHT NEED TO SPECIFY SPECIFIC MODEL PATHS LATER ON IF THERE ARE MULTIPLE RESULTS
SIMILARITY_CSV = DATA_ROOT / "similarity/similarity_results.csv"


SUPPORT_NAME_MAP = {
    "latin": "Latin",
    "cyrillic": "Cyrillic", 
    "japanese": "Katakana",
    "devanagari": "Devanagari",
    "arabic": "Arabic",
    "telugu": "Telugu", 
    "tamil": "Tamil", 
    "bengali": "Bengali",
    "chinese-traditional": "Han", "chinese-simplified": "Han", "chinese-hongkong": "Han",
}

TARGET_SCRIPTS = ["Latin", "Devanagari", "Arabic", "Bengali", "Tamil",
                  "Telugu", "Han", "Katakana", "Cyrillic"]

# ===========================================================================
# Data Loading
# ===========================================================================

def load_all_data(repo: Path = REPO_ROOT) -> pd.DataFrame:
    """Load and merge all four indices."""

    script_dir = Path(__file__).parent.resolve()

    def find_csv(repo_rel: str, filename: str) -> Path:
        p = repo / repo_rel
        if p.exists():
            return p
        p2 = script_dir / filename
        if p2.exists():
            return p2
        raise FileNotFoundError(f"Cannot find {filename} — tried {p} and {p2}")

    # Exposure
    exp = pd.read_csv(EXPOSURE_CSV, names=["script", "exposure"], header=0)
    exp = exp[exp["script"].isin(TARGET_SCRIPTS)].copy()

    # Support

    sup_raw = pd.read_csv(SUPPORT_CSV, names=["script_raw", "count"], header=0)
    records = [{"script": SUPPORT_NAME_MAP[r.script_raw], "support": r.count}
               for r in sup_raw.itertuples() if r.script_raw in SUPPORT_NAME_MAP]
    sup = pd.DataFrame(records).groupby("script", as_index=False)["support"].sum()

    # Mean cosine distance from the similarity analysis
    similarity = pd.read_csv(SIMILARITY_CSV)[["script", "mean_cosine_distance"]]
    logger.info(f"Similarity: \n{SIMILARITY_CSV.name}")

    # Merge (use similarity_index instead)
    master = exp.merge(sup, on="script").merge(similarity, on="script")

    # Log-scale exposure and support for visualization
    master["log_exposure"] = np.log10(master["exposure"])
    master["log_support"] = np.log10(master["support"])

    # ------------------------------------------------------------------
    # Script Servedness Score (SSS) — raw-input sensitivity formula
    # ------------------------------------------------------------------
    #   effective_choice = log10(support) * mean_cosine_distance      # quantity * variety
    #   SSS = effective_choice / log10(exposure)                  # choice per (log) demand

    effective_choice = master["log_support"] * master["mean_cosine_distance"]
    master["sss"] = effective_choice / master["log_exposure"]

    # Sort by raw score (best served first)
    master = master.sort_values("sss", ascending=False).reset_index(drop=True)

    logger.info(f"Loaded {len(master)} scripts")
    return master





def calculate_sensitivity(master: pd.DataFrame, shock_percent: float = 0.10) -> pd.DataFrame:
    """Apply one-at-a-time shocks to raw inputs and report raw SSS changes."""
    if not 0 < shock_percent < 1:
        raise ValueError("shock_percent must be between 0 and 1")

    indexed = master.set_index("script")
    baseline_inputs = {
        "support": indexed["support"].astype(float),
        "mean_cosine_distance": indexed["mean_cosine_distance"].astype(float),
        "exposure": indexed["exposure"].astype(float),
    }

    def score(inputs: dict[str, pd.Series]) -> pd.Series:
        return (
            np.log10(inputs["support"])
            * inputs["mean_cosine_distance"]
            / np.log10(inputs["exposure"])
        )

    def rank_scores(scores: pd.Series) -> pd.Series:
        ordered = scores.sort_values(kind="stable")
        return pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index)

    baseline_sss = score(baseline_inputs)
    baseline_rank = rank_scores(baseline_sss)
    results = []

    for script in baseline_sss.index:
        for parameter, values in baseline_inputs.items():
            for direction, factor in (("Up", 1 + shock_percent), ("Down", 1 - shock_percent)):
                shocked_inputs = {name: series.copy() for name, series in baseline_inputs.items()}
                baseline_input = float(values.loc[script])
                shocked_value = baseline_input * factor
                shocked_inputs[parameter].loc[script] = shocked_value

                shocked_sss = score(shocked_inputs)
                base_value = float(baseline_sss.loc[script])
                new_value = float(shocked_sss.loc[script])
                results.append({
                    "script": script,
                    "parameter": parameter,
                    "direction": direction,
                    "baseline_input": baseline_input,
                    "shocked_input": shocked_value,
                    "baseline_sss": base_value,
                    "shocked_sss": new_value,
                    "sss_change": new_value - base_value,
                    "sss_pct_change": (
                        (new_value - base_value) / base_value * 100
                        if base_value != 0 else np.nan
                    ),
                    "baseline_rank": int(baseline_rank.loc[script]),
                    "shocked_rank": int(rank_scores(shocked_sss).loc[script]),
                })

    return pd.DataFrame(results)

def parse_arguments():
    """Handles terminal parsing and documentation configuration."""
    parser = argparse.ArgumentParser(
        description="Sensitivity Analysis of the Script Servedness Score value. " \
        "This script applies shocks (percentage changes to key values used in " \
        "the calculation of the SSS) in order to determine their affect on the final scores."
    )
    parser.add_argument(
        "pct_shock", 
        type=float,
        default=0.1,
        help="Percentage of change (shock) applied during the sensitivity analysis. " \
        "Use a floating value between 0-1 to represent the percent change"
    )
    return parser.parse_args()


def main() -> None:
    # TODO: uses slightly different calculation than normal in order to get results
    # We use a normalized score in our other calculations but this won't work here
    # Because some normalized results cannot be relatively changed (0's)
    # We might wanna swap everything over to this method though 

    args = parse_arguments()

    master = load_all_data()
    results = calculate_sensitivity(master, args.pct_shock)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sss_sensitivity.csv"
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved sensitivity results to {output_path}")
    print("SSS percent change is undefined when baseline raw SSS is zero.")
    print()
    print("SENSITIVITY ANALYSIS REPORT:")

    print("\nMedian absolute SSS change by parameter:")
    print(results.assign(abs_change=results["sss_change"].abs())
        .groupby("parameter")["abs_change"].median().sort_values(ascending=False))
    print("\nMedian absolute SSS change by script:")
    print(results.assign(abs_change=results["sss_change"].abs())
        .groupby("script")["abs_change"].median().sort_values(ascending=False))
    print("\nCurrently Utilizing a different calculation method to avoid baseline SSS = 0")
    print("Applied one-at-a-time +/- relative shocks to support, mean cosine distance, and exposure.")



if __name__ == "__main__":
    main()