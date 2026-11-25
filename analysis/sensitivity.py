import logging
import sys
from pathlib import Path

import argparse
import numpy as np
import pandas as pd

try:
    from paths import REPO_ROOT, DATA_ROOT
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import REPO_ROOT, DATA_ROOT

try:
    from .scoring import compute_servedness
except ImportError:
    from analysis.scoring import compute_servedness

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

    # Exposure
    exp = pd.read_csv(EXPOSURE_CSV, names=["script", "exposure"], header=0)
    exp = exp[exp["script"].isin(TARGET_SCRIPTS)].copy()

    # Support
    sup_raw = pd.read_csv(SUPPORT_CSV, names=["script_raw", "count"], header=0)
    records = [{"script": SUPPORT_NAME_MAP[r.script_raw], "support": r.count}
               for r in sup_raw.itertuples() if r.script_raw in SUPPORT_NAME_MAP]
    sup = pd.DataFrame(records).groupby("script", as_index=False)["support"].sum()

    # Diversity index from the similarity analysis (matches final scores)
    diversity = pd.read_csv(SIMILARITY_CSV)[["script", "diversity_index"]]
    logger.info(f"Similarity: \n{SIMILARITY_CSV.name}")

    # Merge
    master = exp.merge(sup, on="script").merge(diversity, on="script")

    # Log-scale exposure and support for visualization
    master["log_exposure"] = np.log10(master["exposure"])
    master["log_support"] = np.log10(master["support"])

    # Compute canonical servedness score (matches analysis/scoring.py)
    sss = compute_servedness(
        support=master.set_index("script")["support"],
        diversity=master.set_index("script")["diversity_index"],
        exposure=master.set_index("script")["exposure"],
    )
    master = master.merge(sss[["servedness_score"]].rename(columns={"servedness_score": "sss"}),
                          left_on="script", right_index=True, how="left")
    master = master.sort_values("sss", ascending=False).reset_index(drop=True)

    logger.info(f"Loaded {len(master)} scripts")
    return master





def calculate_sensitivity(master: pd.DataFrame, shock_percent: float = 0.10) -> pd.DataFrame:
    """Apply one-at-a-time shocks to raw inputs and report SSS changes.

    Uses scoring.compute_servedness so the baseline and shocked scores match
    the values reported elsewhere in the project.
    """
    if not 0 < shock_percent < 1:
        raise ValueError("shock_percent must be between 0 and 1")

    indexed = master.set_index("script")
    baseline_inputs = {
        "support": indexed["support"].astype(float),
        "diversity": indexed["diversity_index"].astype(float),
        "exposure": indexed["exposure"].astype(float),
    }

    def score(inputs: dict[str, pd.Series]) -> pd.Series:
        result = compute_servedness(
            support=inputs["support"],
            diversity=inputs["diversity"],
            exposure=inputs["exposure"],
        )
        return result["servedness_score"]

    def rank_scores(scores: pd.Series) -> pd.Series:
        ordered = scores.sort_values(ascending=False, kind="stable")
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
        nargs="?",
        help="Percentage of change (shock) applied during the sensitivity analysis. " \
        "Use a floating value between 0-1 to represent the percent change (default: 0.1)"
    )
    return parser.parse_args()


def main() -> None:
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
    print("How do shocks and changes in the parameters affect the final SSS Score?")
    print("\nMedian Absolute SSS % change by parameter:")
    print(
        results.assign(abs_change=results["sss_pct_change"].abs())
        .groupby("parameter")["abs_change"].median()
        .sort_values(ascending=False)
        .map("{:.2f}%".format))

    print("How stable is each script to shocks/changes in parameters?")
    print("\nMedian Absolute SSS % change by script:")
    print(
    results.assign(abs_change=results["sss_pct_change"].abs())
           .groupby("script")["abs_change"].median()
           .sort_values(ascending=False)
           .map("{:.2f}%".format)
)
    
    print("\nHow Stable are the current SSS rankings?")

    num_changed = len(results[results["baseline_rank"] != results["shocked_rank"]])
    num_tests = len(results)
    print(f"\nWith a {args.pct_shock*100}% shock to the individually tested parameters.")
    print(f"{num_changed}/{num_tests} tests resulted in a change in overall Script Rankings.")
    print()
    if(num_changed > 0):
        print(
            results[results["baseline_rank"] != results["shocked_rank"]]
        )



    print("Applied one-at-a-time +/- relative shocks to support, diversity, and exposure.")



if __name__ == "__main__":
    main()