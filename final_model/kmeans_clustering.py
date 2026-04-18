"""
===============================================================================
Final K-Means Clustering — Script Underservedness Tiers (Real Data)
===============================================================================
Project:  TRC / Monotype — Identifying Underserved Scripts
Author:   Aditya (UCF MIT2 Lab)

Data Sources (all real, no hardcoding):
    exposure_filtered_results.csv  — CrUX web exposure per script (BigQuery)
    google_support_toplist.csv     — Google Fonts family count per script
    diversity_index_summary.csv    — ViT diversity index (100-glyph preferred,
                                     10-glyph fallback)
    similarity_index_summary.csv   — fontTools complexity index (if available)

Usage:
    python kmeans_clustering.py

    Place all CSVs in the same directory as this script.
    Outputs go to ./clustering_outputs/
===============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("./clustering_outputs")

# Scripts we care about — non-Latin only
TARGET_SCRIPTS = ["Cyrillic", "Katakana", "Devanagari", "Arabic",
                  "Han", "Bengali", "Tamil", "Telugu"]

# Google Fonts script name → our canonical name
SUPPORT_NAME_MAP = {
    "cyrillic":          "Cyrillic",
    "japanese":          "Katakana",  # Japanese fonts cover Katakana
    "devanagari":        "Devanagari",
    "arabic":            "Arabic",
    "telugu":            "Telugu",
    "tamil":             "Tamil",
    "bengali":           "Bengali",
    # Han = sum of all Chinese variants
    "chinese-traditional": "Han",
    "chinese-simplified":  "Han",
    "chinese-hongkong":    "Han",
}

# Hardcoded complexity from fontTools pipeline (used only if no CSV found)
COMPLEXITY_FALLBACK = {
    "Cyrillic": 0.167, "Katakana": 0.000, "Devanagari": 0.847,
    "Arabic": 0.541, "Han": 0.002, "Bengali": 0.615,
    "Tamil": 0.290, "Telugu": 0.433,
}


# ===========================================================================
# STEP 1: Load and merge all real data
# ===========================================================================

def load_exposure(path: Path) -> pd.DataFrame:
    """
    Load CrUX exposure data.
    Filter to non-Latin target scripts only.
    """
    df = pd.read_csv(path)
    df.columns = ["script", "exposure"]
    df = df[df["script"].isin(TARGET_SCRIPTS)].copy()
    logger.info(f"  Exposure: {len(df)} scripts from {path.name}")
    return df[["script", "exposure"]]


def load_support(path: Path) -> pd.DataFrame:
    """
    Load Google Fonts support data and aggregate to canonical script names.

    Google Fonts splits Chinese into traditional/simplified/hongkong.
    We sum them into a single Han count.
    Font families supporting Katakana are listed as 'japanese'.
    """
    df = pd.read_csv(path)
    df.columns = ["script_raw", "count"]

    records = []
    for _, row in df.iterrows():
        canonical = SUPPORT_NAME_MAP.get(row["script_raw"])
        if canonical:
            records.append({"script": canonical, "count": row["count"]})

    support = pd.DataFrame(records).groupby("script", as_index=False)["count"].sum()
    support.columns = ["script", "support"]
    support = support[support["script"].isin(TARGET_SCRIPTS)]
    logger.info(f"  Support: {len(support)} scripts from {path.name}")
    return support


def load_diversity(paths: list) -> pd.DataFrame:
    """
    Load diversity index — try 100-glyph first, fall back to 10-glyph.
    100-glyph is preferred because it covers base forms, conjuncts,
    numerals, and signs — a more comprehensive visual sample.
    """
    for path in paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            df = df[["script", "diversity_index"]].copy()
            df = df[df["script"].isin(TARGET_SCRIPTS)]
            logger.info(f"  Diversity: {len(df)} scripts from {Path(path).name}")
            return df
    raise FileNotFoundError("No diversity CSV found. Run the ViT pipeline first.")


def load_complexity(path: Path) -> pd.DataFrame:
    """
    Load complexity from fontTools similarity pipeline.
    Complexity = 1 - similarity_S (higher = more engineering effort).
    Falls back to hardcoded values if CSV not found.
    """
    if path.exists():
        df = pd.read_csv(path)
        df["complexity"] = 1.0 - df["similarity_S"]
        df = df[["script", "complexity", "similarity_S"]]
        df = df[df["script"].isin(TARGET_SCRIPTS)]
        logger.info(f"  Complexity: {len(df)} scripts from {path.name}")
        return df
    else:
        logger.warning(f"  Complexity: {path.name} not found — using fontTools pipeline results")
        df = pd.DataFrame([
            {"script": s, "complexity": c, "similarity_S": 1 - c}
            for s, c in COMPLEXITY_FALLBACK.items()
        ])
        return df


def build_master(base_dir: Path = Path(".")) -> pd.DataFrame:
    """
    Merge all four indices into one master DataFrame.
    All data comes from CSVs — no hardcoded values for exposure/support/diversity.
    """
    logger.info("Loading data...")

    exposure = load_exposure(base_dir / "exposure_filtered_results.csv")
    support = load_support(base_dir / "google_support_toplist.csv")
    diversity = load_diversity([
        base_dir / "vit_outputs_100" / "diversity_index_summary.csv",
        base_dir / "diversity_index_summary.csv",          # 100-glyph uploaded
        base_dir / "vit_outputs" / "diversity_index_summary.csv",
    ])
    complexity = load_complexity(base_dir / "similarity_index_summary.csv")

    # Merge everything on script
    master = exposure \
        .merge(support, on="script", how="inner") \
        .merge(diversity, on="script", how="inner") \
        .merge(complexity[["script", "complexity", "similarity_S"]], on="script", how="left")

    # Fill any missing complexity with fallback
    for idx, row in master.iterrows():
        if pd.isna(row["complexity"]):
            master.at[idx, "complexity"] = COMPLEXITY_FALLBACK.get(row["script"], 0.5)
            master.at[idx, "similarity_S"] = 1 - master.at[idx, "complexity"]

    # Derived feature
    master["exposure_per_font"] = master["exposure"] / master["support"]

    master = master[master["script"].isin(TARGET_SCRIPTS)].reset_index(drop=True)
    logger.info(f"\n  Master: {len(master)} scripts × {len(master.columns)} features")
    return master


# ===========================================================================
# STEP 2: Feature preparation
# ===========================================================================

def prepare_features(master: pd.DataFrame) -> tuple:
    """
    Select and standardize features for K-Means.

    Features:
        log_exposure    — Log-scaled CrUX web exposure
        log_support     — Log-scaled Google Fonts family count
        complexity      — Infrastructure complexity (0-1, fontTools)
        diversity_index — Visual diversity of fonts (0-1, ViT)

    Why log-scale exposure and support?
        Raw values span orders of magnitude. Log compresses the range
        while preserving relative ordering. Without this, K-Means Euclidean
        distance would be dominated by these two raw count features.

    Why StandardScaler?
        After log-scaling, features still have different ranges. Z-score
        normalization (mean=0, std=1) ensures all four indices contribute
        equally to the clustering.
    """
    master = master.copy()
    master["log_exposure"] = np.log1p(master["exposure"])
    master["log_support"] = np.log1p(master["support"])

    feature_cols = ["log_exposure", "log_support", "complexity", "diversity_index"]
    X_raw = master[feature_cols].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    logger.info(f"\nFeatures: {feature_cols}")
    logger.info(f"Means post-scaling: {X.mean(axis=0).round(3)}")
    logger.info(f"Stds  post-scaling: {X.std(axis=0).round(3)}")

    return X, feature_cols, scaler


# ===========================================================================
# STEP 3: K-Means with model selection
# ===========================================================================

def run_kmeans(X: np.ndarray, scripts: list, k_range=range(2, 6)) -> dict:
    """
    Run K-Means for k = 2, 3, 4, 5 and evaluate with silhouette score.

    With only 8 scripts, k=4 or 5 would give clusters of 1-2 scripts —
    too granular to be meaningful. Silhouette score automatically penalizes
    over-clustering.
    """
    logger.info(f"\nRunning K-Means for k = {list(k_range)}...")
    results = {}

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=100, max_iter=1000, random_state=42)
        labels = km.fit_predict(X)

        sil_avg = silhouette_score(X, labels)
        sil_samples = silhouette_samples(X, labels)

        results[k] = {
            "kmeans": km,
            "labels": labels,
            "inertia": km.inertia_,
            "silhouette_avg": sil_avg,
            "silhouette_per_script": dict(zip(scripts, sil_samples.tolist())),
            "centroids": km.cluster_centers_,
        }
        logger.info(f"  k={k}: silhouette={sil_avg:.3f}, inertia={km.inertia_:.2f}")

    return results


def select_k(results: dict) -> int:
    """Select k with best silhouette. Prefer simpler if within 0.05."""
    best_k = max(results, key=lambda k: results[k]["silhouette_avg"])
    best_sil = results[best_k]["silhouette_avg"]
    for k in sorted(results):
        if k < best_k and results[k]["silhouette_avg"] >= best_sil - 0.05:
            logger.info(f"  Preferring simpler k={k} (within 0.05 of best)")
            best_k = k
            break
    logger.info(f"\n  Optimal k={best_k} (silhouette={results[best_k]['silhouette_avg']:.3f})")
    return best_k


# ===========================================================================
# STEP 4: Label tiers
# ===========================================================================

def label_tiers(master: pd.DataFrame, labels: np.ndarray,
                centroids: np.ndarray, feature_cols: list,
                scaler) -> tuple:
    """
    Assign underservedness tier labels to clusters.

    Underservedness score per cluster centroid:
        High exposure + low support + high complexity + low diversity
        = most underserved (highest score → Tier 1 = Critical)
    """
    master = master.copy()
    master["cluster"] = labels

    centroids_orig = scaler.inverse_transform(centroids)
    centroid_df = pd.DataFrame(centroids_orig, columns=feature_cols)

    centroid_df["underservedness"] = (
          centroid_df["log_exposure"]
        - centroid_df["log_support"] * 1.5
        + centroid_df["complexity"] * 2.0
        - centroid_df["diversity_index"]
    )
    centroid_df["rank"] = centroid_df["underservedness"].rank(ascending=False).astype(int)

    n = len(centroid_df)
    tier_labels = {
        2: {1: "Underserved",             2: "Well served"},
        3: {1: "Critically underserved",   2: "Moderately served", 3: "Well served"},
        4: {1: "Critically underserved",   2: "Underserved",
            3: "Moderately served",        4: "Well served"},
    }
    label_map = tier_labels.get(n, {i: f"Tier {i}" for i in range(1, n + 1)})
    centroid_df["tier"] = centroid_df["rank"].map(label_map)

    cluster_to_tier = dict(zip(centroid_df.index, centroid_df["tier"]))
    master["tier"] = master["cluster"].map(cluster_to_tier)
    tier_to_rank = {v: k for k, v in label_map.items()}
    master["tier_rank"] = master["tier"].map(tier_to_rank)

    return master, centroid_df


# ===========================================================================
# STEP 5: Report
# ===========================================================================

def print_report(master, centroid_df, results, k, feature_cols):
    print("\n" + "=" * 80)
    print("FINAL SCRIPT UNDERSERVEDNESS TIERS — K-MEANS (REAL DATA)")
    print("=" * 80)

    print("\n📊 Model selection:\n")
    for ki in sorted(results):
        marker = " ← selected" if ki == k else ""
        print(f"  k={ki}: silhouette={results[ki]['silhouette_avg']:.3f}, "
              f"inertia={results[ki]['inertia']:.1f}{marker}")

    print(f"\n\n📐 Cluster centroids (k={k}):\n")
    print(centroid_df[feature_cols + ["underservedness", "tier"]].to_string(float_format="%.3f"))

    print("\n\n🏷️  Script tier assignments:\n")
    display = master.sort_values("tier_rank")[[
        "script", "tier", "exposure", "support",
        "complexity", "diversity_index", "exposure_per_font"
    ]].copy()
    display["exposure"] = display["exposure"].apply(lambda x: f"{x:,.0f}")
    display["support"] = display["support"].apply(lambda x: f"{x:,}")
    display["complexity"] = display["complexity"].round(3)
    display["diversity_index"] = display["diversity_index"].round(3)
    display["exposure_per_font"] = display["exposure_per_font"].round(1)
    print(display.to_string(index=False))

    print("\n\n📋 Tier summaries:\n")
    for tier in master.sort_values("tier_rank")["tier"].unique():
        grp = master[master["tier"] == tier]
        scripts_list = ", ".join(grp["script"].tolist())
        print(f"  {tier}:")
        print(f"    Scripts:        {scripts_list}")
        print(f"    Avg exposure:   {grp['exposure'].mean():,.0f}")
        print(f"    Avg fonts:      {grp['support'].mean():.1f}")
        print(f"    Avg complexity: {grp['complexity'].mean():.3f}")
        print(f"    Avg diversity:  {grp['diversity_index'].mean():.3f}")
        print()

    print("\n🔍 Per-script silhouette scores:\n")
    sil = results[k]["silhouette_per_script"]
    for script in master.sort_values("tier_rank")["script"]:
        s = sil[script]
        tier = master[master["script"] == script]["tier"].iloc[0]
        fit = "strong" if s > 0.5 else "moderate" if s > 0.2 else "weak"
        print(f"  {script:12s}  sil={s:.3f}  {fit} fit → '{tier}'")

    print("\n\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)

    worst = master.sort_values("exposure_per_font", ascending=False).iloc[0]
    print(f"\n  Highest demand gap: {worst['script']}")
    print(f"    {int(worst['exposure']):,} web appearances → only "
          f"{int(worst['support'])} font families → "
          f"{worst['exposure_per_font']:.0f} pages per font")

    critical = master[master["tier"].str.contains("Critical|Underserved", na=False)]
    if not critical.empty:
        print(f"\n  Underserved scripts ({len(critical)}): "
              f"{', '.join(critical.sort_values('exposure_per_font', ascending=False)['script'].tolist())}")

    print("\n" + "=" * 80)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    repo_root = Path.home() / "TheScriptGap_clean"
    exposure = load_exposure(repo_root / "exposure_research/output/exposure_filtered_results.csv")
    support = load_support(repo_root / "support_research/output/google_support_toplist.csv")
    diversity = load_diversity([
        repo_root / "similarity_research/diversity_research/vit_outputs_100/diversity_index_summary.csv",
        repo_root / "similarity_research/diversity_research/vit100_outputs/diversity_index_summary.csv",
        repo_root / "similarity_research/diversity_research/vit_outputs/diversity_index_summary.csv",
    ])
    complexity = load_complexity(repo_root / "similarity_research/similarity_index_summary.csv")
    master = exposure.merge(support, on="script", how="inner").merge(diversity, on="script", how="inner").merge(complexity[["script","complexity","similarity_S"]], on="script", how="left")
    for idx, row in master.iterrows():
        if __import__('pandas').isna(row["complexity"]):
            master.at[idx, "complexity"] = COMPLEXITY_FALLBACK.get(row["script"], 0.5)
            master.at[idx, "similarity_S"] = 1 - master.at[idx, "complexity"]
    master["exposure_per_font"] = master["exposure"] / master["support"]
    master = master[master["script"].isin(TARGET_SCRIPTS)].reset_index(drop=True)

    print("\n📊 Master DataFrame (real data):\n")
    print(master[["script", "exposure", "support", "complexity",
                   "diversity_index", "exposure_per_font"]].to_string(index=False))

    X, feature_cols, scaler = prepare_features(master)
    results = run_kmeans(X, master["script"].tolist())
    k = select_k(results)
    master, centroid_df = label_tiers(
        master, results[k]["labels"], results[k]["centroids"], feature_cols, scaler
    )

    print_report(master, centroid_df, results, k, feature_cols)

    # Save outputs
    master.to_csv(OUTPUT_DIR / "final_script_tiers.csv", index=False)
    centroid_df.to_csv(OUTPUT_DIR / "cluster_centroids.csv", index=False)
    pd.DataFrame([
        {"k": ki, "silhouette": r["silhouette_avg"], "inertia": r["inertia"]}
        for ki, r in results.items()
    ]).to_csv(OUTPUT_DIR / "clustering_analysis.csv", index=False)

    # Clean summary
    summary = master[["script", "tier", "tier_rank", "exposure", "support",
                       "complexity", "diversity_index", "exposure_per_font"]].sort_values("tier_rank")
    summary.to_csv(OUTPUT_DIR / "tier_summary.csv", index=False)

    logger.info(f"\nAll outputs saved to {OUTPUT_DIR}/")
    logger.info("  final_script_tiers.csv")
    logger.info("  tier_summary.csv")
    logger.info("  cluster_centroids.csv")
    logger.info("  clustering_analysis.csv")


if __name__ == "__main__":
    main()
