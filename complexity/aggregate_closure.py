"""
Aggregate the v1.1 closure measurement to scripts and compare with v1.0.

Usage (from the repo root):
    uv run python -m complexity.aggregate_closure
Reads data/complexity_v11/per_family_script.csv and the v1.0 summary in
data/complexity/, writes by-script results, weight-grid ranks and a summary
to data/complexity_v11/.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["Latin", "Cyrillic", "Han", "Katakana", "Arabic", "Devanagari", "Bengali", "Tamil", "Telugu"]
B, SEED = 2000, 3
W_DEFAULT = (0.5, 0.3, 0.2)


def boot_median_ci(x: np.ndarray, rng) -> tuple[float, float]:
    if len(x) < 3:
        return np.nan, np.nan
    meds = np.median(rng.choice(x, (B, len(x)), replace=True), axis=1)
    return tuple(np.percentile(meds, [2.5, 97.5]))


def minmax(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min())


def composite(agg: pd.DataFrame, w) -> pd.Series:
    return w[0] * minmax(agg["E"]) + w[1] * minmax(agg["V"]) + w[2] * minmax(agg["F"])


def main(closure_csv: str, v1_csv: str, out_dir: str | None = None, verbose: bool = True):
    """Returns (by-script table, summary dict); writes CSVs when out_dir is given."""
    d = pd.read_csv(closure_csv)
    status = d["status"].fillna("missing").str.split(":").str[0].value_counts()
    ok = d[d.status == "ok"].copy()
    rng = np.random.default_rng(SEED)

    rows = []
    for s in ORDER:
        x = ok[ok.script == s]
        r = {"script": s, "families": x.family_dir.nunique()}
        for m in ["E", "E_all", "E_v0", "E_v1", "V", "V_ink", "F", "F_v1"]:
            v = x[m].dropna().to_numpy()
            r[m] = np.median(v)
            r[f"{m}_q25"], r[f"{m}_q75"] = np.percentile(v, [25, 75])
        r["E_ci_lo"], r["E_ci_hi"] = boot_median_ci(x["E"].to_numpy(), rng)
        r["V_ci_lo"], r["V_ci_hi"] = boot_median_ci(x["V"].dropna().to_numpy(), rng)
        r["share_with_script_record"] = x.has_script_record.mean()
        rows.append(r)
    agg = pd.DataFrame(rows).set_index("script")

    v1 = pd.read_csv(v1_csv).set_index("script")
    agg["E_v1_committed"] = v1["median_expansion_ratio"]
    agg["V_v1_committed"] = v1["median_vertical_footprint"]
    agg["F_v1_committed"] = v1["median_friction_score"]
    agg["C_v1_committed"] = v1["complexity_C"]
    agg["files_v1_committed"] = v1["font_count"]

    agg["C"] = composite(agg, W_DEFAULT)
    agg["rank_C"] = agg["C"].rank(ascending=False).astype(int)          # 1 = most complex
    agg["rank_C_v1"] = agg["C_v1_committed"].rank(ascending=False).astype(int)
    for m in ["E", "V", "F"]:
        agg[f"rank_{m}"] = agg[m].rank(ascending=False, method="min").astype(int)

    # weight sensitivity over the simplex, step 0.05
    grid = [(a / 20, b / 20, (20 - a - b) / 20) for a in range(21) for b in range(21 - a)]
    ranks = pd.DataFrame({w: composite(agg, w).rank(ascending=False) for w in grid}).T
    agg["rank_min_over_weights"] = ranks.min().astype(int)
    agg["rank_max_over_weights"] = ranks.max().astype(int)
    rho = ranks.apply(lambda r: r.corr(agg["rank_C"], method="spearman"), axis=1)

    summary = {
        "rows_ok": int(len(ok)), **{f"status_{k}": int(v) for k, v in status.items()},
        "rho_C_v1_vs_v11": float(agg["C"].corr(agg["C_v1_committed"], method="spearman")),
        "rho_E_v1_vs_closure": float(agg["E"].corr(agg["E_v1_committed"], method="spearman")),
        "rho_E_closure_vs_F": float(agg["E"].corr(agg["F"], method="spearman")),
        "rho_E_v1_vs_F_v1_committed": float(agg["E_v1_committed"].corr(agg["F_v1_committed"], method="spearman")),
        "rho_V_vs_V_ink": float(agg["V"].corr(agg["V_ink"], method="spearman")),
        "weight_grid_size": len(grid),
        "weight_grid_rho_min": float(rho.min()),
        "weight_grid_rho_median": float(rho.median()),
        # per-font: is E still a file-level constant across scripts?
    }
    multi = ok.groupby("family_dir").filter(lambda g: g.script.nunique() >= 2)
    spread = multi.groupby("family_dir").E.agg(lambda s: s.max() / s.min())
    summary["closure_E_within_file_spread_median"] = float(spread.median())
    summary["closure_E_share_files_within_10pct"] = float((spread <= 1.10).mean())
    spread_v1 = multi.groupby("family_dir").E_v1.agg(lambda s: s.max() / s.min())
    summary["v1_E_within_file_spread_median"] = float(spread_v1.median())
    summary["v1_E_share_files_within_10pct"] = float((spread_v1 <= 1.10).mean())

    if out_dir is not None:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        agg.to_csv(out / "by_script.csv")
        ranks.to_csv(out / "weight_grid_ranks.csv")
        pd.Series(summary).to_csv(out / "summary.csv", header=["value"])

    if verbose:
        pd.set_option("display.width", 250, "display.max_columns", 40)
        cols = ["families", "files_v1_committed", "E_v0", "E_v1_committed", "E", "E_ci_lo", "E_ci_hi", "E_all",
                "V", "V_ink", "F", "F_v1_committed", "C_v1_committed", "C", "rank_C_v1", "rank_C",
                "rank_min_over_weights", "rank_max_over_weights", "share_with_script_record"]
        print(agg[cols].round(2).sort_values("C", ascending=False).to_string())
        for k, v in summary.items():
            print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
    return agg, summary


if __name__ == "__main__":
    REPO = Path(__file__).resolve().parents[1]
    main(str(REPO / "data/complexity_v11/per_family_script.csv"),
         str(REPO / "data/complexity/complexity_index_summary.csv"),
         str(REPO / "data/complexity_v11"))
