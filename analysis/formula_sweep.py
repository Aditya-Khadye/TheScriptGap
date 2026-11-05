"""
===============================================================================
Formula-weight sensitivity for the Script Servedness Score (SSS)
===============================================================================
Project:  TheScriptGap  |  v1.0  |  Monthly plan, October 2026 item:
          "Sweep the weighting of the similarity discount and the relative
           contribution of exposure and support. Plot rank stability."

Question: are the v1.0 conclusions a property of the data, or of how the three
inputs (support, diversity, demand) happen to be weighted in the formula?

Two parts, both computed from committed data only:

  Part A. Can the shipped formula's demand term move the ranking at all?
          v1.0 divides effective choice by log10(exposure). We raise that
          denominator to a power gamma and find the smallest gamma that changes
          the ordering. If gamma has to be absurdly large, demand is inert in
          the shipped score.

  Part B. Relative-contribution sweep. Each input is log-scaled and z-scored so
          one unit of weight buys the same leverage on every input, then

              score(ws, wd, we) = ws*z(log support) + wd*z(log diversity)
                                  - we*z(log demand)

          is evaluated over the whole weight simplex (ws + wd + we = 1,
          default step 0.05 -> 231 points). Higher score = better served,
          matching v1.0's direction. This is v1.0's multiplicative form
          (support x diversity / demand) in log space, with the exponents set
          by the sweep instead of fixed. Raw mean cosine distance is used for
          diversity so no script is pinned at zero by min-max endpoints.

          At every point we record each script's rank and whether three
          headline claims survive:
            1. Indic four {Tamil, Bengali, Devanagari, Telugu} are the bottom 4
            2. Latin is the best-served script
            3. The six v1.0 "Underserved" scripts are exactly the bottom 6

Outputs (data/analysis/):
    formula_sweep_grid.csv       every weight point: ranks, Spearman vs v1.0, claims
    formula_sweep_by_demand.csv  claim survival rate at each demand weight
    formula_sweep.md             written report
    formula_sweep.png            figure (unless --no-plot)

Usage:
    uv run python analysis/formula_sweep.py
    uv run python analysis/formula_sweep.py --step 0.02 --no-plot
===============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from paths import DATA_ROOT
    from analysis.scoring import compute_servedness
    from analysis.robustness import load_exposure, load_support, spearman
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import DATA_ROOT
    from analysis.scoring import compute_servedness
    from analysis.robustness import load_exposure, load_support, spearman

SIMILARITY_CSV = DATA_ROOT / "similarity" / "similarity_results.csv"
CANONICAL_CSV = DATA_ROOT / "final" / "script_servedness.csv"
OUT_DIR = DATA_ROOT / "analysis"

INDIC_FOUR = {"Tamil", "Bengali", "Devanagari", "Telugu"}
CLAIMS = {
    "indic_bottom4": "Indic four are the 4 most underserved",
    "latin_best": "Latin is the best-served script",
    "underserved6": "v1.0's six Underserved scripts are the bottom 6",
}


# =============================================================================
# Data
# =============================================================================

def load_inputs() -> tuple[pd.DataFrame, pd.Series]:
    """Return (v1.0 baseline table, raw mean cosine distance per script).

    The baseline is recomputed with analysis.scoring.compute_servedness from the
    same inputs the tests use, then checked against the committed canonical CSV
    so the sweep is anchored to exactly what v1.0 publishes.
    """
    sim = pd.read_csv(SIMILARITY_CSV).set_index("script")
    base = compute_servedness(load_support(), sim["diversity_index"], load_exposure())

    canon = pd.read_csv(CANONICAL_CSV).set_index("script")
    drift = (base["servedness_score"] - canon.loc[base.index, "servedness_score"]).abs().max()
    if drift > 1e-9 or not (base["tier"] == canon.loc[base.index, "tier"]).all():
        raise RuntimeError(
            f"Recomputed v1.0 SSS does not match {CANONICAL_CSV.name} (max drift {drift:.2e}). "
            "Regenerate the canonical output before running the sweep."
        )
    return base, sim.loc[base.index, "mean_cosine_distance"]


def zscore(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std(ddof=0)


def standardized_terms(base: pd.DataFrame, mean_cos_dist: pd.Series) -> pd.DataFrame:
    """Log-scaled, z-scored inputs. Signs are applied in composite(), not here."""
    return pd.DataFrame({
        "support": zscore(np.log10(base["support"].astype(float))),
        "diversity": zscore(np.log10(mean_cos_dist.astype(float))),
        "demand": zscore(np.log10(base["exposure"].astype(float))),
    })


# =============================================================================
# Part A: is demand inert in the shipped formula?
# =============================================================================

def demand_exponent_check(base: pd.DataFrame, max_gamma: int = 500) -> dict:
    """Smallest integer gamma at which effective_choice / log10(exposure)**gamma
    reorders the scripts relative to gamma = 1 (the shipped formula)."""
    log_e = np.log10(base["exposure"].astype(float))
    ref = list((base["effective_choice"] / log_e).sort_values(kind="stable").index)
    first_change = None
    for g in range(2, max_gamma + 1):
        order = list((base["effective_choice"] / log_e ** g).sort_values(kind="stable").index)
        if order != ref:
            first_change = g
            break
    return {
        "log_demand_min": float(log_e.min()),
        "log_demand_max": float(log_e.max()),
        "log_demand_spread_pct": float((log_e.max() / log_e.min() - 1) * 100),
        "raw_demand_ratio": float(base["exposure"].max() / base["exposure"].min()),
        "raw_support_ratio": float(base["support"].max() / base["support"].min()),
        "first_reordering_gamma": first_change,
    }


# =============================================================================
# Part B: relative-contribution sweep over the weight simplex
# =============================================================================

def composite(Z: pd.DataFrame, ws: float, wd: float, we: float) -> pd.Series:
    """Higher = better served (more, more varied fonts per unit demand)."""
    return ws * Z["support"] + wd * Z["diversity"] - we * Z["demand"]


def simplex(step: float):
    """All (ws, wd, we) on the simplex at the given resolution, exact on the grid."""
    n = int(round(1 / step))
    if not np.isclose(n * step, 1.0):
        raise ValueError("--step must divide 1 evenly (e.g. 0.1, 0.05, 0.02)")
    for i in range(n + 1):
        for j in range(n + 1 - i):
            yield i / n, j / n, (n - i - j) / n


def evaluate(score: pd.Series, base: pd.DataFrame, underserved: set[str]) -> dict:
    order = list(score.sort_values(kind="stable").index)  # most underserved first
    ranks = score.rank(method="min").astype(int)          # 1 = most underserved
    return {
        "rho_vs_v1": spearman(score.tolist(), base.loc[score.index, "servedness_score"].tolist()),
        "indic_bottom4": set(order[:4]) == INDIC_FOUR,
        "latin_best": order[-1] == "Latin",
        "underserved6": set(order[:len(underserved)]) == underserved,
        "most_underserved": order[0],
        "order": " < ".join(order),
        **{f"rank_{s}": int(ranks[s]) for s in base.index},
    }


def run_sweep(base: pd.DataFrame, Z: pd.DataFrame, step: float) -> pd.DataFrame:
    underserved = set(base.index[base["tier"] == "Underserved"])
    rows = [
        {"w_support": ws, "w_diversity": wd, "w_demand": we,
         **evaluate(composite(Z, ws, wd, we), base, underserved)}
        for ws, wd, we in simplex(step)
    ]
    return pd.DataFrame(rows)


def summarize_by_demand(grid: pd.DataFrame) -> pd.DataFrame:
    """For each demand weight, the share of support/diversity splits where each claim holds."""
    g = grid.groupby(grid["w_demand"].round(6))
    out = g[list(CLAIMS)].mean()
    out["median_rho_vs_v1"] = g["rho_vs_v1"].median()
    out["n_splits"] = g.size()
    return out.reset_index()


def demand_free_edge(grid: pd.DataFrame) -> pd.DataFrame:
    edge = grid[np.isclose(grid["w_demand"], 0.0)].copy()
    edge["support_share"] = edge["w_support"]  # wd = 1 - ws on this edge
    return edge.sort_values("support_share").reset_index(drop=True)


def holding_interval(edge: pd.DataFrame, mask: pd.Series) -> tuple[float, float] | None:
    """Longest contiguous run of support_share values where mask is True."""
    best, cur = None, None
    for share, ok in zip(edge["support_share"], mask):
        if ok:
            cur = (cur[0], share) if cur else (share, share)
            if best is None or cur[1] - cur[0] > best[1] - best[0]:
                best = cur
        else:
            cur = None
    return best


# =============================================================================
# Report
# =============================================================================

def _fmt_interval(iv: tuple[float, float] | None) -> str:
    return "never" if iv is None else f"{iv[0]:.0%} to {iv[1]:.0%}"


def write_report(base, A, grid, by_demand, edge, step) -> str:
    all_hold = edge[list(CLAIMS)].all(axis=1)
    band = holding_interval(edge, all_hold)
    center = grid[np.isclose(grid["w_support"], 0.5) & np.isclose(grid["w_demand"], 0.0)]

    survives_majority = by_demand.loc[by_demand[list(CLAIMS)].min(axis=1) >= 0.5, "w_demand"]
    demand_ceiling = float(survives_majority.max()) if len(survives_majority) else 0.0
    dead_at = {
        c: (float(by_demand.loc[by_demand[c] == 0, "w_demand"].min())
            if (by_demand[c] == 0).any() else None)
        for c in CLAIMS
    }

    L = ["# Formula-weight sensitivity: Script Servedness Score\n",
         f"Generated by `analysis/formula_sweep.py` (step {step}, {len(grid)} weight points) "
         "from committed v1.0 data. The baseline was verified against "
         "`data/final/script_servedness.csv` before sweeping.\n"]

    # ---- headline
    L.append("## Headline\n")
    same_order = center["order"].iloc[0] == " < ".join(base.index)
    L.append(
        f"- **Support vs diversity trade-off: robust.** With demand left out, every v1.0 claim "
        f"holds for any support share from **{_fmt_interval(band)}** (rest on diversity). "
        + (f"An even 50/50 split gives the identical v1.0 ordering (Spearman "
           f"{center['rho_vs_v1'].iloc[0]:.3f}; below 1 only because v1.0 ties Tamil and "
           f"Bengali at zero)." if same_order else
           f"An even 50/50 split gives Spearman {center['rho_vs_v1'].iloc[0]:.3f} vs v1.0."))
    L.append(
        f"- **Demand in the shipped formula is inert.** log10(exposure) varies only "
        f"{A['log_demand_spread_pct']:.1f}% across the nine scripts, so raising the denominator "
        f"to a power changes nothing until gamma = **{A['first_reordering_gamma']}**. "
        "v1.0 is, in practice, a supply-side score (support x diversity).")
    L.append(
        f"- **Giving demand real weight breaks the conclusions.** Once inputs are standardized, "
        f"all three claims survive a majority of support/diversity splits only up to a demand "
        f"weight of about **{demand_ceiling:.0%}**. The six-script Underserved set never survives "
        f"past {dead_at['underserved6']:.0%}, the Indic-four claim dies at "
        f"{dead_at['indic_bottom4']:.0%}, and Latin stops being best served at "
        f"{dead_at['latin_best']:.0%}.\n")

    # ---- part A
    L.append("## Part A: demand exponent in the shipped formula\n")
    L.append("| Quantity | Value |\n|---|---|")
    L.append(f"| log10(exposure) range | {A['log_demand_min']:.3f} to {A['log_demand_max']:.3f} |")
    L.append(f"| Spread of the demand term | {A['log_demand_spread_pct']:.2f}% |")
    L.append(f"| Raw demand max/min | {A['raw_demand_ratio']:.2f}x |")
    L.append(f"| Raw support max/min (for scale) | {A['raw_support_ratio']:.0f}x |")
    L.append(f"| Smallest gamma that reorders any pair | {A['first_reordering_gamma']} |\n")
    L.append("Raw demand itself only spans 1.25x across scripts, consistent with the bundler-font "
             "confound documented in `exposure_research/DEMAND_PROVENANCE.md`. Any transform that "
             "stretches it enough to matter (min-max, z-score) is amplifying a mostly-confounded "
             "signal.\n")

    # ---- part B, edge
    L.append("## Part B1: support vs diversity (demand weight = 0)\n")
    L.append("Rank 1 = most underserved.\n")
    scripts = list(base.index)
    L.append("| Support share | " + " | ".join(scripts) + " | All claims hold |")
    L.append("|---|" + "---|" * (len(scripts) + 1))
    for _, r in edge.iterrows():
        if round(r["support_share"] * 100) % 10:
            continue  # table at 10% resolution; full grid is in the CSV
        ranks = " | ".join(str(int(r[f"rank_{s}"])) for s in scripts)
        ok = "yes" if all(r[c] for c in CLAIMS) else "no"
        L.append(f"| {r['support_share']:.0%} | {ranks} | {ok} |")
    L.append("")
    L.append("Where each Indic script stays in the bottom four (support share, demand = 0):\n")
    for s in ["Tamil", "Bengali", "Devanagari", "Telugu"]:
        iv = holding_interval(edge, edge[f"rank_{s}"] <= 4)
        L.append(f"- {s}: {_fmt_interval(iv)} (rank range {edge[f'rank_{s}'].min()} to "
                 f"{edge[f'rank_{s}'].max()})")
    L.append("\nThe two edges fail for different reasons. Diversity-only lets Katakana (highest "
             "visual variety, 67 families) overtake Latin. Support-heavy (80%+) drops Han, which "
             "has only 26 open-source families, into the bottom four ahead of Devanagari (62).\n")

    # ---- part B, demand
    L.append("## Part B2: claim survival as demand weight rises\n")
    L.append("Share of support/diversity splits (at that demand weight) where each claim holds.\n")
    L.append("| Demand weight | Indic bottom 4 | Latin best | Underserved 6 | Median Spearman vs v1.0 |")
    L.append("|---|---|---|---|---|")
    for _, r in by_demand.iterrows():
        pct = round(r["w_demand"] * 100)
        if pct % 10 and pct not in (5, 15, 25):
            continue  # show 10% steps, plus 5/15/25 where the claims start to break
        L.append(f"| {r['w_demand']:.0%} | {r['indic_bottom4']:.0%} | {r['latin_best']:.0%} | "
                 f"{r['underserved6']:.0%} | {r['median_rho_vs_v1']:.2f} |")
    L.append("")

    # ---- interpretation
    L.append("## What this means for the paper\n")
    L.append("1. The robustness claim that is defensible: the tiering is stable across a wide "
             "range of support/diversity trade-offs. That is the sentence to write.")
    L.append("2. The README's \"real font choice per unit of web demand\" framing overstates what "
             "the shipped score does. Either drop demand from the score and report it as a separate "
             "axis (as complexity already is), or keep it and say plainly that its numerical "
             "effect on the ranking is nil.")
    L.append("3. Demand cannot be given meaningful weight until the `subset=` upgrade lands. At "
             "current quality, any weighting that lets it matter mostly re-sorts scripts by a "
             "bundler-font artifact.")
    L.append("4. Han's tier is the least stable result in the set. It depends on whether you "
             "reward its high visual variety or penalize its low family count, and it has the "
             "lowest measured demand.\n")

    L.append("## Method notes\n")
    L.append("- Inputs: Google Fonts family counts, ViT-B/16 mean cosine distance (raw, not "
             "min-max), HTTP Archive font-request volume; all log10 then z-scored (population sd).")
    L.append("- Because ranks are invariant to rescaling all weights together, the simplex covers "
             "every non-negative weighting up to scale.")
    L.append("- This tests the formula, not measurement error. For input noise see "
             "`analysis/sensitivity.py` (one-at-a-time shocks) on the Sensitivity-Analysis branch.")
    return "\n".join(L) + "\n"


# =============================================================================
# Figure
# =============================================================================

PALETTE = {  # reference data-viz palette, slots 1-3 (validated all-pairs, light mode)
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "context": "#b9b7b0", "surface": "#fcfcfb", "grid": "#e9e8e4",
    "ink": "#0b0b0b", "ink2": "#52514e", "band": "#eef4fc",
}


def plot(edge: pd.DataFrame, by_demand: pd.DataFrame, band, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": PALETTE["grid"], "axes.labelcolor": PALETTE["ink2"],
        "xtick.color": PALETTE["ink2"], "ytick.color": PALETTE["ink2"],
    })
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200,
                                 gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.42})
    fig.patch.set_facecolor(PALETTE["surface"])

    # --- Panel 1: rank stability along the demand-free edge (bump chart) -----
    x = edge["support_share"].to_numpy() * 100
    scripts = [c.removeprefix("rank_") for c in edge.columns if c.startswith("rank_")]
    if band:
        a1.axvspan(band[0] * 100, band[1] * 100, color=PALETTE["band"], zorder=0, lw=0)
        a1.text((band[0] + band[1]) * 50, 0.42, "all v1.0 claims hold",
                ha="center", va="center", fontsize=8.5, color=PALETTE["ink2"])
    for s in sorted(scripts, key=lambda s: (s in INDIC_FOUR or s == "Latin")):
        y = edge[f"rank_{s}"].to_numpy()
        color = (PALETTE["blue"] if s in INDIC_FOUR else
                 PALETTE["orange"] if s == "Latin" else PALETTE["context"])
        lw = 2.0 if (s in INDIC_FOUR or s == "Latin") else 1.5
        a1.plot(x, y, color=color, lw=lw, solid_capstyle="round", zorder=3)
        # direct labels carry the rank too, so no tick labels are needed
        a1.text(x[-1] + 2.5, y[-1], f"{y[-1]}  {s}", va="center", fontsize=9, color=PALETTE["ink"])
        a1.text(x[0] - 2.5, y[0], f"{s}  {y[0]}", va="center", ha="right", fontsize=9,
                color=PALETTE["ink"])
    a1.set_ylim(9.5, 0.1)
    a1.set_yticks(range(1, 10))
    a1.set_yticklabels([])
    a1.set_xlim(0, 100)
    a1.set_xticks([0, 25, 50, 75, 100])
    a1.set_xticklabels(["0%\nall diversity", "25%", "50%", "75%", "100%\nall support"])
    a1.set_xlabel("Weight on support (rest on diversity), demand weight = 0", labelpad=8)
    a1.set_title("Rank stability: support vs diversity trade-off\n", loc="left",
                 fontsize=11, color=PALETTE["ink"])
    a1.text(0, 1.035, "Rank 1 = most underserved. Blue = Indic four, orange = Latin.",
            transform=a1.transAxes, fontsize=8.5, color=PALETTE["ink2"])
    a1.grid(axis="y", color=PALETTE["grid"], lw=1)
    a1.tick_params(length=0)
    for sp in a1.spines.values():
        sp.set_visible(False)

    # --- Panel 2: claim survival vs demand weight ------------------------------
    # Lines tangle where they matter, so identity is carried by a legend placed in
    # the empty upper-right; the report's Part B2 table is the full table view.
    xd = by_demand["w_demand"].to_numpy() * 100
    a2.axvline(0, color=PALETTE["ink2"], lw=1, ls="-", zorder=2)
    a2.text(1.5, 97, "shipped v1.0 (demand weight ~0)", fontsize=8.5,
            color=PALETTE["ink2"], va="center")
    lines = [("indic_bottom4", "Indic four in bottom 4", PALETTE["blue"]),
             ("latin_best", "Latin best served", PALETTE["orange"]),
             ("underserved6", "Underserved six intact", PALETTE["aqua"])]
    for key, label, color in lines:
        y = by_demand[key].to_numpy() * 100
        a2.plot(xd, y, color=color, lw=2, label=label, zorder=3, solid_capstyle="round")
        a2.scatter(xd, y, s=14, color=color, zorder=4, edgecolors=PALETTE["surface"], linewidths=0.8)
    a2.set_xlim(-1, 101)
    a2.set_ylim(-3, 103)
    a2.set_xticks([0, 20, 40, 60, 80, 100])
    a2.set_xticklabels([f"{v}%" for v in [0, 20, 40, 60, 80, 100]])
    a2.set_yticks([0, 25, 50, 75, 100])
    a2.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    a2.set_xlabel("Weight on demand (standardized)", labelpad=8)
    a2.set_ylabel("Support/diversity splits where claim holds")
    a2.set_title("Claims survive only while demand weight stays small\n", loc="left",
                 fontsize=11, color=PALETTE["ink"])
    a2.text(0, 1.035, "Each point averages over every support/diversity split at that demand weight.",
            transform=a2.transAxes, fontsize=8.5, color=PALETTE["ink2"])
    a2.grid(color=PALETTE["grid"], lw=1)
    a2.tick_params(length=0)
    leg = a2.legend(loc="center right", frameon=False, fontsize=9, bbox_to_anchor=(1.0, 0.68),
                    handlelength=1.8)
    for t in leg.get_texts():
        t.set_color(PALETTE["ink"])
    for sp in a2.spines.values():
        sp.set_visible(False)
    for ax in (a1, a2):
        ax.set_facecolor(PALETTE["surface"])

    fig.text(0.01, -0.04, "Source: TheScriptGap v1.0 committed data. Inputs log10 + z-scored; "
             "ViT-B/16 diversity; Google Fonts support; HTTP Archive demand.",
             fontsize=7.5, color=PALETTE["ink2"])
    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("Usage:")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", type=float, default=0.05,
                   help="Simplex grid resolution; must divide 1 evenly (default 0.05).")
    p.add_argument("--no-plot", action="store_true", help="Skip the PNG figure.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    base, mcd = load_inputs()
    Z = standardized_terms(base, mcd)
    A = demand_exponent_check(base)
    grid = run_sweep(base, Z, args.step)
    by_demand = summarize_by_demand(grid)
    edge = demand_free_edge(grid)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT_DIR / "formula_sweep_grid.csv", index=False)
    by_demand.to_csv(OUT_DIR / "formula_sweep_by_demand.csv", index=False)
    report = write_report(base, A, grid, by_demand, edge, args.step)
    (OUT_DIR / "formula_sweep.md").write_text(report, encoding="utf-8")
    if not args.no_plot:
        band = holding_interval(edge, edge[list(CLAIMS)].all(axis=1))
        plot(edge, by_demand, band, OUT_DIR / "formula_sweep.png")

    print(report)
    print(f"Saved -> {OUT_DIR}/formula_sweep_[grid|by_demand].csv, formula_sweep.md"
          + ("" if args.no_plot else ", formula_sweep.png"))


if __name__ == "__main__":
    main()
