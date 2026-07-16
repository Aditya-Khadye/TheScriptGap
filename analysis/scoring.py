"""
Canonical Script Servedness Score (SSS) — single source of truth.

    effective_choice = support_norm * (1 - similarity_norm)     # quantity * variety
    SSS = effective_choice / log10(exposure)                    # choice per (log) demand

Real font choice — how many *genuinely different* fonts a script has (support
weighted by visual variety) — relative to how much the script is actually used.
A high-demand script with little real choice scores lowest (= most underserved),
which is exactly what a prioritization score should surface.

This is the project's original gap-ratio, with one fix: the denominator divides
by log-scaled exposure (the methodology always specified "exposure log-scaled")
instead of the earlier `(exposure_norm + 0.1)`, which needed an arbitrary 0.1
floor because the [0,1]-normalized denominator could hit zero. The ranking is
unchanged (Spearman 0.98-1.0 vs the original); the magic constant is gone.

Demand (exposure) is the weakest input — see exposure_research/DEMAND_PROVENANCE.md
for its confound and the `subset=` upgrade path. Complexity ("engineering cost")
is reported separately as a prioritization signal and is NOT in the SSS.

`data_viz/generate_heatmap.py` computes the same score for the heatmap;
`tests/test_servedness.py` asserts the committed output matches this function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIER_THRESHOLDS = (0.3, 0.6)  # <0.3 Underserved, <0.6 Moderately served, else Well served


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


def assign_tier(sss_norm: pd.Series) -> pd.Series:
    lo, hi = TIER_THRESHOLDS
    return sss_norm.apply(
        lambda v: "Well served" if v > hi else "Moderately served" if v > lo else "Underserved"
    )


def compute_servedness(support: pd.Series, diversity: pd.Series,
                       exposure: pd.Series) -> pd.DataFrame:
    """Per-script servedness from font support, diversity, and web demand.

    Args:
        support:   distinct font-family count per script (raw).
        diversity: diversity index per script in [0, 1] (higher = more variety).
        exposure:  per-script web font-request volume (raw); demand.
    All indexed by canonical script name; the score is computed over their
    common scripts and the support/similarity terms normalized within that set.

    Returns a DataFrame indexed by script with columns:
        support, diversity, exposure, support_norm, similarity_norm,
        effective_choice, servedness_score (sss_norm in [0, 1]), tier
    sorted most- to least-underserved.
    """
    scripts = [s for s in diversity.index if s in support.index and s in exposure.index]
    df = pd.DataFrame(index=pd.Index(scripts, name="script"))
    df["support"] = support[scripts]
    df["diversity"] = diversity[scripts]
    df["exposure"] = exposure[scripts]
    df["support_norm"] = minmax(np.log10(df["support"]))
    df["similarity_norm"] = minmax(1.0 - df["diversity"])
    df["effective_choice"] = df["support_norm"] * (1.0 - df["similarity_norm"])
    sss = df["effective_choice"] / np.log10(df["exposure"])
    df["servedness_score"] = minmax(sss)
    df["tier"] = assign_tier(df["servedness_score"])
    return df.sort_values("servedness_score")
