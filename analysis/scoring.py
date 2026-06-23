"""
Canonical Script Servedness Score (SSS) — single source of truth.

    SSS = log_support_norm − similarity_norm        (similarity = 1 − diversity)

Two co-equal, trustworthy signals: open-source font support and visual diversity.
Demand and complexity are deliberately excluded (see DEMAND_PROVENANCE.md and the
README). `data_viz/generate_heatmap.py` computes the same score for the heatmap;
`tests/test_servedness.py` asserts the committed canonical output matches this
function, so the two cannot silently drift.
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


def compute_servedness(support: pd.Series, diversity: pd.Series) -> pd.DataFrame:
    """Per-script servedness from font support + diversity.

    Args:
        support:   distinct font-family count per script (raw).
        diversity: diversity index per script in [0, 1] (higher = more variety).
    Both indexed by canonical script name; the score is computed over their
    intersection and normalized within that set.

    Returns a DataFrame indexed by script with columns:
        support, diversity, log_support_norm, similarity_norm,
        servedness_score (sss_norm in [0, 1]), tier
    sorted most- to least-underserved.
    """
    scripts = [s for s in diversity.index if s in support.index]
    df = pd.DataFrame(index=pd.Index(scripts, name="script"))
    df["support"] = support[scripts]
    df["diversity"] = diversity[scripts]
    df["log_support_norm"] = minmax(np.log10(df["support"]))
    df["similarity_norm"] = minmax(1.0 - df["diversity"])
    sss = df["log_support_norm"] - df["similarity_norm"]
    df["servedness_score"] = minmax(sss)
    df["tier"] = assign_tier(df["servedness_score"])
    return df.sort_values("servedness_score")
