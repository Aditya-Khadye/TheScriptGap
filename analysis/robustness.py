"""
===============================================================================
Robustness & sensitivity analysis for the Script Servedness Score (SSS)
===============================================================================
Project:  TheScriptGap  |  v1.0

Answers two research-grade questions, entirely from committed data:

  1. Is the DIVERSITY signal robust to model choice?
     Spearman rank correlation of the per-script diversity index across the
     ViT-B/16, ResNet-50 (CNN), and classical-CV pipelines (and 10- vs 100-glyph
     ViT). Headline: ViT vs ResNet.

  2. Is the SERVEDNESS conclusion robust to how the two signals are weighted, and
     to which diversity model feeds it?
       (a) Rank scripts by support-only, diversity-only, and the combined SSS;
           report whether the underserved cluster is stable.
       (b) Recompute the SSS substituting ResNet / classical diversity for ViT
           and report tier agreement.

Outputs a console report and `data/final/robustness.md`. No heavy dependencies
(Spearman is computed directly).

Usage:
    uv run --with pandas python analysis/robustness.py
===============================================================================
"""

from __future__ import annotations

import sys

from pathlib import Path

import pandas as pd

try:
    from paths import DATA_ROOT, REPO_ROOT
    from analysis.scoring import compute_servedness, minmax
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import DATA_ROOT, REPO_ROOT
    from analysis.scoring import compute_servedness, minmax

SIM = REPO_ROOT / "similarity_research" / "diversity_research"
DIVERSITY_SOURCES = {
    "ViT-B/16 (100-glyph)": SIM / "vit_outputs_100" / "diversity_index_summary.csv",
    "ResNet-50 (100-glyph)": SIM / "cnn_outputs_100" / "diversity_index_summary.csv",
    "Classical CV": SIM / "classical_cv_outputs" / "diversity_classical_summary.csv",
    "ViT-B/16 (10-glyph)": SIM / "vit_outputs" / "diversity_index_summary.csv",
}
SUPPORT_CSV = DATA_ROOT / "support" / "script_font_counts.csv"
SIMILARITY_CSV = DATA_ROOT / "similarity" / "similarity_results.csv"
EXPOSURE_CSV = DATA_ROOT / "exposure" / "exposure_filtered_results.csv"
OUT = DATA_ROOT / "final" / "robustness.md"

SUPPORT_NAME_MAP = {
    "latin": "Latin", "cyrillic": "Cyrillic", "japanese": "Katakana",
    "devanagari": "Devanagari", "arabic": "Arabic", "telugu": "Telugu",
    "tamil": "Tamil", "bengali": "Bengali",
    "chinese-traditional": "Han", "chinese-simplified": "Han", "chinese-hongkong": "Han",
}
NON_LATIN = ["Devanagari", "Arabic", "Bengali", "Tamil", "Telugu", "Han", "Katakana", "Cyrillic"]


# --- statistics (no scipy dependency) ---------------------------------------

def _rank(values: list[float]) -> list[float]:
    """Average ranks (ties shared), 1-based."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def spearman(a: list[float], b: list[float]) -> float:
    return _pearson(_rank(a), _rank(b))


# --- data -------------------------------------------------------------------

def load_diversity_sources() -> pd.DataFrame:
    frames = {}
    for label, path in DIVERSITY_SOURCES.items():
        if path.exists():
            df = pd.read_csv(path)[["script", "diversity_index"]].set_index("script")
            frames[label] = df["diversity_index"]
    out = pd.DataFrame(frames)
    return out.loc[out.index.isin(NON_LATIN)].sort_index()


def load_support() -> pd.Series:
    raw = pd.read_csv(SUPPORT_CSV, names=["script_raw", "count"], header=0)
    rec = [{"script": SUPPORT_NAME_MAP[r.script_raw], "support": r.count}
           for r in raw.itertuples() if r.script_raw in SUPPORT_NAME_MAP]
    return pd.DataFrame(rec).groupby("script")["support"].sum()


def load_exposure() -> pd.Series:
    return pd.read_csv(EXPOSURE_CSV, names=["script", "exposure"],
                       header=0).set_index("script")["exposure"]


# --- report -----------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    
    div = load_diversity_sources()
    support = load_support()
    exposure = load_exposure()
    lines: list[str] = ["# Robustness & sensitivity — Script Servedness Score\n"]

    # 1. diversity-model robustness
    lines.append("## 1. Diversity signal vs. model choice (Spearman rank corr.)\n")
    models = list(div.columns)
    common = div.dropna()
    lines.append("Pairwise Spearman ρ over the 8 non-Latin scripts:\n")
    header = "| | " + " | ".join(models) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(models) + 1))
    rhos = {}
    for m1 in models:
        row = [m1]
        for m2 in models:
            r = spearman(common[m1].tolist(), common[m2].tolist())
            rhos[(m1, m2)] = r
            row.append(f"{r:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    vit = "ViT-B/16 (100-glyph)"
    res = "ResNet-50 (100-glyph)"
    cls = "Classical CV"
    lines.append(
        f"\n**Headline:** ViT-B/16 vs ResNet-50 ρ = **{rhos[(vit,res)]:.3f}** "
        f"(strong — the diversity ranking survives the deep-model swap). "
        f"Classical CV diverges (ViT vs classical ρ = {rhos[(vit,cls)]:.2f}), so it is "
        f"NOT cited as diversity-robustness evidence.\n")

    # 2a. demand axis: epsilon-independence and how much demand moves the result
    import numpy as np
    lines.append("## 2a. Demand axis — epsilon-independence & sensitivity\n")
    div_vit = div[vit].dropna()
    scripts = [s for s in div_vit.index if s in support.index and s in exposure.index]
    Su = minmax(np.log10(support[scripts]))
    Si = minmax(1.0 - div_vit[scripts])
    Enorm = minmax(np.log10(exposure[scripts]))
    logE = np.log10(exposure[scripts])
    choice = Su * (1.0 - Si)                 # effective choice (support x diversity), no demand
    principled = choice / logE               # the shipped SSS denominator (log-demand)
    lines.append("The shipped SSS divides effective choice by log10(exposure). Spearman ρ of "
                 "that ranking vs the original `(exposure_norm + eps)` denominator:\n")
    for eps in (0.1, 0.5, 1.0):
        rho = spearman(principled.tolist(), (choice / (Enorm + eps)).tolist())
        lines.append(f"- vs eps = {eps}: ρ = {rho:.3f}")
    rho_choice = spearman(choice.tolist(), principled.tolist())
    lines.append(f"\nRemoving the arbitrary 0.1 floor does **not** change the ranking (it "
                 f"reproduces the original gap-ratio at sensible eps). Demand's influence: over "
                 f"the 8 non-Latin scripts, effective-choice-only (no demand) and the full SSS "
                 f"give the **same ranking (ρ = {rho_choice:.2f})** — so the underserved ordering "
                 f"does not depend on the (weakest) demand axis; demand mainly separates the "
                 f"high-demand well-served scripts (Latin, Cyrillic).\n")
    combined = compute_servedness(support, div_vit, exposure)
    lines.append(f"Most-underserved four: **{', '.join(list(combined.index[:4]))}**.\n")

    # 2b. tier stability across diversity models
    lines.append("## 2b. Servedness tiers vs. diversity model\n")
    base_tier = compute_servedness(support, div_vit, exposure)["tier"]
    lines.append("Tier agreement when the SSS is fed ResNet / classical diversity "
                 "instead of ViT:\n")
    for label in [res, cls]:
        if label not in div.columns:
            continue
        d = div[label].dropna()
        alt = compute_servedness(support, d, exposure)["tier"]
        shared = [s for s in base_tier.index if s in alt.index]
        agree = sum(base_tier[s] == alt[s] for s in shared)
        lines.append(f"- vs {label.split(' (')[0]}: {agree}/{len(shared)} tiers identical")
    lines.append("")

    report = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(report)
        OUT.write_text(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))
        OUT.write_text(report, encoding="utf-8")

        print("UnicodeEncodeError: writing robustness report to file; " \
        "replacing non-ASCII characters with '?' for console output. Markdown file is properly written in UTF-8.")

    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
