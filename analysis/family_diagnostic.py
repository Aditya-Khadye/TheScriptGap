"""
Family-bundling diagnostic for the Script Diversity Index (ViT-B/16, 100-glyph).

Question: the diversity pipeline samples font FILES (rglob over .ttf/.otf), while
the support term counts Google Fonts FAMILIES. Static weight/style files of one
family (HindMadurai-Bold, -Light, ...) are near-duplicates by construction, so
file-level mean pairwise distance mixes "distance between designs" with
"distance between weights of one design". Does the script ranking survive when
the unit of analysis is the family?

Inputs (committed in TheScriptGap @ v1.0):
  similarity_research/diversity_research/vit_outputs_100/font_similarity_pairs/*.csv
      all within-script font-file pairs with ViT cosine similarity
  data/similarity/similarity_results.csv      committed diversity (incl. Latin)
  data/final/script_servedness.csv            committed support + exposure + SSS

Measures per script:
  file_level      mean cosine distance over all file pairs (reproduces committed)
  between_family  mean over pairs whose files belong to different families
  family_rep      one representative file per family, mean over rep pairs
  design_rep      as family_rep after also merging UI / SC sibling families
                  (heuristic sensitivity check, not a primary result)
  within_family   mean distance between files of the SAME family (calibration:
                  what "same design, different weight/style" looks like)

Uncertainty: cluster bootstrap over families (B = 5000, seed 7). Each replicate
resamples a script's families with replacement, keeps each drawn family's
representative, and averages distances over pairs from different original
families (self-pairs of a duplicated family are excluded so they cannot drag
the mean to zero). Replicates are drawn jointly for all scripts so per-replicate
rankings can be summarized.

Usage (from the repo root):
    uv run python -m analysis.family_diagnostic          # writes data/diagnostics/
"""

from __future__ import annotations

import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

STYLE_WORDS = (
    "Hairline|Thin|ExtraLight|UltraLight|Light|Book|Regular|Medium|SemiBold|"
    "DemiBold|Bold|ExtraBold|UltraBold|Black|Heavy|Italic|Oblique"
)
STYLE_RE = re.compile(rf"^(?:{STYLE_WORDS})+$")
B = 5000
SEED = 7


def family_of(file_stem: str) -> str:
    """Google Fonts family key from a font file stem.

    'HindMadurai-Bold' -> 'HindMadurai'; 'NotoSerifTamil-Italic[wdth,wght]' ->
    'NotoSerifTamil'; 'Lohit-Tamil' -> 'Lohit-Tamil' (suffix is not a style).
    """
    base = file_stem.split("[")[0]
    if "-" in base:
        head, tail = base.rsplit("-", 1)
        if STYLE_RE.match(tail):
            return head
    return base


def design_of(family: str) -> str:
    """Heuristic design-lineage key: merge UI and small-caps sibling families."""
    return re.sub(r"(UI|SC)$", "", family)


def pick_rep(files: list[str]) -> str:
    """Representative file: variable roman > Regular > non-italic > first."""
    def score(f: str) -> tuple:
        is_var = "[" in f
        is_italic = "Italic" in f or "Oblique" in f
        is_regular = f.split("[")[0].endswith("-Regular")
        return (is_italic, not is_var, not is_regular, f)
    return sorted(files, key=score)[0]


def load_pairs(path: Path) -> tuple[pd.DataFrame, list[str]]:
    d = pd.read_csv(path)
    d["dist"] = 1.0 - d["similarity"]
    files = sorted(set(d.font_name1) | set(d.font_name2))
    return d, files


def dist_matrix(d: pd.DataFrame, files: list[str]) -> np.ndarray:
    idx = {f: i for i, f in enumerate(files)}
    m = np.full((len(files), len(files)), np.nan)
    i = d.font_name1.map(idx).to_numpy()
    j = d.font_name2.map(idx).to_numpy()
    m[i, j] = d.dist.to_numpy()
    m[j, i] = d.dist.to_numpy()
    np.fill_diagonal(m, 0.0)
    assert not np.isnan(m).any(), "pairs file is not a complete pairwise set"
    return m


def mean_offdiag(m: np.ndarray, members: list[int], groups: list) -> float:
    """Mean distance over pairs drawn from DIFFERENT groups."""
    vals = [m[a, b] for (a, ga), (b, gb) in combinations(zip(members, groups), 2) if ga != gb]
    return float(np.mean(vals)) if vals else np.nan


def analyse_script(name: str, d: pd.DataFrame, files: list[str]) -> tuple[dict, dict]:
    m = dist_matrix(d, files)
    fam = [family_of(f) for f in files]
    fam_to_files: dict[str, list[int]] = {}
    for k, f in enumerate(fam):
        fam_to_files.setdefault(f, []).append(k)

    iu = np.triu_indices(len(files), 1)
    same = np.array([fam[a] == fam[b] for a, b in zip(*iu)])
    all_d = m[iu]

    reps = {f: files.index(pick_rep([files[k] for k in ks])) for f, ks in fam_to_files.items()}
    rep_idx = list(reps.values())
    rep_fams = list(reps.keys())

    designs = [design_of(f) for f in rep_fams]
    # design_rep: one rep per design; first family (sorted) within a design wins
    seen: dict[str, int] = {}
    for f, r, g in sorted(zip(rep_fams, rep_idx, designs)):
        seen.setdefault(g, r)

    row = {
        "script": name,
        "files": len(files),
        "families": len(fam_to_files),
        "files_per_family": len(files) / len(fam_to_files),
        "within_family_pair_share": float(same.mean()),
        "file_level": float(all_d.mean()),
        "between_family": float(all_d[~same].mean()),
        "within_family": float(all_d[same].mean()) if same.any() else np.nan,
        "family_rep": mean_offdiag(m, rep_idx, rep_fams),
        "design_rep": mean_offdiag(m, list(seen.values()), list(seen.keys())),
        "designs": len(seen),
    }
    boot_state = {"m": m, "rep_idx": np.array(rep_idx)}
    families_df = pd.DataFrame({
        "script": name,
        "family": rep_fams,
        "n_files": [len(fam_to_files[f]) for f in rep_fams],
        "representative": [files[r] for r in rep_idx],
    })
    return row | {"_families": families_df}, boot_state


def bootstrap(states: dict[str, dict]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    scripts = list(states)
    out = np.empty((B, len(scripts)))
    for b in range(B):
        for s_i, s in enumerate(scripts):
            st = states[s]
            n = len(st["rep_idx"])
            draw = rng.integers(0, n, n)                 # family positions
            r = st["rep_idx"][draw]
            sub = st["m"][np.ix_(r, r)]
            diff = draw[:, None] != draw[None, :]        # exclude same-family pairs
            iu = np.triu_indices(n, 1)
            mask = diff[iu]
            out[b, s_i] = sub[iu][mask].mean()
    return pd.DataFrame(out, columns=scripts)


def minmax(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min())


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def main(repo: Path, out: Path | None = None, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the diagnostic. Returns (per-script table, SSS comparison, summary).

    Writes CSVs to `out` when given; pass out=None to compute only (tests).
    """
    pair_dir = repo / "similarity_research/diversity_research/vit_outputs_100/font_similarity_pairs"
    rows, fams, states = [], [], {}
    for p in sorted(pair_dir.glob("font_similarity_pairs_*.csv")):
        name = p.stem.replace("font_similarity_pairs_", "")
        d, files = load_pairs(p)
        row, st = analyse_script(name, d, files)
        fams.append(row.pop("_families"))
        rows.append(row)
        states[name] = st
    res = pd.DataFrame(rows).set_index("script")

    committed = pd.read_csv(repo / "data/similarity/similarity_results.csv").set_index("script")
    res["committed_mean_dist"] = committed.loc[res.index, "mean_cosine_distance"]
    res["reproduces_committed"] = np.isclose(res.file_level, res.committed_mean_dist, atol=1e-6)

    for col in ["file_level", "family_rep", "between_family", "design_rep"]:
        res[f"div_{col}"] = minmax(res[col])
        res[f"rank_{col}"] = res[col].rank().astype(int)   # 1 = most homogeneous

    boot = bootstrap(states)
    ranks = boot.rank(axis=1)
    res["family_rep_ci_lo"] = boot.quantile(0.025)
    res["family_rep_ci_hi"] = boot.quantile(0.975)
    res["rank_ci_lo"] = ranks.quantile(0.025).astype(int)
    res["rank_ci_hi"] = ranks.quantile(0.975).astype(int)
    res["p_most_homogeneous"] = (ranks == 1).mean()

    # SSS under family-level diversity, Latin held at its committed value
    canon = pd.read_csv(repo / "data/final/script_servedness.csv").set_index("script")
    sys.path.insert(0, str(repo))
    from analysis.scoring import compute_servedness  # noqa: E402

    def sss_with(div_nonlatin_raw: pd.Series) -> pd.DataFrame:
        lo, hi = div_nonlatin_raw.min(), div_nonlatin_raw.max()
        div = (div_nonlatin_raw - lo) / (hi - lo)
        latin_raw = committed.loc["Latin", "mean_cosine_distance"]
        div.loc["Latin"] = (latin_raw - lo) / (hi - lo)
        return compute_servedness(canon["support_gf_families"], div, canon["exposure_demand"])

    sss_file = sss_with(res["file_level"])
    sss_fam = sss_with(res["family_rep"])
    sss_cmp = pd.DataFrame({
        "sss_committed": canon["servedness_score"],
        "sss_recomputed_file": sss_file["servedness_score"],
        "tier_committed": canon["tier"],
        "sss_family": sss_fam["servedness_score"],
        "tier_family": sss_fam["tier"],
    }).sort_values("sss_family")
    nl = [s for s in sss_cmp.index if s != "Latin"]

    summary = {
        "rho_diversity_file_vs_family_rep": spearman(res.file_level, res.family_rep),
        "rho_diversity_file_vs_between_family": spearman(res.file_level, res.between_family),
        "rho_diversity_file_vs_design_rep": spearman(res.file_level, res.design_rep),
        "rho_sss_committed_vs_family_nonlatin": spearman(sss_cmp.loc[nl, "sss_committed"], sss_cmp.loc[nl, "sss_family"]),
        "tiers_identical": int((sss_cmp.tier_committed == sss_cmp.tier_family).sum()),
        "max_abs_sss_reproduction_error": float((sss_cmp.sss_committed - sss_cmp.sss_recomputed_file).abs().max()),
    }

    # Calibration: how far apart are different designs relative to the spread
    # between weights/styles of ONE design, within the same script and encoder?
    res["within_family_pairs"] = (res.within_family_pair_share * res.files * (res.files - 1) / 2).round().astype(int)
    res["design_to_weight_ratio"] = res.family_rep / res.within_family

    # Latin comparability: Latin was embedded from 10 capitals; non-Latin from
    # ~90-100 glyphs. Place Latin on the 10-glyph non-Latin scale instead.
    ten = pd.read_csv(repo / "similarity_research/diversity_research/vit_outputs/diversity_index_summary.csv").set_index("script")
    latin_raw = committed.loc["Latin", "mean_cosine_distance"]
    lo10, hi10 = ten.mean_cosine_distance.min(), ten.mean_cosine_distance.max()
    summary["latin_div_on_100glyph_scale"] = float(committed.loc["Latin", "diversity_index"])
    summary["latin_div_on_10glyph_scale"] = float((latin_raw - lo10) / (hi10 - lo10))
    summary["median_ratio_10glyph_over_100glyph_dist"] = float(
        (ten.mean_cosine_distance / res.file_level.reindex(ten.index)).median())

    # Population mismatch: families in the diversity sample vs families counted
    # by the support term (Google Fonts API subset tags).
    res["support_term_families"] = canon.loc[res.index, "support_gf_families"]
    res["total_files_in_corpus"] = committed.loc[res.index, "total_fonts"]

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        res.to_csv(out / "family_diagnostic_by_script.csv")
        pd.concat(fams).to_csv(out / "family_representatives.csv", index=False)
        boot.to_csv(out / "bootstrap_family_rep.csv", index=False)
        sss_cmp.to_csv(out / "sss_file_vs_family.csv")
        pd.Series(summary).to_csv(out / "family_diagnostic_summary.csv", header=["value"])

    if verbose:
        pd.set_option("display.width", 200, "display.max_columns", 40)
        print(res.drop(columns=[c for c in res.columns if c.startswith("div_")]).round(4).sort_values("family_rep"))
        print()
        print(sss_cmp.round(3))
        print()
        for k, v in summary.items():
            print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
    return res, sss_cmp, summary


if __name__ == "__main__":
    REPO = Path(__file__).resolve().parents[1]
    main(REPO, REPO / "data" / "diagnostics")
