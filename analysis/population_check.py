"""
Population-alignment check for the Script Diversity Index.

The support term counts Google Fonts families tagged with a script's subset
(Google Fonts API / METADATA.pb `subsets`). The diversity pipeline admits any
font file with >= 10 codepoints in the script's Unicode block(s). This script
maps every sampled file to its Google Fonts family and asks:

  1. What share of the sampled families are NOT tagged with the script's subset
     (i.e. fonts whose own script is something else: Grantha, Chakma, Korean...)?
  2. Does the ranking hold when diversity is computed only over subset-tagged
     families (one representative each), i.e. the same population as support?

Family metadata comes from google/fonts HEAD (METADATA.pb only), indexed by
gf_index.py. The paper's corpus is a March 2026 clone, so a handful of families
added since then are absent from the sample; subset tags of existing families
are stable, which is all this check relies on.

Usage (from the repo root; gf_index.csv is committed in data/diagnostics/):
    uv run python -m analysis.population_check
To rebuild the index from a google/fonts checkout (METADATA.pb files suffice):
    uv run python -m analysis.gf_index <google/fonts dir> data/diagnostics/gf_index.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.family_diagnostic import dist_matrix, family_of, load_pairs, mean_offdiag, pick_rep, spearman  # noqa: E402

SUBSETS = {
    "Tamil": {"tamil"}, "Bengali": {"bengali"}, "Telugu": {"telugu"},
    "Devanagari": {"devanagari"}, "Arabic": {"arabic"}, "Cyrillic": {"cyrillic"},
    "Han": {"chinese-simplified", "chinese-traditional", "chinese-hongkong"},
    "Katakana": {"japanese"},
}
B, SEED = 5000, 11


def main(repo: Path, index_csv: Path, out: Path | None = None,
         verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the check. Returns (per-script table, SSS comparison, summary)."""
    gf = pd.read_csv(index_csv).fillna("")
    by_stem = gf.drop_duplicates("stem").set_index("stem")
    fam_meta = gf.drop_duplicates("family_dir").set_index("family_dir")
    pair_dir = repo / "similarity_research/diversity_research/vit_outputs_100/font_similarity_pairs"

    rows, members, rng = [], [], np.random.default_rng(SEED)
    boot = {}
    for p in sorted(pair_dir.glob("font_similarity_pairs_*.csv")):
        script = p.stem.replace("font_similarity_pairs_", "")
        d, files = load_pairs(p)
        m = dist_matrix(d, files)

        fam_dir = []
        for f in files:
            if f in by_stem.index:
                fam_dir.append(by_stem.loc[f, "family_dir"])
            else:
                key = family_of(f).lower().replace("-", "").replace("_", "")
                fam_dir.append(key if key in fam_meta.index else f"?{family_of(f)}")
        tagged = []
        for fd in fam_dir:
            subs = set(fam_meta.loc[fd, "subsets"].split("|")) if fd in fam_meta.index else set()
            tagged.append(bool(subs & SUBSETS[script]))

        groups: dict[str, list[int]] = {}
        for k, fd in enumerate(fam_dir):
            groups.setdefault(fd, []).append(k)
        reps = {fd: files.index(pick_rep([files[k] for k in ks])) for fd, ks in groups.items()}
        on = [fd for fd in groups if tagged[groups[fd][0]]]
        off = [fd for fd in groups if not tagged[groups[fd][0]]]
        unmatched = [fd for fd in groups if fd.startswith("?")]

        rep_on = [reps[fd] for fd in on]
        n_tagged_total = int(fam_meta.subsets.apply(lambda s: bool(set(s.split("|")) & SUBSETS[script])).sum())
        rows.append({
            "script": script,
            "sampled_families": len(groups),
            "on_population_families": len(on),
            "off_population_families": len(off),
            "unmatched_files_families": len(unmatched),
            "off_population_share": len(off) / len(groups),
            "gf_families_with_subset_today": n_tagged_total,
            "sample_coverage_of_subset_families": len(on) / n_tagged_total,
            "family_rep_all": mean_offdiag(m, list(reps.values()), list(reps)),
            "family_rep_aligned": mean_offdiag(m, rep_on, on),
            "off_population_examples": ", ".join(sorted(off)[:8]),
        })
        for fd in groups:
            members.append({"script": script, "family_dir": fd, "on_population": fd in on,
                            "n_files": len(groups[fd]), "representative": files[reps[fd]]})

        # cluster bootstrap on the aligned population
        r = np.array(rep_on)
        n = len(r)
        iu = np.triu_indices(n, 1)
        vals = np.empty(B)
        for b in range(B):
            draw = rng.integers(0, n, n)
            sub = m[np.ix_(r[draw], r[draw])][iu]
            vals[b] = sub[(draw[:, None] != draw[None, :])[iu]].mean()
        boot[script] = vals

    res = pd.DataFrame(rows).set_index("script")
    bt = pd.DataFrame(boot)
    rk = bt.rank(axis=1)
    res["aligned_ci_lo"] = bt.quantile(0.025)
    res["aligned_ci_hi"] = bt.quantile(0.975)
    res["aligned_rank"] = res.family_rep_aligned.rank().astype(int)
    res["aligned_rank_ci"] = [f"{int(a)}-{int(b)}" for a, b in zip(rk.quantile(0.025), rk.quantile(0.975))]
    res["p_most_homogeneous"] = (rk == 1).mean()

    committed = pd.read_csv(repo / "data/similarity/similarity_results.csv").set_index("script")
    file_level = committed.loc[res.index, "mean_cosine_distance"]
    canon = pd.read_csv(repo / "data/final/script_servedness.csv").set_index("script")
    sys.path.insert(0, str(repo))
    from analysis.scoring import compute_servedness  # noqa: E402

    raw = res.family_rep_aligned
    div = (raw - raw.min()) / (raw.max() - raw.min())
    div.loc["Latin"] = (committed.loc["Latin", "mean_cosine_distance"] - raw.min()) / (raw.max() - raw.min())
    sss = compute_servedness(canon["support_gf_families"], div, canon["exposure_demand"])
    nl = [s for s in sss.index if s != "Latin"]
    cmp_ = pd.DataFrame({"sss_committed": canon["servedness_score"], "tier_committed": canon["tier"],
                         "sss_aligned": sss["servedness_score"], "tier_aligned": sss["tier"]}).sort_values("sss_aligned")
    summary = {
        "rho_div_file_vs_aligned": spearman(file_level, raw),
        "rho_sss_committed_vs_aligned_nonlatin": spearman(cmp_.loc[nl, "sss_committed"], cmp_.loc[nl, "sss_aligned"]),
        "tiers_identical": int((cmp_.tier_committed == cmp_.tier_aligned).sum()),
    }

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        res.to_csv(out / "population_check_by_script.csv")
        pd.DataFrame(members).to_csv(out / "population_members.csv", index=False)
        cmp_.to_csv(out / "sss_file_vs_aligned.csv")
        pd.Series(summary).to_csv(out / "population_summary.csv", header=["value"])
    if verbose:
        pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 90)
        print(res.drop(columns=["off_population_examples"]).round(3).sort_values("family_rep_aligned"))
        print(res["off_population_examples"])
        print(cmp_.round(3))
        print(summary)
    return res, cmp_, summary


if __name__ == "__main__":
    REPO = Path(__file__).resolve().parents[1]
    main(REPO, REPO / "data" / "diagnostics" / "gf_index.csv", REPO / "data" / "diagnostics")
