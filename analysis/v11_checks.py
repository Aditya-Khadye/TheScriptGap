"""Checks behind the v1.1 diversity results (the white paper's §3.5 and Table 4).

v1.1 re-ran the diversity stage for all nine scripts in one pass, with text shaping
pinned on (libraqm) and Latin on the same 100-font, ~100-glyph protocol as the
other scripts. This script compares v1.1 with v1.0 and tests how far the v1.1
tiers hold. Everything runs from committed data in seconds; no GPU or font files.

    python -m analysis.v11_checks

Checks (each writes one CSV under data/analysis/):

  v11_scores.csv          v1.0 vs v1.1 mean cosine distance, diversity, SSS and tier,
                          with exposure and with exposure removed (constant denominator)
  v11_noise.csv           one-at-a-time +/-10% and +/-20% shocks to each script's
                          support, raw mean cosine distance and exposure
  v11_encoder.csv         ViT-B/16 vs ResNet-50 v1.1 diversity, SSS and tier
  v11_latin_primary.csv   Latin diversity from only the sampled fonts whose Google
                          Fonts primary script is Latin (39 of the 92 matched files in
                          the Latin sample are fonts for other scripts that also
                          cover Latin)
  v11_checks_summary.csv  the headline numbers from all four

Family-level and subset-aligned diversity on v1.1 are in analysis/family_diagnostic.py
and analysis/population_check.py (outputs in data/diagnostics/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.scoring import compute_servedness, minmax  # noqa: E402

DIV = ROOT / "similarity_research" / "diversity_research"
VIT = DIV / "vit_outputs_v1_1"
CNN = DIV / "cnn_outputs_v1_1"
DATA = ROOT / "data"
OUT = DATA / "analysis"
INDIC = ["Tamil", "Bengali", "Devanagari", "Telugu"]
SHOCKS = (0.10, 0.20)


def load_inputs() -> tuple[pd.Series, pd.Series]:
    final = pd.read_csv(DATA / "final" / "script_servedness.csv").set_index("script")
    return final["support_gf_families"].astype(float), final["exposure_demand"].astype(float)


def mcd(results_csv: Path) -> pd.Series:
    return pd.read_csv(results_csv).set_index("script")["mean_cosine_distance"].astype(float)


def no_exposure(exposure: pd.Series) -> pd.Series:
    """A constant denominator: the score with exposure taken out."""
    return pd.Series(10.0, index=exposure.index)


def score(support: pd.Series, mean_dist: pd.Series, exposure: pd.Series) -> pd.DataFrame:
    return compute_servedness(support, minmax(mean_dist), exposure).reindex(support.index)


def ranks(sss: pd.Series) -> pd.Series:
    """1 = least served; ties broken by order, as in the published tables."""
    return sss.rank(method="first").astype(int)


def compare_versions(support, exposure) -> pd.DataFrame:
    v10 = mcd(DATA / "similarity" / "similarity_results_v1_0.csv")
    v11 = mcd(DATA / "similarity" / "similarity_results.csv")
    out = pd.DataFrame({"mcd_v1_0": v10, "mcd_v1_1": v11,
                        "diversity_v1_0": minmax(v10), "diversity_v1_1": minmax(v11)})
    for tag, m in (("v1_0", v10), ("v1_1", v11)):
        for lab, e in (("", exposure), ("_no_exposure", no_exposure(exposure))):
            s = score(support, m, e)
            out[f"sss_{tag}{lab}"], out[f"tier_{tag}{lab}"] = s["servedness_score"], s["tier"]
    return out.sort_values("sss_v1_1")


def noise(support, exposure) -> pd.DataFrame:
    base_mcd = mcd(DATA / "similarity" / "similarity_results.csv")
    rows = []
    for lab, e0 in (("with_exposure", exposure), ("no_exposure", no_exposure(exposure))):
        base = score(support, base_mcd, e0)
        base_rank = ranks(base["servedness_score"])
        inputs = ["support", "mean_cosine_distance"] + (["exposure"] if lab == "with_exposure" else [])
        for pct in SHOCKS:
            for inp in inputs:
                for script in support.index:
                    for sign in (1, -1):
                        s, m, e = support.copy(), base_mcd.copy(), e0.copy()
                        {"support": s, "mean_cosine_distance": m, "exposure": e}[inp][script] *= 1 + sign * pct
                        new = score(s, m, e)
                        new_rank = ranks(new["servedness_score"])
                        moved = [x for x in new.index if new.loc[x, "tier"] != base.loc[x, "tier"]]
                        rows.append({
                            "exposure": lab, "shock": f"{'+' if sign > 0 else '-'}{int(pct * 100)}%",
                            "size": pct, "input": inp, "script": script,
                            "max_rank_move": int((new_rank - base_rank).abs().max()),
                            "tier_changes": "; ".join(f"{x} {base.loc[x, 'tier']} -> {new.loc[x, 'tier']}" for x in moved),
                            "n_tier_changes": len(moved),
                            "indic_bottom_four": set(new_rank.nsmallest(4).index) == set(INDIC),
                        })
    return pd.DataFrame(rows)


def encoder(support, exposure) -> pd.DataFrame:
    vit, cnn = mcd(VIT / "diversity_index_results.csv"), mcd(CNN / "diversity_index_results.csv")
    out = pd.DataFrame({"mcd_vit": vit, "mcd_resnet": cnn,
                        "diversity_vit": minmax(vit), "diversity_resnet": minmax(cnn.reindex(vit.index))})
    for tag, m in (("vit", vit), ("resnet", cnn)):
        s = score(support, m, exposure)
        out[f"sss_{tag}"], out[f"tier_{tag}"] = s["servedness_score"], s["tier"]
    return out.sort_values("sss_vit")


def latin_primary(support, exposure) -> tuple[pd.DataFrame, dict]:
    pairs = pd.read_csv(VIT / "font_similarity_pairs" / "font_similarity_pairs_Latin.csv")
    gf = pd.read_csv(DATA / "diagnostics" / "gf_index.csv", dtype=str).drop_duplicates("stem").set_index("stem")
    stems = sorted(set(pairs.font_name1) | set(pairs.font_name2))
    matched = [s for s in stems if s in gf.index]
    primary = gf["primary_script"]
    # Google Fonts leaves primary_script empty for Latin fonts
    latin_primary = [s for s in matched if pd.isna(primary[s]) or primary[s] == "Latn"]
    keep = pairs[pairs.font_name1.isin(latin_primary) & pairs.font_name2.isin(latin_primary)]
    m_latin = float((1 - keep.similarity).mean())

    v11 = mcd(DATA / "similarity" / "similarity_results.csv")
    alt = v11.copy()
    alt["Latin"] = m_latin
    out = pd.DataFrame({"mcd_v1_1": v11, "mcd_latin_primary": alt,
                        "diversity_v1_1": minmax(v11), "diversity_latin_primary": minmax(alt)})
    for tag, m in (("v1_1", v11), ("latin_primary", alt)):
        s = score(support, m, exposure)
        out[f"sss_{tag}"], out[f"tier_{tag}"] = s["servedness_score"], s["tier"]
    info = {"latin_sample_files": len(stems), "latin_sample_matched": len(matched),
            "latin_sample_latin_primary": len(latin_primary),
            "latin_mcd_latin_primary": m_latin}
    return out.sort_values("sss_v1_1"), info


def spearman(a: pd.Series, b: pd.Series) -> float:
    a, b = a.align(b, join="inner")
    return float(a.rank().corr(b.rank()))


def main(out: Path | None = OUT, verbose: bool = True) -> dict:
    support, exposure = load_inputs()
    versions = compare_versions(support, exposure)
    shocks = noise(support, exposure)
    enc = encoder(support, exposure)
    lat, lat_info = latin_primary(support, exposure)

    nl = [s for s in enc.index if s != "Latin"]
    summary = {
        "v1_1_tier_changes_vs_v1_0": int((versions.tier_v1_0 != versions.tier_v1_1).sum()),
        "v1_1_max_abs_sss_change_without_exposure": float((versions.sss_v1_1 - versions.sss_v1_1_no_exposure).abs().max()),
        "v1_1_tiers_same_without_exposure": int((versions.tier_v1_1 == versions.tier_v1_1_no_exposure).sum()),
        "encoder_spearman_mcd_all9": spearman(enc.mcd_vit, enc.mcd_resnet),
        "encoder_spearman_mcd_nonlatin8": spearman(enc.mcd_vit[nl], enc.mcd_resnet[nl]),
        "encoder_tiers_same": int((enc.tier_vit == enc.tier_resnet).sum()),
        "encoder_tiers_moved": ", ".join(enc.index[enc.tier_vit != enc.tier_resnet]),
        **lat_info,
        "latin_diversity_latin_primary": float(lat.loc["Latin", "diversity_latin_primary"]),
        "latin_primary_tiers_moved": ", ".join(lat.index[lat.tier_v1_1 != lat.tier_latin_primary]),
    }
    for (lab, pct), g in shocks.groupby(["exposure", "size"]):
        k = f"noise_{lab}_{int(pct * 100)}pct"
        summary[f"{k}_shocks"] = len(g)
        summary[f"{k}_shocks_moving_a_rank"] = int((g.max_rank_move > 0).sum())
        summary[f"{k}_max_rank_move"] = int(g.max_rank_move.max())
        summary[f"{k}_shocks_changing_a_tier"] = int((g.n_tier_changes > 0).sum())
        summary[f"{k}_indic_always_bottom_four"] = bool(g.indic_bottom_four.all())

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        versions.to_csv(out / "v11_scores.csv", index_label="script")
        shocks.to_csv(out / "v11_noise.csv", index=False)
        enc.to_csv(out / "v11_encoder.csv", index_label="script")
        lat.to_csv(out / "v11_latin_primary.csv", index_label="script")
        pd.Series(summary, name="value").to_csv(out / "v11_checks_summary.csv", index_label="check")
    if verbose:
        pd.set_option("display.width", 220)
        print(versions.round(4).to_string(), "\n")
        print(enc.round(4).to_string(), "\n")
        print(lat.round(4).to_string(), "\n")
        tiered = shocks[shocks.n_tier_changes > 0]
        print(tiered[["exposure", "input", "script", "shock", "tier_changes"]].to_string(index=False), "\n")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    return {"versions": versions, "noise": shocks, "encoder": enc, "latin_primary": lat, "summary": summary}


if __name__ == "__main__":
    main()
