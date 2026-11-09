"""Does text shaping change the v1.0 diversity results?

The v1.0 diversity run (similarity_research/diversity_research/script_diversity_vit_100.py)
never sets a Pillow layout engine. Pillow then uses libraqm (HarfBuzz shaping) when it is
installed and its basic, unshaped layout when it is not, and the committed outputs match the
basic layout. Unshaped, Indic conjuncts are drawn as unjoined component glyphs, and some Arabic
fonts draw base letters only through shaping, so those glyphs come out blank and are skipped.

This script re-embeds the fonts of the committed v1.0 run for each script with the same code,
once per layout engine, and then recomputes the diversity index, the SSS and the tiers with
the shaped mean cosine distances.

    python -m analysis.shaping_check --fonts-dir /path/to/google-fonts-checkout

`--fonts-dir` is a google/fonts checkout. Commit 07d64d5a (2026-04-09) holds every font file
in the v1.0 samples; under the basic layout it reproduces the committed Tamil and Bengali
similarities to float precision. Shaped rendering needs a Pillow
built with libraqm (`python -c "from PIL import features; print(features.check('raqm'))"`).
CPU only; the five default scripts take about 80 minutes on two cores.

Writes data/analysis/shaping_check.csv (per script and engine) and
data/analysis/shaping_check_scores.csv (v1.0 vs shaped diversity, SSS and tiers).
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageFont, features

ROOT = Path(__file__).resolve().parents[1]
DIV = ROOT / "similarity_research" / "diversity_research"
PAIRS = DIV / "vit_outputs_100" / "font_similarity_pairs"
OUT = ROOT / "data" / "analysis"
SHAPED_SCRIPTS = ["Tamil", "Bengali", "Devanagari", "Telugu", "Arabic"]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DIV))
import script_diversity_vit_100 as V  # noqa: E402
from analysis.scoring import compute_servedness, minmax  # noqa: E402


def committed_fonts(script: str) -> tuple[pd.DataFrame, list[str]]:
    pairs = pd.read_csv(PAIRS / f"font_similarity_pairs_{script}.csv")
    names = sorted(set(pairs.font_name1) | set(pairs.font_name2))
    return pairs, names


def locate(fonts_dir: Path, stems: list[str]) -> list[Path]:
    """Map each committed font name (a file stem) to its file in the checkout."""
    index: dict[str, Path] = {}
    for ext in ("*.ttf", "*.otf"):
        for p in fonts_dir.rglob(ext):
            index.setdefault(p.stem, p)
    missing = [s for s in stems if s not in index]
    if missing:
        raise SystemExit(f"{len(missing)} committed fonts not found in {fonts_dir}: {missing[:5]}")
    return [index[s] for s in stems]


def embed(paths: list[Path], ref: list[str], extractor, engine) -> tuple[list[str], np.ndarray, int]:
    """Per-font normalized mean glyph embedding, exactly as the v1.0 run, under one layout engine."""
    orig = ImageFont.truetype
    orig_engine = getattr(V, "LAYOUT_ENGINE", None)
    V.ImageFont.truetype = functools.partial(orig, layout_engine=engine)  # v1.0 code: no engine argument
    if orig_engine is not None:
        V.LAYOUT_ENGINE = engine  # v1.1 code pins the engine itself
    try:
        names, embs, n_glyphs = [], [], 0
        for fp in paths:
            supported = V.check_font_has_chars(str(fp), ref)
            if len(supported) < 5:
                continue
            images = [img for c in supported if (img := V.render_glyph(str(fp), c)) is not None]
            if len(images) < 5:
                continue
            e = extractor.extract_batch(images).mean(axis=0)
            embs.append(e / np.linalg.norm(e))
            names.append(fp.stem)
            n_glyphs += len(images)
    finally:
        V.ImageFont.truetype = orig
        if orig_engine is not None:
            V.LAYOUT_ENGINE = orig_engine
    return names, np.vstack(embs), n_glyphs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fonts-dir", type=Path, required=True)
    ap.add_argument("--scripts", default=",".join(SHAPED_SCRIPTS))
    ap.add_argument("--engines", default="RAQM,BASIC",
                    help="RAQM = shaped, BASIC = unshaped (reproduces v1.0)")
    args = ap.parse_args()

    engines = args.engines.split(",")
    if "RAQM" in engines and not features.check("raqm"):
        raise SystemExit("This Pillow has no libraqm, so RAQM would silently fall back to BASIC.")

    extractor = V.ViTFeatureExtractor(device="cpu")
    rows = []
    for script in args.scripts.split(","):
        ref = V.get_flat_reference_chars(script)
        pairs, stems = committed_fonts(script)
        paths = locate(args.fonts_dir, stems)
        for eng in engines:
            names, M, n_glyphs = embed(paths, ref, extractor, getattr(ImageFont.Layout, eng))
            sims = M @ M.T
            idx = {n: i for i, n in enumerate(names)}
            j = pairs[pairs.font_name1.isin(idx) & pairs.font_name2.isin(idx)]
            recomputed = np.array([sims[idx[a], idx[b]] for a, b in zip(j.font_name1, j.font_name2)])
            diff = np.abs(recomputed - j.similarity.to_numpy())
            rows.append({
                "script": script, "engine": eng, "fonts": len(names), "glyphs": n_glyphs,
                "multi_codepoint_refs": sum(len(c) > 1 for c in ref),
                "pairs_compared": len(j), "max_abs_diff_vs_v1": diff.max(),
                "mean_abs_diff_vs_v1": diff.mean(),
                "pearson_r_vs_v1": np.corrcoef(recomputed, j.similarity)[0, 1],
                "mean_cosine_distance": V.compute_diversity_metrics(M)["mean_cosine_distance"],
            })
            print(rows[-1], flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "shaping_check.csv", index=False)

    if "RAQM" not in engines:
        return
    final = pd.read_csv(ROOT / "data" / "final" / "script_servedness.csv").set_index("script")
    sim = pd.read_csv(ROOT / "data" / "similarity" / "similarity_results_v1_0.csv").set_index("script")
    mcd_v1 = sim["mean_cosine_distance"].astype(float)
    mcd_sh = mcd_v1.copy()
    for r in rows:
        if r["engine"] == "RAQM":
            mcd_sh[r["script"]] = r["mean_cosine_distance"]
    support, exposure = final["support_gf_families"], final["exposure_demand"]
    no_exposure = pd.Series(10.0, index=exposure.index)  # constant denominator = exposure removed

    out = pd.DataFrame({"mcd_v1": mcd_v1, "mcd_shaped": mcd_sh,
                        "diversity_v1": minmax(mcd_v1), "diversity_shaped": minmax(mcd_sh)})
    for label, exp in (("", exposure), ("_no_exposure", no_exposure)):
        for tag, mcd in (("v1", mcd_v1), ("shaped", mcd_sh)):
            s = compute_servedness(support, minmax(mcd), exp)
            out[f"sss_{tag}{label}"] = s["servedness_score"]
            out[f"tier_{tag}{label}"] = s["tier"]
    out = out.sort_values("sss_shaped")
    out.to_csv(OUT / "shaping_check_scores.csv", index_label="script")

    print(out.round(4).to_string())
    for col in ("mcd", "sss", "sss_no_exposure"):
        a = col.replace("sss", "sss_v1") if col != "mcd" else "mcd_v1"
        b = col.replace("sss", "sss_shaped") if col != "mcd" else "mcd_shaped"
        same = list(out[a].sort_values(kind="stable").index) == list(out[b].sort_values(kind="stable").index)
        print(f"{col}: order unchanged = {same}")
    for suffix in ("", "_no_exposure"):
        moved = out.index[out[f"tier_v1{suffix}"] != out[f"tier_shaped{suffix}"]].tolist()
        print(f"tiers changed{suffix or ' (with exposure)'}: {moved or 'none'}")


if __name__ == "__main__":
    main()
