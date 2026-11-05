"""
Infrastructure Complexity Index, v1.1 measurement (GSUB closure).

Unit of analysis: one representative font file per Google Fonts family
(upright variable font if present, otherwise Regular). A family belongs to a
script if its METADATA.pb lists that script's subset (same population as the
support term). Each (family, script) pair is measured independently.

Per (family, script):
  n_cps          codepoints in the script's Unicode ranges that the font maps
  n_base         distinct glyphs those codepoints map to (cmap)
  n_closure      glyphs reachable from n_base through GSUB lookups registered
                 for the script's OpenType tags (or DFLT), restricted to the
                 features a shaping engine applies by default (DEFAULT_GSUB)
  n_closure_all  same closure with every GSUB feature (sensitivity)
  E              n_closure / n_cps          shaping expansion, measured
  E_all          n_closure_all / n_cps
  E_v0           maxp.numGlyphs / n_cps     original formula (multi-script overcount)
  E_v1           v1.0 heuristic (base + 0.15*base per matched GSUB FeatureRecord, font-wide)
  V              (sTypoAscender + |sTypoDescender| + sTypoLineGap) / UPM   (v1 definition)
  V_ink          (95th pct glyph yMax - 5th pct glyph yMin) / UPM over the script's base glyphs
  F              shaping categories present for THIS script (features registered
                 under its OpenType tags or DFLT; RTL only for RTL scripts)
  F_v1           v1.0 friction (font-wide features; RTL if the font maps any RTL codepoint)

Verified against the public `pyftsubset` CLI (same unicodes, scripts and
features): identical glyph counts, except that pyftsubset also adds glyf
composite components after the GSUB closure, which this measure deliberately
excludes (they are building blocks, not shaped forms).

Usage (from the repo root; fontTools pinned in uv.lock):
    uv run python -m complexity.closure <google/fonts checkout> [n_workers]
    uv run python -m complexity.aggregate_closure
Writes data/complexity_v11/representatives.csv and per_family_script.csv.
A full checkout is not needed: the representatives are listed in
representatives.csv and can be fetched with a sparse checkout (escape [ and ]
in the patterns).
"""
from __future__ import annotations

import io
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from fontTools.pens.boundsPen import BoundsPen
from fontTools.subset import Options, Subsetter
from fontTools.ttLib import TTFont

# v1.0 Unicode ranges (complexity/main.py), kept identical for comparability
RANGES = {
    "Latin": [(0x0000, 0x007F), (0x0080, 0x00FF), (0x0100, 0x017F)],
    "Cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "Katakana": [(0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0xF900, 0xFAFF)],
    "Bengali": [(0x0980, 0x09FF)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
}
SUBSET_TO_SCRIPT = {
    "latin": "Latin", "cyrillic": "Cyrillic", "devanagari": "Devanagari",
    "arabic": "Arabic", "bengali": "Bengali", "tamil": "Tamil", "telugu": "Telugu",
    "japanese": "Katakana", "chinese-simplified": "Han",
    "chinese-traditional": "Han", "chinese-hongkong": "Han",
}
OT_TAGS = {
    "Latin": ["latn"], "Cyrillic": ["cyrl"], "Devanagari": ["deva", "dev2"],
    "Bengali": ["beng", "bng2"], "Tamil": ["taml", "tml2"], "Telugu": ["telu", "tel2"],
    "Arabic": ["arab"], "Han": ["hani"], "Katakana": ["kana"],
}
RTL_SCRIPTS = {"Arabic"}

# GSUB features applied by default in horizontal text by OpenType shaping
# engines (common + Arabic + Indic shapers). Discretionary features (dlig,
# salt, ssXX, smcp, ...) and vertical-only features are excluded.
DEFAULT_GSUB = [
    "ccmp", "locl", "rlig", "rclt", "calt", "clig", "liga",
    "isol", "init", "medi", "med2", "fina", "fin2", "fin3", "mset", "stch",
    "nukt", "akhn", "rphf", "rkrf", "pref", "blwf", "abvf", "half", "pstf",
    "vatu", "cjct", "pres", "abvs", "blws", "psts", "haln",
]

# v1.0 definitions, reproduced for the before/after comparison
V1_SHAPING = {"init", "medi", "fina", "isol", "calt", "liga", "rlig", "dlig", "blwf", "half",
              "pres", "abvs", "blws", "psts", "nukt", "akhn", "rphf", "vatu", "haln", "ccmp"}
FRICTION = {
    "contextual_joining": {"init", "medi", "fina", "isol", "calt"},
    "indic_shaping": {"blwf", "half", "pres", "abvs", "blws", "psts", "haln", "nukt", "akhn", "rphf", "vatu"},
    "ligature_heavy": {"liga", "rlig", "dlig", "hlig"},
    "mark_positioning": {"mark", "mkmk", "abvm", "blwm"},
}
V1_RTL = [(0x0590, 0x05FF), (0x0600, 0x06FF), (0x0700, 0x074F), (0x0750, 0x077F),
          (0x0780, 0x07BF), (0x07C0, 0x07FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]
MIN_CPS = 10


# Compatibility blocks that encode already-shaped forms (Arabic positional
# forms and ligatures) or duplicate ideographs. They are excluded from the
# nominal codepoint set: a modern shaper reaches those shapes through GSUB from
# the nominal letters, so counting them in the denominator would hide exactly
# the expansion being measured.
COMPAT_EXCLUDE = [(0xFB50, 0xFDFF), (0xFE70, 0xFEFF), (0xF900, 0xFAFF)]


def script_cps(cmap: dict, script: str) -> set[int]:
    return {cp for cp in cmap
            if any(a <= cp <= b for a, b in RANGES[script])
            and not any(a <= cp <= b for a, b in COMPAT_EXCLUDE)}


def closure(data: bytes, cps: set[int], script: str, features) -> tuple[int, int]:
    font = TTFont(io.BytesIO(data), fontNumber=0)
    opts = Options()
    opts.layout_scripts = OT_TAGS[script] + ["DFLT"]
    opts.layout_features = features
    opts.notdef_glyph = False
    opts.recommended_glyphs = False
    opts.layout_closure = True
    opts.ignore_missing_glyphs = True
    opts.ignore_missing_unicodes = True
    sub = Subsetter(options=opts)
    sub.populate(unicodes=cps)
    sub._prune_pre_subset(font)
    sub._closure_glyphs(font)
    return len(sub.glyphs_cmaped), len(sub.glyphs_gsubed)


def all_features(font: TTFont) -> set[str]:
    tags = set()
    for t in ("GSUB", "GPOS"):
        if t in font and font[t].table.FeatureList:
            tags |= {fr.FeatureTag.strip() for fr in font[t].table.FeatureList.FeatureRecord}
    return tags


def script_features(font: TTFont, script: str) -> set[str]:
    """Feature tags reachable from the script's ScriptRecords, else from DFLT."""
    tags = set()
    for t in ("GSUB", "GPOS"):
        if t not in font or not font[t].table.ScriptList or not font[t].table.FeatureList:
            continue
        tbl = font[t].table
        recs = {sr.ScriptTag: sr.Script for sr in tbl.ScriptList.ScriptRecord}
        chosen = [recs[k] for k in OT_TAGS[script] if k in recs] or ([recs["DFLT"]] if "DFLT" in recs else [])
        for sc in chosen:
            langsys = ([sc.DefaultLangSys] if sc.DefaultLangSys else []) + [lr.LangSys for lr in sc.LangSysRecord]
            for ls in langsys:
                for fi in ls.FeatureIndex:
                    tags.add(tbl.FeatureList.FeatureRecord[fi].FeatureTag.strip())
    return tags


def friction(tags: set[str], rtl: bool) -> int:
    return sum(bool(tags & s) for s in FRICTION.values()) + int(rtl)


def e_v1(font: TTFont, cmap: dict, cps: set[int]) -> float:
    base = len({cmap[cp] for cp in cps if cp in cmap})
    extra = 0
    if "GSUB" in font and font["GSUB"].table.FeatureList:
        for fr in font["GSUB"].table.FeatureList.FeatureRecord:
            if fr.FeatureTag.strip() in V1_SHAPING:
                extra += int(base * 0.15)
    return (base + extra) / len(cps)


def ink_extent(font: TTFont, cmap: dict, cps: set[int], upm: int) -> float:
    gs = font.getGlyphSet()
    lo, hi = [], []
    for g in {cmap[cp] for cp in cps}:
        pen = BoundsPen(gs)
        try:
            gs[g].draw(pen)
        except Exception:
            continue
        if pen.bounds:
            lo.append(pen.bounds[1]); hi.append(pen.bounds[3])
    if not lo:
        return np.nan
    return (np.percentile(hi, 95) - np.percentile(lo, 5)) / upm


def measure_family(args) -> list[dict]:
    root, row = args
    path = Path(root) / row["path"].lstrip("/")
    out = []
    try:
        data = path.read_bytes()
        font = TTFont(io.BytesIO(data), fontNumber=0)
        cmap = font.getBestCmap() or {}
        upm = font["head"].unitsPerEm
        os2 = font["OS/2"] if "OS/2" in font else None
        V = ((os2.sTypoAscender + abs(os2.sTypoDescender) + os2.sTypoLineGap) / upm) if os2 else np.nan
        feats_font = all_features(font)
        rtl_any = any(a <= cp <= b for cp in cmap for a, b in V1_RTL)
        scripts = sorted({SUBSET_TO_SCRIPT[s] for s in row["subsets"].split("|") if s in SUBSET_TO_SCRIPT})
        for script in scripts:
            cps = script_cps(cmap, script)
            rec = {"family_dir": row["family_dir"], "file": path.name, "script": script, "n_cps": len(cps)}
            if len(cps) < MIN_CPS:
                rec["status"] = "below_min_cps"
                out.append(rec)
                continue
            n_base, n_clo = closure(data, cps, script, DEFAULT_GSUB)
            _, n_all = closure(data, cps, script, ["*"])
            sf = script_features(font, script)
            rec.update({
                "status": "ok",
                "n_base": n_base, "n_closure": n_clo, "n_closure_all": n_all,
                "total_glyphs": font["maxp"].numGlyphs,
                "E": n_clo / len(cps), "E_all": n_all / len(cps),
                "E_v0": font["maxp"].numGlyphs / len(cps),
                "E_v1": e_v1(font, cmap, cps),
                "V": V, "V_ink": ink_extent(font, cmap, cps, upm),
                "F": friction(sf, script in RTL_SCRIPTS),
                "F_v1": friction(feats_font, rtl_any),
                "has_script_record": any(
                    t in font and font[t].table.ScriptList and
                    any(sr.ScriptTag in OT_TAGS[script] for sr in font[t].table.ScriptList.ScriptRecord)
                    for t in ("GSUB", "GPOS")),
            })
            out.append(rec)
    except Exception as e:
        out.append({"family_dir": row["family_dir"], "file": path.name, "status": f"error: {type(e).__name__}: {e}",
                    "trace": traceback.format_exc(limit=2)})
    return out


TARGET_SUBSETS = set(SUBSET_TO_SCRIPT)


def build_representatives(root: Path) -> pd.DataFrame:
    """One representative file per family tagged with a target subset:
    upright variable font > Regular > other upright > first (by name)."""
    import re
    from analysis.gf_index import build_index
    idx = build_index(root).fillna("")
    fam = idx.groupby(["license", "family_dir"]).agg(subsets=("subsets", "first"),
                                                     stems=("stem", list)).reset_index()
    fam = fam[fam.subsets.apply(lambda s: bool(set(s.split("|")) & TARGET_SUBSETS))].copy()

    def pick(stems):
        return sorted(stems, key=lambda f: ("Italic" in f, "[" not in f,
                                            not f.split("[")[0].endswith("-Regular"), f))[0]

    paths = []
    for _, r in fam.iterrows():
        md = (root / r.license / r.family_dir / "METADATA.pb").read_text(errors="ignore")
        rep = pick(r.stems)
        fn = next(m for m in re.findall(r'filename:\s*"([^"]+)"', md) if Path(m).stem == rep)
        paths.append(f"/{r.license}/{r.family_dir}/{fn}")
    fam["rep"] = fam.stems.apply(pick)
    fam["path"] = paths
    return fam[["license", "family_dir", "subsets", "rep", "path"]]


def main(root: str, reps_csv: str, out_csv: str, workers: int = 2, limit: int | None = None) -> None:
    reps = pd.read_csv(reps_csv).fillna("")
    if limit:
        reps = reps.sample(limit, random_state=0)
    jobs = [(root, r) for r in reps.to_dict("records")]
    # big files first so the pool does not end on a long CJK tail
    jobs.sort(key=lambda j: -(Path(root) / j[1]["path"].lstrip("/")).stat().st_size)
    rows = []
    with Pool(workers) as pool:
        for i, recs in enumerate(pool.imap_unordered(measure_family, jobs, chunksize=1), 1):
            rows.extend(recs)
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} families", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("done", len(rows), "rows")


if __name__ == "__main__":
    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO))
    gf_root = Path(sys.argv[1])
    out_dir = REPO / "data" / "complexity_v11"
    out_dir.mkdir(parents=True, exist_ok=True)
    reps_path = out_dir / "representatives.csv"
    build_representatives(gf_root).to_csv(reps_path, index=False)
    main(str(gf_root), str(reps_path), str(out_dir / "per_family_script.csv"),
         int(sys.argv[2]) if len(sys.argv) > 2 else 2)
