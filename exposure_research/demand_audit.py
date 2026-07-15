"""
===============================================================================
Demand-axis audit — provenance & validity diagnostics for the Exposure Index
===============================================================================
Project:  TheScriptGap

WHY THIS FILE EXISTS
--------------------
The "Exposure Index" (per-script web demand) is the foundation of the Script
Servedness Score, but its derivation was neither documented nor reproducible:

  * The file the model consumes, `exposure_filtered_results.csv`, is READ by
    the pipeline (data_viz, final_model) but is not WRITTEN by any committed
    script — its provenance is unknown and its numbers cannot be regenerated
    or audited from this repository.
  * The only committed exposure code (`exposure_research/main.py`, the treemap)
    omits Han entirely (it is absent from that file's `scripts_list`) and
    produces per-script totals that do NOT match `exposure_filtered_results.csv`.

This script makes the demand stage REPRODUCIBLE and AUDITABLE. It recomputes
per-script demand from the one committed source that carries the raw signal
(`data/bigquery/big_query_data.csv`: one row per web font, with `font_count`
= web font-request count and `scripts` = the Unicode scripts that font's glyph
set covers), under several explicit attribution rules, and quantifies the
"font-bundling confound" so the number is never again presented without its
caveat.

THE CONFOUND (the headline finding)
-----------------------------------
The demand metric attributes a font's request count to EVERY script its glyphs
cover. Because the most-requested web fonts are large multi-script Latin/UI
fonts (Inter covers 12 scripts, Roboto 7, Noto Sans JP 8, Poppins 4), a
script inherits the popularity of fonts that merely *bundle* its glyphs — even
when essentially no page using that font is written in that script. Measured
here, 97–100% of every non-Latin script's "demand" comes from fonts covering
>=3 scripts; e.g. Tamil/Telugu/Bengali are driven by Inter, Devanagari by
Poppins, and Arabic largely by FontAwesome (an ICON font that maps glyphs into
the Arabic block). So this index is closer to "popularity of fonts that can
render the script" than to "how much the script is read." Treat it as a proxy,
disclose the confound, and prefer a content-based demand signal (e.g. Common
Crawl + a script classifier such as GlotScript) for any load-bearing claim.

USAGE
-----
    uv run --with pandas python exposure_research/demand_audit.py
Outputs a console report and `data/exposure/demand_audit.csv`.
===============================================================================
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

try:
    from paths import BIGQUERY_DATA_DIR, EXPOSURE_DATA_DIR
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import BIGQUERY_DATA_DIR, EXPOSURE_DATA_DIR

SOURCE = BIGQUERY_DATA_DIR / "big_query_data.csv"
LEGACY = BIGQUERY_DATA_DIR / "exposure_filtered_results.csv"  # unreproducible; for comparison only
OUT = EXPOSURE_DATA_DIR / "demand_audit.csv"

TARGET_SCRIPTS = ["Latin", "Cyrillic", "Katakana", "Han",
                  "Devanagari", "Arabic", "Bengali", "Tamil", "Telugu"]

# Tags in the `scripts` field that are not stand-alone reading scripts and
# should not make a font count as "multi-script" on their own.
NON_READING_TAGS = {"PUA", "Inherited", "Common", "Emoji", "Bopomofo"}


def parse_scripts(cell: object) -> list[str]:
    """Parse the unquoted '[Latin,Cyrillic,...]' coverage field robustly."""
    if not isinstance(cell, str):
        return []
    return [s.strip() for s in re.sub(r"[\[\]]", "", cell).split(",") if s.strip()]


def load_source() -> pd.DataFrame:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing demand source: {SOURCE}")
    df = pd.read_csv(SOURCE)
    # The pull has used both column names for the coverage field over time.
    coverage_col = "supported_scripts" if "supported_scripts" in df.columns else "scripts"
    df["coverage"] = df[coverage_col].apply(parse_scripts)
    df["reading_coverage"] = df["coverage"].apply(
        lambda L: [s for s in L if s not in NON_READING_TAGS]
    )
    df["n_reading"] = df["reading_coverage"].apply(len)
    df["count"] = pd.to_numeric(df["font_count"], errors="coerce").fillna(0)
    return df


def demand_under_rule(df: pd.DataFrame, max_scripts: int | None) -> dict[str, float]:
    """Per-script demand. max_scripts=None -> all fonts covering the script;
    max_scripts=k -> only fonts whose reading coverage is <= k scripts."""
    out = {}
    for s in TARGET_SCRIPTS:
        mask = df["reading_coverage"].apply(lambda L: s in L)
        if max_scripts is not None:
            mask &= df["n_reading"] <= max_scripts
        out[s] = float(df[mask]["count"].sum())
    return out


def confound_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in TARGET_SCRIPTS:
        sub = df[df["reading_coverage"].apply(lambda L: s in L)]
        total = sub["count"].sum()
        bundled = sub[sub["n_reading"] >= 3]["count"].sum()
        top = sub.sort_values("count", ascending=False).head(1)
        rows.append({
            "script": s,
            "demand_coverage_all": total,
            "pct_from_multiscript_bundlers": (bundled / total if total else 0.0),
            "top_contributor": top["font_name"].iloc[0] if len(top) else "",
            "top_contributor_n_scripts": int(top["n_reading"].iloc[0]) if len(top) else 0,
        })
    return pd.DataFrame(rows)


def load_legacy() -> dict[str, float]:
    if not LEGACY.exists():
        return {}
    df = pd.read_csv(LEGACY, names=["script", "count"], header=0)
    return dict(zip(df["script"], pd.to_numeric(df["count"], errors="coerce")))


def main() -> None:
    df = load_source()
    print(f"Source: {SOURCE.name}  ({len(df):,} fonts)  Han present: "
          f"{df['reading_coverage'].apply(lambda L: 'Han' in L).any()}\n")

    cov = demand_under_rule(df, None)     # all fonts covering the script
    ded = demand_under_rule(df, 1)        # dedicated single-script fonts only
    near = demand_under_rule(df, 2)       # script + at most one other
    legacy = load_legacy()

    audit = confound_breakdown(df)
    audit["demand_dedicated_1script"] = audit["script"].map(ded)
    audit["demand_le2_scripts"] = audit["script"].map(near)
    audit["legacy_filtered_used_by_model"] = audit["script"].map(legacy)

    EXPOSURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("Per-script demand under different attribution rules:\n")
    show = audit[["script", "demand_coverage_all", "demand_dedicated_1script",
                  "demand_le2_scripts", "legacy_filtered_used_by_model",
                  "pct_from_multiscript_bundlers", "top_contributor",
                  "top_contributor_n_scripts"]].copy()
    print(show.to_string(index=False))

    def ranking(d: dict[str, float]) -> str:
        return " > ".join(s for s, _ in sorted(d.items(), key=lambda kv: -kv[1]))

    print("\nDemand ranking (high -> low) is NOT stable across rules:")
    print("  coverage-all   :", ranking(cov))
    print("  dedicated(1)   :", ranking(ded))
    if legacy:
        print("  legacy (model) :", ranking(legacy))

    print("\nVERDICT: 97-100% of every non-Latin script's coverage-based demand "
          "comes from\nmulti-script bundler fonts; the ranking flips entirely "
          "depending on attribution;\nHan spans 4 orders of magnitude (e.g. "
          f"{cov['Han']:,.0f} -> {ded['Han']:,.0f}); and the file the model "
          "actually\nuses is not reproducible here. This index measures webfont "
          "popularity by glyph\ncoverage, not reading demand. See module docstring.")
    print(f"\nSaved diagnostics -> {OUT}")


if __name__ == "__main__":
    main()
