"""Index Google Fonts METADATA.pb files.

Maps every font file stem to its Google Fonts family directory, family name,
license folder, subset tags, and primary_script. Only METADATA.pb files are
needed, so a blob-less sparse checkout of google/fonts is enough:

    git clone --filter=blob:none --no-checkout --depth 1 https://github.com/google/fonts gf
    git -C gf sparse-checkout set --no-cone '/*/*/METADATA.pb' && git -C gf checkout
    uv run python -m analysis.gf_index gf data/diagnostics/gf_index.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


def build_index(root: Path) -> pd.DataFrame:
    rows = []
    for md in sorted(root.glob("*/*/METADATA.pb")):
        text = md.read_text(errors="ignore")
        name = re.search(r'^name:\s*"([^"]+)"', text, re.M)
        prim = re.search(r'^primary_script:\s*"([^"]+)"', text, re.M)
        subsets = re.findall(r'^subsets:\s*"([^"]+)"', text, re.M)
        for filename in re.findall(r'filename:\s*"([^"]+)"', text):
            rows.append({
                "stem": Path(filename).stem,
                "family_dir": md.parent.name,
                "license": md.parent.parent.name,
                "family": name.group(1) if name else md.parent.name,
                "primary_script": prim.group(1) if prim else "",
                "subsets": "|".join(subsets),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_index(Path(sys.argv[1]))
    df.to_csv(sys.argv[2], index=False)
    print(len(df), "files;", df.family_dir.nunique(), "families")
