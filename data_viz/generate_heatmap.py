"""
===============================================================================
Script Servedness Score (SSS) — Heatmap Visualization
===============================================================================
Project:  TRC / Monotype — Identifying Underserved Scripts
Author:   Aditya (Sawyer Lab)
Issue:    #27 — heatmap comparing all values across scripts

Generates two outputs:
    1. heatmap.html  — interactive heatmap (open in browser)
    2. heatmap.png   — static image for presentations/report

Usage:
    python generate_heatmap.py
    Place in repo root or final_model/ — auto-detects CSV paths.
===============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("./viz_outputs")

SUPPORT_NAME_MAP = {
    "cyrillic": "Cyrillic", "japanese": "Katakana",
    "devanagari": "Devanagari", "arabic": "Arabic",
    "telugu": "Telugu", "tamil": "Tamil", "bengali": "Bengali",
    "chinese-traditional": "Han", "chinese-simplified": "Han",
    "chinese-hongkong": "Han",
}

TARGET_SCRIPTS = ["Devanagari", "Arabic", "Bengali", "Tamil",
                  "Telugu", "Han", "Katakana", "Cyrillic"]


# ===========================================================================
# Data Loading
# ===========================================================================

def find_repo_root() -> Path:
    here = Path(__file__).parent.resolve()
    for _ in range(6):
        if (here / "exposure_research").exists():
            return here
        here = here.parent
    return Path.home() / "TheScriptGap_clean"


def load_all_data(repo: Path) -> pd.DataFrame:
    """Load and merge all four indices."""

    # Helper to find a file in repo OR same directory as script
    script_dir = Path(__file__).parent.resolve()

    def find_csv(repo_rel: str, filename: str) -> Path:
        # Try repo path first
        p = repo / repo_rel
        if p.exists():
            return p
        # Fall back to script directory
        p2 = script_dir / filename
        if p2.exists():
            return p2
        raise FileNotFoundError(f"Cannot find {filename} — tried {p} and {p2}")

    # Exposure
    exp_path = find_csv("exposure_research/output/exposure_filtered_results.csv",
                        "exposure_filtered_results.csv")
    exp = pd.read_csv(exp_path, names=["script", "exposure"], header=0)
    exp = exp[exp["script"].isin(TARGET_SCRIPTS)].copy()
    logger.info(f"  Exposure: {exp_path.name}")

    # Support
    sup_path = find_csv("support_research/output/google_support_toplist.csv",
                        "google_support_toplist.csv")
    sup_raw = pd.read_csv(sup_path, names=["script_raw", "count"], header=0)
    records = [{"script": SUPPORT_NAME_MAP[r.script_raw], "support": r.count}
               for r in sup_raw.itertuples() if r.script_raw in SUPPORT_NAME_MAP]
    sup = pd.DataFrame(records).groupby("script", as_index=False)["support"].sum()
    logger.info(f"  Support: {sup_path.name}")

    # Complexity (from similarity pipeline)
    sim_path = find_csv("similarity_research/similarity_index_summary.csv",
                        "similarity_index_summary.csv")
    sim = pd.read_csv(sim_path)
    sim["complexity"] = 1.0 - sim["similarity_S"]
    sim = sim[["script", "complexity"]]
    logger.info(f"  Complexity: {sim_path.name}")

    # Diversity (100-glyph preferred)
    div_paths = [
        repo / "similarity_research/diversity_research/vit_outputs_100/diversity_index_summary.csv",
        repo / "similarity_research/diversity_research/vit100_outputs/diversity_index_summary.csv",
        repo / "similarity_research/diversity_research/vit_outputs/diversity_index_summary.csv",
        script_dir / "diversity_index_summary.csv",
    ]
    div = None
    for p in div_paths:
        if p.exists():
            div = pd.read_csv(p)[["script", "diversity_index"]]
            logger.info(f"  Diversity: {p.name}")
            break
    if div is None:
        raise FileNotFoundError("Cannot find diversity_index_summary.csv")

    # Merge
    master = exp.merge(sup, on="script").merge(sim, on="script").merge(div, on="script")
    master = master[master["script"].isin(TARGET_SCRIPTS)].reset_index(drop=True)

    # Log-scale exposure and support for visualization
    master["log_exposure"] = np.log10(master["exposure"])
    master["log_support"] = np.log10(master["support"])

    # Normalize all to 0-1 for heatmap
    for col in ["log_exposure", "log_support", "complexity", "diversity_index"]:
        cmin, cmax = master[col].min(), master[col].max()
        master[f"{col}_norm"] = (master[col] - cmin) / (cmax - cmin) if cmax > cmin else 0.5

    # Gap score: high exposure, low support, high complexity, low diversity
    master["sss"] = (
        master["log_exposure_norm"]
        - master["log_support_norm"] * 1.5
        + master["complexity_norm"] * 2.0
        - master["diversity_index_norm"]
    )
    # Normalize gap score to 0-1
    gs_min, gs_max = master["sss"].min(), master["sss"].max()
    master["sss_norm"] = (master["sss"] - gs_min) / (gs_max - gs_min)

    # Sort by gap score (most underserved first)
    master = master.sort_values("sss_norm", ascending=False).reset_index(drop=True)

    logger.info(f"Loaded {len(master)} scripts")
    return master


# ===========================================================================
# HTML Heatmap (interactive)
# ===========================================================================

def generate_html_heatmap(master: pd.DataFrame) -> str:
    """Generate an interactive HTML heatmap with tooltips."""

    scripts = master["script"].tolist()
    n = len(scripts)

    metrics = [
        ("log_exposure_norm",    "Web Exposure",   "Higher = more readers",      "#3B8BD4"),
        ("log_support_norm",     "Font Support",   "Higher = more fonts",        "#1D9E75"),
        ("complexity_norm",      "Complexity",     "Higher = harder to engineer","#E24B4A"),
        ("diversity_index_norm", "Diversity",      "Higher = more visual choice","#EF9F27"),
        ("sss_norm",       "SSS",      "Higher = more underserved",  "#7F77DD"),
    ]

    raw_cols = {
        "log_exposure_norm":    ("exposure",       lambda v: f"{int(v):,}"),
        "log_support_norm":     ("support",        lambda v: f"{int(v)}"),
        "complexity_norm":      ("complexity",     lambda v: f"{v:.3f}"),
        "diversity_index_norm": ("diversity_index",lambda v: f"{v:.3f}"),
        "sss_norm":       ("sss",      lambda v: f"{v:.3f}"),
    }

    # Build cell data as JSON
    rows = []
    for _, row in master.iterrows():
        row_data = {"script": row["script"], "cells": []}
        for norm_col, label, desc, color in metrics:
            val = row[norm_col]
            raw_col, fmt = raw_cols[norm_col]
            raw_val = fmt(row[raw_col])
            row_data["cells"].append({
                "norm": round(val, 4),
                "raw": raw_val,
                "label": label,
                "color": color,
            })
        rows.append(row_data)

    import json
    data_json = json.dumps(rows)
    metrics_json = json.dumps([
        {"label": m[1], "desc": m[2], "color": m[3]} for m in metrics
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Script Underservedness Heatmap</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117;
    color: #e8e6df;
    padding: 32px;
    min-height: 100vh;
  }}
  h1 {{
    font-size: 22px;
    font-weight: 500;
    margin-bottom: 6px;
    color: #f0ede6;
  }}
  .subtitle {{
    font-size: 13px;
    color: #888;
    margin-bottom: 32px;
  }}
  .heatmap-wrap {{
    overflow-x: auto;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    min-width: 680px;
  }}
  th {{
    font-size: 12px;
    font-weight: 500;
    color: #aaa;
    padding: 8px 12px;
    text-align: center;
    letter-spacing: 0.03em;
    white-space: nowrap;
  }}
  th.script-col {{
    text-align: left;
    min-width: 120px;
  }}
  td.script-name {{
    font-size: 14px;
    font-weight: 500;
    color: #e8e6df;
    padding: 6px 12px;
    white-space: nowrap;
  }}
  td.cell {{
    padding: 4px 6px;
    text-align: center;
    cursor: default;
    position: relative;
  }}
  .cell-inner {{
    border-radius: 6px;
    padding: 10px 8px;
    font-size: 12px;
    font-weight: 500;
    transition: transform 0.15s, box-shadow 0.15s;
    user-select: none;
  }}
  .cell-inner:hover {{
    transform: scale(1.08);
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    z-index: 10;
    position: relative;
  }}
  .tooltip {{
    display: none;
    position: fixed;
    background: #1e2030;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #e8e6df;
    z-index: 1000;
    pointer-events: none;
    max-width: 200px;
  }}
  .tooltip.show {{ display: block; }}
  .tooltip-metric {{ font-weight: 600; margin-bottom: 4px; }}
  .tooltip-raw {{ color: #aaa; }}
  .legend {{
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    margin-top: 24px;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #aaa;
  }}
  .legend-dot {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
  }}
  .tier-badge {{
    display: inline-block;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 10px;
    margin-left: 8px;
    vertical-align: middle;
  }}
  tr:hover td.script-name {{ color: #fff; }}
  .note {{
    margin-top: 24px;
    font-size: 12px;
    color: #555;
  }}
</style>
</head>
<body>

<h1>Script Servedness Score (SSS)</h1>
<p class="subtitle">
  TRC / Monotype / Sawyer Lab · All values normalized 0–1 · Sorted by SSS (most underserved first)
</p>

<div class="heatmap-wrap">
  <table id="heatmap">
    <thead>
      <tr>
        <th class="script-col">Script</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<div class="legend" id="legend"></div>
<p class="note">Hover over any cell for the raw value. SSS = Script Servedness Score — combines all four indices.</p>

<div class="tooltip" id="tooltip"></div>

<script>
const data = {data_json};
const metrics = {metrics_json};

function hexToRgb(hex) {{
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return {{r,g,b}};
}}

function cellColor(norm, hexColor) {{
  const rgb = hexToRgb(hexColor);
  const alpha = 0.15 + norm * 0.75;
  const lightness = 1 - norm * 0.6;
  return {{
    bg: `rgba(${{rgb.r}},${{rgb.g}},${{rgb.b}},${{alpha}})`,
    text: norm > 0.55 ? `rgba(${{rgb.r*0.4}},${{rgb.g*0.4}},${{rgb.b*0.4}},1)` : `rgba(${{Math.min(255,rgb.r+80)}},${{Math.min(255,rgb.g+80)}},${{Math.min(255,rgb.b+80)}},0.9)`,
  }};
}}

// Build header
const thead = document.querySelector('thead tr');
metrics.forEach(m => {{
  const th = document.createElement('th');
  th.textContent = m.label;
  th.title = m.desc;
  thead.appendChild(th);
}});

// Build rows
const tbody = document.getElementById('tbody');
data.forEach((row, ri) => {{
  const tr = document.createElement('tr');

  // Script name cell with tier badge
  const tdName = document.createElement('td');
  tdName.className = 'script-name';
  const gapNorm = row.cells[4].norm;
  const tier = gapNorm > 0.6 ? ['Underserved','#E24B4A','#FCEBEB'] :
               gapNorm > 0.3 ? ['Moderate','#EF9F27','#FAEEDA'] :
                               ['Well served','#1D9E75','#E1F5EE'];
  tdName.innerHTML = `${{row.script}}<span class="tier-badge" style="background:${{tier[2]}};color:${{tier[1]}}">${{tier[0]}}</span>`;
  tr.appendChild(tdName);

  // Metric cells
  row.cells.forEach((cell, ci) => {{
    const td = document.createElement('td');
    td.className = 'cell';
    const colors = cellColor(cell.norm, metrics[ci].color);
    const inner = document.createElement('div');
    inner.className = 'cell-inner';
    inner.style.background = colors.bg;
    inner.style.color = colors.text;
    inner.textContent = (cell.norm * 100).toFixed(0) + '%';

    // Tooltip
    inner.addEventListener('mousemove', (e) => {{
      const tt = document.getElementById('tooltip');
      tt.innerHTML = `<div class="tooltip-metric" style="color:${{metrics[ci].color}}">${{cell.label}}</div>
        <div>Normalized: <strong>${{(cell.norm*100).toFixed(1)}}%</strong></div>
        <div class="tooltip-raw">Raw value: ${{cell.raw}}</div>
        <div class="tooltip-raw" style="margin-top:4px;font-style:italic">${{metrics[ci].desc}}</div>`;
      tt.classList.add('show');
      tt.style.left = (e.clientX + 14) + 'px';
      tt.style.top = (e.clientY - 10) + 'px';
    }});
    inner.addEventListener('mouseleave', () => {{
      document.getElementById('tooltip').classList.remove('show');
    }});

    td.appendChild(inner);
    tr.appendChild(td);
  }});

  tbody.appendChild(tr);
}});

// Legend
const legend = document.getElementById('legend');
metrics.forEach(m => {{
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.innerHTML = `<div class="legend-dot" style="background:${{m.color}}"></div>${{m.label}}: ${{m.desc}}`;
  legend.appendChild(item);
}});
</script>
</body>
</html>"""

    return html


# ===========================================================================
# Static PNG heatmap using matplotlib
# ===========================================================================

def generate_png_heatmap(master: pd.DataFrame, output_path: Path):
    """Generate a static PNG heatmap for reports and slides."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        logger.warning("matplotlib not found — skipping PNG. Install with: pip install matplotlib")
        return

    scripts = master["script"].tolist()
    metrics_info = [
        ("log_exposure_norm",    "Web\nExposure",  "#3B8BD4"),
        ("log_support_norm",     "Font\nSupport",  "#1D9E75"),
        ("complexity_norm",      "Complexity",     "#E24B4A"),
        ("diversity_index_norm", "Diversity",      "#EF9F27"),
        ("sss_norm",       "Gap\nScore",     "#7F77DD"),
    ]

    matrix = np.array([
        [master.loc[master["script"] == s, col].values[0] for col, _, _ in metrics_info]
        for s in scripts
    ])

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    n_scripts, n_metrics = matrix.shape

    for i, script in enumerate(scripts):
        for j, (col, label, color) in enumerate(metrics_info):
            val = matrix[i, j]

            # Parse hex color
            rgb = mcolors.to_rgb(color)
            alpha = 0.15 + val * 0.75

            rect = FancyBboxPatch(
                (j + 0.05, n_scripts - i - 1 + 0.05),
                0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=(*rgb, alpha),
                edgecolor="none",
            )
            ax.add_patch(rect)

            text_color = "white" if val < 0.55 else mcolors.to_hex([c * 0.4 for c in rgb])
            ax.text(j + 0.5, n_scripts - i - 0.5,
                    f"{val*100:.0f}%",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color=text_color)

    # Script labels
    for i, script in enumerate(scripts):
        ax.text(-0.15, n_scripts - i - 0.5, script,
                ha="right", va="center",
                fontsize=11, color="#e8e6df", fontweight="500")

    # Metric labels
    for j, (col, label, color) in enumerate(metrics_info):
        ax.text(j + 0.5, n_scripts + 0.2, label,
                ha="center", va="bottom",
                fontsize=10, color=color, fontweight="500")

    ax.set_xlim(-2, n_metrics)
    ax.set_ylim(-0.1, n_scripts + 0.6)
    ax.axis("off")

    fig.suptitle("Script Servedness Score (SSS)",
                 fontsize=14, color="#f0ede6", y=0.98, fontweight="500")
    ax.set_title("All values normalized 0–100% · Sorted by SSS",
                 fontsize=9, color="#666", pad=4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0f1117", edgecolor="none")
    plt.close()
    logger.info(f"Saved PNG → {output_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    repo = find_repo_root()
    logger.info(f"Repo root: {repo}")

    master = load_all_data(repo)

    print("\n📊 Data loaded:\n")
    print(master[["script", "exposure", "support", "complexity",
                   "diversity_index", "sss_norm"]].to_string(index=False))

    # HTML heatmap
    html = generate_html_heatmap(master)
    html_path = OUTPUT_DIR / "heatmap.html"
    html_path.write_text(html)
    logger.info(f"Saved HTML → {html_path}")
    print(f"\nOpen in browser: {html_path.resolve()}")

    # PNG heatmap
    generate_png_heatmap(master, OUTPUT_DIR / "heatmap.png")

    # Also save the normalized data as CSV for reference
    master.to_csv(OUTPUT_DIR / "heatmap_data.csv", index=False)
    logger.info(f"Saved data → {OUTPUT_DIR / 'heatmap_data.csv'}")


if __name__ == "__main__":
    main()
