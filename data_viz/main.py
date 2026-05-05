"""Data visualization pipeline wrapper.

This module exposes a lightweight entrypoint for generating the final heatmap
visualization from the existing data-viz implementation.

Inputs:
    - viz_outputs/heatmap.html (generated)

Responsibilities:
    - ensure visualization output directory exists
    - invoke the heatmap generation module
    - optionally skip rerun when output already exists

Scope:
    - focuses on visualization orchestration only
"""

from pathlib import Path
from paths import VIZ_DATA_DIR
from data_viz.generate_heatmap import main as generate_heatmap_main

OUTPUT_DIR = VIZ_DATA_DIR
OUTPUT_HTML = OUTPUT_DIR / "heatmap.html"


def run_viz_pipeline(force: bool = False) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_HTML.exists() and not force:
        print(f"Visualization output already exists: {OUTPUT_HTML} (use --force to rerun)")
        return OUTPUT_HTML

    print("Running data visualization pipeline...")
    generate_heatmap_main()
    print(f"Completed visualization. Output written to: {OUTPUT_HTML}")
    return OUTPUT_HTML


def main() -> None:
    run_viz_pipeline(force=True)


if __name__ == "__main__":
    main()
