"""TheScriptGap pipeline orchestrator.

This entrypoint is intentionally scoped to the current active research areas:
- support_research
- exposure_research
- data_viz

The following areas are intentionally ignored for this iteration:
- similarity_research (complexity / diversity pipelines)
- final_model (clustering / tier assignment)
- cnn / ViT model training and similarity analysis
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

AVAILABLE_STAGES = ["support", "exposure", "viz"]


def run_support_stage(force: bool = False) -> None:
    from support_research import run_support_pipeline

    print("\n=== Support stage ===")
    run_support_pipeline(force=force)


def run_exposure_stage(force: bool = False) -> None:
    from exposure_research import run_exposure_pipeline

    print("\n=== Exposure stage ===")
    run_exposure_pipeline(force=force)
    print("Exposure data is ready. You can now run the dashboard with `exposure_research/dashboard.py`.")


def run_viz_stage(force: bool = False) -> None:
    from data_viz import run_viz_pipeline

    print("\n=== Data Viz stage ===")
    run_viz_pipeline(force=force)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TheScriptGap pipeline"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=AVAILABLE_STAGES + ["all"],
        default=["all"],
        help="Pipeline stages to execute.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun stages even when output files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    requested = args.stages
    if "all" in requested:
        requested = AVAILABLE_STAGES

    # TODO: ADD A BIG QUERY SECTION

    if "support" in requested:
        run_support_stage(force=args.force)
    if "exposure" in requested:
        run_exposure_stage(force=args.force)
    # TODO: Add similarity and complexity, and CNN clustering stages
    if "viz" in requested:
        run_viz_stage(force=args.force)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
