"""TheScriptGap pipeline orchestrator.

The pipeline is organized around one idea: the Script Servedness Score (SSS)
measures an OUTCOME, how well a script's readers are served by the typography
available to them. The signals that feed it are demand side and supply side:

    Servedness signals (feed the SSS):
        support    Google Fonts family counts          (supply)
        exposure   web usage / reader demand            (demand)
        diversity  visual variety of available fonts    (real choice, an outcome)

Complexity is handled separately, on purpose. It measures creation difficulty,
which is a CAUSE of under-service, not a dimension of it. A script is no better
served because its fonts were easy to build and no worse served because they
were hard. Han is the clear case: expensive to build, well served anyway. So
complexity does not enter the SSS. It runs as its own stage and feeds the
prioritization lens, which answers a different question: for the scripts that
are underserved, how big is the lift to close the gap.

    Prioritization input (NOT in the SSS):
        complexity  per-script creation difficulty

Note: which signals the SSS combines lives in data_viz. This file only
groups the stages by the role they play, so the grouping above is intent, not
the scoring math itself.

Tier assignment / clustering is paused until the servedness signals are locked,
since removing complexity changes what separates the tiers. It is intentionally
not wired in here yet.

Run examples:
    python pipeline.py                          # servedness pipeline + viz
    python pipeline.py --stages exposure        # a single stage
    python pipeline.py --stages all complexity  # servedness + prioritization data
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Signals that feed the Script Servedness Score.
SERVEDNESS_STAGES = ["support", "exposure", "diversity"]

# Every stage this orchestrator can run. complexity is included so it stays
# runnable, but it is deliberately NOT part of SERVEDNESS_STAGES, so "all"
# does not pull it in.
AVAILABLE_STAGES = SERVEDNESS_STAGES + ["complexity", "viz"]

# Canonical execution order. viz runs last because it consumes the others.
STAGE_ORDER = ["support", "exposure", "diversity", "complexity", "viz"]


def run_support_stage(force: bool = False) -> None:
    from support_research import run_support_pipeline
    print("\n=== Support stage (supply: Google Fonts families) ===")
    run_support_pipeline(force=force)


def run_exposure_stage(force: bool = False) -> None:
    from exposure_research import run_exposure_pipeline
    print("\n=== Exposure stage (demand: web usage) ===")
    run_exposure_pipeline(force=force)
    print("Exposure data is ready. Dashboard: exposure_research/dashboard.py")


def run_diversity_stage(force: bool = False) -> None:
    # Visual variety of the available fonts (ViT embeddings). This is the signal
    # Lou's code currently exposes as "similarity"; named "diversity" here
    # because that is what it contributes to the SSS. It still imports the same
    # module, so update the import if you rename the module.
    from new_similarity import run_similarity_pipeline
    print("\n=== Diversity stage (visual variety, feeds the SSS) ===")
    run_similarity_pipeline(force=force)


def run_complexity_stage(force: bool = False) -> None:
    # Creation difficulty. Feeds the prioritization lens, NOT the SSS.
    from complexity import run_complexity_pipeline
    print("\n=== Complexity stage (prioritization input, not in the SSS) ===")
    run_complexity_pipeline(force=force)


def run_viz_stage(force: bool = False) -> None:
    from data_viz import run_viz_pipeline
    print("\n=== Data Viz stage (heatmap) ===")
    run_viz_pipeline(force=force)


STAGE_RUNNERS = {
    "support": run_support_stage,
    "exposure": run_exposure_stage,
    "diversity": run_diversity_stage,
    "complexity": run_complexity_stage,
    "viz": run_viz_stage,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TheScriptGap pipeline")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=AVAILABLE_STAGES + ["all"],
        default=["all"],
        help=(
            "Stages to run. 'all' runs the servedness pipeline plus viz "
            "(support, exposure, diversity, viz). complexity is separate and "
            "must be named, e.g. '--stages complexity' or '--stages all "
            "complexity'. Choices: " + ", ".join(AVAILABLE_STAGES + ["all"])
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun selected stages even if their output files already exist.",
    )
    return parser.parse_args()


def resolve_stages(requested: list[str]) -> list[str]:
    """Expand 'all' to the servedness pipeline plus viz, union any explicitly
    named stages, and return them in canonical run order."""
    wanted = set(requested)
    if "all" in wanted:
        wanted.discard("all")
        wanted.update(SERVEDNESS_STAGES + ["viz"])
    return [stage for stage in STAGE_ORDER if stage in wanted]


def main() -> None:
    args = parse_args()
    stages = resolve_stages(args.stages)

    print(f"Running stages: {', '.join(stages)}")

    # TODO: wire the BigQuery / CrUX pull into the exposure stage.
    for stage in stages:
        STAGE_RUNNERS[stage](force=args.force)

    # TODO: re-add tier assignment / clustering once the servedness signals are
    # locked. It depends on which signals feed the SSS, so it stays out until
    # that is decided.
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()