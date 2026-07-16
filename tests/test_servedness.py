"""
Regression + research-claim tests for the Script Servedness Score (v1.0).

Run:  uv run --with pytest --with pandas --with numpy pytest -q

These tests pin the committed result to the documented formula (so the heatmap
output and the score can't silently drift) and assert the robustness claims made
in the README are actually true of the committed data.
"""

import pandas as pd
import pytest

from paths import DATA_ROOT
from analysis.scoring import compute_servedness, assign_tier
from analysis.robustness import spearman, load_diversity_sources, load_support, load_exposure

CANONICAL = DATA_ROOT / "final" / "script_servedness.csv"
SIMILARITY = DATA_ROOT / "similarity" / "similarity_results.csv"
UNDERSERVED_CLUSTER = {"Tamil", "Bengali", "Devanagari", "Telugu"}

VIT = "ViT-B/16 (100-glyph)"
RESNET = "ResNet-50 (100-glyph)"
CLASSICAL = "Classical CV"


@pytest.fixture(scope="module")
def support():
    return load_support()


@pytest.fixture(scope="module")
def diversity():
    return pd.read_csv(SIMILARITY).set_index("script")["diversity_index"]


@pytest.fixture(scope="module")
def exposure():
    return load_exposure()


@pytest.fixture(scope="module")
def canonical():
    return pd.read_csv(CANONICAL).set_index("script")


# --- 1. the committed result matches the documented formula -----------------

def test_canonical_output_matches_formula(support, diversity, exposure, canonical):
    """data/final/script_servedness.csv must equal compute_servedness() recomputed
    from the raw committed inputs — ties generate_heatmap's output to the formula."""
    got = compute_servedness(support, diversity, exposure)
    assert set(got.index) == set(canonical.index)
    merged = got.join(canonical, rsuffix="_canon")
    pd.testing.assert_series_equal(
        merged["servedness_score"], merged["servedness_score_canon"],
        check_names=False, atol=1e-9,
    )
    assert (merged["tier"] == merged["tier_canon"]).all()


def test_tier_thresholds():
    s = pd.Series({"a": 0.0, "b": 0.3, "c": 0.31, "d": 0.6, "e": 0.61})
    t = assign_tier(s)
    assert t["a"] == "Underserved" and t["b"] == "Underserved"
    assert t["c"] == "Moderately served" and t["d"] == "Moderately served"
    assert t["e"] == "Well served"


# --- 2. the headline conclusion --------------------------------------------

def test_underserved_cluster_is_bottom_four(support, diversity, exposure):
    got = compute_servedness(support, diversity, exposure)
    assert set(got.index[:4]) == UNDERSERVED_CLUSTER


def test_latin_is_best_served(support, diversity, exposure):
    got = compute_servedness(support, diversity, exposure)
    assert got.index[-1] == "Latin"
    assert got["tier"].iloc[-1] == "Well served"


# --- 3. robustness claims (must match the README) ---------------------------

def test_diversity_robust_vit_vs_resnet():
    div = load_diversity_sources().dropna()
    rho = spearman(div[VIT].tolist(), div[RESNET].tolist())
    assert rho > 0.9, f"ViT vs ResNet Spearman {rho:.3f} should be > 0.9"


def test_classical_cv_not_a_robust_substitute():
    div = load_diversity_sources().dropna()
    rho = spearman(div[VIT].tolist(), div[CLASSICAL].tolist())
    assert rho < 0.3, f"classical CV should diverge from ViT, got {rho:.3f}"


def test_tiers_stable_under_resnet(support, exposure):
    div = load_diversity_sources()
    base = compute_servedness(support, div[VIT].dropna(), exposure)["tier"]
    alt = compute_servedness(support, div[RESNET].dropna(), exposure)["tier"]
    shared = [s for s in base.index if s in alt.index]
    agree = sum(base[s] == alt[s] for s in shared)
    assert agree >= len(shared) - 1, f"tiers should be near-identical under ResNet, got {agree}/{len(shared)}"


# --- 4. the demand confound is real (demand is the weakest score input) ------

def test_demand_bundling_confound_is_severe():
    from exposure_research.demand_audit import load_source, confound_breakdown
    audit = confound_breakdown(load_source()).set_index("script")
    for script in ["Tamil", "Telugu", "Bengali", "Devanagari"]:
        share = audit.loc[script, "pct_from_multiscript_bundlers"]
        assert share > 0.9, f"{script} bundler share {share:.2f} should be > 0.9"
