"""
Claims the white paper makes from the v1.1 diversity run (analysis/v11_checks.py).

Run:  uv run --with pytest --with pandas --with numpy pytest -q
"""

import pytest

from analysis import v11_checks

INDIC = {"Tamil", "Bengali", "Devanagari", "Telugu"}
BOUNDARY = {"Arabic", "Han"}


@pytest.fixture(scope="module")
def res():
    return v11_checks.main(out=None, verbose=False)


def test_tiers(res):
    v = res["versions"]
    assert set(v.index[v.tier_v1_1 == "Underserved"]) == INDIC
    assert set(v.index[v.tier_v1_1 == "Moderately served"]) == BOUNDARY
    assert set(v.index[v.tier_v1_1 == "Well served"]) == {"Cyrillic", "Latin", "Katakana"}


def test_exposure_barely_matters(res):
    s = res["summary"]
    assert s["v1_1_tiers_same_without_exposure"] == 9
    assert s["v1_1_max_abs_sss_change_without_exposure"] < 0.01


def test_noise_keeps_indic_bottom_four_and_only_boundary_tiers_move_at_10pct(res):
    n = res["noise"]
    assert n.indic_bottom_four.all()
    assert n.max_rank_move.max() <= 2
    ten = n[(n["size"] == 0.10) & (n.n_tier_changes > 0)]
    moved = {c.split(" ")[0] for row in ten.tier_changes for c in row.split("; ")}
    assert moved <= BOUNDARY


def test_encoder_agrees_on_ranking(res):
    s = res["summary"]
    assert s["encoder_spearman_mcd_all9"] > 0.95
    assert s["encoder_spearman_mcd_nonlatin8"] == pytest.approx(1.0)
    e = res["encoder"]
    moved = set(e.index[e.tier_vit != e.tier_resnet])
    assert moved <= BOUNDARY


def test_latin_primary_population(res):
    s = res["summary"]
    assert s["latin_sample_latin_primary"] == 53
    assert s["latin_diversity_latin_primary"] == pytest.approx(0.533, abs=0.005)
    assert set(s["latin_primary_tiers_moved"].split(", ")) <= BOUNDARY
