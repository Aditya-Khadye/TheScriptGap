"""
Family-bundling and population-alignment diagnostics for the diversity term.

Run:  uv run --with pytest --with pandas --with numpy pytest -q

These pin the claims the white paper's robustness section makes about the
Script Diversity Index when the unit of analysis moves from font files to
font families, and when membership moves from a codepoint threshold to Google
Fonts subset tags.
"""

import pytest

from paths import DATA_ROOT, REPO_ROOT
from analysis import family_diagnostic, population_check
from analysis.family_diagnostic import family_of

INDIC = {"Tamil", "Bengali", "Devanagari", "Telugu"}


@pytest.mark.parametrize("stem,family", [
    ("HindMadurai-Bold", "HindMadurai"),
    ("NotoSerifTamil-Italic[wdth,wght]", "NotoSerifTamil"),
    ("AnekTamil[wdth,wght]", "AnekTamil"),
    ("IBMPlexSansArabic-ExtraLight", "IBMPlexSansArabic"),
    ("Amiri-BoldItalic", "Amiri"),
    ("Lohit-Tamil", "Lohit-Tamil"),          # suffix is not a style word
])
def test_family_key(stem, family):
    assert family_of(stem) == family


@pytest.fixture(scope="module")
def family():
    return family_diagnostic.main(REPO_ROOT, out=None, verbose=False)


@pytest.fixture(scope="module")
def aligned():
    return population_check.main(REPO_ROOT, DATA_ROOT / "diagnostics" / "gf_index.csv",
                                 out=None, verbose=False)


def test_reproduces_committed_before_changes(family):
    res, sss, summary = family
    assert res["reproduces_committed"].all()
    assert summary["max_abs_sss_reproduction_error"] < 1e-6   # committed CSV is rounded


def test_ranking_survives_family_level(family):
    res, sss, summary = family
    assert summary["rho_diversity_file_vs_family_rep"] > 0.9
    assert summary["tiers_identical"] == 9
    assert set(res["family_rep"].nsmallest(4).index) == INDIC


def test_ranking_survives_population_alignment(aligned):
    res, sss, summary = aligned
    assert summary["rho_div_file_vs_aligned"] > 0.9
    assert summary["tiers_identical"] == 9
    assert set(res["family_rep_aligned"].nsmallest(4).index) == INDIC


def test_indic_order_is_not_identified(aligned):
    """Every Indic script's bootstrap rank interval includes rank 1, so the
    paper may claim the four-script cluster but not an order within it."""
    res, _, _ = aligned
    for s in INDIC:
        assert res.loc[s, "aligned_rank_ci"].startswith("1-"), s


def test_han_sample_is_mostly_off_population(aligned):
    res, _, _ = aligned
    assert res.loc["Han", "off_population_share"] > 0.75
