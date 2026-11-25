"""
Claims about the v1.1 Infrastructure Complexity Index (GSUB closure).

Run:  uv run --with pytest --with pandas --with numpy pytest -q

Pinned on the committed per-family measurements in data/complexity_v11/, so
the paper's methods and results text cannot drift from the data.
"""

import pandas as pd
import pytest

from paths import DATA_ROOT
from complexity.aggregate_closure import main as aggregate

V11 = DATA_ROOT / "complexity_v11"
NO_SHAPING = ["Latin", "Cyrillic", "Han", "Katakana"]
SHAPING = ["Telugu", "Devanagari", "Bengali", "Arabic", "Tamil"]


@pytest.fixture(scope="module")
def rows():
    return pd.read_csv(V11 / "per_family_script.csv")


@pytest.fixture(scope="module")
def result():
    return aggregate(str(V11 / "per_family_script.csv"),
                     str(DATA_ROOT / "complexity" / "complexity_index_summary.csv"),
                     out_dir=None, verbose=False)


def test_every_representative_measured(rows):
    assert set(rows.status) <= {"ok", "below_min_cps"}
    assert (rows.status == "ok").sum() >= 2470


def test_original_formula_overcounts_small_scripts_in_cjk_fonts(rows):
    """v0 (maxp.numGlyphs / script codepoints): Kosugi Katakana ~84x, the bug
    found in April 2026. The measured closure for the same font is 1.0."""
    k = rows[(rows.family_dir == "kosugi") & (rows.script == "Katakana")].iloc[0]
    assert k.E_v0 > 80
    assert k.E == pytest.approx(1.0)


def test_closure_is_script_specific_v1_was_not(result):
    _, s = result
    assert s["v1_E_share_files_within_10pct"] > 0.95       # v1: one number per file
    assert s["closure_E_share_files_within_10pct"] < 0.70  # v1.1: differs by script


def test_no_shaping_scripts_have_unit_expansion(result):
    agg, _ = result
    for script in NO_SHAPING:
        assert agg.loc[script, "E"] == pytest.approx(1.0, abs=0.02), script


def test_shaping_scripts_expand(result):
    agg, _ = result
    assert agg.loc[SHAPING, "E"].min() > 2.0
    assert agg.loc[SHAPING, "E"].min() > agg.loc[NO_SHAPING, "E"].max()


def test_composite_tracks_v1_and_is_weight_robust_at_the_extremes(result):
    agg, s = result
    assert s["rho_C_v1_vs_v11"] >= 0.85
    assert s["weight_grid_rho_median"] >= 0.9
    assert agg.loc["Katakana", "rank_C"] == 9   # least complex under default weights
