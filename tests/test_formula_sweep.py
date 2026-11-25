"""
Tests pinning the claims in data/analysis/formula_sweep.md to the committed data.

Run:  uv run --with pytest pytest -q tests/test_formula_sweep.py
"""

import numpy as np
import pytest

from analysis.formula_sweep import (
    CLAIMS, composite, demand_exponent_check, demand_free_edge, holding_interval,
    load_inputs, run_sweep, standardized_terms, summarize_by_demand,
)


@pytest.fixture(scope="module")
def sweep():
    base, mcd = load_inputs()          # also asserts the baseline matches canonical
    Z = standardized_terms(base, mcd)
    grid = run_sweep(base, Z, step=0.05)
    return base, Z, grid


def test_even_split_matches_the_tiers(sweep):
    """An even support/diversity split gives the SSS's bottom six in the same
    order; only the order inside the well-served three differs."""
    base, Z, _ = sweep
    order = list(composite(Z, 0.5, 0.5, 0.0).sort_values(kind="stable").index)
    assert order[:6] == list(base.index[:6])
    assert set(order[-3:]) == set(base.index[-3:])


def test_demand_term_is_inert_in_shipped_formula(sweep):
    base, _, _ = sweep
    A = demand_exponent_check(base)
    assert A["log_demand_spread_pct"] < 2.0
    # Demand first reorders anything only at gamma 9, and then only a pair of
    # well-served scripts (Latin and Katakana): no tier membership changes.
    assert A["first_reordering_gamma"] is None or A["first_reordering_gamma"] >= 9
    if A["first_reordering_gamma"] is not None:
        assert set(A["first_reordered_pair"].split(" and ")) <= {"Cyrillic", "Latin", "Katakana"}


def test_claims_hold_across_wide_support_diversity_band(sweep):
    _, _, grid = sweep
    edge = demand_free_edge(grid)
    band = holding_interval(edge, edge[list(CLAIMS)].all(axis=1))
    assert band is not None
    assert band[0] <= 0.3 + 1e-9 and band[1] >= 0.7 - 1e-9
    indic = holding_interval(edge, edge["indic_bottom4"])
    assert indic[0] <= 1e-9 and indic[1] >= 0.7 - 1e-9


def test_large_demand_weight_breaks_claims(sweep):
    _, _, grid = sweep
    by_demand = summarize_by_demand(grid)
    heavy = by_demand[by_demand["w_demand"] >= 0.5]
    assert (heavy[list(CLAIMS)] == 0).all().all()
    assert np.isclose(heavy["w_demand"].min(), 0.5)
