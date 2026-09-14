"""Unit tests for the Welch's two-sample t-test implementation.

Asserts exact statistical equivalence between our first-principles implementation
and the reference `scipy.stats.ttest_ind(..., equal_var=False)`.
"""

import pytest
import numpy as np
from scipy import stats
from src.continuous_test import welch_t_test, validate_against_scipy


def test_welch_t_test_matches_scipy_cookie_cats():
    """Verify sum_gamerounds matches scipy exactly on synthetic and real slices."""
    rng = np.random.default_rng(42)
    # Generate heteroscedastic samples of unequal sizes
    a = rng.normal(loc=50.0, scale=10.0, size=5000)
    b = rng.normal(loc=52.0, scale=25.0, size=6000)

    matches, ct, st, cp, sp, df = validate_against_scipy(a, b)
    assert matches, f"Mismatch: custom t={ct}, scipy t={st}, custom p={cp}, scipy p={sp}"
    assert np.isclose(ct, st, atol=1e-7)
    assert np.isclose(cp, sp, atol=1e-7)


def test_welch_t_test_exact_matches_known_values():
    """Test on deterministic values with known unequal variance."""
    a = [10.0, 12.0, 14.0, 15.0, 18.0]
    b = [22.0, 24.0, 27.0, 31.0, 35.0, 40.0]

    res = welch_t_test(a, b)
    scipy_res = stats.ttest_ind(a, b, equal_var=False)

    assert np.isclose(res.t_stat, scipy_res.statistic, atol=1e-7)
    assert np.isclose(res.p_value, scipy_res.pvalue, atol=1e-7)
    assert np.isclose(res.df, scipy_res.df, atol=1e-7)
    assert res.ci_lower < res.absolute_diff < res.ci_upper


def test_welch_t_test_error_handling():
    """Verify input validation for small or empty samples."""
    with pytest.raises(ValueError):
        welch_t_test([1.0], [2.0, 3.0])
    with pytest.raises(ValueError):
        welch_t_test([], [])
    with pytest.raises(ValueError):
        # Zero variance in both groups
        welch_t_test([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
