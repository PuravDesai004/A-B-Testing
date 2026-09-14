"""Unit tests for the two-proportion z-test implementation.

Asserts exact statistical equivalence between our first-principles implementation
and the reference `statsmodels.stats.proportion.proportions_ztest`.
"""

import pytest
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from src.proportion_test import proportion_z_test, validate_against_statsmodels


def test_proportion_z_test_matches_statsmodels_cookie_cats():
    """Verify retention metrics match statsmodels on real Cookie Cats values."""
    # retention_1: gate_30 vs gate_40
    matches_r1, cz1, sz1, cp1, sp1 = validate_against_statsmodels(
        successes_a=20034, n_a=44699, successes_b=20119, n_b=45489
    )
    assert matches_r1, f"Mismatch in retention_1: custom z={cz1}, sm z={sz1}, custom p={cp1}, sm p={sp1}"
    assert np.isclose(cz1, sz1, atol=1e-7)
    assert np.isclose(cp1, sp1, atol=1e-7)

    # retention_7: gate_30 vs gate_40
    matches_r7, cz7, sz7, cp7, sp7 = validate_against_statsmodels(
        successes_a=8501, n_a=44699, successes_b=8279, n_b=45489
    )
    assert matches_r7, f"Mismatch in retention_7: custom z={cz7}, sm z={sz7}, custom p={cp7}, sm p={sp7}"
    assert np.isclose(cz7, sz7, atol=1e-7)
    assert np.isclose(cp7, sp7, atol=1e-7)


def test_proportion_z_test_synthetic_edge_cases():
    """Test synthetic balanced, asymmetric, and small sample scenarios."""
    # Test identical proportions (null holds exactly)
    res_identical = proportion_z_test(50, 100, 50, 100)
    assert np.isclose(res_identical.z_stat, 0.0)
    assert np.isclose(res_identical.p_value, 1.0)
    assert np.isclose(res_identical.absolute_diff, 0.0)

    # Test extreme divergence
    count = [90, 10]
    nobs = [100, 100]
    sm_z, sm_p = proportions_ztest(count, nobs)
    res_divergent = proportion_z_test(90, 100, 10, 100)
    assert np.isclose(res_divergent.z_stat, sm_z, atol=1e-7)
    assert np.isclose(res_divergent.p_value, sm_p, atol=1e-7)


def test_proportion_z_test_invalid_inputs():
    """Test boundary validation and error handling."""
    with pytest.raises(ValueError):
        proportion_z_test(successes_a=150, n_a=100, successes_b=50, n_b=100)
    with pytest.raises(ValueError):
        proportion_z_test(successes_a=50, n_a=0, successes_b=50, n_b=100)
    with pytest.raises(ValueError):
        proportion_z_test(successes_a=-5, n_a=100, successes_b=50, n_b=100)
