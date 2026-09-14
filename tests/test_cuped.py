"""Unit tests for Phase 3 CUPED variance reduction engine."""

import pytest
import numpy as np
from scipy import stats
from src.synthetic_cuped_data import generate_cuped_dataset
from src.cuped import (
    compute_covariance,
    compute_variance,
    compute_correlation,
    compute_optimal_theta,
    apply_cuped_adjustment,
    run_cuped_analysis,
    validate_theta_against_scipy,
)


def test_covariance_variance_correlation_exact():
    """Verify custom basic statistical moments match numpy and scipy reference implementations."""
    rng = np.random.default_rng(99)
    x = rng.normal(10.0, 3.0, size=500)
    y = 2.5 * x + rng.normal(0.0, 2.0, size=500)

    # 1. Covariance
    custom_cov = compute_covariance(x, y)
    np_cov = np.cov(x, y, ddof=1)[0, 1]
    assert np.isclose(custom_cov, np_cov, atol=1e-7)

    # 2. Variance
    custom_var_x = compute_variance(x)
    np_var_x = np.var(x, ddof=1)
    assert np.isclose(custom_var_x, np_var_x, atol=1e-7)

    # 3. Correlation
    custom_corr = compute_correlation(x, y)
    scipy_corr, _ = stats.pearsonr(x, y)
    assert np.isclose(custom_corr, scipy_corr, atol=1e-7)


def test_theta_matches_scipy_ols():
    """Verify optimal theta coefficient exactly matches OLS regression slope from scipy."""
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y = np.array([12.0, 22.0, 28.0, 42.0, 46.0])

    matches, c_theta, s_slope = validate_theta_against_scipy(x, y)
    assert matches, f"Theta mismatch: custom={c_theta}, scipy={s_slope}"
    assert np.isclose(c_theta, s_slope, atol=1e-7)


def test_cuped_unbiased_mean_preservation():
    """Verify that CUPED preserves the sample mean of the outcome metric."""
    dataset = generate_cuped_dataset(num_users=3000, true_tau=2.0, target_rho=0.70, random_seed=12)
    x = dataset.df["pre_rounds"].values
    y = dataset.df["post_rounds"].values

    y_cuped, theta = apply_cuped_adjustment(x, y)

    # Mean of Y and mean of Y_cuped should be virtually identical
    assert np.isclose(np.mean(y), np.mean(y_cuped), atol=1e-5)


def test_cuped_variance_reduction_and_power():
    """Verify empirical variance reduction closely aligns with theoretical 1 - rho^2."""
    dataset = generate_cuped_dataset(num_users=5000, true_tau=1.50, target_rho=0.80, random_seed=42)
    df_out, res = run_cuped_analysis(dataset.df)

    # Check theoretical vs empirical variance reduction
    # With rho ~ 0.80, variance reduction ~ 64%
    assert 0.55 <= res.empirical_var_reduction <= 0.70
    assert np.isclose(res.empirical_var_reduction, res.theoretical_var_reduction, atol=0.03)

    # Standard error and CI width must strictly decrease
    assert res.cuped_ttest.se_diff < res.raw_ttest.se_diff
    assert res.ci_width_cuped < res.ci_width_raw
    assert res.ci_width_reduction > 0.20  # At least 20% narrower CI

    # T-statistic magnitude must increase
    assert abs(res.cuped_ttest.t_stat) > abs(res.raw_ttest.t_stat)


def test_cuped_edge_cases():
    """Test input validation for invalid or mismatched inputs."""
    with pytest.raises(ValueError):
        compute_covariance([1.0], [2.0])  # Length < 2
    with pytest.raises(ValueError):
        compute_covariance([1.0, 2.0], [1.0])  # Mismatched length
    with pytest.raises(ValueError):
        compute_optimal_theta([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])  # Var(X) == 0
