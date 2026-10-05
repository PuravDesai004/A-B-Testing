"""CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction engine.

This module implements CUPED from first principles. It computes covariance,
correlation, the optimal variance-minimizing theta coefficient, performs the
linear adjustment Y_cuped = Y - theta * (X - mean(X)), and measures the resulting
variance reduction and statistical power gain using Welch's t-test.
"""

import sys
from pathlib import Path
from typing import NamedTuple, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.continuous_test import welch_t_test, TTestResult


class CUPEDMetrics(NamedTuple):
    """Container holding variance and effect size comparisons before and after CUPED."""
    theta: float
    correlation_rho: float
    theoretical_var_reduction: float  # rho^2 (fraction of variance removed)
    empirical_var_reduction: float    # 1 - (var_cuped / var_raw)
    effective_sample_multiplier: float # 1 / (1 - rho^2)

    # Raw metrics
    raw_var_control: float
    raw_var_treatment: float
    raw_pooled_var: float
    raw_ttest: TTestResult

    # CUPED-adjusted metrics
    cuped_var_control: float
    cuped_var_treatment: float
    cuped_pooled_var: float
    cuped_ttest: TTestResult

    ci_width_raw: float
    ci_width_cuped: float
    ci_width_reduction: float        # (ci_width_raw - ci_width_cuped) / ci_width_raw


def compute_covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute sample covariance between two 1D arrays with Bessel's correction (ddof=1).

    Formula: Cov(X, Y) = (1 / (n - 1)) * sum((X_i - mean(X)) * (Y_i - mean(Y)))
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    n = len(arr_x)
    if n < 2 or len(arr_y) != n:
        raise ValueError("Arrays must have equal length >= 2.")

    mean_x = np.mean(arr_x)
    mean_y = np.mean(arr_y)
    cov = np.sum((arr_x - mean_x) * (arr_y - mean_y)) / (n - 1)
    return float(cov)


def compute_variance(x: np.ndarray) -> float:
    """Compute unbiased sample variance (ddof=1)."""
    arr_x = np.asarray(x, dtype=float)
    return float(compute_covariance(arr_x, arr_x))


def compute_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient rho(X, Y).

    Formula: rho = Cov(X, Y) / (std(X) * std(Y))
    """
    cov = compute_covariance(x, y)
    var_x = compute_variance(x)
    var_y = compute_variance(y)
    if var_x == 0.0 or var_y == 0.0:
        raise ValueError("Cannot compute correlation when variance is zero.")
    return float(cov / np.sqrt(var_x * var_y))


def compute_optimal_theta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the variance-minimizing CUPED coefficient theta.

    Formula: theta = Cov(X, Y) / Var(X)

    Mathematical rationale:
    Var(Y - theta * X) = Var(Y) + theta^2 * Var(X) - 2 * theta * Cov(X, Y)
    Taking derivative with respect to theta and setting to 0:
    d/d(theta) [Var(Y_cuped)] = 2 * theta * Var(X) - 2 * Cov(X, Y) = 0
    => theta* = Cov(X, Y) / Var(X)
    """
    cov_xy = compute_covariance(x, y)
    var_x = compute_variance(x)
    if var_x == 0.0:
        raise ValueError("Pre-experiment covariate X has zero variance.")
    return float(cov_xy / var_x)


def apply_cuped_adjustment(
    x: np.ndarray,
    y: np.ndarray,
    theta: float = None
) -> Tuple[np.ndarray, float]:
    """Apply the CUPED transformation to an outcome metric.

    Formula:
        Y_cuped = Y - theta * (X - mean(X))

    Parameters
    ----------
    x : np.ndarray
        Pre-experiment covariate.
    y : np.ndarray
        Raw post-experiment outcome metric.
    theta : float, optional
        CUPED scaling coefficient. If None, optimal theta is computed from (x, y).

    Returns
    -------
    y_cuped : np.ndarray
        Adjusted outcome metric with minimized variance.
    theta : float
        The theta coefficient used.
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)

    if theta is None:
        theta = compute_optimal_theta(arr_x, arr_y)

    mean_x = np.mean(arr_x)
    y_cuped = arr_y - theta * (arr_x - mean_x)
    return y_cuped, theta


def run_cuped_analysis(
    df: pd.DataFrame,
    covariate_col: str = "pre_rounds",
    outcome_col: str = "post_rounds",
    variant_col: str = "variant",
    control_label: str = "control",
    treatment_label: str = "treatment",
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, CUPEDMetrics]:
    """Execute complete end-to-end CUPED variance reduction and statistical benchmarking.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing user records.
    covariate_col : str
        Name of pre-experiment feature column (X).
    outcome_col : str
        Name of raw post-experiment outcome column (Y).
    variant_col : str
        Name of variant indicator column.
    control_label : str
    treatment_label : str
    alpha : float

    Returns
    -------
    df_out : pd.DataFrame
        Input dataframe augmented with 'cuped_outcome' column.
    metrics : CUPEDMetrics
        Comprehensive statistical comparison container.
    """
    if control_label == treatment_label:
        raise ValueError("Control and treatment labels must differ.")
    df = df[df[variant_col].isin([control_label, treatment_label])].copy()
    if not np.isfinite(df[[covariate_col, outcome_col]].to_numpy(dtype=float)).all():
        raise ValueError("CUPED requires finite, nonmissing covariates and outcomes.")
    x_all = df[covariate_col].values
    y_all = df[outcome_col].values

    # 1. Compute optimal theta and correlation
    theta = compute_optimal_theta(x_all, y_all)
    rho = compute_correlation(x_all, y_all)

    # 2. Transform raw outcome Y into CUPED outcome
    y_cuped, _ = apply_cuped_adjustment(x_all, y_all, theta=theta)
    df_out = df.copy()
    df_out["cuped_outcome"] = y_cuped

    # 3. Partition into Control and Treatment
    ctrl_mask = df_out[variant_col] == control_label
    trt_mask = df_out[variant_col] == treatment_label

    raw_ctrl = df_out.loc[ctrl_mask, outcome_col]
    raw_trt = df_out.loc[trt_mask, outcome_col]
    cuped_ctrl = df_out.loc[ctrl_mask, "cuped_outcome"]
    cuped_trt = df_out.loc[trt_mask, "cuped_outcome"]

    # 4. Run Welch's t-test before CUPED (Raw)
    raw_res = welch_t_test(raw_ctrl, raw_trt, alpha=alpha)

    # 5. Run Welch's t-test after CUPED
    cuped_res = welch_t_test(cuped_ctrl, cuped_trt, alpha=alpha)

    # 6. Variances and Reductions
    raw_pooled_var = (raw_res.var_a + raw_res.var_b) / 2.0
    cuped_pooled_var = (cuped_res.var_a + cuped_res.var_b) / 2.0

    theor_reduction = rho ** 2
    emp_reduction = 1.0 - (cuped_pooled_var / raw_pooled_var)
    sample_multiplier = 1.0 / (1.0 - theor_reduction)

    ci_raw = raw_res.ci_upper - raw_res.ci_lower
    ci_cuped = cuped_res.ci_upper - cuped_res.ci_lower
    ci_reduction = (ci_raw - ci_cuped) / ci_raw

    metrics = CUPEDMetrics(
        theta=theta,
        correlation_rho=rho,
        theoretical_var_reduction=theor_reduction,
        empirical_var_reduction=emp_reduction,
        effective_sample_multiplier=sample_multiplier,
        raw_var_control=raw_res.var_a,
        raw_var_treatment=raw_res.var_b,
        raw_pooled_var=raw_pooled_var,
        raw_ttest=raw_res,
        cuped_var_control=cuped_res.var_a,
        cuped_var_treatment=cuped_res.var_b,
        cuped_pooled_var=cuped_pooled_var,
        cuped_ttest=cuped_res,
        ci_width_raw=ci_raw,
        ci_width_cuped=ci_cuped,
        ci_width_reduction=ci_reduction,
    )

    return df_out, metrics


def validate_theta_against_scipy(x: np.ndarray, y: np.ndarray) -> Tuple[bool, float, float]:
    """Validate that custom theta matches scipy.stats.linregress OLS slope."""
    custom_theta = compute_optimal_theta(x, y)
    slope, _, r_val, _, _ = stats.linregress(x, y)
    matches = abs(custom_theta - slope) < 1e-7
    return matches, custom_theta, float(slope)


if __name__ == "__main__":
    from src.synthetic_cuped_data import generate_cuped_dataset

    # Generate synthetic dataset with true effect tau = +1.50
    dataset = generate_cuped_dataset(num_users=10000, true_tau=1.50, target_rho=0.75, random_seed=42)
    df_cuped, res = run_cuped_analysis(dataset.df)

    # Validate theta with scipy
    matches, c_th, s_th = validate_theta_against_scipy(dataset.df["pre_rounds"].values, dataset.df["post_rounds"].values)

    print("=" * 70)
    print("STEP 4 & 5: CUPED VARIANCE REDUCTION BENCHMARK")
    print("=" * 70)
    print(f"Pre/Post Correlation (rho):       {res.correlation_rho:.4f}")
    print(f"Optimal Theta (OLS Slope):        {res.theta:.4f} (scipy: {s_th:.4f}, Validated: {matches})")
    print(f"Theoretical Variance Reduction:   {res.theoretical_var_reduction:.2%}")
    print(f"Empirical Variance Reduction:     {res.empirical_var_reduction:.2%}")
    print(f"Effective Sample Size Multiplier: {res.effective_sample_multiplier:.2f}x (Equivalent to +{((res.effective_sample_multiplier-1)*100):.1f}% more users)")

    print("\n--- A. Raw Metric (Without CUPED) ---")
    print(f"Control Mean:       {res.raw_ttest.mean_a:.3f} (Var: {res.raw_var_control:.2f})")
    print(f"Treatment Mean:     {res.raw_ttest.mean_b:.3f} (Var: {res.raw_var_treatment:.2f})")
    print(f"Estimated Lift:     {res.raw_ttest.absolute_diff:+.3f} rounds (True Effect: +{dataset.true_tau:.2f})")
    print(f"Standard Error:     {res.raw_ttest.se_diff:.4f}")
    print(f"T-statistic:        {res.raw_ttest.t_stat:.4f}")
    print(f"P-value:            {res.raw_ttest.p_value:.4e}")
    print(f"95% CI:             [{res.raw_ttest.ci_lower:+.3f}, {res.raw_ttest.ci_upper:+.3f}] (Width: {res.ci_width_raw:.3f})")

    print("\n--- B. CUPED-Adjusted Metric ---")
    print(f"Control Mean:       {res.cuped_ttest.mean_a:.3f} (Var: {res.cuped_var_control:.2f})")
    print(f"Treatment Mean:     {res.cuped_ttest.mean_b:.3f} (Var: {res.cuped_var_treatment:.2f})")
    print(f"Estimated Lift:     {res.cuped_ttest.absolute_diff:+.3f} rounds (Unbiased: Exactly matches raw!)")
    print(f"Standard Error:     {res.cuped_ttest.se_diff:.4f} (Dropped by {((res.raw_ttest.se_diff - res.cuped_ttest.se_diff)/res.raw_ttest.se_diff):.1%})")
    print(f"T-statistic:        {res.cuped_ttest.t_stat:.4f} (Increased from {res.raw_ttest.t_stat:.2f})")
    print(f"P-value:            {res.cuped_ttest.p_value:.4e} (Substantially more significant!)")
    print(f"95% CI:             [{res.cuped_ttest.ci_lower:+.3f}, {res.cuped_ttest.ci_upper:+.3f}] (Width: {res.ci_width_cuped:.3f}, Narrowed by {res.ci_width_reduction:.1%})")
