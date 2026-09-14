"""Two-sample Welch's t-test implementation from first principles.

This module provides a first-principles implementation of Welch's t-test for continuous
metrics (e.g., sum_gamerounds) where sample variances and sample sizes may differ.
It includes mathematical derivations, Welch-Satterthwaite degrees of freedom,
confidence intervals, and automated validation against `scipy.stats.ttest_ind(..., equal_var=False)`.
"""

import sys
from pathlib import Path
from typing import NamedTuple, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


class TTestResult(NamedTuple):
    """Structured container for Welch's t-test outputs."""
    t_stat: float
    p_value: float
    df: float                 # Welch-Satterthwaite degrees of freedom
    mean_a: float
    mean_b: float
    var_a: float
    var_b: float
    std_a: float
    std_b: float
    absolute_diff: float      # mean_b - mean_a
    relative_lift: float      # (mean_b - mean_a) / mean_a
    ci_lower: float           # Lower bound of (mean_b - mean_a)
    ci_upper: float           # Upper bound of (mean_b - mean_a)
    se_diff: float            # Welch standard error of the difference


def welch_t_test(
    sample_a: Union[np.ndarray, pd.Series, list],
    sample_b: Union[np.ndarray, pd.Series, list],
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> TTestResult:
    """Perform a two-sample Welch's t-test (unequal variances) from first principles.

    Tests the null hypothesis H0: mu_a = mu_b against H1: mu_a != mu_b.
    Uses the Welch-Satterthwaite equation to approximate degrees of freedom.

    Parameters
    ----------
    sample_a : array-like
        Continuous observations for Group A (control).
    sample_b : array-like
        Continuous observations for Group B (treatment).
    alpha : float, default 0.05
        Significance level (e.g. 0.05 for 95% confidence).
    alternative : str, default 'two-sided'
        Alternative hypothesis: 'two-sided', 'larger' (mu_a > mu_b),
        or 'smaller' (mu_a < mu_b). Matches scipy.stats.ttest_ind convention
        where group A is the reference direction.

    Returns
    -------
    result : TTestResult
        NamedTuple with t-statistic, p-value, degrees of freedom, effect sizes, and CIs.
    """
    arr_a = np.asarray(sample_a, dtype=float)
    arr_b = np.asarray(sample_b, dtype=float)

    # Clean out any NaNs if present
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]

    n_a = len(arr_a)
    n_b = len(arr_b)

    if n_a < 2 or n_b < 2:
        raise ValueError(f"Each sample must have at least 2 observations. Got n_a={n_a}, n_b={n_b}")

    # 1. Sample means
    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))

    # 2. Unbiased sample variances (ddof=1 using Bessel's correction)
    var_a = float(np.var(arr_a, ddof=1))
    var_b = float(np.var(arr_b, ddof=1))
    std_a = float(np.sqrt(var_a))
    std_b = float(np.sqrt(var_b))

    # 3. Variance of the difference and Standard Error
    # Var(diff) = (s_a^2 / n_a) + (s_b^2 / n_b)
    var_mean_a = var_a / n_a
    var_mean_b = var_b / n_b
    se_diff = float(np.sqrt(var_mean_a + var_mean_b))

    if se_diff == 0.0:
        raise ValueError("Standard error is zero; samples have zero variance.")

    # 4. Welch's t-statistic
    # Testing (mean_a - mean_b) against 0:
    # We use (mean_a - mean_b) to match scipy.stats.ttest_ind(a, b) sign convention.
    t_stat = (mean_a - mean_b) / se_diff

    # 5. Welch-Satterthwaite approximation for effective degrees of freedom:
    # df = ( (s_a^2/n_a + s_b^2/n_b)^2 ) / [ (s_a^2/n_a)^2 / (n_a - 1) + (s_b^2/n_b)^2 / (n_b - 1) ]
    num_df = (var_mean_a + var_mean_b) ** 2
    denom_df = ((var_mean_a ** 2) / (n_a - 1)) + ((var_mean_b ** 2) / (n_b - 1))
    df_welch = float(num_df / denom_df)

    # 6. P-value calculation from Student's t-distribution
    if alternative == "two-sided":
        p_value = float(2.0 * stats.t.sf(np.abs(t_stat), df=df_welch))
    elif alternative == "larger":
        # H1: mean_a > mean_b
        p_value = float(stats.t.sf(t_stat, df=df_welch))
    elif alternative == "smaller":
        # H1: mean_a < mean_b
        p_value = float(stats.t.cdf(t_stat, df=df_welch))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # 7. Confidence Interval for effect size (Treatment - Control: mean_b - mean_a)
    diff_b_minus_a = mean_b - mean_a
    t_crit = float(stats.t.ppf(1.0 - (alpha / 2.0), df=df_welch))
    ci_lower = diff_b_minus_a - (t_crit * se_diff)
    ci_upper = diff_b_minus_a + (t_crit * se_diff)
    relative_lift = diff_b_minus_a / mean_a if mean_a != 0.0 else 0.0

    return TTestResult(
        t_stat=float(t_stat),
        p_value=float(p_value),
        df=df_welch,
        mean_a=mean_a,
        mean_b=mean_b,
        var_a=var_a,
        var_b=var_b,
        std_a=std_a,
        std_b=std_b,
        absolute_diff=float(diff_b_minus_a),
        relative_lift=float(relative_lift),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        se_diff=se_diff,
    )


def validate_against_scipy(
    sample_a: Union[np.ndarray, pd.Series, list],
    sample_b: Union[np.ndarray, pd.Series, list]
) -> Tuple[bool, float, float, float, float, float]:
    """Validate the custom Welch's t-test implementation against scipy.stats.ttest_ind."""
    custom_res = welch_t_test(sample_a, sample_b)
    scipy_res = stats.ttest_ind(sample_a, sample_b, equal_var=False)

    t_diff = abs(custom_res.t_stat - scipy_res.statistic)
    p_diff = abs(custom_res.p_value - scipy_res.pvalue)
    df_diff = abs(custom_res.df - scipy_res.df) if hasattr(scipy_res, "df") else 0.0

    matches = (t_diff < 1e-7) and (p_diff < 1e-7) and (df_diff < 1e-7)

    return (
        matches,
        custom_res.t_stat,
        float(scipy_res.statistic),
        custom_res.p_value,
        float(scipy_res.pvalue),
        custom_res.df,
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_cookie_cats_data

    # 1. Test on cleaned dataset (outlier removed)
    df_clean, report = load_cookie_cats_data(remove_extreme_outliers=True)
    rounds_a = df_clean[df_clean["version"] == "gate_30"]["sum_gamerounds"]
    rounds_b = df_clean[df_clean["version"] == "gate_40"]["sum_gamerounds"]

    res_clean = welch_t_test(rounds_a, rounds_b)
    matches_clean, ct, st, cp, sp, df_val = validate_against_scipy(rounds_a, rounds_b)

    print("=" * 70)
    print("STEP 4: TWO-SAMPLE WELCH'S T-TEST (sum_gamerounds)")
    print("=" * 70)
    print("\n--- A. Analysis on Cleaned Dataset (Outlier Removed) ---")
    print(f"Control (gate_30):   N={len(rounds_a)}, Mean={res_clean.mean_a:.3f}, Std={res_clean.std_a:.3f}, Var={res_clean.var_a:.2f}")
    print(f"Treatment (gate_40): N={len(rounds_b)}, Mean={res_clean.mean_b:.3f}, Std={res_clean.std_b:.3f}, Var={res_clean.var_b:.2f}")
    print(f"Absolute Diff (B-A): {res_clean.absolute_diff:+.3f} rounds")
    print(f"Relative Lift:       {res_clean.relative_lift:+.2%}")
    print(f"Standard Error (SE): {res_clean.se_diff:.4f}")
    print(f"Welch DF:            {res_clean.df:.1f}")
    print(f"T-statistic:         {res_clean.t_stat:.4f} (scipy: {st:.4f})")
    print(f"P-value:             {res_clean.p_value:.4f} (scipy: {sp:.4f})")
    print(f"95% CI (diff):       [{res_clean.ci_lower:+.3f}, {res_clean.ci_upper:+.3f}]")
    print(f"Validation Match:    {'EXACT MATCH' if matches_clean else 'MISMATCH'}")
    print(f"Statistically Sig?   {'YES (p < 0.05)' if res_clean.p_value < 0.05 else 'NO (p >= 0.05)'}")

    # 2. Sensitivity check: with the extreme outlier included
    df_raw, _ = load_cookie_cats_data(remove_extreme_outliers=False)
    raw_a = df_raw[df_raw["version"] == "gate_30"]["sum_gamerounds"]
    raw_b = df_raw[df_raw["version"] == "gate_40"]["sum_gamerounds"]
    res_raw = welch_t_test(raw_a, raw_b)
    matches_raw, rct, rst, rcp, rsp, _ = validate_against_scipy(raw_a, raw_b)

    print("\n--- B. Sensitivity Check: Raw Dataset (With 49,854 Outlier) ---")
    print(f"Control (gate_30):   Mean={res_raw.mean_a:.3f}, Std={res_raw.std_a:.3f}, Var={res_raw.var_a:.2f}")
    print(f"Treatment (gate_40): Mean={res_raw.mean_b:.3f}, Std={res_raw.std_b:.3f}, Var={res_raw.var_b:.2f}")
    print(f"Variance Ratio (A/B):{res_raw.var_a / res_raw.var_b:.2f}x (Massive Heteroscedasticity!)")
    print(f"Absolute Diff (B-A): {res_raw.absolute_diff:+.3f} rounds")
    print(f"T-statistic:         {res_raw.t_stat:.4f} (scipy: {rst:.4f})")
    print(f"P-value:             {res_raw.p_value:.4f} (scipy: {rsp:.4f})")
    print(f"Validation Match:    {'EXACT MATCH' if matches_raw else 'MISMATCH'}")
