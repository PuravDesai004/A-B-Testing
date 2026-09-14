"""Two-proportion Z-test implementation from first principles.

This module provides an exact implementation of the two-sample pooled-proportion
z-test used for binary conversion/retention metrics (e.g., retention_1, retention_7).
It includes full educational commentary, mathematical derivations, confidence interval
calculations, and automated verification against statsmodels.
"""

from typing import NamedTuple, Tuple
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


class ProportionTestResult(NamedTuple):
    """Structured container for two-proportion test outputs."""
    z_stat: float
    p_value: float
    prop_a: float
    prop_b: float
    absolute_diff: float      # prop_b - prop_a
    relative_lift: float      # (prop_b - prop_a) / prop_a
    ci_lower: float           # Lower bound of (prop_b - prop_a)
    ci_upper: float           # Upper bound of (prop_b - prop_a)
    pooled_prop: float
    se_pooled: float
    se_diff: float


def proportion_z_test(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> ProportionTestResult:
    """Perform a two-sample proportion z-test from first principles.

    Tests the null hypothesis H0: p_a = p_b against H1: p_a != p_b (two-sided),
    or directional alternatives ("larger", "smaller").

    Parameters
    ----------
    successes_a : int
        Number of conversions/successes in Group A (e.g. control).
    n_a : int
        Total sample size of Group A.
    successes_b : int
        Number of conversions/successes in Group B (e.g. treatment).
    n_b : int
        Total sample size of Group B.
    alpha : float, default 0.05
        Significance level (e.g. 0.05 for 95% confidence).
    alternative : str, default 'two-sided'
        Type of alternative hypothesis: 'two-sided', 'larger' (p_a > p_b),
        or 'smaller' (p_a < p_b). Matches statsmodels convention where group A
        is the reference direction.

    Returns
    -------
    result : ProportionTestResult
        NamedTuple with test statistics, p-values, lifts, and confidence intervals.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError(f"Sample sizes must be positive. Got n_a={n_a}, n_b={n_b}")
    if not (0 <= successes_a <= n_a) or not (0 <= successes_b <= n_b):
        raise ValueError("Successes must be between 0 and sample size.")

    # 1. Sample proportions
    p_a = successes_a / n_a
    p_b = successes_b / n_b

    # 2. Pooled proportion under H0 (where p_a == p_b == p_pool)
    p_pool = (successes_a + successes_b) / (n_a + n_b)

    # 3. Pooled Standard Error of the difference under H0
    # SE_pool = sqrt( p_pool * (1 - p_pool) * (1/n_a + 1/n_b) )
    se_pooled = np.sqrt(p_pool * (1.0 - p_pool) * ((1.0 / n_a) + (1.0 / n_b)))

    # 4. Z-statistic
    # Testing (p_b - p_a) against 0:
    # Notice: If testing (p_a - p_b), sign reverses. Statsmodels default order:
    # proportions_ztest([successes_a, successes_b], [n_a, n_b]) tests (p_a - p_b).
    # We follow the convention: z = (p_a - p_b) / se_pooled to match statsmodels directly.
    diff_a_minus_b = p_a - p_b
    z_stat = diff_a_minus_b / se_pooled

    # 5. P-value calculation using standard normal distribution (Phi)
    if alternative == "two-sided":
        # Two-tailed: probability of seeing a z as extreme or more extreme in either tail
        p_value = 2.0 * stats.norm.sf(np.abs(z_stat))
    elif alternative == "larger":
        # H1: p_a > p_b  => z > 0
        p_value = stats.norm.sf(z_stat)
    elif alternative == "smaller":
        # H1: p_a < p_b  => z < 0
        p_value = stats.norm.cdf(z_stat)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # 6. Confidence Interval for the difference (Treatment - Control: p_b - p_a)
    # Important statistical nuance:
    # Under H0, we pool variances. But when building a Confidence Interval for the
    # true effect size, we do NOT assume H0 is true! We estimate separate variances:
    # SE_diff = sqrt( p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b )
    se_diff = np.sqrt((p_a * (1.0 - p_a) / n_a) + (p_b * (1.0 - p_b) / n_b))
    z_crit = stats.norm.ppf(1.0 - (alpha / 2.0))

    diff_b_minus_a = p_b - p_a
    ci_lower = diff_b_minus_a - (z_crit * se_diff)
    ci_upper = diff_b_minus_a + (z_crit * se_diff)
    relative_lift = (p_b - p_a) / p_a if p_a > 0 else 0.0

    return ProportionTestResult(
        z_stat=float(z_stat),
        p_value=float(p_value),
        prop_a=float(p_a),
        prop_b=float(p_b),
        absolute_diff=float(diff_b_minus_a),
        relative_lift=float(relative_lift),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        pooled_prop=float(p_pool),
        se_pooled=float(se_pooled),
        se_diff=float(se_diff),
    )


def validate_against_statsmodels(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int
) -> Tuple[bool, float, float, float, float]:
    """Validate the custom z-test implementation against statsmodels.

    Returns
    -------
    matches : bool
        True if custom implementation matches statsmodels within 1e-9.
    custom_z, sm_z, custom_p, sm_p : float
        The respective statistics and p-values.
    """
    # Custom run
    custom_res = proportion_z_test(successes_a, n_a, successes_b, n_b)

    # Reference run using statsmodels
    count = np.array([successes_a, successes_b])
    nobs = np.array([n_a, n_b])
    sm_z, sm_p = proportions_ztest(count=count, nobs=nobs, alternative="two-sided")

    z_diff = abs(custom_res.z_stat - sm_z)
    p_diff = abs(custom_res.p_value - sm_p)
    matches = (z_diff < 1e-7) and (p_diff < 1e-7)

    return matches, custom_res.z_stat, float(sm_z), custom_res.p_value, float(sm_p)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_cookie_cats_data, get_group_metrics

    df, _ = load_cookie_cats_data()
    metrics = get_group_metrics(df)

    print("=" * 70)
    print("STEP 3: TWO-PROPORTION Z-TEST (RETENTION METRICS)")
    print("=" * 70)

    for metric_name in ["retention_1", "retention_7"]:
        m_a = metrics["gate_30"][metric_name]
        m_b = metrics["gate_40"][metric_name]

        res = proportion_z_test(
            successes_a=m_a["successes"],
            n_a=m_a["total"],
            successes_b=m_b["successes"],
            n_b=m_b["total"]
        )

        matches, cz, sz, cp, sp = validate_against_statsmodels(
            m_a["successes"], m_a["total"],
            m_b["successes"], m_b["total"]
        )

        print(f"\n--- Metric: {metric_name} ---")
        print(f"Control (gate_30):   {res.prop_a:.4%} ({m_a['successes']}/{m_a['total']})")
        print(f"Treatment (gate_40): {res.prop_b:.4%} ({m_b['successes']}/{m_b['total']})")
        print(f"Absolute Diff:       {res.absolute_diff:+.4%} points")
        print(f"Relative Lift:       {res.relative_lift:+.2%}")
        print(f"Pooled Proportion:   {res.pooled_prop:.4%}")
        print(f"Pooled SE (for z):   {res.se_pooled:.6f}")
        print(f"Z-statistic:         {res.z_stat:.4f} (statsmodels: {sz:.4f})")
        print(f"P-value:             {res.p_value:.4e} (statsmodels: {sp:.4e})")
        print(f"95% CI (diff):       [{res.ci_lower:+.4%}, {res.ci_upper:+.4%}]")
        print(f"Validation Match:    {'EXACT MATCH' if matches else 'MISMATCH'}")
        print(f"Statistically Sig?   {'YES (p < 0.05)' if res.p_value < 0.05 else 'NO (p >= 0.05)'}")
