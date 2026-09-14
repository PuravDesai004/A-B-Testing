"""Retrospective power and Minimum Detectable Effect (MDE) analysis module.

This module computes the Minimum Detectable Effect (MDE) for an experiment given its
actual sample size, significance level (alpha=0.05), and target power (80%).
It translates standardized effect sizes (Cohen's h and Cohen's d) into real-world
business units (percentage points and gamerounds) and compares them against observed effects.
"""

import sys
from pathlib import Path
from typing import NamedTuple, Dict, Any
import numpy as np
from statsmodels.stats.power import NormalIndPower, TTestIndPower


class ProportionPowerResult(NamedTuple):
    """Container for proportion metric power analysis."""
    metric_name: str
    n_control: int
    n_treatment: int
    control_rate: float
    treatment_rate: float
    observed_abs_diff: float
    observed_rel_lift: float
    cohens_h_mde: float
    mde_abs_percentage_points: float
    mde_relative_percentage: float
    was_effect_detectable: bool
    verdict: str


class ContinuousPowerResult(NamedTuple):
    """Container for continuous metric power analysis."""
    metric_name: str
    n_control: int
    n_treatment: int
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    pooled_std: float
    observed_abs_diff: float
    observed_rel_lift: float
    cohens_d_mde: float
    mde_abs_rounds: float
    mde_relative_percentage: float
    was_effect_detectable: bool
    verdict: str


def compute_proportion_mde(
    metric_name: str,
    n_control: int,
    n_treatment: int,
    control_rate: float,
    treatment_rate: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> ProportionPowerResult:
    """Compute retrospective MDE for a binary conversion/retention metric.

    Parameters
    ----------
    metric_name : str
        Human-readable name of the metric (e.g. 'retention_1').
    n_control : int
        Sample size in control group (gate_30).
    n_treatment : int
        Sample size in treatment group (gate_40).
    control_rate : float
        Observed baseline proportion (e.g. 0.4482).
    treatment_rate : float
        Observed treatment proportion (e.g. 0.4423).
    alpha : float, default 0.05
        Two-sided significance level.
    power : float, default 0.80
        Target statistical power (1 - beta).

    Returns
    -------
    result : ProportionPowerResult
    """
    ratio = n_treatment / n_control
    power_analysis = NormalIndPower()

    # Solve for Minimum Detectable Effect in Cohen's h
    # Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    h_mde = power_analysis.solve_power(
        effect_size=None,
        nobs1=n_control,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative="two-sided"
    )

    # Convert Cohen's h back to absolute percentage points from baseline p1:
    # arcsin(sqrt(p2)) = arcsin(sqrt(p1)) +- (h / 2)
    # p2 = sin(arcsin(sqrt(p1)) +- (h / 2))^2
    p1 = control_rate
    phi1 = np.arcsin(np.sqrt(p1))

    # We evaluate upper and lower boundaries and average their distance to p1
    p2_upper = np.sin(phi1 + (h_mde / 2.0)) ** 2
    p2_lower = np.sin(phi1 - (h_mde / 2.0)) ** 2
    mde_abs = float((abs(p2_upper - p1) + abs(p1 - p2_lower)) / 2.0)
    mde_rel = float(mde_abs / p1) if p1 > 0 else 0.0

    obs_abs = float(abs(treatment_rate - control_rate))
    obs_rel = float((treatment_rate - control_rate) / control_rate) if p1 > 0 else 0.0

    was_detectable = obs_abs >= mde_abs

    if was_detectable:
        verdict = (
            f"WELL-POWERED: The observed change of {obs_abs:.2%} points exceeds the "
            f"80% power detection threshold ({mde_abs:.2%} points)."
        )
    else:
        verdict = (
            f"POTENTIALLY UNDERPOWERED FOR OBSERVED EFFECT: The observed change of "
            f"{obs_abs:.2%} points is smaller than the minimum effect this experiment "
            f"could reliably catch at 80% power ({mde_abs:.2%} points). "
            f"A non-significant result cannot rule out a real effect up to {mde_abs:.2%} points."
        )

    return ProportionPowerResult(
        metric_name=metric_name,
        n_control=n_control,
        n_treatment=n_treatment,
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        observed_abs_diff=treatment_rate - control_rate,
        observed_rel_lift=obs_rel,
        cohens_h_mde=float(h_mde),
        mde_abs_percentage_points=mde_abs,
        mde_relative_percentage=mde_rel,
        was_effect_detectable=was_detectable,
        verdict=verdict,
    )


def compute_continuous_mde(
    metric_name: str,
    n_control: int,
    n_treatment: int,
    control_mean: float,
    treatment_mean: float,
    control_std: float,
    treatment_std: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> ContinuousPowerResult:
    """Compute retrospective MDE for a continuous metric (e.g. sum_gamerounds).

    Parameters
    ----------
    metric_name : str
        Human-readable name of the metric.
    n_control : int
        Sample size in control group.
    n_treatment : int
        Sample size in treatment group.
    control_mean : float
        Sample mean of control group.
    treatment_mean : float
        Sample mean of treatment group.
    control_std : float
        Sample standard deviation of control group.
    treatment_std : float
        Sample standard deviation of treatment group.
    alpha : float, default 0.05
    power : float, default 0.80

    Returns
    -------
    result : ContinuousPowerResult
    """
    ratio = n_treatment / n_control
    power_analysis = TTestIndPower()

    # Solve for Minimum Detectable Effect in Cohen's d: d = |mu1 - mu2| / pooled_std
    d_mde = power_analysis.solve_power(
        effect_size=None,
        nobs1=n_control,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative="two-sided"
    )

    # Pooled standard deviation
    pooled_var = (((n_control - 1) * (control_std ** 2)) + ((n_treatment - 1) * (treatment_std ** 2))) / (n_control + n_treatment - 2)
    pooled_std = float(np.sqrt(pooled_var))

    mde_abs_rounds = float(d_mde * pooled_std)
    mde_rel = float(mde_abs_rounds / control_mean) if control_mean > 0 else 0.0

    obs_abs = float(abs(treatment_mean - control_mean))
    obs_rel = float((treatment_mean - control_mean) / control_mean) if control_mean > 0 else 0.0

    was_detectable = obs_abs >= mde_abs_rounds

    if was_detectable:
        verdict = (
            f"WELL-POWERED: The observed change of {obs_abs:.3f} units exceeds the "
            f"80% power detection threshold ({mde_abs_rounds:.3f} units)."
        )
    else:
        verdict = (
            f"EFFECT SMALLER THAN MDE: The observed change of {obs_abs:.3f} rounds "
            f"is much smaller than the 80% power detection threshold ({mde_abs_rounds:.3f} rounds). "
            f"Given the sample sizes, this experiment was powered to detect changes of ~{mde_abs_rounds:.2f} units (~{mde_rel:.1%}). "
            f"The observed {obs_abs:.3f} unit difference is practically negligible."
        )

    return ContinuousPowerResult(
        metric_name=metric_name,
        n_control=n_control,
        n_treatment=n_treatment,
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        control_std=control_std,
        treatment_std=treatment_std,
        pooled_std=pooled_std,
        observed_abs_diff=treatment_mean - control_mean,
        observed_rel_lift=obs_rel,
        cohens_d_mde=float(d_mde),
        mde_abs_rounds=mde_abs_rounds,
        mde_relative_percentage=mde_rel,
        was_effect_detectable=was_detectable,
        verdict=verdict,
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_cookie_cats_data, get_group_metrics

    df, _ = load_cookie_cats_data(remove_extreme_outliers=True)
    metrics = get_group_metrics(df)

    print("=" * 70)
    print("STEP 5: RETROSPECTIVE POWER CHECK & MINIMUM DETECTABLE EFFECT (MDE)")
    print("=" * 70)

    # Retention 1
    pwr_r1 = compute_proportion_mde(
        metric_name="retention_1",
        n_control=metrics["gate_30"]["n"],
        n_treatment=metrics["gate_40"]["n"],
        control_rate=metrics["gate_30"]["retention_1"]["rate"],
        treatment_rate=metrics["gate_40"]["retention_1"]["rate"],
    )

    # Retention 7
    pwr_r7 = compute_proportion_mde(
        metric_name="retention_7",
        n_control=metrics["gate_30"]["n"],
        n_treatment=metrics["gate_40"]["n"],
        control_rate=metrics["gate_30"]["retention_7"]["rate"],
        treatment_rate=metrics["gate_40"]["retention_7"]["rate"],
    )

    # Gamerounds
    pwr_gr = compute_continuous_mde(
        metric_name="sum_gamerounds",
        n_control=metrics["gate_30"]["n"],
        n_treatment=metrics["gate_40"]["n"],
        control_mean=metrics["gate_30"]["sum_gamerounds"]["mean"],
        treatment_mean=metrics["gate_40"]["sum_gamerounds"]["mean"],
        control_std=metrics["gate_30"]["sum_gamerounds"]["std"],
        treatment_std=metrics["gate_40"]["sum_gamerounds"]["std"],
    )

    for res in [pwr_r1, pwr_r7]:
        print(f"\n--- Metric: {res.metric_name} ---")
        print(f"Sample Sizes:         N_control = {res.n_control:,}, N_treatment = {res.n_treatment:,}")
        print(f"Baseline Rate:        {res.control_rate:.4%}")
        print(f"Observed Diff:        {res.observed_abs_diff:+.4%} points ({res.observed_rel_lift:+.2%} relative)")
        print(f"Cohen's h MDE (80%):  {res.cohens_h_mde:.4f}")
        print(f"Absolute MDE (80%):   ±{res.mde_abs_percentage_points:.4%} points")
        print(f"Relative MDE (80%):   ±{res.mde_relative_percentage:.2%}")
        print(f"Verdict:              {res.verdict}")

    print(f"\n--- Metric: {pwr_gr.metric_name} ---")
    print(f"Sample Sizes:         N_control = {pwr_gr.n_control:,}, N_treatment = {pwr_gr.n_treatment:,}")
    print(f"Baseline Mean:        {pwr_gr.control_mean:.3f} rounds (Pooled Std: {pwr_gr.pooled_std:.2f})")
    print(f"Observed Diff:        {pwr_gr.observed_abs_diff:+.3f} rounds ({pwr_gr.observed_rel_lift:+.2%} relative)")
    print(f"Cohen's d MDE (80%):  {pwr_gr.cohens_d_mde:.4f}")
    print(f"Absolute MDE (80%):   ±{pwr_gr.mde_abs_rounds:.3f} rounds")
    print(f"Relative MDE (80%):   ±{pwr_gr.mde_relative_percentage:.2%}")
    print(f"Verdict:              {pwr_gr.verdict}")
