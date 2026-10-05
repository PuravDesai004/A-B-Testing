"""Novelty effect detection and time-windowed decay analysis module.

This module detects transient behavioral changes ('novelty effects') where an experiment's
treatment effect is initially strong due to user curiosity, but exponentially decays
as the novelty wears off. It performs windowed Welch's t-tests comparing early vs. late
experiment windows.
"""

import sys
from pathlib import Path
from typing import NamedTuple, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.continuous_test import welch_t_test, TTestResult


class NoveltyCheckResult(NamedTuple):
    """Container for time-windowed novelty effect analysis."""
    daily_effects: List[float]          # Daily observed lift (Treatment - Control)
    daily_days: List[int]
    early_window_days: Tuple[int, int]  # e.g. (1, 3)
    late_window_days: Tuple[int, int]   # e.g. (12, 14)
    early_ttest: TTestResult
    late_ttest: TTestResult
    full_ttest: TTestResult
    decay_percentage: float             # (early_lift - late_lift) / early_lift
    has_novelty_decay: bool
    verdict: str
    decay_p_value: float


def generate_novelty_decay_data(
    num_users_per_day: int = 1000,
    num_days: int = 14,
    baseline_mean: float = 50.0,
    baseline_std: float = 10.0,
    initial_tau: float = 5.0,           # Strong Day 1 lift (+5.0 rounds)
    decay_rate: float = 0.35,           # Exponential decay constant lambda
    random_seed: int = 42
) -> pd.DataFrame:
    """Generate daily user activity telemetry with a known, decaying treatment effect.

    The model:
        Control: Y_it ~ Normal(baseline_mean, baseline_std^2)
        Treatment: Y_it ~ Normal(baseline_mean + tau(t), baseline_std^2)
        where tau(t) = initial_tau * exp(-decay_rate * (t - 1))

    Parameters
    ----------
    num_users_per_day : int, default 1000
        New players enrolled per group per day (500 control, 500 treatment).
    num_days : int, default 14
        Duration of the experiment in days.
    baseline_mean : float, default 50.0
    baseline_std : float, default 10.0
    initial_tau : float, default 5.0
    decay_rate : float, default 0.35
    random_seed : int, default 42

    Returns
    -------
    df : pd.DataFrame
        Table with ['day', 'variant', 'rounds'].
    """
    rng = np.random.default_rng(random_seed)
    records = []

    users_per_variant = num_users_per_day // 2

    for day in range(1, num_days + 1):
        # Ground-truth decaying effect for day t
        tau_t = initial_tau * np.exp(-decay_rate * (day - 1))

        # Control observations
        ctrl_rounds = rng.normal(loc=baseline_mean, scale=baseline_std, size=users_per_variant)
        # Treatment observations with decaying lift
        trt_rounds = rng.normal(loc=baseline_mean + tau_t, scale=baseline_std, size=users_per_variant)

        for val in ctrl_rounds:
            records.append({"day": day, "variant": "control", "rounds": max(0.0, float(val))})
        for val in trt_rounds:
            records.append({"day": day, "variant": "treatment", "rounds": max(0.0, float(val))})

    return pd.DataFrame(records)


def analyze_novelty_effect(
    df: pd.DataFrame,
    day_col: str = "day",
    variant_col: str = "variant",
    metric_col: str = "rounds",
    early_window: Tuple[int, int] = (1, 3),
    late_window: Tuple[int, int] = (12, 14),
    alpha: float = 0.05,
    control_label: str = "control",
    treatment_label: str = "treatment",
) -> NoveltyCheckResult:
    """Analyze novelty decay assuming independent observations across windows.

    Repeated measurements of the same users require a clustered/paired model
    instead. The decay contrast is a one-sided early-minus-late effect test.

    Parameters
    ----------
    df : pd.DataFrame
        Daily user activity records.
    day_col : str
        Column denoting experiment day (1, 2, ...).
    variant_col : str
        Column indicating 'control' vs 'treatment'.
    metric_col : str
        The continuous metric evaluated (e.g. 'rounds').
    early_window : tuple of (int, int), default (1, 3)
        Days inclusive defining the initial window.
    late_window : tuple of (int, int), default (12, 14)
        Days inclusive defining the late window.
    alpha : float, default 0.05

    Returns
    -------
    result : NoveltyCheckResult
    """
    if control_label == treatment_label:
        raise ValueError("Control and treatment labels must differ.")
    if early_window[0] > early_window[1] or late_window[0] > late_window[1] or early_window[1] >= late_window[0]:
        raise ValueError("Early and late windows must be ordered and non-overlapping.")
    df = df[df[variant_col].isin([control_label, treatment_label])].copy()
    df = df.dropna(subset=[metric_col])
    if not np.isfinite(df[[day_col, metric_col]].to_numpy(dtype=float)).all():
        raise ValueError("Novelty checks require finite days and outcomes.")
    days = sorted(df[day_col].unique())
    daily_lifts = []

    # 1. Compute daily average lift across all individual days
    for d in days:
        day_sub = df[df[day_col] == d]
        ctrl = day_sub[day_sub[variant_col] == control_label][metric_col]
        trt = day_sub[day_sub[variant_col] == treatment_label][metric_col]
        if ctrl.empty or trt.empty:
            raise ValueError(f"Both groups need observations on day {d}.")
        lift = float(trt.mean() - ctrl.mean())
        daily_lifts.append(lift)

    # 2. Windowed slices
    early_mask = df[day_col].between(early_window[0], early_window[1])
    late_mask = df[day_col].between(late_window[0], late_window[1])

    early_df = df[early_mask]
    late_df = df[late_mask]

    # Run Welch's t-test on Early Window
    early_ctrl = early_df[early_df[variant_col] == control_label][metric_col]
    early_trt = early_df[early_df[variant_col] == treatment_label][metric_col]
    early_res = welch_t_test(early_ctrl, early_trt, alpha=alpha)

    # Run Welch's t-test on Late Window
    late_ctrl = late_df[late_df[variant_col] == control_label][metric_col]
    late_trt = late_df[late_df[variant_col] == treatment_label][metric_col]
    late_res = welch_t_test(late_ctrl, late_trt, alpha=alpha)

    # Run Welch's t-test on Full Timeline
    full_ctrl = df[df[variant_col] == control_label][metric_col]
    full_trt = df[df[variant_col] == treatment_label][metric_col]
    full_res = welch_t_test(full_ctrl, full_trt, alpha=alpha)

    # Compute Decay Rate: (early_lift - late_lift) / early_lift
    early_lift = early_res.absolute_diff
    late_lift = late_res.absolute_diff

    if early_lift != 0.0:
        decay_pct = float((early_lift - late_lift) / early_lift)
    else:
        decay_pct = 0.0

    # Direct difference-in-differences test for independent observations across
    # four cells. A change in significance alone is not evidence of decay.
    cells = [early_ctrl, early_trt, late_ctrl, late_trt]
    variance_terms = np.array([cell.var(ddof=1) / len(cell) for cell in cells])
    decay_se = float(np.sqrt(variance_terms.sum()))
    decay_df = float(variance_terms.sum() ** 2 / sum(
        term ** 2 / (len(cell) - 1) for term, cell in zip(variance_terms, cells)))
    decay_p = float(stats.t.sf((early_lift - late_lift) / decay_se, decay_df))
    has_decay = bool(early_lift > 0 and early_res.p_value < alpha
                     and decay_pct >= 0.60 and decay_p < alpha)

    if has_decay:
        verdict = (
            f"NOVELTY EFFECT DETECTED: Treatment lift was strongly positive in the early window "
            f"({early_lift:+.2f} rounds, p = {early_res.p_value:.4e}), but decayed by {decay_pct:.1%} "
            f"to {late_lift:+.2f} rounds (p = {late_res.p_value:.4f}) in the late window. "
            f"The early-to-late decrease is significant (one-sided p = {decay_p:.4e}); "
            f"this is evidence of attenuation, not proof of its cause or future persistence."
        )
    else:
        verdict = (
            f"NO NOVELTY DECAY DETECTED: Insufficient evidence of a positive lift decaying by at least 60%. Early ({early_lift:+.2f} rounds) "
            f"and late ({late_lift:+.2f} rounds) windows (decay: {decay_pct:.1%})."
        )

    return NoveltyCheckResult(
        daily_effects=daily_lifts,
        daily_days=[int(d) for d in days],
        early_window_days=early_window,
        late_window_days=late_window,
        early_ttest=early_res,
        late_ttest=late_res,
        full_ttest=full_res,
        decay_percentage=decay_pct,
        has_novelty_decay=has_decay,
        verdict=verdict,
        decay_p_value=decay_p,
    )


if __name__ == "__main__":
    df_decay = generate_novelty_decay_data(num_users_per_day=1000, num_days=14, initial_tau=5.0, decay_rate=0.35)
    res = analyze_novelty_effect(df_decay)

    print("=" * 70)
    print("PHASE 4: NOVELTY EFFECT & TIME-WINDOWED DECAY ANALYSIS")
    print("=" * 70)
    print(f"Early Window (Days {res.early_window_days[0]}-{res.early_window_days[1]}): Lift = {res.early_ttest.absolute_diff:+.3f} rounds, p = {res.early_ttest.p_value:.4e}")
    print(f"Late Window  (Days {res.late_window_days[0]}-{res.late_window_days[1]}): Lift = {res.late_ttest.absolute_diff:+.3f} rounds, p = {res.late_ttest.p_value:.4f}")
    print(f"Full Window  (Days 1-14):    Lift = {res.full_ttest.absolute_diff:+.3f} rounds, p = {res.full_ttest.p_value:.4e}")
    print(f"Observed Effect Decay:       {res.decay_percentage:.1%}")
    print(f"Novelty Effect Flag:         {res.has_novelty_decay}")
    print(f"\nVerdict:\n{res.verdict}")
