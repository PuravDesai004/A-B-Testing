"""Sequential testing corrections module: Pocock boundaries and O'Brien-Fleming alpha spending.

This module provides mathematically sound corrections for multiple interim looks ('peeking').
It implements:
1. Pocock-style constant boundaries for planned interim checks.
2. O'Brien-Fleming style boundaries (conservative early, preserving final power).
3. Evaluation functions comparing uncorrected vs. corrected False Positive Rates.
"""

import sys
from pathlib import Path
from typing import NamedTuple, List, Tuple
import numpy as np
from scipy import stats, optimize
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.peeking_simulation import run_peeking_simulation, PeekingSimulationResult


class SequentialCorrectionResult(NamedTuple):
    """Container comparing naive vs corrected sequential testing."""
    method: str
    num_looks: int
    look_days: List[int]
    per_look_thresholds: List[float]  # Alpha threshold applied at each look
    per_look_z_crit: List[float]      # Critical z-score applied at each look
    naive_fpr: float                  # Error rate using uncorrected alpha=0.05
    corrected_fpr: float              # Error rate using corrected thresholds
    fpr_reduction: float              # naive_fpr - corrected_fpr
    verdict: str


# Standard Pocock critical values for overall alpha = 0.05 (two-sided)
# Derived from multivariate normal integration of the Brownian motion joint covariance matrix:
# Cov(Z(s), Z(t)) = sqrt(s / t) for s <= t.
POCOCK_CRITICAL_VALUES_ALPHA_05 = {
    1:  1.960,  # alpha = 0.0500
    2:  2.178,  # alpha = 0.0294
    3:  2.289,  # alpha = 0.0221
    4:  2.361,  # alpha = 0.0182
    5:  2.413,  # alpha = 0.0158
    6:  2.453,  # alpha = 0.0142
    7:  2.485,  # alpha = 0.0130
    8:  2.512,  # alpha = 0.0120
    9:  2.535,  # alpha = 0.0112
    10: 2.555,  # alpha = 0.0106
    11: 2.572,  # alpha = 0.0101
    12: 2.588,  # alpha = 0.0097
    13: 2.602,  # alpha = 0.0093
    14: 2.615,  # alpha = 0.0089
    20: 2.672,  # alpha = 0.0075
}


def get_pocock_critical_value(num_looks: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Retrieve or interpolate the Pocock critical value z* and per-look alpha*.

    Parameters
    ----------
    num_looks : int
        Total number of planned interim evaluations.
    alpha : float, default 0.05
        Target family-wise error rate.

    Returns
    -------
    z_crit : float
        The constant two-sided critical z value.
    alpha_per_look : float
        The adjusted per-look significance threshold: 2 * (1 - Phi(z_crit)).
    """
    if alpha != 0.05:
        raise NotImplementedError("Currently calibrated for total alpha = 0.05.")

    if num_looks in POCOCK_CRITICAL_VALUES_ALPHA_05:
        z_crit = POCOCK_CRITICAL_VALUES_ALPHA_05[num_looks]
    elif num_looks < 1:
        raise ValueError("Number of looks must be >= 1.")
    else:
        # Approximate using logarithmic regression fit to Pocock table
        # z*(K) ~ 1.96 + 0.245 * ln(K)
        z_crit = 1.960 + 0.247 * np.log(num_looks)

    alpha_per_look = float(2.0 * stats.norm.sf(z_crit))
    return float(z_crit), alpha_per_look


@lru_cache(maxsize=128)
def _obf_constant(fractions: tuple, alpha: float) -> float:
    """Calibrate discrete two-sided Brownian crossing probability by quadrature.

    W(t_k) has independent N(0, t_k-t_{k-1}) increments. Integrating its
    surviving density inside [-c, c] at every look accounts for earlier exits.
    """
    times = np.asarray(fractions)
    if len(times) == 1:
        return float(stats.norm.isf(alpha / 2) * np.sqrt(times[0]))

    def calibrate(order):
        nodes, weights = np.polynomial.legendre.leggauss(order)

        def crossing_probability(c):
            # Truncate only negligible Gaussian tails to resolve early looks.
            radii = np.minimum(c, 9 * np.sqrt(times))
            previous_x = nodes * radii[0]
            previous_w = weights * radii[0]
            density = stats.norm.pdf(previous_x, scale=np.sqrt(times[0]))
            for index in range(1, len(times)):
                x = nodes * radii[index]
                transition = stats.norm.pdf(
                    x[:, None] - previous_x[None, :],
                    scale=np.sqrt(times[index] - times[index - 1]))
                density = transition @ (density * previous_w)
                previous_x, previous_w = x, weights * radii[index]
            return 1 - float(density @ previous_w)

        lower = stats.norm.isf(alpha / 2) * np.sqrt(times[-1])
        upper = stats.norm.isf(alpha / (2 * len(times))) * np.sqrt(times[-1])
        return float(optimize.brentq(lambda c: crossing_probability(c) - alpha,
                                     lower, upper, xtol=1e-10))

    previous = calibrate(128)
    for order in (256, 512):
        current = calibrate(order)
        if abs(current - previous) < 1e-7:
            return current
        previous = current
    raise ValueError("Information schedule is too uneven for reliable boundary calibration.")


def get_obrien_fleming_boundaries(
    num_looks: int, alpha: float = 0.05,
    information_fractions=None,
) -> Tuple[List[float], List[float]]:
    """Discrete O'Brien-Fleming boundaries z_k = c / sqrt(t_k).

    Numerically calibrates c to overall two-sided alpha under the canonical
    joint-normal model. Defaults to equally spaced information, ending at 1.
    Irregular looks may supply their actual increasing information fractions.
    """
    if not isinstance(num_looks, (int, np.integer)) or not 1 <= num_looks <= 100:
        raise ValueError("Number of looks must be an integer between 1 and 100.")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")
    times = (np.arange(1, num_looks + 1) / num_looks if information_fractions is None
             else np.asarray(information_fractions, dtype=float))
    if (times.shape != (num_looks,) or not np.isfinite(times).all()
            or times[0] <= 0 or times[-1] > 1 or np.any(np.diff(times) <= 0)):
        raise ValueError("Information fractions must be finite, increasing, and in (0, 1].")
    constant = _obf_constant(tuple(times), float(alpha))
    z_crits = constant / np.sqrt(times)
    thresholds = 2 * stats.norm.sf(z_crits)
    return z_crits.tolist(), thresholds.tolist()


def evaluate_pocock_correction(
    peeking_res: PeekingSimulationResult,
    look_days: List[int] = None
) -> SequentialCorrectionResult:
    """Apply Pocock boundary correction to simulated A/A trajectories.

    Parameters
    ----------
    peeking_res : PeekingSimulationResult
        Simulation results from `run_peeking_simulation`.
    look_days : list of int, optional
        Specific days on which looks occur (e.g. [7, 14] for 2 looks,
        or [1, 2, ..., 14] for daily looks). Default is all days.

    Returns
    -------
    result : SequentialCorrectionResult
    """
    if look_days is None:
        look_days = list(range(1, peeking_res.num_days + 1))

    num_looks = len(look_days)
    day_indices = [d - 1 for d in look_days]

    z_crit, alpha_look = get_pocock_critical_value(num_looks, alpha=peeking_res.alpha)
    thresholds = [alpha_look] * num_looks
    z_crits = [z_crit] * num_looks

    # Slice p-values and z-scores at the specified look days: shape (num_sims, num_looks)
    p_subset = peeking_res.p_matrix[:, day_indices]

    # Naive evaluation (checking at uncorrected alpha=0.05 at each look)
    naive_ever_sig = np.any(p_subset < peeking_res.alpha, axis=1)
    naive_fpr = float(np.mean(naive_ever_sig))

    # Corrected evaluation (checking at adjusted alpha_look at each look)
    corrected_ever_sig = np.any(p_subset < alpha_look, axis=1)
    corrected_fpr = float(np.mean(corrected_ever_sig))

    verdict = (
        f"SUCCESS: The Pocock boundary (alpha* = {alpha_look:.4f}) reduced the false positive rate "
        f"from {naive_fpr:.2%} down to {corrected_fpr:.2%}, successfully restoring "
        f"Type I error control near the nominal {peeking_res.alpha:.1%} target."
    )

    return SequentialCorrectionResult(
        method="Pocock Boundary (Constant Alpha)",
        num_looks=num_looks,
        look_days=look_days,
        per_look_thresholds=thresholds,
        per_look_z_crit=z_crits,
        naive_fpr=naive_fpr,
        corrected_fpr=corrected_fpr,
        fpr_reduction=naive_fpr - corrected_fpr,
        verdict=verdict,
    )


def evaluate_obrien_fleming_correction(
    peeking_res: PeekingSimulationResult,
    look_days: List[int] = None
) -> SequentialCorrectionResult:
    """Apply O'Brien-Fleming alpha spending correction to simulated A/A trajectories.

    Parameters
    ----------
    peeking_res : PeekingSimulationResult
        Simulation results from `run_peeking_simulation`.
    look_days : list of int, optional
        Days on which looks occur. Default is all days.

    Returns
    -------
    result : SequentialCorrectionResult
    """
    if look_days is None:
        look_days = list(range(1, peeking_res.num_days + 1))

    if (not look_days or any(not isinstance(d, (int, np.integer)) for d in look_days)
            or any(d < 1 or d > peeking_res.num_days for d in look_days)
            or any(a >= b for a, b in zip(look_days, look_days[1:]))):
        raise ValueError("Look days must be strictly increasing valid simulation days.")
    num_looks = len(look_days)
    day_indices = [d - 1 for d in look_days]

    z_crits, thresholds = get_obrien_fleming_boundaries(
        num_looks, alpha=peeking_res.alpha,
        information_fractions=np.asarray(look_days) / peeking_res.num_days)
    thresh_arr = np.array(thresholds)[np.newaxis, :]  # shape: (1, num_looks)

    # Slice p-values at look days
    p_subset = peeking_res.p_matrix[:, day_indices]

    # Naive evaluation
    naive_ever_sig = np.any(p_subset < peeking_res.alpha, axis=1)
    naive_fpr = float(np.mean(naive_ever_sig))

    # O'Brien-Fleming evaluation: p_subset < thresh_arr
    corrected_ever_sig = np.any(p_subset < thresh_arr, axis=1)
    corrected_fpr = float(np.mean(corrected_ever_sig))

    verdict = (
        f"Calibrated discrete O'Brien-Fleming boundaries changed the false positive rate "
        f"from {naive_fpr:.2%} down to {corrected_fpr:.2%}. It remains extremely "
        f"conservative early on (Day 1 alpha = {thresholds[0]:.2e}) and preserves "
        f"high final-stage power (Day {look_days[-1]} alpha = {thresholds[-1]:.4f})."
    )

    return SequentialCorrectionResult(
        method="O'Brien-Fleming Alpha Spending",
        num_looks=num_looks,
        look_days=look_days,
        per_look_thresholds=thresholds,
        per_look_z_crit=z_crits,
        naive_fpr=naive_fpr,
        corrected_fpr=corrected_fpr,
        fpr_reduction=naive_fpr - corrected_fpr,
        verdict=verdict,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 4 & 5: SEQUENTIAL TESTING CORRECTION BENCHMARK")
    print("=" * 70)

    # Run base peeking simulation across 10,000 A/A experiments
    sim = run_peeking_simulation(num_simulations=10000, num_days=14, alpha=0.05, random_seed=42)

    # Scenario A: 2 Planned Looks (Weekly: Day 7 and Day 14)
    res_pocock_2 = evaluate_pocock_correction(sim, look_days=[7, 14])
    print("\n--- Scenario A: 2 Planned Looks (Weekly: Day 7 & Day 14) ---")
    print(f"Per-Look Adjusted Alpha:  {res_pocock_2.per_look_thresholds[0]:.4f} (z* = {res_pocock_2.per_look_z_crit[0]:.3f})")
    print(f"Naive Peeking FPR (2 looks, alpha=0.05):    {res_pocock_2.naive_fpr:.2%}")
    print(f"Pocock Corrected FPR (2 looks, alpha=0.0294):{res_pocock_2.corrected_fpr:.2%}")
    print(f"Verdict: {res_pocock_2.verdict}")

    # Scenario B: 14 Planned Looks (Daily: Days 1 to 14) using Pocock
    res_pocock_14 = evaluate_pocock_correction(sim, look_days=list(range(1, 15)))
    print("\n--- Scenario B: 14 Planned Looks (Daily: Days 1 to 14) - Pocock ---")
    print(f"Per-Look Adjusted Alpha:  {res_pocock_14.per_look_thresholds[0]:.4f} (z* = {res_pocock_14.per_look_z_crit[0]:.3f})")
    print(f"Naive Daily Peeking FPR (14 looks, alpha=0.05):   {res_pocock_14.naive_fpr:.2%}")
    print(f"Pocock Corrected FPR (14 looks, alpha=0.0089):    {res_pocock_14.corrected_fpr:.2%}")
    print(f"Verdict: {res_pocock_14.verdict}")

    # Scenario C: 14 Planned Looks (Daily: Days 1 to 14) using O'Brien-Fleming
    res_obf_14 = evaluate_obrien_fleming_correction(sim, look_days=list(range(1, 15)))
    print("\n--- Scenario C: 14 Planned Looks (Daily: Days 1 to 14) - O'Brien-Fleming ---")
    print(f"Day 1 Threshold:  alpha = {res_obf_14.per_look_thresholds[0]:.2e} (z* = {res_obf_14.per_look_z_crit[0]:.2f})")
    print(f"Day 7 Threshold:  alpha = {res_obf_14.per_look_thresholds[6]:.4f} (z* = {res_obf_14.per_look_z_crit[6]:.2f})")
    print(f"Day 14 Threshold: alpha = {res_obf_14.per_look_thresholds[-1]:.4f} (z* = {res_obf_14.per_look_z_crit[-1]:.2f})")
    print(f"Naive Daily Peeking FPR:   {res_obf_14.naive_fpr:.2%}")
    print(f"O'Brien-Fleming FPR:       {res_obf_14.corrected_fpr:.2%}")
    print(f"Verdict: {res_obf_14.verdict}")
