"""Naive daily peeking simulation module.

This module simulates the real-world engineering and product fallacy of checking
experiment metrics every day ('peeking') and stopping immediately if p < 0.05.
It quantifies the exact Type I error inflation (False Positive Rate) over time
compared to standard fixed-horizon evaluation.
"""

import sys
from pathlib import Path
from typing import NamedTuple, Dict, Any
import numpy as np
from scipy import stats

# Ensure src modules are discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.aa_simulator import generate_aa_experiments, AASimulationData


class PeekingSimulationResult(NamedTuple):
    """Container for peeking simulation outputs."""
    num_simulations: int
    num_days: int
    alpha: float
    fixed_horizon_fpr: float              # False positive rate at day 14 only
    naive_peeking_fpr: float              # False positive rate across all 14 days
    fpr_by_day: np.ndarray                # Cumulative FPR at days 1..14
    first_stop_day_counts: Dict[int, int] # Day on which test first stopped
    z_matrix: np.ndarray                  # Shape: (num_sims, num_days)
    p_matrix: np.ndarray                  # Shape: (num_sims, num_days)


def run_peeking_simulation(
    num_simulations: int = 10000,
    num_days: int = 14,
    daily_users_per_group: int = 1000,
    true_conversion_rate: float = 0.10,
    alpha: float = 0.05,
    random_seed: int = 42
) -> PeekingSimulationResult:
    """Simulate daily peeking across thousands of A/A experiments.

    Parameters
    ----------
    num_simulations : int, default 10000
        Number of A/A experiments.
    num_days : int, default 14
        Number of daily looks (e.g. 14 days).
    daily_users_per_group : int, default 1000
        Users entering each group per day.
    true_conversion_rate : float, default 0.10
        True baseline rate (identical in both groups).
    alpha : float, default 0.05
        Nominal significance threshold.
    random_seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    result : PeekingSimulationResult
        Contains empirical error rates, daily trajectories, and stopping stats.
    """
    # 1. Generate data streams
    sim_data = generate_aa_experiments(
        num_simulations=num_simulations,
        num_days=num_days,
        daily_users_per_group=daily_users_per_group,
        true_conversion_rate=true_conversion_rate,
        random_seed=random_seed,
    )

    # 2. Vectorized cumulative proportion testing across all (sims, days)
    # cum_users is (num_days,) -> reshaped to (1, num_days) for broadcasting
    cum_n = sim_data.cum_users[np.newaxis, :]  # shape: (1, 14)
    cum_xa = sim_data.cum_successes_a          # shape: (num_sims, 14)
    cum_xb = sim_data.cum_successes_b          # shape: (num_sims, 14)

    p_a = cum_xa / cum_n
    p_b = cum_xb / cum_n
    p_pool = (cum_xa + cum_xb) / (2.0 * cum_n)

    # Standard error of difference under H0
    se_pool = np.sqrt(p_pool * (1.0 - p_pool) * (2.0 / cum_n))

    # Z-statistic matrix: shape (num_sims, num_days)
    z_matrix = (p_a - p_b) / se_pool

    # Two-sided p-values: shape (num_sims, num_days)
    p_matrix = 2.0 * stats.norm.sf(np.abs(z_matrix))

    # 3. Determine significance flags
    # sig_matrix[i, t] = True if experiment i showed p < alpha on day t
    sig_matrix = p_matrix < alpha

    # 4. Fixed Horizon (Only checking on the final day, Day 14)
    fixed_sig = sig_matrix[:, -1]
    fixed_fpr = float(np.mean(fixed_sig))

    # 5. Naive Peeking (Stopping at the FIRST day where p < alpha)
    # cum_ever_sig[i, t] = True if experiment i was significant at ANY day <= t
    cum_ever_sig = np.maximum.accumulate(sig_matrix, axis=1)
    naive_sig_final = cum_ever_sig[:, -1]
    naive_fpr = float(np.mean(naive_sig_final))

    # Cumulative false positive rate by day: shape (num_days,)
    fpr_by_day = np.mean(cum_ever_sig, axis=0)

    # 6. Distribution of first stopping day
    # For experiments that ever stopped, identify the index of the first True
    first_stop_day_counts = {}
    ever_stopped_mask = cum_ever_sig[:, -1]
    # argmax on boolean array returns first index where True appears
    first_stop_indices = np.argmax(sig_matrix[ever_stopped_mask], axis=1)
    unique_days, counts = np.unique(first_stop_indices + 1, return_counts=True)
    for day, count in zip(unique_days, counts):
        first_stop_day_counts[int(day)] = int(count)

    return PeekingSimulationResult(
        num_simulations=num_simulations,
        num_days=num_days,
        alpha=alpha,
        fixed_horizon_fpr=fixed_fpr,
        naive_peeking_fpr=naive_fpr,
        fpr_by_day=fpr_by_day,
        first_stop_day_counts=first_stop_day_counts,
        z_matrix=z_matrix,
        p_matrix=p_matrix,
    )


if __name__ == "__main__":
    res = run_peeking_simulation(num_simulations=10000, num_days=14, alpha=0.05)
    print("=" * 70)
    print("STEP 2: NAIVE DAILY PEEKING SIMULATION RESULTS (10,000 A/A EXPERIMENTS)")
    print("=" * 70)
    print(f"Nominal Target Alpha (Per-Look):     {res.alpha:.1%}")
    print(f"Fixed Horizon FPR (Day 14 only):     {res.fixed_horizon_fpr:.2%}")
    print(f"Naive Daily Peeking FPR (Any day):   {res.naive_peeking_fpr:.2%}")
    print(f"Error Rate Inflation Multiplier:     {res.naive_peeking_fpr / res.fixed_horizon_fpr:.2f}x")
    print("\nCumulative False Positive Rate Growth by Number of Looks:")
    for day, fpr in enumerate(res.fpr_by_day, 1):
        print(f"  Day {day:2d} ({day:2d} looks): {fpr:.2%}")

    print("\nFirst Day of False Alarm Triggered (Early Stoppages):")
    total_stopped = sum(res.first_stop_day_counts.values())
    for day in range(1, 15):
        cnt = res.first_stop_day_counts.get(day, 0)
        pct_of_all = (cnt / res.num_simulations) * 100
        print(f"  Day {day:2d}: {cnt:4d} false alarms ({pct_of_all:.2f}% of all tests)")
