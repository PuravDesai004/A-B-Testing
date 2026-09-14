"""Synthetic A/A test stream generator for simulating multi-day experimentation.

This module generates synthetic A/A experimentation streams where the ground truth
difference between Group A and Group B is mathematically zero (p_A == p_B).
Because ground truth is known with certainty, any test that rejects H0 is guaranteed
to be a False Positive (Type I error).
"""

from typing import NamedTuple
import numpy as np


class AASimulationData(NamedTuple):
    """Container holding vectorized daily conversion data for A/A experiments."""
    num_sims: int
    num_days: int
    daily_users: int
    true_rate: float
    successes_a: np.ndarray  # Shape: (num_sims, num_days)
    successes_b: np.ndarray  # Shape: (num_sims, num_days)
    cum_successes_a: np.ndarray
    cum_successes_b: np.ndarray
    cum_users: np.ndarray    # Shape: (num_days,) cumulative users per group


def generate_aa_experiments(
    num_simulations: int = 10000,
    num_days: int = 14,
    daily_users_per_group: int = 1000,
    true_conversion_rate: float = 0.10,
    random_seed: int = 42
) -> AASimulationData:
    """Generate vectorized daily conversion streams for thousands of A/A experiments.

    Parameters
    ----------
    num_simulations : int, default 10000
        Number of independent A/A experiments to simulate.
    num_days : int, default 14
        Duration of each experiment in days (number of daily looks).
    daily_users_per_group : int, default 1000
        Number of newly enrolled users in each group each day.
        Total sample size per group at day 14 = 14 * 1000 = 14,000.
    true_conversion_rate : float, default 0.10
        The true, fixed conversion probability for both groups.
    random_seed : int, default 42
        Seed for reproducibility.

    Returns
    -------
    data : AASimulationData
        Container with daily and cumulative success matrices.
    """
    if not (0.0 < true_conversion_rate < 1.0):
        raise ValueError(f"Conversion rate must be in (0, 1). Got {true_conversion_rate}")
    if num_simulations <= 0 or num_days <= 0 or daily_users_per_group <= 0:
        raise ValueError("Simulation dimensions must be positive integers.")

    rng = np.random.default_rng(random_seed)

    # Simulate daily binomial conversions for both groups simultaneously:
    # successes_a[i, t] = number of conversions in experiment i on day t
    successes_a = rng.binomial(
        n=daily_users_per_group,
        p=true_conversion_rate,
        size=(num_simulations, num_days)
    )
    successes_b = rng.binomial(
        n=daily_users_per_group,
        p=true_conversion_rate,
        size=(num_simulations, num_days)
    )

    # Compute running cumulative sums across days (axis=1)
    cum_successes_a = np.cumsum(successes_a, axis=1)
    cum_successes_b = np.cumsum(successes_b, axis=1)

    # Cumulative users enrolled per group by day t: [1000, 2000, ..., 14000]
    cum_users = np.arange(1, num_days + 1) * daily_users_per_group

    return AASimulationData(
        num_sims=num_simulations,
        num_days=num_days,
        daily_users=daily_users_per_group,
        true_rate=true_conversion_rate,
        successes_a=successes_a,
        successes_b=successes_b,
        cum_successes_a=cum_successes_a,
        cum_successes_b=cum_successes_b,
        cum_users=cum_users,
    )


if __name__ == "__main__":
    data = generate_aa_experiments(num_simulations=5, num_days=14, daily_users_per_group=1000)
    print("=== Step 1: Synthetic A/A Generator Demo ===")
    print(f"Simulations generated: {data.num_sims}")
    print(f"Days per experiment:   {data.num_days}")
    print(f"Users per day/group:   {data.daily_users:,}")
    print(f"Total N at Day 14:     {data.cum_users[-1]:,} per group")
    print(f"True Conversion Rate:  {data.true_rate:.1%}")
    print("\nSample Experiment #1 Daily Successes (Group A):", data.successes_a[0])
    print("Sample Experiment #1 Daily Successes (Group B):", data.successes_b[0])
    print("Sample Experiment #1 Cumulative Successes (Group A):", data.cum_successes_a[0])
    print("Sample Experiment #1 Cumulative Successes (Group B):", data.cum_successes_b[0])
