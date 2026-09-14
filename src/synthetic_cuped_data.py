"""Synthetic pre/post experiment dataset generator for CUPED variance reduction.

Because the Cookie Cats dataset only tracks players starting at install (no pre-experiment
history exists), this module generates a synthetic mobile gaming experimentation dataset
with a genuine pre-experiment covariate (X) correlated with post-experiment outcome (Y)
and an injected, known ground-truth treatment effect (tau).
"""

from typing import NamedTuple
import numpy as np
import pandas as pd


class CUPEDDataset(NamedTuple):
    """Container holding synthetic experiment data and population parameters."""
    df: pd.DataFrame
    num_users: int
    true_tau: float           # Known injected treatment effect
    target_correlation: float # Population correlation rho(X, Y)
    empirical_correlation: float


def generate_cuped_dataset(
    num_users: int = 10000,
    true_tau: float = 1.50,
    target_rho: float = 0.75,
    mean_pre: float = 50.0,
    std_pre: float = 20.0,
    random_seed: int = 42
) -> CUPEDDataset:
    """Generate synthetic experiment data with correlated pre- and post-metrics.

    The model:
        X_i (pre-experiment rounds) ~ Normal(mean_pre, std_pre^2)
        Y_i (raw post-experiment rounds) = beta * X_i + epsilon_i + tau * Treatment_i

    Parameters
    ----------
    num_users : int, default 10000
        Total users enrolled in the experiment (50% control, 50% treatment).
    true_tau : float, default 1.50
        Ground-truth treatment effect (lift in gamerounds for treatment group).
    target_rho : float, default 0.75
        Desired linear correlation between pre-metric X and post-metric Y.
    mean_pre : float, default 50.0
        Mean pre-experiment rounds played.
    std_pre : float, default 20.0
        Standard deviation of pre-experiment rounds played.
    random_seed : int, default 42
        Seed for reproducibility.

    Returns
    -------
    dataset : CUPEDDataset
        NamedTuple with pandas DataFrame and summary metadata.
    """
    if not (0.0 < target_rho < 1.0):
        raise ValueError(f"Target correlation must be in (0, 1). Got {target_rho}")
    if num_users <= 0:
        raise ValueError("Number of users must be positive.")

    rng = np.random.default_rng(random_seed)

    # 1. Simulate Pre-experiment Covariate X (e.g. rounds played in prior 14 days)
    # Ensure non-negative rounds by clipping at 0
    raw_x = rng.normal(loc=mean_pre, scale=std_pre, size=num_users)
    x = np.clip(raw_x, a_min=0.0, a_max=None)

    # 2. Random 50/50 Treatment Assignment
    # 0 = Control, 1 = Treatment
    variant = rng.choice(["control", "treatment"], size=num_users, p=[0.5, 0.5])
    is_treatment = (variant == "treatment").astype(float)

    # 3. Derive regression slope beta and residual noise variance to hit target rho
    # For Y = beta * X + epsilon:
    # Var(Y) = beta^2 * Var(X) + Var(epsilon)
    # Cov(X, Y) = beta * Var(X)
    # Corr(X, Y) = Cov(X, Y) / (std(X) * std(Y)) = beta * std(X) / std(Y)
    # Choosing std(Y) ~ std(X): beta = target_rho, and Var(epsilon) = std(X)^2 * (1 - target_rho^2)
    beta = target_rho
    std_eps = std_pre * np.sqrt(1.0 - (target_rho ** 2))

    epsilon = rng.normal(loc=0.0, scale=std_eps, size=num_users)

    # Baseline intercept chosen so E[Y | control] == E[X]
    beta_0 = mean_pre * (1.0 - beta)

    # Post-experiment outcome Y:
    # Notice: treatment effect tau only affects the post-period outcome Y, NOT pre-period X!
    # Pre-period X is measured before randomization, so it cannot be affected by treatment.
    y = beta_0 + (beta * x) + epsilon + (true_tau * is_treatment)
    y = np.clip(y, a_min=0.0, a_max=None)

    df = pd.DataFrame({
        "userid": np.arange(100001, 100001 + num_users),
        "variant": variant,
        "pre_rounds": x,       # Covariate X
        "post_rounds": y,      # Outcome Y
    })

    # Compute empirical correlation in control group (uncontaminated by treatment)
    ctrl_mask = df["variant"] == "control"
    emp_corr = float(np.corrcoef(df.loc[ctrl_mask, "pre_rounds"], df.loc[ctrl_mask, "post_rounds"])[0, 1])

    return CUPEDDataset(
        df=df,
        num_users=num_users,
        true_tau=true_tau,
        target_correlation=target_rho,
        empirical_correlation=emp_corr,
    )


if __name__ == "__main__":
    dataset = generate_cuped_dataset(num_users=10000, true_tau=1.50, target_rho=0.75)
    print("=== Step 2: Synthetic CUPED Dataset Generated ===")
    print(f"Total Users:              {dataset.num_users:,}")
    print(f"Known Treatment Effect:   +{dataset.true_tau:.2f} gamerounds")
    print(f"Target Correlation (rho): {dataset.target_correlation:.2f}")
    print(f"Empirical Control Corr:   {dataset.empirical_correlation:.4f}")
    print("\nSample Data Records:")
    print(dataset.df.head(10))
    print("\nGroup Summary Statistics:")
    print(dataset.df.groupby("variant")[["pre_rounds", "post_rounds"]].agg(["mean", "std", "count"]))
