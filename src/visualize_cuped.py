"""Visualization script for Phase 3: CUPED Variance Reduction.

Generates:
1. reports/figures/cuped_before_after.png:
   - Panel A: Kernel density comparison of raw Y vs. CUPED-adjusted Y (showing variance narrowing).
   - Panel B: 95% Confidence Interval contraction comparing raw vs. CUPED treatment effect.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.synthetic_cuped_data import generate_cuped_dataset
from src.cuped import run_cuped_analysis


def generate_cuped_plots():
    figures_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate dataset and run analysis
    dataset = generate_cuped_dataset(num_users=10000, true_tau=1.50, target_rho=0.75, random_seed=42)
    df_cuped, metrics = run_cuped_analysis(dataset.df)

    # Setup plotting canvas
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    sns.set_theme(style="whitegrid")

    # Panel A: Distribution Variance Narrowing (Control Group)
    ctrl_data = df_cuped[df_cuped["variant"] == "control"]
    raw_ctrl = ctrl_data["post_rounds"]
    cuped_ctrl = ctrl_data["cuped_outcome"]

    sns.kdeplot(
        raw_ctrl,
        ax=axes[0],
        color="#e74c3c",
        linewidth=2.5,
        fill=True,
        alpha=0.25,
        label=f"Raw Outcome Y (Var: {metrics.raw_var_control:.1f})"
    )
    sns.kdeplot(
        cuped_ctrl,
        ax=axes[0],
        color="#2ecc71",
        linewidth=2.5,
        fill=True,
        alpha=0.35,
        label=f"CUPED Outcome Y_cuped (Var: {metrics.cuped_var_control:.1f})"
    )

    axes[0].set_title(
        f"A. Variance Reduction via CUPED (-{metrics.empirical_var_reduction:.1%})",
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    axes[0].set_xlabel("Rounds Played", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    axes[0].set_xlim(0, 100)
    axes[0].legend(frameon=True, facecolor="white", loc="upper right")

    # Panel B: Confidence Interval Contraction
    y_positions = [1, 0]
    estimates = [metrics.raw_ttest.absolute_diff, metrics.cuped_ttest.absolute_diff]
    ci_lowers = [metrics.raw_ttest.ci_lower, metrics.cuped_ttest.ci_lower]
    ci_uppers = [metrics.raw_ttest.ci_upper, metrics.cuped_ttest.ci_upper]
    errors = [
        [estimates[0] - ci_lowers[0], ci_uppers[0] - estimates[0]],
        [estimates[1] - ci_lowers[1], ci_uppers[1] - estimates[1]],
    ]
    colors = ["#e74c3c", "#2ecc71"]
    labels = [
        f"Raw Y (Width: {metrics.ci_width_raw:.2f})\np = {metrics.raw_ttest.p_value:.2e}",
        f"CUPED Y (Width: {metrics.ci_width_cuped:.2f})\np = {metrics.cuped_ttest.p_value:.2e}"
    ]

    for i in range(2):
        axes[1].errorbar(
            estimates[i],
            y_positions[i],
            xerr=[[errors[i][0]], [errors[i][1]]],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            elinewidth=3,
            capsize=8,
            capthick=2,
            markersize=8,
            label=labels[i]
        )

    # Injected true effect line
    axes[1].axvline(
        dataset.true_tau,
        color="#2980b9",
        linestyle="--",
        linewidth=1.8,
        label=f"True Known Effect (tau = +{dataset.true_tau:.2f})"
    )

    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(["Raw Analysis", "CUPED Analysis"], fontsize=11, fontweight="bold")
    axes[1].set_title(
        f"B. 95% Confidence Interval Contraction (-{metrics.ci_width_reduction:.1%})",
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    axes[1].set_xlabel("Estimated Treatment Effect (Rounds)", fontsize=11, fontweight="bold")
    axes[1].legend(frameon=True, facecolor="white", loc="lower right")

    plt.tight_layout()
    plot_path = figures_dir / "cuped_before_after.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Generated CUPED figures at: {plot_path}")


if __name__ == "__main__":
    generate_cuped_plots()
