"""Visualization script for Phase 2: Peeking Problem & Sequential Testing.

Generates:
1. false_positive_rate_by_peeks.png: Compares naive vs. corrected false positive rates.
2. pvalue_trajectories.png: Illustrates sample random-walk p-value paths under H0.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.peeking_simulation import run_peeking_simulation
from src.sequential_correction import get_pocock_critical_value


def generate_peeking_plots():
    figures_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run simulation across 10,000 experiments
    sim = run_peeking_simulation(num_simulations=10000, num_days=14, alpha=0.05, random_seed=42)

    # 2. Compute cumulative FPR for Pocock corrected threshold
    z_crit_pocock, alpha_pocock = get_pocock_critical_value(14, alpha=0.05)
    sig_pocock = sim.p_matrix < alpha_pocock
    cum_sig_pocock = np.maximum.accumulate(sig_pocock, axis=1)
    pocock_fpr_by_day = np.mean(cum_sig_pocock, axis=0) * 100
    naive_fpr_by_day = sim.fpr_by_day * 100
    days = np.arange(1, 15)

    # Plot 1: False Positive Rate by Peeks
    plt.figure(figsize=(10, 6), dpi=150)
    sns.set_theme(style="whitegrid")

    plt.plot(days, naive_fpr_by_day, marker="o", linewidth=2.5, color="#e74c3c", label=f"Naive Daily Peeking (Threshold = 0.05, End FPR = {naive_fpr_by_day[-1]:.1f}%)")
    plt.plot(days, pocock_fpr_by_day, marker="s", linewidth=2.5, color="#2ecc71", label=f"Pocock Corrected (Threshold = {alpha_pocock:.4f}, End FPR = {pocock_fpr_by_day[-1]:.1f}%)")
    plt.axhline(5.0, color="#34495e", linestyle="--", linewidth=1.5, label="Target Nominal Rate (5.0%)")

    plt.title("The Peeking Problem: False Positive Rate vs. Number of Interim Looks", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Daily Interim Looks (Days Elapsed)", fontsize=12, fontweight="bold")
    plt.ylabel("Cumulative False Positive Rate (%)", fontsize=12, fontweight="bold")
    plt.xticks(days)
    plt.ylim(0, 26)

    # Annotate inflation
    plt.annotate(
        f"4.4x Error Inflation!\n({naive_fpr_by_day[-1]:.1f}% False Positives)",
        xy=(14, naive_fpr_by_day[-1]),
        xytext=(10.5, 23.5),
        arrowprops=dict(facecolor="#e74c3c", arrowstyle="->", lw=1.5),
        fontweight="bold",
        color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#e74c3c")
    )

    plt.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    fig1_path = figures_dir / "false_positive_rate_by_peeks.png"
    plt.savefig(fig1_path)
    plt.close()

    # Plot 2: P-value Random Walk Trajectories under H0
    plt.figure(figsize=(10, 6), dpi=150)

    # Pick 20 sample trajectories from the 10,000 simulations
    rng = np.random.default_rng(101)
    sample_indices = rng.choice(10000, size=20, replace=False)

    # Find one experiment that had a false alarm early (e.g. day 3) but ended > 0.05 at day 14
    false_alarm_candidates = np.where((np.min(sim.p_matrix[:, :7], axis=1) < 0.05) & (sim.p_matrix[:, -1] > 0.20))[0]
    highlight_idx = false_alarm_candidates[0] if len(false_alarm_candidates) > 0 else sample_indices[0]

    for idx in sample_indices:
        if idx != highlight_idx:
            plt.plot(days, sim.p_matrix[idx], color="#bdc3c7", alpha=0.5, linewidth=1.2)

    # Highlight the false alarm trajectory
    plt.plot(days, sim.p_matrix[highlight_idx], color="#e74c3c", linewidth=2.8,
             label=f"Sample False Alarm Path (Dips below 0.05, ends at p={sim.p_matrix[highlight_idx, -1]:.2f})")

    plt.axhline(0.05, color="#e74c3c", linestyle="--", linewidth=1.5, label="Standard Significance Threshold (alpha = 0.05)")
    plt.axhline(alpha_pocock, color="#2ecc71", linestyle=":", linewidth=1.8, label=f"Pocock Adjusted Threshold (alpha* = {alpha_pocock:.4f})")

    plt.title("Sample P-Value Trajectories Over Time Under H0 (Ground Truth Effect = 0)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Experiment Day (Look)", fontsize=12, fontweight="bold")
    plt.ylabel("Observed P-Value", fontsize=12, fontweight="bold")
    plt.xticks(days)
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    fig2_path = figures_dir / "pvalue_trajectories.png"
    plt.savefig(fig2_path)
    plt.close()

    print(f"Generated Phase 2 figures in: {figures_dir}")


if __name__ == "__main__":
    generate_peeking_plots()
