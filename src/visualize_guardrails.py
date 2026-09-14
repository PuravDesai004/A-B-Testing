"""Visualization script for Phase 4: Guardrail Diagnostics.

Generates:
1. reports/figures/novelty_effect_decay.png: Time-windowed novelty decay curve.
2. reports/figures/simpsons_paradox_segments.png: Visual reversal in Simpson's Paradox.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.novelty_check import generate_novelty_decay_data, analyze_novelty_effect
from src.simpsons_check import generate_cookie_cats_segmented_data, detect_simpsons_paradox


def generate_guardrail_plots():
    figures_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Novelty Decay Plot
    df_decay = generate_novelty_decay_data(num_users_per_day=1000, num_days=14, initial_tau=5.0, decay_rate=0.35)
    res_novelty = analyze_novelty_effect(df_decay)

    plt.figure(figsize=(10, 5), dpi=150)
    days = np.array(res_novelty.daily_days)
    lifts = np.array(res_novelty.daily_effects)

    plt.plot(days, lifts, marker="o", color="#e74c3c", linewidth=2.5, label="Observed Daily Lift (Treatment - Control)")
    plt.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1.2)

    # Shading Early Window
    plt.axvspan(0.8, 3.2, color="#3498db", alpha=0.18, label=f"Early Window (Days 1-3: +{res_novelty.early_ttest.absolute_diff:.2f} rnds)")
    # Shading Late Window
    plt.axvspan(11.8, 14.2, color="#95a5a6", alpha=0.25, label=f"Late Window (Days 12-14: +{res_novelty.late_ttest.absolute_diff:.2f} rnds)")

    # Trend line
    z = np.polyfit(days, lifts, 2)
    p = np.poly1d(z)
    plt.plot(days, p(days), color="#c0392b", linestyle=":", linewidth=2.0, label="Quadratic Trend Fit (Decaying)")

    plt.title(f"Guardrail Check: Novelty Effect Decay (-{res_novelty.decay_percentage:.1%} Attenuation)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Experiment Day Elapsed", fontsize=11, fontweight="bold")
    plt.ylabel("Treatment Lift (Rounds Played)", fontsize=11, fontweight="bold")
    plt.xticks(days)
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    fig1_path = figures_dir / "novelty_effect_decay.png"
    plt.savefig(fig1_path)
    plt.close()

    # 2. Simpson's Paradox Plot (Cookie Cats Platform Confounding)
    df_cc = generate_cookie_cats_segmented_data()
    res_cc = detect_simpsons_paradox(
        df_cc,
        group_col="version",
        outcome_col="retention_7",
        segment_col="platform",
        group_a="gate_30",
        group_b="gate_40"
    )

    plt.figure(figsize=(10, 5.5), dpi=150)
    categories = ["Android (Low Baseline)", "iOS (High Baseline)", "Combined Aggregate"]
    gate_30_rates = [
        res_cc.segment_rates["Android"]["gate_30"] * 100,
        res_cc.segment_rates["iOS"]["gate_30"] * 100,
        res_cc.aggregate_rates["gate_30"] * 100,
    ]
    gate_40_rates = [
        res_cc.segment_rates["Android"]["gate_40"] * 100,
        res_cc.segment_rates["iOS"]["gate_40"] * 100,
        res_cc.aggregate_rates["gate_40"] * 100,
    ]

    x = np.arange(len(categories))
    width = 0.32

    rects1 = plt.bar(x - width/2, gate_30_rates, width, label="gate_30", color="#2980b9")
    rects2 = plt.bar(x + width/2, gate_40_rates, width, label="gate_40", color="#e67e22")

    plt.ylabel("7-Day Retention Rate (%)", fontsize=11, fontweight="bold")
    plt.title("Simpson's Paradox in Cookie Cats: Platform Subgroups vs. Aggregate Reversal", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, categories, fontsize=11, fontweight="bold")
    plt.ylim(0, 35)
    plt.legend(frameon=True, facecolor="white")

    # Annotate percentages
    for rect in rects1:
        h = rect.get_height()
        plt.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        plt.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Annotate the paradox on aggregate bar
    plt.annotate(
        "PARADOX REVERSAL!\nGate 40 wins aggregate\ndespite losing on iOS & Android",
        xy=(2 + width/2, gate_40_rates[2]),
        xytext=(1.5, 27),
        arrowprops=dict(facecolor="#e74c3c", arrowstyle="->", lw=1.8),
        fontweight="bold",
        color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#e74c3c")
    )

    plt.tight_layout()
    fig2_path = figures_dir / "simpsons_paradox_segments.png"
    plt.savefig(fig2_path)
    plt.close()

    print(f"Generated Phase 4 figures in: {figures_dir}")


if __name__ == "__main__":
    generate_guardrail_plots()
