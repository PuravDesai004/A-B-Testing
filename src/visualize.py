import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.data_loader import load_cookie_cats_data, get_group_metrics


def generate_visualizations():
    figures_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_clean, _ = load_cookie_cats_data(remove_extreme_outliers=True)
    metrics = get_group_metrics(df_clean)

    # 1. Retention Plot
    plt.figure(figsize=(9, 5), dpi=150)
    sns.set_theme(style="whitegrid")

    labels = ["Day 1 Retention", "Day 7 Retention"]
    gate30_rates = [
        metrics["gate_30"]["retention_1"]["rate"] * 100,
        metrics["gate_30"]["retention_7"]["rate"] * 100,
    ]
    gate40_rates = [
        metrics["gate_40"]["retention_1"]["rate"] * 100,
        metrics["gate_40"]["retention_7"]["rate"] * 100,
    ]

    x = np.arange(len(labels))
    width = 0.32

    rects1 = plt.bar(x - width/2, gate30_rates, width, label="Gate 30 (Control)", color="#3498db")
    rects2 = plt.bar(x + width/2, gate40_rates, width, label="Gate 40 (Treatment)", color="#e74c3c")

    plt.ylabel("Retention Rate (%)", fontsize=12, fontweight="bold")
    plt.title("Cookie Cats: Retention Rates by Gate Placement", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, labels, fontsize=11, fontweight="bold")
    plt.legend(frameon=True, facecolor="white")
    plt.ylim(0, 52)

    for rect in rects1:
        height = rect.get_height()
        plt.annotate(f"{height:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    for rect in rects2:
        height = rect.get_height()
        plt.annotate(f"{height:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plot_path1 = figures_dir / "retention_comparison.png"
    plt.savefig(plot_path1)
    plt.close()

    # 2. Gamerounds distribution plot (clipped to 100 for readability)
    plt.figure(figsize=(9, 5), dpi=150)
    sns.boxplot(
        data=df_clean[df_clean["sum_gamerounds"] <= 100],
        x="version",
        y="sum_gamerounds",
        hue="version",
        palette=["#3498db", "#e74c3c"],
        notch=True,
        legend=False,
    )
    plt.xlabel("Experiment Version", fontsize=12, fontweight="bold")
    plt.ylabel("Rounds Played (first 14 days, <= 100)", fontsize=12, fontweight="bold")
    plt.title("Rounds Played Distribution (Median & IQR)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plot_path2 = figures_dir / "gamerounds_distribution.png"
    plt.savefig(plot_path2)
    plt.close()

    print(f"Generated plots at: {figures_dir}")


if __name__ == "__main__":
    generate_visualizations()
