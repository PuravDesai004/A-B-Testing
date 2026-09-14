"""Simpson's Paradox detection and subgroup confounding analysis module.

This module demonstrates Simpson's Paradox where an aggregate effect size reverses
or disappears when conditioned on confounding subgroups. It includes:
1. The classic 1986 Charig et al. kidney stone benchmark.
2. Cookie Cats device platform segmentation data (iOS vs Android retention across gate_30 vs gate_40).
3. A general-purpose Simpson's Paradox detector for segmented experimentation data.
"""

from typing import NamedTuple, Dict, List
import numpy as np
import pandas as pd
from pathlib import Path


class SimpsonsParadoxResult(NamedTuple):
    """Container holding subgroup and aggregate comparison results."""
    dataset_name: str
    group_col: str
    outcome_col: str
    segment_col: str
    aggregate_rates: Dict[str, float]
    aggregate_lift: float             # Treatment - Control
    segment_rates: Dict[str, Dict[str, float]]
    segment_lifts: Dict[str, float]
    has_simpsons_paradox: bool
    reversal_type: str
    explanation: str


def get_kidney_stone_data() -> pd.DataFrame:
    """Return the classic 1986 Charig et al. kidney stone treatment dataset.

    Treatment A: Open surgery (invasive)
    Treatment B: Percutaneous nephrolithotomy (minimally invasive)
    Subgroup: Stone Size ('Small' vs 'Large')
    Outcome: Success (1 = success, 0 = failure)
    """
    records = []
    # Small Stones:
    # Treatment A: 81 successes / 87 total (93.1%)
    for _ in range(81):
        records.append({"treatment": "A", "stone_size": "Small", "success": 1})
    for _ in range(87 - 81):
        records.append({"treatment": "A", "stone_size": "Small", "success": 0})
    # Treatment B: 234 successes / 270 total (86.7%)
    for _ in range(234):
        records.append({"treatment": "B", "stone_size": "Small", "success": 1})
    for _ in range(270 - 234):
        records.append({"treatment": "B", "stone_size": "Small", "success": 0})

    # Large Stones:
    # Treatment A: 192 successes / 263 total (73.0%)
    for _ in range(192):
        records.append({"treatment": "A", "stone_size": "Large", "success": 1})
    for _ in range(263 - 192):
        records.append({"treatment": "A", "stone_size": "Large", "success": 0})
    # Treatment B: 55 successes / 80 total (68.8%)
    for _ in range(55):
        records.append({"treatment": "B", "stone_size": "Large", "success": 1})
    for _ in range(80 - 55):
        records.append({"treatment": "B", "stone_size": "Large", "success": 0})

    return pd.DataFrame(records)


def generate_cookie_cats_segmented_data(output_path: Path = None) -> pd.DataFrame:
    """Generate Cookie Cats player activity records with device platform segmentation demonstrating Simpson's Paradox.

    Compares 7-Day Retention for gate_30 vs gate_40 across Device Platforms ('iOS' vs 'Android').

    Real-world Experimentation Context:
    In mobile gaming, iOS users often exhibit higher baseline engagement and retention than Android users.
    If an unstratified or phased feature rollout assigns 80% of Gate 40 players to iOS and 75% of Gate 30
    players to Android, Gate 40 can appear to 'win' in aggregate even though Gate 30 wins decisively on both platforms!

    Exact Parameters:
    - On iOS (High Baseline Retention):
      * gate_30: 500 players, 24.0% retention (120 retained)
      * gate_40: 2,000 players, 21.0% retention (420 retained)
      * gate_30 wins on iOS by +3.0% points!
    - On Android (Lower Baseline Retention):
      * gate_30: 1,500 players, 14.0% retention (210 retained)
      * gate_40: 500 players, 11.0% retention (55 retained)
      * gate_30 wins on Android by +3.0% points!
    - In Aggregate (Ignoring Platform):
      * gate_30: (120 + 210) / (500 + 1500) = 330 / 2,000 = 16.50%
      * gate_40: (420 + 55) / (2000 + 500) = 475 / 2,500 = 19.00%
      * Gate 40 appears to win in aggregate by +2.50% points! (QUALITATIVE REVERSAL)
    """
    records = []

    # 1. iOS: High baseline retention
    # gate_30: 500 players, 24% retention
    n_g30_ios = 500
    ret_g30_ios = int(0.24 * n_g30_ios)  # 120
    for _ in range(ret_g30_ios):
        records.append({"version": "gate_30", "platform": "iOS", "retention_7": 1})
    for _ in range(n_g30_ios - ret_g30_ios):
        records.append({"version": "gate_30", "platform": "iOS", "retention_7": 0})

    # gate_40: 2000 players, 21% retention
    n_g40_ios = 2000
    ret_g40_ios = int(0.21 * n_g40_ios)  # 420
    for _ in range(ret_g40_ios):
        records.append({"version": "gate_40", "platform": "iOS", "retention_7": 1})
    for _ in range(n_g40_ios - ret_g40_ios):
        records.append({"version": "gate_40", "platform": "iOS", "retention_7": 0})

    # 2. Android: Lower baseline retention
    # gate_30: 1500 players, 14% retention
    n_g30_android = 1500
    ret_g30_android = int(0.14 * n_g30_android)  # 210
    for _ in range(ret_g30_android):
        records.append({"version": "gate_30", "platform": "Android", "retention_7": 1})
    for _ in range(n_g30_android - ret_g30_android):
        records.append({"version": "gate_30", "platform": "Android", "retention_7": 0})

    # gate_40: 500 players, 11% retention
    n_g40_android = 500
    ret_g40_android = int(0.11 * n_g40_android)  # 55
    for _ in range(ret_g40_android):
        records.append({"version": "gate_40", "platform": "Android", "retention_7": 1})
    for _ in range(n_g40_android - ret_g40_android):
        records.append({"version": "gate_40", "platform": "Android", "retention_7": 0})

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def detect_simpsons_paradox(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    segment_col: str,
    group_a: str = None,
    group_b: str = None,
    dataset_name: str = "Dataset"
) -> SimpsonsParadoxResult:
    """Audit an experiment for Simpson's Paradox across subgroups.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with group, outcome, and segment columns.
    group_col : str
        Column denoting variant/treatment (e.g. 'deck' or 'treatment').
    outcome_col : str
        Binary outcome column (1 = win/success, 0 = loss/failure).
    segment_col : str
        Categorical segment column (e.g. 'arena_tier' or 'stone_size').
    group_a : str, optional
        Control or baseline group label. Defaults to first unique group.
    group_b : str, optional
        Treatment or comparison group label. Defaults to second unique group.
    dataset_name : str

    Returns
    -------
    result : SimpsonsParadoxResult
    """
    unique_groups = sorted(df[group_col].unique())
    if len(unique_groups) != 2:
        raise ValueError(f"Simpson's check requires exactly 2 groups. Found: {unique_groups}")

    if group_a is None:
        group_a = unique_groups[0]
    if group_b is None:
        group_b = unique_groups[1]

    # 1. Aggregate Rates
    agg_rates = {}
    for g in [group_a, group_b]:
        sub = df[df[group_col] == g]
        agg_rates[g] = float(sub[outcome_col].mean())

    agg_lift = agg_rates[group_b] - agg_rates[group_a]

    # 2. Segment-level Rates
    segments = sorted(df[segment_col].unique())
    segment_rates = {}
    segment_lifts = {}

    for s in segments:
        seg_df = df[df[segment_col] == s]
        seg_rates = {}
        for g in [group_a, group_b]:
            g_sub = seg_df[seg_df[group_col] == g]
            seg_rates[g] = float(g_sub[outcome_col].mean()) if len(g_sub) > 0 else 0.0

        segment_rates[s] = seg_rates
        segment_lifts[s] = seg_rates[group_b] - seg_rates[group_a]

    # 3. Check for Paradox / Direction Reversal
    # Paradox occurs if:
    # Lift in ALL individual segments has one sign (+ or -),
    # but the aggregate lift has the OPPOSITE sign!
    seg_lift_vals = list(segment_lifts.values())
    all_positive = all(l > 0.0 for l in seg_lift_vals)
    all_negative = all(l < 0.0 for l in seg_lift_vals)

    has_paradox = False
    reversal_type = "None"

    if all_positive and agg_lift < 0.0:
        has_paradox = True
        reversal_type = f"QUALITATIVE REVERSAL: {group_b} wins in every subgroup, but {group_a} wins in aggregate!"
    elif all_negative and agg_lift > 0.0:
        has_paradox = True
        reversal_type = f"QUALITATIVE REVERSAL: {group_a} wins in every subgroup, but {group_b} wins in aggregate!"

    if has_paradox:
        explanation = (
            f"SIMPSON'S PARADOX CONFIRMED: {reversal_type}\n"
            f"Aggregate Lift ({group_b} - {group_a}): {agg_lift:+.2%} points.\n"
            f"Subgroup Lifts:\n" +
            "\n".join([f"  - [{seg}]: {segment_lifts[seg]:+.2%} points ({group_b}: {segment_rates[seg][group_b]:.1%}, {group_a}: {segment_rates[seg][group_a]:.1%})" for seg in segments]) +
            f"\nRoot Cause: Severe sample size imbalance across segments (confounding variable '{segment_col}')."
        )
    else:
        explanation = (
            f"NO PARADOX DETECTED: Aggregate lift ({agg_lift:+.2%} points) is directionally consistent "
            f"with subgroup lifts."
        )

    return SimpsonsParadoxResult(
        dataset_name=dataset_name,
        group_col=group_col,
        outcome_col=outcome_col,
        segment_col=segment_col,
        aggregate_rates=agg_rates,
        aggregate_lift=agg_lift,
        segment_rates=segment_rates,
        segment_lifts=segment_lifts,
        has_simpsons_paradox=has_paradox,
        reversal_type=reversal_type,
        explanation=explanation,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 4: SIMPSON'S PARADOX & SUBGROUP CONFOUNDING AUDIT")
    print("=" * 70)

    # 1. Benchmark 1: Classic 1986 Kidney Stone Study
    df_kidney = get_kidney_stone_data()
    res_kidney = detect_simpsons_paradox(
        df_kidney,
        group_col="treatment",
        outcome_col="success",
        segment_col="stone_size",
        group_a="A",
        group_b="B",
        dataset_name="1986 Kidney Stone Study (Charig et al.)"
    )
    print(f"\n--- Benchmark 1: {res_kidney.dataset_name} ---")
    print(res_kidney.explanation)

    # 2. Benchmark 2: Cookie Cats Device Platform Confounding
    cc_path = Path(__file__).resolve().parent.parent / "data" / "cookie_cats_segmented.csv"
    df_cc = generate_cookie_cats_segmented_data(output_path=cc_path)
    res_cc = detect_simpsons_paradox(
        df_cc,
        group_col="version",
        outcome_col="retention_7",
        segment_col="platform",
        group_a="gate_30",
        group_b="gate_40",
        dataset_name="Cookie Cats Device Platform Confounding"
    )
    print(f"\n--- Benchmark 2: {res_cc.dataset_name} ---")
    print(res_cc.explanation)
