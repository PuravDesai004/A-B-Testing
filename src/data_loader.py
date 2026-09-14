"""Data loading and exploratory validation module for the Cookie Cats dataset.

This module loads the Cookie Cats A/B testing dataset, performs health checks
(checking for nulls, duplicates, and extreme anomalies), and extracts descriptive
group-level summaries for continuous and binary metrics.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd


def get_data_path() -> Path:
    """Resolve the dataset path across project directory structures."""
    candidate_paths = [
        Path(__file__).resolve().parent.parent / "data" / "cookie_cats.csv",
        Path(__file__).resolve().parent.parent / "cookie_cats.csv",
        Path("data/cookie_cats.csv"),
        Path("cookie_cats.csv"),
    ]
    for p in candidate_paths:
        if p.exists():
            return p
    raise FileNotFoundError("cookie_cats.csv could not be located in data/ or root directory.")


def load_cookie_cats_data(remove_extreme_outliers: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load the Cookie Cats CSV and perform data hygiene and audit checks.

    Parameters
    ----------
    remove_extreme_outliers : bool, default True
        If True, filters out impossible/unrealistic outliers (e.g., the infamous
        single player with 49,854 rounds played in 14 days, averaging ~3,500 games/day).

    Returns
    -------
    df : pd.DataFrame
        The cleaned dataset.
    audit_report : dict
        A dictionary containing data health diagnostics:
        - raw_rows: int
        - cleaned_rows: int
        - null_counts: dict
        - duplicate_user_ids: int
        - outliers_removed: list of dicts
        - group_sample_sizes: dict
    """
    file_path = get_data_path()
    df_raw = pd.read_csv(file_path)

    # 1. Check for duplicates in primary identifier
    dup_users = int(df_raw["userid"].duplicated().sum())

    # 2. Check for missing values
    null_counts = df_raw.isnull().sum().to_dict()

    # 3. Detect extreme gamerounds outliers
    # A single player played 49,854 gamerounds. Over 14 days, that is ~3,561 rounds/day,
    # or ~2.5 games every minute without sleeping for 14 continuous days.
    outlier_mask = df_raw["sum_gamerounds"] > 10000
    outliers = df_raw[outlier_mask].to_dict(orient="records")

    if remove_extreme_outliers:
        df_clean = df_raw[~outlier_mask].copy()
    else:
        df_clean = df_raw.copy()

    # Ensure retention boolean types
    df_clean["retention_1"] = df_clean["retention_1"].astype(bool)
    df_clean["retention_7"] = df_clean["retention_7"].astype(bool)

    group_sizes = df_clean["version"].value_counts().to_dict()

    audit_report = {
        "raw_rows": len(df_raw),
        "cleaned_rows": len(df_clean),
        "null_counts": null_counts,
        "duplicate_user_ids": dup_users,
        "outliers_removed": outliers,
        "group_sample_sizes": group_sizes,
    }

    return df_clean, audit_report


def get_group_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute summary statistics per variant (gate_30 vs. gate_40).

    Returns
    -------
    summary : dict
        Nested dictionary containing descriptive metrics for gate_30 (control)
        and gate_40 (treatment).
    """
    summary = {}
    for version in ["gate_30", "gate_40"]:
        sub = df[df["version"] == version]
        n = len(sub)
        r1_success = int(sub["retention_1"].sum())
        r7_success = int(sub["retention_7"].sum())
        rounds = sub["sum_gamerounds"]

        summary[version] = {
            "n": n,
            "retention_1": {
                "successes": r1_success,
                "total": n,
                "rate": r1_success / n,
            },
            "retention_7": {
                "successes": r7_success,
                "total": n,
                "rate": r7_success / n,
            },
            "sum_gamerounds": {
                "count": n,
                "mean": float(rounds.mean()),
                "std": float(rounds.std(ddof=1)),
                "var": float(rounds.var(ddof=1)),
                "median": float(rounds.median()),
                "q25": float(rounds.quantile(0.25)),
                "q75": float(rounds.quantile(0.75)),
                "iqr": float(rounds.quantile(0.75) - rounds.quantile(0.25)),
            },
        }
    return summary


if __name__ == "__main__":
    df, report = load_cookie_cats_data()
    print("=== Data Audit Report ===")
    print(f"Total Rows: {report['raw_rows']}")
    print(f"Null Values: {report['null_counts']}")
    print(f"Duplicate User IDs: {report['duplicate_user_ids']}")
    print(f"Outliers Identified: {report['outliers_removed']}")
    print(f"Group Split: {report['group_sample_sizes']}")
    print("\n=== Group Descriptive Metrics ===")
    metrics = get_group_metrics(df)
    for grp, vals in metrics.items():
        print(f"\nVariant [{grp}]:")
        print(f"  N = {vals['n']}")
        print(f"  Day 1 Retention: {vals['retention_1']['rate']:.4%} ({vals['retention_1']['successes']}/{vals['retention_1']['total']})")
        print(f"  Day 7 Retention: {vals['retention_7']['rate']:.4%} ({vals['retention_7']['successes']}/{vals['retention_7']['total']})")
        print(f"  Gamerounds Mean: {vals['sum_gamerounds']['mean']:.3f}, Std: {vals['sum_gamerounds']['std']:.3f}, Median: {vals['sum_gamerounds']['median']:.1f}")
