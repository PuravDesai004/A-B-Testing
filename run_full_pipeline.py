"""Master runner for Phase 6: End-to-end pipeline execution across all phases.

Executes the complete experimentation pipeline across Cookie Cats and synthetic
experimentation data, and outputs unified consolidated JSON and terminal reports.
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import ExperimentPipeline
from src.schemas import ExperimentAnalysisRequest


def run_complete_suite():
    print("=" * 80)
    print("PHASE 6: END-TO-END EXPERIMENTATION PIPELINE EXECUTION")
    print("=" * 80)

    # 1. Run Cookie Cats Retention 7 (Proportion Test + Power Check)
    req_r7 = ExperimentAnalysisRequest(
        dataset_name="Cookie Cats Day 7 Retention",
        metric_type="proportion",
        metric_column="retention_7",
        variant_column="version",
        control_value="gate_30",
        treatment_value="gate_40",
        alpha=0.05,
        power=0.80,
        interim_looks_planned=1
    )
    rep_r7 = ExperimentPipeline.run(req_r7)
    print("\n--- 1. Real Cookie Cats Experiment (retention_7) ---")
    print(f"Test Type:       {rep_r7.core_test.test_type}")
    print(f"Sample Sizes:    {rep_r7.sample_sizes}")
    print(f"Control Rate:    {rep_r7.core_test.control_rate_or_mean:.2%}")
    print(f"Treatment Rate:  {rep_r7.core_test.treatment_rate_or_mean:.2%}")
    print(f"Absolute Lift:   {rep_r7.core_test.absolute_diff:+.4%} points ({rep_r7.core_test.relative_lift_pct:+.2f}%)")
    print(f"P-Value:         {rep_r7.core_test.p_value:.4e} (Significant: {rep_r7.core_test.is_significant})")
    print(f"95% CI:          [{rep_r7.core_test.ci_lower:+.4%}, {rep_r7.core_test.ci_upper:+.4%}]")
    print(f"Recommendation:  {rep_r7.executive_recommendation}")

    # 2. Run Cookie Cats Gamerounds (Continuous Test + Power Check)
    req_gr = ExperimentAnalysisRequest(
        dataset_name="Cookie Cats Gamerounds",
        metric_type="continuous",
        metric_column="sum_gamerounds",
        variant_column="version",
        control_value="gate_30",
        treatment_value="gate_40",
        alpha=0.05,
        power=0.80,
        interim_looks_planned=1
    )
    rep_gr = ExperimentPipeline.run(req_gr)
    print("\n--- 2. Real Cookie Cats Experiment (sum_gamerounds) ---")
    print(f"Test Type:       {rep_gr.core_test.test_type}")
    print(f"Control Mean:    {rep_gr.core_test.control_rate_or_mean:.3f} rounds")
    print(f"Treatment Mean:  {rep_gr.core_test.treatment_rate_or_mean:.3f} rounds")
    print(f"Absolute Lift:   {rep_gr.core_test.absolute_diff:+.3f} rounds ({rep_gr.core_test.relative_lift_pct:+.2f}%)")
    print(f"P-Value:         {rep_gr.core_test.p_value:.4f} (Significant: {rep_gr.core_test.is_significant})")
    print(f"80% Power MDE:   +-{rep_gr.power_check.mde_absolute:.3f} rounds")
    print(f"Recommendation:  {rep_gr.executive_recommendation}")

    # 3. Run Pipeline with CUPED Covariate on Synthetic Telemetry
    from src.synthetic_cuped_data import generate_cuped_dataset
    cuped_csv = ROOT_DIR / "data" / "synthetic_cuped_stream.csv"
    syn_ds = generate_cuped_dataset(num_users=5000, true_tau=1.50, target_rho=0.75, random_seed=42)
    syn_ds.df.to_csv(cuped_csv, index=False)

    req_cuped = ExperimentAnalysisRequest(
        dataset_name="Synthetic Mobile Game Feature Test",
        dataset_path=str(cuped_csv),
        metric_type="continuous",
        metric_column="post_rounds",
        variant_column="variant",
        control_value="control",
        treatment_value="treatment",
        covariate_column="pre_rounds",
        alpha=0.05,
        power=0.80,
        interim_looks_planned=1
    )
    rep_cuped = ExperimentPipeline.run(req_cuped)
    print("\n--- 3. CUPED Variance-Reduced Telemetry Pipeline ---")
    print(f"Covariate Used:  {rep_cuped.cuped_adjustment.covariate_used} (rho = {rep_cuped.cuped_adjustment.correlation_rho:.3f})")
    print(f"Var Reduction:   -{rep_cuped.cuped_adjustment.variance_reduction_pct:.1f}%")
    print(f"Sample Bonus:    {rep_cuped.cuped_adjustment.effective_sample_multiplier:.2f}x effective users")
    print(f"Raw P-Value:     {rep_cuped.core_test.p_value:.4e} -> CUPED P-Value: {rep_cuped.cuped_adjustment.adjusted_p_value:.4e}")
    print(f"Recommendation:  {rep_cuped.executive_recommendation}")

    # Save consolidated output to reports/end_to_end_output.json
    out_path = ROOT_DIR / "reports" / "end_to_end_output.json"
    consolidated = {
        "cookie_cats_retention_7": rep_r7.model_dump(),
        "cookie_cats_gamerounds": rep_gr.model_dump(),
        "synthetic_cuped_experiment": rep_cuped.model_dump(),
    }
    out_path.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")
    print(f"\nSaved consolidated JSON report to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_complete_suite()
