"""Unified experimentation pipeline orchestrator.

Chains together all statistical modules from Phases 1-4 into an automated execution
engine with graceful degradation when optional data columns (covariates or segments)
are not provided.
"""

import sys
from pathlib import Path
from typing import Optional
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import get_data_path
from src.proportion_test import proportion_z_test
from src.continuous_test import welch_t_test
from src.power_check import compute_proportion_mde, compute_continuous_mde
from src.sequential_correction import get_pocock_critical_value
from src.cuped import run_cuped_analysis
from src.novelty_check import analyze_novelty_effect
from src.simpsons_check import detect_simpsons_paradox
from src.schemas import (
    ExperimentAnalysisRequest,
    ExperimentReportResponse,
    PowerAnalysisOutput,
    CoreTestOutput,
    SequentialSafetyOutput,
    CUPEDOutput,
    GuardrailOutput,
)


class ExperimentPipeline:
    """Orchestrates end-to-end experiment analysis across all platform stages."""

    @classmethod
    def run(cls, request: ExperimentAnalysisRequest, df: Optional[pd.DataFrame] = None) -> ExperimentReportResponse:
        """Execute the sequential stages of the experimentation platform."""
        # 1. Ingest Data
        if df is None:
            if request.dataset_path:
                csv_path = Path(request.dataset_path)
            else:
                csv_path = get_data_path()
            df = pd.read_csv(csv_path)

        # Basic hygiene: remove extreme anomalies if gamerounds column exists.
        # The outlier (user 6390605 with 49,854 rounds) is a data quality issue
        # that should be excluded regardless of which metric is being analyzed.
        if "sum_gamerounds" in df.columns:
            df = df[df["sum_gamerounds"] < 10000].copy()

        ctrl_sub = df[df[request.variant_column] == request.control_value]
        trt_sub = df[df[request.variant_column] == request.treatment_value]

        n_ctrl = len(ctrl_sub)
        n_trt = len(trt_sub)
        sample_sizes = {request.control_value: n_ctrl, request.treatment_value: n_trt}

        # 2. Stage 1: Core Hypothesis Testing & Retrospective Power
        if request.metric_type == "proportion":
            # Binary conversion/retention metric
            succ_ctrl = int(ctrl_sub[request.metric_column].sum())
            succ_trt = int(trt_sub[request.metric_column].sum())

            core_res = proportion_z_test(
                successes_a=succ_ctrl,
                n_a=n_ctrl,
                successes_b=succ_trt,
                n_b=n_trt,
                alpha=request.alpha
            )

            pwr_res = compute_proportion_mde(
                metric_name=request.metric_column,
                n_control=n_ctrl,
                n_treatment=n_trt,
                control_rate=core_res.prop_a,
                treatment_rate=core_res.prop_b,
                alpha=request.alpha,
                power=request.power
            )

            core_output = CoreTestOutput(
                test_type="Two-Proportion Pooled Z-Test",
                control_rate_or_mean=core_res.prop_a,
                treatment_rate_or_mean=core_res.prop_b,
                absolute_diff=core_res.absolute_diff,
                relative_lift_pct=core_res.relative_lift * 100.0,
                statistic=core_res.z_stat,
                p_value=core_res.p_value,
                ci_lower=core_res.ci_lower,
                ci_upper=core_res.ci_upper,
                is_significant=core_res.p_value < request.alpha
            )

            power_output = PowerAnalysisOutput(
                status="COMPLETED",
                target_power=request.power,
                mde_absolute=pwr_res.mde_abs_percentage_points,
                mde_relative_pct=pwr_res.mde_relative_percentage * 100.0,
                observed_diff=pwr_res.observed_abs_diff,
                was_effect_detectable=pwr_res.was_effect_detectable,
                verdict=pwr_res.verdict
            )

        else:
            # Continuous metric (e.g. gamerounds)
            vals_ctrl = ctrl_sub[request.metric_column]
            vals_trt = trt_sub[request.metric_column]

            core_res = welch_t_test(vals_ctrl, vals_trt, alpha=request.alpha)

            pwr_res = compute_continuous_mde(
                metric_name=request.metric_column,
                n_control=n_ctrl,
                n_treatment=n_trt,
                control_mean=core_res.mean_a,
                treatment_mean=core_res.mean_b,
                control_std=core_res.std_a,
                treatment_std=core_res.std_b,
                alpha=request.alpha,
                power=request.power
            )

            core_output = CoreTestOutput(
                test_type="Welch's Two-Sample T-Test (Unequal Variances)",
                control_rate_or_mean=core_res.mean_a,
                treatment_rate_or_mean=core_res.mean_b,
                absolute_diff=core_res.absolute_diff,
                relative_lift_pct=core_res.relative_lift * 100.0,
                statistic=core_res.t_stat,
                p_value=core_res.p_value,
                ci_lower=core_res.ci_lower,
                ci_upper=core_res.ci_upper,
                is_significant=core_res.p_value < request.alpha
            )

            power_output = PowerAnalysisOutput(
                status="COMPLETED",
                target_power=request.power,
                mde_absolute=pwr_res.mde_abs_rounds,
                mde_relative_pct=pwr_res.mde_relative_percentage * 100.0,
                observed_diff=pwr_res.observed_abs_diff,
                was_effect_detectable=pwr_res.was_effect_detectable,
                verdict=pwr_res.verdict
            )

        # 3. Stage 2: Sequential Safety Check (Pocock)
        if request.interim_looks_planned > 1:
            z_crit, alpha_look = get_pocock_critical_value(request.interim_looks_planned, alpha=request.alpha)
            seq_verdict = (
                f"SAFETY GUARD APPLIED: For {request.interim_looks_planned} planned interim looks, "
                f"per-look alpha is adjusted to {alpha_look:.4f} (z* = {z_crit:.3f}) to guarantee overall "
                f"Type I error does not exceed {request.alpha:.1%}."
            )
            seq_output = SequentialSafetyOutput(
                status="ACTIVE_ADJUSTMENT",
                looks_evaluated=request.interim_looks_planned,
                nominal_alpha=request.alpha,
                adjusted_alpha=alpha_look,
                verdict=seq_verdict
            )
        else:
            seq_output = SequentialSafetyOutput(
                status="FIXED_HORIZON",
                looks_evaluated=1,
                nominal_alpha=request.alpha,
                adjusted_alpha=request.alpha,
                verdict="Single fixed-horizon evaluation: standard alpha=0.05 applies."
            )

        # 4. Stage 3: CUPED Variance Reduction
        if request.covariate_column and request.covariate_column in df.columns:
            df, cuped_metrics = run_cuped_analysis(
                df,
                covariate_col=request.covariate_column,
                outcome_col=request.metric_column,
                variant_col=request.variant_column,
                control_label=request.control_value,
                treatment_label=request.treatment_value,
                alpha=request.alpha
            )
            cuped_output = CUPEDOutput(
                status="APPLIED",
                covariate_used=request.covariate_column,
                correlation_rho=cuped_metrics.correlation_rho,
                optimal_theta=cuped_metrics.theta,
                variance_reduction_pct=cuped_metrics.empirical_var_reduction * 100.0,
                effective_sample_multiplier=cuped_metrics.effective_sample_multiplier,
                adjusted_lift=cuped_metrics.cuped_ttest.absolute_diff,
                adjusted_p_value=cuped_metrics.cuped_ttest.p_value,
            )
        else:
            cuped_output = CUPEDOutput(
                status="SKIPPED_NO_PRE_COVARIATE",
                variance_reduction_pct=0.0,
                effective_sample_multiplier=1.0,
            )

        # 5. Stage 4: Guardrail Diagnostics
        guardrail_details = {}
        novelty_flag = None
        simpsons_flag = None

        # Check Novelty (evaluates CUPED-adjusted metric if CUPED was applied)
        if request.day_column and request.day_column in df.columns:
            metric_for_novelty = (
                "cuped_outcome"
                if (cuped_output.status == "APPLIED" and "cuped_outcome" in df.columns)
                else request.metric_column
            )
            nov_res = analyze_novelty_effect(
                df,
                day_col=request.day_column,
                variant_col=request.variant_column,
                metric_col=metric_for_novelty,
                alpha=request.alpha
            )
            novelty_flag = nov_res.has_novelty_decay
            guardrail_details["novelty_decay_pct"] = nov_res.decay_percentage * 100.0
            guardrail_details["novelty_verdict"] = nov_res.verdict
            guardrail_details["novelty_metric_evaluated"] = metric_for_novelty
            nov_status = "EVALUATED"
        else:
            nov_status = "SKIPPED_NO_DAY_COLUMN"

        # Check Simpson's Paradox
        if request.segment_column and request.segment_column in df.columns:
            simp_res = detect_simpsons_paradox(
                df,
                group_col=request.variant_column,
                outcome_col=request.metric_column,
                segment_col=request.segment_column,
                group_a=request.control_value,
                group_b=request.treatment_value,
            )
            simpsons_flag = simp_res.has_simpsons_paradox
            guardrail_details["simpsons_reversal"] = simp_res.reversal_type
            guardrail_details["simpsons_explanation"] = simp_res.explanation
            simp_status = "EVALUATED"
        else:
            simp_status = "SKIPPED_NO_SEGMENT_COLUMN"

        guardrail_output = GuardrailOutput(
            novelty_status=nov_status,
            novelty_flagged=novelty_flag,
            simpsons_status=simp_status,
            simpsons_flagged=simpsons_flag,
            details=guardrail_details
        )

        # 6. Synthesize Executive Recommendation
        rec_parts = []

        # Effective alpha threshold is adjusted if sequential testing looks are planned
        effective_alpha = (
            seq_output.adjusted_alpha
            if request.interim_looks_planned > 1
            else request.alpha
        )

        # Use CUPED-adjusted results as primary decision if applied, else raw core test
        if cuped_output.status == "APPLIED" and cuped_output.adjusted_p_value is not None:
            eval_p = cuped_output.adjusted_p_value
            eval_diff = cuped_output.adjusted_lift
            is_significant = eval_p < effective_alpha
        else:
            eval_p = core_output.p_value
            eval_diff = core_output.absolute_diff
            is_significant = eval_p < effective_alpha

        if is_significant:
            if eval_diff > 0:
                rec_parts.append(
                    f"STATISTICALLY SIGNIFICANT WIN: Treatment produced a statistically significant lift "
                    f"(+{eval_diff:.3f}, p = {eval_p:.4e})."
                )
            else:
                rec_parts.append(
                    f"STATISTICALLY SIGNIFICANT LOSS: Treatment produced a statistically significant decline "
                    f"({eval_diff:+.3f}, p = {eval_p:.4e}). DO NOT SHIP."
                )
        else:
            rec_parts.append(
                f"INCONCLUSIVE / NO SIGNIFICANT EFFECT: P-value ({eval_p:.4f}) does not meet "
                f"significance threshold (alpha={effective_alpha:.4f})."
            )

        if novelty_flag is True:
            rec_parts.append("WARNING: Novelty effect detected! Early gains decayed significantly by the end of the test.")
        if simpsons_flag is True:
            rec_parts.append("CRITICAL WARNING: Simpson's Paradox detected! Subgroup effects contradict aggregate numbers.")

        rec = " ".join(rec_parts)

        return ExperimentReportResponse(
            experiment_name=request.dataset_name or "Experiment",
            metric_analyzed=request.metric_column,
            metric_type=request.metric_type,
            sample_sizes=sample_sizes,
            power_check=power_output,
            core_test=core_output,
            sequential_safety=seq_output,
            cuped_adjustment=cuped_output,
            guardrails=guardrail_output,
            executive_recommendation=rec
        )
