"""Pydantic data schemas for the experimentation analysis API."""

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class ExperimentAnalysisRequest(BaseModel):
    """Input payload requesting end-to-end experiment analysis."""
    dataset_name: Optional[str] = Field("cookie_cats", description="Identifier or nickname of the experiment")
    dataset_path: Optional[str] = Field(None, description="Local path to CSV dataset. If None, resolves default data path.")
    metric_type: Literal["proportion", "continuous"] = Field("proportion", description="Type of metric: 'proportion' (binary) or 'continuous'")
    metric_column: str = Field("retention_7", description="Column name of the primary outcome metric")
    variant_column: str = Field("version", description="Column name of the variant/group indicator")
    control_value: str = Field("gate_30", description="Value identifying the control group")
    treatment_value: str = Field("gate_40", description="Value identifying the treatment group")
    covariate_column: Optional[str] = Field(None, description="Pre-experiment covariate for CUPED (if available)")
    day_column: Optional[str] = Field(None, description="Day column for time-windowed novelty checks (if available)")
    segment_column: Optional[str] = Field(None, description="Segment column for Simpson's paradox checks (if available)")
    alpha: float = Field(0.05, ge=0.001, le=0.20, description="Nominal significance level")
    power: float = Field(0.80, ge=0.50, le=0.99, description="Target statistical power for MDE")
    interim_looks_planned: int = Field(1, ge=1, le=100, description="Total interim looks planned (1 = fixed horizon)")


class PowerAnalysisOutput(BaseModel):
    """Results from retrospective power & MDE check."""
    status: str
    target_power: float
    mde_absolute: float
    mde_relative_pct: float
    observed_diff: float
    was_effect_detectable: bool
    verdict: str


class CoreTestOutput(BaseModel):
    """Statistical hypothesis test results."""
    test_type: str
    control_rate_or_mean: float
    treatment_rate_or_mean: float
    absolute_diff: float
    relative_lift_pct: float
    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    is_significant: bool


class SequentialSafetyOutput(BaseModel):
    """Pocock sequential safety check results."""
    status: str
    looks_evaluated: int
    nominal_alpha: float
    adjusted_alpha: float
    verdict: str


class CUPEDOutput(BaseModel):
    """CUPED variance reduction results."""
    status: str
    covariate_used: Optional[str] = None
    correlation_rho: Optional[float] = None
    optimal_theta: Optional[float] = None
    variance_reduction_pct: Optional[float] = None
    effective_sample_multiplier: Optional[float] = None
    adjusted_lift: Optional[float] = None
    adjusted_p_value: Optional[float] = None


class GuardrailOutput(BaseModel):
    """Diagnostics for novelty decay and Simpson's Paradox."""
    novelty_status: str
    novelty_flagged: Optional[bool] = None
    simpsons_status: str
    simpsons_flagged: Optional[bool] = None
    details: Dict[str, Any] = {}


class ExperimentReportResponse(BaseModel):
    """Complete consolidated analysis report returned by the API."""
    experiment_name: str
    metric_analyzed: str
    metric_type: str
    sample_sizes: Dict[str, int]
    power_check: PowerAnalysisOutput
    core_test: CoreTestOutput
    sequential_safety: SequentialSafetyOutput
    cuped_adjustment: CUPEDOutput
    guardrails: GuardrailOutput
    executive_recommendation: str
