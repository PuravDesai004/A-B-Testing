"""Regression coverage for analysis-population and statistical accuracy bugs."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.power import TTestIndPower

from src.pipeline import ExperimentPipeline
from src.schemas import ExperimentAnalysisRequest
from src.cuped import run_cuped_analysis
from src.novelty_check import analyze_novelty_effect, generate_novelty_decay_data
from src.power_check import compute_continuous_mde
from src.peeking_simulation import run_peeking_simulation
from src.sequential_correction import (
    get_obrien_fleming_boundaries, evaluate_obrien_fleming_correction,
)
from src.synthetic_cuped_data import generate_cuped_dataset


def request(**kwargs):
    return ExperimentAnalysisRequest(
        variant_column="variant", control_value="A", treatment_value="B",
        metric_column="outcome", **kwargs)


def test_missing_binary_outcomes_are_not_failures():
    df = pd.DataFrame({"variant": ["A"] * 4 + ["B"] * 4,
                       "outcome": [1, 0, np.nan, np.nan, 1, 0, 1, 0]})
    report = ExperimentPipeline.run(request(), df)
    assert report.sample_sizes == {"A": 2, "B": 4}
    assert report.core_test.control_rate_or_mean == 0.5
    assert report.core_test.absolute_diff == 0


@pytest.mark.parametrize("bad", [0.5, 2, -1, np.inf])
def test_invalid_binary_outcomes_are_rejected(bad):
    df = pd.DataFrame({"variant": ["A", "A", "B", "B"],
                       "outcome": [bad, 0, 1, 0]})
    with pytest.raises(ValueError):
        ExperimentPipeline.run(request(), df)


def test_unrelated_rounds_do_not_change_retention_population():
    df = pd.DataFrame({"variant": ["A"] * 4 + ["B"] * 4,
                       "outcome": [1, 1, 0, 0, 1, 0, 0, 0]})
    expected = ExperimentPipeline.run(request(), df)
    df["sum_gamerounds"] = [10000, 50000, np.nan, 1, 1, 1, 1, 1]
    actual = ExperimentPipeline.run(request(), df)
    assert actual == expected


def test_missing_continuous_outcomes_use_same_power_population():
    df = pd.DataFrame({"variant": ["A"] * 4 + ["B"] * 4,
                       "outcome": [1, 2, 4, np.nan, 2, 4, 5, 6]})
    actual = ExperimentPipeline.run(request(metric_type="continuous"), df)
    expected = ExperimentPipeline.run(request(metric_type="continuous"), df.dropna())
    assert actual == expected


def test_gamerounds_outliers_are_retained_when_analyzing_rounds():
    df = pd.DataFrame({"variant": ["A"] * 3 + ["B"] * 3,
                       "sum_gamerounds": [1, 2, 50000, 2, 3, 4]})
    req = request(metric_type="continuous").model_copy(update={"metric_column": "sum_gamerounds"})
    report = ExperimentPipeline.run(req, df)
    assert report.sample_sizes == {"A": 3, "B": 3}
    assert report.core_test.control_rate_or_mean == pytest.approx(50003 / 3)


def test_cuped_ignores_unselected_variants():
    df = generate_cuped_dataset(num_users=2000, random_seed=42).df
    _, expected = run_cuped_analysis(df)
    extra = pd.DataFrame({"variant": ["unrelated"] * 10,
                          "pre_rounds": np.arange(10) * 1000,
                          "post_rounds": -np.arange(10) * 10000})
    adjusted, actual = run_cuped_analysis(pd.concat([df, extra], ignore_index=True))
    assert actual == expected
    assert set(adjusted.variant) == {"control", "treatment"}


def test_novelty_custom_labels_through_pipeline():
    df = generate_novelty_decay_data()
    expected = analyze_novelty_effect(df)
    df = df.rename(columns={"rounds": "outcome"})
    df["variant"] = df.variant.map({"control": "A", "treatment": "B"})
    report = ExperimentPipeline.run(request(metric_type="continuous", day_column="day"), df)
    assert report.guardrails.novelty_flagged == expected.has_novelty_decay
    assert report.guardrails.novelty_flagged is True


def test_equal_lifts_with_different_precision_are_not_decay():
    rows = []
    # Symmetric samples give exactly the same +2 lift in each window, but
    # the late window has very little precision.
    for day, offsets in [(1, np.tile([-1., 1.], 100)), (12, np.array([-100., 100.]))]:
        for variant, lift in [("control", 0), ("treatment", 2)]:
            rows.extend({"day": day, "variant": variant, "rounds": 200 + lift + value}
                        for value in offsets)
    result = analyze_novelty_effect(pd.DataFrame(rows))
    assert result.early_ttest.p_value < 0.05
    assert result.late_ttest.p_value > 0.05
    assert result.decay_percentage == 0
    assert result.has_novelty_decay is False


def test_negative_effect_recovery_is_not_positive_novelty():
    df = generate_novelty_decay_data(initial_tau=-5)
    result = analyze_novelty_effect(df)
    assert result.early_ttest.absolute_diff < 0
    assert result.has_novelty_decay is False


def test_novelty_rejects_overlapping_windows():
    with pytest.raises(ValueError, match="non-overlapping"):
        analyze_novelty_effect(generate_novelty_decay_data(), late_window=(3, 5))


def test_welch_mde_matches_target_power_with_unequal_variances():
    result = compute_continuous_mde("metric", 100, 1000, 20, 21, 10, 1)
    # Independently reconstruct the reference noncentral-t power at the MDE.
    se = np.sqrt(10 ** 2 / 100 + 1 / 1000)
    df = (1 + .001) ** 2 / (1 / 99 + .001 ** 2 / 999)
    cutoff = stats.t.isf(.025, df)
    ncp = result.mde_abs_rounds / se
    achieved = stats.nct.sf(cutoff, df, ncp) + stats.nct.cdf(-cutoff, df, ncp)
    assert achieved == pytest.approx(.80, abs=1e-8)
    assert result.mde_abs_rounds > 2.8


def test_welch_mde_matches_standard_t_for_balanced_equal_variance():
    result = compute_continuous_mde("metric", 100, 100, 20, 21, 5, 5)
    reference = TTestIndPower().solve_power(nobs1=100, alpha=.05, power=.8) * 5
    assert result.mde_abs_rounds == pytest.approx(reference, rel=1e-5)


@pytest.mark.parametrize("days", [list(range(1, 15)), [1, 3, 14]])
def test_obf_controls_false_positives_for_daily_and_irregular_looks(days):
    simulation = run_peeking_simulation(num_simulations=50000, random_seed=123)
    result = evaluate_obrien_fleming_correction(simulation, look_days=days)
    assert .046 < result.corrected_fpr < .054


def test_obf_single_look_is_fixed_horizon():
    z, alpha = get_obrien_fleming_boundaries(1, alpha=.01)
    assert z[0] == pytest.approx(stats.norm.isf(.005))
    assert alpha[0] == pytest.approx(.01)


@pytest.mark.parametrize("fractions", [[.5, .5], [.8, .2], [0, 1], [.5, np.nan]])
def test_obf_rejects_invalid_information_schedule(fractions):
    with pytest.raises(ValueError):
        get_obrien_fleming_boundaries(2, information_fractions=fractions)
