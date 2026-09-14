"""Unit tests for Phase 4 Guardrails: Novelty Effects & Simpson's Paradox."""

import numpy as np
import pandas as pd
from src.novelty_check import generate_novelty_decay_data, analyze_novelty_effect
from src.simpsons_check import (
    get_kidney_stone_data,
    generate_cookie_cats_segmented_data,
    detect_simpsons_paradox,
)


def test_novelty_effect_detection():
    """Verify that novelty effect is correctly flagged when decay is present."""
    df_decay = generate_novelty_decay_data(
        num_users_per_day=800,
        num_days=14,
        initial_tau=5.0,
        decay_rate=0.35,
        random_seed=42
    )
    res = analyze_novelty_effect(df_decay)

    assert res.has_novelty_decay is True
    assert res.early_ttest.absolute_diff > res.late_ttest.absolute_diff
    assert res.decay_percentage > 0.50
    assert len(res.daily_effects) == 14


def test_novelty_effect_stable_treatment():
    """Verify that a persistent, non-decaying effect is NOT flagged as novelty."""
    # When decay_rate = 0.0, treatment effect is flat over time
    df_stable = generate_novelty_decay_data(
        num_users_per_day=800,
        num_days=14,
        initial_tau=3.0,
        decay_rate=0.0,
        random_seed=101
    )
    res = analyze_novelty_effect(df_stable)
    assert res.has_novelty_decay is False


def test_simpsons_paradox_kidney_stone_benchmark():
    """Verify Simpson's Paradox on the peer-reviewed 1986 Charig et al. dataset."""
    df_kidney = get_kidney_stone_data()
    res = detect_simpsons_paradox(
        df_kidney,
        group_col="treatment",
        outcome_col="success",
        segment_col="stone_size",
        group_a="A",
        group_b="B"
    )
    assert res.has_simpsons_paradox is True
    # In both small and large stones, Treatment A has higher success rate (lift B - A is negative)
    assert res.segment_lifts["Small"] < 0.0
    assert res.segment_lifts["Large"] < 0.0
    # In aggregate, Treatment B has higher success rate (lift B - A is positive)
    assert res.aggregate_lift > 0.0


def test_simpsons_paradox_cookie_cats_segmented():
    """Verify Simpson's Paradox on Cookie Cats device platform segmentation."""
    df_cc = generate_cookie_cats_segmented_data()
    res = detect_simpsons_paradox(
        df_cc,
        group_col="version",
        outcome_col="retention_7",
        segment_col="platform",
        group_a="gate_30",
        group_b="gate_40"
    )
    assert res.has_simpsons_paradox is True
    # gate_30 wins on both iOS and Android (gate_40 - gate_30 lift is negative)
    for seg, lift in res.segment_lifts.items():
        assert lift < 0.0
    # gate_40 appears to win in aggregate (gate_40 - gate_30 lift is positive)
    assert res.aggregate_lift > 0.0
    assert np.isclose(res.segment_lifts["iOS"], -0.03, atol=1e-3)
    assert np.isclose(res.segment_lifts["Android"], -0.03, atol=1e-3)
    assert np.isclose(res.aggregate_lift, 0.025, atol=1e-3)
