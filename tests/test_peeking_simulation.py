"""Unit tests for Phase 2 peeking simulation and sequential corrections."""

import pytest
import numpy as np
from src.aa_simulator import generate_aa_experiments
from src.peeking_simulation import run_peeking_simulation
from src.sequential_correction import (
    get_pocock_critical_value,
    get_obrien_fleming_boundaries,
    evaluate_pocock_correction,
)


def test_aa_generator_structure_and_bounds():
    """Verify synthetic A/A generator generates valid binomial streams."""
    sim = generate_aa_experiments(
        num_simulations=100,
        num_days=7,
        daily_users_per_group=500,
        true_conversion_rate=0.10,
        random_seed=123
    )
    assert sim.successes_a.shape == (100, 7)
    assert sim.successes_b.shape == (100, 7)
    assert np.all(sim.successes_a >= 0)
    assert np.all(sim.successes_b >= 0)
    assert np.all(sim.successes_a <= 500)
    assert np.all(sim.successes_b <= 500)
    # Check cumulative sums monotonically increase
    assert np.all(np.diff(sim.cum_successes_a, axis=1) >= 0)
    assert np.all(np.diff(sim.cum_successes_b, axis=1) >= 0)


def test_naive_peeking_inflates_type_1_error():
    """Verify that daily peeking inflates false positives far above 5%."""
    # 5,000 simulations for fast, statistically stable testing
    res = run_peeking_simulation(
        num_simulations=5000,
        num_days=14,
        daily_users_per_group=500,
        alpha=0.05,
        random_seed=42
    )
    # Fixed horizon should be approximately 5%
    assert 0.040 <= res.fixed_horizon_fpr <= 0.060, f"Unexpected fixed FPR: {res.fixed_horizon_fpr}"

    # Naive daily peeking must be heavily inflated (expected ~20% - 25%)
    assert res.naive_peeking_fpr > 0.18, f"Peeking did not inflate FPR: {res.naive_peeking_fpr}"

    # FPR must grow monotonically with number of looks
    assert np.all(np.diff(res.fpr_by_day) >= 0)


def test_pocock_correction_restores_error_rate():
    """Verify Pocock correction controls overall alpha near 5%."""
    res = run_peeking_simulation(
        num_simulations=5000,
        num_days=14,
        daily_users_per_group=500,
        alpha=0.05,
        random_seed=42
    )

    # 1. Two-look weekly check (Day 7 and Day 14)
    pocock_2 = evaluate_pocock_correction(res, look_days=[7, 14])
    assert pocock_2.naive_fpr > 0.075
    assert 0.040 <= pocock_2.corrected_fpr <= 0.060

    # 2. Daily 14-look check
    pocock_14 = evaluate_pocock_correction(res, look_days=list(range(1, 15)))
    assert 0.040 <= pocock_14.corrected_fpr <= 0.060


def test_boundary_helpers():
    """Check boundary calculations for standard looks."""
    z2, a2 = get_pocock_critical_value(2, alpha=0.05)
    assert np.isclose(z2, 2.178)
    assert np.isclose(a2, 0.0294, atol=1e-3)

    z_obf, a_obf = get_obrien_fleming_boundaries(14, alpha=0.05)
    assert len(z_obf) == 14
    assert z_obf[0] > z_obf[-1]  # Strict early, lenient late
    assert np.isclose(z_obf[-1], 1.960, atol=1e-3)
