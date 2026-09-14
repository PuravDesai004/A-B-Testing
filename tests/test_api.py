"""Integration tests for FastAPI experimentation service."""

import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.synthetic_cuped_data import generate_cuped_dataset

client = TestClient(app)


def test_health_check_endpoint():
    """Verify health probe returns status 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "two_proportion_z_test" in data["supported_tests"]


def test_analyze_cookie_cats_retention_7():
    """Verify complete analysis on Cookie Cats retention_7 metric."""
    payload = {
        "dataset_name": "Cookie Cats Day 7 Retention",
        "metric_type": "proportion",
        "metric_column": "retention_7",
        "variant_column": "version",
        "control_value": "gate_30",
        "treatment_value": "gate_40",
        "alpha": 0.05,
        "interim_looks_planned": 1
    }
    response = client.post("/experiment/analyze", json=payload)
    assert response.status_code == 200
    report = response.json()

    # Verify Core Test Findings
    assert report["core_test"]["is_significant"] is True
    assert report["core_test"]["p_value"] < 0.01
    assert report["core_test"]["absolute_diff"] < 0.0  # Significant drop

    # Verify Graceful Degradation for unsupplied optional stages
    assert report["cuped_adjustment"]["status"] == "SKIPPED_NO_PRE_COVARIATE"
    assert report["guardrails"]["novelty_status"] == "SKIPPED_NO_DAY_COLUMN"
    assert report["guardrails"]["simpsons_status"] == "SKIPPED_NO_SEGMENT_COLUMN"
    assert "DO NOT SHIP" in report["executive_recommendation"]


def test_analyze_cookie_cats_continuous_gamerounds():
    """Verify continuous analysis on Cookie Cats sum_gamerounds metric."""
    payload = {
        "dataset_name": "Cookie Cats Gamerounds",
        "metric_type": "continuous",
        "metric_column": "sum_gamerounds",
        "variant_column": "version",
        "control_value": "gate_30",
        "treatment_value": "gate_40",
        "alpha": 0.05
    }
    response = client.post("/experiment/analyze", json=payload)
    assert response.status_code == 200
    report = response.json()
    assert report["core_test"]["test_type"] == "Welch's Two-Sample T-Test (Unequal Variances)"
    assert report["core_test"]["is_significant"] is False  # p ~ 0.95


def test_analyze_sequential_safety_interim_looks():
    """Verify that specifying multiple interim looks triggers Pocock adjustment."""
    payload = {
        "dataset_name": "Cookie Cats Daily Peeking",
        "metric_type": "proportion",
        "metric_column": "retention_1",
        "variant_column": "version",
        "control_value": "gate_30",
        "treatment_value": "gate_40",
        "alpha": 0.05,
        "interim_looks_planned": 14
    }
    response = client.post("/experiment/analyze", json=payload)
    assert response.status_code == 200
    report = response.json()

    seq = report["sequential_safety"]
    assert seq["status"] == "ACTIVE_ADJUSTMENT"
    assert seq["looks_evaluated"] == 14
    assert seq["adjusted_alpha"] < 0.01  # Approximately 0.0089


def test_analyze_cuped_pipeline_with_covariate(tmp_path):
    """Verify pipeline executes CUPED when pre-experiment covariate is provided."""
    # Generate temporary synthetic CUPED CSV
    temp_csv = tmp_path / "temp_cuped_data.csv"
    dataset = generate_cuped_dataset(num_users=2000, true_tau=1.50, target_rho=0.75, random_seed=42)
    dataset.df.to_csv(temp_csv, index=False)

    payload = {
        "dataset_name": "Synthetic Gaming Experiment",
        "dataset_path": str(temp_csv),
        "metric_type": "continuous",
        "metric_column": "post_rounds",
        "variant_column": "variant",
        "control_value": "control",
        "treatment_value": "treatment",
        "covariate_column": "pre_rounds",
        "alpha": 0.05
    }
    response = client.post("/experiment/analyze", json=payload)
    assert response.status_code == 200
    report = response.json()

    cuped = report["cuped_adjustment"]
    assert cuped["status"] == "APPLIED"
    assert cuped["variance_reduction_pct"] > 40.0
    assert cuped["effective_sample_multiplier"] > 1.5
