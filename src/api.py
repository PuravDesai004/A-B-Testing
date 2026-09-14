"""FastAPI microservice exposing the unified experimentation platform.

Provides HTTP REST endpoints for automated A/B test analysis, sequential safety,
CUPED variance reduction, and guardrail diagnostics.
"""

import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.schemas import ExperimentAnalysisRequest, ExperimentReportResponse
from src.pipeline import ExperimentPipeline

app = FastAPI(
    title="A/B Testing & Experimentation Engine API",
    version="1.0.0",
    description=(
        "Production-grade experimentation microservice chaining hypothesis testing, "
        "retrospective power checks, Pocock sequential corrections, CUPED variance reduction, "
        "and guardrail diagnostics (novelty decay & Simpson's paradox)."
    ),
)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
def health_check():
    """Service health and liveness probe."""
    return {
        "status": "healthy",
        "service": "ab-testing-engine",
        "version": "1.0.0",
        "supported_tests": ["two_proportion_z_test", "welch_t_test", "pocock_sequential", "cuped", "novelty_decay", "simpsons_paradox"]
    }


@app.post(
    "/experiment/analyze",
    response_model=ExperimentReportResponse,
    status_code=status.HTTP_200_OK,
    tags=["Experimentation"]
)
def analyze_experiment(request: ExperimentAnalysisRequest):
    """Execute end-to-end experiment analysis across all platform stages.

    Chains:
    1. Data hygiene & validation
    2. Statistical hypothesis test (Z-test or Welch's t-test)
    3. Retrospective power & Minimum Detectable Effect (MDE) check
    4. Pocock sequential safety adjustment (if multiple interim looks planned)
    5. CUPED variance reduction (if pre-experiment covariate provided)
    6. Guardrail checks (novelty decay and Simpson's Paradox)
    """
    try:
        report = ExperimentPipeline.run(request)
        return report
    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file could not be found: {str(fnf_err)}"
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Statistical validation error: {str(val_err)}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal experimentation pipeline error: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
