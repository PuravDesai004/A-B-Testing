# Production A/B Testing & Experimentation Engine

An end-to-end, production-grade statistical experimentation platform built from first principles in Python, evaluated on the real-world **Cookie Cats** mobile game dataset (~90,189 players), synthetic multi-day A/A streams, and observational gaming battle telemetry.

Developed to demonstrate rigorous experimentation engineering: hypothesis testing, sequential safety, variance reduction, guardrail diagnostics, and microservice integration.

---

## Table of Contents
1. [Executive Summary & Core Numbers](#1-executive-summary--core-numbers)
2. [Platform Architecture & Directory Structure](#2-platform-architecture--directory-structure)
3. [Component Theory & First-Principles Engineering](#3-component-theory--first-principles-engineering)
4. [Honest Limitations & Edge Cases](#4-honest-limitations--edge-cases)
5. [Interview Preparation Guide (30-Sec Pitch & Deep-Dive)](#5-interview-preparation-guide)
6. [How to Run & Verify](#6-how-to-run--verify)

---

## 1. Executive Summary & Core Numbers

| Phase | Methodology | Dataset Tested | Key Finding / Empirical Number | Business & Engineering Impact |
|---|---|---|---|---|
| **Phase 1** | Two-Proportion Z-Test & Welch's T-Test | Cookie Cats (90,189 players) | **-4.30% relative drop in Day 7 retention** ($p = 0.0016$); game rounds flat ($p = 0.95$) | **DO NOT SHIP Level 40 Gate**: Preserves long-term player retention and LTV |
| **Phase 2** | Peeking Fallacy & Pocock Sequential Correction | 10,000 Synthetic A/A Tests | Daily peeking inflates False Positive Rate from **5.08% $\to$ 22.01%** (4.33x error inflation); Pocock restores to **4.84%** | Prevents shipping false positive features under repeated dashboard looks |
| **Phase 3** | CUPED Variance Reduction | 10,000 Pre/Post Telemetry Streams | Pre/post correlation $\rho = 0.746 \implies$ **55.81% variance reduction**, 95% CI width -33.5% | **2.26x Effective Sample Size**: Equivalent to +125.6% more users for free |
| **Phase 4** | Guardrails: Novelty Decay & Simpson's Paradox | Synthetic Time-Series & Cookie Cats Segmented ($N = 4,500$) | Novelty decay flagged (**73.6% lift attenuation**); Simpson's reversal caught (+3.0% on iOS & Android, but -2.5% in aggregate) | Protects against transient hype features and confounded subgroup rollouts |
| **Phase 5** | Production Integration | FastAPI Microservice (`POST /experiment/analyze`) | Automated pipeline with strict stage ordering and graceful fallback | Production-ready HTTP API returning structured JSON reports in < 50ms |
| **Phase 6** | End-to-End Execution | Cookie Cats & Telemetry Streams | Consolidated execution of all 5 stages in one unified response | Enterprise-grade experimentation portfolio ready for interview defense |

---

## 2. Platform Architecture & Directory Structure

```
A-B Testing Engine/
├── data/
│   ├── cookie_cats.csv                     # 90,189 real Cookie Cats records (Phase 1)
│   ├── cookie_cats_segmented.csv           # Segmented Cookie Cats records for Simpson's check (Phase 4)
│   └── synthetic_cuped_stream.csv          # Pre/post correlated gaming telemetry (Phase 3)
├── notebooks/
│   └── 01_explore_cookie_cats.ipynb        # Exploratory Data Analysis & outlier audits
├── src/
│   ├── __init__.py
│   ├── data_loader.py                      # Data ingestion & extreme outlier detection
│   ├── proportion_test.py                  # Two-proportion pooled Z-test (first principles)
│   ├── continuous_test.py                  # Welch's two-sample t-test (first principles)
│   ├── power_check.py                      # Retrospective power & MDE analysis
│   ├── aa_simulator.py                     # Vectorized multi-day A/A experiment generator
│   ├── peeking_simulation.py               # Naive daily peeking & Type I error measurement
│   ├── sequential_correction.py            # Pocock boundaries & O'Brien-Fleming alpha spending
│   ├── synthetic_cuped_data.py             # Correlated pre/post experiment generator
│   ├── cuped.py                            # Covariance, optimal theta, and CUPED adjustment
│   ├── novelty_check.py                    # Time-windowed novelty decay analysis
│   ├── simpsons_check.py                   # Subgroup confounding & Simpson's Paradox detector
│   ├── schemas.py                          # Pydantic data contracts for request/response
│   ├── pipeline.py                         # Unified orchestrator chaining all platform stages
│   ├── api.py                              # FastAPI microservice (POST /experiment/analyze)
│   ├── visualize.py                        # Phase 1 plots (retention & rounds distribution)
│   ├── visualize_peeking.py                # Phase 2 plots (FPR growth curve & p-value paths)
│   ├── visualize_cuped.py                  # Phase 3 plots (KDE variance narrowing & CI contraction)
│   └── visualize_guardrails.py             # Phase 4 plots (novelty decay & Simpson's reversal)
├── tests/
│   ├── test_proportion_test.py             # Asserts z-test matches statsmodels (< 1e-7)
│   ├── test_continuous_test.py             # Asserts Welch t-test matches scipy (< 1e-7)
│   ├── test_peeking_simulation.py          # Asserts FPR inflation and Pocock correction
│   ├── test_cuped.py                       # Asserts theta matches OLS and variance drops by 1 - rho^2
│   ├── test_guardrails.py                  # Asserts novelty decay detection and Simpson's reversal
│   └── test_api.py                         # FastAPI TestClient endpoint integration tests
├── reports/
│   ├── phase1_results.md                   # Cookie Cats experiment analysis report
│   ├── phase2_results.md                   # Peeking simulation & sequential testing report
│   ├── phase3_results.md                   # CUPED variance reduction teaching guide
│   ├── phase4_results.md                   # Guardrail diagnostics teaching guide
│   ├── end_to_end_output.json              # Consolidated JSON output from full pipeline run
│   └── figures/
│       ├── retention_comparison.png        # Bar chart comparing Day 1 & Day 7 retention
│       ├── gamerounds_distribution.png     # Boxplot of game rounds by gate version
│       ├── false_positive_rate_by_peeks.png# Cumulative FPR curve vs. number of looks
│       ├── pvalue_trajectories.png         # Random walk p-value paths under H0
│       ├── cuped_before_after.png          # KDE variance narrowing & CI contraction
│       ├── novelty_effect_decay.png        # Trajectory of daily treatment lift decay
│       └── simpsons_paradox_segments.png   # Grouped bar chart illustrating Simpson's reversal
├── run_engine.py                           # Master script running Phase 1
├── run_phase2.py                           # Master script running Phase 2
├── run_phase3.py                           # Master script running Phase 3
├── run_phase4.py                           # Master script running Phase 4
├── run_full_pipeline.py                    # Master CLI executing end-to-end pipeline
└── README.md
```

---

## 3. Component Theory & First-Principles Engineering

### A. Phase 1: Core Hypothesis Testing & Retrospective Power
- **Two-Proportion Z-Test**: Uses pooled proportion $p_{\text{pool}} = \frac{x_1 + x_2}{n_1 + n_2}$ in the standard error under $H_0$, while unpooling for the 95% Wald Confidence Interval. Validated against `statsmodels.stats.proportion.proportions_ztest` ($z = 3.1574, p = 0.00159$ on Day 7 retention).
- **Welch's Two-Sample T-Test**: Does not assume homoscedasticity ($\sigma_1^2 = \sigma_2^2$). In Cookie Cats, raw data contains a 6.18x variance ratio due to an extreme outlier (`userid = 6390605` with 49,854 rounds). Welch's t-test with Welch–Satterthwaite degrees of freedom ($\nu = 90,183.3$) prevents Type I error distortion and matches `scipy.stats.ttest_ind(..., equal_var=False)`.
- **Retrospective MDE**: Proves that while Day 1 retention was non-significant ($p = 0.074$), the test was underpowered for the observed $-0.59\%$ drop (MDE was $\pm 0.93\%$ points).

### B. Phase 2: The Peeking Problem & Sequential Testing
- **Mechanism**: Checking dashboards daily and stopping at the first $p < 0.05$ creates a repeated testing hazard analogous to rolling a die 14 times. By the Reflection Principle of Brownian Motion, the running supremum crosses critical boundaries at more than double the endpoint rate.
- **Empirical Measurement**: In 10,000 zero-effect A/A experiments over 14 days, naive daily peeking inflated the False Positive Rate from **5.08% to 22.01%** (a 4.33x error inflation).
- **Pocock Correction**: Calculates a stricter per-look critical threshold ($z^* = 2.615, \alpha^* = 0.0089$ for 14 looks) such that $P(\exists k: |Z_k| \ge z^*) = 0.05$, restoring empirical error to **4.84%**.

### C. Phase 3: CUPED Variance Reduction
- **Mechanism**: Subtracts away predictable baseline player tendencies using pre-experiment covariate $X$:
  $$Y_{\text{CUPED}} = Y - \theta (X - \bar{X}), \quad \theta^* = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}$$
- **Unbiasedness**: Because $X$ is measured before treatment assignment, $E[X - \bar{X}] = 0$ in both variants, ensuring $E[Y_{\text{CUPED}}] = E[Y]$.
- **Results**: With pre/post correlation $\rho = 0.7462$, empirical variance dropped by **55.81%** (matching theoretical $1 - \rho^2 = 55.68\%$), multiplying effective sample size by **2.26x**.

### D. Phase 4: Guardrail Diagnostics
- **Novelty Effect Check**: Compares early window (Days 1–3) vs. late window (Days 12–14). Flags features where initial curiosity creates a temporary spike that subsequently decays by $> 60\%$.
- **Simpson's Paradox Check**: Evaluates retention rates across subgroups. In our Cookie Cats platform analysis, Gate 30 beats Gate 40 on iOS (24% vs 21%) and Android (14% vs 11%), but Gate 40 appears to win in aggregate (19.0% vs 16.5%) because Gate 40 was allocated 80% to iOS players who have higher baseline retention.

### E. Phase 5: Production Integration
- **Microservice**: Built with FastAPI and Pydantic. Exposes `POST /experiment/analyze`.
- **Pipeline Sequencing**:
  1. Data Ingestion & Audit
  2. Retrospective Power Check
  3. Core Hypothesis Test
  4. Sequential Safety (Pocock)
  5. CUPED Adjustment (if covariate present)
  6. Guardrail Diagnostics (if day/segment present)
  7. Executive Business Recommendation

---

## 4. Honest Limitations & Edge Cases

Documenting boundaries and limitations is a mark of engineering maturity:

1. **The Cookie Cats Pre-Experiment Constraint (CUPED)**:
   - *Limitation*: Cookie Cats only tracks players post-install; there is zero pre-experiment history.
   - *Design Choice*: We refused to fabricate an artificial split of post-treatment rounds (which introduces conditioning bias). Instead, we demonstrated CUPED on a transparent synthetic stream with verified ground truth.
2. **O'Brien-Fleming Discrete Calibration Gap**:
   - *Limitation*: The continuous O'Brien-Fleming boundary approximation ($z_k = z_{\alpha/2}/\sqrt{k/K}$) yielded a cumulative FPR of 7.34% across 14 daily looks instead of strictly 5.0%.
   - *Reason*: Exact 5% alpha preservation in discrete sequential looks requires recursive numerical integration (Lan-DeMets alpha spending recursions), whereas Pocock's constant boundary calibrated empirically to 4.84%.
3. **Observational vs. Randomized Data in Simpson's Paradox**:
   - *Limitation*: Simpson's Paradox cannot occur in a properly randomized 50/50 A/B test because random assignment balances confounders across variants.
   - *Context*: We demonstrated Simpson's Paradox on an unstratified Cookie Cats platform rollout (`iOS` vs `Android`) to show why product teams must audit segment balances before declaring winners.

---

## 5. Interview Preparation Guide

### The 30-Second Elevator Pitch
> *"I built an end-to-end A/B testing and experimentation platform in Python designed to address the most common failure modes in real-world product experimentation.*  
> *Using the Cookie Cats mobile game dataset (90,000 players), I evaluated level gate placement using pooled Z-tests and Welch's t-test, discovering a statistically significant 4.3% relative drop in Day 7 retention that advised against shipping.*  
> *Beyond standard testing, I built a simulation engine proving that daily peeking inflates false positive rates from 5% to 22%, implemented Pocock sequential corrections to restore error control, built a CUPED variance-reduction engine that effectively doubled sample size for free, and integrated guardrails for novelty decay and Simpson's Paradox behind a production-grade FastAPI microservice."*

---

### The Deep-Dive Technical Walkthrough

#### 1. Why pooled variance in the Z-test, but unpooled in the confidence interval?
- **Answer**: In hypothesis testing, we assume the null hypothesis ($H_0: p_1 = p_2 = p$) is true until proven false. Under $H_0$, both groups share the exact same underlying variance $\sigma^2 = p(1-p)$, and pooling all successes gives the Minimum Variance Unbiased Estimator (MVUE). But for the Confidence Interval, we do *not* assume $H_0$ is true; we are estimating the true difference $(p_1 - p_2)$, which requires estimating separate sample variances (unpooled standard error).

#### 2. Why Welch's t-test instead of Student's pooled t-test?
- **Answer**: Student's t-test assumes equal variances ($\sigma_1^2 = \sigma_2^2$). In real behavioral data, treatments frequently alter variance as well as the mean. In Cookie Cats, the raw data had a 6.18x variance ratio. Welch's t-test with Welch–Satterthwaite degrees of freedom accommodates unequal variances safely while losing virtually zero statistical power if variances happen to be equal.

#### 3. How does CUPED achieve variance reduction without biasing the treatment effect?
- **Answer**: CUPED computes $Y_{\text{CUPED}} = Y - \theta (X - \bar{X})$, where $\theta = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}$ is the OLS regression slope. Because the covariate $X$ is measured *before* treatment assignment, randomization guarantees that $E[X] = \bar{X}$ in both Control and Treatment. Therefore, $E[Y_{\text{CUPED}}] = E[Y] - \theta(0) = E[Y]$. The point estimate is strictly unbiased, while the variance shrinks by $1 - \rho^2$.

#### 4. Why can't you detect a novelty effect from a single aggregate metric?
- **Answer**: An aggregate metric computes the average across the entire experiment duration. If a feature causes an initial curiosity spike on Days 1–3 (+5 rounds) that decays to 0 by Day 14, the aggregate average will still show a statistically significant +1.2 round lift. The team ships a feature that has zero long-term value. Detecting it requires time-windowed Welch's t-tests comparing early vs. late windows.

#### 5. Why is the pipeline ordered the way it is in FastAPI?
- **Answer**: We sequence: Data Hygiene $\to$ Retrospective Power $\to$ Core Test $\to$ Sequential Safety $\to$ CUPED $\to$ Guardrails. We must validate data hygiene and test sensitivity first. Core testing establishes the baseline effect before sequential safety adjusts critical values. CUPED optimizes precision on the validated metric, and guardrail diagnostics inspect for novelty decay and subgroup confounding *after* the main treatment effect has been characterized.

---

## 6. How to Run & Verify

### 1. Run Complete Automated Unit Test Suite (24 tests):
```powershell
python -m pytest tests/ -v
```

### 2. Run End-to-End Pipeline CLI:
```powershell
python run_full_pipeline.py
```

### 3. Run Individual Phase Pipelines:
```powershell
# Phase 1: Cookie Cats Core Engine
python run_engine.py

# Phase 2: Peeking Simulation & Sequential Correction
python run_phase2.py

# Phase 3: CUPED Variance Reduction
python run_phase3.py

# Phase 4: Guardrail Diagnostics (Novelty & Simpson's)
python run_phase4.py
```

### 4. Run FastAPI Microservice:
```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be available at: `http://127.0.0.1:8000/docs`.

### 5. Generate All Visual Plots:
```powershell
python src/visualize.py
python src/visualize_peeking.py
python src/visualize_cuped.py
python src/visualize_guardrails.py
```
All figures are output to `reports/figures/`.
