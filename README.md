# A/B Testing & Experimentation Engine

A Python project for comparing two versions of a product, measuring uncertainty, and demonstrating common mistakes in experiment analysis. It includes statistical calculations, simulated experiments, diagnostic checks, charts, and a FastAPI HTTP interface.

New to the concepts? See [teaching.md](teaching.md).

This is an educational analysis engine with automated tests, not a guarantee of correct business decisions or a fully hardened production platform. Its conclusions depend on the experiment design, input data, and statistical assumptions. The descriptions and numerical examples below were checked against the code after the accuracy fixes in commit `0d93d53`.

## Contents

1. [What problem does this solve?](#1-what-problem-does-this-solve)
2. [Statistics from zero](#2-statistics-from-zero)
3. [What each component does](#3-what-each-component-does)
4. [The datasets](#4-the-datasets)
5. [Install and run](#5-install-and-run)
6. [Analyze your own data](#6-analyze-your-own-data)
7. [Use the HTTP API](#7-use-the-http-api)
8. [Read the report correctly](#8-read-the-report-correctly)
9. [Verified example results](#9-verified-example-results)
10. [How the pipeline works](#10-how-the-pipeline-works)
11. [Project map](#11-project-map)
12. [Tests and accuracy fixes](#12-tests-and-accuracy-fixes)
13. [Limitations and troubleshooting](#13-limitations-and-troubleshooting)
14. [Optional formulas and further reading](#14-optional-formulas-and-further-reading)

## 1. What problem does this solve?

Suppose you make a mobile game. You currently place a progression gate at level 30 and wonder whether moving it to level 40 will make players return more often.

You randomly assign players to two groups:

- **A, the control:** the existing version, with the gate at level 30.
- **B, the treatment:** the proposed version, with the gate at level 40.

You measure whether each player returns seven days later. Maybe 19% return in A and 18% in B. Is the new version harmful, or could random differences between the groups explain that gap?

This project estimates the difference and its uncertainty. It also explores what happens when you repeatedly check results, use earlier player behavior to improve precision, or combine groups with different characteristics.

The engine **analyzes collected data**. It does not assign real users to variants, collect events, deploy features, or automatically run an experiment. The simulator generates artificial assignments and outcomes only for demonstrations.

A sensible experiment starts by choosing the primary outcome, the smallest worthwhile improvement, the assignment method, and the stopping plan **before** looking at results.

## 2. Statistics from zero

The [teaching guide](teaching.md#2-statistics-from-zero) covers rates, uncertainty, p-values, confidence intervals and power.

## 3. What each component does

The engine compares proportions and averages, estimates power, and includes demonstrations of sequential testing, CUPED, novelty effects and Simpson's paradox. The [teaching guide](teaching.md#3-what-each-component-does) explains the methods and their assumptions.

## 4. The datasets

| Dataset or generator | What it contains | Real or synthetic? |
|---|---|---|
| `data/cookie_cats.csv` | 90,189 players; level-30 versus level-40 gate; retention and game rounds | Real-world example dataset bundled in this repository |
| `data/cookie_cats_segmented.csv` | 4,500 rows constructed with the platform rates in the [teaching guide](teaching.md#simpsons-paradox-the-overall-comparison-can-reverse) | **Synthetic**; not measured device metadata from the original experiment |
| `data/synthetic_cuped_stream.csv` | Example pre/post activity stream | **Synthetic**; overwritten with 5,000 generated users by `run_full_pipeline.py` |
| `generate_aa_experiments` | Daily independent binomial successes, equal true rate in both groups | Synthetic |
| `generate_novelty_decay_data` | Independent daily cohorts with a declining injected effect | Synthetic |
| `get_kidney_stone_data` | Expanded success/failure records from a classic aggregate example | Illustrative reconstructed records, not original individual medical records |

Cookie Cats columns:

| Column | Meaning |
|---|---|
| `userid` | Player identifier |
| `version` | `gate_30` or `gate_40` |
| `sum_gamerounds` | Rounds played during the first 14 days |
| `retention_1` | Whether the player returned after one day |
| `retention_7` | Whether the player returned after seven days |

Cookie Cats has no genuine pre-experiment behavior column for CUPED. The separate synthetic generator demonstrates that method. Its nominal treatment effect is added before clipping outcomes at zero, so clipping and sampling mean the realized difference need not exactly equal the input effect.

## 5. Install and run

You need Python, a terminal, and the files in this repository. Git is needed only if you clone rather than download the repository. Verification here used Python 3.14 on Windows; the repository does not declare a complete supported-version matrix. Dependencies have minimum versions in `requirements.txt`, not a lockfile.

From a terminal:

```bash
git clone https://github.com/PuravDesai004/A-B-Testing.git
cd A-B-Testing
python -m venv .venv
```

Activate the environment in **Windows PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on **macOS/Linux**:

```bash
source .venv/bin/activate
```

On systems where Python is named `python3`, use that command to create the environment. Once activated, install and test:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -v
```

The current suite contains **46 test cases**. If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` directly in place of `python`; activation is only a convenience.

### Run a complete example

```bash
python run_full_pipeline.py
```

This prints analyses for Cookie Cats day-seven retention, Cookie Cats game rounds, and a synthetic CUPED experiment. It writes `reports/end_to_end_output.json` and regenerates `data/synthetic_cuped_stream.csv`. It demonstrates the unified pipeline on three requests; it does **not** exercise every optional guardrail or sequential method.

### Run individual teaching demonstrations

| Command | What it runs or writes |
|---|---|
| `python run_engine.py` | Phase 1; writes `reports/phase1_results.md`; uses the legacy filtered Cookie Cats loader |
| `python run_phase2.py` | A/A peeking and sequential comparisons; writes `reports/phase2_results.md` |
| `python run_phase3.py` | CUPED demonstration with 10,000 generated users; writes `reports/phase3_results.md` |
| `python run_phase4.py` | Novelty and Simpson examples; writes `reports/phase4_results.md` and regenerates the segmented CSV |

Some prose in these legacy report templates and [STUDY_GUIDE.md](STUDY_GUIDE.md) predates the fixes. Treat those as historical teaching material, not a specification of the current pipeline. Regenerating a report updates its computed values but does not correct every hardcoded explanatory statement in its template.

### Generate charts

Run from the repository root:

```bash
python -m src.visualize
python -m src.visualize_peeking
python -m src.visualize_cuped
python -m src.visualize_guardrails
```

These write PNG files under `reports/figures/`. The Phase 1 chart path uses the legacy data loader, so its population can differ from the current API pipeline. The notebook in `notebooks/` contains exploratory analysis; Jupyter is not included in the runtime requirements.

## 6. Analyze your own data

Use one independent observation per randomized unit for the main comparison—for example, one row per user containing that user's total rounds. Multiple events from one user are not independent users. The pipeline does not deduplicate or aggregate them for you.

A tiny illustrative CSV could look like this:

```csv
user_id,variant,converted,revenue
1,control,1,12.5
2,control,0,0
3,treatment,1,15
4,treatment,0,0
```

This shows the **format**, not a sufficient sample size for reliable inference. For binary outcomes, prefer `0` and `1`; booleans are also accepted. Avoid strings such as `yes` and `no`.

### Run the bundled retention analysis from Python

Save a script in the repository root or use an interactive Python session there:

```python
from src.pipeline import ExperimentPipeline
from src.schemas import ExperimentAnalysisRequest

request = ExperimentAnalysisRequest(
    dataset_name="Cookie Cats day-seven retention",
    metric_type="proportion",
    metric_column="retention_7",
    variant_column="version",
    control_value="gate_30",
    treatment_value="gate_40",
)
report = ExperimentPipeline.run(request)
print(report.model_dump_json(indent=2))
```

For your own CSV, set `dataset_path="data/my_experiment.csv"` and map the metric and group column names to your file. For revenue or rounds, use `metric_type="continuous"`. Relative paths are resolved from the process's working directory.

### Run CUPED without creating a CSV

```python
from src.synthetic_cuped_data import generate_cuped_dataset
from src.pipeline import ExperimentPipeline
from src.schemas import ExperimentAnalysisRequest

dataset = generate_cuped_dataset(num_users=10000, random_seed=42)
request = ExperimentAnalysisRequest(
    dataset_name="Synthetic CUPED example",
    metric_type="continuous",
    metric_column="post_rounds",
    variant_column="variant",
    control_value="control",
    treatment_value="treatment",
    covariate_column="pre_rounds",
)
report = ExperimentPipeline.run(request, df=dataset.df)
print(report.cuped_adjustment.model_dump())
```

### Input handling you should know

- Only the named control and treatment variants enter the pipeline. Other variants are excluded.
- Missing primary outcomes are excluded before sample sizes, tests, and power calculations. This estimates effects among observed outcomes; missingness related to treatment or behavior can still bias the analysis.
- Binary outcomes must be 0 or 1. Infinite outcomes are rejected. Group labels must differ and the selected groups need usable observations.
- Requested columns must exist. Optional stages are skipped when their column parameter is omitted; explicitly requesting a nonexistent column raises an error.
- CUPED requires finite, nonmissing pre-period covariates on the retained rows. Missing covariates are not imputed.
- The unified pipeline does not remove users because their `sum_gamerounds` is unusually large. Any exclusion policy should be justified and defined before examining treatment effects.
- The separate `load_cookie_cats_data()` helper still defaults to filtering values above 10,000 rounds. That is a legacy demonstration policy, not a proven rule that those users are invalid.

## 7. Use the HTTP API

An API lets another program request an analysis over HTTP. Start this local server from the repository root:

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Open [the interactive API documentation](http://127.0.0.1:8000/docs). Expand `POST /experiment/analyze`, choose **Try it out**, enter the JSON below, and choose **Execute**.

```json
{
  "dataset_name": "Cookie Cats day-seven retention",
  "metric_type": "proportion",
  "metric_column": "retention_7",
  "variant_column": "version",
  "control_value": "gate_30",
  "treatment_value": "gate_40",
  "alpha": 0.05,
  "power": 0.8,
  "interim_looks_planned": 1
}
```

Omitting `dataset_path` selects the bundled Cookie Cats CSV. This endpoint accepts a path **on the server's filesystem**, not a CSV upload or a URL to download.

| Request field | Default | Meaning |
|---|---|---|
| `dataset_name` | `cookie_cats` | Label used in the report |
| `dataset_path` | `null` | Server-local CSV path; omitted means the default dataset |
| `metric_type` | `proportion` | `proportion` or `continuous` |
| `metric_column` | `retention_7` | Outcome to compare |
| `variant_column` | `version` | Column identifying experiment groups |
| `control_value` | `gate_30` | Control group label |
| `treatment_value` | `gate_40` | Treatment group label |
| `covariate_column` | `null` | Enables CUPED using this pre-experiment column |
| `day_column` | `null` | Enables novelty analysis; use integer experiment-day numbers |
| `segment_column` | `null` | Enables subgroup reversal analysis |
| `alpha` | `0.05` | Schema range 0.001–0.20; multiple-look Pocock currently requires 0.05 |
| `power` | `0.80` | Target power, allowed range 0.50–0.99 |
| `interim_looks_planned` | `1` | Planned total looks, allowed range 1–100; 1 means fixed horizon |

`GET /health` reports service status and advertised capabilities. It does not test your dataset or guarantee that every method is selectable through the analyze endpoint.

Expected errors include **404** for a missing CSV, **422** for request or statistical validation errors, and **500** for unhandled failures. One current example of a 500 is requesting multiple looks with alpha other than 0.05, because the Pocock helper does not implement that combination.

## 8. Read the report correctly

| Response section | What it tells you |
|---|---|
| `experiment_name`, `metric_analyzed`, `metric_type` | Which analysis this is |
| `sample_sizes` | Retained observations in the selected groups, after missing outcomes are removed |
| `core_test` | Raw group means/rates, difference, relative lift, statistic, p-value, fixed-horizon interval, and nominal-alpha significance flag |
| `power_check` | Raw-analysis sensitivity at the requested power and nominal alpha |
| `sequential_safety` | Fixed-horizon status or the Pocock per-look threshold |
| `cuped_adjustment` | Whether CUPED ran; coefficient, correlation, variance reduction, adjusted difference and p-value |
| `guardrails` | Novelty and Simpson flags plus explanations; omitted checks are marked skipped |
| `executive_recommendation` | Rule-based interpretation using CUPED results if applied, otherwise raw results, with the sequential threshold when requested |

### Units and signs

For proportions, `absolute_diff = -0.0082` means approximately **-0.82 percentage points**. A `relative_lift_pct` of `-4.31` means **-4.31% relative lift**. Proportion confidence limits and `power_check.mde_absolute` also use fraction units; multiply by 100 to express percentage points.

For continuous outcomes, absolute differences, intervals, and MDE use the outcome's units, such as rounds per player. `cuped_adjustment.adjusted_lift` is an absolute difference, not a percentage.

The z and t statistics use **control minus treatment** to match reference library conventions, while reported effects use **treatment minus control**. A positive test statistic can therefore accompany a negative treatment effect. Read the named difference field to determine direction.

### Why different sections can disagree

`core_test.is_significant` uses raw data and nominal alpha. The recommendation can use a stricter sequential threshold or a CUPED-adjusted p-value. A raw p-value of 0.02 is significant at 0.05 but fails the roughly 0.0089 threshold for 14 Pocock looks.

Core confidence intervals and power estimates are not recalculated for those later adjustments. `looks_evaluated` contains the requested planned count; the API does not store or inspect a history of previous looks.

The recommendation assumes **higher is better**. It also appends guardrail warnings without automatically removing a headline “WIN.” Read the complete report and apply the metric's business meaning before acting.

## 9. Verified example results

These are rounded outputs from the current functions, not promised results for other datasets or environments. Seeded simulations are reproducible within the relevant software environment; finite simulation counts still have sampling uncertainty.

### Cookie Cats through the current unified pipeline

Settings: bundled CSV, no game-rounds filtering, fixed horizon, alpha 0.05, target power 0.80. All three comparisons retain **44,700 control + 45,489 treatment = 90,189 players**.

| Metric | Control | Treatment | Treatment minus control | p-value | Raw 95% interval for difference |
|---|---:|---:|---:|---:|---:|
| Day-one retention | 44.8188% | 44.2283% | -0.5905 percentage points | 0.074410 | [-1.2392, +0.0582] percentage points |
| Day-seven retention | 19.0201% | 18.2000% | -0.8201 percentage points | 0.001554 | [-1.3282, -0.3121] percentage points |
| Game rounds | 52.4563 | 51.2988 | -1.1575 rounds | 0.375924 | [-3.7197, +1.4047] rounds |

Day-seven retention has evidence of a decline under this analysis; its relative lift is **-4.3119%**. The other two comparisons do not meet the 0.05 threshold. They do not establish that the versions are equivalent. These are individual metric tests, not a multiple-metric-adjusted experiment-wide conclusion.

The corresponding raw 80%-power MDE estimates are approximately **0.9278 percentage points**, **0.7322 percentage points**, and **3.6624 rounds**.

Older reports show game-rounds p ≈ 0.95 after removing the extreme player. The current pipeline retains that observation and gives p ≈ 0.376. Neither preprocessing choice should be hidden; deleting a large value is a substantive analysis decision.

### Sequential A/A demonstration

Settings: 10,000 simulations, 14 days, 1,000 new users per group per day, true rate 0.10 in both groups, alpha 0.05, seed 42.

| Decision procedure | Simulated false-positive rate |
|---|---:|
| Ordinary test only at the final day | 5.08% |
| Ordinary test every day, stop at first significance | 22.01% |
| Pocock threshold at all 14 daily looks | 4.84% |
| Calibrated discrete O'Brien-Fleming at all 14 daily looks | 5.06% |

The previous uncalibrated O'Brien-Fleming formula produced roughly 7.3% false positives in this example. The corrected calculation accounts for earlier boundary crossings. The two corrected methods reduce repeated-testing errors in this design; they do not make every individual decision correct.

Reproduce these figures without writing reports:

```python
from src.peeking_simulation import run_peeking_simulation
from src.sequential_correction import (
    evaluate_pocock_correction,
    evaluate_obrien_fleming_correction,
)

sim = run_peeking_simulation(num_simulations=10000, random_seed=42)
print("Final look only:", sim.fixed_horizon_fpr)
print("Naive daily looks:", sim.naive_peeking_fpr)
print("Pocock:", evaluate_pocock_correction(sim).corrected_fpr)
print("O'Brien-Fleming:", evaluate_obrien_fleming_correction(sim).corrected_fpr)
```

### Other synthetic demonstrations

| Example and settings | Verified result | Meaning |
|---|---|---|
| CUPED: 10,000 users, nominal injected effect 1.5, target correlation 0.75, seed 42 | Correlation 0.7462; empirical variance reduction 55.81%; CI width reduction 33.52%; reported effective sample multiplier 2.256 | Useful pre-period information improved precision in this generated example |
| Same CUPED run | Raw estimated effect +1.6972; adjusted effect +1.8420 | Adjustment can change the sample estimate; neither must equal the generator input exactly |
| Novelty generator defaults, seed 42 | Early lift +3.2793; late +0.8653; estimated decay 73.61%; direct decrease p ≈ 0.00000160 | Positive early lift attenuated enough to trigger the check |
| Constructed platform example | B wins overall by 2.5 points, loses within each platform by 3 points | Simpson's reversal is detected |

The end-to-end runner uses **5,000**, not 10,000, CUPED users, so its results differ from the CUPED row above.

## 10. How the pipeline works

```text
CSV file or pandas DataFrame
          |
          v
Validate columns and labels; select A/B; exclude missing outcomes
          |
          v
Run raw proportion or Welch test, then raw MDE calculation
          |
          v
Choose fixed-horizon or Pocock decision threshold
          |
          v
Apply CUPED if a covariate column was requested
          |
          v
Check novelty if days were requested (uses CUPED outcome when available)
Check Simpson reversal if segments were requested (uses raw outcome)
          |
          v
Return a structured report and rule-based recommendation
```

FastAPI parses and validates request fields using Pydantic, then calls `ExperimentPipeline.run`. The pipeline returns a Pydantic response object. FastAPI serializes that object to JSON, a structured text format other programs can read.

This is a synchronous, single-request analysis. It does not persist experiments, schedule repeated analyses, or maintain sequential stopping state. Statistical functions can also be used independently of the API.

## 11. Project map

| File or folder | Responsibility |
|---|---|
| [`src/data_loader.py`](src/data_loader.py) | Locate Cookie Cats; legacy audit and optional outlier filtering; group summaries |
| [`src/proportion_test.py`](src/proportion_test.py) | Binary rate comparison and statsmodels cross-check |
| [`src/continuous_test.py`](src/continuous_test.py) | Welch comparison and SciPy cross-check |
| [`src/power_check.py`](src/power_check.py) | Approximate MDE calculations |
| [`src/aa_simulator.py`](src/aa_simulator.py) | Generate independent daily no-effect experiment streams |
| [`src/peeking_simulation.py`](src/peeking_simulation.py) | Cumulative p-values and false-alarm measurement |
| [`src/sequential_correction.py`](src/sequential_correction.py) | Pocock thresholds and calibrated O'Brien-Fleming boundaries |
| [`src/synthetic_cuped_data.py`](src/synthetic_cuped_data.py) | Generate correlated pre/post data |
| [`src/cuped.py`](src/cuped.py) | Fit and apply CUPED; compare raw and adjusted analyses |
| [`src/novelty_check.py`](src/novelty_check.py) | Generate a decay example and compare early/late effects |
| [`src/simpsons_check.py`](src/simpsons_check.py) | Construct examples and detect aggregate/subgroup reversal |
| [`src/schemas.py`](src/schemas.py) | Request and response field definitions |
| [`src/pipeline.py`](src/pipeline.py) | Orchestrate the analysis stages |
| [`src/api.py`](src/api.py) | FastAPI routes and error handling |
| `src/visualize*.py` | Matplotlib/seaborn chart generation |
| `run_engine.py`, `run_phase2.py`–`run_phase4.py` | Individual teaching report generators |
| [`run_full_pipeline.py`](run_full_pipeline.py) | Three unified-pipeline examples and JSON export |
| [`tests/`](tests/) | Formula comparisons, simulations, API tests, and accuracy regressions |
| [`data/`](data/) | Bundled real and synthetic CSVs |
| [`notebooks/`](notebooks/) | Cookie Cats exploration |
| [`reports/`](reports/) | Saved reports and charts; historical artifacts may predate fixes |
| [`STUDY_GUIDE.md`](STUDY_GUIDE.md) | Extended historical study material; see the caveat in section 5 |
| [`requirements.txt`](requirements.txt) | Runtime, plotting, and test dependencies |

NumPy handles arrays and simulation, pandas handles tables, SciPy supplies probability distributions and numerical solvers, and statsmodels supplies power utilities and reference comparisons. FastAPI/Pydantic/Uvicorn provide the HTTP service; pytest/httpx support tests; Matplotlib/seaborn draw charts.

## 12. Tests and accuracy fixes

The current suite has **46 passing cases**, including **22 new regression cases** for the accuracy fixes. Passing tests is evidence about covered behavior, not proof that all inputs or statistical designs are valid.

| Area | What is checked |
|---|---|
| Core calculations | z-test agrees with statsmodels; Welch statistic, p-value, and degrees of freedom agree with SciPy on tested cases |
| Sequential behavior | A/A peeking inflates errors; corrected procedures are checked by simulation |
| O'Brien-Fleming regressions | 50,000 simulations each for daily and irregular looks; acceptance range 4.6%–5.4% at nominal 5% |
| CUPED | Coefficient checks, variance reduction, and independence from unrelated variants |
| Guardrails | Known decay, stable effects, negative-effect recovery, custom labels, and known Simpson examples |
| API | Health endpoint and representative binary, continuous, sequential, and CUPED requests |
| Data handling and power | Missing outcomes, nonbinary values, retained large values, consistent sample counts, and unequal-variance MDE |

The fixes removed seven specific sources of incorrect or misleading results: missing outcomes treated as failures, automatic post-treatment filtering, uncalibrated O'Brien-Fleming boundaries, false novelty flags from changing precision, hardcoded novelty labels, pooled-variance power estimates for Welch tests, and unrelated variants entering CUPED fitting.

## 13. Limitations and troubleshooting

### Limits that affect interpretation

- **Assignment and measurement:** The engine cannot verify randomization, exposure timing, logging completeness, or whether outcomes matured. It does not perform a sample-ratio-mismatch test or enforce unique user IDs.
- **Independence:** Repeated user events, clustered assignments, and network effects need models not implemented here. Aggregate or deduplicate according to the actual randomization unit before using these tests.
- **Approximate inference:** Binary tests use a normal approximation and Wald intervals. Very small samples and extreme rates can give unreliable intervals. Both groups having all zeros or all ones can produce undefined statistics; constant continuous samples raise an error. These cases are not comprehensively handled by the API.
- **Relative effects at zero:** Some helpers return zero relative lift when the control mean/rate is zero. Mathematically that relative effect is undefined; use the absolute difference instead.
- **Sequential restrictions:** Pocock's table and fallback approximation have limited calibration. Its helper rejects alpha other than 0.05. The pipeline does not accept an information schedule, record past looks, or offer always-valid inference. Its safety wording should not be read as a guarantee outside the calibrated setting.
- **Multiple comparisons:** Repeated-look correction is not correction for testing many metrics, variants, segments, or experiments. The project does not apply general family-wise or false-discovery-rate control across them.
- **CUPED assumptions:** It uses a single fitted linear adjustment and an ordinary Welch test on adjusted data; it is not a full regression-inference framework accounting for every nuisance-estimation issue. Degenerate covariates or near-perfect correlations can fail. The variance summary averages the two within-group variances equally, and its theoretical sample multiplier need not describe unequal-size groups exactly.
- **Novelty:** Default windows require usable data in days 1–3 and 12–14. Shorter studies need the Python helper with explicit windows. Unsupported or sparse data can fail rather than yield a partial report.
- **Simpson detector:** It checks strict sign reversal in every segment, not every possible form of confounding. It currently substitutes zero for a group missing from a segment; use complete, nonmissing segments containing both groups. No flag is not proof that groups are comparable.
- **Recommendations:** Higher values are always called a win. Guardrail warnings do not automatically block that label, and the program does not measure rollout costs, harms, long-term value, or practical significance.
- **Deployment:** No authentication, dataset-path allowlist, job queue, persistent experiment store, or production load benchmark is implemented. Keep the demonstrated server local; exposing server-local file paths to untrusted callers requires additional controls.

### Common problems

| Symptom | What to check |
|---|---|
| `No module named ...` | Install `requirements.txt` using the same Python executable that runs the command |
| `cookie_cats.csv could not be located` | Run from the repository root and check the bundled file or `dataset_path` |
| HTTP 422 | Read the error detail: check field names, labels, required columns, valid binary values, covariates, and usable sample sizes |
| Multiple-look request fails with nondefault alpha | Current pipeline Pocock support requires alpha 0.05 |
| Novelty check has insufficient observations | Confirm both groups exist in the default early and late windows |
| A new p-value differs from an old report | Compare filtering, sample sizes, metric, seed, number of generated users, and dependency versions |
| Core significance differs from recommendation | Check the sequential threshold and whether CUPED was applied |

## 14. Optional formulas and further reading

See the [formulas and references](teaching.md#14-optional-formulas-and-further-reading) in the teaching guide.
