# Experimentation concepts

This guide contains the concept explanations and formulas that were previously in the main README. For a longer walkthrough, see [STUDY_GUIDE.md](STUDY_GUIDE.md).

## 2. Statistics from zero

### A metric is simply something you measure

A **binary metric** has two outcomes: yes/no or 1/0. A purchase, a click, or returning on day seven can be recorded this way. If 20 of 100 players return, the retention rate is `20 / 100 = 20%`.

A **numerical metric** measures an amount, such as revenue or rounds played. This project calls these metrics `continuous`, even when the values are whole-number counts. It compares their averages: total rounds divided by number of players.

### A percentage point is different from a percent

Imagine conversion rises from 20% to 22%:

| Quantity | Calculation | Result |
|---|---|---|
| Absolute difference | 22% minus 20% | **+2 percentage points** |
| Relative lift | (22% minus 20%) divided by 20% | **+10%** |
| Approximate practical meaning | 2% of 1,000 comparable users | **20 additional conversions per 1,000 users** |

The engine reports effects as **treatment minus control**. A positive difference means the measured outcome increased. Whether that is good depends on the metric: more retention may be desirable; more crashes is not.

### Random variation

Two groups can differ even when both see exactly the same product. Like tossing a fair coin ten times, you should not expect an exact 50/50 split every time.

A statistical test asks whether the observed gap is unusually large relative to the variation expected under a model of no effect. Larger samples usually improve precision, but more data cannot repair biased assignment or incorrect measurement.

### Null hypothesis and p-value

The **null hypothesis** is the starting model of no population difference between the two versions.

A **p-value** measures how often the test would produce a result at least as extreme as the one observed, assuming that no-difference model and the test's assumptions are true. The pipeline uses two-sided tests, which consider differences in either direction.

A p-value of 0.03 does **not** mean there is a 3% probability that the new version has no effect. It also does not mean there is a 97% probability that shipping it is a good decision.

### Alpha and false positives

**Alpha** is the planned significance threshold, usually 0.05. The code labels a result significant when its p-value is below the relevant threshold.

A **false positive** happens when a test detects a difference although there is no true effect. Under a correctly calibrated procedure and its assumptions, alpha = 0.05 targets about five false alarms per 100 repeated no-effect experiments. This is a long-run error rate, not a probability attached to one particular conclusion.

### Confidence intervals

A confidence interval shows uncertainty around the estimated effect. An effect interval from **-1.3 to -0.3 percentage points** points toward a decline; one from **-1.3 to +0.1 points** leaves both a decline and a small improvement compatible with the analysis.

A 95% confidence procedure would cover the true effect in about 95% of repeated experiments under its assumptions. It does not assign a 95% probability to the fixed true effect after the interval has been calculated. See [NIST's explanation of confidence limits](https://itl.nist.gov/div898/handbook/eda/section3/eda352.htm).

The API's core intervals describe the **raw, fixed-horizon** comparison. They are not automatically CUPED-adjusted or valid for arbitrary repeated stopping.

### Spread, standard error, and power

| Term | Plain-language meaning | Example |
|---|---|---|
| Mean | Average value | Total rounds / players |
| Variance / standard deviation | How spread out individual values are | Some players play 2 rounds; others play 200 |
| Standard error | How uncertain an estimated average or difference is | More independent users usually reduce it |
| Power | Chance of detecting a specified true effect under the model | 80% power means detection in about 80 of 100 repeated experiments with that effect |
| Minimum detectable effect (MDE) | Approximate effect size needed for the chosen power at the available sample size | A test may reliably detect a 2-point change but struggle with a 0.2-point change |

**Not significant does not mean identical.** A small study can miss an important effect. A statistically significant effect may also be too small to matter economically. MDE is a sensitivity estimate, not a confidence interval or a definition of business importance.

## 3. What each component does

### Two-proportion z-test: compare rates

Use this for binary outcomes such as retention or conversion. It compares successes divided by users in each group. The implementation combines the groups to estimate noise under the no-difference hypothesis, then uses separate group estimates for its confidence interval.

Example: compare 200 returning players out of 1,000 in A with 220 out of 1,000 in B. The observed difference is +2 percentage points, but a test is still needed to assess its uncertainty.

This is a large-sample approximation, **not an exact test**. Rare events, tiny samples, and rates near 0% or 100% need extra care; the engine does not automatically switch to an exact or score-based method.

### Welch's t-test: compare averages

Use this for amounts such as rounds played. Welch's test allows the groups to have different sizes and different variability.

For example, A may contain mostly regular players while B contains a mix of very light and very heavy players. Treating their variability as identical can misrepresent uncertainty. Welch uses each group's variability separately. It still assumes independent observations and does not make extreme observations harmless.

### Power and MDE: understand sensitivity

After the core test, the engine estimates the detectable effect using observed sample sizes and variation. The binary calculation uses a standardized rate difference called **Cohen's h**. The numerical calculation uses a noncentral-t approximation with Welch's standard error and degrees of freedom.

This is a **retrospective sensitivity calculation**. It should not replace sample-size planning before an experiment or turn a nonsignificant result into evidence of equivalence. For binary metrics, the reported MDE averages upward and downward distances, so it is not an exact symmetric detection boundary, especially near 0% or 100%.

### A/A simulation and the peeking problem

An **A/A experiment** gives both groups the same underlying success rate. Any declared difference is therefore a false alarm in the simulation.

Checking an ordinary p-value every day and stopping as soon as it falls below 0.05 creates repeated opportunities for a false alarm. The project simulates this using fresh users each day and cumulative results.

| Method | Idea | Where it is available |
|---|---|---|
| Pocock-style | Use the same stricter threshold at each planned look | API pipeline and simulation functions |
| Discrete O'Brien-Fleming | Require stronger evidence early; relax later while accounting for all planned looks | Python boundary helper and simulation evaluator; **not selectable in the HTTP pipeline** |

The Pocock helper supports overall alpha **0.05 only**. It uses tabulated thresholds for 1–14 and 20 looks, and a logarithmic approximation for other counts. Those approximations and arbitrary look schedules are not a universal error-control guarantee.

The O'Brien-Fleming helper numerically calibrates the probability of crossing any planned boundary under a joint-normal model. It accepts 1–100 looks and optional increasing **information fractions**: how much of the planned statistical information is available at each look. With equal fresh samples per day in the simulator, day divided by total days supplies that fraction. Time alone is not generally equivalent to information in real experiments.

### CUPED: use pre-experiment history to reduce noise

CUPED means **Controlled-experiment Using Pre-Experiment Data**. Players who were active before an experiment are often active afterward too. CUPED uses that relationship to subtract some predictable variation before comparing groups.

A simple illustration: if the fitted coefficient is 0.5, a player's pre-period value is 60, and the overall pre-period average is 50, CUPED subtracts `0.5 × (60 - 50) = 5` from that player's post-period value. The same fitted adjustment is applied across the selected groups.

The code estimates this relationship using only the requested control and treatment variants, transforms their outcomes, and runs Welch's test on the adjusted values. If the covariate is useful, uncertainty can shrink. See [Microsoft Research's CUPED explanation](https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/articles/deep-dive-into-variance-reduction/).

The covariate must genuinely precede treatment. Reusing a value influenced by the feature can bias the answer. The engine cannot verify measurement timing from a column name. Adjustment can change the estimated difference in a particular sample; it does **not** promise exactly the same lift before and after adjustment. The reported effective sample multiplier is an approximate precision comparison, not additional real users.

### Novelty check: did an early improvement fade?

Imagine a feature initially adds five rounds per player but later adds only one. An overall average can hide that decline.

The check compares days **1–3** with days **12–14** by default. It flags decay only when:

1. The early effect is positive and statistically significant.
2. The estimated lift falls by at least 60%.
3. A direct, one-sided early-minus-late effect test is significant.

A significant early result and a nonsignificant late result alone are insufficient. The later sample may simply be noisier.

The check assumes independent observations across windows. Repeated measurements of the same users require a paired or clustered model, which is not implemented here. A flag identifies evidence of attenuation; it cannot establish curiosity as the cause or predict the permanent effect. The Python function supports custom windows; the HTTP request does not expose them.

### Simpson's paradox: the overall comparison can reverse

Consider this deliberately constructed example:

| Platform | A returning / total | A rate | B returning / total | B rate |
|---|---:|---:|---:|---:|
| iOS | 120 / 500 | 24% | 420 / 2,000 | 21% |
| Android | 210 / 1,500 | 14% | 55 / 500 | 11% |
| Overall | 330 / 2,000 | **16.5%** | 475 / 2,500 | **19%** |

B is worse by three percentage points on **each** platform, yet looks better overall. Why? Most B users belong to the higher-retention iOS group. The combined rates use different weights.

The detector checks whether every segment has an effect in one direction while the aggregate effect goes the other way. It reports a warning; it does not adjust the estimate for confounding or prove a causal explanation. Random assignment balances groups in expectation, not perfectly in every realized sample.

## 14. Optional formulas and further reading

You can use the project without memorizing these. They connect the plain-language explanation to the implementation.

Let A be control, B treatment, `n` the number of observations, `x` the number of binary successes, `mean` an average, and `s²` a sample variance.

| Calculation | Formula or idea |
|---|---|
| Rate | `p = x / n` |
| Reported effect | `p_B - p_A` or `mean_B - mean_A` |
| Relative lift | `effect / control_value` when the baseline is nonzero |
| Pooled binary rate | `(x_A + x_B) / (n_A + n_B)` |
| Binary test standard error | `sqrt(p_pool × (1 - p_pool) × (1/n_A + 1/n_B))` |
| Binary interval standard error | `sqrt(p_A × (1-p_A)/n_A + p_B × (1-p_B)/n_B)` |
| Welch standard error | `sqrt(s_A²/n_A + s_B²/n_B)` |
| Test statistic | `(control - treatment) / standard_error` |
| Two-sided interval | `effect ± critical_value × standard_error` |
| CUPED adjustment | `Y_adjusted = Y - theta × (X - mean(X))` |
| CUPED coefficient | `theta = covariance(X,Y) / variance(X)` |
| Idealized residual variance ratio | `1 - rho²`, where rho is pre/post correlation |
| O'Brien-Fleming boundary | `z_k = c / sqrt(t_k)`; c is calibrated across the entire planned schedule |
| Novelty contrast | `(early_B - early_A) - (late_B - late_A)` |

For Welch's test, the critical value comes from a t distribution with estimated degrees of freedom. Degrees of freedom describe how much information is available for estimating variability. For the novelty contrast, independent contributions to its variance come from all four group/window cells.

Useful primary references:

- [NIST: confidence limits](https://itl.nist.gov/div898/handbook/eda/section3/eda352.htm).
- [SciPy: independent-sample t-test, including Welch with `equal_var=False`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html).
- [statsmodels: two-proportion z-test](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html).
- [Original CUPED paper: Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data](https://ai.stanford.edu/~ronnyk/2013-02CUPEDImprovingSensitivityOfControlledExperiments.pdf).

For the exact behavior of this implementation, follow the module links in the [README](README.md) and the regression tests rather than assuming that every feature described in a reference is implemented here.
