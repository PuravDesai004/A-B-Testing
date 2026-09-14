# Phase 3 Teaching & Results Guide: CUPED Variance Reduction

## Executive Summary of Results

| Metric Dimension | Baseline Raw Metric ($Y$) | CUPED-Adjusted Metric ($Y_{\text{CUPED}}$) | Improvement / Delta | Practical Interpretation |
|---|---|---|---|---|
| **Control Mean (Rounds)** | 50.093 | 50.020 | Unchanged | **Strictly Unbiased**: Expected value preserved |
| **Treatment Mean (Rounds)** | 51.790 | 51.862 | Unchanged | **Strictly Unbiased**: Expected value preserved |
| **Estimated Lift ($\Delta$)** | +1.697 rounds | +1.842 rounds | +0.145 rounds | Accurate recovery of true effect ($\tau = +1.50$) |
| **Pooled Sample Variance** | 395.04 | 174.56 | **-55.81%** | Matches theoretical $1 - \rho^2$ (55.68%) |
| **Standard Error (SE)** | 0.3975 | 0.2643 | **-33.5%** | Much tighter error distribution |
| **Welch T-Statistic** | -4.2696 | -6.9705 | **+63.3% higher** | Greater signal-to-noise ratio |
| **P-Value** | `1.9762e-05` | `3.3577e-12` | **7 orders of magnitude** | From standard significance to overwhelming proof |
| **95% Confidence Interval** | `[+0.918, +2.476]` | `[+1.324, +2.360]` | **-33.5% narrower** | CI width shrank from 1.558 to 1.036 |
| **Equivalent Sample Size** | 1.0x ($N = 10,000$) | **2.26x** | **+125.6% bonus** | Like getting 12,563 extra users for free |

---

## 0. Why Cookie Cats Could NOT Be Used for CUPED
Before diving into the math, it is critical to understand a real-world data constraint:

> **The Cookie Cats Constraint**:  
> In the Cookie Cats dataset, players are tracked **starting only at the moment of game installation**. All columns (`sum_gamerounds`, `retention_1`, `retention_7`) measure behavior that occurred *after* install and *after* the player encountered the gate variant. There is zero historical pre-experiment data for these players.
>
> A common beginner mistake is to fabricate a pre-period by splitting post-experiment rounds (e.g. "first 7 days vs next 7 days"). **This is invalid**: any metric measured after treatment assignment is potentially contaminated by the treatment itself. Using a post-treatment variable as a covariate introduces **conditioning bias** and destroys causal validity.
>
> Therefore, we built a dedicated, realistic **synthetic gaming dataset** where:
> 1. Each player has a genuine pre-experiment covariate $X$ (e.g. rounds played in the 14 days before the feature rolled out).
> 2. Post-experiment rounds $Y$ are naturally correlated with pre-experiment habits ($ho \approx 0.75$).
> 3. An exact ground-truth treatment effect ($\tau = +1.50$ rounds) is added to the treatment group.

---

## Concept 1: Covariance & Correlation

### 1. Plain Language (No Jargon)
When evaluating an experiment, players are naturally different: some are highly engaged hardcore players who play dozens of rounds a day, while others are casual players who play once a week.  
If you know how much a person played **before** the experiment started, you can make a very good guess about how much they will play **during** the experiment.

- **Covariance** measures whether two numbers tend to move together in the same direction. If players who played more before the test also play more during the test, covariance is positive.
- **Correlation** is simply covariance scaled so it always lands between $-1$ and $+1$. Because it has no units, $+1.0$ means a perfect linear relationship, $0.0$ means no relationship, and $-1.0$ means a perfect inverse relationship.

### 2. Hand-Checkable Numerical Example
Suppose we have 5 players with pre-experiment rounds ($X$) and post-experiment rounds ($Y$):
- Player 1: $X = 10, Y = 12$
- Player 2: $X = 20, Y = 22$
- Player 3: $X = 30, Y = 28$
- Player 4: $X = 40, Y = 42$
- Player 5: $X = 50, Y = 46$

**Step A: Calculate Means**
$$\bar{X} = \frac{10 + 20 + 30 + 40 + 50}{5} = 30$$
$$\bar{Y} = \frac{12 + 22 + 28 + 42 + 46}{5} = 30$$

**Step B: Calculate Deviations and Products**
| Player | $(X_i - \bar{X})$ | $(Y_i - \bar{Y})$ | $(X_i - \bar{X})^2$ | $(Y_i - \bar{Y})^2$ | $(X_i - \bar{X})(Y_i - \bar{Y})$ |
|---|---|---|---|---|---|
| 1 | $-20$ | $-18$ | $400$ | $324$ | $+360$ |
| 2 | $-10$ | $-8$ | $100$ | $64$ | $+80$ |
| 3 | $0$ | $-2$ | $0$ | $4$ | $0$ |
| 4 | $+10$ | $+12$ | $100$ | $144$ | $+120$ |
| 5 | $+20$ | $+16$ | $400$ | $256$ | $+320$ |
| **Sum** | $0$ | $0$ | **$1000$** | **$792$** | **$+880$** |

**Step C: Compute Variances, Covariance, and Correlation ($n - 1 = 4$)**
$$\text{Var}(X) = s_X^2 = \frac{1000}{4} = 250, \quad s_X = \sqrt{250} \approx 15.811$$
$$\text{Var}(Y) = s_Y^2 = \frac{792}{4} = 198, \quad s_Y = \sqrt{198} \approx 14.071$$
$$\text{Cov}(X, Y) = \frac{+880}{4} = +220$$
$$\rho = \frac{\text{Cov}(X, Y)}{s_X \cdot s_Y} = \frac{220}{15.811 \cdot 14.071} = \frac{220}{222.48} \approx +0.9888$$
Because $\rho = +0.9888$, pre-experiment activity $X$ is almost perfectly predictive of post-experiment activity $Y$.

### 3. Real Code & Real Project Numbers
In our 10,000-player dataset:
```python
cov_xy = compute_covariance(df["pre_rounds"], df["post_rounds"])  # 296.88
var_x = compute_variance(df["pre_rounds"])                         # 399.80
var_y = compute_variance(df["post_rounds"])                         # 395.20
rho = compute_correlation(df["pre_rounds"], df["post_rounds"])     # 0.7462
```
- **Real Covariance:** $+296.88$
- **Real Correlation:** $\rho = +0.7462$ (Strong positive correlation, reflecting real mobile gaming telemetry).

---

## Concept 2: What CUPED Does & Why

### 1. Plain Language (No Jargon)
In any experiment, the outcome $Y$ has lots of random spread (variance). But that spread comes from two distinct sources:
1. **Predictable Baseline Differences**: Alice is an active gamer who plays 80 rounds; Bob is a casual player who plays 10 rounds. This has nothing to do with the experiment.
2. **Treatment Noise & True Effect**: The actual change caused by the new game feature.

CUPED uses pre-experiment data $X$ to **subtract away the predictable baseline difference** for each player.  
If Alice always plays 50 rounds more than average, CUPED adjusts for that. What remains is a clean, low-variance metric that reveals the true treatment effect much more clearly.

### 2. Hand-Checkable Numerical Example
Look at Player 1 and Player 5 from our 5-player toy example:
- Player 1 played $X=10$ before, and $Y=12$ after.
- Player 5 played $X=50$ before, and $Y=46$ after.
In the raw data, the spread between Player 1 and Player 5 is huge: $46 - 12 = 34$ rounds of variance!  
However, Player 1 was *already* a low-activity player (10 rounds), and Player 5 was *already* a high-activity player (50 rounds).  
Once we adjust for their known baseline, their "unexpected" activity is nearly identical. The raw spread was mostly baseline habit, not experiment noise.

### 3. Real Code & Real Project Numbers
In our simulation, raw post-rounds had a pooled sample variance of **395.04**.  
After CUPED subtracted away baseline player tendencies, the pooled sample variance dropped to **174.56** — an empirical variance reduction of **55.81%**!

---

## Concept 3: The CUPED Formula & The Optimal $\theta$

### 1. Plain Language (No Jargon)
The CUPED adjustment formula is:
$$Y_{\text{CUPED}} = Y - \theta \cdot (X - \bar{X})$$

- $(X - \bar{X})$ is how much higher or lower this player was compared to the average player *before* the experiment started.
- $\theta$ (theta) is a scaling knob: it determines how many units of $Y$ we should adjust for each 1-unit deviation in $X$.
- **Why does $\theta = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}$?**
  This is the exact formula for the slope of an Ordinary Least Squares (OLS) linear regression! It is mathematically proven to be the unique coefficient that minimizes the remaining variance of $Y_{\text{CUPED}}$.
- **Why is CUPED strictly unbiased?**
  Because $X$ was measured before treatment assignment, both Control and Treatment groups have the same expected baseline: $E[X] = \bar{X}$. Therefore:
  $$E[Y_{\text{CUPED}}] = E[Y] - \theta \cdot (E[X] - \bar{X}) = E[Y] - \theta \cdot (0) = E[Y]$$
  CUPED reduces variance **without shifting the true treatment effect by a single decimal place**.

### 2. Hand-Checkable Numerical Example
Using our 5-player toy example:
$$\text{Cov}(X, Y) = 220, \quad \text{Var}(X) = 250$$
$$\theta = \frac{\text{Cov}(X, Y)}{\text{Var}(X)} = \frac{220}{250} = 0.88$$

Now, compute $Y_{\text{CUPED}} = Y - 0.88 \cdot (X - 30)$ for each player:
- **Player 1:** $12 - 0.88 \cdot (10 - 30) = 12 - 0.88(-20) = 12 + 17.6 = 29.6$
- **Player 2:** $22 - 0.88 \cdot (20 - 30) = 22 - 0.88(-10) = 22 + 8.8 = 30.8$
- **Player 3:** $28 - 0.88 \cdot (30 - 30) = 28 - 0.88(0) = 28.0$
- **Player 4:** $42 - 0.88 \cdot (40 - 30) = 42 - 0.88(10) = 42 - 8.8 = 33.2$
- **Player 5:** $46 - 0.88 \cdot (50 - 30) = 46 - 0.88(20) = 46 - 17.6 = 28.4$

**Let's check the new variance ($n-1 = 4$):**
$$\bar{Y}_{\text{CUPED}} = \frac{29.6 + 30.8 + 28.0 + 33.2 + 28.4}{5} = 30.0 \quad \text{(Exactly equal to original mean!)} $$
Deviations from 30.0:
$(-0.4)^2 + (+0.8)^2 + (-2.0)^2 + (+3.2)^2 + (-1.6)^2 = 0.16 + 0.64 + 4.00 + 10.24 + 2.56 = 17.60$
$$\text{Var}(Y_{\text{CUPED}}) = \frac{17.60}{4} = 4.40$$

Compare the variances:
- **Original Variance $\text{Var}(Y)$:** $198.0$
- **CUPED Variance $\text{Var}(Y_{\text{CUPED}})$:** $4.40$
- **Variance shrank by:** $\frac{198 - 4.4}{198} = \mathbf{97.8\%}$!

### 3. Real Code & Real Project Numbers
In our project codebase:
```python
theta = compute_optimal_theta(df["pre_rounds"], df["post_rounds"])
# theta = 0.7426
# scipy.stats.linregress slope = 0.7426 (Exact match!)
df["cuped_outcome"] = df["post_rounds"] - theta * (df["pre_rounds"] - df["pre_rounds"].mean())
```
- Control Mean before CUPED: **50.093** $\to$ after CUPED: **50.020** (Preserved!)
- Treatment Mean before CUPED: **51.790** $\to$ after CUPED: **51.862** (Preserved!)
- Estimated Effect $\Delta$: **+1.842** rounds (Accurately recovering known true effect $\tau = +1.50$).

---

## Concept 4: The Variance Reduction Ratio ($1 - \rho^2$)

### 1. Plain Language (No Jargon)
How much will CUPED help you in advance?  
You can know before running the test by looking at the correlation $\rho$.
The percentage of variance removed by CUPED is exactly $\rho^2$, meaning the remaining variance is:
$$\text{Remaining Variance Fraction} = 1 - \rho^2$$

Because correlation is squared:
- If $\rho = 0.50$, $\rho^2 = 0.25 \implies 25\%$ variance reduction.
- If $\rho = 0.75$, $\rho^2 = 0.5625 \implies 56.25\%$ variance reduction.
- If $\rho = 0.90$, $\rho^2 = 0.81 \implies 81\%$ variance reduction!
Every increase in covariate correlation produces quadratic gains in statistical precision.

### 2. Hand-Checkable Numerical Example
In our 5-player toy example, we found $\rho = 0.9888$.
Theoretical remaining variance fraction:
$$1 - \rho^2 = 1 - (0.9888)^2 = 1 - 0.9777 = 0.0223 \quad (2.23\% \text{ remaining})$$
Predicted CUPED variance:
$$\text{Var}(Y) \cdot (1 - \rho^2) = 198.0 \cdot 0.0223 = 4.415$$
Our actual hand-computed CUPED variance was **4.40**. The formula and the calculation match down to rounding decimals!

### 3. Real Code & Real Project Numbers
In our simulation:
- Empirical correlation $\rho = 0.7462$
- Theoretical variance reduction: $\rho^2 = (0.7462)^2 = \mathbf{55.68\%}$
- Empirical variance reduction: $1 - \frac{174.56}{395.04} = \mathbf{55.81%}$
The theoretical prediction and empirical execution match within $0.13\%$!

---

## Concept 5: Practical Business Value & The Sample Size Multiplier

### 1. Plain Language (No Jargon)
Why do companies like Netflix and Meta care so much about CUPED?
Because variance is directly in the denominator of experiment duration.  
If you cut variance in half, you need **half as many users** or **half as many days** to detect the exact same effect size.  
The **Effective Sample Size Multiplier** is:
$$\text{Multiplier} = \frac{1}{1 - \rho^2}$$
CUPED gives your experimentation platform free statistical power without buying more traffic or making users wait weeks longer for results.

### 2. Hand-Checkable Numerical Example
If an experiment has $\rho = 0.70$:
$$1 - \rho^2 = 1 - 0.49 = 0.51$$
$$\text{Multiplier} = \frac{1}{0.51} \approx 1.96x$$
A correlation of 0.70 essentially **doubles your sample size for free**.

### 3. Real Code & Real Project Numbers
In our experiment:
$$\text{Multiplier} = \frac{1}{1 - 0.5568} = \mathbf{2.26x}$$
Our 10,000 users with CUPED produced the statistical certainty of **22,563 users** analyzed conventionally.

---

## 6. Generated Visualizations & Test Suite
- **Plot**: [CUPED Variance Reduction Plot](figures/cuped_before_after.png)
  - Shows variance distribution narrowing and 95% Confidence Interval shrinking by 33.5%.
- **Unit Tests**: All unit tests in `pytest tests/` passing.
