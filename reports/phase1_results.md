# Phase 1 Results Report: Cookie Cats A/B Testing Engine

**Dataset:** Mobile Games A/B Testing (`cookie_cats.csv`)  
**Experiment Design:** Level Gate Placement (`gate_30` Control vs. `gate_40` Treatment)  
**Total Players Analyzed:** 90,188 (outlier user 6390605 with 49,854 rounds excluded)  
**Control Group (gate_30):** 44,699 players  
**Treatment Group (gate_40):** 45,489 players  

---

## 1. Executive Summary Table

| Metric | Metric Type | Test Used | Control (`gate_30`) | Treatment (`gate_40`) | Absolute Difference | Relative Lift | P-Value | 95% Confidence Interval | Statistically Significant? ($\alpha=0.05$) | MDE at 80% Power | Power Check Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **`retention_1`** | Binary | Two-Proportion Z-Test | 44.8198% | 44.2283% | -0.5915% pts | -1.32% | `0.0739` | `[-1.2403%, +0.0572%]` | **No** ($p \ge 0.05$) | $\pm$0.9278% pts | *Underpowered for observed effect* |
| **`retention_7`** | Binary | Two-Proportion Z-Test | 19.0183% | 18.2000% | -0.8183% pts | -4.30% | `1.5918e-03` | `[-1.3263%, -0.3103%]` | **Yes** ($p < 0.05$) | $\pm$0.7322% pts | **Well-Powered** (Observed > MDE) |
| **`sum_gamerounds`** | Continuous | Welch's t-test (unequal var) | 51.342 rnds | 51.299 rnds | -0.043 rnds | -0.08% | `0.9495` | `[-1.384, +1.297]` | **No** ($p = 0.95$) | $\pm$1.916 rnds | *Effect is negligible* |

---

## 2. Metric Deep Dives & Statistical Rationale

### A. Day 1 Retention (`retention_1`)
- **Null Hypothesis ($H_0$):** $p_{gate\_30} = p_{gate\_40}$ (Day 1 retention is identical across gate placements).
- **Alternative Hypothesis ($H_1$):** $p_{gate\_30} \neq p_{gate\_40}$.
- **Findings:**
  - Day 1 retention dropped from **44.82%** in control to **44.23%** in treatment (a drop of **0.59% percentage points**, or **-1.32%** relative).
  - Test Statistic: $z = 1.7871$, $p = 0.0739$.
  - The 95% Confidence Interval is `[-1.2403%, +0.0572%]`, spanning zero.
  - **Verdict:** We fail to reject $H_0$ at the 5% significance level.
  - **Power Insight:** The 80% power Minimum Detectable Effect for this sample size was $\pm 0.93%$ points. Because the observed drop (0.59% points) is smaller than the MDE, this test was insufficiently powered to reliably detect an effect of this modest magnitude.

### B. Day 7 Retention (`retention_7`)
- **Null Hypothesis ($H_0$):** $p_{gate\_30} = p_{gate\_40}$ (Day 7 retention is identical across gate placements).
- **Alternative Hypothesis ($H_1$):** $p_{gate\_30} \neq p_{gate\_40}$.
- **Findings:**
  - Day 7 retention dropped from **19.02%** in control to **18.20%** in treatment (a drop of **0.82% percentage points**, or **-4.30%** relative).
  - Test Statistic: $z = 3.1574$, $p = 1.5918e-03$.
  - The 95% Confidence Interval is `[-1.3263%, -0.3103%]`, strictly below zero.
  - **Verdict:** Reject $H_0$ ($p < 0.01$). Moving the gate from level 30 to level 40 causes a statistically significant and damaging drop in 7-day retention.
  - **Power Insight:** The MDE at 80% power was $\pm 0.73%$ points. The observed drop (0.82% points) exceeded this threshold, confirming the experiment was well-powered.

### C. Rounds Played in 14 Days (`sum_gamerounds`)
- **Null Hypothesis ($H_0$):** $\mu_{gate\_30} = \mu_{gate\_40}$ (Average game rounds played is identical).
- **Alternative Hypothesis ($H_1$):** $\mu_{gate\_30} \neq \mu_{gate\_40}$.
- **Findings:**
  - Control played an average of **51.342** rounds ($s = 102.06$), while Treatment played **51.299** rounds ($s = 103.29$).
  - Difference: **-0.043** rounds (-0.08%).
  - Test Statistic: $t = 0.0634$, Welch Degrees of Freedom: $\nu = 90183.3$, $p = 0.9495$.
  - 95% Confidence Interval: `[-1.384, +1.297]` rounds.
  - **Verdict:** Fail to reject $H_0$. There is no detectable difference in rounds played between gate 30 and gate 40.
  - **Power Insight:** The MDE at 80% power was $\pm 1.92$ rounds. An observed difference of -0.043 rounds is 45x smaller than the detectable limit, indicating that aggregate rounds played remained virtually unchanged.

---

## 3. Product & Business Decision

> **Business Recommendation: DO NOT MOVE THE GATE TO LEVEL 40. Keep the gate at level 30.**
>
> 1. **Retention is King:** Mobile game monetization relies on long-term habituation. While Day 1 retention shows a directional decline (-1.32%), **Day 7 retention suffers a statistically significant 4.30% relative decrease**.
> 2. **Hedonic Treadmill & Churn Dynamics:** Forcing players to wait until level 40 before encountering the first progression pause prevents natural rest intervals and leads to player fatigue and churn before the first week concludes.
> 3. **Engagement Illusion:** Although overall average rounds played in the first 14 days appears flat (51.34 vs. 51.30 rounds), players who churn by day 7 cease contributing to future ad impressions, in-app purchases (IAP), and long-term Lifetime Value (LTV).

---

## 4. Verification & Statistical Validation

All custom mathematical functions were validated side-by-side against reference packages:
- `proportion_z_test` validated against `statsmodels.stats.proportion.proportions_ztest`: **PASSED (Tolerance < 1e-7)**
- `welch_t_test` validated against `scipy.stats.ttest_ind(..., equal_var=False)`: **PASSED (Tolerance < 1e-7)**
- Automated test suite: `pytest tests/` passing 6/6 unit tests.
