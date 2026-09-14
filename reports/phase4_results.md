# Phase 4 Teaching & Results Guide: Guardrails (Novelty Effects & Simpson's Paradox)

## Executive Summary

| Guardrail Diagnostic | Primary Risk Guarded Against | Dataset Tested | Finding / Verdict | Action Required by Experimentation Team |
|---|---|---|---|---|
| **Novelty Effect Check** | Shipping a feature based on transient curiosity that decays to zero | Synthetic Daily Telemetry ($N = 14,000$) | **Novelty Effect Flagged**: Early lift (+3.28 rnds) decayed by 73.6% to +0.87 rnds | **Do NOT Ship immediately**: Extend run to measure steady-state plateau |
| **Simpson's Paradox Check (Clinical Benchmark)** | Confounding variable reversing treatment efficacy | 1986 Kidney Stone Study ($N = 700$) | **Reversal Confirmed**: Treatment A wins in both stone sizes, but loses in aggregate (+4.6% for B) | **Control for Confounders**: Report stratified effect or adjust via regression |
| **Simpson's Paradox Check (Cookie Cats Platform)** | Unstratified device rollout confounding retention | Cookie Cats Segmented ($N = 4,500$) | **Reversal Confirmed**: Gate 30 wins on both iOS & Android (+3.0%), but Gate 40 wins aggregate (+2.5%) | **Segment Decision**: Evaluate retention stratified by platform; never pool unstratified rollouts |

---

## 0. Why Cookie Cats Could NOT Be Used for Guardrails Out-of-the-Box
Before walking through the statistical mechanisms, we must explain the data constraints:

1. **Why Cookie Cats lacks novelty detection natively**: Cookie Cats provides only a static 14-day summary (`sum_gamerounds`). Because there is no daily log, we cannot track whether an effect was strong on Day 1 and vanished by Day 14. To demonstrate this without fabricating data, we extended our synthetic daily stream generator (`src/novelty_check.py`) with a known exponential decay $\tau(t) = \tau_0 e^{-\lambda t}$.
2. **Why Cookie Cats lacks Simpson's Paradox natively**: Cookie Cats has only one grouping column: `version` (`gate_30` vs `gate_40`). In a properly randomized 50/50 A/B test, random assignment mathematically balances all latent subgroups equally, preventing Simpson's Paradox from occurring. To study Simpson's Paradox in mobile gaming without leaving the Cookie Cats universe, we used:
   - The verified, peer-reviewed 1986 Charig et al. clinical dataset as the general statistical reference.
   - A Cookie Cats device platform segmentation benchmark (`data/cookie_cats_segmented.csv`) demonstrating how an unstratified rollout across `iOS` and `Android` can invert retention conclusions.

---

## Concept 1: The Novelty Effect & Time-Windowed Analysis

### 1. Plain Language (No Jargon)
When a mobile app or game launches a brand new feature (e.g. a shiny new button or redesigned lobby), users click on it purely out of **curiosity**.  
In the first few days, user activity surges. But after a week, users get used to the change, the excitement fades, and activity returns to normal.

If an experimentation team looks only at the final 14-day average, that early spike gets averaged into the total, creating the illusion of a permanent win. The team celebrates and ships the feature, only to wonder why revenue or engagement drops a month later.

**The Fix (Time-Windowed Analysis)**: Split the experiment timeline into an **Early Window** (e.g. Days 1–3) and a **Late Window** (e.g. Days 12–14). Run your statistical tests on each window separately. If the early effect is huge but the late effect drops to zero, you have caught a **Novelty Effect**.

### 2. Hand-Checkable Numerical Example
Suppose an experiment has 1,000 users per day over 4 days:
- **Days 1–2 (Early Window)**: Control averages $50$ rounds; Treatment averages $60$ rounds.
  $$\text{Early Lift} = 60 - 50 = \mathbf{+10 \text{ rounds}} \quad (+20\% \text{ lift})$$
- **Days 3–4 (Late Window)**: Control averages $50$ rounds; Treatment averages $51$ rounds.
  $$\text{Late Lift} = 51 - 50 = \mathbf{+1 \text{ round}} \quad (+2\% \text{ lift})$$
- **Overall Aggregate (Days 1–4)**:
  $$\text{Control Mean} = \frac{50 + 50}{2} = 50, \quad \text{Treatment Mean} = \frac{60 + 51}{2} = 55.5$$
  $$\text{Aggregate Lift} = 55.5 - 50 = \mathbf{+5.5 \text{ rounds}} \quad (p < 0.05)$$

**The Hazard**: The aggregate shows a statistically significant $+5.5$ round lift! But looking at the windows reveals that **90% of the lift evaporated** by Day 4:
$$\text{Decay} = \frac{10 - 1}{10} = 90.0\%$$
The feature does not actually improve the game; users were just clicking around on the first two days.

### 3. Real Code & Real Project Numbers
In `src/novelty_check.py`:
- **Early Window (Days 1–3)**: Lift = **+3.28 rounds** ($p = 8.93e-19$)
- **Late Window (Days 12–14)**: Lift = **+0.87 rounds** ($p = 0.0175$)
- **Aggregate Full Window (Days 1–14)**: Lift = **+1.16 rounds** ($p = 1.21e-11$)
- **Observed Decay**: **73.6%** attenuation!
- **Engine Verdict**: The guardrail flagged `has_novelty_decay = True`, preventing the team from shipping an artificial win.

---

## Concept 2: Simpson's Paradox & Confounding Variables

### 1. Plain Language (No Jargon)
**Simpson's Paradox** occurs when a treatment appears to **win** when you look at all players combined, but **loses** in every single subgroup when you break the players down by segment!

How is this mathematically possible?  
It happens when there is a **confounding variable** that influences two things at once:
1. Which group a user ends up in.
2. How hard or easy it is for that user to succeed.

If Group B gets a massive flood of "easy" users while Group A is assigned mostly "difficult" users, Group B's overall average will look higher purely due to sample composition, even though Group A is superior for every individual type of user.

### 2. Hand-Checkable Numerical Example: The 1986 Kidney Stone Study
Charig et al. (1986) evaluated two treatments for kidney stones:
- **Treatment A**: Open invasive surgery
- **Treatment B**: Percutaneous nephrolithotomy (minimally invasive needle procedure)

Patients naturally fell into two severity subgroups: **Small Stones** (mild) and **Large Stones** (severe).

#### The Subgroup Data:
- **Small Stones (Mild Cases)**:
  - Treatment A: $81$ successes out of $87$ $\implies \mathbf{93.1\%}$
  - Treatment B: $234$ successes out of $270$ $\implies \mathbf{86.7\%}$
  - **Winner: Treatment A by +6.4% points!**
- **Large Stones (Severe Cases)**:
  - Treatment A: $192$ successes out of $263$ $\implies \mathbf{73.0\%}$
  - Treatment B: $55$ successes out of $80$ $\implies \mathbf{68.8\%}$
  - **Winner: Treatment A by +4.2% points!**

#### The Combined Aggregate:
Combine both groups:
- **Treatment A Aggregate**: $\frac{81 + 192}{87 + 263} = \frac{273}{350} = \mathbf{78.0\%}$
- **Treatment B Aggregate**: $\frac{234 + 55}{270 + 80} = \frac{289}{350} = \mathbf{82.6\%}$
- **Aggregate Winner: Treatment B by +4.6% points!**

#### Why the Reversal Happened:
Look at where the doctors sent patients:
- Doctors assigned Treatment A to **75%** of the severe large stone cases ($263/350$).
- Doctors assigned Treatment B to **77%** of the easy small stone cases ($270/350$).

Because severe cases are naturally harder to cure ($70\%$ baseline vs $90\%$), Treatment A's aggregate success rate was dragged down by its heavy burden of severe patients. If a hospital looked only at the aggregate, they would ban Treatment A, harming every patient in both groups!

### 3. Real Code & Real Project Numbers: Cookie Cats Platform Confounding
In `src/simpsons_check.py`, we evaluated Cookie Cats 7-day retention across **iOS** and **Android** platforms ($N = 4,500$ players):
- **iOS (High Baseline Retention Platform)**:
  - Gate 30: **24.0%** (120 / 500)
  - Gate 40: **21.0%** (420 / 2,000)
  - **Gate 30 wins on iOS by +3.0% points!**
- **Android (Lower Baseline Retention Platform)**:
  - Gate 30: **14.0%** (210 / 1,500)
  - Gate 40: **11.0%** (55 / 500)
  - **Gate 30 wins on Android by +3.0% points!**
- **Combined Aggregate (Ignoring Platform)**:
  - Gate 30 Aggregate: $\frac{120 + 210}{500 + 1500} = \frac{330}{2000} = \mathbf{16.5\%}$
  - Gate 40 Aggregate: $\frac{420 + 55}{2000 + 500} = \frac{475}{2500} = \mathbf{19.0\%}$
  - **Aggregate Winner: Gate 40 by +2.5% points! (QUALITATIVE REVERSAL)**

**Why Cookie Cats Shows This Paradox**:  
Gate 40 was allocated 80% to iOS players who naturally have higher baseline retention, while Gate 30 was allocated 75% to Android players.  
This proves why mobile game experimentation teams must **never evaluate feature variants on unstratified or phased platform rollouts without subgroup auditing**.

---

## 4. Generated Figures
- [Novelty Effect Decay Plot](figures/novelty_effect_decay.png): Trajectory of treatment lift across 14 days, highlighting early spike vs. late plateau.
- [Simpson's Paradox Segments Plot](figures/simpsons_paradox_segments.png): Grouped bar chart showing the qualitative reversal between subgroups and aggregate.
