# Process-valued donation under valid e-processes: exploratory SupFDR experiment

## Status and claim ceiling

This is the first statistical stress test attached to the process-valued compound donation work in #9894. It evaluates the process-level accounting proposal under constructed, conditionally valid coordinate likelihood-ratio martingales and a running-maximum adjuster.

The results are **exploratory only**:
- 500 simulation streams per statistical cell;
- 32 hypotheses and 80 looks;
- no simultaneous confidence correction;
- no Rust execution;
- the exact committed runner was not executed from a connected checkout. The reported local run used a separate NumPy reimplementation aligned to the checked-in runner's formulas, scenario definitions, seeds and routing logic.

No theorem, confirmatory FDR/SupFDR result, production-performance result, or SOTA claim is established by this document.

## 1. Valid-null evidence generator

For each coordinate, the null observation probability is \(p_0=1/2\). The likelihood-ratio e-factor is registered against \(p_1=0.7\):

\[
L(y)=\begin{cases}
0.7/0.5=1.4,&y=1,\\
0.3/0.5=0.6,&y=0.
\end{cases}
\]

For any true-null coordinate, if the current observation is conditionally Bernoulli\((1/2)\) given the entire global past,

\[
\mathbb E[L(Y_{i,t})\mid\mathcal F_{t-1}]
=\tfrac12(1.4)+\tfrac12(0.6)=1.
\]

Thus the product \(M_{i,t}=\prod_{s\le t}L(Y_{i,s})\), with \(M_{i,t}=1\) while dormant, is a nonnegative test martingale for each true null in the declared common global filtration.

Three dependence constructions were used:

1. **Independent:** fresh independent uniforms for each coordinate and time.
2. **Shared-uniform common shock:** one fresh uniform \(U_t\) per stream and time, with \(Y_{i,t}=1\{U_t\le p_{i,t}\}\). Under every null, its conditional success probability given the global past remains exactly \(1/2\), although null coordinates are strongly dependent.
3. **Balanced negative:** each round chooses a uniformly random half of the true-null coordinates to succeed (for odd null counts, floor/ceiling half are chosen with equal probability). Each null coordinate has marginal conditional success probability \(1/2\), while null outcomes have negative dependence.

Alternatives use scenario-specific \(p_{i,t}\), including weak, strong, far-cluster and switching signals. Arrivals are deterministic and ordered by immutable candidate ID; these are predictable, but this first run does not yet cover random outcome-dependent arrivals.

## 2. Time-uniform adjustment

We do not feed raw running maxima into e-BH. The coordinate e-process is adjusted using

\[
A(x)=\sqrt{x}-1,\qquad x\ge1,
\]

applied to the running maximum of the LR martingale:

\[
X_{i,t}=\sqrt{\max_{s\le t}M_{i,s}}-1.
\]

The adjuster has integral budget

\[
\int_1^\infty\frac{\sqrt{x}-1}{x^2}\,dx=1.
\]

It therefore gives a nonnegative, pathwise nondecreasing e-process under the stated null assumptions, with \(\mathbb E[X_{i,t}]\le1\). This is the A2-style construction used to obtain a time-uniform e-process from a running maximum. See Tavyrikov, Goeman & de Heide (2026), [Carefree multiple testing with e-processes](https://doi.org/10.1214/26-EJS2546).

## 3. Donation reserve and compound score

Candidate weights are fixed globally:

\[
\gamma_i=\frac{1}{(i+1)(i+2)},\qquad i=0,\ldots,31,
\]

so the first 32 weights sum to \(32/33<1\).

For each donor, the cumulative capped-overshoot reserve is

\[
D_{i,t}=\min\{(\gamma_iX_{i,t}-1)_+,\gamma_i\}.
\]

Only \(\Delta D_{i,t}\) is newly available at look t. It satisfies

\[
0\le D_{i,t}\le\gamma_i\min(X_{i,t},1),\qquad
0\le\Delta D_{i,t}\le\gamma_i\Delta X_{i,t}.
\]

All policies allocate only this increment. Incoming capital is never used to enlarge a donor's reserve.

The weighted compound score is

\[
Z_{i,t}=\gamma_iX_{i,t}+I_{i,t}-O_{i,t},
\]

where \(I\) and \(O\) are cumulative credited and source-reserved capital. Every route is forward-only. For every prefix m, the simulation checks

\[
\sum_{i\le m}(I_{i,t}-O_{i,t})\le0.
\]

It also checks the donor cap, own-increment cap, nonnegative and nondecreasing scores, share mass, and exact-in-model escrow conservation. These checks are numerical simulation checks, not the independent integer verifier in PR #9919.

## 4. Rejection rule and estimands

At every look, weighted e-BH is applied to the compound weighted scores \(Z_{i,t}\). For an active rejection set \(R_t\), every rejected coordinate satisfies

\[
i\in R_t\quad\Longrightarrow\quad
Z_{i,t}\ge\frac{1}{\alpha|R_t|}.
\]

The simulation records:

- \(\mathrm{SupFDR}=\mathbb E[\max_t\mathrm{FDP}_t]\);
- terminal FDR;
- \(P(\text{any false discovery})\);
- terminal power (fraction of nonnull hypotheses rejected);
- reserve/credit/escrow volumes.

These are different estimands. In particular, \(P(\text{any false})\) is not SupFDR when alternatives are present.

## 5. Policies compared

1. No donation.
2. Fixed next recipient.
3. Fixed telescoping distance prior \(\rho_d=1/[d(d+1)]\), retaining mass beyond the finite horizon in an outside-horizon escrow sink.
4. Fixed geometric prior \(\rho_d=2^{-d}\), with its exact tail in the same way.
5. Prior-weighted Hedge, whose round-t shares use only recipient adjusted-evidence increments observed through round \(t-1\).
6. Fixed-share Hedge, mixing the posterior allocation back toward the original prior.
7. Raw running-maximum e-BH as an explicitly invalid diagnostic, not a valid competitor.

The two adaptive allocation algorithms are predictable for each fixed online stream. They can use higher-ID recipient history and are therefore not strongly horizon-projective. This experiment does not claim projectivity for them.

## 6. Executed exploratory results

Configuration: 500 streams per cell, 32 hypotheses, 80 looks, \(\alpha=0.10\), seed 10033, three dependence modes and six signal/arrival scenarios. The one-sided 95% Hoeffding half-width for a bounded mean at n=500 is approximately 5.47 percentage points. There is no simultaneous correction across the grid.

### All-null calibration screen

Observed SupFDR and \(P(\text{any false})\) were zero for every adjusted method in all three dependence modes in this 500-stream sample. The raw running-maximum diagnostic observed:

| Dependence | Raw-supremum SupFDR |
|---|---:|
| Independent | 6.6% |
| Shared-uniform | 5.4% |
| Balanced-negative | 8.6% |

These point estimates are below the nominal 10% in this finite run. They do not validate the raw-supremum method, nor do they demonstrate a violation in this particular all-null experiment. The known theoretical warning is that e-BH on raw running suprema need not control FDR-sup under arbitrary dependence; an experiment that fails to exhibit a violation is not a rebuttal.

### Purpose-built donation bridge

The bridge scenario has 20 alternatives: a strong donor at ID 0, one borderline immediate recipient at ID 1, and 18 strong alternatives at later IDs. It has 12 nulls. The target is chosen so that the immediate recipient's base evidence is close to, but below, the weighted e-BH self-consistency boundary while the remaining strong discoveries lower that boundary.

Terminal power:

| Dependence | No donation | Fixed next | Telescope | Geometric | Prior Hedge | Fixed-share |
|---|---:|---:|---:|---:|---:|---:|
| Independent | 91.85% | 94.24% | 92.98% | 92.90% | 92.07% | 92.13% |
| Shared-uniform | 89.72% | 91.60% | 90.40% | 90.39% | 89.92% | 90.02% |
| Balanced-negative | 91.59% | 93.89% | 92.48% | 92.42% | 91.82% | 91.92% |

No adjusted method recorded false discoveries in these three 500-stream cells. The fixed-next policy gains 1.88–2.39 percentage points of terminal power in this deliberately constructed setting. The distance priors dilute capital across future recipients and obtain smaller gains; the adaptive policies do not learn to concentrate on the borderline immediate recipient quickly enough to beat fixed-next here.

This is a hypothesis-generating result, not a universal ranking. The scenario was constructed to favor the immediate-next route, and paired power-difference confidence intervals were not computed.

### Far-recipient mismatch

A separate scenario has one strong donor and 10 alternatives far into the candidate order. None of the adjusted donation policies improves terminal power over no donation in this configuration. This is important: a large reserve does not help if the allocation policy sends it to recipients who are not the hypotheses near the rejection boundary.

The raw running-supremum diagnostic has materially higher \(P(\text{any false})\) in this setting (14.4–23.6%) while the observed SupFDR point estimates are 3.35–5.93%. The difference between those metrics reinforces why “any false” and SupFDR must not be interchanged.

Other weak, switching and delayed-arrival scenarios showed only small power differences (generally tenths of a percentage point). They do not support superiority claims.

## 7. Interpretation and next experiments

The experiment supports three directions for the next round of work:

1. **Recipient-boundary-aware allocation:** the useful donor is not necessarily the one with the largest evidence, nor is the recipient necessarily the closest. A policy should measure how much new reserve is needed for a candidate to cross the current self-consistency boundary, while using only predictable information.
2. **Power-versus-projectivity frontier:** fixed projective priors preserve horizon coherence but dilute capital; predictable adaptive policies can focus capital on a borderline recipient but may be non-projective. Benchmark both as separate product modes.
3. **Proof-carrying process tests:** store evidence-process specification, filtration/arrival receipt, adjuster identity, weight-sequence digest, reserve policy and credit lineage with each rejection certificate. The ledger alone cannot authenticate its upstream e-process inputs.

For confirmatory work, use the preregistered #9714 matrix and at least 50,000 paired streams per statistical cell, with simultaneous uncertainty bounds. Include random predictable arrivals, more varied valid LR mixtures, effect sizes, stopping schedules, high dimensions and out-of-distribution utility regimes. The strongest published one-shot comparator remains donation e-LOND from Xu, Fischer & Ramdas (UAI 2026), but its one-e-value-per-arrival contract is not the same as these evolving coordinate processes. [UAI 2026 paper](https://proceedings.mlr.press/v337/xu26a.html).

## Claim boundary

This is a simulation of one valid e-process family, an A2-adjusted running maximum, and the process-valued source-only ledger policy. It does not prove the compound-null-mass theorem, validate real-world source timestamps or filtration completeness, implement the Rust writer, or beat donation e-LOND / dynamic e-closure. The observed power gain is specific to the bridge construction and is not yet statistically resolved.
