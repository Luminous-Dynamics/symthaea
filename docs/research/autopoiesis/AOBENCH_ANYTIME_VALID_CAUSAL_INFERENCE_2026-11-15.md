# AOBench — Anytime-Valid Causal Effect Inference Under Adaptive Stopping

**Status:** Initial bounded-outcome Rust evaluator committed; exact-head qualification pending  
**Date:** 2026-11-15  
**Tracking:** [AOBENCH-017 #12242](https://github.com/Luminous-Dynamics/symthaea/issues/12242)

## 1. Why this track exists

AOBENCH-011 evaluates which experiment to run and when to stop. AOBENCH-012/013 evaluate model adequacy and effect bounds. AOBENCH-017 adds a separate requirement: an effect interval must remain statistically valid at every sample size at which an agent could have stopped.

A conventional fixed-sample confidence interval can be invalidated by repeatedly checking results and stopping when it looks favorable. A confidence sequence instead targets time-uniform coverage:

\[
\Pr\left(\forall t\geq 1:\ \theta\in C_t\right)\geq 1-\alpha.
\]

This is stronger than checking whether a single interval happens to cover the truth at the final stopping sample.

## 2. Research landscape

- **Howard, Ramdas, McAuliffe & Sekhon (2021),** *Time-Uniform, Nonparametric, Nonasymptotic Confidence Sequences*, Annals of Statistics 49(2):1055–1080. Develops nonasymptotic time-uniform intervals from nonnegative supermartingales and related concentration methods. [Paper](https://arxiv.org/abs/1810.08240). <Cite refs={["turn525037academia38"]}/>
- **Lindon, Ham, Tingley & Bojinov (2026),** *Anytime-Valid Inference in Linear Models with Applications to Regression-Adjusted Causal Inference*, JASA, DOI 10.1080/01621459.2026.2692052. Provides sequential tests and confidence sets for linear models and heteroscedasticity-robust asymptotic procedures for randomized causal estimands. Reproducibility code and data are public. [Paper](https://doi.org/10.1080/01621459.2026.2692052) · [Code/data](https://github.com/michaellindon/avlm_paper). <Cite refs={["turn525037search0","turn525037search4"]}/>
- **Molitor & Gold (2025),** *Anytime-Valid Inference in Adaptive Experiments: Covariate Adjustment and Balanced Power*. MADCovar and MADMod combine anytime-valid inference with adaptive allocation and power balancing; their paper reports up to 60% narrower confidence sequences than its MAD baseline in the evaluated simulations. [Paper](https://arxiv.org/abs/2506.20523). <Cite refs={["turn525037academia36"]}/>
- **Lindon & Kallus (2026),** *Design-Based Anytime-Valid Inference for Randomized Experiments with Delayed Outcomes and Staggered Entry*, ICML/PMLR 306. Extends anytime-valid randomized inference to delayed outcomes, staggered entry and calendar-time cumulative reward. [Paper](https://proceedings.mlr.press/v306/lindon26a.html). <Cite refs={["turn525037search3"]}/>
- **Lindon (2026),** *Anytime-Valid Inference in Linear Models with Clustered Outcomes*. Develops exact multivariate Gaussian confidence sequences and asymptotic cluster-robust sequences. [Paper](https://arxiv.org/abs/2610.25613). <Cite refs={["turn525037academia37"]}/>

These are not interchangeable methods. Some are exact under parametric models; some are asymptotic/robust; some are design-based for specific randomization and outcome processes. AOBench must compare them only where their assumptions and estimands match.

## 3. Initial v0 contract

The first Rust implementation is deliberately narrow and auditable:

- A registered family of Q causal queries.
- Each query has bounded outcomes Y in [L,U] and a fixed treatment-minus-control mean effect.
- A two-arm randomized experiment with predictable allocation.
- One ordered stream of outcomes, each bound to an immutable evidence receipt.
- The agent selects a global stopping prefix and declares one effect interval / decision per query.
- Evaluator truth and evidence adjudications remain in a separate oracle.

### 3.1 Hoeffding alpha-spending confidence sequence

For each query and each arm, allocate:

\[
\alpha_{\mathrm{arm}}=\frac{\alpha_{\mathrm{FWER}}}{2Q}
\]

At arm sample size n, allocate:

\[
\alpha_n=\frac{\alpha_{\mathrm{arm}}}{n(n+1)}.
\]

Since \(\sum_{n=1}^{\infty}1/[n(n+1)]=1\), the total error allocation for one arm across all sample sizes is at most alpha_arm.

For observed arm mean \(\bar Y_n\) and support width R=U−L, use:

\[
r_n=R\sqrt{\frac{\log(2/\alpha_n)}{2n}}.
\]

The arm interval is the sample mean plus/minus r_n, clipped to [L,U]. At n=0, the interval remains the full support [L,U].

The treatment-effect interval is:

\[
C_n^{\tau}=
\left[
L_{1,n}-U_{0,n},
U_{1,n}-L_{0,n}
\right].
\]

A union bound over both arms, all Q queries and all n gives the registered family-wise error guarantee, under the declared bounded-outcome martingale/fixed-mean assumptions. The implementation computes log alpha in the log domain to avoid overflowing n(n+1).

This construction is a conservative baseline, not a claim of best achievable width or power.

### 3.2 Evidence gates

Each observation references an agent-visible receipt. The evaluator independently checks five dispositions:
- randomized/predictable assignment;
- intervention delivery;
- isolation;
- outcome measurement;
- fixed-mean outcome contract.

Rejected evidence blocks the claim. Unavailable evidence produces Inconclusive. Neither is silently converted into an observation or zero.

The fixed-mean gate matters: a confidence sequence for a fixed arm mean is not automatically valid if outcomes drift, interference is unmodeled, or the assignment/observation process violates the declared martingale contract.

### 3.3 Claim verification

The evaluator recomputes the confidence sequence at the exact prefix supplied by the agent. The reported interval must contain the recomputed interval. A narrower reported interval is NotQualified.

A decisive action is allowed only if the agent's own reported interval is entirely on the supporting side of the registered decision threshold. Defer is always available; unnecessary defer is reported as conservative rather than treated as a scientific failure.

The evaluator reports whether the hidden true effect is covered at the chosen stopping prefix. A miss is a possible probabilistic event at nominal alpha; it is measured across Monte Carlo worlds rather than reclassified as a deterministic code-contract failure.

## 4. Current executable module

The nested Rust package now contains src/anytime_valid.rs, exported from the crate root.

It provides:
- hoeffding_alpha_spending_radius;
- registered query and bounded-outcome contracts;
- ordered arm-specific observations with evidence receipt IDs;
- hidden per-receipt adjudication;
- family-wise alpha allocation across query/arm pairs;
- arbitrary global stopping-prefix validation;
- per-query control/treatment confidence sequences and treatment-effect interval;
- reported-interval containment checks;
- robust Act / DoNotAct / Defer validation;
- family-wise true-effect coverage reporting.

Fourteen deterministic tests cover formula validity/invalid inputs, zero-observation support, multi-query alpha allocation, a separated positive effect, false narrowing, unavailable randomization evidence, rejected delivery, rejected fixed-mean assumptions, stopping/budget violations, unsupported outcomes, missing query claims, unsafe decisions and evaluation at arbitrary prefixes.

These tests are source-authored fixtures. No compiler, runtime, Monte Carlo coverage or power result is claimed until exact-head qualification and statistical simulation complete.

## 5. Fair baselines

- Fixed-horizon classical intervals (valid only at the preregistered horizon; diagnostic comparator).
- Repeated fixed-sample p-values/intervals with optional stopping (deliberately invalid negative control).
- Always-defer / full-support interval.
- AOBench Hoeffding alpha-spending v0.
- Howard-style mixture/stitching confidence sequences.
- E-process / likelihood-ratio confidence sequences.
- Lindon et al. AVLM regression-adjusted inference when the data and model assumptions match.
- MADCovar / MADMod under matched adaptive-allocation assumptions.
- Delayed-outcome design-based confidence sequences when staggered-entry outcomes are in scope.
- Oracle interval / oracle stopping policy (upper bound only).

For comparisons, freeze the estimand, outcome support, assignment policy, covariate schema, cluster/delay structure, missingness, stopping rule, alpha, query family, sample budget and analysis version. Do not compare a model-based parametric result with a bounded-outcome distribution-free interval as if assumptions were identical.

## 6. Proposed qualification targets (not achieved)

- 100% correct deterministic algebraic and contract fixtures.
- At nominal family-wise alpha=0.05, at least 95% simultaneous empirical coverage across all registered queries and all evaluated time prefixes, with a confidence interval for the coverage estimate.
- False decisive-action rate at most 5% on sealed null and threshold-straddling controls.
- Zero statistical updates from rejected/unavailable randomization, delivery, isolation, measurement or fixed-mean evidence.
- Zero accepted reported intervals narrower than evaluator-recomputed intervals.
- Report power at preregistered minimum effect sizes, interval width and stopping-time distribution against mixture, e-process and regression-adjusted baselines.
- Preserve unresolved/low-power outcomes; do not force a decision to improve the average.

The empirical 95% coverage target is a Monte Carlo evaluation target, not a deterministic per-run promise. The mathematical v0 coverage statement is conditional on the bounded-outcome, predictable-randomization and fixed-mean contract.

## 7. Next increments

1. Monte Carlo generator for Bernoulli, bounded continuous, heteroscedastic and adaptive allocation worlds.
2. Optional stopping adversaries that choose a stop based on the current interval or p-value.
3. Family-wise coverage tests over simultaneous queries and many time points.
4. Compare against Howard mixture/stitching constructions.
5. Add delayed outcomes and staggered entry; use the design-based filtration appropriate to that estimand.
6. Add regression-adjusted AVLM and cluster-robust/clustered-outcome comparators in their applicable regimes.
7. Add mutation tests for alpha allocation, summability, evidence gates, prefix handling and interval containment.

## 8. Claim boundary

This v0 is a randomized two-arm bounded-outcome mean-effect evaluator. It is not an observational causal estimator, a general confidence sequence for arbitrary effect estimands, or a substitute for causal identification. Coverage does not validate an exclusion restriction, no-interference assumption, measurement source or physical intervention. Those remain separate identification and evidence gates.
