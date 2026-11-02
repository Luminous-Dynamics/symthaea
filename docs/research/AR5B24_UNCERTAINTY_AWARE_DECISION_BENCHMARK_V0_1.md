# AR-5B24: Uncertainty-Aware Decision Benchmark Protocol v0.1

**Status:** research protocol draft; no benchmark execution or performance result is claimed.  
**Tracking:** https://github.com/Luminous-Dynamics/symthaea/issues/9971  
**Branch:** `research/ar5b24-benchmark-protocol-v0.1-20261102`  
**Scope:** AR-5B19 policy frontier, AR-5B20 TV robustness, AR-5B21 finite-support calibration, AR-5B23 decision-functional confidence, AR-5B22 robust dynamic programming (future wave).

## 1. Research question

Does a policy-selection system that accounts for sampling uncertainty, transport uncertainty and model/specification uncertainty make better decisions per unit of evidence and compute than strong, correctly implemented statistical and robust-optimization baselines?

The benchmark is not a single accuracy/power leaderboard. A method can be statistically valid and operationally useless (for example, by always abstaining); it can be efficient and invalid; or it can perform well only because the benchmark's declared model family omits the shift that breaks it.

The primary comparison is therefore a **validity–decision-quality–cost frontier**, accompanied by explicit assumptions and per-regime results.

## 2. Separate the three uncertainty problems

Do not compress these into one radius or one boolean called robust.

| Layer | Question | Evidence/method | What it does not establish |
|---|---|---|---|
| Sampling uncertainty | How uncertain are estimated expectations under the calibration law? | Finite-sample confidence bounds/sets | Equality with the confirmatory law |
| Transport uncertainty | Does the calibration law describe the confirmation/deployment target? | Same-target-law protocol or separately justified transport bound and monitoring | Omitted mechanisms not represented by the transport model |
| Specification uncertainty | Is the candidate model/null/mechanism family incomplete? | Held-out mechanism families, model criticism, adversarial discovery, abstention | Universal completeness of the model family |

A SHA-256 digest binds exact bytes; it does not prove data provenance, sampling independence, stationarity, exchangeability, or target-law equality.

### Calibration-to-confirmation contract

Every run must declare one of:

- `SAME_TARGET_LAW_BY_DESIGN`: calibration and confirmatory samples are intended to follow the same law under the same target population, eligibility/exposure rules, assignment mechanism, policy implementation, outcome definitions and environment version. This remains a design assumption, not a fact proven by a digest.
- `BOUNDED_TRANSPORT`: a separately supported discrepancy bound maps the calibration law to the target law. The discrepancy, evidence and failure budget are explicit and composed into the decision bounds.
- `UNKNOWN_TARGET_RELATION`: statistical qualification is `INDETERMINATE`; an independent calibration stream is not enough.

Required negative control: generate independent calibration and confirmation streams from different laws. The method must not qualify the confirmation guarantee merely because the samples are independent.

## 3. Estimands and error-budget contract

For policy π and model (m), retain the original receipt semantics:

- (s_{π,m}): null rejection probability;
- (p_{π,m}): alternative rejection probability/power;
- (c_{π,m}): expected evaluator cost;
- (u_{π,m}): expected decision utility.

Use the exact same path-level functions and decision semantics when calibrating these quantities. A nominal expectation over a declared path support must reconstruct the frozen receipt before its uncertainty bound is accepted.

For a null-family simultaneous coverage failure probability (δ_0<α), and a conditional confirmatory error budget (α_c), the required spending identity is:

[
δ_0+(1-δ_0)α_cleqα,
qquad
α_c=rac{α-δ_0}{1-δ_0}.
]

This calculation is useful only if:
1. the simultaneous null-coverage event covers the complete candidate null family and policy family;
2. the calibration-to-confirmation target relation is valid;
3. the confirmatory inference is valid conditional on the calibration result (independence is one sufficient design route);
4. any policy selected after calibration belongs to the simultaneously covered family, or a valid uniform/adaptive-selection correction is used.

Alternative-family coverage failure is reported separately; it must not be silently represented as Type-I spending.

## 4. Methods under comparison

### Policy-frontier methods

| ID | Method | Role |
|---|---|---|
| P0 | Nominal exact finite frontier (#9899) | Exact small finite-family reference |
| P1 | TV robust frontier with known/oracle radii (#9924) | Distribution-level mathematical reference |
| P2 | Weissman finite-support TV calibration (#9925) | Finite-alphabet calibrated TV baseline |
| P3 | Decision-functional scalar Hoeffding (#9951) | Support-free bounds for a frozen finite vector of bounded expectations |
| P4 | Vector-valued empirical Bernstein | Variance-adaptive joint confidence-set comparator |
| P5 | Generalized TV (GTV) | Cost-/decision-aware ambiguity geometry on matched finite-support instances |
| P6 | Exact multinomial/likelihood-ratio confidence region | Tiny-support coverage oracle/comparator |

P3 is not a full-distribution ambiguity set. It protects only the predeclared bounded functionals. A rectangular product of simultaneous coordinate intervals is an outer confidence region for the declared expectation vector, not necessarily a realizable distribution.

P4 must follow the theorem's actual conditions. Martinez-Taboada & Ramdas (2026) develop vector-valued empirical-Bernstein confidence sets in 2-smooth separable Banach spaces, including batch and stopping-time settings. Do not assume the result applies to arbitrary dependence, changing target laws or unbounded metrics.

P5 must faithfully reproduce the Fontem–Ji (2026) GTV construction on matched finite-support cost problems. Do not describe a different weighted TV ball as a reproduction of their algorithm.

### Sequential inference and acquisition methods

- fixed horizon;
- Bonferroni repeated-look control;
- Lan–DeMets group-sequential alpha spending;
- always-valid confidence sequence/e-process;
- induced anytime-valid fixed-endpoint test where assumptions apply;
- SERPANT pairwise e-process ranking and adaptive tournament only on ranking/top-k tasks;
- #8795 frozen-reference mixture likelihood-ratio drift monitor;
- random, uncertainty-only, cost-only, plug-in EVSI and robust-EVSI acquisition.

### Robust-MDP geometries (later wave)

- ((s,a))-rectangular ambiguity;
- (s)-rectangular ambiguity, coupled across actions within each state;
- globally coupled/factor nonrectangular uncertainty;
- exhaustive tiny path/LP oracle.

These geometries are not interchangeable. A Bellman result requires a sufficient Markov state and a valid ambiguity factorization. Do not claim ordinary Bellman recursion for globally coupled uncertainty without a theorem or a valid state augmentation.

## 5. Frozen campaign families

The generator must expose known evaluator truth while withholding latent parameters and counterfactual outcomes from candidate methods.

| Family | Design | Expected diagnostic |
|---|---|---|
| F00 clean IID | Frozen target law, well-specified null/alternative | Robustness tax and efficiency |
| F01 orthogonal shift | Large path-law change, declared decision functionals unchanged | TV can be conservative; DFCE should remain decision-tight |
| F02 aligned shift | Rejection, cost or utility expectation changes | Decision-functional bounds must widen |
| F03 mixture contamination | Rare contamination and multiple contamination fractions | Sensitivity to tail mass |
| F04 omitted null | Add a valid null mechanism absent from the declared family | Exposes finite-family false assurance |
| F05 support growth | Increasing categorical support and continuous/high-cardinality observations | Complete-path curse versus functional scaling |
| F06 rare events | Rare null rejections, including zero observed events | Prevent zero-count certainty |
| F07 variance regimes | Same support/range, low versus high empirical variance | Potential value of empirical Bernstein |
| F08 clusters | Clustered randomization/repeated measures | Reject IID-only calibration claims |
| F09 serial dependence | Markov/AR processes and repeated subjects | Reject invalid independent-path assumptions |
| F10 drift | Abrupt, gradual, seasonal and recurring shifts | Detection delay, transport risk and recalibration |
| F11 independent shift | Independent calibration/confirmation streams with different laws | Independence alone must not qualify |
| F12 delayed/censored | Delayed, missing and informative censoring | Preserve missingness assumptions and uncertainty |
| F13 adaptive family | Policies/functions introduced after calibration | Detect selection outside the covered family |
| F14 cost inversion | Gross information value positive, net value negative after all-in cost | Avoid selecting expensive, low-net-value learning |
| F15 learning-only | Regression, conclusive neutral, hypothesis elimination, no winner | Measure learning beyond win rate |

For each family, predeclare ranges, seeds, sample budgets, policy/model counts, confidence allocation and independent experimental unit. Keep a development generator subset, calibration split, confirmatory split and untouched lockbox mechanism families.

Use paired seeds/common random numbers where valid. Statistical uncertainty must be computed at the independent unit (experiment, cluster, episode or task family), not at correlated steps within a trajectory.

## 6. Metrics and estimators

### Validity

- empirical unconditional Type-I error and one-sided confidence limit;
- simultaneous null and alternative functional/distribution coverage;
- transport coverage on the target law;
- anytime false-alarm/familywise error over all monitored times;
- false authorization/promotion on every frozen negative control.

For a simulation estimate \hat{p}=x/N), report an appropriate binomial interval only when the (N) campaign replications are independent Bernoulli units. Do not count correlated time steps as independent replications.

### Decision quality

- false-ship and false-stop loss;
- downstream regret versus evaluator-only oracle;
- worst-case and tail regret by shift family;
- decision reversals under held-out mechanisms;
- useful-learning yield: validated win, regression detection, conclusive neutral, hypothesis elimination or material reduction in decision uncertainty;
- feasibility, abstention and `INDETERMINATE` rates.

A conservative method that always abstains or denies cannot win on decision quality.

### Cost and efficiency

- expected and tail sample count;
- achieved power at matched unconditional Type-I error;
- sample-size prediction error versus achieved target power;
- evaluator, calibration, annotation, solver and independent verification cost;
- wall time, memory, energy where measurable;
- p50/p95/p99 decision latency;
- useful learning per all-in cost.

### Geometry and scaling

- TV radius / robust coefficient width;
- scalar and vector confidence-region width;
- ambiguity conservatism gap against exact tiny-instance oracle;
- robust frontier feasibility fraction;
- sensitivity to support size (K), functional dimension (d), policy/model count, horizon, state/action size and confidence budget;
- transport penalty and robustness tax under clean and shifted regimes.

Report a metric vector and Pareto frontiers. Any scalar decision score requires pre-registered weights, hard floors and tie-breaks.

## 7. Preregistered comparison protocol

1. Freeze source commits, dependency lockfiles, compiler/toolchain, generator and evaluator versions.
2. Freeze the candidate method family, policy/function definitions, hypotheses, split identities, seeds, confidence allocation and stopping rules.
3. Freeze the primary estimands, utility/cost ranges, aggregation unit, multiplicity correction, missing-run treatment and acceptance thresholds.
4. Generate calibration and confirmatory data under declared target-law relations. The evaluator retains hidden ground truth.
5. Run all methods on identical permitted observations and matched all-in budgets.
6. Preserve every failure, unsupported case, abstention, invalid run and infrastructure-indeterminate result.
7. Run independent solvers/verifiers on exact small fixtures and independently recompute aggregate metrics from per-run outputs.
8. Open lockbox families once the protocol and method artifacts are frozen. Any tuning after opening creates a new protocol version and new lockbox.
9. Publish per-family results, negative controls, confidence intervals, compute accounting, source SHAs and claim boundaries.

## 8. Qualification gates (internal targets, not results)

- Every known-answer negative control is caught; zero false promotion/authorization on the frozen fault suite.
- Tiny LP/DP/oracle agreement within (10^{-10}) absolute error; independent solver objective gap within (10^{-8}).
- Simultaneous coverage meets the declared level in independent simulations, with uncertainty intervals and valid experimental units.
- No unconditional Type-I violation after coverage-failure spending.
- DFCE/vector methods improve the predeclared high-support/low-functional-dimension cost–coverage frontier over Weissman TV without degrading protection on decision-aligned shifts.
- Sample savings are assessed at matched achieved power and unconditional Type-I error, including stopping and multiplicity costs.
- Learning-yield improvements are assessed at matched all-in spend, while separately reporting regression/harm detection and neutral/hypothesis-elimination yield.
- No lockbox regime is used for method, threshold, radius or generator tuning.
- Independent verifier reconstructs promoted summaries from canonical per-run records.
- No cryptographic digest is described as proof of IID, stationarity, authenticity, or target-law equality.

The proposed numeric gates in the tracking issue are engineering hypotheses. They are not achieved results, external standards or a claim of superiority.

## 9. Industry and literature comparison

- Spotify's Confidence post dated 14 July 2026 reports 8–20% simulated sample savings and 9.5% median savings across 713 metrics from 283 production experiments for its always-valid sample-size correction. Use the paper's matched calculation problem and report achieved power/sample savings; treat production figures as vendor-reported.
- SERPANT (Gu et al., ICML 2026) is a relevant anytime-valid pairwise ranking baseline with adaptive tournament sampling and familywise error control. It is only a fair comparator for ranking/top-k tasks.
- Koning & van Meer (2026) show how to induce an anytime-valid sequential test that matches a given valid fixed-(N) endpoint test at (N). Include where the hypotheses/stream assumptions match.
- Fontem & Ji (2026) show GTV ambiguity sets can yield finite-sample generalization bounds competitive with TV/Wasserstein on their finite-support Newsvendor and portfolio experiments. Reproduce on matched tasks rather than infer universal superiority.
- Martinez-Taboada & Ramdas (2026) provide a vector empirical-Bernstein route for bounded random vectors, including stopping-time confidence sets. This is a promising lower-dimensional/correlation-aware comparator, subject to its theorem's exact assumptions.
- Ho, Petrik & Wiesemann (2026) provide s-rectangular robust-MDP algorithms and report large solver improvements on their own synthetic/standard workloads. Reproduce matched instances and include a separate coupled-uncertainty comparison.

A result is "better than SOTA" only when the competitor is faithfully implemented/reproduced, estimands and information access match, all-in compute/data budgets match, and independent uncertainty estimates support the comparison. Otherwise say "better on this declared benchmark family."

## 10. Kill conditions

A method/certificate fails qualification if:
- calibration is applied to confirmation with unknown target-law relation;
- the generated data violate an accepted IID/cluster/Markov premise;
- adaptive policy/model/functional selection is outside the simultaneous family or a valid complexity correction;
- a required negative control passes as qualified;
- unsupported or unidentified cases receive a confident decision instead of `INDETERMINATE`;
- a robust method wins only by always denying/abstaining or consuming an unmatched budget;
- the reported result omits failures, failed strata or negative outcomes;
- a SOTA claim relies on unmatched estimands, data, computation, randomization or evaluator access.

## 11. Execution waves

**Wave 1 — independent of robust-DP runners**
- canonical campaign manifest and deterministic scenario generator;
- independent evaluator with hidden truth;
- nominal #9899, oracle-TV #9924, Weissman #9925 and Hoeffding-DFCE #9951;
- F01 orthogonal-shift and F02 decision-aligned-shift fixtures;
- target-law mismatch negative control;
- exact tiny categorical oracle and independent summary verifier.

**Wave 2**
- vector empirical-Bernstein confidence set under verified assumptions;
- faithful GTV reproduction for compatible finite-support problems;
- fixed-horizon, group-sequential and always-valid sample-size/power comparisons;
- integrate #8795 drift monitoring and bounded transport evidence.

**Wave 3**
- (s,a)-rectangular, s-rectangular and globally coupled robust DP;
- continuous/high-dimensional and real public-data replay;
- untouched lockbox evaluation and independent review.

## 12. References

1. Belleh Fontem & Ran Ji (2026), *Distributionally robust optimization with generalized total variation ambiguity sets*, European Journal of Operational Research 328(3), 894–911. https://doi.org/10.1016/j.ejor.2025.09.024
2. Diego Martinez-Taboada & Aaditya Ramdas (2026), *Empirical Bernstein in Smooth Banach Spaces*, Annals of Applied Probability 36(3), 2199–2236. https://doi.org/10.1214/25-AAP2277
3. Chin Pang Ho, Marek Petrik & Wolfram Wiesemann (2026), *Efficient Algorithms for Robust Markov Decision Processes with s-Rectangular Ambiguity Sets*. https://arxiv.org/abs/2602.05591
4. Runzhe Gu, Wenguang Sun, Bowen Gang & Xintao Xia (2026), *Anytime-Valid Inference for Online Ranking of Large Language Models*, ICML 2026, PMLR 306. https://proceedings.mlr.press/v306/gu26l.html
5. Nick W. Koning & Sam van Meer (2026), *Anytime validity is free: inducing sequential tests*, JRSS-B. https://doi.org/10.1093/jrsssb/qkag050
6. Spotify Confidence (14 July 2026), *Accurate Sample Size for Always-Valid Inference*. https://confidence.spotify.com/blog/accurate-sample-size-always-valid
