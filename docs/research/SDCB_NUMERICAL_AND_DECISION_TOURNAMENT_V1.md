# SDCB numerical and scientific-decision tournament v1

**Status:** preregistration protocol, not benchmark results. This document separates numerical integration quality from downstream scientific decision quality and end-to-end scientific-agent capability. All thresholds below are targets unless explicitly identified as published comparator results.

## 1. Competitive thesis

SDCB should not try to win a single leaderboard by mixing unlike claims. It needs to establish three independently testable advantages:

1. **Numerical reliability:** joint predictive probabilities remain accurate, calibrated, bounded, and explicit about unsupported covariance geometry.
2. **Measurement intelligence:** temporal process covariance, shared causes, reference ancestry, device drift, model uncertainty and unequal acquisition windows are represented without false independence.
3. **Decision quality:** the resulting likelihoods select better experiments and produce lower terminal scientific decision loss at equal reference cost, while abstaining when the candidate model set is inadequate.

The intended contribution is the composition of these layers with explicit evidence and authority boundaries. Genz integration, Kalman filtering, Bayesian model averaging, conformal calibration, and expected value of sample information are established methods; this programme must not claim to have invented them.

## 2. Tournament tracks

### Track A — low-dimensional numerical truth

Purpose: qualify the bivariate/trivariate reference before optimization.

Frozen families:
- independent channels;
- positive and negative moderate correlation;
- near-perfect positive and negative correlation;
- asymmetric finite rectangles;
- one-sided and mixed semi-infinite rectangles;
- rare joint tails from 5 sigma through the numerical resolution boundary;
- exact duplicate channels and rank-one PSD support;
- rank-zero and one-deterministic-channel cases;
- small and large marginal variances, including scale heterogeneity;
- covariance symmetry, PSD and unsupported-condition controls.

Reference hierarchy:
- closed-form identities where available (e.g. bivariate Gaussian orthants);
- high-precision conditional integration for low dimensions;
- independent quadrature implementation with different numerical primitives;
- exact marginal and partition identities;
- established package outputs as comparison methods, not as unquestionable ground truth.

Every fixture must bind mean, covariance, rectangle bounds, input units, support rank, expected probability or interval, reference method, absolute/relative tolerance and source digest.

### Track B — published Genz workload reproduction

Purpose: measure competitiveness on the same SPD workloads as modern MVN solvers.

Required methods:
- Genz–Bretz randomized QMC via mvtnorm;
- Miwa where its dimension and non-singularity assumptions fit;
- TVPACK for supported 2D/3D semi-infinite regions;
- tlrmvnmvt where its matrix and QMC-budget semantics fit;
- Peterlin's 2026 doubly blocked Genz implementation;
- SDCB deterministic bivariate/trivariate references on low dimensions;
- SDCB general Genz/QMC implementation only after it exists.

Reproduce the published covariance families, bounds and dimensions before creating SDCB-only workloads:
- random SPD: \(\Sigma=I+UU^\top\);
- equicorrelated-style: \(\Sigma=I+\mathbf 1\mathbf 1^\top\);
- Matérn \(\nu=1.5\), \(d=0.001n\);
- Matérn \(\nu=1.5\), \(d=0.0001n\).

Peterlin's 2026 paper uses standardized symmetric bounds chosen to keep most integral values in an approximate 0.25–0.8 range. Reproduce this setting as one track, then add application-shaped asymmetric and rare-tail rectangles as a separate track.

Published comparator context:
- The 2026 doubly blocked Genz paper reports speedups usually above 16× for tested \(n\in[2^4,2^7]\), and often near/above 100× for \(n>2^9\), against the compared implementations that support those dimensions. These are the paper's tested workloads and hardware, not an SDCB result.
- That paper evaluates only SPD matrices; it explicitly notes that singular covariance requires a modified method such as Genz–Kwong and that singular support is available in mvtnorm but not the other compared packages.
- mvtnorm documents Genz–Bretz for broad SPD dimensions, Miwa for smaller dimensions (up to 20), and TVPACK for 2D/3D semi-infinite integration. Miwa's non-orthant path replaces infinite limits with maxval and warns; that approximation must be surfaced in comparisons.
- The paper notes mvtnorm's maximum covariance dimension of 1000 for its comparison; it excludes that package above this dimension. Do not describe those comparisons as all-method results at dimensions it cannot process.

Fair-comparison rules:
- Match covariance matrices, bounds, mean transformations, dimension, probability scale, tail probability, stopping tolerance, total QMC points, number of independent randomizations/replicates, hardware, thread count, BLAS library and precision.
- Normalize the different QMC-budget conventions. In the paper's comparison, tlrmvnmvt interprets N as points per repetition and runs 20 repetitions, whereas other methods interpret the configured budget as total points.
- Include preprocessing in end-to-end timing: covariance scaling, permutation, factorization and data allocation. Also publish kernel-only timing separately.
- Run at least 100 timed repetitions after warm-up; report median, p95, minimum and dispersion. Include the paper's minimum-time statistic for direct replication, but do not rely on minimum alone for deployment claims.
- Report FP64 as primary. FP32/low-rank variants are separate accuracy–speed trade-off arms.
- Use exact or independently high-precision fixtures for error measurements; do not judge numerical accuracy by solver-to-solver agreement alone.

### Track C — PSD geometry and dependence integrity

Purpose: ensure singular and near-singular measurements do not manufacture evidence.

Families:
- exact duplicate channels;
- linear combinations and rank-one / rank-k covariance;
- one or more zero-variance coordinates;
- rank-zero deterministic states;
- nearly singular but SPD matrices over a condition-number ladder;
- non-PSD matrices and asymmetric input matrices;
- repeated/overlapping observation windows;
- correlated residual acquisition noise and multiple explicit common factors.

Required outputs:
- explicit covariance support rank;
- accepted / unsupported / invalid classification;
- joint rectangle probability or explicit unresolved status;
- probability bounds or numerical error diagnostics;
- marginal recovery and partition normalization;
- no diagonal jitter or unreported covariance repair.

Compare against mvtnorm's singular-covariance path where applicable. Do not benchmark a solver that only supports SPD covariance on rank-deficient cases and score its unsupported behavior as a speed loss; compare capability coverage separately.

### Track D — temporal metrology and calibration

Purpose: establish the part that generic MVN integrators do not provide.

Baselines:
- IID and static full-covariance conditioning;
- scalar OU and vector continuous-time Kalman references;
- Gaussian-process temporal covariance where tractable;
- hierarchical/random-effects calibration;
- control-standard and control-chart recalibration;
- OLS, GDR and C-GDR for mobile-reference collocation where their assumptions match;
- fixed-window, exponential-weight, change-point-reset and drift-triggered calibration;
- ordinary split conformal only on justified exchangeable units;
- dependence-aware online time-series conformal methods where the task matches.

Frozen scenarios:
- unequal and overlapping acquisition windows;
- delayed reference labels / delayed residual maturation;
- common reference drift plus device-local drift;
- shared environmental, power or firmware disturbances;
- calibration drift versus actual physical-process drift;
- abrupt, gradual, recurring and out-of-support regimes;
- missing, censored, invalid, provisional and revised reference observations;
- clock skew, out-of-order delivery and duplicate evidence;
- exact duplicate devices / reference ancestry and confounded reference chains.

Do not treat non-overlapping time blocks as automatically exchangeable. Compare #13943's fresh-fold and epoch-ID contract with #13280's delayed-feedback online calibration and #13924's anytime-valid rank-martingale drift monitor. AcMCP and other time-series conformal methods are specific baselines for dependent multi-horizon residuals and delayed feedback; their assumptions and estimands must not be transferred to arbitrary metrology data without verification.

### Track E — adaptive scientific decisions

Purpose: show the measurement model changes decisions, not just probabilities.

Use the existing owners:
- #13567: exact prior-free sequential policy-tree evaluator (and separate Bayesian expected-terminal-loss evaluator);
- #13739: integration of actual Symthaea experiment selectors;
- #13691: hidden-family active scientific intelligence episodes;
- #13805: evidence-gated hypothesis/common-cause/device posterior transitions.

Baselines:
- random legal action;
- fixed sequence / uniform schedule;
- greedy entropy and expected information gain;
- A-, D-, c- and E-optimal design where assumptions fit;
- one-step EVSI;
- multi-step Bayesian EVSI / exact Bayesian terminal-loss oracle;
- robust maximin terminal-loss policy;
- current HDC disagreement and Spark EIG selectors;
- simple Kalman/GP + classical design baselines;
- learned HDC/CfC proposals only after exact small-world references exist;
- evaluator-only exact finite-horizon oracle as an upper reference.

Hard known-answer controls:
- high-entropy nuisance measurement that does not change terminal decision loss;
- lower-entropy measurement that changes the correct terminal decision;
- non-myopic branch unlock where a low-immediate-value measurement makes a valuable follow-up possible;
- common-cause discovery that should invalidate apparent independent replication;
- duplicate-channel measurement with zero incremental information after conditioning on the first;
- model-set misspecification where the best registered model is still inadequate;
- out-of-support regime where calibrated abstention is better than forced selection;
- hard reference-budget, deadline and pathwise-resource boundary cases.

For each episode, compare policies on identical public evidence, action sets, time windows, observation semantics, terminal loss, reference budget and compute budget. A single action mapping must apply across hidden robust scenarios. Do not give the candidate hidden generator truth, future outcomes, calibration labels or evaluator scores.

Do not claim adaptivity from a known-covariance, fixed linear-Gaussian, variance-only problem if outcome values cannot affect the optimal next action. Construct explicit outcome-dependent value through model mixtures, discrete common-cause states, different dynamic regimes, action-dependent resets or terminal decisions.

### Track F — end-to-end scientific agents

Purpose: evaluate the full Symthaea proposal → experiment → observation → verification → scientific claim cycle.

Use external benchmarks as complementary evidence, not substitutes for the SDCB physical-measurement benchmark:
- **Terminal-Bench-Science 0.1** has 70 expert-curated tasks across life, physical, earth, mathematical and engineering sciences. Its task families explicitly include sensor calibration, model fitting, inverse problems, signal processing, statistical inference and simulation. The initial August 2026 release reported 30% resolution for its best evaluated system after three trials per task; the benchmark is designed to evolve through new releases. Freeze the release, harness and leaderboard snapshot when comparing. This is the closest public workflow bridge for an SDCB calibration task.
- **ASI-Bench** is a 60-task, 11-domain project-level benchmark with a B1–B4 guidance gradient. Its public catalog includes LiDAR/IMU time-offset and extrinsic calibration, transmission-line dispersion inferred from calibration data, correlated range-noise mapping, and stochastic dynamics. These tasks are especially relevant to SDCB's calibration, temporal-model and causal-dependence tracks. Record the exact task set and benchmark revision.
- **FrontierPhysics** tests literature review, planning, experiments and structured research outputs over 56 tasks contributed by 42 physicists across nine physics domains. Its initial 2026 paper reports that no evaluated configuration passed more than 46 of 168 rollouts.
- **AstaBench** covers more than 2,400 scientific problems across literature, coding/execution, data analysis and end-to-end discovery. Its April 2026 update reports about 53% overall for Asta v0, while the best E2E-Bench-Hard agent completed only 3% of tasks perfectly end-to-end.
- **ScienceAgentBench** provides 102 expert-validated data-driven discovery tasks from peer-reviewed scientific work.
- **DiscoverPhysics** uses 22 procedurally generated worlds with deliberately nonstandard physical laws and requires iterative experimental discovery.
- **Autonomous Science Bench 0.1** is a six-task pilot with a public leaderboard; interpret its resolution rates as pilot evidence, not a broad field ranking.
- **SciAgentArena** offers approximately 200 interactive scientific research tasks with stepwise verification.

Map each SDCB task onto the benchmark capabilities it genuinely tests. A paper-generation score is not a metrology score; a solver score is not evidence of discovery; a synthetic win is not physical validation.

## 3. Primary metrics and target gates

| Dimension | Metric | Target |
|---|---|---:|
| Numerical | Low-dimensional known-answer agreement | 100% within frozen tolerance |
| Numerical | Probability mass normalization | Absolute error <= 1e-9 for declared finite tables |
| Numerical | Marginal recovery | Absolute error <= 1e-9 |
| Integrity | False independent-confirmation claims | 0 on all registered negative controls |
| Integrity | Invalid / stale / non-PSD input accepted | 0 |
| Integrity | Silent jitter / repair events | 0 |
| Calibration | Nominal predictive interval coverage | 90% and 95%, reported by family and regime |
| Calibration | Unsupported / out-of-envelope forced decisions | 0 |
| Policy | Approximate-policy regret vs exact small-world oracle | <= 2% |
| Policy | Terminal decision-loss reduction per equal reference cost | >= 10% vs strongest matched baseline, paired CI excludes zero |
| Operations | Hard-budget / deadline violations | 0 |
| Evidence | Invalid receipt or stale version accepted | 0 |
| Evidence | Accepted-transition replay mismatches | 0 |

The 10% decision-loss target is a preregistered research goal, not a measured achievement. It only qualifies on held-out task families after paired confidence intervals exclude zero and all safety/integrity gates pass.

## 4. Required reporting

For every solver and policy publish:
- source commit, build/toolchain/runtime and configuration digests;
- hardware model, CPU/GPU, memory, thread count, BLAS and compiler flags;
- raw input matrix/bounds digest and generator seed or task manifest;
- dimension, covariance support rank, condition estimate, marginal scales and tail probability;
- QMC point budget, replicate count, randomization and stopping reason;
- absolute/relative error against the independent reference, reported error estimate and empirical error across replicates;
- runtime distribution, peak memory, allocations and energy where measurable;
- probability normalization, marginalization error, unsupported cases and all failed trials;
- coverage, interval width, proper predictive scores, log score, Brier score and decision regret;
- action/reference cost, turnaround time, elapsed time, budget utilization and terminal loss;
- abstentions, invalid observations, model-adequacy alarms and false alarms;
- accepted/rejected evidence receipts, state-version changes and replay results.

Publish Pareto frontiers. Do not collapse probability error, decision loss, safety, latency, energy and provenance into one score that allows one dimension to compensate for failure in another.

## 5. Paradigm shifts to pursue

1. **Support geometry, not forced positive-definiteness.** Treat rank as structure. Duplicate or linearly dependent channels live on a lower-dimensional support; integrate there or return an explicit unsupported result. Never hide singularity with diagonal jitter.
2. **Measurement as a time-windowed causal observation.** A timestamp is not an acquisition interval. Preserve each instrument's integration kernel, elapsed-time state evolution and endpoint/observation cross-covariance.
3. **Causal dependence accounting.** Keep physical-process correlation, shared measurement causes, reference-chain ancestry, device-local drift and residual acquisition covariance distinct. Correlation is evidence of dependence, not proof of a named physical cause.
4. **Open-world model adequacy.** Model weights that sum to one do not prove a candidate is adequate. Preserve observations and diagnostics, but block automatic scientific-decision updates when the registered set fails a separately justified adequacy test.
5. **Calibration as a lifecycle with a validity domain.** Track label maturity, time alignment, calibration epoch, model digest, drift mode and fresh-fold custody. Use a dependence-aware method or make no exchangeability claim.
6. **Decision value, not information volume.** Optimize terminal scientific loss and value of sample information under full resource budgets. High entropy reduction is not inherently useful.
7. **Evidence-carrying adaptation.** Symthaea may propose the next experiment; the policy oracle evaluates it; authorization and physical execution remain separate; observations update beliefs only through independently verified receipts and replay.
8. **Decision-aware numerical refinement.** Do not compute every rectangle probability to an arbitrary uniform tolerance. Carry an integration estimate and its uncertainty diagnostics into the policy evaluator; allocate more points/refinement only where action-value intervals overlap or a decision boundary may change. Stop early when the preferred action is separated by a valid bound, and abstain/escalate when it is not. Randomized-QMC replicate spread and Simpson refinement deltas are diagnostics unless a separate calibration/proof justifies a bound; never describe them as certified intervals by default.

## 6. Implementation sequence

1. Finish M0 bivariate reference and exact rank-one/rank-zero fixtures (#13951).
2. Add independent trivariate known-answer integration and randomized-QMC comparison.
3. Add a decision-aware refinement controller that trades numerical effort against policy action-margin, and evaluates compute regret and wrong-action frequency on exact small worlds.
3. Implement a Genz-style SPD path with randomized replicates, reproducible streams, variable reordering and reported error diagnostics.
4. Add general PSD support reduction only with rank-revealing factorization and exact support tests; never jitter.
5. Bind joint likelihood tables to #13940 interval-observation state predictions, conditional on each model/common/device state.
6. Feed the likelihood interface into #13567 and #13739; preserve the #13805 evidence authority boundary.
7. Integrate delayed matured-label and dependence-aware calibration owners (#13280, #13924); keep #13943's exchangeability caveat.
8. Run the TC0–TC14 measurement tournament and hidden mechanism/device/reference/time splits in #13691.
9. Only then train HDC/CfC proposals against exact policy references and report generalization/regret.
10. Reproduce the published Genz double-blocked SPD benchmarks on matched hardware before making any numerical SOTA claim.

## 7. Sources and comparator references

- Peterlin, J. (2026), *Doubly blocked variant of Genz's method for multivariate normal integration*, Computational Statistics 41, 117. https://doi.org/10.1007/s00180-026-01794-8
- Genz, A. and Bretz, F. (2009), *Computation of Multivariate Normal and t Probabilities*, Springer. https://doi.org/10.1007/978-3-642-01689-9
- mvtnorm package documentation, algorithms Genz–Bretz, Miwa and TVPACK. https://cran.r-project.org/package=mvtnorm
- Wang, X. and Hyndman, R. (2026), *Online conformal inference for multi-step time series forecasting* (AcMCP), 5 November 2026. https://www.robjhyndman.com/publications/cpts.html
- Yang et al. (2026), *Sequential Conformal Calibration under Recursive Selection*. https://arxiv.org/abs/2610.12937
- You et al. (2026), *FrontierPhysics: Evaluating Agents for End-to-End Physics Research*. https://arxiv.org/abs/2611.06910
- Ai2 (2026), *AstaBench update: New results, plus adoption from industry*. https://allenai.org/blog/astabench-update-spring-2026
- Chen et al. (2025), *ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery*. https://hal.cs.princeton.edu/scienceagentbench
- Wiemann et al. (2026), *DiscoverPhysics: Benchmarking LLMs for Out-of-the-Box Scientific Thinking*. https://arxiv.org/abs/2605.26087
- Wen and Liu (2026), *Autonomous Science Bench: Evaluating AI Agents for Autonomous Scientific Discovery*. https://arxiv.org/abs/2610.18449

The numerical and scientific decision tracks must be reported separately. An external agent benchmark can demonstrate workflow capability; it cannot validate a sensor model, causal interpretation or physical measurement.


### Conditional likelihood custody

Every multichannel likelihood supplied to Track E must carry a state-conditional identity envelope: prediction/model IDs and versions, the complete scientific/common-cause/device conditioning assignment, per-channel device evidence versions, each channel's own acquisition-window ID and start/end, per-channel calibration epoch/context, and measurement-action identity. The output also carries the conditional mean/covariance, threshold arrays and numerical method configuration. This prevents a table from becoming an unlabelled probability array or losing the provenance of unequal acquisition windows.

The envelope is identity propagation, not authentication. The likelihood provider does not verify signatures, mutate #13805, authorize physical execution, or replace the independent receipt path. Each table must remain conditional on the relevant model and common/device states until the exact Bayesian or robust decision estimand specifies how to aggregate them.
