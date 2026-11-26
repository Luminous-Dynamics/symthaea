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


## 13. Exact orthogonal/aligned pilot fixtures

These known-answer categorical fixtures make F01/F02 concrete. Categories are indexed 0 through 7.

Nominal distribution:

`P0 = (0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125)`

Four bounded path functionals, in this order:

- rejection A: `g1 = (0,0,1,1,0,0,1,1)`;
- rejection B: `g2 = (0,1,0,1,0,1,0,1)`;
- scaled cost: `g3 = (1,1,0,0,1,1,0,0)`;
- scaled utility: `g4 = (0,0,0,0,1,1,1,1)`.

Each functional has nominal expectation 0.5.

### F01 orthogonal shift

`Q_orth = (0.20, 0.05, 0.15, 0.10, 0.10, 0.15, 0.05, 0.20)`

The total-variation distance is `TV(P0, Q_orth) = 0.20`, but all four expectations remain exactly `(0.5, 0.5, 0.5, 0.5)`. This is a distribution shift that is irrelevant to this declared functional family, not a claim that all omitted functions are unchanged.

### F02 decision-aligned shift

`Q_aligned = (0.10, 0.10, 0.15, 0.15, 0.10, 0.10, 0.15, 0.15)`

Here `TV(P0, Q_aligned) = 0.10`, and expectations become `(0.6, 0.5, 0.4, 0.5)`. Rejection A rises by 0.10 and scaled cost falls by 0.10. The method must report the decision-relevant shift rather than dismissing it as harmless.

### Pilot calibration settings and formula checks

For `K = 8`, `n = 5,000`, four functionals and family failure budget `delta = 0.01`, Bonferroni gives `delta_j = 0.0025`. Scalar Hoeffding gives:

```text
r_j = sqrt(ln(1 / 0.0025) / (2 * 5000)) = 0.02447747
```

Using the conservative Weissman slack with family failure 0.01:

```text
r_TV_slack = sqrt((8 * ln(2) + ln(100)) / (2 * 5000)) = 0.03185961
```

An independent NumPy formula-level pilot (seed 20261102, 20,000 IID multinomial replications per scenario) observed simultaneous DFCE coverage of 99.855% for F01 and 99.845% for F02. The corresponding exact 95% binomial intervals were approximately [99.792%, 99.903%] and [99.780%, 99.895%]. The sample-based Weissman radius (empirical TV to P0 plus slack) had mean/median/5th–95th percentiles:
- F01: 0.23190 / 0.23186 / [0.22106, 0.24266].
- F02: 0.13189 / 0.13186 / [0.12046, 0.14326].

These are formula-level simulations, not Rust tests, a proof, a comprehensive coverage study, or an end-to-end Type-I result. The comparison has different guarantee scopes: TV protects arbitrary bounded path functionals, whereas DFCE protects only the four declared functions. Preserve that distinction in all reports.


## 14. Rectangularity comparison: no universal conservatism ordering

Do not assume that s-rectangular ambiguity is always less conservative than (s,a)-rectangular ambiguity. The geometry differs—(s,a)-rectangular sets factorize across state-action rows, while s-rectangular sets factorize across states and may couple action-specific rows within each state—but robust-value comparisons also depend on the radii, distance, normalization and shape of the ambiguity sets.

Ramani & Ghate, *A Family of s-Rectangular Robust MDPs: Relative Conservativeness, Asymptotic Analyses, and Finite-Sample Properties*, SIAM Journal on Optimization, DOI https://doi.org/10.1137/23M1559920, show that for their family an s-rectangular model can be compared with an (s,a)-rectangular model in both directions by changing radii; under lower-semicontinuous convex distances, identical robust optimal values can be obtained under their stated conditions.

Therefore compare the geometries at matched statistical coverage and matched calibration evidence, not by assigning the same arbitrary scalar radius. Report ambiguity-set inclusion when provable, the induced robust-value gap, and the calibration/solver cost. The objective is to measure geometry–calibration–computation trade-offs, not to assume one class universally dominates another.


## 15. Policy-specific calibration data and estimand identification

A policy-specific functional expectation requires policy-specific evidence. The same observed trajectory does not reveal the realized counterfactual outcome under every candidate policy.

Every calibration row must identify its evidence mode:

- randomized execution of the candidate policy;
- simulator/evaluator data with explicitly available ground truth;
- off-policy estimation from behavior-policy data.

For randomized execution, bind assignment unit, randomization probabilities, cluster IDs, eligibility/exposure rules, policy implementation/version, intervention timing, outcome and censoring definitions, and any interference assumptions. In confirmation, execute the selected randomized policy mixture under a declared assignment protocol matching the receipt estimand.

For off-policy estimates, require positivity/support and a typed estimator/weight contract. V1 DFCE must not apply bounded-IID Hoeffding to unbounded importance weights, unsupported actions, unmodeled interference, or informative missingness. Such inputs are unsupported/indeterminate until a separate valid finite-sample method is implemented.

Do not count repeated observations from one subject, user, cluster, experiment or episode as independent calibration paths. Common random numbers are appropriate for synthetic paired-seed evaluation when their coupling is declared; they do not create observable real-world counterfactuals.

Required negative controls: deterministic policy assignment with no overlap, zero-probability policy arms, cluster leakage, interference between units, and informative missingness. None may qualify as IID per-policy calibration rows.


## 16. Noncompensatory guardrails and rollback quality

The policy utility objective must not compensate for a mandatory guardrail violation. Track required global guardrails and applicable local/stratum guardrails separately from the utility objective. A policy mixture is eligible only if every applicable guardrail's simultaneous one-sided bound clears its predeclared floor/cap. Missing or non-estimable strata return `INDETERMINATE`.

For a linear expected guardrail functional `g`, a randomized mixture with weights `q_i` can use:
- upper-bound guardrail: `sum_i(q_i * U[i,g,s,m]) <= threshold[g,s,m]`;
- lower-bound guardrail: `sum_i(q_i * L[i,g,s,m]) >= threshold[g,s,m]`.


### Support-size scaling calculation (formula only)

At `n = 5,000` and familywise `delta = 0.01`, compare one TV distribution row with scalar DFCE over `d` predeclared functionals. The TV column is only the Weissman concentration slack `sqrt((K * ln(2) + ln(1/delta)) / (2*n))`; it excludes the additional empirical TV distance to the nominal law. The DFCE column is the scalar Hoeffding radius `sqrt(ln(d/delta) / (2*n))` after Bonferroni across the `d` functionals.

| Support K | Weissman TV slack | DFCE radius d=4 | DFCE radius d=64 |
|---:|---:|---:|---:|
| 8 | 0.03186 | 0.02448 | 0.02960 |
| 32 | 0.05176 | 0.02448 | 0.02960 |
| 128 | 0.09661 | 0.02448 | 0.02960 |
| 512 | 0.18960 | 0.02448 | 0.02960 |
| 2,048 | 0.37738 | 0.02448 | 0.02960 |
| 8,192 | 0.75385 | 0.02448 | 0.02960 |
| 32,768 | 1.50724 (radius capped at 1) | 0.02448 | 0.02960 |

This isolates only the concentration term at one family row and matched 99% confidence. The full Weissman radius adds empirical TV distance to the nominal path law, and the family allocation must expand to include every policy/model row. The calculation is not a coverage simulation or an end-to-end decision result.


### Vector confidence and observed cross-policy dependence

The vector empirical-Bernstein comparator may exploit only dependence actually observed under the sampling design. Multiple decision/guardrail functionals scored on the same independent experimental unit within a policy/model arm form an observed bounded random vector; a valid vector confidence set may account for coordinate dependence.

Different policy arms generally reveal different potential outcomes in a real randomized experiment. Do not invent cross-policy covariance or paired potential-outcome coupling. Use per-policy vector confidence sets plus simultaneous coverage across the frozen policy family, unless a simulator, randomization design or identified estimator genuinely supports paired cross-policy contrasts. State the Banach norm and confidence-set geometry explicitly because the optimizer's support function depends on them.


## 18. Compose all null-side confidence failures before alpha spending

When transport bounds or state-conditional confidence sets are added, their failure probabilities must be included in the same good event required by the unconditional null Type-I proof.

Let the required null-side events include, as applicable:
- calibration-radius / functional-confidence coverage;
- calibration-to-confirmation transport coverage;
- transition-row confidence-set coverage;
- any other probabilistic event required to bound the confirmatory null path law.

Define the joint event as their intersection. A conservative union bound is:

`delta_null_total <= sum(delta_k)`

for all required null-side confidence events. Use the total in the conditional alpha calculation:

`alpha_cond = (alpha - delta_null_total) / (1 - delta_null_total)`

and require `delta_null_total < alpha`. Do not spend only the calibration delta while leaving transport/transition confidence failure unaccounted for.

The event used for this calculation must be independent of the confirmatory stream, or the method needs a separate joint/sequential proof. If the transport estimator consumes confirmatory outcomes, the independent-split derivation does not apply automatically.

Keep alternative-performance confidence failure and guardrail false-safe confidence failure as separately named budgets because they support different claims. If a top-level authorization claim composes several gates, state exactly which failure events govern that claim and how they combine; never relabel guardrail or alternative failure as Type-I error.


## 19. Learning yield: classify outcomes and prevent metric gaming

Spotify's approximately 64% learning / 12% win figures originate in its September 2025 *Beyond Winning: Spotify's Experiments with Learning Framework* and are reiterated in its May 2026 *Better Experiments with LLM Evals* article, not a September 2026 publication:
- https://engineering.atspotify.com/2025/9/spotifys-experiments-with-learning-framework
- https://engineering.atspotify.com/2026/5/better-experiments-with-llm-evals-a-funnel-not-a-fork

Spotify's definition requires a valid, decision-ready experiment. Its learning outcomes include a win without regressions, a detected regression, or a sufficiently powered neutral result. It states that every relevant metric must be powered for a neutral result to qualify and warns against improving the learning rate by sacrificing win rate, experiment volume or precision. These are Spotify's inclusion criteria and company-reported outcomes, not universal industry standards.

For this benchmark:
- require health/setup checks before classifying a learning outcome;
- require decision readiness across the complete predeclared metric and guardrail family, not only the primary metric;
- classify win, regression, powered-neutral, hypothesis-elimination, invalid, underpowered and indeterminate separately;
- report per-metric achieved power, precision/MDE, restarts/attempts, experiment volume, false-learning rate and all-in cost;
- treat learning yield as an outcome vector, with win-rate, volume and precision guardrails, not a scalar optimization target.

Do not set a target to reproduce Spotify's 64% learning rate. Compare methods at matched inclusion rules, sample/evaluator budget, metric families and power requirements.

References:
- Spotify Engineering (September 2025), *Beyond Winning: Spotify's Experiments with Learning Framework*. https://engineering.atspotify.com/2025/9/spotifys-experiments-with-learning-framework
- Spotify Engineering (May 2026), *Better Experiments with LLM Evals — A funnel, not a fork*. https://engineering.atspotify.com/2026/5/better-experiments-with-llm-evals-a-funnel-not-a-fork


## 20. Implementation work opened after protocol v0.1

The following follow-on artifacts separate the mathematical/engineering concerns into independently reviewable work:

- [PR #9999 — bounded-IID decision-functional confidence](https://github.com/Luminous-Dynamics/symthaea/pull/9999): scalar Hoeffding DFCE reference, same-target-law-only V1, sampling-unit and randomized-arm contracts. Draft; tests authored, not executed.
- [PR #10005 — noncompensatory guardrail verifier](https://github.com/Luminous-Dynamics/symthaea/pull/10005): fixed-mixture PASS/BLOCKED/INDETERMINATE gate with global/local requirements and row-level provenance. Draft; tests authored, not executed. This is a verifier, not yet the guardrail-constrained optimizer.
- [Issue #10006 — guardrail-constrained frontier LP](https://github.com/Luminous-Dynamics/symthaea/issues/10006): integrate hard guardrail inequalities directly into the exact policy-frontier optimization.
- [Issue #10008 — calibration-to-confirmation transport certificates](https://github.com/Luminous-Dynamics/symthaea/issues/10008): finite-sample calibration/target transport and composition of transport failure into the null-side error budget.
- [Issue #10009 — vector empirical-Bernstein confidence sets](https://github.com/Luminous-Dynamics/symthaea/issues/10009): theorem-faithful joint confidence regions and directional support functions.

These are draft research artifacts. No Rust compile/test, CI pass, performance result or SOTA superiority is claimed for the new implementations. The benchmark waves must qualify them at exact heads and with independent solvers/verifiers before promotion.


Wave-1 now has an executable categorical generator and CLI:
- [PR #10011 — functional shift benchmark](https://github.com/Luminous-Dynamics/symthaea/pull/10011): seeded F01/F02 generation, declared functional vectors, omitted-functional trap, Hoeffding/Weissman confidence calculations, empirical coverage and TV radius summaries. Draft; tests authored, not executed.
