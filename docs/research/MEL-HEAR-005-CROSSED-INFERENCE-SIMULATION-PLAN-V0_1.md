# MEL-HEAR-005 — Crossed-inference simulation and method-selection plan

**Status:** preregistration scaffold; estimators not selected or implemented  
**Date:** 28 November 2026  
**Depends on:** #13947 property-relative reducer, #14019 source-work estimator, #14026 sampling design, #14168 admission classifier

## 1. Decision to be made

Do not select a crossed estimator because it produces the narrowest interval. Select it because it preserves the declared estimand and achieves calibrated error/coverage under the sampling structures the musician study can actually produce.

The unit of the scientific question is not automatically the audio clip. A source work can contribute several windows, a performer can contribute several works, a work can have several performers, and sessions can be shared or nested. The inferential procedure must follow the recruited design rather than pretending these observations are independent.

## 2. Freeze the estimand before comparing methods

At least three estimands are plausible and not interchangeable:

1. **Equal-source-work:** each sampled source work receives equal total weight; within-work items are averaged before the population mean.
2. **Equal-performer:** each sampled performer receives equal total weight; works performed by that person are averaged under a declared rule.
3. **Crossed work × performer target:** expected performance on a work and performer drawn from the declared target populations, with an explicit rule for unobserved work–performer cells and ensembles.

The current #14019 estimator implements only the first estimand under a one-way independent-source-work assumption. A crossed bootstrap must not silently change it to item-weighted or equal-cell weighting. Multi-performer observations require a preregistered membership/weighting rule; duplicating one ensemble recording once per musician would create pseudo-replication.

The confirmatory endpoint set must retain separate Performance-STAY false-difference, Musical-FLIP miss, and joint exact rates. The STAY and FLIP arms are paired; resampling must apply identical sampled design weights to baseline and candidate. Do not hide an endpoint regression in a composite.

## 3. Candidate methods — evaluate, do not presume

### A. Design-based performer-disjoint cohort

Recruit independent work/performer/session groups so the relevant sampling units do not recur across the clusters used for inference. Then use the simplest justified cluster estimator. This is the preferred reference when feasible because it minimizes statistical-model dependence. It may be expensive and may narrow the target population; state that trade-off.

### B. Existing one-way source-work bootstrap (#14019)

Retain as the v1 comparator for panels whose design supports independent source-work clusters. A 30-work floor is only a reporting gate, not a power calculation. It is invalid as population inference when a performer or session induces dependence across works and that dependence is not handled by design or model.

### C. Two-way/multiway cluster-robust variance

Cameron, Gelbach and Miller derive a multiway cluster-robust sandwich estimator for non-nested cluster dimensions using inclusion–exclusion across cluster intersections. This is an important candidate for a work × performer design, but applying it to nonlinear binary rates requires a defensible linearization/influence-function derivation or an appropriate estimating-equation formulation. Do not bolt a linear-regression variance formula onto the current report's three ratios without deriving the estimand and score.

Reference: Cameron, Gelbach & Miller (2011), *Robust Inference with Multi-way Clustering*, Journal of Business & Economic Statistics 29(2), 238–249, DOI 10.1198/jbes.2010.07136. https://doi.org/10.1198/jbes.2010.07136

### D. Crossed / pigeonhole bootstrap

Owen's pigeonhole bootstrap separately resamples the row and column dimensions of crossed, potentially unbalanced data. It is a principled candidate for crossed work × performer incidence, and has a mean-consistency result under crossed random-effects settings; it is not an exact universal bootstrap for every statistic or sparse design. The correct cell multiplicity, unequal cluster weighting, repeated sessions and multi-performer memberships must be specified and tested for the Melothaea estimand.

Reference: Owen (2007), *The Pigeonhole Bootstrap*, Annals of Applied Statistics 1(2), 386–411, DOI 10.1214/07-AOAS122. https://doi.org/10.1214/07-AOAS122

### E. Few-cluster corrections

CR2 / bias-reduced linearization with Satterthwaite degrees of freedom, wild-cluster bootstrap, and cluster jackknife methods are candidates when one or more axes have few independent clusters. Their assumptions and supported models differ. The CR2 literature includes linear-model settings; any adaptation to paired Bernoulli outcomes or the equal-work estimand must be validated, not presumed.

References:
- Pustejovsky & Tipton (2018), *Small-Sample Methods for Cluster-Robust Variance Estimation and Hypothesis Testing in Fixed Effects Models*, Journal of Business & Economic Statistics 36(4), 672–683, DOI 10.1080/07350015.2016.1247004. https://doi.org/10.1080/07350015.2016.1247004
- MacKinnon, Nielsen & Webb (2023), *Cluster-robust inference: A guide to empirical practice*, Journal of Econometrics 232(2), 272–299, DOI 10.1016/j.jeconom.2022.04.001. https://doi.org/10.1016/j.jeconom.2022.04.001
- Webb (2023), *Reworking wild bootstrap-based inference for clustered errors*, Canadian Journal of Economics 56(3), 839–858, DOI 10.1111/caje.12661. https://doi.org/10.1111/caje.12661

Small-sample corrections are not free power. Published cluster-randomized-trial simulations show that type-I error and coverage can vary substantially with cluster count, ICC, cluster-size imbalance and degrees-of-freedom corrections. Therefore, do not turn “30 works” into a universal threshold or choose a correction without design-matched simulation.

### F. Hierarchical / crossed-effects model

A model with work, performer, session and possibly work × performer effects can be a useful sensitivity analysis and may support partial pooling. It introduces distributional and covariance assumptions; it is not a substitute for documenting the sampling frame or an automatic rescue for sparse axes.

## 4. Simulation grid to preregister

Simulate paired baseline/candidate outcomes under known truth. Preserve the pairing between model answers, all three arms, and repeated items from the same design units.

### Sampling structures

- Independent source works and performers.
- Performers repeated across multiple works.
- Several performers nested within a work.
- Multiple performances/takes within performer × work.
- Sessions nested within performer × work.
- Recording sessions reused across works.
- Performer × work interaction effects.
- Multi-performer ensemble arms with explicit membership.
- Unbalanced work and performer degrees; a few high-leverage clusters.
- Sparse target-property × STAY/FLIP-factor strata.
- One axis with few independent units.
- Missing, inconsistent, or rekeyed work/performer/session identities.

### Data-generating mechanisms

Use at least two outcome families:
- a continuous paired candidate-minus-baseline outcome with additive work, performer, session and work × performer effects, plus heteroskedastic residuals;
- a paired Bernoulli/triad generator for STAY errors, FLIP misses and joint exactness, with shared latent effects so paired baseline/candidate outcomes are correlated.

Vary work and performer variance components, work–performer interaction, cluster-size imbalance, within-cell repetition and the baseline–candidate correlation. Include null effects, small effects, effects near the intended minimum important difference, and endpoint trade-offs where STAY improves while FLIP regresses.

### Comparators

At minimum:
- item-wise bootstrap as a deliberately invalid negative control;
- #14019 one-way source-work bootstrap;
- design-based performer-disjoint / work-disjoint estimator;
- two-way pigeonhole bootstrap;
- two-way cluster-robust influence-function estimator where the estimand derivation is valid;
- small-sample adjusted/wild-bootstrap variants only in scenarios matching their assumptions;
- a hierarchical crossed-effects sensitivity model.

Do not rank methods by interval width alone.

## 5. Qualification outputs and targets

For every preregistered design cell, report:
- point-estimate bias and RMSE;
- nominal 95% interval coverage and interval width;
- null type-I error at 5%;
- power curves under the predeclared effect-size grid;
- estimator convergence/failure rate;
- sensitivity to cluster-size imbalance, sparse cells, influential performers and missingness;
- exact resampling units and paired baseline/candidate draw reuse;
- seed, PRNG, replicate count, algorithm revision and output digests.

**Proposed simulation precision:** begin with at least 10,000 independent Monte Carlo replications per primary design cell. At a true rate of 0.05, the binomial Monte Carlo standard error is about 0.0022; at 0.95 coverage it is also about 0.0022. Report the Monte Carlo interval rather than treating a single percentage as exact. Increase replication for boundary decisions or very rare failures.

**Proposed calibration gates, to be frozen before simulation:**
- the 95% Monte Carlo interval for null rejection should include 0.05 in each design cell designated as supported;
- the 95% Monte Carlo interval for nominal 95% coverage should include 0.95 in each supported design cell;
- absolute standardized bias should be no greater than 0.05 in the declared estimand's standard-deviation units;
- identical baseline and candidate must yield exactly zero paired effect;
- item/row/column order permutations must not change point estimates, and fixed-seed replay must be deterministic;
- invalid or underidentified sampling designs must withhold population intervals rather than fall back to item-wise resampling.

These are simulation qualification targets, not empirical performance claims. The Monte Carlo acceptance rule should be preregistered with its multiplicity handling across design cells; an overall average cannot hide a badly calibrated crossed or sparse scenario.

## 6. Known-answer and independent-oracle requirement

Before general simulation, implement an exhaustive tiny crossed fixture (e.g. a complete 2-work × 2-performer panel). Enumerate every resampling-index combination and calculate the exact finite bootstrap distribution independently of the production sampler. Compare:
- row and column multiplicities;
- joint cell weights;
- equal-work/equal-performer estimand handling;
- paired baseline/candidate deltas;
- percentile extraction and degenerate/zero-effect behavior.

Include an independently hand-checked fixture or independent reference implementation. A test that compares a function with itself is not an oracle.

## 7. Method-selection policy

- Work-disjoint design with complete identities and explicit independence basis → #14019 is conditionally admissible.
- Crossed performer or session axis → refuse #14019 intervals; continue with descriptive estimates until a crossed method passes simulation.
- Missing / unstable pseudonyms, ambiguous multi-performer membership, unknown session reuse, or undefined sampling frame → descriptive-only.
- Few-cluster or highly unbalanced axis → use only a method whose small-sample behavior is demonstrated in matching simulations.
- If no method is calibrated for the observed design, change recruitment to a defensible independent design or do not make a population claim.

No single numeric cluster count substitutes for this rule.

## 8. Claim ceiling

Synthetic calibration establishes only that a named method behaves as expected under the declared data-generating processes. It does not authenticate musicians, establish representativeness or exchangeability, validate musical annotations, prove human musical understanding, or demonstrate model superiority. Those are separate evidence gates owned by the sampling, annotation, rights and held-out model-evaluation workstreams.
