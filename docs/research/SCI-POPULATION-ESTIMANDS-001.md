# SCI-POPULATION-ESTIMANDS-001 — Generalized benchmark accuracy and hierarchical uncertainty

**Status:** Protocol proposal and implementation plan. No statistical backend, empirical estimate, or generalized-performance claim is implemented or qualified by this document.

**Tracking:** [Issue #9433](https://github.com/Luminous-Dynamics/symthaea/issues/9433).

## 1. Purpose and boundaries

This is the cross-benchmark inference layer for the Symthaea evaluation programme. It composes with individual judge metrology (#9022), judge score geometry (#9130), factorial bias measurement (#9243), mutation validity (#9270), reference assurance (#9081), and the correlated judge ensemble programme (#9431).

It does not duplicate those evaluators. It addresses a different question: what population-level conclusion, if any, is justified by the items, families, sources, judges and time periods actually sampled?

A benchmark score is conditional on its measurement instrument, sampling frame, task mixture, reference protocol and decision rule. The statistical report must expose these conditions rather than turn a fixed-set score into a universal model property.

## 2. Distinct estimands

### 2.1 Finite benchmark performance

For the exact frozen item set B:

\[
\widehat{\mu}_{B}=\frac{1}{|B|}\sum_{i\in B}Y_i.
\]

This is a descriptive result for B. If every item in B was evaluated, uncertainty about the finite-set arithmetic mean is not the same as uncertainty about future tasks. Any uncertainty interval must state what source of randomness it represents (e.g. judge stochasticity or repeated measurements).

### 2.2 Within-frame generalized performance

Declare a target task-family distribution P(T) and an item distribution P(I|T):

\[
\mu_{P}=\mathbb{E}_{T\sim P(T),\ I\sim P(I\mid T)}[Y(T,I)].
\]

This estimand requires a defensible sampling frame or an explicit transport assumption. A convenience benchmark is not automatically representative of production tasks.

### 2.3 Cross-family generalization

Performance on previously unseen families is a separate target. Use family-level holdouts and enough independently sampled families to support the intended inference. Random item splits within already-seen families do not establish cross-family generalization.

### 2.4 Temporal transport

Future-time performance is separate from random holdout. Freeze a temporal cutoff and evaluate on later tasks without using their labels for model, judge, rubric, threshold or weight selection.

### 2.5 Decision risk

For policy π, action a, outcome y and a predeclared loss matrix L(a,y):

\[
R_{P}(\pi)=\mathbb{E}_{X,Y\sim P}\left[\sum_a \pi(a\mid X)L(a,Y)\right].
\]

Accuracy and decision risk are not interchangeable. False-positive and false-negative costs, abstention consequences and operational resource constraints must be declared before results are observed.

## 3. Sampling and lineage contract

The study plan must freeze:
- target population, sampling frame, strata, inclusion/exclusion rules and target weights;
- task-family, source, item, semantic-cluster, modality, difficulty and time identifiers;
- judge, evaluator, generator, rubric, prompt and policy revisions;
- calibration, policy-selection, validation and held-out splits;
- duplicate/near-duplicate clustering and lineage rules;
- planned repeats, stopping rules, missingness treatment and resource budget;
- estimands, contrasts, statistical model, diagnostics and multiplicity policy.

Split by the highest shared source/cluster lineage before creating item variants. Paraphrases, repeated calls, generated mutations and tasks derived from a shared source are not independent population samples. Preserve every attempted trial, timeout, retry, abstention, missing outcome and exclusion with its reason and denominator.

Where probability sampling is used, retain inclusion probabilities and apply predeclared weights. Where probability sampling is unavailable, state the transport assumptions and keep conclusions conditional on the observed frame.

## 4. Outcome-appropriate statistical models

Do not force every outcome into one model.

- Binary correctness or critical failure: logistic mixed-effects models.
- Ordinal rubric ratings: cumulative-link mixed-effects models.
- Continuous scores: linear or robust mixed-effects models after score-scale diagnostics.
- Paired preference decisions: Bradley–Terry or logistic mixed-effects models.
- Counts and rates: appropriate Poisson, negative-binomial or binomial models with explicit exposure denominators.

A simple binary starting model may be:

\[
\operatorname{logit}\Pr(Y_{fij}=1)=
\beta_0+\beta_{\mathrm{system}}+\beta_{\mathrm{condition}}+
u_f+v_{fi}+b_j.
\]

Here u_f, v_fi and b_j represent declared family, item/cluster and judge effects. The appropriate structure depends on the actual crossed or nested design. Rubric, prompt, source, generator, modality and time effects or random slopes should be added only when supported by the sampling design and identifiable sample sizes.

A model formula is not a substitute for adequate independent sampling. Report the number of independent families and clusters, not merely the raw number of calls.

## 5. Uncertainty, diagnostics and fallback

Compare finite-benchmark summaries with population estimates. Use a family/cluster bootstrap at the actual independently sampled unit or a preregistered hierarchical model. Never bootstrap individual judge calls as independent tasks when the sampling unit is the item, source or task family.

Report:
- convergence, singular fits, separation and model warnings;
- variance components and intraclass correlations;
- leave-one-family-out influence and sensitivity;
- outcome-specific residual and calibration diagnostics;
- missingness, weighting and selection sensitivity;
- prior/penalty sensitivity for sparse hierarchical models;
- effective independent sample counts;
- interval coverage and Type-I error in known-answer simulations.

When there are too few independent families, when variance components are not identifiable, or when diagnostics show material misspecification, use a predeclared conservative fallback and narrow the claim. Do not select the narrowest interval as the preferred method.

## 6. Synthetic qualification benchmark

Before empirical use, test simulated populations with known truth and:
- item and task-family variance components;
- crossed judge and prompt effects;
- family × system interactions;
- unequal family prevalence and unequal inclusion probabilities;
- repeated calls and within-cluster dependence;
- missing-not-at-random outcomes and selective abstention;
- duplicate semantic clusters and split leakage;
- temporal and leave-family-out distribution shifts;
- rare critical failures, including zero events in small samples.

Verify estimand recovery, bias, interval coverage, Type-I error, power under preregistered effects, and sensitivity to omitted or misspecified variance components. Include negative controls demonstrating how naive item-level standard errors become overconfident under clustering.

Monte Carlo tolerances, simulation counts and expected coverage ranges must be fixed in advance and reported. Synthetic success qualifies the analysis implementation under those data-generating conditions; it does not establish empirical Symthaea performance.

## 7. Method comparison

On identical synthetic populations with known truth, compare:
1. finite-set arithmetic summaries and conventional item-level intervals;
2. cluster-aware bootstrap;
3. an appropriate hierarchical / mixed-effects model;
4. a conservative finite-sample fallback when model assumptions fail.

Evaluate bias, interval coverage, Type-I error, power, robustness to imbalance, calibration and decision error. A narrower interval is not evidence of a better method unless coverage remains valid.

For empirical comparisons, preserve paired tasks and matched sampling frames. Use the same external test population for all systems and report task-family and temporal strata separately. An estimate based on a different task mixture is not a fair head-to-head comparison.

## 8. Implementation sequence

1. Freeze this protocol and the target-population / sampling plan.
2. Add strict study and sampling-frame schemas plus exact analysis-plan commitments.
3. Build a deterministic synthetic population generator with known crossed effects and hostile missingness / leakage conditions.
4. Implement a Rust-first manifest auditor for sampling units, weights, lineage, split leakage, denominators and immutable receipts.
5. Use a pinned, reviewable statistical backend for hierarchical model fitting and diagnostics; do not pretend a bespoke arithmetic routine is a validated GLMM implementation.
6. Independently check backend outputs against small exact cases and synthetic populations with known parameters.
7. Add a report validator that checks estimand compatibility, sampling claims, model diagnostics, uncertainty and claim ceilings.
8. Integrate estimates into judge, rubric, mutation, metamorphic and ensemble scorecards without allowing those components to self-certify sampling or model assumptions.

The Rust component should own deterministic data contracts, receipts and independent checks. The statistical engine should be separately versioned and reproducible.

## 9. Qualification gates

- Synthetic known-answer estimands and malformed sampling manifests are handled exactly.
- Interval coverage and Type-I error meet preregistered Monte Carlo tolerances across balanced, imbalanced and clustered designs.
- No population-performance claim is accepted from a convenience sample without explicit, defensible transport assumptions.
- Leave-family-out and temporal sensitivity are reported for claims involving new families or future performance.
- Non-identifiable, underpowered or materially misspecified models fail closed or produce a clearly bounded result.
- All denominators, exclusions, weights, missingness and effective independent sample counts are present.

Power, minimum worthwhile effect and sample size must be derived from the intended use case and pilot variance structure; no universal numeric target is asserted here.

## 10. Research anchor

NIST AI 800-3 (February 2026), *Expanding the AI Evaluation Toolbox with Statistical Models*, distinguishes fixed-benchmark accuracy from generalized accuracy and demonstrates the value of statistical models that account for task-level variation:
https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models

The NIST result motivates this work but does not imply that a GLMM alone fixes biased sampling, bad references or distribution shift.

## 11. Claim boundary

A valid model fit does not prove that the sampling frame represents deployment. A narrow interval does not cure selection bias. A passing synthetic simulation does not establish real-world performance. Generalized-performance claims require a declared target population, defensible sampling or transport assumptions, adequate independent families, model adequacy and independent replication.
