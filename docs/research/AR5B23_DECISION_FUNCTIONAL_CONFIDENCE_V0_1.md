# AR-5B23: Decision-Functional Confidence Calibration v0.1

**Status:** Rust implementation draft; tests authored but not executed.  
**Tracking:** https://github.com/Luminous-Dynamics/symthaea/issues/9951  
**Implementation:** `crates/domains/symthaea-architecture-engineering/src/decision_functional_confidence.rs`  
**Base dependency:** AR-5B21 / PR #9925 at `2727cc9e0bbcaef9bc42ca9864ecafa88b97630f`.

## 1. Scope

V1 estimates a predeclared finite vector of bounded expectations consumed by the existing policy-frontier solver:

- null rejection probability;
- alternative rejection probability/power;
- expected evaluator cost;
- expected decision utility.

It does not estimate the full path distribution. Its statistical guarantee applies only to the declared functionals and the stated calibration target law. A change orthogonal to those functionals may remain invisible by design.

## 2. Finite-sample bounds

For independent sampling units (X_1,ldots,X_n) and a predeclared function (g(X)in[L,U]), the empirical mean is:

[
widehatmu = rac{1}{n}sum_{i=1}^{n} g(X_i).
]

For a one-sided failure probability (delta), V1 uses Hoeffding:

[
r=(U-L)sqrt{rac{log(1/delta)}{2n}}.
]

The upper and lower bounds are clipped to the known support:

[
U_g=min(U,widehatmu+r),qquad
L_g=max(L,widehatmu-r).
]

The implementation caps the radius at the support width when the bound is already uninformative. It rejects empty samples, non-finite values, support violations and non-finite support widths.

### Simultaneous family allocation

For (P) candidate policies and (M_0) null models, the null failure budget (delta_0) is Bonferroni-allocated as:

[
delta_{0,i}=rac{delta_0}{P M_0}.
]

For (P) policies, (M_1) alternatives and three alternative functionals (power, cost and utility), the alternative budget is:

[
delta_{1,i}=rac{delta_1}{3 P M_1}.
]

No independence between rows or metric coordinates is needed for the union bound. Each row's own sampling-unit assumptions still need to hold.

The null-side confidence event is used to spend Type-I error:

[
alpha_{mathrm{cond}} =
rac{alpha-delta_0}{1-delta_0},qquad delta_0<alpha.
]

On the simultaneous null-coverage event, the robustified frontier has worst-case null size no larger than (alpha_{mathrm{cond}}). Outside that event, the conservative rejection bound is one. The report reconstructs:

[
delta_0+(1-delta_0),alpha_{mathrm{cond}}leqalpha.
]

This is conditional on calibration being independent of the confirmatory stream and sharing its target law by design. A same-design digest is an identity binding, not proof of stationarity, exchangeability, provenance or equality of the actual data distributions.

## 3. Sampling and identification contract

The input records the evidence mode:

- `RANDOMIZED_POLICY_EXECUTION`;
- `SIMULATOR_GROUND_TRUTH`;
- `OFF_POLICY_ESTIMATE` (rejected by V1).

For randomized execution, V1 requires positive assignment probability for every candidate policy arm, assignment probabilities summing to one within each model family, and globally disjoint sampling-unit IDs across all randomized policy/model evidence rows. Repeated observations must be aggregated to the declared independent sampling unit before entry. Unmodeled interference is rejected by the V1 declaration contract.

For simulator ground truth, paired/common-random-number experiments may reuse seed/unit identities across policy rows because the evaluator observes counterfactuals inside the simulator. This is not an assertion that real-world counterfactual outcomes are observed.

V1 rejects off-policy estimates and importance weights. Positivity, weight truncation and finite-sample weighted confidence sets require a separate implementation.

## 4. Target-law boundary

V1 accepts only `SAME_TARGET_LAW_BY_DESIGN` and requires identical calibration/confirmation design digests. It rejects:

- `UNKNOWN` target relation;
- `BOUNDED_TRANSPORT` (not yet implemented);
- unequal design identities.

A digest and caller declaration do not prove actual target-law equality. If the environment, eligibility, assignment, policy implementation, metric definition or population changes, use an independently justified transport bound and spend its confidence failure in the total null-side budget before using the alpha formula. V1 does not implement this transport composition.

## 5. Optimizer integration

The calibrator:
1. binds to the exact canonical `PolicyFrontierPlan` digest;
2. requires every policy × null/alternative model pair exactly once;
3. binds alternative utility/cost samples to the policy's existing utility-contract digest;
4. checks nominal cost/utility receipts lie inside the declared path-level support ranges;
5. constructs simultaneous one-sided bounds;
6. replaces the nominal size/power/cost/utility coefficients in a cloned plan;
7. reduces the plan's conditional significance level using the null-family delta;
8. invokes the unchanged `analyze_policy_frontier` solver;
9. reports confidence components, source/target identities, robustified-plan digest, frontier and unconditional Type-I reconstruction.

No existing nominal plan or optimizer semantics are mutated.

## 6. Tests authored

The module currently defines tests for:
- exact family confidence allocation and unconditional Type-I reconstruction;
- zero-width support / zero radius;
- unknown and bounded-transport target relations;
- independent streams with mismatched target design;
- off-policy rejection;
- calibration/confirmation dependence and non-independent sampling declaration;
- randomized-arm support and assignment-probability accounting;
- duplicate/reused sampling units;
- missing policy/model rows;
- NaN, infinite and out-of-range outcomes;
- null coverage failure consuming the entire alpha budget;
- row-order canonicalization;
- report tampering.

These tests are authored, not run. No Rust compile, format, Clippy, unit-test, CI or performance result is claimed.

## 7. Known limitations and next work

- Hoeffding is the transparent finite-sample baseline; it may be conservative in low-variance settings.
- The vector empirical-Bernstein comparator must be implemented from its exact 2-smooth Banach-space assumptions; do not substitute a scalar variance estimate or assume arbitrary dependence is supported.
- The separate ICML 2026 sharp empirical-Bernstein variance result assumes constant conditional mean and variance for its sequential guarantees and has a Hilbert-space extension. It is a different comparator, not a drop-in replacement.
- Bounded transport, clustered/Markov data, off-policy estimation, informative censoring, cross-policy counterfactual identification and adaptively created policy/function families remain unsupported.
- The included report verifier deterministically reruns this implementation. An independently implemented verifier remains a qualification requirement.
- A declaration or SHA-256 digest does not prove the asserted data-generating assumptions.

## 8. Research references

1. Martinez-Taboada & Ramdas (2026), *Empirical Bernstein in Smooth Banach Spaces*, Annals of Applied Probability 36(3), 2199–2236. https://doi.org/10.1214/25-AAP2277
2. Martinez-Taboada & Ramdas (ICML 2026), *Sharp Empirical Bernstein Inequalities for the Variance of Bounded Random Variables*, PMLR 306. https://proceedings.mlr.press/v306/martinez-taboada26a.html
3. Fontem & Ji (2026), *Distributionally robust optimization with generalized total variation ambiguity sets*, EJOR 328(3), 894–911. https://doi.org/10.1016/j.ejor.2025.09.024
