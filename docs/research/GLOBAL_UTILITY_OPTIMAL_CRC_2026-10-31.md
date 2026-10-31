# Global utility-optimal CRC under exact finite-exchangeability constraints

**Date:** 2026-10-31  
**Status:** 8/8 local reference tests passed, including 40 random objective-weight vectors and a two-risk instance. This is a two-type synthetic optimization result, not a CIFAR experiment.

## Paradigm shift

Equal-share recertification improves a controller pointwise: it spends available slack to make each calibration state no more conservative, while preserving every hypothetical augmented-sample constraint. That is useful, but it does not generally solve a global application objective because valid controllers need not form a lattice and the optimal trade-off depends on how often each calibration state occurs.

For a finite observation alphabet, write the controller as one action a_D per calibration multiset D. For linear losses

    L_r(z,a) = g[r,z] * (1-a),   a in [0,1],

finite-exchangeability validity is equivalent to the finite family of constraints

    sum_z S_z * g[r,z] * (1-a_(S-e_z)) <= (n+1)*alpha_r

for every augmented multiset S of size n+1 and every risk r.

These constraints define a polytope. We can optimize a frozen utility over the entire polytope rather than follow a local equal-share update.

## New optimizer

The prototype solves:

    maximize sum_D w_D * (1-a_D)

subject to every augmented-multiset validity constraint and 0<=a_D<=1.

The reference weights w_D affect only the objective. They do not remove, relax or reweight any certificate constraint. Therefore, the resulting deterministic controller remains valid under every finite exchangeable law represented by this state space, even when the true law differs from the reference objective weights.

A second formulation maximizes worst-state utility, equivalent to minimizing max_D a_D when utility is 1-a_D. This is a maximin objective over arbitrary distributions on calibration states.

The implementation also supports multiple simultaneous risk vectors with a shared action and separate alpha budgets.

## Exact two-type result

Use n=2, two types H/L, g_H=1, g_L=0.5, alpha=0.5, and U(a)=1-a. Equal-share recertification converges to:

| Calibration multiset | Equal-share action |
|---|---:|
| LL (0,2) | 0.125 |
| HL (1,1) | 0.375 |
| HH (2,0) | 0.500 |

Under a binomial reference distribution with P(H)=0.1, the global LP returns (0, 0.5, 0.5). It passes every augmented-multiset constraint and achieves expected utility 0.905, compared with 0.82625 for equal-share: a 0.07875 (7.875 percentage-point) improvement in this toy utility.

| P(H) reference | Equal-share utility | Global LP utility | Gain |
|---:|---:|---:|---:|
| 0.10 | 0.82625 | 0.90500 | +0.07875 |
| 0.20 | 0.78000 | 0.82000 | +0.04000 |
| 0.50 | 0.65625 | 0.65625 | 0 |
| 0.80 | 0.55500 | 0.55500 | 0 |
| 0.90 | 0.52625 | 0.52625 | 0 |

The improvement is a trade-off, not pointwise dominance. When P(H)=0.1, the optimizer lowers action for LL, but increases action for HL. It spends less budget on common states and protects less frequent states with more conservative actions, while preserving all universal validity constraints.

For the distribution-free maximin objective, the optimal worst-state utility in this example is 0.5, matching the equal-share controller's worst-state utility. This is the correct contrast: reference-weighted optimization can exploit a declared prior over calibration states, while maximin optimization sacrifices that gain to protect the least favorable state.

## Assumption and governance requirements

The reference weights must be frozen before observing the calibration sample, or estimated from an independent dataset with a separate justification. Learning weights from the same calibration sample and then treating the resulting state-dependent controller as a fixed map is not covered by this prototype.

The certificate should record:
- the full action vector and canonical state ordering;
- all risk coefficients and alpha levels;
- the complete set of augmented states checked;
- the objective weights and their source/commitment time;
- the LP solver version, tolerances, primal residuals and dual optimality residuals;
- the model and loss digests, without publishing raw labels.

## Local validation

8/8 reference tests passed:
- all augmented-state constraints for reference-weighted solutions;
- the skewed-reference utility improvement;
- reference-weight sensitivity and the P(H)=0.5 equality case;
- maximin solution;
- a two-risk, three-type simultaneous-constraint instance;
- 40 random reference weight vectors, each followed by exact augmented-state verification;
- invalid inputs and negative controls;
- direct recomputation of expected utility from returned actions.

The numerical execution used an independent local reference copy implementing the committed LP equations. The exact GitHub blob has not yet been run in a separate clean checkout, and remote CI was not awaited.

## How this changes the research programme

We now have three distinct optimizers that should not be conflated:

1. **Equal-share recertification:** pointwise improvement, independent of a reference calibration-state distribution.
2. **Fixed-order coordinate sweep:** finite-domain admissibility, with order-dependent valid controllers.
3. **Global utility LP:** globally optimal for a predeclared reference utility and finite-state risk polytope, allowing explicit trade-offs between states.

Next, extend the optimizer to common-depth CIFAR with a hierarchy-specific specificity utility and the published type-count weights, then evaluate sensitivity to reference-weight misspecification. Keep a maximin comparator and the original equal-share method. The result would still be finite-exchangeability valid, but the utility claim would be conditional on the frozen objective weights.

