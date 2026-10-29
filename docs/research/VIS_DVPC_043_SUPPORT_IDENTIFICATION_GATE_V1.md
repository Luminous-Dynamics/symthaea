# VIS-DVPC-043 — Support identification and fail-closed OPE decision gate

**Status:** Rust implementation with frozen controls; exact-head CI must pass before it is qualified.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8873

## Paradigm shift: an estimate is not evidence of identifiability

Off-policy evaluation begins with a support contract, not an estimator. A target policy may assign probability to an action that the logger never takes. In that case, the observed logged data do not identify the counterfactual reward for that action without additional assumptions. A narrow interval from a downstream routine cannot fix missing support.

This module is estimator-agnostic: it computes support status before MOPE, IPS, DR, HCOPE, or a policy-selection method is allowed to publish a numeric conclusion or recommend promotion. It separates three questions:

1. **Identifiability:** is every action with positive target-policy mass covered by positive logging probability?
2. **Statistical overlap:** even if support exists, how concentrated can the importance ratios be?
3. **Decision eligibility:** should this evidence be allowed to authorize a deployment decision?

A positive support answer does not mean the estimate is accurate. The gate only grants eligibility for a separate estimator and decision review; it does not authorize deployment by itself.

## Exact quantities

Given categorical logging probabilities h(a) and target probabilities pi(a), the implementation validates that both are finite probability vectors of equal length and sum to one within 1e-9.

Unsupported target mass:

m_unsupported = sum_a pi(a) * 1[h(a)=0].

If this is greater than 1e-12, status is NOT_IDENTIFIED. The module suppresses maximum ratio, second moment, population ESS, and downstream value/interval disclosure. It never divides by zero or fabricates an importance weight for an unsupported target action.

When support is complete, compute:

w(a) = pi(a)/h(a),
E_h[w²] = sum_a pi(a)² / h(a),
population ESS fraction = 1 / E_h[w²].

The maximum supported w is also reported. A low maximum-importance-ratio cap or low population ESS threshold marks the case WEAK_SUPPORT and returns INSUFFICIENT_EVIDENCE. WEAK_SUPPORT does not claim the parameter is mathematically unidentified; it says this gate will not authorize a decision from overlap alone. The current thresholds (max weight 20; population ESS fraction 0.05) are explicit benchmark policy knobs, not universal constants.

Empirical ESS, (sum_i w_i)² / sum_i w_i², is reported as a separate diagnostic and explicitly does not prove population support. A sample can simply fail to contain rare actions.

## Frozen controls and known answers

| Control | Logger h | Target pi | Expected result |
|---|---|---|---|
| balanced | (0.5, 0.5) | (0.5, 0.5) | IDENTIFIED, max w=1, ESS fraction=1 |
| weak overlap | (0.01, 0.99) | (0.5, 0.5) | WEAK_SUPPORT, max w=50, E[w²]≈25.252525, ESS fraction≈0.0396 |
| near-zero overlap | (0.000001, 0.999999) | (0.5, 0.5) | WEAK_SUPPORT, ESS fraction≈0.000004 |
| zero support | (0, 1) | (0.5, 0.5) | NOT_IDENTIFIED, unsupported target mass=0.5; metrics suppressed |
| logger-zero action also target-zero | (0, 1) | (0, 1) | IDENTIFIED; no false NOT_IDENTIFIED alarm |
| invalid normalization | (0.4, 0.4) | (0.5, 0.5) | INVALID_INPUT, no decision |

The implementation verifies logged propensity ratios against the frozen probability vectors. A mismatch or an observed action with zero logging probability fails closed. Invalid shapes, nonfinite values, negative probabilities, or invalid gate thresholds return INVALID_INPUT.

## Status and consumer contract

- IDENTIFIED → ELIGIBLE_FOR_EVIDENCE_REVIEW. A downstream estimator may compute its own statistically valid interval; this gate does not authorize the decision itself.
- WEAK_SUPPORT → INSUFFICIENT_EVIDENCE. A value may still be estimated with explicit uncertainty, but this layer will not allow a deployment decision.
- NOT_IDENTIFIED → SUPPRESS_VALUE_AND_INTERVAL. Do not present a point estimate or confidence interval as if it were learned from these data.
- INVALID_INPUT → INVALID_INPUT_NO_DECISION. Suppress results.

This contract is intended to compose with the earlier VIS-DVPC-039 through 042 MOPE work. A zero-support policy must not look “safe” merely because its estimator returned a finite numeric value.

## Hard gates

The focused Rust workflow requires:
- exact known answers for the six frozen controls;
- unsupported target mass triggers NOT_IDENTIFIED and suppression;
- target-zero mass on unlogged actions is not a false alarm;
- weak / near-zero support cannot be decision-eligible;
- propensity-ratio mismatches and invalid inputs fail closed;
- empirical ESS is kept distinct from population ESS;
- pinned source/output SHA-256 receipt and exact-head test execution.

No production or state-of-the-art claim is made. These are deterministic contract and algebra tests; the next stage is integration with a shared policy-evaluation API and stress testing under changing/misspecified logging propensities.

## Research comparison context

SCOPE-RL implements Direct Method, trajectory/per-decision importance sampling, doubly robust and self-normalized estimators, along with Hoeffding, empirical-Bernstein, Student-t and bootstrap high-confidence intervals; its evaluation metrics include MSE, Regret@k, rank correlation, Type-I/II error and safety violations. This support gate is designed to sit before such estimators, not replace them:
https://scope-rl.readthedocs.io/en/latest/documentation/evaluation_implementation.html

Ryu, Kwon, Koppe & Jun (COLT 2025) study betting-based off-policy selection and variance-aware freezing. The gate provides an independent support/identification contract for comparing these selection methods:
https://proceedings.mlr.press/v291/ryu25a.html

A recent 2026 diagnostic paper argues that offline policy estimates can mislead when reward alignment and learnability are not assessed separately. This supports evaluating decision readiness beyond a lone value estimate:
https://arxiv.org/abs/2608.11560

This PR only builds the support/identification guard. It is not a complete OPE system, does not establish estimator coverage, and does not establish superior decision quality.
