# EGDDM ShiftBench D6.17.1 — Partial identification of dependent evidence channels

**Date:** 2026-11-17  
**Status:** evaluator extension; synthetic known-answer fixture; no Rust execution or empirical calibration claim  
**Related:** [D6.17 issue #12529](https://github.com/Luminous-Dynamics/symthaea/issues/12529), [D6.17.1 issue #12560](https://github.com/Luminous-Dynamics/symthaea/issues/12560), [D6.17 draft PR #12558](https://github.com/Luminous-Dynamics/symthaea/pull/12558).

## Research question

Can an agent use evidence from correlated or incompletely characterized sources without either multiplying them as independent or discarding all of their information?

D6.17 v0.1 initially refused to condition on a multi-source trace unless an evaluator-held exact-set factorization certificate existed. This was safe but coarse. D6.17.1 adds a partial-identification fallback based on sharp Fréchet bounds for joint observed events.

## Mathematical contract

For hypothesis \(h_i\), suppose the observed event from channel \(j\) has independently qualified marginal probability:

\[
\ell_{ij} \leq P(E_j\mid h_i) \leq u_{ij}.
\]

For \(n\) channels with no dependence information, the sharp Fréchet bounds on the joint event are:

\[
\underline q_i =
\max\left(0,\sum_{j=1}^{n}\ell_{ij}-(n-1)\right),
\qquad
\overline q_i =
\min_j u_{ij}.
\]

This is the intersection probability of the actual observed outcome events. It is not a claim that the channels are independent. It uses only marginal interval information and is sharp over all joint distributions consistent with those marginals.

For a finite precision scale of 10,000 basis points, lower endpoints round down and upper endpoints round up. The lower support bound is subsequently optimized over the bounded-simplex prior. Zero lower support remains an abstention condition, not zero risk.

### Dependence groups

Group channels that share a plausible common cause: optical path, power, clock, firmware, calibration regime, test fixture, upstream provider, network, operator or data source.

1. Within each dependence group, use Fréchet bounds over all acquired channel outcomes.
2. Across groups, multiply those joint intervals only if the evaluator holds a certificate for the exact selected factor IDs and exact group set.
3. The certificate must state mutual conditional independence, include hypothesis in its conditioning scope, identify its authority and action scope, and be valid at each observation and at the decision cutoff.
4. Without such a certificate, use one unrestricted Fréchet bound over all selected channel events. Do not multiply the groups.
5. Pairwise independence certificates never imply mutual independence for a three-or-more-group factorization.

The certificate qualifies independence between group-level observations, not independence of individual sensors inside a dependence group.

### Nominal baseline semantics

The point-product likelihood remains available only as a naive independence comparator. It is useful to calculate an optimism gap but does not authorize an action. The safety decision uses the combined worst-case prior and joint-likelihood ambiguity set.

This preserves apples-to-apples comparison with D6.16 and makes it visible when a naive product is overconfident.

## Synthetic fixture results

The existing 13-trace fixture gains a Fréchet partial-dependence fallback.

**Two optical sensors sharing one path**

- Naive independence baseline posterior risk: 0.12%.
- Fréchet lower evidence-support bound: 34.50%.
- Combined worst-case posterior risk: 11.54%.
- Registered reversible envelope: 7%.
- Result: reject the action, while retaining the bounded evidence rather than discarding it.

**Three factors with pairwise-only certificates**

The fixture includes all three pairwise certificates across the selected groups, with exact factor IDs and conditioning scope, but no three-way mutual-independence certificate. The evaluator does not multiply the three factors. It uses the unrestricted Fréchet intersection bound:

- Naive independence baseline risk: 0%.
- Combined worst-case posterior risk: 3.57%.
- Registered reversible envelope: 7%.
- Result: the action is justified under the declared conservative model.

This illustrates an important property: refusing an unjustified independence assumption does not necessarily force abstention. Some decisions remain justified under the broader partial-identification set.

## Validation

- Candidate/oracle copies of actions, priors, option availability and factor likelihoods are structurally checked.
- Every marginal outcome partition has feasible lower/upper bounds and nominal probabilities summing to 10,000 bps.
- Every calibration record covers the factor and every factor references an evaluator-held calibration.
- Factorization certificates bind exact factor IDs, groups, conditioning scope, authority, actions and validity.
- Expected report includes dependence mode, robust risk, support lower bound and prior-/likelihood-only ablations.
- Independent randomized check: 10,000 generated joint distributions over 2–6 binary channels; the Fréchet interval contained every generated joint-event probability under marginal intervals containing the generated marginals. This checks interval coverage on generated cases, not the Rust binary or a proof of the implementation.
- Rust tests are authored but not executed: no Rust toolchain is available in this environment.

## Required benchmark dimensions

Report by source type, consequence class and channel count:
- joint-likelihood interval coverage on held-out joint outcomes;
- interval sharpness / proper interval score, so vacuous bounds do not win;
- worst-case posterior risk and false-safe rate;
- optimism gap from the naive independence baseline;
- useful action coverage and unnecessary abstention;
- regret relative to the exact joint-channel oracle on small enumerable cases;
- cost / latency growth as channel count and dependence groups increase.

Development controls should cover 2–6 channels, 2–4 groups, multiple common-cause structures, unknown dependence, pairwise-only certificates, calibration uncertainty shared across sources and coordinated compromise.

## Claim ceiling

Fréchet bounds are valid for the intersection of events when only marginal probabilities are known. They do not identify the full joint distribution, detect source compromise, prove the supplied marginal intervals cover reality, or guarantee a finite useful posterior under zero support. This prototype does not solve arbitrary dependence with side constraints; that remains future work involving joint-distribution polytopes, common-cause latent variables and copula ambiguity classes.
