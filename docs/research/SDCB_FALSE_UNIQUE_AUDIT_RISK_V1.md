# Selective-audit false-unique risk: independent reference v1

Status: analytic reference / public known-answer fixture. This is not production policy code, a qualified SOWRC implementation, or an empirical Symthaea result.

## Purpose

This fixture gives an independent, dependency-free finite-cohort reference for issue [#14130](https://github.com/Luminous-Dynamics/symthaea/issues/14130). It distinguishes:
1. Pointwise structural ambiguity: distinct parameterized mechanisms induce the same complete observation law under the protocol actually executed.
2. Composite-family overlap: allowed nuisance-parameter sets for mechanism families intersect exactly in observation-law space, even when a nominal parameter pair is distinguishable.
3. Finite-budget ambiguity: distinct laws cannot be reliably discriminated at the frozen sample/design budget and declared power criterion.

It contains six episodes, exact dyadic Bernoulli inclusion probabilities, three nested mechanism-set operating points, and three nested family-set operating points. Mechanism and family singleton denominators intentionally differ.

## Estimands

For mechanism-instance output A[t,j], family output B[t,j], and evaluator-only compatibility sets G_struct[t], G_budget[t], and G_family[t], define:
- C_mech[t,j] = 1 when A[t,j] is one substantive mechanism instance.
- C_family[t,j] = 1 when B[t,j] is one substantive family.
- M_struct = 1{G_struct is not a subset of A}.
- M_budget = 1{G_budget is not a subset of A}.
- M_family = 1{G_family is not a subset of B}.
- F_struct = C_mech * M_struct.
- F_budget = C_mech * M_budget.
- F_family = C_family * M_family.

Use separate frozen commitment rates c_mech = sum(C_mech)/N and c_family = sum(C_family)/N. For each positive denominator, conditional false-unique risk is the corresponding full-horizon joint risk divided by its matching commitment rate. A zero denominator is NO_SINGLETON_COHORT, never a zero-risk PASS.

A generating mechanism being present in the candidate set is weaker than compatibility-set containment. A nominal parameter pair being distinguishable is weaker than proving the allowed composite families are disjoint.

## Exact link to #14122 SOWRC

For a fixed pre-registered operating point, each F[t] is a bounded binary loss. Encode its nonconformity score as r[t]=1 when F[t]=1 and r[t]=0 otherwise. At the one predeclared threshold lambda=1/2, the SOWRC loss 1{r[t]>lambda} is exactly F[t].

Use the existing #14122 pointwise martingale-mixture upper boundary with the #14124/#14126 observed-label indicator and predictable q[t]. Keep weights a[t]=1 and the full event horizon; do not filter the sequence down to singleton cases. The conditional targets are met only when:
- U_struct(1/2; delta_struct) <= alpha_struct * c_mech;
- U_budget(1/2; delta_budget) <= alpha_budget * c_mech;
- U_family(1/2; delta_family) <= alpha_family * c_family.

Allocate delta over all applicable endpoints before labels. If #14122 exposes only a selected threshold, reuse/expose its pointwise U calculation rather than duplicating the SOWRC mathematics. This reference does not implement the SOWRC boundary; it independently checks the estimand arithmetic and Horvitz–Thompson expectation.

For nested operating-point selection, use monotone compatibility-set miss losses M, not direct false-unique losses F. Direct F can be non-monotone: an empty set becomes a wrong singleton and then a larger set. Nested selection still requires the exact theorem assumptions, simultaneous error allocation, and a safe terminal set.

## Known answers

At lambda index 0:
- Mechanism singleton count = 4/6.
- Pointwise-structural false singletons = 3/4 (conditional rate 0.75).
- Finite-budget false singletons = 4/4 (conditional rate 1.0).
- Family singleton count = 3/6.
- Composite-family false singletons = 3/3 (conditional rate 1.0).

At lambda index 1:
- Mechanism singleton count = 2/6; structural and finite-budget false-singleton counts are each 1/2.
- Family singleton count = 2/6; family false-singleton count is 1/2.

At lambda index 2:
- Both mechanism and family singleton denominators are zero. All conditional risk results must be withheld as NO_SINGLETON_COHORT.

Episode e2 is pointwise structurally identifiable at its nominal parameter point but has both a larger finite-budget compatibility set and a composite-family overlap set. Episode e6 demonstrates direct structural false-unique loss 0 -> 1 -> 0 across nested sets, while its compatibility-set miss loss is 1 -> 1 -> 0.

## Exact Bernoulli-mask enumeration

The fixture uses q[t] = q_num[t]/65536 with numerators 16384, 32768, 49152, 65536, 32768, 49152. All inclusion probabilities are strictly positive and dyadic. The Rust reference enumerates all 64 independent Bernoulli masks using their actual product probabilities and verifies that the expected Horvitz–Thompson estimate equals the complete-cohort joint risk for each endpoint and operating point.

This is a small independent-design arithmetic check, not a test of the production #14124 2^64 probability grid, not history-adaptive q, and not a live audit.

## Run

```bash
bash scripts/research/run_sdcb_false_unique_risk_tests.sh
```

The harness uses only rustc and the Rust standard library. It has no Cargo dependencies.

## Negative controls and limits

Reject or withhold claims when: q=0; a singleton mask changes after labels; a family ID is substituted; audit selection depends on the hidden current outcome; a selected label is missing/revoked/unqualified; a finite nuisance grid is presented as a continuous-family proof; or Bernoulli selection stops at a realized quota.

The fixture is synthetic. It does not prove SOWRC, authenticate the custodian, establish conditional ignorability, validate an adjudicator, prove continuous-family disjointness beyond the supplied analytic witness, establish drift control, qualify a live research policy, or demonstrate SOTA.
