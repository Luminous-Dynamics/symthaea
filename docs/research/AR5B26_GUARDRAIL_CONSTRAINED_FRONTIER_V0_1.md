# AR-5B26 — Guardrail-Constrained Exact Policy Frontier v0.1

**Status:** implementation draft; Rust tests authored but not executed.  
**Tracking:** [Issue #10006](https://github.com/Luminous-Dynamics/symthaea/issues/10006)  
**Implementation:** **crates/domains/symthaea-architecture-engineering/src/guardrail_constrained_policy_frontier.rs**  
**Solver extension:** **crates/domains/symthaea-architecture-engineering/src/policy_frontier_lp.rs**  
**Verifier correction:** **crates/domains/symthaea-architecture-engineering/src/noncompensatory_guardrails.rs**  
**Branch:** **research/ar5b26-guardrail-constrained-frontier-v0.1-20261102**  
**Dependency base:** AR-5B25 / PR #10005, whose base includes AR-5B23 / PR #9999.

## 1. Paradigm shift

The previous sequence was:

1. Optimize utility over the policy family.
2. Inspect the selected mixture for mandatory guardrail violations.
3. Block the result if unsafe.

That is insufficient when a different mixture would be both feasible and useful. It also invites unsafe post-selection trade-offs.

This implementation changes the feasible decision space itself:

\[
\max_{q,z} z
\]

subject to the existing frontier constraints and every mandatory confidence-bound guardrail. Utility is optimized **only inside** the robustly feasible region.

For policy weights \(q_i \ge 0\), \(\sum_i q_i=1\), a declared upper-bound guardrail is:

\[
\sum_i q_i U_{i,g,s,m} \le \tau_{g,s,m}
\]

A lower-bound guardrail is:

\[
\sum_i q_i L_{i,g,s,m} \ge \tau_{g,s,m}
\]

The LP implementation stores all inequalities in \(a^\top q \le b\) form. Upper guards use \(a_i=U_i\); lower guards use \(a_i=-L_i\) and \(b=-\tau\).

An unavailable, non-estimable, missing, or under-sampled policy/requirement cell is **not** assigned a zero confidence bound. The policy is excluded from the feasible support with the explicit equality \(q_i=0\), and the report records the exact reason. If no eligible mixture remains, the LP is infeasible and returns an error rather than a permissive result.

## 2. Implementation shape

### 2.1 Generic hard inequalities in the exact LP

**policy_frontier_lp.rs** now exports:

- **FrontierPolicyCoefficient { policy_id, coefficient }**
- **FrontierAdditionalLinearConstraint { constraint_id, policy_coefficients, upper_bound }**
- **analyze_policy_frontier_with_constraints(plan, constraints)**

The original **analyze_policy_frontier** API delegates to the new method with an empty additional-constraint list, preserving existing callers.

Every coefficient is keyed by stable policy ID rather than a positional vector. The solver rejects missing/duplicate/unknown IDs and reorders coefficients after canonicalizing the plan. This prevents a reordered policy list from silently binding a safety coefficient to the wrong policy.

Additional inequalities are validated for unique non-empty constraint IDs and finite coefficients/bounds. They are appended to the same active-constraint set before exhaustive basis enumeration. Every candidate is also checked against every additional inequality after reconstruction.

The objective and existing constraints remain unchanged: worst-case alternative utility, composite-null Type-I ceilings, and optional alternative-specific expected evaluator-cost caps. The existing deterministic tie-break order is preserved.

### 2.2 Typed guarded-frontier wrapper

**guardrail_constrained_policy_frontier.rs** introduces:

- **GuardrailConstrainedFrontierRequest**
- **PolicyEligibilityRecord**
- **GuardrailConstrainedRequirementResult**
- **GuardrailConstrainedFrontierReport**
- **analyze_guardrail_constrained_frontier**
- **verify_guardrail_constrained_frontier_report**

The wrapper binds the base policy-frontier plan to the existing typed global/local guardrail evidence request. It requires exact policy-family identity, a structurally valid applicable requirement roster, the same-target-law-by-design relation, independent sampling units, the no-cross-unit-interference declaration, the linear-mixture estimand declaration, and the simultaneous confidence-family receipt digest.

For every frozen requirement, the optimizer uses the bound direction and confidence limit. It emits the policy exclusions, selected mixture, all requirement-level mixture confidence intervals, signed slack, active guardrail identifiers, evidence digests, and an outer deterministic report digest. The report stores the frontier's experiment Type-I significance level separately from the guardrail simultaneous failure probability. It also embeds immutable solver settings (algorithm ID, policy ceiling, active-basis ceiling, and feasibility/pivot/objective tolerances), and the verifier checks those settings against the implementation constants.

The embedded **simultaneous_coverage_receipt_sha256** binds an evidence artifact; **a digest is not proof that the artifact is valid, complete, or statistically calibrated**. The calibration producer and its coverage proof remain separately governed.

### 2.3 Independent post-selection check

After optimization, the result is passed to the existing fixed-mixture **analyze_noncompensatory_guardrails** and **verify_noncompensatory_guardrail_report** functions. The selected mixture is expanded to the full original policy roster, with explicit zero weights for excluded policies.

The fixed-mixture verifier was also corrected so an exactly zero-mass policy does not poison the selected mixture merely because it lacks one of the guardrail bounds. That policy's exclusion is explicit in the constrained optimizer report; it cannot receive positive probability.

**verify_guardrail_constrained_frontier_report** does **not** call the optimizer or its internal feasibility routine. It independently reconstructs:

- selected support and probability normalization;
- every composite-null Type-I constraint;
- alternative-specific power, cost, and utility values;
- worst-case utility, cost, power, null size, entropy, and alpha slack;
- every selected policy's guardrail evidence eligibility;
- each mixture lower/upper confidence bound and signed threshold slack;
- the independent fixed-mixture guardrail report and outer report digest.

This is a feasibility and integrity checker, **not an independent optimality proof**. An external LP oracle is still required to check optimality independently.

## 3. Statistical and semantic boundaries

V1 only accepts expected-value guardrails that are linear under the declared ex-ante policy mixture. It does not convert those statements into pathwise safety.

Not supported by this LP schema:

- CVaR, quantiles, maxima, tail probabilities, or distributional constraints;
- nonlinear ratios or metrics whose denominators change with the policy;
- cross-policy interactions, interference, or adaptive mixture-dependent estimands;
- transport between calibration and confirmation target laws;
- any claim that the declared guardrail catalogue captures every relevant harm.

The current evidence request requires **SAME_TARGET_LAW_BY_DESIGN**. Unknown or shifted target relations remain non-qualifying until a transport certificate and its confidence event are implemented. The guardrail false-safe error budget remains separate from the experiment's null Type-I budget.

The solver uses floating-point arithmetic and fixed feasibility/pivot tolerances. Exhaustive active-basis enumeration is an exact *combinatorial reference strategy* for the bounded LP family, not exact rational arithmetic. A numerical feasibility tolerance is not a proof of real-number feasibility.

## 4. Tests authored

The new module includes eight tests:

1. **Utility trap / hard boundary:** the unsafe high-utility policy is capped at the exact confidence-bound boundary; the independent checker passes the selected mixture.
2. **Non-estimable evidence:** a policy with a NOT_ESTIMABLE required cell is excluded and receives zero selected mass.
3. **Missing evidence:** an absent policy/requirement row is not imputed as zero and cannot receive positive mass.
4. **Family expansion:** adding a policy after calibration is rejected.
5. **Tampering:** changes to guardrail results, solver settings, or the outer report digest are rejected.
6. **Inclusive boundary:** guardrail feasibility is checked using the frozen tolerance.
7. **Local-stratum trap:** a local lower-bound guardrail binds even though the global upper-bound guardrail has positive slack.
8. **Infeasible polytope:** when every policy violates a mandatory global guardrail, optimization fails closed.

The generic policy-frontier module also adds two tests for policy-ID-bound coefficient order invariance and missing/duplicate policy identities. The utility-trap test additionally asserts that the experiment Type-I alpha (0.05) and guardrail confidence failure budget (0.01) remain separately recorded.

The earlier noncompensatory verifier module adds a regression test showing that a zero-weight policy with a missing bound does not make an otherwise fully evidenced mixture indeterminate.

**Tests are authored, not run.** No cargo test, cargo check, rustfmt, Clippy, CI, benchmark-runtime, or performance result is claimed.

## 5. Benchmark protocol and comparison

The benchmark should compare at least these methods on the same frozen policy family, guardrail roster, confidence intervals, and random instances:

1. Utility-only unconstrained frontier.
2. Unconstrained frontier followed by a post-selection guardrail verifier.
3. Guardrail-constrained frontier (this implementation).
4. Fixed safe fallback policy.
5. Always-abstain / always-deny control (must not count as optimizer success).
6. Independent HiGHS LP oracle on generated instances.

The independent LP oracle should receive the same coefficient matrix, bounds, and objective. Compare weights, objective, active constraints, primal residuals, and infeasibility decisions. Use scale-aware tolerances and retain the original unscaled residuals. HiGHS documentation explicitly notes that floating-point solvers satisfy feasibility and optimality conditions within tolerances and that poor scaling can undermine numerical reliability; therefore the independent verifier must reconstruct the original constraints rather than trust a solver status alone:
- [HiGHS numerical considerations](https://ergo-code.github.io/HiGHS/dev/guide/numerics/)
- [HiGHS feasibility and optimality](https://ergo-code.github.io/HiGHS/dev/guide/kkt/)
- [HiGHS LP/MIP specification and solver families](https://ergo-code.github.io/HiGHS/dev/)

### Frozen metric vector

Report the following per instance and as a distribution across seeds:

- known-answer false-safe rate and false-rollback rate;
- mandatory global and worst-local guardrail violation probability and magnitude;
- utility regret against the exact constrained LP optimum;
- worst-stratum regret, overall utility, and the guardrail/robustness tax;
- feasible, infeasible, excluded-policy, and abstention rates;
- selected support size and mass assigned to ineligible policies;
- active guardrail count, basis count, and solver/verifier wall time;
- independent LP primal residual, objective gap, and weight-vector distance;
- null Type-I error, alternative power, and expected evaluation cost;
- calibration/confirmation sample cost and confidence-family size.

### Proposed qualification targets (targets, not results)

- Zero known-answer guardrail violations among qualified mixtures.
- Zero positive mixture mass on a policy missing any required estimable bound.
- Exact fixture objective and weights within \(10^{-10}\) of analytical solutions, where conditioning permits.
- Independent LP objective gap at most \(10^{-8}\) on well-scaled fixtures, with raw primal residuals also published.
- No always-abstain strategy scored as successful optimization.
- Explicit infeasibility whenever the hard-guardrail polytope is empty.
- No false-safe claim outside the declared guardrail roster, target law, sampling-unit contract, estimand, and simultaneous confidence procedure.

These are internal engineering targets. They are not claimed as industry standards or achieved results.

## 6. Industry positioning and next research

Experimentation platforms commonly separate success metrics from safety metrics and support metric-triggered rollback. That is operationally valuable, but a rollout gate does not by itself solve the policy-selection problem: it can block an unsafe candidate without finding the best safe mixture.

The intended complement is:

- a platform's staged experiment and rollback lifecycle for deployment;
- the guarded LP for selecting a mixture that is feasible under the declared confidence bounds;
- an independent verifier for the post-selection record;
- sequential monitoring/drift detection for freshness after deployment;
- transport certification before reusing calibration bounds under a changed target law.

This does not replace domain-specific authorization or a rollout system. It adds a machine-checkable optimization layer before rollout.

The bounded reference solver currently supports at most six policies and caps active-basis enumeration at 100,000 bases. Additional guardrails increase the number of possible bases combinatorially. The next scalability steps are a frozen randomized-instance campaign, an independent HiGHS oracle, sparse constraint construction, and a separate scalable solver adapter whose output is always checked by the independent feasibility verifier.

## 7. Example commands

From the repository root:

~~~bash
cargo test --locked -p symthaea-architecture-engineering guardrail_constrained_policy_frontier
cargo test --locked -p symthaea-architecture-engineering noncompensatory_guardrails
~~~

The tests should be run against the exact PR head. Until then, this remains an implementation draft with static review only.
