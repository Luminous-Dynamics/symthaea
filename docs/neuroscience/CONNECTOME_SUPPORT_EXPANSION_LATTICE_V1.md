# Exhaustive support-expansion lattice benchmark V1

**Tracking:** [#10590 — exhaustive support-expansion monotonicity lattice](https://github.com/Luminous-Dynamics/symthaea/issues/10590)  
**Support-expansion implementation:** [#10588](https://github.com/Luminous-Dynamics/symthaea/pull/10588)  
**Harness:** connectome_support_expansion_lattice_audit

## 1. Purpose

Hand-authored positive and negative controls are necessary, but they do not establish that a monotonicity invariant holds over the entire bounded finite state space. This benchmark enumerates every ordered inclusion pair for a tiny complete support universe and checks exact adaptive and fixed-plan costs.

It is a regression theorem check: every pair satisfies \(J(w)\subseteq J'(w)\) for each world. The harness then verifies that the optimum cannot become cheaper or newly feasible.

## 2. Exhaustive universe

- Two worlds with opposite frozen query decisions.
- Two categorical observation actions, each with binary outcome alphabet \(\{0,1\}\).
- Four possible joint tuples per world: \((0,0),(0,1),(1,0),(1,1)\).
- Every non-empty subset of those four tuples is a valid support relation: 15 possible base supports per world.
- For each base support \(S\), enumerate every superset \(S'\supseteq S\).

There are 65 ordered support-inclusion pairs per world because:

\`\\sum_{S\\subseteq U,\\ S\\ne\\varnothing} 2^{|U\\setminus S|}=3^{|U|}-2^{|U|}=65\`

For the two worlds, the complete cross-product is:

\`65\\times65=4,225\`

The harness enumerates every one of these 4,225 pairs. It does not sample or randomize the support lattice.

## 3. Exact objectives

The adaptive solver uses a memoized minimax decision tree over the two action bits. It stops only when one of the opposing-decision worlds has no compatible joint tuple in the observed branch; otherwise it must inspect another action. Its objective is minimum worst-case total effort with action costs 1 and 2.

The independent fixed-plan solver enumerates every non-empty action subset and uses direct projected-set disjointness. Both return either an exact minimum cost or no feasible plan.

For each inclusion pair, the harness checks:
- adaptive optimum does not decrease;
- fixed-bundle optimum does not decrease;
- an infeasible base policy does not become feasible;
- an infeasible base fixed bundle does not become feasible.

It separately counts unchanged cost, increased cost and loss of feasibility. Any violation is serialized as a counterexample and causes non-zero exit.

## 4. Implementation independence

The Rust harness uses 4-bit support masks, an action-bit state and a memoized minimax recursion. The Python oracle uses immutable sets of joint tuples, tuple-coordinate filtering, direct projection sets and its own memoized recursion. The two implementations share only the mathematical specification and the explicit finite universe; the Python verifier checks the complete Rust report, not just its pass flag.

## 5. Promotion targets

| Metric | Target |
|---|---:|
| Ordered support-inclusion pairs evaluated | 4,225 exactly |
| Adaptive cost monotonicity violations | 0 |
| Fixed cost monotonicity violations | 0 |
| Adaptive policy resurrections | 0 |
| Fixed-plan resurrections | 0 |
| Counterexample report mismatch vs independent oracle | 0 |
| Exact report equality across repeated runs | 100% |

A clean result establishes a bounded exhaustive implementation invariant only. It does not establish the monotonicity of a different implementation, larger outcome alphabets, infinite domains, approximate solvers or real biological observation models.

## 6. Connection to the full programme

The exhaustive lattice harness qualifies the abstract monotonicity property. The support-expansion audit in [#10588](https://github.com/Luminous-Dynamics/symthaea/pull/10588) applies that property to provenance-bound challenger tuples in a realistic (but still finite) query scenario. The adaptive policy in [#10568](https://github.com/Luminous-Dynamics/symthaea/pull/10568) chooses observations sequentially under correlated joint supports, while [#10528](https://github.com/Luminous-Dynamics/symthaea/pull/10528) defines those joint relations.

The full benchmark progression is therefore:

1. exhaustive mathematical finite-lattice property test;
2. synthetic evidence-bound support challenge;
3. larger adversarial and mutation corpora;
4. blinded expert-adjudicated real support expansions;
5. prospective measurement of query-critical error and review effort.

No biological truth, source authenticity, support completeness or independence is inferred from the lattice pass.
