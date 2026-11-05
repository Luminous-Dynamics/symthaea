# Worst-case adaptive query-directed observation policy V1

**Tracking:** [#10568 — adaptive query-directed observation policies](https://github.com/Luminous-Dynamics/symthaea/issues/10568)  
**Joint support input contract:** [#10528 — correlated joint observation support](https://github.com/Luminous-Dynamics/symthaea/pull/10528)  
**Fixed-plan predecessors:** [#10506](https://github.com/Luminous-Dynamics/symthaea/pull/10506), [#10525](https://github.com/Luminous-Dynamics/symthaea/pull/10525).  
**Status:** Research specification on branch `research/connectome-adaptive-query-policy-v1`. Implementation and validation have not yet been claimed.

## 1. Research gap

The fixed-plan planners choose all observation actions before any measurement is performed. That is safe and appropriate for batched work, but it can waste effort when an early observation already makes all compatible worlds agree on the scientific query.

The next step should plan sequentially: observe one action, condition the joint support on its actual outcome, then choose the next action based on what remains possible.

The planner must not assume that observations are independent. A measurement on one section can constrain what outcomes remain possible for a later morphology or synapse inspection. This conditioning must be performed against the original joint tuples, not reconstructed from separate marginals.

## 2. Paradigm shift: query unanimity, not world identification

Let each world (w) have a frozen query decision (q(w) \in \{0,1\}), and let (R_w) be the set of joint observation tuples still compatible with the observation history.

A policy may stop exactly when all remaining compatible worlds agree:

`\left|\{q(w): w \text{ remains compatible}\}\right| = 1`

The policy is not required to identify which world is true. Distinct biological reconstructions that imply the same frozen query answer may remain together in a terminal leaf.

This is the key efficiency change: spend review effort only when a remaining ambiguity can still change the scientific conclusion.

## 3. State transition with correlation preserved

A policy state contains:
- the currently compatible worlds;
- for each world, the subset of its original joint tuples compatible with every observation so far;
- the set of observation actions not yet used.

When the planner selects action (a), it enumerates every outcome (y) admitted by at least one remaining joint tuple. For each such outcome, the child state is:

`R'_w = \{t \in R_w : t[a] = y\}`

Worlds with empty (R'_w) are removed from that branch. The remaining joint tuples are retained in full over all not-yet-performed actions, preserving correlation under conditioning.

The child branch is terminal only if all retained worlds agree on the query. Otherwise, the policy must choose another unused action. If the child contains conflicting decisions and no remaining action can resolve them, that parent action is not a valid policy choice.

Any actual observed category that is absent from every declared joint tuple is an out-of-model observation. Runtime execution must fail closed with `OUT_OF_MODEL_OBSERVATION`; it must not map the observation to a nearest category or silently ignore it.

## 4. Exact worst-case objective

V1 uses no priors. The objective is minimum worst-case expert-review effort:

`V(B) = 0 \quad\text{if the query is unanimous in state } B`

Otherwise:

`V(B) = \min_{a \in A(B)} \left[c(a) + \max_{y \in Y(a,B)} V(B_{a=y})\right]`

The minimum is over actions that produce a valid policy for every declared admissible outcome branch. The maximum is over every outcome admitted by the currently compatible joint support.

No expected cost, expected information gain or probability-weighted value is computed. Realized terminal path costs can be reported individually, along with the worst-case bound.

## 5. Proposed exact implementation bounds

A first implementation should be deliberately bounded:

- At most 12 actions, 12 worlds, and 64 joint tuples per world.
- Joint tuple support and action definitions reuse the #10528 input contract and digest rules.
- Each world's compatible tuple subset can be represented by a 64-bit mask.
- Memoize states by (remaining-action mask, per-world compatible-tuple masks).
- Enumerate each action's distinct currently admissible outcomes; recursively solve every child.
- Return a deterministic optimal policy, worst-case cost, minimum and maximum terminal path cost, and a bounded count of tied optimal policies.
- Cap memoized states (proposed 25,000 initially). If the cap is reached, emit no optimality claim; report `SEARCH_LIMIT_REACHED` or fail closed.
- Never describe an incomplete or approximate search as exact.

The same immutable source/model digests and dependence ancestry labels from the joint-support input remain attached to the policy.

## 6. Required fixtures

1. **Adaptive savings:** four worlds. A cheap first measurement divides them into two opposing-decision branches. One branch needs a morphology action; the other needs a synapse action. The exact adaptive policy cost is the first action plus the more expensive branch, while a fixed non-adaptive bundle pays for both follow-up actions.
2. **Correlation-preserving conditioning:** observing action A filters joint tuples and changes the admissible outcomes for B. The planner must not use B's unconditional marginal set after observing A.
3. **Query unanimity without world identification:** a terminal leaf contains multiple worlds with the same query decision.
4. **Unresolvable ambiguity:** at least one branch retains opposing decisions after all remaining actions; the parent action cannot be selected as a valid policy node.
5. **Out-of-model category:** an outcome absent from all declared tuples produces an explicit fail-closed disposition.
6. **Tied exact policies:** multiple actions/policies with equal worst-case cost; canonical deterministic selection and exact tie count.
7. **Budget cutoff:** minimum worst-case plan exceeds budget but is still reported as the exact minimum.
8. **Ordering invariance:** action, world and tuple order permutations preserve the canonical policy and digest.

## 7. Industry and research comparison

Adaptive observation planning is an established research area, so the contribution should be a connectomics-specific assurance formulation, not a claim to invent adaptive testing.

- The **IJCAI 2026 survey on active feature acquisition** organizes embedded cost-aware predictors, model-based policies, model-free policies and hybrids, and emphasizes the connection to POMDP planning. [Survey](https://ijcai.org/proceedings/2026/860)
- **Optimal Decision Tree and Adaptive Submodular Ranking with Noisy Outcomes (JMLR 2024)** studies cost-aware adaptive tests with uncertain and persistent outcomes, and provides approximation algorithms. [JMLR](https://jmlr.org/papers/v25/23-1484.html)
- **Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization (JAIR)** establishes guarantees when the adaptive submodularity conditions hold. [JAIR](https://ocs.aaai.org/Library/JAIR/Vol42/jair42-012.php)

These methods are comparison baselines, not assumptions to import automatically. Correlated joint-support models may violate independence or adaptive-submodularity conditions. Any approximation guarantee must be justified against the actual model.

## 8. Benchmark matrix

| Metric | Proposed target |
|---|---:|
| Exact worst-case policy vs independent oracle on bounded cases | 100% |
| False terminal leaves with mixed query decisions | 0 |
| Correlation lost after conditioning | 0 |
| Out-of-model observations silently accepted | 0 |
| Worst-case cost gap vs exact oracle on bounded cases | 0 |
| Equivalent action/world/tuple ordering changes policy | 0 |
| Expert-review effort reduction vs optimal fixed plan at matched query reliability | At least 25% prospectively |
| Exact/approximate/limit-reached states explicitly distinguished | 100% |

For comparisons report both worst-case and realized terminal path effort. Do not report an average without a declared prior and a defensible sampling design.

## 9. Claim boundary

This specification is limited to exact finite support over declared worlds, actions and joint outcome tuples. A policy certificate cannot prove that the joint-support model includes every physically possible observation, that evidence sources are independent, or that a biological hypothesis is true.

No expected value of information, likelihood, posterior, causal effect or biological truth claim is made in V1.
