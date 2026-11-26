# MACE Exact Memoized Adaptive Policy Solver V1

**Snapshot:** 2026-11-26  
**Status:** Candidate solver implementation and deterministic differential tests authored. Rust compilation, formatting, Clippy and test execution have not been performed in this environment. No speedup or SOTA result is claimed.

## 1. Why optimize after building the exhaustive oracle

The exhaustive V1 evaluator is the independent correctness reference for small finite joint-world scenarios. It explores query orders repeatedly, even when different orders lead to the same unresolved world set and the same set of queries already acquired.

For example, two paths can observe q-a then q-b or q-b then q-a and reach the identical support of possible worlds with the same set of used queries. Under the current V1 contract, the remaining decision problem is identical.

The new candidate solver memoizes those equivalent subproblems. It does not change the objective, permissions, likelihood model, costs, resource budgets, or policy representation.

## 2. Exact memoization key

The memoization key is:

[
K = (W, Q)
]

where:
- (W) is a 64-bit bitset of still-possible latent worlds;
- (Q) is a bitset of query indices already used.

The key is sufficient for V1 because:
1. Each query is single-use.
2. A query's tool-call, latency, context-token, inspected-byte and utility cost are fixed in the scenario.
3. Resource costs are additive and non-negative.
4. The observation returned by each query is fixed for each latent world.
5. The action utilities and admissibility are fixed by state.
6. No hidden state beyond the possible-world support and used-query set changes future query or action eligibility.

Therefore:
- depth equals the population count of (Q);
- cumulative resource spend is the sum of the fixed costs of queries in (Q);
- the current evidence history's effect on possible states is represented by (W);
- remaining unused queries are the complement of (Q) over the scenario's query set.

Two recursive histories with the same (W,Q) have exactly the same feasible continuation policies and objective values. Caching the best continuation is therefore semantics-preserving.

### Cache invalidation boundary

This proof does not automatically extend to future contracts with history-dependent acquisition costs, dynamic query permissions, expiring receipts, changing source rights, bitemporal freshness transitions, asynchronous revocation, new calibration state or policy-dependent resource costs.

If any such field affects a future transition, it must either be represented in the memoization key or be independently revalidated outside the cached subproblem before each transition. Never let a cache hit bypass current authority, lifecycle, scope or rights checks.

The source-origin lineage still matters for how the joint-world scenario is constructed. In this V1 oracle, correlation is represented by fixed joint outcomes in each world, not inferred by multiplying marginal query likelihoods.

## 3. Implemented candidate

New module:
`research/broca-mace-transfer-bench-v0-1/aml-adapter/src/adaptive_policy_optimized.rs`

Public entry points:
- `evaluate_expected_memoized_v1`
- `evaluate_robust_memoized_v1`

The candidate:
- reuses the public V1 scenario validation;
- preserves all possible worlds, including zero-probability worlds, for hard action admissibility;
- computes expected utility and expected query cost using the original integer ppm numerators;
- computes probability-free worst-case regret plus acquisition cost;
- derives cumulative budgets from the used-query bitset;
- uses exact canonical policy tie-breaking;
- keeps separate memo tables for expected and robust objectives;
- reports recursive calls, unique expanded states, cache hits and misses;
- bounds unique state expansion at 50,000 and recursive calls at 250,000;
- returns the same policy/result types as the reference oracle, with the additional search statistics.

The reference oracle remains unchanged as the semantic source of truth except for the separately reviewed crate-visible canonical-key method and the support-safety fix in the stacked base PR.

## 4. Differential test strategy

The new test module includes:
- 16 deterministic scenario variants for expected-value comparison;
- 16 probability-free robust comparisons;
- exact policy-tree equality, not only equal scalar scores;
- exact expected utility, expected acquisition cost and net-utility numerator equality;
- exact robust regret and worst acquisition-cost equality;
- a targeted zero-probability possible-world permission case;
- an order-convergent query-history case that must exercise cache hits;
- query enumeration permutation checks for both objectives.

The fixtures include correlated joint observations, multiple action utilities, hard action restrictions, zero probability mass, distinct query costs and all six resource budgets.

The tests are deterministic fixtures, not an empirical performance benchmark. They are written but unexecuted here.

## 5. Qualification gates before promoting the candidate

### Gate A: exact differential equivalence

For every frozen scenario:
- compare expected utility numerator, expected cost numerator and net utility numerator;
- compare the complete policy tree and terminal actions;
- compare robust worst-case regret and acquisition cost;
- compare results after action/query enumeration permutations;
- exercise zero-mass possible worlds and contradictory observations;
- assert all hard constraints and budgets path-by-path.

Any semantic mismatch blocks promotion.

### Gate B: generated adversarial corpus

Extend deterministic fixtures or property-based tests to cover:
- 1–64 possible worlds;
- 1–8 queries;
- duplicate, overlapping and disjoint source roots;
- identical and partially identical observation partitions;
- zero-mass possible worlds;
- budget boundaries at exactly zero and exactly the permitted maximum;
- actions forbidden in one or more states;
- ties in utility and policy canonical order;
- no-information and contradictory queries;
- search limits and oversized policies;
- query-order and action-order permutations.

For each generated small-world scenario, the exhaustive implementation is the oracle. A candidate result is accepted only when the entire objective and policy match.

### Gate C: performance

Measure:
- unique memo states;
- total recursive calls;
- cache hit ratio;
- wall-clock p50/p95;
- peak memory;
- policy node count;
- input-world/query dimensions.

Compare both solvers on the exact same frozen scenarios, compiler version, build profile and host. The proposed research target is at least a 10x reduction in expanded subproblems on convergent-query scenario families, with zero semantic mismatches. It is not a present result and should not become a release claim until measured.

### Gate D: production boundaries

The memoized solver only proposes a policy. It must not execute a tool call or authorize a real-world action. Every production transition must revalidate:
- current consumer DecisionNeed and capability;
- exact query/action permission;
- evidence rights and purpose;
- current source revision and origin;
- current lifecycle, correction, quarantine, revocation and expiry;
- remaining resources and signed receipts;
- independent final action authorization.

A stale cache entry is never authority.

## 6. External benchmark connection

This solver provides a small, exact test of adaptive acquisition semantics. It does not replace external memory evaluation.

Use the official matched tracks of:
- TimlMemBench for revoke, supersede, scope, provenance, activation and proactive restraint;
- StreamMemBench for chronological evidence use, feedback incorporation and future reuse;
- COLLABMEM for role-aware sharing, access, routing, provenance, conflict resolution and retention;
- DeepControl for retrieval continuation and granularity under matched models, benchmarks and budgets;
- UtilMem for distributed evidence integration and distractor resistance;
- MemUse for appropriate natural integration and deliberate non-use.

Never collapse these into one scalar. Keep official measures and denominators intact, with MACE governance and safety strata reported separately.

## 7. Industry comparison principles

Compare against Mem0, Letta, A-Mem, Graphiti/Zep, MemoryOS and other relevant systems through their pinned official implementations and their supported benchmark adapters. Separate:
- implementation capability (what the product actually offers);
- protocol conformance (what the adapter can demonstrate);
- measured outcome (what happened on the matched task set);
- assurance evidence (what an independent replay verifies).

A system that does not expose a capability should be reported as unsupported or unmeasured, not scored as a failure under a made-up proxy. A system that scores better on recall but fails revocation or authorization must retain those distinct results.

The defensible MACE differentiator is a complete, replayable evidence-to-decision chain with independent authority and lifecycle enforcement—not a claim to win every memory leaderboard.

## 8. Current validation boundary

- The exact memoized solver and test suite are authored.
- The candidate module is exported from the AML adapter crate.
- The exhaustive reference oracle is retained.
- The memoization assumptions and invalidation boundary are documented.
- No local Rust toolchain was available in the execution environment.
- No Rust compile, rustfmt, Clippy, Cargo test, CI pass, speedup measurement, external benchmark execution or SOTA claim is reported.

Next promotion gate: obtain an exact-head Rust toolchain run and independent review of the differential test results before the candidate is used by any production policy path.
