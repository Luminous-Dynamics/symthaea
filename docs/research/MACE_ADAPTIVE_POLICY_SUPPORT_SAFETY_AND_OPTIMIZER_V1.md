# MACE Adaptive Policy Support Safety and Optimizer V1

**Snapshot:** 2026-11-26  
**Status:** Support-safety regression fixed in the companion branch; memoized optimizer is a research/implementation specification. No Rust compilation or test execution is claimed.

## 1. Support and probability are different objects

The finite-world evaluator has two logically distinct inputs:

- The **possibility support** says which worlds remain admissible under current evidence and hard constraints.
- The **probability measure** assigns weights to worlds for expected-utility optimization.

A possible world with probability mass zero contributes nothing to an expectation. It does not thereby become impossible, and it must still constrain any hard action-permission check.

The previous expected-value entry point filtered the search support to worlds satisfying both `possible` and `probability_ppm > 0`. That allowed an action to be selected even when a still-possible zero-mass world prohibited it.

The correction retains every `possible` world in the policy search. The existing probability mass continues to weight expected utility and expected acquisition cost. The probability-free robust evaluator is unchanged.

### Minimal regression

| World | Possible | Probability | `commit` permitted |
|---|---:|---:|---:|
| `bad` | yes | 0 ppm | no |
| `good` | yes | 1,000,000 ppm | yes |

Both worlds permit `abstain`. With no query budget, the expected-value evaluator must select `abstain`, not `commit`. The probability-zero world contributes zero expected utility but remains part of the hard permission support.

The companion source change adds this as a deterministic regression test:
`possible_zero_mass_worlds_still_constrain_expected_action_permissions`.

## 2. Formal invariant

For history (h), let (Omega(h)) be the set of possible worlds consistent with every observation on that path. Let (p(omega)) be the declared probability mass.

Expected utility is evaluated over the declared measure:

[
V_{mathrm{EV}}(h,a)=
sum_{omegainOmega(h)} p(omega)U(a,omega).
]

Action admissibility is evaluated over the entire possibility support:

[
mathrm{Admissible}(amid h)
iff
orallomegainOmega(h), mathrm{Allowed}(a,omega).
]

These predicates must never be conflated. In particular, (p(omega)=0) is not equivalent to (omega
otinOmega(h)).

A later design may explicitly model probability intervals or an untrusted calibration set. It must not silently reinterpret the support set to improve expected utility.

## 3. Next solver: exact memoization, not an approximate oracle replacement

The exhaustive reference solver should remain the correctness oracle. The candidate optimized solver should exploit an invariant in the current scenario contract:

- Queries are single-use.
- Query resource costs are fixed per query.
- The observation returned by a query is fixed by each latent world.
- There is no hidden history-dependent query-cost or permission state inside V1.

Under those conditions, a subproblem is identified by:

1. The bitset of still-possible world rows.
2. The bitset of queries already acquired.

The cumulative resource spend is the sum of fixed costs for the acquired-query set; the horizon is the set cardinality. Thus two different query orders that reach the same world support and same acquired-query set have the same remaining optimization problem.

The proposed memoization key is therefore:

[
K=(mathrm{possibleWorldMask},mathrm{queriedMask}).
]

Expected-value and robust objectives require separate memo tables. Query transitions partition the world mask by the query's observation, then recurse into each child. The candidate must preserve the reference solver's action admissibility, resource caps, policy-node limits and deterministic canonical tie-breaking.

### Important invalidation condition

The two-mask key is valid only while the above V1 assumptions hold. If a future contract adds history-dependent costs, query availability, changing rights, freshness deadlines, revocation events, model calibration updates or receipt expiry, the relevant state/digest must be added to the key. A cache hit must never bypass lifecycle revalidation.

## 4. Differential qualification protocol

For every frozen scenario, compare the optimized candidate against the exhaustive reference:

- Exact terminal action for every possible world.
- Exact expected utility, expected cost and net utility numerators.
- Exact worst-case regret and acquisition-cost outputs in robust mode.
- Hard permission preservation, including zero-mass possible worlds.
- Every resource budget on every policy path.
- Query/observation branch completeness.
- Policy node cap and deterministic canonical output under input permutation.

Generated scenarios should vary world count, query count, duplicate and overlapping source roots, contradictory observations, zero-mass possible worlds, tied actions, zero-cost queries, uneven query costs and resource caps. Keep a small exhaustive subset as a permanent oracle corpus.

Performance reporting must include both solver time and explored unique memo states, not just wall-clock latency. Compare the exact same frozen scenario set, build profile and host. The reference's explored-history count and the candidate's cache-hit/miss counters should remain visible.

No speedup target is a result until both outputs match for the complete frozen corpus. A reasonable experimental objective is at least a 10x median reduction in evaluated subproblems on scenarios with convergent query orders, with zero semantic mismatches. This is a proposed target, not a measured result.

## 5. External benchmark bridge

The solver corpus is not a replacement for external memory benchmarks. It is an exact small-world test of acquisition-policy semantics.

The next evaluation layer should connect its traces to:

- COLLABMEM: role visibility, provenance, authority, conflict and retention.
- TimlMemBench: revoke, supersede, preserve, scope, provenance, boundary, activation and proactive restraint.
- StreamMemBench: fidelity, initial evidence use, feedback incorporation and follow-up reuse.
- UtilMem: distributed synthesis, implicit relevance and distractor resistance.
- MemUse: natural integration rather than prompted recall.
- DeepControl: retrieval continuation and granularity under matched model, task and resource budgets.

Every external run should record the benchmark release, exact harness and data digests, model/checkpoint, prompts, tool versions, candidate commit, task denominator, omissions, safety strata, resources, matched baseline, uncertainty estimate and independent-replay state in the MACE benchmark evidence capsule.

## 6. Validation boundary

The support-safety correction and regression test are authored on a branch based on the #13785 head. The new regression test has not been compiled or executed in this environment. The memoized optimizer described above is a specification, not an implemented or benchmarked solver. No SOTA or performance improvement is claimed.
