# Exact rational action-mask cluster bounds

This is the first implementation step for
[PGM-UMDP-PARTITION-BNB-001](https://github.com/Luminous-Dynamics/symthaea/issues/11178):
an inner branch-and-bound relaxation for deterministic finite-horizon Markov
policies over an explicitly enumerated finite family of UMDPs.

## Mathematical contract

For a cluster of environments C, let A[t,s] be a nonempty set of actions
still permitted at each time/state decision slot. For each model M, exact
backward dynamic programming computes:

```
U*_M       = max over all deterministic time-dependent Markov policies pi of V_M(pi)
U*_M(A)    = max over pi with pi[t,s] in A[t,s] of V_M(pi)
gap_M(A)   = U*_M - U*_M(A)
LB_C(A)    = max over M in C of gap_M(A)
```

For every common policy pi that obeys A, we have
V_M(pi) <= U*_M(A), and therefore
U*_M - V_M(pi) >= gap_M(A) for each model. Taking the maximum over the
cluster proves:

```
LB_C(A) <= min_pi max_{M in C} (U*_M - V_M(pi))
```

The bound is **admissible** for minimizing cluster worst-case regret. At a
leaf mask—exactly one allowed action at every slot—there is only one policy,
so the bound equals its exact cluster regret.

This does not assume one transition model is shared across different
policies; each environment is fixed while all policies are evaluated. The
common policy constraint is what makes the max-over-models of independent
masked optima a lower bound rather than a feasible common-policy value.

## Why there is no big-M reward penalty

The KAPS implementation uses a finite negative reward penalty to discourage
forbidden actions. This implementation instead carries allowed-action sets
directly into the Bellman recurrence. Every masked Bellman maximization
considers only permitted actions. There is no arbitrary penalty magnitude to
prove adequate and no risk that a forbidden action remains optimal under a
poorly chosen bound.

## Branch split

For an allowed action a at slot (t,s), `split_on_action` returns:
- child 1 forces a by making it the only allowed action at (t,s);
- child 2 forbids a and retains all other previously allowed actions.

The children are disjoint and exhaustive over the parent policy set. Empty
action sets, invalid slots, forbidden branch actions, and singleton slots are
rejected. Child masks are bound to the source family SHA-256 and carry their
own canonical mask SHA-256.

## Exact witness and replay

`compute_policy_mask_cluster_lower_bound` returns:
- source family and action-mask digests;
- ordered model IDs and per-model source digests;
- unrestricted and masked exact rational optimal values;
- each model's exact regret-gap lower bound;
- the cluster maximum lower bound;
- the deterministic DP multiply-add estimate;
- a canonical SHA-256 over the whole bound transcript.

`verify_policy_mask_cluster_lower_bound` recomputes all model optima, masked
optima, gaps, the aggregate lower bound and digest. Mutated witnesses are
rejected.

The current implementation supports an explicit finite family, common
state/action/horizon and initial-distribution dimensions, deterministic
time-dependent Markov policies and checked rational arithmetic. It does not
certify continuous uncertainty polytopes, shared-parameter uncertainty,
history-dependent policies or randomized policies.

## Resource limits

`PolicyActionMaskBudget` caps model count, action-mask entries, unrestricted
plus masked Bellman operation estimates, rational numerator/denominator bit
width (including generated intermediates), and estimated peak result plus DP
scratch memory. Checked arithmetic and fallible reservations fail closed.

## Exact fixtures

The two-model specialist fixture uses one state and two actions:
- M0 rewards (1,0);
- M1 rewards (0,1);
- discount 1/2 (irrelevant at horizon one).

With both actions allowed, both models achieve their unrestricted value 1 and
the cluster lower bound is 0. If action 0 is fixed, the per-model gaps are
(0,1), and the cluster lower bound is 1. If action 1 is fixed, the gaps are
(1,0), and the cluster lower bound is 1.

A two-step one-state fixture checks the time-varying action mask and discount:
unrestricted value 3/2, masked value 1/2, exact lower bound 1.

The source test module currently contains 21 authored tests across the finite
family portfolio oracle and action-mask extension. These tests are not
represented as compiled or executed in this repository environment. Exact
branch-head CI evidence is required before promotion beyond draft.

## Next step

Implement the outer partition frontier using the same admissible cluster
bounds. A partial cluster's optimum regret cannot decrease when more models
are added, so the maximum of current cluster bounds is a sound lower bound
for every completion of that outer node. Each branch must assign exactly one
unassigned model to one cluster, with symmetry breaking for empty/new
clusters. Maintain a global incumbent upper bound and the minimum bound over
the complete live frontier. Exact optimality is claimed only when those
bounds meet exactly or the frontier is exhausted.

## Exact single-cluster branch-and-bound

`solve_exact_cluster_policy_bnb` is the first search consumer of the mask
lower bound. It starts with the lexicographically least deterministic policy
as a valid incumbent, then traverses masks depth-first in action-index order.
At each mask:
1. Recompute or reuse the exact per-model unrestricted/masked Bellman values.
2. Prune only if the maximum exact modelwise regret gap is greater than or
   equal to the current incumbent regret.
3. If not a leaf, split the first non-singleton (time,state) slot into
   force-lowest-action and forbid-lowest-action children.
4. At a leaf, the action mask identifies a unique policy and the modelwise
   masked values/gaps are its exact value/regret witnesses.

The complete frontier must be exhausted before the function returns a
`VerifiedClusterPolicyOptimum`. A node, aggregate rational-work, frontier
count, frontier-byte or total-memory cap returns an explicit error. It does
not return an unproven incumbent with an optimality label.

The returned witness binds the exact family, cluster, discount, selected
policy index and action vector, each model's optimal/policy value/regret,
worst-case model set, search-node count, pruned-node count, and total
rational Bellman work into a canonical SHA-256. The verifier reruns the whole
deterministic cluster search and compares the complete result.

The search currently returns the single lexicographically least optimum,
not every tied cluster policy. That is sufficient as a deterministic
partition subproblem oracle; complete portfolio tie enumeration remains the
responsibility of the finite-family exhaustive reference.

### Cluster search known answers

- Two specialist models (M0 rewards [1,0], M1 rewards [0,1]), cluster [0,1],
  discount 1/2: policy index 0, actions [0], model values [1,0], regrets
  [0,1], worst model [1], cluster regret 1; 3 masks visited, 2 pruned,
  54 rational DP multiply/add terms.
- One model with rewards [0,1], cluster [0], discount 1/2: initial policy 0
  is suboptimal; B&B finds policy index 1/actions [1], value 1, regret 0;
  3 masks visited, 1 pruned, 27 rational DP multiply/add terms.

This is an exact **inner cluster solver**, not yet the outer partition
search. It is not evidence of a runtime win over KAPS. The outer search must
still assign all environments to at most k clusters, maintain a complete
frontier lower bound, and prove global portfolio optimality.
