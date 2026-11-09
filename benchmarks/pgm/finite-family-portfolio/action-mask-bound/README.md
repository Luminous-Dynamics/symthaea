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

## Exact outer policy-portfolio partition branch-and-bound

`solve_exact_finite_family_policy_portfolio_bnb` now composes the exact
single-cluster solver into a global finite-family solver for at most k policies.

### Partition equivalence

Let (r(C)=min_{\pi}\max_{M\in C}(V^*_M-V_M^\pi)) be the exact minimax
regret of one shared policy for a nonempty cluster C. Then:

```
min_{|Pi| <= k} max_M (V*_M - max_{pi in Pi} V_M(pi))
=
min_{partition C_1,...,C_j; j <= k} max_i r(C_i)
```

Proof sketch:
- Given a partition, choose a minimax-regret policy for each cluster.
  Every model is covered by its assigned policy, so portfolio regret is no
  greater than the maximum cluster regret.
- Given any portfolio, assign each model to a portfolio member that gives
  it the highest value (ties resolved deterministically). Each induced
  cluster's shared-policy optimum is no worse than that assigned member,
  so the best partition has objective no greater than the portfolio regret.

The two optima are equal. At a globally optimal partition, the returned
cluster-optimal policies must therefore have portfolio coverage regret equal
to the partition objective. The implementation checks that equality by
re-evaluating every selected policy on every model.

### Outer admissible bound

A partial partition assigns a prefix of the ordered model family into at most
k nonempty clusters. For each current cluster C, the inner B&B proves r(C).
Adding models cannot lower r(C), because the maximum in the cluster regret
objective is taken over a superset. Thus

```
LB(partial partition) = max over current clusters C of r(C)
```

is a valid lower bound for every completion. Unassigned models contribute
nonnegative regret and do not invalidate the bound.

The search uses canonical restricted-growth partitions: at each model, assign
it to one existing cluster in ascending order, or create exactly one new
cluster. Cluster labels are not permuted, so each unlabeled set partition is
visited once. DFS visits them lexicographically. The initial feasible upper
bound is the exact one-cluster solution.

A node is pruned only when its exact lower bound is greater than or equal to
the incumbent. Completed partitions are evaluated exactly. The function
returns only after the full outer frontier has been exhausted. It exposes
equal global lower/upper bounds, node/complete/prune counts, cached inner
cluster certificates, total inner-search nodes and complete DP work. Budget
exhaustion returns an error, not a partial result labelled optimal.

The transcript hashes every popped partial partition, ordered cluster
certificate digests, exact lower bound and event (expanded/pruned/complete).
The final portfolio certificate binds the family, policy manifest, selected
partition, inner certificates, every model-by-policy value, selected
portfolio values, regrets, worst-case models, global bounds, counters and
transcript digest. The verifier re-runs the complete deterministic search.

### Outer exact known answer

Two one-state specialist models, M0 rewards [1,0] and M1 rewards [0,1],
discount 1/2, maximum portfolio size k=2:
- Optimal partition: [[0],[1]].
- Selected cluster policies: action 0 for M0's cluster, action 1 for M1's.
- Model-by-policy values: [[1,0],[0,1]].
- Model portfolio values: [1,1]; model regrets: [0,0].
- Minimax portfolio regret and global lower/upper bounds: 0.
- Outer nodes: 4; complete partitions: 2; pruned outer nodes: 0.
- Cached unique cluster solutions: 3 (the full family, [0], [1]).
- Total inner cluster nodes: 7; total exact DP work including cross-model
  portfolio replay: 104 multiply/add terms.

Known identities:
- Family SHA-256: `31c37db3d062ca761438ef3d506049aaebc667984a442605f3c52aa41ea2447f`.
- One-model M0 cluster certificate: `bdfca086755479f1cff8fe09303fabdbeabb12182e50789c12f2628d365cb5c7`.
- One-model M1 cluster certificate: `551f6f315468270f5a5b3a74f04665ba3e5a9894cc8ae41312cd1bbb5cdaa222`.
- Outer partition transcript: `c69c11ded97c6634b743ba30e1f5bfcaadd5cd0de6c3db65cca867c8a3d38eaa`.
- Outer optimum certificate: `f75dc2c13a38445ffb22df19526d0d4628d533b864dcec8265c2c65d03eb6d17`.

Hashes are independently reconstructed from the documented canonical
little-endian rational/integer encodings. They are regression targets;
they are not evidence of a Rust test having executed.

## Comparative benchmark programme

Primary research references:
- [KAPS, Lutz et al., arXiv:2608.02509 (3 Aug 2026)](https://arxiv.org/abs/2608.02509), pinned code revision `4d35eef6c442521139d1786e31c6d6fbf322788f`.
- [Adaptive Policy Portfolios, Engelen et al., arXiv:2608.17929 (18 Aug 2026)](https://arxiv.org/abs/2608.17929).
- [ORBE / Best-Effort Policies, Abate et al., AAAI 2026](https://doi.org/10.1609/aaai.v40i43.40929).
- [Generic, Reliable, Efficient Robust MDP solving, Meggendorfer et al., AAAI 2025](https://doi.org/10.1609/aaai.v39i25.34865).

Keep four test regimes distinct:

1. **Exact finite UMDP comparison against KAPS** (SUMI-lab/KAPS, commit
   `4d35eef6c442521139d1786e31c6d6fbf322788f`). Use the same finite model
   tensors, discount, initial state, reward timing, k, and deterministic
   memoryless policy class. For H-step finite-horizon inputs, clock-augment
   each state to (t,s) with an absorbing zero-reward terminal state. Compare
   the exact Rust reference and KAPS's float64 values with a preregistered
   tolerance. Report regret, proof status/gap, nodes, per-cluster solve/cache
   statistics, wall time, memory, and proof replay separately.
2. **Continuous/parametric RMDP coverage and selection** against
   *Adaptive Policy Portfolios for Robust MDPs* (arXiv:2608.17929). This is a
   different model class: the paper discretizes two-parameter uncertainty
   domains into cells, clusters regret profiles, and uses UCB for online
   member selection. It reports Datacenter Climate Control (330 states) and
   UAV-small/medium/large (98/358/1813 states), with k in {1,2,3,5,7,10}.
   Its finite-sample 1000-point regret estimate is an under-approximation,
   not an exact robust guarantee. Port these as approximate/continuous
   tracks, and do not compare them as if they were finite enumerated KAPS
   instances.
3. **Robust best-effort tie breaking** against AAAI-26's ORBE work. When a
   robust value or regret objective has many tied solutions, measure nominal
   or non-adversarial performance, Pareto dominance, safety margins and
   stability of the selected policy. Do not collapse this into worst-case
   regret alone.
4. **Adjacent robust-MDP solver baseline** against *Solving Robust Markov
   Decision Processes: Generic, Reliable, Efficient* (AAAI 2025). That work
   supports several uncertainty-set/objective classes and reports million-state
   RMDP solving in under a minute. It is not a direct portfolio-regret
   comparator: compare its robust Bellman/value-iteration throughput only on
   matched rectangular single-policy subproblems, then separately report the
   extra cost of multi-policy portfolio optimization and proof certificates.

KAPS's public study reports its exact solver achieved the lowest
single-policy regret in 11 of 13 benchmark instances and proved optimality
in 9, versus 3 for the exact MILP baseline within a five-minute limit. Its
Maintenance scaling experiment reports proofs up to 500 states and 100
distinct MDPs in under 100 seconds. These are the concrete external scale
targets: first reproduce the 13-instance k=1 comparison, then match the
500-state/100-model maintenance result under the same machine and protocol,
then seek a clear proof-time and memory improvement. (The published numbers
are reference points, not measurements from this repository.)

The 2026 adaptive-portfolio paper reports a sharp early gain from K=1 to K=2,
then diminishing but continued improvements. In its exact experimental setup,
the offline candidate pipeline discretizes each of two uncertainty parameters
into 10 bins (100 cells), estimates portfolio regret on 1,000 sampled parameter
points (explicitly an under-approximation), and uses UCB with empirical
Bernstein bounds over length-100 trajectories. The reported construction runs
use a 2022 MacBook Pro M1 with six threads and three seeds. These details must
be mirrored or clearly normalized before interpreting runtime or regret
differences. Its experiments include
UAV-small/medium/large (98/358/1813 states) and Datacenter Climate Control
(330 states), with empirical K values 1,2,3,5,7,10. At K=10, after 10,000
UCB iterations only about half the runs recommend the best arm, although
the value gap between the recommended and best member is small. Reproduce
those metrics as a separate continuous/parametric and online-selection
track. Do not claim that the finite-family deterministic reference has
already beaten the randomized-policy, polytopic RMDP results.

## Paradigm shifts to pursue next

- **Beyond offline coverage to evidence-conditioned action:** use observations
  to update a posterior/confidence set over models and select among
  pre-certified policies, with a regret decomposition into offline coverage
  gap plus online identification regret.
- **Proof-aware active sensing:** choose observations/actions by expected
  information gain per unit cost, while maintaining a safety envelope and
  never deploying a policy whose certificate does not cover the current
  credible model set.
- **Regret-profile geometry:** [the exact profile quotient / Pareto frontier
  track](../regret-profile-frontier/README.md) now groups identical checked-
  rational regret vectors, preserves original policy indices, and provides
  strict componentwise-dominance witnesses. The next step is to use these
  profiles in threshold set-cover / branch-and-price without weakening the
  exact optimum.
- **One-run policy-budget frontier:** reuse cluster and partition certificates
  across k instead of rerunning from scratch for every budget; report the
  full (policy count, regret) Pareto frontier.
- **Risk-sensitive and service-constrained objectives:** augment minimax
  regret with tail loss, chance constraints, safety invariants, switching
  costs and resource limits. Keep these as separately defined objectives.
- **Randomized and continuous uncertainty:** first support a finite grid of
  rational randomized policies, then parameterized policies with interval or
  algebraic bounds. Do not silently generalize the deterministic finite
  family verifier to arbitrary polytopes.

## Current scope and limitations

The current exact implementation handles a finite, explicitly enumerated
family of finite-horizon, fully observable UMDPs with common state/action
spaces and initial distribution, checked-rational data, deterministic
time-dependent Markov policies, and up to k policies. The inner and outer
searches are currently deterministic and all-or-error under configured caps;
they are not yet resumable, distributed or anytime gap-reporting. The policy
space cardinality must still fit in usize for the current canonical policy
index manifest. Continuous rational polytopes, randomized policies, online
model inference, partially observable models, risk constraints and
incremental Pareto-frontier reuse and online model inference remain separate
work. The new profile analysis is an exact post-processing layer over the
fully evaluated finite policy table; it is not yet integrated into the outer
partition B&B's runtime path.
