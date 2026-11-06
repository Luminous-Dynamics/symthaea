# Query-directed adaptive connectome intervention policies V1

**Tracking:** #10823 — SCI-CONNECTOMICS-094
**Parent benchmark:** #10670 — query-discriminating interventions
**Composes:** #10603, #10656, #10661, #9522, #9587, #9549, #9575, #9578, #9255
**Status:** deterministic synthetic policy contract and independent reference oracle; no biological intervention or performance claim.

## 1. Objective

Choose a complete outcome-contingent intervention policy, not merely the next intervention ranked by entropy or value-of-information. The policy must guarantee that every possible observation path ends only when the remaining compatible biological worlds agree on the frozen scientific query.

V1 deliberately uses a bounded deterministic potential-outcome table. It does not pretend that categorical potential outcomes are calibrated probability distributions. Probabilistic posterior-risk selection remains a separate #10670 objective.

## 2. Finite latent-state model

- A biological world `w` has a frozen query decision `q(w)` and exact graph/evidence digests.
- A nuisance model `m` is one coherent, fixed model choice throughout a complete experimental path.
- A latent state is `(w,m)`. Its query decision is `q(w)`; nuisance uncertainty does not change the biological query label.
- Each intervention `a` has one declared potential outcome `o(w,m,a)` in V1.
- The full potential-outcome vector over all interventions is jointly specified for each `(w,m)`. Cross-intervention dependence is therefore represented by one vector, not inferred by multiplying marginal distributions.

Every world × nuisance × action cell must exist exactly once. Missing outcomes, duplicate cells, unregistered worlds, or unknown action references reject the run.

## 3. Adaptive policy semantics

A policy maps the complete observed action/outcome history to the next feasible intervention or `STOP`.

At a node with compatible latent-state set S:

1. Stop only if `{q(w) : (w,m) in S}` has cardinality one.
2. Otherwise, choose an unused feasible action a.
3. Partition S by the declared potential outcome for a.
4. For each nonempty outcome branch, recurse on exactly the latent states that predict that outcome, with a marked used.
5. The action's worst-case continuation cost is `cost(a) + max(child_cost)`.
6. The exact adaptive optimum is the minimum such worst-case cost over all valid actions and all possible outcome branches.

If any possible branch has no guaranteed resolving continuation, that action is not a valid policy node. If no action can guarantee resolution, return `NO_GUARANTEED_QUERY_RESOLUTION`.

A nuisance ID remains attached to each state after every observation. It cannot be selected independently at each step, which would manufacture impossible switches between incompatible nuisance models.

## 4. Fixed-plan comparator

A fixed plan chooses a subset of feasible actions before observing any outcome, executes every selected action, and then resolves the query from the resulting joint outcome tuple. No action is chosen conditionally on an earlier response, and no outcome-dependent early stopping is allowed.

A fixed plan is resolving only when no two latent states with opposing biological query decisions share the same joint outcome tuple over the selected actions.

Enumerate all feasible action subsets within the V1 bound and select the minimum total effort resolving bundle. The adaptive and fixed planners use identical worlds, nuisance models, potential outcomes, feasibility and costs.

Every fixed plan can be executed as a non-branching adaptive policy, so the exact adaptive worst-case cost must never exceed the exact fixed-plan optimum. This is a metamorphic invariant.

## 5. Known-answer fixture A: adaptivity saves effort

Four biological worlds have query decisions `true, false, true, false`. Two coherent nuisance models preserve the same response partitions but use different outcome labels.

| Action | Cost | W1 | W2 | W3 | W4 |
|---|---:|---|---|---|---|
| `screen-group` | 1 | group-A | group-A | group-B | group-B |
| `resolve-A` | 2 | A-plus | A-minus | indeterminate | indeterminate |
| `resolve-B` | 2 | indeterminate | indeterminate | B-plus | B-minus |

The second nuisance model uses distinct labels for the same partitions. This prevents the result from depending on a single outcome vocabulary.

Exact adaptive policy:
- run `screen-group`;
- after either group-A label, run `resolve-A`;
- after either group-B label, run `resolve-B`;
- stop on the two distinct resolution outcomes.

Expected worst-case adaptive cost: 3 effort units. Exact fixed optimum: `resolve-A + resolve-B`, cost 4. Adaptivity saving: 1 unit, 2,500 basis points (25%).

## 6. Known-answer fixture B: nuisance destroys the apparent gain

Append a second coherent nuisance model in which `screen-group` produces one shared outcome for all four worlds. The planner must retain all four worlds after that observation and cannot pretend it learned the group.

Expected adaptive worst-case cost: 4. Exact fixed optimum: 4. Adaptivity saving: zero. A planner that still reports cost 3 has mixed nuisance states or discarded an admissible world.

## 7. Known-answer fixture C: no guaranteed resolution

All feasible interventions have identical outcomes across the opposing-query worlds. Both adaptive and fixed-plan searches must return no guaranteed resolving plan. This is not a biological equivalence claim: it is limited to the declared finite world, nuisance and intervention universe.

## 8. Required report and audit outputs

- exact raw input SHA-256;
- canonical digests for sorted world, nuisance, action and potential-outcome sets;
- latent-state count, action count, memoized state count and search-complete flag;
- exact adaptive status, worst-case cost and full outcome-contingent policy tree;
- exact fixed-plan status, minimum effort and action bundle;
- adaptivity savings in effort units and integer basis points;
- no-guarantee disposition when no policy exists;
- explicit claim ceiling: deterministic finite potential outcomes, no probabilities, no population or biological guarantee.

## 9. Adversarial and metamorphic controls

1. Missing / duplicate world × nuisance × action cell.
2. Duplicate IDs and dangling references.
3. Infeasible action made artificially cheap; it must never enter either optimum.
4. A STOP leaf with both true and false biological worlds remaining.
5. An invented outcome branch unsupported by any latent state.
6. A repeated action on one root-to-leaf path.
7. A nuisance model that erases an apparent screening partition.
8. A nuisance ID switched between steps; no such transition is allowed.
9. Input row permutations preserve report optima and canonical digests.
10. Increasing an action cost cannot lower either optimum.
11. Adding a nuisance model cannot improve a worst-case guarantee.
12. Removing admissible worlds cannot worsen the optimum.
13. An all-overlap case returns no policy; bounded failure is not called biological equivalence.
14. Any search cap exhaustion returns `INCOMPLETE`, never `EXACT`.

## 10. Industry comparison and benchmark targets

Compare with random feasible trees, expert-authored trees, one-step greedy query-value (#10670), entropy/EIG-greedy, non-adaptive fixed-bundle optimum, degree/centrality heuristics, and matched rewired/activity-only controls. Use identical world sets, nuisance models, action universes and cost tables.

Aspirational synthetic targets (not results):
- 100% exact adaptive and fixed optimum recovery within 12 actions and 96 world × nuisance states, subject to a measured search-state cap;
- 0 false STOPs with an opposing biological world still compatible;
- 0 unsupported outcome branches or repeated/infeasible actions;
- 100% correct no-guarantee classification on overlap controls;
- adaptive cost never exceeds fixed-plan optimum;
- at least 20% lower worst-case effort on preregistered cases where adaptivity is useful, with no loss of the guarantee;
- 100% expected monotonicity on nuisance-addition, world-set and action-cost controls.

Do not present these synthetic criteria as wet-lab time savings. Prospective biological trials must compare randomized adaptive and fixed policies with blinded adjudication, matched feasibility and total cost.

## 11. Claim ceiling

An exact result proves optimality only over the declared finite biological worlds, coherent nuisance set, deterministic potential-outcome table and feasible interventions. It does not prove biological hypothesis-space completeness, causal synaptic mechanisms, source authenticity, stochastic calibration or in-vivo efficacy.