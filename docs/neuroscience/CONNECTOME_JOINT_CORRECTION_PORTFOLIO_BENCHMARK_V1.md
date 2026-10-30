# Joint correction interaction and portfolio benchmark V1

**Status:** protocol; no joint-portfolio optimizer or biological interaction result is implemented.
**Tracking:** [SCI-CONNECTOMICS-011 / #9250](https://github.com/Luminous-Dynamics/symthaea/issues/9250).

## 1. Problem: isolated candidate scores are not compositional

The current query-impact ranker evaluates individual mutations against an immutable baseline and its retrospective policy replay applies proposals sequentially. That is a useful safety improvement over assuming all edits are additive, but isolated sensitivity scores do not reveal the value of jointly selecting a set of corrections.

Two corrections can be redundant, synergistic, antagonistic, mutually exclusive, or order-dependent. Directed reachability, feed-forward motifs, cycles, and downstream query metrics are nonlinear. Stale segment/root identities and shared morphology subgraphs also make edit order consequential.

## 2. Formal estimand

Freeze baseline graph G, candidate edit definitions, edit semantics, query bundle Q, query weights, oracle graph, and review-minute costs. Define U(S) as the preregistered signed reduction in weighted query error after applying compatible candidate set S under the versioned edit semantics.

Pairwise interaction:

I_ij = U({i,j}) - U({i}) - U({j}) + U(empty).

Positive values indicate synergy; negative values indicate antagonism or redundancy under the chosen utility. Pairwise terms are not generally sufficient: measure higher-order residuals for sampled sets and report how well the pairwise approximation predicts joint utility.

## 3. Edit semantics and evidence contract

Each candidate must carry exact source graph and segmentation digests; candidate and source identities; operation type; affected roots, segments, edges, and morphology subgraphs; prerequisite/conflict links; review-minute estimate or measured time with provenance; and an immutable evidence digest.

Define whether edits are applied sequentially or as a simultaneous batch, how stale references are detected, whether operations commute, and how two edits to the same object are resolved. The semantics cannot be selected after seeing oracle utility.

Oracle query values remain evaluator-only. Do not pass oracle labels or outcomes to candidate generation, ranking, or portfolio search during lockbox evaluation.

## 4. Interaction discovery

Start with a sparse dependency graph over candidates. Add an interaction edge when candidates share or structurally affect a graph object, query path, motif, root, segment, morphology subgraph, or spatial boundary. Preserve the reason for each edge.

Evaluate isolated utilities and joint two-candidate counterfactuals for all interacting pairs. For pairs with no known dependency, sample a random subset to estimate the omitted-interaction rate. Test order permutations on noncommuting edit sets.

For larger portfolios, compare:
- isolated-score greedy baseline;
- probability-only, uncertainty-only, and deterministic-random controls;
- conflict-aware greedy selection;
- pairwise-interaction-aware selection;
- bounded beam search / portfolio search with explicit compute budget;
- exact optimum on enumerable small candidate sets.

All policies share the same candidate pool, review-minute budget, edit semantics, and oracle. The portfolio optimizer is an offline selection algorithm; it does not make reviewer-time savings causal.

## 5. Benchmark fixtures

Construct finite synthetic graph suites with exact optimal sets and controlled cases:
- independent edits whose utilities are additive;
- duplicate / redundant corrections;
- complementary edits that only help jointly;
- mutually exclusive or contradictory edits;
- edits whose utility depends on order;
- shared-edge and stale-root conflicts;
- path-existence changes through alternate paths;
- feed-forward-loop and directed-cycle interactions;
- two- and three-query trade-offs;
- higher-order interactions not captured by pairwise terms.

Vary graph size, candidate count, interaction density, review costs, uncertainty, and invalidation rates. For small portfolios, enumerate all subsets and all valid orderings to provide exact oracle optima.

## 6. Metrics

- Joint signed query-error reduction at fixed expert-minute / edit-count budgets.
- Regret relative to exact compatible portfolio optimum on enumerable cases.
- Pairwise interaction sign and magnitude error; high-order residual error.
- Invalid/stale edit rate, order sensitivity, conflict-resolution time, and rework.
- Robustness to query-weight uncertainty and missing / indeterminate oracle values.
- Runtime and memory versus candidate count and dependency density.

## 7. Proposed targets (not results)

- Zero regret on fully enumerable synthetic cases with ≤20 candidates when exact search is enabled.
- At least 10% better held-out query-error reduction than the strongest frozen greedy baseline at identical review budget on larger synthetic and then biological instances.
- No increase in false accepted-correction risk; all changes remain human-approved and reversible.
- No causal human-efficiency claim without the prospective trial in #9242.

## 8. Promotion boundary

Portfolio-level gains are first reported as counterfactual algorithmic utility, not observed scientific improvement. Biological evaluation requires an independent adjudicated oracle and lockbox split by volume/neuron/source operation. Actual expert-minute and correction-quality effects require randomized prospective workflow assignment.