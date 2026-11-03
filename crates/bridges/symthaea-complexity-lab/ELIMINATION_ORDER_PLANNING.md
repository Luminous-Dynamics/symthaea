# ETH-017: output-algebra- and certificate-aware elimination ordering

**Status:** Exhaustive tiny-instance reference pilot executed; no Rust implementation, broad benchmark, or SOTA claim yet.

## Research positioning and prior art

Elimination order is a first-order determinant of variable-elimination and tensor-network contraction cost. Finding an optimal order is NP-hard in general. Standard baselines include min-neighbors/min-degree, min-weight, min-fill and weighted min-fill. Their rankings are not universally interchangeable; variable cardinalities can change the result. See:

- Peyrard et al., *Exact or approximate inference in graphical models: why the choice is dictated by the treewidth, and how variable elimination can be exploited*, ANZJS 2019: https://doi.org/10.1111/anzs.12257
- Koller & Friedman, variable-elimination ordering discussion (min-neighbors, min-weight, min-fill and weighted min-fill): https://web.stanford.edu/~shervine/teaching/cs-221/cheatsheet-variables-models/
- Derkinderen et al., *Ordering Variables for Weighted Model Integration*, UAI 2020, including bottom-up min-fill for hybrid discrete-continuous inference: https://proceedings.mlr.press/v124/derkinderen20a.html
- Stoian, Milbradt & Mendl, *On the Optimal Linear Contraction Order of Tree Tensor Networks, and Beyond*, SIAM J. Sci. Comput.: https://doi.org/10.1137/23M161286X
- Tensor-network ordering work also evaluates total FLOPs and memory/read-write costs rather than relying only on largest intermediate rank: https://www.cpsjournals.cn/en/article/doi/10.1088/1674-1056/adbee2

This work does **not** claim novelty for elimination ordering, min-fill, weighted min-fill, or output-sensitive ordering generally. The hypothesis to test is narrower: for the same factor graph and requested output, the cheapest order can depend on the exact coefficient algebra and its intermediate representation; a proof-carrying path must also price trace size and replay/checking work. This must be compared against weighted model integration, tensor-network contraction planners and algebraic inference baselines before any novelty claim.

## Paradigm shift

Do not ask only: “Which order minimizes induced width?” Ask:

> Which valid order minimizes the cost vector for this requested exact output, coefficient representation and verification contract, subject to time and memory constraints?

The planner exposes a Pareto set, not one unmotivated weighted sum. Candidate metrics:

1. semiring additions and multiplications;
2. algebra-specific work proxy (e.g. support-pair products, coefficient bit-height work, Pareto dominance comparisons);
3. total generated factor entries and peak live table entries;
4. total/peak profile or Pareto support;
5. coefficient bit-height;
6. certificate bytes, with checker replay cost measured separately in the production benchmark.

The planner selects an algebra from the output contract first. It then generates candidates from structural heuristics and bounded search, executes or estimates their algebra-specific cost, and returns the nondominated plans. A caller can choose, for example, lowest estimated arithmetic work subject to a memory limit, or smallest certificate subject to a solve-time budget.

## Executed finite pilot

Reference implementation: src/elimination_order_planner.py. Full per-order output: src/elimination_order_planner_results.json. Compact summary: src/elimination_order_planner_summary.min.json.

Frozen graph portfolio: 10 five-variable mixed-cardinality factor graphs with domain sizes [2,3,2,3,2]. Two are hand-designed (mixed cycle/chords and lopsided hypergraph); eight are deterministically generated from fixed seeds 20261103 through 20261110. These are public/reproducible pilot cases, not hidden holdouts.

For each graph, every one of its 120 variable permutations was executed under MaxCount (optimum and multiplicity), two-objective ParetoCount, and one-variable polynomial profile. Total: **3,600 exact order/algebra executions**. All 120 orders in every graph/algebra combination returned the same result. The direct full-assignment oracle agreed in all **30/30** graph/algebra cases. **193 selected certificates** were replayed by the independent checker.

### Aggregate observed results

| Metric across 30 graph/algebra cases | Result |
|---|---:|
| Deterministic min-fill regret vs exhaustive best algebra-work proxy, median | 16.96% |
| Same regret, mean | 17.07% |
| Same regret, maximum | 40.94% |
| 32-start stochastic min-fill gain vs deterministic min-fill, median | 10.47% |
| Same stochastic gain, mean | 13.33% |
| Minimum certificate-size reduction vs min-fill, median | 5.54% |
| Same certificate-size reduction, maximum | 16.27% |
| Graphs where best work-proxy order differs across the three algebras | 9/10 |

The exact order minimizing the algebra-work proxy differed among MaxCount, ParetoCount and polynomial profiles on 9 of the 10 graphs. The 32-start stochastic min-fill pool narrowed the gap on average but did not consistently recover the exhaustive best. Across cases, cost-vector Pareto frontiers ranged from 1 to 23 orders.

These differences compare deterministic min-fill (variable-ID tie-break) with the best of all 120 orders **under this proxy on these tiny synthetic instances**. They are not runtime speedups. The algebra-work proxy is transparent but model-dependent: it charges support operations, approximate coefficient-bit work and, for the reference Pareto implementation, quadratic dominance comparisons. The reference certificate byte count is exact UTF-8 JSON size under certificate v1, not production Rust trace size.

## What this proves and does not prove

**It does show:** exact output invariance across all 120 orders on each frozen case; measurable algebra-specific intermediate-value costs in the finite reference; different best proxy orders for different output algebras; trade-offs between operation proxy, table size and certificate bytes; and that selected proof traces can be replayed against the separate checker.

**It does not show:** a new approximation or a polynomial-time optimizer for optimal elimination order; that the proxy predicts real CPU/GPU time or cache behavior; a scalable reduction in solve time/RSS; an improvement over current SOTA heuristics on public benchmark suites; or that the planner's own output is a proof of optimality.

## Next work and fair benchmark protocol

1. Expand from tiny exhaustive graphs to a frozen public portfolio: Bayesian networks, UAI/PGM, weighted model integration, SAT/#SAT incidence graphs, MaxCut/QUBO, sparse CSPs and tensor networks. Keep compatible semantics separate.
2. Baselines: min-degree, min-fill, min-weight, weighted min-fill, min-factor, bottom-up min-fill where applicable, randomized multi-start, tree-decomposition heuristics, and exact best-order enumeration only on small cases.
3. Add online cost probes and sampled partial-order rollouts so order candidates do not require full execution just to be ranked. Compare predicted cost to realized cost; calibrate separately per algebra.
4. Record raw time/CPU, peak RSS, operations, peak live entries, support sizes, coefficient bit-height, certificate bytes, checker time, verifier RSS, and result/certificate hashes.
5. Use paired hardware/time budgets, pinned commits, frozen train/tune/held-out sets and multiple seeds. Include order-search overhead and certificate verification in end-to-end costs.
6. Report performance profiles and per-family results, not only means. Publish the Pareto frontier of time, memory and proof overhead.
7. Formalize the basic VE correctness invariant independently of the order-selection heuristic. Any heuristic-selected order is only required to be a permutation; optimality claims need separate evidence.
8. Use exact order enumeration on n <= 9 (where feasible) as an oracle for heuristic regret; scale via treewidth bounds and randomized/local search.

Aspirational qualification gates—not achieved: zero exact-result mismatch across all supported algebras and order sources; measured calibration error for each cost proxy, reported on held-out graphs; planner overhead included and kept below 5% of total solve time on an agreed medium-instance slice, or justified where the search returns value; at least 10% reduction in a preregistered primary metric (real solve time or verified end-to-end time) on a held-out structural slice, with no correctness regression and with RSS/certificate trade-offs reported; no broad SOTA claim without public benchmarks, matched budgets and third-party reproduction.
