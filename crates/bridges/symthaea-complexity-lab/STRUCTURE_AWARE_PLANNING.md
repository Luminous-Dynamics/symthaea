# ETH-019: structure-aware exact computation planning

**Status:** Research specification plus a finite Python component-decomposition reference prototype. The broader structural portfolio and all performance claims remain open.

## 1. Research objective

ETH-017 searches elimination orders. ETH-018 searches algebraically equivalent computation representations and certifies local rewrites. ETH-019 broadens the search space to structural decompositions that may expose tractability not visible from the primal graph alone.

The planner should ask:

1. What is the requested exact output and coefficient algebra?
2. Which representation best exposes the problem's structure?
3. Which structural witness yields a feasible exact plan under time and memory limits?
4. Can an independent verifier validate that witness and the resulting computation?
5. What is the total measured cost after including proof construction and checking?

This is not a new treewidth, hypertree-width, or tensor contraction algorithm by itself. The hypothesis is a proof-aware portfolio whose candidate structural witnesses are compared using algebra-specific resource costs and independent verification cost.

## 2. Structural views

Given factors over finite-domain variables, derive and preserve several views:

- **Primal graph:** variables adjacent when they share a factor. Candidate parameters include treewidth, pathwidth, separator size and elimination width.
- **Factor hypergraph:** variables are vertices, factor scopes are hyperedges. Track rank, edge cover, generalized/fractional hypertree decompositions and adaptive width where feasible.
- **Incidence graph:** variable nodes and factor nodes. Connected components and articulation separators can expose decompositions hidden by primal clique expansion.
- **Tensor network:** factors are tensors, shared variables are indices. Track contraction tree, index dimensions, bond dimensions, slicing and largest intermediate.
- **Symmetry structure:** automorphism orbits and stabilizers. Use quotienting only with a proof that preserves the requested output and multiplicities.
- **Conditioning structure:** cutsets whose assignments split the residual graph into smaller components.

Do not collapse these views to one scalar width. Each has distinct semantics and extraction cost.

## 3. Prior art and claim boundary

Relevant established results include:

- Fractional hypertree width as a structural parameter for CSP tractability: Marx, SODA 2010, https://doi.org/10.1137/1.9781611973068.98
- Exact FPT algorithms for generalized/fractional hypertree width and adaptive width under bounded-rank/degree parameterisations: Lanzinger, Razgon & Unterberger, PACMMOD 2026, https://doi.org/10.1145/3801900
- Threshold treewidth and hypertree width incorporate problem-specific solving costs into structural decomposition: Ganian et al., IJCAI 2020, https://doi.org/10.24963/ijcai/2020/263
- opt_einsum dynamic programming already performs trace extraction and splits disconnected contraction components: https://optimized-einsum.readthedocs.io/en/stable/dp_path.html
- cotengra already searches over contraction trees, subtree reconfiguration, slicing, dynamic slicing, and objectives such as FLOPs or largest intermediate size: https://cotengra.readthedocs.io/en/stable/advanced.html
- Exact inference complexity is closely related to treewidth and elimination order: Peyrard et al., 2019, https://doi.org/10.1111/anzs.12257

The opportunity is integration and qualification: algebra-aware structural portfolios with replayable decomposition witnesses and proof-aware end-to-end costs. No novelty or SOTA claim is justified until compared with those systems.

## 4. Candidate portfolio

Initial candidate generators (not all need be enabled at once):

| Family | Candidate | Witness to check |
|---|---|---|
| Primal graph | min-degree, min-fill, weighted min-fill | elimination order and induced fill trace |
| Cutset | small conditioning sets, residual component split | chosen cutset, domains, residual components |
| Hypergraph | bounded-width decompositions, fractional covers | bags, guards, fractional-cover feasibility |
| Incidence graph | component/articulation decomposition | component partition and boundary |
| Tensor | contraction trees, slicing, subtree reconfiguration | contraction tree, index sets, slice coverage |
| Symmetry | orbit quotient | group action, orbit partition, stabilizer/multiplicity proof |

The first implementation should use bounded candidate search and exact small-instance checking. It must not claim an optimal decomposition when it only finds a heuristic upper bound.

## 5. Algebra-aware cost model

Cost vectors should include:

- number of semiring additions/multiplications or field operations;
- estimated and observed bucket/contraction operation counts;
- peak live table entries and actual peak bytes;
- coefficient bit-height and arbitrary-precision arithmetic cost;
- polynomial support or Pareto-frontier cardinality;
- number of tensor slices and recombination work;
- certificate size, proof-construction time, checker time and checker peak RSS;
- planner time, number of candidate plans, and timeout/memory-cap feasibility.

Predicted work must remain separate from observed wall time. A candidate should not be scored as "verified cost" unless the certificate is actually checked.

Use a Pareto set rather than one weighted sum until calibration data supports a well-justified scalar objective.

## 6. Benchmark matrix

Freeze public instances by structural family and use exact output contracts:

1. CSP hypergraphs: sparse high-arity factors, bounded-rank instances, bounded fractional hypertree width, and examples where primal clique expansion is poor.
2. Factor graphs: chains, trees, grids, disconnected components, articulation separators, high-cardinality variables, and dense local factors.
3. Tensor networks: chain, tree, 2-D/3-D grids, random sparse hypergraphs, heterogeneous bond dimensions and memory-limited cases requiring slicing.
4. Symmetric instances: graph coloring, independent sets and model-counting cases with nontrivial automorphism groups.
5. Conditioning: small cutsets that turn a high-width graph into multiple low-width components.

Baselines:
- min-degree/min-fill/weighted-min-fill and bounded exact ordering search;
- tree decomposition and available hypergraph-width tools for matching structural goals;
- opt_einsum, cotengra and cuQuantum-compatible planners for matching tensor objectives;
- exact counting baselines such as Ganak/D4/CPOG only when the requested output matches.

No comparison of a contraction FLOP estimate against an exact solver's wall-clock completion is a fair performance result.

## 7. Qualification gates

- Every structural witness passes an independent checker.
- Every supported candidate plan agrees with a direct small-instance oracle.
- Slicing certificate proves disjoint, complete assignment/index coverage with exact recombination.
- Cutset certificate proves the residual components and domains cover all original assignments exactly once.
- Symmetry certificate proves the action preserves constraints, weights and requested output multiplicities.
- A held-out structured family demonstrates a better *measured verified* result than the graph-only baseline.
- Target at least 10% improvement in verified end-to-end solved instances or cost on a preregistered slice, under identical CPU, RAM and time budgets, with planning overhead below 5%. These are aspirations, not achieved results.

## 8. Work sequence

1. Canonicalize factor hypergraph and incidence representations.
2. Add structural descriptors and connected-component witnesses.
3. Add cutset/conditioning candidate generation and compositional exact certificates.
4. Add tensor contraction and slicing witnesses.
5. Investigate fractional hypertree decompositions and fractional-cover certificates.
6. Add symmetry only after group-action and multiplicity proofs are available.
7. Connect all candidate plans to ETH-015 output algebras, ETH-016 certificates, ETH-017 cost-aware ordering and ETH-018 rewrite proofs.
8. Benchmark on frozen held-out instances against compatible external tools.

## 9. Executed finite component-witness reference

Prototype files:
- `reference/component_decomposition.py`
- `reference/component_decomposition_checker.py`
- `reference/component_decomposition_results.json`

The producer builds canonical connected components using union-find over the primal variable graph. The independent checker reconstructs the expected components using BFS, not union-find, then validates canonical ordering, exact variable/factor coverage, factor-scope containment, table dimensions, instance digest and the closed algebra tag.

Deterministic seed `0xE7019` results:
- 1,000 random structural factor-graph instances.
- 240 additional random instances checked semantically.
- 720 direct-enumeration versus component-product comparisons: 240 each for Boolean feasibility, natural-number weighted count and MaxCount (optimum plus multiplicity).
- 9 special semantic controls covering empty instances, isolated variables, disconnected factors and zero-scope scalar factors.
- Independent subprocess checker accepted 1,729/1,729 positive witnesses and rejected 6/6 malformed witnesses.
- Up to 11 components appeared in the structural random corpus.

The component product is computed exactly as the product of component-local finite sums. This uses the commutative-semiring decomposition identity; the prototype's executable evaluator covers only bool/nat/MaxCount. The closed witness tag catalogue also lists Pareto and polynomial algebras, but those are not included in this ETH-019 semantic experiment.

This is finite reference evidence, not a generalized/fractional hypertree-width result, tensor-contraction planner, slicing optimizer, symmetry reduction, Rust qualification, Lean proof or performance benchmark. The reference tests use small domains and direct enumeration; no runtime advantage is claimed.

## 10. Claim ceiling

This document defines the research question and test programme. It contains no structural planner implementation, proof of an FPT result, measured speedup, or SOTA result.
