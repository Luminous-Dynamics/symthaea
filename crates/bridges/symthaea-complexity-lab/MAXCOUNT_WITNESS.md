# ETH-018 — exact MaxCount witness and optimal-solution DAG

**Status:** Python exact reference prototype passes. Rust integration, Lean proofs and hardware benchmarks remain open.

## Why this is the next extension

An optimum value alone is not operationally useful in many applications. A solver should return a concrete assignment, prove that it attains the optimum, report exact MaxCount multiplicity, and expose the structure of all optimum assignments without re-solving the instance once per variable.

Existing work already certifies optimization. VeriPB is a general pseudo-Boolean proof format used by combinatorial solvers; formally verified checking for PB optimization is practically feasible in RoundingSat/Sat4j pipelines. Recent work also gives proof logging for state-of-the-art branch-and-bound MaxSAT and certifying Pareto-optimality for multi-objective MaxSAT. This project must not claim that proof-logging optimization or Pareto certification is new.

References:
- VeriPB, proof format and ecosystem: https://veripb.org/
- Koops et al., *Practically Feasible Proof Logging for Pseudo-Boolean Optimization*, CP 2025: https://doi.org/10.4230/LIPIcs.CP.2025.21
- Vandesande, Coll & Bogaerts, *Certified Branch-and-Bound MaxSAT Solving*, AAAI 2026: https://doi.org/10.1609/aaai.v40i17.38449
- Jabs et al., *Certifying Pareto-Optimality in Multi-Objective Maximum Satisfiability*, TACAS 2025: https://doi.org/10.1007/978-3-031-90653-4_6
- Ihalainen et al., *Certified MaxSAT Preprocessing*, IJCAR 2024: https://arxiv.org/abs/2404.17316

The narrower hypothesis is to extend Symthaea's factor-elimination / MaxCount path with an exact, replayable witness interface, including tie-aware local decisions and an optimal-assignment DAG, while preserving the semiring's exact optimum and multiplicity. For overlapping MaxSAT/PB cases, benchmark interoperability with VeriPB rather than reinventing a proof language.

## Algorithm

When eliminating variable x with separator assignment y, evaluate every state a:

```
T_a(y) = product of factors in bucket(x), evaluated at (y,a)
M_x(y) = max-plus-count sum over a of T_a(y)
```

Store the MaxCount output and **all** states attaining its maximum, together with each branch's local multiplicity. The backpointer table for each eliminated variable is indexed by its separator assignment.

After forward elimination, reverse the order. The separator variables have already been assigned; select the smallest maximizing state to obtain a canonical witness. Branch over all stored maximizing states to enumerate the optimal-assignment support as an implicit decision DAG. Enumeration may itself be exponential in the number of witnesses; the DAG is the compact representation, not a promise that materializing every witness is cheap.

## Important semantic distinction

MaxCount's second component is the semiring multiplicity, not necessarily the number of distinct variable assignments. With unit local multiplicities, these coincide. If local factors encode weights/multiple derivations, a single assignment can contribute multiplicity greater than one.

The exact witness DAG represents distinct variable assignments only. The dual-count extension also computes the distinct assignment total during elimination, rather than requiring DAG enumeration. The checker separately verifies:
1. every local argmax set against the original bucket tables;
2. every intermediate MaxCount message and tie multiplicity;
3. the final optimum and total semiring multiplicity;
4. the reconstructed canonical witness against the original factor system;
5. the complete optimum-assignment support represented by the DAG on small exhaustive tests.

Example: triangle MaxCut has optimum 2, multiplicity 6 and six distinct optimal assignments. A weighted mixed-cardinality example has optimum 6, semiring multiplicity 4, but only one distinct optimum assignment. Conflating these quantities would be incorrect.

## Dual-count extension: multiplicity plus distinct assignment support

The same elimination pass can compute a second count that must not be confused with MaxCount multiplicity:

- The **multiplicity channel** multiplies factor multiplicities and sums them across score ties.
- The **distinct-support channel** initializes each feasible original factor-table cell to support 1 and each infeasible cell to 0. During elimination, support counts multiply across factors whose eliminated-variable sets are disjoint, and add across tied values of the currently eliminated variable.

This is valid for the current API because factors are created from original tables, where each table cell is one local assignment. It does not silently support pre-aggregated factors that encode overlapping hidden assignment sets; any future API admitting those must supply provenance or reject the distinct-support output contract.

Rust additions in `semiring.rs`:
- `MaxCountBackpointerRow.branch_assignment_counts` alongside `branch_multiplicities`;
- `MaxCountWitnessResult.distinct_optimal_assignments`;
- support-count operation counters with checked overflow;
- `OutputContract::OptimumWithWitnessMultiplicityAndDistinctAssignmentCount` and `SuggestedAlgebra::MaxCountWithDistinctSupport`.

The certificate format is now v2 dual-count. The independent checker recomputes each intermediate MaxCount table, each support table, the branch tie list and both final counts. It rejects rehashed false support values, not just bad hashes.

## Executed Python reference

- `reference/maxcount_witness_oracle.py`: producer, separate replay checker, direct assignment oracle, and backpointer-DAG enumerator.
- `reference/maxcount_witness_results.json`: deterministic run output.

Results (seed 20261103):
- 240/240 randomized mixed-cardinality factor graphs: exact MaxCount result equals direct assignment enumeration;
- 240/240: independently checked certificate accepted;
- 240/240: backpointer DAG's distinct optimum-assignment support equals direct enumeration support;
- 2/2 handcrafted cases, including triangle MaxCut and weighted multiplicity-versus-distinct-witness distinction;
- 8/8 deliberately rehashed false/stale/mutated certificates rejected.

All integer arithmetic uses Python arbitrary-precision integers. The checker has a separate replay implementation and recomputes local argmax sets; recomputing the certificate digest after mutation does not make a false trace acceptable.

## Exact uniform sampling of optima

The distinct-support counts permit exact uniform sampling without enumerating every optimum. At each reverse backpointer step, select a maximizing state with probability proportional to its branch-support count. The branch count is the product of distinct-support counts for disjoint child blocks in the bucket; prior backpointers then sample those child blocks conditionally. The probability products telescope to 1 / total distinct optimum support.

Rust adds ExactFactorGraph<MaxCountValue>::sample_optimal_assignment. The caller supplies a uniform_below(bound) callback that must return an unbiased integer in [0,bound). This avoids a dependency on a specific RNG crate. The sampler validates the variable permutation, separator ordering, row encoding, parallel branch-array lengths, positive branch counts, root-support product, and the returned assignment's score. It trusts the supplied argmax/support contents; externally supplied results must be independently certificate-checked before sampling.

The Python v2 reference checks 1,383 path probabilities exactly with fractions.Fraction: each tested optimum has probability exactly 1 / support_size; all six triangle MaxCut optima have probability 1/6. This is an exact finite probability check, not a runtime benchmark.

## Benchmarks and qualification targets

Compare on identical frozen factor systems and budgets:
- optimum-only MaxCount;
- MaxCount plus one canonical witness via backpointers;
- optimum-support DAG via backpointers;
- repeated conditional self-reduction (one or more solver calls per variable/state);
- plain max-sum/treewidth DP with unverified backpointers;
- certified PB/MaxSAT baselines (VeriPB, MaxCDCL/MaxSAT) where input semantics match.

Report separately: optimum correctness, multiplicity correctness, witness validity, optimal-support precision/recall, backpointer bytes, certificate bytes, solve time, backtrack time, independent-check time, peak RSS, and additional solver calls avoided. Include low/high treewidth graph/CSP families, tie-heavy and unique-optimum cases, weighted factors, negative scores, infeasible instances, and unconstrained variables.

Aspirational gates (not achieved): 100% agreement with exhaustive support on tractable cases; zero accepted mutated certificates; no re-solving required for the canonical witness; exact multiplicity preserved; median backpointer overhead below 25% on a declared low-width slice; verified solve+check overhead below 2x against the same unverified elimination baseline; clear Pareto comparison with PB proof systems on overlapping MaxSAT cases.

## Claim ceiling

- Python finite reference != Rust implementation or Lean proof;
- MaxCount multiplicity != distinct witness count in weighted encodings;
- an optimal-solution DAG != a polynomial-size DAG for every instance;
- proof-carrying backpointers != a new general proof system;
- no hardware runtime, broad benchmark win or SOTA claim yet.
