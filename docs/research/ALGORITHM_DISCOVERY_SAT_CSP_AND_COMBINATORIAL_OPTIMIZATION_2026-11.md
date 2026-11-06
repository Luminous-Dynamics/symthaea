# SAT, 3-SAT, CSP and Combinatorial Algorithm Discovery

**Status:** research plan and repository audit; not an implementation or performance claim  
**Date:** 2026-11-06  
**Owner:** Luminous Dynamics / Symthaea research  
**Scope:** Symthaea, Mycelix, Symtropy, Sol-Atlas and shared Luminous infrastructure

## Executive decision

Make Boolean satisfiability and constraint solving the first shared algorithm-discovery workstream. Use **3-SAT as the theoretical core**, but do not reduce the programme to 3-SAT alone. Build a shared, typed problem contract and evidence pipeline; retain small in-house solvers as transparent references; compare production performance against mature external solvers; let Symthaea evolve solver components and portfolios; require independent validation of every result.

The first task is correctness, not speed. The current Symthaea propositional CNF path has a concrete semantic defect (see Repository audit). It must not be used as a research baseline for arbitrary formulas until repaired and exhaustively cross-checked.

The theoretical and applied programmes should be separate but connected:

- **Theory:** SAT, #SAT, 3-SAT, parameterized algorithms, lower-bound assumptions, formal proof and complexity analysis.
- **Engineering:** SAT/SMT/PB/CSP, MaxSAT, graph optimization, scheduling, routing and industrial decision problems.
- **Discovery:** LLM/HDC-guided construction, mutation, selection and composition of algorithms, with executable correctness gates.

## Why 3-SAT is central

3-SAT is NP-complete and is a useful bridge between complexity theory, exact algorithms, model counting, constraint learning and practical SAT. It also has several non-interchangeable evaluation targets:

- **Decision 3-SAT:** return SAT with a valid assignment, or UNSAT with a verifiable proof.
- **#3-SAT:** return the exact number of satisfying assignments.
- **Random 3-SAT threshold:** characterize the asymptotic satisfiable/unsatisfiable transition for a specified random formula distribution.
- **Max-3-SAT:** maximize the number (or total weight) of satisfied clauses.
- **Weighted / projected / incremental variants:** solve different operational or probabilistic questions; do not report them as ordinary 3-SAT.

A benchmark result for one target does not establish performance on the others.

## Direct cross-problem consequences claimed by the OpenAI #SAT preprint

The October 25 preprint contains a particularly relevant corollary, not just a 3-SAT result. It claims deterministic subexponential-time exact counting for:

- Independent sets in a simple graph with v vertices and e edges: 2^o(v) time and polynomial space.
- Proper q-colorings, for each fixed q >= 2: 2^o(v) time and polynomial space.
- Perfect matchings: 2^o(v+e) time and polynomial space.

The paper gives counting-preserving encodings: independent sets map directly to 2-CNF; fixed-q graph colorings use q variables per vertex with exactly-one and adjacency constraints; perfect matching uses one variable per edge and a linear-size checking circuit. These are **claims inherited from the preprint's main theorem**, not independently reproduced results.

Important reduction boundary: the preprint explicitly warns that 2^o(n) time does not automatically transfer through arbitrary polynomial-time reductions. The number of variables in the fixed-width SAT encoding must remain controlled. Its graph-counting corollary is therefore concrete; it does not imply that every NP problem has a subexponential algorithm.

Research priority: graph coloring and independent-set counting should join #3-SAT as a first-class theoretical reproduction track; perfect-matching counting is a separate algebraic/counting track. Also test the encodings against native graph algorithms on small instances and verify counts exhaustively.

## Current literature worth implementing or reproducing

### 1. OpenAI: “The Exponential Time Hypothesis is False” (2026-10-25)

[OpenAI preprint and Lean materials](https://github.com/openai/math/tree/main/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026)

The October 25 manuscript claims a deterministic, polynomial-space algorithm that exactly counts satisfying assignments of arbitrary fixed-width k-CNF formulas, for every fixed k >= 2, in time

$$
(n+m)^{O_k(1)} 2^{O_k(n\log\log n/\sqrt{\log n})}.
$$

For sparse formulas with m <= Dn (fixed D), the manuscript states the stronger bound

$$
(n+m)^{O_{k,D}(1)} 2^{O_{k,D}(n/\sqrt{\log n})}.
$$

The public Lean scope note is deliberately recorded separately: for canonical encodings of k-CNF, it states a formally proved time bound of

$$
(L+1)^{O_k(1)}2^{O_k(n(\log\log n)^2/\sqrt{\log n})},
$$

where L is the encoding bit length, with polynomially bounded visited tape cells. Do not silently replace this Lean scope by the sharper manuscript bound. A theorem-statement/comparator audit is required to establish exactly which paper statement each checked proof supports.

Since a formula is satisfiable exactly when its model count is positive, the claimed manuscript algorithm refutes ETH; its fixed-k consequence also conflicts with SETH. Track any additional claimed consequences (including gap or nondeterministic variants) against their exact promise problems and reductions rather than repeating a headline list without checking the statement.

**Interpretation:** if the manuscript theorem and proof correspondence are correct, this is a major change to fine-grained complexity and a high-priority formal-reproduction target. The stated algorithms remain superpolynomial; they do not give a polynomial-time SAT algorithm and do not prove P = NP. The asymptotic bound is not evidence of a practical SAT-solver speedup. Hidden constants, input encoding, instance size, reductions, proof-to-paper correspondence and practical resource use all matter.

[OpenAI preprint: full version](https://github.com/openai/math/blob/main/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/build/main_full_version.tex)  
[Independent community mirror and scope notes, updated 2026-11-04](https://mathcommunity.github.io/oai-math-revised/papers/The-Exponential-Time-Hypothesis-is-False-October-25-2026/)

The community mirror explicitly warns that no one has yet checked whether the Lean statements exactly match the paper's claims. OpenAI's announced external review should be treated as scrutiny in progress, not as a certification. Our reproduction record must distinguish: theorem statement, Lean kernel-checked proof, paper correspondence, algorithm extraction, asymptotic analysis, implementation, and practical evaluation.

### 2. OpenAI: “Computing the Random 3-SAT Threshold” (2026-09-27)

[Preprint and formal materials](https://github.com/openai/math/tree/main/preprints/Computing-the-Random-3-SAT-Threshold-September-27-2026)

This work claims that the limiting satisfiability threshold for the specified uniform random 3-SAT distribution is a computable real: one finite deterministic machine can approximate it to arbitrary requested precision. The paper credits Gaia Carenini's result for establishing threshold existence.

Crucial scope boundary: this computes a distributional asymptotic threshold; it does **not** solve a supplied 3-SAT instance, does not give a practical SAT algorithm, and states no efficiency bound for the approximation procedure. It also leaves behavior exactly at the threshold open.

**Use for Symthaea:** probabilistic reasoning, finite-certificate search, distribution-aware benchmark generation, and separation of asymptotic distributional claims from worst-case algorithmic claims.

### 3. Kutzkov: “Complementary Branching for #SAT” (IJCAI 2026)

[IJCAI proceedings](https://www.ijcai.org/proceedings/2026/254)

The paper introduces complementary branching based on
$$
#(F AND G) = #F - #(F AND NOT G),
$$
using subformula-level branching in addition to variable branching. It reports a deterministic O*(1.637^n) upper bound for #3-SAT, improving the preceding O*(1.6423^n) result, and parameterized bounds for sparse/bounded-degree #SAT.

**Use:** implement a small, separately reviewed research prototype; test whether complementary branching helps on bounded-degree formula families; do not conflate exact counting with decision SAT.

### 4. Jiang and Cai: “A Better Analysis for PPSZ for 3-SAT” (2026)

[arXiv:2607.10697](https://arxiv.org/abs/2607.10697)

The paper improves the analysis of PPSZ and reports a randomized worst-case decision bound of O*(1.307031578^n) for general 3-SAT. It is an analytical improvement to the algorithm's bound, not a 30% or similar practical runtime claim.

**Use:** theoretical comparison point; reproduce numerical/certified analysis if the project has the relevant exact-arithmetic infrastructure. Compare the randomized decision guarantee separately from deterministic #3-SAT counting.

### 5. AutoModSAT: LLM discovery inside a modular SAT solver (Nature Communications, 2026-07-17)

[Paper](https://www.nature.com/articles/s41467-026-74949-2) · [Code/data record](https://doi.org/10.5281/zenodo.17856395)

AutoModSAT modularizes a CDCL solver into seven heuristic functions and uses LLM-generated candidates, prompt diversification, presearch and a (1 + lambda) evolutionary process. It reports 40% improvement over its modular baseline and 30% over the compared state-of-the-art solvers across its selected datasets, with wins on most (not all) of the 11 datasets.

This is directly relevant to Symthaea. Treat those reported improvements as results on that paper's test corpus, not proof of universal superiority over Kissat or CaDiCaL. Its approach and failure cases should be reproduced before adapting it.

### 6. LACE: complementary heuristic evolution (Nature Machine Intelligence, 2026-10-01)

[Paper](https://www.nature.com/articles/s42256-026-01307-8) · [Code/data](https://github.com/PJ-NTU/LACE)

LACE's input/output/tool/heuristic-portfolio interface and complementary evolution are a strong model for a domain-independent algorithm-discovery layer. It evaluates a broad CO benchmark and structurally new problems. Reuse its interface and evaluation ideas where licensing and engineering review permit; do not assume its results transfer unchanged to SAT or industrial workloads.

## Additional OpenAI 2026 algorithmic results with direct transfer potential

The broader OpenAI math release contains several results that are useful beyond SAT. Keep these as separate tracks; they solve different problems and carry different guarantees.

### General-graph perfect-matching counting FPRAS (2026-09-23)

[Paper and source](https://github.com/openai/math/tree/main/preprints/A-Fully-Polynomial-Randomized-Approximation-Scheme-for-Perfect-Matchings-in-General-Graphs-September-23-2026)

The paper claims an FPRAS for the number of perfect matchings in every finite simple undirected graph. For rational accuracy 0 < epsilon < 1 and failure probability 0 < delta < 1/2, it returns a nonnegative rational estimate with relative error at most epsilon with probability at least 1-delta, in worst-case bit time polynomial in input length, epsilon^-1 and log(delta^-1). It returns zero with certainty when no perfect matching exists. This is **approximate counting**, not an exact counter and not a maximum-weight matching solver.

**Transfer:** Mycelix resource/partner matching and allocation analytics; Symtropy assignment and maintenance scheduling; Sol-Atlas network matching. Benchmark estimator bias, variance, confidence calibration, runtime and exact small-graph agreement against brute force.

### Almost-linear-time maximum-cardinality matching (2026-09-24)

[Paper and source](https://github.com/openai/math/tree/main/preprints/Almost-Linear-Time-Maximum-Cardinality-Matching-in-Sparse-General-Graphs-September-24-2026)

The paper claims one uniform randomized algorithm finding a maximum-cardinality matching in a simple undirected graph with n vertices and m edges in (n+m)^(1+o(1)) time, with probability at least 2/3. The time bound holds on every computation path in the stated logarithmic-word model; the result is a randomized success guarantee, not deterministic correctness.

**Transfer:** graph preprocessing and operational matching in transport, supply chains, workforce and resource allocation. Compare exact matching size and certificate validity; report failure probability and time distribution separately.

### FPRAS for common integer polymatroid bases (2026-10-05)

[OpenAI Math preprint](https://github.com/openai/math/tree/main/preprints/An-FPRAS-for-Common-Integer-Polymatroid-Bases-with-Binary-Capacities-October-5-2026)

The paper claims an FPRAS for counting common integer bases of two integral polymatroids of equal total rank, with exact rank-value oracles and binary-encoded capacities. Its matroid-oracle corollaries include common independent sets under several cardinality regimes.

**Transfer:** constrained resource allocation, intersection of capacity policies, and counting feasible allocations. Keep oracle complexity and binary-encoded numeric sizes explicit; do not assume that a practical rank oracle is free.

### FPRAS for contingency tables (2026-09-24)

[Paper and source](https://github.com/openai/math/tree/main/preprints/An-FPRAS-for-Cell-Bounded-Contingency-Tables-September-24-2026)

The work claims sampling and approximate-counting results for nonnegative integer matrices with prescribed row/column margins and arbitrary cell bounds, including structural zeros. This directly relates to synthetic data, privacy-preserving statistics, allocation tables and constrained planning.

**Transfer:** benchmark exact small cases, marginal preservation, cell restrictions, sampling uniformity/total variation where claimed, and approximation error.

### Approximation-hardness boundaries: Vertex Cover and Max-Cut

[Factor-Two Hardness Threshold for Vertex Cover](https://github.com/openai/math/tree/main/preprints/The-Factor-Two-Hardness-Threshold-for-Vertex-Cover-September-23-2026) claims NP-hardness of approximating minimum Vertex Cover within every fixed factor below 2, even on simple unweighted graphs. The accompanying Lean scope note specifies a formalized gap reduction from binary 3-SAT. [Scope note](https://github.com/openai/math/blob/main/lean/docs/102.md).

The same OpenAI release includes a direct proof of Max-Cut hardness beyond the Goemans–Williamson approximation constant. These are **hardness boundaries**, not new approximation algorithms. They are useful for setting honest targets: a heuristic should be compared against the best possible known guarantees and should not promise a polynomial-time approximation factor that conflicts with established hardness unless it exposes a new assumption or a restricted input class.

### Competitive online algorithms: k-server on arbitrary metrics

[OpenAI Math overview](https://github.com/openai/math/blob/main/overview.tex) lists a result establishing a randomized competitive ratio O(log^2(k+1)) for k-server on arbitrary metrics, matching the worst-case lower-bound order. This is relevant to online dispatch and dynamic infrastructure, where future requests are unknown. It is not a static TSP solver and the competitive guarantee is not a per-instance approximation ratio.

**Transfer:** dynamic Sol-Atlas dispatch and re-routing; benchmark adversarial and oblivious request sequences, movement cost, competitive ratio and online computation separately from offline optimal routing.

### Research lesson

The Luminous portfolio should distinguish at least five guarantee classes:

1. Exact optimization/counting with proof or independently checkable certificate.
2. Randomized exact algorithms with one-sided/two-sided error.
3. FPRAS / approximate counting with explicit relative-error and confidence guarantees.
4. Approximation algorithms with a proven ratio or gap.
5. Online algorithms with competitive guarantees.

A single “accuracy” metric cannot compare these classes. The type of guarantee must be part of each problem contract and every benchmark receipt.

## State of practice: competition and benchmark baselines

- [SAT Competition 2026](https://satcompetition.github.io/2026/): Main, Parallel, Experimental and Cloud tracks, with AI-generated/AI-tuned subtracks. Main-track proof-checker support and proof-checking rules are especially relevant. Use the published benchmark list, per-instance outcomes, solver sources and result logs.
- [SAT Competition 2026 results](https://github.com/satcompetition/2026/blob/main/results.html): compare SAT and UNSAT solved counts and scores separately; portfolio/VBS results are an upper-bound reference, not a single executable solver.
- [MaxSAT Evaluation 2026](https://maxsat-evaluations.github.io/2026/): exact, anytime and incremental tracks, with unweighted and weighted variants. Use its benchmark collection, metadata, solver sources and logs.
- [CO-Bench / LACE benchmark implementation](https://github.com/PJ-NTU/LACE): a broad set of combinatorial optimization tasks, useful for transfer and cross-domain evaluation.
- [TSPLIB](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/): established TSP benchmark family.

For practical SAT, integrate mature baselines such as **Kissat** and **CaDiCaL** through a stable subprocess/FFI boundary. Do not initially attempt to replace mature CDCL engines with a new HDC solver. Keep the in-house DPLL solver as a small reference until it passes semantic, exhaustive and proof-validation gates.

## Repository audit: fix the foundation first

Relevant existing code:

- [logic_engine.rs](../../crates/core/symthaea-core/src/hdc/logic_engine.rs): propositional formula representation, NNF/CNF conversion and DPLL.
- [constraint_solver.rs](../../crates/core/symthaea-core/src/hdc/constraint_solver.rs): CSP types, AC-3 and backtracking/forward checking.
- [z3_bridge.rs](../../src/z3_bridge.rs): runtime Z3 subprocess with an internal DPLL fallback.
- [symthaea-lean-bridge](../../crates/bridges/symthaea-lean-bridge/): Lean source emission/checking bridge.

### Confirmed CNF conversion defect

LogicEngine::to_cnf currently calls to_nnf, then collect_cnf_clauses and collect_literals. The latter traverses Or, but does not distribute Or over And and does not introduce Tseitin variables. For example:

$$
(p AND q) OR r
$$

is visited as one disjunction; the And(p,q) subtree contributes no literals, so the resulting clause is just [r]. The equivalent CNF is (p OR r) AND (q OR r). This can reject a genuinely satisfiable formula.

There is a second concrete issue: False contributes no literal, and collect_cnf_clauses drops empty clauses. Thus False can become an empty clause list, which the DPLL base case interprets as SAT.

The module documentation says “CNF conversion (Tseitin)”; the inspected implementation does not implement Tseitin encoding. Correct the contract and implementation before claiming general propositional SAT support.

### Required regression tests

At minimum, test:

1. False is UNSAT; True is SAT.
2. (p AND q) OR r is SAT under p=true, q=true, r=false.
3. p OR (q AND r) is SAT under p=false, q=true, r=true.
4. Contradictory unit clauses are UNSAT.
5. Tautological clauses and duplicate literals normalize without changing semantics.
6. Exhaustively enumerate all assignments for generated formulas with up to a small fixed number of variables and compare the CNF encoding and solver result against direct evaluation.
7. For every returned SAT witness, evaluate the **original** formula/CNF independently.
8. For every returned UNSAT answer in the proof-producing path, require an independently checked proof; ProofResult's narrative step list is not a DRAT/LRAT certificate.

Do not call a solver's Unknown, timeout, crash or missing proof UNSAT.

## Architecture proposal

Keep the low-level solver independent of HDC and cognitive-loop code. Add a narrow shared problem/solver interface first; only introduce a new crate after reviewing the workspace's existing math and logic boundaries.

### Typed problem contract

For SAT, use canonical integer literals (non-zero signed variable IDs), an explicit declared-variable count and clauses where an empty clause is preserved. Define parsing/validation and normalization separately from solving.

A result should be a typed status, not Option<Model>:

- Sat { model, validation }
- Unsat { proof_format, proof_digest, checker, checker_version }
- Unknown { reason }
- Timeout { limit, elapsed }
- Error { category }

For optimization, return a feasible incumbent, objective value, lower/upper bound as applicable, gap, and termination reason. Do not label a heuristic solution optimal without a certificate or exhaustive proof.

### Shared components

1. problem-contract: canonical typed inputs, constraints, objective semantics and stable serialization.
2. solver-adapter: subprocess/FFI adapter for external solvers, with pinned versions and explicit capabilities.
3. reference-solvers: small transparent exact solvers for tiny instances and regression testing.
4. instance-features: graph/formula features, sparsity, degree distributions, clause-length distribution, component structure, treewidth proxies and detected symmetries.
5. portfolio-selector: predicts which solver/heuristic to try; records selection rationale and pilot-run evidence.
6. algorithm-evolution: generates/mutates small isolated heuristic functions, compiles in a sandbox and rejects invalid candidates.
7. certificate-check: independent model validation, DRAT/LRAT proof checking, optimization-bound checking and proof artifact provenance.
8. benchmark-runner: deterministic instance splits, seeds, resource limits, machine metadata and per-instance logs.
9. evidence-receipts: content-addressed records for inputs, binaries, source revisions, configuration, stdout/stderr, results, proof artifacts and checker verdicts.

Use Lean for theorem statements and verified transformations where appropriate. Lean-checking a theorem statement is not, by itself, evidence that an extracted production binary implements the theorem's algorithm.

## Problem portfolio: beyond 3-SAT

Prioritize problems that give us theory, cross-domain leverage and operational utility.

| Priority | Problem family | Why it matters | Key baselines / evidence |
|---|---|---|---|
| P0 | SAT / 2-SAT / 3-SAT / k-SAT | Core Boolean reasoning; complexity and verification | Kissat, CaDiCaL; SAT Competition |
| P0 | #SAT / weighted model counting | Counts explanations/configurations; probabilistic inference | Exact counting baselines; IJCAI complementary branching; OpenAI preprint tracked separately |
| P0 | MaxSAT / weighted MaxSAT / pseudo-Boolean | Soft constraints, preferences and resource trade-offs | MaxSAT Evaluation 2026 |
| P1 | SMT (QF_BV, QF_LIA, arrays) | Hardware/software verification, engineering constraints and symbolic execution | Z3 and other mature SMT solvers |
| P1 | General CSP / graph coloring / Latin square / N-Queens | Scheduling, allocation, configuration and propagation research | AC-3, MAC, min-conflicts, CP-SAT |
| P1 | Vertex Cover / Hitting Set / Independent Set / Clique | Strong graph structure and parameterized algorithms | Exact, kernelized and treewidth-based methods |
| P1 | Set Cover / Set Packing / Exact Cover | Resource selection, coverage, supply chains and scientific design | Greedy approximation, ILP/MaxSAT and exact baselines |
| P1 | Knapsack / Subset Sum / partition | Pseudopolynomial algorithms, approximation and parameterization | DP, meet-in-the-middle, branch-and-bound |
| P1 | TSP / VRP / Steiner Tree / Facility Location | Transport, ports, logistics, energy and infrastructure | Concorde, LKH-3, OR-Tools, exact MILP/CP-SAT |
| P1 | Job-shop / flow-shop / RCPSP scheduling | Manufacturing, maintenance, workforce and project planning | CP-SAT, MILP and public scheduling benchmarks |
| P2 | Matching / max-flow / min-cost flow / shortest path | Polynomial-time controls and subroutines | Blossom, network-flow and shortest-path references |
| P2 | XOR-SAT / linear systems over GF(2) | Algebraic control group; cryptography and parity reasoning | Gaussian elimination and hybrid SAT-XOR |
| P2 | Model checking / bounded model checking | Safety properties for autonomous systems and control | SMT/SAT encodings plus independently checked counterexamples |
| P2 | Bayesian inference / probabilistic graphical models | Connect exact counting and constraints to epistemic uncertainty | Weighted #SAT and established inference engines |

Polynomial-time problems are deliberate controls. They help identify when the research system is needlessly applying exponential machinery and allow us to compare structure discovery against well-understood algorithms.

## Paradigm shifts to test

### A. From one solver to a solver portfolio

Use a family of experts: CDCL, local search, exact branching, algebraic/counting, dynamic programming, decomposition and domain-specific heuristics. Select them based on instance features and short pilot runs. Preserve quality-diversity; a specialist that wins on a rare family should not be discarded merely because it loses on the average.

### B. From raw search to structure-first compilation

Before search, detect and exploit connected components, forced assignments, equivalent variables, symmetries, low-treewidth regions, small separators, backdoor sets, bounded degree and decomposable subproblems. Record which reduction was applied and how its semantics are preserved.

### C. From model output to proof-carrying output

For SAT, validate the model against the original input. For UNSAT, require a DRAT/LRAT or other supported proof checked by an independent checker. For optimization, provide a feasible incumbent and a valid bound. Use a verifier that is simpler and more trusted than the generator.

### D. From monolithic algorithms to evolvable components

Follow AutoModSAT's modularization and LACE's problem-contract/portfolio design. Let the search process alter one isolated heuristic at a time (branching, phase selection, restart, clause reduction, propagation, neighborhood move, crossover, destroy/repair). Require compilation, property tests, proof checking and held-out evaluation before promotion.

### E. From average score to distributional competence

Keep separate hidden test families and evaluate distribution shift. The system should report where it wins, where it loses, and when it abstains or hands off. Never use benchmark instances used during evolution as the only evidence of generalization.

### F. From single objective to anytime trade-offs

For operations, combine feasibility, cost, time, energy, risk, fairness and resilience. Return Pareto fronts or explicitly defined scalarizations. Do not hide infeasibility or trade-off choices inside an opaque score.

## Benchmark protocol and target metrics

### SAT / 3-SAT

Create a stratified corpus:

- Uniform random 3-SAT at several clause densities, including below, near and above the phase transition; use fixed seeds and report the exact generator.
- Planted-solution and planted-unsatisfiable distributions.
- Crafted hard instances.
- Real application families from SAT Competition (hardware, software, planning, cryptography, graph and scheduling encodings).
- Sparse/bounded-degree CNFs and formulas with controlled variable/clause graph structure.
- Held-out whole families, not merely random held-out instances from the same source.

Report SAT and UNSAT separately: solved count, PAR-2 or competition score, median and tail runtime, memory, proof-generation time, proof-check time, timeouts, crashes and correctness failures. Keep random seeds and machine conditions fixed. Include small brute-force truth-table agreement tests as a mandatory gate.

### #SAT and MaxSAT

For #SAT, compare exact counts against brute force on small instances and established exact counters on larger ones; report time/space separately from decision SAT. For MaxSAT, use the 2026 exact, anytime and incremental tracks, report objective quality at fixed time budgets, optimality certificates, and the quality of lower/upper bounds.

### Combinatorial optimization

Use TSPLIB for TSP, CO-Bench for broad CO coverage, and public scheduling/VRP/graph benchmarks. Report objective gap to known optimum/best-known, feasibility rate, time-to-target, anytime curve, memory and energy. Compare against specialist solvers (e.g. Concorde for exact symmetric TSP; LKH-3 for strong TSP/VRP heuristics; OR-Tools/CP-SAT for routing and scheduling), not only generic LLM-generated baselines.

### Research gates

- **Correctness gate:** no silent wrong model, wrong status or invalid objective.
- **Reproducibility gate:** exact source revision, input digest, configuration, seed, environment and checker result.
- **Generalization gate:** held-out families and adversarial counterexamples.
- **Performance gate:** statistically meaningful improvement over pinned baselines at equivalent resources.
- **Theory gate:** proof of worst-case bound, parameter assumptions and reduction direction; experiments alone do not establish asymptotic complexity.
- **Promotion gate:** queued != pass; a candidate remains experimental until all required independent evidence is present.

## Work packages

### WP0 — semantic repair and reference tests (first)

- Add tests demonstrating the False and OR-over-AND defects.
- Replace the misleading Tseitin description with the actual contract.
- Implement a typed CNF representation and either a correct Tseitin transformation or a bounded, fully correct distributive conversion for tiny formulas. Prefer Tseitin for general formula compilation.
- Exhaustively compare conversion/solver outcomes to direct evaluation on small generated formulas.
- Independently validate every SAT model against the original formula.
- Keep proof status distinct from the current narrative ProofResult.

### WP1 — benchmark and solver adapters

- Add strict DIMACS CNF parsing and canonical serialization.
- Build adapters for pinned Kissat and CaDiCaL binaries; record solver and checker versions.
- Add a small brute-force oracle for tiny instances.
- Wire model validation and an external UNSAT proof checker.
- Add reproducible per-instance JSON/CSV receipts.

### WP2 — SAT heuristic discovery

- Make a modular experimental CDCL/heuristic surface; do not rewrite Kissat/CaDiCaL first.
- Expose isolated branching, phase, restart, clause-retention and presearch policies.
- Reproduce AutoModSAT's reported method and ablations on a held-out corpus.
- Compare LACE-style complementary portfolios against single experts and a virtual-best-solver oracle.
- Keep proof checking enabled for every promoted UNSAT result.

### WP3 — 3-SAT and #SAT theory reproduction

- Reproduce the published PPSZ bound as a theoretical analysis, not an implementation claim.
- Implement and test complementary branching for #SAT on small and sparse formula families.
- Independently audit the OpenAI ETH preprint's theorem-to-Lean correspondence, assumptions and extracted algorithm; record unresolved gaps and do not mark it validated prematurely.
- Reproduce the random 3-SAT threshold certificate procedure separately from instance solving.

### WP4 — transfer to other problems

- Lower graph coloring, exact cover, vertex cover, set cover and scheduling into the shared problem contract.
- Compare native CSP/CP-SAT/ILP paths with SAT/PB/SMT encodings.
- Extend to TSP/VRP/Steiner tree and port/logistics scenarios after the benchmark/evidence layer is stable.

## Project responsibilities

- **Symthaea:** discover representations, features, conjectures, strategies and heuristic candidates; use HDC as a learned guidance/selection representation, never as the correctness oracle.
- **Mycelix:** attest source/data/solver provenance and make experiment receipts independently reproducible.
- **Symtropy:** run optimization portfolios for industrial scheduling, routing, operations and simulation planning.
- **Sol-Atlas:** apply graph and constraint optimization to transport, port, energy and infrastructure networks.
- **Luminous platform:** host common contracts, solver registry, benchmark registry, evaluation protocol and evidence lineage.

## Explicit non-goals and claim discipline

- Do not claim P = NP or P != NP from heuristic results, large solved instances, subexponential algorithms or random-distribution results.
- Do not call a timeout or unknown result UNSAT.
- Do not call a solver's output optimal without a certificate or proof.
- Do not compare a #SAT counting bound with a decision-SAT bound as if they were the same task.
- Do not present a conditional sparse-graph result as an unconditional general-graph result.
- Do not promote a learned heuristic because of training-set performance alone.
- Do not treat Lean formalization, paper correctness, extracted algorithm correctness and practical binary correctness as one undifferentiated status.

## First success criterion

The first successful increment is not a claim to beat Kissat. It is a small, reproducible SAT research foundation that (1) cannot silently miscompile the formulas it accepts, (2) agrees with exhaustive truth tables on small cases, (3) validates SAT models and UNSAT proofs independently, and (4) provides a fair harness in which future algorithm discoveries can be measured against current specialist solvers.
