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

### 3. Vorobyev: “An upper bound of 4.268 for the random 3-SAT satisfiability threshold” (2026-10-06)

[arXiv:2610.10599](https://arxiv.org/abs/2610.10599)

This paper proves that a uniform random 3-SAT formula with `floor(4.268 n)` clauses is UNSAT with high probability. It brings the rigorous upper estimate close to the cavity-method prediction around 4.267 and formalizes the finite calculation in Lean 4. This is an **upper bound on the random threshold**, not the exact threshold itself and not an algorithm for arbitrary worst-case 3-SAT.

**Use for Symthaea:** create an exact-rational, proof-backed threshold certificate benchmark. Compare lower certificates, upper certificates, convergence of bounds, and machine-checkable arithmetic. Keep distributional guarantees separate from per-instance decision.

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

### 7. From static heuristic evolution to solver–instance co-evolution

Three 2026 results suggest that Symthaea should evolve more than a solver's local heuristic functions:

- **ASRO — Algorithm Space Response Oracles (ICML 2026, PMLR 306):** co-evolves solver-program and instance-generator strategy pools using best-response oracles. Its motivation is that static instance distributions can overfit and generalize poorly under distribution shift. [Paper](https://proceedings.mlr.press/v306/ke26b.html).
- **HeurEvo (September 2026 preprint):** jointly evolves high-level algorithm plans, their implementations and reusable components, while adapting runtime allocation. [Preprint](https://arxiv.org/abs/2609.36303).
- **CP-SynC-XL (May 2026 preprint):** 100 combinatorial problems and 4,577 instances; highlights that representation/formulation choice changes correctness. Its reported search-heuristic speedups are small and bimodal (median 1.03–1.12×), and some optimizations reduce correctness. This is a caution against promoting LLM-generated search changes without separate verification. [Preprint](https://arxiv.org/abs/2605.12421).
- **Mathematical programming model mining survey (May 2026):** reports the field lacks standardized fitness/precision/generalization measures for model conformance, and that many approaches remain at low technology readiness. This motivates a separate semantic-fidelity benchmark before solver optimization. [Survey](https://doi.org/10.1016/j.cosrev.2026.100905).

**Proposed Symthaea extension:** jointly search over (i) proof-carrying representation/reduction paths, (ii) complementary solver plans and reusable components, and (iii) adversarial instance generators. Keep the problem contract, evaluator, hidden labels and certificate checker frozen outside both evolving populations. Use a restricted empirical game / double-oracle loop, not an unsupported claim of Nash equilibrium or minimax optimality. Every generated instance must satisfy the original problem contract and pass independent validation. This is tracked in [#10848 — co-evolve proof-carrying representations, solver plans and adversarial instances](https://github.com/Luminous-Dynamics/symthaea/issues/10848).

This adds an adversarial discovery curriculum to LACE-style portfolio search and HeurEvo-style plan/component co-evolution. It composes with #10811, #10835, #10807 and #10836; it must not replace the frozen benchmark runner or leak sealed confirmation instances.

A prerequisite is to verify the mathematical formulation itself before evolving a solver for it. [#10850 — benchmark semantic fidelity of discovered mathematical formulations](https://github.com/Luminous-Dynamics/symthaea/issues/10850) defines exact small-instance conformance checks and separate formulation-correctness metrics for allocation, scheduling and transport models. An optimizer returning a feasible point for an incomplete or incorrectly weighted model is not a successful discovery.

## Counting-to-sampling and Max-k-SAT consequences in the ETH manuscript

The October 25 manuscript also derives two useful consequences of exact #k-SAT counting. Treat both as claims within that preprint, subject to the same proof-scope and review caveats as the main theorem.

- **Witness extraction:** testing whether the exact count is positive decides SAT; standard self-reduction fixes variables one at a time, querying the remaining conditional counts to construct a satisfying assignment.
- **Uniform feasible-solution sampling:** choosing each variable according to the two conditional model counts yields an exactly uniform random satisfying assignment, with polynomial expected overhead as stated in the manuscript. This is not ordinary heuristic sampling and depends on access to the exact counter. It may become a valuable research primitive for sampling feasible schedules/configurations or measuring solution-space diversity, if the asymptotic algorithm ever becomes practical.
- **Additive Max-k-SAT approximation:** the manuscript derives a randomized algorithm that, for fixed k and epsilon > 0, returns an assignment satisfying at least OPT(F) - epsilon*m clauses with probability 1 - 2^(-Omega(n)), in subexponential time and polynomial space. This is an **additive error in clause count**, not a multiplicative approximation ratio, and it remains a superpolynomial-time theoretical guarantee.

For Symthaea, the interesting experiment is to compare (a) exact-count-guided uniform sampling, (b) approximate model counters and (c) learned solution samplers on small instances where exact distributions are enumerable. Measure distributional calibration, mode coverage, sample diversity and validity separately. For planning systems, feasible-solution sampling should be used to characterize flexibility and fragility—not to silently choose a legally or morally binding allocation.

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

## Measured industry/research baseline snapshot (SAT Competition 2026)

### Authoritative per-instance source

Use the organizers' [current instance-wise `scores.csv`](https://github.com/satcompetition/2026/blob/main/downloads/scores.csv), not the repository's stale `results.html` summary or the earlier July slide snapshot.

Pinned source identity observed 2026-11-07:
- Repository head commit: `3e2dee7cf5aa643c77bf946280a3c1c520756ef9` (2026-08-25).
- CSV Git blob SHA: `9c926b2bd35ac0fb4f3d0cf8f58c97178d43faef`.
- File size: 1,262,695 bytes.
- Header: `solverid,instanceid,runtime,status,score,vresult`.
- 13,200 records: 33 solver IDs × 400 instances. The relevant Main Track groups contain 19 `[main]` solvers and 12 `[main-ai]` solvers; `[exp-ai]` entries are kept outside these Main Track comparisons.
- Independent input-integrity audit: 13,200 unique solver/instance pairs, all 33 solvers have the identical 400-instance universe, all score strings have exactly two decimal places, and zero verified-status/result-label mismatches were found. The exact status histogram and recomputed baseline answers are frozen in [`SATCOMP26_RESULTS_AUDIT_EXPECTED_V1.json`](SATCOMP26_RESULTS_AUDIT_EXPECTED_V1.json) for the Rust auditor's regression tests.
- The organizers' index records an August 10, 2026 update adding two solvers omitted by manual intake and publishing updated scores, slides and sources: [official competition index](https://github.com/satcompetition/2026/).

### Updated Main Track results

Recomputed from the pinned CSV; PAR-2 is the sum of the 400 per-instance score fields divided by 400.

| Main Track solver | Solved | SAT verified | UNSAT verified | PAR-2 |
|---|---:|---:|---:|---:|
| `anders_satsuma-iter-kissat[main]` | 276 | 127 | 149 | 3647.019675 |
| `anders_satsuma-iter-ae-kissat-mab[main]` | 269 | 136 | 133 | 3696.843000 |
| `zheng_kissat-mab-hypre[main]` | 255 | 134 | 121 | 3996.293425 |
| `zheng_kissat-mab-hypre-v2[main]` | 255 | 134 | 121 | 3997.276200 |
| `oertel_satsuma-lex-kissat[main]` | 249 | 125 | 124 | 4279.583500 |

These counts use accepted statuses `sat-verified` and `unsat-verified`. Raw `unsat`, `unknown`, timeout, crash and checker-failure statuses are not accepted as verified solutions.

The AI SAT subtrack's Lymphosat result remains a separate metric: **146 SAT instances**. The full `green_lymphosat[main-ai]` row in the instance-wise file has 251 verified results (146 SAT and 105 UNSAT), while `green_lymphosat-verified[main-ai]` is the SAT-only 146-result row used for the AI SAT subtrack comparison. Do not conflate the SAT-only recognition result with total Main Track solves.

The competition's **award podiums are separate from the raw ordering of every submitted solver configuration**. The 2026 presentation's combined Main Track podium is 276 / 255 / 240 solved (satsuma-iter+kissat, Kissat-MAB-HyPre, EDA-Kissat-v2). The instance-wise CSV also has the same-team `anders_satsuma-iter-ae-kissat-mab[main]` row at 269; preserve it in raw solver comparisons even though the official award slide lists the next eligible podium entry. Do not discard a high-performing configuration from raw baseline analysis merely because prize eligibility yields a different podium.

### Separate SAT and UNSAT track results

The SAT-only ranking must not be inferred from total Main Track solves. Recomputed from the 191 SAT-labeled rows per solver in the pinned CSV (SAT PAR-2 uses the sum of the 191 SAT-instance score fields divided by 191), the leading rows are:

| Main Track SAT solver | SAT solved | SAT PAR-2 |
|---|---:|---:|
| `anders_satsuma-iter-ae-kissat-mab[main]` | 136 | 3430.953508 |
| `zheng_kissat-mab-hypre[main]` | 134 | 3445.917749 |
| `guo_kissat-ae-hucb[main]` | 133 | 3507.773298 |
| `guo_kissat-ae-eg[main]` | 133 | 3521.796230 |

The AI SAT subtrack includes Lymphosat at 146 / PAR-2 2813.498010. The official SAT award table lists satsuma-iter+ae-kissat-mab (136), Kissat-MAB-HyPre (134), and AE-kissat-HUCB (133); its AI row is listed separately.

For the UNSAT-only table, the leading Main rows are satsuma-iter+kissat at 149 / PAR-2 1743.835904, Kissat-MAB-HyPre-V2 at 121 / PAR-2 3073.614337, and EDA-Kissat-v3 at 111 / PAR-2 3843.298434. This is a different optimization surface from SAT solving and should have separate policy and specialist analysis.

The source PDF's overall and subtrack tables are at slides 13, 16 and 18; the official index records the August 10 correction that included two initially omitted solvers and changed the Main SAT podium. Link: [updated SAT Competition 2026 presentation](https://satcompetition.github.io/2026/downloads/satcomp26slides.pdf) · [official index and correction notice](https://satcompetition.github.io/2026/).

### Recomputed virtual-best diagnostics

The CSV does not provide a trusted explicit VBS row, so the following are **our derived per-instance oracles**, not official competition leaderboard rows.

Method:
1. Partition solvers by the `[main]` or `[main-ai]` suffix.
2. For each of the 400 instance IDs, admit only `sat-verified` and `unsat-verified` outcomes.
3. Select the lowest `score` among admitted outcomes.
4. If no solver has an admitted result, assign the 10,000 PAR-2 timeout penalty (2 × 5,000 seconds).
5. Sum all 400 selected scores and divide by 400. Count SAT and UNSAT from the selected verified outcomes separately.

| Derived oracle universe | Solvers | Solved | SAT | UNSAT | PAR-2 |
|---|---:|---:|---:|---:|---:|
| Main-only VBS | 19 | 324 / 400 | 161 | 163 | 2261.119825 |
| Main + Main-AI VBS | 31 | 344 / 400 | 181 | 163 | 1723.650175 |

The Main-only VBS gap to the best single `[main]` solver is **48 instances** (324 − 276). The combined Main + Main-AI diagnostic closes 68 instances beyond that best single solver, but mixes the AI subtrack and must not be called the conventional Main-only VBS. Neither oracle is a deployable solver: each selects with hindsight after seeing all solver outcomes.

### Stale-summary warning

The currently checked-in [`results.html`](https://github.com/satcompetition/2026/blob/main/results.html) still shows `VBS=347` and `kissat-sc2024=306`. Those are the older SAT Competition 2025/2024-era rows, not the pinned 2026 instance-wise source. The official index says the updated per-instance scores were published on August 10, 2026. For Symthaea research, pin and aggregate `scores.csv`; do not use the stale HTML rows.

### Competitive targets

Use the current 2026 single-solver baseline and the derived Main-only VBS diagnostic:

| Stage | Target | Meaning |
|---|---|---|
| T0 | 100% small-instance semantic agreement; zero invalid accepted models/proofs | Correctness prerequisite |
| T1 | Reproduce pinned 2026 baselines within 1% PAR-2 with identical per-instance outcomes | Baseline integrity |
| T2 | On sealed held-out families, beat the best component by ≥3 solved instances or ≥5% PAR-2, selector overhead ≤5% | First deployable portfolio utility |
| T3 | ≥280 Main Track instances | +4 over 276 winner |
| T4 | ≥290 Main Track instances | +14; close about 29% of the 48-instance Main-only oracle gap |
| T4-stretch | ≥300 Main Track instances | +24; close half of the Main-only oracle gap |
| AI-SAT | ≥150, then ≥160 SAT instances on the same AI SAT subtrack contract | Separate SAT-only objective; do not add UNSAT subtrack counts |
| T5 | A new worst-case/FPT/approximation theorem with complete proof | A theoretical breakthrough, not a runtime claim |

These remain prospective targets. Report PAR-2, solved count, SAT/UNSAT split, per-instance receipts and family-held-out performance. Keep oracle metrics strictly diagnostic.

## Model-counting specialist baselines and task separation

The 2021–2023 model-counting competition review, published in 2026, identifies a useful group of established reference systems:

- **SharpSAT-TD:** top-down compilation with tree-decomposition heuristics.
- **Ganak:** SharpSAT-derived with probabilistic component caching.
- **D4:** Decision-DNNF knowledge compilation with dynamic decomposition.
- **GPMC:** compilation-based and includes projected-counting optimizations.
- **ExactMC:** CCDD compilation, for non-projected counting in the cited comparison.
- **ApproxMC:** hashing-based approximate model counting.

The review reports different leaders across exact/projected/approximate tracks and a sizeable virtual-best-solver gap; this is evidence for complementary methods, not one universal #SAT ranking. See [Model Counting in the Wild](https://proceedings.kr.org/2024/71/kr2024-0071-shaw-et-al.pdf) and the [2021–2023 competition review](https://doi.org/10.1016/j.artint.2026.104605).

For our benchmark contracts, distinguish at least:
1. Exact non-projected #SAT.
2. Exact projected #SAT.
3. Approximate model counting with explicit confidence/relative-error guarantees.
4. Weighted model counting.
5. Knowledge compilation, where the compiled representation supports repeated downstream queries.

Do not compare projected counts with full model counts, or approximate counters with exact counters, using a single solved-count metric. Compare counts, proof/precision guarantees, preprocessing, compile time, query time, memory and reuse across repeated queries.

## Actual Model Counting Competition 2026 baseline: Ganak

The current MC Competition 2026 comparison source includes a detailed, reproducible Ganak entry with exact/approximate result tables across the official tracks. Use it as a pinned specialist baseline, while respecting the metric/solver caveats below.

| Track | Contract | Ganak reported configuration | Solved / denominator |
|---|---|---|---:|
| 1 | Unweighted #SAT | Exact MPZ integer arithmetic | 133 / 200 |
| 1 | Unweighted #SAT | PAC approximate counting (MPZ output) | 143 / 200 |
| 2B | Weighted #SAT, including negative weights | Exact MPQ rationals | 120 / 200 |
| 2B | Weighted #SAT | MPFR high-precision floating point | 121 / 200 |
| 3 | Projected model counting | Exact MPZ integer arithmetic | 125 / 200 |
| 3 | Projected model counting | PAC approximate counting (MPZ output) | 167 / 200 |
| 4 | Mixed projected weighted counting (WMC/PMC/PWMC) | Exact MPQ rational arithmetic | 135 / 200 |
| 5B | Complex-weight algebraic model counting | Exact MPQ configuration | 49 / 100 public instances |
| 5B | Complex-weight algebraic model counting | MPFR high-precision floating point | 53 / 100 public instances |

Sources: [MC Competition 2026 track definitions and rules](https://mccompetition.org/2026/mc_description) and [Ganak's 2026 entry, run scripts and per-track results](https://github.com/meelgroup/mccomp-ganak-appmc-2026). The competition budget is 3,600 seconds and 32 GB; exact ranking requires arbitrary precision and zero relative error, while the PAC track requires its stated multiplicative error/confidence guarantee.

**Critical metric caveats from the authors:**

- The entry states that Ganak ran in probabilistic mode for all displayed configurations. "Exact MPZ/MPQ" here describes arithmetic/output precision, not a claim that the whole solver execution is a deterministic, formally certified computation. Record algorithmic randomization and its error guarantee separately from arithmetic exactness and independent proof/certificate status.
- For Tracks 1 and 3, "approx" is PAC approximate counting with a multiplicative `epsilon` and success probability `1-delta`; it is not comparable to exact MPZ as if it had the same guarantee.
- For weighted Track 2B and complex Track 5B, "approx" is MPFR high-precision floating point, **not PAC**. The authors warn that absolute error near a true zero can yield a nonzero result. It must not be treated as a valid relative-error approximation or an exact zero-preserving method.
- Track 4's entry explicitly reports a header/semantics problem: some instances labeled PWMC were actually PMC or WMC, and Ganak selected PAC or MPFR methods based on the header. Only the exact MPQ total is reported; this is a data-contract warning, not clean evidence of homogeneous PWMC coverage.
- Track 5B's published table lists 100 public instances and no private denominator; do not describe 49/100 as a full 200-instance total.

### Symthaea's model-counting targets

First reproduce the entry on the same public/private files, hardware limits and per-instance manifests. Then compare each contract independently:
- Track 1: exact MPZ baseline 133/200 and PAC 143/200; maintain separate exact and PAC scorecards.
- Track 2B: exact MPQ baseline 120/200. Add negative-weight and exact-zero cancellation controls; MPFR's 121/200 is a diagnostic only, not an exact/PAC target.
- Track 3: exact MPZ baseline 125/200 and PAC 167/200; enforce projection-variable sets and existential hidden-witness semantics.
- Track 4: do not use the reported 135/200 as a clean PWMC target until each instance's actual PMC/WMC/PWMC header and semantics are independently audited. Re-run a homogeneous PWMC subset with a typed parser.
- Track 5B: exact Q(i) output / zero-cancellation correctness is the priority; current Ganak published result is only 49/100 public instances. Keep MPFR 53/100 separate.
- For all tracks, record solver randomization, arithmetic domain, error/confidence guarantee, parsing/contract validation, preprocessing, exact output comparison, runtime, peak RSS, and independent checker evidence separately.

A promising research advantage is not simply beating Ganak's totals. It is combining algebra-aware representation/decomposition discovery with a fail-closed contract auditor and independent replay/certification, especially for projected support, negative rational weights and complex cancellation. Only claim a performance win after a same-instance, same-budget comparison with the full end-to-end path measured.

## Proof-complexity-aware model counting: optimize the derivation, not only the answer

A further paradigm shift is to make the *structure of an exact count derivation* part of the algorithm-discovery objective. A numeric #SAT answer alone hides whether the solver found reusable decomposition, whether its cache was effective, and whether its computation admits a compact independently checkable representation.

Two 2026 results give this direction a concrete theoretical foundation:

- **AAAI-26, Proof Systems That Tightly Characterise Model Counting Algorithms:** an adapted form of MICE is polynomially equivalent to annotated Decision-DNNF proof representations for a DPLL-style model-counting algorithm; the resulting proof systems tightly characterize the runtime of that solver class. The paper also formalizes semantic caching and shows that the corresponding proof system is characterized by Decision-DNNF representation size (ignoring annotation size), motivating semantic reuse as a research variable. [Paper](https://doi.org/10.1609/aaai.v40i17.38431) · [Open PDF](https://ojs.aaai.org/index.php/AAAI/article/download/38431/42393).
- **Journal of Automated Reasoning (July 20, 2026), The Relative Strength of #SAT Proof Systems:** CPOG polynomially simulates both MICE and kcps, while MICE and kcps are exponentially incomparable. This establishes a proof-size simulation hierarchy among those systems, not a claim that CPOG is the fastest practical checker or the smallest certificate for every instance. [Paper](https://doi.org/10.1007/s10817-026-09757-w) · [PubMed record](https://pubmed.ncbi.nlm.nih.gov/42483024/).

This should compose with, not duplicate, the typed tractable-circuit and proof-format adapter work in [ETH-026 #10301](https://github.com/Luminous-Dynamics/symthaea/issues/10301), scope-complete circuit IR in [ETH-027 #10305](https://github.com/Luminous-Dynamics/symthaea/issues/10305), and exact model-counting benchmark in [ETH-004 #10090](https://github.com/Luminous-Dynamics/symthaea/issues/10090).

### New research hypothesis

For exact #SAT, a solver's useful capability is not only how quickly it produces the correct integer. It is how effectively it discovers a compact derivation composed of:
- decision partitions, using the exact identity `#F = #F[x] + #F[¬x]`;
- variable-disjoint decompositions, using `#(F₁ ∧ … ∧ Fₖ) = ∏ᵢ #Fᵢ` only when the scopes are disjoint;
- sound component caching with a canonical residual-formula identity and explicit remaining-variable universe;
- preprocessing / rewriting steps whose count-preservation and auxiliary-variable multiplicities are separately established;
- a proof or circuit representation that an independent checker can validate.

Search should jointly optimize branching, decomposition discovery, cache keys and reuse, and proof-target selection. Semantic cache hits require an actual equivalence obligation; a hash match is an identity test, not proof that two residual formulas are equivalent. Every cache key must bind the residual formula semantics and the variables still free, so unconstrained-variable multiplicities are not lost.

### Metrics to add to #SAT experiments

Keep these as a multi-objective vector rather than collapsing them into one score:

| Dimension | Required measurement |
|---|---|
| Solver quality | Exact count, solved status, wall/CPU time, peak RSS |
| Search structure | Decision nodes, decomposition nodes, component count, cache hits/misses, repeated residual states |
| Certificate quality | Proof-system ID/version, proof bytes, DAG/circuit nodes and edges, annotation bytes |
| Assurance cost | Independent checker time, peak RSS, rejection of mutated/invalid proofs |
| End-to-end cost | Preprocessing + solving + proof generation + independent verification |
| Transfer | Held-out family/generator performance and proof-size/runtime distributions |

Run matched ablations: no cache, syntactic component cache, semantic cache, branch-only, decomposition-first, and proof-target-aware discovery. Compare CPOG, MICE and annotated Decision-DNNF only where their precise contracts match. Keep proof-system simulation theorems, measured certificate sizes, checker runtime and solver runtime as separate evidence.

For initial qualification, use exhaustive small #SAT instances to verify every partition/decomposition/cache identity, then compare the emitted proof/circuit with an independently implemented checker. The proof-size/runtime characterization applies to the paper's stated algorithm class; do not generalize it to every CDCL or modern model-counting implementation without a separate reduction argument.

## State of practice: competition and benchmark baselines

- [SAT Competition 2026](https://satcompetition.github.io/2026/): Main, Parallel, Experimental and Cloud tracks, including AI-generated/AI-tuned participation. The rules require a model for SAT instances in all tracks and UNSAT proof certificates in the Main Track; source code and a 1–2 page system description are required, including disclosure of AI-authored code and AI-tuned heuristics. Pure portfolios combining solvers developed by different groups are restricted outside the Cloud Track, so an internal portfolio benchmark is not automatically a valid competition submission. Use the published benchmark list, per-instance outcomes, solver sources and result logs.
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

## Paradigm shift: a proof-carrying representation/reduction graph

Algorithm discovery should search over **representations** as well as search heuristics. But every transition must say exactly what it preserves. This is especially important because the OpenAI ETH manuscript derives graph-counting consequences through controlled counting-preserving encodings, while arbitrary polynomial-time reductions can destroy subexponential parameter bounds.

Create a directed graph of problem representations. Each node is a typed problem contract; each edge is a transformation with an explicit guarantee:

| Reduction/transform class | Preserved property | Typical use |
|---|---|---|
| Decision-preserving many-one | SAT/UNSAT yes/no equivalence | NP-completeness and decision algorithms |
| Parsimonious | Exact number of solutions | #SAT, graph colouring/independent-set counting |
| Unique-extension | Each source solution has exactly one target extension | Tseitin encodings and count-preserving compilation |
| Known-factor counting | Count related by a proved, explicit multiplier | Specialized counting reductions |
| Approximation-preserving | Approximation quality transfers under stated parameters | FPRAS and approximation algorithms |
| Optimization-preserving | Optimum/objective maps by a proved transformation | MaxSAT, ILP, graph optimization |
| Parameter-preserving/FPT | Controlled parameter and size blow-up | Treewidth, backdoor and kernel algorithms |

Every transformation record should contain:
- source/target schemas and versions;
- source and target input digests;
- forward mapping and, where needed, reverse witness mapping;
- semantic guarantee class;
- proof/certificate for correctness;
- variable/vertex/clause/edge blow-up;
- parameter mapping (e.g. k -> k', treewidth w -> w');
- runtime and space overhead of the transformation;
- whether the transformation is deterministic or randomized and its failure/approximation guarantee.

The discovery engine can then ask a more useful question than “which solver is fastest?”:

**Which verified path through the representation graph gives the lowest end-to-end cost for this instance family, while preserving the exact property the task requires?**

This is how we avoid invalid transfers: a decision-preserving reduction is not automatically parsimonious; an equisatisfiable encoding is not automatically count-preserving; and a subexponential algorithm does not automatically remain subexponential after a reduction with a superlinear variable blow-up.

This representation graph should be a first-class Symthaea research capability and a shared transformation registry in the Luminous platform. It is also the right place to store the unique-extension guarantee of the new Tseitin encoder.

### Extension: proof-carrying AND/OR hyperedges

A one-source/one-target edge is not expressive enough for the transformations that matter most in #SAT and combinatorial optimization. Branching creates several subproblems whose solution sets must be combined; disconnected-component decomposition creates multiple subproblems whose counts multiply; projected counting must union visible support with Boolean OR; tensor slicing creates multiple slices with a typed aggregation law.

The registry should therefore evolve into a **proof-carrying AND/OR hypergraph**. A hyperedge binds the source problem, a canonical set of target subproblems, a typed aggregation operator, a partition/decomposition witness, reconstruction semantics, and composed time/space/parameter costs. Initial operators should be deliberately narrow: `BranchSum`, `IndependentProduct`, `ExistentialSupportUnion`, `OptimizationCombine`, and `TensorSliceReduce`.

The aggregation law is part of the theorem, not metadata. For example:
- `BranchSum` requires an exhaustive, disjoint partition;
- `IndependentProduct` requires disjoint variable scopes and explicit free-variable handling;
- `ExistentialSupportUnion` collapses duplicate hidden witnesses with Boolean OR, not arithmetic addition;
- optimization and tensor operators must name their exact output algebra and prove their combination rule.

This is tracked in [#10868 — proof-carrying AND/OR hyperedges for decomposition](https://github.com/Luminous-Dynamics/symthaea/issues/10868), composing with #10811/#10835, the structural witnesses in ETH-019 #10244, and the typed circuit/proof adapters in ETH-026 #10301 and ETH-027 #10305. It should reuse their schemas and proof artifacts, not introduce a parallel circuit IR or receipt system.

#### Canonical finite known-answer fixtures

Freeze these three fixtures before implementing generic hypergraph search:

1. **BranchSum:** (F=(x\lor y)\land(\lnot x\lor z)), declared universe ({x,y,z}). The (x=0) cofactor is (y) with (z) free (2 assignments); the (x=1) cofactor is (z) with (y) free (2 assignments). The exhaustive, disjoint partition gives (#F=2+2=4). A missing cofactor, overlapping branch condition or dropped free variable must be rejected.
2. **IndependentProduct:** let (C_1=(a\lor b)\land(\lnot a\lor b)), (C_2=(c\lor d)), and (F=C_1\land C_2). (C_1) has 2 assignments, (C_2) has 3, and their variable scopes are disjoint, so (#F=2\times3=6). If unused variable (e) is also in the declared source universe, the answer is 12. A shared variable or omitted free variable must invalidate an unsupported product claim.
3. **ExistentialSupportUnion:** let (F(v,h_1,h_2)=(v\land h_1)\lor(v\land h_2)), with visible (v) and hidden (h_1,h_2). There are 3 satisfying full assignments but only 1 supported visible assignment ((v=1)). Hidden-witness branches overlap on visible support, so arithmetic addition double-counts; Boolean OR must collapse support before counting.

These cases are small enough for exhaustive truth-table checking. They distinguish SAT branching, #SAT component products and projected counting semantics without making a performance claim.



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

## Quantified targets: how to beat SOTA without benchmark theatre

The pinned August-updated 2026 instance-wise `scores.csv` shows a best single Main Track solver at 276/400. A derived Main-only virtual-best oracle over 19 `[main]` solvers solves 324/400 (48 more); a separate Main+AI oracle over 31 solvers solves 344/400. Both are hindsight diagnostics, not deployable portfolios. The source commit, CSV blob SHA, per-instance admission method and exact PAR-2 calculations are recorded in the baseline section above.

Set staged targets:

| Stage | Target | Admission rule |
|---|---|---|
| T0: semantic trust | 100% agreement on exhaustive small formula/graph corpora; zero invalid SAT models | Any false SAT/UNSAT or invalid witness blocks promotion |
| T1: baseline reproduction | Reproduce pinned 2026 top solver and relevant specialist baselines within 1% PAR-2 on identical conditions and per-instance outcomes | Compare exact binaries, flags, benchmark manifests and resource limits |
| T2: selector utility | On sealed held-out families, beat the best deployable component by at least 3 additional solved instances **or** 5% PAR-2, with selector overhead <=5% | No materially catastrophic held-out family regression hidden by an aggregate |
| T3: first competition-grade SAT target | Reach >=280 solved on the exact 2026 Main Track corpus and resource setup | Aspirational +4 vs the 276 single-solver winner; valid SAT models and independently checked UNSAT proofs required |
| T4: strong portfolio target | Reach >=290 solved on that same corpus without hindsight oracle information | Aspirational +14 vs winner; close about 29% of the 48-instance Main-only VBS gap |
| T4-stretch | Reach >=300 solved on that same corpus | Aspirational +24; close half of the Main-only VBS gap |
| A1: AI SAT-subtrack target | Reach >=150 SAT instances on the same 2026 Main Track SAT AI subtrack corpus | Lymphosat's SAT-only result is 146; keep this separate from total Main solves and UNSAT results |
| A2: strong AI SAT-subtrack target | Reach >=160 SAT instances on that same SAT-only corpus | Aspirational; disclose AI-generated/tuned components and report family-/generator-held-out performance |
| T5: theory breakthrough | Publish an improved worst-case upper bound, FPT result or approximation guarantee with a complete proof and exact assumptions | Benchmarks are supporting evidence, never a substitute for proof |

For AutoModSAT reproduction, do not target its paper's reported 30% speedup as a universal expected outcome. First reproduce its 11-family setup, then use an entirely held-out family split. Report which families win and lose; its authors explicitly report a case where parameter-tuned Kissat beats AutoModSAT and note the human-in-the-loop baseline preparation.

For model counting, maintain separate target tables for exact non-projected, exact projected, approximate and weighted counting. The appropriate comparator set includes SharpSAT-TD, D4, ExactMC, Ganak/GPMC and ApproxMC, selected according to each task's semantics. A claim to beat one exact counter on a projected task is not meaningful unless both solve the same projected-counting contract.

For operational optimization, define the practical target by Pareto improvement or an agreed SLA:
- no feasibility regressions;
- a statistically meaningful objective-gap reduction at the same wall-clock budget, or the same quality at lower latency/energy;
- bounded replan time after graph/constraint changes;
- independently validated solution and bounds;
- explicit cost/time/energy/risk trade-offs rather than one hidden scalar score.

Use confidence intervals or paired instance-level tests where appropriate. Report failed/timeout instances individually, use PAR-2 only where its competition semantics apply, and never average away invalid outputs.

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

### Machine-readable hyperedge fixtures and reference DPLL work

The AND/OR hyperedge tranche now has a frozen machine-readable known-answer fixture:
[`REPRESENTATION_HYPEREDGE_KNOWN_ANSWERS_V1.json`](REPRESENTATION_HYPEREDGE_KNOWN_ANSWERS_V1.json).

The three fixture identities were independently enumerated in a small direct truth-table check:
- BranchSum: source count 4, cofactor counts 2 and 2.
- IndependentProduct: component counts 2 and 3, with one unconstrained declared variable, so 2 × 3 × 2 = 12.
- ExistentialSupportUnion: three full satisfying assignments project to one visible assignment.

The five negative controls cover incomplete/overlapping branches, overlapping product scopes, dropped free variables and arithmetic addition in place of projected support union. These are finite semantic fixtures only; they do not establish scalable decomposition or Rust implementation correctness.

A separate implementation workstream, [#10885 — trail-based DPLL with deterministic budgets](https://github.com/Luminous-Dynamics/symthaea/issues/10885), follows from source inspection of #10801. The current copy-based branch isolation is a useful correctness-first reference, but it clones accumulated assignments and proof-step vectors at each branch. The new work proposes indexed tri-state assignments, a reversible trail, decision-frame checkpoints, explicit backtrack events and deterministic decisions/propagations/literal-inspection/event budgets. It remains a reference DPLL, not a claimed competitor to Kissat. Conversion budgets are separate and ordered: NNF expansion (#10849), CNF distribution (#10849), Tseitin output/depth (#10809/#10804), truth-table oracle (#10870), then solver search (#10885). Any phase exhaustion yields UNKNOWN/resource-limited, never UNSAT.

The shared anytime envelope now also has a normative semantics note:
[`ANYTIME_RESULT_ENVELOPE_V1_SEMANTICS.md`](ANYTIME_RESULT_ENVELOPE_V1_SEMANTICS.md).
It defines exact objective comparison, lower/upper-bound direction, request/contract binding, monotone accepted incumbent sequences, disposition invariants, exhaustive completion and forbidden status upgrades. The JSON Schema remains structural validation; independent Rust cross-record/objective/state-transition verification is tracked in [#10887](https://github.com/Luminous-Dynamics/symthaea/issues/10887).

## Cross-domain verified anytime optimization

A solver should be allowed to return a verified feasible incumbent before it proves optimality, without losing that useful result when a deadline or worker failure occurs. But the result must distinguish **candidate feasibility**, **search completeness**, and **optimality**.

This is now a shared Symthaea contract in [#10884 — evidence-qualified anytime result envelope](https://github.com/Luminous-Dynamics/symthaea/issues/10884), with a versioned schema and known-answer fixtures:

- [`ANYTIME_RESULT_ENVELOPE_V1.schema.json`](ANYTIME_RESULT_ENVELOPE_V1.schema.json)
- [`ANYTIME_RESULT_ENVELOPE_V1.examples.json`](ANYTIME_RESULT_ENVELOPE_V1.examples.json)

The v1 envelope is intentionally scoped to optimization workloads (allocation, scheduling, routing, energy dispatch and proof-packet selection). SAT's decision statuses remain owned by #10810 and are adapted without status upgrades. The envelope binds every incumbent validator, bound checker, completion certificate, infeasibility certificate and optimality certificate to the same request and contract digests. A digest is an identity binding; the independent checker must still validate the witness/proof.

### Result dimensions

| Claim | Minimum evidence | What it does not imply |
|---|---|---|
| Feasible incumbent | Complete witness plus independent validation against the original request | Search completeness or optimality |
| Search complete | Exhaustive completion for the declared model, with checked completion/search-tree evidence | Source truth or real-world safety |
| Bound available | Independently checked lower/upper bound with exact objective and assumption binding | A feasible candidate |
| Proven optimal | Admitted incumbent plus an independently accepted bound/optimality certificate under the same objective and tie policy | Operational authorization |
| Proven infeasible | Exhaustive checked search or independently accepted infeasibility proof | Infeasibility of a different model revision |

The result dispositions are deliberately explicit: `exact_complete`, `feasible_unproven`, `proven_infeasible`, `bound_only`, `limit_without_candidate`, `unknown`, `input_rejected`, `unsupported`, and `error`. A timeout cannot become infeasibility. A partial witness cannot replace an admitted complete incumbent. Optimality cannot be inferred from a small heuristic gap.

### Benchmark protocol

Compare the full verified decision path, not only the core solver:
- time to first candidate;
- time to first independently verified candidate;
- incumbent quality at fixed budgets;
- time to first independently checked bound;
- time to proof of optimality / infeasibility;
- verified objective-gap trajectory only when incumbent and bound have the same model/objective binding;
- rate of runs with a valid incumbent before deadline;
- checker/proof overhead, peak memory, CPU and energy;
- tail latency, worst-family performance and failure disposition.

This matches how industry systems expose bounded optimization results. OR-Tools CP-SAT distinguishes `FEASIBLE` (solution found without proof of optimality) from `OPTIMAL`, `INFEASIBLE`, and `UNKNOWN`; Gurobi exposes incumbents and best objective bounds under time limits, with a relative MIP gap computed from those endpoints. Its documentation notes that relative gap is not universally monotonic when incumbent/bound signs differ. We therefore track objective endpoints and their certificates separately rather than treating one scalar gap as an always-improving confidence score. [OR-Tools status definitions](https://github.com/google/or-tools/blob/stable/ortools/sat/docs/troubleshooting.md) · [Gurobi gap definition](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html) · [Gurobi benchmarking guidance](https://support.gurobi.com/hc/en-us/articles/37638046123921-What-to-measure-during-benchmarking).

### Reuse existing domain work

This is a shared semantics/adaptation layer, not a new optimizer or receipt system:
- [Mycelix #9425](https://github.com/Luminous-Dynamics/symthaea/issues/9425): independently checkable optimality/lower-bound certificates for proof-packet selection.
- [Symtropy #2608](https://github.com/Luminous-Dynamics/symtropy/issues/2608): preserve verified clean incumbents and measure time-to-first-clean.
- [Sol-Atlas #417](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/417): audited energy-dispatch incumbents under bounded search.
- #10810: SAT decision/proof status.
- #10805: benchmark receipts and per-run resource accounting.

The first adapter work should map Sol-Atlas's `feasible_unproven`-style candidate semantics, Mycelix's witness/lower-bound proof, and Symtropy's clean incumbent into the common result model while retaining each project's authoritative validator and exact request bytes.

### Qualification boundaries

The JSON Schema checks structure, digest shape, admissible status combinations and mandatory evidence fields. It does not prove objective ordering, that a bound is mathematically valid, or that an incumbent sequence is monotone. A semantic verifier must additionally:
- check every certificate and witness against the exact request/contract digest;
- check lower/upper bound direction against minimization/maximization;
- compare incumbent values using the declared exact objective domain and units;
- enforce non-worsening incumbents for fixed-objective minimization/maximization sequences;
- validate tie handling and only prune on a strictly justified bound;
- reject stale request revisions and incomplete search-frontier claims.

The three sample envelopes and negative controls are fixtures, not production run evidence or performance results.

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
