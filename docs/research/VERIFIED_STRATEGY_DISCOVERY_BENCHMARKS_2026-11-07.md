# Verified Strategy Discovery — Benchmark Matrix and Targets

Date: 2026-11-07  
Status: research benchmark plan; proposed targets are not achieved results  
Owner workstreams: #10805, #10848, #10884, #10887, #10899

## 1. Evaluation thesis

The primary evaluation question is:

> How much independently accepted computational value does a strategy deliver under a fixed, reproducible resource budget, and how well does that result generalize beyond the data used to discover the strategy?

Do not reduce this to one aggregate "AI score". Use correctness and evidence as hard admission gates, then report a Pareto surface over solution quality, time, proof production, checking, memory, CPU and energy.

A candidate that returns an unverified model, an invalid bound, or an incorrect exact count has not delivered a qualified result. A result that is correct only on the discovery distribution has not demonstrated robust algorithm discovery.

## 2. SAT Competition 2026 — pinned external baseline

Authoritative source: [SAT Competition 2026 instance-wise scores](https://github.com/satcompetition/2026/blob/main/downloads/scores.csv)

Pinned audit in [#10805](https://github.com/Luminous-Dynamics/symthaea/issues/10805):

- organizer repository head at retrieval: `3e2dee7cf5aa643c77bf946280a3c1c520756ef9`;
- CSV blob SHA: `9c926b2bd35ac0fb4f3d0cf8f58c97178d43faef`;
- 1,262,695 bytes; 13,200 data rows; 33 configurations × 400 instances;
- Main and Main-AI must remain separate; exclude experimental-AI configurations from those universes;
- only `sat-verified` / `unsat-verified` outcomes count as solved.

Recomputed results recorded in #10805:

| Measurement | Solved | Meaning |
|---|---:|---|
| Best individual Main configuration, `anders_satsuma-iter-kissat[main]` | 276/400 | Deployable single configuration |
| Second Main configuration | 269/400 | Deployable single configuration |
| Main-only virtual-best | 324/400 | Per-instance hindsight oracle, not a solver |
| Main + Main-AI virtual-best | 344/400 | Broader hindsight oracle, not a solver |

The Main-only VBS gap is 48 instances over the best individual solver. Do not present this as an achievable selector gain. The oracle can choose the winning solver after seeing each instance's result.

### Symthaea targets

| Stage | Main Track target | Qualification requirement |
|---|---:|---|
| Reproduce | 276/400 | Exact pinned corpus, status rules, resource envelope and PAR-2 |
| Initial improvement | 280/400 | Better verified solved count; PAR-2 no worse than 3647.019675 |
| Strong | 290/400 | Improvement on both solved count and PAR-2, plus family/structural holdout |
| Stretch | 300/400 | At least 24 more verified instances than the best 2026 single solver; no correctness regressions |

Also report:
- SAT and UNSAT solved counts separately;
- PAR-2, not solved count alone;
- time to first candidate and time to first independently verified candidate;
- proof generation time/bytes and independent checker time;
- failures, timeouts, unknowns and checker failures separately;
- worst-family and lower-tail performance;
- selector/runtime overhead and energy where measured reliably.

The AI SAT subtrack target remains separate: 150 initial, 160 stretch. The 146 SAT result and the 251 total result in the Main-AI row describe different metrics and must not be interchanged.

## 3. Model Counting Competition 2026 — task semantics are distinct

Authoritative rules and track definitions:
- https://mccompetition.org/2026/mc_description
- https://mccompetition.org/2026/submission

The competition has separate tracks for unweighted model counting, weighted counting with negative weights, projected model counting, mixed projected/weighted input and complex algebraic model counting. The standard run limit is 3,600 seconds and 32 GB. Exact ranking requires zero relative error; any wrong exact solution removes the submission from the ranking. The PAC track uses a declared approximation guarantee (alpha = 0.8). These are different contracts, not interchangeable result types.

### Ganak 2026 reference results

Source: https://github.com/meelgroup/mccomp-ganak-appmc-2026

| Track | Exact reference | Other reference | Important caveat |
|---|---:|---:|---|
| Unweighted #SAT | 133/200 (MPZ) | 143/200 (PAC + MPZ) | PAC guarantee must be reported |
| Weighted #SAT | 120/200 (MPQ) | 121/200 (MPFR) | MPFR can report non-zero for a true zero |
| Projected counting | 125/200 (MPZ) | 167/200 (PAC + MPZ) | Projection is not arithmetic summation over witnesses |
| Mixed projected/weighted | 135/200 (MPQ) | — | Input headers and semantics were inconsistent; treat as qualified exact-only reference |
| Complex AMC | 49/100 (MPQ) | 53/100 (MPFR) | MPFR zero/cancellation errors are possible |

Ganak reports that the solver ran probabilistically in all these configurations. MPZ/MPQ arithmetic is exact arithmetic, but that alone does not turn the algorithm into a deterministic procedure. Preserve the distinction between arithmetic representation, algorithmic randomness and correctness guarantee.

### Symthaea targets

First reproduce the published results and verify exact track semantics. Then pursue at least five additional exact instances on one fully qualified track without losing correctness on any other track. A +10-instance improvement is a stretch target only after the baseline and private/family-held-out corpus are reproducible.

Do not claim an exact WMC/AMC win by using floating-point arithmetic that can change an exact zero into a non-zero value. Do not compare a PAC result with an exact result without reporting its theorem-backed epsilon/delta or alpha contract.

## 4. MaxSAT Evaluation 2026 — exact and anytime optimization

Authoritative results and artifacts:
- https://maxsat-evaluations.github.io/2026/rankings
- https://zenodo.org/records/21358014

### Exact tracks

| Track | Best individual solver | Solved | VBS | VBS meaning |
|---|---|---:|---:|---|
| Exact unweighted | UWrMaxSAT2026 | 424 | 477 | Hindsight per-instance oracle |
| Exact weighted | UWrMaxSAT2026 | 456 | 470 | Hindsight per-instance oracle |

The 2026 evaluation also includes general-purpose/encoding baselines such as CPMpy+HiGHS and CPMpy+OR-Tools. On the exact unweighted ranking they solved 215 and 213, respectively; on the exact weighted ranking they solved 216 and 264. These values compare those specific MaxSAT encodings and competition configurations. They do not prove that dedicated MaxSAT solvers dominate general-purpose optimization for every application.

The weighted anytime ranking provides a useful fixed-horizon comparison for Aperture-KBG:
- 60 seconds: 0.91386;
- 300 seconds: 0.94158;
- 1,707 seconds: 0.96311.

The virtual-best scores at those horizons are 0.94926, 0.98368 and 0.99797. These are anytime quality scores, not exact solved counts.

### Symthaea targets

- Reproduce exact unweighted and weighted baselines before claiming improvement.
- Then target +5 verified optimal instances on one track with no incorrect result; +10 is a stretch objective.
- On anytime tracks, compare the full per-instance quality trajectory at 60, 300 and 1,707 seconds, not only the final incumbent.
- Report time to first feasible incumbent, time to first independently checked incumbent, best verified objective over time, and time to a proof of optimality separately.
- Include the incremental track via IPAMIR. Test repeated related instances and assumptions, verifying that cached clauses, decompositions and certificates remain valid under the updated request.

A solver disqualified for incorrect output is not a competitive baseline, even when it has attractive speed or objective values.

## 5. Algorithm-discovery and combinatorial optimization benchmarks

Use [CO-Bench](https://ojs.aaai.org/index.php/AAAI/article/view/40596) as a broad algorithm-search benchmark: it includes 36 real-world combinatorial optimization problems.

The 2026 LACE paper reports average CO-Bench score 0.945, compared with 0.870 for its strongest prior LLM-based method and 0.571 for direct prompting. It reports 0.97–0.99 on four structurally new tasks. These are benchmark-specific results, not SAT/MaxSAT solved-count improvements and not evidence that an LLM strategy beats every exact industrial optimizer.

### Symthaea targets

1. Reproduce LACE and its baselines on the same CO-Bench version and scoring contract.
2. Add typed transformation/decomposition contracts and independent feasibility checking.
3. Compare hand-authored typed strategies, static portfolios, LACE-style complementary evolution, ASRO-style solver/generator co-evolution and the joint Symthaea approach.
4. Require improvement on family- and structure-held-out tasks, not just the public IID score.
5. Report performance against domain solvers. Depending on the application and licensing, include SCIP, HiGHS, OR-Tools, Gurobi or CPLEX with pinned versions, configurations and equal hardware/resource envelopes. Do not imply a commercial solver comparison if the corresponding product was not actually run.

## 6. The cross-domain Symthaea scorecard

Use hard gates first, then a Pareto scorecard.

### Hard admission gates

- Source problem contract and input identity are immutable.
- The output is independently checked against the source task.
- Any reduction/decomposition has its required qualified guarantee.
- Exact arithmetic and approximation guarantees are not confused.
- No timeout/crash/resource-limit outcome is upgraded to UNSAT, infeasible, complete or optimal.
- A cached item is reused only when its full semantic key matches the current request.
- The confirmatory corpus is sealed from strategy and generator search.

### Reported metrics

| Dimension | Required measurement |
|---|---|
| Correctness | Independently admitted SAT models, UNSAT proofs, counts, witnesses and feasible incumbents |
| Completeness | Fully resolved search obligations and independently accepted completion |
| Optimality | Exact objective, valid bounds, proof of optimality, certified gap |
| Anytime value | Quality-vs-time curve; first candidate and first verified candidate |
| Proof cost | Generation time, bytes, proof events, peak proof memory |
| Checking cost | Independent checker time, CPU, memory, throughput, failures |
| Efficiency | Wall time, CPU time, peak memory, energy, strategy overhead |
| Robustness | Worst-family performance, lower-tail score, timeout/crash/regression rates |
| Generalization | IID, leave-family-out, leave-generator-out, structural OOD, time-forward, private holdout |
| Discovery | New qualified representations, strategies, counterexamples and theorem obligations discharged |
| Economics | Cost per qualified improvement, including model/API, compute, checker and human-review cost |

For optimization, add the area under the **certified** optimality-gap curve. Do not compute a gap from uncertified or mismatched bounds.

For SAT, report verified SAT and verified UNSAT separately. For model counting, report each algebra/semantics track separately. For approximate counting, record the guarantee and empirical coverage. For MaxSAT, preserve exact, anytime and incremental tracks.

## 7. Proposed experiment ladder

| Phase | Experiment | Promotion condition |
|---|---|---|
| 0 | Freeze source contracts, corpora, solver versions and resource limits | Reproducible identities and independent ground truth |
| 1 | Re-run the finite #10868 fixtures with source enumeration | All supported examples and negative controls agree |
| 2 | Compare direct solver vs. qualified decomposition on small SAT/#SAT/PMC | No semantic mismatch; independently checked outputs |
| 3 | Add pinned Kissat/CaDiCaL/Ganak/MaxSAT adapters | Correct adapter contracts and reproducible receipts |
| 4 | Compare fixed portfolio, static selector and typed strategies | Same total CPU/wall/memory budgets; overhead reported |
| 5 | Enable #10848 generator co-evolution | Adversarial instances independently admitted; held-out improvement |
| 6 | Add cross-family and structural-OOD evaluation | Improvement survives sealed confirmatory corpus |
| 7 | Integrate #10884 anytime results in Mycelix, Symtropy and Sol-Atlas | Domain-specific independent checkers accept each admitted result |

Do not start automatic strategy co-evolution before the interpreter, type checker and evidence admission path are qualified.

## 8. What would count as a defensible SOTA claim?

A paper/release claim should state, at minimum:

- the benchmark name, exact version, instance universe and split;
- the source blob/commit SHA and all solver versions/configurations;
- hardware, threads, memory, time, energy and total compute budget;
- exact/approximate semantics and checker policy;
- per-instance results and independent re-aggregation;
- best single solver, deployable selector, and hindsight VBS in separate columns;
- all ablations and failed/invalid candidates;
- confidence intervals or paired statistical tests where appropriate;
- a separate result on sealed family/structural holdouts;
- a reproduction package and independent checker.

A single aggregate score, a VBS comparison, or performance on strategy-generated instances is insufficient.

## 9. Research opportunity: optimize the strategy's verification profile

SAT research has demonstrated that proof-checking architecture can materially change the cost of trusted solving: one on-the-fly LRAT approach reported experiments up to 2,432 cores and median proof-checking overhead of at most 42% over untrusted solving.

This suggests a specific Symthaea direction: discover not only algorithms that solve quickly, but decompositions and proof-production schedules that make independent checking cheaper and more parallelizable. Evaluate proof generation, streaming/online checking and final artifact size together. Do not assume that all proof formats or all solver families admit the same trade-off.

Source: Dominik Schreiber, *Trusted Scalable SAT Solving with On-The-Fly LRAT Checking*, SAT 2024, https://doi.org/10.4230/LIPIcs.SAT.2024.25.


## 10. Proof-system-aware benchmark requirements

Add proof backend as a controlled experimental factor. For SAT, compare proof-capable pinned baselines such as CaDiCaL 3.0 alongside the relevant Kissat baseline, keeping incremental capability and proof requirements visible rather than comparing unlike interfaces. CaDiCaL 3.0 reports adapting Kissat techniques to incremental solving and efficient linear proof production with hints; see https://doi.org/10.4230/LIPIcs.SAT.2026.40.

For exact #SAT, evaluate certificate-capable systems separately. The July 2026 analysis of kcps, MICE, CPOG and CLIP finds CPOG simulates MICE and kcps, while MICE and kcps are exponentially incomparable. CLIP simulates all three. These are proof-system strength statements, not a guarantee of practical proof-size or checker-time dominance. See https://doi.org/10.1007/s10817-026-09757-w and https://pmc.ncbi.nlm.nih.gov/articles/PMC13385077/.

For every proof-aware experiment report: solving time, proof-generation overhead, certificate bytes, peak proof/checker memory, checker wall/CPU time, streaming feasibility, independent checker outcome and time-to-trusted-result. Keep solver search seed/configuration fixed between proof/no-proof ablations where feasible; disclose when proof logging changes search behaviour.


## 11. Active experimental design: learn which benchmark to run next

A strategy-discovery system should optimize not only which algorithm to execute, but also **which experiment to perform next**. Benchmark runs are expensive, and many easy or redundant instances do not materially distinguish competing strategies.

Research basis: Fuchs, Bach & Iser, *Active Learning for SAT Solver Benchmarking*, Journal of Automated Reasoning (2025), https://doi.org/10.1007/s10817-025-09729-6. The work treats benchmark selection as dynamic active learning: select informative instances to estimate a new solver's rank while reducing benchmarking cost.

The paper reports approximately **92% PAR-2 ranking prediction accuracy at about 10% of full-benchmark runtime** on the SAT Competition 2022 Anniversary Track. This is a historical published reference, not a demonstrated result on the 2026 Main Track and not a guarantee of transfer to Symthaea's evolving strategy population.

For Symthaea, distinguish three processes:

1. **Development-set active selection:** choose known, non-confirmatory instances that are most likely to change a strategy-selection decision.
2. **Adversarial instance generation:** create new contract-valid cases that expose weaknesses or distinguish hypotheses.
3. **Sealed confirmation:** evaluate the frozen successor on an untouched full corpus, or a preregistered statistically valid confirmatory design. The selector and generator cannot alter its membership, truth labels, scoring, or resource limits.

### Baselines

- Full fixed development set.
- Uniform random sample.
- Stratified sample by benchmark family, generator lineage and structural features.
- Active-learning benchmark selector.
- Adversarial generation alone.
- Combined active selection + adversarial generation + strategy evolution.

First reproduce the published historical reference (approximately 92% ranking accuracy at approximately 10% evaluation time on the 2022 Anniversary Track). Then test a prospective Symthaea stretch target of **greater than 95% rank accuracy at no more than 10% of full development-set evaluation cost** on the 2026 development pool, with family-wise ranking error, calibration and selection regret reported. This is a hypothesis, not an achieved result.

### Measurements

- Strategy-rank accuracy and rank correlation.
- Decision regret: quality lost by selecting the wrong strategy from the limited sample.
- Compute spent before the selector reaches a stable decision.
- Number of instances and total solver/checker CPU time required.
- Selection-model calibration and uncertainty quality.
- Coverage of families, generators and structural clusters.
- Frequency with which a selected experiment actually changes the next research or engineering decision.
- Selector overhead and the cost of false confidence.

The acquisition function should estimate expected value of information: probability that an experiment changes a decision, multiplied by the consequence of that decision, balanced against experiment cost and coverage obligations. An experiment can be valuable because it falsifies a proposed transformation or invalidates an attractive strategy, not only because it improves a score.

**No adaptive peeking at the confirmatory set.** Repeatedly selecting the most informative examples from a reused test set creates selection bias. Active sampling is for development and diagnosis; final promotion still requires a sealed evaluation and valid uncertainty accounting. Use #10805's existing receipt/manifest format to record the sampling policy, selected instance IDs and selection-model version; do not create a parallel evidence format.


## 12. Future quantified benchmark family: QBF / #QBF / MaxQBF

Do not merge QBF results with SAT, #SAT or PMC tracks. The recent ECCC report TR26-279 (posted 27 October 2026), *On Knowledge Compilation Languages for QBFs*, introduces tree/DAG compilation languages and proof systems for #QBF, and reports an initial tree-like compilation/counting implementation that can represent doubly exponential counts. It compares against d4-QBF. This is a new research baseline, not yet a Symthaea result: https://eccc.weizmann.ac.il/report/2026/279/.

Projected knowledge compilation also now has a formally validated proof framework that carries Skolem assignments, with a C checker and a CakeML/HOL4-verified checker (Bryant, Tan & Heule, SAT 2025): https://doi.org/10.4230/LIPIcs.SAT.2025.8.

Before benchmarking QBF, pin a distinct contract for:
- quantifier prefix and alternation depth;
- existential Skolem dependencies;
- truth / #QBF / MaxQBF semantics;
- count magnitude and exact representation;
- proof backend and checker acceptance.

Evaluate d4-QBF and reproducible artifacts for TR26-279 when available, tree versus DAG compilation, relevant #QBF proof systems and uncompiled QBF baselines. Record alternation depth, dependency width, Skolem artifact size, compilation size, exact count bit width, checker time/memory and time-to-trusted-result. No QBF implementation or SOTA claim follows from the structural prefix validator alone.


## 13. Claim-specific proof and checker baselines

Add proof capability as a typed benchmark factor, not just a Boolean "proof enabled" flag.

| Claim | Candidate evidence family | Important non-equivalence |
|---|---|---|
| SAT / UNSAT | Direct source-level model validation; DRAT/LRAT checking for UNSAT | A solver's UNSAT status without a checked certificate is not a proved result |
| Exact #SAT / WMC | CPOG, MICE, kcps and certified compiled representations | Decision UNSAT proofs do not establish a model count |
| Projected count | Skolem-annotated projected knowledge-compilation certificates | Full-model count does not establish visible support cardinality |
| QBF / #QBF | Quantifier prefix + Skolem strategy evidence; Q-MICE and QBF compilation proof systems | Unordered hidden-variable scopes erase dependency semantics |
| MaxSAT optimum | MaxSAT-specific core-guided proof logging or a verified pseudo-Boolean proof such as VeriPB | An UNSAT check at one bound alone does not certify the complete optimum derivation |
| PAC count | Theorem-backed approximation guarantee, epsilon/delta or alpha, seed/policy and checker | MPFR/high precision is not a PAC guarantee |

Relevant research:
- Q-MICE proof system for counting winning strategies / Skolem functions: https://eccc.weizmann.ac.il/report/2026/101/
- MaxSAT-specific proof logging for core-guided OLL solvers: https://conf.researchr.org/details/ase-2025/ase-2025-papers/49/Efficient-and-Verifiable-Proof-Logging-for-MaxSAT-Solving
- VeriPB general pseudo-Boolean proof ecosystem: https://veripb.org/

Benchmark each producer/checker pair on the same instance family. Report proof bytes, generation overhead, independent checking time/memory, checker rejection rate under mutations, streaming support, and time-to-trusted-result. Do not compare only raw proof generation or only solver runtime. For MaxSAT, include proof-carrying lower-bound derivation cost in addition to incumbent feasibility.


## 14. Incremental query workloads and amortized strategy reuse

Industry-facing evaluation must include sequences of related requests, not only isolated instances. CaDiCaL 3.0 emphasizes incremental assumptions and proof logging; PBCount2 adds exact projected and incremental pseudo-Boolean model counting and reports completing at least 1.40x the projected and 1.18x the incremental benchmark count of competing methods; MaxSAT Evaluation 2026 includes an incremental IPAMIR track. These motivate a new workload benchmark, not a claimed Symthaea win.

Sources:
- CaDiCaL 3.0: https://doi.org/10.4230/LIPIcs.SAT.2026.40
- PBCount2: https://doi.org/10.1609/aaai.v39i11.33240
- MaxSAT Evaluation 2026: https://maxsat-evaluations.github.io/2026/rankings

Compare restart-per-query, native incremental solver APIs, fixed cached portfolios, typed strategy reuse and a no-cross-query-reuse ablation. Use exact same workload traces and aggregate resource budgets.

Required workload measurements:
- total CPU/wall time and per-update p50/p95/worst latency;
- time to first and first independently verified result;
- proof generation/check time and bytes per query;
- peak and retained memory;
- cache reuse rate, invalidation rate, recomputation cost and false-reuse count;
- cumulative verified result value across the full workload.

Every reusable artifact must bind to source digest, task kind, assumptions/update revision, variable universe, objective and units, quantifier prefix/dependencies, transformation/hyperedge registry revision, solver/proof backend and checker contract. Any semantic mismatch must cause recomputation or rejection. Test stale bounds, witnesses, clauses, projected support, compiled representations and Skolem dependencies after query updates.

The workload trace and per-transition evidence must use #10805's existing manifest/receipt format. Do not create a second provenance system. See [#10965](https://github.com/Luminous-Dynamics/symthaea/issues/10965) for the implementation workstream.


## 12. Quantified Boolean formulas — strategy quality is the result

Add a separate QBF evaluation lane. The central output is not only a TRUE/FALSE decision: it is a winning strategy whose dependencies and behavior can be checked against the original matrix.

### External evaluation surface

[QBFGallery 2026](https://qbf.pages.sai.jku.at/gallery26/) evaluates QBF and DQBF and defines these distinct tracks:
- PCNF main track;
- crafted instances;
- random formulas;
- 2QBF;
- prenex non-CNF;
- DQBF.

It accepts QDIMACS, QCIR and DQDIMACS inputs. The gallery's 2026 event presentation took place at the QBF Workshop on 19 July 2026. The public page identifies a results presentation and benchmark collection, but this plan does not invent a numeric best-solver count where a machine-readable result table has not been pinned.

### Competitive references and proof research

- CAQE is described in a SAT 2026 paper as the strongest practice-wise QBF algorithmic paradigm and as clearly dominating recent QBF competitions. This is a literature characterization; it is not a complete current leaderboard or a strategy-extraction score. See *Towards Understanding the Complexity of CAQE: A Proof-Theoretic Analysis of Its Core Procedure*, SAT 2026 proceedings: https://drops.dagstuhl.de/entities/volume/LIPIcs-volume-377.
- *Proof Systems for QBF Synthesis: Extracting Skolem and Herbrand Functions* (SAT 2026) introduces a sound and complete proof system with polynomial-time extraction of Skolem/Herbrand functions and intermediate game values: https://doi.org/10.4230/LIPIcs.SAT.2026.3.
- *On Proof Systems for #QBF* (SAT 2026) studies proof systems for counting winning strategies and proposes Q-MICE. #QBF counts Skolem functions; it is not QBF truth evaluation: https://doi.org/10.4230/LIPIcs.SAT.2026.32.

### Symthaea comparison protocol

Keep four result layers separate:

1. **Decision:** independently verified QBF TRUE and FALSE.
2. **Strategy:** explicit Skolem function for TRUE or Herbrand function for FALSE.
3. **Certificate:** proof object and independently validated proof, when a supported proof backend exists.
4. **Counting:** #QBF count of distinct winning strategies, a separate research task and result type.

Compare pinned versions of CAQE, DepQBF and Qute where their supported interfaces and configurations are available. Report decision-only solver runs separately from strategy-producing or proof-logging runs. A solver's proof may permit extraction, but do not assume it provides the same output or cost as a native strategy interface.

### Required metrics

- verified TRUE count and verified FALSE count separately;
- time to solver decision, strategy candidate, and independently verified strategy;
- strategy function/table/circuit size and dependency count;
- proof generation time/bytes and proof checker time, where available;
- strategy extraction overhead relative to decision-only solving;
- fraction of solved instances for which strategy extraction succeeds;
- checker throughput, memory and counterexample rates;
- results by quantifier alternation depth, PCNF/crafted/random/2QBF/non-CNF family and held-out application;
- preprocessing traceability: bind extracted functions to the original preprocessed/unpreprocessed contract and verify reconstruction where preprocessing changes variables or dependencies.

### Proposed milestones

| Stage | Target |
|---|---|
| Finite reference | All exhaustive small formulas and prefixes agree with an independent oracle; every extracted function independently validates |
| External reproduction | Reproduce pinned QBFGallery 2026 decision results for the selected public corpus and solver builds |
| Strategy baseline | Extract and validate strategies on every instance where the baseline solver exposes adequate proof/strategy data |
| Initial improvement | Improve verified strategy extraction coverage without reducing decision correctness |
| Strong result | Improve time-to-verified-strategy and/or strategy size on held-out families while retaining the same resource envelope |

Do not set a numeric solved-count target until the exact QBFGallery results artifact, solver versions, benchmark splits and scoring rules are pinned in #10805.

### Semantics boundary

Prenex QBF dependencies are determined by quantifier order. DQBF instead supplies explicit, potentially non-prenex dependency sets and cannot be represented by the standard prefix rule alone. Keep DQBF behind a different typed task/dependency contract. Likewise, do not treat #QBF (count of winning strategies) as ordinary #SAT or as a QBF decision result.

The bounded implementation in #10991 is an exhaustive reference producer/checker over CNF matrices. Its truth-table strategies are a correctness oracle, not the target representation for scalable synthesis. Production strategy discovery should explore AIG/BDD/circuit representations and proof-derived intermediate game values, and optimize verified output size plus checking cost.


## 13. Strategy-function representation and search optimization

QBF strategy evaluation must measure the complete artifact, not only whether a formula was decided.

### Representation pipeline

For every independently verified source strategy:

1. **Semantic support reduction.** Prove which declared dependencies actually influence each Boolean strategy function. Remove only inessential variables, retain prefix order, and exhaustively verify equivalence over the original dependency universe.
2. **Canonical ROBDD construction.** Compile the reduced truth table with a separate BDD decision order. The BDD decision order is a permutation of the support indices; it must never rewrite or broaden the semantic dependency scope.
3. **Order optimization.** Use exact permutation search as a small-scope oracle when (n!) is inside the explicit evaluation budget. Use deterministic bounded sifting above that range. Record initial and final node counts, order evaluations and the selected order.
4. **Independent checking.** Check BDD topology, reachability, unique/reduced nodes, order constraints, declared support, and every truth-table row. The original QBF strategy must still be independently checked against the original matrix and prefix.

### Research context

- The 2026 frontier-aware OBDD work links prefix-frontier width with pathwidth for CNF compilation and reports substantial reductions from bounded-insertion ordering refinement. Its results are for feature-interaction graphs and CNF OBDD compilation; they are a candidate structural seed for strategy BDDs, not direct performance evidence for Symthaea: https://www.nature.com/articles/s41598-026-74296-8.
- The January 2026 feature-model BDD study combines SAT preprocessing, variable ordering, clause ordering and alternative/multithreaded construction methods; it reports compilation of previously infeasible eCos feature models: https://doi.org/10.1016/j.jss.2025.112566.
- The SAT 2026 QBF synthesis proof-system work provides a stronger long-term destination: a proof representation from which Skolem/Herbrand functions and intermediate game values can be extracted in polynomial time: https://doi.org/10.4230/LIPIcs.SAT.2026.3.

### Required measurements

| Layer | Measure |
|---|---|
| Source scope | Original dependencies per function |
| Semantic support | Essential dependencies, removed dependencies, pair checks |
| Truth table | Original and reduced entries, bytes, build time |
| ROBDD | Reachable node count, edge count, decision order |
| Ordering | Exact minimum where budgeted, sifting result, order evaluations |
| Build | Peak nodes, peak memory, time-to-first-valid BDD |
| Verification | Structural check time, assignments checked, checker memory |
| QBF search | Quantifier-node visits, partial-matrix checks, early-TRUE/FALSE prunes, matrix leaf evaluations |
| End-to-end | Time and memory to independently verified strategy plus compact representation |

### Fair comparisons

- Compare natural prefix order, reverse order, exact optimum (small scopes), bounded sifting and future graph-derived seeds.
- Do not compare a strategy BDD with a source truth table as if the row count and node count were equivalent units.
- Report both original and minimized support; a smaller BDD caused by dropping an unused dependency is valid only when support equivalence is checked.
- Report decision time separately from strategy synthesis, support reduction, ordering search and independent verification.
- For larger functions, include construction peak size, not just final BDD size; a small final diagram can still require an impractical peak intermediate.
- Keep a hidden held-out set and record the function family. Boolean-function micro-corpora are correctness/heuristic tests, not substitutes for application-derived QBF strategies.

### Near-term finite qualification

The current finite research lane uses exhaustive small QBF oracles, generated multi-alternation QBF instances, exhaustive three-input Boolean-function support/BDD equivalence, exact order search, and node/tampering/resource negative controls. These establish bounded semantics and heuristic behavior only. They do not establish scalable solver superiority.

### Candidate next paradigm: frontier-seeded multi-start ordering

Use the Boolean function's interaction graph (pairwise non-zero mixed Boolean derivatives) to create deterministic min-fill/frontier-style seed orders. Run the same bounded sifting from several seeds and retain the smallest independently verified ROBDD. Compare this against natural-order sifting and exact optima on small functions. Do not assume pairwise interaction graphs capture every high-order dependency; retain exact small-scope results as the oracle and test the seeds on held-out functions.


## 14. Frontier-refined multi-start ordering

The current ROBDD ordering pipeline now compares deterministic starts from:
- natural dependency order;
- reverse dependency order;
- pairwise Boolean-interaction graph min-fill;
- reverse min-fill;
- min-fill refined by bounded insertion over the vertex-separation frontier;
- reverse frontier-refined order.

The interaction graph is computed exactly from the truth table. An edge ((i,j)) exists if the mixed Boolean derivative (f_{00}\oplus f_{01}\oplus f_{10}\oplus f_{11}) is nonzero for at least one assignment to the other inputs. This graph is only an ordering heuristic; it is not the semantic dependency graph and does not claim to capture higher-order structure completely.

Frontier refinement optimizes the lexicographic tuple:
1. maximum live frontier;
2. total live-frontier mass;
3. frontier profile.

Only strict improvements are accepted. Therefore the maximum frontier cannot exceed the min-fill seed's maximum frontier. BDD node count, not graph score, decides which independently verified final BDD is selected.

### Qualification experiments

- Exhaustively compare the pairwise-interaction graph with an independently written mixed-derivative oracle over all 256 three-input Boolean functions.
- Use a fixed five-vertex graph where min-fill has frontier profile `[0,1,2,3]`; bounded insertion produces `[0,1,2,1]`, reducing maximum frontier from 3 to 2.
- Test order permutation, graph symmetry, self-edge and out-of-range rejection.
- Enforce separate interaction-check, frontier-order and BDD-order evaluation budgets.
- Verify the final selected BDD against every truth-table row.

These finite experiments establish exact graph construction on three-input functions and a frontier improvement on a specific graph. They do not establish improved average BDD size on application QBF strategies. The benchmark must report whether lower frontier actually predicts lower ROBDD nodes across held-out function families, and include counterexamples where it does not.

### New target metrics

| Metric | Required comparison |
|---|---|
| Interaction graph cost | Pair checks and runtime versus truth-table size |
| Structural seed | Min-fill versus frontier-refined max/total frontier |
| Multi-start coverage | Unique seeds actually evaluated under global budget |
| Final ROBDD quality | Nodes versus natural-order and natural-sifting baselines |
| Optimality gap | Exact minimum on scopes where (n!\) is affordable |
| Construction resources | Peak nodes, peak bytes and order-evaluation count |
| Correctness | Exhaustive BDD equivalence and original QBF strategy acceptance |

No claim of dominance is warranted unless the frontier seed reduces verified BDD node count on held-out, application-derived strategy functions at matched memory and time budgets.


## 15. Exact #SAT proof logging — CPOG as a unifying candidate

Exact #SAT proof logging is a separate evaluation lane from QBF strategy synthesis and #QBF strategy counting.

### Current proof-complexity baseline

Beyersdorff, Fichte, Hecher, Hoffmann and Kasche, *The Relative Strength of #SAT Proof Systems* (Journal of Automated Reasoning 70(2), article 12, published 20 July 2026; DOI 10.1007/s10817-026-09757-w), determines the simulation order among:
- **kcps** — knowledge compilation proof system;
- **MICE** — model-counting induction by claim extension;
- **CPOG** — certified partitioned-operation graphs;
- **CLIP** — circuit-linear introduction proposition, a stronger theoretical proof system.

The paper establishes that CPOG polynomially simulates both MICE and kcps, while MICE and kcps are exponentially incomparable. CLIP simulates all three, but is primarily a theoretical strength baseline. The paper motivates CPOG as a plausible unified proof-logging target for #SAT solving approaches. These are proof-system simulation/size results, not evidence that a particular CPOG implementation will be faster or smaller on every benchmark.

Sources:
- https://doi.org/10.1007/s10817-026-09757-w
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13385077/

### Symthaea comparison protocol

Keep these claims distinct:
1. exact #SAT result (count of satisfying assignments);
2. proof object accepted by an independent CPOG checker;
3. proof-system simulation/certification of another proof format;
4. projected #SAT (support union, not arithmetic sum);
5. weighted/algebraic counting (requires its own coefficient/aggregation contract);
6. #QBF (counts winning Skolem functions), which is not #SAT.

Compare:
- direct enumeration and current exact decomposition reference;
- pinned external exact #SAT solvers in count-only mode;
- proof-emitting modes of compatible systems (kcps, MICE, CPOG);
- translation from MICE/kcps into CPOG where the formal simulation is implemented;
- CLIP as a theoretical proof-size reference unless a practical implementation exists.

At matched source instances and compute limits, report exact count agreement, solved instances, proof size/bytes, generation time, independent checker time/bytes, peak memory, aggregation/decomposition structure, free-variable multiplicity preservation and worst-family performance.

### Promotion gates

- The complete CPOG formal rules and supported artifact/version must be pinned.
- Independent proof checking must validate each implemented rule and the exact original CNF/variable universe.
- No count is accepted from producer-supplied values/acceptance flags alone.
- Direct enumeration differential tests cover every supported finite fixture.
- Negative controls include omitted free variables, overlapping components, incomplete branches, incorrect sum/product/support union, stale source identity, invalid proof nodes and overflow.
- No solver-performance claim until a pinned external corpus and matched proof/count interfaces are benchmarked.

The CPOG workstream is tracked separately in #11036. It must reuse #10868/#10899/#10884/#10887/#10805 rather than introduce a second proof/result/receipt contract.


## 17. Arithmetic weighted interactions — a second structural signal

The current pairwise Boolean-interaction graph uses the GF(2) mixed derivative
\[
f_{00}\oplus f_{01}\oplus f_{10}\oplus f_{11}.
\]
This correctly captures nonzero Boolean pair interactions, but affine functions such as XOR have zero pairwise GF(2) mixed derivatives. A second ordering signal now measures the arithmetic mixed finite difference of the Boolean truth table after its fixed (0/1) embedding:
\[
w_{ij}=\sum_{x\in\{0,1\}^{n-2}}
\left|f_{00}(x)-f_{01}(x)-f_{10}(x)+f_{11}(x)\right|.
\]
These weights are **not** a semantic dependency certificate or a theorem about optimal BDD order. They provide a second deterministic heuristic signal.

### Weighted ordering

- Build a symmetric, zero-diagonal matrix with explicit overflow and context-check budgets.
- Project positive weights to an adjacency graph for frontier measurement.
- Weighted min-fill ranks candidate eliminations by weighted induced fill cost, then fill-edge count, weighted degree, active degree and variable ID.
- When an elimination introduces a fill edge between (a,b), its induced weight is the checked product of the selected variable's current coupling weights to (a) and (b). The induced weight is propagated to later elimination steps.
- Refine the weighted seed with the existing bounded frontier insertion and pass it to the same BDD sifting engine.
- Keep the original GF(2) graph, natural order, reverse order and their refined variants as separate ablation arms. Select solely by final independently verified BDD node count under one shared search budget.

### Frozen known-answer experiments

1. **XOR3:** the GF(2) interaction graph has no pairwise edges. The arithmetic weight matrix has weight 4 on each of its three pairs (total 12; six context checks). This illustrates a signal difference, not that weighted ordering must improve XOR's BDD size.
2. **Four-variable constructed function:** for truth-table values (variable 0 is the least-significant assignment bit)
`[1,0,0,1,1,0,0,1,1,0,1,0,1,0,0,1]`,
natural order and GF(2) min-fill produce 7 nodes. Arithmetic weighted min-fill returns `[2,3,0,1]` and produces 6 nodes. Exact search finds 5; the bounded multi-start pipeline reaches 5 on this fixture.

The second result is one constructed finite example. It is not an average-case result, an application result, or an external SOTA comparison.

### Required ablation

Compare under matched truth tables and identical resource limits:
- natural / reverse;
- GF(2) min-fill and frontier-refined GF(2);
- arithmetic weighted min-fill and its frontier-refined variant;
- bounded sifting from each unique seed;
- exact permutation minimum for scopes where (n!\) fits.

Report per function: source family, support size, both interaction matrices (or their digests and exact weights), pair/context checks, seed orders, induced-fill policy, frontier profiles, unique seed count, order evaluations, search-complete flag, initial/final nodes, exact minimum when available, build peak nodes/bytes, and independent checker time.

Hold out complete function generators and QBF application families. The confirmatory split must not be used to tune weights or tie-breaks. The metric of interest is *verified node-count reduction at equal total construction + checking budget*, not a lower graph score alone.


## 18. QBF external solver interface audit — do not conflate decisions with strategies

### CAQE as the practice-oriented baseline

The SAT 2026 proof-theoretic analysis characterizes CAQE as the most successful practice-wise QBF algorithmic paradigm and says it clearly dominated recent QBF competitions. This is an important baseline to include; it is not a numerical QBFGallery 2026 score for Symthaea or a claim that the basic CAQE core and all optimized CAQE releases are equivalent.

The public CAQE repository documents:
- `caqe` for prenex-CNF QBF in QDIMACS;
- `dcaqe` as a separate DQBF solver using DQDIMACS;
- `--qdo` for QDIMACS output containing partial assignments.

A partial assignment must not be automatically classified as a complete Skolem/Herbrand strategy. The benchmark adapter must independently establish completeness, dependency legality and matrix validity before promoting output into the strategy-extraction column.

### Interface classes

Every external result is assigned exactly one capability class per run:

| Class | Meaning | Counts as verified strategy extraction? |
|---|---|---|
| Decision only | TRUE/FALSE decision | No |
| Partial assignment | One or more partial QDIMACS assignments | No, until completed and independently checked |
| Proof emitting | Proof/certificate is returned | No, until proof checking and extraction both succeed |
| Complete strategy | Full Skolem or Herbrand functions in a recognized representation | Only after independent matrix/prefix verification |
| Intermediate game values | Values for quantified subgames | Only after scope, recurrence and source binding checks |

### Baseline protocol

Pin solver commit/release, compiler/toolchain, command line, input digest and resource limits. Run PCNF QBF baselines in their supported interface modes; run DQBF baselines only on the separate DQBF corpus. The 2026 gallery separates PCNF main, crafted, random, 2QBF, prenex non-CNF and DQBF, and accepts QDIMACS, QCIR and DQDIMACS. Do not mix track totals or formats.

For every solver, report separately:
- decision coverage and time;
- proof availability, proof bytes, proof-check outcome and time;
- partial assignment availability and completion rate;
- strategy extraction coverage and time;
- time-to-independently-verified strategy;
- strategy function/circuit/ROBDD size and checker cost.

The 2026 proof-system synthesis result gives a second comparison path: compile a QBF proof in the supported proof system, extract Skolem/Herbrand functions and intermediate game values, then independently verify the resulting functions against the original matrix and prefix. Compare this to solver-native proof/strategy output rather than treating these as equivalent by assumption.

### Promotion gate

The selected QBFGallery corpus and result artifact must be pinned before publishing any numeric external target. No current numeric CAQE-vs-Symthaea score is claimed by this plan. CAQE's practice strength is a literature baseline; the numerical head-to-head remains future benchmark work.


## 19. Multi-representation strategy compilation — ROBDD and AIG

Strategy functions should not be tied permanently to a single circuit representation. The implementation now has two bounded compilation paths from the same finite Boolean source table:

- **ROBDD:** reduced ordered decision DAG, whose size depends on decision order and whose canonicality is relative to a fixed order.
- **AIG:** AND nodes with complemented edges, compiled by Shannon decomposition and structural hashing. Its graph is not a canonical representation across all decomposition orders.

### Fair size and cost comparison

Do not compare BDD node count directly with AIG AND-node count as if these were the same unit. For each representation record:
- input count, internal node count and edge/reference count;
- exact deterministic serialized byte length under a pinned encoding;
- construction time and peak node/byte count;
- evaluation time over the same assignments;
- independent checker time and checker memory;
- search/order optimization cost and whether the search was complete;
- original and minimized semantic support.

### Representation trust boundary

The AIG/ROBDD checker validates only that the compiled circuit is equivalent to its source strategy function. The original QBF matrix/prefix verifier remains a separate gate and must run before a circuit is promoted as a winning strategy. An AIG is not a QBF proof certificate.

### Required finite and application evaluation

- Exhaust all 256 three-input Boolean functions and compare both circuit outputs with the same source truth tables.
- Include constant, identity, negation, parity, multiplexer, majority, comparator and carry-like function families.
- For small functions, use exact ROBDD variable-order search as a reference, but do not treat an exact BDD size as a lower bound for AIGs or vice versa.
- Evaluate both AIG decomposition orders and ROBDD decision orders under the same aggregate compilation budget.
- On application-derived QBF strategies, report strategy extraction time separately from AIG/ROBDD compilation and checking.

The promotion objective is a lower verified serialized artifact cost at equal total construction plus verification budget, with held-out families and unchanged semantic acceptance. A smaller representation is useful only when it remains independently equivalent and the original strategy remains a winning strategy.


## 20. Portable QBF game proofs and end-to-end evidence portfolios

The finite QBF reference lane now adds a second certificate alongside the winning Skolem/Herbrand strategy: a post-order game-proof tree. It records matrix leaves, all-clauses-satisfied early-TRUE rules, falsified-clause early-FALSE witnesses, and quantified decision nodes with their EXISTS/OR or FORALL/AND recurrence.

The proof is bound to the ordered prefix and source CNF by an independent verifier. The in-memory proof is serialized as canonical `LD-QBFPROOF1` bytes, decoded under variable/node/byte limits, and independently reverified after decoding. The portfolio assembler then combines:
1. independently validated game-proof bytes;
2. independently validated winning strategy;
3. support-minimized ROBDD;
4. AIG with complemented edges;
5. exact BDD/AIG verification reports and aggregate artifact sizes.

The source game proof, source winning strategy and compiled graph equivalence are three separate trust checks. Passing one does not imply the others.

### New proof/portfolio metrics

| Layer | Metric |
|---|---|
| Game proof | Post-order node count by node kind |
| Pruning | Early-TRUE count, early-FALSE clause witnesses, complete matrix leaves |
| Expansion baseline | Full binary prefix-tree node bound (2^{n+1}-1) |
| Proof construction | Builder visits and time |
| Proof checking | Independent verifier visits/time, source-clause inspections |
| Serialization | LD-QBFPROOF1 byte length, decode time, malformed-input rejects |
| Strategy | Opponent assignments checked and strategy table entries |
| Compiled graphs | ROBDD nodes/bytes, AIG AND nodes/bytes |
| End-to-end | Total proof + BDD + AIG bytes and time to full verified bundle |

For the expansion comparison, report the full-tree bound and actual proof-tree count, but do not call the node reduction a polynomial proof-size guarantee. Early pruning is input-dependent; worst-case proof size remains exponential in prefix length for this tree reference. The verifier's clause-evaluation work also depends on formula size, not only proof nodes.

### Known external certificate practice

- The QBFcert framework includes DepQBF, QRPcheck, QRPcert and CertCheck. It offers a real external syntactic-proof and certificate-extraction baseline, rather than assuming every QBF solver emits the same certificate type: https://fmv.jku.at/qbfcert/.
- CAQE's public QDIMACS interface distinguishes decision output from `--qdo` partial-assignment output. The latter is not automatically a complete strategy; Symthaea must check function completeness, prefix-legal dependencies and the original matrix: https://github.com/ltentrup/caqe.
- The SAT 2026 QBF synthesis proof-system paper explicitly frames intermediate game values and Skolem/Herbrand extraction as an alternative target, with a sound/complete system and polynomial-time extraction for proofs in that system: https://doi.org/10.4230/LIPIcs.SAT.2026.3.
- #QBF is a further distinct task: counting winning Skolem functions. The Q-MICE work is a proof-system reference for that counting task, not a QBF truth-decision baseline: https://doi.org/10.4230/LIPIcs.SAT.2026.32.

### Promotion and comparison rules

- Compare our finite game-tree proof size against our own exhaustive full-expansion tree first; do not compare its node count directly with QRP lines, Q-res proof bytes, or the SAT 2026 synthesis proof system without an explicit translation and common semantics.
- Benchmark syntactic certificates, semantic strategy functions, and game-value traces as separate certificate classes.
- A serialized proof is accepted only after bounded decode, source/prefix verification and deterministic re-encoding equality.
- Report proof-check time separately from strategy-check time and BDD/AIG equivalence time.
- Confirm results on QBFGallery 2026 instances after pinning its actual public benchmark archive, selected tracks, solver builds and licenses. The gallery advertises PCNF main, crafted, random, 2QBF, prenex non-CNF and DQBF tracks; no numeric external solved-count target is claimed here.


## 21. Algebraic interaction hypergraphs and Pareto-qualified representation selection

The BDD ordering pipeline now includes three distinct structural descriptions of a strategy function:

1. **GF(2) pairwise interaction graph:** edges from nonzero mixed Boolean derivatives.
2. **Arithmetic weighted pairwise graph:** integer mixed finite-difference magnitude. This can reveal coupling in XOR functions whose GF(2) pairwise graph is empty.
3. **ANF interaction hypergraph:** Boolean Möbius transform recovers the full algebraic normal form. Each nonzero monomial of degree at least two becomes a hyperedge; linear terms and the constant term are retained separately.

These are ordering heuristics, not semantic scope evidence. The exact support checker still determines which dependencies actually affect the function. Hyperedge vertices are positions in the ordered dependency scope, never global variable IDs. The ANF transform itself is tested by reconstructing every row of all 256 three-input Boolean functions.

### Ordering ablation

For each source-verified Boolean strategy function, compare:

- natural and reverse dependency order;
- GF(2) pairwise min-fill and reverse;
- arithmetic weighted min-fill and reverse;
- frontier-refined pairwise and weighted seeds;
- ANF hypergraph min-fill and reverse;
- bounded sifting from each seed;
- exact minimum order for small scopes.

Record seed collisions/deduplication, transform checks, pair-context checks, ANF hyperedge count/rank, frontier profile, order evaluations, whether the bounded search completed, final ROBDD nodes, and exact-optimum gap wherever exact search fits the budget.

The key hypothesis is not that ANF always wins. It is that higher-order algebraic structure supplies an additional ordering signal when pairwise graphs are empty, weak, or misleading. Require held-out function families and source-derived QBF strategies before promoting an ordering method.

### Multi-objective representation cost

Each independently verified function now has both a canonical ROBDD and AIG candidate. Record a cost vector:

| Dimension | Meaning |
|---|---|
| Encoded bytes | Actual canonical LD-ROBDD1 or LD-AIG-1 payload size |
| Internal nodes | ROBDD decision nodes or AIG AND nodes |
| Logic depth | Maximum source-to-terminal/output decision/gate depth |
| Worst-case evaluation work | ROBDD decision visits; current AIG interpreter's complete stored-node passes |
| Estimated verification work | Explicit structural work estimate plus exhaustive assignments times worst-case evaluation steps |
| Measured runtime | Wall/CPU time on pinned hardware; do not substitute the estimate |
| Peak memory / energy | Measured separately for construction, evaluation and checking |

The compiler computes the nondominated (Pareto) set rather than assuming one representation dominates in every workload. The default selection is the smallest serialized member of that frontier; both encodings remain available so a caller can choose a different point under its storage, latency or checker constraints.

**Important:** the current verification-work field is a deterministic structural estimate, not a measured instruction count, runtime or energy. The benchmark harness must report empirical construction, evaluation and checker timings before any performance claim.

### External QBF comparison — match task and output contract

QBFGallery 2026 exposes PCNF main, crafted, random, 2QBF, prenex non-CNF and DQBF tracks. Its documented command-line result convention is exit 10 for TRUE and 20 for FALSE; that is a decision interface, not by itself a complete Skolem/Herbrand function or independently checkable certificate. Keep DQBF separate because its explicit dependency semantics are not the prenex QBF prefix semantics used by this reference engine.

CAQE is a necessary practice-oriented baseline; compare pinned CAQE, DepQBF and Qute configurations where runnable artifacts and compatible formats exist. Separate:
- decision-only runs;
- proof-logging runs;
- partial-assignment / dependency-scheme outputs;
- complete extracted strategy functions;
- independently checked proof certificates;
- intermediate game-value outputs.

CAQE's QDIMACS partial assignment output is not automatically a complete strategy. A candidate receives strategy credit only after all controlled variables/functions are present, legal prefix dependencies are established, and the original matrix is checked for every opposing-player assignment.

### Historical score anchor and target discipline

The 2023 QBF Gallery report provides a reproducible historical anchor: CAQE-pre solved 231 PCNF main-track instances, while QuAbS-CAQE solved 286 PNCNF instances. They are different tracks and must not be presented as one ranking or directly compared. PAR-2 must also be reported; lower is better. These are historical figures, not 2026 SOTA.

The 2026 public Gallery page identifies the 2026 tracks and benchmark collection, but this programme has not yet pinned a complete machine-readable 2026 score table, exact benchmark archive, solver artifact hashes and resource limits. Therefore no numerical 2026 "beat SOTA" solved-count target is claimed yet.

### Promotion gates

1. Exact finite correctness: complete reconstruction for all three-input Boolean functions; 0 false accepts under mutated BDD/AIG/strategy/proof controls.
2. Ordering evidence: report exact-optimum gaps on small functions and held-out multi-input functions; no global-optimality claim from bounded sifting.
3. Portfolio evidence: demonstrate Pareto tradeoffs with actual canonical byte lengths and measured target-hardware latency/checker cost.
4. External reproduction: pin QBFGallery 2026 benchmark hashes and solver versions, reproduce the selected competition track and scoring protocol.
5. Strategy comparison: report verified strategy extraction coverage and time-to-verified-strategy separately from decision count.
6. Research claim: only claim improvement where the same instances, hardware, resource limits, task semantics and output obligations are used.

### Literature and baselines

- QBFGallery 2026 tracks and exit-code interface: https://qbf.pages.sai.jku.at/gallery26/
- SAT 2026 proof system for extracting Skolem/Herbrand functions and intermediate game values in polynomial time from proofs in that system: https://doi.org/10.4230/LIPIcs.SAT.2026.3
- CAQE proof-theoretic analysis, which describes CAQE as the strongest practice-wise QBF algorithmic paradigm and clearly dominating recent competitions: https://drops.dagstuhl.de/entities/volume/LIPIcs-volume-377
- Historical 2023 QBFGallery results: https://qbf.pages.sai.jku.at/gallery23/
- 2026 frontier-aware OBDD refinement on feature interaction graphs: https://www.nature.com/articles/s41598-026-74296-8. It is a CNF/feature-graph result; its numerical results are not transferable directly to QBF strategy functions.


## 14. QBF proof trees versus residual-state proof DAGs

A new QBF knowledge-compilation paper is directly relevant: Chede, Krishan & Shukla, *On Knowledge Compilation Languages for QBFs*, ECCC TR26-279, report published 27 October 2026. They study tree-based and DAG-based QBF compilation languages, their proof-system consequences for #QBF, and an initial compiled #QBF solver compared with d4-QBF. They report a minor Q-MICE extension that efficiently simulates the proposed static proof systems and is exponentially stronger than each of them. This establishes that tree-to-DAG knowledge compilation and #QBF are active research directions; Symthaea must not claim the general DAG idea as unprecedented. Source: https://eccc.weizmann.ac.il/report/2026/279/.

### Our narrower experimental hypothesis

The residual-CNF DAG in #11130 uses a specific state identity:
`(quantifier-prefix position, structurally canonical residual CNF)`.

For each branch, the verifier recomputes the residual CNF after assigning the next quantified variable. A state is shared only when both the remaining prefix and canonical residual formula are identical. This is stronger than a hash-only match: the intern key is compared structurally, so collisions cannot merge non-equivalent games.

The proposed comparison is:
- existing contextual proof tree with source-matrix terminal witnesses;
- residual-CNF proof DAG with shared equivalent suffix states;
- proof tree / proof DAG crossed with strategy extraction and BDD/AIG compilation;
- decision-only solver output;
- #QBF counting as a separate target and output type.

### Metrics

- verified instances and root truth values;
- proof tree nodes versus DAG nodes;
- unique residual states, memo hits and shared-child references;
- residual clause/literal storage and bytes;
- construction visits, canonicalization work, peak memory and time;
- verifier visits, residual recomputation work, memory and time;
- time and memory to a verified strategy;
- strategy and compiled BDD/AIG sizes, reported separately from proof size;
- count correctness and proof-checking cost for a separate #QBF lane.

A DAG that reduces proof nodes but increases residual storage or checker work is not automatically better. Report the full resource vector and time-to-trusted-result.

### External comparison and limits

QBFGallery 2026 separates PCNF main, crafted, random, 2QBF, prenex non-CNF and DQBF tracks, and supports QDIMACS, QCIR and DQDIMACS benchmark inputs. The first external phase should use the public PCNF/crafted/random/2QBF corpora for which exact source files, licenses, solver versions and outcome artifacts can be pinned. Keep non-CNF and DQBF separate until their semantics are implemented and independently qualified.

CAQE is a strong practical QBF baseline: a 2026 proof-theoretic analysis describes it as the leading practical QBF algorithmic paradigm, and its public implementation supports QDIMACS plus a QDO output mode. DepQBF is an additional search-based baseline, with its exact binary and preprocessing configuration pinned. Do not invent a current ranking from solver availability or mix decision-only execution with strategy/proof output modes.

The QBFGallery page lists results as presented at SAT but does not expose a machine-readable score table in the retrieved page content. Numeric solved-count targets must wait until the official results and exact benchmark artifacts are pinned. Until then, use reproducibility and per-instance verified outcome tables rather than invented aggregate SOTA claims.

### #QBF boundary

The October 2026 knowledge-compilation paper treats #QBF as counting winning strategies and studies proof systems for this task. It is not the same as deciding whether a QBF is true. Any Symthaea #QBF prototype must count distinct Skolem strategy profiles under the exact dependency semantics, independently validate the count on finite instances, and compare with d4-QBF/Q-MICE-related references where compatible. Do not report #QBF values as QBF decision results or ordinary #SAT model counts.

## 15. Corrected official SAT Competition 2026 Main Track baseline

The current official [SAT Competition 2026 results table](https://github.com/satcompetition/2026/blob/main/results.html) reports these Main Track outcomes:

| Configuration | Total solved | SAT solved | UNSAT solved |
|---|---:|---:|---:|
| Official virtual best solver (VBS) | 347 / 400 | 168 | 179 |
| Best individual listed configuration, kissat-sc2024 | 306 / 400 | 153 | 153 |
| Kissat_MAB-DC | 290 / 400 | 146 | 144 |
| Kissat_MAB_ESA | 287 / 400 | 141 | 146 |
| hKis-bva | 285 / 400 | 143 | 142 |

The VBS is a per-instance hindsight oracle, not a deployable portfolio; its 41-instance gap over the best individual configuration is an upper-bound diagnostic, not a forecast of selector gain.

The competition website records an August 10, 2026 correction that added two initially omitted solvers and changed the third-place Main Track ranking. See the official [competition update](https://github.com/satcompetition/2026/blob/main/index.html) and [results table](https://github.com/satcompetition/2026/blob/main/results.html).

**Historical reconciliation:** the earlier instance-wise research note reported 276 / 269 individual configurations and 324 / 344 derived VBS values. Those figures are preserved as a historical local recomputation with a different extraction/selection basis. They must not be labelled the current official Main Track leaderboard. Before using them for head-to-head claims, regenerate the per-instance table from the corrected scores.csv and reconcile inclusion rules, solver configurations, track filters and timeout/score semantics.

### Symthaea competitive targets

Do not compare the QBF/strategy engine with SAT Main Track solved counts. For the SAT portfolio work, use the official table as the public baseline:
- reproduce 306/400 for the exact pinned kissat-sc2024 baseline before claiming parity;
- measure selector improvement against that pinned individual baseline under identical CPU, memory, timeout and per-instance scoring;
- report total, SAT, UNSAT, PAR-2/official score, family breakdown and selector overhead;
- require improvement on a sealed held-out set and never use the 347 VBS score as the practical target.

For QBF / #QBF, use QBFGallery 2026 and compatible #QBF corpora separately. The SAT score table is not a QBF benchmark.


## 22. Full-history game-value synthesis is a distinct result from winning strategies

The SAT 2026 paper *Proof Systems for QBF Synthesis: Extracting Skolem and Herbrand Functions* (Akshay et al., DOI: [10.4230/LIPIcs.SAT.2026.3](https://doi.org/10.4230/LIPIcs.SAT.2026.3)) makes a critical distinction:

- An existential/universal **winning strategy** only has to win from the initial position of a true/false QBF.
- A **Skolem/Herbrand function** is a full-history, win-whenever-possible function: it chooses correctly from every partial play where its player can still force the preferred result.
- A game-value function (g_i) maps every assignment to the first (i) prefix moves to the truth value of the remaining quantified suffix.

For a prefix variable (x_i), the paper derives the full-history function from the next game-value layer:
- if (Q_i = \exists), (f_i(x_{<i}) = g_i(x_{<i}, x_i=1));
- if (Q_i = \forall), (f_i(x_{<i}) = g_i(x_{<i}, x_i=0)).

These functions depend on **all preceding prefix variables**, not merely the opposing player's preceding variables. This is a different type from the existing opponent-only WinningStrategy interface. Keep both types, validators and claims separate.

### Current finite reference implementation

The new `quantified_game_values.rs` module extracts all game-value layers from the independently verified residual-CNF proof DAG. It emits complete full-history Skolem and Herbrand tables for every quantified variable, including both function families regardless of whether the root QBF is true or false.

Its independent verifier checks:
1. the final layer against the original CNF for every complete assignment;
2. every adjacent layer against the exact EXISTS/OR or FORALL/AND recurrence;
3. every full-history function's exact ordered prefix scope and every table row against the corresponding next-layer restriction;
4. the root layer against the declared truth value.

This is a semantic game-value profile and finite table extractor. It is **not** a SynQBF proof object or a full implementation of the paper's CEGC proof-construction algorithm. The paper's CEGC algorithm assumes an AllSATRep subroutine that returns a circuit for all satisfying assignments; implementation of that subroutine is explicitly outside the paper's scope.

### Required benchmarks

- Complete layer coverage by prefix position and quantifier alternation depth.
- Full-history function count, dependency width, truth-table bytes and ROBDD/AIG sizes.
- Independent recurrence-check time and source-matrix-check time.
- Proof-DAG nodes versus complete prefix-tree expansion and profile-table size.
- Difference between initial winning-strategy size and full-history function size.
- Held-out multi-variable formulas, quantified planning, synthesis and model-checking instances.

Do not report a full-history synthesis pass merely because the root QBF truth value and its one winning strategy validate. Every game-value layer and both player function families must be checked.


## 23. Portable SynQBF artifacts and QDIMACS-to-verifier vertical slice

The finite SynQBF proof is now a portable versioned artifact, not merely an in-memory AIG. Version 1 uses a deterministic little-endian format with:
- magic and explicit format version;
- variable count, claimed root truth value and exact ordered quantifier prefix;
- paired-variable dependency order and AIG decomposition order;
- tagged input/AND nodes, constants, complemented edges and output reference.

The artifact does not embed or replace the source CNF. Structural decoding is not semantic acceptance. The only acceptance path is decode followed by the independent three-obligation SynQBF verifier against the original CNF and validated prefix.

### Required artifact measurements

| Measure | Why it matters |
|---|---|
| Canonical byte length | Storage, transfer and auditability |
| Encode/decode time and peak memory | End-to-end artifact overhead |
| Decoder structural checks | Graph validity and malformed-input rejection |
| Source-equivalence assignments | First SynQBF obligation coverage |
| Representability conditions | Per-prefix local OR/AND obligations |
| Root-value check | Truth claim binding |
| AIG node visits / circuit evaluations | Deterministic verification work |
| Bytes per independently checked instance | Comparable evidence density |

Record format version, codec implementation revision, source QDIMACS digest, prefix digest, proof-circuit digest and verifier configuration. The proof artifact itself does not self-authenticate the source or certify its own truth. Bind those external digests using the existing benchmark manifest/receipt system.

### New finite vertical slice

The new test path composes:
1. bounded strict QDIMACS PCNF parsing;
2. residual-CNF QBF game proof-DAG construction;
3. independently checked game-value profile extraction;
4. finite SynQBF circuit construction and three-condition verification;
5. deterministic SynQBF proof encoding;
6. artifact decoding and independent verification against the original parsed CNF and prefix.

It includes an exists-x QDIMACS fixture. Separate two-variable construction tests cover 273 CNFs, two variable orders and all four quantifier patterns (2,184 formula/prefix cases), plus empty-prefix TRUE/FALSE constant proofs. These are finite exhaustive/reference checks, not external competition performance results.

### Adversarial artifact corpus

Add malformed-artifact fixtures for:
- unknown magic/version/tag and invalid booleans/quantifiers;
- duplicate/out-of-range prefix variables;
- wrong paired-input and decomposition orders;
- invalid references, non-topological edges, duplicate/unreachable AIG nodes;
- truncated records, trailing bytes and impossible count/byte combinations;
- byte, node, AND-node, variable, circuit-evaluation and verifier-work budget exhaustion;
- valid, structurally well-formed circuits bound to the wrong source CNF or prefix.

The decoder must reject impossible declared node counts before allocating node storage. Budget failures and malformed artifacts remain errors/UNKNOWN, never a QBF truth value.

### External QBF interface compatibility

QBFGallery 2026 decision-only execution is a distinct interface: its documented command interface accepts one input file, reports TRUE via exit code 10 and FALSE via 20; other return codes are failures. It has PCNF main, crafted, random, 2QBF, prenex non-CNF and DQBF tracks, with QDIMACS, QCIR and DQDIMACS input formats. The current parser supports bounded QDIMACS PCNF only. Do not run the PCNF parser on non-CNF/QCIR or DQBF inputs, and do not treat non-10/20 exits as truth values.

For every external comparison, report separate tracks for:
1. decision-only QBF;
2. proof production and independent proof checking;
3. Skolem/Herbrand or SynQBF proof extraction;
4. independent strategy/proof verification;
5. artifact size and serialization overhead.

A decision-only solver is not required to return a proof artifact; its solved count must not be compared to verified strategy coverage as if they were the same output contract.
