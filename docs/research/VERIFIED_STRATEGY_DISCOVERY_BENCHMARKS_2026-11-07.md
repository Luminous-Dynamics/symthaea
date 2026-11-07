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
