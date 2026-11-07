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
