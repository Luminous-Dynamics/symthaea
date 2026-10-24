# QOBLIB-MIS-COHORT-001: pinned maximum-independent-set cohort

Status: **source inventory pinned; no solver run or performance result claimed**

## Immutable source

This cohort is pinned to QOBLIB commit d61f234b9fc816deb3d4e1dea4c031595b35acd7, whose website index reports a build time of 2026-10-19T09:44:36Z. The MIS directory tree at that commit has Git tree ID f2b6f6d6f7b705851b4f04253de1972b5930f568. The companion JSON lists the 50 .gph files, immutable raw URLs, Git blob SHA-1 object IDs, and byte sizes. It excludes the directory README from the graph count and does not vendor graph bytes.

The official website reports 1,264 instances across ten problem classes and identifies 50 instances for MIS, with declared sizes from 17 to 4,000 variables. Those aggregate site counts describe the snapshot; they are not head-to-head solver evidence.

## Format and feasibility

The pinned QOBLIB instance README specifies DIMACS edge format: lines beginning with c are comments; p edge n m declares node and edge counts; e u v specifies an undirected edge; nodes are one-based. The official checker verifies that a proposed set is feasible (no selected adjacent pair) and explicitly does not verify optimality. Feasibility evidence and the objective's optimal/best-known provenance must therefore remain separate.

**Important complement warning:** QOBLIB states that several original maximum-clique graphs have already been inverted into independent-set instances. Do not complement these graphs a second time. The JSON keeps the immutable source and the warning together; read the source README at the pinned commit when implementing the parser or QUBO encoding.

## Experimental design to run against this cohort

### Primary question

Does changing the reducer's information source improve independently verified final MIS quality or reduce all-in cost to a preregistered target, on the same graph instances, under the same stopping and candidate-generation rules?

### Required arms

1. Reducer-only baseline, without an informer.
2. Seeded random informer as a negative control.
3. Predeclared inexpensive graph-structural informer.
4. Strong classical local-search/population informer, charging all search and aggregation costs.
5. Classical solver or solver portfolio used as both solution baseline and cost baseline.
6. Exact-oracle signal on tiny fixtures only as a positive plumbing control.
7. Quantum-informed policy as a separate evidence track; distinguish replayed published outputs, circuit export, ideal simulation, approximate simulation and real hardware.

The experiment must isolate the information source as the intended treatment. Candidate pool, reducer version/configuration, tie-breaking, verifier, graph, repetition schedule, target rule, and stopping budget must otherwise be paired. A quantum policy compared only with random or with a weak single-run heuristic is not a meaningful SOTA comparison.

### Outcomes and cost boundary

Retain raw per-instance objective value, feasibility verdict, verifier output, target attainment, objective gap where an authoritative optimum/best-known value is available, and full resource vector. Record candidate observations, physical device shots and circuit batches separately. Include preprocessing, classical compute, compilation, calibration, queue and wall time without double-counting elapsed time. Report cost to target and probability of hitting that target, not just best sampled bitstring or informer accuracy.

A target must be defined before confirmatory evaluation for each graph from an authoritative optimum/best-known record or an explicit policy goal. The local checker does not establish optimality. Update result provenance if a result improves the current best-known bound.

### Pin comparison records separately from the graph cohort

The JSON manifest now pins three public comparison records to a separate QOBLIB snapshot, commit 1e0e7c523cd9707dd541cc9e5a3086125d4da3c7 (2026-10-19T09:52:56Z), including each submission README's Git blob ID. This keeps the graph-source snapshot and the evolving submission/best-known record snapshot independently auditable.

At this pinned submission snapshot, the classical ParallelILS README reports farm objective 10 (best-known), 10/10 successful trials, 0.0004 s average wall-clock time on an Apple M5 Max with 18 CPU cores. For es60fst02 it reports objective 88, 10/10, and 0.0006 s. The complete per-instance publication table is now frozen in [QOBLIB-MIS-BASELINES-001.json](QOBLIB-MIS-BASELINES-001.json), with its interpretation in [QOBLIB-MIS-BASELINES-001.md](QOBLIB-MIS-BASELINES-001.md). It joins all 50 ParallelILS rows to the official current best-known table from the same pinned QOBLIB commit. All 50 graph file Git blob IDs were checked identical between the graph-source snapshot and the newer published-record snapshot. The ParallelILS objective matches the official best-known value on all 50 instances; 38 are marked optimal and 12 best known. Its per-instance report records 487 successful trials out of 500 by the reported criterion, with a median across the 50 per-instance mean runtimes of 0.0009 s and a maximum of 120.05 s on brock800-1. These are external published records, not measurements produced by Symthaea.

The HSQC hybrid record reports objective 88, 5/5, and 2.5608 s total (2.4765 s QPU, 0.0843 s CPU). The QAOA record reports objective 88, 5/20, and 23,365.8542 s total (78 s QPU, 23,287.8542 s CPU).

These are source-pinned published records, not reproduced by Symthaea and not fair head-to-head timings. They use different hardware, algorithms, resource allocations and success conventions. Use them as comparison context and as motivation for a controlled experiment, not as a causal claim that one paradigm is faster.

### Independently rechecked public solution artifacts

The machine-readable [QOBLIB-MIS-REFERENCE-AUDIT-001.json](QOBLIB-MIS-REFERENCE-AUDIT-001.json) pins the graph and three published solution files for es60fst02 to QOBLIB commit 1e0e7c523cd9707dd541cc9e5a3086125d4da3c7, recording the file-level Git object IDs.

A separate JavaScript source parser and edge checker inspected all three submitted 88-vertex solution sets against the exact 186-vertex, 280-edge graph. ParallelILS, the HSQC hybrid submission, and the QAOA submission each had 88 unique in-range vertices and zero selected-edge conflicts. This establishes **feasibility of the submitted lists only**; it does not prove that 88 is globally optimal, validate the reported repeated-run success rates, reproduce any algorithm, or measure runtime. The QOBLIB record explicitly leaves the quantum submissions' optimality bounds as N/A.

This audit was executed during authoring but the one-off JavaScript checker is not a committed standalone tool. The new `--verify-solution` mode in the Symthaea smoke runner (PR #8324) makes the same class of feasibility check repeatable from the repository when Rust becomes executable: it emits an audit receipt and exits nonzero on an invalid candidate or mismatched reported objective. That Rust path remains authored but uncompiled in this environment.

### Report both QOBLIB-native and all-in cost

QOBLIB's current contributor guide specifies that submitted runtime fields should exclude compilation and queue time, while reporting total runtime, time-to-solution and CPU/GPU/QPU/other-hardware runtime. For apples-to-apples comparisons with a QOBLIB row, reproduce those conventions and report the hardware and averaging method. Do not add local stage timers to a published QOBLIB total unless their definitions match.

For Symthaea's decision-value question, also retain an explicitly separate all-in operational account that can include compile/transpile, calibration, queueing, data transfer, preprocessing, classical inference, retries, and end-to-end latency. Preserve each component separately and do not sum overlapping component durations with end-to-end wall time. This internal all-in account is a different metric from the QOBLIB-native submission field, not a retroactive correction to the public leaderboard.

Reference records are pinned separately to QOBLIB result snapshot commit 1e0e7c523cd9707dd541cc9e5a3086125d4da3c7 (2026-10-19T09:52:56Z), and are preserved in the companion JSON with file-level Git blob IDs. The [ParallelILS result table](https://github.com/ZIB-AOPT/QOBLIB/blob/1e0e7c523cd9707dd541cc9e5a3086125d4da3c7/07-independentset/submissions/20260806_ParallelILS_Hegade/README.md) reports best-known objective 10 for farm over 10/10 trials with 0.0004 s total runtime on an Apple M5 Max with 18 CPU cores; on es60fst02 it reports objective 88, 10/10 successful trials and 0.0006 s. The [HSQC hybrid record](https://github.com/ZIB-AOPT/QOBLIB/blob/1e0e7c523cd9707dd541cc9e5a3086125d4da3c7/07-independentset/submissions/20260804_HSQC_Hegade/es60fst02/README.md) reports objective 88, 5/5 successful trials and 2.5608 s total including 2.4765 s QPU time. The [QAOA hardware record](https://github.com/ZIB-AOPT/QOBLIB/blob/1e0e7c523cd9707dd541cc9e5a3086125d4da3c7/07-independentset/submissions/20260907_QAOA_Vivek/es60fst02/README.md) reports objective 88, 5 successful runs out of 20 and 23,365.8542 s total, with 78 s QPU and 23,287.8542 s CPU. These published records use different hardware and experimental protocols, so they are **not controlled head-to-head timings**; the point is to preserve the real cost boundary and strong classical reference, not claim a speedup by matching objective alone.

### Current frontier: two distinct literature-reproduction tracks

A paper accepted by *Physical Review Research* on 19 October 2026, *Scalable quantum-enhanced greedy algorithm for maximum independent set problems on random regular graphs* (DOI [10.1103/ds1n-wsdm](https://doi.org/10.1103/ds1n-wsdm), [preprint arXiv:2601.21923](https://arxiv.org/abs/2601.21923)), combines precomputed tree-derived QAOA local expectation values with sequential minimum-degree greedy decisions. It reports a 20-qubit IQM hardware implementation and tensor-network simulation on larger graphs. The study uses random 3-regular graphs, not QOBLIB's named benchmark instances.

**Important baseline correction:** for regular graphs, bare minimum-degree greedy is not the classical SOTA comparator. The paper explicitly compares with Marino and Kirkpatrick's *linear-prioritized local search* for random d-regular graphs ([arXiv:2003.12293](https://arxiv.org/abs/2003.12293)) in addition to ordinary greedy. The prioritized method uses contractions/virtual nodes to defer decisions and is tailored to this graph distribution. Symthaea must include it before interpreting a quantum-informed increase as SOTA on random regular graphs.

The accepted-paper text has an important sample-count inconsistency: Section IV says it generated 10^5 random 3-regular instances per system size for the SOTA comparison, while the Figure 3 caption reports 200 graph instances for each QGreedy depth curve (100 for the p=3 hardware series) and lists N=50, 100, 200, 500, 1000, 2000 and 5000. Treat the exact sample count for each reported result as **unresolved** until official raw data, supplementary materials or author clarification determine which cohort corresponds to which curve. Do not silently select one count as the reproduction target. The article reports significant p=4 gains over the prioritized baseline for considered sizes up to N <= 2000 and discusses p=4 results through about N = 5000. These are **external published claims**, not Symthaea results. The companion manifest records this discrepancy and requires a per-curve sample count to be frozen before a replication.

An official source repository for the October 2026 accepted paper was not found in GitHub repository/code searches performed on 24 October 2026. That does not prove no implementation exists elsewhere. Until official code is located, a replication must be labelled either blocked on source recovery or as an independent implementation with its own commit, exact fixed-angle table, tests and independent graph generator. Do not call it an exact reproduction without the code/configuration match.

A second paper, qReduMIS (*Physical Review Research*, published 10 September 2026; DOI [10.1103/nh3b-1wv5]), uses quantum information to guide classical MIS reductions rather than expecting a QPU to solve the whole input graph. Its pinned open-source repository and source-artifact gaps are recorded separately under **QREDUMIS-UJ-REPRO-001**. That study's random unit-disk / Union-Jack graph distribution is also distinct from random regular graphs and from QOBLIB.

Required arms for the random-regular QAOA track:
1. Ordinary minimum-degree greedy with randomized tie selection.
2. Deterministic/index-tie minimum-degree greedy as a control.
3. Marino–Kirkpatrick linear-prioritized search with its contraction, tie and stopping rules pinned.
4. The QAOA-informed choice among eligible minimum-degree candidates using frozen tree-derived angles.
5. A matched negative control replacing QAOA-informed tie choices with random choices under the same candidate/decision budget.

Freeze graph-generation version, degree, seeds, size grid, QAOA depth/angle source, shots, estimator, cache/recalculation policy, stopping rules and primary metric before confirmatory runs. Report mean independent-set ratio |I|/|V| and per-instance paired deltas by graph size, uncertainty and multiplicity correction, feasibility/failure rates, and full resource use. Track circuit batches, shots, cached/local expectation recalculations, classical graph updates, compilation/calibration if charged, CPU/QPU time and end-to-end latency. Keep physical hardware, ideal simulation, tensor-network simulation and published-output replay in separate evidence classes. Use fixed-target hit probability/time-to-target only as a preregistered secondary measure.

The method for exceeding SOTA is to reproduce the literature's actual workload and strong classical baseline first, then test a new Symthaea informer on fresh held-out seeds and unseen sizes. Neither the QOBLIB cohort nor results from random unit-disk graphs can substitute for this separate random-regular comparison, and no universal quantum-advantage claim follows from this track alone.

### Stratification and claims

- Report results by source family and graph size/density; set strata before seeing confirmatory outcomes.
- Separate model development/tuning from confirmatory comparisons. Since benchmark instances may already have appeared in prior research, claims of out-of-distribution generalization require additional held-out graph families.
- Run the checker independently on every returned set. Preserve infeasible runs, timeouts, abstentions and failures.
- Pair statistical units at graph-instance level. Repeated seeds within one graph are nested repeats, not independent new graph instances.
- Submit useful positive or negative results to QOBLIB only under its source-specific contribution rules and licenses.
- Never infer quantum advantage from website totals, solver-reported claims, a simulation, a lower local score, or a circuit that was exported but not executed.

## Frozen provisional family partition

The companion JSON now assigns every pinned instance to a deterministic, family-held-out development partition: 27 train, 9 validation and 14 test instances, across 18 named source groups. For the test split, all BROCK graphs, the C-series graphs, and all FRB graphs stay together by group. C-series membership is not itself a claim that every instance was complemented; apply the upstream inversion warning only to its enumerated instances. The validation split holds out all Hamming, Insecta and Sloane-named instances. All other source groups are assigned to train. Per-instance assignments and the family-to-split map are recorded in the JSON manifest.

This is a *provisional source-name family grouping*, not proof that each group is an independent graph-generation distribution. It is also not a blinded test set: QOBLIB graphs and best-known values are public and prior research may already have influenced development. Use the partition to prevent obvious within-family leakage during selector tuning; claims of out-of-distribution generalization still need newly generated or independently held-out graph families. A changed split must use a new partition ID rather than silently rewriting this one.

## Separate qReduMIS reproduction track

QOBLIB gives Symthaea broad heterogeneous graphs and published solver records. The qReduMIS paper answers a different question on hardware-native random unit-disk instances with Union-Jack connectivity. The pinned open-source results README specifies radius sqrt(2), density 0.8, varied grid length/seed, QAA, qReduMIS and randomReduMIS; it says backend runs use 1,000 shots and the same annealing schedule, with common postprocessing plus loading-error correction for hardware. The published paper also discusses classical simulated annealing. Capture exact graph bytes/coordinates, seeds, reducer rules, policy configuration, raw shots/results, cost boundaries and software environment before replication.

The public qReduMIS tree is Apache-2.0 and pinned in the companion JSON, along with blob IDs for its two result notebooks and README. It is suitable as a paper-reproduction reference, not as an automatic dependency for Symthaea's Rust production path. One reproducibility gap is recorded explicitly: the pinned result README mentions generator/generate.py, but that path is absent from the pinned tree inspected for this manifest. No generator reconstruction or paper result is claimed here.

## Current state-of-the-art comparison implication (reviewed 2026-10-24)

The 10 September 2026 *Physical Review Research* qReduMIS paper is an important comparator for this project: [Schuetz et al., DOI 10.1103/nh3b-1wv5](https://doi.org/10.1103/nh3b-1wv5). It uses a quantum coprocessor to identify likely frozen vertices inside a repeated exact classical-kernelization loop, rather than asking the QPU to solve the full graph directly. Its evaluation includes up to 231 qubits on Rydberg hardware and 39 random union-jack instances.

Most importantly for experimental design, the paper also reports a *SaReduMIS* arm: the same broader reduction framework powered by classical simulated annealing. With 500 replicas and two annealing steps (a stated total computational budget of 1000, on par with the quantum component budget), that classical variant achieved reported success probability 1 on all 39 union-jack instances in the paper's test bed. The authors therefore explicitly call for future comparisons of qReduMIS against classical ReduMIS and acknowledge the importance of the classical reduction wrapper.

This does not establish SaReduMIS beats qReduMIS in all regimes, and it is not a result on the QOBLIB graphs. It does establish a necessary standard for Symthaea: **the primary causal ablation must keep the reduction wrapper fixed and change only the information/inner-solver source**. A qReduMIS-style arm compared only with standalone QAOA, random picking, or an underpowered heuristic is insufficient.

### Mandatory matched-information ablation

| Arm | Same reduction rules? | Inner signal / solver | Cost that must be charged |
|---|---|---|---|
| A | Yes | No informer / deterministic reduction | Reduction and reconstruction compute |
| B | Yes | Seeded random frozen-node signal | Signal generation and all reduction work |
| C | Yes | Degree / local graph-structure signal | Feature extraction plus reduction work |
| D | Yes | Classical simulated annealing | All replicas, annealing, CPU/GPU time and overhead |
| E | Yes | Classical ILS / population portfolio | Full portfolio compute, worker count and synchronization |
| F | Yes | Quantum-derived marginal/frozen-node signal | Preprocessing, compilation, queue, calibration, QPU shots/batches, failed attempts, classical post-processing and all reduction work |

For every arm, freeze the original graph digest, reduction operators/order, kernel policy, candidate budget, stopping condition, solution reconstruction and independent verifier. Pair by graph and predeclared replicate; count repeated trials within graph as clustered repeats. Report target-hit probability and cost-to-target curves, not just best objective and not just average final runtime.

### Two distinct benchmark tracks

1. **External portfolio benchmark:** the pinned 50-instance QOBLIB MIS cohort in the companion JSON; report QOBLIB-native fields and preserve the upstream feasibility/optimality distinction.
2. **Reduction-mechanism benchmark:** a separate random unit-disk / union-jack cohort with frozen generation seeds, hardness strata, and a same-wrapper SaReduMIS/ReduMIS-style classical baseline. Do not pool these instances with QOBLIB as though they came from the same distribution.

For both tracks, keep development choices separate from confirmatory graph families. The public QOBLIB train/validation/test split is only a visible family-leakage control, not a blinded test set. Newly generated seeds and independent graph families are needed for an actual held-out generalization claim.

## Next executable gates

1. Add a strict DIMACS parser test suite for comments, duplicate edges, self-loops, malformed headers, count mismatch, out-of-range vertices, and the large C4000 file.
2. Independently validate graph and solution parsing against QOBLIB fixtures and the pinned checker behavior.
3. Establish a deterministic baseline report on tiny exact instances and then on a frozen small subset; record content IDs and resource counters.
4. Run the complete 50-instance cohort only after the parser/verifier and cost accounting are validated.
5. Run a separate **QREDUMIS-UJ-REPRO-001** track against the paper's random unit-disk workload with Union-Jack connectivity (radius sqrt(2), density 0.8), pinned to the open-source [qReduMIS repository at commit 43feb7502ca4c90e10bde24e077c1d54427f8fef](https://github.com/jpmorganchase/qReduMIS/tree/43feb7502ca4c90e10bde24e077c1d54427f8fef). Use its QAA, qReduMIS and randomReduMIS arms as references, then add matched classical structural and solver-population informers. Upstream describes 20 classical shots for the qReduMIS policy and 1,000 backend shots per execution, with a shared annealing schedule; pin exact configs and raw graph artifacts before claiming reproduction. The upstream instructions include Python 3.9 and a custom simulator branch, so isolate paper reproduction from the Rust-first Symthaea production path. The pinned results README refers to generator/generate.py, but that file was not present in the inspected pinned tree; locate its exact source or freeze graph-coordinate bytes and seeds independently before regeneration. Keep this track completely separate from QOBLIB's heterogeneous graph cohort.

## Validation boundary

This contribution only inventories the immutable source cohort. It does not parse graph bytes, run a solver, reproduce QOBLIB submissions, evaluate a QPU, or provide evidence of a quantum advantage. The pinned Git object IDs are Git SHA-1 object identifiers, not SHA-256 checksums of raw files.
