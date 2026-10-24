# QCOMP-COST-RISK-001: Cost-sensitive evaluation of quantum-informed optimization

Status: **protocol draft; no performance result claimed**

## Research question

When does an information source improve the final verified optimization outcome enough to justify its all-in cost?

For maximum independent set (MIS), a good local score is not itself a good final solution. The primary causal comparison holds instance, candidate pool, reducer, seeds, verifier, and stopping/quality target fixed while changing only the information source. A quantum-derived score must be compared with both a reducer-only arm and a strong matched-budget classical informer.

This protocol develops the objective in issue [#8403](https://github.com/Luminous-Dynamics/symthaea/issues/8403), alongside the signal work in [#8348](https://github.com/Luminous-Dynamics/symthaea/pull/8348), batch accounting in [#8387](https://github.com/Luminous-Dynamics/symthaea/pull/8387), and graph-level oracle/verifier work in [#8324](https://github.com/Luminous-Dynamics/symthaea/pull/8324).

## Paradigm shift: decision value per operational cost

Report the full outcome/cost vector before reducing it to any scalar:
- independently verified feasibility and final solution objective;
- normalized loss under a definition frozen before confirmatory evaluation;
- attainment of a predeclared solution-quality target;
- candidate observations, unique physical shots, and circuit batches;
- compilation, calibration, preprocessing, classical execution, queue and end-to-end wall time;
- energy and direct monetary cost only when recorded, with units and measurement method.

Candidate observations are not automatically physical QPU shots. A single bitstring may update multiple observables, and separate circuits can have different compilation/setup costs. The joint sampling model, batch count and cost boundary must be explicit. Never add wall time to its constituent stage times in a scalar risk.

For a predeclared coefficient vector, define

    risk = quality_weight * normalized_quality_loss
         + infeasible_penalty * 1[not feasible]
         + target_miss_penalty * 1[target not reached]
         + sum(weight_j * observed_cost_j / reference_amount_j)

All weights and reference amounts must be fixed before the confirmatory cohort is run. Report the raw dimensions and Pareto frontier as well: one scalar risk is decision-specific, not a universal ranking. Repeat for several predeclared cost preferences rather than choosing one after observing results.

## Implemented pure-Rust primitive

The files crates/domains/symthaea-quantum-comp/src/mis_cost_risk.rs and mis_pareto.rs add:
- typed cost dimensions, optional energy/money fields, explicit evidence classes (synthetic arms, classical replay/solver, ideal simulator, tensor-network simulator, circuit export, physical QPU), and required evidence references;
- scale-normalized, preregistered cost-risk calculation with explicit quality, feasibility and target-miss terms;
- paired method comparison using shared instance/repetition keys plus an explicit workload identity binding cohort, graph digest, repetition seed, reducer configuration, candidate pool and verifier version; mismatches fail closed;
- fail-closed checks for duplicate/missing comparison keys, mixed policy IDs, any unrecorded configured cost dimension (including ordinary counters), invalid currency metadata, duplicated weights, invalid scales, and wall-clock/component-time double counting;
- a separate Pareto-frontier function that retains non-dominated policies over normalized quality loss, infeasibility rate, target-miss rate and selected shared-scale cost dimensions, without collapsing trade-offs into one weighted sum; each point retains its evidence class;
- each run requires an evidence reference, each policy's records must use one evidence class, and the paired comparison returns the evidence class of each arm. This prevents one policy leaderboard row from silently mixing synthetic, simulated, and physical-QPU runs.

This module scores caller-supplied run records. It does not measure hardware, independently verify solutions, choose an appropriate normalized-loss function, establish provenance, or certify the validity of a selected weight vector. The caller must keep the configuration and outcome normalization fixed and preserve raw records.

### Pareto-frontier stability across graph resamples

The new module `mis_pareto_stability` complements the point-estimate frontier with a graph-cluster bootstrap. Inputs must contain exactly one pre-aggregated `MisParetoPoint` per policy per graph cluster, with a shared policy set, cohort identity, normalization and homogeneous evidence class for every policy. The function resamples graph clusters with replacement, recomputes equal-graph-weighted policy aggregates and frontier membership, and reports each policy's membership count and frequency next to the original-cohort point estimate.

Frontier membership frequency is a **stability diagnostic**, not a posterior probability or confidence that a policy is truly Pareto-optimal. Aggregate repeated trials within each graph using a frozen rule before supplying the per-graph point. If the graph set contains strong within-family dependence, graph-level resampling may still overstate independence; use family-level clusters or a hierarchical model when the sampling design warrants it. A frontier of point estimates is not itself a confidence region.

The helper fails closed on missing/duplicate policy rows, inconsistent policy sets, empty cluster/cohort IDs, mixed evidence classes, invalid point/cost records, cross-cluster currency changes and zero bootstrap replicates. It does not correct multiple comparisons or select normalization scales.



## Study design

### Arm set

1. Reducer only, with no informer.
2. Deterministic random signal as a negative/control arm.
3. Structural classical signals (degree, neighborhood and other inexpensive predeclared graph features).
4. Population frequencies generated by a pinned classical local-search portfolio, including all portfolio cost.
5. Signal from a strong classical solver ensemble, including all ensemble cost.
6. Exact oracle signal on tiny graphs only as a positive plumbing control.
7. Published qReduMIS reproduction cohort and the fixed-angle QAOA-informed greedy cohort as distinct tracks; label source data, tensor-network estimates, simulator output and physical-QPU records separately.

Use the same graph, candidate pool, reducer, decision count, seeds, verifier and cost boundary for each paired comparison. Keep paper reproduction separate from generalization.

### Instance cohorts and baselines

- Tiny hand-checkable fixtures with an exact optimum.
- A preregistered, source-hashed QOBLIB MIS cohort.
- Separate random-regular, native unit-disk and general-graph cohorts.
- Difficulty strata established against a pinned portfolio of strong classical MIS solvers, not selected after quantum results are inspected.

QOBLIB is a public community benchmark spanning ten optimization classes and accepts positive, negative, classical and quantum submissions. It is the appropriate shared comparison track; every dataset snapshot and instance identity should be pinned in the experiment manifest.

### Statistical analysis

- Pair outcomes by instance digest and replicate seed.
- Report paired quality delta, failure/feasibility rate, target attainment, risk delta, raw per-instance data and uncertainty intervals.
- Separate fixed-budget best-arm identification, fixed-confidence best-arm identification, anytime best-arm identification and good-arm threshold objectives.
- Include uniform, Successive Rejects, Continuous Rejects, the current conservative adaptive planner and Almost Tracking only under their matching sampling contracts.
- For the actual graph objective, also compare against reducer-only and classical informer arms. Arm accuracy is not a substitute for graph-level quality.
- Preserve null results, infeasible outputs, abstentions and backend runs that did not actually call a quantum device.

## Recent baseline obligations (October 2026)

The accepted paper by Wybo et al., *A scalable quantum-enhanced greedy algorithm for maximum independent set problems* (Physical Review Research, accepted 19 October 2026; https://doi.org/10.1103/ds1n-wsdm; preprint https://arxiv.org/abs/2601.21923), is a distinct random-regular-graph track. It reports fixed-angle QAOA expectation-guided greedy selection, experiments on a 20-qubit IQM device, tensor-network simulation, and comparison against minimal-degree greedy plus Marino–Kirkpatrick prioritized search. Its main QGreedy analyses use 200 instances at N = 50..5000; its p=4 prioritized-search comparison reports a statistically significant improvement for N <= 2000 on a 100,000-instance-per-size cohort. These are the paper's reported results, not Symthaea results.

The Symthaea benchmark should reproduce the random 3-regular task separately, using identical generated graph files for all arms. The new [random-regular protocol](https://github.com/Luminous-Dynamics/symthaea/blob/research/qcomp-portfolio-001a-mis-verifier-current/docs/research/quantum/benchmarks/QCOMP-RRG-001.md) requires the paper's named prioritized-search baseline, not only greedy. Do not pool it with QOBLIB's fixed named graph cohort or with qReduMIS's quantum-informed kernelization study.

A July 2026 cross-family hardware benchmark, *From Circuits to Hardware* (https://arxiv.org/abs/2607.11637), reports a matched uniform-random control and finds that many feasible low-fidelity outputs in its tested gate-based configurations fall within random-control performance. That reinforces these minimum controls: random/permuted signals, post-transpilation circuit size/depth, shot/batch counts, backend calibration, output feasibility, and full preprocessing/postprocessing resource accounting. Its findings are bounded to the tested implementations; do not generalize them to all QAOA variants.

## Decision gates

1. **Integrity:** tiny exact oracle and independent feasible-set verifier pass; fixed seeds replay deterministically.
2. **Cost integrity:** no missing configured metric defaults to zero; physical shots, candidate observations and circuit batches remain separately observable.
3. **Fair comparison:** matched instances, paired keys, pinned solvers, fixed quality target and preregistered cost configuration.
4. **Generalization:** report results by held-out family and hardness stratum, including failures and uncertainty.
5. **Advantage claim:** no quantum advantage from synthetic independent-arm accuracy, a surrogate, a topology cache hit, a simulated result or a better local score. Require graph-level outcome improvement or lower all-in cost to the same target, measured under matched rules and independently checkable evidence.

## Proposed aspirational target

Target at least a 2x reduction in end-to-end cost to a fixed, preregistered solution-quality target at non-inferior verified solution quality on held-out families. This is an aspiration, not a current result or an industry standard. A statistically supported quality improvement at matched all-in cost is also useful. Negative or inconclusive outcomes are valuable research outcomes when reproducible.

## Validation boundary

Twelve unit tests are authored in the cost-risk module and four in the Pareto module. At the time of this commit they have not been confirmed compiled or executed. The existing paired-difference helper uses a compact normal-approximation interval; its output is a local diagnostic, not publication-grade statistical inference. Publication-grade reports should include appropriate paired bootstrap/randomization methods and multiple-comparison handling. The cost and frontier-stability modules do not contain or claim any measurements. Pareto stability has four authored tests for deterministic replay, stable dominated-policy exclusion, graph-composition sensitivity, mismatched/duplicate policy rows, mixed evidence and cross-cluster currency; these remain uncompiled and unexecuted in this environment. An independent JavaScript semantic reference reproduces the composition-sensitive fixture: both policies are on the full-cohort point frontier, while their bootstrap membership frequencies are approximately 0.767 and 0.758 across 1,000 graph-cluster resamples. This is a reference calculation, not Rust execution.

## Related prior art

- Wang, Tzeng & Proutiere, *Best Arm Identification with Fixed Budget: A Large Deviation Perspective*, NeurIPS 2023, Continuous Rejects: https://proceedings.neurips.cc/paper_files/paper/2023/hash/364d565b4b726c607aa40e1632045873-Abstract-Conference.html
- Komiyama, Jang & Honda, *Rate-optimal Design for Anytime Best Arm Identification*, AISTATS 2026, Almost Tracking: https://proceedings.mlr.press/v300/komiyama26a.html
- Harding & Kandasamy, *Balancing Performance and Costs in Best Arm Identification*, NeurIPS 2025, DBCARE: https://papers.nips.cc/paper_files/paper/2025/hash/b0dfe8236981ba84b4d98a7ed0426f0b-Abstract-Conference.html
- Schuetz et al., *Quantum-informed reduction algorithm for the maximum independent set problem*, accepted Physical Review Research paper: https://doi.org/10.1103/nh3b-1wv5
- Wybo et al., *A scalable quantum-enhanced greedy algorithm for maximum independent set problems on random regular graphs*, accepted Physical Review Research paper: https://doi.org/10.1103/ds1n-wsdm
- QOBLIB: https://zib-aopt.github.io/QOBLIB/

These methods use distinct workloads, assumptions and metrics; this protocol does not combine their results into a common SOTA claim without a matched comparison.
