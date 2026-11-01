# Unified Temporal Resource Control and Qualification Protocol v0.1

**Tracking:** NEURO-SILICON-009 / #9666  
**Implementation branch:** `research/temporal-joint-resource-control-20261101`  
**Status:** candidate allocator and evaluator exact-oracle source authored; no Rust compilation, tests, benchmark runs, hardware measurements or security deployment are claimed.

## 1. Research thesis

The system should not independently optimize what to remember, when to observe, when to compute and when to verify. Those choices are coupled: a new observation may make an old memory obsolete; retaining a memory can make a later forecast cheaper; a compute refinement may be worthless without a fresh sample; verification can prevent a high-cost wrong action but consumes time, energy and receipt bandwidth.

The target contribution is therefore **closed-loop temporal resource control**: predict the value of a complete joint action, enforce non-fungible resource limits, preserve the exposure trade-off surface, and measure realized effects with an evaluator-only oracle.

This is more precise than claiming that a larger liquid neural state is intrinsically more intelligent. The architecture's advantage must appear as improved decisions under the same information and resource budgets.

## 2. Four coupled decision families

| Decision | Candidate action families | Principal value signal |
|---|---|---|
| Sensing | observe now, defer, periodic, event-triggered, multi-rate | expected reduction in future exposure from new evidence |
| Memory | retain, evict, compress, refresh, invalidate | expected reduction in future exposure over retained byte-time |
| Computation | skip, update, refine, forecast, verify a computation | expected decision improvement from additional compute |
| Evidence verification | local consistency check, signature/receipt check, witness quorum check | expected reduction in undetected evidence or state-integrity failure |

Verification is deliberately treated as a resource-consuming action, not an invisible evaluator freebie. The final benchmark should report its latency, operations, energy and receipt bytes.

## 3. Candidate and evaluator boundary

### Candidate-visible inputs

- current observations and their event-time / arrival-time metadata;
- candidate-owned memory and the candidate's own state;
- public task schema, resource ceilings and allowed action families;
- the candidate's predictions, calibration metadata and estimated complete-plan costs;
- previously accepted public receipts and non-secret checkpoint status.

### Evaluator-only inputs

- hidden future stream and event intervals;
- exact ground truth and realized wrong-belief, abstention and missing-output exposure;
- evaluator-only alias probes and hidden event boundaries;
- paired intervention assignments and exact small-instance joint oracle;
- evaluator seeds, signing keys and witness private keys.

The candidate cannot observe any field whose availability timestamp is after the decision origin. A future covariate is allowed only when its known-at-origin provenance is explicit. Event time and arrival/availability time must remain separate throughout ingestion, storage and scoring.

## 4. Implemented reference in this branch

### Candidate allocator

`research/temporal-joint-control` defines complete joint plan alternatives. Each plan has three separate predicted exposure reductions and twelve resource estimates:

1. persistent bytes;
2. retained byte-microseconds;
3. read bytes;
4. write bytes;
5. compute microseconds;
6. refresh operations;
7. observations;
8. sensor-active microseconds;
9. end-to-end latency microseconds;
10. energy nanojoules;
11. verification operations;
12. receipt bytes.

The allocator first applies hard resource and maximum-gap limits, then returns the Pareto frontier over the three exposure axes and all resource dimensions. It uses an explicit caller-supplied exposure priority for the final deterministic choice. It never converts an infeasible plan into a feasible one through a weighted penalty.

The allocator accepts complete-plan estimates. Therefore, it can represent non-additive interactions: the predicted value and cost of an observe+retain+compute+verify plan need not equal the sum of four isolated estimates.

### Evaluator exact oracle

`research/temporal-joint-oracle` is a separate evaluator-side package. For up to 16 binary action factors, it requires all (2^n) combinations exactly once, checks identical episode commitments, enforces measured resource ceilings, and reports componentwise regret plus the full non-dominated frontier.

A four-factor fixture enumerates all 16 sensing/memory/compute/verification combinations. This captures all pairwise, three-way and four-way interactions for that finite action set. It does not establish global optimality over continuous policies, arbitrary action spaces or unenumerated horizons.

## 5. The paradigm shift: measure the value of not remaining ignorant

The key quantity is not merely prediction accuracy. It is the *marginal reduction in future decision exposure* produced by information acquisition, retention, computation or verification, after charging the complete resource cost.

A future value estimator should predict three effects separately:

`DeltaV(a) = (DeltaW(a), DeltaA(a), DeltaO(a))`

where `W` is wrong-belief exposure, `A` is abstention exposure, and `O` is missing-output exposure. These are not interchangeable. A system must not mask a large regression in one axis with a gain in another.

Resource costs remain a vector:

`C(a) = (M, M_t, R, W, C_t, F, N_s, T_s, L, E, V, B_r)`

with the twelve dimensions defined above. Feasibility is componentwise: `C_i(a) <= B_i` for every resource dimension `i`.

Only after feasibility should the controller compare the three-axis benefit and twelve-axis cost Pareto surface. Future adaptive versions may use explicit per-resource shadow prices or a receding-horizon controller, but hard limits remain separate and every conversion/price must be recorded.

The main open research hypothesis is that a shared estimator of marginal future exposure reduction will outperform local memory, sensing, compute and verification heuristics on interaction-heavy tasks without regressing on null tasks.

## 6. Benchmark design

### Task strata

| Stratum | What it tests | Essential control |
|---|---|---|
| Null-value | sampling/retention/verification are unnecessary | extra actions must not be rewarded |
| Transient and aliasing | events occur between samples | exact alias-pair schedules |
| Irregular cadence | bursts, long gaps, missingness, jitter | event-time/arrival-time separation |
| Informative missingness | observation probability depends on latent state | missingness mechanism held out |
| Abrupt and gradual drift | update timing and stale beliefs | unseen shift times and magnitudes |
| Recurring regimes | useful old knowledge returns | recurrence with distractor periods |
| Retraction / correction | old evidence becomes invalid | hard invalidation beats memory priority |
| Asynchronous multimodal | modalities have different clocks and causes | modality-specific masks and availability times |
| Adversarial evidence | forged, replayed, stale or conflicting receipts | invalid-signature and rollback negative controls |
| Tight physical budgets | actual memory/energy/latency trade-offs | end-to-end measurement, not operation counts alone |

### Competitor matrix

| Research question | Required baselines |
|---|---|
| Continual retention | no memory; LRU/LFU/TTL; temporal-value-per-byte-time; reservoir/replay methods; BrainTrace; BPTT; GRU/LSTM/GRU-D |
| Active sensing | fixed periodic; random; event-triggered; uncertainty-triggered; information-gain/value-of-information policy; APN time-aware adaptive patching as a forecasting-side comparator |
| Irregular forecasting | interpolation/last-value; GRU-D; mTAND; GraFITi; tPatchGNN; HyperIMTS; QuITE++; Warpformer; NeuralFlow; CRU/Latent-ODE; APN; KAFNet |
| Multi-timescale state | LIF/ALIF; STDP; PMSN; GRU/continuous-time RNN; HDC-LTC |
| Joint allocation | fixed quotas; each subsystem alone; independently tuned subsystems; greedy one-resource-at-a-time; Pareto joint controller; explicit primal-dual controller; receding-horizon joint controller; evaluator-only exact joint oracle |

The model/forecasting rows and the action-allocation rows answer different questions. Forecasting accuracy cannot substitute for measuring whether the system chose the right time to observe, retain or verify.

## 7. External benchmark and industry alignment (verified 1 November 2026)

### Irregular time series

- **APN (AAAI-26)** is a serious efficiency baseline, not an optional footnote. It learns time-aware patch boundaries/aggregation and uses a shallow prediction head; its authors report improved accuracy and efficiency across real datasets. Include its official implementation and identical data splits where feasible. <https://doi.org/10.1609/aaai.v40i28.39563>
- **KAFNet (AAAI-26)** revisits canonical pre-alignment with temporal kernel aggregation and frequency-domain inter-series attention. Its authors report 7.2x fewer parameters and 8.4x training/inference acceleration versus their comparison setting. Reproduce the paper's exact task and protocol before using those ratios as an expected target. <https://doi.org/10.1609/aaai.v40i34.40149>
- **Time-IMM / IMM-TSF (NeurIPS 2025 Datasets & Benchmarks)** is particularly relevant because it represents nine irregularity types across trigger-, constraint- and artifact-based mechanisms and supports asynchronous multimodal forecasting. Use it as a real-data validation set, not as a substitute for the evaluator's hidden event-time controls. <https://proceedings.neurips.cc/paper_files/paper/2025/hash/4199594d3c15736df2bf5274fa3155f4-Abstract-Datasets_and_Benchmarks_Track.html>
- A public USHCN m=24 to horizon=3 leaderboard currently lists QuITE++ at 4.84 MSE, HyperIMTS at 4.96 and tPatchGNN at 5.00. Treat this as a lead for baseline reproduction, not an authoritative universal SOTA claim: verify dataset preprocessing, split, horizon, metric scaling and run provenance locally before comparison. <https://www.sota2.com/research/sota/irregular-time-series-forecasting-on-ushcn-m-24-3-horizon>

### Continual learning

**Continual Learning Bench 1.0** measures expert-validated task sequences where systems are expected to change through experience, and reports gain relative to a stateless baseline. It has six domains, including software engineering, signal processing, outbreak forecasting, database querying, strategic games and demand forecasting. Use the official tasks and usage conditions for a separate retention/transfer track; do not claim a CL-Bench score from synthetic temporal episodes. <https://continual-learning-bench.com/> <https://sky.cs.berkeley.edu/project/continual-learning-bench/>

### Neuromorphic and edge systems

**NeuroBench** supplies an algorithm track and a system track. Its algorithm-side metrics include footprint, connection/activation sparsity and synaptic operations; system comparisons are task-level and require timing and power/energy methodology. Report its metrics where the Symthaea workload fits, then add full-path memory traffic, tail latency and verified energy boundaries. <https://www.nature.com/articles/s41467-025-56739-4> <https://neurobench.ai/>

**MLPerf Tiny v1.3** covers image classification, person detection, keyword spotting, anomaly detection and streaming wakeword, with single-stream latency/quality and optional energy measurements. It is not a general irregular-time-series or continual-learning leaderboard. Use it only for compatible streaming or embedded inference deployments, preserving the official model, quality and measurement rules. <https://mlcommons.org/benchmarks/inference-tiny/>

## 8. Preregistered target gates (targets, not results)

1. **Validity:** zero future-boundary leaks; zero candidate access to evaluator-only fields; zero missing/unknown resource dimensions silently treated as zero.
2. **Safety:** zero accepted plans over any hard resource cap or maximum-observation-gap limit.
3. **Exactness:** exact coverage of all (2^n) assignments for small oracle fixtures; exact agreement with a second independent implementation on hand-calculated cases.
4. **Benefit:** at matched per-resource ceilings, target at least 10% lower held-out temporal error debt on one preregistered primary exposure axis, with paired uncertainty intervals excluding zero and no practically meaningful regression in the other exposure axes.
5. **Efficiency alternative:** at equivalent three-axis exposure, target at least 20% lower measured cost in at least one resource dimension without exceeding any other constrained dimension.
6. **Interaction:** outperform the best subsystem-local policy on at least one preregistered interaction-heavy regime and do not materially regress on null controls.
7. **Calibration:** predicted marginal value must be calibrated on held-out episodes; report reliability, signed error and calibration drift, not only rank correlation.
8. **Physical measurement:** any energy claim includes the full declared path (sensor/idle listening, host, memory, transfers, preprocessing, model execution, verification and receipt work) with instrument, sampling rate, idle baseline and uncertainty reported.

Report the three exposure axes separately. Report resource usage, overrun count, p50/p95/p99 latency, worst-regime performance and Pareto regret. Any hypervolume summary must disclose its normalization and publish the raw vectors.

## 9. Work still required

- Run both Cargo test suites, formatting and Clippy on the exact branch head.
- Add a second independent oracle implementation and hand-derived golden vectors.
- Add explicit online dual-price adaptation with bounded updates and non-fungible hard constraints.
- Add candidate-value calibration and prediction-versus-realized gain analysis.
- Add real-data adapters for Time-IMM/IMM-TSF and matched APN/KAFNet/QuITE++ baselines.
- Add randomized paired factorial interventions for large action sets.
- Profile actual allocation, byte traffic, memory bandwidth, tail latency and calibrated joules per correct decision.
- Integrate the resulting evaluator records with #9630 signed receipts and #9640 witness verification. The current branch does not deploy remote witnesses, TPM counters, CSPRNG challenges or protected durable witness storage.

## 10. Claim ceiling

This branch adds source-level reference implementations and a qualification protocol only. It does not show that Symthaea beats APN, KAFNet, QuITE++, CL-Bench systems, NeuroBench submissions or MLPerf Tiny results. It does not establish a deployed rollback-resistant evaluator, a calibrated energy advantage, or globally optimal temporal control. Those claims require exact-head execution, valid matched baselines, independent evidence verification and hardware measurements.
