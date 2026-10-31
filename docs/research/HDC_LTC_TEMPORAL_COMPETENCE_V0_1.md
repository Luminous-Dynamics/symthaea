# HDC-LTC Multi-timescale Temporal Competence Benchmark (v0.1)

**Status:** preregistered protocol; not implemented or executed  
**Tracking:** NEURO-SILICON-007 / issue #9394  
**Parent research:** #9369 / #9370 (fast/slow memory), #9371 / #9372 (input trajectories), #9373 (Fourier/RK4 semantics), #7749 (HDC-LTC architecture)  
**Version:** 0.1.0

## 1. Purpose

This benchmark asks whether HDC-LTC/CfC can exploit temporal structure more effectively than matched recurrent, spiking, and temporal-convolution baselines. It targets capabilities that ordinary shuffled-window classification can conceal:

- immediate reaction to sparse events;
- recall after controlled delays and distractor interference;
- interactions among fast, medium, and slow latent processes;
- online adaptation after hidden regime changes;
- irregular observation times and missing data;
- selective retention and invalidation of stale context;
- phase-sensitive prediction under temporal warping.

The suite is an evaluation layer, not a new neuron implementation. It does not change the active HDC-LTC equations, integration defaults, memory policy, or model dimensions.

## 2. Research motivation and external alignment

An October 8, 2026 IEEE TNNLS review of SNN temporal processing identifies limitations in commonly used neuromorphic benchmarks for multi-timescale dependencies. Its abstract describes a four-task suite and reports remaining challenges in online learning and long-range dependency modeling relative to non-spiking sequence models (Ma et al., DOI: 10.1109/TNNLS.2026.3736737; PubMed PMID 42848555). The full task definitions and implementation have not yet been inspected; v0.1 does not claim to reproduce those tasks.

NeuroBench separates hardware-independent algorithm qualification from deployed-system qualification. MLPerf Tiny v1.4 fixes workloads and quality targets, and evaluates latency and energy under declared measurement methods. This suite should map to those conventions:

- **Algorithm track:** task quality, temporal integrity, state/compute complexity, and reproducible seeds;
- **System track:** measured end-to-end latency, persistent and peak memory, memory traffic, and calibrated energy on named hardware;
- **No blended score:** publish a Pareto frontier and task-specific metrics rather than weighting correctness, safety, and efficiency into one number.

## 3. Core hypothesis

The hypothesis under test is that *causal state allocation* can improve the temporal capability/resource frontier:

1. fast state handles new observations and low-latency reactions;
2. medium state retains task-relevant working context;
3. slow state preserves useful context across long delays;
4. novelty/event control allocates state updates, refresh, compression, and invalidation;
5. elapsed time, missingness, uncertainty, and discontinuities remain explicit.

This is not a presumption that three-timescale memory must win. Include null workloads where a small GRU, TCN, or fixed-state model should be more efficient.

## 4. Task contract

### T1 — Fast reaction and event timing

Use quiet intervals, isolated events, bursts, and distractor noise. Report event precision/recall, false triggers per unit idle time, and event-to-decision latency p50/p95/p99. Count preprocessing and input encoding in the system result.

### T2 — Delayed-cue associative recall

Present a cue and later query a dependent fact after 1, 4, 16, 64, 256, or 1,024 steps. Vary distractor count and cue/distractor similarity. Report accuracy by delay and interference level. Do not hide long-delay collapse inside a pooled mean.

### T3 — Multiple interacting timescales

Mix latent processes with independently declared periods/decays. Include phase shifts, amplitude changes, coupling, and regime reversal. Evaluate both state classification and short-horizon forecasting against exact generated ground truth.

### T4 — Online change-point adaptation

Use hidden abrupt, gradual, recurring, and false change points. The model receives only the legal online stream; it cannot reset or retrain at a hidden boundary. Report change detection delay, performance recovery delay, cumulative regret, false alarms, and retention/forgetting on previous regimes.

### T5 — Irregular sampling and missingness

Vary timestamp jitter, inter-observation gaps, bursts, and missingness. Elapsed simulated time—not invocation count—defines physical decay. Reject out-of-order, duplicate, non-finite, and non-monotonic timestamps according to the input contract. Report performance by gap and missingness stratum.

### T6 — Memory interference and selective retention

Insert distractor traces that compete with a relevant cue. Compare no memory, fixed decay, learned decay, fast/slow state, and event/novelty-triggered refresh. Include explicit invalidation/retraction: a stale trace must not persist as a correct answer after its validity ends.

### T7 — Periodic and phase-sensitive prediction

Predict phase, transitions, and next events under controlled time scaling and phase shifts. When Fourier features are used as continuous-time forcing, evaluate at correct integration stage times. An interval-held snapshot is a distinct approximation and must be named as such.

## 5. Data and split discipline

Keep two strata separate:

- **Synthetic exact controls:** deterministic generators and analytical/exhaustive oracles for causality, delay, time warping, change points, and invalid input.
- **Real workloads:** begin with SHD, SSC, DVS Gesture, S-MNIST/PS-MNIST, and MLPerf Tiny streaming wake-word. Add others only after licensing, provenance, timestamps, labels, preprocessing, and split semantics are pinned.

For real data, split by subject/session/source recording *before* windowing. A source recording and overlapping windows from it must not cross train, development, calibration, or confirmatory partitions. Hash the source manifest, preprocessing, label map, and split manifest. Preserve original timing/missingness.

Synthetic generated instances must be source-family split, not merely seed split. The confirmatory evaluator/oracle is not accessible to the candidate model or tuning loop.

## 6. Baseline registry

Required baseline families, where semantics are appropriate:

- exact incumbent HDC-LTC/CfC;
- static HDC and memoryless HDC sequence encoding;
- GRU and LSTM;
- GRU-D or another missingness-aware recurrent variant;
- ODE-RNN/latent-ODE or another reproducible continuous-time recurrent reference;
- adaptive LIF/ALIF or a representative SNN on event-native tasks;
- BrainTrace online-learning methods on compatible SNN tasks: D-RTRL (parameter-dimensional traces), pp-prop / ES-D-RTRL (input/output-factorized traces), and e-prop where supported by the matched model;
- a small temporal convolution network;
- matched parameter/state-budget and compute-budget controls.

**BrainTrace is a required named competitor, not a generic SNN checkbox.** The paper *Model-agnostic linear-memory online learning in spiking neural networks* (Nature Communications, 19 January 2026; DOI 10.1038/s41467-026-68453-w) introduces model-agnostic online learning/eligibility-trace compilation. Its published abstract and full text report D-RTRL and pp-prop/ES-D-RTRL methods, and online training across several SNN dynamics. The current public package inspected for this protocol is BrainTrace v0.2.8 at commit `4fd426e1ec58a61a11d47a8d95c32346778e8b11`; the public DMS experiment repository is pinned at `81332436b3a0167bc83059878e43a0f0e2eb62c7`. See `BRAINTRACE_REFERENCE_AUDIT_V0_1.md` for the code-level task and split audit.

BrainTrace's algorithms are online **learning** baselines, not drop-in inference-only architectures. Compare them on online adaptation and long-horizon credit assignment, count trace memory and compute, and include BPTT as a small-scale gradient reference. Do not interpret pp-prop's factorized trace as an exact full-parameter gradient; keep its approximation/rank/decay explicit. Paper-reported metrics remain literature results, not Symthaea reproductions.

For every baseline pin repository, exact commit/release, framework, configuration, training budget, precision, data pipeline, random seed, hardware, and all deviations. A paper's reported score is a literature reference, not a reproduced score. Compare only within the same task definition and test split.

## 7. Metrics

### Capability and temporal behavior

- task-native accuracy, macro-F1, AUROC, or AUPRC;
- recall as a function of delay and distractor similarity;
- event precision/recall and false alarms during idle periods;
- change detection and recovery latency;
- cumulative regret and prior-regime retention;
- phase error and forecast horizon;
- calibration/robustness under jitter, gaps, and missingness;
- worst-family and worst-delay performance.

### Temporal integrity

- future-sample leakage negative controls;
- causal event-order and interval-endpoint tests;
- time-scale and phase transformation sensitivity;
- duplicate/out-of-order/non-finite timestamp handling;
- stale-memory false-recall and invalidation behavior;
- reset, discontinuity, and long-gap behavior.

### Resource accounting

- model parameter bytes;
- persistent state bytes and peak resident/working-set bytes;
- allocation count and temporary/copy buffers;
- read/write bytes by stage and memory tier;
- operation/counter estimates (clearly marked if modeled);
- batch-1 event, burst, idle, and continuous-stream latency p50/p95/p99;
- joules per correct decision and complete-stream energy;
- idle listening energy and preprocessing/encoding energy;
- throughput under declared stream concurrency.

Do not report only multiply-accumulate counts. For memory-bound HDC, bytes moved and cache behavior may dominate arithmetic.

## 8. Experimental protocol

- Pilot: at least 10 paired seeds on synthetic controls.
- Confirmatory: at least 30 independent paired seeds, or a preregistered power-based alternative.
- Identical generated instances and legal observations across compared models.
- Group uncertainty by generated source family or real subject/session; timesteps/windows are not independent replicates.
- Keep development, calibration, and confirmatory source families separate.
- Freeze thresholds, model-selection budget, and final splits before confirmatory runs.
- Count failed, invalid, timed-out, and inconclusive attempts in cost.
- Use paired confidence intervals and account for multiple comparisons across broad ablations.
- Publish mean, median, uncertainty, and worst-family results.

## 9. Proposed qualification gates

These are targets, not measured results:

1. Zero future-sample leakage on exhaustive synthetic controls.
2. Zero silent acceptance of invalid time inputs; errors are explicit and deterministic.
3. Scalar analytical controls converge to tolerance under integration step refinement.
4. Delay-conditioned recall and interference curves are published for every preregistered delay.
5. Online adaptation reports detection, recovery, regret, false alarms, and old-regime retention.
6. Any memory reduction includes persistent and peak bytes, all intermediate buffers, and read/write traffic.
7. Energy claims use calibrated hardware measurement and disclose the full boundary; simulator estimates are not hardware measurements.
8. Improvement claims are supported on held-out source families and state all Pareto gains and regressions.
9. Null tasks visibly expose cases where simpler baselines are superior.
10. No production behavior changes until exact-head build/tests, independent numerical controls, and matched baselines pass.

## 10. Execution ladder

1. Validate the JSON matrix and deterministic generator contract.
2. Implement small Rust synthetic generators plus an independent scalar oracle.
3. Add the runner adapter without modifying active neuron code.
4. Execute causal/invalid-input negative controls and delay/interference sweeps.
5. Add recurrent/SNN baselines under matched interfaces and resource budgets.
6. Add real datasets only after split/provenance qualification.
7. Add measured device/energy campaigns after algorithm-track evidence is stable.

## 11. Evidence and claim boundary

The repository contains an isolated, dependency-free Rust prototype under research/temporal-bench:
- a delayed-cue generator with controlled key similarity and distractor interference;
- a multi-timescale forecasting generator with hidden periods/phases, a hidden middle-third regime boundary, and delayed online feedback;
- an evaluator-side trace verifier with negative-control tests for altered labels, early feedback, missing feedback, and oracle identity mismatch.

The generated public event stream and oracle remain returned in one process. The verifier has been implemented, but the confirmatory oracle is **not yet sealed by process or permission isolation**. Rust unit tests are authored but unrun in this environment. The full seven-family generator, streaming runner, dataset adapters, separate evaluator service, benchmark execution, model comparison, hardware measurement, and energy result are not yet included. A passing benchmark would establish a result only for the declared tasks, splits, model versions, and hardware. It would not establish consciousness, general intelligence, universal superiority, or neuromorphic hardware efficiency if measured only on conventional CPU/GPU execution.
