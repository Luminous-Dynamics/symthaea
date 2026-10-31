# Temporal-Value Allocation and Continuous-Time Risk Calibration (v0.1)

**Tracking:** NEURO-SILICON-010 / #9557  
**Status:** research prototype authored; no Rust build, test, benchmark, or hardware run claimed  
**Branch:** `research/temporal-value-allocation-20261031`  
**Parent:** temporal trajectory/CSE draft #9529 at `05220001fc9dda9cf0570506bd2f06e89c8f08b4`  
**Scope:** standalone research crate only; no production neuron or HDC-LTC defaults changed

## 1. Research questions

This track asks two connected but independently falsifiable questions.

1. Under a fixed byte capacity, can a candidate-side policy preserve more future decision quality when it ranks memory by estimated future Temporal Error Debt avoided per byte-time, rather than by recency, frequency, fixed TTL, novelty, or confidence alone?
2. Can sampling-aware CSE/CAE estimates be calibrated against an independently computed continuous-time reference for a precisely declared prediction path, while exposing the assumptions and failure modes induced by nonuniform sampling and informative missingness?

The intended paradigm shift is from treating memory as a passive cache to treating retention as a constrained allocation of future decision-error reduction. The value estimate is always a candidate prediction. The evaluator's TED outcome remains protected evaluation data and is never an input to the candidate policy.

## 2. Workstream A — temporal-value memory allocation

### 2.1 Candidate-visible contract

`TemporalMemoryCandidate` contains only candidate-visible fields:

- stable item ID and byte size;
- candidate-estimated future wrong-belief exposure avoided (microseconds);
- confidence, relevance, freshness and novelty in parts per million;
- last access and last update times;
- expected useful retention horizon;
- optional hard validity deadline;
- access count.

The caller supplies an estimator SHA-256 digest, which is copied into the returned plan as provenance. This is not a signature and does not authenticate the caller. A future runner receipt must bind the complete canonical candidate feature record, policy, capacity, clock, estimator digest, task digest and output plan.

There is deliberately no oracle value, expected answer, future query, protected seed, or evaluator-only field in the policy API.

### 2.2 Policy arms

The prototype defines six candidate-side arms:

- **Temporal value per byte-time:** rank by candidate-estimated TED reduction adjusted by confidence, relevance and freshness, divided by bytes times expected retention duration.
- **LRU:** most recently accessed item first.
- **LFU:** highest access count first.
- **Fixed TTL:** exclude items older than the declared TTL; rank surviving items by remaining TTL.
- **Novelty-first:** highest novelty estimate first.
- **Confidence-first:** highest confidence estimate first.

All arms enforce the same hard validity boundary. A known-expired/retracted item is ineligible regardless of its predicted value. The temporal-value path caps its expected retention horizon at the hard validity deadline.

### 2.3 Fixed-point score

Let (D_i) be the candidate-estimated TED reduction in microseconds, (c_i,r_i,f_i) the confidence, relevance and freshness fractions, (b_i) the item byte size, and (h_i) the effective retention horizon in microseconds.

The idealized score is:

[
S_i = rac{D_i c_i r_i f_i}{b_i h_i}
]

The implementation uses sequential integer PPM scaling and a fixed (10^{18}) score scale to keep arithmetic deterministic and avoid floating-point policy ordering. The denominator uses checked (u128) multiplication. Input validation rejects duplicate IDs, zero-size/zero-horizon items, future timestamps, invalid PPM values and zero TTL. Equal scores break by ascending stable item ID.

Selection is a deterministic greedy density heuristic under a hard byte budget. It is **not** an exact knapsack solver and is not claimed to be globally optimal. Items that do not fit are skipped so smaller lower-ranked items can still use remaining capacity. A zero predicted value does not consume temporal-value budget.

### 2.4 Current implementation boundary

Implemented in `research/temporal-bench/src/temporal_value.rs`:

- candidate-visible feature schema and estimator provenance digest;
- six policy variants;
- hard-validity enforcement;
- deterministic ordering and byte-budget selection;
- explicit selected/evicted IDs, rank scores, eligibility and bytes used;
- eleven authored unit tests.

Not implemented yet:

- an online TED-value predictor/calibrator;
- refresh/write-cost prediction;
- tiered memory placement or measured bytes moved;
- integration with production HDC-LTC state;
- a task runner or isolated candidate/evaluator process;
- execution of the Rust tests.

The selector is a policy primitive. It does not establish that the predicted item values are accurate or that the policy improves retention quality.

## 3. Workstream B — exact continuous-time reference

### 3.1 Declared prediction path

For each channel, define residual samples (e_i = hat y(t_i)-y(t_i)). The reference path is the piecewise-linear interpolant between consecutive observed residual endpoints. For segment duration (Delta t) and endpoints (e_0,e_1):

[
int_{t_0}^{t_1} e(t)^2,dt =
rac{Delta t}{3}(e_0^2+e_0e_1+e_1^2)
]

The absolute-error integral uses the trapezoid formula when the residual does not cross zero. For a sign crossing it integrates the two triangular areas exactly:

[
int |e(t)|,dt =
rac{Delta t(e_0^2+e_1^2)}{2(|e_0|+|e_1|)}
quad 	ext{when } e_0e_1<0
]

The implementation returns integrated and duration-normalized squared/absolute risk for each of the three channels. It rejects fewer than two samples, mismatched lengths, duplicate/reversed timestamps and nonfinite residuals.

### 3.2 Claim boundary

This reference is exact **for the declared piecewise-linear residual interpolant on the interval from the first to last observation**. It does not prove that the true latent process or candidate output followed a straight line between samples. It does not extrapolate before the first or after the last timestamp. A long unobserved gap may hide transients; that is a sampling limitation to measure, not interpolate away.

The irregular forecast scorer now returns this reference alongside MSE, MAE, estimated CSE, estimated CAE, bandwidth and effective sample size. This enables direct calibration diagnostics without treating the reference as a universal latent-process oracle.

Implemented in `research/temporal-bench/src/continuous_risk.rs`:

- exact closed-form squared and absolute integration;
- explicit sign-crossing handling;
- duration-normalized results;
- eight authored closed-form and invalid-input tests.

Integrated into `irregular_metrics.rs` as `piecewise_linear_reference`; the existing MSE/MAE and CSE/CAE values remain separately visible. CSE follows the cited paper's squared-error estimator; CAE is a locally added absolute-error companion and must be qualified separately rather than attributed to that paper.

## 4. Sampling and missingness qualification

CSE addresses timestamp-density weighting under its statistical assumptions. It must not be described as a universal correction for informative missingness. If observation probability depends on an unobserved target/error state, timestamp density alone does not identify the missing outcomes. A propensity-based correction would require a known or independently estimable observation probability and positivity/support checks.

Preregister separate strata:

1. uniform observations;
2. clustered observations;
3. bursty observations;
4. sparse observations with long gaps;
5. missing completely at random (MCAR);
6. structured missingness patterns;
7. state-dependent/informative missingness as a stress test, with no unbiasedness claim unless observation propensities are supplied and validated.

For each schedule, retain a fixed underlying continuous path and sample it under multiple observation distributions. Report:

- MSE and MAE over observed timestamps;
- CSE and CAE with bandwidth and estimator details;
- exact piecewise-linear interpolant risk;
- effective sample size and weight distribution;
- relative bias/error against the interpolant reference;
- ranking changes across schedules;
- coverage, gap-length distribution and any missingness propensity assumptions.

The CSE bandwidth is evaluator-controlled and frozen before confirmatory runs. A bandwidth sweep is a development-stage sensitivity analysis, not an after-the-fact selection on confirmatory outcomes.

## 5. Combined temporal-value allocation experiment

### 5.1 Workloads

Use the existing delayed recall, selective retention/invalidation, bitemporal T10 and repeated-checkpoint TED tasks. The current suite already includes ten synthetic/reference task families; external task parity should be added only after exact data, licences, source versions and split semantics are pinned.

Use both synthetic exact controls and external benchmarks. Do not collapse task scores across different semantics.

### 5.2 Factorial design

- Policy arms: temporal value, LRU, LFU, fixed TTL, novelty-first, confidence-first, plus full-retention and no-retention controls.
- Capacity: preregister fixed byte budgets and relative strata such as 25%, 50%, 75% and 100% of a declared reference footprint. Record actual bytes, not config labels.
- Refresh budget: equal number of refresh opportunities and equal update compute for every policy.
- Seeds: at least 10 paired pilot seeds and at least 30 paired confirmatory seeds, or a preregistered power-based alternative.
- Split: partition by source family / generated process family, not merely random timestep or overlapping windows.
- Scoring: evaluator-side TED after candidate execution; candidate receives only public events and its own estimated feature stream.
- Failure accounting: invalid, missing, timeout, abstention and out-of-budget runs remain in denominators.

### 5.3 Outcomes — separate Pareto axes

**Capability:** task-native accuracy/F1, delay-conditioned recall, change detection/recovery, false alarms, prior-regime retention and worst-family results.

**Reliability:** wrong-belief exposure, abstention exposure, missing-output exposure, stale-memory false recall, correction/retraction latency, answer oscillation and future-information leakage.

**Memory economics:** persistent bytes, peak bytes, byte-seconds, refresh/write bytes, read bytes, eviction/reload count and state traffic by memory tier.

**Runtime:** p50/p95/p99 event latency, throughput, idle exposure and wall-clock time to a fixed quality target.

**Energy:** calibrated joules per continuous stream-hour, joules per correct decision and total energy at fixed quality. No energy inference from operation count, event sparsity or simulator estimates.

**Estimator quality:** calibration of predicted TED reduction, CSE/CAE error against the declared interpolant reference, bandwidth sensitivity and effective sample size.

Do not create one weighted aggregate score. Publish quality-versus-TED, TED-versus-byte-seconds, quality-versus-energy, and estimator-calibration plots separately.

## 6. Proposed qualification targets (not results)

Initial stretch targets for the confirmatory programme:

- at matched byte and refresh budgets, at least 20% lower wrong-belief exposure than the strongest simple retention baseline on at least one preregistered task family;
- alternatively, equal TED within a preregistered noninferiority margin while using at least 30% fewer byte-seconds;
- no increase in stale-memory false recall or future-information leakage;
- 100% preservation of exact historical-as-known answers on exhaustive bitemporal fixtures;
- the predicted TED reduction must be calibrated on held-out source families; report calibration slope, intercept, reliability bins and uncertainty, not only correlation;
- continuous-risk reference tests pass exact hand-calculated positive, negative and sign-crossing cases;
- CSE/CAE estimates must be evaluated against the reference across sampling schedules with preregistered finite-sample tolerances and ESS floors.

These are target gates, not achieved results. The final CSE tolerance and minimum ESS should be fixed from development-only calibration experiments, before confirmatory evaluation.

## 7. Industry and research alignment

- **Temporal SNN qualification:** Ma et al., IEEE TNNLS (8 October 2026), DOI 10.1109/TNNLS.2026.3736737, introduces a four-task, multi-timescale temporal-processing suite and reports remaining challenges in online learning and long-range dependencies. Add compatible parity tasks; do not claim the internal suite replaces the paper's benchmark. https://pubmed.ncbi.nlm.nih.gov/42848555/
- **BrainTrace:** Wang et al., Nature Communications (19 January 2026), DOI 10.1038/s41467-026-68453-w. Required online-learning comparator with D-RTRL and pp-prop/ES-D-RTRL where compatible. Paper numbers remain literature references until independently reproduced. https://www.nature.com/articles/s41467-026-68453-w
- **STDL:** Ma et al., IEEE TNNLS 37(9), September 2026, DOI 10.1109/TNNLS.2026.3671461. Required spatial/temporal training comparator on compatible event/static vision, not a drop-in baseline for every synthetic sequence task. https://pubmed.ncbi.nlm.nih.gov/41824348/
- **TEMPO:** a stochastic temporal-robustness benchmark using Morse timing derived from 44 recordings; the published proof-of-concept reports a 75.7% baseline and a spike-order randomization control. This is a valuable external timing-only task to investigate and reproduce under its licence and exact protocol. DOI 10.1088/2634-4386/ae84f7. https://doi.org/10.1088/2634-4386/ae84f7
- **BITS:** an irregular-time benchmark proposal covering 11 datasets across nine domains and multiple irregularity dimensions. Use it as an external parity/adapter target after licensing, manifests and evaluation protocol are pinned; do not mix its metrics into synthetic CSE calibration. https://arxiv.org/abs/2609.33303
- **CSE:** Li et al. (18 August 2026), arXiv:2608.17293, motivates separating sample-density effects from continuous-time predictive risk. The exact interpolant reference here is a local calibration oracle for a declared prediction path, not a replacement for the paper's statistical estimator. https://arxiv.org/abs/2608.17293
- **NeuroBench:** maintain the distinction between its hardware-independent algorithm track and system-aware deployment track. Its maintained leaderboards include few-shot class-incremental learning, event-camera object detection, non-human-primate motor prediction and chaotic-function prediction. https://neurobench.ai/ and https://github.com/NeuroBench/neurobench/blob/main/leaderboard.rst
- **MLPerf Tiny v1.4:** use closed/open divisions and fixed workload/quality targets for edge-system comparisons. Its streaming wake-word task includes idle listening; energy must use a calibrated measurement boundary. https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/

The comparisons must be task compatible. STDL's spatial auxiliary learning, BrainTrace's online credit assignment, GRU/LSTM/GRU-D/ODE-RNN sequence modelling and HDC-LTC temporal memory answer related but distinct questions.

## 8. Security and evidence requirements

This branch inherits the current #9529 limitation: task and oracle artifacts are still generated in the same process. Candidate-side policy separation in a Rust module is not OS-level isolation.

Before confirmatory claims:
- complete #9437 with distinct generator, candidate and evaluator authorities;
- deny candidate access to oracle storage, seeds, evaluator keys and inherited descriptors;
- bind task, model, config, runtime, prediction, evaluator profile and raw measurements in canonical artifacts;
- sign receipts using an evaluator-owned key;
- reject replayed or mismatched receipts;
- execute filesystem, symlink, TOCTOU, descriptor, environment, mutation and replay negative controls on the target host;
- retain raw failed runs and independent verifier outputs.

## 9. Implementation and validation status

Authored in this branch:
- `temporal_value.rs`: six policy arms and ten unit tests;
- `continuous_risk.rs`: exact interpolation integral and eight unit tests;
- `irregular_metrics.rs`: exposes exact piecewise-linear reference beside MSE/MAE/CSE/CAE;
- `lib.rs`: exports both new modules.

This adds 19 authored tests to the 208-test parent branch description (227 authored test functions expected across the combined branch). The sources have not been compiled, formatted, linted or executed: this environment has no `cargo`, `rustc` or `rustfmt`. The counts are source-level authored tests, not passing tests.

No temporal-value predictor, HDC-LTC production integration, baseline training, data adapter, confirmatory evaluation, hardware measurement or energy result is claimed.

## 10. Next sequence

1. Obtain an independent source-level review and run formatting, build, all tests and Clippy on the exact branch head.
2. Review the temporal-value score semantics, ineligible-item handling, fixed-point ordering and hard invalidation.
3. Review the exact interpolant integration against an independent implementation.
4. Add candidate-side predicted-value calibration from development data only.
5. Integrate the selector with T6/T10/TED trajectories through the isolated evaluator (#9437).
6. Add BITS and TEMPO adapters after manifest, licence and protocol reconciliation.
7. Reproduce compatible BrainTrace, STDL, GRU/LSTM/GRU-D and continuous-time recurrent baselines.
8. Run paired confirmatory seeds and publish every Pareto axis.
9. Measure deployed system performance and energy only on calibrated target hardware.
