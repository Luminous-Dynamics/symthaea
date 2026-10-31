# HDC-LTC Multi-timescale Temporal Competence Benchmark (v0.1)

**Status:** preregistered protocol with ten prototype Rust task generators; benchmark not built or executed  
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

NeuroBench has two explicit tracks: its algorithm track evaluates correctness and static complexity independent of deployment target, while its system track measures deployment-aware timing and efficiency. MLPerf Tiny v1.4 offers a complementary edge-deployment standard: accuracy, latency, and calibrated energy per inference at fixed task-quality targets. Its five workloads include keyword spotting, visual wake words, image classification, anomaly detection, and continuous streaming wake-word detection. Streaming wake-word is particularly relevant because it measures idle listening as well as detection events.

Use the divisions correctly:
- **MLPerf Tiny Closed Division:** fixed reference model, dataset, and processing pipeline; useful for apples-to-apples hardware/runtime comparison.
- **MLPerf Tiny Open Division:** allows different models, training methods, and optimizations while retaining the same task and quality target; the right comparison path for a new HDC-LTC architecture.
- **NeuroBench Algorithm Track:** task correctness, temporal integrity, and static complexity under hardware-independent conditions.
- **NeuroBench System Track:** actual deployment timing, memory, power, and energy with the full system boundary.

The 2026 v1.4 round reported nine organizations and 25 configurations across Closed and Open divisions. MLPerf's own results show why we must distinguish latency from energy and include idle duty cycle: Syntiant reported a 3.3% duty cycle on streaming wake-word, while Qualcomm reported sub-0.30 ms on several legacy sensing workloads. Those figures are workload/platform-specific external reference points, not Symthaea comparisons. Use the official streaming wake-word workload as a deployment anchor where compatible; do not claim that our custom delayed-recall suite is itself an MLPerf submission.

This suite should align to these conventions without collapsing them into one score:
- **Algorithm track:** task quality, temporal integrity, state/compute complexity, and reproducible seeds;
- **System track:** measured end-to-end latency, persistent and peak memory, memory traffic, and calibrated energy on named hardware;
- **No blended score:** publish task-specific metrics and capability/resource Pareto frontiers rather than weighting correctness, safety, and efficiency into one number.

## 3. Core hypothesis

The hypothesis under test is that *causal state allocation* can improve the temporal capability/resource frontier:

1. fast state handles new observations and low-latency reactions;
2. medium state retains task-relevant working context;
3. slow state preserves useful context across long delays;
4. novelty/event control allocates state updates, refresh, compression, and invalidation;
5. elapsed time, missingness, uncertainty, and discontinuities remain explicit.

This is not a presumption that three-timescale memory must win. Include null workloads where a small GRU, TCN, or fixed-state model should be more efficient.

## 4. Task contract

### T1 — Always-on sparse-event detection and event timing

A Rust synthetic prequalification generator is implemented in `research/temporal-bench/src/fast_event.rs`. Its reference configuration provides a 16,384-sample continuous scalar stream at 1 ms/sample, with eight target motifs, sixteen similar non-target motifs, bounded background noise, and randomized placements in non-overlapping temporal blocks. A quiet-idle mode contains no motifs. The candidate receives only timestamped samples; event boundaries, labels, codes, and amplitudes remain in the oracle.

This is an event-detection control for temporal semantics, not an audio wake-word benchmark and not a reproduction of MLPerf Tiny. The target signature is a fixed eight-sample binary motif. High-similarity distractors differ in 1–3 of eight positions; low-similarity distractors differ in 4–7. All baselines receive exactly the same public stream.

Report target-event precision/recall, misses, duplicate activations, distractor false alarms, quiet-idle false alarms, monitored quiet exposure, idle false triggers per hour, and event-to-decision latency p50/p95/p99. Publish raw false-positive counts alongside rates; do not present a rate without its measured exposure. Count preprocessing, input encoding, event scheduling, and decision post-processing in the system result. Add burst-cluster variants as a subsequent task extension; v0.1 currently qualifies rare events, distractors, and quiet idle, not a separately parameterized burst process.

This task is aligned with the *measurement principle* behind MLPerf Tiny v1.4's continuous streaming wake-word workload, which explicitly considers idle listening and calibrated energy, but it is not that workload's audio data, model, or official metric implementation. [MLPerf Tiny v1.4 overview](https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/).

### T2 — Delayed-cue associative recall

Present a cue and later query a dependent fact after 1, 4, 16, 64, 256, or 1,024 steps. Vary distractor count and cue/distractor similarity. Report accuracy by delay and interference level. Do not hide long-delay collapse inside a pooled mean.

### T3 — Multiple interacting timescales

Mix latent processes with independently declared periods/decays. Include phase shifts, amplitude changes, coupling, and regime reversal. Evaluate both state classification and short-horizon forecasting against exact generated ground truth.

### T4 — Online change-point adaptation

Use hidden abrupt, gradual, recurring, and false change points. The model receives only the legal online stream; it cannot reset or retrain at a hidden boundary. Report change detection delay, performance recovery delay, cumulative regret, false alarms, and retention/forgetting on previous regimes.

### T5 — Irregular sampling and missingness

Vary timestamp jitter, inter-observation gaps, bursts, and missingness. Elapsed simulated time—not invocation count—defines physical decay. Reject out-of-order, duplicate, non-finite, and non-monotonic timestamps according to the input contract. Report performance by gap and missingness stratum.

A fixed-point synthetic generator now exists in `research/temporal-bench/src/irregular.rs`: three latent triangle-wave processes with independent periods/phases are observed on a jittered microsecond schedule; a declared fraction of scheduled observations are dropped; bounded integer noise is added to retained samples. The oracle stores exact latent values and the dropped-slot manifest. Its verifier checks both canonical digests, strict timestamp monotonicity, schedule bounds across missing slots, and observed-vs-latent noise limits. This is a synthetic T5 control, not yet a production dataset adapter or model benchmark.

### T6 — Memory interference and selective retention

Insert distractor traces that compete with a relevant cue. Compare no memory, fixed decay, learned decay, fast/slow state, and event/novelty-triggered refresh. Include explicit invalidation/retraction: a stale trace must not persist as a correct answer after its validity ends.

A deterministic synthetic task now exists in `research/temporal-bench/src/retention.rs`: it stores a target association, interleaves high- or low-similarity distractors, queries while the target is valid, explicitly invalidates the target record, and queries again. The oracle requires recall before invalidation and abstention afterward. The verifier checks the event schedule, association uniqueness, Hamming-distance bands, invalidation identity/time, expected answers, and SHA-256 content digests. This tests selective retention and stale-memory rejection—not merely whether old content can be recalled.

### T7 — Periodic and phase-sensitive prediction

A fixed-point phase workload is implemented in `research/temporal-bench/src/phase.rs`. It generates timestamped noisy samples from a periodic triangle oscillator under four regimes: stable, phase jump, frequency/period shift, and combined phase+frequency shift. The oracle stores cycle phase, effective period, and the exact next rising zero-crossing time at every sample. Frequency changes map the fractional cycle phase continuously at the hidden transition; phase jumps create a declared discontinuity. Candidate predictions are scored with circular phase error and absolute next-crossing timing error.

The v0.1 generator uses a piecewise-linear triangle oscillator to keep the oracle deterministic with integer arithmetic. It is a phase/timing control, not a sinusoidal Fourier benchmark or a reproduction of an external dataset. A sinusoidal/frequency-sweep variant and explicit burst-cluster controls remain extensions. When Fourier features are used as continuous-time forcing, evaluate at correct integration stage times. An interval-held snapshot is a distinct approximation and must be named as such.

**Affine time-dilation metamorphic qualification is now implemented.** The evaluator can transform a complete phase trace by an exact rational factor: sample times, oscillator periods, phase offsets, phase jumps, and crossing deadlines all scale together, while normalized phase and observed samples remain unchanged. The transform fails closed if any microsecond value cannot be represented exactly. Phase scoring also reports expected points, missing predictions, and prediction coverage in parts per million against the entire stream; sparse predictions can no longer appear as a complete result by reporting error only on a selected subset.

## 5. Data and split discipline

Keep two strata separate:

- **Synthetic exact controls:** deterministic generators and analytical/exhaustive oracles for causality, delay, time warping, change points, and invalid input.
- **Real workloads:** begin with SHD, SSC, DVS Gesture, S-MNIST/PS-MNIST, and MLPerf Tiny streaming wake-word. Add others only after licensing, provenance, timestamps, labels, preprocessing, and split semantics are pinned.

For real data, split by subject/session/source recording *before* windowing. A source recording and overlapping windows from it must not cross train, development, calibration, or confirmatory partitions. Hash the source manifest, preprocessing, label map, and split manifest. Preserve original timing/missingness.

Synthetic generated instances must be source-family split, not merely seed split. The confirmatory evaluator/oracle is not accessible to the candidate model or tuning loop.

### External reference parity tasks from BrainTrace's public experiments

These are added as explicitly named reference tasks, not as evidence that the public repository contains a final-paper test split. Their exact public-code semantics and caveats are recorded in `BRAINTRACE_REFERENCE_AUDIT_V0_1.md`.

- **T8 — Delayed match-to-sample (DMS):** eight motion directions; sample, 1,000 ms delay, then rotated match/non-match test; 1 ms step in the inspected entry point; 100 input channels and a 200-neuron recurrent SNN reference. A Rust generator and structural/oracle verifier are now implemented under `research/temporal-bench/src/dms.rs`. Preserve the 1,000 ms anchor, then add preregistered shorter/longer delays and distractor variants. Use independent seed families and a sealed test split.
- **T9 — Sequential evidence accumulation:** seven 100 ms cues separated by 50 ms intervals, followed by a 1,000 ms delay and a 150 ms recall period; the inspected generator creates left/right cue assignments plus recall/noise channels and a binary majority label. A Rust generator and structural/oracle verifier are now implemented under `research/temporal-bench/src/evidence.rs`. Preserve the original semantics as an anchor, then add cue-count, evidence-strength, and delay strata.

These generators reproduce task *semantics and parameterization*, not the exact original NumPy/JAX pseudorandom byte stream or the BrainTrace model/training implementation. They generate a common deterministic stream for all candidate baselines; the exact serialized stream digest must be recorded.

The public DMS script uses NumPy global randomness and its inspected entry point hardcodes the I/O-factorized trainer despite README examples for other methods. The evidence-accumulation path seeds BrainState from an explicit argument but its inspected trainer uses an infinite training generator and stops on a training-accuracy threshold; it does not show a separate held-out test path. Our reproduction must supply the missing reproducibility and confirmatory split contract rather than treating those scripts' training curves as test performance.

### T10 — Bitemporal continual-knowledge stream

A new structured task is implemented in `research/temporal-bench/src/knowledge_stream.rs`. It maintains an append-only fact history with separate valid time and transaction/observation time. The candidate sees an initial assertion, high-similarity distractor facts, a correction that arrives after its valid-time boundary, a later retraction, and five queries with explicit `(valid_at, known_at)` coordinates.

The required behavior is not simply “remember the newest value.” It must preserve both:
- what the system had learned by a historical knowledge cutoff; and
- what it currently believes applied at an earlier valid time after a late correction.

The query suite includes same-valid-time queries before and after correction arrival, same-valid-time queries before and after retraction arrival, and an earlier-valid-time query after both updates. This distinguishes a proper bitemporal ledger from recency decay, a current-state key/value cache, and overwrite-on-update memory.

This task is an executable synthetic adapter to the existing bitemporal source-event and proof-DAG work (#9075, #9149, #9427), not a replacement for those runtime/persistence implementations. It adds no production ledger or proof-DAG code.

### OAKS continual-knowledge benchmark alignment

Kim et al., ACL 2026, introduce OAKS with OAKS-BABI and OAKS-Novel, evaluating answers at each context interval as facts evolve. The paper reports state-tracking delays and distraction susceptibility across 14 evaluated models. The official code repository is `kaistAI/OAKS` and the dataset repository is `adobe-research/OAKS`; pinned repository snapshots, README blob hashes, and the data-license digest are recorded in the machine-readable fixture.

OAKS is an external natural-language benchmark, not identical to T10's structured bitemporal oracle. Use both: T10 to test valid-time/known-time correctness under an exact oracle; OAKS to measure language-level tracking and distraction at every chunk. Do not combine their scores. The implementation checklist, data/adapter schema, official baseline mapping, scoring trajectory, leakage controls, license rules, and manifest reconciliation steps are specified in `OAKS_ADAPTER_CONTRACT_V0_1.md`.

The Adobe dataset repository's license restricts use to noncommercial research and teaching, excluding commercial product development or other commercial-gain activities. OAKS data must therefore remain in a noncommercial research lane unless a separate license is obtained. The OAKS evaluation-code README reports 435 OAKS-Novel questions while the dataset README reports 870; reconcile the precise data-file manifest and question-count convention before attempting a paper-score reproduction. No OAKS dataset has been downloaded or evaluated in this PR.

## 6. Baseline registry

Required baseline families, where semantics are appropriate:

- exact incumbent HDC-LTC/CfC;
- static HDC and memoryless HDC sequence encoding;
- GRU and LSTM;
- GRU-D or another missingness-aware recurrent variant;
- ODE-RNN/latent-ODE or another reproducible continuous-time recurrent reference;
- adaptive LIF/ALIF or a representative SNN on event-native tasks;
- BrainTrace online-learning methods on compatible SNN tasks: D-RTRL (parameter-dimensional traces), pp-prop / ES-D-RTRL (input/output-factorized traces), and e-prop where supported by the matched model;
- STDL spatiotemporal-decoupled SNN training on compatible event/static vision tasks, with its memory-constrained auxiliary-subnetwork variants and ELL/DECOLLE local-learning controls;
- a small temporal convolution network;
- matched parameter/state-budget and compute-budget controls.

**BrainTrace is a required named competitor, not a generic SNN checkbox.** The paper *Model-agnostic linear-memory online learning in spiking neural networks* (Nature Communications, 19 January 2026; DOI 10.1038/s41467-026-68453-w) introduces model-agnostic online learning/eligibility-trace compilation. Its published abstract reports D-RTRL and pp-prop/ES-D-RTRL methods, and online training across several SNN dynamics. The current public package inspected for this protocol is BrainTrace v0.2.8 at commit `4fd426e1ec58a61a11d47a8d95c32346778e8b11`; the public DMS experiment repository is pinned at `81332436b3a0167bc83059878e43a0f0e2eb62c7`. See `BRAINTRACE_REFERENCE_AUDIT_V0_1.md` for the code-level task and split audit.

BrainTrace's algorithms are online **learning** baselines, not drop-in inference-only architectures. Compare them on online adaptation and long-horizon credit assignment, count trace memory and compute, and include BPTT as a small-scale gradient reference. Do not interpret pp-prop's factorized trace as an exact full-parameter gradient; keep its approximation/rank/decay explicit. Paper-reported metrics remain literature results, not Symthaea reproductions.

**STDL is a second required named training comparator.** Ma et al., *Spatiotemporal Decoupled Learning for Spiking Neural Networks*, IEEE TNNLS 37(9), September 2026, DOI 10.1109/TNNLS.2026.3671461, combine memory-constrained auxiliary subnetworks for spatial decoupling with detached temporal dependencies. The abstract reports comparable accuracy to BPTT and 4x lower GPU memory on ImageNet (paper result only). The public repo is pinned at `ChenxiangMA/STDL` commit `e5e483332ade02c94d8354583ab46aa63e99ecf1`; source audit and caveats are in `STDL_REFERENCE_AUDIT_V0_1.md`. STDL is directly applicable to matched static/event vision training, but not a drop-in baseline for every synthetic scalar/DMS workload. Public CIFAR/SVHN/CIFAR10DVS/HARDVS trainer paths select checkpoints using a loader populated from the dataset's test split; reproduce with separate dev and one-shot confirmatory splits. Do not import its paper score into Symthaea results.

For every baseline pin repository, exact commit/release, framework, configuration, training budget, precision, data pipeline, random seed, hardware, and all deviations. A paper's reported score is a literature reference, not a reproduced score. Compare only within the same task definition and test split.

## 7. Metrics

### Capability and temporal behavior

- task-native accuracy, macro-F1, AUROC, or AUPRC;
- recall as a function of delay and distractor similarity;
- event precision/recall and false alarms during idle periods;
- change detection and recovery latency;
- cumulative regret and prior-regime retention;
- phase error and forecast horizon;
- phase prediction coverage, missing-prediction count, and full-stream denominator;
- affine time-dilation equivariance: phase/sample invariance and linearly scaled physical crossing error;
- calibration/robustness under jitter, gaps, and missingness;
- worst-family and worst-delay performance;
- exact-match accuracy at each `(valid_at, known_at)` bitemporal coordinate;
- future-knowledge leakage rate, obsolete-claim use after retraction, correction propagation latency, and preservation of historical as-known answers;
- OAKS per-chunk accuracy trajectory, answer-change precision/recall, and delay-to-correct-answer (reported separately from T10).

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
3. Zero transaction-time leakage on exhaustive bitemporal fixtures: a late correction must not change any answer whose `known_at` precedes the correction's observed time.
4. Exact agreement with an independent symbolic bitemporal oracle at all preregistered `(valid_at, known_at)` coordinates; historical as-known answers survive later corrections and retractions.
5. Scalar analytical controls converge to tolerance under integration step refinement.
6. Delay-conditioned recall and interference curves are published for every preregistered delay.
7. Online adaptation reports detection, recovery, regret, false alarms, and old-regime retention.
8. Any memory reduction includes persistent and peak bytes, all intermediate buffers, and read/write traffic.
9. Energy claims use calibrated hardware measurement and disclose the full boundary; simulator estimates are not hardware measurements.
10. Improvement claims are supported on held-out source families and state all Pareto gains and regressions.
11. Null tasks visibly expose cases where simpler baselines are superior.
12. No production behavior changes until exact-head build/tests, independent numerical controls, and matched baselines pass.

## 10. Execution ladder

1. Validate the JSON matrix and deterministic generator contracts.
2. Add burst-cluster variants to T1 and sinusoidal/frequency-sweep variants to T7.
3. Implement the isolated taskgen/candidate/evaluator runner in #9437; verify SHA-256 content binding and signed receipt/replay semantics before any confirmatory claim.
4. Compile and run the standalone Rust tests and negative controls.
5. Add the runner adapter without modifying active neuron code.
6. Execute causal/invalid-input negative controls and delay/interference sweeps.
7. Add recurrent/SNN baselines under matched interfaces and resource budgets.
8. Add real datasets only after split/provenance qualification.
9. Add measured device/energy campaigns after algorithm-track evidence is stable.

When a Rust toolchain is available, the local verification sequence is:

```sh
cargo fmt --manifest-path research/temporal-bench/Cargo.toml -- --check
cargo test --manifest-path research/temporal-bench/Cargo.toml
cargo clippy --manifest-path research/temporal-bench/Cargo.toml --all-targets -- -D warnings
```

These commands are a runbook, not evidence that checks have passed.

## 11. Evidence and claim boundary

The repository contains a standalone Rust research crate under research/temporal-bench (using the standard sha2 crate, not a dependency-free implementation). Prototype generators and task/oracle verifiers are authored for all ten task families:
- a delayed-cue generator with controlled key similarity and distractor interference;
- a multi-timescale forecasting/change-point generator with hidden periods/phases, a hidden middle-third regime boundary, delayed online feedback, and labels recomputed by the verifier from the hidden process;
- a DMS generator/verifier matching the pinned eight-direction task timing and population-tuned spike input;
- an evidence-accumulation generator/verifier matching the pinned seven-cue/1,000 ms delay/150 ms recall task structure;
- an irregular-time/missingness generator with jittered microsecond timestamps, explicit dropped-slot manifests, bounded noise, and latent oracle points;
- a selective-retention/invalidation generator with before/after query semantics and stale-memory abstention;
- an always-on sparse-event generator with rare target motifs, controlled distractor similarity, a quiet-idle control, event-level scoring, and false-trigger exposure accounting;
- a phase-sensitive prediction generator with fixed-point oscillator phase, hidden phase/frequency shifts, exact next-crossing targets, circular phase/timing scoring, full-stream missing-prediction accounting, and exact affine time-dilation metamorphic controls;
- a bitemporal continual-knowledge stream with retroactive correction, delayed retraction, distractor facts, and explicit `(valid_at, known_at)` query coordinates;
- seed-bound deterministic stream regeneration and replay verification for delayed cue, multiscale forecasting/change-point, DMS, evidence accumulation, irregular/missingness, retention/invalidation, fast-event detection, phase prediction, and bitemporal knowledge stream;
- canonical SHA-256 digests for public streams and oracle records, with mutation negative controls.

Seed-bound replay checks regenerate expected tasks and oracles from evaluator-held seed/configuration and reject recommitted but seed-inconsistent event streams. The seed/configuration remains evaluator-owned and is not given to the candidate. These checks improve reproducibility; they are not a cryptographic security boundary.

The public task and oracle are still returned together in one process. The confirmatory oracle is **not yet sealed by process or permission isolation**; there is no candidate sandbox, evaluator-owned signature, replay-resistant receipt, or secure taskgen/evaluator runner. **170 Rust unit tests are authored but unrun** in this environment; cargo, rustc, and rustfmt are unavailable here.

No dataset adapter, baseline training run, model comparison, hardware measurement, or energy result has been executed. The ten synthetic/reference-task generators are authored, but burst-cluster T1 and sinusoidal/frequency-sweep T7 extensions remain open. The optional OAKS external dataset adapter is specified but not implemented. The evaluator runner, real-dataset adapters, independent process isolation, signed receipts, benchmark execution, system measurements, and energy result are not yet included. A passing benchmark would establish a result only for the declared tasks, splits, model versions, and hardware. It would not establish consciousness, general intelligence, universal superiority, or neuromorphic hardware efficiency if measured only on conventional CPU/GPU execution.
