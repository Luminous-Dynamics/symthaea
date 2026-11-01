# Symthaea versus TinyML: comparative benchmark v0.1

Tracking: #9775  
Branch: `research/symthaea-tinyml-benchmark-v0.1`

## Executive answer

Symthaea could outperform conventional TinyML on some classes of tasks, but "TinyML" is a deployment field and ecosystem, not one model. It is not credible to claim overall superiority. Optimized INT8 CNNs on current MCU kernels/NPUs are likely to remain much stronger on static image/audio inference per millijoule. Symthaea's plausible advantage is persistent temporal state, irregular-time dynamics, associative memory, low-label adaptation, uncertainty-aware decisions and active sensing—provided those capabilities survive dimension reduction and are cheaper at the complete-event level.

The current HDC-LTC default is not yet MCU-ready at its default size. This note makes the memory problem explicit and starts a matched benchmark programme.

## What the industry actually measures

### MLPerf Tiny

MLPerf Tiny v1.4 (July 2026) reported 25 system configurations from nine organizations. It uses matched benchmark tasks, datasets and quality targets to compare latency and energy. The five task families are:
- image classification (CIFAR-10);
- visual wake word (COCO);
- keyword spotting (Speech Commands);
- anomaly detection (ToyADMOS);
- streaming wake word.

The published result highlights dedicated accelerators and sensing hubs. ST's reported STM32U3 hardware signal processing reduced image-inference latency up to 76% and power 23.3% on a specific comparison; an STM32H7P preview with Neural-ART NPU improved latency up to 96% relative to its CM7-only baseline. These are vendor-reported, workload-specific results, not universal ratios.

### ExecuTorch versus TFLite Micro

Arm's 19 October 2026 Cortex-M report compares ExecuTorch + CMSIS-NN to TFLite Micro on an Alif Ensemble E8, using one Cortex-M55 at 400 MHz with NPUs disabled. Arm reports:
- 2.3–10.6% lower inference latency on three of four MLPerf Tiny models;
- up to 21.1% lower measured flash footprint on all four;
- lower RAM on three of four, including a 46.1% reduction for its autoencoder; ResNet-8 used 1.2% more RAM.

This is a useful modern CPU-only baseline. We should not compare Symthaea only to an unoptimized TFLite Micro build, and accelerator-enabled results belong in a separate hardware tier.

### TinyML already includes adaptation

A September 2026 IEEE Access review of 156 TinyML studies treats on-device adaptation as a major category (40 studies), including continual/incremental learning, few-shot/transfer, unsupervised/self-supervised and federated methods. Thus Symthaea cannot claim novelty merely because it adapts on-device. The benchmark must compare against strong adaptive baselines.

## Three tracks, not one leaderboard

### Track A — fixed inference / MLPerf-compatible

Goal: determine whether Symthaea can compete on established fixed tasks.

Compare official/reference and optimized:
- TFLite Micro + CMSIS-NN;
- ExecuTorch Cortex-M + CMSIS-NN;
- microTVM where a reproducible port exists;
- Symthaea HDC-only;
- compact CfC-only;
- compact HDC-LTC hybrid.

Keep CPU-only and accelerator/NPU configurations separate. Meet the official quality threshold before comparing speed/energy. Publish flash, peak SRAM, latency distribution, energy/inference and quality.

### Track B — temporal state and adaptation

Goal: test the capabilities that make Symthaea different.

Use a causal stream with:
- stationary warm-up;
- sudden, gradual and recurring concept drift;
- irregular timestamps and jitter;
- missing sensors and feature corruption;
- open-set/novel events;
- device/session shift;
- label budgets 0, 1, 5 and 20 per new regime;
- reset, cold-start and recovery periods.

Baselines:
1. static INT8 DS-CNN/TCN/GRU;
2. adaptive TinyML baseline (prototype/centroid or supported online update);
3. threshold/EMA controls;
4. HDC-only;
5. CfC-only;
6. HDC-LTC hybrid;
7. hybrid with confidence-gated learning.

Use prequential evaluation: predict, score when the delayed label becomes available, and only then update. No future labels, test-session normalization or hidden labels in the feature path.

Metrics:
- running balanced accuracy/F1 and false alarms per hour;
- event miss rate and detection delay;
- samples and wall time to recover a fixed fraction of pre-drift quality;
- retention/forgetting across earlier regimes;
- open-set AUPRC/AUROC and selective risk versus coverage;
- Brier score/ECE with a frozen calibration protocol;
- adaptation energy and bytes written;
- p50/p95/p99 latency and peak SRAM/flash;
- energy per accepted correct deadline-met event.

### Track C — active sensing / full-event efficiency

Goal: test whether Symthaea can save energy by deciding when to sense and compute.

Compare fixed-rate, periodic-adaptive, event-triggered, prediction-error-triggered and value-of-information sensing. Include a low-power threshold control. Match application quality, false-alarm limits, missed-event cost, latency and duty-cycle/workload.

Metrics include sensor samples/hour, sensor-active duty cycle, event energy, wake/sleep energy, radio bytes/retries, communication and adaptation energy, missed events, deadline misses, peak RAM/flash and uncertainty.

A battery-life result is a separate workload projection; it is not a direct measurement.

## Key implementation reality: current HDC-LTC is too large at default width

The standalone crate `symthaea-hdc-ltc` uses continuous hypervectors backed by `Vec<f32>`. At 16,384 dimensions, one vector has:

`16,384 × 4 bytes = 65,536 bytes = 64 KiB`

Each current neuron stores eight such vector payloads: state, weight, input mask, tau modulator, gate weight, gate bias and two momentum accumulators. A default-width neuron therefore needs 512 KiB just for those vector payloads.

The default network is 4+8+4 = 16 neurons over three layers. The network also retains one layer binding and one cached output vector per layer. The persistent payload lower bound is:

`(16 × 8 + 3 × 2) × 16,384 × 4 = 8,781,824 bytes = 8.375 MiB`

One input vector adds 65,536 bytes, for a minimum working vector payload of 8,847,360 bytes (8.4375 MiB). This excludes:
- Vec headers and allocator overhead;
- temporary/intermediate vectors;
- sensor staging and outputs beyond one input;
- runtime, firmware, stacks, DMA and interrupt state;
- alignment, fragmentation and linker placement.

At 256 dimensions, the same default network has 137,216 bytes of persistent vector payload plus a 1,024-byte input vector: 138,240 bytes before those excluded costs. That may fit some MCU SRAM budgets with enough reserve, but it does not establish that the complete firmware fits. The HDC-LTC crate currently uses heap-backed Vec and `rand`; it is not a no_std, statically bounded MCU runtime.

The new CLI provides a payload lower-bound preflight:

```bash
cargo run -p symthaea-architecture-engineering --bin hdc_ltc_memory_budget -- 16384 4,8,4 1048576 131072
```

A compact profile can be checked separately:

```bash
cargo run -p symthaea-architecture-engineering --bin hdc_ltc_memory_budget -- \
  256 4,8,4 262144 65536
```

It outputs the estimated vector payload, input buffer, protected reserve and a conservative fit/no-fit result. A fit is only a lower-bound feasibility signal, not a linker-map approval.

## Paradigm shifts worth pursuing

### 1. Split fast state from associative memory

Do not make every state vector 16K float. Use a 32–256D continuous liquid core for fast dynamics, coupled to a separate, compact HDC memory. Keep the memory packed (binary/ternary/int8) and use it only when recall or binding is useful.

### 2. Persistent/event-driven computation

Process new events and elapsed time instead of recomputing an entire fixed sensor window. The CfC update is independent of the size of dt, but it is still O(D) in vector dimension and scales with neuron count. O(1) in elapsed time is not O(1) in computation.

### 3. Confidence-gated adaptation

Predict, measure error, estimate drift and uncertainty, then adapt only when the update is justified. Bound update size and retain a safe fallback/abstention path. This must be compared against adaptive TinyML baselines.

### 4. Active sensing

Use expected information gain / value of information to choose sensor cadence, rather than running every sensor at a fixed rate. Measure the additional compute overhead; an intelligent scheduler can consume more energy than it saves.

### 5. Hardware-aware representation selection

Let the application choose between conventional INT8 tensors, sparse CfC, compact HDC, or a hybrid. A good system does not insist that every task run on its most distinctive architecture.

### 6. Local fast path + optional higher-level deliberation

Run a small, bounded controller locally. Escalate rare ambiguous/novel cases to a phone or gateway. Include communications, privacy and availability costs in the comparison so offloading is not hidden.

## Proposed research gates (aspirational, not current results)

| Track | Gate |
|---|---|
| A | Meet the official MLPerf Tiny quality threshold, then publish comparable energy/latency/memory |
| B | Retain at least 95% of pre-drift quality after a regime shift and recover to 95% within a preregistered label/energy budget |
| B | Keep earlier-regime retention loss within 5 percentage points |
| C | At least 20% lower energy per successful event than the best matched fixed-rate baseline at equal quality/deadline, with uncertainty interval excluding zero improvement |
| All | 100% exact artifact/runtime/compiler/data/target/measurement provenance for promoted results |
| MCU | Fit peak SRAM and flash with a protected reserve; no unbounded allocations in MCU deployment tier |

These are hypothesis gates. No result should be promoted because it satisfies a target chosen after seeing the test set. Publish full Pareto fronts; do not hide tradeoffs in one weighted score.

## Causal data path and prequential audit

The synthetic stream now separates two different data types:
- `TemporalObservation`: timestamp, sensor features and missing-feature mask only; no ground-truth label field exists on this model-facing type. Dropped sensor values are actually zero-filled before the observation is emitted; labels still use the latent complete signal.
- private `DelayedGroundTruth`: evaluator-only label, release timestamp, regime transition and OOD metadata.

The combined evaluator-owned dataset intentionally does not implement `Debug`, `Serialize` or `Deserialize`; both its delayed-label field and drift-schedule configuration are private. Candidate implementations should receive only the observation slice, not the evaluator-owned container. This is an API/data-boundary improvement, not a process sandbox: a real deployment benchmark must physically isolate the evaluator label and schedule channels.

A prequential execution trace is a time-ordered sequence of:
1. `Prediction` — timestamp, predicted class/abstention, positive-class Q15 probability and the model-state digest used;
2. `LabelReleased` — timestamp and delayed ground truth, after the frozen availability time;
3. `ModelUpdate` — timestamp plus before/after state digests.

The auditor enforces one prediction and one delayed-label release per event, prediction strictly before that event's label-availability timestamp, matching label values at release, globally non-decreasing action time, no update before label release, one update at most per event, and a continuous model-state digest chain. It computes accuracy, balanced accuracy where both classes occur, coverage/selective accuracy, Brier score, false alarms/hour, missed events/hour, prediction latency and update count.

This is a chronology/provenance check, not a sandbox that proves a candidate binary could not read the evaluator's memory. In a physical benchmark, the observation process and label evaluator should be isolated (separate processes/MCU or data interface) and the execution/action trace independently collected. State hashes bind claims but do not prove that the recorded model state was honestly produced.

## Current implementation status

The new Rust module currently provides:
- source-layout HDC-LTC memory payload estimation and SRAM reserve assessment;
- deterministic temporal stream generation with sudden/gradual/recurring drift, timestamp jitter, true zero-filled feature dropout, out-of-distribution events and delayed labels;
- model-facing observation records with no label field, and an evaluator-owned combined dataset whose delayed-label field is private and which does not implement Debug/Serialize;
- dataset source replay and digest validation over a private canonical representation;
- a frozen three-track manifest with split digests and required baseline-family checks;
- prequential execution-trace auditing with prediction-before-label availability, delayed label release, post-label update ordering, a continuous model-state SHA-256 chain, and event-level accuracy, balanced accuracy, coverage, selective accuracy, Brier score, false alarms/hour, missed events/hour and prediction latency;
- exact hash commitments from the prequential trace to per-run energy and metrology receipts.

The CLI reports HDC-LTC memory preflight. The next increment is to verify the contents of the bound phase-energy and metrology receipts from #9755/#9744, compute adaptation/recovery and retention metrics by regime, and evaluate all candidates on matched tracks to return a multi-objective Pareto frontier. Fourteen synthetic tests are authored but not compiled or executed in this authoring environment. No on-device result or SOTA claim is made.

## Sources

- MLPerf Tiny v1.4 results (July 2026): https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/
- MLPerf Tiny official task definitions and quality targets: https://mlcommons.org/benchmarks/inference-tiny/
- Arm, Bringing ExecuTorch to Arm Cortex-M CPUs (19 October 2026): https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/bringing-executorch-to-arm-cortex-m-cpus
- TensorFlow Lite Micro: https://github.com/tensorflow/tflite-micro
- MLPerf Tiny repository: https://github.com/mlcommons/tiny
- Jose et al., A Systematic Review of TinyML Techniques for Efficient Inference and On-Device Learning, IEEE Access (23 September 2026): https://doi.org/10.1109/ACCESS.2026.3736856
- Configurable Hardware Acceleration for HDC Extension on RISC-V, IEEE Transactions on Computers (February 2026): https://doi.org/10.1109/TC.2025.3642250
- HyperMetric: Efficient Hyperdimensional Computing With Metric Learning for Robust Edge Intelligence (2026): https://doi.org/10.1109/TCAD.2025.3579323
- On-device training and pruning for energy saving and continuous learning on resource-constrained MCUs, Future Generation Computer Systems (March 2026): https://doi.org/10.1016/j.future.2025.108194
