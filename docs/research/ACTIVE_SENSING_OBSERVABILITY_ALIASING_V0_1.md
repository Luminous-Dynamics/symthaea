# Active Sensing Observability and Aliasing Qualification (v0.1)

**Tracking:** NEURO-SILICON-012 / #9563  
**Evaluator implementation:** `research/temporal-evaluator/src/sensing.rs`  
**Candidate policy:** #9616 / `research/temporal-sensing`  
**Security dependency:** #9437 / #9621  
**Status:** evaluator-side deterministic controls authored; no Rust tests or schedule experiments executed

## 1. Why a separate evaluator is necessary

The candidate-side scheduler chooses observation times using only received history and candidate-estimated risk/uncertainty/novelty/value. The evaluator alone knows the hidden transient windows and continuous latent paths.

A schedule with no sample during a transient has not directly observed that event. That is not the same as a model receiving evidence and then failing to recognize it. The evaluator must report both conditions separately.

Likewise, distinct latent processes can produce identical values at every scheduled observation and diverge between them. A candidate cannot uniquely reconstruct which path occurred from those samples alone unless other candidate-visible evidence resolves the ambiguity. The benchmark must mark such cases as observationally aliased rather than reward hallucinated reconstruction.

## 2. Transient observability scorer

The evaluator module consumes:
- episode duration, maximum gap, observation/energy budgets and an evaluator-declared minimum detectable amplitude;
- candidate observation timestamps;
- evaluator-owned hidden transient intervals and amplitudes;
- candidate event decisions and decision timestamps;
- measured energy when an energy budget is declared.

Each event uses a half-open interval `[start_us, end_us)`. An event is directly observable if at least one observation timestamp falls inside that interval.

The scorer separates:
- directly observable events;
- directly observed and detected events;
- observable but missed events;
- events with no direct observation;
- candidate inferences without a direct sample;
- false detections for unknown event IDs;
- detection latency from event onset.

It also reports event-observability coverage and recall among directly observable events. Observation budget overrun, energy overrun and maximum-gap deadline miss remain explicit flags. An event with no sample during its transient is not counted as a conventional model miss; it is counted as unobserved. If a candidate claims it anyway, that is kept as an inference without direct evidence, not credited as an observed hit.

The evaluator rejects duplicate event/decision IDs, out-of-range or nonmonotonic observation times, invalid event intervals, future-event decision leakage and missing energy measurement when an energy budget is declared.

## 3. Exact alias-pair metamorphic control

The module accepts two evaluator-owned piecewise-linear signals on integer microsecond timestamps and integer milli-unit values.

The control:
1. evaluates both signals at every candidate observation timestamp;
2. verifies that their values are equivalent within a declared integer tolerance;
3. evaluates both signals at hidden probe timestamps strictly between observations;
4. requires at least one hidden probe where their values differ beyond tolerance.

Interpolation and comparison use checked integer rational arithmetic rather than floating-point equality. The result records observed timestamp count, hidden probe count, maximum hidden difference and divergence locations.

This establishes a concrete identifiability limit for the declared pair and schedule. It does not claim that all real latent signals are piecewise linear. It gives a controlled negative example where more latent truth exists than the observation schedule reveals.

## 4. Required experiment matrix

Pair the candidate policies in #9616:
- fixed interval;
- deterministic jittered interval;
- multi-rate;
- event threshold;
- novelty threshold;
- uncertainty threshold;
- past-residual threshold;
- value-of-information per observation nJ.

With evaluator schedule controls:
- uniform and quasi-uniform samples;
- clustered and bursty samples at equal count;
- short transients between observations;
- transient phase shifts relative to sampling;
- abrupt/gradual regime changes at adverse phases;
- alias pairs with identical observed values;
- quiet intervals followed by rare critical events;
- MCAR, structured and state-dependent missingness;
- clock jitter, timestamp quantization and delayed arrival.

For every policy, publish:
- TED wrong-belief, abstention and missing-output exposure separately;
- event observability coverage;
- detection recall and miss rate conditional on observability;
- unobserved-event rate and inference-without-direct-sample count;
- false detection rate;
- detection latency by duration/amplitude/phase;
- maximum gap and deadline misses;
- observations, energy and compute;
- retained byte-time and communication traffic;
- p50/p95/p99 decision latency.

Compare both fixed observation/energy budget and fixed decision quality. Do not hide impossible or unobserved cases inside average detection accuracy.

## 5. Source-level implementation

The package exports `pub mod sensing` from the evaluator-only binary crate. Twelve authored tests cover:
- directly observed transient detection and latency;
- observable-but-missed event;
- transient entirely between samples;
- inference without a direct sample;
- future-event leakage;
- invalid timing and missing energy;
- observation/energy/max-gap overruns;
- signals that match sampled values but diverge between samples;
- aliased pair rejected when observed timestamps differ;
- hidden probe and no-divergence controls.

No Cargo build or test run is claimed. The module is not linked into candidate code. The package is a source-level evaluator primitive, not an active-sensing benchmark result.

## 6. Claim ceiling

Observability is defined by the declared sensor/task semantics. A sampled event may be inferred from correlated context, so the scorer reports unobserved-event claims separately rather than universally treating them as false. Alias pairs demonstrate schedule-specific non-identifiability, not a universal impossibility theorem for every model or sensor. No SOTA or energy claim follows without isolated confirmatory evaluation and calibrated hardware measurement.
