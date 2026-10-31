# Timestep Scaling Paradox and Four-Clock Qualification (v0.1)

**Status:** deterministic shared-input generator and K-profile metadata implemented; no model solver, training adapter, benchmark run, or hardware measurement  
**Tracking:** NEURO-SILICON-009 / issue #9496  
**Related:** NEURO-SILICON-007 / #9394 / draft PR #9395  
**Code:** `research/temporal-bench/src/time_resolution.rs`

## 1. Research motivation

Ye et al., “Resolving the Timestep Scaling Paradox in Spiking Neural Networks with a Timestep-Scalable Neuron Model,” ICML 2026, PMLR 306:148430–148449, report a timestep-scaling paradox: increasing SNN simulation timesteps can degrade accuracy, which the paper links to long-term temporal gradient vanishing and weakened cross-timestep dependencies. Their TS neuron introduces long-term memory reconsolidation and periodic temporal forgetting; the paper reports improvements as timestep counts scale.

Canonical paper record: https://proceedings.mlr.press/v306/ye26k.html

This is an important baseline for Symthaea's temporal programme. We must test the mechanism rather than assume continuous-time dynamics, HDC state, or adaptive integration automatically avoid it. The TS neuron is a paper-specified comparator at present: the official implementation and supplement have not yet been pinned and audited. No paper result is imported as a Symthaea measurement.

## 2. The four-clock model

A common experimental error is to overload "timestep" with four different quantities. The qualification separates them:

1. **World/physical time:** how long the modeled process lasts and when its events occur.
2. **Observation time:** when the sensor provides data; changing this changes the information available to the candidate.
3. **Internal update time:** solver/neuron substeps performed between the same two observations; changing this changes computation and numerical resolution, not sensor information.
4. **Credit-assignment horizon:** how many internal updates and how much physical time a training method must bridge.

These clocks are varied independently and are stored in each experiment manifest. A comparison is invalid if it silently increases the physical horizon, supplies more observations, or changes the target alignment when only internal K was intended to vary.

## 3. Implemented synthetic input family

`research/temporal-bench/src/time_resolution.rs` defines one deterministic continuous-time signal family:

- fixed physical duration: 4,000,000 microseconds (4 seconds);
- a fixed-point triangle-wave signal;
- period before transition: 64,000 microseconds;
- period after transition: 96,000 microseconds;
- a frequency-regime transition at 2,000,000 microseconds;
- a phase jump at the transition;
- bounded deterministic noise keyed by absolute physical timestamp.

The public task is a sampled observation stream. The oracle contains exact phase, latent sample and observed sample at every observation timestamp. Both have canonical SHA-256 digests and are verified against the physical-time signal function. Seed-bound replay is available through `verify_time_resolution_seeded`.

### Observation-rate variants

Reference observation intervals of 10,000, 20,000, 40,000 and 80,000 microseconds all divide the same four-second physical duration. The signal and noise are keyed by absolute time, so observations at shared physical timestamps are byte-identical across the variants. This lets the experiment measure information-rate effects without changing the underlying process or noise at matched timestamps.

This is a deterministic synthetic control, not a sensor-data benchmark, physical neuron implementation, or reproduction of the TS-neuron paper's tasks.

## 4. Internal-substep sweep

For a fixed public task, the generator exposes internal profile arms:

`K = 1, 2, 4, 8, 16`

For an observation interval `dt_obs` and K that exactly divides it:

```text
internal_dt = dt_obs / K
total_internal_updates = observation_count * K
```

The task, observation count, physical start/end interval, input bytes and SHA-256 digest are identical across these K profiles. Only the declared internal update budget changes.

The helper rejects unsupported K values and intervals that cannot be divided exactly. It does not actually execute a model integrator at those substeps. Candidate adapters must:
- name the input interpolation/hold rule;
- use no future observation at an internal substep;
- keep the same physical start/end times and task targets;
- record the integrator and precision;
- count all internal state updates and intermediate buffers;
- preserve the original task digest across K.

**Implementation boundary:** v0.1 creates the common physical-time input and metadata for K profiles. It does not yet implement the solver loop, an HDC-LTC/CfC model adapter, a TS-neuron adapter, a gradient/eligibility measurement, or an energy meter.

## 5. Experimental design

### A. Fixed-world-time / internal-substep sweep

Hold the complete public stream fixed. Run K = 1, 2, 4, 8, 16 where supported. Report:
- task quality and worst-horizon quality at each K;
- numerical error against the hidden physical-time oracle;
- gradient/eligibility sensitivity and training stability for trainable arms;
- total internal updates, wall time, memory, bytes moved and measured energy.

Do not give a candidate interpolated future sensor observations. Internal substeps may only use the current legal observation and the declared causal hold/interpolation scheme.

### B. Observation-rate sweep

Hold physical duration and continuous-time source fixed. Sample at the declared intervals. Report quality as a function of information rate, including aliasing/undersampling controls. Keep it separate from the K sweep.

### C. Physical-horizon sweep

Hold observation rate and internal dt fixed while varying the length of the temporal dependency. Reuse T2 delayed cue, T3/T4 multiscale/change-point, T8 DMS and T9 evidence-accumulation workloads. This isolates long-range dependency from numerical resolution.

### D. Credit-assignment sweep

For differentiable SNN/recurrent models, compare gradient norm and cosine alignment at different temporal depths against a small full-horizon reference. For online learners, report eligibility-trace bytes, update time and time-to-target quality. Distinguish exact BPTT/D-RTRL gradients from factorized pp-prop approximations and local e-prop traces.

## 6. Required baselines

- HDC-LTC/CfC incumbent, using the existing active equations.
- Refined numerical reference on the same equations and physical trajectory.
- Vanilla LIF and ALIF on compatible event tasks.
- STDL on compatible static/event vision training tasks.
- BrainTrace D-RTRL, pp-prop/ES-D-RTRL and e-prop on compatible online-learning tasks.
- TS neuron from Ye et al. when official source code/supplement are located, pinned and reproduced.
- A small GRU/LSTM or ODE-RNN control to ensure the SNN comparison does not hide an easier/stronger non-spiking baseline.

The TS-neuron baseline is currently paper-specified/source-not-pinned. Do not implement or claim to reproduce it from the abstract alone. Keep training methods and inference architectures distinct.

## 7. Metrics and benchmark targets

Publish separate task-quality, numerical, learning, resource and integrity results. Do not collapse them into one weighted score.

| Dimension | Report | Proposed stretch target (not a result) |
|---|---|---|
| Fixed-world-time quality | accuracy/error by K and worst temporal horizon | no unexplained quality degradation as K increases |
| Numerical fidelity | error against exact physical-time oracle | monotonic convergence to declared tolerance as K increases |
| Temporal credit | gradient norm and cosine alignment by depth | reduce long-horizon gradient collapse vs matched vanilla LIF/ALIF |
| Online adaptation | time-to-target quality, change recovery, forgetting | 2x faster recovery or 30% lower measured energy per correct decision without worse leakage/forgetting |
| Resource | updates/physical second, state/trace bytes, peak memory and traffic | better quality/energy Pareto frontier, not just lower MACs |
| Deployment | p50/p95/p99 latency, energy per correct decision, energy per stream-hour incl. idle | 20% lower error at equal whole-system energy on compatible long-horizon tasks |

These are research goals to test, not benchmark results. Published TS-neuron SOTA claims are not comparable until the task definitions, splits, training budgets and hardware are matched.

## 8. Validation controls

- All K variants share the exact task digest, sample count and physical duration.
- Same absolute physical timestamps produce identical noisy observations across observation-rate variants.
- Changing K must not change public sensor bytes.
- No future sample may be read by an internal substep.
- A negative control that lengthens physical time while claiming only K changed must be rejected.
- A negative control that inserts extra observations into a fixed-input K sweep must be rejected.
- Model-specific substep execution and training tests must run before any TSP claim.

## 9. Current status and claim boundary

Implemented:
- deterministic continuous-time synthetic source;
- same physical signal sampled at multiple observation rates;
- fixed-world-time K = 1/2/4/8/16 metadata arms with exact update counts;
- SHA-256 public/oracle digests, semantic verification and seed-bound replay;
- ten unit tests authored for deterministic generation, seed replay, K/input separation, fixed duration, cross-rate matched samples and mutation/invalid-parameter controls.

Not implemented:
- actual model solver at each K;
- HDC-LTC/CfC or SNN training adapters;
- TS-neuron code/reproduction;
- gradient/eligibility traces and credit-horizon instrumentation;
- matched baseline runs, hardware measurements or energy results;
- independent evaluator OS isolation and signed receipts.

The current code therefore establishes a reusable, controlled *input and resolution experiment design*, not a result that Symthaea solves the Timestep Scaling Paradox. The tests are authored but unrun because this environment does not provide Cargo/rustc/rustfmt and external toolchain download is unavailable.
