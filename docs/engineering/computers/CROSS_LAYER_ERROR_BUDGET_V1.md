# Cross-layer error budgeting for physical computing — v1 research contract

**Owner:** RELIABILITY-COMPUTE-001 #10993, composed with PHYS-COMPUTE-ADAPT-001 #10933.  
**Status:** software reference draft; no physical model calibration or hardware result.  
**Purpose:** replace a single “fault rate” or “analog precision” number with a typed, compositional account of where errors enter, how the physical/computational pipeline transforms them, and how they affect task correctness.

## 1. Research motivation

The 2026 cross-layer review by Ding et al. organizes analog in-memory computing non-idealities by their effects on computation: weight perturbations, multiplicative distortions, nonlinear transfer effects, additive errors, and quantization/accumulation errors. It traces those errors across device, array, circuit, data-conversion and system levels. This matters because two systems with the same headline device variation can produce very different task errors after readout, accumulation, calibration and mapping.

Reference: Ding et al., *From device non-idealities to computation accuracy: A cross-layer review of error mechanisms and mitigation in analog in-memory computing*, Journal of Semiconductors (2026, in press / uncorrected proof dated 2026-11-03), DOI: https://doi.org/10.1088/1674-4926/26060006

The 2026 IEEE Micro paper *Let Analog Be Analog* argues that direct transfer of digital abstractions to analog accelerator design can mishandle analog-specific error sources. This motivates a native analog execution track in addition to same-algorithm digital/analog comparisons: https://doi.org/10.1109/MM.2026.3666132

The September 2026 readout review also highlights application-driven dynamic resolution and an area-aware compute-in-memory figure of merit. This is a reminder that ADC/DAC/readout precision, sampling rate, area and peripheral power are part of the compute architecture, not bookkeeping outside it: https://doi.org/10.1016/j.mee.2026.112466

## 2. Typed error channels

Preserve these mechanisms separately:

| Error kind | Physical / computational examples | Typical observable |
|---|---|---|
| Weight perturbation | Conductance programming variation, stuck-at devices, drift | Effective matrix deviation |
| Multiplicative distortion | Gain error, reference/current scaling, line-drop attenuation | Relative output scaling |
| Nonlinear transfer residual | Device I–V nonlinearity, ADC/DAC INL, saturation | Input-dependent transfer residual |
| Additive offset | Sense-amplifier offset, bias/reference error | Output offset |
| Input conversion | DAC quantization, input coding error | Perturbed operand |
| Output quantization | ADC quantization and clipping | Quantized result |
| Accumulation | Finite accumulator width, rounding and truncation | Sum / reduction error |
| Read noise | Thermal, shot, RTN and read-to-read noise | Repeated-read distribution |
| Timing jitter | Clock phase, aperture uncertainty, settling variation | Time-domain error |
| Spatial fault | Correlated row/column/block failures, gradients | Geometry-dependent error |
| Other | A mechanism with a documented definition | Mechanism-specific |

A fault rate is not an error magnitude. A 10% stuck-cell rate, 10% conductance coefficient of variation, 10% ADC quantization error and 10% output noise are different objects and must never be collapsed into one number.

## 3. Compositional operator model

For a matrix-vector operation, one useful first-order decomposition is:

`y = Wx`

with a candidate physical result approximated by:

`ŷ = (1 + G)(W + ΔW)(x + δx) + b + n + q + a + r_nl`

where the terms represent gain/multiplicative distortion, weight perturbation, input-conversion error, additive offset, read noise, output quantization, accumulation error and nonlinear-transfer residual.

This expression is a bookkeeping decomposition, not a universal device equation. The actual order of nonlinear transfer, clipping, quantization and accumulation must follow the implementation. A nonlinear operation generally cannot be commuted across a noise or quantization term without changing the model.

A conservative first-order norm budget can be built by assigning each local contribution:
- a typed mechanism;
- a local magnitude and units;
- a propagation gain to the declared operator output;
- either a deterministic upper bound or a standard deviation;
- a calibration / measurement / model provenance reference;
- a dependence assumption.

The current Rust artifact, `research/physical_compute/error_budget_v1.rs`, accepts these explicit local magnitudes and gains. It computes:
- a linear sum of deterministic output bounds;
- a conservative L1 sum of stochastic standard deviations when correlations are unknown;
- an RSS combination only when the caller declares an independence assumption with an evidence digest;
- a second propagation through a caller-supplied task-sensitivity bound.

**Important:** the current utility checks the independence digest's syntax only. It does not retrieve or validate the cited evidence and does not establish independence. The RSS field is therefore named as conditional on a declared assumption, and is absent by default. Even a valid independence declaration requires distributional assumptions before RSS becomes a confidence guarantee.

## 4. Task sensitivity and correctness

Local physical error is not the same as task error. Each workload needs an explicit sensitivity certificate or bound.

Examples:
- Linear solve: perturbation sensitivity depends on conditioning and the relevant backward-stability assumptions; do not substitute `κ(A)` as a universal sensitivity multiplier without deriving the applicable bound.
- HDC retrieval: task sensitivity depends on the margin between the best and second-best Hamming distances, not merely vector dimension.
- Temporal state: error can grow, decay or remain bounded depending on the dynamics, horizon and stability/Lipschitz properties.
- ODE/PDE: local defect and parameter perturbations need a stability or Grönwall-style argument before being translated into a trajectory-error bound.
- Classification: feature perturbation should be related to the decision margin and task-level error, not only per-layer RMS error.

The compiler should compare mitigation by its reduction in task-level error, not by reducing an isolated device variance.

## 5. New optimization problem: allocate the next unit of precision

For candidate mitigation actions j, estimate:
- marginal reduction in task error / risk;
- marginal area, energy, latency, conversion, calibration and service cost;
- whether the evidence for the error channel is measured, derived or synthetic;
- whether the action improves a deterministic bound or only a stochastic variance model.

Then solve a constrained allocation problem, for example:

`minimize total system cost`

subject to:

`deterministic_task_error_bound + risk_budget(stochastic_terms) ≤ task_tolerance`

and the declared latency, energy, area/resource and availability constraints.

Candidate actions include:
- selectively increasing ADC resolution only on sensitive outputs;
- using lower precision or fewer readout bits where task margins permit;
- adding bit slices, RNS, ECC or mixed-precision refinement only where their marginal value is positive;
- relocating critical weights away from measured fault clusters;
- separating high-sensitivity operations onto independent power/clock/controller islands;
- calibrating only when the expected reduction in risk justifies its energy and latency cost.

This makes adaptive readout resolution and physical precision allocation compiler decisions rather than fixed global settings.

## 6. Required benchmark ablations

Compare:
A. Unmodeled nominal hardware / fixed precision.  
B. One aggregate error percentage (deliberately weak baseline).  
C. Layered error sources with independent parameters.  
D. Layered sources with measured/derived propagation gains.  
E. Sensitivity-aware mitigation allocation.  
F. Dynamic readout resolution / calibration.  
G. Combined error-aware compiler with independent verifier and recovery.

For each, keep the underlying workload, fault topology, physical resource budget and acceptance contract fixed.

Report:
- error by physical layer and mechanism;
- operator output error and task-level quality;
- uncertainty and cross-device/cycle variation;
- false accept/reject and fault-detection coverage;
- effective bits / ENOB where meaningful, without using ENOB as a substitute for task quality;
- ADC/DAC/readout power, area, sampling rate, conversion latency and control overhead;
- calibration frequency, reprogramming and data movement;
- p50/p95/p99 task latency and joules per correct accepted task;
- held-out performance under new fault geometry, temperature, drift and operating conditions.

A technique only earns credit if its complete cost is included. Lower ADC power is not an advantage if it makes the task violate its correctness contract.

## 7. Software reference tests

The current dependency-free reference has five authored tests:
- deterministic bounds add linearly and propagate through a task sensitivity factor;
- unknown correlation uses the conservative L1 sigma bound;
- RSS is exposed only under an explicitly declared independence assumption with a syntactically valid digest;
- invalid magnitudes, sensitivity and digest declarations fail closed;
- an empty error ledger returns a zero budget.

Test command:

```sh
rustc --edition=2021 --test research/physical_compute/error_budget_v1.rs -o /tmp/error_budget_tests
/tmp/error_budget_tests
```

The Rust tests are authored but have not been compiled or executed in the current environment.

## 8. Claim ceiling

This is an error-budget calculator over supplied magnitudes and sensitivity factors. It does not derive those values from device physics, infer correlations, verify calibration, or guarantee a probability of task correctness. Synthetic values are for logic tests only. Measured claims require identified hardware, raw repeated observations, environmental metadata, uncertainty analysis, and independent reproduction.
