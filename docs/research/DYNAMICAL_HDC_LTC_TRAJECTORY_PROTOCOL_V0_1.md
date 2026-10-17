# Dynamical HDC-LTC Trajectory Protocol v0.1

**Status:** preregistered target contract; not executed. Thresholds below are proposed starting gates to freeze before measuring candidates, not achieved results or universal scientific constants.

## Purpose

This protocol evaluates the active `symthaea-core::hdc::hdc_ltc_unified` architecture as a dynamical system. It must not conflate packed BinaryHV kernels with continuous-f32 state evolution, or static similarity with time-evolving trajectory preservation.

## Architecture axes

- **Semantic width:** 16K / 32K / 64K / 128K / 256K. The first three are baseline tiers; 128K and 256K remain exploratory.
- **Local dynamical width:** candidate 1K / 4K / 16K / 32K, independently declared where manifold compression is tested.
- **Representation:** ContinuousHV f32. One vector at 128K is 512 KiB and one at 256K is 1 MiB; neither number is a network working-set measurement. Count weights, state, gates, tau modulation, allocations and peak temporaries separately.

## Comparison design

Compare active scalar/SIMD, fused closed-form, bounded iterative, and SIMD/cache-tiled candidates. Use fine-step RK4 as an independent reference only when it can implement the same equations and input schedule; refine its step size until the change in the reference is less than 10% of the candidate's allowed error. Streaming/chunked execution is eligible only if it preserves update dependencies.

For semantic-width versus dynamical-width experiments, compare full-width dynamics against matched-width random projection, deterministic orthogonal/nested projection where applicable, and manifold projection with an explicit residual/novelty path. Learned task-specific projection comes only after the non-learned controls. Approximate/analogue profiles are versioned separately.

## Workload coverage

At minimum, cover fixed step sizes, jittered irregular steps, burst-gap schedules, abrupt input discontinuities, and long-horizon recovery after a disturbance. Evaluate temporal associative recall, streaming sequence classification, temporal prediction/control, and a representative edge-state update when each task has a pinned corpus and an executable reference.

Every run pins the source commit and dirty-state, toolchain/build flags, hardware/OS/power mode, corpus hash, seed, configuration hash, and measurement boundary.

## Proposed gates to freeze before results

### A. Numerical-integrator fidelity

Only apply this gate when an independent, refinement-checked reference exists for the same equations. Initial target: trajectory NRMSE ≤ 1e-3, maximum normalized state error ≤ 1e-2, and relative tau/gate/equilibrium distortion ≤ 0.5%. The definition of normalized state error and epsilon protection for near-zero reference states must be fixed in the runner. Reject NaN/Inf states, unbounded runs, and mutation after invalid or over-budget requests. If the reference cannot meet the refinement criterion, the comparison is unresolved—not a pass.

### B. Compressed-dynamics task preservation

Compare on held-out task sequences against full-width dynamics. Initial target: classification loss no more than 1 percentage point; control-return loss no more than 2% relative; reconstructed trajectory NRMSE ≤ 1e-2. Also require zero non-finite states and zero deadline misses for tasks declared hard real time. These gates are independent: a trajectory match does not prove task quality, and a task score alone does not prove dynamical equivalence.

The claimed compressed path must not hide full-width recurrent computation or move the full source vector through a residual path at every step. Report the quality-versus-local-state-width Pareto frontier and the residual traffic explicitly.

### C. End-to-end system improvement

To claim a system-level breakthrough, first pass the applicable task-quality and numerical/trajectory gates. Then require either ≥ 2.0× end-to-end throughput **or** ≥ 30% lower measured system energy versus the same active incumbent. Keep p95 and p99 latency regression at ≤ 5% unless a stricter declared task deadline applies. Declare a real memory budget for the target platform, including resident state and peak temporaries. These are candidate targets, not evidence that the targets have been met.

## Measurement discipline

For each configuration: at least 5 data seeds, 10 warm-ups, at least 30 independent measurement blocks with at least 1,000 timed invocations per block. Report p50/p95/p99 and a 95% bootstrap interval over independent blocks, stating the resampling unit. Measure allocation and transfer overhead, projection/lift, input/output processing, initialization policy, and peripherals whenever they belong to the deployed system; report energy-sensor calibration/uncertainty.

MLPerf Tiny's current published methodology illustrates why fixed task/quality targets and explicit measurement boundaries matter: vendors' energy figures can differ when one includes peripherals/startup/I/O and another measures only the inference core ([MLPerf Tiny v1.4 results and methodology](https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/)). Recent analogue in-memory HDC research is relevant inspiration, but its physical device, task and representation are not a proxy for Symthaea's continuous dynamics ([Nature Communications, 2026](https://www.nature.com/articles/s41467-026-76067-5)).

## Evidence ladder

- **E0:** schema/protocol only; no empirical claim.
- **E1:** operator correctness/cost; no trajectory claim.
- **E2:** trajectory fidelity against a declared reference.
- **E3:** held-out task quality.
- **E4:** end-to-end measured system cost on a declared platform.
- **E5:** independent hardware evidence or external replication, limited to what was actually measured.

Missing, duplicated, malformed, hash-mismatched or unpinned evidence fails closed. An unexecuted or unmeasured matrix cell remains unknown, not a pass and not a failure.

## Canonical owners — compose, do not duplicate

- [#6560](https://github.com/Luminous-Dynamics/symthaea/issues/6560): state-continuous resolution changes.
- [#6561](https://github.com/Luminous-Dynamics/symthaea/issues/6561): trajectory-resolution matrix.
- [#6619](https://github.com/Luminous-Dynamics/symthaea/issues/6619) / [#6630](https://github.com/Luminous-Dynamics/symthaea/pull/6630): fail-closed operator-to-trajectory dependency.
- [#6631](https://github.com/Luminous-Dynamics/symthaea/issues/6631): resource admissibility.
- [#7374](https://github.com/Luminous-Dynamics/symthaea/issues/7374): bounded timestep work and invalid-input rejection.
- [#3348](https://github.com/Luminous-Dynamics/symthaea/issues/3348): manifold-compressed liquid dynamics.
- [#7749](https://github.com/Luminous-Dynamics/symthaea/issues/7749): cross-layer dynamical HDC-LTC co-design.

## Important limitation

This is a preregistration artifact only. No Rust tests, trajectory trials, GPU/FPGA runs, power measurements, or hardware measurements have been executed by adding this contract. Do not record an E1–E5 receipt until its evidence exists.
