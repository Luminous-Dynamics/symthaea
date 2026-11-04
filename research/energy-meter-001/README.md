# Hardware energy measurement and qualification protocol

This directory isolates the exact production energy-counter and energy-receipt
modules from the full Symthaea workspace. It is not a replacement for full
workspace integration tests.

## Evidence classes

Keep these classes separate in every report:

1. **Direct cumulative energy counter** — a device/package/system sensor reports
   accumulated energy over an interval (for example NVML millijoules or Linux
   powercap microjoules).
2. **Sampled power integration** — a power sensor reports watts at a cadence;
   energy is estimated by integrating samples. Publish cadence, interpolation
   rule, missing-sample handling and uncertainty.
3. **External wall meter** — a calibrated meter measures mains or DC input.
   This may include conversion losses, display, fans, memory and other rails
   outside the scope of an on-device counter.
4. **Model estimate** — FLOP count, nominal TDP/clock, architecture or
   theoretical lower bound. This is not a hardware energy measurement.

Never silently combine different classes. A missing counter is Unsupported,
not zero. A package counter is not automatically whole-system energy.

## Receipt contract

One run receipt must bind:

- exact source commit and optional dirty-tree digest;
- workload identity and digest of frozen inputs;
- effective configuration digest;
- device identity/model, driver and runtime versions;
- meter backend, sensor ID, unit and physical scope;
- raw cumulative counter start/end, counter generation and monotonic interval;
- warm-up, measurement iterations, batch size and concurrency;
- completed work units and the exact unit label;
- observed task-quality metric and threshold.

Energy remains in the receipt even when the quality gate fails. A failed quality
gate disqualifies the observation from matched-quality efficiency claims; it
does not delete an unfavourable data point.

The receipt records total observed energy for the selected sensor scope and
exposes a BLAKE3 content fingerprint over its schema version, provenance,
quality gate, raw counter readings and derived interval. The digest is a content
identity, not a digital signature or proof that metadata is truthful. Idle
subtraction is a separately derived view and requires its own idle baseline
receipt, matching sensor, scope, device, power state and measurement protocol.
Do not subtract a GPU idle counter from a CPU package total.

## Controlled workload protocol

For each workload/configuration:

1. Freeze source SHA, model/configuration digest, input corpus, runtime, driver,
   device identity, output cadence and correctness/quality threshold.
2. Record sensor capability and scope before warm-up. Keep unsupported sensors
   explicit.
3. Execute warm-up iterations and disclose whether clocks, compilation, caches,
   allocator state and thermal state were stabilized.
4. Start and stop measurements only after synchronizing asynchronous device
   work. Record raw counter readings and monotonic timestamps.
5. Preserve batch size, concurrency, worker/thread count and work-unit count.
6. Repeat at least 20 independent measurement windows per configuration.
   Interleave baseline and candidate order where practical to reduce drift.
7. Report median, p05/p95, dispersion and a 95% confidence interval. Use
   bootstrap intervals over run-level observations, not individual tokens or
   samples treated as independent when they share one run.
8. Keep failures, timeouts, OOMs and invalid outputs in the denominator.
   Do not discard high-energy runs or failed runs silently.
9. Repeat at matched task quality. An energy reduction purchased by degraded
   accuracy, calibration, robustness or completion rate is not an efficiency
   win.

## Core benchmark metrics

Report the full vector, not a single opaque score:

- total joules per run and sensor scope;
- joules per completed inference, token, update, rollout or decision;
- samples/joule or tokens/joule where work units match;
- average watts over the measurement interval;
- latency p50/p95 and throughput;
- peak memory and, where available, memory bandwidth;
- task quality, calibration, robustness and failure rate;
- energy and latency dispersion plus confidence intervals;
- sensor coverage, idle baseline and unmeasured power domains.

Comparisons must use equivalent work units. Tokens/joule is not comparable to
samples/joule, and GPU joules are not comparable to whole-system joules.

## Initial comparison families

### ML inference

- Reproduce an ML.ENERGY workload where hardware and model access permit.
- Compare at least three JEPA/Symthaea configurations on the same device.
- Include precision, batch size, concurrency, sequence length and memory
  traffic in the configuration digest.
- Compare only when task quality is within a predeclared tolerance.
- Measure both single-request latency and sustained/batched throughput.

### Cognitive workloads

- Choose task-level workloads such as calibrated prediction, memory retrieval,
  online learning update and multi-step planning.
- Report joules per successful task completion, not joules per FLOP.
- Include exact task quality (accuracy, Brier/log score, calibration error,
  retrieval recall/precision, constraint violations) for each family.
- A faster or lower-energy cognitive loop that violates safety or epistemic
  calibration is not a win.

### Physics and simulation

- Use the same equations, initial state, units, event logic, solver tolerance,
  output cadence and horizon for Symtropy and external engines.
- Report work-precision curves, invariant/constraint residuals, event-time error,
  penetration/tunnelling, convergence, failure rate, memory and energy.
- For batch simulation, preserve rollout IDs and seeds independent of thread
  scheduling. Compare ordered output streams against a sequential reference.
- Do not compare dissimilar contact formulations as if they were equivalent.

## External reference baselines

- **ML.ENERGY (NeurIPS 2025)**: realistic inference-energy benchmark and open
  methodology. Its early corpus covered 40 architectures across six tasks and
  reported optimizations up to 40% in selected case studies without changing
  the computed model operation. Use it to reproduce methodology and selected
  workloads, not as a guarantee of expected savings.
- **MLPerf Power (IEEE Micro 2026)**: cross-scale methodology and shared
  samples/joule and tokens/joule metrics. It reports 1,841 reproducible
  submissions across 60 systems. Use its measurement principles and public
  results to calibrate reporting and identify comparable systems.
- **MuJoCo 3.17.0**: batch API results are documented as bit-identical to the
  sequential loop across thread counts. Treat deterministic multi-run replay as
  a comparator target, alongside contact, stiffness and work-precision tests.
- **SUNDIALS 7.9.0**: current ODE/DAE/multirate numerical-solver comparator.
  Use on matched mathematical systems, not as a rigid-body contact comparator.
- **ASME V&V 20**: quantify agreement at specified validation points while
  accounting for both numerical solution uncertainty and experimental data
  uncertainty. Keep verification, validation and calibration distinct.

## Improvement targets

These are proposed programme targets, not achieved results:

- 100% of published energy claims carry a machine-readable receipt and an
  explicit sensor scope.
- 0 unsupported sensors converted to numeric zero or synthetic joule values.
- 20+ repeated measurement windows per published comparison.
- Matched-quality comparisons with predeclared quality tolerance and confidence
  intervals.
- A first optimization goal of at least 10% lower measured energy at matched
  quality on three independent workload families, without unacceptable
  regression in p95 latency, memory, calibration, robustness or failure rate.
- For deterministic simulation, exact output equality where the engine's
  arithmetic order promises it; otherwise a documented tolerance and
  determinism class, never a post-hoc tolerance chosen to make a test pass.

## What counts as an industry-beating result

A claim must specify:
- which baseline engine/model/hardware/runtime/version was compared;
- which task and work unit were measured;
- which sensor domain the energy covers;
- whether the comparison is direct-counter, sampled-power or external-meter;
- whether task quality and uncertainty were matched;
- exact source/config/input digests and raw evidence.

A superiority claim is valid only for the measured comparison domain. No
universal "more efficient than SOTA" statement follows from one device,
one workload or one sensor scope.
