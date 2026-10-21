# ODE-VERIFY-001 — Numerical Contract and Work–Precision Protocol v0.1

**Status:** research protocol; the isolated test harness uses test-only HDC stubs. It runs the checked-in numerical source, but it does not validate production HDC behavior or establish physical-model validity.

## Purpose

Qualify Symthaea's ODE integration boundary before comparing solver performance. The goal is a reproducible work–precision frontier—error achieved versus computational work—not a claim that one integrator wins every problem.

The implementation remains Rust-first. External solvers may be used as independently qualified reference implementations or adapters; they are not production dependencies of this test harness.

## Claim ladder

Keep these separate in every report:

1. **API correctness:** invalid ranges, step sizes, dimensions, states, and solver configurations are rejected.
2. **Numerical verification:** an implementation solves its stated equations to a declared numerical tolerance, with expected convergence and invariant behavior.
3. **Independent numerical comparison:** another solver produces consistent results under matched equations, units, initial conditions, boundary conditions, and tolerances.
4. **Physical validation:** measured data from the intended physical system supports the model within a declared applicability envelope and uncertainty.
5. **Engineering qualification:** the model and workflow are appropriate for a specific use, risk level, and decision.

Passing an earlier level does not imply later levels.

## Benchmark ladder

### A — Contract and hostile-input tests

Required controls include zero timestep, negative/non-finite timestep, non-finite or reversed time interval, overflowed time span/step estimate, mismatched initial-state dimension, NaN/Inf state, malformed adaptive tolerances, integrator output with wrong dimension, non-finite state output, and excessive step/state budgets.

Target: every invalid fixture is rejected explicitly; no invalid case is presented as a successful trajectory; resource bounds are checked before allocation and during adaptive stepping. Do not weaken gates to turn a failure green.

### B — Analytical solutions and convergence

Use the harmonic oscillator first, then extend to linear systems and additional smooth problems with known solutions. Compare the final state and full trajectory against analytic or independently computed high-precision reference values.

For a smooth problem in the asymptotic convergence regime, halve the step size and measure observed order:

- Forward Euler: expected global order approximately 1; error ratio near 2.
- RK4: expected global order approximately 4; error ratio near 16.
- Verlet: expected second-order state accuracy and bounded/oscillatory energy error on appropriate conservative Hamiltonian systems.

The proposed first regression tolerances are diagnostic bands, not universal numerical guarantees. Record absolute and relative errors, norm definition, exact times, step sizes, floating-point type, and excluded singular/transient regimes. For the current harmonic-oscillator RK4 case, the test uses two consecutive step halvings and an error-ratio band of 12–20.

### C — Conservation and long-horizon behavior

For systems with known invariants, report initial value, maximum absolute and relative drift, median/P95 drift, duration, timestep, and drift versus time. Examples are energy and momentum for appropriately configured conservative mechanics and exact mass balance in declared closed systems.

Do not demand conservation from dissipative, driven, stochastic, or chaotic systems where that invariant does not apply. For chaotic dynamics, compare short-horizon error and suitable statistics/attractor properties instead of treating long-horizon pointwise trajectory divergence as automatic solver failure.

### D — Adaptive-step correctness

The recorded simulation time must advance by the **accepted** adaptive step, not the requested maximum step. Retain accepted-step count, rejected-step count if available, derivative/function evaluation count, achieved error estimate, min/median/max step sizes, and any tolerance shortfall. An adaptive run that reaches its minimum step while violating tolerance must not be silently described as meeting tolerance.

**Fail-closed rule:** if RK45 reaches `dt_min` while its embedded error estimate still exceeds `tol`, the checked simulation returns `SimulationError::AdaptiveToleranceNotMet`. The checked path must never turn this into a successful-looking trajectory. The legacy tuple-returning RK45 API remains a panic-on-failure compatibility wrapper; new checked callers should use `try_simulate` / `try_adaptive_step`, or `try_simulate_with_tolerances` / `try_adaptive_step_with_tolerances` for scale-aware control. A minimum-step failure is a model/solver resolution limit to report, not permission to relax tolerance after observing the result.

**Scale-aware extension:** the new `try_simulate_with_tolerances` path uses a component-specific error scale `atol[i] + rtol * max(abs(y_old[i]), abs(y_new[i]))` and a weighted-RMS norm. Each accepted step reports the normalized error, maximum absolute embedded difference, proposed next step, attempt/rejection counts, and derivative-evaluation count. A caller-owned `Rk45Controller` preserves the proposal across steps; the integrator itself remains immutable. The legacy scalar-tolerance APIs retain their existing tuple shape and weighted-infinity acceptance rule for compatibility.

Acceptance controls for the scale-aware path:
- A uniform rescaling of a linear problem and its component absolute tolerances should preserve the accepted/rejected step sequence (within floating-point tolerance).
- The normalized error of every accepted step must be `<= 1`; invalid tolerance lengths, non-finite values, non-positive absolute tolerances and negative relative tolerances are rejected.
- The aggregate derivative-evaluation count must equal six times the total accepted-plus-rejected Cash–Karp attempts.
- The persistent controller's second step must use the previous step's recorded proposal, rather than restart from the original initial timestep.
- Report the controller's work and error diagnostics separately from the physical-model validity claim.


### E — Work–precision comparison

Use public benchmark families represented in [SciMLBenchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/): non-stiff problems (Enright–Pryce, FitzHugh–Nagumo, Lotka–Volterra, Pleiades, rigid-body and three-body dynamics), stiff problems (ROBER, HIRES, Van der Pol, and chemistry/battery examples), and suitable dynamical systems. Add PDE/engineering cases only after choosing the correct qualified model/solver interfaces.

For every common error threshold, report:
- achieved state/trajectory error and invariant diagnostics;
- wall time (repeat runs and report median plus spread);
- accepted steps, derivative/function evaluations, rejected steps where relevant;
- peak memory and allocations where measured;
- hardware, OS, compiler/toolchain, optimization settings, floating-point type, and exact command.

Do not compare solvers at the same nominal timestep if their methods or adaptive behavior differ. Compare at matched achieved error and matched problem semantics. Publish work–precision plots and Pareto fronts; list cases where a method loses as well as wins.

### F — Independent references and physical data

Select a reference per model family and document why it is suitable. Potential anchors include [OpenFOAM verification/validation cases](https://doc.openfoam.com/2306/examples/verification-validation/) for CFD/heat transfer and specialist structural or multibody engines when Symtropy's actual solver adapters are ready. Inputs, version/license, units, tolerances, controls, and provenance must be source-pinned.

Keep manufactured-solution tests (which verify a numerical implementation) separate from experimental validation (which assesses whether the model represents reality). For coupled fields, compare vectors/fields across space and time, not just one scalar summary; use model-appropriate methods such as [ASME VVUQ 20.1](https://www.asme.org/codes-standards/find-codes-standards/vvuq201-vvuq-20-1-multivariate-metric-for-validation) as methodological guidance where appropriate.

## Provisional targets

These are proposed project targets, not published standards or achieved results:

- 100% rejection of enumerated malformed-input controls.
- Zero non-finite or wrong-dimension accepted states in qualified numerical runs.
- Expected observed convergence order on at least three refinements for each smooth analytical test.
- Repeated runs produce deterministic inputs/configurations and stable result manifests; output differences are quantified rather than ignored.
- No performance claim without a matched-error work–precision comparison and independent validation of the reference semantics.
- Before claiming a Pareto-front improvement, preregister cases and thresholds, repeat measurements, report uncertainty/spread, and disclose both wins and regressions. An initial performance objective could be at least 20% lower runtime or memory at matched error on several representative cases; treat this as an aspiration until baselines are collected, never as a gate to fudge.

## Relation to standards and external state of practice

Use the active [NASA-STD-7009B](https://standards.nasa.gov/standard/NASA/NASA-STD-7009) and its [February 2026 implementation handbook](https://standards.nasa.gov/standard/NASA/NASA-HDBK-7009) to structure intended use, model credibility evidence, and the life-cycle record. ASME's [VVUQ standards portfolio](https://www.asme.org/codes-standards/vvuq-standards) provides discipline-specific guidance for CFD/heat transfer, solid mechanics, and uncertainty quantification. These references guide the protocol; they do not certify this project.

## Receipt requirements

Every published comparison should retain: exact repository/source commit, lockfile digest, Rust/Cargo versions, target triple, command and configuration, reference dataset/license/digest, tolerances and norm definitions, raw outputs, machine-readable summary, run identity, negative controls, and an explicit status for missing evidence.

Queued, skipped, formatting-only, and synthetic-known-answer passes are never described as complete physical validation. The source-visible open-world law evaluator is a separate model-selection benchmark; it must not be conflated with ODE numerical accuracy or blind physical-law discovery.
