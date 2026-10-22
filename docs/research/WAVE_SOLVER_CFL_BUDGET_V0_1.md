# 1-D wave solver CFL-budget contract v0.1

**Status:** implementation proposal requiring exact-head test execution. It is
separate from heat-equation verification because the wave solver uses a
leapfrog/Störmer–Verlet method and a different Courant stability condition.

## Failure mode

The existing solver chose
`dt_max = 0.9 Δx / c`, but then clamped the required count using
`min(MAX_STEPS)`. The returned time step could therefore exceed the stated
CFL margin while still producing a trajectory result. The legacy API also
accepted finite inputs whose arithmetic update could generate infinities, and
a zero-duration solve reached an energy diagnostic that divides by `dt=0`.

## Contract and implementation

The new `try_wave_equation_1d` API returns a typed `WaveEquationError` and:

- validates displacement/velocity vector sizes, non-empty grid, finite positive
  wave speed and domain length, finite non-negative end time, and finite input
  values;
- computes the needed step count before integration and returns
  `StabilityBudgetExceeded` rather than clamping to `MAX_STEPS`;
- verifies the actual Courant number `c Δt / Δx ≤ 0.9 + 10⁻¹²`;
- rejects non-finite updated displacement values, reporting the first failed
  step and component index;
- treats zero duration as the identity map, returning the input displacement
  without performing a finite-difference velocity calculation with zero dt.

The old `wave_equation_1d` signature remains as a wrapper, but over-budget or
invalid operations now fail loudly rather than returning a false-success
record. New callers needing recoverable errors should use the fallible API.

## Immediate regression gates

- **Budget negative control:** `n_x=128, L=1, c=1, t_end=1000` requires more
  than 100,000 CFL-safe steps and must return `StabilityBudgetExceeded`.
- **Nominal case:** `n_x=20, L=1, c=1, t_end=0.5` should use 12 steps and
  produce Courant number `0.875`.
- **Zero duration:** `t_end=0` returns the initial displacement unchanged,
  uses zero integration steps, and has finite diagnostics.
- **Invalid inputs:** NaN wave speed, invalid length/time, vector length
  mismatch, and non-finite displacement/velocity each return typed errors.
- **Arithmetic overflow:** finite values whose stencil update overflows must
  return `NonFiniteSolution`; no non-finite result may be marked successful.

These are authored tests until run by the pinned toolchain at the exact head.

## Benchmark ladder

### W0 — CFL contract corpus

Generate at least 1,000 deterministic cases across grid size, wave speed,
length, duration, and input amplitude. Independently recompute the required
CFL step count and actual Courant number. Report accepted, budget-refused,
invalid-input and arithmetic-failure cases separately. Goal: zero false
successful results in the over-budget/adversarial set.

### W1 — analytical mode error

For the fixed-end wave equation, use
`u(x,t) = sin(mπx/L) cos(c mπt/L)` with initial velocity zero.
Compare the full returned grid against the analytical mode at multiple times.
Report relative `L2`, max absolute error, temporal/spatial resolution, and
energy drift independently. Energy conservation alone is not an accuracy
metric.

### W2 — convergence/refinement

Refine dt and dx in controlled schedules; the current CFL-based dt couples to
dx, so document that coupling. Report observed order, phase error, amplitude
error, and error by mode. Do not infer second-order convergence from a single
final-time sample or when the error is at roundoff.

### W3 — cost-aware solver comparison

Compare leapfrog against an independently implemented method using equal
compute/memory budgets. Report error at equal cost and cost at equal error;
runtime cannot excuse a CFL violation.

### W4 — scope audit

This change does not address 2-D wave propagation or all ODE/PDE solvers.
Related heat solvers are being addressed independently. Every numerical method
must own its derived stability condition and negative controls.

## Evidence receipt

Pin exact source SHA, toolchain and target, input corpus digest, solver
parameters, expected/actual step counts, actual Courant number, output digest,
raw analytical errors, energy drift, runtime and error dispositions. A skipped,
missing, queued or stale-head workflow is not a pass.

## Research basis

- NASA [NASA-STD-7009B](https://standards.nasa.gov/standard/NASA/NASA-STD-7009)
  and [NASA-HDBK-7009B](https://standards.nasa.gov/standard/NASA/NASA-HDBK-7009):
  evidence-driven credibility for model/simulation outputs.
- Salari and Knupp, [Code Verification by the Method of Manufactured Solutions](https://doi.org/10.1115/1.1436090):
  analytical references and systematic refinement help reveal implementation
  errors.
- NASA Glenn, [Verification Assessment](https://www.grc.nasa.gov/www/wind/valid/tutorial/verassess.html):
  convergence/error assessment should accompany qualitative invariants.
- Srinivasan, [NIST AMS 100-85 (2026)](https://doi.org/10.6028/NIST.AMS.100-85):
  uncertainty and credibility evidence should match intended-use risks.

These sources support the verification process, not the chosen threshold values
or any claim of real-world model validity.
