# 2-D heat stability-budget contract v0.1

**Status:** proposed software-verification contract; exact-head execution required. This work is stacked on the 1-D heat stability-budget change because it shares the typed heat-solver error contract.

## Failure mode

The previous two-dimensional explicit heat solver computed a step count from
`dt_max = 0.4 / (α (1/Δx² + 1/Δy²))`, then clamped the count with
`min(MAX_STEPS)`. When a fine grid or long integration horizon required more
than the budget, it increased the actual timestep and still returned a
`PDE2DResult`. A computed trajectory must not imply a valid solve when the
declared stability rule was not honored.

## Contract

The new `try_heat_equation_2d` API returns `Result<PDE2DResult, HeatEquationError>`
and validates:

- checked grid-size product and exact initial-vector dimensions;
- at least two interior points per axis, preserving the existing solver's
  documented grid requirement;
- finite, positive diffusivity and domain lengths;
- finite, non-negative end time and finite initial-condition components;
- representable positive grid spacings and a finite positive stable timestep;
- required stable step count within `MAX_STEPS`, without clamping;
- the actual ratio `α Δt (1/Δx² + 1/Δy²) ≤ 0.4 + 10⁻¹²`;
- finite state updates, with the failing step and flat component index reported.

The historic `heat_equation_2d` signature remains as a wrapper, but now fails
loudly for invalid or over-budget cases instead of returning a result that
violates its own stability envelope. Callers requiring recoverable failure
handling should use the new fallible API.

## Provisional acceptance gates

1. **No false completion:** every over-budget input returns
   `StabilityBudgetExceeded` before stepping; target zero over-budget cases
   returned as successful solutions.
2. **Actual stability:** every accepted run has an independently recomputed
   timestep ratio within the declared limit.
3. **Bad-data rejection:** invalid dimensions, invalid/non-finite parameters,
   and non-finite initial values are explicit errors.
4. **Arithmetic rejection:** computed NaN/Inf values return
   `NonFiniteSolution`; they are never encoded as successful field solutions.
5. **Independent correctness:** later qualification must compare analytical
   2-D eigenmodes against the full grid, not merely observe decay of the
   aggregate energy score.
6. **Claim ceiling:** a stable, finite numerical result is not automatically
   accurate, validated against measurements, or suitable for a safety-critical
   application.

## Immediate tests

- **Budget negative control:** `nx=400, ny=400, Lx=Ly=1, α=0.01, t_end=20`.
  The declared stable schedule requires more than 100,000 steps and must
  refuse before stepping.
- **Nominal case:** `nx=ny=4, Lx=Ly=1, α=0.01, t_end=0.1`. The schedule
  should use one step and report an independently calculated ratio no greater
  than `0.4 + 1e-12`.
- **Invalid parameter controls:** NaN diffusivity, zero domain length,
  infinite end time, mismatched vector length, undersized grid, and NaN
  initial condition must be refused.
- **Finite arithmetic control:** any finite initial vector that overflows
  during the update must return the exact failed step/index.

These are authored tests until compiled and executed against the exact PR head.

## Benchmark ladder

### H0 — frozen contract corpus

Freeze at least 1,000 deterministic parameter combinations across nx, ny,
domain lengths, diffusivity, duration, and amplitude scale. Independently
derive required steps and the stability ratio. Report a matrix of accepted,
budget-refused, invalid, and non-finite outcomes. Goal: zero false successes in
the over-budget/adversarial set.

### H1 — analytical eigenmode accuracy

Use separable Dirichlet modes
`sin(mπx/Lx) sin(nπy/Ly)`, with analytical decay
`exp(-απ²((m/Lx)²+(n/Ly)²)t)`. Compare full-grid relative L2 and maximum
absolute error, grouped by mode, aspect ratio, resolution, and timestep.
Energy decay stays a separate qualitative invariant.

### H2 — refinement and cost

Refine x and y independently and together. Report observed convergence order,
grid dimensions, actual timestep, steps, measured runtime, and memory. Check
that a pass is not caused by a solver using too few stable steps or by
threshold tuning on the held-out cases.

### H3 — algorithm choice under equal budget

Compare explicit Euler against an independently implemented implicit or
adaptive method under declared equal compute/memory constraints. Report error
at equal cost and cost at equal error. A faster unstable result cannot win.

### H4 — sibling solver audit

The 1-D wave method also contains a `min(MAX_STEPS)` count clamp. It needs a
separate wave-equation CFL condition and energy/trajectory tests. Do not infer
that the heat fixes qualify wave integration.

## Reproducibility receipt

Pin source SHA, compiler and target, feature flags, test/corpus digests, all
parameters, expected/actual step count, actual stability ratio, per-cell raw
output digest, error metrics, runtime, and every refusal disposition. Missing
provenance or skipped/missing/queued/stale CI is not a pass.

## Research basis

- NASA [NASA-STD-7009B](https://standards.nasa.gov/standard/NASA/NASA-STD-7009)
  and [NASA-HDBK-7009B](https://standards.nasa.gov/standard/NASA/NASA-HDBK-7009):
  evidence-driven model/simulation credibility and implementation guidance.
- Salari and Knupp, [Code Verification by the Method of Manufactured Solutions](https://doi.org/10.1115/1.1436090):
  independent known solutions plus systematic refinement.
- NASA Glenn, [Verification Assessment](https://www.grc.nasa.gov/www/wind/valid/tutorial/verassess.html):
  error estimation and spatial/time convergence checks.
- Srinivasan, [NIST AMS 100-85 (2026)](https://doi.org/10.6028/NIST.AMS.100-85):
  model credibility and uncertainty evaluation aligned to intended use/risk.

The sources support the verification approach, not the proposed thresholds or
any claim that this solver is accurate for real-world physical systems.
