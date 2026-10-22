# Heat solver stability-budget contract v0.1

**Status:** implementation and test proposal; exact-head Rust execution is required before qualification. This contract covers the 1-D explicit-Euler heat solver only. It does not qualify the wave or 2-D heat solvers, physical models, or safety-critical use.

## Failure mode addressed

The old code selected a stability-limited step count, then clamped it with `min(MAX_STEPS)`. When the required count exceeded the cap, it still returned a `PDEResult` despite having increased `dt` beyond the documented stability envelope. A resource budget must not silently convert an unsupported simulation into an apparent success.

The new fallible API, `DifferentialEquationsEngine::try_heat_equation_1d`, returns a typed `HeatEquationError` if:
- the initial-condition dimensions or grid are invalid;
- diffusivity, length, end time, or initial values are invalid/non-finite;
- no finite stability step can be represented;
- the number of stable steps exceeds `MAX_STEPS`;
- the computed ratio `r = α Δt / Δx²` exceeds the declared 0.4 margin;
- a numerical update produces a NaN or infinity.

The historical `heat_equation_1d` signature remains available as a wrapper, but now fails loudly instead of returning an under-resolved or unstable result. Callers that require recoverable control flow should migrate to `try_heat_equation_1d`.

## Invariants and provisional acceptance targets

1. **No budget clamp:** if the required stable step count exceeds 100,000, the fallible API must return `StabilityBudgetExceeded` before integrating.
2. **Actual-step verification:** every accepted nontrivial solve must satisfy `α Δt / Δx² ≤ 0.4 + 10⁻¹²` for the actual computed `Δt`.
3. **Fail closed on bad input:** invalid coefficients, dimensions, and non-finite initial values must produce typed errors, not successful result records.
4. **Finite trajectory:** non-finite computed state components must return a typed failure with step and component index.
5. **Regression anchors:** one valid small-grid case must return a deterministic expected step count and finite values; a known over-budget case must refuse.
6. **Claim ceiling:** a successful result proves only that this implementation completed within the declared stability contract for these inputs. It does not prove the PDE is a valid model of a real system or that the solution is accurate enough for a particular use.

The constants and thresholds here are software contract choices. They are not claimed as universal physical or numerical standards.

## Immediate tests

- Over-budget case: `n_x=2500`, `L=1`, `α=0.01`, `t_end=1`; the stability schedule requires approximately 156,376 steps, exceeding the 100,000-step budget. Expected outcome: `StabilityBudgetExceeded`, not a solution.
- Nominal case: `n_x=20`, `L=1`, `α=0.01`, `t_end=0.5`; expected step count is 6. Recompute and assert the actual stability ratio and finite state values.
- Invalid-data controls: NaN diffusivity, non-positive length, negative end time, wrong initial-vector length, empty grid, and infinite initial value all refuse.
- Arithmetic-overflow control: an input may contain only finite values yet overflow during stencil evaluation; expected outcome is `NonFiniteSolution`, not a successful result.

These are authored source-level tests until executed at the exact PR head with the pinned toolchain.

## Next benchmark ladder

### S0 — contract census

Run all explicit valid/invalid cases and report counts by error variant. Target: 100% of invalid cases explicitly refused, zero invalid cases represented as successful solutions.

### S1 — parameter sweep

Generate at least 1,000 deterministic combinations over grid size, domain length, diffusivity, duration and initial-condition scale. Independently recompute the expected stable step budget. Every accepted run must satisfy the observed stability ratio; every over-budget case must refuse. Freeze seeds and a corpus digest before qualification.

### S2 — accuracy under budget

For accepted inputs, compare analytical eigenmodes with the exact returned grid and report relative discrete `L2` error, maximum absolute error, grid spacing, timestep, actual step count, stability ratio, runtime and memory. Keep accuracy, stability, convergence, and completion as distinct dimensions. A stable solution may still be inaccurate.

### S3 — budget-aware algorithms

Compare explicit Euler to an implicit or adaptive method on stiff/high-resolution regimes under equal declared compute and memory budgets. Compare error at equal wall-clock/step budgets and cost at equal error. Do not claim superiority from an unconstrained comparison or extrapolate synthetic benchmarks to physical performance.

### S4 — nearby-method audit

Audit the 1-D wave and 2-D heat implementations for the same pattern: required CFL/stability step count clamped to `MAX_STEPS`. For each method, derive the correct condition, add a typed refusal path, and build analytical references/refinement tests before modifying production behavior.

## Reproducibility receipt

A qualified run must record exact source SHA, compiler/toolchain, target/features, test IDs, input corpus digest, parameters, predicted and actual step count, actual stability ratio, output/error disposition and raw test output. A skipped, missing, queued, stale-head or absent workflow is not a pass.

## Research basis

- NASA, [NASA-STD-7009B: Standard for Models and Simulations](https://standards.nasa.gov/standard/NASA/NASA-STD-7009) and [NASA-HDBK-7009B implementation guide](https://standards.nasa.gov/standard/NASA/NASA-HDBK-7009): credibility requires planned, documented evidence rather than a plausible output alone.
- Salari and Knupp, [Code Verification by the Method of Manufactured Solutions](https://doi.org/10.1115/1.1436090): use independently known solutions and systematic refinements to find numerical implementation errors.
- NASA Glenn, [Verification Assessment](https://www.grc.nasa.gov/www/wind/valid/tutorial/verassess.html): error and convergence checks complement qualitative conservation or stability behavior.
- Srinivasan, [NIST AMS 100-85 (2026)](https://doi.org/10.6028/NIST.AMS.100-85): credibility and uncertainty evaluation should be commensurate with intended use and risk.

These sources motivate rigorous verification and evidence contracts; they do not endorse the numeric threshold values or establish performance for this code.
