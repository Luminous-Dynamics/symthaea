# Thermal lethality numerical-oracle benchmark

**Track:** CULINARY-THERMAL-NUM-001  
**Initial version:** 1.0.0  
**Status:** implementation proposed; Rust execution not yet verified  
**Scope:** numerical correctness of the implemented piecewise-linear D/z integration only

## Paradigm shift

Do not accept a scientific-looking scalar merely because it is generated consistently by the same implementation. Validate it against an independent equation-derived oracle and properties that must hold regardless of implementation.

The crate defines core temperature as piecewise linear between trajectory samples. For the conventional log-linear thermal-inactivation model, instantaneous lethality rate is

`r(T) = 10^((T - T_ref)/z) / D_ref`

where `D_ref` is the decimal-reduction time at reference temperature `T_ref`, and `z` is the temperature increase that changes the D-value by a factor of ten. Accumulated modelled log reduction is the time integral of that rate.

On a segment of duration `dt` with a linear temperature ramp from `T0` to `T1`, let

`x = ln(10) * (T1 - T0) / z`.

The integral has a closed form:

`L_segment = dt * r(T0) * expm1(x) / x`

with the limiting factor defined as 1 when `x = 0`. Rust's `exp_m1` preserves precision for small `x). This is exact for the model's piecewise-linear temperature assumption, aside from floating-point rounding; it removes the arbitrary 64-substep trapezoid discretization and its ramp-dependent integration error. The integration tests use a separate endpoint-rate-difference expression as the analytic oracle rather than copying the production expression.

## Reproduction

```bash
cargo test -p symthaea-culinary --test thermal_numeric_oracles
cargo test -p symthaea-culinary thermal::
```

Bind any claimed test result to the exact commit SHA and a clean checkout. A queued check is not a pass.

## Frozen tests and proposed gates

| ID | Property | Initial gate |
|---|---|---|
| TN-001 | Constant-temperature hold versus closed-form rate | Relative error ≤ 2e-14 for finite fixtures |
| TN-002 | Heating and cooling ramps versus independent analytic integral | Relative error ≤ 2e-12 for the fixed finite fixture range |
| TN-003 | Insert a collinear sample into the same piecewise-linear curve | Integrated result invariant within 2e-14 relative error |
| TN-004 | Multiply all durations by three | Integrated lethality multiplies by three within 2e-14 relative error |
| TN-005 | Shift all temperatures by one z-value | Modelled lethality multiplies by ten within 2e-13 relative error |
| TN-006 | Subdivide a ramp into 1,000 collinear segments | Result invariant within 5e-13 relative error |
| TN-007 | Raise the whole trajectory while holding its shape/time fixed | Modelled lethality must not decrease |
| TN-008 | Near-constant temperature delta | The `exp_m1(x)/x` limit remains numerically stable |

These are proposed engineering tolerances for the represented mathematical model, not microbiological uncertainty limits. If a platform cannot meet a bound, report the failure and numerical evidence; do not loosen a gate just to make the patch pass.

## Scientific limitations and safety boundary

The D/z log-linear framework is widely used to summarize time-temperature lethality, including non-isothermal processes. However, reviews document deviations from log-linear survival due to microbial heterogeneity, clumping, matrix protection, heat adaptation and process measurement/heat-transfer limitations. Therefore:

- An analytical integral only establishes numerical correctness conditional on the chosen D/z model and the supplied temperature curve.
- It does not prove that a modeled core-temperature curve is physically achievable, that the cold spot was measured, or that the input D/z values describe the actual food, strain, matrix, equipment or handling context.
- It is not an autonomous cooking instruction, real-world pasteurization certification, or food-safety release gate.
- Parameter validation, uncertainty, cold-spot thermal modelling, real measurement and jurisdiction-appropriate process validation remain separate work.

## References

1. [Inactivation kinetics of food-borne pathogens subjected to thermal treatments: a review](https://doi.org/10.1080/02656736.2017.1372643) (2017). Reviews the definitions and use of D- and z-values and the limits of log-linear assumptions.
2. [Development of a Log-Quadratic Model To Describe Microbial Inactivation](https://pmc.ncbi.nlm.nih.gov/articles/PMC2786541/), Applied and Environmental Microbiology. Describes integrated non-isothermal lethality under the D/z framework and discusses departures from log-linear survival.
3. Current crate model and its explicitly simplified parameters: `src/thermal.rs` and `src/thresholds.rs`. Threshold provenance and food-safety scope require separate review; a software test is not validation of a scientific parameter.
