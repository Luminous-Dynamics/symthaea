# symthaea-thermofluids

Applied fluid & thermal engineering for Symthaea, completing the engineering
stack alongside `symthaea-structural` (mechanics) and `symthaea-circuits`
(electrical).

Pure `std`, zero deps, no `symthaea-core` link. Checked vs textbook values.

- `fluids` — Reynolds number & regime, Bernoulli head, Darcy-Weisbach head loss,
  continuity.
- `thermal` — Carnot efficiency, Fourier conduction, Newton cooling, engine work.
- `checked` — `Result`-returning wrappers for non-finite/domain validation of
  fluid and thermal equations.
- `joule_heating` — lumped electrical-to-thermal energy accounting. The adiabatic
  reference step and the exact constant-conductance convection step both report
  signed energy exchange and a residual; neither is a spatial EM/thermal solver.

Use `checked::*` for values originating from telemetry, user input, or composed
models. The legacy scalar functions remain available for compatibility, but do
not establish domain validity and may propagate NaN/infinity for invalid inputs.

```bash
cargo test -p symthaea-thermofluids
```
