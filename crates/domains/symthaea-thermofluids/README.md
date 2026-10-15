# symthaea-thermofluids

Applied fluid & thermal engineering for Symthaea, completing the engineering
stack alongside `symthaea-structural` (mechanics) and `symthaea-circuits`
(electrical).

Pure `std`, zero dependencies, no `symthaea-core` link. Checked against textbook
reference values.

- `fluids` — Reynolds number & regime, Bernoulli head, Darcy-Weisbach head loss,
  continuity.
- `thermal` — Carnot efficiency, Fourier conduction, Newton cooling, engine work.
- `cycles` — input-validated ideal Carnot, Brayton, air-standard Otto, and
  air-standard Diesel efficiency baselines. These are analytic screening models,
  **not** predictions for real engines; losses, working-fluid behavior, hardware
  limits, and system auxiliaries are not included.

Run tests:

```bash
cargo test -p symthaea-thermofluids
```

Research context and validation requirements:
[Symthaea Energy-Conversion Discovery Program](../../../docs/research/ENERGY_CONVERSION_DISCOVERY.md)
