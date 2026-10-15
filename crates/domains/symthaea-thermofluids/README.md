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
  air-standard Diesel efficiency baselines. These are analytical screening models,
  **not** predictions for real engines; losses, working-fluid behavior, hardware
  limits, and system auxiliaries are not included.
- `energy_balance` — steady-state energy conservation checks and net/gross
  efficiency accounting. Results are accepted only when the declared power-flow
  balance closes within an explicit relative tolerance.

The energy-balance API assumes a steady-state boundary. Transient storage changes
must be included separately, and comparisons are valid only when systems use the
same input boundary and account for auxiliary loads consistently. Select and
justify the residual tolerance before evaluating candidates.

Run tests:

```bash
cargo test -p symthaea-thermofluids
```

Research context and validation requirements:
[Symthaea Energy-Conversion Discovery Program](../../../docs/research/ENERGY_CONVERSION_DISCOVERY.md)
