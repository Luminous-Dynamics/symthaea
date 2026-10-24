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
- `energy_balance` — steady-state conservation checks and net/gross efficiency
  accounting. Results are accepted only when the declared power-flow balance
  closes within an explicit relative tolerance.
- `energy_interval` — transient interval-integrated conservation checks that
  account for changes in stored energy, plus a duty-cycle aggregator that validates
  every interval before combining totals. It reports net cycle efficiency only when stored energy finishes within the
  declared tolerance of its starting state and does not show net discharge.

The steady-state API assumes a steady-state boundary. Transient tests must account
for stored energy gained or released over a clearly registered interval. Duty-cycle
aggregation validates every interval before summing. Energy flows must share the
same boundary and use compatible units. Select and justify the relative residual
tolerance before evaluating candidates, from solver and/or sensor uncertainty.
Tolerance values must be finite and in [0, 1).

Run tests:

```bash
cargo test -p symthaea-thermofluids
```

Research context and validation requirements:
[Symthaea Energy-Conversion Discovery Program](../../../docs/research/ENERGY_CONVERSION_DISCOVERY.md)
