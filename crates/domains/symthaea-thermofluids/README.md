# symthaea-thermofluids

Applied fluid & thermal engineering for Symthaea, completing the engineering
stack alongside `symthaea-structural` (mechanics) and `symthaea-circuits`
(electrical).

Pure `std`, zero deps, no `symthaea-core` link. Checked vs textbook values.

- `fluids` — Reynolds number & regime, Bernoulli head, Darcy-Weisbach head loss,
  continuity.
- `thermal` — Carnot efficiency, Fourier conduction, Newton cooling, engine work.

```bash
cargo test -p symthaea-thermofluids
```
