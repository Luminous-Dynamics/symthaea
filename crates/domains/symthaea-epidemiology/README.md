# symthaea-epidemiology

Infectious-disease dynamics for Symthaea, starting with the SIR compartment
model (Kermack–McKendrick). The workspace had clinical/medical crates but no
epidemic modelling.

Pure `std`, zero dependencies, no `symthaea-core` link. Closed-form results plus
a population-conserving simulation, checked against known values.

## Capabilities

`Sir { beta, gamma }` →
- `basic_reproduction_number` (R₀ = β/γ)
- `herd_immunity_threshold` (1 − 1/R₀)
- `final_size` (solves `R∞ = 1 − e^(−R₀·R∞)`)
- `simulate` (Euler stepping, conserves S+I+R, tracks peak infected)

## Example

```rust
use symthaea_epidemiology::Sir;
let flu = Sir { beta: 0.3, gamma: 0.1 };          // R0 = 3
assert!((flu.herd_immunity_threshold() - 0.6667).abs() < 1e-3);
assert!(flu.final_size() > 0.9);                  // ~94% eventually infected
```

```bash
cargo test -p symthaea-epidemiology
```

## Not yet

SEIR (exposed compartment), age structure, spatial/network models.
