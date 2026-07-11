# symthaea-earth-system

A self-contained **earth-system climate physics** layer for Symthaea. Fills a
gap — the workspace encoded geophysics only as exploratory HDC concept vectors
with no quantitative climate model.

Pure `std`, zero dependencies, no `symthaea-core` link. Closed-form and
root-solved physics, checked against canonical values.

## Capabilities

| Area | API |
|------|-----|
| Energy balance | `EnergyBalanceModel`, `effective_temperature`, `grey_atmosphere_surface_temperature` |
| Forcing / sensitivity | `co2_radiative_forcing` (Myhre 1998), `equilibrium_warming` |
| Ice-albedo feedback | `IceAlbedoModel::equilibria` (multiple stable states, snowball Earth) |
| Carbon budgets | `warming_from_cumulative_carbon/co2` (TCRE), `remaining_carbon_budget` |

## Example

```rust
use symthaea_earth_system::{IceAlbedoModel, warming_from_cumulative_carbon};

// Present Earth has a habitable stable state above freezing.
let warm = IceAlbedoModel::earth().warm_stable_temperature().unwrap();
assert!(warm > 273.15);

// 1000 GtC of cumulative emissions ⇒ ~1.65 °C warming (TCRE).
assert!((warming_from_cumulative_carbon(1000.0) - 1.65).abs() < 0.01);
```

## Validation

Checked against canonical values: Earth effective temperature ≈ 255 K, CO₂
doubling forcing ≈ 3.7 W/m², ~33 K greenhouse warming, ~3 K ECS, snowball
bistability, TCRE budgets.

```bash
cargo test -p symthaea-earth-system
```

## Not yet

1-D latitudinal EBM, radiative-convective columns, general circulation.
