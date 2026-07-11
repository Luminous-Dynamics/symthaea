# symthaea-economics

Economics & finance for Symthaea. Fills a confirmed gap — the workspace had only
behavioral-game fragments (Ultimatum, Public Goods in psych-bench), no
quantitative economics.

Pure `std`, zero dependencies, no `symthaea-core` link. All results closed-form,
checked against textbook values.

## Capabilities

| Area | API |
|------|-----|
| Finance | `finance::{future_value, present_value, npv, irr, compound_interest, annuity_payment}` |
| Markets | `market::{Demand, Supply, equilibrium, price_elasticity_of_demand}` |
| Inequality | `inequality::gini` |
| Game theory | `game::Game2x2` (pure Nash equilibria, dominant strategies) |

## Example

```rust
use symthaea_economics::finance::{npv, irr};
assert!((npv(0.10, &[-1000.0, 500.0, 500.0, 500.0]) - 243.426).abs() < 0.01);
assert!((irr(&[-1000.0, 600.0, 600.0]).unwrap() - 0.13066).abs() < 1e-4);
```

```bash
cargo test -p symthaea-economics
```
