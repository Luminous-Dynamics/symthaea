# symthaea-economics

A pure-`std`, zero-dependency economics kernel for Symthaea. Version 0.2 makes
invalid domains, undefined answers, and numerical failures explicit through
`Result<T, EconomicsError>` rather than silently returning plausible `f64`
values.

## Design boundaries

- deterministic mathematical primitives live here;
- cognition, institutions, and agent simulation remain in higher layers;
- public inputs reject `NaN` and infinities;
- economic non-solutions are distinct from invalid models;
- no `unsafe` code and no dependency on `symthaea-core`.

## Example

```rust
use symthaea_economics::{Demand, Supply, equilibrium, gini};

let demand = Demand::new(100.0, 2.0)?;
let supply = Supply::new(20.0, 2.0)?;
let point = equilibrium(&demand, &supply)?;
assert_eq!(point.price, 20.0);
assert!(gini(&[10.0, 10.0, 10.0, 70.0])? > 0.4);
# Ok::<(), symthaea_economics::EconomicsError>(())
```

Run with:

```bash
cargo test -p symthaea-economics
```

## Finance depth

The finance module includes stable annuity calculations, amortization schedules,
nominal/effective rate conversion, MIRR, and a bounded multi-root IRR analyzer.
`irr` remains deliberately strict; use `irr_analysis` for non-conventional cash
flows.

## Welfare and distribution

Linear markets now expose consumer and producer surplus, administered-price
shortages/surpluses, per-unit tax incidence, revenue, and deadweight loss.
Distribution analysis includes Lorenz curves, finite-sample-normalized Gini,
Hoover, Theil T, and Atkinson indices. Every metric operates on the same
explicit non-negative-population contract.

## Strategic analysis

The 2×2 game kernel now treats both players symmetrically: best responses,
strict and weak dominance, pure and interior mixed Nash equilibria, Pareto
frontiers, social-welfare maximizers, constant-sum detection, and transposition
are independently inspectable.
