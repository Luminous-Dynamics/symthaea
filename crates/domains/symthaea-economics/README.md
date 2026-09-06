# symthaea-economics

A pure-`std`, zero-dependency economics kernel for Symthaea. Version 0.2 makes
invalid domains, undefined answers, and numerical failures explicit through
`Result<T, EconomicsError>` rather than silently returning plausible `f64`
values.

## Design boundaries

- deterministic mathematical primitives live here;
- theory-neutral scientific contracts may live here when they remain pure data/invariants;
- cognition, institutions, agent simulation, empirical ingestion, and governance remain in higher layers;
- public floating-point inputs reject `NaN` and infinities;
- economic non-solutions are distinct from invalid models;
- no `unsafe` code and no dependency on `symthaea-core`.

## Economic Science foundation

The additive Economic Science v1 foundation separates three kinds of statement:

- hard constraints and accounting identities;
- falsifiable empirical claims;
- explicit normative propositions.

It adds exact integer-atom double-entry accounting, a small theory-neutral state
ontology, decomposed economic claims with predeclared falsifiers, and **ETIR v1**
(Economic Theory Intermediate Representation). ETIR permits multiple model
paradigms to implement the same scientific claim without making any model family
or economic school the owner of that claim.

The constitutional boundary is documented in
[`docs/ECONOMIC_SCIENCE_CONSTITUTION_V1.md`](docs/ECONOMIC_SCIENCE_CONSTITUTION_V1.md).
The foundation does **not** add a policy recommender, welfare optimizer, Mycelix
data access, or execution/governance authority.

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
