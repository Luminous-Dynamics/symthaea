# Marine acquisition and lifecycle cost benchmark (V0)

This standalone, dependency-free Rust crate implements the first executable accounting slice for [MARINE-COST-CLASS-001 #8215](https://github.com/Luminous-Dynamics/symthaea/issues/8215).

It answers a bounded question: given declared, source-referenced cost intervals and a matched useful-service contract, what do first-unit build cost, repeat-build cost, discounted lifetime cost and cost per delivered service unit look like—and at what production volume might programme/tooling NRE break even?

## Run locally

~~~sh
cargo test --locked --offline --manifest-path crates/tools/marine-cost-model-bench/Cargo.toml
cargo run --locked --offline --manifest-path crates/tools/marine-cost-model-bench/Cargo.toml --example synthetic
~~~

The package is a nested Cargo workspace to remain isolated from the large Symthaea workspace.

## Inputs and boundaries

- **Money basis:** all values in one currency_basis_id, which must identify currency, price year and geographic/market basis. Conversion and escalation happen before this calculator; do not mix nominal years or currencies.
- **Cost scope:** exact cost_scope_id match is required. The scope should state what is included (e.g. yard-gate construction plus service-period costs), including taxes, freight, commissioning, inspection and safety outfit as appropriate.
- **Useful service:** exact service_contract_id match is required across alternatives. Freeze payload, route, speed/arrival policy, operating envelope, utilisation/availability and the unit of completed service before comparison.
- **Programme engineering and tooling:** modelled as non-recurring engineering/programme cost, reported separately from per-vessel build cost and amortized at the selected production volume.
- **Lifecycle cost:** annual owner operating costs, explicitly scheduled replacement events, service life, discount rate and terminal residual value are included. Annual operating cost is treated as paid at the end of each year; a replacement event is discounted at its declared one-based year; residual value is discounted at the end of service life.
- **Evidence:** each estimate needs a stable evidence reference and an origin (SupplierQuote, Invoice, Measured, Modelled, Assumption or Synthetic). References should point into an independently reviewable source ledger. The crate does not validate the truth, age, rights, or authenticity of a reference.
- **Uncertainty:** lower/nominal/upper are conservative bounds, not probabilities or confidence intervals. Bounds are added conservatively; correlated uncertainty is not modelled. A comparison is robustly cheaper only when the candidate upper bound is strictly below the baseline lower bound on cost per discounted service unit. Overlap is Indeterminate.
- **Missing is not zero:** unknown programme cost, any required build/operating/residual estimate, or the replacement plan makes evaluation invalid. An explicit empty replacement list and an explicit zero residual value are allowed.
- **Authority:** this is not a cost database, naval-architecture solver, stability or structural analysis, certification tool, or construction approval. A cheaper result can never override engineering, inspection, legal or operating requirements.

## Output

For each case and chosen production volume, the report computes:

1. first-unit build cost, with the full programme NRE allocated to unit one;
2. repeat/amortized build cost per vessel;
3. discounted lifetime cost per vessel and for the whole production programme;
4. discounted useful-service units and lifetime cost per discounted service unit;
5. an indicative nominal break-even production volume where one design has higher NRE but lower recurring lifecycle cost.

The break-even value is a mathematical crossover of the supplied nominal inputs, not a confidence guarantee. A design may be cheaper after that volume only if the model's service and cost assumptions remain valid.

## Benchmark cases and falsification

Unit tests cover hand-calculable costs, NRE amortization, lifecycle costs, robust non-overlapping bands, overlapping/indeterminate bands, unknown residual value, unmatched mission contracts and NRE/recurring-cost break-even.

The synthetic example is only an arithmetic fixture. Do not label its outputs as boat-market prices, actual cost savings or vessel performance. Before industry comparison, populate a separate reviewed dataset with date-stamped quotes, invoices or permitted public prices, comparable delivery scope, production volume, service duty cycle, operating geography and uncertainty. Keep first-unit prototypes separate from mature series builds.

## Research comparison

- The [IMO GreenVoyage2050 hull-form optimization overview](https://greenvoyage2050.imo.org/technology/hull-form-optimization/) says main particulars are often already optimized but operating profiles outside a design point can be underweighted. It estimates traditional tank-tested hull optimization savings in total fuel consumption at 0.1–4%, while higher percentages may refer to main-engine fuel at a particular design point. Use the proper denominator and scope.
- The [ITTC resistance and propulsion benchmark repository](https://www.ittc.info/benchmark-repository/ship-resistance-and-propulsion/) is a validation source family for hydrodynamics, not a vessel-cost benchmark.
- The 2026 review of [shipyard production planning digitalisation](https://doi.org/10.3390/jmse14040396) finds hybrid discrete-event simulation and optimization comparatively operationally credible, while noting limited public yard datasets and inadequate longitudinal yard-wide KPI validation in much digital-twin/AI work.
- A 2025 [battery-electric ROPAX ferry techno-economic and life-cycle assessment](https://doi.org/10.1016/j.jclepro.2025.146935) is a reminder to compare capital, replacements, operations and life-cycle boundaries rather than purchase price alone.

V0 is intentionally a cost-accounting tool; it must not duplicate Symthaea's energy-budget/hydrodynamics work, Symtropy's shipyard scheduling engine, Mycelix's provenance contracts, or Sol-Atlas's spatial evidence projection.
