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

## Initial public procurement anchors

The companion file `references/public-procurement-anchors.csv` is a **source register**, not a normalized price dataset. It records source type, date, quantity when explicitly stated, scope clues, missing fields, and a conservative comparability disposition.

Initial leads include:

- CanadaBuys award record [CW2350457-000](https://canadabuys.canada.ca/en/tender-opportunities/contract-history/cw2350457-000): a maximum-6.0 m aluminium hull jet-boat contract with a published total value of CAD 187,726.35 including amendments. It is associated with a standing-offer contract over a 23-month term and the quantity/scope should not be assumed to equal one delivered vessel.
- CanadaBuys award record [CW2349465-000](https://canadabuys.canada.ca/en/tender-opportunities/contract-history/cw2349465-000): a published total contract value of CAD 428,538.96 including amendments for a 6.1 m welded-aluminium commercial vessel; the procurement description includes twin engines and a trailer for one vessel. The line-item cost and tax/delivery breakdown is still missing.
- CanadaBuys award record [CW2342791-000](https://canadabuys.canada.ca/en/tender-opportunities/contract-history/cw2342791-000): CAD 123,488.40 for a 4.8–5.5 m aluminium jet-boat contract with trailer, including amendments. Engine rating and detailed equipment/tax/commissioning scope are unresolved.
- South African Department of Water and Sanitation tender [WQ-32149](https://www.dws.gov.za/Tenders/Docs/2025/Tender%20Document%20%20WQ%20%20-%2032149.pdf): a 5.2 m SAMSA-accredited RIB specification with a 100 hp four-stroke outboard, licensed trailer and accessories. The surfaced price fields are blank; this is a specification lead, **not** price evidence.
- South African DWS tender notice [WQ-32541](https://www.dws.gov.za/Tenders/tendersCurrent.aspx/Docs/2025/Oelsona%40dws.gov.za) lists manufacturing, supply, delivery and commissioning of a rigid inflatable boat with a 4 December 2025 closing date. The official award/outcome was not established in this pass; treat this as a discovery lead only, not an awarded transaction.
- The [National Treasury Transparency Portal](https://data.etenders.gov.za/) publishes procurement exports in OCDS-oriented CSV/JSON/Excel and an API, with records from May 2021 under CC BY 4.0; its notice says only a subset of eTender records is captured. Each boat-specific record still needs retrieval, rights/scope review and a source digest.

**Do not compare these totals as though they were equivalent boat prices.** They differ by vessel form, propulsion, equipment, quantity/contract term, amendments, currency, country, date and likely tax/delivery conditions. Currency conversions and inflation escalation are not applied by the Rust calculator. To admit an award as an actual comparable, obtain the detailed procurement specification, award and amendments, quantity, price components, taxes, commissioning/delivery boundary, and use an agreed price-year/region basis. If these details cannot be resolved, keep it as context-only evidence.

This first register makes a missing-evidence gap visible; it does not yet establish South African market prices, a current average, or any percentage savings.

## Research comparison

- The [IMO GreenVoyage2050 hull-form optimization overview](https://greenvoyage2050.imo.org/technology/hull-form-optimization/) says main particulars are often already optimized but operating profiles outside a design point can be underweighted. It estimates traditional tank-tested hull optimization savings in total fuel consumption at 0.1–4%, while higher percentages may refer to main-engine fuel at a particular design point. Use the proper denominator and scope.
- The [ITTC resistance and propulsion benchmark repository](https://www.ittc.info/benchmark-repository/ship-resistance-and-propulsion/) is a validation source family for hydrodynamics, not a vessel-cost benchmark.
- The 2026 review of [shipyard production planning digitalisation](https://doi.org/10.3390/jmse14040396) finds hybrid discrete-event simulation and optimization comparatively operationally credible, while noting limited public yard datasets and inadequate longitudinal yard-wide KPI validation in much digital-twin/AI work.
- A 2025 [battery-electric ROPAX ferry techno-economic and life-cycle assessment](https://doi.org/10.1016/j.jclepro.2025.146935) is a reminder to compare capital, replacements, operations and life-cycle boundaries rather than purchase price alone.

V0 is intentionally a cost-accounting tool; it must not duplicate Symthaea's energy-budget/hydrodynamics work, Symtropy's shipyard scheduling engine, Mycelix's provenance contracts, or Sol-Atlas's spatial evidence projection.
