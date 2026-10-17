# Aircraft Ownership Economics: Scenario Model and Evidence Contract

**Date:** 2026-10-17  
**Status:** cost-model specification and pure-Rust implementation; not a quote, market average, or prediction of a proposed aircraft's actual cost.  
**Tracking:** [AIRFAMILY-002 #7753](https://github.com/Luminous-Dynamics/symthaea/issues/7753)  
**Implementation:** [Operations-research aircraft ownership module](https://github.com/Luminous-Dynamics/symthaea/blob/feat/aircraft-ownership-cost-model-2026-10-17/crates/domains/symthaea-operations-research/src/aircraft_ownership.rs)

## 1. Why cost needs an evidence model

Aircraft ownership comparisons often understate the real cost by omitting items that do not appear in a simple fuel-burn number: annual insurance and storage, inspection/compliance, unscheduled maintenance, engine and propeller overhaul reserves, avionics/electrification replacement reserves, owner labor, and downtime. Fixed costs per hour change sharply with how often the aircraft is flown. A model must expose assumptions and let owners replace them with local quotes, records and observed data.

AOPA's operating-cost calculator separates fixed costs from variable costs. Its ownership guidance highlights how omitting overhaul reserves and unexpected maintenance can make an hourly number misleading. We use these as model-coverage references—not as default price data:
- AOPA [Operating Costs Calculator](https://aopa.org/go-fly/aircraft-and-ownership/operating-costs-calculator)
- AOPA [Operating costs and financing](https://www.aopa.org/go-fly/aircraft-and-ownership/buying-an-aircraft/tips-on-buying-used-aircraft/operating-costs-and-financing)
- AOPA [Hypothetical operating-cost calculation](https://www.aopa.org/go-fly/aircraft-and-ownership/buying-an-aircraft/tips-on-buying-used-aircraft/hypothetical-operating-cost-calculation)

## 2. What the Rust model calculates

The first implementation is a dependency-free module inside the existing symthaea-operations-research crate. It accepts explicit inputs; it does not load guessed aircraft prices or silently select a default utilization.

### Annual fixed operating cash

Separately records insurance, hangar or tie-down, annual inspections and compliance, recurrent training, subscriptions and data, registration/taxes/fees, and other fixed costs.

### Variable operating cost per flown hour

Separately records fuel or electricity, oil and fluids, routine scheduled maintenance, allowance for unscheduled maintenance, engine and propeller overhaul reserve, avionics replacement reserve, battery replacement reserve, wear parts/consumables, and landing/handling/usage fees.

A reserve is a budgeting provision, not a promise about when a part will fail. For electric aircraft, the energy item represents the declared electricity boundary; battery replacement reserve is separate. Do not count a replacement twice under both the reserve and an annual expense.

### Acquisition capital cost

The model does not add loan payments on top of depreciation or annualized acquisition expense. It reports an equivalent annual capital cost given acquisition price, residual value at the horizon, ownership horizon and discount rate.

At zero discount rate:

\[
C_{capital,annual} = \frac{P-R}{n}
\]

At a positive annual rate r:

\[
CRF=\frac{r(1+r)^n}{(1+r)^n-1}
\]

\[
C_{capital,annual}=\left(P-\frac{R}{(1+r)^n}\right)CRF
\]

where P is acquisition price, R is the declared end-horizon residual, and n is the horizon in years. This is an economic annualization, not the actual cash-flow schedule, tax treatment, or a loan quote. Residual value cannot exceed purchase price in this simplified model.

### Owner labor and downtime

Owner time is calculated separately from operating cash costs using an hourly value, annual administration/logbook time, and maintenance hours per flown hour. This avoids a false "cheap ownership" comparison that treats all owner labor as free.

Maintenance downtime support cost is explicit, as is extra non-hourly cost per aborted mission. If an expense is already included in actual flown-hour costs, it must not also be entered as downtime cost. Aborted flights still count toward flown-hour variable costs.

## 3. Accounting formulas and output metrics

Let F be total annual fixed operating cash cost; V the variable contribution per actual flown hour; H actual flown hours, including aborted attempts; D annual downtime support cost; A attempts minus completed missions; CA extra non-hourly cost per aborted mission, excluding costs already included elsewhere; K equivalent annual capital cost; and L monetized annual owner labor.

Then:

\[
C_{operating,cash} = F + VH + D + AC_A
\]

\[
C_{annual,economic} = C_{operating,cash} + K + L
\]

The result reports annual fixed operating cash, variable cost rate and annual variable cash, annual downtime/aborted-mission cost, annual operating cash cost, equivalent annual capital cost, owner labor hours and economic value, fully loaded annual economic cost, cost per actual flown hour, cost per successful flight hour, cost per completed mission, and cost per delivered passenger-kilometre.

A rate whose denominator is zero is returned as **unavailable**, not zero, NaN, infinity, or an invented fallback. Annual costs may still be present even if the aircraft completes no missions. If there are no completed missions, the input cannot simultaneously claim successful flight hours or delivered passenger-kilometres.

## 4. Fair comparisons: freeze the mission and the accounting basis

A comparison requires all scenarios to share the mission-family ID and identical SHA-256 mission-contract digest, currency code, capital horizon, and annual discount-rate assumption.

Different utilization values and different cost structures are deliberately allowed within a common mission family. The output is a sorted, per-scenario breakdown, not one opaque score. It does not automatically convert currencies or declare a winning aircraft.

The mission-family contract should state at minimum:
- passenger, baggage/cargo and payload assumptions;
- route or route class and mission definition;
- fuel/energy reserve and alternate/abort policy;
- performance and environmental conditions;
- successful-mission and completed passenger-kilometre accounting; and
- included/excluded operating and downtime costs.

A family identifier or digest cannot itself prove two mission profiles are physically comparable. A model owner must retain the source assumptions.

## 5. Validation and planned benchmarks

The implementation includes synthetic unit tests for:
- capital annualization at zero and positive discount rates;
- cost per hour falling as utilization increases when fixed costs are spread over more flown hours;
- owner labor being visible in fully loaded economic cost but not operating cash;
- zero-utilization metrics being unavailable;
- rejection of different mission-contract digests/currencies/capital assumptions;
- deterministic comparison ordering;
- invalid currencies, non-finite/negative values, impossible mission counts and duplicate scenario IDs; and
- passenger-kilometre denominators that only exist when service was delivered.

Test declarations are not test passes until executed on a recorded commit with the Rust toolchain. The initial test values are synthetic fixtures only—not Skyhawk operating costs or estimates for Wren 2, Wayfarer 4, Ranger 4 or Wren-E.

For the product family, prepare at least three utilization scenarios (low, planning-baseline and high). Choose those values from intended owner research and the region of first sale, rather than declaring one utilization level universal. Compare:
1. Cessna Skyhawk and relevant two-seat trainer/utility references using source-linked assumptions;
2. Wren 2, Wayfarer 4 and Ranger 4 once candidate-specific cost evidence exists; and
3. Wren-E only if the energy, battery replacement, charging, downtime and infrastructure data are supportable.

The family-level target is **at least 15% lower fully loaded cost per flight hour** than a frozen, mission-appropriate baseline across multiple utilization scenarios. That target remains a proposal: it can only be claimed after assumptions are auditable and real quotes, task-time data, utilization and fleet maintenance records replace synthetic inputs.

## 6. Explicit limitations

The first model does not automatically derive real fuel flow, electricity use, maintenance intervals, insurance quotes, hangar prices, component failure rates, residual-value curves, financing cash flows, taxes, currency exchange, shared-ownership allocation, or the value of a missed trip. All must be supplied and sourced by the study that uses them. It is not yet connected to the aircraft-energy, maintenance-task or factory modules.

The goal is to make ownership comparisons honest and reproducible, not to make an aircraft appear economical by choosing friendly assumptions.

## Nonclaims

This document does not assert that any proposed aircraft is cheaper than a Cessna or competitor, predict real-world ownership cost, or establish maintenance, safety, airworthiness or certification. It defines an explicit planning model and the evidence needed to use it responsibly.
