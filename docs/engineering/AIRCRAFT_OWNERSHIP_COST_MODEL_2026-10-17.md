# Aircraft Ownership Cost Model — Contract, Formulae and Benchmarks

**Date:** 2026-10-17  
**Status:** first-pass accounting model; synthetic example only, not an observed aircraft cost estimate.  
**Implementation:** `crates/domains/symthaea-operations-research/src/aircraft_ownership.rs`  
**Tracking:** [AIRCRAFT-COST-001 #7763](https://github.com/Luminous-Dynamics/symthaea/issues/7763)  
**Product context:** [owner-maintainable aircraft family](https://github.com/Luminous-Dynamics/symthaea/pull/7761)

## 1. Why this model exists

A plane can look cheap when a comparison counts only fuel and purchase price. That comparison can be badly misleading if it excludes insurance, storage, scheduled inspections, engine/propeller reserves, batteries, maintenance labor/parts, failed dispatches, replacement parts waiting time, and the trips the aircraft did not complete.

The Rust model in Symthaea's existing zero-dependency operations-research crate is designed to make cost boundaries explicit and comparisons repeatable. It reports:
- annual fully loaded owner/operator cost;
- expected successful and failed dispatch counts;
- flown hours and completed passenger-kilometres;
- separate fixed, per-attempt, failed-dispatch, completed-mission and hourly cost components;
- cost per flown hour, completed mission and completed passenger-kilometre.

It does not predict a future aircraft's reliability or costs. Its output is only as credible as the assumptions and measured data supplied to it.

## 2. Input contract

All monetary inputs in one scenario must use the same declared three-letter currency code. Every scenario also carries a cost-boundary ID, cost-assumption digest, source locator and ISO calendar date. A comparison rejects candidates with different currencies, cost boundaries, data dates, or mission contracts. The crate checks digest syntax but does not calculate the digest or retrieve the referenced input artifact; the report producer must preserve that evidence. This crate does not perform currency conversion or silently populate missing data.

### Annual fixed cost inputs

- Annual capital charge, produced by an explicitly documented method.
- Insurance.
- Hangar/storage.
- Calendar-based inspection.
- Taxes, registration and fees.
- Other fixed costs.

The annual capital charge is an input, not a financial model. The scenario manifest must explain whether and how it represents depreciation, financing/opportunity cost or other capital allocation. Do not charge the same cost again in another category.

### Cost per flown hour

- Fuel/electricity.
- Oil and consumables.
- Owner maintenance labor, valued using an explicit opportunity-cost rate.
- Professional/authorized maintenance labor.
- Maintenance parts.
- Engine overhaul reserve.
- Propeller reserve.
- Battery replacement reserve, if relevant.
- Other hourly costs.

Hourly-based inspection and labor should be included once. Calendar-based inspections should be separate; don't count the same work twice. Reserve amounts must be explicitly supplied from an approved/credible overhaul or replacement assumption, not optimized away to lower the cost result.

### Cost per attempted/completed mission

- Preparation cost per dispatch attempt, including attempts later cancelled.
- Additional failed-dispatch cost per cancelled attempt.
- Cost per completed mission, such as landing/handling fees where applicable.

The failed-dispatch input should include incremental cancellation costs not captured elsewhere (such as abort-specific preparation, ground activity or recovery). This first implementation does not infer fuel burned or maintenance time during each failed attempt; scenarios must include those incremental costs in the failed-dispatch field or extend the model with a measured cost boundary. Owner and professional maintenance labor are separate hourly categories so a maintenance-first design cannot make the human labor burden disappear into a single maintenance figure.

### Mission and utilization

- Annual planned dispatch attempts.
- Annual scheduling time budget.
- Block time for one completed mission.
- Route distance.
- Number of passengers transported, excluding flight crew.
- Dispatch-success fraction.
- Mission-family identifier and SHA-256 digest.

The comparison function requires the same mission family/digest, currency, number of scheduled attempts, route distance, transported passenger count and available annual scheduling budget. Candidate mission duration, cost assumptions and dispatch-success fraction may differ. Any different assumptions must be disclosed in the manifest. The routine rejects an annual plan whose planned block hours exceed its declared scheduling budget.

## 3. Calculation

Let:
- (N) be annual planned dispatch attempts;
- (p) be dispatch-success fraction;
- (t) be block hours per completed mission;
- (d) be route distance in kilometres;
- (q) be passengers per completed mission;
- (F) be annual fixed cost;
- (C_a) be preparation cost per attempt;
- (C_f) be additional cost per failed dispatch;
- (C_m) be per-successful-mission cost;
- (C_h) be total variable cost per flown hour.

Expected counts and usage:

[
N_s = N p,qquad N_f = N(1-p)
]

[
H = N_s t,qquad PKM = N_s q d
]

Annual cost:

[
C_{annual} = F + N C_a + N_f C_f + N_s C_m + H C_h
]

Output metrics:

[
C_{hour} = rac{C_{annual}}{H}
]

[
C_{mission} = rac{C_{annual}}{N_s}
]

[
C_{passenger	ext{-}km} = rac{C_{annual}}{PKM}
]

These are *fully loaded* costs: annual fixed ownership costs are allocated across expected completed flight hours/missions/passenger-kilometres. The report also retains variable cost per flight hour separately so users can distinguish fixed-cost dilution from direct operating expense.

Because expected outcomes can be fractional (e.g. 90 expected completions from 100 attempts at 90% reliability), counts are analytical expectations, not integer predictions for a particular calendar year. Field comparison must use actual numerators and denominators, not just expected values.

Zero/negative/non-finite costs, invalid probabilities, malformed mission identity, an impossible annual schedule, arithmetic overflow and non-comparable mission/currency cases fail closed. A zero success fraction is rejected rather than returning a misleading zero-denominator cost.

## 4. Worked synthetic example

The following uses invented monetary inputs solely to verify arithmetic. They are **not real aircraft prices, maintenance figures, fuel prices, or a cost estimate for S4 or any Cessna**.

Assumptions for all rows: same 200 km route, two passengers, 1.5 block hours per mission, 90% dispatch success, 500-hour annual schedule budget, annual fixed cost 16,000 currency units, hourly variable cost 155 units, preparation 20 per attempt, failed-dispatch increment 100 per failure, and 50 per completed mission.

| Planned attempts/year | Expected completed missions | Expected failures | Flown hours | Passenger-km completed | Annual total cost | Cost / flown hour | Cost / completed mission | Cost / passenger-km |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 45 | 5 | 67.5 | 18,000 | 30,212.50 | 447.59 | 671.39 | 1.6785 |
| 100 | 90 | 10 | 135 | 36,000 | 44,425.00 | 329.07 | 493.61 | 1.2340 |
| 200 | 180 | 20 | 270 | 72,000 | 78,850.00 | 291.67 | 438.06 | 1.0951 |

The important result is not the numbers themselves; they are synthetic. It is the utilization effect: the annual fixed-cost contribution is spread over more completed work as utilization rises, while hourly variable cost remains. If dispatch reliability falls, the model charges for failed attempts and reduces completed passenger-kilometres, so cost per completed passenger-kilometre rises.

A real analysis must add sensitivity cases for fuel/energy price, insurance, financing/depreciation method, maintenance-hour rate, parts costs, overhaul/battery life, dispatch reliability and utilization. Publish the assumptions alongside every result.

## 5. Benchmark protocol for the aircraft family

Run each reference and candidate against the same frozen mission-family manifest at **50, 100 and 200 planned dispatch attempts per year**, subject to a realistic documented time budget and mission duration. In addition to the common family, preserve a separate family for genuinely different missions rather than comparing a touring mission to a short training circuit as if equivalent.

Required workflow:
1. Record source and revision for every cost/price/reserve assumption.
2. Separate annual fixed, per-attempt, per-completed-mission and per-flight-hour costs.
3. Record actual cancellation costs and successful/failed dispatch denominator.
4. Keep engine/propeller/battery reserves funded with a sourceable method.
5. Use a consistent route, passenger count, payload definition, route distance and scheduled attempt count when ranking candidates.
6. Record route completion, payload and reserve from the candidate-specific mission model. This cost module does not validate those inputs.
7. Compare costs with and without capital allocation, but clearly label the boundaries; do not combine a cash operating cost for one plane with an annualized capital cost for another.
8. Publish a full component decomposition and uncertainty/sensitivity report.
9. Replace synthetic cost data with invoices, maintenance logs, task-time measurements, parts lead times, insurance quotes and actual operating records as they become available.
10. Never improve a cost result by omitting a mandated inspection, underfunding overhaul/replacement reserves, excluding dispatch failures or reducing the payload/route/reserve requirements.

### Metrics to gather in the field

- Owner and licensed maintenance labor hours, separated by task/authority class.
- Scheduled and unscheduled parts/labor expenses, including repeat discrepancies.
- Engine/propeller/energy-storage overhaul or replacement reserve and life basis.
- Dispatch attempts, cancellations, failures, route completion and time grounded.
- Actual fuel/energy used at an explicit measurement boundary.
- Parts fill rate and order-to-owner lead time, especially parts that ground the aircraft.
- Flight hours and completed passenger-kilometres by actual mission profile.
- Aircraft age, configuration, environment and utilization, to avoid mixing incomparable samples.

AOPA's public ownership-cost guidance is a useful framing source because it calls out direct costs, fixed costs, reserves and variable expenses, and cautions against omitting reserves from a headline hourly estimate: https://www.aopa.org/go-fly/aircraft-and-ownership/buying-an-aircraft/tips-on-buying-used-aircraft/operating-costs-and-financing.

## 6. Implementation scope and limitations

The Rust implementation lives in the existing `symthaea-operations-research` crate and adds no third-party dependency. Its fourteen unit-test declarations cover a synthetic known answer, a 50/100/200-attempt utilization curve, owner/professional labor decomposition, dispatch effects, explicit zero components, negative/non-finite inputs, currency and mission-family mismatch, cost boundary/date mismatch, digest/source-reference errors, schedule budget and zero denominators, deterministic ranking, route/passenger mismatch, invalid calendar dates, and duplicate identifiers.

This is a deterministic accounting helper—not:
- a price database, operating-cost quote or investment-return promise;
- an aircraft performance, energy, weather or reserve solver;
- an inspection/maintenance predictor or reliability estimator;
- a method for authorizing owner maintenance;
- a certified design or evidence of cost superiority.

Local Rust test execution remains a separate evidence gate. Exact-head workflow results must be checked before claims of CI/test success.

## 7. Source context

- AOPA, [Operating Costs and Financing](https://www.aopa.org/go-fly/aircraft-and-ownership/buying-an-aircraft/tips-on-buying-used-aircraft/operating-costs-and-financing).
- Textron Aviation, [Skyhawk](https://cessna.txtav.com/en/piston/cessna-skyhawk), [Skylane](https://cessna.txtav.com/piston/cessna-skylane), and [Turbo Stationair HD](https://cessna.txtav.com/en/piston/cessna-turbo-stationair-hd) manufacturer baseline pages.
- Pipistrel, [Velis Electro endurance and powertrain service-life notes](https://www.pipistrel.eu/products/general-aviation/velis-electro/).

## Nonclaims

The example numbers are synthetic. The calculator does not prove that any concept is affordable or that any cost reduction has been achieved. A comparative cost claim requires identical mission/cost boundaries, sourceable inputs, realistic reserves, data quality/uncertainty, and independent review.
