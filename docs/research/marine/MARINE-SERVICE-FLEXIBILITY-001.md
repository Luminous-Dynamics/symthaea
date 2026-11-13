# MARINE-SERVICE-FLEXIBILITY-001
## Co-design schedule slack, battery storage and fleet delivery reliability

**Status:** research specification v0.1.0  
**Created:** 11 November 2026  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Related:** #11429 (route/service robustness), #11510 (WAPS 3-DOF coupling), #11473 (forecast-regret EMS)

> Research specification only. Schedule slack is a temporal service allowance, not stored energy, and must never be converted directly into kWh or fuel savings.

## 1. Paradigm shift: time is a constrained design resource

Ship design often treats battery capacity, generator rating, fuel reserve and propulsion efficiency as engineering variables while treating the commercial schedule as a fixed external input. Real cargo contracts and port operations frequently contain a mix of hard deadlines, arrival windows, berth windows, turnaround times and rotation slack.

The research hypothesis is that this temporal flexibility has measurable option value. A vessel may be able to:
- reduce speed when favourable wind, current or weather-routing conditions permit;
- avoid carrying some battery capacity solely to cover short propulsion transients or arrival-time recovery;
- shift energy use across legs while preserving the contracted arrival windows;
- reduce peak plant sizing or reserve requirements where dispatch and schedule flexibility interact;
- coordinate port calls, charging, maintenance and fleet rotations.

This does **not** make time equivalent to energy. It creates a separate temporal flexibility budget that must be co-optimized with energy storage and propulsion while preserving cargo service and reliability.

## 2. Published seed case: Kamsarmax 1-DOF versus 3-DOF routing

Kytariolou and Themelis (JMSE 14(2), 148, 9 January 2026; DOI 10.3390/jmse14020148) compare wind-assisted Kamsarmax weather-routing approaches.

The published Table 4 route comparison reports:
- 1-DOF-optimized route, evaluated at 3-DOF: total fuel consumption 187.16 t, route distance 2,781.29 nm, duration 10.48 days;
- 3-DOF-optimized route: total fuel consumption 184.64 t, route distance 2,804.91 nm, duration 10.57 days;
- difference: 2.52 t less total fuel, 23.62 nm longer (about 0.85%) and approximately 2.16 hours longer;
- the reported FOC-saving comparison is 7.07% versus 8.32%, a 1.25 percentage-point difference.

The fuel boundary includes main-engine and diesel-generator fuel. The dataset is identified as Zenodo DOI 10.5281/zenodo.17859015 under CC BY 4.0. Reproduce the paper's own cases and accounting before using the values elsewhere.

The crucial question is not whether 2.52 t is attractive in isolation. It is whether the additional time is eligible under the cargo contract, port slot, downstream departure, fleet rotation and recovery margin. A hard berth/arrival miss can reverse the apparent advantage.

## 3. Hypotheses to falsify

**H1 — Schedule slack can substitute for some onboard flexibility.** On selected route/service contracts, a schedule-flexible speed policy may achieve part of a battery-assisted fuel/peak-power benefit without carrying the same battery mass/volume.

**H2 — The trade-off is route- and port-dependent.** Slack at one call may be unusable because the next berth window, tide, pilotage, terminal throughput or fleet rotation is binding.

**H3 — Battery and schedule flexibility are complementary in some regimes.** Batteries may still be valuable for fast transients, reserve and short high-power events even when schedule slack is plentiful.

**H4 — Reliability tails determine the real value.** Mean arrival delay or mean fuel cannot hide missed berths, cancellation, missed onward connections or reserve depletion. Evaluate p90/p95 delay and missed-window probability.

**H5 — The value of slack depends on forecast skill.** An apparently flexible route may be fragile when forecasts are biased, stale or missing. Compare policy performance using only forecasts available at the decision time.

## 4. Service and time-window contract

For each port call, declare:
- earliest permitted arrival;
- target or nominal arrival;
- latest permitted arrival;
- berth/pilotage/tidal access window where available;
- minimum turnaround and earliest/latest departure;
- downstream rotation and onward-service commitments;
- contractual late-arrival consequence and any permitted early-arrival/waiting cost;
- cargo work and required delivered service.

For each candidate schedule, record actual/planned arrival and departure by port call. Compute per-call early margin, late slack available, late slack consumed and late slack remaining.

Do not sum per-call slack into one route-wide number unless the model proves it is conserved and transferable. Per-call windows can be independent, coupled or non-transferable.

## 5. Matched benchmark arms

Use the same vessel, cargo, service fingerprint, route/port calls, forecast realization, loading distribution and lifecycle boundary.

| Arm | Policy | Purpose |
|---|---|---|
| S0 | Fixed schedule, conventional powertrain | Baseline |
| S1 | Schedule-flexible speed policy, no added storage | Temporal flexibility only |
| S2 | Battery/hybrid sizing, fixed schedule | Storage value only |
| S3 | Schedule-flexible policy + battery/hybrid sizing | Interaction and complementarity |
| S4 | Forecast-aware schedule/energy MPC | Deployable joint policy |
| S5 | Perfect-foresight joint schedule/energy optimization | Offline upper bound only |
| S6 | Degraded forecasts, berth disruption and equipment outage | Tail-risk/fallback assessment |

A candidate must not be credited for lower energy if it reduces delivered cargo work, violates arrival/berth constraints, needs an extra vessel, increases missed calls or ends a voyage with a lower battery state of charge.

## 6. Required outcomes

Keep these distinct:
- fuel by consumer and whole-vessel total fuel;
- shaft/electrical/primary energy and conversion losses;
- battery usable capacity, mass, volume, degradation and replacement;
- arrival-time distribution, early waiting, late delay and window violations;
- berth/port waiting, turnaround, schedule recovery and fleet-rotation impact;
- annual delivered cargo work, fleet size and vessel availability;
- lifecycle cost and well-to-wake GHG;
- reserve and hard-gate status.

Report Pareto frontiers, not just a weighted total. A scalar objective may be reported only after the service owner declares the penalty and risk weights.

## 7. Deterministic first audit increment

The first standalone contract/auditor checks:
- exact service fingerprint shared by all comparison arms;
- exactly one baseline and at least one candidate;
- every contracted port call represented exactly once in every arm;
- RFC3339 arrival/departure timestamps;
- earliest <= target <= latest arrival;
- planned arrival within the declared hard window;
- planned departure respects minimum turnaround and port departure cutoff;
- per-call temporal slack arithmetic and explicit non-transferability;
- synthetic cases always remain INCOMPLETE and cannot produce fuel-savings claims.

It intentionally does not calculate vessel resistance, fuel savings, battery sizing, route optimization or schedule-to-energy conversion.

## 8. Research targets

Targets are aspirations, not results:
- 100% known-answer agreement for time-window and turnaround arithmetic;
- zero candidate schedules silently accepted outside hard arrival/berth windows;
- zero missing/duplicate port calls in comparison cases;
- 0% reduction in delivered cargo service or required availability;
- report p50/p90/p95 late arrival, early waiting and missed-window probability;
- report battery mass/volume/capacity and lifecycle replacement for every storage arm;
- schedule-flexibility advantage must persist on held-out weather/port/season blocks;
- a candidate ranks superior only if its fuel/cost benefit exceeds combined model/metrology uncertainty and all service hard gates pass.

## 9. Competitive positioning

Weather routing, hybrid-energy management, ship design optimization and port call optimization already exist as industry capabilities. The research contribution is their service-matched co-design: schedule slack, battery sizing, route policy, port windows, fleet rotation and evidence uncertainty are evaluated together under one declared cargo-service contract.

Do not claim the concept of flexible scheduling is new. Demonstrate measurable improvement in the joint Pareto frontier and independent reproducibility against the published Kamsarmax and permissioned operator cases.

## 10. Claim ceiling

Temporal slack is not energy storage. A schedule audit is not a voyage optimizer. Published route results are case-specific. Synthetic fixtures do not establish fuel savings. A schedule-flexible policy must never compromise safe navigation, bridge authority, statutory limits, port requirements or cargo-service commitments.
