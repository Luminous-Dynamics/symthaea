# MARINE-SAIL-LIVEABOARD-CO-DESIGN-001

**Status:** Research contract / P0 composition profile  
**Created:** 2026-11-19  
**Parent issue:** [Symthaea #12884](https://github.com/Luminous-Dynamics/symthaea/issues/12884)  
**Claim ceiling:** Cross-owner composition only. Synthetic contract conformance is not a physical performance, safety, seaworthiness or regulatory result.

## 1. Purpose

Join two mature but separate workstreams:

- wind-propelled sailcraft route candidates, uncertainty-aware VMG, tack/gybe costs and configuration-bound capabilities;
- long-duration liveaboard mission energy, power traces, resource-service scheduling, multi-resource autonomy and service continuity.

The benchmark asks which sailing/propulsion policy completes the **same mission and delivers the same onboard services** with the lowest defensible resource burden, subject to non-compensable safety and service constraints.

It is not another sailing simulator, weather router, energy model, habitat simulator or controller. It consumes owner-native outputs and verifies cross-domain identity, completeness, temporal availability, comparability and outcome reconciliation.

## 2. Why single-objective sailing is insufficient

A route with the highest steady VMG may not minimize full-mission burden. Tack and gybe losses, motor-assist demand, solar availability, shore-power windows, hotel/HVAC loads, watermaker duty, battery degradation, water and waste constraints, arrival windows and watchkeeping workload can change the ranking.

The opposite may also occur: a longer sailing route may create a more reliable service outcome or a better total-energy result. Both outcomes are hypotheses to test.

## 3. Canonical owners

| Owner | Consumed output | Must not be reimplemented here |
|---|---|---|
| #12153 / #12158 | Sailcraft route-candidate contract | Wind/polar route kernel |
| #12161 / #12170 | Robust VMG / wind-current-polar intervals | Robust route evaluator |
| #12310 / #12311 | Tack/gybe time and signed progress envelopes | Manoeuvre dynamics |
| Symtropy #5254 / #5219 | Sail capability graph and exact-scope disposition | Qualification compiler |
| Symtropy #3893, #3865; #11185, #12214 | ODD, mode, manoeuvring, COLREG and oversight references | Navigation authority / legal interpreter |
| #8092 / #8102 | Mission energy budget and voyage power traces | Energy arithmetic / trace integrator |
| #11008 | Resource-service schedules and flexible-load decisions | Service scheduler |
| #10956 / #10994 | Resource-autonomy vector and watermaker evidence | Life-support / RO process physics |
| #10988 / #11115 / #11126 | Mission evidence join, service continuity and restoration | Evidence ontology / service-continuity evaluator |
| #8082 / #11442 / #11513 / #11765 | Hydro/propulsor, model V&V, fidelity and robust Pareto | CFD, hydrodynamics, optimizer, Pareto engine |
| #11064 / #11093 | Physical campaign binding and staged demonstrator | Instrument registry / acquisition hardware |
| Symtropy #2048 | Shipbuilding design-for-manufacture and assembly | Shipyard scheduler / manufacturing ledger |

## 4. Comparison protocol

Every policy arm receives the same craft identity, loading, mission/service fingerprint, forecast issue-time snapshots, route constraints, resource inventory, hard gates and compute budget.

| Arm | Baseline / candidate |
|---|---|
| S0 | Fixed conservative schedule and propulsion policy |
| S1 | Polar-table VMG / conventional tack selection |
| S2 | S1 plus evidence-bound tack/gybe transition costs |
| S3 | Forecast-aware receding-horizon weather routing |
| S4 | S3 plus energy reserves and essential-service floors |
| S5 | Integrated advisory evaluation of route candidates and resource-service schedules |
| S6 | Perfect-foresight evaluator-only upper bound; never deployable |

ORC VPP 2026 is a reference for steady-state sailing speed and aerodynamic/hydrodynamic force-balance practice. It is not the integrated liveaboard benchmark. Autopilot and commercial weather-routing products remain comparators; no superiority is inferred from feature counts.

## 5. Time and information contract

Each decision step declares its decision time, forecast source issue time, availability time, valid interval and forecast receipt. A candidate may only consume observations/forecasts available at its decision time. The realized environment is evaluator-only. A perfect-foresight arm is useful as a lower-bound cost / upper-bound performance reference, but not as a deployable competitor.

Cross-clock comparisons, stale or missing critical forecasts, source/configuration mismatches and out-of-domain polar requests produce explicit unknown/incomplete/blocked outcomes; they do not silently inherit prior capability.

## 6. Energy and resource boundaries

Keep all of these separate:

- propulsion electrical input;
- propulsion fuel chemical energy and fuel mass;
- hotel / auxiliary electricity;
- shore electricity;
- generator fuel and operating hours;
- solar energy delivered to the bus, actually used and curtailed;
- battery usable energy, state of health, throughput and conversion loss;
- water produced, used and diverted off-spec;
- food/nutrient coverage, waste-storage capacity, consumables, spares and emergency reserves.

No nameplate PV capacity is credited as delivered energy. No absent energy channel is zero. No energy, water, food, waste or crew workload is collapsed into an arbitrary common scalar. The autonomy result remains a per-dimension vector.

## 7. Frozen scenario families

1. Coastal passage with useful solar and a shore-power arrival window.
2. Offshore passage with no shore-power and constrained resupply.
3. Variable wind where point and interval VMG rankings differ.
4. Upwind no-go zone with tack/gybe and motor-assist decisions.
5. Stale forecast, missing current or forecast disagreement.
6. Cloud/solar shortfall plus increased hotel/HVAC demand.
7. Battery ageing, temperature derating and changed charge limits.
8. Generator unavailable or fuel inventory uncertain.
9. Watermaker duty/flush windows compete with propulsion and domestic loads.
10. Guests, cooking, refrigeration, work and rest requirements change.
11. Water, food, waste or critical consumables become limiting before energy.
12. Sail/steering sensor or actuator fault, failed transition, or reef/furl capability withdrawal.
13. Traffic, visibility or weather pushes a candidate outside its ODD.
14. Common-cause failure affects purportedly redundant generation/sensing/communications.
15. Configuration change or evidence revocation invalidates only dependent capabilities.
16. No policy meets all gates: return no feasible mission plan, with reasons.

## 8. Metrics

**Hard gates:** false-feasible plan count; essential-service floor breaches; ODD/COLREG/authority gate failures; unknown or stale inputs treated as valid; future-information leakage; accounting incompleteness; unauthorized commands (must remain zero).

**Sailing and service:** distance, passage time, arrival-window success, VMG, manoeuvre count, signed transition losses, motor-assist hours, delivered occupied berth-days, service availability, comfort, workload and maintenance burden.

**Energy:** propulsion electric kWh; fuel mass and chemical energy; hotel/aux kWh; shore kWh; generator hours; conversion losses; renewable energy used/curtailed; battery throughput and degradation proxy. Compare only like-for-like boundaries.

**Resource autonomy:** water, food/nutrients, waste capacity, consumables, spares and emergency-reserve margin by dimension. Report the minimum dimension and uncertainty, not one “autonomy days” value.

**Robustness:** worst-case route/service regret, forecast regret, robust feasibility coverage, plan churn, held-out condition degradation, abstentions, p50/p95/p99 runtime and compute budget.

## 9. Research targets

These are preregistered challenge targets, not achieved performance:

- Zero false-feasible results on frozen missing/stale/revoked/unsupported negative controls.
- Zero hard essential-service violations in accepted plans.
- 100% matched mission/configuration/occupancy/service fingerprints.
- 100% energy/resource accounting completeness; missing remains unknown.
- Zero future-observation leakage.
- Exact agreement with an independent hand-calculation oracle on synthetic join cases.
- At least 10% lower purchased-energy per occupied berth-day (or total purchased energy per matched mission) than the strongest information-matched baseline over at least three preregistered wind/solar/load strata, with conservative improvement interval above zero and no service/schedule/safety regression.
- At least 10% lower worst-case regret than the reserve-aware baseline on held-out forecast errors, with no increase in hard-gate failures.

Do not force an energy reduction target where it would trade away safety, schedule, comfort or service. If uncertainty intervals overlap, report unresolved dominance. Publish full per-stratum results, failed/abstained cases and Pareto trade-offs.

## 10. P0/P1/P2/P3 release sequence

- **P0:** Strict case schema, owner-reference map, synthetic positive/negative cases, independent join oracle, time-availability checks, no physics/planner.
- **P1:** Candidate comparison by consuming native receipts and independently rechecking identity, service match, gate completeness and objective reconciliation.
- **P2:** Rights-reviewed shadow replay and held-out route/weather/load cases; outputs advisory/read-only.
- **P3:** Extend LOS-18 staged demonstrator with sailcraft/propulsion electrical interfaces; bench/dockside instrumentation first, then permissioned sheltered-water trials after qualified review.

## 11. Standards and external references

- [ORC 2026 VPP documentation](https://orc.org/organization/rules-regulations): steady sailing speed prediction using hull/appendage/sail force models and equilibrium; comparison anchor only.
- [ISO 16315:2026](https://www.iso.org/standard/84203.html): AC/DC electrical propulsion and hybrid electrical systems for small craft up to 24 m LH.
- [ISO/DIS 12217-2.2:2026](https://www.iso.org/standard/88606.html): draft stability/buoyancy assessment for sailing boats, principally 6–24 m; includes relevant habitable-multihull inversion/escape subjects; not yet final.
- [IMO MASS Code](https://www.imo.org/en/mediacentre/pressbriefings/pages/imo-adopts-mass-code.aspx): non-mandatory code adopted May 2026, effective 1 July 2026 for SOLAS cargo ships; application to a small private liveaboard must not be presumed.
- [ISO/AWI 25930](https://www.iso.org/standard/92074.html): approved work item, under development; its ODD/OE taxonomy is a reference, not a released requirement in this profile.

## 12. Claim ceiling

A P0 pass means only that the composition contract, temporal-information rules and frozen synthetic examples behave as specified. It does not validate polar data, weather forecasts, hull stability, actual sailing performance, battery endurance, solar yield, water quality, habitability, autonomous navigation, COLREG compliance, seaworthiness, class/flag certification or permission to operate. Every physical claim requires source-bound, independent validation on the exact configuration and operating envelope.
