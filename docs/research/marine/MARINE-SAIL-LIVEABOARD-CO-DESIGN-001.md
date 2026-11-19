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
| Sol-Atlas #115 / #654 / #809 | Source-qualified spatial/temporal maritime projection and metocean/access context; CRS/datum, coverage, uncertainty, rights and reversible source links | Navigation, weather forecasting or bathymetric authority |
| Mycelix #5968 / #5974 / #6154 | Cross-stack golden-trace coordination and serialized vessel/component configuration, manufacturing, inspection and repair lineage | Naval-architecture approval or engineering qualification |
| Symtropy #2048 | Shipbuilding design-for-manufacture and assembly | Shipyard scheduler / manufacturing ledger |

Sol-Atlas context is read-only and source-reversible: it may qualify where/when a metocean or access observation applies, but does not turn a mapped value into a forecast or navigation instruction. Mycelix binds exact component/configuration/maintenance evidence and the cross-stack golden trace; it does not make the originating engineering claim true.

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
14. A commercial or research router returns a route only after automatically relaxing a declared hard comfort limit; preserve it as BLOCKED for that hard-comfort mission rather than counting it as compliant.
15. Common-cause failure affects purportedly redundant generation/sensing/communications.
16. Configuration change or evidence revocation invalidates only dependent capabilities.
17. No policy meets all gates: return no feasible mission plan, with reasons.

## 8. Metrics

**Hard gates:** false-feasible plan count; essential-service floor breaches; ODD/COLREG/authority gate failures; unknown or stale inputs treated as valid; future-information leakage; accounting incompleteness; unauthorized commands (must remain zero); route results automatically relaxed beyond declared hard comfort limits (must remain zero).

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

- **P0:** Strict case schema, explicit hard-gate evaluations, synthetic fixture, and standalone Rust audit CLI (`tools/marine-sail-liveaboard-audit`) with semantic/time-availability checks and a matched-case oracle; no physics/planner.
- **P1:** Candidate comparison by consuming native receipts and independently rechecking identity, service match, gate completeness and objective reconciliation.
- **P2:** Rights-reviewed shadow replay and held-out route/weather/load cases; outputs advisory/read-only.
- **P3:** Extend LOS-18 staged demonstrator with sailcraft/propulsion electrical interfaces; bench/dockside instrumentation first, then permissioned sheltered-water trials after qualified review.

## 11. Standards and external references

- [ORC 2026 VPP documentation](https://orc.org/organization/rules-regulations): steady sailing speed prediction using hull/appendage/sail force models and equilibrium; comparison anchor only.
- [ISO 16315:2026](https://www.iso.org/standard/84203.html): AC/DC electrical propulsion and hybrid electrical systems for small craft up to 24 m LH.
- [ISO/DIS 12217-2.2:2026](https://www.iso.org/standard/88606.html): draft stability/buoyancy assessment for sailing boats, principally 6–24 m; includes relevant habitable-multihull inversion/escape subjects; not yet final.
- [IMO MASS Code](https://www.imo.org/en/mediacentre/pressbriefings/pages/imo-adopts-mass-code.aspx): non-mandatory code adopted May 2026, effective 1 July 2026 for SOLAS cargo ships; application to a small private liveaboard must not be presumed.
- [ISO/AWI 25927](https://www.iso.org/standard/92071.html): collision-avoidance scenario-generation process for MASS; approved work item under development. Useful for the scenario corpus, not an established compliance procedure.
- [ISO/AWI 25929](https://www.iso.org/standard/92073.html): minimum safe lookout area method for MASS; approved work item under development. Useful for specifying perception/observation coverage, not evidence that a sensor suite is safe.
- [ISO/AWI 25930](https://www.iso.org/standard/92074.html): ODD/OE taxonomy and definition format for MASS; approved work item under development. It explicitly does not define attribute test procedures or monitoring requirements.
- [ISO/AWI 25922](https://www.iso.org/standard/92067.html): capability/responsibility allocation between a MASS and remote operators; approved work item under development.
- [ISO/CD 25928](https://www.iso.org/standard/92072.html): software functional-safety management for MASS; committee draft under development.
- [ISO/CD 25934.2](https://www.iso.org/standard/92078.html): guidance for developing verification procedures for autonomous navigation systems; committee draft under development.
These projects are standards-watch inputs only. Do not treat a work item or committee draft as a published requirement or claim conformance from schema similarity.

## 12. Industry comparison: what is already commercially available

The comparison must recognize capabilities that already exist in commercial sailing systems:

- [PredictWind Sail Routing](https://help.predictwind.com/en/articles/2884532-how-to-use-weather-routing-for-sailboats-on-the-forecast-website) offers fastest-time and comfort routing, sail polars, forecast-driven routes and motion metrics (RMS roll, vertical acceleration and slamming). Its published guidance says that when no route meets the selected comfort limits, the product automatically reruns without those limits so the user still receives a route. Our benchmark must preserve the returned route's actual constraint status: an automatically relaxed route cannot be counted as satisfying a hard comfort requirement.
- [PredictWind AI Polars](https://www.predictwind.com/features/ai-polars) uses connected vessel data to adapt boat polars and includes wave-state features and night behaviour. The provider describes the result as a five-dimensional proprietary performance dataset and says it cannot be downloaded/exported for use in other systems. Personalised polar learning is therefore not a differentiator. An independent comparison needs an authorized black-box trial with captured settings/forecast provenance, or a separate rights-cleared measured polar; it must not pretend an exported GPX reproduces the commercial polar or forecast inputs.
- [Raymarine wind-vane / steer-to-wind modes](https://docs.raymarine.com/81406/en-US/latest/AutopilotWindVaneMode-6A57E8F8.html) include apparent-wind, true-wind and polar-target wind-angle steering. This means holding a wind angle or following a polar target is a baseline capability, not a novel autonomy claim.
- [Garmin SmartDrive (announced 1 September 2026)](https://www.garmin.com/en-US/newsroom/press-release/marine/garmin-launches-innovative-smartdrive-autopilot-for-sailboats-and-catamarans/) extends the installed autopilot ecosystem with a brushless linear actuator, integrated ECU and rudder feedback for sailboats/catamarans up to 45 ft and declared displacement limits. This is hardware/control-system evidence, not proof of autonomous mission planning.

**What would actually differentiate this programme:** not steering accuracy, a learned polar, or another weather route. It is a reproducible, owner-independent evaluation of route choices against complete onboard service delivery and multidimensional resource autonomy, with temporal evidence admissibility, fault/revocation propagation, per-configuration capability scope, comfort-constraint non-relaxation accounting, and blind forecast-regret evaluation. Commercial products may already optimize comfort, route time and boat-specific polars; our benchmark must compare those outputs fairly rather than dismiss them.

## 13. Claim ceiling

A P0 pass means only that the composition contract, temporal-information rules and frozen synthetic examples behave as specified. It does not validate polar data, weather forecasts, hull stability, actual sailing performance, battery endurance, solar yield, water quality, habitability, autonomous navigation, COLREG compliance, seaworthiness, class/flag certification or permission to operate. Every physical claim requires source-bound, independent validation on the exact configuration and operating envelope.


## 14. P0 implementation status (2026-11-19)

- The P0 standalone Rust audit is source-committed under `tools/marine-sail-liveaboard-audit/`.
- The CLI separates `INVALID`, `BLOCKED`, `INCOMPLETE`, `SYNTHETIC_ONLY` and `ELIGIBLE_FOR_COMPARISON`; only the first four can arise from the current synthetic fixture.
- Pair comparison checks identical mission/service/configuration/environment/gate inputs and excludes the sail-route candidate output from shared-input identity comparison.
- Sixteen audit-engine tests and seven corpus-validator tests are authored, including end-to-end replay of all sixteen frozen mutation vectors through the shared Rust library.
- Exact Rust formatting, compilation, Clippy and test execution are pending because the current environment does not expose a Rust toolchain. The workflow is path-scoped and will run independently; no CI pass is claimed.


## 15. Frozen paired-mission adversarial corpus

`docs/research/marine/fixtures/marine-sail-liveaboard-corpus-0001.json` is the versioned synthetic scenario registry; its contract is `docs/research/marine/marine-sail-liveaboard-corpus.schema.json`. The initial corpus freezes 16 cases and all seven arms S0-S6. It includes temporal forecast leakage, impossible issue/availability order, realized-truth leakage, failed or omitted hard gates, missing/duplicated autonomy dimensions, hard-comfort fallback, missing fallback provenance, malformed identity, and four paired-case mismatch/upper-bound controls.

Each vector declares a command (`audit_case` or `compare_pair`), mutation target, JSON-Pointer mutation operations, expected disposition and either expected finding codes or mismatched fields. The audit implementation is now a shared Rust library; the `validate_corpus` binary applies the frozen mutations to the original synthetic fixture, calls the same `audit_case` / `compare_pair` implementation used by the primary CLI, and compares actual outcomes with every expected disposition and finding/mismatch assertion. Its output is `CORPUS_EXECUTED` only when all 16 cases match. The validator has seven authored tests, including end-to-end execution of the frozen corpus. The exact Rust format/build/Clippy/test run remains pending CI; these are not reported as passing.

All policy arms use the same declared budget profile. S0-S5 are deployable information sets restricted to data available at each decision time. S6 is an evaluator-only perfect-foresight bound, never a competitor or deployable policy. Scenario identifiers and expected outputs are frozen; changing either requires a corpus-version increment and review.

## 16. Benchmark design and SOTA comparison protocol

### Information-matched baselines

Compare against the strongest feasible, rights-cleared baseline at the exact craft and configuration:
- S0 fixed conservative schedule;
- S1 measured or independently validated polar-table VMG and conventional tack selection;
- S2 transition-cost-aware VMG;
- S3 a current commercial-grade weather route (PredictWind where licensed and reproducibly exportable) or a documented equivalent using the same forecast snapshot;
- S4 the best reserve-aware route/service policy available from existing owners;
- S5 this integrated advisory composition;
- S6 a perfect-foresight evaluator bound used only to estimate the information gap and regret.

PredictWind already provides fastest-time and comfort routing, wave-motion metrics, connected-data personalized AI polars and night adaptation. Its current help material says it attempts the requested comfort constraints and then reruns without them when no compliant route is found. Therefore the decisive test is not whether this programme can produce a route: it is whether hard/soft status is preserved, the constraints are never silently relaxed, and the route is compared on identical services, forecast information and craft inputs. Proprietary systems are not assumed to export all intermediate state; when they do not, mark the comparison incomplete rather than reverse-engineering or inventing missing evidence.

ORC VPP is a recognized steady-state force-balance and speed-prediction reference for sailing performance. Reproduce/compare that narrow sailing layer before making hull or polar claims; it is not a liveaboard service-availability benchmark. Raymarine wind-angle steering and Garmin's sailboat autopilot hardware remain control-system comparators, not proof of an integrated autonomous mission planner.

### Primary and secondary measures

Report the complete resource vector and carrier-specific energy ledger. The preregistered headline efficiency measure is purchased-energy equivalent per delivered occupied berth-day, with shore electricity, fuel LHV, generator output and conversion losses disclosed separately. Never let that headline replace the carrier-specific ledger or hide water, food, waste, spares, reserve, service or comfort regressions.

For robust performance, report paired mission regret and CVaR95 across held-out forecast-error/weather episodes, plus route/arrival regret, missed service floors, plan churn and p50/p95/p99 compute latency. Report all safety and service gates as counts and denominators, including unknown, abstained and infeasible cases.

### Statistical protocol

Freeze vessel/configuration, displacement/loading, occupancy, service profile, start/arrival constraints, forecast issue snapshots, policy versions and compute budgets before evaluation. Split by whole voyage or independent weather episode—not adjacent timesteps—so near-duplicate weather does not leak between development and evaluation. Stratify at minimum across three preregistered wind/sea-state, solar/load and forecast-error regimes. Determine sample size prospectively from pilot variance and a power analysis; do not call an arbitrary small scenario count statistically conclusive. Use paired trajectories and block-bootstrap confidence intervals by voyage/episode. Publish all strata, failures, abstentions, incomplete records and Pareto trade-offs.

### Challenge targets, not achieved results

- Zero false-feasible plans on the frozen negative controls.
- Zero accepted hard-gate or essential-service violations; zero future-information leakage and zero unauthorized commands.
- 100% exact mission/configuration/occupancy/service matching before a paired score is emitted.
- 100% independent-oracle agreement on hand-checkable synthetic cases.
- At least 10% lower purchased-energy equivalent per delivered occupied berth-day than the strongest information-matched deployable baseline, across at least three strata, with a conservative paired improvement interval above zero and no degradation in arrival, comfort, service continuity, resource autonomy or hard constraints.
- At least 10% lower CVaR95 mission regret than the strongest reserve-aware baseline on held-out forecast-error episodes, with no increase in gate failures.
- Runtime and compute-budget reporting at p50/p95/p99; quality improvement must be measured at equal information and declared compute budgets.

The 10% figures are aspirational research gates, not forecasts or reported performance. Failure to beat the baseline is a valid result. If uncertainty intervals overlap, or the vendor baseline cannot be reproduced with equivalent information, report no demonstrated superiority.



## 17. Forecast replay source qualification

The source inventory is pinned in docs/research/marine/sources/marine-forecast-replay-source-register-0001.json; deterministic implementation and actual dataset retrieval are tracked separately in [Symtropy #6209 — MARINE-FORECAST-BLIND-REPLAY-001](https://github.com/Luminous-Dynamics/symtropy/issues/6209).

The key distinction is information class, not just dataset format:

- **Deployable forecast candidates:** historical NOAA/NCEP GFS forecast cycles with cycle issue time, forecast lead, first demonstrable publication/availability time, valid time, object digest, parameter set, grid and units preserved. GFS is a forecast product; archived copies and source migration/retention limitations still require per-object qualification.
- **Evaluator-only retrospective context:** ERA5 atmospheric reanalysis and Copernicus Global Ocean Waves Reanalysis. They can help reconstruct a retrospective environment, but reanalysis assimilates observations and is not an operational forecast or direct in-situ ground truth. They must never be visible to a deployable policy.
- **Independent point checks:** NOAA NDBC buoy/C-MAN observations, subject to station coverage, sensor metadata, quality flags and representativeness uncertainty. They do not provide a continuous route-wide field.
- **Commercial route references:** PredictWind GPX/KML/CSV exports, where access is licensed. A route geometry file alone does not reproduce the forecast cycle, vessel polar, route constraints, comfort settings or fallback behaviour; absent inputs make the benchmark incomplete.

For each forecast object, preserve source_issue_time, first_available_at, valid_time, decision_cutoff, forecast_lead, exact model/cycle/version, product/grid/CRS/units, URI and immutable SHA-256 digest. retrieval_time is a separate field and may not be substituted for first availability. An archive's ingest time is not presumed to equal original operational publication time.

No raw dataset has been downloaded or declared rights-cleared by this source inventory. The first replay milestone is a small rights-reviewed set of whole weather episodes, frozen with raw object manifests and availability receipts, independent evaluator-side reanalysis, representative buoy checks and held-out voyage/episode splits. If original forecast availability cannot be reconstructed, classify the episode as non-deployable replay or incomplete rather than backfilling with reanalysis.


## 18. Grouped forecast-replay holdout protocol

The Symtropy replay-admission work [#6209 / PR #6215](https://github.com/Luminous-Dynamics/symtropy/pull/6215) now includes a versioned split-manifest schema and validator. The split unit is explicitly joint voyage/weather-event grouping: no complete voyage or declared weather event may cross development, validation or test. An identical replay-bundle digest may not be reused across folds, and overlapping episode windows within one voyage are rejected.

Each frozen episode binds its voyage ID, weather-event ID, replay-bundle digest, exact craft-configuration digest, start/end times, environment strata and ordered decision-snapshot references. The split manifest pins the dataset-manifest digest, split algorithm and seed, and carries its own immutable output identity. The current synthetic fixture contains nine independent episode groups (three per fold) and eighteen decision samples; the held-out test fold covers multiple environment strata.

Random row-level and adjacent-timestep train/test splits are not admissible for forecast-regret or mission-generalization claims. The validator establishes only that the declared partition is grouped and internally consistent. It does not prove statistical independence between real weather events, guarantee sample size, establish statistical power or establish superior performance. Sample size must still be set by prospective power analysis, and inference should use whole-voyage/weather-episode blocks.
