# Clean Airline Public-Data Benchmark Contract

**Status:** source register and data-contract proposal; no real-world benchmark has been run.  
**Owner:** CLEAN-AVIATION-DATA-001 (#8986), connected to CLEAN-AVIATION-001 (#8930) and rolling-horizon issue #8981.  
**Review date:** 2026-10-29.

## 1. Purpose and claim boundary

This contract defines how public flight-performance data and public solar/meteorological data may be used to establish a reproducible aviation/airport benchmark. It exists to prevent a common methodological error: combining observable flight operations with synthetic aircraft energy, tariffs or airport load, then presenting the resulting simulation as an observed clean-airline saving.

The benchmark has three independent evidence tracks:

1. **Observed operations:** flight events, schedule/actual timing, delays, cancellations and other fields actually present in a selected dataset.
2. **Resource context:** historical solar/meteorological series and, only where sources exist, grid/tariff/airport resource measurements.
3. **Modelled clean-airline energy:** aircraft charging demand, propulsion/mission limits, electricity/fuel conversion and infrastructure constraints. These are separate inputs from validated models, measured airport data or clearly labelled synthetic assumptions.

A joined result receives the weakest applicable evidence label for every derived claim. The public flight data does not automatically provide electric-aircraft charging energy. Solar irradiance does not equal metered airport PV output. Synthetic demand or tariff inputs remain synthetic in every downstream result.

## 2. Source register

Every ingestion run must materialize a source manifest with canonical URL, publisher, dataset/table name, release/schema/methodology version, coverage, retrieval timestamp, access/license/terms, accepted use, exclusions, units and time basis, exact partition, file byte count and cryptographic digest. Record the digest of the source archive and of every normalized partition. Never replace a version in place.

### S1 — EUROCONTROL / OpenSky Open Performance Data Initiative (OPDI)

- Landing page: https://www.opdi.aero/
- Data/schema: https://www.opdi.aero/data
- Flight-event downloads: https://www.opdi.aero/flight-event-data
- Methodology: https://www.opdi.aero/methodology
- Concepts: https://www.opdi.aero/concepts
- Roadmap: https://www.opdi.aero/roadmap
- Publicly listed release on the review date: v0.0.2, coverage January 2022 through September 2026, refreshed October 2026.
- The homepage reports more than 79 million flights, 1.9 billion flight events and 3.8 billion measurements.
- Published entities include FLIGHT, EVENT and MEASUREMENT; trajectory data comes from OpenSky Network state vectors. The methodology describes inferred trajectory-based flight milestones. Current event examples include take-off, landing, top-of-climb/descent and specific parking-position events, depending on version.
- Important caveat: the methodology page lists v0.0.2 methodology versions from May/June 2024 while the current release is refreshed in October 2026. Pin both dataset release and methodology version, inspect changelog/roadmap, and do not assume the release refresh means the event-inference algorithm changed.
- Not equivalent by default to airline schedule/gate timestamps, actual off-block/in-block times, certified taxi records, airport energy meters, aircraft type/variant, or electric-aircraft charge requirements.
- License/terms/access scope must be recorded from the publisher portal at download time; “publicly available/open data initiative” is not a substitute for archiving the applicable terms.

### S2 — U.S. Bureau of Transportation Statistics (BTS), Reporting Carrier On-Time Performance

- Table / field selector: https://transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=&gnoyr_VQ=FGJ
- Table description: https://transtats.bts.gov/TableInfo.asp?QO_fu146_anzr=b0-gvzr&V0s1_b0yB=D&gnoyr_VQ=FGJ
- Official Aug 2026 release (published 19 Oct 2026): https://www.bts.gov/newsroom/airline-service-quality-performance-data-august-2026
- Official delay/cancellation charts: https://transtats.bts.gov/OT_Delay/ot_delaycause1.asp
- On review date, latest available data is August 2026. Reporting Carrier On-Time Performance is monthly and described as non-stop domestic U.S. flights, covering 1987–present.
- Listed fields include scheduled/actual departure and arrival, cancellations/diversions, taxi-out/taxi-in, delay causes, air time and non-stop distance.
- Use this as a U.S.-specific observed operations benchmark, not a global airline distribution or a substitute for a South African airport dataset.
- Preserve definitions for delayed flights, cancellations, diverted flights, causes, and missing/late reports. Do not infer passenger connections, crew legality or aircraft rotation unless joined data and validity checks exist.
- Field selection, data dictionary version and selected field names must be pinned in the source manifest.

### S3 — ECMWF / Copernicus Atmosphere Monitoring Service (CAMS) solar radiation time series

- Publisher: https://www.ecmwf.int/en/forecasts/datasets/cams-solar-radiation-time-series
- Includes global horizontal, direct horizontal, diffuse horizontal and direct normal irradiation time series. Publisher lists historical coverage 2004 through 30 June 2026 on the review date, with all-sky coverage over Africa within the specified Meteosat/MSG satellite footprint.
- Offered formats include ASCII and NetCDF; archive the chosen product, time basis, grid/location and retrieval information.
- Irradiance is not actual airport PV power. Any PV electricity calculation must declare module/inverter/temperature/loss assumptions and carry the modelled evidence status; no claim of measured production is permitted without meter evidence.

### S4 — European Commission Joint Research Centre PVGIS 6

- Publisher: https://photovoltaic-geographic-information-system.ec.europa.eu/en/about-pvgis-6
- Public solar-radiation and PV performance information for most worldwide locations, excluding the poles. PVGIS 6 documentation describes hourly solar radiation and meteorological data for 2013–2024 and notes the web tool/API is beta.
- Pin the product version, coordinates, horizon/elevation assumptions, PV configuration, API request parameters, response payload, retrieval timestamp and digest. PV output computed by PVGIS is a modelled estimate, not an airport meter record.
- Do not mix CAMS and PVGIS periods/series silently. Differences in temporal coverage, spatial resolution, cloud representation and model assumptions must be reported.

### S5 — Publication baselines (protocol references until reproducible data/formulation is available)

1. Flight-aware charging under airport-specific pricing and demand charges, deterministic/SAA/robust optimization, six airports in China's Greater Bay Area, published cost savings up to 5.77% under high demand uncertainty: https://www.sciencedirect.com/science/article/pii/S0967070X26002398
2. Aircraft charging coordinated with landside EV-to-grid using a linear program, real grid prices and synthetic parking-garage occupancy; reported cost savings up to 32% depending on schedule and EV participation: https://ieeexplore.ieee.org/document/11616483/
3. Solar airport microgrid with aircraft-to-grid and fixed storage; study-specific reported cost reduction approximately 10.35%: https://ieeexplore.ieee.org/abstract/document/11578850
4. Physics-based reinforcement learning for airport thermo-electric microgrid energy management, evaluated against a classical optimizer on an airport microgrid: https://ieeexplore.ieee.org/abstract/document/11701290
5. Collaborative airline/network-manager recovery with fairness protection, evaluated on real operational data: https://www.sciencedirect.com/science/article/abs/pii/S1366554526003169

These are separate formulations with different boundaries, datasets, assumptions and objective functions. They do not form one comparable leaderboard. The published headline percentages must not be transferred to Symthaea, added together, or treated as targets already achieved. Where full text/data or assumptions cannot be obtained, label the work “related synthetic comparison,” not exact reproduction.

## 3. Required evidence labels

Use one of these labels for every input and downstream claim:

- **Observed:** directly measured or recorded by the cited source at the declared granularity.
- **Derived:** deterministic transformation of source-observed values, with code/version and units pinned.
- **Modelled:** output of a named model with version, assumptions, calibration domain and uncertainty.
- **Supplier-reported:** supplied by a vendor/operator and not yet independently measured.
- **Synthetic:** intentionally constructed test value or scenario.
- **Unavailable:** the required field/source has not been found or access is unavailable.
- **Unresolved:** data exist but their meaning, units, validity, join or applicability has not been established.

Do not silently promote Derived to Observed, Modelled to Measured, or Synthetic to Real. Unknown values are not zero.

## 4. Join and time contract

- Internally represent timestamps in UTC with explicit timestamp precision. Convert to local civil time only for known tariff/airport schedule rules, using a pinned IANA time-zone identifier and database version where possible.
- Keep event timestamps, schedule timestamps and observation/ingestion timestamps separate.
- Preserve event source, algorithm/methodology version and event semantics on every normalized event.
- Do not join flights only by callsign or approximate timestamp where duplicate flights are possible. Define a stable join key, validate cardinality, and quantify ambiguous/unmatched rows.
- Keep arrival, landing, in-block, on-block, gate arrival, take-off, off-block, taxi and departure notions distinct. An inferred trajectory milestone is not automatically a gate-event record.
- Store source units and normalized units together. Every conversion must be explicit and tested.
- Do not join a trajectory event to solar data by date alone: use the declared airport/location, UTC interval and spatial/temporal resolution.
- A conventional flight record does not specify battery size, usable energy, electrical charging power, turnaround charger compatibility, aircraft mass, or propulsion efficiency. Those require a separate aircraft configuration and qualified mission model or remain synthetic.
- Airport electricity tariffs and demand charges must be tied to exact tariff schedules and effective dates. If no source is available, label them synthetic rather than borrowing a different jurisdiction's tariff.

## 5. Temporal split and leakage policy

Every benchmark release must freeze a split manifest before tuning:

- chronological train / development / calibration / held-out test windows;
- disruption-event clusters kept intact within one partition;
- no random row-wise split where adjacent observations from one flight/event can cross partitions;
- for rolling policy tests, the policy at time t may use only observations/forecasts available at t; realized future values are forbidden;
- calibration data are not a second test set; any repeated test-set tuning invalidates that release's test score;
- source files, normalization code, join rules, feature manifest, target definition, model/solver versions and output checksums are pinned together;
- run at least one negative control where an intentionally inserted future-leak feature must trigger a fail-closed rejection;
- report missingness, unmatched join rate, coverage by airport/route/day, out-of-support episodes and source drift alongside aggregate performance.

## 6. First reproducible benchmark (proposed)

### Dataset A — observed flight-operation replay

Start with a bounded airport subset and frozen month range supported by the actual schema. Extract only fields actually present. Benchmark event/timing quality and recovery/dispatch baselines separately. OPDI and BTS may be separate tracks because they have different scope and event semantics; do not force a misleading cross-dataset join.

### Dataset B — renewable availability replay

For the same geographic region, construct solar/meteorological time series from one pinned CAMS or PVGIS release. If the airport-level grid/load/price/charger measurements are unavailable, create a synthetic energy-hub scenario parameterized by the solar time series and explicitly label tariff, PV conversion, airport load, aircraft demand, storage and charger quantities as synthetic/modelled.

### Dataset C — integrated synthetic energy/recovery replay

Join observed or derived flight timing with an explicitly declared aircraft/energy scenario. Keep evidence fields on every input. Use identical workload, horizon, observation timing, service demand, compute budget and constraint set for:
- frozen static plan;
- renewable-first greedy baseline;
- exact exhaustive oracle on tiny cases;
- robust/recourse reference;
- nonanticipative policy;
- later learned/heuristic policies, always checked by an independent feasibility verifier.

Report flight forecast skill, decision value, energy value, lifecycle CO2e and any causal claim separately. An offline replay estimates simulated counterfactual performance only; it does not establish causal deployed savings.

## 7. Benchmark vector

At minimum report:
- data coverage, missingness, invalid/ambiguous joins and event-definition coverage;
- timing errors with unit/definition (for example MAE/p50/p95, only for comparable target definitions);
- delays/cancellations and transport-service completion;
- energy and cost by source, grid import/peak, renewable generation/use/curtailment, storage losses and terminal reserve;
- accounted lifecycle CO2e with boundary/factor evidence, and non-CO2 indicators separately;
- model and input uncertainty, distribution shift and unsupported-case weight;
- feasibility, definitely infeasible and unresolved weights separately;
- optimization runtime (P50/P95/P99), time-to-first-feasible, work counters, memory and valid optimality gap;
- baseline ablations so the incremental value of forecast skill, energy coupling, robust recourse and learned policy is visible;
- burden/fairness distribution when recovery changes delays across airports, carriers or passenger groups.

## 8. Acceptance / falsifiers

- 100% material inputs have source/provenance, version, units, evidence label, coverage and exact digest.
- Every join has a cardinality assertion and a missing/unmatched report.
- UTC/event chronology and as-of decision-time features are tested.
- Frozen train/calibration/test partitions and leakage negative controls exist.
- No silent zero-imputation for missing energy, cost, emissions, demand or event times.
- Every feasible plan is independently reverified against the exact input digest and all hard constraints.
- Dataset/access/terms unavailable -> keep that field/claim unavailable. Do not fabricate the missing source.
- Published benchmark details/data unavailable -> report the limitation; use a clearly labelled related synthetic comparison.
- A measured energy-saving claim is prohibited until airport energy/aircraft demand are actually observed or a model is independently calibrated and the result is correctly called simulated/modelled.

## 9. Current status

This document is a source and methodology contract. No OPDI/BTS/CAMS/PVGIS records have yet been downloaded, normalized, joined or scored in this workstream. No flight-recovery metrics, airport energy saving, lifecycle reduction or SOTA improvement has been measured by this document.
