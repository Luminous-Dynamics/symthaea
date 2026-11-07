# Marine resource-service operations benchmark — V0.1

**Status:** synthetic benchmark and interface proposal; not an executable controller or physical vessel qualification.  
**Parent:** [MARINE-RESOURCE-OPS-001 #11008](https://github.com/Luminous-Dynamics/symthaea/issues/11008)

## 1. Objective

Schedule onboard services against the vessel's time-varying electrical and resource state without treating all loads as equally flexible.

The scheduler receives a frozen mission/operating case, time-aligned forecasts and measured/modelled service state. It proposes service start times and energy dispatch requests. Existing local equipment controllers retain authority.

The benchmark asks:

> Under identical environment forecasts, service requirements, vessel state and hardware constraints, can a marine-specific service scheduler preserve every required onboard service while reducing avoidable peaks, generator operation and battery wear against simple baselines?

No answer is assumed in V0.

## 2. Existing technology to compose

- `symthaea-grid-physics` already has battery energy/SoH, solar/load scenario replay, a naive greedy policy and a reserve-aware policy. Reuse it; do not clone the battery or generic grid scheduler.
- Symtropy Habitat Metabolism owns authoritative water, food, waste, thermal, maintenance and material stocks/flows.
- MARINE-DESIGN-001 #7703 owns itinerary and ship/route co-optimization.
- MARINE-ENERGY-BUDGET-001 #8092 / MARINE-TRACE-001 #8102 own mission energy accounting and measured voyage traces.
- MARINE-WATERMAKER-BENCH-001 #10994 owns qualified RO output and electrical input evidence.
- MARINE-AUTONOMY-VECTOR-001 #10956 owns per-resource horizon assessment.
- MARINE-HABITABILITY-SEAKEEPING-001 #10966 owns task/location comfort evidence.
- MARINE-LIVEABOARD-EVIDENCE-JOIN-001 #10988 owns exact-case cross-domain evidence alignment.
- SMDC-ELECTRICAL-001 #3846 owns three-phase model, protection, coordination, transfer sequences and electrical assurance.

The shared grid-physics scenario harness currently receives an aggregate load profile, generation profile and battery policy. This V0 marine contract adds the service-origin semantics and constraints needed to construct those profiles; it does not claim that the existing harness already models watermakers, generator dynamics, refrigeration thermal state or full vessel operation.

## 3. Paradigm shift: service scheduling rather than load shifting

A kilowatt is not a complete service description.

A watermaker needs feed within its qualified envelope, prefilter/membrane state, a minimum production run, flush and preservation stages, product-water quality diversion and available tank headroom. A refrigerator has stored thermal state and temperature limits. Hot water has a thermal store. HVAC has comfort and humidity bounds. A workshop task has a deadline and safe operating prerequisites. Propulsion, steering, navigation and emergency systems cannot be treated as discretionary loads.

The optimizer should schedule *service requests*, not manipulate arbitrary kW bars.

Each service request therefore declares:
- required output and acceptance criterion;
- earliest/latest start and deadline;
- process stages and minimum on/off times;
- time-varying power envelope and energy interval;
- state prerequisites (temperature, tank level, feed conditions, maintenance state);
- hard operating limits and allowed flexibility;
- consequences of deferment, interruption or failure;
- source, evidence class, revision and exact mission-case binding.

## 4. Multi-carrier service accounting (electricity is not the whole habitat)

Recent ship-EMS work extends beyond electrical dispatch:
- A September 2026 rolling-horizon multi-energy ship study co-optimizes electricity, thermal energy, hydrogen, freshwater, waste-heat recovery, route selection and flexible voyage scheduling: https://doi.org/10.1080/20464177.2026.2728778
- A 2026 Applied Energy review frames hybrid-ship EMS around source/storage dynamics, lifecycle degradation, transient buffers, DC-grid stability and deployment evidence: https://doi.org/10.1016/j.apenergy.2026.128503
- A September 2026 hybrid ship prediction study uses a multi-model ensemble for propulsion/load prediction; it reports lower MAE/RMSE than its best single-network comparator on its own study dataset. It is not evidence that the same method will outperform for small-craft hotel loads: https://doi.org/10.1016/j.oceaneng.2026.127503

- Bao et al., *Research on prediction and coordinated optimization for the energy management system of hybrid power ships*, Ocean Engineering 365 (1 Sep 2026), DOI https://doi.org/10.1016/j.oceaneng.2026.127503. The paper reports an EMTOA-DW multi-network forecasting approach (CNN/BiLSTM/ELM/Elman) with lower MAE/RMSE than its best single-network comparator in its study. Treat the percentages as paper-specific; LOS requires its own out-of-vessel/time/condition validation.
- Kotsampopoulos, *Testing Ship Electric Propulsion and Shipboard Microgrids: Standards, Techniques and New Trends*, Energies 19(9), published 22 Apr 2026, DOI https://doi.org/10.3390/en19092016. Supports staged holistic CHIL/PHIL validation rather than relying only on offline simulation.
- A 2026 field-validated shipboard microgrid paper reports 360 half-hour records across 12 event classes at 4 s sampling. Useful for supervisory diagnostics and fault-aware energy reconfiguration; its primary telemetry is operator-confidential and its sample rate is not fast-transient evidence: https://doi.org/10.3390/en19174074.

### Forecast-quality benchmark (separate from dispatch-policy scoring)

The forecast feeding a scheduler is itself a model and must be benchmarked independently:
- **Persistence / last-value / seasonal baseline:** simple, transparent floor.
- **Physics / operating-point baseline:** speed, draft, propeller/RPM, resistance/added resistance and declared weather where the required model is validated.
- **Single learned predictor:** one predeclared architecture, no test-case tuning.
- **Multi-model / ensemble predictor:** include a literature comparator such as Bao et al.'s EMTOA-DW multi-network study, but do not transfer its reported results to LOS.
- **Symthaea forecast/advisory:** only after its model, inputs, calibration and compute budget are frozen.

For every model report MAE, RMSE, signed bias, p95 / worst-stratum absolute error, forecast interval coverage and width, calibration error where valid, OOD/abstention rate, lead time, runtime and missing/stale input behavior. Report propulsion shaft-power prediction separately from auxiliary electrical / hotel-load prediction; never use one as a proxy for the other. Hold out complete voyages/vessels/time periods or operational regimes, not random adjacent minute/hour rows. Scheduler comparisons must give every policy the same forecast information and forecast horizon.

**Candidate restricted propulsion dataset:** Shifts 2.0 Vessel Power Estimation / Shifts Marine Cargo Vessel Power Consumption Prediction Dataset, Zenodo v2.0 (DOI 10.5281/zenodo.7684813). The Shifts project describes four years of merchant-vessel minute-level data with 10 features, a main-engine shaft-power target, real and synthetic partitions, and canonical in-domain/shifted splits along wind/time (to capture weather regime, fouling and sensor drift). Shifts lists CC BY-NC-SA 4.0, while the current Zenodo Rights field is not populated; rights intake remains pending through MARINE-REF #8146. It is a non-commercial research candidate only until exact rights are confirmed; do not use it in commercial Luminous training, distribution or product claims without permission. It is not an onboard electrical/hotel-load dataset and not a small-craft validation set. Sources: https://shifts.ai/dataset ; https://zenodo.org/records/7684813

The service schema now records both `input_requirements[]` and `required_output.carrier`. Examples:
- A watermaker consumes electrical energy and seawater feed; its delivered output is potable water only after the separate quality acceptance owner has accepted it.
- Domestic hot water consumes electricity and may consume recoverable heat from a generator/range extender or a thermal store; the useful heat must satisfy the grade/temperature/exchanger constraints.
- Fuel, electricity, recoverable heat, potable water and other material inputs remain distinct carriers and units. Do not sum kWh_e and kWh_th as if they were the same energy quality or double-count recovered heat already included in the generator/fuel energy balance.
- Waste heat receives no credit until source temperature, duty, exchanger effectiveness, contamination/isolation, availability, parasitic pump energy and thermal storage losses are declared and measured/validated. It is not assumed to be available at generator idle or during an offline state.

For LOS-18, heat recovery should first target low-risk domestic hot-water / space-heating service if a selected generator and exchanger support it. Do not assume RO feed preheating improves membrane performance or potable-water quality; that requires OEM membrane operating-envelope and water-quality review.

## 5. Decision hierarchy

Use constrained/lexicographic decisions, not a free weighted sum:

1. Hard safety, electrical, thermal, battery, machinery, fire/flood and route constraints.
2. Non-deferrable critical services and degraded-mode minimums.
3. Qualified potable-water, refrigeration, ventilation and emergency reserves.
4. Occupant service deadlines, comfort and privacy/noise limits.
5. Reduce unserved services, emergency generator starts, peak demand, fuel, battery throughput/degradation, energy cost and maintenance burden.
6. Preserve a Pareto view of trade-offs; no average score may mask a failed hard constraint.

V0 is shadow/recommendation mode only. It cannot command the watermaker, generator, battery, HVAC, refrigerator, propulsion or protection devices. Physical controllers, certified local interlocks and operator authority remain independent.

## 6. Weather-routing research context and authority boundary

Recent weather-routing literature reinforces why route, speed and energy must be co-designed under uncertainty, but that work belongs to the existing MARINE-DESIGN-001 owner:

- Chen et al., *State-of-the-art optimization algorithms in weather routing — ship decision support systems: challenge, taxonomy, and review*, Ocean Engineering (1 July 2025), reviews optimization methods and common terminology issues: https://doi.org/10.1016/j.oceaneng.2025.121198
- Timas and Mohammadi, *Integrating weather-informed routing and energy optimization for sustainable maritime transportation*, Ocean Engineering (30 July 2025), present a two-stage route/energy optimization with intermediate waypoints, weather, power, speed, safety and refuelling constraints: https://doi.org/10.1016/j.oceaneng.2025.121463
- Golsefidi et al., *Containerships routing problem: Incorporating uncertain weather impact on fuel consumption by considering speed optimization*, Ocean Engineering (30 November 2025), report in their container-ship simulation that ignoring weather could underestimate fuel by up to 11%, while variable-speed policies saved up to 21% fuel in the studied scenario at the cost of delays up to 20 hours: https://doi.org/10.1016/j.oceaneng.2025.122410

These are research results from different vessel classes and scenarios, not transferable performance claims for LOS-18. They justify matched weather/forecast uncertainty and explicit fuel-versus-delay trade-offs. The marine resource scheduler consumes a frozen route/propulsion demand profile; it does not generate the route or change speed in V0. A future joint study can compose the route owner and service scheduler using exact shared case identities, then compare the full Pareto frontier.

## 7. Benchmark policy baselines

Compare on identical frozen cases and the same information:
1. Fixed equipment defaults / static timetable.
2. Greedy solar-surplus execution.
3. Reserve-aware priority rules.
4. Constrained receding-horizon / MPC where the models are sufficiently specified.
5. Symthaea advisory proposal.
6. Perfect-foresight schedule as a clearly labelled non-realizable oracle upper bound.

Do not compare a forecast-informed policy against a baseline deprived of the same forecast. Freeze compute budgets and replanning intervals.

## 8. Metrics

Report separately:
- critical service violations (non-compensable hard fail);
- service completion, delivery quality, postponement and failure by service class;
- energy by propulsion, auxiliaries, hotel, thermal and flexible service;
- battery SOC minimum/reserve events, charge/discharge throughput and equivalent full cycles;
- generator starts, runtime, fuel, ramp/step-load violations and maintenance implications;
- renewable energy used and curtailed based on actual time-series power, not panel nameplate;
- potable water accepted by quality criteria, flush/preservation energy and autonomy-vector impact;
- refrigerator/freezer temperature excursions and duration (not a food-safety conclusion);
- comfort-band / humidity violations by occupied location and time;
- peak and p95 electrical demand;
- deferred service age, deadline slack, operator overrides and maintenance-window collisions;
- forecast sensitivity, uncertainty and worst-case outcomes by scenario;
- policy runtime and data requirements.

No one score combines these. Publish the full service and energy ledger plus hard-gate disposition.

## 9. Frozen scenario corpus

The companion JSON defines four synthetic profiles with critical, hotel, thermal-protection and propulsion load series. Their sum is checked against the aggregate load profile passed to the existing grid-physics harness. Each scenario also declares synthetic generator availability/ramp/start/minimum-run/fuel state, shore-power availability, battery temperature/SoH, sea state/wind/air/sea temperature, solar forecast error bound and RO feed salinity/temperature qualification. These are synthetic contract fields, not vendor- or class-qualified equipment parameters. It also carries separate recoverable-heat and thermal-store availability, plus per-service input carrier requirements (for example, electricity + seawater feed -> qualified potable water; electricity + recoverable heat -> hot-water service). This enables multi-carrier dispatch tests without treating thermal energy as electrical energy or crediting unqualified heat.

The four cases are:
- clear solar surplus with scheduled watermaking and hot-water service;
- cloud / low-solar event concurrent with fixed underway propulsion load;
- low battery reserve with increased water demand and a watermaker service deadline;
- watermaker feed outside its qualified envelope, where scheduling production must be refused or deferred even if electrical energy is available.

These profiles are deliberately synthetic. Their numbers are contract-test inputs, not recommended vessel sizing or measured operating assumptions.

## 10. Negative controls

- Missing or stale battery state must not be treated as reserve headroom.
- Watermaker feed outside qualified limits must block the production request.
- Water production that fails its quality criterion does not count toward potable stock.
- Flush/preservation energy may not be omitted from watermaker service energy.
- Generator noise restrictions at anchorage may constrain operation, but cannot justify violating critical safety/electrical limits.
- A refrigeration temperature excursion must not be hidden by a lower energy cost.
- High solar nameplate capacity does not imply contemporaneous generation.
- Recoverable generator heat is not available by assumption: if the source is offline, below usable temperature, exchanger state is unknown, or the measurement/energy boundary would double-count it, no thermal-service credit is permitted.
- Thermal kWh and electrical kWh remain separate carriers; never sum or substitute them without a reviewed conversion/process model.
- Propulsion load supplied by a route case is fixed in V0; the service scheduler may not change speed or route to improve its score.
- A schedule with a lower energy total but missed service deadlines does not win.
- Reordered service rows must produce equivalent deterministic output.
- Loss of network, forecast, Symthaea or Mycelix cannot disable local equipment safety controls.

## 11. Acceptance targets — goals, not results

- Zero hard-constraint violations in the frozen adversarial corpus.
- 100% service accounting: completed, postponed, failed, blocked, or indeterminate with reason.
- 100% deterministic replay for the same frozen scenario, policy and toolchain where promised.
- Zero off-spec water credited as potable service.
- Zero false-safe scheduling recommendations when safety-critical telemetry or prerequisites are unknown.
- Demonstrate improvement over at least one simple baseline on a preregistered metric without worsening hard constraints or service delivery.
- Report paired results, tails and worst scenario; no universal kWh saving is predeclared.
- Use held-out scenario families for policy confirmation; do not split time samples from one synthetic trace into train and test.

## 12. Claim ceiling

A PASS establishes only that the frozen scheduling contract behaves as specified on the synthetic replay corpus. It does not establish physical energy savings, generator fuel reduction, battery-life improvement, potable-water availability, food safety, habitability, seaworthiness or authority to operate equipment.
