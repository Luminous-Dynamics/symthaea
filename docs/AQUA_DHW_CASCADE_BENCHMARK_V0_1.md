# Luminous Aqua — Whole-System DHW and Drain-Heat Cascade Benchmark v0.1

**Status:** research / preregistration draft; no Aqua thermal configuration has been tested  
**Parent:** [AQUA-BATHING-SERVICE-001 #12660](https://github.com/Luminous-Dynamics/symthaea/issues/12660)  
**Workstream:** [AQUA-DHW-CASCADE-001 #12709](https://github.com/Luminous-Dynamics/symthaea/issues/12709)  
**Composes:** shower benchmark, wet-room envelope, Symtropy thermal / HP / resource-closure owners, Mycelix metrology and water assessment.

## 1. Research question

Which coupling of hot-water source, heat pump, storage, shower drain-water heat recovery (DWHR), fixture mixing and control provides the lowest total purchased energy for an equivalent, comfort-qualified bathing service, after standby, distribution, fouling, parasitic power and maintenance?

The comparison target is **whole-system DHW service**, not nominal exchanger efficiency.

## 2. Current research baseline

| Source | Result / evidence | Limitation |
|---|---|---|
| Bouvenot & Beaudet (2025), 14-month in-situ DWHR field study | 66% exchanger efficiency after fouling/unsteady effects; fouling reduced nominal efficiency by 8%; 52% complete shower-system efficiency after heat loss from showerhead to drain | Exact exchanger and installation; system-level result is the comparable boundary, not a universal device value |
| Bouvenot & Beaudet (2026), DWHR coupling with electric/HP storage | Global COP up to 6.3 experimentally and 8.0 in simulation; configuration-dependent, with static heat-loss and HP COP penalties in some topologies; best tested result was unbalanced DWHR connected to shower mixer rather than storage tank | Study-specific models / installation / service; not a guaranteed Aqua target |
| Stec et al. (2026), centralized multi-tube DWHR | Integrated 4E method and explicit fouling/pumping/uncertainty; 7.89 kW maximum recovered heat in a centralized system; lower in degraded cases | Centralized greywater plant is not directly rankable against one residential shower |

Sources:
- https://doi.org/10.1016/j.enbuild.2025.115341
- https://doi.org/10.1016/j.enbuild.2026.117761
- https://doi.org/10.1016/j.applthermaleng.2026.131985
- https://doi.org/10.1016/j.jobe.2025.112062

## 3. Configurations

Compare under identical shower task, inlet water, flow/pressure, room conditions, demand profile and service interval:

| Arm | Architecture |
|---|---|
| H0 | Resistance storage heater, no DWHR |
| H1 | Heat-pump storage heater, no DWHR |
| D1 | DWHR preheats cold feed to shower mixer (unbalanced) |
| D2 | Balanced DWHR preheats both hot-side / cold-side paths |
| D3 | DWHR transfers heat to storage tank through indirect circuit |
| D4 | HP storage + mixer-side DWHR + bounded bypass / isolation |
| D5 | HP / ventilation or dehumidifier heat integration through a separate closed loop |
| D6 | High-performance low-flow shower + HP, no DWHR |

No fluid connection is permitted between drain water, potable water, shower same-session recirculation, greywater reuse, pool water or immersion water. Heat transfer is indirect and metered.

## 4. Functional unit and boundary

Primary unit: one completed 10-minute shower at a declared delivered temperature, flow, pressure and task profile. Add 5- and 15-minute strata; bath-fill and simultaneous fixture demand are separate profile families.

Start at request and end at post-use ready state. Count:
- all source water and hot-water delivery;
- heater / HP purchased electricity or fuel;
- compressor, fan, pump, control, defrost and auxiliary energy;
- storage standby and cycling losses;
- pipe / distribution / first-draw losses;
- drain heat available, transferred and actually useful at the receiving boundary;
- exchanger pressure loss, bypass and fouling;
- cleaning, purge, reject and maintenance water / energy;
- room ventilation / dehumidification increment when material;
- repair labour, consumables, downtime and replacement.

Use fixed system windows for shared loads and allocate each resource once. Do not credit the same joule at the exchanger and again at the HP/storage level.

## 5. Primary outcome vector

Report independently:
- litres of hot/mixed water at the fixture;
- delivered temperature and stability;
- total purchased electricity and fuel;
- heat-pump COP and complete DHW system COP/SPF, with numerator/denominator definitions;
- exchanger effectiveness and complete shower-system efficiency as separate metrics;
- useful recovered heat measured at the receiving boundary;
- tank/storage and distribution losses;
- parasitic pump/fan/control energy;
- water, reject and cleaning burden;
- pressure loss, fouling and service interval;
- lifecycle cost per qualified hot-water service;
- uncertainty, missing values, all retries and failed runs.

If exergy analysis is used, show temperature-grade inputs/outputs and destroyed exergy as an additional metric; do not use exergy to replace measured kWh or fixture service.

## 6. Candidate targets — hypotheses only

- ≥10–15% lower purchased whole-system energy per completed shower service than the best matched incumbent topology as an initial stretch.
- Pursue ≥20% only after measured baseline variance and room-specific duty cycles are established.
- ≥60% whole-system recovery may be tested only at the useful receiving boundary after showerhead-to-drain loss, fouling, parasitic energy, storage and distribution losses.
- The 2026 study's COP 6.3 experiment / 8.0 simulation are comparators to understand, not guaranteed target values.
- No degradation in user task, temperature stability, low-pressure operation, access, hygiene or maintainability.
- 100% of energy credit is assigned once; zero double counting or unmeasured storage/standby losses.
- 100% indirect water separation with fault-isolated bypass.

## 7. Adversarial cases

1. cold / warm start and long standby;
2. continuous shower vs repeated short events;
3. seasonal inlet-temperature / ambient shifts;
4. low / nominal / high fixture flow and pressure;
5. heat-pump high-lift / low-source temperature;
6. storage stratification and sensor misplacement;
7. drain-side fouling / biofilm and cleanout;
8. partial blockage / high differential pressure;
9. bypass valve stuck or incorrect state;
10. indirect-exchanger integrity fault;
11. simultaneous bath / basin draw;
12. power failure and pump failure;
13. expired or missing calibration;
14. missing / stale drain or hot-water measurement;
15. cleaning and maintenance overdue.

Unknown energy or water paths receive no favorable credit. The physical design must have a professionally reviewed safe bypass/isolation path.

## 8. Owners and non-duplication

- Symtropy's native thermal, heat-pump and thermofluid owners own the physics; this benchmark creates no second solver.
- Symtropy #2152 owns physical water/energy/material/residual closure; #4978 water-service T0 arithmetic; #5425 water economics.
- Symtropy #4531 owns the unoccupied room-scale integration bench.
- Mycelix #6025/#6029/#6034/#6048/#6054/#6057 own channel identity, measurement, calibration, uncertainty/timebase and applicability.
- Mycelix #6100 owns water use/quality assessment; Aqua cannot self-authorize drain or recirculated water for another use.
- Sol-Atlas #765/#766/#770/#772 owns pool thermal / evaporation / microclimate.
- Aqua owns product topology comparison and completed bathing-service context.

## 9. Evidence gates

- G0 exact configuration and comparator manuals captured.
- G1 service and measurement boundaries frozen.
- G2 deterministic arithmetic / conservation and missing-evidence negative controls pass.
- G3 unoccupied clean-water bench and meter uncertainty qualified.
- G4 fouling / maintenance / pressure-loss sequence completed.
- G5 matched physical confirmation with exact device and room revisions.
- G6 longer-duration field operation before any field superiority or lifetime savings claim.

A synthetic result is not a physical performance result. A simulation is bounded by its model. A short unoccupied bench does not establish lifetime fouling or commercial payback.

## 10. Claim ceiling

This protocol does not establish a COP, heat-recovery rate, efficiency, energy savings, economic payback, hygiene outcome, product superiority, building approval or procurement/construction authority. All values are hypotheses until tested on exact configurations with independent measurement.


## 11. South African hot-water hygiene and delivery-temperature boundaries

The IOPSA notice for SANS 10252-1:2018 Edition 3.2 quotes clause 7.1.2 as requiring conditions that are as far as possible unfavorable to Legionella development in the cited stored-water installations; it further quotes a stored-hot-water provision of at least 55 °C with 60 °C reached for at least one hour per day. The same notice quotes clause 7.5.1.4.2 on terminal fitting discharge temperature (maximum 60 °C unless otherwise required, with thermostatic mixers potentially required to achieve a lower delivery temperature).

Source: https://www.iopsa.org/news/7216337

This is an industry-association summary, not the controlled SABS standard. Confirm the current controlled text, exact scope, amendments and project applicability with SABS and the responsible professional.

For Aqua engineering, keep these as separate boundaries:
- storage / source-side temperature-time regimen;
- distribution and return temperature / residence-time profile;
- point-of-use mixing and terminal delivery temperature;
- actual water-quality and biofilm assessment;
- purchased energy and storage / distribution losses.

Do not raise the user-delivery temperature to satisfy a storage-side hygiene regimen. Do not treat a tank reaching a specified temperature as proof that the entire installation, shower branch, hose, head or aerosol exposure is microbiologically controlled. Do not use an assumed pasteurization cycle as a substitute for Mycelix / competent water-quality assessment and wet-path hygiene evidence. The DHW optimizer must account for any required source-side thermal regimen and its energy burden without overriding independent anti-scald protection.

## 12. Power-coupled electric shower pathway

The UK ETL changed its non-domestic electric-shower criteria on 15 September 2026. The current criteria keep 4.5 L/min as the standard electric-shower maximum but allow up to 9 L/min where electric power is modulated to minimize energy while maintaining stable output temperature with ETL-compliant wastewater heat recovery; up to 10.6 L/min is possible with an additional qualifying user-behaviour feature. This is a UK non-domestic programme condition, not a South African requirement or domestic certification.

The Aqua comparison must measure purchased electrical energy and useful recovered heat at the actual receiver, and include exchanger fouling, showerhead-to-drain losses, storage/standby losses, pump/control parasitics, warm-up, pause/boost, purge and completed-task quality. Never infer system savings by multiplying nominal exchanger efficiency and nominal heater COP.

The October 2026 DWHR coupling study reports that mixer-side coupling performed best in its tested whole-system configurations, while some storage-side configurations increased losses and reduced heat-pump COP. This is a topology hypothesis for Aqua, not a universal rule. The 2025 14-month DWHR fouling study reports 66% overall exchanger efficiency and 52% system efficiency after showerhead-to-drain losses in its specific installation.

Detailed preregistration: [AQUA_POWER_COUPLED_SHOWER_DWHR_BENCHMARK_V0_1.md](AQUA_POWER_COUPLED_SHOWER_DWHR_BENCHMARK_V0_1.md), tracked by [#12816](https://github.com/Luminous-Dynamics/symthaea/issues/12816).
