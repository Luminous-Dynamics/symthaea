# Luminous Aqua — Power-Coupled Shower and Wastewater Heat-Recovery Benchmark v0.1

**Status:** architecture and experiment preregistration; no prototype, physical test or energy result exists  
**Workstream:** [AQUA-POWER-RECOVERY-001 #12816](https://github.com/Luminous-Dynamics/symthaea/issues/12816)  
**DHW system owner:** #12709  
**Task-phase owner:** #12798  
**Transient supply owner:** #12801  
**Product anchor:** AQS-1 / Aqua

## 1. Decision thesis

Do not optimize the showerhead, electric heater, drain heat exchanger, storage tank or control policy as isolated products. Optimize one completed bathing task as a coupled water–power–temperature–heat-recovery service.

The important distinction is between:
- heat recovered at the exchanger;
- heat delivered to a useful receiver at the time it is needed;
- purchased electricity displaced at the whole-system boundary;
- fresh water and thermal service required to complete the task.

Only the last two establish a product-level efficiency result. A high exchanger efficiency can coexist with high static losses, poor timing, fouling, excess pumping, unstable outlet temperature or little actual reduction in purchased energy.

## 2. Newly relevant external benchmark — UK ETL 2025/26

The UK Department for Energy Security and Net Zero published its 2025/26 ETL criteria update on 15 September 2026. The current ETL shower criteria explicitly describe non-domestic electric showers and a conditional higher-flow path:

- Standard electric-shower maximum available flow: 4.5 L/min.
- Up to 9 L/min where the shower modulates electrical input to minimize electricity while maintaining stable output temperature when connected to an ETL-compliant wastewater heat-recovery system.
- Up to 10.6 L/min where the system also incorporates user-behaviour features (UBFs).
- UBFs include features such as push valves, duration/resource displays, timed shutoff, and water or temperature breaks.

Sources:
- Current ETL shower criteria, sections 1.1–1.4: https://etl.energysecurity.gov.uk/products/showers/showers
- 2025/26 Annual Research Programme update, 15 September 2026: https://etl.energysecurity.gov.uk/news-events/news/202526-annual-research-programme-updated-criteria-now-live

**Scope:** the ETL shower category is for non-domestic applications such as commercial and hospitality settings. These flow values are not South African law, not a domestic shower certification, and not evidence that any particular Aqua configuration achieves them. The criteria use the Unified Water Label procedure and its exact system assumptions. Aqua should use this as a system-architecture research reference, not a shortcut to a conformity claim.

## 3. Literature anchors and their limitations

### 3.1 DWHR fouling and useful system efficiency

Bouvenot and Beaudet (Energy and Buildings, 2025, 331, 115341; DOI 10.1016/j.enbuild.2025.115341) report a 14-month in-situ study of a commercial shower drain-water recovery plate heat exchanger. Their highlights report:
- overall heat-exchanger efficiency of 66% after fouling and unsteady-use effects;
- fouling reduced nominal efficiency by 8%;
- overall system efficiency of 52% when heat losses between the showerhead and drain were included.

These are study-specific measured/modelled values and cannot be copied as Aqua performance or treated as universal exchanger limits. They establish why Aqua must measure fouling and shower-to-drain losses across the whole service.

Source: https://doi.org/10.1016/j.enbuild.2025.115341

### 3.2 System-level coupling topology

Bouvenot and Beaudet (Energy and Buildings, 2026, 368, 117761; DOI 10.1016/j.enbuild.2026.117761) study DWHR coupling with electric and heat-pump storage water heaters. Their abstract reports:
- global DHW COP up to 6.3 experimentally and 8.0 in simulation in their studied configurations;
- DWHR can increase static heat losses and reduce heat-pump COP by around 15–20% in some configurations;
- the best tested overall system configuration connected an unbalanced DWHR to the shower mixer rather than simply charging the storage tank.

These results are configuration-, climate- and boundary-specific. The model is reported as validated against a year of measurements with cv(RMSE) below 7%; simulated COP is not measured field performance. Aqua must reproduce its own matched system boundary and must not assume that mixer-side topology always wins.

Source: https://doi.org/10.1016/j.enbuild.2026.117761

## 4. Architectures to compare

All variants use one frozen completed-task definition and matched inlet / room conditions. Each architecture has its own exact configuration and control digest.

| Arm | Architecture | Decision question |
|---|---|---|
| A0 | Direct electric shower, fixed-power / conventional control | Baseline purchased energy and water |
| A1 | Direct electric shower, modulated power, no DWHR | Value of modulation alone |
| A2 | Modulated electric shower + compliant drain-to-inlet / receiver DWHR | Incremental useful heat and energy displaced |
| A3 | A2 + explicit UBF (pause, time/resource display or timed shutoff) | Incremental water / energy change from user behaviour |
| A4 | Heat-pump / storage DHW + mixer-side DWHR | Compare against the 2026 system-topology finding |
| A5 | Heat-pump / storage DHW + storage-side DWHR | Expose static loss / COP penalty |
| A6 | Heat-pump / storage DHW without DWHR | Heat-pump baseline |
| A7 | Recirculating shower / water reuse | Separate research only; requires its own wet-path and exposure qualification |

Do not compare A2 against A0 if the task, output-temperature profile, inlet temperature, user behavior, pressure, room or accounting boundary differ. Do not credit heat recovery when the recovered heat is rejected, dissipated, stored with excessive loss or unavailable when the task needs it.

## 5. Functional unit and service boundary

The primary functional unit is **one completed, independently assessed bathing task**, not one minute of shower operation or one litre of nominal flow.

The protocol must define, before test runs:
- task identity and task phase sequence;
- required outlet / bather-plane surrogate thermal profile;
- fixed inlet cold-water temperature profile;
- starting room and surface thermal conditions;
- exact shower mode and fixture geometry;
- allowed pauses, boosts and water-break behavior;
- independent task-quality endpoint and non-inferiority margin;
- test duration rule and task completion criterion;
- cleaning, purge, startup and return-to-ready boundary.

Report both:
- litres of fresh water per completed task;
- purchased kWh per completed task.

Do not allow a shorter or incomplete task to appear more efficient. An aborted / failed task remains in the denominator as an unserved attempt and its water / energy burden remains in the ledger.

## 6. Required synchronized measurements

### Electrical and source-side
- true electrical power and cumulative energy for heater, pump, control and other active auxiliaries;
- operating voltage/current/power factor where applicable;
- heat-pump compressor and source-side energy where applicable;
- standby energy between tasks.

### Water and thermal
- fresh-water mass/volume and flow over time;
- hot and cold inlet temperatures;
- heater inlet/outlet and mixer inlet/outlet temperatures;
- spray / bather-plane surrogate temperature at fixed locations;
- drain inlet temperature and flow;
- DWHR hot-side and cold-side inlet/outlet temperatures and flow;
- actual preheat delivered at the receiver;
- storage-tank temperatures at defined locations and time intervals;
- ambient / room conditions.

### Dynamics and maintenance
- start-up, stabilization, pause, boost, shutoff and recovery event timestamps;
- pressure at relevant dynamic datums;
- exchanger fouling state and pressure drop;
- cleaning intervention, time, water, electricity, chemicals and labour;
- failed starts, temperature excursions, invalid or missing channels.

Mycelix owns instrument identity, calibration, uncertainty, timebase and evidence qualification. Symtropy owns canonical hydraulic/thermal models, energy balance and residual closure. Aqua consumes those records and reports complete service outcomes; it does not create another heat-balance authority.

## 7. Heat and energy accounting

Report separately:

1. Purchased electricity at the whole-system meter boundary.
2. Useful heat delivered directly by the heater / heat pump.
3. Heat transferred by DWHR at the actual receiver.
4. Recovered heat rejected or unused.
5. Thermal losses in pipework, storage and waiting periods.
6. Electrical parasitic loads (pumps, controls and any active components).
7. Fresh-water heating energy demand and the measured output-temperature service.
8. Drain-side residual energy and heat not recoverable under the installed geometry.

A first-principles water-side thermal quantity may be estimated from measured mass flow and temperature difference, using water properties from the frozen method. It is not a substitute for electrical metering or a fully closed system balance. Publish uncertainty and residuals; do not silently distribute residual error into the recovered-heat claim.

For heat-pump systems, distinguish instantaneous COP, session COP and whole-cycle COP including standby / recovery. Never mix simulated COP with measured COP.

## 8. Experimental sequence

1. Freeze the complete architecture bill of materials, rated components, installation drawing, firmware/control revision and protocol digests.
2. Calibrate all electrical, flow and temperature channels and verify the timebase.
3. Run baseline stabilization and repeatability checks.
4. Run matched task sequences under A0–A6; randomize/block by day, water batch, specimen and operator where practical.
5. Run a full session including startup, task phases, pauses/boosts, shutdown and purge.
6. Repeat at clean, aged/fouled and post-maintenance exchanger states using a preregistered fouling / cleaning profile.
7. Include low and high inlet-water temperature seasons or controlled boundary profiles, without conflating them.
8. Include transient supply scenarios from #12801 as a separate event block.
9. Hold back independent confirmation runs and freeze the analysis before unblinding.
10. Reconcile mass and energy balances; unresolved residuals block superiority claims.

No human exposure or occupied installation is authorized. Task-quality proxies and surrogate thermal setups are analytical/test-design tools only until separate independent human-factors approval exists.

## 9. Proposed research targets — not achieved results

These ambitious targets are hypotheses to size and confirm using pilot variance and measurement uncertainty:

- At least 15% lower purchased electricity per completed matched task than the strongest eligible direct-electric baseline, without worse independent task-quality or output-temperature endpoints.
- At least 20% lower fresh water per completed task for a qualified UBF configuration versus matched always-on operation, including warm-up, pause, rinse, purge and failed attempts.
- At least 90% useful recovered heat is a stretch hypothesis only when the recovered energy is measured at the actual receiver and remains useful after fouling, standby and parasitic losses. It is not a 90% exchanger-efficiency target.
- No claimed energy savings from recovered heat that is rejected, stored with excessive loss or unavailable at the moment of demand.
- Lower lifecycle cost per qualified task after maintenance, service labour, replacement parts and downtime are included.

Targets are not standard limits and must not override safety, hygiene, electrical or anti-scald constraints.

## 10. Selection and claim rules

- Use a Pareto set for water/task, purchased electricity/task, thermal stability, maintenance and lifecycle cost; do not create one opaque weighted score.
- Hold task quality and thermal delivery as hard constraints, not variables to trade away for lower flow.
- A UBF must measurably change completed-task water/energy or behavior; a display alone is not evidence of savings.
- A controller may recommend or display; it cannot override local heater power limits, mechanical thermostatic protections, electrical isolation, backflow protection, water-quality gates or manual control.
- Any non-potable, reused or recirculated water architecture needs its own source/use-class, wet-path, biofilm and exposure qualification; it is not implicitly authorized by this protocol.
- A result is only valid for its tested architecture, climate, electrical supply, inlet conditions, user/task profile and fouling state.
- Do not claim ETL compliance or product certification from an Aqua matched-task experiment.

## 11. Claim ceiling

This document authorizes architecture comparison, evidence review, analytical modelling, synthetic traces and unoccupied test design only. It does not authorize a pressurized or energized prototype, procurement, plumbing/electrical modification, human exposure, occupied pilot or construction release. Such work requires separate professional design and safety admission.
