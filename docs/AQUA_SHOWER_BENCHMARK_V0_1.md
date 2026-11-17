# Luminous Aqua — Shower Benchmark Protocol v0.1

**Status:** preregistration draft; no physical test has been performed under this protocol  
**Parent:** [AQUA-BATHING-SERVICE-001 #12660](https://github.com/Luminous-Dynamics/symthaea/issues/12660)  
**Comparator ledger:** [AQUA_COMPETITOR_LEDGER_V0_1.md](AQUA_COMPETITOR_LEDGER_V0_1.md)  
**Service contract:** [AQUA_SHOWER_SERVICE_CONTRACT_V0_1.md](AQUA_SHOWER_SERVICE_CONTRACT_V0_1.md)

## 1. Research question

Can a Luminous Aqua shower deliver the same or better independently assessed bathing task, user comfort, temperature stability, spray coverage, accessibility and reliability as a matched incumbent while reducing complete-service fresh water, useful energy and lifecycle burden—without worsening hygiene or serviceability?

Do not optimize a single metric in isolation. A design that reduces water by degrading task completion, increasing cleaning labor, shifting energy to another system, or creating an unresolved microbiological exposure is not superior.

## 2. Functional unit and primary boundary

Primary functional unit: **one completed 10-minute shower service** for a declared profile under frozen inlet water quality, supply pressure, inlet temperature, ambient temperature/RH, installation, spray geometry and task conditions.

Mandatory duration strata: 5, 10 and 15 minutes. Add task-specific hair-wash/rinse and seated/assisted profiles after their task definitions are frozen.

The accounting interval begins at the user request and ends only after the fixture reaches the admitted post-use ready state. Count:
- warm-up and cold-start flush;
- priming, reservoir fill and startup purge;
- all fresh make-up during active use;
- recirculation volume, age and treatment circulation separately;
- post-use Rapid Clean / flushing / purge / drain-down;
- allocated periodic deep-clean and maintenance consumables;
- water and energy from failed, aborted and retry attempts;
- allocated energy from room ventilation/dehumidification when that load is materially changed by the fixture configuration.

Do not allocate shared resources twice. Do not exclude hidden operations, standby or maintenance solely because they are not visible to the user.

## 3. Comparator arms

Freeze exact SKU, market variant, installation and configuration before testing.

| Arm | Role | Evidence to acquire before test |
|---|---|---|
| C0 — premium once-through shower | Quality control | Exact head/valve, flow curve, pressure compensation, installation instructions |
| C1 — efficient once-through showerhead | Efficiency baseline | WaterSense-labelled model where available or exact local-market equivalent; confirm spray and pressure method |\n| C1-SA — local pressure-compensating showerhead | South African comparator | Reproduce local 150 / 300 kPa test points; CLASP's 2022 sample and 2025 WRC synthesis are context, not substitute for a named SKU test |\n| C1-CR — independent lab-tested direct-flow head | High-performing direct-flow comparator | Freeze an exact 2026-tested model with public measured flow around 1.5–1.6 gpm; use comparable coverage, force and temperature-drop methods |\n| C1-HS — Hansgrohe EcoSmart+ | Premium local-market comparator | Confirm exact model and actual flow curve; the family claim is ≤6 L/min at 3 bar, not independent test evidence |
| C2 — KOHLER Anthem EvoCycle | Commercial recirculation frontier | Current installation/operating/cleaning manuals, exact Cycle/Standard mode, base/fixture requirements, service and filter intervals |
| C3 — Orbital Shower | Commercial recirculation comparator | Exact market model, duration basis for ~20 L statement, cleaning/maintenance, fresh-water and energy boundaries |
| T0 — Luminous direct-flow test article | Our low-flow architecture | Frozen nozzle/spray geometry and valve/heat-recovery configuration |
| T1 — Luminous within-session recirculation article | Research architecture only | No human exposure until independent hygiene and safety admission; test on unoccupied rig |
| T2 — hybrid | Direct flow + indirect drain heat recovery, with recirculation only if independently qualified | Component and boundary identity; parasitic energy, pressure drop, cleanability |

Vendor figures are not substituted for test results. If a comparator's boundary is unknown, record it as UNKNOWN; do not guess.

## 4. Frozen conditions and blocking

Before collecting confirmation data, preregister:
- inlet water temperature and its stability/uncertainty;
- static and dynamic supply pressure at defined reference locations; include at least 150 and 300 kPa to align with the South African CLASP proposal, then add any exact comparator-specific standard points such as 3 bar;
- ambient air temperature and relative humidity;
- room airflow/ventilation state;
- exact spray mode, nozzle geometry, head height/orientation and measured spray pattern;
- user/task profile, duration and permitted movement;
- water hardness / chemistry and source identity;
- device age, filter/media age, fouling condition and maintenance history;
- firmware, controller configuration and cleaning protocol;
- instrument identities, installation positions, calibration and uncertainty;
- operator, day, water batch, test cell, device serial and randomized test order.

Use blocked/randomized comparator order where practical. Separate development/tuning runs from locked confirmation runs. Group repeated trials by device, day, water batch and operator; do not treat multiple readings or images from one service run as independent trials.

For EPA WaterSense comparability, obtain and follow the applicable test method and official certification programme documents. The public programme criteria include pressure compensation, spray force and spray coverage and use multiple pressure points; do not invent a substitute method or claim WaterSense certification.

## 5. Measurements

### Water
- fresh water from source boundary;
- flow and volume delivered to user;
- recirculated volume and circulation rate;
- water age/residence-time trace;
- all discharge, purge, clean, reject and backwash streams;
- storage change and tank residual;
- leakage and spill;
- cleaning and periodic maintenance water.

### Thermal and energy
- inlet, heater outlet, user outlet and drain temperatures;
- useful thermal energy delivered to user;
- thermal energy available in drain and actually transferred across heat exchanger;
- heat loss between showerhead and exchanger;
- heater electricity/fuel;
- pump, controls, treatment, UV/ozone and cleaning energy;
- ventilation/dehumidification increment where applicable;
- pressure drop and additional hydraulic pumping energy.

### Delivered bathing task
- spray coverage using the chosen repeatable method;
- spray force / impact distribution using the chosen repeatable method;
- outlet temperature stability and overshoot;
- flow variation under the declared pressure envelope;
- noise and vibration;
- user comfort and preference using a preregistered instrument, only after safety review;
- task completion / rewash / repeat request;
- accessibility task success and assistance burden.

### Cleanability, reliability and lifecycle
- time and labour per standard clean;
- time to disassemble / inspect / clean / reassemble / requalify;
- filter/media loading, pressure loss and replacement interval;
- visible scale, biofilm/fouling observations under competent methods;
- component wear and faults;
- service downtime and spare-part availability;
- chemical use and waste handling;
- repair cost and end-of-life route.

## 6. Calculation rules

### Whole-service fresh water

W_all_in = W_warmup + W_prime + W_active_fresh + W_clean + W_purge + W_periodic + W_retry

If periodic cleaning is required, allocate its measured water over the number of services completed before that maintenance action, and also report the unallocated absolute maintenance total.

Saving_percent = 100 × (1 - W_candidate / W_matched_control)

Use only equal service tasks, comparable duration and identical accounting boundaries. Never compare a 10-minute candidate task with an unspecified-duration vendor claim and call the result superiority.

### Energy

E_all_in = E_heater + E_pump + E_treatment + E_control + E_clean + E_room_allocated

Report both purchased energy and useful thermal energy delivered. Keep fuel, electricity and thermal energy quantities distinct; any primary-energy or emissions conversion must name its declared factor and vintage.

### Heat recovery

Q_useful_recovered = m × cp × (T_preheat_out - T_preheat_in)

where flow, temperature, fluid properties, time synchronization and uncertainty are measured. Report separately:
- drain-side thermal availability;
- exchanger effectiveness under the declared reference;
- heat delivered to the intended supply;
- showerhead-to-drain loss;
- parasitic pump/fan/control energy;
- pressure drop and fouling state.

Do not label exchanger effectiveness as whole-system recovery.

### Mass and energy closure

Report every boundary stream and the residual:

Residual = inputs - useful_outputs - measured_losses - change_in_storage

A result is admissible only when closure falls within a preregistered uncertainty interval and the residual is explained. Unknown flows cannot be treated as zero.

## 7. Initial target hypotheses

Targets are experimental hypotheses, not product claims.

| Metric | Initial hypothesis |
|---|---|
| Direct-flow primary target | ≤6.0 L/min at 300 kPa, while meeting the selected independent spray/pressure criteria and maintaining acceptable performance at 150 kPa |\n| Direct-flow stretch target | ≤4.5 L/min at 300 kPa, with matched spray force, spray coverage, temperature stability and task outcome; stretch target, not a published standard |
| Recirculation 10-minute service | ≤18 L fresh water including startup, active use, after-session cleaning and purge |
| Whole-service fresh-water saving | ≥80% against matched 2.5 US gpm once-through control, only when all-in boundary and comfort/hygiene gates pass |
| System-level drain heat recovery | ≥60% useful recovery after declared fouling/maintenance, with all parasitic energy and upstream losses counted |
| Comfort and task quality | Non-inferior to matched premium control under a preregistered margin and instrument |
| Lifecycle cost | ≥20% lower cost per verified service as a stretch goal, with no burden-shifting |

Any target that cannot be achieved without violating safety, hygiene, task quality, accessibility or repairability is rejected or revised with a documented decision. Do not tune thresholds after seeing locked confirmation results.

## 8. Adversarial and fault-injection matrix

At minimum exercise, on an unoccupied test article:
1. low supply pressure and pressure transient;
2. high supply pressure within declared test envelope;
3. cold start after long idle;
4. warm restart under short inter-session interval;
5. scale/hard-water accumulation;
6. loaded or partially blocked filter;
7. torn/mis-seated filter or seal;
8. pump degradation / loss of circulation;
9. stuck or ambiguously positioned valve;
10. water-temperature sensor bias, drift, open circuit and stale sample;
11. flow-meter pulse loss or wrong calibration factor;
12. failed/shortened cleaning cycle;
13. stagnant or over-age recirculation batch;
14. power loss during each active state;
15. network/comms loss (core safe function must remain local);
16. overflow, leak and blocked drain;
17. unexpected chemistry / feed excursion;
18. treatment-barrier fault;
19. purge/clean cycle interrupted;
20. repeated-session accumulation / filter exhaustion;
21. measurement clock drift or unsynchronized channels;
22. maintenance lockout bypass attempt.

For each case record expected physical safe state, observed state, evidence required for recovery, water/energy consequences and whether an operator can detect the fault without cloud services.

## 9. Hygiene and exposure qualification gate

No human recirculation trial is permitted until:
- an independent specialist defines the exposure assessment and applicable acceptance limits;
- all water-contact materials and treatment stages have an evidence-backed envelope;
- sample locations, timing, methods, laboratory competence, detection limits and uncertainty are preregistered;
- biofilm, aerosol, stagnation, cleaning, post-use disinfection and cross-user carryover are addressed across water, biofilm and aerosol phases and across shower hoses, showerheads and pipes;\n- the risk-based microbiological panel is selected by competent specialists. Published reviews identify fecal indicators (E. coli, total coliforms, enterococci), heterotrophic plate counts, Pseudomonas aeruginosa and Legionella spp. as important candidates; supplementary organisms and viruses/protozoa may be relevant to the specific treatment/exposure question. This list is not a universal pass/fail panel;
- fault injection proves no unqualified water reaches the shower outlet;
- a competent safety review admits the exact tested configuration and operating envelope.

Turbidity, ORP, pH, conductivity, temperature, UV intensity, ozone residual or filter differential pressure are useful process observations where fit for purpose. None is a standalone proof of pathogen removal or exposure safety.

## 10. Statistical and evidence reporting

- Use a pilot to estimate variance and cluster structure; then preregister confirmation sample size and stopping rule.
- Distinguish independent physical service runs from repeated sensor samples, droplets, images or within-run measurements.
- Report every attempt, exclusion, failure, retry and missing observation.
- Report distributions, confidence intervals / uncertainty intervals, device-to-device variance and cluster-aware analysis.
- Use blinded task assessors where feasible.
- Publish protocol revision, raw-data digests, calibration, units, methods, exact device configuration and analysis code revision.
- A synthetic corpus validates software logic, not physical performance. A room-scale simulation validates only the modeled quantities within its assumptions. Bench evidence is not field deployment evidence.

## 11. Decision gates

- **Gate 0:** competitor specifications and boundary complete enough to freeze comparison; unknowns explicit.
- **Gate 1:** service contract, hazards, measurement plan, data rights and source boundaries reviewed.
- **Gate 2:** deterministic arithmetic and negative controls pass on frozen synthetic fixtures.
- **Gate 3:** unoccupied clean-water direct-flow / heat-recovery rig qualifies its measurement system and energy/water balance.
- **Gate 4:** recirculation unoccupied evidence independently qualifies the exposure process; otherwise recirculation is stopped or redesigned.
- **Gate 5:** human-factors testing only after all relevant safety/ethics approvals.
- **Gate 6:** field pilot only after local professional approvals and an explicit stop/continue decision.

Passing one gate does not imply another. A queued/skipped workflow is not a pass; an exact-head failure remains a failure until repaired and rerun.


## 12. South African comparator protocol addendum

The CLASP South African showerhead study measured 19 models across approximately 250–550 kPa, with spray patterns recorded at 300 kPa. The report recommends considering 150 and 300 kPa test points; WRC's 2025 summary reports pressure-compensating models averaging 6.6 L/min at 300 kPa and fixed-orifice models averaging roughly 24–27 L/min. Aqua's controlled bench should include:
- 150 kPa low-pressure operation;
- 300 kPa reference operation;
- 3 bar for direct comparison with Hansgrohe EcoSmart+ product-family claims;
- 500 kPa as a high-pressure stress point where the exact test article and installed components are rated for it.

Use the same declared inlet temperature, water chemistry, nozzle orientation and task profile across comparator arms. Do not treat CLASP's anonymized 19-product population averages as the measured result of a named comparator.

The 2026 Consumer Reports laboratory programme tests coverage, force, water consumption, temperature drop and usability. Aqua should record the same dimensions where methods can be obtained and reproduced. An exact CR product/model may be included as an external reference only after its market variant is identified and physically tested in the Aqua rig.

## 13. Recirculation exposure addendum

The 2025 review of recirculating-shower microbial indicators reports limited study of water quality in this emerging category and identifies bacterial indicators and waterborne pathogens that may matter. A 2026 review expands the system view to water, biofilms and aerosols across hoses, showerheads and pipes, with opportunistic pathogens including Legionella, Mycobacterium and Pseudomonas. This supports a stronger research gate: treatment performance in bulk water alone cannot be assumed to establish user exposure safety.

Sources:
- Pérez Guillemette & Albrechtsen (2025), https://doi.org/10.1016/j.scitotenv.2025.178610
- Ren et al. (2026), https://doi.org/10.1016/j.jes.2026.02.034
- CLASP South African showerhead report, https://www.clasp.ngo/research/all/south-african-shower-heads-testing-report/
- WRC 2025 water-efficiency update, https://www.wrc.org.za/wp-content/uploads/mdocs/WW%20July%20-%20Sept%202025_web.pdf
- Consumer Reports 2026 lab testing, https://www.consumerreports.org/home-garden/showerheads/best-showerheads-of-the-year-a7580523069/
