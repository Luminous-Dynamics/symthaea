# Luminous Aqua — Thermal Suite Benchmark Protocol v0.1

**Status:** research / preregistration draft; no thermal-suite product has been tested under this protocol  
**Parents:** [AQUA-BATHING-SERVICE-001 #12660](https://github.com/Luminous-Dynamics/symthaea/issues/12660), [AQUA-THERMAL-SUITE-001 #12694](https://github.com/Luminous-Dynamics/symthaea/issues/12694), [AQUA-WETROOM-ENVELOPE-001 #12702](https://github.com/Luminous-Dynamics/symthaea/issues/12702)  
**Owner composition:** Symtropy #4531/#2152 and existing thermal/moisture owners; Mycelix #6025/#6029/#6034/#6048/#6054/#6057/#6100; Sol-Atlas #765/#766/#770/#772.

## 1. Research question

Can Aqua provide a declared bathing / thermal exposure service with non-inferior comfort, temperature/humidity field, ventilation, access, safety and serviceability while reducing complete-session purchased energy, water and lifecycle burden?

The comparison unit is not heater nameplate kW, room temperature alone, a wellness feature list, or a marketing claim. It is a **completed, condition-qualified occupant service**, counted from request to post-session ready state.

Do not combine dry sauna, infrared sauna, steam room, cold plunge, whirlpool and pool into one score. They have materially different physics, water-quality classes, room envelopes and exposure risks.

## 2. Market / technical references

### Dry sauna

- A Finnish household-sauna energy guide checked 8 October 2026 cites a Motiva energy-efficiency expert estimate of **8–16 kWh for a little over one hour** in a basic household sauna. The guide identifies room volume, heater rating, insulation, warm-up and time powered as the main drivers. This is a sanity-check range, not a matched laboratory result or appliance specification: https://saunatutka.fi/en/opas/saunan-lammityskustannus-ja-turvallisuus
- HUUM's STEEL product page lists 6, 9 and 10.5 kW variants for nominal sauna volumes of 5–9, 8–13 and 10–16 m³ respectively, with 150 kg stone capacity. These are equipment sizing/catalogue values, not measured energy per occupant-hour or thermal uniformity: https://huum.eu/product/steel-electric-sauna-heater/
- HUUM's installation guidance notes that uninsulated brick/tile/glass surfaces affect effective room sizing and heater selection; it also requires electrical supply, clearances and room volume checks. This is a manufacturer installation reference, not a substitute for local electrical design: https://huumsauna.com/support-category/electric-sauna-heaters/

### Infrared sauna

There is no directly comparable power-only SOTA figure. Emitter spectrum, radiant intensity, distance, surface temperature, enclosure, body-plane uniformity, warm-up, duty cycle and total occupancy determine the delivered service. Compare measured radiant/thermal fields and kWh per qualified occupant-hour, not nominal electrical watts alone.

### Steam / hammam

Treat steam generation, water feed, scale, room vapor envelope, condensate, drainage, surface dew point, exhaust and post-use dry-out as one service boundary. A steam generator's kW rating or advertised warm-up time does not measure the room's moisture safety or all-in service energy.

### Cold plunge / whirlpool / pool

- Shared immersion and whirlpool water are distinct sanitation/exposure classes. CDC hot-tub guidance highlights disinfectant residual, pH, sediment/biofilm, temperature and water age as relevant Legionella-control considerations: https://www.cdc.gov/control-legionella/php/toolkit/hot-tub-module.html
- Indoor pools are a natatorium / high-dew-point building class. ASHRAE recommends a holistic envelope/HVAC approach involving humidity, room pressure, ventilation, air distribution, evaporation, water chemistry and total energy; HVAC alone cannot fix envelope defects: https://handbook.ashrae.org/Handbooks/A23/IP/A23_Ch06/A23_Ch06_ip.aspx
- Pool thermal / evaporation / water-feature and heat-recovery claims remain with Sol-Atlas #765/#766/#770/#772. Aqua must not create a duplicate pool solver.

### Relevant current standards / guidance references

- **IEC 60335-2-53:2026, Edition 5.0**, published 15 October 2026, is the current IEC particular safety standard found for household/similar sauna heating appliances and infrared cabins within its scope. It replaces the 2011 edition with 2017 and 2021 amendments and includes revised thermal-storage-heater and access-to-live-parts provisions. Confirm local adoption and applicable conformity route: https://webstore.iec.ch/en/publication/106068
- **ISO 21542:2021** is the published international accessibility and usability reference for the built environment; use the controlled standard and applicable national rules for actual fixture/room dimensions: https://www.iso.org/standard/71860.html
- CDC's **2024 fifth-edition Model Aquatic Health Code** is model guidance for public aquatic venues; it is not automatically local law: https://www.cdc.gov/model-aquatic-health-code/php/our-work/index.html

## 3. Functional units and system boundaries

Each class gets its own functional unit and exact conditions.

| Class | Primary unit | Full system boundary |
|---|---|---|
| Dry sauna | one qualified session and occupant-hour | start request, warm-up, occupied session, door openings, standby during occupancy, post-session ventilation/cool-down, return to ready |
| Infrared sauna | one qualified session and occupant-hour | emitter warm-up, delivered radiant field, occupied session, control duty, post-session state |
| Steam / hammam | one qualified session and occupant-hour | generator startup, feed water, steam delivery, room heat/moisture response, drainage/condensate, ventilation/dry-out and scale burden |
| Cold plunge | one admitted immersion service and occupant-hour | source/make-up, cooling, circulation, treatment, user load, temperature recovery, cleaning and sanitation |
| Whirlpool / hydrotherapy | one admitted recirculating-water service | fill/turnover, recirculation, disinfection, filtration, aerosols, cleaning, purge, make-up, energy and maintenance |
| Pool / natatorium | qualified occupancy-hour and declared water-temperature duty | pool heat, evaporation, make-up, filtration, treatment, building HVAC/dehumidification, ventilation, heat recovery and water-quality controls |

All service windows begin at the user's request and end only after the unit has returned to its declared ready state. Count:
- cold start and warm start separately;
- preheat / standby energy;
- all heating, pump, fan, controls, treatment and dehumidification energy;
- water feed, make-up, evaporation, purge, cleaning, backwash and reject streams;
- thermal storage charge/discharge and storage loss;
- ventilation/dry-out after occupancy;
- consumables, service time, cleaning labour, faults and retries.

The room size, envelope, outdoor weather, occupancy, door events, water/air setpoints and session length must be matched across comparator arms. Shared-room loads are allocated once per full system window, never double-counted across fixtures.

## 4. Comparator arms

Freeze exact models, market variants, rated power, room sizing instructions, electrical supply, stone mass, controls, firmware, clearances, installation and service documentation before testing.

| Arm | Role | Evidence class / limitation |
|---|---|---|
| H0 — matched incumbent dry-sauna heater | quality and energy control | exact SKU, room and installation required; nameplate power is not performance |
| H1 — HUUM STEEL 6/9/10.5 kW family | current premium heater sizing reference | catalogue data only; room sizing and energy must be independently measured |
| H2 — matched Harvia heater/room package | second incumbent | select exact model and manual before test; no cross-model generic claim |
| IR0 — matched infrared cabin | radiant baseline | compare measured radiant field and occupancy, not just kW |
| S0 — matched steam generator + room | steam baseline | include envelope, water feed, scale, ventilation and dry-out |
| C0 — conventional cooled plunge | cooling baseline | use actual duty cycle, water volume, insulation and user load |
| W0 — conventional whirlpool / hot tub | recirculating aquatic baseline | sanitation and aerosol evidence required; do not expose humans on unqualified setup |
| T0 — Aqua passive/envelope-first suite | design candidate | room envelope, passive stratification, local capture and serviceability must be frozen |
| T1 — Aqua indirect heat-cascade suite | design candidate | indirect heat recovery and thermal store; count parasitics and useful heat at receiver |
| T2 — Aqua bounded adaptive suite | optional advanced candidate | local control only within qualified envelope; compare against passive/manual control |

Do not select a single model as the SOTA winner based on market marketing. Benchmark at least one exact, current SKU per relevant class and retain all unknowns.

## 5. Frozen conditions and metrology

Preregister:
- room geometry and envelope revision, surface materials and thermal-bridge details;
- heater/generator/heat-pump model, installation and supply;
- outdoor and indoor temperature/humidity, weather data source and timebase;
- occupancy count, body/task profile, duration and door/opening schedule;
- measured air temperature at multiple heights/locations;
- globe/radiant and surface temperatures as distinct measurands;
- water source/temperature/flow/quality and immersion-water sample points where relevant;
- airflow, exhaust, pressure differential and ventilation operating mode;
- electrical energy at appliance and whole-room boundaries;
- condensate/drain flows, storage change, purge and make-up water;
- calibration, uncertainty, installation, device serial, operator, day and test order.

Use Mycelix-approved measurand definitions and channel identities. A thermostat's displayed reading is not a spatial thermal-field measurement. A humidity sensor does not establish that concealed surfaces are condensation-safe. A vendor kWh estimate is not a meter.

## 6. Class-specific endpoints

### A. Dry sauna

Measure:
- request-to-ready time;
- energy by warm-up, occupied, idle and post-session phase;
- air temperature stratification and radiant/surface field;
- response to door openings and occupancy;
- ventilation and air-quality state;
- stones, heater thermal storage, standby and post-session loss;
- seating ergonomics, safe clearances, access and egress;
- service/cleaning time and component condition.

### B. Infrared cabin

Measure:
- emitter type, spectrum and measured radiant flux distribution;
- user-plane exposure distribution at declared seat positions;
- air temperature vs radiant/surface temperature;
- warm-up, controller duty, kWh/session and standby;
- enclosure / surface temperature and electrical safety;
- comfort and user preference only after independent safety/ethics approval.

Do not treat infrared cabins as low-energy winners unless equal service and measured radiant exposure are demonstrated. No detoxification or medical claims.

### C. Steam / hammam

Measure:
- generator warm-up and delivered steam/heat;
- feed water and kWh;
- indoor air temperature, RH, dew point and spatial distribution;
- surface temperature and condensation at envelope junctions;
- ventilation and post-session dry-out;
- condensate/drainage and scale accumulation;
- generator service access, descaling and downtime.

No occupied use until the exact enclosure/waterproofing/vapor/ventilation design is professionally reviewed.

### D. Cold plunge / whirlpool

Measure:
- water temperature and spatial stability;
- cooling energy and temperature recovery under matched bather load;
- water volume, turnover, make-up, purge and cleaning;
- filtration/disinfection state and independent sample results;
- biofilm / aerosol assessment where recirculation or spray exposure exists;
- electrical isolation, entrapment/egress, slip and emergency shutoff.

No human immersion trial while quality, sanitation, circulation, electrical or egress evidence is missing or expired.

### E. Pool / natatorium

Aqua records the room-service requirement and consumes Sol-Atlas pool thermal/water/evaporation results. It does not calculate pool heat, evaporation or treatment. Report whole-room and pool system energy at separate meters and retain the owner's exact evidence boundary.

## 7. Candidate targets — hypotheses, not achieved results

| Metric | Target hypothesis | Non-compensatory condition |
|---|---|---|
| Whole-session energy | ≥15–20% lower kWh per qualified occupant-hour vs matched incumbent as a stretch | equal room volume/envelope, target thermal field, occupancy, ventilation and session |
| Ready time | no slower than matched incumbent, or declare a distinct ready-time / energy Pareto tradeoff | do not improve warm-up by omitting the post-session state |
| Thermal uniformity | non-inferior or improved spatial temperature/radiant field | use declared measurement locations and uncertainty |
| Steam-room envelope | no visible or concealed condensation within the declared qualified operating envelope | professional envelope review and surface/dew-point measurements |
| Water / sanitation | zero use-class mismatch and zero release on missing/expired critical quality evidence | independent quality method and authority |
| Local safety | 100% essential interlocks/manual fallback operate without cloud, network or Symthaea | independently reviewed fault injections |
| Lifecycle | lower cost per qualified service without transferring cleaning or maintenance burden to residents | cost, parts, labour, downtime and material end-of-life declared |

The 15–20% energy figure is an optimization hypothesis, not a universal efficiency criterion. If a product cannot achieve this without degraded comfort, moisture control, serviceability or safety, the result is a documented Pareto tradeoff or a stop/pivot decision.

## 8. Adversarial / fault corpus

1. cold start and warm start;
2. repeated sessions with insufficient room recovery;
3. door held open and realistic occupant ingress/egress;
4. heater sensor drift or failure;
5. independent high-limit trip / contactor fault;
6. stone packing / heater loading outside declared installation;
7. ventilation failure or wrong pressure direction;
8. cold-surface condensation bridge;
9. steam-generator scale / low-water fault;
10. condensate drain blockage;
11. water-quality monitoring overdue;
12. disinfectant residual / pH excursion in recirculating immersion water;
13. pump loss / filter loading / blocked suction-return;
14. temperature sensor disagreement;
15. power/network/cloud loss;
16. manual maintenance lockout and service restart;
17. shared plant failure that affects multiple modes;
18. mode switch while the previous zone has not returned to ready state.

Unknown or invalid evidence must produce a Hold / Isolate / Refer disposition, not a green mode state.

## 9. Room integration and commissioning

Use a test cell linked to Symtropy #4531. Compare independent, shared-utility and hybrid/local-autonomy configurations under identical event traces. Shared resources may reduce capex/energy but also create common-mode faults, contamination paths, repair dependency and delayed recovery.

Commission:
- wet/dry/steam/sauna/aquatic zone pressure and airflow;
- room surface/dew-point and concealed-risk inspection;
- drainage and overflow paths;
- water, electrical and thermal measurement closure;
- post-session moisture recovery and service access;
- local isolation, manual operation and safe egress;
- exact device / room / model / sensor revisions and owner receipts.

## 10. Data and evidence tiers

- Literature/vendor reference: context only.
- Synthetic scenario: validates arithmetic/contract only.
- Digital twin: validates modeled outputs only under declared assumptions.
- Unoccupied bench: validates the measured test article and conditions.
- Independent lab / exposure study: qualifies only its tested sample and method.
- Human factors: requires independent safety review, informed consent and any applicable ethics approval.
- Field pilot: requires local professional and regulatory review.

No tier inherits proof from a lower tier. Repeated sensor samples, aerosol particles or images from one run are not independent physical trials.

## 11. Claim ceiling

This protocol does not establish efficiency, health benefit, safe thermal exposure, pathogen control, accessibility certification, sauna or steam regulatory compliance, pool-water safety, product superiority, affordability or commercial readiness. All targets remain hypotheses until the exact physical system and comparison protocol are independently tested.
