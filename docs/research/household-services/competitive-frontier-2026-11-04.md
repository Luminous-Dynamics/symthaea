# Verified Household Services — competitive frontier and benchmark corrections

**Snapshot:** 4 November 2026  
**Status:** source-backed competitive research and proposed matched-test protocol; not product qualification.

## Executive conclusion

Smart features are no longer a defensible novelty boundary. Camera-based load recognition, programme adaptation, automatic detergent dosing, remote start, connected diagnostics, AI-assisted leak detection and automatic water shutoff are all present in incumbent products or announced roadmaps.

The defensible opportunity is to demonstrate **more independently verified household service per full system resource and lifecycle burden**, while retaining repairability, local operation and safe degraded modes.

The competitive objective should be a measured Pareto improvement, not a claim that an appliance is more intelligent.

## 1. Dishwasher incumbent frontier

### Miele G 7655 SCVi XXL AutoDos — published product fiche

Miele's manufacturer fiche for this 14-place-setting model lists:

| Published specification | Value |
|---|---:|
| Capacity | 14 place settings |
| Eco electricity | 0.54 kWh/cycle |
| Eco water | 8.4 L/cycle |
| Eco duration | 230 min |
| Automatic-programme water | from 6.0 L |
| Networked standby | 0.90 W |
| Noise | 42 dB(A) |

The 6.0 L figure belongs to the Automatic programme; the 8.4 L and 0.54 kWh figures belong to Eco. These must never be combined into a fictitious 0.54 kWh / 6 L Eco baseline. The fiche also lists AutoDos, networking, AI Diagnostics, Smart Home Integration, and a manufacturer statement of testing equivalent to 20 years' service life based on a stated development test-cycle programme. That manufacturer statement is not a warranty or field-derived 20-year reliability distribution.

Manufacturer fiche:
https://media.miele.com/downloads/n-/ie/FS_12423980_IEE_IE-en-IE.pdf

### Beko DIN28428 — South African product-page comparator

Beko South Africa's published page for DIN28428 lists 14 place settings, 0.93 kWh/cycle, 9.5 L/cycle and 46 dBA. It shows the older A++ energy-class label, so that class must not be compared directly with the current EU A–G label scale. The numeric values are useful as a regional market reference, but the exact market variant, programme protocol and availability must be frozen before a physical comparison.

Manufacturer page:
https://www.beko.com/za-en/products/integrated-dishwashers/dishwasher-14-place-settings-full-size-din28428

### Miele G8 Diamond — roadmap comparator, not a 2026 retail test unit

Miele announced the G8 Diamond on 2 September 2026. The announcement describes an integrated camera, AI-assisted load detection and wash-cycle adaptation, automatic dispensing, variable hot-air drying and app-based start automation. Miele says launch in Germany, Austria and Switzerland is planned for April 2027, with other countries following.

As of this research snapshot, the G8 Diamond is an announced future product, not a retail unit available for our November 2026 matched physical benchmark. No independently measured energy, water, cleaning, rinse-residue, drying, reliability or repairability result follows from the feature announcement.

Official announcement:
https://www.miele.de/en/m/world-premiere-for-the-g8-diamond-8453.htm

The Cary laundry-care announcement is a concept study, not an established retail product:
https://www.miele.de/en/m/how-miele-eliminates-household-decisions-8441.htm

### What the current product data mean for the stretch target

The current internal stretch objective is 0.39 kWh, 6 L and 16 place settings. Against the Miele G 7655 Eco fiche (0.54 kWh, 8.4 L, 14 place settings), that objective implies:

- 27.8% less Eco electricity per cycle;
- 28.6% less Eco water per cycle;
- 14.3% greater nominal rated capacity.

These are arithmetic comparisons between a research target and a manufacturer product fiche, not matched physical performance results. They are not valid superiority claims until the same standard edition, loading protocol, detergent/reference soil, water condition, endpoint rules and measurement boundaries are used. The 6 L target must be compared to the Eco programme only if the target is also defined and qualified as an Eco-equivalent service.

A capacity number is not a cleaning endpoint. The full load must satisfy cleaning, final-rinse residue, cross-transfer and drying acceptance criteria, with failures and rewashes included.

## 2. Standards frontier

### IEC 60436:2025, corrected April 2026

IEC 60436:2025 COR1:2026 is the current household-dishwasher performance measurement baseline for the programme. The fifth edition includes:
- a new reference machine and reference detergent type E;
- updated low-power-mode measurement, including network standby;
- a method to assess variation between units of one model;
- multi-compartment dishwasher testing;
- updated loading/handling and ballast-soil preparation procedures.

IEC explicitly states that IEC 60436 is neither a safety standard nor a minimum-performance requirement. Use the licensed corrected edition for formal reproduction.

Official IEC record:
https://webstore.iec.ch/en/publication/80459

Corrigendum:
https://webstore.iec.ch/en/publication/112799

This means that multi-compartment architecture, unit variation and network standby are now explicit comparator dimensions. Their mere presence is not a novelty claim.

### Microbiological efficacy is not sanitization

IEC TS 63331:2024 defines reproducible measurement of relative microbial reduction on household dish items. IEC restricts the resulting statements to relative comparisons and expressly does not cover sanitization, disinfection or sterilization.

Official IEC record:
https://webstore.iec.ch/en/publication/66647

NSF/ANSI 184 is a separate residential sanitization certification path. NSF describes a minimum 5-log (99.999%) bacterial reduction and a 150°F final rinse on the certified sanitizing cycle. These claims apply to the relevant NSF-certified product and cycle, not to generic high temperature, camera detection or a relative IEC TS 63331 result.

NSF overview:
https://www.nsf.org/consumer-resources/articles/dishwasher-certification

Cleaning, final-rinse chemical residue, relative microbiological reduction and certified sanitization must remain separate endpoints.

## 3. Optimization and metrology state of the art

### Mechanistic detergent/cycle co-optimization

An October 2026 paper in Computers & Chemical Engineering describes mechanistic models of major dishwasher-cycle stages linked by event-based algorithms and a genetic-algorithm optimization framework. On its reference formula/cycle, it reports:
- 31% shorter wash time;
- 59% shorter drying time;
- 5% higher stain-removal index.

The article states that the data used are confidential. We must therefore treat these as reported publication results, not as a fully reproducible public benchmark with independently inspectable raw data.

DOI:
https://doi.org/10.1016/j.compchemeng.2026.109771

Competitive implication: a genetic algorithm, event-based cycle model, or detergent/cycle co-optimization is not by itself a novel claim. Our differentiator must be public, frozen inputs; independently qualified endpoints; total resource accounting; held-out physical confirmation; production-unit transfer; and external replication.

### Independent cleaning observation

A May 2026 Food Research International study reports fluorescence imaging with 365 nm excitation and FM-YOLO/area correction for oatmeal/porridge residue on rice bowls. It reports 86.7% overall classification accuracy after surface-area correction, but class-level accuracy includes 65% and 70% for two intermediate classes.

DOI:
https://doi.org/10.1016/j.foodres.2026.118853

This is a promising one-soil/one-substrate assay lead, not a universal clean detector. It reinforces the need for:
- soil-family and surface-specific validity envelopes;
- independent reference chemistry/gravimetry;
- uncertainty and abstention around ambiguous intermediate outcomes;
- false-acceptance reporting, not only aggregate accuracy;
- held-out material, shape, lighting, detergent and machine-unit validation.

A 2025 interlaboratory cleaning-chemistry method-development paper also documents the need to refine controlled soil application and stainless-steel coupon procedures for reproducibility:
https://pmc.ncbi.nlm.nih.gov/articles/PMC12453126/

## 4. WaterLoop incumbent frontier

### JoJo municipal backup

JoJo's South African municipal-backup solution already combines:
- automatic filling while municipal supply is available;
- a household storage tank;
- a pump returning stored water to the house;
- a pressure controller that starts/stops the pump;
- float-valve filling and non-return/isolation components.

JoJo also recommends tank cleaning every three months and regular water rotation (every two weeks in its guidance). It explicitly notes that municipal water quality can vary and that additional filtration may be appropriate for drinking/cooking.

Official product description:
https://www.jojo.co.za/water-storage-solutions/municipal-backup-water-storage-solutions/

### Moen Flo

Moen Flo already markets:
- flow, temperature and pressure monitoring;
- AI-assisted learning of household usage patterns;
- daily micro-leak checks;
- real-time abnormal-usage/leak alerts;
- automatic shutoff.

Official product page:
https://flo.moen.com/pages/flo-smart-water-monitor

Competitive implication: a backup tank/pump, pressure controller, smart meter, AI leak detector or automatic shutoff alone is not a defensible WaterLoop novelty claim.

WaterLoop must instead demonstrate end-use-qualified, pressure-qualified household service under matched municipal outage, staggered restoration, low-pressure, power-failure, pump-failure and leakage scenarios. Measure actual litres delivered at named fixture boundaries, usable reserve range, minimum pressure, flow, restoration stability, false shutoff and missed-leak rates, pump/standby energy, maintenance, filter lifecycle and water-quality qualification for each specific use.

Gross tank inventory is not deliverable reserve. A pressure sensor is not a water-quality assay. A pump command is not measured flow. Potable and non-potable routes must remain distinct.

## 5. Repairability incumbent frontier

The French repairability index already scores documentation, disassembly/access/tools/fasteners, spare-parts availability and delivery time, parts pricing, and product-specific criteria, producing a consumer-facing score out of 10.

Official French ministry page:
https://www.ecologie.gouv.fr/politiques-publiques/indice-reparabilite

Our contribution should be realized repair performance, not another score:
- exact part availability and delivered lead time in a named market;
- time to diagnose and restore by task, reporting median and 90th percentile;
- actual tool and access burden;
- first-time restoration and affected-interface requalification;
- parts, labour, downtime and consumable cost;
- recurrence per recorded operating exposure;
- offline diagnostics, firmware rollback and calibration;
- safe referral/hold when the necessary test capability is absent.

The European Commission opened a 3 November 2026 consultation on possible future repairability requirements for additional product groups. It considers spares, repair information, software/firmware, disassembly and repairability information. The consultation is open until 9 February 2027. It creates no new binding product obligations by itself; final product scope and measures remain subject to the process.

Consultation reporting:
https://agenceurope.eu/en/bulletin/article/13974/10/european-commission-launches-public-consultation-on-product-reparability

Do not assume EU obligations apply automatically to products placed on the South African market. South African requirements require their own product-specific legal and technical assessment.

## 6. Recommended competitive benchmark protocol

### Freeze before comparison

For every incumbent and candidate, record:
- exact model, country variant, hardware revision, firmware and accessories;
- exact programme, options, load matrix and loading instructions;
- reference detergent/consumable type and batch;
- inlet water hardness, temperature, pressure and ambient conditions;
- measurement boundary and calibrated instrument identity;
- independent soil, rinse-residue, cross-transfer and drying methods;
- uncertainty budget and endpoint acceptance rules;
- retry, intervention, failure, purge, pre-treatment and rewash policy.

For water, also freeze fixture topology, demand profile, source pressure, outage waveform, pump/electrical dependencies, end-use class and water-quality assay.

For repair, freeze fault/task definition, technician skill/tool profile, part revision, market stock/quote, start/stop event definitions and post-repair test.

### Report a vector, not one weighted score

InstantClean:
- quality-qualified items/hour;
- first independently verified item latency;
- first-pass acceptance and rewash rate;
- soil residue, rinse residue and cross-transfer by item/soil/surface stratum;
- total kWh, water, chemistry, noise and operator time per accepted item;
- standby/networked-standby and whole-system concurrent loads;
- unit variation and failure/maintenance burden.

WaterLoop:
- pressure-qualified litres delivered per fixture;
- reserve-duration interval under an observed demand profile;
- service availability during outage and recovery;
- leak detection latency, missed leaks and false shutoffs;
- recovery time after source/power restoration;
- kWh per qualified delivered litre and water-quality coverage.

HomeRenew:
- qualified restoration time, median and P90;
- first-time repair rate;
- parts fill rate and lead time;
- cost and downtime per qualified repair;
- post-repair requalification pass rate;
- recurrence per operating-hour or service-cycle exposure.

Cross-product:
- availability of the named service;
- time to first useful service and time to full restoration;
- actual total resource use over the same service window;
- cost of ownership and repair over observed exposure;
- uncertainty, missingness, abstention and worst-performing strata.

Never trade away safety, hygiene, cleaning, water quality or material integrity through a weighted average. Apply hard acceptance constraints first, then compare feasible candidates on a published Pareto frontier.

## 7. Evidence and superiority gates

Keep four separate evidence grades:
1. within-unit repeatability;
2. independent-laboratory reproducibility;
3. production-unit variation;
4. real-household validity.

Set confirmatory sample sizes only after pilot variance components and the intended confidence/precision are known. Respect clustering by machine, lab, day, operator, soil/consumable batch and household. Do not count dishes, images, frames or repeated assays from one physical cycle as independent machine trials.

Proposed programme gates:
- E1: repeatable prototype improvement with the full uncertainty budget;
- E2: independent laboratory reproduction of the preregistered primary result;
- E3: production-representative unit distribution retains the claimed improvement;
- E4: representative households retain the improvement under actual demand, maintenance and environment.

The 10%/20% improvement targets in the programme are hypotheses to test after the matched baseline is measured. They are not industry standards or existing performance results.

## 8. Strategic conclusion

The innovation target is not an appliance with the largest feature list. It is a household service that:
- delivers a better independently accepted outcome;
- consumes less total resource for that same outcome;
- remains available through declared disturbances;
- makes uncertainty, failure and abstention visible;
- can be independently repaired and requalified;
- retains performance across production units and real use.

One external replication of such a result is more defensible than a collection of untested smart features or synthetic benchmark wins.


## 9. SOTA refresh — 22 November 2026

**Evidence rule:** the following Topten values are supplier-declared / energy-label data curated in a third-party selection database. They are desk-screening references, not independent Luminous measurements and not automatically comparable with South African legacy-label products.

### Washer frontier

Topten's European list (last updated 17 August 2026) features:
- AEG LR9WW80409: 10 kg; 20 kWh/100 Eco 40–60 cycles; 50 L/cycle; 235 minutes; spin-drying class B; noise class A; maximum spin speed shown as 1,351 rpm.
- Samsung WF9000 / WF90F09C4SU5: 9 kg; 22 kWh/100 cycles; 50 L/cycle; 228 minutes; spin-drying class B.

Topten's methodology identifies supplier declarations, national Topten sites and Encodex as data sources. Its EU programme, reference load and current IEC 60456:2024 test-method changes must be preserved; do not map these numbers to older SANS/legacy label classes.

Source: https://www.topten.eu/private/products/washing_machines
Selection-method context: https://www.topten.eu/private/selection-criteria/washing-machines

### Heat-pump dryer frontier

Topten's European list (last updated 19 May 2026) includes:
- AEG TR9HH9AY: 9 kg; 79 kWh/100 cycles; efficiency index 42.9%; condenser efficiency class A; noise class A.
- AEG TR948V4BC: 8 kg; 73 kWh/100 cycles; efficiency index 42.9%; condenser efficiency class A; noise class A.

The lower 73 kWh/100-cycle number belongs to an 8 kg model, not a 9 kg service capacity. A noise class is not a numeric dB(A) result. Use the exact PNC / regional fiche, initial moisture, final moisture, programme and matched textile load for physical selection.

Source: https://www.topten.eu/private/products/tumble_driers

### Refrigerator frontier

Topten's European list (last updated 10 September 2026) lists the Liebherr CBNbsa20 575i at:
- 362 L total net volume (259 L refrigerator + 103 L freezer);
- 92 kWh/year label energy;
- ambient operating range 10–43°C;
- noise class A; frost-free.

This provides a useful high-ambient desk comparator. It does not prove energy consumption or food-temperature performance at every point from 10–43°C, nor outage holdover, warm-load pull-down or food safety.

Source: https://www.topten.eu/private/products/refrigerators

### HeatBank direct-DC PV water heating — new 2026 publication

Walker, Thomas and Booysen, *Energy for Sustainable Development* 94 (October 2026), DOI 10.1016/j.esd.2026.102053, evaluates a 1.68 kWp / four-module direct-DC PV-to-electric-water-heater arrangement across seven Sub-Saharan African cities including Johannesburg. The paper reports a 5-minute simulation using a two-node tank model, stochastic household hot-water draws and 2024 satellite meteorology. Its abstract reports:
- over 2,200 kWh/year accepted by the heater in the Johannesburg heavy-user scenario;
- 21–47% simulated water-heating-bill savings across studied cities and demand profiles;
- up to 10.3% avoidable cold draws for Johannesburg heavy users under the PV-priority configuration.

These are model outputs, not measured household savings or a qualified commercial product. The public code/results repository is:
https://github.com/WalkerFelix/Simulation-Based-Solar-PV-Assisted-Domestic-Water-Heating-in-Sub-Saharan-African-Cities

An initial static inspection of the public repository found several reproducibility questions, tracked separately under Symthaea HEATBANK-DIRECTPV-REPRO-001:
- repository configuration currently contains nine South African city keys, while the paper's abstract describes seven Sub-Saharan African cities; the exact commit/data bundle corresponding to the journal results needs pinning;
- `config.py` uses a developer-specific absolute Windows ROOT_DIR and expects external `solar_data` and `user_data` directories;
- `tariff.py` assigns a flat 0.186 USD/kWh rate to all nine configured cities;
- `simulate.py` values PV energy at that flat tariff and constructs `cost_without_solar` as grid cost plus PV energy valued at the tariff, while the separate grid-only tank is used for cold-event comparison and its baseline electricity/cost is not accumulated into the returned KPI set;
- `tank.py` applies draw-related energy removal first to the top node (then bottom if needed), but the inspected step does not explicitly advect/replace cold mains water into the lower tank node, so the two-node stratification trajectory needs independent mass/energy validation;
- the current code snapshot's root `README.md` and `requirements.txt` were not found by the public file fetch; a reproducible run manifest and dependency/data checksums are needed.

These are source-level questions, not a finding that the article's results are false. Do not use the published percentage or kWh values as HeatBank design guarantees until the exact paper result is reproduced and the physical tank/controller model is independently checked.

### Critical performance-method scope correction for HeatBank

IEC 60379:2023 is a method for household electric storage water heaters, but explicitly excludes heaters that use electricity as a **secondary** heating source and does not specify product safety. It must not be applied indiscriminately to direct-DC solar-primary, solar-thermal, hybrid-backup or heat-pump architectures.

Use an architecture-specific method map:
- conventional electric resistance storage: investigate IEC 60379:2023 for performance, and IEC 60335-1 with IEC 60335-2-21 for safety;
- solar-only system: investigate ISO 9459-2:1995 outdoor method within scope; note ISO/AWI 9459-2 Edition 2 is under development and is not yet a published replacement;
- solar system with integral backup / preheat: investigate ISO 9459-4:2013 component-test plus simulation method;
- packaged heat-pump water heater: investigate EN 16147 and applicable national adoption / local conformity route; do not substitute IEC 60379 merely because a storage tank is present.

All methods remain subject to South African SANS adoption, product classification, current SANAS scope, NRCS/VC 9006 and installation requirements. This crosswalk does not certify applicability.

Sources:
- IEC 60379:2023: https://webstore.iec.ch/en/publication/61520
- ISO 9459-2:1995: https://www.iso.org/standard/17187.html
- ISO/AWI 9459-2: https://www.iso.org/standard/95343.html
- ISO 9459-4:2013: https://www.iso.org/standard/41695.html
- EN 16147 reference context: https://www.ehpa.org/wp-content/uploads/2022/07/EHPA_TestReg_DHW-HP_V1.8_February_2021.pdf

## 10. Next decision consequence

The SOTA scan confirms that an energy-label win alone is unlikely to be a durable differentiator. The immediate design questions are now more precise:
1. Can our washer-to-dryer service beat the 20 kWh/100-cycle washer reference and 73–79 kWh/100-cycle dryer references at equivalent accepted textile quality and final moisture while retaining repair and textile-life advantages?
2. Can FoodSteward's refrigeration architecture deliver better useful cold-storage service and outage performance at a competitive whole-life cost than a 362 L, 92 kWh/year, 10–43°C desk reference?
3. Can HeatBank reproduce the direct-DC model's useful service / cold-draw trade-off with an independently validated tank model, local tariffs and exact installed-system costs?
4. Can the fan and appliance measurement cell yield independent repeatability and laboratory correlation rather than merely producing plausible plots?

No percentage superiority claim follows from this desk update.


## 11. South African direct-PV water-heating incumbent frontier — 22 November 2026

**Novelty correction:** direct-PV geyser conversion, MPPT control, AC backup, retrofit installation and basic remote monitoring are already commercial capabilities in South Africa. Luminous must not describe those features as novel.

### Local incumbent evidence

**Geyserwise PV Water Heating / PV2Heat**
- Official product page describes conversion of existing geysers, PTC AC/DC elements and 100 L, 150 L and 200 L configurations.
- Its MWS Controller page states MPPT maximum performance of 99% over 120–350 V DC.
- Its Geyserwise Dual page describes AC/DC setpoint ranges, timer modes, fault codes and mechanical over-temperature cut-out.

Manufacturer sources:
- https://www.geyserwise.com/our-products/pv-water-heating-systems/
- https://www.geyserwise.com/mws-controller/
- https://www.geyserwise.com/our-products/geyserwise-electronic-thermostats/geyserwise-dual/

**Microcare Hot Water Generator / Kushushu**
- 2025.1 manual describes a local retrofit using 2, 3 or 4 kW geyser elements, PV and automatic AC mains override, thermistor control and existing thermostat/element compatibility.
- Its manual says solar heating efficiency is greater than 95%; a separate August 2025 product datasheet claims greater than 96%. Keep these as two manufacturer-source claims; do not average them or treat either as independent testing.
- The manual restricts installation/service to qualified registered electrical personnel and warns against unauthorised opening/modification.

Sources:
- https://microcare.co.za/wp-content/uploads/2025/09/Microcare-HWG-Manual-2025v1.1.pdf
- https://microcare.co.za/wp-content/uploads/2025/08/Microcare-Solar-Geyser-SolutionData-Sheet.pdf

**Apollo NeoPV MPPT**
- Product page describes existing-geyser retrofit, MPPT and AC mains backup / bypass, optional app/cloud monitoring and listed 100/150 L (3 × 580 W), 200 L (4 × 580 W) and 300 L (5 × 580 W) configurations.
- It states MPPT efficiency greater than 99% and advertises up to 60–90% water-heating bill savings.
- These are manufacturer-published claims; the product page's two input-voltage / input-power configuration columns need exact SKU mapping, and no independent service result follows from the marketing figures.

Source:
https://apollotechnology.co.za/neo-pv/

**Market maturity context**
- The May 2024 South African Water Heating DSM study estimated more than 25,000 direct-PV geyser systems installed nationally and over 7,000 installed during 2022 alone. It placed a direct-PV system at an estimated R19,000 excluding the geyser and identified DC arcing due to incorrect installation as a fire risk. These are historical study estimates, not a current product census or price quote.
  https://www.savingenergy.org.za/reports/June%202024/FINAL%20REPORT-%20South%20Africa%20Water%20Heating%20DSM%20Study%205.26.2024.pdf
- The IEA SHC Task 69 market report dated 10 June 2026 says South Africa had experienced a flux of PV water-heating entrants, many exiting or rebranding, and names Elon and Geyserwise among the most prominent systems then available.
  https://task69.iea-shc.org/Data/Sites/1/publications/IEA-SHC-Task69-A.1-Report-Dominant-Solar-Water-Heating.pdf

These market sources establish that basic direct-PV water heating is not novel; they do not establish the current market share, service life, price competitiveness or performance ranking of these exact systems.

### Controller standards frontier: IEC 63349

IEC 63349-1:2025 (published 4 December 2025) now gives a directly relevant standard for photovoltaic direct-driven appliance controller functions and operational performance. It covers controllers coordinating power from PV, grid, storage and appliances (including water heaters, air conditioners, refrigerators and pumps) for systems up to 1,500 V DC or 1,000 V AC.

Crucially, IEC 63349-1 explicitly does **not** cover safety or EMC requirements for the controller. It references separate safety requirements for converters and IEC 62093 for converter performance. It is not a product safety certificate or an appliance energy test.

- IEC 63349-1:2025: https://webstore.iec.ch/en/publication/67424
- IEC TS 63349-2:2022, operation modes and example graphic display: https://webstore.iec.ch/en/publication/67425

### Revised HeatBank differentiation hypothesis

Do not compete on the existence of a PV-geyser controller, MPPT, timer, AC backup or app. Compete on **independently verified accepted hot-water service and whole-system lifecycle performance**.

Compare the exact Microcare, Geyserwise and Apollo local-market configurations against the conventional geyser, solar thermal and heat-pump alternatives. Record exact SKU, hardware/firmware, tank/element identity, panel/module datasheet, installed cost, installer, protection devices, warranty, service route and evidence class before physical comparison.

The candidate controller/system should be evaluated on:
- accepted mixed hot-water litres per declared temperature-time service band;
- grid kWh, PV generated, PV accepted by heater, curtailed PV, element/converter/control losses and tank/distribution losses;
- cold-draw litres, temperature, duration, recovery and unmet demand;
- performance across irradiance ramps, partial shading, module mismatch, hot/cold ambient, inlet-water temperature and heavy morning/evening draw profiles;
- source-transfer behaviour, DC disconnect/isolation, abnormal-state handling, fail-safe thermal limits and manual/offline fallback;
- peak grid demand and service continuity through grid/PV disturbances;
- maintenance/repair, component compatibility, parts lead time and verified post-repair thermal/service output.

A manufacturer's MPPT efficiency claim of 95–99% is not a household seasonal service efficiency and cannot substitute for a whole-system measurement.

## 12. Direct-PV paper/code provenance clarification

The linked public repository's nine South African city configuration is consistent with a June 2026 SSRN preprint titled *Simulation-Based Assessment of Solar PV-Assisted Domestic Water Heating in South African Cities*, whose abstract describes a 1.68 kWp five-minute simulation across nine South African cities:
https://papers.ssrn.com/sol3/Delivery.cfm/82011b4b-e1c6-4d6a-b907-e55e6f9fef0b-MECA.pdf?abstractid=6929312

The October 2026 *Energy for Sustainable Development* paper is a subsequent journal publication titled *Assessment of solar PV-assisted domestic water heating in Sub-Saharan African cities* and its abstract describes seven cities (Cape Town, Johannesburg, Lusaka, Luanda, Kinshasa, Nairobi and Lagos):
https://doi.org/10.1016/j.esd.2026.102053

This probably explains why the currently retrieved public code lists nine South African cities while the journal abstract lists seven Sub-Saharan African cities. However, the repo-head commit, data files and result files must still be mapped to the exact preprint and journal-paper versions before either is said to be independently reproduced.

The June 2026 preprint is a related research source, not proof that the current public repository exactly reproduces the later October journal publication.

## 13. Research direction after the novelty correction

The stronger opportunity is an open, standards-aware **household energy and thermal service controller** that can coordinate hot water with PV, grid availability, tariff periods and other household resources while preserving independent appliance safety and user control.

The controller is only one layer. The qualification target is the physical service:
- useful hot water at a named delivery boundary;
- temperature/time and hygiene constraints;
- whole-system electrical/thermal balance;
- accepted service under cloudy weather and supply interruption;
- repairability and service continuity;
- independent reproduction across units, labs and households.

The existing HeatBank, household utility coordination, metrology, evidence and manufacturing owners should compose this work; do not open a duplicate scheduler, evidence ledger, safety authority or manufacturing system.
