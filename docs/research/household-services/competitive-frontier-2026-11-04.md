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
