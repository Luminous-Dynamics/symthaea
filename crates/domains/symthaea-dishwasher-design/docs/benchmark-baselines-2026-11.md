# InstantClean benchmark baselines and target ladder (2026-11-02)

This is a research ledger, not a certified comparison. Labelled eco-programme data are not interchangeable with short-cycle data. Always retain exact model, market, test standard, programme, load and measurement boundary.

## Standards and policy

### Household performance

- IEC 60436:2025, corrected version 2026-04, is the current fifth-edition household dishwasher performance-measurement method found in this review.
- The fifth edition introduces a new reference machine and reference detergent type E; updates cleaning/drying and low-power measurement; and adds a multi-compartment testing method (Annex X).
- IEC 60436 defines measurement methods. It does not itself set minimum performance requirements or establish product safety.
- The complete standard is a paid 289-page publication. Obtain the licensed copy and follow the actual clauses before claiming IEC-conforming results. The public product page is not a substitute.

Official reference: https://webstore.iec.ch/en/publication/80459

#### Household electrical safety — separate from performance

IEC 60335-2-5:2025 is the seventh edition of the particular household-dishwasher safety requirements, published 18 December 2025. It replaces the 2012 edition plus Amendment 1:2018. The public IEC summary identifies relevant technical updates including surface-temperature limits, testing for leakage of the rinse-agent reservoir, requirements concerning simultaneous operation of multiple loads and updated detergent/rinse-agent references. Use it together with the applicable latest IEC 60335-1 edition and national adoption; obtain the licensed text and perform a clause-level compliance review before design freeze. IEC 60436 remains a performance-measurement method and is not a substitute for this safety assessment.

Official reference: https://webstore.iec.ch/en/publication/101702

## United States qualification

ENERGY STAR's current published standard-dishwasher criteria remain <=240 kWh/year and <=3.2 US gallons/cycle; compact criteria are <=155 kWh/year and <=2.0 gallons/cycle. This is a qualification ceiling, not the best-product frontier, and US annual metrics must not be mixed directly with EU per-eco-cycle labels.

Official reference: https://www.energystar.gov/products/dishwashers/key_product_criteria

### EU ecodesign and label context

The EU label reports energy per 100 eco cycles, eco-programme duration, water per eco cycle, capacity and noise. The Commission states that an efficient machine can wash up to 15 place settings using about 9 L, versus about 20 L for handwashing the same crockery volume. That is broad context, not a per-mug comparison.

Official reference: https://energy-efficient-products.ec.europa.eu/product-list/dishwashers_en

EU ecodesign rules also require access to certain spare parts (including hinges/seals, spray arms, filters and racks) for 10 years after a model leaves the market, and relevant professional repair information/parts for at least 7 years. Repairability is therefore a competitive requirement as well as a product principle.

### South Africa

Government Notice 7859, published 28 August 2026 in Gazette 55257, amends VC9008 and states phase 1 and phase 2 effective 12 and 24 months after publication. That implies 28 August 2027 and 28 August 2028 respectively. The public notice page reviewed here does not expose the complete schedule; obtain the full gazette and have NRCS/SANS applicability reviewed before product design freeze or market placement. Do not infer appliance-specific thresholds from the notice headline alone.

Notice record: https://www.acts.co.za/national_/n7859_notice_no__7859_of_2026

## Commercial architecture and sensor precedents

The following are important incumbents, not novelty claims for InstantClean:

- **Independent compartments are already commercial.** Fisher & Paykel's Series 7 Double DishDrawer quick reference guide dated 10 March 2026 describes two independently operated drawers, each a true half load, with simultaneous cycles possible and Quick, Sanitise and Dry+ options. The manufacturer states its Sanitise final-rinse option kills 99.9% of bacteria and is certified to NSF/ANSI 184; the precise model/market guide and certification scope must be checked before making any equivalent claim.
  - 2026 guide: https://assets.fisherpaykel.com/product/2883_d000ae48-46ad-4dc7-ab83-8395baf534ca
- **Single-drawer baseline.** Fisher & Paykel DD60SCX9 is listed as a 7-place-setting single DishDrawer with 136 kWh/year and 6.9 L/wash, plus Quick, Sanitise and Extra Dry options. Annual energy and per-wash water are different rating metrics and must not be combined into EU eco-cycle comparisons.
  - 2026 quick reference: https://assets.fisherpaykel.com/product/2883_71bbf617-1ab5-4c91-8a6a-6cc9ed584e63
- **Soil-sensing auto cycles are established.** ENERGY STAR describes auto-sensing cycles that detect whether water coming off dishes is dirty or clean and adjust operation to save water. GE's current CleanSensor/ExtraClean support documentation describes optical turbidity sensing, including a laser-based variant that adjusts wash time, water temperature, water amount and fills.
  - ENERGY STAR: https://www.energystar.gov/products/dishwashers
  - GE CleanSensor: https://products.geappliances.com/appliance/gea-support-search-content?contentId=22177

Therefore, dual compartments and turbidity sensing alone are not InstantClean differentiators. The defensible wedge must be demonstrated on a matched basis: a genuine one-item / micro-batch cell, measured clean-and-rinse endpoints, per-successful-item resources, time-to-release, cross-contamination controls, repairability and robust fallback when sensors are degraded. Any patent novelty or freedom-to-operate claim requires a separate patent search and legal review.

### 2026 announced camera-AI dishwasher incumbent

Miele announced the G8 Diamond on 2 September 2026. Its AutoVision camera uses image processing and AI-based recognition to estimate load level and item types, then applies load-dependent wash/dry settings (e.g. stronger cleaning for pots, optimized internal bottle cleaning, and drying selection for plastics). It is integrated with AutoDos automatic detergent dispensing, three ExcellentDry levels and app preferences for start scheduling / load thresholds / available photovoltaic energy. Miele states the DACH launch is April 2027, with other countries to follow.

This means camera-based load recognition, item classification, automatic dosing and adaptive drying are not sufficient novelty claims. The differentiator must be demonstrated through quality-calibrated uncertainty, ground-truth cleaning/rinse tests, fault/occlusion detection, privacy-preserving operation and matched resource-per-successful-clean improvement. The press release contains no comparable cycle energy/water or cleaning-index dataset; do not insert invented values into the numeric leaderboard.

Official source: https://www.miele.de/en/m/world-premiere-for-the-g8-diamond-8453.htm

## Exact-model market reference points

| Exact model / reference | Capacity | Labelled eco energy | Water per cycle | Eco duration | Evidence notes |
|---|---:|---:|---:|---:|---|
| V-ZUG AdoraDish V6000 heat-pump, AS6T-41173 | 13 place settings | 0.394 kWh/cycle (39 kWh/100) | 8.6 L | 370 min (Topten listing) | Official V-ZUG product information and independent Topten listing |
| V-ZUG AdoraDish V6000, AS6T-41171 | 14 place settings | 0.46 kWh/cycle | 9.0 L | 175 min | Official V-ZUG data sheet; 38 dB also listed |
| V-ZUG AdoraDish V6000, AS6T-41147 | 14 place settings | 0.53 kWh/cycle | 6.7 L | 405 min | Official data sheet; 38 dB; low-water reference |
| AEG GI9210X2TF (911474011) | 14 place settings | 0.488 kWh/cycle (49 kWh/100) | 8.4 L | 310 min | EPREL product record; cleaning index 1.121, drying index 1.061 |
| Haier XS 63A4M4B | 16 place settings | 0.39 kWh/cycle (39 kWh/100) | 9.5 L | 315 min | Topten current featured-product table |
| V-ZUG AS6T6HVI / AS6T-41171 Sprint programme | 13 settings on brochure | Not separately established for Sprint | Standard-cycle specification lists 6.5 L | Sprint duration 20 min | The 20-minute programme is real; do not attribute the standard-cycle kWh/L figures to Sprint without programme-specific test data |
| Fisher & Paykel Series 9 Single DishDrawer DD60SDFTX9 | 7 place settings | 136 kWh/year (different rating basis) | 6.9 L/wash | Quick modifier; duration not in captured data | Independent-drawer architecture benchmark; annual rating is not directly comparable with EU per-cycle eco data |

Sources:
- V-ZUG AS6T-41173: https://www.vzug.com/fr/en/energy-labels-dishwashers
- V-ZUG AS6T-41147 data sheet (14 settings, 0.53 kWh, 6.7 L, 405 min): https://www.vzug.com/occ/v2/gb-site-spa/pdf/product/euDatasheet/V-ZUG-AS6T6GBHVI-4114700009.pdf?lang=en_GB&productCode=4114700009
- ApplianceBench EU water-use report (updated 2026-09-25; 166 qualified models): https://appliancebench.com/en-gb/reports/dishwasher-water
- V-ZUG AS6T-41171 data sheet: https://www.vzug.com/occ/v2/au-site-spa/pdf/product/euDatasheet/V-ZUG-AS6T6HVI-4117100007.pdf?lang=en_AU&productCode=4117100007
- AEG EPREL: https://eprel.ec.europa.eu/screen/product/dishwashers2019/2400252?navigatingFrom=qr
- Topten 2026 selection: https://www.topten.eu/private/products/dishwashers
- V-ZUG Sprint brochure: https://www.vzug.com/occ/v2/gb-site-spa/pdf/product/brochure/V-ZUG-AS6T6HVI-4117100010.pdf?lang=en_GB&productCode=4117100010
- Fisher & Paykel DD60SDFTX9 guide: https://assets.fisherpaykel.com/product/2883_28c01e3b-bb71-4656-a71a-5aa082ea17aa

Important: the table mixes manufacturer/EPREL product declarations and independent product-selection data. It is a competitive screening set, not a laboratory head-to-head.

## Fast-cycle and rapid-turnaround incumbents

Fast cycles need soil class and service context; published minutes are not interchangeable:

| Product / programme | Published time | Stated use case | Important boundary |
|---|---:|---|---|
| Miele QuickPowerWash (South Africa range) | 58 min | Normally soiled crockery; Miele markets best cleaning results in under one hour | Product-family programme statement; exact model, load, energy and IEC test report required for head-to-head |
| V-ZUG Sprint | 20 min | Lightly soiled pre-dinner glasses | Not suitable for normally or heavily soiled plates/pans; manufacturer guidance |
| V-ZUG Party | 11 min | Repeated rapid turnaround between courses | 20-minute heating phase before repeated loading; five consecutive cycles described by manufacturer. Count preheat, refill, water and total event energy, not only the 11-minute interval |
| Fisher & Paykel DishDrawer Quick modifier | Published as faster wash while maintaining performance | Faster wash modifier on an independently operated drawer | Exact duration, soil rating, consumption and model-specific test evidence must be obtained; do not invent a duration |

| Miele G8 Diamond / AutoVision | Announced; DACH launch April 2027 | Camera + AI load/type recognition; pots intensify wash, bottle interiors optimized, plastics get drying selection | No public matched energy/water/IEC cleaning values yet; future competitor to test, not numeric SOTA row |

Sources:
- Miele South Africa QuickPowerWash: https://www.miele.co.za/domestic/range-of-dishwashers-2498.htm
- V-ZUG Sprint / Party and soil suitability: https://www.vzug.com/be/en/service/device/v-zug-know-how-faq?query=%3Arelevance-desc%3AtabItemType%3AFaq%3AproductCategoryCode%3Adishwasher
- Fisher & Paykel Series 7 Double DishDrawer guide, 17 April 2026: https://assets.fisherpaykel.com/product/2883_f16c0631-b9fe-4395-812f-cb2c5c6ee4bc

### Implication for QuickCell target definition

The previous blanket target of <=4 minutes for a "single cup/plate wash-and-rinse" is too broad to be defensible without a soil qualification. Replace it with two separately benchmarked service classes:

1. **Rapid reuse / lightly soiled glassware**: <=4 minutes ready-state is an aspirational target to test against V-ZUG's 20-minute Sprint and 11-minute Party interval, but compare equal soil severity, temperature, loading and finish. Report cold-start and party/preheat overhead separately. This target is not for normally or heavily soiled plates.
2. **Normally soiled single item**: do not pre-commit to a sub-4-minute claim. First target a matched endpoint materially faster than the 58-minute Miele QuickPowerWash incumbent; measure energy, water and rinse residue. A <=15-minute ready-state target may be used as a stretch hypothesis for the one-item cell, but only if it preserves the defined normal-soil residual threshold.

For party / multi-course use, benchmark an **event service level** rather than only cycle time: time to first released item, median inter-batch refill-to-release time, total items released in 60 / 120 minutes, total energy and fresh water including preheat, plus the fraction meeting the required soil endpoint. V-ZUG's Party sequence shows why an 11-minute run time alone can understate the total service boundary.

## Recent research relevant to the engineering plan

1. **Hydraulic network modelling (Simulation Modelling Practice and Theory, Volume 152, November 2026, Article 103329).** A modular lumped-parameter hydraulic model was compared with numerical and test-bench evidence. The authors report up to 14% deviation in key results, transient pump-ramp deviations below 9.1%, motor-current estimate deviations below 10.7%, and a design modification that raised spray-arm flow by 12% without raising pump power/current. The authors say the data will be made available on request. This supports a measured-pump-curve + branch-loss model as our first digital-twin layer; target better accuracy on an open, reproducible test rig rather than claiming superiority from equations alone.
   - DOI: https://doi.org/10.1016/j.simpat.2026.103329

2. **Jet instability / Helmholtz nozzles (Physics of Fluids 38, 034112, 11 March 2026).** The authors report increased pressure oscillations, jet breakup and splash generation; their oatmeal/oil cleaning experiments and dishwasher testbed showed improved indirect cleaning, particularly in regions where secondary splashes dominate. This is a promising nozzle hypothesis, not evidence that all soils, all dish positions or a complete household product improve.
   - DOI: https://doi.org/10.1063/5.0314894

3. **Soil-specific cleaning kinetics (Chemical Engineering Science 320B, 15 January 2026, Article 122486).** The study fits separate non-linear cleaning models for carbohydrate and protein soils. Amylase is identified as the main carbohydrate cleaning driver; bleaching compounds can reduce amylase effectiveness. Protein removal depends on amylase and protease, with bleaching also changing stain colour. The underlying datasets are confidential. This supports separate, experimentally calibrated soil models and cautions against one generic soil sensor score.
   - DOI: https://doi.org/10.1016/j.ces.2025.122486

4. **Joint detergent and cycle optimization (Computers & Chemical Engineering 213, October 2026, Article 109771).** The abstract reports a model-based framework with a reference-case improvement of 31% shorter washing, 59% shorter drying and 5% higher stain removal. The data are confidential. Treat this as a lead to independently reproduce, not a performance claim for InstantClean.
   - DOI: https://doi.org/10.1016/j.compchemeng.2026.109771

5. **Thermoelectric heating/cooling with thermal energy storage (Applied Thermal Engineering 296, June 2026, Article 130695).** The study reports 8.7% lower total energy, 40% lower internal water-heating energy, 42% shorter drying at the same remaining moisture content and an RC thermal model within ±1 K. The abstract reports 0.952 kWh/cycle in its modified setup. This is a study-specific prototype and not a matched result against the current ~0.39 kWh Eco-label frontier. Reproduce its baseline and measurement boundary before deciding whether thermoelectric storage is useful, and compare it against heat-pump, zeolite/desiccant and passive drying options.
   - DOI: https://doi.org/10.1016/j.applthermaleng.2026.130695
   - ORNL research record: https://impact.ornl.gov/en/publications/thermoelectric-heating-and-coolingintegrated-dishwasher-with-ther/

### Pre-wetting and flexible thermal scheduling incumbent

V-ZUG's AdoraDish V6000 heat-pump product page documents OptiStart: cold-water pre-wetting during the delay period softens soil so the subsequent wash can run 5–8 °C cooler, with claimed savings up to approximately 10%. It also documents load/soil sensors, partial-load operation, automatic airflow/door-opening drying and an 11-minute Party programme. These are model-specific manufacturer claims, but they show that pre-soaking and time-shifting are already commercial strategies.

Official product information: https://www.vzug.com/gb/en/gb/en/products/kitchen/dishwashers/ultra-efficient/adoradish-v6000-with-heat-pump--p4112100010

## Proposed target ladder

These targets are intentionally stricter than broad ENERGY STAR thresholds and aim at the exact-market products above. They are aspirational engineering gates, not predictions or achieved performance.

### Full-load mode

| Metric | First prototype gate | Competitive target | Stretch target |
|---|---:|---:|---:|
| Capacity | 14 place settings under the applicable IEC loading method | 16 place settings | 16 place settings |
| Eco-cycle electricity | <=0.53 kWh | <=0.39 kWh (market-frontier parity) | <=0.35 kWh (energy-leading target) |
| Fresh water | <=6.7 L | <=6.0 L | <=5.5 L |
| Programme duration | <=240 min | <=180 min | <=120 min |
| Cleaning / drying | pass the applicable IEC 60436 assessment | non-inferior to the best tested matched reference | statistically superior on pre-registered soil set without drying regression |
| Noise | measure | <=38 dB(A) | <=36 dB(A) |

Against the exact AS6T-41147 data sheet (14 place settings, 0.53 kWh, 6.7 L, 405 min, 38 dB(A)), the 16-place competitive target would nominally reduce energy by about 26%, water by about 10%, and duration by about 56%. These are arithmetic target comparisons, not achieved results. Against the current Topten Haier listing, 0.39 kWh is energy parity, the 16-place target matches capacity, and 180 minutes would be about 43% shorter than its 315-minute Eco duration. The 0.35 kWh stretch target would be about 10% below the listed 0.39 kWh. None of these calculations proves cleaning or drying superiority. The model's declared data sheet uses its stated EU test basis; formal product claims must be checked using the current IEC 60436:2025 corrected method and matched loading/soil procedures.

### QuickCell (one item or micro-batch)

Do not compare the QuickCell to eco labels. Define a separate, pre-registered per-item protocol with fixed soil, surface, detergent, inlet temperature, final-rinse requirement and measured drying boundary.

| Metric | First engineering gate | Competitive target |
|---|---:|---:|
| Rapid reuse: lightly soiled glassware | <=6 min ready-state | <=4 min ready-state; aspirational vs incumbent 11–20 min rapid programmes |
| Normally soiled single item | Establish matched baseline | <=15 min ready-state stretch hypothesis; no achieved claim |
| Fresh water | <=1.0 L/item | <=0.7 L/item |
| Electricity | <=0.07 kWh/item | <=0.05 kWh/item |
| Cleaning | Meet fixed residual-soil threshold for each declared soil class | Match conventional reference with >=10% lower resource per successful clean |
| Rinse carryover | below assay limit defined by protocol | no worse than reference |
| Cross-contamination | no detectable transfer under negative-control design | no detectable transfer across validated worst case |

Ready-state means preheated/reservoir conditions are separately declared. Cold-start energy and latency must always be reported. Drying is a separate optional operation and is not included in the wash-and-rinse time unless explicitly stated.

### Multi-compartment mode

Benchmark three operating states independently:
- One compartment active.
- Two compartments simultaneously active.
- All compartments active.

Report total and per-item energy/water, programme duration, flow stability, pressure cross-talk, thermal cross-talk, detergent transfer and combined capacity. Use IEC 60436 Annex X where applicable.

## Decision rules

1. Cleaning and rinse/hygiene criteria are hard constraints, not variables that may be traded away for shorter time.
2. Minimize energy, fresh water, duration, noise, chemical dose and lifecycle burden only after the relevant cleaning endpoint passes.
3. Always compare total system energy and water, including standby, warm-up, drying, pumping, water preheating and discarded/flush volumes.
4. Report medians, spread and confidence intervals across independent runs; do not report only the best run.
5. Keep market-label data, internal measurements, simulation outputs and aspirations in separate columns.
6. Freeze protocol, data, firmware, geometry and analysis commits before confirmatory runs.


## Regulatory update checked 2026-11-02

### United States — distinguish ENERGY STAR from federal minimum standards

- ENERGY STAR's currently published Version 7.0 dishwasher criteria remain effective from 19 July 2023: standard models (>=8 place settings + 6 serving pieces) <=240 kWh/year and <=3.2 US gal/cycle; compact models <=155 kWh/year and <=2.0 US gal/cycle. The certified-products dataset was last updated 27 October 2026.
- Separately, US DOE federal minimum conservation standards apply to products manufactured on or after 23 April 2027: standard-size <=223 kWh/year and <=3.3 US gal/cycle; compact <=174 kWh/year and <=3.1 US gal/cycle.
- The DOE 2027 rule states that the standard-size **energy** limit does not apply to normal cycles of 60 minutes or less. Do not read this exception as removing the per-cycle water limit.
- ENERGY STAR's 240 kWh / 3.2 gal qualification and the DOE 2027 federal 223 kWh / 3.3 gal minimum are different programmes and must not be merged.

Sources:
- ENERGY STAR criteria: https://www.energystar.gov/products/dishwashers/key_product_criteria
- ENERGY STAR product list (updated 2026-10-27): https://data.energystar.gov/Active-Specifications/ENERGY-STAR-Certified-Residential-Dishwashers/q8py-6w3f
- DOE consumer dishwasher page: https://www.energy.gov/cmei/buildings/consumer-dishwashers
- DOE compliance date / standard table: https://www.law.cornell.edu/cfr/text/10/430.32
- Federal Register confirmation of 2027 date: https://www.federalregister.gov/documents/2024/10/17/2024-23908/energy-conservation-program-energy-conservation-standards-for-dishwashers

### V-ZUG product-level correction / comparison

The exact V-ZUG AS6T-41147 product data sheet reports 14 place settings, 0.53 kWh and 6.7 L per standard cleaning cycle, 6 h 45 min, cleaning index 1.12, drying index 1.065 and 38 dB(A). This makes it a strong water-use and cleaning/drying reference, but not the lowest-energy model in the current Topten comparison.

Official product data sheet:
https://www.vzug.com/occ/v2/gb-site-spa/pdf/product/euDatasheet/V-ZUG-AS6T6GBHVI-4114700009.pdf?lang=en_GB&productCode=4114700009

V-ZUG AS6T-41173 official label reports 39 kWh/100 cycles, 8.6 L/cycle, 13 place settings and 6 h 10 min. It is a lower-energy benchmark with higher water use and lower place-setting capacity than AS6T-41147:
https://www.vzug.com/medias/sys_master/root/h97/h1b/11172887560222/1228067_1171426-00_AC/1228067-1171426-00-AC.pdf

### 2026 market-frontier interpretation

Topten's current featured-product table lists two separate 0.39 kWh/eco-cycle frontier examples:
- Haier XS 63A4M4B: 16 place settings, 39 kWh/100 cycles, 9.5 L and 315-minute programme.
- V-ZUG AdoraDish V6000 Heatpump: 13 place settings, 39 kWh/100 cycles, 8.6 L and 370-minute programme.

A separate water-use screening report updated 25 September 2026 identifies 6.7 L/eco cycle as the lowest value in its 166-model qualified EU EPREL set, with Bosch SMV6ECX23E and V-ZUG AS6T-41147 at that value. The exact V-ZUG AS6T-41147 official data sheet lists 14 place settings, 0.53 kWh, 6.7 L and 405 minutes. This is the relevant captured low-water frontier; 6.0 L is about 10.4% below 6.7 L. The energy and water leaders are therefore distinct exact models, not one benchmark product.

Do not combine those figures into a fictional single competitor. Both are listed at the same 0.39 kWh/cycle, but the Haier comparison is 16 places / 9.5 L / 315 min and the V-ZUG comparison is 13 places / 8.6 L / 370 min. The 0.39 kWh target is **energy parity with this 2026 market frontier**, not an energy-beating claim. The <=0.35 kWh stretch target is the first target in this ladder that aims to beat the listed 0.39 kWh frontier on energy. The <=6.0 L water target is nominally about 36.8% below the Haier's 9.5 L and 30.2% below the V-ZUG's 8.6 L, but these are target-vs-label arithmetic comparisons, not matched lab results.

Competitive implications:
- Keep capacity at 16 place settings for the competitive and stretch targets if we claim parity with the 16-place Haier reference. The first prototype can target 14 while the loading system is being developed.
- A 0.39 kWh / <=6.0 L / <=180 min 16-place target would match the captured market energy frontier, target lower water than the V-ZUG reference, and aim for a programme materially shorter than the 315-minute Haier and 370-minute V-ZUG programmes, while targeting lower water than both displayed product values. None of those cross-model comparisons establishes matched cleaning/drying superiority.
- A <=0.35 kWh / <=5.5 L / <=120 min target is an energy-leading stretch hypothesis, not an engineering prediction. It requires an efficient hydraulic architecture, low-loss heat management and verified soil/rinse endpoints.
- A 120-180 minute programme is a distinct convenience objective and must retain the applicable IEC cleaning/drying result. A short programme is not equivalent merely because its energy or water is lower.
