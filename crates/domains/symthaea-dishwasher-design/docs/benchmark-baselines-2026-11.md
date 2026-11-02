# InstantClean benchmark baselines and target ladder (2026-11-02)

This is a research ledger, not a certified comparison. Labelled eco-programme data are not interchangeable with short-cycle data. Always retain exact model, market, test standard, programme, load and measurement boundary.

## Standards and policy

### Household performance

- IEC 60436:2025, corrected version 2026-04, is the current fifth-edition household dishwasher performance-measurement method found in this review.
- The fifth edition introduces a new reference machine and reference detergent type E; updates cleaning/drying and low-power measurement; and adds a multi-compartment testing method (Annex X).
- IEC 60436 defines measurement methods. It does not itself set minimum performance requirements or establish product safety.
- The complete standard is a paid 289-page publication. Obtain the licensed copy and follow the actual clauses before claiming IEC-conforming results. The public product page is not a substitute.

Official reference: https://webstore.iec.ch/en/publication/80459

### United States qualification

ENERGY STAR's current published standard-dishwasher criteria remain <=240 kWh/year and <=3.2 US gallons/cycle; compact criteria are <=155 kWh/year and <=2.0 gallons/cycle. This is a qualification ceiling, not the best-product frontier, and US annual metrics must not be mixed directly with EU per-eco-cycle labels.

Official reference: https://www.energystar.gov/products/dishwashers/key_product_criteria

### EU ecodesign and label context

The EU label reports energy per 100 eco cycles, eco-programme duration, water per eco cycle, capacity and noise. The Commission states that an efficient machine can wash up to 15 place settings using about 9 L, versus about 20 L for handwashing the same crockery volume. That is broad context, not a per-mug comparison.

Official reference: https://energy-efficient-products.ec.europa.eu/product-list/dishwashers_en

EU ecodesign rules also require access to certain spare parts (including hinges/seals, spray arms, filters and racks) for 10 years after a model leaves the market, and relevant professional repair information/parts for at least 7 years. Repairability is therefore a competitive requirement as well as a product principle.

### South Africa

Government Notice 7859, published 28 August 2026 in Gazette 55257, amends VC9008 and states phase 1 and phase 2 effective 12 and 24 months after publication. That implies 28 August 2027 and 28 August 2028 respectively. The public notice page reviewed here does not expose the complete schedule; obtain the full gazette and have NRCS/SANS applicability reviewed before product design freeze or market placement. Do not infer appliance-specific thresholds from the notice headline alone.

Notice record: https://www.acts.co.za/national_/n7859_notice_no__7859_of_2026

## Exact-model market reference points

| Exact model / reference | Capacity | Labelled eco energy | Water per cycle | Eco duration | Evidence notes |
|---|---:|---:|---:|---:|---|
| V-ZUG AdoraDish V6000 heat-pump, AS6T-41173 | 13 place settings | 0.394 kWh/cycle (39 kWh/100) | 8.6 L | 370 min (Topten listing) | Official V-ZUG product information and independent Topten listing |
| V-ZUG AdoraDish V6000, AS6T-41171 | 14 place settings | 0.46 kWh/cycle | 9.0 L | 175 min | Official V-ZUG data sheet; 38 dB also listed |
| AEG GI9210X2TF (911474011) | 14 place settings | 0.488 kWh/cycle (49 kWh/100) | 8.4 L | 310 min | EPREL product record; cleaning index 1.121, drying index 1.061 |
| Haier XS 63A4M4B | 16 place settings | 0.39 kWh/cycle (39 kWh/100) | Not captured in this record | 315 min | Topten listing; do not infer water performance |
| V-ZUG AS6T6HVI / AS6T-41171 Sprint programme | 13 settings on brochure | Not separately established for Sprint | Standard-cycle specification lists 6.5 L | Sprint duration 20 min | The 20-minute programme is real; do not attribute the standard-cycle kWh/L figures to Sprint without programme-specific test data |
| Fisher & Paykel Series 9 Single DishDrawer DD60SDFTX9 | 7 place settings | 136 kWh/year (different rating basis) | 6.9 L/wash | Quick modifier; duration not in captured data | Independent-drawer architecture benchmark; annual rating is not directly comparable with EU per-cycle eco data |

Sources:
- V-ZUG AS6T-41173: https://www.vzug.com/fr/en/energy-labels-dishwashers
- V-ZUG AS6T-41171 data sheet: https://www.vzug.com/occ/v2/au-site-spa/pdf/product/euDatasheet/V-ZUG-AS6T6HVI-4117100007.pdf?lang=en_AU&productCode=4117100007
- AEG EPREL: https://eprel.ec.europa.eu/screen/product/dishwashers2019/2400252?navigatingFrom=qr
- Topten 2026 selection: https://www.topten.eu/private/products/dishwashers
- V-ZUG Sprint brochure: https://www.vzug.com/occ/v2/gb-site-spa/pdf/product/brochure/V-ZUG-AS6T6HVI-4117100010.pdf?lang=en_GB&productCode=4117100010
- Fisher & Paykel DD60SDFTX9 guide: https://assets.fisherpaykel.com/product/2883_28c01e3b-bb71-4656-a71a-5aa082ea17aa

Important: the table mixes manufacturer/EPREL product declarations and independent product-selection data. It is a competitive screening set, not a laboratory head-to-head.

## Recent research relevant to the engineering plan

1. **Hydraulic network modelling (Simulation Modelling Practice and Theory, Volume 152, November 2026, Article 103329).** A modular lumped-parameter hydraulic model was compared with numerical and test-bench evidence. The authors report up to 14% deviation in key results, transient pump-ramp deviations below 9.1%, motor-current estimate deviations below 10.7%, and a design modification that raised spray-arm flow by 12% without raising pump power/current. The authors say the data will be made available on request. This supports a measured-pump-curve + branch-loss model as our first digital-twin layer; target better accuracy on an open, reproducible test rig rather than claiming superiority from equations alone.
   - DOI: https://doi.org/10.1016/j.simpat.2026.103329

2. **Jet instability / Helmholtz nozzles (Physics of Fluids 38, 034112, 11 March 2026).** The authors report increased pressure oscillations, jet breakup and splash generation; their oatmeal/oil cleaning experiments and dishwasher testbed showed improved indirect cleaning, particularly in regions where secondary splashes dominate. This is a promising nozzle hypothesis, not evidence that all soils, all dish positions or a complete household product improve.
   - DOI: https://doi.org/10.1063/5.0314894

3. **Soil-specific cleaning kinetics (Chemical Engineering Science 320B, 15 January 2026, Article 122486).** The study fits separate non-linear cleaning models for carbohydrate and protein soils. Amylase is identified as the main carbohydrate cleaning driver; bleaching compounds can reduce amylase effectiveness. Protein removal depends on amylase and protease, with bleaching also changing stain colour. The underlying datasets are confidential. This supports separate, experimentally calibrated soil models and cautions against one generic soil sensor score.
   - DOI: https://doi.org/10.1016/j.ces.2025.122486

4. **Joint detergent and cycle optimization (Computers & Chemical Engineering 213, October 2026, Article 109771).** The abstract reports a model-based framework with a reference-case improvement of 31% shorter washing, 59% shorter drying and 5% higher stain removal. The data are confidential. Treat this as a lead to independently reproduce, not a performance claim for InstantClean.
   - DOI: https://doi.org/10.1016/j.compchemeng.2026.109771

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
| Single cup/plate wash-and-rinse | <=6 min ready-state | <=4 min ready-state |
| Fresh water | <=1.0 L/item | <=0.7 L/item |
| Electricity | <=0.07 kWh/item | <=0.05 kWh/item |
| Cleaning | meet fixed residual-soil threshold | match conventional reference with >=10% lower resource per successful clean |
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

Topten's current selection (last updated 22 May 2026) lists two separate 0.39 kWh/eco-cycle frontier examples:
- Haier XS 63A4M4B: 16 place settings, 39 kWh/100 cycles and 315-minute programme. The captured Topten listing does not provide its water value.
- V-ZUG AdoraDish V6000 Heatpump: 13 place settings, 39 kWh/100 cycles and 370-minute programme. The exact AS6T-41173 label separately reports 8.6 L/cycle.

Do not combine those figures into a fictional single competitor. The Haier is the current capacity/energy/time comparison; the V-ZUG is a separate low-energy and documented-water comparison. The 0.39 kWh target is **energy parity with this 2026 market frontier**, not an energy-beating claim. The <=0.35 kWh stretch target is the first target in this ladder that aims to beat the listed 0.39 kWh frontier on energy. The <=6.0 L water target is below the V-ZUG's 8.6 L figure, but must not be described as beating the Haier on water until its exact-model declaration or a matched test is obtained.

Competitive implications:
- Keep capacity at 16 place settings for the competitive and stretch targets if we claim parity with the 16-place Haier reference. The first prototype can target 14 while the loading system is being developed.
- A 0.39 kWh / <=6.0 L / <=180 min 16-place target would match the captured market energy frontier, target lower water than the V-ZUG reference, and aim for a programme materially shorter than the 315-minute Haier and 370-minute V-ZUG programmes. None of those cross-model comparisons establishes matched cleaning/drying superiority.
- A <=0.35 kWh / <=5.5 L / <=120 min target is an energy-leading stretch hypothesis, not an engineering prediction. It requires an efficient hydraulic architecture, low-loss heat management and verified soil/rinse endpoints.
- A 120-180 minute programme is a distinct convenience objective and must retain the applicable IEC cleaning/drying result. A short programme is not equivalent merely because its energy or water is lower.
