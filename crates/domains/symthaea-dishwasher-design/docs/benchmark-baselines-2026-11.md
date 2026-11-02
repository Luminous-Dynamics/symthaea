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
| Capacity | 14 place settings under the applicable IEC loading method | 14 place settings | 14 place settings |
| Eco-cycle electricity | establish baseline, then <=0.46 kWh | <=0.39 kWh | <=0.35 kWh |
| Fresh water | establish baseline | <=7.0 L | <=6.0 L |
| Programme duration | establish baseline | <=150 min | <=120 min |
| Cleaning / drying | pass the applicable IEC 60436 assessment | non-inferior to the best tested matched reference | statistically superior on pre-registered soil set without drying regression |
| Noise | measure | <=40 dB(A) | <=38 dB(A) |

The competitive target would, if achieved under matched conditions, improve on the AS6T-41171 reference by about 15% in electricity, 22% in water and 14% in duration. It would match the 0.39 kWh/eco-cycle energy figure seen in the 13-place heat-pump model while targeting more capacity, less water and a shorter programme. This comparison is a target calculation, not proof that the different tests are directly interchangeable.

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
