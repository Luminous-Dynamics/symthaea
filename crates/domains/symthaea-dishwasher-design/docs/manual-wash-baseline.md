# InstantClean manual dishwashing baseline: one item and micro-batches

Status: research protocol, 3 November 2026. No manual-versus-QuickCell performance result is claimed. Published full-load/manual comparisons do not establish the resource use for washing one cup or one plate; this protocol measures that gap directly.

## Why this baseline matters

A QuickCell for one cup or plate must compete with a person washing that same item in a sink. Comparing the QuickCell only with a full dishwasher's Eco cycle would miss the most important user choice.

Published evidence establishes that full-load manual washing can be much more resource-intensive than a full dishwasher, but it also shows large behavioural variability:

- A 2026 Which? comparison estimates 4.5 L per place setting for handwashing with a bowl, versus 0.696 L per place setting in Eco and 0.926 L in the main programme across a mixed sample of 33 dishwashers. A place setting includes 11 individual pieces (cup, saucer, plate, bowl, glass, cutlery), so this cannot be used as a one-cup estimate.
  - https://www.which.co.uk/news/article/which-research-reveals-how-little-water-dishwashers-use-compared-to-hand-washing-aUDng9Y2iK8E
- A 2022 review compiling participant studies reports manual full-load water use from roughly 34.7–160.1 L in one multi-country study and 18.3–472.8 L across participants in a separate comparison. Results vary substantially with practice. It also lists multiple studies where manual water and energy exceeded machine cycles, but the test boundaries and behaviour differ.
  - https://www.degruyterbrill.com/document/doi/10.1515/tsd-2022-2423/html
- A 2010 UK study of 150 participants found mean manual full-load use of 49 L and 1.7 kWh, compared with 13 L and 1.3 kWh for its 2007 dishwasher reference, under its test conditions. This is useful methodological precedent, not a current-market benchmark.
  - https://onlinelibrary.wiley.com/doi/10.1111/j.1470-6431.2009.00840.x
- A 2026 study of dishwashing pre-treatment behaviors in five European countries reports 6.8–24.4 L across different pre-treatment methods in its central-location testing. Pre-treatment water must not be omitted from total-use accounting.
  - https://link.springer.com/article/10.1007/s43832-026-00403-5

## Questions

1. At what item count does QuickCell use less water and energy than a person's normal sink method?
2. At what item count does a basin-based method become more efficient than running-tap washing?
3. Does QuickCell materially reduce active human time, or only unattended elapsed time?
4. Does quick washing retain the same residual-soil and rinse-residue endpoint?
5. How do results change with electric geyser, gas, heat-pump or solar preheated water?
6. Is the crossover different for a lightly soiled glass, beverage-stained mug and normally soiled plate?
7. Does pre-soaking improve cleaning enough to justify its additional time/water/energy?

## Controlled item set

Use identical item models and surface materials for all methods:
- glass tumbler;
- ceramic mug with handle and internal base;
- flat ceramic plate;
- stainless-steel cutlery item or standardized cutlery bundle.

Soils are separate:
- lightly soiled beverage glass (standardized tea/coffee residue);
- starch/carbohydrate;
- protein;
- fat/oil;
- mixed soil only after single-soil comparisons.

For each soil, record application mass, surface area/geometric pattern, drying/aging time, and the independent cleaning acceptance assay. Do not describe a visually rinsed item as clean without endpoint measurement.

## Manual methods to compare

A. **Running-tap wash**
- Pre-wet, apply a defined quantity of dishwashing liquid, scrub with a standardized sponge/brush procedure, rinse under running water, drain.
- Record water flow continuously or meter volume at the inlet.
- Record water temperature throughout; include upstream hot-water energy and first-draw cold-water wastage.

B. **Single-basin wash + separate rinse**
- Use a defined basin volume at a measured temperature.
- Apply a defined soap mass and standard scrubbing procedure.
- Rinse in a separate measured water stream or second basin.
- Record initial basin fill, top-ups, rinse volume, and disposal.

C. **Two-basin wash/rinse**
- One basin for wash, one for rinse; quantify each independently.

D. **Participant's habitual method**
- Record participant's normal practice without coaching on efficiency.
- Capture tap-on time, water, soap and actual active time.
- Use this as an ecological comparator, not a controlled substitute for A–C.

E. **Pre-soak then wash**
- Separate pre-soak time, volume, temperature and energy.
- Compare with no pre-soak under identical cleaning endpoint.

## Machine comparators

1. InstantClean QuickCell at declared ready-state and cold-start states.
2. A current drawer/compact dishwasher's one-item or Quick programme.
3. A full-size incumbent using its manufacturer-supported Quick/Auto/Half-load option.
4. If relevant, the full-size machine's Eco cycle as a separate full-load context, not a one-item head-to-head.

Record exact SKU, region, firmware, programme, options, item count, preheat state and published rating. Never compare a QuickCell one-item result directly against a label that represents an Eco programme and full load without clearly stating the mismatch.

## Measurement boundary

Per item and per successful clean, capture:

- Total cold and hot water volume, including pre-rinse, pre-soak, wash, rinse, tap flushing and refill.
- Actual hot-water energy supplied. Prefer an inline thermal energy meter; otherwise measure inlet/outlet mass and temperature with a declared calibrated method. Record the heating source and efficiency boundary. Do not estimate with a generic efficiency factor without reporting it.
- Electricity for QuickCell/machine operation, including warm-up, standby during the declared duty window, pump, heater, controls and drying.
- Soap/detergent mass, product and batch.
- Active human time (hands-on) separately from elapsed time and machine unattended time.
- Cleaning residual, rinse residue, rewash, drying endpoint and material damage.
- Wastewater volume, temperature and detergent residue where feasible.
- Water hardness at the point of use, ambient temperature and humidity.
- Instrument calibration IDs, raw meter files, video only where consented, protocol/version hashes and run IDs.

Resource per successful clean must include failed attempts and rewash cycles. Do not discard failed manual or machine runs. Keep cleaning, rinse, drying, hygiene and sanitization as distinct outcomes.

## Experimental design

- Use a crossover design: each operator performs each manual method across the same item/soil conditions, with randomized order and counterbalanced method sequence.
- Use several operators with a range of habitual practices; report participant-level spread, not only the pooled mean.
- Standardize sponge/brush type, soap dose and replacement/cleaning interval for controlled methods.
- For habitual method, do not alter behavior; record the operator's actual actions.
- Block by day, water source and soil batch. Hold out whole participants and days for confirmatory analysis if modelled.
- Pilot at least five independent cycles/runs per condition to estimate variance; this is not a final powered sample size. Calculate confirmatory size from preregistered effect size, variance and test family.
- Randomize machine/manual run order to avoid systematically favoring warmed machines or fresher soil.
- Use independent cleaning/rinse ground truth and negative controls.

## Johannesburg / South African water and energy context

Use the Johannesburg hardness matrix already defined in the benchmark ledger: controlled levels 20, 50, 75, 100, 150 and 200 mg/L as CaCO3 plus water actually sampled at the appliance inlet.

For upstream heating, test at least:
- electric resistance geyser;
- gas water heater where available;
- heat-pump geyser where available;
- solar preheated water plus backup where available.

Measure real hot-water energy at the system boundary. Report the household heater separately and a combined appliance-plus-heater total. Tariff/cost comparisons must use current local tariffs and be reported separately from physical kWh/L results.

## Decision metrics

Report Pareto fronts, not one opaque weighted score:
- L fresh water per verified clean;
- kWh total upstream + appliance energy per verified clean;
- grams of soap/chemical per verified clean;
- median/p95 elapsed time;
- median/p95 active human time;
- rewash / endpoint failure rate;
- drying result and time;
- noise and reliability for machine options.

Key crossover plots:
- one item, two items, four items, six items;
- separate glass/cup/plate;
- running tap versus each basin method;
- cold-start versus ready-state QuickCell;
- water heating source;
- cost per successful clean using an explicitly named tariff scenario.

## Proposed QuickCell go/no-go criterion

QuickCell should demonstrate at least one meaningful advantage over the best matched manual method at identical cleaning/rinse endpoints:
- >=20% lower combined water + upstream/appliance energy on a separately reported Pareto dimension, or
- >=50% lower active human time without increasing total energy/water unacceptably,
while preserving cleaning, rinse and material-compatibility hard gates.

These are screening goals, not expected or achieved results. If manual washing is better for one item, QuickCell should honestly route that use case to manual washing or improve the cell; do not force an appliance usage claim.

## Immediate deliverables

1. Manual-wash operator script and data sheet.
2. Metered running-tap and basin rigs.
3. Matched one-item QuickCell/manual/Drawer trials.
4. Resource and labor crossover report at 1/2/4/6 items.
5. Independent cleaning and rinse endpoint evidence, all failed runs retained.

Related issue: #10228
