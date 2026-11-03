# InstantClean grouped generalization and independent confirmation benchmark

Status: research protocol draft, 3 November 2026. This document defines a proposed experiment and analysis plan. It is not a certification protocol, a claim of product performance, or a replacement for licensed standards.

## 1. Why this is the next bottleneck

The existing `quality::qualify_pilot_evidence` contract is intentionally narrow: it qualifies a fixed machine/programme/soil/material/item-set/batch/environment/protocol scope from independent item-level outcome assays. The current five-run floor is a pilot eligibility check only.

The next scientific risk is **generalization leakage**. A model can appear accurate when dishes, images, sensor windows or repeated cycles from the same physical run, machine, day or reagent lot appear in both development and test data. A large number of correlated records is not a large number of independent experiments.

The next qualification tier must answer a harder question:

> Does the observer and process policy retain its endpoint guarantees on new physical runs drawn from genuinely new dates, units, soil preparations, detergent/water lots and operating conditions—without silently expanding its authorized envelope?

No random row-level train/test split is acceptable for a release-critical claim.

## 2. Paradigm shift: from model accuracy to bounded risk under distribution shift

The object being qualified is not a classifier in isolation. It is the complete chain:

1. physical process and its operating envelope;
2. sensors and their calibration / health state;
3. observer version and uncertainty estimate;
4. deterministic supervisor and fallback policy;
5. independent endpoint assay;
6. item-release decision and any rewash / reject outcome.

A model score does not qualify the chain. Measure the rate at which the complete chain falsely releases an item or load that fails an independent endpoint assay.

Keep six outcomes separate:

- residual soil on each declared item/surface;
- final-rinse chemical residue / carryover;
- cross-contamination to initially clean witness surfaces;
- residual moisture / drying;
- relative microbial reduction, where explicitly studied;
- sanitization/disinfection, which requires its own applicable standard and certification.

A successful result on one endpoint never substitutes for another. IEC TS 63331:2024 supports relative microbiological efficacy statements and expressly does not address sanitization, disinfection or sterilization. NSF/ANSI 184 is a separate residential sanitization path.

## 3. Independent experimental units and leakage controls

### 3.1 Define the unit before collecting data

- **Physical wash run:** one complete appliance cycle / operation under one declared load, programme and process history. This is the primary independent trial for run-level false-release analysis.
- **Item observation:** one item × endpoint × run. It is nested in the physical run.
- **Image, crop, sensor frame, sample aliquot or repeated assay reading:** a measurement, not a new independent physical trial.
- **Test block:** a predeclared group sharing a date/session, machine unit, soil preparation, chemistry lot, water lot or operator, according to the factor being held out.

For a run-level release decision, define failure conservatively: if any required item or endpoint fails its pre-registered acceptance criterion, the run is a false release if the observer released it. Continue to publish item-level results; the run-level aggregation must not hide a weak item or surface.

### 3.2 Split by groups, never by rows

All data derived from one physical run stay in one partition: images, crops, sensor windows, assay replicates, process logs and item observations.

Use pre-registered grouped partitions:

| Partition | Purpose | Permitted use |
|---|---|---|
| Development | Fit candidate physical/observer models and select features | Training and exploratory model selection |
| Calibration | Calibrate probabilities/intervals and choose a release threshold | Threshold selection only; no final performance claim |
| Locked confirmation | Evaluate the frozen observer, threshold and supervisor | One-time confirmatory result |
| External replication | Independent operator/lab and new unit or site | Test transfer beyond the development environment |

For each generalization axis, hold out complete groups:
- physical run IDs;
- test day/session IDs;
- appliance unit serial IDs;
- soil preparation batch IDs;
- detergent/formulation batch IDs;
- water batch/source IDs;
- operator and, where feasible, assay laboratory IDs.

Do not place one member of a held-out group in training and another in confirmation. Do not tune after opening the locked-confirmation outcomes. If the protocol or threshold changes after unblinding, create a new version and a new confirmation set.

A single split cannot prove every type of generalization. Run distinct, pre-registered transfer tests for new days, new soil batches, new detergent/water lots, new units and new operators. Then run a harder external test that combines several shifts. Record which axes were actually held out; do not describe a same-unit/new-day test as cross-unit validation.

### 3.3 Model and threshold development

- Use grouped nested cross-validation inside the development set when selecting model families or hyperparameters.
- Use calibration data that are group-disjoint from development to estimate probability calibration, residual distributions, interval width and abstention thresholds.
- Freeze model weights, feature definitions, sensor calibration IDs, software commit, runtime policy, endpoint criteria and decision threshold before confirmation.
- Report all tried model families and threshold-selection steps. Do not present the best of many unreported tests as a confirmatory result.
- Keep the locked confirmation archive inaccessible to model developers until the frozen evaluation is run.
- Preserve rejected, failed, rew ash, aborted and missing-data runs in the denominator and raw archive. Exclusion is permitted only for a predeclared protocol deviation with a documented reason and independent adjudication; publish sensitivity analysis with all runs retained.

## 4. Experimental design

### 4.1 Discovery and response-surface phase

Use a staged design, not an enormous full factorial:

1. **Method qualification:** quantify repeatability, reproducibility, blank response, detection limit, recovery and expanded uncertainty for each independent assay. Establish whether fluorescence, gravimetry, TOC/targeted chemistry or witness-coupon assays are fit for the exact endpoint.
2. **Screening:** blocked fractional-factorial or space-filling design to identify important interactions among soil, material, geometry, soil age, water hardness, temperature, detergent formulation/dose, mechanical action and cycle phase. Randomize order within practical blocks.
3. **Mechanism and response surface:** investigate selected interactions with richer nonlinear models. Keep carbohydrate, protein, fat/oil and beverage/tannin soils as separate response families until evidence supports a combined model.
4. **Locked confirmation:** compare preselected candidates under a frozen protocol and held-out groups. Do not use confirmation results to select the model that is then reported as validated.

Use matched test blocks when comparing architectures or competitor appliances: same soil preparation, soil age, item geometry/material, detergent class, water hardness, ambient condition and endpoint assay. Randomize appliance/programme order within the block and blind the endpoint assessor to the selected design where practical.

### 4.2 Core controlled factors

Record exact values and lot identities, not labels such as “soft water” or “normal soil”.

- Soil family, recipe/reference, deposited mass, application method, preparation lot and age at cycle start.
- Surface material, finish, geometry, location/orientation, fixture and load-set revision.
- Detergent brand/formulation/version/lot, component masses commanded and independently measured, and release phase.
- Water source/batch, hardness as mg/L CaCO3, conductivity, pH, inlet temperature and any softener/filter treatment.
- Ambient temperature and relative humidity.
- Machine model, individual unit serial, programme/software build, pump/nozzle/arm/filter identifiers and maintenance state.
- Branch pressure/flow, pump speed/current/real power, phase temperature, real wall energy, fresh water, purge/flush volume, recirculated volume and elapsed time.
- Operator, test session, randomized block and blinded assay code.

For the Johannesburg prototype, use the existing hardness matrix of 20, 50, 75, 100, 150 and 200 mg/L as CaCO3, plus direct samples at the appliance inlet. The Rand Water municipal mean is not a household inlet measurement.

### 4.3 Negative and adversarial controls

Every observer/supervisor evaluation must include:
- known-clean negative controls and known-soiled positive controls close to the acceptance boundary;
- clean witness surfaces co-loaded with heavily soiled items to expose cross-transfer;
- occluded surfaces, mug interiors, rims, bottle interiors and narrow recesses;
- blocked/partially blocked nozzle, restricted filter, dirty sump and reduced pump flow;
- low, high and mistimed detergent release; incorrect formulation or incompatible phase;
- failed, disconnected, frozen, stale, biased and drifting sensors;
- sensor disagreement (e.g. clear-looking recirculated water but failing surface coupon);
- out-of-envelope water hardness, soil age, temperature, material and geometry;
- interrupted power, incomplete stage, leak/overflow and incomplete final-rinse challenges.

The test must demonstrate that an invalid or contradictory observation cannot increase release authority. It must trigger a validated conservative process or abstention.

## 5. Endpoint reference methods and observer metrics

### 5.1 Reference methods

- Soil removal: standardized coupons and independent residual-mass or calibrated chemical/optical assay. Validate 365-nm fluorescence only for each soil/material combination where recovery and specificity are established.
- Rinse carryover: conductivity/pH are useful process observations, but pair them with formulation-specific residue chemistry (e.g. TOC or a validated marker assay) for ground truth.
- Cross-contamination: clean witness surfaces with a traceable marker, quantified after the run.
- Drying: item-specific gravimetric or other validated residual-moisture measure after a fixed conditioning period.
- Microbiology: separate protocol and laboratory; do not translate relative reduction into a sanitization statement.

### 5.2 Report the full error and uncertainty profile

For each endpoint × soil × material × geometry × shift stratum, report:
- false-pass / false-release count and denominator;
- false-fail count and denominator;
- sensitivity, specificity and their confidence intervals;
- indeterminate / abstain rate;
- risk-coverage curve (false-release risk as a function of fraction of runs automatically released);
- calibration curve, Brier score or log loss for probabilistic outputs;
- prediction-interval empirical coverage and interval width;
- detection latency for faults and quality degradation;
- fallback and rewash frequency;
- resource burden of conservative fallback and resources spent on every failed/rewashed run.

Do not rely on aggregate accuracy, AUROC, ECE or a macro-average alone. Rare false-success events need their own count and bound. Publish the worst-performing declared stratum and the entire confusion matrix.

### 5.3 Statistical rules

The current exact zero-failure helper calculates a one-sided Clopper–Pearson bound under an IID Bernoulli assumption:
- 299 independent zero-failure trials give an upper bound below 1% at one-sided 95% confidence;
- 2,995 independent zero-failure trials give an upper bound below 0.1%;
- with 20 simultaneous strata and simple Bonferroni allocation for family-wise 95% confidence, the corresponding 0.1% per-stratum zero-event floor is approximately 5,989 trials per stratum.

These numbers are **not automatically valid** if runs share a soil lot, detergent lot, day, appliance unit or operator in a way that creates cluster correlation. In that case, plan and analyse the cluster structure explicitly (e.g. hierarchical/mixed-effects models and cluster-aware bootstrap or exact cluster-level bounds). More repeated cycles from one batch do not substitute for independent batches. Estimate intracluster correlation during discovery and size confirmation using the number of independent groups as well as runs per group.

A single cycle containing many items is one run-level trial for the false-release bound. Item/image/frame counts must not inflate the effective sample size. Pre-register the confidence level, error event, grouping, multiplicity correction, missing-data rule and stopping rule before confirmation.

## 6. Comparator design and current performance frontier

### 6.1 FullLoad

For the household full-load claim:
- use the licensed IEC 60436:2025 corrected 2026 text and its applicable reference machine, detergent, load/handling, repeatability and multi-compartment methods;
- report exact SKU, market, programme, rated capacity, measured capacity/load, energy boundary, water boundary, eco duration, cleaning and drying results, uncertainty and number of units;
- separately evaluate within-model unit-to-unit variation and service/repair requirements;
- do not infer compliance or safety from an IEC performance result. Review IEC 60335-2-5 and applicable national requirements separately.

A current September 2026 exact-model market screen reports:
- lowest captured EU eco energy: V-ZUG AS6T-41173 at 39 kWh/100 cycles (0.39 kWh/eco cycle);
- lowest captured EU eco water: Bosch SMV6ECX23E and V-ZUG AS6T-41147 at 6.7 L/eco cycle.

These are separate exact-model frontiers and the screen is not a complete market census. Treat them as acquisition targets for independent replication, not as a single best-in-class product. Preserve the difference between EPREL/IEC eco values and DOE/ENERGY STAR annual metrics.

**InstantClean research target ladder (not achieved):**

| Metric | Competitive target | Stretch target |
|---|---:|---:|
| FullLoad energy, matched IEC basis | <=0.39 kWh/cycle | <=0.35 kWh/cycle |
| FullLoad fresh water, matched IEC basis | <=6.0 L/cycle | <=5.5 L/cycle |
| FullLoad programme duration | <=150 min | <=120 min |
| FullLoad airborne noise | <=38 dBA | <=36 dBA |

The 6.0 L target is about 10.4% below the current 6.7 L captured water frontier. Energy parity at 0.39 kWh is not superiority; 0.35 kWh is the energy-leading stretch hypothesis. Capacity and exact test basis must be matched.

### 6.2 QuickCell

A one-item cell is not a miniature full-load dishwasher benchmark. It needs a dedicated protocol and matched alternatives:
- best available independently operated drawer / compact-dishwasher Quick or Auto mode;
- best relevant mainstream machine's manufacturer-supported quick/auto/half-load mode;
- manual washing of the same single item, measured at point of use (including tap flow, hot-water energy, pre-rinse and operator time).

Do not extrapolate a full-place-setting handwashing study to one cup. Freeze exact item geometry, soil, soil age, initial temperature, detergent, water hardness and acceptance criteria. No pre-rinse should be allowed unless the same action is included and measured for all comparators.

Separate service classes:
- lightly soiled glassware / rapid reuse: <=4 min ready-state is an aspirational target only;
- normally soiled single item: <=15 min is a stretch hypothesis, not a product claim, and must beat matched incumbent modes at non-inferior soil/rinse/cross-transfer outcomes.

Initial QuickCell target ladder, not achieved:
- competitive hypothesis: <=4 min for lightly soiled glassware, <=0.7 L and <=0.05 kWh per successful item;
- normally soiled item: <=15 min stretch hypothesis with independently passed endpoints;
- always report cold-start, warm/ready-state, preheat, purge and idle/standby overhead separately.

For multi-course/event use, report time-to-first-released item, items released in 60 and 120 minutes, median and p95 inter-release time, successful-clean yield, energy, fresh water, detergent and user interventions. A short isolated programme time is not event throughput.

### 6.3 Don't let incomparable metrics masquerade as a win

- EU eco kWh/100 cycles and US kWh/year are different reporting systems.
- Labelled eco cycle and rapid programme are different services.
- Full-load per-cycle and one-item per-item values are different denominators.
- Ready-state and cold-start are different boundaries.
- A failed/rewashed item is not a successful clean; count all resources spent before release.
- Compare physical outcomes first, then Pareto vectors. Do not collapse safety or quality gates into a weighted score.

## 7. Paradigm-shift hypotheses worth testing

1. **Outcome-qualified release, not timer completion.** The controller estimates whether the declared endpoint has been achieved, carries calibrated uncertainty, and abstains whenever evidence is weak.
2. **Quality-separated fluid topology.** Physically separate soil-rich wash return from clean final-rinse supply; instrument purge and carryover boundaries. Prove isolation with tracers under valve faults, dirty filters and simultaneous compartments.
3. **One platform, multiple service cells.** QuickCell and FullLoad share only those utilities whose measured contamination, thermal and hydraulic coupling remain acceptable. Select architecture by a physical topology bake-off, not by aesthetic preference.
4. **Soil-specific chemistry and mechanics.** Optimize phase, temperature, enzyme compatibility, dose mass and mechanical action jointly for each soil family; avoid the false assumption that more detergent, more heat or more pressure always improves removal.
5. **Adaptive hydraulics from deposited energy, not nominal flow.** Measure spatial wetting and impact on witness surfaces; compare steady branch models with transient pump/arm/jet measurements. Test promising unstable-jet/splash hypotheses against conventional nozzles under identical loads and soils.
6. **Thermal cascading.** Reuse heat between drain/pre-rinse, wash, incoming water and drying only where hygiene, carryover and total wall energy remain validated; compare with heat-pump, desiccant/zeolite and auto-open/hot-air incumbents.
7. **Fault-aware low-complexity sensing.** Fuse a small number of independently calibrated signals; add sensors only when their information gain improves held-out false-release bounds or fault detection enough to justify cost, calibration and service burden.
8. **Repairable and interoperable by design.** Make filters, spray arms, seals, dosing channels, sensors and pumps serviceable; keep manual fallback and privacy/offline operation. Total lifecycle cost and repair time belong beside water/energy in the benchmark.

These are falsifiable engineering hypotheses, not claims that the architecture has already achieved them.

## 8. Immediate execution sequence

1. Acquire at least one efficient full-size reference, one independently operated drawer/compact reference and the candidate prototype / hydraulic skid.
2. Freeze the common soil recipes, coupon geometry, fixtures, assay methods, acceptance criteria, water sampling and run randomization.
3. Validate the independent assays and measurement uncertainty before training any observer.
4. Run a randomized blocked discovery matrix and retain every run, including failures.
5. Create grouped development/calibration/locked-confirmation partitions before modelling.
6. Train and calibrate candidate observers only on the allowed partitions; compare simple physical baselines before complex ML.
7. Freeze the observer and deterministic supervisor, then run held-out fault/shift challenges in shadow mode.
8. Compare topology and scheduler alternatives using verified-successful item throughput and total resource vectors.
9. Publish the raw manifest, analysis script, pre-registration, exact software SHA, instrument calibration and every failed/aborted run. A digest identifies a file; it does not authenticate its scientific validity.
10. Only after independent confirmation consider an actual early-stop trial. Until then, no observer may shorten a qualified physical process.

## 9. Sources and status boundary

- IEC 60436:2025, corrected 2026-04: https://webstore.iec.ch/en/publication/80459
- ENERGY STAR dishwasher criteria: https://www.energystar.gov/products/dishwashers/key_product_criteria
- EU dishwasher label / ecodesign: https://energy-efficient-products.ec.europa.eu/product-list/dishwashers_en
- Exact-model EU eco energy screen (updated 25 September 2026): https://appliancebench.com/en-gb/reports/dishwasher-energy
- Exact-model EU eco water screen (updated 25 September 2026): https://appliancebench.com/en-gb/reports/dishwasher-water
- 2026 oatmeal fluorescence study: https://doi.org/10.1016/j.foodres.2026.118853
- Soil-specific carbohydrate/protein cleaning kinetics: https://doi.org/10.1016/j.ces.2025.122486
- IEC TS 63331:2024 relative microbiological efficacy: https://webstore.iec.ch/en/publication/66647
- NSF dishwasher sanitization overview: https://www.nsf.org/consumer-resources/articles/dishwasher-certification

No performance, reliability, cleaning, rinse, drying, microbiological, sanitization or superiority result is claimed by this document.
