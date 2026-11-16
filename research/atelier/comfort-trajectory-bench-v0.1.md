# Atelier Comfort Trajectory Bench v0.1

**Status:** research protocol proposal; not a validated scale, product result, or industry standard  
**Owner:** EverydayWearBench [#12157](https://github.com/Luminous-Dynamics/symthaea/issues/12157)  
**Applies to:** AO-01 Everyday Overshirt and CR-01 Continuum Wrap  
**Date:** 2026-11-15

## Machine-readable session contract

The versioned session schema and synthetic examples live in [the schema/fixture directory](comfort-trajectory-bench-v0.1/README.md). The Draft 2020-12 schema fixes the 0/15/30/60/90/120-minute schedule, dimension-specific 1–7 ratings, consent, pseudonymous participant identity, exact garment/pattern/BOM/assembly/configuration identities and material lot. It explicitly represents missing observations and observations not collected after a voluntary stop.

The fixture set includes complete AO-01, early-stop CR-01 and explicit-missing positive examples, plus negative cases for a transposed observation schedule, an out-of-range rating and missing consent. JSON Schema checks record shape; additional cross-field semantic checks are listed in the directory README and remain a separate validator implementation task.

## 1. Purpose and boundary

This document extends EverydayWearBench with a time-resolved, wearer-led evaluation of comfort. It does not create a competing benchmark, replace the existing textile-test programme, or define a universal comfort score.

The research question is:

> Does the garment remain acceptable to its wearer as time, posture, activity, temperature and moisture conditions change?

A five-minute fitting cannot answer that question. Equally, one pooled score cannot represent thermal, tactile, moisture, fit, mobility and aesthetic experience without hiding meaningful trade-offs.

The benchmark therefore produces a **comfort trajectory vector**: separate time series for each declared comfort dimension, accompanied by task outcomes, environmental context, adjustments, breaks, withdrawals and uncertainty.

Comfort is a wearer-reported outcome. Instrument readings can describe the conditions around an experience; they cannot overrule the wearer.

## 2. Paradigm shift: garment performance as a trajectory

Conventional product tests often reduce a garment to a static fit check, a fabric coupon result, or a final satisfaction rating. Atelier should test the garment as a system evolving through a sequence of real wearing states.

The system boundary is:

wearer intent -> body/fit profile -> pattern revision -> material lot -> assembly revision -> wear session -> care/repair events -> requalification

Every result must be tied to the exact garment configuration tested. A changed collar, seam, fabric lot, lining, closure, fit block or warmth panel must not silently inherit an earlier qualification.

### What the output is

For each dimension d, record the trajectory S_d(t) and the within-session change:

Delta_d(t) = S_d(t) - S_d(0)

where S_d(0) is the post-donning baseline for that session. The response scale and its direction must be declared before testing.

Report, per dimension:

- the observed trajectory at each scheduled time;
- change from the session baseline;
- the lowest observed acceptability and when it occurred;
- the participant's stated reason and body region when discomfort is reported;
- whether an adjustment, layer change, pause or garment removal was requested;
- recovery or non-recovery after a voluntary pause;
- task completion and movement restriction;
- missing observations, early stopping and withdrawals.

**Do not average these dimensions into one Atelier Comfort Score.** Overall comfort may be collected as one additional wearer-reported dimension, but it cannot replace the separate trajectories.

## 3. Formative and confirmatory study design

### Phase A — formative instrument and prototype debugging

Start with the proposed 5–8 consenting wearers, where feasible across different body/fit profiles and fit preferences. This phase is for discovering failure modes, checking that questions are understandable, and improving the task protocol. It is not a statistical demonstration of superiority.

Use the formative phase to establish:

- whether the response anchors are understandable and repeatable;
- whether the measurement schedule causes unacceptable interruption;
- which activities reveal garment-specific restriction or pressure;
- whether different participants interpret a question differently;
- which adverse or unacceptable conditions require an immediate stop;
- whether sensor attachments, if used, alter garment feel.

Revise and version the instrument before confirmatory testing. Do not retroactively mix measurements collected under materially different instrument versions.

### Phase B — matched, randomized crossover

Where feasible, the same participant wears the Atelier garment and a commercially available comparator in separate sessions. Randomize and counterbalance garment order (AB/BA). Use the same participant, declared fit profile, underlayer, activity sequence and as-similar-as-practical environmental conditions.

Record unavoidable differences rather than pretending they do not exist. Use separate garment sizes/fit profiles only when the fit rationale is documented. A comparator must be appropriate to the product's actual use, not merely visually similar.

For CR-01, include at least one travel-oriented wrap comparator and one conventional robe or over-layer where available. A weatherproof changing robe is a boundary comparator, not an equivalent product.

### Session duration and observation schedule

For the first AO-01 and CR-01 samples, use a manageable 120-minute session after the formative protocol is stable. Record at:

- post-donning baseline (after a declared settling period);
- 15 minutes;
- 30 minutes;
- 60 minutes;
- 90 minutes;
- 120 minutes.

Only extend CR-01 to 180–240 minutes after the shorter protocol demonstrates acceptable participant burden and the garment passes basic fit/closure safety checks. Do not require a participant to finish a session. Participants may pause, adjust, remove the garment or withdraw at any time without penalty.

The interval schedule is an initial protocol proposal, not an ISO-prescribed schedule. Freeze it before confirmatory data collection.

### Repeated task sequence

A session should contain safe, reproducible blocks covering:

1. standing and ordinary walking;
2. desk/chair sitting and a reclined seated posture;
3. forward and overhead reach;
4. bending and turning;
5. sit-to-stand transitions;
6. pocket access while standing and seated;
7. closure operation and independent don/doff;
8. a short carry/luggage or bag-strap interaction where relevant.

Record task order, duration, completion, garment adjustment and any failure. Use a safe stationary mock-up for travel tasks. Do not test inside a moving vehicle or aircraft.

## 4. Wearer-reported dimensions

Use separate items with explicit response anchors. A suggested seven-point acceptability scale is:

- 1 — completely unacceptable;
- 2 — very unacceptable;
- 3 — somewhat unacceptable;
- 4 — neither acceptable nor unacceptable;
- 5 — somewhat acceptable;
- 6 — very acceptable;
- 7 — completely acceptable.

Include “not applicable / unable to judge” where appropriate. Do not treat that response as a neutral score.

At each scheduled time, collect these separately:

| Dimension | Prompt focus | Additional context |
|---|---|---|
| Tactile | Surface feel, cling, scratchiness and contact comfort | Contact region; fabric or seam suspected |
| Thermal | Whether the garment feels thermally acceptable now | Too warm / about right / too cool |
| Moisture | Dampness, cling and perceived moisture accumulation | Dry / slightly damp / uncomfortable; drying after activity |
| Fit and pressure | Pulling, pressure, constriction and garment migration | Body region; posture when noticed |
| Mobility | Freedom to reach, bend, sit, walk and transition | Task completed; adjustment needed |
| Overall comfort | Overall acceptability at this moment | Reason for rating; not a substitute for the dimensions above |

At the start and end, collect aesthetic/style acceptance and whether the wearer would choose to wear the garment in its intended context. Keep aesthetics separate from functional comfort.

Do not require body photographs, body scans, medical information or inferred demographic attributes. If a body-region diagram is useful, use a neutral, optional, participant-controlled body-region selector without requiring an image of the participant.

## 5. Environmental and session context

Record only what is needed to interpret the session:

- ambient temperature and relative humidity;
- activity/rest state and time spent in each block;
- underlayer and other worn layers;
- garment ID, size/fit profile, revision and measured mass;
- fabric/material lot and care/wash history;
- session start/end, observation times, breaks and adjustments;
- whether the participant had recently exercised or changed environment, when voluntarily disclosed and relevant.

Optional local garment microclimate temperature/humidity sensors may be used in a separate instrumented lane. Sensor locations, attachment methods and calibration must be recorded. First establish that the sensor does not materially change drape, pressure, moisture or tactile sensation. A sensor reading is not a comfort score or a health measurement.

Climate profiles must be built from measured test conditions (for example, hot/dry, hot/humid, cool/dry and indoor/outdoor transitions). Do not treat a country or city as a single climate.

## 6. Analysis plan

### Formative phase

Use participant-level plots and structured qualitative coding to identify recurring failure modes. Do not calculate inferential superiority claims from the 5–8-person formative group.

### Confirmatory phase

Before recruitment, preregister the comparator, primary dimension(s), primary timepoint or trajectory contrast, exclusion rules, missing-data handling, environmental envelope and analysis code/version.

For ordinal acceptability responses, prefer an ordinal mixed-effects model (or another justified repeated-measures model) that accounts for within-participant observations. Include garment, time, garment-by-time interaction and prespecified session/order/environment covariates as justified. Report effect estimates, uncertainty intervals and participant-level trajectories. Do not assume the seven response categories are equal-interval measurements without justification.

Use a pilot to estimate variance, within-person correlation, feasibility and measurement reliability. Determine confirmatory sample size from the selected primary endpoint, a wearer- and use-case-justified minimum meaningful difference, desired power, and expected attrition. The formative sample size is not a substitute for that calculation.

For multiple comfort dimensions, preregister the primary family and multiplicity strategy. Show lower-tail and individual participant outcomes; do not publish only a pooled mean. Preserve withdrawals, incomplete sessions and adverse experiences in the flow and denominator reporting.

### Candidate measures (report as separate outcomes)

- dimension-by-time response distribution;
- within-person change from baseline;
- model-estimated garment-by-time contrasts;
- task completion and adjustment/stop events;
- time to participant-requested change or removal (exploratory; censored observations retained);
- recovery after voluntary rest/adjustment;
- missingness and participant burden.

A threshold-based “time-to-discomfort” endpoint may be added only after wearer feedback and the formative phase define a defensible threshold. Do not invent a universal two-point decline or label a participant's decision to stop as a failure of the participant.

## 7. Garment-specific engineering tasks

### AO-01 Everyday Overshirt

Test the exact Quiet Architecture block and the selected fabric/lot.

- Shoulder, armscye, underarm and elbow: assess forward/overhead reach, crossed arms and repeated flexion without using excessive torso ease to conceal poor sleeve geometry.
- Back yoke/pleat: compare clean-back and movement-allowance versions with the same fit model and tasks.
- Collar and collar stand: check contact pressure, roll, edge stiffness and seated neck posture over time.
- Cuffs and sleeve placket: test both adjustment positions, independent operation and pressure when the wrist is flexed.
- Pockets: test access while seated and standing, retention of the declared everyday items, pocket-bag migration and local pressure. Use declared item masses rather than arbitrary loads.
- Stress zones: inspect placket, pocket mouth, cuff, vent ends, shoulder and underarm after movement and laundering.

Do not grade the fit block until the first physical toile is accepted. Record body measurements, finished garment measurements and ease independently. Keep regional ease explicit: chest, waist, seat, upper arm, elbow, wrist, neck and layering allowance.

### CR-01 Continuum Wrap

The robe must be attractive without the warmth panel and secure without relying on a belt alone.

- Measure front overlap and coverage in standing, sitting, reclined sitting, bending, reaching and walking positions.
- Test the concealed retention closure and external belt separately; document accidental opening, shifting, pressure and one-hand operation.
- Test seated pocket access with a bag strap and seatbelt mock-up; do not claim travel security.
- Check sleeve/hem/belt interaction with walking, steps and a stationary luggage-handle mock-up.
- Test warmth-panel attachment, removal, attachment-point pressure, shifting and independent lap use.
- Compare natural-fibre warmth and technical insulation coupons at matched declared conditions before selecting a panel.
- Keep the robe visually and functionally complete when the panel is removed.

For the wrap, define an overlap/coverage envelope by posture and activity rather than a single standing measurement. A design fails if it unexpectedly opens, exposes a wearer-declared protected area, blocks movement or creates an unacceptable pressure point.

## 8. Material and construction qualification

Use matched coupons and record the exact fabric lot. Separate fabric-coupon results from assembled garment results and from wearer reports.

Candidate method register (verify applicability, access to the full standard and the latest national adoption before formal testing):

| Standard | Verified edition/status at 2026-11-15 | Use and limitation |
|---|---|---|
| ISO 11092 | 2026 edition published May 2026 | Thermal and water-vapour resistance under steady-state sweating-guarded-hotplate conditions; not a universal comfort rating |
| ISO 13935-2 | 2026 edition published April 2026 | Grab-method maximum force to rupture straight sewn seams; not a curved-seam test |
| ISO 6330 | 2021 edition published | Domestic wash/dry procedures; specify machine, detergent, ballast, wash and drying recipe |
| ISO 3759 | 2011 edition confirmed current in 2022 | Specimen/garment marking and measurement for dimensional change |
| ISO 5077 | 2007 edition confirmed current in 2022 | Dimensional change after specified washing/drying |
| ISO 9237 | 1995 edition confirmed current in 2023 | Fabric air permeability |
| ISO 12947-2 | 2016 edition confirmed current in 2023 | Martindale specimen breakdown; not a complete garment lifetime |
| ISO 12945-2 | 2020 edition published; under review | Modified Martindale pilling/fuzzing/matting; recheck status before test order |
| ISO 20932-1 / -2 | 2018 editions confirmed current in 2024 | Fabric stretch/recovery where the fabric actually uses elasticity |
| ISO 15487 | 2018 edition published; revision ISO/DIS 15487 is in enquiry | Appearance of apparel after domestic wash/dry; record which edition is used and monitor replacement |
| ISO 8559-1 | 2017 edition confirmed current in 2026 | Anthropometric body-measurement vocabulary and procedure |
| ISO 8559-2 | 2025 edition published; supersedes 2017 edition | Primary and secondary body-dimension indicators for clothing size designation |

A standard result is bounded by its method, specimen, instrument, conditioning and stated uncertainty. It is not automatically transferable to a whole garment, another fabric lot or a person's lived comfort.

For every test, record specimen ID/orientation, material lot, conditioning, method/edition, equipment and calibration, operator, repetitions, raw values, uncertainty, deviations and failed/inconclusive results.

## 9. Digital-to-physical calibration

Digital garment tools are valuable for rapid screening, but their outputs must be calibrated against the physical garment.

A 2026 exploratory study comparing 3D simulations with physical fit reported limitations involving avatar posture, body-fat distribution and fabric representation. It used four models and 21 experienced assessors; treat it as a warning and research signal, not a universal estimate of all CAD systems.

Reference: Guo, “A comparative study of virtual and real fit in customised garments,” *International Journal of Fashion Design, Technology and Education* (2026), https://doi.org/10.1080/17543266.2026.2618270.

For AO-01 and CR-01, compare predicted and measured:

- key body/garment landmarks and finished dimensions;
- seam and pattern-piece correspondence;
- sleeve/armscye, collar/neck and wrap-overlap geometry;
- drape/hem position and garment migration in the named postures;
- material mass, thickness and mechanical/thermal inputs;
- observed fit failures and the pattern correction required.

Keep simulation version, avatar/measurement profile, fabric parameter source and solver configuration with the result. Report virtual-to-physical residuals by feature, not just one global visual-similarity score. A simulation pass must never replace a physical toile or wearer acceptance gate.

## 10. Industry benchmark: compare by use, not by marketing category

Repair, travel comfort and durable construction are not novel claims. Existing market programmes establish meaningful baselines:

- Patagonia reports repairing garments since the 1970s, operating repair through repair centres, tours and stores; its 2026 corporate release reports nearly 160,000 Worn Wear items sold in fiscal 2026 and more than 230,000 items processed by its team in the prior year: https://www.patagoniaworks.com/press/2026/10/14/a-new-milestone-for-patagonias-resale-business
- Nudie Jeans offers free repairs for its jeans through repair stores/partners and repair kits, while acknowledging that it does not currently provide mail-in repair in all markets: https://www.nudiejeans.com/help-center/where-can-i-get-my-jeans-repaired
- ASKET publishes repair guides and offers spare parts such as buttons, thread, fabric pieces and zippers: https://mp.asket.com/us/lifecycle/repair-guides
- MERSEA's Modern Travel Wrap uses a securing loop and matching travel bag; it is a direct wrap/portability reference, not a tailored robe: https://mersea.com/products/modern-travel-wrap
- Dryrobe Tech-Lite is a weatherproof changing robe with a stated 1 kg mass including its bag, 10,000 mm waterproofness and a three-year performance guarantee. It is a technical/weather-changing product, not a like-for-like luxury travel wrap: https://dryrobe.com/products/dryrobe-lite

Atelier must not claim that repair, modular warmth, a travel wrap, or a hidden pocket is unprecedented.

### Matched comparator protocol

For each garment, freeze at least three commercially obtainable references where the market permits. Record exact model/SKU, size, purchase date, country/market, retail price and currency, fibre/care label, measured mass, construction observations and the reason it is a fair comparator. Include a lower-cost conventional baseline and a premium reference where feasible. Do not compare unlike products without declaring the use case.

The competitive question is not “Is Atelier better than fashion?” It is:

> For which wearer, posture, climate and use profile does this exact garment improve a declared outcome against the best relevant available alternative, without sacrificing another hard requirement?

## 11. Proposed performance gates

These are research targets, not achieved results or industry standards:

- 100% of accepted hard wearer constraints remain traceable through brief, pattern, BOM, assembly, revision and tested garment.
- Zero unsupported comfort, composition, durability, weather, sustainability or safety claims are promoted from a supplier description, simulation or marketing page.
- Zero critical wrap-closure or garment-integrity failures are admitted in the frozen synthetic and bench-cycle acceptance corpus.
- 100% of comfort observations, including missing observations, voluntary stops and adverse experiences, are represented in the analysis record.
- 100% of published physical results identify the exact garment revision and material lot tested.
- Against matched commercial references, show a preregistered, wearer-valued improvement in at least one use-case-specific outcome while meeting non-inferiority margins for all declared critical outcomes. Set margins and sample size only after the formative pilot.
- For repairability, demonstrate that common failures (button/closure, pocket-bag seam, cuff/hem and a selected structural seam) can be inspected and repaired with declared common tools/parts, then re-inspected. Publish repair time, cost, parts, appearance and residual strength; do not call a repair successful merely because it looks intact.
- For digital fit, report physical residuals and reduce avoidable physical rework against a matched incumbent workflow only after a measured baseline exists.

A win in one dimension does not authorize a universal “most comfortable,” “most durable” or “most sustainable” claim.

## 12. Adversarial and negative controls

The corpus should include at least these cases:

1. A garment passes at time zero but becomes unacceptable at 60 or 120 minutes.
2. An early stop caused by discomfort is omitted from the result.
3. A pooled mean hides an individual or subgroup failure.
4. A participant changes an underlayer or environment and the change is not recorded.
5. Sensor tape, cabling or a microclimate device changes the garment's tactile feel.
6. Thermal balance is misreported as “cooling” or “warmth” without the participant's direction of preference.
7. High air permeability is treated as equivalent to water-vapour comfort.
8. A fabric coupon is promoted to whole-garment comfort.
9. A standing-only robe fit conceals seated gaping or pocket inaccessibility.
10. The belt remains tied but the concealed retention closure fails.
11. A removable panel shifts, creates pressure, or cannot function independently as a lap cover.
12. A changed material lot or pattern revision inherits stale test evidence.
13. A virtual fit is represented as a physical fit result.
14. A repair is counted as successful without an inspection/requalification record.
15. A supplier's fibre declaration is presented as independent composition analysis.

## 13. Evidence record and release discipline

Every result should link to:

- benchmark/protocol revision and preregistration identifier;
- consent and participant pseudonym (no direct identifying body data in public records);
- session, comparator, environment and activity sequence;
- product/style, pattern revision, BOM revision and sewing-plan digest;
- material supplier/SKU/lot and evidence class for composition;
- instrument/test method/edition/calibration where applicable;
- raw observation and analysis version;
- missingness, deviations, withdrawals and adverse events;
- reviewer decision and permitted claim scope.

Public repository fixtures should be synthetic and rights-cleared. Do not publish participants' body measurements, photographs, identifiable session logs or private preferences. A public sample report may publish only properly consented, aggregated or de-identified information with an explicit data-governance basis.

## 14. Immediate execution sequence

1. Add this protocol as an extension under EverydayWearBench #12157.
2. Update the shared standards register to ISO 8559-2:2025 and the 2026 editions of ISO 11092 and ISO 13935-2; flag ISO/DIS 15487 as a draft under development.
3. Build a synthetic session corpus with complete, missing, early-stop, adjustment, adverse, stale-revision and subgroup-masking cases.
4. Implement a deterministic validator for session completeness, timestamp ordering, stable garment/material identity, declared scale direction, missingness preservation and stale-evidence rejection.
5. Freeze the AO-01 and CR-01 task maps and select commercial comparators.
6. Acquire only swatches and coupons for the first screening. Do not purchase production quantities until lot, price, care, composition and stock are confirmed.
7. Produce the first AO-01 toile and record physical fit corrections.
8. Recalibrate the 3D profile against the toile; keep the physical fit gate.
9. Proceed to CR-01 toile and warmth-panel tests only after the basic wrap geometry and closure are stable.
10. Conduct wearer studies only with consent, participant stop rights and the approved protocol version.

## 15. Claim ceiling

This protocol qualifies the design and reporting of a time-resolved wear study. It does not establish that a garment is comfortable, safe, durable, sustainable, protective or commercially superior.

The small formative group can reveal design failures; it cannot establish population-wide comfort. The textile standards qualify only their specified tests. Digital simulation is a screening aid. Only physical garments, matched material lots, controlled evaluation and properly analyzed wearer observations can support the corresponding product claims.
