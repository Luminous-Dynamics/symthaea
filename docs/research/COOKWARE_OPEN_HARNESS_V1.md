# COOKWARE OPEN HARNESS V1
## A reproducible, low-cost path from pan comparison to independent evidence

**Status:** proposed research protocol. No physical cookware was tested to produce this document.  
**Related:** [COOKWARE-CRAFT-001 #8182](https://github.com/Luminous-Dynamics/symthaea/issues/8182), [COOKWARE-LIFETIME-001 #8191](https://github.com/Luminous-Dynamics/symthaea/issues/8191), [COOKWARE-HARNESS-001 #8197](https://github.com/Luminous-Dynamics/symthaea/issues/8197), [benchmark PR #8190](https://github.com/Luminous-Dynamics/symthaea/pull/8190), [lifecycle PR #8193](https://github.com/Luminous-Dynamics/symthaea/pull/8193).

## 1. Research thesis

Build a public, reproducible test method that lets independent kitchens, repair shops, schools and small labs produce comparable evidence with affordable instrumentation—without confusing cheap sensors with ground truth and without requiring a vendor-controlled composite score.

The paradigm shift is from **one product, one ranking** to **a profile-indexed evidence surface**:

`candidate × physical specimen × cooktop × task × load × wear state × measurement method`

A candidate is only comparable with another candidate where the declared conditions and intended task align. An induction-only test cannot establish performance on gas. A benchmark of egg release on nonstick cannot establish every stainless-steel pan's overall value. Different tasks should expose trade-offs, not be averaged into a universal “best pan” number.

The harness is a research/measurement method, not a cookware product, food-contact certification, legal-compliance decision, or proof of a lifetime claim.

## 2. External baselines and what we add

Consumer Reports describes a profile-specific test portfolio: evenness across pans; cast-iron/carbon-steel tests suited to searing and cornbread; nonstick-release and surface-durability tests; plus cleaning, handle and other usability dimensions. Its March 2026 article describes heated steel-wool abrasion followed by egg-release testing, with up to 2,000 strokes in its top durability category. That is an industry reference, **not** our protocol and not an equivalence claim.

- https://www.consumerreports.org/home-garden/cookware/best-frying-pans-of-the-year-a7222946160/
- https://www.consumerreports.org/home-garden/cookware/least-durable-nonstick-frying-pans-what-to-buy-instead-a4466494459/

The open harness adds transparent raw-data definitions, repeatability checks, explicit calibration and sensor limitations, a frozen scenario matrix, profile-specific missingness and failure reporting, and longitudinal linking to the physical specimen. Its objective is independent reproducibility—not claiming to replace proprietary test labs on the basis of documentation alone.

A 2026 stakeholder-centered repairability study describes repairability ratings that vary between designers, consumers and maintenance professionals, supporting separate stakeholder tasks rather than a single supposedly universal score:
https://doi.org/10.1016/j.cesys.2026.100419

EU Regulation 2024/1781 provides a broad policy framework that includes durability, repairability, maintenance/refurbishment, resource use and end-of-life. This protocol does not claim that a cookware-specific delegated requirement is already in force:
https://eur-lex.europa.eu/eli/reg/2024/1781/oj/eng

## 3. Phase 0 — preregistration before unblinding

Freeze and publish:

1. Candidate and reference identities, design revision, dimensions, mass, construction family, intended tasks, compatible heat sources, and available supplier/material evidence.
2. Protocol version, primary and secondary outcomes, units, direction of improvement, exclusion rules, missing-data handling, randomization procedure and analysis plan.
3. Instrument make/model/serial or anonymous instrument ID, range, resolution, stated accuracy, calibration method/date, placement, acquisition cadence, drift checks and known failure modes.
4. Exact cooktop, control settings, burner/power zone, cookware location, ingredient batch and mass, starting temperature, oil/mass if any, room/environment conditions, operator instructions and endpoint definition.
5. The practical difference that matters: define an engineering threshold or non-inferiority margin **before** results are known. Do not choose a favorable endpoint after exploring the data.
6. For human studies, target participant mix, consent, survey/instrument versions, randomization/order, stop rules, data retention, and the separate practitioner-defined outcomes. No covert audio/video capture or workplace-performance surveillance.

Retain a dated, versioned preregistration artifact and immutable raw-data digest. A digest helps identify byte changes; it does not prove that a record is true or that the equipment was calibrated.

## 4. Phase 1 — affordable fixture, qualified only for its own measurands

### 4.1 Instrumentation and capability declaration

Use instrumentation appropriate to each primary metric. The minimum candidate setup may include:

- A fixed contact-temperature grid using matched, documented thermocouples or RTDs and a logger.
- A separate reference temperature probe for the fluid/load and ambient readings.
- A scale with documented capacity, resolution and repeatability for food/load/oil/mass-loss measurements.
- A timer/data logger and a record of cooktop control setting and operating conditions.
- A suitable electrical power/energy meter for electric/induction cooktops or a documented fuel-use measurement for gas where feasible.
- Optional calibrated force fixture for food-release measurement and a fixed-location handle-temperature measurement arrangement.
- A reference straightedge/flatness fixture and repeatable measurement method for post-test warp.

For every instrument, record the actual unit, resolution and calibration status. Resolution is not accuracy. A supplier datasheet is not a current calibration certificate. A temperature sensor's contact, response time and attachment can disturb the local temperature field and must be documented.

**Infrared camera warning:** apparent temperature depends on emissivity, viewing geometry, reflections, wavelength and calibration. Do not use uncorrected IR images as primary comparative surface-temperature truth across shiny stainless steel, coated surfaces and cast iron. Use a qualified method for each surface or keep IR evidence explicitly exploratory and surface-specific.

**Energy boundary warning:** record electricity or fuel delivered at a defined boundary. Do not treat metered electrical input, gas fuel energy, burner output and heat absorbed by the food as the same quantity. If an across-fuel comparison is planned, preregister the conversion and system boundary and publish component results.

### 4.2 Fixture qualification

Before testing candidates, assess the fixture with a reference setup:

- Sensor agreement and drift before/after a session.
- Repeatability with a fixed reference pan and an identical controlled procedure.
- Day-to-day reproducibility and operator-to-operator variation where more than one operator is used.
- Missing readings, logger clock alignment, spatial position repeatability, and load/power-control variation.
- Known-answer checks for unit conversions, energy integration, timestamps, and data export.

Use at least three repeated readings within a session and repeated sessions on separate days as an initial **metrology shakedown**. This is not a sample-size claim about the cookware population, and it is not enough by itself to establish product superiority. The protocol must state whether repeatability error is small enough for the planned effect threshold; if not, improve the fixture first.

## 5. Phase 2 — freeze the operating envelope

An operating envelope lists the conditions for which a comparison is intended to support a conclusion. Start narrow and add profiles only where the use case warrants them.

### Core thermal/electrical profile
- A defined empty-pan heating sequence, with a safe upper bound consistent with the candidate's intended use.
- A fixed sensor coordinate grid, fixed sampling interval and preregistered stabilization/endpoint rule.
- Temperature spread, edge-to-centre gradient, response time, overshoot and cooldown/recovery where defined.
- Energy consumed to reach the specified state at the declared appliance input boundary.

### Controlled load/recovery profile
- Frozen load material, mass, starting temperature, placement and timing.
- Defined surface/load starting state and recovery threshold.
- Repeatable recording of transient temperature response and consumed energy.

### Task profiles, selected by intended use
- Searing/browning: fixed ingredient cut/geometry, mass, starting condition, endpoint, surface-preparation method and image/colour measurement conditions.
- Food release: only for a task/surface for which the comparison is meaningful; define food, temperature, oil, timing, spatula, lifting method and force measurement. Do not use “egg release” to claim universal cooking superiority across dissimilar construction families.
- Simmer/heat retention: only where relevant; fixed vessel contents, heat setting, start/end temperatures, evaporation/mass-loss and control method.
- Cleaning: standardized residue, cooling delay, cleaning water/temperature, tool/detergent, operator protocol, endpoint and time/consumable recording.
- Ergonomics: handle temperature, total/filled mass, centre-of-mass or balance method, grip/one-handed/two-handed task, pour-control task, and practitioner context.

### Cooktop profiles
Use induction, electric resistance, and gas only if the intended candidates and fixtures support each. Freeze the equipment model, power settings and location of each pan. Where a candidate is not compatible or a profile is not tested, record **not applicable**, **not tested**, or **failed** with a reason; never carry a value from another cooktop into that cell.

No cross-profile grand ranking is allowed unless a justified, user-facing task distribution and weighting policy were preregistered. Default output is one metric vector and uncertainty summary per candidate and profile.

## 6. Phase 3 — randomized blocked trial

1. Use at least three appropriate commercial reference families for the chosen task envelope where practical (for example, clad stainless, carbon steel/cast iron, and a mainstream nonstick pan for the tasks that suit each). Include a proposed design only after its specification is frozen.
2. Use independently acquired physical units. Give each unit a stable ID; repeated runs of one unit are repeated measures, not additional independent cookware samples.
3. Within a block, randomize candidate order. Balance run order across sessions to reduce warm-up, room-temperature, operator-learning and ingredient-batch effects.
4. Record candidate, unit, run, block, day, operator pseudonym where required, cooktop, task, condition profile, raw observations, method, calibration, and protocol identity.
5. Keep every trial outcome, including failed runs, broken parts, unsafe behavior, missing readings and protocol deviations. Exclusion requires a predefined reason; preserve the original record.
6. Blind downstream outcome assessment to candidate identity where feasible. Do not pretend operators can be blinded to obvious pan constructions during hands-on use.
7. Repeat the run protocol when predefined fixture-validity checks fail; retain the failed attempt and reason in the dataset.

### Sample-size discipline

First run the fixture shakedown and a separate exploratory pilot. Estimate within-unit, between-unit, between-day and between-operator variation for each primary outcome. Then preregister the confirmatory sample size based on the intended minimum effect or non-inferiority margin, uncertainty level, and relevant paired/repeated-measures structure.

Do **not** declare that a fixed number of pans is universally sufficient. A small study can qualify the fixture, reveal gross defects and estimate variability; it generally cannot establish general superiority, population-wide service life, or a safety claim. If resources are limited, explicitly call the study exploratory and narrow the claim.

## 7. Phase 4 — wear, maintenance and repair

Use the lifecycle module/protocol proposed in #8193 once independently qualified. At each preregistered wear checkpoint:

- Record cumulative thermal, cooking-load, cleaning, abrasion and declared misuse/abuse cycles separately.
- Re-run only the declared task/profile measurements under the same or explicitly versioned conditions.
- Record corrosion, warp, surface/coating degradation, joint/handle condition, and all observed failures.
- If maintenance or repair occurs, preserve pre-intervention and post-intervention state, parts, tools, labor time, downtime, cost, and any condition-specific qualification that needs renewed evaluation.
- Keep units that fail or retire in the study. A failed or discontinued specimen is an outcome—not missing nuisance data.

Accelerated cycles are exposure units, not calendar years. Converting cycles into years requires independent empirical correlation and uncertainty; absent that, report the observed duty-cycle profile only.

## 8. Data and analysis contract

### Required record fields
- Protocol ID/version and preregistration digest.
- Candidate and design revision; physical specimen ID.
- Exact profile IDs (cooktop, task, load, environment and wear checkpoint).
- Run/block/order identity and outcome state.
- Raw observation and unit, instrument/method/calibration refs, uncertainty or explicit “not supplied.”
- Evidence type (observed instrument, observed practitioner, manufacturer-declared, simulation or synthetic).
- Protocol deviation, failure, missingness and exclusion fields with reason.
- Source hashes or stable artifact references, without treating hashes as authenticity.

### Report, do not hide
- Per-run raw values and per-specimen trajectories.
- Sample counts at both physical-unit and repeated-measure levels.
- Mean/median and dispersion or uncertainty summaries appropriate to the preregistered design.
- Paired/block-level differences and their uncertainty when the design supports them.
- Instrument repeatability separately from between-pan variation.
- Missingness, failures, protocol deviations, exclusions, adverse events and subgroup/task effects.
- Per-profile comparisons and trade-offs. Use an existing qualified robust-frontier utility if/when its shared PR is merged and validated; do not create a second generic Pareto engine within cookware.
- Stakeholder repairability judgments separately (consumer, repair technician, designer, kitchen operator) if collected; do not bury divergent perspectives in one weight.

A spreadsheet, chart, simulation, or digest is a delivery format—not independent scientific validation. Retain reproducible scripts and the raw records used to compute all derived results.

## 9. Suggested acceptance matrix

These are protocol goals, not current results.

| Gate | Pass criterion |
|---|---|
| Fixture | Reference configuration remains within the preregistered repeatability/drift limits across declared sessions |
| Traceability | 100% of primary observations map to profile, unit, method and available calibration evidence |
| Coverage | 100% of expected candidate × profile × block cells are observed or explicitly classified with a reason |
| Determinism | An independent recalculation of deterministic derived fields matches the published result within declared numeric rules |
| Missingness | Zero silent imputation; missing, inapplicable, failed and excluded observations remain distinct |
| Independence | Physical-unit count is reported separately from repeated readings |
| Analysis | All primary outcomes, thresholds, exclusion rules and sample size are preregistered before confirmatory analysis |
| Safety ceiling | No performance result compensates for missing or unresolved qualification evidence |
| Reproducibility | An independent operator can repeat the method within the declared uncertainty or the deviation is reported |
| Craft | Human outcomes are consented, practitioner-informed and separate from instrumental performance |

## 10. Product-decision principles

- Prefer a task-appropriate option only within the evidence scope; do not name a universal winner where tasks materially differ.
- A candidate should beat a strong, relevant reference on a preregistered primary outcome, with uncertainty supporting the improvement, and meet the non-inferiority/hard constraints.
- If the preferred option changes with cooktop, load, user type, measurement uncertainty or lifecycle scenario, report that sensitivity rather than concealing it.
- Treat safety and food-contact qualifications as separate gates. The harness only records evidence references; it does not certify the reference or replace competent laboratory/regulatory review.
- Report when conventional cookware is the better choice. A framework that cannot return “no improvement demonstrated” is not a benchmark.

## 11. Existing standards/research boundaries

- Consumer Reports testing is an industry comparator; we do not have its full proprietary protocol and should not claim direct equivalence.
- ASTM F1786-97(2021) applies to commercial gas/electric braising-pan energy and performance testing; use as a methods analogue only for a skillet project.
- NSF/ANSI 51 pertains to materials used in commercial food equipment; determine scope/applicability before using it to support any specific cookware claim.
- EU Regulation 2024/1781 is a general ecodesign framework. This protocol makes no claim that a product-specific cookware delegated act already applies.
- Food-contact suitability depends on the specific material/article, manufacturer, use conditions and jurisdiction. Thermal durability is not food-contact evidence.

## 12. What would count as surpassing current practice?

Not a lower-cost fixture alone, and not an attractive single number. A meaningful advance would be that independent groups can reproduce the result; another candidate can challenge it under the same preregistered conditions; raw observations and uncertainty are auditable; the outcome remains useful across explicitly declared scenarios; repair and wear histories are connected to initial performance; and practitioner agency is not traded away unnoticed.

The first milestone is **repeatability and correct accounting**, not a predetermined percentage of improvement. The second is a properly powered, profile-specific physical comparison. The third is an independently repeatable longitudinal study. Product and sustainability claims can only be made once those levels of evidence exist.
