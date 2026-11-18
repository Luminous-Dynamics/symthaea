# Luminous Aqua — Spray Geometry Design-of-Experiments Protocol v0.1

**Status:** experimental design preregistration template; no nozzle geometry or bench result is qualified  
**Workstream:** [AQUA-SPRAY-DOE-001 #12772](https://github.com/Luminous-Dynamics/symthaea/issues/12772)  
**Parent:** [AQUA-SPRAY-FIELD-001 #12692](https://github.com/Luminous-Dynamics/symthaea/issues/12692)  
**Analysis:** `symthaea-aqua-bench::spray_field`

## 1. Decision objective

Select a manufacturable passive nozzle/manifold architecture that provides the best **robust spray-field performance per litre** while meeting all required standard-method, task, temperature, pressure, cleaning and lifecycle constraints.

This is not a geometry beauty contest or a nominal-flow optimization. The winning design must survive manufacturing variation and declared pressure / water-quality / maintenance challenges.

## 2. Factor register

The exact values and low / centre / high settings must be established from CAD, material, manufacturing process capability and rated component limits before runs are generated. Do not copy universal dimensions into this protocol.

| Factor | Physical meaning | Candidate levels to freeze | Principal interactions to test |
|---|---|---|---|
| F1 | Total outlet area | Manufacturable low / centre / high | F2, F5 |
| F2 | Outlet-area spatial allocation | Uniform / task-zoned / candidate distribution | F1, F3 |
| F3 | Outlet count and grouping | Feasible count / sector layouts | F2, F4 |
| F4 | Outlet angle / convergence | Manufacturable angle bounds | F3, F6 |
| F5 | Manifold flow-path distribution | Candidate passive channel geometries | F1, F7 |
| F6 | Outlet diameter distribution | Process-capable tolerance range | F4, F7 |
| F7 | Passive pressure compensation | None / qualified commercial element / candidate passive concept | F5, F6 |
| F8 | Cassette face / seal / retention geometry | Serviceable, manufacturable interface alternatives | F1, F5 |
| F9 | Passive feedback-channel / oscillator geometry (optional research arm) | No oscillator / single-feedback / double-feedback; feasible cavity and channel ratios from a defined design envelope | F1, F3, F5, F6 |
| F10 | Nozzle-area ratio and resistance factor | Exact dimensionless definitions and manufacturable bounds taken from the replicated literature method / CAD | F1, F2, F5, F6 |

Each factor entry must include: units, exact CAD parameter, physical rationale, manufacturable range, tolerance distribution, process route, material/finish, rated pressure/temperature constraints, and a digest of the source CAD / configuration. If a factor changes another dimension unintentionally, record the dependency rather than treating the variables as independent.

## 3. Response vector and hard constraints

### Standardized endpoints

Use the current controlled edition and applicable test procedure for any formal standard claim. WaterSense's published specification uses the incorporated ASME/CSA spray-force and spray-coverage methods as well as pressure-compensation / flow tests. Aqua's spatial map is supplementary, not a substitute.

### Aqua diagnostics

For each complete, externally qualified deposition map, report:
- target-area coverage fraction at the preregistered deposition threshold;
- area-weighted target deposition mean and coefficient of variation;
- off-target mapped mass share;
- total mapped mass and target/off-target subtotals.

The Rust helper in `symthaea-aqua-bench` calculates these descriptive values and matched state deltas. It does not qualify the collector boundary or infer comfort.

### Other independent outcomes

Retain separate outputs for flow, dynamic inlet pressure, standardized force/coverage, temperature drop at declared bather-plane location(s), spray-to-skin heat-transfer coefficient only where the measurement method is validated, splash/off-target capture, noise, scale, partial blockage, cleaning labour, post-clean performance recovery, manufacturing yield and unit-to-unit variation.

No single weighted score may compensate for a hard-constraint violation. Do not infer comfort, cleaning efficacy or aerosol safety from deposited-water density.

## 4. Sequential experimental design

### Phase A — comparator and measurement-system anchor

1. Freeze exact comparator SKU, mode, installation, pressure and test-method revision.
2. Establish flow/pressure/temperature measurement repeatability and calibration status.
3. Establish spray-force and standard coverage method reproducibility under the selected controlled method.
4. Establish deposition-grid collector repeatability, cell-area identity, detection floor and complete capture boundary.
5. Run blank / leak / missing-cell / saturation / drift negative controls.
6. Do not rank candidate geometries if the measurement system cannot distinguish the declared effect from its uncertainty.

### Phase B — factor screening

Use a preregistered factorial or fractional-factorial design appropriate to the number of active factors and interactions. A resolution-IV-or-better fractional design is a candidate when a full factorial is not practical; the final design must explicitly list aliased effects and why those aliases are acceptable.

- Include centre points to detect curvature where meaningful.
- Randomize run order within practical blocks.
- Block on specimen/build batch, water batch, test day, operator and apparatus setup.
- Use independent specimens for between-build variation; repeated readings on one specimen are not extra specimens.
- Freeze the analysis model and primary endpoints before unblinding the screening results.
- Record every aborted, failed, clogged, leaking or invalid run and its disposition.

### Phase C — response-surface / sequential refinement

Only after screening and measurement-system checks indicate stable factors:
- narrow the design space to physically plausible, manufacturable candidates;
- use a response-surface design (for example, central-composite or Box–Behnken) when factor continuity and feasible operating points support it;
- constrained Bayesian optimization may prioritize the next candidate only if the data volume, model uncertainty and design space justify it;
- keep standard, safety, pressure, temperature and hygiene requirements as hard constraints outside the optimizer;
- use existing Symtropy / qualified specialist physical models for hydraulic/thermal interpretation; do not create an Aqua duplicate CFD solver.

The optimizer proposes experiments; it does not approve hardware or replace measured evidence.

### Phase D — robustness and lifecycle challenges

For shortlisted geometries, run a separate challenge sequence:
- declared local low-pressure point (150 kPa);
- separately declared WaterSense comparison points (140, 310 and 550 kPa) only within rated / approved fixture limits;
- 300 kPa local / nominal 3 bar comparator point;
- cold-start and stabilized temperature states;
- measured hardness / water-chemistry profile and a preregistered scale-aging procedure;
- single-nozzle and sector partial blockage;
- inlet-strainer loading;
- damaged / missing insert;
- post-clean and requalification state;
- process tolerance corners and independent build batches.

Do not treat 140, 150, 300, 310 and 550 kPa as interchangeable. Do not test beyond the lowest rated component or approved apparatus envelope.

### Phase E — confirmatory holdout

Freeze exact geometry, process, settings, analysis, endpoints, margin and exclusions. Test independent confirmation specimens, held-out runs and at least one different test day / water batch where practical. The final report must separate exploratory selection data from confirmatory data.

Determine sample size from pilot variance, chosen primary endpoint, acceptable margin, desired confidence / power and between-specimen variability. Do not set an arbitrary sample count as proof of significance.

## 5. Run-order and provenance requirements

Every run record must bind:
- run ID and randomized order;
- exact specimen, build batch and geometry/CAD digest;
- material / finish and manufacturing process revision;
- factor settings and actual measured dimensions / tolerances;
- installed orientation and test apparatus revision;
- exact comparator identity / mode where applicable;
- pressure trace, inlet-water temperature and chemistry;
- room conditions and ventilation for tests where relevant;
- map grid / target region / collection interval / threshold revision;
- instrument IDs, calibration, uncertainty, timebase and operator;
- cleaning / fouling / blockage state and preceding interventions;
- raw observations, processing-code digest, failed-run status and exclusions.

The test-context digest used by `compare_spray_field_to_baseline` must bind identical task, pressure, water temperature, ambient conditions, mounting, grid geometry, target mask, collection duration, map method and deposition threshold. Geometry digest must bind the exact CAD/manufacturing revision. Baseline and challenged observations must be from the same specimen for degradation-recovery comparisons.

## 6. Statistical and anti-gaming rules

- Declare primary endpoints, margins, model terms, multiplicity handling and exclusion rules before confirmation.
- Report run-level data, not only cell averages or best-performing specimens.
- Treat grid cells as spatial observations nested in a test run, not independent replicates.
- Report confidence intervals and between-specimen variation.
- Keep screen / refine / confirm datasets distinct.
- Publish Pareto frontiers and hard-constraint failures.
- Do not select a pressure point, target polygon, deposition threshold or spray mode after seeing which candidate wins.
- Do not silently discard pressure collapses, blocked nozzles, leaks, failed cleans or aborted runs.
- Use a separate challenge sequence for aged / fouled / post-clean states; do not assume clean new hardware represents service-life performance.
- Report model uncertainty and prediction error on held-out physical runs.

## 7. Selection rule

A geometry is eligible for selection only if it meets all preregistered hard requirements and has adequate evidence at the required pressure, temperature, standard-method, cleanability and manufacturing conditions.

Among eligible geometries:
- report the Pareto set for water, standard spray force/coverage, task-region map, temperature, robustness, serviceability and manufacturing cost;
- select only with a documented product-level decision rule agreed before confirmation;
- where trade-offs remain, report them explicitly rather than inventing one global score;
- if no candidate meets the hard constraints, stop or pivot; do not lower the criteria post hoc.

## 8. Research targets

The 6.0 L/min primary and 4.5 L/min stretch points remain investigation targets, not novelty or superiority claims. Commercial products already publish 4.5–5.7 L/min modes. The differentiator must be a statistically supported matched improvement in the robust response vector and completed service—not the minimum nominal flow alone.

The first study should compare:
- A0: exact local premium comparator modes;
- A1: passive zoned manifold;
- A2: replaceable nozzle cassette;
- A3: passive fluidic / flow-shaping geometry only if it is manufacturable and measurable;
- A4: fixed plus handheld delivery as separate task modes.

Do not begin with powered variable zones until passive designs demonstrate a robust, maintainable advantage. A fluidic oscillator is a research hypothesis, not a shower-proven technology: a March 2026 review of wall-attachment oscillating jets reports double-feedback oscillation at Reynolds numbers around 220 in certain studied configurations, but those configurations are not showerheads and the threshold cannot be transferred directly. The study motivates a bounded test arm only. Source: Jaafarian et al., International Communications in Heat and Mass Transfer (2026), DOI 10.1016/j.icheatmasstransfer.2025.110243, https://www.sciencedirect.com/science/article/pii/S0735193325016690.

## 9. Evidence ownership

- Aqua #12692 / #12772: geometry, DOE, spray-field product comparison and task context.
- Mycelix: instrument identity, calibration, uncertainty, timebase and applicability.
- Symtropy: existing hydraulic / thermal model outputs and canonical water/energy/material/residual closure.
- Mycelix / competent water-quality authority and #12703: wet-path hygiene, source quality and aerosol/exposure evidence.
- Independent standard-method laboratory: formal ASME/CSA / WaterSense test procedures and certification claims.
- Independent human-factors evaluator: task-quality instruments and non-inferiority margins.

## 10. Claim ceiling

This protocol defines experiment design, not physical performance. It authorizes no prototype procurement, pressurized article fabrication, human exposure, health/aerosol claim, standards certification, occupied installation or construction release.


## 11. New passive-fluidics hypothesis — oscillating coverage without moving parts

A potentially high-value concept is a passive fluidic oscillator that uses internal feedback geometry / wall attachment to sweep a coherent jet or distribute a flow pattern without a motor, solenoid, rotating faceplate or external power. This may allow a broad task-region pattern without permanently dividing a low-head supply into many small orifices.

The research basis is adjacent, not direct: a March 2026 review compares cavity jets and single-/double-feedback wall-attachment oscillators. It reports that the reviewed double-feedback geometries could sustain oscillation at substantially lower Reynolds number (around Re ≈ 220) than the cavity-jet configurations discussed. This result is configuration-specific and from thermal/jet research, not a showerhead qualification. It does not establish suitable flow, pressure, droplet, aerosol, cleaning or user performance for Aqua.

### Controlled investigation

- Start with a non-oscillating fixed-orifice reference and a passive multi-nozzle reference.
- Use existing Symtropy / specialist flow models to screen channel ratios and expected pressure loss; do not create a separate Aqua CFD solver.
- Measure oscillation startup threshold, frequency, sweep angle, spatial deposition map, total flow, dynamic pressure drop and temporal stability.
- Test 50 / 75 / 100 kPa only within component and apparatus ratings, then 140 / 150 / 300 kPa for the declared comparison purposes.
- Challenge with measured hardness/scale, particle contamination, partial obstruction, orientation changes and post-clean requalification.
- Characterize droplet-size and aerosol time series separately; do not assume sweeping improves exposure.
- Reject any design whose oscillation becomes intermittent, asymmetric, noisy, hard to clean or unstable after fouling.

**Falsification:** if the device needs more inlet pressure than the target installation can supply, or its pressure loss, aerosol profile, clogging or cleanability erases any useful coverage gain, it is not an Aqua breakthrough. A simple fixed passive manifold may remain superior.


## 12. Prior evidence: spray pattern and heat-transfer coupling

Zhang, Mui and Wong (Water, 2024, 16(7), 928) tested five showerheads and 18 spray patterns. Their study reports that nozzle area ratio, supply pressure and resistance factor significantly affected the spray-to-skin heat-transfer coefficient; it reports that a higher heat-transfer coefficient could reduce energy use by up to 20% while maintaining the same thermal sensation in that study's conditions.

This is important prior evidence: nozzle geometry should not be optimized independently of the thermal-service boundary. Aqua should preserve the paper's dimensionless predictor variables where definitions can be reproduced, and separately measure water flow, pressure, delivered temperature, bather-plane temperature and any validated heat-transfer coefficient. A product-level thermal or human-comfort claim requires a matched Aqua test and its own independent endpoint method; the paper's 20% result is not an Aqua target or guarantee.

Source: https://www.mdpi.com/2073-4441/16/7/928
