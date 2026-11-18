# Luminous Aqua — Spray-Field Metrology and Robustness Protocol v0.1

**Status:** preregistration / test-method design; no physical spray-field result exists  
**Owner:** Aqua spray-field engineering #12692  
**Parent:** integrated bathing service #12660  
**Related:** matched-service adapter PR #12662, wet-path hygiene #12703, AQS-1 #12717

## 1. Research question and claim boundary

Can a serviceable shower delivery module provide an equal-or-better bathing task with less fresh water across the declared pressure, inlet-temperature, water-quality, installation and maintenance envelope?

The engineering object is a **spatial field delivered to a task**, not a flow-rate label. Water deposition, impact force, temperature, task completion, splash and aerosol are distinct observations. They must not be collapsed into one proxy.

This protocol adds an Aqua spatial-map method to established showerhead tests. It does not replace the applicable ASME/CSA test method, EPA WaterSense certification, independent human-factors work, Mycelix metrology admission, Symtropy resource closure, or a competent hygiene/exposure assessment.

No numeric threshold for comfortable impact, human wetting, acceptable aerosol exposure or non-inferiority is invented here. Those thresholds must be selected and preregistered by the relevant independent method owner before confirmation data are unblinded.

## 2. Updated competitor picture

The 4.5 L/min figure is **not** a world-first or evidence of superiority:

| Comparator | Published specification / result | Evidence class and limitation |
|---|---|---|
| U.S. EPA WaterSense showerheads | Maximum rated flow 2.0 US gpm (7.6 L/min); independent certification framework also evaluates spray force, spray coverage and pressure compensation | Programme specification; U.S. label, not South African law or Aqua certification |
| Brondell Nebia Corre, model family N400RO | Manufacturer sheet lists 1.5 gpm (5.7 L/min) in three modes and 1.2 gpm (4.5 L/min) in Super Saver | Manufacturer-declared mode flow; sheet does not establish matched coverage/force at Aqua's 300 kPa point |
| Hansgrohe Raindance Select E 120 3jet PowderRain EcoSmart+, SKU 28729000 | South African professional datasheet: PowderRain 5.2 L/min, Rain 5.4 L/min, massage 5.2 L/min at 3 bar; maximum 6 L/min; minimum flow pressure 1.5 bar | Exact local-market product declaration; not an independent matched test |
| Hansgrohe Activera Select S 95 2jet EcoSmart, SKU 28087000 | South African datasheet: Rain 7.2 L/min, ImpactRain 6.5 L/min at 3 bar; minimum flow pressure 0.5 bar | Exact local-market product declaration; retain as a second pressure / spray-style comparator |
| Consumer Reports 2026 test population | Public ratings show measured examples around 1.4–1.6 gpm (approximately 5.3–6.1 L/min); tests include coverage, force, consumption, temperature drop and ease of use | Independent consumer-lab measurements; acquire exact current SKU and method before a formal replication |
| South African CLASP sample, summarized by WRC in 2025 | 19 products; pressure-compensating models averaged around 6.6 L/min at 300 kPa; fixed-orifice products averaged substantially higher | Dated local sample, not an exhaustive 2026 market census |

Sources:
- EPA WaterSense showerhead programme and Version 1.1 specification: https://www.epa.gov/watersense/showerheads and https://19january2021snapshot.epa.gov/sites/static/files/2018-07/documents/ws-products-specification-showerheads-v1-1.pdf
- Brondell Nebia Corre manufacturer specification: https://www.brondell.com/content/Nebia_Corre_Fixed_Showerhead_Spec_Sheet_web_010623.pdf
- Hansgrohe South African SKU 28729000 datasheet: https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28729000
- Hansgrohe South African SKU 28087000 datasheet: https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28087000
- Consumer Reports 2026 test overview and ratings: https://www.consumerreports.org/home-garden/showerheads/best-showerheads-of-the-year-a7580523069/ and https://www.consumerreports.org/home-garden/showerheads/c33121/
- WRC 2025 summary of the CLASP 19-product study: https://www.wrc.org.za/wp-content/uploads/mdocs/WW%20July%20-%20Sept%202025_web.pdf

**Comparator rule:** freeze exact SKU, market variant, spray mode, installation, supply conditions, specimen age and maintenance state. Vendor values remain vendor values until reproduced. A model with an unknown duration, mode or pressure point is not silently normalized.

## 3. Benchmark alignment: standard method first, Aqua map second

The test plan must retain the applicable ASME A112.18.1/CSA B125.1 procedures and WaterSense methods when claiming comparability. EPA states that WaterSense evaluates spray force, spray coverage and pressure compensation as well as flow. Its Version 1.1 specification references the harmonized plumbing-fittings test method and defines separate force and coverage criteria. Use the current official controlled method; do not reconstruct it from a summary.

### Pressure points

The matrix should distinguish local research points from WaterSense comparability points:

| Point | Purpose | Admission |
|---:|---|---|
| 140 kPa | EPA / WaterSense spray-force reference point (20 psi nominal) | Use the method's actual tolerance and flowing-pressure definition |
| 150 kPa | South African low-pressure robustness point used in Aqua research | Measure dynamic pressure at the defined inlet reference |
| 300 kPa | South African comparator point and nominal 3 bar manufacturer point | Exact pressure trace and regulator behavior recorded |
| 310 kPa | EPA / WaterSense flow-test reference point (45 psi nominal) | Retain as a distinct declared point; do not relabel 300 kPa as 310 kPa |
| 550 kPa | EPA / WaterSense high-pressure flow-test reference point (80 psi nominal) | Test only if every test-article component and installation is rated for it; otherwise mark not tested / not comparable at this point |

The pressure points are not interchangeable. A unit tested only at 150 and 300 kPa has useful local engineering evidence but has not completed the full WaterSense pressure matrix. Never apply a pressure beyond the lowest rated component or approved test fixture envelope.

### Standard endpoints

Capture the exact outputs required by the selected standard method, including maximum/minimum flow where applicable, spray force, spray coverage, markings and specimen rules. EPA Version 1.1 specifies a 0.56 N minimum spray force at nominal 140 kPa for high-efficiency showerheads and hand showers, and 0.40 N for high-efficiency rain showers. These are standardized test endpoints, not a complete model of every user's comfort or a universal pain/comfort threshold.

### Aqua spatial-map extension

The Aqua map is a supplementary spatial diagnostic. It must not be marketed as an alternative WaterSense or ASME/CSA result.

Use a repeatable bather-plane coordinate frame with:
- fixed origin, orientation, plane distance and fixture mounting geometry;
- versioned cell boundaries and cell areas;
- named target regions and off-target catch regions;
- documented collector/sensor response, detection floor and calibration;
- full capture boundary or an explicit incomplete-capture status;
- synchronized inlet pressure, flow, temperature and room conditions.

Define the intended task regions before testing. A standing general shower, seated shower, hair-rinse and handheld localized-rinse profile are separate tests; they must not share a post-hoc target polygon. Use a non-human surrogate for early bench mapping. Human-factors validation is a separate, later approval gate.

## 4. Keep independent physical fields separate

### A. Water-deposition field

Collect mass or volume on a spatial grid using a frozen collection interval. Convert each cell to deposited mass per unit area only after its area and collector calibration are established. The map describes where collected water lands; it is not a direct measure of comfort, cleaning, impact or aerosol exposure.

### B. Impact-force field

Measure force using the applicable standard method for formal comparison. Any supplementary spatial force array needs its own sensor geometry, dynamic response, calibration, range, saturation check and uncertainty budget. Report the force map separately from deposited-water density. Do not infer comfort from a force map without a preregistered human-factors relationship.

### C. Temperature field

Measure inlet, showerhead outlet and bather-plane temperature with synchronized, calibrated channels. If a spatial thermal field is studied, preserve the individual positions and time response. Do not use a single outlet temperature to hide cooling between the head and the bather plane.

### D. Aerosol / droplet field

Aerosol or droplet-size work is a separate exposure study. Hot and cold shower aerosols can differ in concentration, particle size and temporal dynamics; a visually attractive spray or reduced flow is not evidence of lower inhalation exposure. Use competent aerosol instrumentation and a separately approved protocol. No human exposure trial is authorized by this document.

## 5. Deterministic map metrics

The symthaea-aqua-bench spray-field helper implements a small, dependency-free summary of a **complete, externally qualified deposition grid**. It is a descriptive data transformation, not a physical solver, calibration engine, safety assessor or task-quality oracle.

For cell i:
- area A_i is in cm² after unit conversion;
- collected mass m_i is in mg;
- deposition density d_i = m_i/A_i is in mg/cm²;
- target cells are defined by the preregistered task region.

Report these independently:

1. **Target-area coverage fraction:** sum of target-cell area meeting a preregistered deposition threshold, divided by all target-cell area.
2. **Mean target deposition density:** target collected mass divided by target area.
3. **Area-weighted target deposition coefficient of variation:** weighted standard deviation of target-cell densities divided by their weighted mean. Undefined when mean deposition is zero.
4. **Off-target share of mapped captured mass:** off-target mass divided by total mass in the complete mapped capture boundary.
5. **Total mapped mass and target/off-target subtotals:** retain raw totals so derived values can be audited.

Thresholds are inputs to the function, not embedded Aqua constants. Do not choose the threshold after seeing which design wins. Grid cell counts are not independent physical replicates; the independent unit is the complete randomized test run / specimen as defined in the statistical plan.

The map summary is invalid when:
- any cell is missing, duplicated or has non-positive area;
- any mass is negative or non-finite;
- the target or off-target region is absent;
- the deposited-water threshold is non-finite or non-positive;
- the total mapped mass is zero;
- the capture boundary has not been qualified as complete by the upstream evidence process.

Aqua's helper does not itself prove completeness. The run record must retain the Mycelix/Symtropy evidence binding and the external reconciliation of mapped collectors to the calibrated total-flow / resource-closure boundary.

## 6. Experimental design

### Development and confirmation separation

- Development runs may tune geometry and choose thresholds.
- Freeze geometry CAD/mesh revision, nozzle insert IDs, process, tolerances, surface finish, material and test method before confirmation.
- Hold out confirmation specimens, test days and operating profiles where practical.
- Randomize comparator order within blocks for specimen, day, water batch and operator.
- Record every failure, blockage, cleaning, retry and aborted run. Do not delete them as outliers without a preregistered rule.

### Manufacturing variation

Use at least three independently manufactured specimens per geometry in the exploratory phase to expose gross unit-to-unit variation. Determine the confirmatory sample size from pilot variance, a preregistered endpoint and power / confidence requirements; do not treat repeated readings from one specimen as additional specimens.

### Challenge matrix

At minimum include:
- pressure points in Section 3, limited by rated components;
- cold-start and stabilized operation;
- each declared task mode separately;
- measured local water-quality profiles, including the intended hardness / scale challenge profile;
- new, aged and post-clean states;
- partial blockage of a single nozzle and a sector;
- inlet-strainer loading;
- nozzle insert damaged or missing;
- minimum and maximum allowed installation orientation;
- manufacturing tolerance corners selected from the actual process capability data;
- temperature transient and declared ambient conditions;
- capture-boundary leak, collector saturation, missing cell, sensor drift and timestamp fault as data-quality negative controls.

Cleaning cycles must use a frozen procedure and record labour, tools, water, chemicals, waste and requalification results. The test must show whether performance returns to baseline after cleaning; it must not assume that visible unclogging implies restored performance.

## 7. Robust multi-objective optimization: no single vanity score

Rank designs using a Pareto view, not an opaque weighted sum. Required dimensions include:
- flow and whole-service fresh water;
- standard spray-force and coverage endpoints;
- Aqua task-region coverage and deposition uniformity;
- temperature stability / bather-plane temperature;
- low- and high-pressure behavior;
- off-target deposition / splash;
- scale and partial-blockage degradation;
- cleaning time and post-clean recovery;
- manufacturing yield and tolerance sensitivity;
- replacement cost, service time and consumables.

A candidate can be called better only when it is not worse on required hard constraints and demonstrates a preregistered, statistically supported improvement on at least one declared objective. If it trades one dimension against another, report the Pareto trade-off rather than hiding it in a composite score.

Use a **worst-case envelope** across allowed pressures, water profiles and manufacturing tolerances. A high nominal score that collapses at 150 kPa, after scaling or after one blocked nozzle is not robust superiority.

## 8. Research hypotheses to compare

| Concept | Hypothesis | Main falsifier |
|---|---|---|
| A0 pressure-compensating incumbent | Baseline reference, not assumed optimal | Exact SKU does not maintain its declared curve |
| A1 zoned passive manifold | Spatial allocation can improve target-region coverage per litre | Better target coverage requires excessive flow, force concentration or complexity |
| A2 replaceable nozzle cassette | Geometry can be tuned and serviced without replacing the body | Seals, tolerance stack, scale or service access degrade performance |
| A3 passive fluidic / flow-shaping element | Passive geometry can shape coherent streams without powered valves | Pressure sensitivity, noise, clogging or poor coverage dominates |
| A4 fixed + handheld mode architecture | Different tasks benefit from distinct delivery geometries | User switching burden or total simultaneous demand erases benefit |
| A5 deterministic selectable zones | Selecting a qualified zone improves task efficiency | Added valves, stagnation, failure and cleanability burden exceed task gain |

Start with A0 comparator metrology and passive A1/A2 concepts. Add powered variable zones only after passive designs establish a measurable gain.

## 9. Initial benchmark targets and honest claim rules

- Retain 6.0 L/min at 300 kPa as a practical engineering gate, not a novelty claim.
- Retain 4.5 L/min at 300 kPa as a stretch investigation point, explicitly acknowledging Brondell's manufacturer-declared 4.5 L/min mode.
- The actual research win is non-inferior or improved task-region coverage, standard force/coverage endpoints, temperature stability and maintenance recovery against exact competitor SKUs at matched pressure points.
- Require evidence at 150 kPa; report 140 kPa separately for WaterSense-method comparison.
- Test 310 and 550 kPa only under the appropriate rated envelope; otherwise disclose the gap.
- Set the non-inferiority margin and required confidence interval before confirmation testing with independent human-factors / endpoint input.
- Do not claim global SOTA until the comparator set covers exact models and modes, methods are matched, independent replication is available, and the result is reproducible.

No product target is a compliance limit. No synthetic map is physical validation. No standard-method pass is a health, aerosol, hygiene or universal comfort claim.

## 10. Evidence and ownership

- Aqua #12692 owns geometry, product-specific spray map, design comparison and task context.
- EPA / ASME / CSA methods own their standardized performance methods and certification boundaries.
- Mycelix owns channel identity, calibration, timebase, metrology admission and uncertainty evidence.
- Symtropy owns canonical physical resource closure and any hydraulic / thermal simulation or physical model it already maintains.
- #12703 and competent independent laboratories own wet-path hygiene and aerosol / water-quality qualification.
- Independent human-factors evaluators own the relevant task-quality and non-inferiority instruments.

The Aqua helper may validate grid shape and summarize measured values. It must not create a duplicate metrology authority, water ledger, CFD solver, hygiene score, safety limit or human-comfort oracle.

## 11. Sources and current standard register

- EPA WaterSense showerhead overview: https://www.epa.gov/watersense/showerheads
- EPA WaterSense Specification for Showerheads v1.1: https://19january2021snapshot.epa.gov/sites/static/files/2018-07/documents/ws-products-specification-showerheads-v1-1.pdf
- ASME A112.18.1/CSA B125.1 listing (2024 edition): https://www.asme.org/codes-standards/find-codes-standards/plumbing-supply-fittings-%28with-10-18-errata%29
- Shower aerosol modeling paper, Building and Environment (2025): https://doi.org/10.1016/j.buildenv.2025.112825
- WRC 2025 South African showerhead evidence summary: https://www.wrc.org.za/wp-content/uploads/mdocs/WW%20July%20-%20Sept%202025_web.pdf

## 12. Claim ceiling

This is a preregistration and data-analysis protocol. It does not demonstrate spray-field performance, user comfort, hygiene, aerosol safety, compliance, service life, manufacturing readiness or product superiority. It authorizes no purchase, construction, powered prototype, occupied installation or human-exposure study.
