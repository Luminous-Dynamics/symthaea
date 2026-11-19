# Luminous Aqua — Benchmark Positioning and Breakthrough Roadmap v0.1

**Snapshot:** 19 November 2026  
**Status:** research strategy and proposed benchmark targets; no comparative Aqua physical results  
**Companion:** [Pressure-Transient and Supply-Recovery Protocol v0.1](../AQUA_PRESSURE_TRANSIENT_AND_SUPPLY_RECOVERY_PROTOCOL_V0_1.md)  
**Evidence corpus:** [Aqua evidence corpus v0.1](AQUA_EVIDENCE_CORPUS_V0_1.md)

## 1. Strategic thesis

Do not try to win the showerhead category by advertising a smaller litres-per-minute number. The industry already has efficient low-flow products, established spray-quality measures, pressure-compensation criteria, whole-system drain-water heat recovery (DWHR) products and consumer testing.

The higher-value opportunity is to define and demonstrate a new product class:

> **A household bathing service whose water, thermal, quality, reliability, recovery and maintenance performance is qualified across a measured operating envelope, with reproducible evidence for every claim.**

This moves the unit of comparison from an isolated outlet to a completed task delivered by a household system. A low-flow head, recirculating shower, DWHR system, mixer and hot-water plant are not interchangeable product categories. Compare like-for-like configurations and boundaries; show the complete system result separately.

## 2. What current SOTA and industry benchmarks actually cover

### 2.1 EPA WaterSense — independently certified outlet performance

WaterSense Version 1.1 requires a maximum flow rate of 2.0 US gpm for the label and independent certification. Its performance framework includes minimum-flow retention at specified inlet pressures, spray force and spray coverage. The published minimum-flow thresholds are 60% of maximum at 20 psi and 75% at 45 and 80 psi. Minimum spray force at 20 psi is 0.56 N for high-efficiency / handheld showerheads and 0.40 N for high-efficiency rain showers. These categories and test methods must be matched when benchmarking.

Sources:
- EPA, [Showerheads and WaterSense performance overview](https://www.epa.gov/watersense/showerheads)
- EPA, [WaterSense Specification for Showerheads v1.1](https://19january2021snapshot.epa.gov/sites/static/files/2018-07/documents/ws-products-specification-showerheads-v1-1.pdf)
- EPA, [Performance overview: Showerheads](https://www.epa.gov/system/files/documents/2022-05/ws-products-perfomance-showerheads.pdf)

**What this does well:** establishes independent, user-relevant outlet performance checks, rather than treating low flow as success by itself.

**What Aqua should add:** time-varying supply, concurrent fixture loads, interruption and recovery, completed-task resource closure, household hot-water topology, serviceability and machine-readable trace provenance. These are additional claims, not reasons to bypass or weaken WaterSense methods.

### 2.2 UK Energy Technology List — non-domestic shower performance and user behaviour

The current ETL shower specification explicitly covers non-domestic showers. For the defined categories it includes maximum available flow (9 L/min without user-behaviour features and 10.6 L/min with them), pressure independence and spray coverage. The pressure-independence requirement is at least 60% of maximum flow at 1.5 bar relative to a 3 bar reference, where the reference flow is no greater than 8 L/min. The method uses the specified UWL P2 datum at the faceplate centre and reports flow after stabilization. It also recognizes user-behaviour features such as timed shutoff and water/temperature breaks.

Source: UK DESNZ, [Energy Technology List — Showers](https://etl.energysecurity.gov.uk/products/showers/showers)

**What this does well:** makes pressure-independence, faceplate datum, stabilized flow, spray coverage and resource-saving user features explicit.

**What Aqua should add:** a time-domain qualification alongside the stabilized test, with exact event profiles, independent task outcomes and recovery costs. ETL's scope is non-domestic; do not present it as a domestic South African certification.

### 2.3 Consumer Reports — broad consumer-facing performance evaluation

Consumer Reports' January 2026 buying guide describes testing that considers coverage, spray force, water consumption and ease of use. It highlights that strong performance can coexist with less than 1.8 gpm consumption in some tested products.

Source: [Consumer Reports — Showerhead Buying Guide, updated 28 January 2026](https://www.consumerreports.org/cro/showerheads/buying-guide/index.htm)

**What this does well:** combines physical performance measures with user-facing ease of use and product comparison.

**What Aqua should add:** disclose the complete SKU/configuration, matched site and supply conditions, measurement uncertainty, raw trace evidence, interruption-adjusted resource totals, life-cycle maintenance and the independent assessor behind each task-quality claim.

### 2.4 Utility continuity benchmarking — supply availability is an external condition

The Pacific Water & Wastewater Association's 2025 benchmarking report includes continuity-of-supply hours/day for utilities, using the latest available year per utility. It reports many systems at 24 hours/day and some systems at 2–4 hours/day. The report cautions that the year varies and data should be validated. These are utility-level service indicators, not household pressure waveforms or shower-product performance tests.

Source: PWWA, [Benchmarking Report 2025](https://assets.zyrosite.com/YX4azLpX2VhDW692/pwwa_benchmarking_2025_final-report_v2-W9S59N1SnvsVUyRC.pdf)

**What Aqua should add:** distinguish upstream service availability from fixture performance. Measure the incoming condition at the site and do not claim that a fixture can create water during a complete supply outage. Qualify graceful interruption, storage/backup interfaces only where declared, safe restoration and verified completion when supply returns.

### 2.5 DWHR and domestic hot-water research — optimize the complete system

A 2026 *Energy and Buildings* study of drain-water heat recovery coupled with domestic electrical and heat-pump storage systems reports experimental global COP up to 6.3 and simulated COP up to 8.0 for its studied configurations. Its main lesson is that storage, heat pump and heat recovery must be considered together; an exchanger's component effectiveness is not the same as whole-system efficiency.

Source: Bouvenot & Beaudet (2026), [Optimal coupling of shower drain water heat recovery systems with domestic electrical and heat pump storage water heaters](https://doi.org/10.1016/j.enbuild.2026.117761)

A 2025 *Journal of Cleaner Production* study of Australian and UK domestic hot-water plumbing configurations reports substantial storage standing losses and a large fraction of shower energy leaving in drain water in the studied cases. These values are configuration-specific, not universal design constants.

Source: Hall et al. (2025), [Trade-offs between total and peak energy consumption due to plumbing layouts in Australian and UK domestic hot water systems](https://doi.org/10.1016/j.jclepro.2025.145941)

**What Aqua should add:** compare mixer-side, tank-side, balanced and no-DWHR topologies on one whole-system boundary. Measure purchased electricity/fuel, useful hot-water heat, tank/pipe standing loss, pump/control parasitics, drain recovery, fouling and the task actually delivered. Do not optimize heat-exchanger percentage in isolation.

### 2.6 Domestic plumbing transients — the fixture is part of a network

Marsili et al. (2026) tested a highly instrumented full-scale household minor-system replica with fast solenoid closures at multiple domestic outlets and high-frequency pressure acquisition at 2 kHz. The authors report significant transient pressure variations within the plumbing system, terminal branches as critical locations, and a case where the adjacent branch peak was 1.6 times the peak at the event-generation outlet. These are results from a particular laboratory replica, not a universal residential design pressure or a showerhead rating.

Source: Marsili et al. (2026), [Role of water service lines in the propagation of user-induced transients](https://doi.org/10.1016/j.measurement.2026.120358)

**What Aqua should add:** capture pressure at the event location and at adjacent branches in a separately qualified test rig. Never infer the system's maximum stress from the first peak at the activated valve alone.

## 3. The real gap: no single scorecard joins these layers

The established landscape is strong but fragmented:

| Benchmark / field | Strongest contribution | Distinct gap Aqua should address |
|---|---|---|
| WaterSense | Independently certified flow retention, spray force and coverage | Time-domain service, household demand disturbances, recovery and full-task accounting |
| UK ETL / UWL | Stabilized flow, pressure independence, spray coverage, user-behaviour features | Dynamic pressure/time envelope and post-interruption service restoration |
| Consumer Reports | Consumer-facing comparison of coverage, force, consumption and ease of use | Public raw measurement provenance, site-specific transient profiles, whole-system energy and repair economics |
| Utility continuity indices | Supply hours/day and service availability at utility level | Product-level task completion conditional on the actual incoming supply trace |
| DWHR literature | Heat recovery and whole-system DHW topology analysis | Matched, repeated bathing-task measurements including fouling, storage and parasitics |
| Plumbing transient research | Fast pressure propagation and material/diameter effects | A reproducible link from transient exposure to shower task/service outcomes and qualified component limits |

Aqua must not claim to replace certification schemes, consumer labs, utility benchmarks or independent human-factors testing. The opportunity is a traceable *integration benchmark* that makes their distinct measurements composable without collapsing their authority.

## 4. Proposed Aqua Complete-Task Benchmark (Aqua-CTB)

The unit of comparison is one **qualified completed bathing task**, under a declared task profile and incoming supply profile.

For every attempt, record:

- task outcome: completed, incomplete, aborted or invalid; outcome assessor and method revision;
- fresh water: task delivery plus interruption, purge, restart, cleaning and make-up task water;
- energy: purchased electricity/fuel, useful hot-water heat, distribution/storage loss and declared pump/control parasitics;
- service: pressure, flow and temperature states throughout the event, interruption count/duration and return-to-qualified-service time;
- human factors: reach, control usability, temperature stability, user workarounds and independently assessed experience;
- maintenance: time to inspect, clean, descale, replace wear parts and return the unit to service;
- evidence: exact SKU/variant, device configuration digest, installation, site and event profile, instruments, calibration receipts, raw trace SHA-256, analysis code digest and protocol revision.

Report per-attempt and per-qualified-completed-task denominators. Failed attempts cannot disappear from resource totals. Never divide task water by only the successful subset without publishing failed-task rates and resource use.

### 4.1 Independent, non-compensatory gates

Do not compress the entire result into one weighted score. A high water-saving score must never offset a thermal-limit violation, failed task, hygiene blocker, invalid calibration or exceeded apparatus rating.

Use separate result dimensions:

1. **Safety / validity:** all apparatus, pressure, temperature and measurement evidence gates passed.
2. **Task:** independent completion and quality outcome.
3. **Water:** fresh water per qualified completed task, including recovery and cleaning.
4. **Energy:** purchased energy and useful heat per qualified completed task.
5. **Resilience:** task outcome across site-derived T2 events and pressure/flow/temperature recovery tails.
6. **Durability / repair:** wear, scale/fouling, maintainability, repair time and life-cycle cost.
7. **Evidence quality:** provenance completeness, uncertainty, reproducibility and independent confirmation.

A result with missing critical evidence is Indeterminate, not a low score.

## 5. Proposed research targets — not standards or achieved results

| Dimension | Existing reference | Aqua internal target / stretch objective |
|---|---|---|
| Flow efficiency | WaterSense maximum 2.0 US gpm; market has products below this | Keep the existing AQS-1 candidate target of 4.5–5.5 L/min at 300 kPa only if task quality remains non-inferior; this is a research target, not a universal user requirement |
| Pressure retention | WaterSense minimum 60% at 20 psi and 75% at 45/80 psi; ETL category uses 60% at 1.5 bar vs 3 bar | Research stretch of at least 80% at 20 psi and 85% at 45/80 psi, with the same pressure datum and test method; no certification claim |
| Spray field | WaterSense and UWL spray force/coverage methods | Non-inferior to best exact comparator on category-appropriate force and coverage, with the complete spatial distribution and uncertainty published |
| Task water | Consumer/label systems typically emphasize flow and duration | At least 20% lower fresh water per qualified completed task versus the best exact comparator, including purge/cleaning/restart, without task-quality regression |
| Purchased energy | DHW/DWHR research reports whole-system COP for defined configurations | At least 20% lower purchased energy per qualified completed task for a matched whole-system configuration; independently close the energy boundary |
| Service continuity | Standards mainly test controlled steady points; utility systems measure continuity hours/day | Non-inferior task completion over every preregistered site-derived event family |
| Recovery tail | No common cross-industry fixture P95 recovery threshold identified in this review | At least 20% lower P95 time outside qualified service after event restoration, without thermal or task-quality regression |
| Transient characterization | 2026 full-scale plumbing study used 2 kHz acquisition in its test rig | Freeze sample rate/bandwidth from a measurement-bandwidth analysis; 2 kHz is an initial T3 research reference, not a universal minimum |
| Repair / lifecycle | Consumer product comparisons may mention ease of use, but rarely normalize repair per successful task | 20% lower life-cycle cost per qualified completed task as a stretch goal, with parts, labor, cleaning and replacement disclosed |
| Evidence | Independent test/certification exists in established schemes | Machine-readable, versioned evidence passport with exact product/SKU, raw trace hashes, calibration, analysis code, negative controls and independent replay |

Every target must be evaluated with exact comparator, market SKU/variant, method, installation, task and site conditions. Sample size must follow a preregistered precision/power calculation from pilot variance. A larger number of samples from one physical specimen is not a substitute for independent specimens.

## 6. Paradigm shifts that can change the category

### A. From product to service envelope

Publish the complete tested envelope: steady pressure points, measured transient profiles, concurrent loads, temperature and task outcome. Label tested, untested and out-of-scope regions. Do not interpolate a continuous guarantee from sparse pressure points.

### B. From low-flow to quality-constrained resource minimization

Minimize water and purchased energy only subject to completed task, spray quality, thermal stability, accessibility and hygiene constraints. The objective is not “least flow”; it is “least total resource burden for a qualified task.”

### C. From isolated outlet to network-aware device

Aqua should characterize how the product both responds to and contributes to the household network. Measure terminal branch stress, cross-fixture interference, pressure recovery and pump/control parasitics. Any mitigation needs independent physical qualification; this programme grants no automatic actuator authority.

### D. From average performance to tail reliability

Report median, P95, worst observed, failure rate and uncertainty by event family and site. Preserve rare, severe events. A favorable mean cannot hide recurring task interruption or a pressure/temperature excursion.

### E. From manufacturer claims to an evidence passport

Publish a compact, machine-readable record for every qualified SKU and configuration: product identity, protocol revision, test lab, raw trace digests, calibration/uncertainty receipts, conditions, results, failed/invalid runs, analysis code and claim ceiling. Independent labs and auditors should be able to replay the evidence. A digest is a binding, not a substitute for verifying the artifact bytes or authenticating its issuer.

### F. From component optimization to whole-life system performance

Co-optimize outlet, mixing, hot-water storage, heating, DWHR, water quality, serviceability and user behaviour at the correct system boundary. Report component metrics for diagnosis, but judge savings at the complete task and lifetime boundary.

## 7. Benchmark execution ladder

1. **Software conformance:** schema validation, raw-file SHA verification, negative tests, exact-head Cargo tests, formatter and Clippy. These are necessary but not physical evidence.
2. **Metrology and fixture methods:** independent calibration, WaterSense/UWL/ETL method crosswalk, pressure/flow/temperature instruments, spray field and uncertainty budget.
3. **Unoccupied hydraulic replay:** qualified T2 events and dedicated T3 apparatus with spatial pressure channels; retain raw data, even for failed runs.
4. **Whole-system thermal/resource closure:** compare DWHR and non-DWHR configurations with measured purchased energy and task heat.
5. **Independent task and human-factors assessment:** assess completed service, comfort, accessibility and usability under an appropriate approved protocol.
6. **Durability and serviceability:** scale/fouling, cleaning, wear-part replacement, leakage and lifecycle cost.
7. **Independent confirmation:** hold out product batches and event profiles; use a second lab/apparatus and publish the reproducibility interval.

## 8. Claim discipline

The cited benchmarks establish the capabilities of those methods and schemes; they do not establish that Aqua exceeds them. Aqua has no physical comparative result in this protocol. Internal targets are hypotheses. The programme should say “designed to test,” “authored,” “structurally checked,” “software validated,” “physically measured” and “independently qualified” as distinct states.

The intended breakthrough is to make a complete household bathing task independently measurable across its supply, thermal, network, maintenance and evidence boundaries—not to overclaim that a new test protocol is already a new product performance record.
