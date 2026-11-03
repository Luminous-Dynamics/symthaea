# InstantClean sensor-ablation and adaptive-control benchmark

**Status:** preregistration template / research protocol. This document is not a test result, performance claim, certification, or authorization to change appliance actuators.

## 1. Research question

Can a sensor-informed, physics-constrained observer estimate the required *item-level outcomes* reliably enough to select a lower-resource cleaning process than fixed-duration and incumbent adaptive dishwasher controls, while abstaining safely under uncertainty, sensor faults, and distribution shift?

The benchmark must test whether additional sensing adds decision value. It must not assume that more sensors or a higher prediction score produces a safer or more efficient appliance.

The operational objective is:

> Minimize total measured service resources per unique item released after every required independent endpoint passes, subject to hard quality, hygiene, material, electrical, hydraulic, thermal, and safety constraints.

The primary unit of analysis is a complete physical wash run. Items, coupons, image crops, video frames, acoustic windows, and repeated measurements from the same run are nested observations, not independent physical runs.

## 2. Competitive boundary (3 November 2026)

Adaptive sensing is already an incumbent feature, so "AI wash" or "turbidity sensing" is not a differentiator by itself.

- **LG AI SenseClean (2026 EuroCucina announcement):** LG describes a digital turbidity sensor used during pre-wash, main-wash and rinse; its system can adjust temperature, rinse count and detergent amount across up to 32 combinations. Manufacturer announcement: https://www.lg.com/global/newsroom/news/home-appliance-solution/lg-electronics-to-showcase-new-dishwasher-lineup-at-eurocucina-2026/
- **Samsung Bespoke AI Dishwasher (21 April 2026 announcement):** Samsung describes a high-resolution turbidity sensor plus machine learning, measuring food particles through the cycle and adapting washing/rinsing time. Manufacturer announcement: https://news.samsung.com/global/samsung-expands-kitchen-portfolio-with-intelligent-performance-and-refined-design
- **GE Clean Sensor:** GE describes laser turbidity and temperature sensing used to adapt wash time, temperature, water quantity and fill count. Manufacturer support page: https://products.geappliances.com/appliance/gea-support-search-content?contentId=22177
- **Industrial adjacent evidence:** a November 2026 clean-in-place (CIP) study reports an ultrasound-plus-machine-learning cleaning-performance predictor with R² = 0.882 and reports ultrasonic features outperforming image-derived features in its tested CIP application. This is not a dishwasher result and does not establish that ultrasound can read residual soil on glass, ceramic, steel or polymer dishware. It is a reason to run a contained feasibility experiment, not a transferable performance target. DOI 10.1016/j.jwpe.2026.111025: https://www.sciencedirect.com/science/article/pii/S2214714426015837
- **Industrial mechanism-resolution context:** a 2026 food-processing study used image-derived cleaning-mechanism features to classify cohesive separation, adhesive detachment and viscous shifting, reporting 97.9% accuracy on its independent experimental test data and then analysing more than 800 industrial CIP runs. Its task was mechanism classification in heat-exchanger cleaning—not item-level dishwasher cleanliness. https://www.sciencedirect.com/science/article/pii/S0960308526002142

All OEM values above are manufacturer-described capabilities, not independent proof of false-release rate, endpoint accuracy, or measured service-resource savings. The CIP studies are cross-domain research leads, not direct competitors or dishwasher targets.

## 3. Separate the measurands

Do not collapse all observables into one universal cleanliness score.

| Endpoint | Primary measurand | Ground-truth method class | Explicit non-equivalence |
|---|---|---|---|
| Soil removal | Residual soil mass/concentration by soil family and substrate | Qualified soil-specific extraction, gravimetry, or validated optical/chemical assay | Turbidity is not item-surface soil |
| Final-rinse chemistry | Bulk rinse carryover and item-surface residue | Formulation/analyte-appropriate chemical assay and validated surface extraction | Conductivity alone does not measure all non-ionic residue |
| Cross-transfer | Tracer or soil transferred to initially clean witnesses | Validated tracer/chemical assay with recovery and mass balance | Low turbidity does not rule out local transfer |
| Drying | Retained moisture by item/material | Gravimetric or validated moisture method | Shine/spot score is not dryness |
| Microbiological effect | Organism-specific log reduction | Qualified microbiological challenge method | Relative reduction is not automatically sanitization |
| Material condition | Surface roughness, mass loss, coating/glass/plastic change | Appropriate material metrology | A clean-looking item may still be damaged |

Use endpoint-specific criterion IDs, method IDs, protocols, units, uncertainty and decision guard bands. An invalid or missing assay is unknown, never a pass. Analysts should be blinded to controller tier and competitor identity until reference outcomes are locked.

## 4. Sensor tiers and baselines

Run the same preregistered randomized physical-run allocation through these tiers wherever technically possible:

| Tier | Inputs | Purpose |
|---|---|---|
| B0 | Fixed reference cycle and timer | Non-adaptive control |
| B1 | Turbidity-only adaptive policy | OEM-style adaptive baseline |
| B2 | Turbidity + temperature + measured pressure/flow + pump RPM/current/real power | Process-state fusion |
| B3 | B2 + ultrasound/acoustic features | Test incremental value of acoustic sensing |
| B4 | B2/B3 + validated surface-sensitive fluorescence/optical sensing, if qualified for that soil/material | Test item-surface information |
| B5 | Actual commercial adaptive dishwasher programme | Product-level comparator, separately identified by exact SKU, firmware, market and programme |

B1 is a controlled turbidity-only research baseline unless an OEM product/policy is obtained and run. Do not present a reconstructed policy as the manufacturer's proprietary algorithm. B5 is the actual product comparator.

Keep controller capacity, development budget, tuning opportunities and data access comparable across sensor tiers. Include feature-ablation and sensor-dropout experiments so a tier cannot appear better only because it received more tuning or easier data.

### Sensor interpretation limits

- Turbidity measures properties of circulating liquid and suspended material; it cannot directly establish that a specific item surface is clean.
- Conductivity responds to ionic species and water chemistry; it is not a universal assay for all surfactants or rinse-aid residues.
- Pressure, flow, pump current and power describe process state and energy delivery, not soil removal.
- Ultrasound/acoustic features may respond to deposits, flow, bubbles, wall coupling or geometry; their sensitivity and specificity must be measured for each physical path.
- Fluorescence/optical channels require soil × substrate × surface-finish validation, optical access, repeatability and fouling controls.
- Cross-transfer requires initially clean witness surfaces and a validated tracer/reference assay. No process proxy can certify its own absence.

## 5. Two-stage ultrasound feasibility gate

Before adding ultrasound to an appliance or controller:

**Stage U1 — transducer/material feasibility**
1. Identify a defined measurement path: pipe-wall deposit, witness coupon, water column, or item surface. Do not mix these as one sensor type.
2. Establish calibration, coupling repeatability, bandwidth, temperature sensitivity, bubble/flow interference, material/substrate effects, and cleaning-agent compatibility.
3. Compare the feature against blinded, independent reference assays across clean, near-threshold, and deliberately soiled states.
4. Include negative controls for no-soil, bubbles/air, flow changes, temperature changes, transducer decoupling, surface finish changes, and known residue without suspended particles.
5. Reject the channel if the response is not specific/repeatable enough to add information beyond B2.

**Stage U2 — incremental decision value**
1. Lock B2 and B3 before evaluation.
2. Compare B3 against B2 on the same grouped holdouts.
3. Measure whether ultrasound improves endpoint prediction calibration, false-accept risk, uncertainty/abstention, or service resources—not just training fit or R².
4. Require a reproducible improvement on an external test session or independent build before integration is considered.

If ultrasound does not add measurable decision value after cost, calibration, maintenance, size and power are included, remove it. The benchmark rewards simpler sufficient instrumentation.

## 6. Experimental design and data separation

### Physical factors

Use the factor dictionary in `experiments/sensor-ablation-factor-template.csv`. The initial design should cover, within validated safe ranges:

- sensor tier and controller policy;
- soil family, soil lot, application mass and age at wash start;
- substrate/material, finish, item geometry and spatial location (direct, oblique, shadow, concave, low-point);
- water batch, hardness, inlet temperature and supply pressure;
- detergent formulation/lot and actual delivered mass;
- nozzle/build, measured flow/pressure and wash/rinse phase;
- occupancy/load and operator handling;
- environmental/session factors and deliberate sensor/process faults.

Begin with a small preregistered mechanism-screening design. Use blocked or split-plot randomization where unit, nozzle build or soil preparation is hard to change. Record aliasing and the true randomization unit. One-factor-at-a-time testing is not sufficient for important interactions; an unqualified full factorial is likely wasteful.

### Data partitions

1. **Development:** feature selection, model comparison and policy discovery.
2. **Calibration:** probability/interval calibration and threshold selection.
3. **Locked confirmation:** fixed allocation and one-time evaluation after all artifacts are frozen.
4. **External replication:** new physical unit/build, session/day, operator, site or assay laboratory, with the exact axis named.

Every sensor trace, image, crop, item result and assay derived from one physical run must remain in one partition. Hold out complete runs and groups such as unit, test session, soil lot, detergent lot, water batch, operator and laboratory. Do not call a new day on the same machine a cross-unit test.

Sequential design may query development data only. The locked confirmation allocation is generated and frozen before outcomes are observed. Any repeated looks require a preregistered sequential-testing plan.

## 7. Outcomes and metrics

### Endpoint prediction

For continuous residual/rinse measurements report MAE, RMSE, bias, prediction-interval coverage and width, calibration by stratum, and performance near the acceptance boundary. For categorical decisions report confusion matrices, sensitivity/specificity, Brier/log loss where applicable, calibration and abstention.

Do not rank models by R² alone. A model with high overall R² can still be unsafe near the clean/not-clean boundary or on a rare material/geometry stratum.

### Release assurance

Define the physical-run false-accept event before analysis:

> The controller issues a release decision for a run in which at least one required item/endpoint fails its independently measured criterion.

Report separately:
- false acceptance conditional on a deliberately nonconforming challenge run;
- false release over the operational run population (which also depends on failure prevalence);
- false rejection/rewash and abstention;
- per-endpoint and worst-stratum event rates;
- upper confidence bounds and the exact independent physical-run denominator.

For a zero-event binomial illustration only, 299 IID independent nonconforming challenge runs with zero false accepts give a one-sided 95% upper bound of approximately 1%; 2,995 give approximately 0.1%. These are not guaranteed sample sizes under clustering, multiple strata, adaptive looks, invalid assays or non-IID conditions. Use run/block-aware inference, independent-group counts and an appropriate multiplicity plan. Items/coupons/frames from one cycle do not multiply the physical-run sample size.

### Service resources

Use the service-accounting contract to charge all attempts, failed runs, interrupted runs, preheat/standby/purge, pre-treatment, and rewashes to the verified successful-item denominator. Report separately:
- appliance wall electricity and upstream hot-water energy;
- fresh water, including fills/rinses/purges/rewashes and manual pre-treatment;
- actual delivered detergent and rinse aid;
- active operator time, elapsed time and time-to-release;
- maintenance/fouling and material-damage burden;
- accepted items/hour at declared utilization for batch products.

A quality/safety failure cannot be compensated by a lower resource score. Publish raw Pareto points after feasibility filtering; do not hide a candidate behind an arbitrary weighted scalar.

## 8. Decision and early-stop policy

The observer is advisory until the independent assay, grouped generalization, locked confirmation and external replication gates pass.

Use the documented supervisor states:
- **Continue:** evidence supports continuing the validated process.
- **StopCandidate:** observer predicts the endpoint is likely satisfied; this is not itself a release.
- **FallbackToValidatedProcess:** sensor uncertainty, fault, or envelope violation invokes the full validated reference process.
- **Abstain:** contradictory, stale, missing, out-of-range or uncalibrated signals provide no safe decision.

During development, StopCandidate remains shadow-only. Inject sensor disconnect, stuck/stale values, drift, occlusion, fouling, conflicting sensors, pump/valve faults, pressure/voltage disturbances, and water/soil distribution shifts. Confirm that failures trigger fallback/abstention, not false release.

No sanitization, disinfection or sterilization claim follows from a soil sensor, process model, hot-water setpoint or relative microbiological result.

## 9. Competitive success criteria

A competitive claim must compare exact products/programmes on matched soil, material, geometry, item count, inlet conditions and service boundaries.

Minimum decision gates:
1. All required independent endpoints pass with uncertainty guard bands.
2. No critical false-accept evidence appears on locked confirmation; upper confidence bounds and per-stratum denominators are published.
3. Grouped holdout and external replication support the exact scope of the claim.
4. At equivalent verified outcomes, InstantClean reduces full service energy, water, time, operator effort or chemical use—or measurably improves cleaning without worsening the other constrained outcomes.
5. Added sensors deliver incremental decision value that justifies their cost, failure modes, calibration and maintenance.

Existing project hypotheses such as at least 10% lower total service energy or fresh water at equivalent endpoints, or at least 5% better independent soil removal at no greater total resources, may be used as preregistered screening targets. They are not established SOTA margins, standards, or product results. The 2026 mechanistic detergent/cycle optimization paper reports 31% shorter wash time, 59% shorter drying time and 5% higher stain-removal index on its reference case, but its data are confidential; reproduce a matched reference before claiming superiority. https://doi.org/10.1016/j.compchemeng.2026.109771

For a QuickCell use case, retain separate household single-item/small-batch and commercial throughput leaderboards. A fast glasswasher rack cycle is not directly comparable to a household cycle without accepted-item, utilization, loading, start-up, idle, operator and service resource normalization.

## 10. Required artifacts

- This frozen benchmark protocol and factor dictionary.
- Exact sensor and comparator register, including SKU/firmware/market.
- Run-level randomization manifest and partition assignment.
- Raw synchronized sensor traces and calibration records.
- Independent endpoint assay records and uncertainty budgets.
- Frozen model, controller, calibration, configuration and software digests.
- Sensor-ablation report with per-stratum errors, calibration, risk-coverage, false-accept bounds and service-resource Pareto set.
- External replication report and explicit non-claims.

**Current status:** protocol authored; no physical sensor study has been run; no ultrasound sensor has been validated on dishware; no InstantClean adaptive-control or early-stop performance is claimed. Cargo tests and formatting for the codebase remain separate prerequisites and are not implied by this document.
