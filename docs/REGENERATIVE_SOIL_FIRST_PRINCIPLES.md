# Regenerative Soil Systems: First-Principles Science and Engineering Design

Status: engineering research plan plus an initial deterministic Rust accounting implementation (2026-10-10).

This initiative connects Symthaea's engineering reasoning to Sol Atlas's nutrient-cycle planning and Mycelix's provenance/coordination layer. It deliberately separates physical identities from empirical soil-response claims.

## Executive decision

**Yes: use first principles, and put the model-building and experiment-design workflow in Symthaea.** Do not try to derive all soil behavior from physics alone. First principles provide conservation laws, stoichiometry, thermodynamic bounds and dimensional constraints. Material properties, reaction rates, nutrient availability, microbial ecology, field-scale transport and crop response require measured inputs or empirically calibrated constitutive models.

The useful design pattern is a hybrid model:
1. **Conservation kernel** — exact arithmetic for mass, elemental carbon/nitrogen/phosphorus/potassium, and energy balances.
2. **Physics/chemistry models** — heat transfer, reaction thermodynamics, sorption, water flow and transport, using explicitly sourced property data.
3. **Empirical ecological layer** — decomposition, mineralization, nitrogen fixation, microbial responses, plant uptake, yield and long-term carbon stability, calibrated with suitable experiments.
4. **Decision layer** — multi-objective engineering optimization with hard safety, mass-balance, cost, energy and capacity constraints.
5. **Evidence layer** — every parameter has units, provenance, uncertainty, valid range and status; every output is reproducible and is not presented as more certain than the input evidence.

The first implementation is in `crates/domains/symthaea-agribot/src/soil_process.rs`: an explicit-input biomass pyrolysis mass/carbon/heat-duty calculation and nutrient-recovery mass balance. It is an accounting primitive—not a validated reactor simulator or a fertilizer prescription.

## What first principles can and cannot determine

### Derivable from balances and unit identities

For wet feedstock mass \(m_w\) and wet-basis moisture fraction \(x_w\):

\[
m_{dry}=m_w(1-x_w),\qquad m_{water}=m_w x_w
\]

Given an empirically observed or scenario-assumed char yield \(y_c\) on a dry-feed basis:

\[
m_{char}=y_c m_{dry}
\]

With measured elemental carbon mass fractions \(x_{C,f}\) for feedstock and \(x_{C,c}\) for char:

\[
R_C=\frac{m_{char}x_{C,c}}{m_{dry}x_{C,f}}
\]

This carbon-retention ratio is a mass-balance result conditional on the material analyses and char-yield input. It is not the same as a validated multi-century soil-carbon persistence estimate.

For a simplified heat duty, using sourced specific heat capacities \(c_p\), target temperature \(T_p\), ambient temperature \(T_a\), water boiling temperature \(T_b\), latent heat \(L_v\), and explicit heat-transfer efficiency \(\eta\):

\[
Q_{dry}=m_{dry}c_{p,dry}(T_p-T_a)
\]

\[
Q_{water}=m_{water}[c_{p,l}(T_b-T_a)+L_v+c_{p,v}(T_p-T_b)]
\]

\[
Q_{reactor}=m_{reactor}c_{p,reactor}(T_p-T_a),\qquad
Q_{supplied,estimate}=\frac{Q_{dry}+Q_{water}+Q_{reactor}}{\eta}
\]

The initial Rust function computes these declared terms only. It explicitly excludes reaction enthalpy, detailed product/exhaust heating, reactor transients, pressure-dependent phase behavior, syngas energy recovery and other plant losses. Treat the result as a transparent screening estimate, **not** a complete heat requirement or a machine design rating.

For a recovered nutrient stream of volume \(V\), elemental concentration \(C_i\), process recovery fraction \(\eta_i\), and period-specific plant-available fraction \(f_i\):

\[
m_{i,recovered}=V C_i\eta_i,\qquad
m_{i,available}=V C_i\eta_i f_i
\]

For N, P and K, the arithmetic is simple; the hard questions are whether the concentrations were sampled representatively, the process truly recovers that fraction, the product passes safety requirements, and how much is crop-available during the relevant season. Those fractions must be measured or calibrated per product and process, never set as universal biological constants.

### Not derivable from feed mass alone

No conservation equation by itself supplies an empirical char yield, char surface chemistry, cation exchange capacity (CEC), available phosphorus, ammonia loss, nitrogen immobilization, soil aggregation, microbial community function, root uptake, yield response or decades-long carbon persistence. These need analytical chemistry, well-specified mechanistic models, literature priors with applicability limits, and/or experimental calibration.

A 2024 systematic review drawing on 1,073 datasets from 316 publications reports that biochar properties change with feedstock, temperature and modification; property trends trade off against one another rather than forming one universally optimal recipe. A 2024 global meta-analysis of the soil-carbon cycle found increased carbon sequestration on average, but effects on respiration and total CO₂ flux were more uncertain. A separate global meta-analysis reported average reductions in soil inorganic nitrogen after biochar addition, with strong dependence on feedstock, temperature, application, pH and fertilizer interactions. These results are precisely why an accounting model must not contain a universal 'biochar efficacy' multiplier.

Research:
- Qi et al. (2024), systematic review of biochar for soil degradation, 1,073 datasets / 316 publications: https://doi.org/10.1111/gcbb.13147
- Bekchanova et al. (2024), rapid review/meta-analysis of biochar and the soil carbon cycle: https://doi.org/10.1007/s42773-024-00381-8
- Meta-analysis of biochar effects on soil available inorganic nitrogen: https://doi.org/10.1016/j.geoderma.2016.11.004
- FAO nutrient balance methodology, including the use of experimentally derived availability/equivalence factors rather than assuming all nutrient mass acts like mineral fertilizer: https://www.fao.org/4/y4620e/y4620e09.htm

## Terra preta: what are we trying to reproduce?

Terra preta de Índio is not a recipe with a single specification. Published reviews describe anthropogenic Amazonian dark earths as the result of mixtures of pyrogenic carbon, organic matter, and inorganic/mineral materials—including ash and bone-derived inputs—transformed through long periods of soil biological and geochemical processes. The role and timeline of some formation processes remain uncertain.

Source: Glaser & Birk (2012), *State of the scientific knowledge on properties and genesis of Anthropogenic Dark Earths in Central Amazonia*: https://doi.org/10.1016/j.geoderma.2011.09.014

We should not claim to manufacture a century-old soil on a short timeline. Instead, design **region-specific regenerative soil systems inspired by the functions of terra preta**: durable carbon storage, stable and safe nutrient retention, good soil structure and water behavior, resilient microbial processes, and reliable crop performance.

Could our engineered product be better? **It is plausible to outperform a benchmark on a selected, measurable objective**—for example, yield stability per dollar, available nutrient recovery, reduced nutrient loss or water productivity—in a given crop/soil/climate. It is not yet justified to claim overall equivalence or superiority to mature terra preta, particularly for centuries-long persistence, ecosystem complexity or multi-generational soil resilience. The honest goal is measured, function-specific performance against appropriate controls and real terra preta/anthropogenic dark-earth references when lawful and scientifically comparable material and data are available.

## Symthaea's role: design the process, not just analyze it

Symthaea already has distinct engineering capabilities: `symthaea-thermofluids` for applied heat/flow calculations, `symthaea-materials` for material properties, `symthaea-digital-twin` for telemetry/state, `symthaea-operations-research` for optimization primitives, `symthaea-engineering` as the engineering reasoning facade, and `symthaea-agribot` as the ecological/soil interface. The new `soil_process` module begins with deterministic, auditable calculations in the agribot domain.

Recommended division:
- **Rust deterministic kernels:** balances, unit conversions, dimensional checks, numerical stability and process constraints.
- **Thermofluids and digital twin:** staged heat duty, heat-exchanger/heat-recovery scenarios, moisture control, sensors, process telemetry and calibration against a physical unit.
- **Materials models:** source-specific composition and measured characterization (ultimate/proximate analysis, ash/minerals, pH, electrical conductivity, surface area, pore-size distributions, CEC, H/C and O/C ratios, relevant contaminants).
- **Operations research:** choose feedstock blend, throughput, energy source/heat recovery, process conditions and product blends under explicit constraints; compute Pareto fronts rather than a single opaque 'best' recipe.
- **Engineering facade:** the regenerative screening module independently checks process mass/carbon/heat closure, applies user-specified hard constraints, and refuses to put candidates with a failed or unknown product-quality gate on the Pareto frontier. Scenario candidates remain visibly scenario-tagged.
- **Climate inventory:** candidates can include explicit emission, removal and avoided-emission flows, each with evidence, plus a char-storage eligibility decision and durability fraction at a declared horizon. Climate can be a Pareto objective and an optional hard constraint; missing/incomplete inventory or unknown storage eligibility makes the climate objective indeterminate rather than silently zero.
- **Causal reasoning and Symthaea planning:** recommend the next informative experiment and compare counterfactuals. Keep causal hypotheses distinct from demonstrated effects.
- **Independent evaluator:** recompute balances and check feasibility without trusting the agent that proposed a design.
- **Sol Atlas:** map verified suppliers, crop seasons, nutrient flows, facility capacity, demand, transportation, and uncertainty.
- **Mycelix:** coordinate signed provenance/attestation for feedstock lots, laboratory reports, processing records, shipments and trial outcomes. Decentralized records preserve assertions but do not establish that a physical sample or claimed test is valid.

## Engineering design problem

The target process is not 'maximum biochar yield'. It is a constrained multi-objective design:

**Inputs to characterize**
- sustainable feedstock availability by season and competing uses;
- feedstock moisture, dry matter, elemental composition, ash/mineral composition, contamination and particle-size distribution;
- energy and water availability, local emissions requirements, operators/maintenance capacity and logistics;
- local soil texture, pH, salinity, nutrient status and water limitation;
- target crops, planting windows, harvest removal, yield and economics.

**Process variables to explore**
- feedstock-specific preprocessing and moisture reduction;
- controlled pyrolysis profile, heating rate, residence time and oxygen exclusion;
- heat recovery and clean use of process gases where feasible;
- safe post-processing and conditioning with characterized compost or recovered nutrients;
- nutrient recovery and application timing; and
- blends selected for the target soil rather than one recipe for the planet.

**Constraints and measurable outputs**
- close wet mass, dry-solids and elemental carbon balances within defined tolerances;
- report uncertainty and provenance for every empirical parameter;
- no use of contaminated/unsafe feedstocks or products that fail applicable limits;
- quantify process heat, auxiliary energy, water, throughput, labor, cost, emissions and transport;
- characterize biochar stability indicators, pH, salinity, ash/minerals, plant-relevant nutrients, CEC/sorption and contaminants;
- quantify short- and long-term nutrient availability, leaching/runoff and relevant greenhouse gases;
- compare plant performance and farm net returns under replicated trials.

## Experiment strategy: learn cheaply, fail safely

1. **Parameter acquisition first.** Establish source-specific feedstock and material analyses before choosing a process model. Mark missing values as unknown; use scenario ranges only for exploration.
2. **Bench-scale screening.** Run small, instrumented batches over a designed set of process conditions. Close mass and elemental carbon balances; characterize the resulting materials with suitable laboratories. Use competent operators and appropriate controls for heat, fire, combustible gases, condensate and emissions.
3. **Pre-register the soil experiment.** Use a representative soil and crop, untreated/standard-practice controls where appropriate, randomized replicates, and treatments that separate biochar, compost/nutrients, and their combination. Keep conventional agronomy and safety constraints intact.
4. **Measure what would falsify the design.** Include poor germination, nutrient immobilization, salinity/pH damage, contaminant failure, excess leaching, high emissions/energy cost, yield loss and negative farm economics as explicit failure outcomes.
5. **Use multi-season validation.** Incubation and one harvest are useful screening but are not proof of durable soil-carbon storage or mature terra preta-like behavior. Validate predictions on held-out trials/sites before generalizing.
6. **Promote only evidence-backed recipes.** A recipe is valid only within its characterized feedstock/process/soil/crop range. A changing feedstock lot creates a new configuration requiring characterization and perhaps recalibration.

## First implementation delivered in this branch

`crates/domains/symthaea-agribot/src/soil_process.rs` adds:
- validated wet/dry feedstock, char-yield and elemental-carbon calculations;
- an explicitly limited sensible-heat + water vaporization duty estimate with all property inputs visible;
- separate nutrient-stream elemental mass, recovered mass, and seasonally available mass calculations;
- an evidence-gated batch climate inventory that keeps gross emissions, other removals, avoided emissions and horizon-specific char-storage credit separate;
- no panic fallback for missing carbon-storage parameters, and no complete net climate result when the inventory is partial or carbon-storage eligibility is unknown;
- typed evidence references for feedstock data, empirical process parameters, thermophysical properties, reactor design, stream composition, recovery parameters and plant-availability parameters; result objects retain these references;
- tests for mass/carbon balances, thermal duty, sub-zero ambient temperature, invalid/missing evidence, nutrient recovery bounds, climate accounting and serialization.

The engineering layer also exposes an evidence-linked interval screen for yield, carbon retention, heat, cost, water, and optional climate constraints. Its status describes numeric threshold relations only; it does not override the separate product-quality gate. A full interval must satisfy a constraint for a robust pass; a threshold-crossing interval remains indeterminate. Bounds are supplied by callers and are not automatically treated as statistical confidence intervals.

A first measurement-prioritization utility now lives in `crates/domains/symthaea-engineering/src/measurement_planner.rs`. It ranks explicit measurement options only if they target an unresolved numeric constraint, using a declared relevance weight, an explicit benefit fraction (interval narrowing or acquisition of missing required data), and same-unit measurement cost. It retains unranked options and their dispositions so a useful-looking test is not silently discarded. The missing climate-inventory case can be prioritized for data acquisition, and a climate assay targets the active hard climate threshold when one is declared.

This is an intentionally transparent **triage heuristic**, not an optimal design-of-experiments (DoE) algorithm or calibrated value-of-information calculation. Its interval-reduction estimates, relevance weights, and assay costs are caller inputs with evidence references. It does not account for correlated parameters, statistical power, treatment randomization, interactions, or the probability that an experiment changes the decision. The next research step is to use a proper screening DoE where many factors exist, and then a sequential Bayesian / information-gain method once calibrated data and defensible likelihood models are available.

Research grounding:
- NIST/SEMATECH, [choosing an experimental design](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm) and [screening designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3346.htm): selection depends on objective and number of factors; screening designs identify important effects but do not replace confirmatory experiments.
- NIST, [Sequential Bayesian Experiment Design](https://www.nist.gov/programs-projects/sequential-bayesian-experiment-design): sequential measurement can guide observations toward parameter uncertainty reduction when suitable probabilistic models are available.
- Kumar et al. (2025), [field biochar effects on soil microbial biomass carbon](https://doi.org/10.1007/s42773-024-00391-6): the meta-analysis of 539 paired observations found strong context dependence and called for longer-term field work, reinforcing why site- and outcome-specific evidence must remain explicit.

The agribot layer now also has an independent measured-stream mass-closure assessor. This is important because the simple pyrolysis calculator derives its non-char dry-mass residual by subtraction; that arithmetic identity is not proof that a physical reactor's measured output streams close. The observed-stream assessor requires measured evidence for all enumerated inlet/outlet masses, rejects duplicate/missing stream IDs and unmeasured scenario streams, preserves the boundary/tolerance evidence, and reports the signed residual and relative closure error.

`crates/domains/symthaea-engineering/src/regenerative.rs` adds:
- candidate assessments against versioned yield, carbon-retention, supplied-heat, cost and water requirements;
- independent re-checks of process mass, carbon and declared heat-duty identities;
- strict same-unit cost comparisons, provenance retention, and a quality gate that fails closed on unknown status;
- optional lifecycle-climate objective and net-climate hard limit, with incomplete climate inputs marked indeterminate;
- a sorted, deterministic Pareto frontier that retains trade-offs instead of hiding them inside a weighted composite score, and includes climate only when explicitly required.

### Conservative interval screening

The engineering module now also exposes `MetricInterval`, `RegenerativeMetricIntervals`, and `assess_regenerative_uncertainty()`. This path evaluates constraints against caller-supplied lower/upper bounds. A constraint passes only when its entire interval is on the feasible side of the threshold; it fails robustly only when its entire interval violates the threshold; an interval crossing the threshold is indeterminate. Missing climate bounds remain indeterminate when climate comparison is required.

These bounds are **not automatically generated confidence intervals**. Each interval must cite evidence describing whether it represents measurement uncertainty, a calibrated prediction interval, or a conservative scenario envelope. The current method does not infer distributions, estimate covariance, run Monte Carlo propagation, discover causal effects, or generate an optimal experiment plan. It is a deliberately conservative screening primitive, not a substitute for those future capabilities or for field validation.

The lifecycle function is **not** an ISO-conformant life-cycle assessment or a carbon-credit verifier. It credits char storage only when the caller provides verified biogenic-sourcing eligibility and an explicit, evidence-linked durability fraction for a declared horizon. It does not infer emissions from feedstock carbon absent from char and does not invent counterfactual credits.

Relevant research:
- Qi et al. (2024), systematic review of 1,073 biochar datasets from 316 publications: biochar properties vary with feedstock, temperature and modification, supporting product-specific characterization: https://doi.org/10.1111/gcbb.13147
- 2024 biochar LCA review: lifecycle emissions depend on feedstock, production, application and product properties; system boundaries and uncertain soil-carbon effects matter: https://doi.org/10.1016/j.scitotenv.2024.175448
- 2025 phosphorus review: biochar effects on P availability vary across soils and application conditions: https://doi.org/10.1007/s42773-024-00415-1
- FAOSTAT cropland nutrient balances report N, P and K flows from synthetic fertilizer, manure, atmospheric deposition, crop removal and biological fixation through 2023: https://www.fao.org/statistics/events/events-detail/cropland-nutrient-balance.-global--regional-and-country-trends--1961-2023/en

The test fixture values are **illustrative scenarios, not recommendations or measured plant performance**. The modules are not claimed as compile-verified until CI passes. Evidence IDs currently link to caller-managed immutable records; next integration should verify those receipts against Symthaea's shared evidence plane and require schema/version identifiers for each parameter group.

## Preregistered bench-scale screening design

The new `symthaea_engineering::screening_design` module creates an executable **schedule specification** for a bounded numeric-factor screening experiment. The initial implementation is intentionally a two-level full factorial rather than a fractional factorial: two to six quantitative factors, complete blocks, a fixed randomization seed, independently randomized treatment order within each block, and an exact copy of the declared request in the plan output. This yields (2^k) unique low/high combinations, with all combinations present in each block. The planner is bounded at 1,100 scheduled runs.

Before schedule generation, the request must identify the preregistration and input snapshot, objective, primary hypothesis, analysis-plan ID, each factor's low/high values and evidence references, blocks, independent replicate counts, and one primary endpoint with a minimum practically meaningful difference. Each factor must also declare whether its setting can be randomized independently run-by-run. It also requires a declared bench-scale review approval tied to the exact protocol ID, input-snapshot ID, and SHA-256 fingerprint of the versioned design-input payload. The fingerprint includes the factor ranges and randomization classes, blocks, seed, replicates, center points, hypothesis, analysis plan, and endpoint, plus the schedule algorithm version; changing those inputs invalidates the reviewed digest. The review object is excluded from its own hash to avoid a circular digest. Scenario-only review evidence is rejected. This gate only allows generation of a *bench-scale schedule*—it does not authorize equipment operation, a field trial, product release, or soil application. The current generator accepts only factors declared per-run randomizable. It rejects factors declared restricted or hard-to-change because split-plot and other restricted-randomization designs are not implemented. This class is a caller assertion, not an automatically inferred property of the equipment; the reviewer must verify that the run-order design is feasible. The current API checks the declared review status, non-scenario evidence class, evidence-ID presence, exact protocol ID, exact input-snapshot ID, and digest equality, but does not authenticate or resolve the review record against an independent evidence service. SHA-256 binds content; it does not prove who approved it. Until that trust-plane integration exists, the caller's approval assertion remains an external input rather than a verified authorization.

Each block contains every factorial treatment setting. Treatment runs and optional center-point controls are jointly Fisher–Yates shuffled with a versioned SplitMix64 pseudo-random stream; the seed and algorithm ID are retained so the schedule is reproducible. Center-point controls can be disabled or set to 3–5 per block; when used, every factor is at its numerical midpoint. Randomizing them together with treatment runs avoids always placing controls at the beginning and end, which could confound control status with time/order drift. Center points can help monitor stability and curvature, but do not replace a response-surface study.

The generated schedule can be passed to a separate structural verifier that rechecks contiguous run order, unique IDs, complete treatment/replicate coverage per block, factor levels/units, and center-point placement without calling the generator. This guards against malformed or corrupted plan artifacts, but it runs in the same crate and does not independently authenticate the review record or prove that its source is trustworthy.

There is an important resource trade-off: a full factorial preserves all declared low/high combinations and allows the treatment combinations to be retained without fractional-design aliasing, but the number of settings grows as 2^k. NIST notes that when screening five or more factors, fractional factorial or Plackett–Burman designs are often the more resource-efficient starting point; this implementation does not yet generate those designs. For a high-factor or run-budget-constrained problem, do not blindly run this full factorial—first select a proper fractional design and explicitly document its alias structure. There are explicit limits: this implementation accepts numeric factors only, does not generate fractional designs for many-factor screens, does not calculate power or sample size from variance, and does not analyze responses. The declared meaningful difference is not automatically a detectable effect. The statistical analysis, treatment feasibility, instrument capability, randomization protocol, adverse-event/stopping rules, and product/worker/environment safety review still require an informed reviewer before execution.

Methodological sources:
- NIST/SEMATECH, [selecting an experimental design](https://www.itl.nist.gov/div898/handbook/pri/section3/pri33.htm): design choice depends on objectives and factor count; screening and response-surface objectives need different designs.
- NIST/SEMATECH, [two-level full factorial designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3331.htm): a (2^k) design covers every combination of k factors at low/high levels.
- NIST/SEMATECH, [blocking full factorial designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3333.htm): nuisance variation can be managed by blocks, with explicit attention to effects potentially confounded with blocking.
- NIST/SEMATECH, [adding center points](https://itl.nist.gov/div898/handbook/pri/section3/pri337.htm): center points can assess process stability and curvature and should be distributed across the experiment.

## Preliminary power planning

The optional `symthaea-engineering::screening_power` module adds a first-pass power check for **main effects only**. It requires an evidence-backed lower/upper residual-standard-deviation range in the same unit as the predeclared primary endpoint, a named source context and method, familywise alpha, and target power. Scenario-only variance inputs are rejected, and the upper SD bound is used for projected power and required replication. The formula adjusts familywise alpha with Bonferroni over the k main effects and uses the factorial-contrast variance relationship

[
\operatorname{Var}(\widehat{\text{main effect}})=\frac{\sigma^2}{n\,2^{k-2}},
]

where (n) is the number of complete-factorial replicates (blocks multiplied by within-setting replicates). It estimates current power and required complete-factorial replications using a two-sided normal approximation.

This is a **preliminary screen, not a claim of adequate power**. The computation uses the upper SD bound conservatively but treats that bound as known; it does not calculate exact finite-sample noncentral-t power or establish statistical coverage for the SD bounds, account for block-by-treatment interaction, evaluate interaction effects, include center points in the factorial contrast, or correct for secondary endpoints and post-hoc exploration. It assumes independent, balanced, common-variance errors and additive block effects. Use context-relevant pilot data and a defensible upper SD bound; the interval is not automatically a confidence interval. Obtain statistical review before selecting replication or declaring a confirmatory trial adequately powered.

Useful methodological sources:
- NIST's [sample size guide](https://itl.nist.gov/div898/handbook/prc/section2/prc222.htm) explains that sample size depends on alpha, beta/power, effect size and standard deviation; it warns that a standard-deviation assumption is required.
- Penn State STAT 503 derives the two-level factorial main-effect variance relationship and explains how replication and blocking alter the error term: [factorial effects and variance](https://online.stat.psu.edu/stat503/Lesson06) and [blocking in replicated factorial designs](https://online.stat.psu.edu/stat503/Lesson07).
- USDA-ARS guidance describes power analysis as depending on effect magnitude, error variance, alpha and beta: [Power and replication](https://www.ars.usda.gov/ARSUserFiles/3122/PirkEtAl2013.pdf).

## Closing the screening loop with measured outcomes

The new \`symthaea-engineering::screening_analysis\` module connects the verified schedule to observed primary-endpoint outcomes. Each planned run must have exactly one observation with a unique observation ID, exact run ID, the preregistered endpoint ID, the exact endpoint unit, the preregistered measurement method, a finite outcome value, and a non-empty evidence ID marked **Measured**. Missing runs, duplicate IDs, unknown run IDs, method/unit mismatches, scenario-labelled outcomes, or a structurally corrupted schedule cause analysis to fail closed.

The analysis reports descriptive factorial contrast estimates for every main effect and interaction in the declared full factorial, with the original design request and ordered observation records retained. For center-point designs it also reports, per block, the center-point mean minus the factorial-treatment mean. Randomization and complete blocks support a future inferential model, but these outputs are descriptive estimates only: no p-values, confidence intervals, causal claims, or crop-efficacy conclusions are generated. The center-point difference is not itself a formal curvature test.

This stage deliberately does not impute missing outcomes. A missing observation should trigger the missing-data handling declared in the preregistered analysis plan, and that handling must be implemented and reviewed separately. The next scientific software milestone is a separately specified inferential analysis (including explicit residual degrees of freedom, model diagnostics, multiplicity and treatment-by-block checks), validated against an independent statistical implementation before anyone uses it to claim detection or efficacy.

## Evidence update and implications for the model

A focused literature refresh adds an important constraint: the correct target is not “maximize biochar” but “choose a safe, economically viable intervention for a specific soil × crop × climate × management context.”

- Bekchanova et al. (2024) systematically reviewed 92 articles / 1,609 observations focused on sandy-textured soils. Their synthesis reported positive average responses for several nutrient-cycle indicators, but no average effect on soil mineral nitrogen or nutrient-use efficiency; heterogeneity and publication-bias sensitivity were material, including a sign change for effective CEC after correction. This is a strong reason to model outcome-specific effects and uncertainty, not one universal “soil improvement” score: https://doi.org/10.1186/s13750-024-00326-5
- A 2025 review comparing six biochar standards reports shared attention to feedstock restrictions, total carbon / H:C characterization, heavy metals and PAHs. The product-quality gate should therefore be a versioned set of analyte-specific checks, not a single boolean asserted by the process simulator: https://doi.org/10.1016/j.biteb.2025.102059
- A 2025 field-trial-focused review of biochar combined with other amendments highlights the need to distinguish biochar alone, amendment alone, and the combination with proper controls: https://doi.org/10.1007/s42773-025-00531-6

### Model changes this evidence motivates

1. **No universal effect coefficients.** Store soil/crop/context-stratified evidence, study design, duration, sample size, effect estimate, confidence interval, and risk-of-bias assessment. Do not transport a mean effect to a new site without an explicit applicability check.
2. **Quality gates become typed and analyte-specific.** Track test method, units, lab/sample identity, detection limit, threshold source/version, and result. Unknown, missing, or stale mandatory tests must remain indeterminate, not pass.
3. **Evaluate interventions factorially.** Where feasible, compare untreated control, standard practice, biochar alone, nutrient amendment alone, and biochar + amendment. Track interaction effects instead of attributing every combined-treatment benefit to char.
4. **Report uncertainty and downside risk.** Alongside point estimates, retain uncertainty intervals and a worst-case / sensitivity view. A candidate that looks best only under optimistic assumptions should not dominate the recommendation.
5. **Keep the objective vector explicit.** Crop yield and stability, plant-available nutrients, nutrient losses, water productivity, lifecycle greenhouse-gas balance, contaminant risk, cost per hectare, and labor/logistics are distinct objectives. A Pareto frontier is more honest than one weighted score unless weights are explicitly chosen and justified.
6. **Model nutrient balance at field and regional scales.** Imported fertilizer is only one input. Track manure/compost, biological N fixation, deposition, irrigation inputs, crop removals, losses, and soil-stock changes to distinguish a genuine nutrient deficit from a distribution or affordability problem.

These are research-driven design requirements, not claims that the current code already implements the full statistical, laboratory, or agronomic stack.

## Acceptance criteria for the next milestone

- [ ] Rust formatting and focused crate tests pass.
- [ ] Unit tests verify zero/unknown distinctions, property conversions, mass closure and error bounds.
- [ ] An independent calculation reproduces sample results.
- [ ] At least one versioned, licensed public nutrient dataset and one representative local soil/biomass dataset are ingested with source, units, date and uncertainty.
- [ ] Every synthetic value in demos is conspicuously marked as scenario data.
- [ ] No application-rate recommendation is exposed before soil, amendment safety and crop-specific validation gates pass.
- [ ] A physical pilot has documented mass/energy balances, safety controls, laboratory characterization and pre-registered trial design before any agronomic effectiveness claim is made.

The long-term goal is to discover whether durable, safe and affordable terra preta-inspired soil systems can improve specified outcomes while reducing dependency on imported synthetic nutrients. That question should be answered by reproducible calculations, scientific experiments and farm economics—not by the sophistication of the AI or the elegance of the visualization.


## Experimental validation protocol

The companion [validation and experimental design protocol](REGENERATIVE_SOIL_VALIDATION_PROTOCOL.md) defines evidence classes, randomized treatment comparisons, sample-size planning, batch-level provenance, safety gates, nutrient ledgers, and staged acceptance criteria. It is a research template, not an application-rate recommendation or evidence of field efficacy.
