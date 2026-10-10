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
- **Engineering facade:** the new regenerative screening module independently checks process mass/carbon/heat closure, applies user-specified hard constraints, and refuses to put candidates with a failed or unknown product-quality gate on the Pareto frontier. Scenario candidates remain visibly scenario-tagged.
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
- rejection of NaN/infinite, negative, out-of-range, overflow and physically inconsistent carbon inputs;
- typed evidence references for feedstock data, empirical process parameters, thermophysical properties, reactor design, stream composition, recovery parameters and plant-availability parameters; result objects retain these references;
- tests for mass/carbon balances, thermal duty, sub-zero ambient temperature, invalid/missing evidence, nutrient recovery bounds, and serialization.

`crates/domains/symthaea-engineering/src/regenerative.rs` adds:
- candidate assessments against versioned yield, carbon-retention, supplied-heat, cost and water requirements;
- independent re-checks of mass, carbon and declared heat-duty identities;
- strict same-unit cost comparisons, provenance retention, and a quality gate that fails closed on unknown status;
- a sorted, deterministic Pareto frontier that retains trade-offs instead of hiding them inside a weighted composite score.

The test fixture values are **illustrative scenarios, not recommendations or measured plant performance**. The modules are not claimed as compile-verified until CI passes. Evidence IDs currently link to caller-managed immutable records; next integration should verify those receipts against Symthaea's shared evidence plane and require schema/version identifiers for each parameter group.

## Acceptance criteria for the next milestone

- [ ] Rust formatting and focused crate tests pass.
- [ ] Unit tests verify zero/unknown distinctions, property conversions, mass closure and error bounds.
- [ ] An independent calculation reproduces sample results.
- [ ] At least one versioned, licensed public nutrient dataset and one representative local soil/biomass dataset are ingested with source, units, date and uncertainty.
- [ ] Every synthetic value in demos is conspicuously marked as scenario data.
- [ ] No application-rate recommendation is exposed before soil, amendment safety and crop-specific validation gates pass.
- [ ] A physical pilot has documented mass/energy balances, safety controls, laboratory characterization and pre-registered trial design before any agronomic effectiveness claim is made.

The long-term goal is to discover whether durable, safe and affordable terra preta-inspired soil systems can improve specified outcomes while reducing dependency on imported synthetic nutrients. That question should be answered by reproducible calculations, scientific experiments and farm economics—not by the sophistication of the AI or the elegance of the visualization.
