# CropFoundry: crop intelligence and engineering research plan

**Status:** research proposal / benchmark contract v0; no crop model, breeding recommendation, or performance result is claimed by this document.  
**Owner:** Symthaea, with integration points for Symtropy, Mycelix, and Sol-Atlas.  
**Initial geography:** South Africa, beginning with Gauteng-relevant growing environments while keeping the schema region-neutral.  
**Initial crops:** tomato as a comparatively well-studied reference crop; one locally relevant indigenous/traditional leafy vegetable chosen from amaranth, *Cleome*, *Corchorus*, or *Vigna* based on accessible germplasm, partners, and usable observations. Sweet potato is a next-stage crop. Fruit trees are in the registry from the beginning, but their long evaluation cycles should not dominate the first proof.

## Thesis

Do not build a larger plant encyclopedia and call it an engineering system. Build a **closed-loop, evidence-aware crop engineer** that translates a stated food-system need into candidate interventions, ranks uncertainty, proposes the next most informative safe experiment, and only updates recommendations from traceable observations.

The loop is:

1. **Specify the need.** Define the target population, local environments, farmer and consumer constraints, acceptable trade-offs, and non-negotiable safety conditions.
2. **Assemble evidence.** Link germplasm identity, genotype (where available), traits, experimental methods, soil, weather, management, costs, and post-harvest outcomes. Preserve missingness instead of silently imputing facts.
3. **Model and compare.** Start with transparent baselines and established crop simulators; add mechanistic/statistical/ML hybrids only where they beat baselines on held-out data.
4. **Choose experiments by value of information.** Prefer the experiment expected to reduce a decision-relevant uncertainty, not the most novel-looking model output.
5. **Measure independently.** Use pre-registered protocols, replicated trials, calibrated instruments, blinded sensory work where relevant, and explicit negative results.
6. **Update with provenance.** Store source data, transformations, code/model versions, uncertainty, reviewer decision, and the exact evidence receipt supporting every released claim.

A simulation result is a hypothesis generator. It is not proof of plant performance, food safety, or a new cultivar.

## Paradigm shifts worth pursuing

### 1. Engineer outcomes, not isolated traits

Optimize a transparent Pareto frontier across marketable edible yield, water and nutrient inputs, resilience, nutrition, flavor, storage loss, labor, cost, soil/ecological effects, and seed accessibility. Do not compress every objective into one opaque scalar. Report dominated candidates, constraints, uncertainty, and who benefits or pays.

### 2. Treat genotype × environment × management as the unit of prediction

A cultivar is not simply “drought resistant” or “high yielding.” Performance is conditional on genotype, environment, management, and time. Hold out complete site-years (and, when possible, genotypes) during evaluation to expose leakage and poor transfer.

### 3. Replace data hunger with decision-focused active learning

Estimate which next measurement or field trial would most change the choice of cultivar, cross, or management plan. Compare active selection with random and conventional selection under equal trial budgets. The first active-learning results should be simulated; real-world claims require real observations.

### 4. Join mechanisms with empirical evidence

Use mass balance, phenology, soil-water dynamics, light interception, and known biological constraints to reject impossible or internally inconsistent predictions. Hybridize only when useful. Treat established crop models as benchmarks/adapters, not as unquestioned ground truth.

### 5. Design crop portfolios, not one universal winner

Optimize seasonally complementary crops, cultivar diversity, rotations, and risk diversification at farm or regional scale. A portfolio can be more resilient than maximizing the mean yield of one crop. Include local seed availability, labor peaks, perishability, and household food preferences.

### 6. Make uncertainty and failure first-class outputs

Every recommendation carries a prediction interval, applicability domain, provenance, counterevidence, and missing-data profile. The system must abstain when the test environment falls outside validated coverage. Preserve failed trials and negative results to avoid repeating expensive mistakes.

### 7. Treat farmers and communities as co-designers

Include farmer-defined utility, local culinary knowledge, indigenous crop priorities, labor realities, seed sovereignty, and consent. Do not scrape or publish sensitive location or traditional knowledge without appropriate permissions. Public data does not automatically authorize unrestricted redistribution of every derived dataset.

## Additional paradigm shift: optimize breeding programs, not just candidate rankings

A July 2026 review, [Click Breeding: Toward programmable crop design](https://www.sciencedirect.com/science/article/pii/S1360138526002104), argues that genomic prediction alone can struggle to generalize across environments for complex polygenic traits and describes AI agents that generate and evaluate multi-generation breeding strategies under explicit objectives, constraints, and uncertainty. A 2026 systematic review of agricultural digital twins likewise highlights continuing interoperability and field-validation gaps ([review](https://doi.org/10.1016/j.atech.2026.102283)).

Treat these papers as research directions, not proof that general autonomous breeding agents already work. CropFoundry therefore adds CF-013, a **multi-generation breeding-policy value** benchmark. It is explicitly not evaluable until reliable genotype, pedigree and phenotype histories—and a defensible retrospective simulator or program—exist. First compare sequential policies with a greedy candidate-ranking baseline on synthetic and historical replay under equal budgets, with strict future-information leakage prevention. A simulated improvement is not a realized breeding gain; prospective agronomic validation remains a separate gate.

This suggests a useful long-term transition:

```text
rank current candidates
        ↓
plan a constrained breeding policy over generations
        ↓
select the next experiment with high decision value
        ↓
measure actual offspring and environments
        ↓
update uncertainty and next-generation decisions
```

Each arrow needs its own evidence contract. If pedigree, genotype, license, or phenotype history is absent, the system must say so and stop short of a breeding-policy claim.

A second, separate design shift is to optimize **decision quality rather than prediction score alone**. CF-014 compares candidate recommendations against a declared baseline using decision regret under held-out environment/price scenarios, with hard constraints and sensitivity to utility assumptions. A slightly less accurate yield predictor may still make better planting or resource-allocation choices; conversely, a model with low average error can be harmful when errors cluster around high-cost extremes. Begin with synthetic and retrospective replay, then require local cost data and prospective evaluation before asserting economic benefit.

A third long-horizon research direction is **cold-start transfer to underrepresented crops**. A shared representation of plant physiology, environment response, phenotype measurement, and management context could reduce the amount of trial data needed for an underutilized crop—but transfer can also create negative transfer when traits or measurement methods are not comparable. CF-015 therefore remains blocked until at least three crop domains have compatible, harmonized observations; it requires species/cultivar/site-year holdouts and an explicit target-crop-only baseline. The goal is to help data-sparse local crops, not to treat their lack of records as evidence that they are inferior.

A fourth long-horizon direction is **agroecological portfolio engineering**: optimize combinations of crops and their sequence over seasons, not only the genes or yield of one crop. The model should consider nutritional-basket coverage, land-equivalent ratio where scientifically appropriate, marketable output, seasonal covariance and shortfall risk, soil/resource budgets, labor peaks, market fit, and local cultural priorities. This is not a presumption that every intercrop beats monoculture; poor combinations or timing can create focal-crop yield penalties.

The literature justifies testing the idea while keeping local validation separate. A 2025 *Nature Communications* meta-analysis synthesizing 3,663 paired field-trial yield observations found that crop rotations increased total yield, dietary energy, several nutrients, and revenue relative to continuous monoculture across the analyzed studies ([paper](https://www.nature.com/articles/s41467-025-64567-9)). A July 2026 meta-analysis of maize-based intercropping systems in Africa reports a mean land-equivalent ratio of 1.45 across the analyzed combinations, with substantial crop-combination effects ([paper](https://www.sciencedirect.com/science/article/pii/S1161030126001620)). Neither figure should be copied as a guaranteed local gain or a Johannesburg forecast.

CF-016 therefore benchmarks tail-risk and nutritional-basket shortfall against declared baselines under matched land, water, labor, and cost constraints. It is blocked until usable interaction data or a clearly labeled synthetic scenario set exists, and physical claims require locally replicated multi-season trials.

A fifth direction is **causal intervention evaluation** rather than treating predictive association as treatment efficacy. CF-018 asks whether recommendations based on fertilizer, irrigation, or other crop treatments correspond to independently supported effects under documented assignment, experimental units, blocking, replication, and predeclared outcomes. If allocation is observational or the protocol is missing, the system may report associations but must not claim an identified treatment effect. This is especially relevant for DS-010: the dataset record states that 27 treatment combinations exist, but those labels alone do not demonstrate randomization or sufficient causal identification, and its referenced accompanying documentation is not listed among the Zenodo files. CF-018 therefore remains blocked until the design evidence is acquired and reviewed.

A sixth direction is **sequentially updated plant digital twins** rather than frozen predictors. The July 2026 *Trends in Plant Science* perspective describes virtual plants that integrate plant structure, physiology and environment and update through continuous data assimilation, while noting fidelity, scale and accessibility challenges ([review](https://doi.org/10.1016/j.tplants.2026.06.005)). CropFoundry adds CF-019 to test whether state assimilation improves horizon-specific forecasts and calibrated uncertainty against frozen-model and simple time-series baselines. It must enforce prequential chronology, event-time/receive-time separation, sensor-staleness and drift testing, and strict exclusion of future harvest outcomes. If timestamps, units, sensor calibration or observation windows are undocumented, the twin cannot be considered calibrated; CF-019 remains blocked until a pinned time-series profile exists.

## Additional paradigm shift: physiological signals before visible symptoms

Visible leaf color and fruit morphology are useful, but can be late or ambiguous indicators of stress. Thermal imagery, multispectral/hyperspectral bands, chlorophyll fluorescence, and RGB measure different aspects of plant–environment interactions. A 2025 crop-stress review discusses non-destructive sensing for stress assessment ([review](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1638675/full)); a dedicated thermal-imaging review also emphasizes interpretation and measurement issues that complicate field use ([review](https://doi.org/10.1016/j.compag.2024.109227)). These papers justify testing the modalities, not claiming that one modality always detects stress earlier.

- **CF-020 — cross-modal pre-symptomatic stress detection:** test whether time-aligned thermal/spectral/fluorescence data provide earlier warning at a fixed false-alert budget than RGB-only, single-sensor and threshold/persistence baselines. Pin stress ground truth independently from input channels; preserve sampling cadence, time of day, illumination, humidity, calibration and plant/plot grouping. Hold out entire site-seasons for transfer claims. Blocked until paired, labeled stress-sensor data are pinned.
- **CF-021 — low-cost calibration and transfer:** test whether affordable sensors retain actionable accuracy and uncertainty across devices, sites, seasons and drift versus calibrated reference instruments. Include recalibration frequency, missingness, labor, power, maintenance, replacement and connectivity in total cost of ownership. A cheaper sensor is not a useful substitute if its uncertainty changes the decision. Blocked until paired reference measurements and site/device metadata are pinned.

The broader paradigm is **measurement design as part of crop engineering**: choose the next sensor or observation according to expected value of information, the decision it could change, uncertainty, and fully loaded cost. A visually impressive model that adds no useful lead time or changes no decision should not win the benchmark.

## Additional paradigm shift: optimize water decisions under hard constraints, not soil-moisture prediction alone

A useful crop engineer must decide where and when to allocate scarce water under uncertain forecasts, sensor failures, unequal crop-stage sensitivity, pumping limits, and explicit water budgets. Irrigation ML reviews identify uncertainty quantification, field-scale applicability, physics-informed modelling, data availability and operational outcomes as enduring research needs ([Umutoni & Samadi, 2024](https://doi.org/10.1016/j.agwat.2024.108710); [precision-irrigation decision-support review, 2025](https://doi.org/10.1016/j.dte.2025.100038)). A 2026 study of probabilistic site-specific soil-moisture scheduling likewise discusses how uncertainty shifts with forecast horizon and compares weather-forecast approaches ([SWIM2 study](https://doi.org/10.1016/j.agwat.2026.110300)).

**CF-022** turns that gap into an executable algorithmic starting point: a deterministic synthetic harness compares equal-share calendar allocation, a soil-threshold baseline and an uncertainty-aware marginal-risk policy across a reference scenario, matched water-cap-only, forecast-bias-only and sensor-dropout-only ablations, plus heat/drought and compound-failure scenarios. Those matched controls isolate individual failure modes instead of conflating all of them. It checks the non-negotiable daily water cap and emits per-scenario synthetic stress-exposure and resource metrics. Its runnable implementation and limitations are specified in [CF-022_WATER_POLICY_SYNTHETIC_BENCHMARK_V0.md](./CF-022_WATER_POLICY_SYNTHETIC_BENCHMARK_V0.md).

The initial acceptance criterion is zero daily total-water-budget violations, zero per-plot actuator-limit violations, and deterministic replay. A provisional research target is at least 10% lower median aggregate synthetic stress-exposure proxy than the equal-share baseline on the declared scenario suite. Receipt schema v2 reports per-zone irrigation and stress, worst-zone burden, disparity and allocation dispersion, plus the number of scenarios contributing to each worst-to-best ratio. Review of the dose-matched runs also found that the v1 stress formula clipped negative end-of-step storage before scoring, which erased marginal penalties under severe unmet demand; v2 now preserves the negative water margin as unmet-demand severity while physical state remains clamped. All v1 stress results are historical debugging evidence and must not be compared numerically with v2. All v2 comparisons must still be read as synthetic only: the successful v2 smoke run showed a Pareto trade-off where risk-aware allocation minimized mean aggregate proxy, while greedy minimax improved the mean worst-to-best ratio (about 1.20 versus 1.42) at a higher aggregate proxy. These fixture outputs do not establish an agronomic winner. The harness also includes a dose-matched rotating-uniform counterfactual that redistributes the risk-aware policy's exact daily total across plots; this is a hindsight comparator, not a deployable policy, and isolates allocation pattern from total daily dose. Neither condition establishes biological benefit: the proxy, storage, demand, critical windows and environmental traces are invented test fixtures. The harness is useful now to test policy code, deterministic receipts, hard constraints and robustness to missing/bad inputs. It must not be presented as crop-water savings, yield performance or a farm recommendation. Aggregate score gains alone do not justify operational use; a crop-specific permissible worst-plot stress burden and allocation/fairness constraints must be preregistered before any field-stage trial.

The path to a serious irrigation result is layered: (1) synthetic contract and policy tests; (2) versioned, calibrated soil-water balance with uncertainty; (3) retrospective/hindcast comparison against fixed/calendar, threshold, ET-based and established crop-model baselines; (4) independent local calibration and sensitivity analysis; (5) preregistered replicated field trials reporting metered water, crop outcomes, pump energy, labor, full cost and environmental side effects. All policies must use the same weather inputs, resource caps, held-out scenarios and decision horizon. Report every scenario, not merely the average.

## Additional paradigm shift: optimize delivered food, not harvest yield alone

A crop engineer should optimize **safe, marketable edible output delivered**, not just biomass or harvest yield. Harvest maturity, handling damage, storage conditions, transport delay, water loss, sorting decisions, and affordability can dominate whether edible food reaches a household. Report the physical balance and stage-by-stage loss reasons instead of collapsing them into a single unlabeled "loss" scalar.

FAO's Food Loss Index covers losses between production and retail and its current portal reports a 2023 fruit-and-vegetable aggregate estimate of 25.4%. This is a global commodity-basket estimate, not a tomato-specific value or a Gauteng baseline ([FAO Food Loss Index](https://www.fao.org/sustainable-development-goals-data-portal/data/indicators/1231-global-food-losses/en)). FAO's 2026 technical platform also lists dedicated harvest/post-harvest measurement guidance for fruit and vegetables ([FAO measurement platform](https://www.fao.org/platform-food-loss-waste/en)). A 2025 review identifies substantial regional, commodity and supply-chain-stage gaps in available fruit/vegetable loss data ([review](https://doi.org/10.3389/fhort.2025.1529040)).

**CF-023** turns this into a staged benchmark: first reproduce a lot-level mass-balance and loss taxonomy; then evaluate forecasts on lot-disjoint and chronological splits; then test post-harvest interventions using controlled, independently replicated evidence. Mass retention, marketability, quality, measured nutrient retention and food safety are separate endpoints. Appearance is not a safety assay, and an image trajectory from one tomato is not a set of independent biological replicates.

A candidate source is the [passive-cooling-blanket tomato dataset](https://zenodo.org/records/15856413), which lists eight XLSX workbooks associated with a Uganda tomato supply-chain pilot and measurements including weight, firmness, color, soluble solids and acidity. Its visible Zenodo license field is not populated, and the workbooks have not been acquired or audited; DS-014 remains quarantined. Its study context is not proof of local South African efficacy.

The provisional prospective research target is at least 10% lower measured marketable-mass loss with non-inferior predeclared quality, zero additional safety-gate failures, and no increase in fully loaded cost/energy/emissions per marketable kilogram. This target must be reviewed against baseline variability, a power analysis and partner priorities and locked before outcomes are seen. The data contract, leakage controls, study stages and claim ceiling are specified in [CF-023_POSTHARVEST_EDIBLE_OUTPUT_BENCHMARK_V0.md](./CF-023_POSTHARVEST_EDIBLE_OUTPUT_BENCHMARK_V0.md). No intervention result or local loss rate is claimed.

## Additional paradigm shift: measure nutrition delivered per scarce resource

CropFoundry should move from optimizing yield/area to optimizing **verified safe nutrient delivery per unit of water, land, energy, cost and labor**, through harvest, storage and preparation. A nutrient concentration reading, harvested mass, edible servings and bioavailable nutrient delivery are distinct outcomes. No food basket should earn a good score by shifting shortfall to another nutrient or population group.

Recent synthesis proposes nutritional resilience as the joint preservation of productivity, nutrient density, bioavailability and dietary diversity ([Dissanayaka et al., 2026](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2026.1987519/abstract)). A large meta-analysis of elevated CO₂ effects covers 43 crops and 32 nutrients/antinutrients across 29,524 observation pairs ([ter Haar et al., *Global Change Biology*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12616468/)); a 2026 review likewise documents nutrient-specific and crop/stress-specific variation ([Bello et al., 2026](https://doi.org/10.3390/agriculture16111220)). This evidence justifies a separate benchmark objective; it does not prove CropFoundry can improve human nutrition.

**CF-024** formalizes nutrient mass after measured edible fractions and documented post-harvest/preparation retention, with any bioavailability factor separately sourced and uncertain. The comparison set must include mass-yield, nutrient-density, lowest-cost, diverse-basket and uncertainty-robust baselines under equal resource envelopes. Report every nutrient, requirement group, cost and resource constraint. Keep direct assays distinct from literature factors and model estimates; unknown is not zero. Do not infer nutrient content from images.

The proposal is blocked until source-backed assays, nutrient units and mass basis, edible/serving definitions, retention factors, bioavailability evidence where used, and population-specific requirements are pinned. A provisional 10% improvement target can only be considered after independent nutrition/domain review and pre-registration, with no new nutrient shortfall, affordability or diversity regression, or resource-limit violations. See [CF-024_NUTRITION_DELIVERY_RESOURCE_BENCHMARK_V0.md](./CF-024_NUTRITION_DELIVERY_RESOURCE_BENCHMARK_V0.md). A local-only Rust accounting prototype now checks mass-basis consistency, source/assay provenance, retention factors, optional bioavailability, and nutrient/resource ratios while preserving unknowns as unknown. The current extension propagates explicit source-backed lower/upper bounds for concentration, edible mass, retention and bioavailability; it does not treat missing bounds as zero uncertainty or call arbitrary bounds a confidence interval. Partial bounds, point estimates outside bounds and missing interval provenance fail closed. The synthetic fixture only tests arithmetic and receipt integrity; no dietary-health outcome is claimed. See [CF-024_NUTRITION_ACCOUNTING_AND_UNCERTAINTY_V0.md](./CF-024_NUTRITION_ACCOUNTING_AND_UNCERTAINTY_V0.md).

## Additional paradigm shift: compare complete crop options by nutrient delivery per resource

CF-024 accounts for one batch × nutrient row, but individual-row ratios are not enough to compare a production portfolio. A portfolio-level comparison must verify that each batch has the same declared nutrient set, reconcile the shared water/land/energy/cost ledger once per batch, and keep direct assays distinct from estimated delivery. Otherwise nutrient rows can inadvertently multiply resource use, or a missing lab value can silently become zero.

Nutritional water productivity research formalizes nutrient output relative to water use but notes methodological limitations and limited application ([review in *Water*, 2023](https://www.mdpi.com/2073-4441/15/24/4278)). FAO recommends retaining nutrient content through storage and processing, alongside loss reduction ([FAO agrifood-systems recommendation](https://www.fao.org/evidence-platform-agri-food-systems-nutrition/recommendations/detail/3.2.4.c/en)). This supports a separate system-level accounting target rather than assuming that yield or nutrient concentration alone equals nutrition delivered.

**CF-026** adds a portfolio envelope over CF-024 records. It reconciles multiple production options, sums each batch resource ledger once, reports nutrient-specific totals and per-resource ratios only under compatible boundaries, rejects missing nutrient coverage rather than imputing zero, preserves uncertainty and bioavailability coverage, and fails closed on mixed currencies or numeric overflow. A synthetic two-option example is shared between tests and CI. It emits no recommendation or dietary-adequacy claim.

A future improvement threshold must be chosen with local agriculture and nutrition partners, on a declared site-season/production basis, before outcomes are inspected. For example, a proposed target might be at least 10% higher delivered mass of a predeclared limiting nutrient per scarce resource while retaining lower-bound floors for every other priority nutrient and respecting safety, affordability and environmental constraints. This is a study-design target, not a universal nutrition goal or current result. See [CF-026_NUTRITION_PORTFOLIO_ACCOUNTING_V0.md](./CF-026_NUTRITION_PORTFOLIO_ACCOUNTING_V0.md).

## Additional paradigm shift: robust decisions under hybrid uncertainty

CropFoundry should not call a policy robust because it wins one expected score. A policy can sacrifice one plot, crop, nutrient or future season while improving the mean. CF-022's earlier score design made that failure mode visible; receipt v2 preserves unmet-demand penalties and reports per-zone outcomes. CF-025 adds deterministic stress testing over multiple budgets, weather transitions, forecast biases and sensor failures, then reports whether policy rankings and tail outcomes survive those changes.

A 2026 multi-cropping irrigation study proposes chance-constrained multi-objective robust programming and examines how risk preferences and robustness penalties affect solutions ([Agronomy 2026, 16(18), 1786](https://www.mdpi.com/2073-4395/16/18/1786)). A separate 2026 site-specific irrigation study reports that the dominant source of soil-moisture forecast uncertainty shifts from model parameters toward weather uncertainty at longer forecast horizons ([SWIM2](https://doi.org/10.1016/j.agwat.2026.110300)). These are methodological examples, not validation for Gauteng crops or CropFoundry.

The executable `cropfoundry-robust-policy-demo` now re-simulates the same five policies across a 30-case deterministic sensitivity lattice: the six unchanged CF-022 base cases plus isolated water-cap, forecast-bias, sensor-dropout, drought and heat sweeps, and compound failures. The receipt contains all 150 policy × scenario outcomes, and for each policy it names the two cases forming CVaR95's upper tail (`ceil(30 × 0.05) = 2)). Unique scenario IDs, complete grid coverage, daily water caps, per-plot actuator limits and the dose-matched rotating-uniform counterfactual are explicit invariants.

This is more auditable than a six-case single-worst-point summary, but it is still a fixed sensitivity suite, not a random sample from a climate distribution. Equal-weight sensitivity cases do not define real-world probabilities. CF-025 remains blocked until joint uncertainty sets, evidence-backed dependence assumptions, empirical/site-year holdouts, calibration and the preference protocol are pinned. A robust synthetic algorithm run does not establish crop efficacy, calibrated risk, field transfer, food safety or affordability; see [CF-025_ROBUST_DECISION_UNCERTAINTY_BENCHMARK_V0.md](./CF-025_ROBUST_DECISION_UNCERTAINTY_BENCHMARK_V0.md).

## Additional paradigm shift: design experiments to maximize decision value per farm burden

Before spending scarce field-season, seed, labor, fertilizer and water budgets, estimate whether the proposed trial can distinguish a practically meaningful treatment effect. New work is being added as **CF-027**, a Rust trial-power preflight that records the exact input digest and reports a two-arm normal-approximation sample size with Bonferroni correction for co-primary endpoints, a simple cluster design effect and an attrition allowance.

This direction is supported by on-farm trial-design research. Lark et al. (2025) showed the importance of farm-scale block replication and regional site coverage for micronutrient interventions ([npj Sustainable Agriculture](https://www.nature.com/articles/s44264-025-00101-0)). Responsible agricultural research guidance emphasizes random allocation, replication, blocking, and matching the statistical analysis to the experimental design ([Sichinga-Ligowe et al., 2024](https://doi.org/10.1079/cabireviews.2024.0058)). The preflight is deliberately not a substitute for a mixed-model, spatial, repeated-measures or cluster-aware power simulation.

The next standard should be: *no confirmatory biological claim without a pre-registered endpoint and estimand, independent variance basis, minimum practically important effect, allocation unit, randomization/blocking scheme, power/alpha, expected attrition and a design-specific analysis plan*. Underpowered trials should be identified before planting, not defended afterward by post-hoc observed power. The runnable method and its limitations are specified in [CF-027_TRIAL_POWER_PREFLIGHT_V0.md](./CF-027_TRIAL_POWER_PREFLIGHT_V0.md).

## System boundaries

- **Symthaea:** evidence-aware reasoning, candidate generation, uncertainty handling, multi-objective ranking, experiment selection, and recommendation receipts.
- **Symtropy:** optional spatial/dynamic digital twins for plant growth scenarios, irrigation, greenhouse/orchard systems, and farm-level interactions. Label scenarios and simulation assumptions explicitly.
- **Sol-Atlas:** location/environment layers and geospatial applicability: soils, climate, water access, terrain, infrastructure, markets, and exposure to climate hazards. Keep resolution and date of each layer visible.
- **Mycelix:** consent-aware provenance and collaboration for germplasm, trials, seed lots, observations, outcomes, and data-use terms. Preserve authorship and community control.
- **External model adapters:** standardized inputs/outputs around suitable tools such as APSIM and DSSAT. Avoid rewriting mature crop models before establishing where they fail to meet local needs.

These are intended responsibilities, not assertions that the integrations already exist.

## Interoperability before novel schemas

Map rather than replace relevant standards:

- [MIAPPE](https://www.miappe.org/overview/) for minimum plant-phenotyping experiment metadata.
- [BrAPI v2.1](https://brapi.org/specification) for breeding, germplasm, trials, phenotyping, and genotyping exchange.
- [Crop Ontology](https://cropontology.org/about) for standardized trait + method + scale definitions.
- [FAO Genebank Standards](https://www.fao.org/agriculture/crops/thematic-sitemap/theme/seeds-pgr/gbs/en/) for responsible germplasm conservation and documentation.
- [Genesys PGR](https://www.genesys-pgr.org/) and participating national/institutional collections as discoverable germplasm sources; respect access controls, terms, and material-transfer conditions.
- [FAOSTAT](https://www.fao.org/faostat/) for aggregate agricultural, production, trade, and price context—not for cultivar-level evidence.
- [DSSAT](https://dssat.net/) and [APSIM](https://www.apsim.info/) for crop-model baselines where a crop and context are supported.

The ingestion layer must record source URI/identifier, retrieval date, version/digest, license or terms, units, method, geographic/temporal coverage, and transformation lineage. If provenance or reuse rights cannot be established, quarantine the record from training and redistribution.

## Concrete dataset footholds found on 2026-10-18

### DS-010: small repeated-measures tomato dataset

The canonical [UdeA Zenodo record](https://zenodo.org/records/16745911) reports CC BY 4.0 and provides a 303 kB CSV. The web-accessible CSV preview exposes 40 columns including date, plant identifier, treatment, environmental/manual measurements, fruit counts, harvested-fruit count and total harvest weight. Its record describes 135 plants, 27 treatment combinations and 20 weekly sessions (2,666 rows). The related GitHub mirror README reports a different 2,664-row / 44-column shape and inconsistent repository-level license metadata; therefore use only the canonical Zenodo artifact as the first candidate and resolve the exact file/version before claiming schema or model results.

**Split design:** decide whether the first scientific question is (a) generalization to plants not seen during training or (b) prediction at later dates. These are different questions and need different group-aware splits. Random row splitting is not an acceptable sole evaluation because repeated observations from the same plants and treatment setup can cross train/test partitions. Dataset QA must inspect missingness and semantics; for example, the preview shows fields with blanks and the initial dates with zero fruit/harvest values, so zeros must not be automatically interpreted as sensor failures or missing data.

This dataset is a pipeline and within-site benchmark candidate from Colombia, not validation for Gauteng climate, current cultivars, or treatment recommendations.

### DS-011: tomato image phenotyping

[AerialYield-T2M](https://zenodo.org/records/22071809) has an included LICENSE file stating CC BY 4.0. The record describes 677 frames (535 close-range, 142 drone), 14,303 tomato annotations across six ripeness stages, mask and count labels, a SHA-256 checksum list, and a deterministic block-grouped train/validation/test split. The dataset is approximately 3.6 GB and covers one commercial greenhouse in the UAE.

CF-017 creates a separate perception benchmark for ripeness classification, fruit detection/segmentation and visible-fruit counting. These outputs are plant-phenotyping signals only: **ripeness prediction or visible-fruit counts do not establish marketable yield, harvest quality, disease resistance, crop genetics, nutritional quality, or food safety.** Before fitting models, verify the dataset README, bundled checksums, split grouping and annotation conventions; preserve close-range versus drone performance as separate results and check for near-duplicate frames.

These dataset advances make it possible to start a deterministic schema/split pipeline and a separate computer-vision baseline without inventing data. They do not remove the need for local South African observations or physical crop trials.

## Proposed first proof

### P0 — Evidence and schema inventory

Create a source inventory for tomato and one indigenous/traditional leafy crop. List actual available datasets, trait definitions, licensing/access status, site-years, sample counts, missingness, and known leakage risks. Do not announce coverage counts until each is reproducible from a pinned source snapshot.

### P1 — Honest prediction baseline

Choose one trait with sufficient comparable observations (e.g., marketable yield or days to maturity). Evaluate simple climatology/mean, a transparent statistical baseline, and any applicable established crop model. Split by site-year before model selection; keep a final test set untouched. Publish error, uncertainty calibration, data exclusions, and baseline details.

### P2 — Decision-value benchmark

On training/validation data only, compare random selection, uncertainty sampling, and decision-focused expected value-of-information selection under equal trial budgets. Begin on simulated response surfaces and replay datasets; report sensitivity to model misspecification. No simulated win is presented as an agricultural win.

### P3 — Farmer-defined multi-objective optimization

For candidates with defensible data, generate a Pareto set across a small, explicitly measured objective set. Show trade-offs and constraints in interpretable units. Ask local growers and relevant specialists to review whether objectives and weights reflect real decisions.

### P4 — Independent physical validation

Only proceed with willing, qualified partners, appropriate approvals, safe protocols, accessible seed/material, and pre-registered endpoints. Use representative locations and seasons, randomization and replication appropriate to the design, reference cultivars, and independent outcome measurements. Do not market or describe a candidate as an improved cultivar until evidence clears the pre-declared gate.

## Benchmark contract

The machine-readable starting contract is [cropfoundry-benchmark-v0.json](./cropfoundry-benchmark-v0.json). It defines tests and **provisional target thresholds**, not achieved metrics. Before scoring, pin data snapshots and hashes, crop/population, outcome definitions, split rules, baselines, exclusions, and primary metrics. Report confidence intervals and per-environment results; do not let aggregate means hide catastrophic subgroup regressions.

Priority benchmark dimensions:

1. Data lineage, standard compatibility, unit correctness, and duplicate/leakage rejection.
2. Predictive performance against the strongest appropriate baseline on held-out site-years.
3. Prediction interval calibration and out-of-domain abstention.
4. Active-learning sample efficiency against random selection at equal cost.
5. Pareto-front quality and constraint satisfaction, including safety-critical hard constraints.
6. Reproducibility of trial recommendations and independent replication.
7. Real marketable yield, irrigation water productivity, nutrition, sensory quality, post-harvest losses, and economics—measured independently in physical trials.
8. Transfer across environments and inclusion of local/underutilized crops, without conflating data absence with biological inferiority.

## Fail-closed invariants

- Never present synthetic data, simulated outcomes, generated text, or a literature association as a field-validated result.
- Never claim that a variety is improved without a preregistered comparison against explicit local reference cultivars and uncertainty reporting.
- Never infer “safe to eat” from a crop identity or model output alone; toxicity, allergens, contamination, and processing effects require crop-specific evidence and appropriate evaluation.
- Never recommend restricted genetic material, unreviewed gene edits, or laboratory/field interventions outside qualified supervision, applicable biosafety review, and legal approvals.
- Never publish raw farm coordinates, personal data, or sensitive traditional knowledge without a suitable consent and disclosure assessment.
- Keep provenance for all claims, including counterevidence and failed experiments. Missing information must remain visible.
- Do not weaken tests, split integrity, or thresholds to make a benchmark pass. A missing dataset, unavailable runner, or not-yet-measured outcome is **not evidence of success**.

## Executable contract validator

A dependency-light command is integrated into the existing workspace `xtask` instead of creating a separate domain crate or competing registry:

```bash
cargo test -p xtask cropfoundry_validate
cargo run -p xtask -- cropfoundry-validate
```

The validator checks benchmark IDs, required metric/target/split fields, non-passing proposal states, invariant IDs, source IDs, controlled ingestion states, and the permission/provenance prerequisites before any source can be marked training-authorized. Current candidate sources intentionally remain non-trainable. Unit tests include negative mutations for duplicate IDs, a forged passed status, missing targets, and training authorization against unresolved license terms.

The targeted workflow at `.github/workflows/cropfoundry-contract.yml` runs on matching pull requests, including draft PRs. This check validates the integrity of the research contract only: it does not download datasets, certify legal rights, run crop models, or validate biological performance. Local source review and static checks must never be represented as a Rust test pass until the command has actually run in a Rust environment.

## Exit criteria for the first research milestone

The milestone is complete only when (a) a versioned, rights-aware dataset inventory exists, (b) one trait has a pinned and leakage-audited evaluation split, (c) baseline metrics and uncertainty are reproducible, (d) the benchmark contract is validated by a machine-readable checker, and (e) any proposed physical trial has an owner, protocol, comparison, and approval path. This PR defines the research contract only; it does not claim those downstream criteria have been met.

## Research references

- FAO, [Genebank Standards for Plant Genetic Resources for Food and Agriculture](https://www.fao.org/agriculture/crops/thematic-sitemap/theme/seeds-pgr/gbs/en/).
- MIAPPE, [Overview and experiment metadata model](https://www.miappe.org/overview/).
- BrAPI, [Stable V2.1 specification](https://brapi.org/specification).
- Crop Ontology, [Trait dictionary approach](https://cropontology.org/about).
- DSSAT, [official crop-system model and version information](https://dssat.net/).
- APSIM, [model interfaces and coupled crop-system components](https://docs.apsim.info/docs/development/software/interfaces).
- South African Agricultural Research Council, [Indigenous/Traditional African Leafy Vegetables](https://arc.agric.za/arc-vopi/Pages/Plant%20Breeding/Indigenous-Vegetables.aspx) and [roots, tubers and bulbous crops](https://arc.agric.za/arc-vopi/Pages/Plant-Breeding.aspx).
- FAOSTAT, [Food and agriculture statistics](https://www.fao.org/faostat/).

Links were reviewed for this proposal on 2026-10-18. Dataset availability, access terms, and current tool versions must be rechecked at ingestion time.

## Coordination with existing Luminous Dynamics work

CropFoundry is an evaluation/evidence umbrella. It must extend these active workstreams rather than create parallel crop, trial, provenance, or simulation authorities:

- [Symthaea #5967 — agriculture and food-system infrastructure engineering](https://github.com/Luminous-Dynamics/symthaea/issues/5967): owns supported agricultural/food infrastructure classes and explicitly does not confer agronomic or food-safety authority.
- [Symthaea #6213 — hydroponic controlled-environment analysis profile](https://github.com/Luminous-Dynamics/symthaea/issues/6213): reuse the existing Agribot + read-only AnalysisRequest/AnalysisArtifact path; do not reinterpret soil-state channels as nutrient-solution evidence.
- [Symthaea #7441 — evidence-bounded strawberry-picking benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/7441) and [Symtropy #1605 — deterministic harvest scene](https://github.com/Luminous-Dynamics/symtropy/issues/1605): use these for robotics/manipulation. Geometry simulation is not fruit-damage, food-quality, or crop-yield evidence.
- [Symthaea #7828 — agricultural HRI dataset benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/7828): reuse its distinct perception/temporal/runtime/safety boundaries; recorded-data evaluation is not physical machine qualification.
- [Mycelix #3303 — ProductiveLoopV1 evidence contract](https://github.com/Luminous-Dynamics/mycelix/issues/3303), [#3309 — lettuce/DWC research profile](https://github.com/Luminous-Dynamics/mycelix/issues/3309), [#3311 — cultivar admission profile](https://github.com/Luminous-Dynamics/mycelix/issues/3311), and [#3317 — admission-boundary validator](https://github.com/Luminous-Dynamics/mycelix/issues/3317): reuse these for useful-output events, cultivar/source distinctions, and hydroponic evidence boundaries. Supplier claims, cultivar admission, and harvested mass remain separate from measured crop performance, edibility, safety, or marketability.
- [Mycelix #5001 — neighborhood producer cooperative vertical slice](https://github.com/Luminous-Dynamics/mycelix/issues/5001) and [Sol-Atlas #105 — privacy-preserving neighborhood food network](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/105): reuse existing product/lot/provenance and spatial projection domains.
- [Sol-Atlas #107 — Johannesburg/Highveld food resilience profile](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/107): consume its source/scale/rights gates and local environmental profile; do not produce a competing crop-suitability or current-season forecast.

The initial tomato + local leafy-crop proposal is a **data-discovery plan**, not a frozen cultivar selection. The existing lettuce/DWC profile is a separate controlled-environment test lane. Do not force all lanes into one scoring model or claim any lane has demonstrated performance until its own data and evaluation gates are passed.

The first coordination milestone should be to crosswalk these contracts to the benchmark IDs in the JSON, identify missing fields/tests, and assign one canonical owner per entity. Do not duplicate ProductiveLoop, farm/household identity, seed-lot identity, environmental source manifests, robot scene profiles, or safety authorities.

