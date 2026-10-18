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

