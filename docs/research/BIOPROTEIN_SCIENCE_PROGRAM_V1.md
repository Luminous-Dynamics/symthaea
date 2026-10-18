# BIOPROTEIN-001 — Evidence-first animal nutrition and engineered-protein systems

**Status:** research/design proposal; no biological, nutritional, regulatory, economic, or environmental result is established by this document.  
**Version:** 1.0  
**Review date:** 2026-10-18  
**Owner:** Symthaea research profile, composed with existing Symtropy, Mycelix, and Sol Atlas contracts.

## Executive thesis

Build a reusable decision-and-evidence system that works backward from a declared animal-nutrition or food-product requirement to candidate ingredients, production processes, and experiments. The platform should compare conventional, microbial/fungal, fermentation-derived, cultivated-cell, and hybrid options on the same declared requirements and evidence standard.

The first deliverable is **not** an autonomous recipe generator or a claim that cultivated meat is already cost-competitive. It is a deterministic, source-pinned benchmark and feasibility evaluator that makes it possible to falsify promising-looking formulations before expensive experiments, while clearly identifying what still requires laboratory, veterinary, animal-trial, manufacturing, lifecycle, or regulatory evidence.

## Why this is worth starting now

Three independent research threads point to a tractable program:

- A scoping review of cultivated-meat techno-economic analyses found a significant disconnect between industrial scale-up assumptions and the scale/maturity of much primary experimental research. Low-cost media, suitable bioreactors, serum-free differentiation, and facility/process design remain major bottlenecks. [Goodwin et al., *Nature Food* (2024)](https://www.nature.com/articles/s43016-024-01061-3).
- A 2025 review identifies microbial single-cell protein as a potential alternative to soybean meal and fishmeal, while listing cost, scalability, palatability, safety, regulation, life-cycle impacts, and the need for longer animal trials as open constraints. [Wang et al., *Chinese Science Bulletin* (2025)](https://www.sciengine.com/CSB/doi/10.1360/CSB-2025-5190).
- A 2026 review of hybrid cultivated meat treats the product as an integrated system—cell supply, media, scaffold, processing, nutrition, and quality—not merely cultivated cells mixed into a conventional recipe. Most approaches described remain at laboratory/prototype scale. [Hybrid cultured meat review (2026)](https://www.sciencedirect.com/science/article/pii/S0963996926015188).

This profile composes with existing work instead of creating a parallel nutrition stack:
- Symthaea [nutrition-aware reasoning PR #7476](https://github.com/Luminous-Dynamics/symthaea/pull/7476), [NUTRI-SOURCE-001 #7481](https://github.com/Luminous-Dynamics/symthaea/issues/7481), and [CULINARY-BENCH-001 #7859](https://github.com/Luminous-Dynamics/symthaea/issues/7859).
- Symtropy [nutrient accounting PR #1637](https://github.com/Luminous-Dynamics/symtropy/pull/1637) and [NUTRI-PORTFOLIO-001 #1649](https://github.com/Luminous-Dynamics/symtropy/issues/1649).
- Mycelix [food-network issue #5001](https://github.com/Luminous-Dynamics/mycelix/issues/5001) and [REGEN program #940](https://github.com/Luminous-Dynamics/mycelix/issues/940).
- Sol Atlas [ATLAS-FOOD-001 #105](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/105) and [AGRO-CLIMATE-001 #107](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/107).

## Paradigm shifts

### 1. Optimize delivered nutrition, not crude-protein percentage

Crude protein is an inadequate sole objective. Comparisons must distinguish as-fed and dry-matter bases, energy, digestible indispensable amino acids, relevant fatty acids, vitamins/minerals, palatability and intake, species/life-stage requirements, and evidence quality. A cheap protein source may be poor value if digestibility is low, the limiting amino acid needs expensive supplementation, or processing and safety controls are costly.

The denominator should fit the decision. Candidate metrics include cost per kilogram of usable ingredient, cost per 1,000 kcal delivered, and cost per unit of digestible limiting nutrient. They are not interchangeable; publish the complete objective vector and units rather than hiding it behind one score.

### 2. Co-design the product and the bioprocess

Search jointly over ingredient blends and process choices. For microbial/fungal biomass this can include feedstock class, organism identity, nutrient profile, processing burden, downstream recovery, and lot variability. For cultivated cells it can include media composition classes, growth/differentiation phases, cell density/productivity, scaffolds, recovery and hybrid dilution. Keep these as candidate variables for research, not operational culture instructions.

Never compare a bench-scale biomass yield with an industrial total-cost projection without explicitly marking the scale/model boundary.

### 3. Choose the cheapest *safe feasible* solution, not the mathematical cheapest solution

Nutrient adequacy, ingredient/product eligibility, contamination controls, animal welfare, provenance and legal authorization are hard gates. Cost, water, energy, land, greenhouse-gas emissions, resilience, taste/texture and local availability are candidate objectives only after the hard gates are evaluated.

`unknown` is not `safe`, `zero`, `available`, or `compliant`. A lower price or carbon estimate must never compensate for an unresolved safety or nutrition requirement.

### 4. Make uncertainty and regional conditions first-class

Use aligned scenarios for seasonal feedstock availability, transport interruptions, electricity mix, water scarcity, source uncertainty, yield variability and ingredient-price ranges. Report nominal results, per-scenario results, feasibility changes and rank instability. Avoid unsupported independent uncertainty intervals where uncertainties share the same source or condition.

Sol Atlas should contribute spatial context and resource/transport scenarios; it must not become the source of truth for inventory, nutrition composition, or food safety.

### 5. Use active learning to buy information, not just generate more hypotheses

Prioritize the next laboratory assay or pilot measurement by expected decision value: which uncertainty could change feasibility, candidate ranking or an expensive scale-up decision? Compare value-of-information selection with a preregistered random-selection baseline at the same experimental budget. Record negative and inconclusive results, not just winning formulations.

### 6. Make every claim traceable and revisable

Mycelix can preserve source, consent/rights, lot, custody, batch, assay and outcome lineage. Corrected or superseded evidence must remain traceable; derived claims cannot silently inherit stronger status than their inputs. Data access and redistribution rights remain separate from evidence validity.

## System boundaries and project ownership

| System | Responsibility in this program | Must not become |
|---|---|---|
| **Symthaea** | Candidate generation; constraint-aware decision support; evidence reconciliation; uncertainty and experiment prioritization; benchmark receipts | A source of biological truth, a second nutrient database/calculator, a veterinary authority or physical controller |
| **Symtropy** | Deterministic nutrient/resource accounting, process and scenario simulation, feasible/Pareto frontiers | An authority for ingredient safety or real-world inventory |
| **Mycelix** | Permissioned sharing, provenance, batch/source/outcome lineage, community and producer commitments | The assay lab, certification body, or nutritional calculator |
| **Sol Atlas** | Geography, climate/resource layers, transport and regional supply scenarios | A yield oracle, private farm locator, inventory ledger, or safety authority |

Reuse the referenced work's types and evidence identities. Do not fork its data stores, calculators, geography services or ledgers.

## Proposed two-track research program

### Track A — animal food and feed (nearer-term validation)

Start with a narrowly bounded ingredient comparison or complementary product, not an automatically prescribed complete diet. Benchmark a selected microbial/fungal ingredient against a feasible conventional reference within a declared species, life stage, inclusion range, and feeding purpose. Candidate species include dogs/cats, poultry, pigs, ruminants, and aquaculture—but each target requires its own requirements, evidence and evaluation protocol. Do not transfer a result across species without evidence.

A safe first computational case is a **synthetic, clearly labelled feed-formulation cohort**. Advance to actual data only when their source, license, species match, measurement method, unit basis and limitations are recorded. Feeding trials, health outcomes, or commercial formulation remain outside the computational proof boundary.

### Track B — engineered protein and cultivated/hybrid meat (longer-term)

Evaluate microbial biomass, precision-fermentation ingredients, cultivated muscle/fat, scaffolds/texturizers and hybrid formulations under one product specification. Start with a bottom-up techno-economic model and sensitivity analysis before making price-parity or sustainability claims. Compare production systems using the same functional unit, geography, electricity mix, water assumptions, yield boundary, recovery losses, facility utilization and product-quality requirements.

Possible products should be judged separately: pet-food ingredients, human-food ingredients, structured cuts and hybrid mince do not share a single definition of success.

## Benchmark contract

Every benchmark run must be bound to a versioned scenario and emit machine-readable receipts. A run is `pass`, `fail`, `incomplete`, or `not_comparable`; queued, missing, stale, or wrong-head checks never count as pass.

### Gate 0 — source and evidence eligibility

Record canonical source/record IDs, edition, retrieval time, license/terms and scoped permission evidence, artifact/record digest where retaining it is lawful, units, basis, method, geography, species/cultivar/strain where applicable, measured/compiled/estimated/synthetic status, known limitations and review status.

Acceptance:
- 100% of bundled real-data rows have demonstrable redistribution/use permission for that exact use; unknown permission blocks bundling.
- 100% of nutrient/measurement values retain source and transformation lineage.
- Stale, revoked, mismatched, unlicensed or unknown material never silently becomes eligible.
- Public fixtures are synthetic or explicitly redistributable; no private dataset or credential is committed.

### Gate 1 — deterministic accounting and constraint correctness

Create independent hand-checked golden fixtures for unit conversion, as-fed/dry-matter conversion, ingredient blending, missing values, range/boundary behavior, and every declared nutrient/resource constraint. Use adopted, versioned requirements appropriate to the target jurisdiction, species and life stage. The [2025 FEDIAF guidelines](https://europeanpetfood.org/self-regulation/nutritional-guidelines/) are a current reference for cat/dog complete and complementary foods; use appropriate authoritative guidance for other species.

Acceptance:
- 100% exact agreement with independent known-answer fixtures within an explicitly declared numeric tolerance.
- Zero silently imputed required values.
- Zero infeasible formulations admitted to a feasible set.
- Positive controls show that valid known examples remain accepted.
- Results are invariant to input/candidate ordering where the calculation is meant to be order-independent.

Passing this gate proves only the declared arithmetic/contract, not dietary safety or adequacy in an animal.

### Gate 2 — nutritional utility and real-world feasibility

Report each objective separately:
- requirement coverage and distance to every hard limit;
- digestible nutrient yield and relevant limiting amino acids where sound inputs exist;
- energy density and nutrient delivery per unit of cost;
- palatability/intake, animal performance and health outcomes only when measured under an appropriate protocol;
- ingredient variability and uncertainty across batches;
- cost, water, energy, land, emissions, storage and transport with defined boundaries.

For feed trials, preregister controls, inclusion rates, sample-size rationale, endpoints, exclusions, analysis, welfare monitoring and stop criteria with qualified veterinary/animal-nutrition oversight. Relevant measured endpoints may include feed conversion ratio, average daily gain, intake, health/welfare indicators, mortality and output/product quality. These outcomes cannot be inferred from composition tables or simulation.

### Gate 3 — process/economics/sustainability

For microbial and cultivated production, report as applicable:
- volumetric productivity and yield on stated substrate;
- harvestable/recovered mass and downstream recovery losses;
- batch success/contamination/rejection rates and product variability;
- media/feedstock cost share, energy, water, labor, facility utilization, capital assumptions and process consumables;
- cost per kilogram of usable specification-compliant product;
- life-cycle results per declared functional unit and geography, with sensitivity to electricity, feedstock allocation, yields, recovery, scale and end-of-life.

Do not infer a universal cultivated-meat cost floor or emissions advantage from one hypothetical plant model. Make assumptions inspectable and label modeled results as modeled.

### Gate 4 — prediction and experiment-selection skill

Evaluate forecasts on a held-out set grouped by supplier/ingredient family, process or time where feasible; random row splits alone can leak near-duplicate evidence. Compare with simple transparent baselines. Report MAE/RMSE or other suitable endpoint metrics, calibration and interval coverage, abstention/unknown rates, ranking stability, and performance by subgroup. For experiment selection, compare realized decision improvement against random and strongest simple baselines at matched test budgets.

Initial target: 100% agreement on the frozen deterministic corpus, zero hard-gate violations and reproducible receipts. After the real-data baseline is frozen, propose stretch goals (for example, at least 10% improvement in one preregistered cost or resource metric) only where the same safety/nutrition constraints hold and uncertainty intervals support the difference. That 10% is a proposed engineering target, not a published result.

### Gate 5 — external validity and claims

Promotion ladder:

`schema/specification → synthetic golden corpus → public/licensed data replay → independent assay comparison → controlled pilot → approved animal study or food-safety review → contextual replication → permitted deployment`

Each stage proves only its scoped claim. Synthetic correctness is not laboratory validation; a lab result is not a feeding trial; short-term animal performance is not a lifetime-health claim; one facility's lifecycle result is not a universal sustainability conclusion.

## Adversarial / negative-control corpus

At minimum:
1. Required amino acid missing, but crude protein is high.
2. A source row has the wrong animal species, life stage, ingredient identity or preparation state.
3. Dry-matter and as-fed values are mixed without conversion.
4. A nutrient has an unknown unit, denominator or assay method.
5. A safety contaminant measurement is absent, stale, above a declared limit or from a different lot.
6. Ingredient price is current but the ingredient is unavailable during the required season.
7. A local residue is physically present but contaminated, reserved for ecological use, or already committed elsewhere.
8. Novel microorganism or processing method lacks the required safety/authorization evidence.
9. Cost/impact improves by using an infeasible formulation or excluding rejected batches.
10. Scale-up extrapolates from a bench result without identifying the assumed change.
11. A cultured-cell claim is treated as approval for every species, process, jurisdiction or end use.
12. An objective/source/scenario/unit mismatch makes two candidates incomparable.
13. Missing data are silently set to zero or carried forward from another scenario.
14. A simulated feeding outcome is represented as an observed animal outcome.
15. A successful average masks a high-risk batch, harmful subgroup or welfare adverse event.
16. Changing candidate or source order changes a deterministic result.
17. An invalid candidate enters a Pareto frontier because its cost/footprint is favorable.
18. A correction erases the previously used source/batch rather than preserving supersession lineage.

Every negative case needs a stable ID, input fixture, expected disposition, invariant/violation class, and reference/rationale. Include positive controls for every major gate.

## Safety and regulatory boundaries

- No formula generated by this platform is a feeding recommendation until reviewed by a qualified animal nutritionist/veterinarian under a declared product purpose and applicable requirements.
- Do not produce or use a novel ingredient for consumption solely because it meets a computational nutrient profile. Identity, contaminants, digestibility, organism/strain risks, residues, process controls, stability, and intended-use safety require appropriate evidence.
- Regulatory pathways are jurisdiction- and product-specific. The US FDA's [Animal Food Ingredient Consultation guidance and program status](https://www.fda.gov/animal-veterinary/animal-foods-feeds/evaluation-fda-pre-market-animal-food-programs-and-plan-consultations) was updated in September 2026; EU novel foods and microorganism/feed additive pathways also have separate assessment requirements. This document is not regulatory advice or authorization.
- No autonomous purchasing, formulation release, manufacturing control, feed delivery or animal treatment. Any future physical-control work needs a separately reviewed authority and safety boundary.

## Implementation sequence — runners are not a research dependency

1. **R0 — contract review:** review the project boundaries and non-compensatory gates before expanding implementation.
2. **R1 — synthetic contract artifacts drafted:** this PR includes [a JSON Schema](bioprotein/fixtures/bioprotein-gate-schema-v1.json) and a [26-case synthetic gate corpus](bioprotein/fixtures/bioprotein-gate-corpus-v1.json). These fixtures cover permission/freshness, species/identity matching, nutrient basis/unknowns, safety/authorization, availability, scenario alignment and evidence-kind misuse. They are design artifacts; no runtime evaluator has validated them yet.
3. **R2 — validate independently:** check JSON parsing/schema conformance and implement a separate small oracle for the declared gate semantics. The oracle must not import the production evaluator's own decision function.
4. **R3 — compose, don't duplicate:** add a read-only candidate profile adapter on top of the existing nutrition source registry and Symtropy accounting contracts; do not create a second calculator.
5. **R4 — receipts and adversarial tests:** bind exact repository commit, source/scenario manifest digest, fixture counts, expected/actual dispositions and run status. Add permutation/replay invariance and malformed-data cases.
6. **R5 — evidence-backed expansion:** add public or permission-cleared ingredient data; publish source rights and held-out evaluation design before measuring model skill.
7. **R6 — partner-led experiments:** only design a lab/pilot study after modeling identifies a decision-sensitive uncertainty and qualified collaborators review safety, ethics, sample size and stopping criteria.


## Evidence references

- Goodwin, Aimutis & Shirwaiker, “A scoping review of cultivated meat techno-economic analyses to inform future research directions for scaled-up manufacturing,” *Nature Food* (2024): https://www.nature.com/articles/s43016-024-01061-3
- Wang et al., “Single-cell protein in the animals feed industry: advances and future prospects,” *Chinese Science Bulletin* (2025): https://www.sciengine.com/CSB/doi/10.1360/CSB-2025-5190
- “Biomass and precision fermentation inputs for cultivated meat: Advances, sustainability, food safety, and techno-economic perspectives,” *Food Research International* (2026): https://www.sciencedirect.com/science/article/pii/S0963996926005818
- “Hybrid cultured meat as an integrated food production system: Recent advances and commercialization challenges” (2026): https://www.sciencedirect.com/science/article/pii/S0963996926015188
- FEDIAF, Nutritional Guidelines 2025: https://europeanpetfood.org/self-regulation/nutritional-guidelines/
- US FDA, Animal Food Ingredient Consultation and premarket animal-food review program (updated 2026): https://www.fda.gov/animal-veterinary/animal-foods-feeds/evaluation-fda-pre-market-animal-food-programs-and-plan-consultations
- EFSA, Qualified Presumption of Safety (reviewed 2026): https://www.efsa.europa.eu/en/topics/topic/qualified-presumption-safety-qps
- FAO, “Sources of feedborne hazards” (hazards vary by ingredient origin and processing): https://www.fao.org/4/w8901e/w8901e07.htm
