# Culinary Commons: benchmark-first systems design

**Status:** research RFC / proposed work, 2026-10-17  
**Scope:** Symthaea culinary domain, with proposed Mycelix, Symtropy, and Sol Atlas integration  
**Maturity:** design proposal. Targets are proposed acceptance thresholds, not measured results or certification.

## Executive thesis

Build a **Culinary Systems Compiler**: turn a person's intent, pantry, budget, time, tools, local supply, and hard restrictions into a versioned plan that can be checked, explained, executed, revised from evidence, and shared with consent. The system must be able to say **“unknown—verify this”** or **“no safe feasible plan”**, not merely produce a fluent recipe.

The product is not a universal AI chef and should not optimize a single notion of the perfect dish. Culinary success is multi-objective and culturally situated: a meal must be safe, feasible, affordable, enjoyable to the people eating it, nutritionally fit for its declared purpose, and honest about provenance and uncertainty. Waste and environmental impact count only when measured or calculated from traceable data.

The central paradigm shift is from **recipe generation** to **evidence-carrying culinary decisions**.

## What exists now—and what it does not establish

The current Symthaea culinary crate is a substantive foundation:

- Its flavor network uses published ingredient/volatile-compound and recipe data; the Ahn et al. 2011 food-pairing result is a falsifiable regression test.
- CulinarySpec and its validator encode explicit process and composition checks, including emulsion, coagulation, pasteurization, dough hydration, candy stages, dairy acidification, frying fat, sodium, and added sugar.
- Thermal and reaction modules provide simplified process models; the nutrition module implements declared-profile calculations such as Atwater energy and population-level threshold checks.
- The kitchen and active-inference palate modules form a closed simulation loop and document both failures and limitations rather than silently presenting them as solved.

Those facts are not evidence of end-to-end meal quality. Flavor-pairing replication is a pattern in a recipe corpus, not proof that a generated combination tastes good. The balance score is a heuristic, not a sensory instrument. The kitchen is explicitly a toy simulator, not a calibrated digital twin of a real stove, pot, ingredient, or sensor. Nutrition totals are currently supplied by a caller; the existing module does not derive them from a full ingredient database. The palate's own documented experiments report unreliable target convergence and physically invalid transient overshoot under some policies.

These limitations determine the next engineering step: **benchmark the gaps, add data only where it closes a measured gap, and keep simulated, inferred, and physically verified evidence separate.**

## Paradigm shifts

### 1. From prose to a typed meal plan that must compile

Represent a meal as data with ingredient identities, amounts, units, servings, preparation steps, temperatures and durations, required tools, optional substitutions, allergens, provenance, nutritional estimates, cost timestamp, and uncertainty. A plan should pass through explicit stages:

1. **Parse intent:** distinguish hard constraints from soft preferences.
2. **Resolve evidence:** ingredient identity, units, source/licence, local availability, prices, nutrition reference, and freshness of each claim.
3. **Generate candidates:** return multiple feasible options, not one falsely precise answer.
4. **Validate:** run hard invariants before ranking.
5. **Rank:** expose trade-offs between preference fit, time, cost, nutrition, waste, familiarity, novelty, and impact.
6. **Execute and observe:** where a user or hardware reports an outcome, update from that evidence.
7. **Record outcome:** use opt-in, minimised feedback with a versioned recipe identity; never infer enjoyment from a generated rating.

Use explicit outcomes such as **Feasible**, **Infeasible**, **NeedsVerification**, and **Rejected**. Hard safety and allergy constraints are gates, not weighted preferences. Unknown facts required to establish safety must not be papered over with a score.

### 2. From “best recipe” to Pareto front and preference sovereignty

There is no single scalar that safely captures a dish. Return a small Pareto set or clearly explained alternatives. Never let a small carbon or cost win compensate for an allergen conflict. Do not collapse regional, household, religious, ethical, or personal preferences into a global average. Preference learning should be optional, inspectable, and locally controllable; a user must be able to reset it and reject a recommendation without being penalised.

Flavor-compound overlap is one feature, not a universal law. Ahn et al. discuss preparation method and non-flavor ingredient roles as factors their network does not fully represent. Treat cuisines as bodies of practice, not labels for a model to imitate without evidence or community input.

### 3. From open-loop instructions to guarded closed-loop cooking

For a real kitchen, plan–act–observe–revise can be safer and more robust than pretending every ingredient, appliance, portion, and stove behaves identically. But a simulation must not be represented as physical sensing. Each observation needs a source and uncertainty: user observation, actual sensor, database value, or simulation state.

- Keep validated safety constraints outside the learned policy.
- Compare the active-inference agent to deterministic and random baselines at the same initial conditions and seeds.
- Track maximum constraint violation over the whole trajectory, not just final state.
- Add sensor noise, lag, missing measurements, ingredient variation, and actuator faults as labelled simulation conditions first.
- Require a stop/abstain action when uncertainty makes safe action impossible.
- Do not attach Symthaea consciousness or Phi claims to taste, dish quality, or food safety.

### 4. From “personalization” to household feasibility and access

A recommendation is not useful if the user cannot buy the ingredients, afford the basket, understand the instructions, use the available tools, or finish in the available time. Evaluate the complete path from recommendation to prepared meal, including local ingredient names, substitutions, storage, equipment, skill, accessibility, household size, and customary meal patterns. Health-specific dietary advice requires appropriate domain review and must remain outside general recipe ranking unless explicitly and safely scoped.

### 5. From individual recommendation to a community-governed food commons

Recipes and culinary knowledge carry authorship, culture, labour, and sometimes sensitive household details. Mycelix should support provenance and consent rather than turn every contribution into permanent public training data. Separate public product/ingredient facts from private household preferences. Record licensing, attribution, version history, and source evidence; support correction, withdrawal where technically and legally possible, dispute paths, and producer-led attestations. Do not claim a Holochain record proves a physical supply claim: it proves that a record was attested, not that the underlying statement was true.

### 6. From shopping lists to resilient local food infrastructure

Sol Atlas could eventually add spatial context—seasonality, verified suppliers, food access, transport/cold-chain constraints, local water/energy stress, and supply disruptions—only where sourced data has dates and known coverage. This is a proposed integration, not an existing Sol Atlas food layer. Symtropy could help simulate kitchen layout, task timing, energy use, and robotic reach/collision scenarios if a concrete model is built; it is not already a validated food process simulator.

Optimise for resilience as well as the average day: evaluate what happens when a preferred ingredient is unavailable, a price rises, a delivery is late, or a local supply chain is disrupted. A robust plan should produce safe, acceptable substitutions or abstain.

## Proposed benchmark suite

A benchmark is meaningful only with a named dataset, ground-truth method, version, split policy, baseline, uncertainty, and failure report. The thresholds below are **initial pilot goals**, not established industry standards. Report metrics per task, region/cuisine, dietary constraint, and difficulty tier; never let aggregate performance hide one severe failure.

| Track | Evaluation method | Initial acceptance target |
|---|---|---|
| **B0 — Scientific reproduction** | Re-run the existing Ahn flavor-network analysis against pinned source data; publish source hashes, parser version, null-model details, per-cuisine values, and sensitivity analysis. | Preserve current falsifiable sign/direction checks. Investigate divergence; never tune it away. Explicitly state corpus-pattern reproduction is not taste validation. |
| **B1 — Hard safety & invariants** | Curated positive/negative fixtures and adversarial mutations for every implemented CulinarySpec invariant; include boundaries, malformed values, missing evidence, unit errors, and near-threshold trajectories. | **Zero critical unsafe acceptances** in the labelled hazard suite. Every deliberately unsafe fixture is rejected or marked as needing verification. Fail closed for unknown facts required by the safety case. |
| **B2 — Nutrition estimation** | Use at least 300 independently audited recipes with traceable ingredient-level data where applicable and disclosed cooking-yield/loss assumptions. Measure three distinct things: whether targets are justified for the declared profile (**target validity**), whether the menu actually meets those targets (**menu achievement**), and whether reported totals match an independent calculation (**arithmetic agreement**). Compare with laboratory or expert-verified references where feasible. | Pilot goal: median absolute percentage error ≤10% for energy and major macros, and ≤20% for selected micronutrients on the defined reference subset. Pre-register nutrient-specific tolerances. Report target validity, menu achievement, arithmetic agreement, coverage, and uncertainty separately; do not score nutrients without reference data. |
| **B3 — Cost and availability** | Dated local baskets with receipt/market-price ground truth; hold out stores, dates, and geographic areas. Separate unavailable items from estimated items. | Median cost error ≤10% and 95th-percentile error ≤20% for the tested market basket. A hard budget is satisfied only with sufficiently current prices and a computed total within budget; otherwise mark uncertain. |
| **B4 — Time and equipment feasibility** | Time-and-motion observations from real cooking sessions; record active time, elapsed time, equipment, and skill requirements separately. Validate by recipe and equipment class. | Median absolute time error ≤20% or 5 minutes, whichever is larger. At least 95% of generated plans satisfy declared tool/time constraints or explicitly abstain. |
| **B5 — Sensory acceptance** | Randomised, blinded consumer tests designed with sensory-method expertise and aligned with ISO 11136 guidance. Compare with a credible human-authored baseline and stratify by culinary context. | Personalised recommendations are non-inferior to the constraint-only baseline within a pre-registered margin (initial proposal: 0.5 points on a 9-point liking scale), and improve pairwise preference by ≥10% before making a benefit claim. Determine sample size by power analysis; report confidence intervals and disagreement. |
| **B6 — Cultural and preference fit** | Community-reviewed cases, opt-in feedback, separate per-context analysis, and counterexamples where generic ingredient-pair heuristics would erase a local convention. | Pilot coverage across at least 12 materially distinct culinary contexts, with gaps disclosed. Translation alone does not validate a context; review by people with relevant lived/practice knowledge is required. |
| **B7 — Waste and portion fit** | Weigh avoidable kitchen and plate waste per person-week before/after in matched pilots; distinguish edible food, inedible residue, spoilage, and unserved surplus. Track satisfaction and adequacy so the system cannot “win” by under-serving. | Target ≥15% lower measured avoidable waste without a material fall in satisfaction or declared adequacy measures. Publish absolute mass and method, not just a percentage. |
| **B8 — Supply shock resilience** | Stress scenarios: top-three ingredient unavailable, selected prices +20%, one supplier lost, seasonal shift, and delivery delay. Validate substitutions against hard constraints and with human review. | ≥90% of generated plans remain feasible under the defined stress suite, with zero critical safety violations. Break down failures by scenario; do not treat unknown substitutions as safe. |
| **B9 — Provenance and correction** | Trace every source-derived claim to a source/version/date/licence. Inject conflicting supplier, recall, and correction records; measure whether dependent plans are identified. | 100% of externally sourced claims in benchmark output carry provenance or are marked unverified; zero fabricated supplier/certification claims. Target recall-impact analysis in <60 seconds on a bounded pilot corpus, with scope reported. |
| **B10 — Agent reliability** | Paired-seed trials against random, greedy, and deterministic domain baselines. Record target success, time-to-target, overshoot, intermediate violations, and abstentions. | Do not promote on final-state score alone. Require ≥95% safe completion across 100 pre-registered seeds per scenario and no constraint overshoot in the supported safe envelope before calling that scenario reliable. |
| **B11 — Operational performance** | Release-mode runs on named commodity CPU hardware; warm/cold starts, 10,000-spec validation batch, one-week plan generation, memory and binary size. | Initial engineering targets: p95 <100 ms per single-spec validation and <2 s for a seven-day plan on a documented reference machine. Rebaseline only with evidence and an explicit reason. |
| **B12 — Personalization linkage and repeatability** | Paired counterfactual profiles that change one decision-relevant input at a time (budget, available equipment, hard restriction, household serving count, or a preference); include pre-registered no-change cases for irrelevant attributes. Run repeated generations at a frozen model/configuration and compare effects against within-profile run-to-run variation. | Zero violations of declared hard restrictions across the fixture suite; ≥90% directional concordance on pre-adjudicated soft-constraint cases in the initial pilot. Do not claim personalization when changes are smaller than baseline stochastic variation or irrelevant attributes change safety-critical decisions. |

### Guardrails for interpreting the numbers

### Risk-proportionate progression gates

A newly published 13 October 2026 perspective proposes the VALID-Diet framework for generated dietary advice. It is explicitly a testable proposal, not a consensus guideline or regulatory standard; its useful methodological point is to stage claims by evidence rather than treating plausibility as validity ([Xu et al., 2026](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2026.1957211/full)).

Apply that principle to culinary releases:

1. **Gate 0 — intended use and input integrity:** define who the system serves, allowed tasks, risk tier, hard constraints, missing-data behavior, and what it must refuse.
2. **Gate 1 — analytical validity:** independently verify units, recipe decomposition, nutrition arithmetic, physical-model domain, source identity, and uncertainty.
3. **Gate 2 — contextual validity:** test safety, actual availability, personalization, cultural context, equipment/time feasibility, and subgroup failure cases.
4. **Gate 3 — human utility and outcomes:** only after prior gates, test blinded sensory preference, actual preparation completion, waste/cost changes, and workflow burden.

Each gate has its own receipt and failure conditions. Passing a later-looking benchmark such as taste preference does not compensate for missing analytical or safety evidence. Keep nutrition-specific intended uses risk-proportionate; ordinary meal ideation and disease-specific dietary advice must not share the same release claims.

- The zero-unsafe-acceptance goal applies to the defined test suite and modelled conditions; it is not a real-world food-safety certification. Physical safety depends on validated time/temperature conditions, measurement, handling, and jurisdiction-appropriate guidance. Use primary food-safety sources such as USDA FSIS and update jurisdictional cases deliberately.
- Nutrient tables vary by cultivar, brand, batch, storage, cooking yield, and measurement method. Benchmark uncertainty as well as point error. A generic database estimate is not equivalent to lab analysis.
- Sensory tests measure liking in a tested population and context; they do not establish universal quality. Pre-register hypotheses and margins, randomise order, prevent participants/testers from identifying the system's recommendation, and retain per-person distributions rather than only a mean.
- Waste benchmarks need a consistent mass-accounting boundary. ISO/CD TS 20008 is a committee draft under development as of this RFC's date, not a final published standard; use it as a methodological signal, not a compliance claim.
- Thresholds should be reviewed after a versioned pilot reveals actual error distributions. Do not lower a threshold merely because a candidate fails it.

## Cross-project responsibility map

### Symthaea — culinary compiler and evidence-aware planner

**Owns:** ingredient/flavor relations, CulinarySpec, chemical/process validators, nutrient calculations, candidate ranking, simulation benchmarks, uncertainty, and abstention semantics.

**Next implementation increments:**
1. Pin dataset hashes and expose a machine-readable benchmark manifest.
2. Add mutation/property tests for existing validators and prove failures at boundaries.
3. Define a recipe/meal-plan schema that represents source, units, allergens, confidence, cost timestamp, equipment, and provenance independently of prose.
4. Build a deterministic evaluation runner emitting JSON with exact git SHA, dataset digests, seed, case counts, failures, and baseline comparisons.
5. Keep model-only and physical-sensor observations as distinct typed evidence.

### Mycelix — consent, provenance, and producer/community coordination

**Proposed integration, not a claim of existing implementation:** attestable ingredient lots and supplier statements; recipe attribution and licence; author/household consent; correction and recall propagation; cooperative purchasing and fair contribution records; privacy-preserving sharing of household preferences.

**Acceptance test:** an auditor can trace a plan to each sourced ingredient claim and its recorded attestation, identify unknown/disputed claims, and propagate a recall/correction without exposing private household profile data. An attestation must not be misrepresented as proof that a physical item was authentic.

### Symtropy — constrained spatial and operational simulation

**Proposed integration:** model kitchen/workcell layout, equipment reach, collision-free paths, task precedence, concurrent preparation, energy use, and failure/recovery. Link to culinary process profiles only where their parameters have documented validity; no consciousness metric should be used as a proxy for flavor or safety.

**Acceptance test:** deterministic replay of the same scenario produces reproducible task/energy/collision outcomes; adversarial geometry and schedule cases are included; robotic simulation results remain labelled simulated until confirmed by hardware tests.

### Sol Atlas — local resource and resilience context

**Proposed integration:** optional map layers for verified food sources, seasonal availability, community kitchens, food access, transport/cold-chain constraints, local energy/water conditions, and shocks. Each layer needs licensing, timestamp, geographic coverage, and data-quality labels. Do not extrapolate sparse data as if it were live inventory.

**Acceptance test:** map-assisted plans beat a location-blind baseline on measured local availability/cost and retain feasibility during stress scenarios; unsupported areas and data age are reported.

### Shared boundary

Use one portable benchmark-result contract across repositories, but avoid a dependency tangle. Symthaea owns culinary decision logic; Mycelix owns social/evidence coordination; Symtropy owns physical/operational simulation; Sol Atlas supplies spatial context. Share versioned schemas and fixtures through public repository interfaces, not private monorepo imports.

## Benchmark result contract

Every run should emit a machine-readable receipt with:

- schema version and benchmark ID/version;
- repository and exact commit SHA;
- dataset/source hashes, licences, and retrieval dates;
- scenario, cuisine/context, and difficulty;
- seed and model/policy version;
- baseline identity;
- attempted, passed, failed, abstained, and not-applicable counts;
- severe failures and intermediate-trajectory violations;
- per-metric values, denominators, uncertainty/confidence intervals where appropriate;
- hardware/runtime details for performance claims;
- limitations and manual adjudication;
- explicit status: **pass**, **fail**, **incomplete**, or **not comparable**.

A queued CI job is incomplete. A green aggregate is not a pass when required cases are missing, the tested commit differs from the claimed commit, dataset identity cannot be verified, or a negative control stops failing. Save raw failure cases so regressions are diagnosable.

## Execution plan: start without waiting for CI

### Slice 1 — benchmark existing ground truth

- Lock the current Ahn data provenance with content hashes.
- Preserve existing direction checks as a regression test.
- Add a report printing every cuisine result, null-model configuration, recipe count, and sensitivity to random seed/algorithm choices where applicable.
- Label this as corpus analysis, not sensory-quality scoring.

### Slice 2 — adversarially test the current validator

- Enumerate every validator and its valid, boundary, and invalid states.
- Add positive cases just inside and negative cases just outside thresholds.
- Add malformed-data and non-finite-number cases where applicable.
- Record whether each failure is rejected, reported uncertain, or erroneously accepted.
- Do not edit thresholds simply to make the suite pass; resolve failures against cited physical/food-science evidence.

### Slice 3 — close nutrition and feasibility data gaps

- Define provenance-aware ingredient identity and units before importing nutrient data.
- Start with a small, auditable benchmark corpus; record provider identity, licence, version, and limitations.
- Compare with hand calculations and independently reviewed reference recipes.
- Add price, local availability, equipment and timing only as separate evidence sources—never bundle them into an opaque confidence score.

### Slice 4 — compare policies fairly

- Establish random and deterministic domain-heuristic baselines before adjusting the active-inference palate.
- Pre-register seeds, target ranges, score definitions, hard constraints, and stopping conditions.
- Report safe completion, overshoot, time-to-target, target error, and abstention for all seeds.
- Add model complexity only when a measured failure mode identifies what it must fix.

### Slice 5 — run human and community pilots

- Begin with opt-in, low-risk, non-clinical cooking scenarios and informed consent.
- Use blinded, randomised sensory comparisons with an agreed protocol.
- Measure completion, cost, time, waste, satisfaction, and safety separately.
- Review cultural coverage and contribution rights with practitioners, not just model-generated personas.
- Keep deployment claims narrower than the tested population and conditions.

## Scope boundaries

This RFC does **not** claim that any model is a chef, that flavor overlap predicts human preference in general, that a toy simulation validates a physical process, that ingredient databases replace clinical judgement, or that cryptographic provenance proves a physical claim. It does not propose autonomous operation of knives, burners, pressure vessels, or food-safety-critical equipment as a first milestone.

The product bar is a system more honest and more useful than a recipe chatbot: feasible recommendations, explainable trade-offs, traceable evidence, meaningful abstentions, and measurable real-world improvement.

## References and methodological anchors

1. Ahn, Y.-Y. et al. (2011), [Flavor network and the principles of food pairing](https://www.nature.com/articles/srep00196), *Scientific Reports* 1, 196. Supports reproducible analysis of ingredient/compound patterns, not a universal claim about human taste.
2. Xu, J. et al. (2026), [From plausible menus to trustworthy nutrition: a risk-proportionate VALID-Diet framework for artificial intelligence-generated dietary advice](https://doi.org/10.3389/fnut.2026.1957211). Published 13 October 2026; perspective proposing staged, risk-proportionate evaluation. It is not a regulatory standard or consensus guideline.\n3. ISO 11136:2014, [Sensory analysis — guidance for hedonic tests with consumers](https://www.iso.org/standard/50125.html). The ISO page states this edition was reviewed and confirmed in 2025.
3. USDA Food Safety and Inspection Service, [Safe Minimum Internal Temperature Chart](https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/safe-temperature-chart). Use current authoritative limits, measurement location, and rest-time conditions in applicable fixtures.
4. USDA, [FoodData Central](https://fdc.nal.usda.gov/). A possible nutrient-data source; record dataset type and limitations.
5. ISO/CD TS 20008, [Quantification, measurement, monitoring and reporting methods for excess food and food loss and waste](https://www.iso.org/standard/88144.html). Committee draft under development as of 2026-10-17; not cited as a final standard.
6. Zheng et al. (2026), [AI-driven personalization of food systems: From precision nutrition to climate-resilient supply chains](https://doi.org/10.1038/s41538-026-00901-9), *npj Science of Food*. Context for personalisation, resilience, data silos, and bias; this RFC still requires direct measurement for product claims.
7. [VALID-Diet framework for AI-generated dietary advice](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2026.1957211/full) (2026). Methodological context for feasibility, repeatability, risk-proportionate evaluation, and culturally appropriate review; not a substitute for clinical validation.
