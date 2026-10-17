# Evidence-aware Pareto decisions for culinary candidates

**Track:** CULINARY-PARETO-001  
**Schema:** v1  
**Implementation:** `src/pareto.rs`  
**Status:** committed proposal; Rust tests not yet executed in this environment

## Paradigm shift

Do not ask the model to compress nutrition, cost, waste, preparation time, availability and environmental burden into a single "best recipe" score. A weighted score can hide whose preferences determine the weights and allow an improvement in one dimension to compensate for a hard safety failure in another.

Instead, use two distinct decisions:

1. **Eligibility:** hard gates determine whether a candidate is eligible for objective comparison. Any explicit failed gate blocks the candidate; an unresolved gate makes it indeterminate. Neither state enters the Pareto frontier.
2. **Trade-offs:** candidates with every gate passing and every objective measured with a finite, referenced value are compared using Pareto dominance. There is no implicit conversion of units, no imputation of missing values, and no universal winner when candidates trade off objectives.

This supports a practical question: "Which candidates are defensible options, and what is gained or sacrificed by choosing each?" It does not claim to solve nutrition science, infer actual allergen safety, or authorize food preparation.

## Input contract

Every candidate in one comparison set must use the same gate keys and same objective keys/directions.

A hard gate contains:
- stable key, such as `allergen_evidence_reviewed` or `food_safety_constraints`;
- explicit `pass`, `fail`, or `unknown` state;
- a non-empty evidence reference for Pass/Fail. Unknown may omit a reference.

An objective contains:
- stable key and direction (`minimize` or `maximize`);
- a finite number plus a non-empty provenance reference, or `None` to mean unknown/missing.

The evidence reference is a pointer supplied by the caller, not cryptographic verification. Before comparison, the caller must normalize each objective to consistent units and definitions, version source data/calculation receipts, and establish provenance. This module checks shape and deterministic comparison semantics only.

## Status semantics

| Gate disposition | Objective data | Ranking disposition | Enters frontier? |
|---|---|---|---|
| Any gate fails | Any | `not_ranked` | No |
| No failure, at least one gate unknown | Any | `not_ranked` | No |
| All gates pass | One or more objective values missing | `incomplete_metrics` | No |
| All gates pass | Every objective finite and referenced | `ranked` | Only if non-dominated |

A candidate with a known failed gate is `blocked` even if another gate is unknown. This reports the decisive exclusion without pretending the remaining gate is resolved. A missing objective does not become zero, average, a penalty, or a made-up default. Such a candidate remains visible in the report but is not directly ranked.

## Pareto rule

For each registered objective, values must share comparable units and direction. Candidate A dominates B iff:
- A is no worse than B on every objective; and
- A is strictly better than B on at least one objective.

A candidate is on the frontier when it is fully rankable and no other rankable candidate dominates it. Ties do not dominate one another. Frontier results and explanation lists are deterministically sorted by candidate ID, independent of input order or objective declaration order.

## Initial acceptance targets

These are proposed gates, not measured performance:

| Dimension | Proposed target |
|---|---|
| Hard-gate integrity | Zero blocked or indeterminate candidates appear on the frontier, across all fixed fixtures |
| Missing-data integrity | 100% of candidates with any missing objective are explicitly `incomplete_metrics` and never silently imputed |
| Schema integrity | 100% of empty/duplicate keys, inconsistent schemas, unreferenced values and non-finite objectives are rejected |
| Oracle correctness | 100% exact agreement on preregistered hand-computed frontiers, ties, mixed objective directions and dominance relations |
| Deterministic replay | Candidate input permutation and objective-key ordering do not change the report |
| Performance | Initial target: 1,000 candidates × 8 objectives in p95 ≤100 ms in release mode on documented hardware; measure separately for all-nondominated and heavily-dominated cases |
| Decision utility | Before any superiority claim, compare choices against transparent baselines and later test whether selected options improve measured household outcomes |

Performance values are new targets to benchmark, not current results. Pareto enumeration is pairwise and has worst-case work (O(n^2m)), where (n) is candidate count and (m) is objective count. The target must be measured on representative and adversarial distributions before adding acceleration structures or approximate frontiers.

## Integration plan

- **Symthaea:** computes estimates, hard-gate evaluations, uncertainty and Pareto summaries. A source uncertainty or failed gate never becomes a recipe recommendation through optimization.
- **Mycelix:** holds versioned evidence, consent, product/lot/supplier links and correction/recall events. A signature or provenance pointer is not proof that the underlying food claim is true.
- **Symtropy:** can later rehearse preparation-workcell schedules and resources; simulated outcomes remain marked simulated until compared with measured execution.
- **Sol Atlas:** can later add dated/local context for prices, access and supply disruptions. It must report coverage and age; unknown inventory cannot be substituted with a prediction labelled as fact.

Do not connect to a health-specific personalization pathway until intended-use, clinical boundaries, source quality and independent review are specified. The first milestone is auditable non-clinical meal candidate comparison.

## Required follow-on benchmarks

1. **Known-truth oracle set:** hand-computed 2D/3D frontiers, dominated and nondominated trade-offs, exact ties, minimize/maximize mixtures, and shuffled inputs.
2. **Adversarial semantics:** a zero-cost candidate with failed safety gate, unknown allergen evidence, missing cost/environment values, NaN/infinity, mismatched units/objective schema, duplicate IDs, stale price references, and conflicting data versions.
3. **Uncertainty and robust choices:** a later layer should use validated intervals/scenarios rather than invent point estimates. Report candidates whose dominance relationship is stable across uncertainty scenarios separately from candidates whose ordering changes.
4. **Real-context comparison:** local, dated basket prices; portion-size and household feasibility; measured prep time/waste; cultural review; and environmental indicators only where boundaries/data support them.
5. **Human outcome:** blind usability/acceptability study with pre-registered baseline and non-inferiority/benefit definitions. Do not claim measured affordability, waste or nutrition gains from simulated plans.

## Research anchors

- [Multicriteria Measures to Assess the Sustainability of Diets: A Systematic Review](https://academic.oup.com/nutritionreviews/article/83/3/e980/7701449), *Nutrition Reviews*. The review identified six existing multicriteria measures from only seven eligible studies; sociocultural dimensions appeared in only one measure. This is a reason to expose criteria and gaps, not evidence that this implementation is better.
- [Optimizing Food Systems for Health, Affordability, and Sustainability](https://engineering.jhu.edu/designcenter/2026/04/27/optimizing-food-systems-for-health-affordability-and-sustainability/), Johns Hopkins Engineering Design Center (27 April 2026). A student research project uses multi-objective food bundles across US and Argentina contexts and describes cost/nutrition/emissions trade-offs; it is a useful design example, not a broad clinical validation.
- [Multi-objective optimization of national dietary guidelines: balancing nutrition, environment, and economy](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2026.1758724/full), *Frontiers in Nutrition* (2026). Reports country-dependent cost and emissions trade-offs in a four-country modeling study, supporting the need to retain local context and disclose objective conflict.
- [Improving school lunch menus with multi-objective optimisation: nutrition, cost, consumption and environmental impacts](https://pmc.ncbi.nlm.nih.gov/articles/PMC10410403/), *International Journal of Behavioral Nutrition and Physical Activity*. Demonstrates that an outcome can vary by which objective is optimized; no single priority should be assumed universal.

## Qualification limit

A passing unit-test suite would establish only that this module obeys its typed gate, schema, and dominance contract. It would not prove that objective values are accurate, that evidence references are trustworthy, that a candidate is culturally appropriate, or that a recommended meal is safe or nutritious in the real world. Exact-head Rust test evidence is required before calling the implementation verified.
