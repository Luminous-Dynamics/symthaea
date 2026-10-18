# Scenario-aligned robust Pareto evaluation (CULINARY-PARETO-002)

**Implementation:** `src/robust_pareto.rs`  
**Status:** proposed implementation on a branch stacked on PR #7929; tests are authored but have not been compiled or executed here.  
**Purpose:** preserve the baseline answer while showing whether it survives a preregistered set of plausible, jointly aligned scenarios.

## Paradigm shift

A point-estimate Pareto frontier answers: “Which candidates are non-dominated under these supplied values?” It does not answer whether a candidate remains competitive when assumptions change.

The evaluator accepts named scenarios with caller-supplied version/digest identifiers. Every candidate must provide one metric vector per scenario, and each vector must match the registered scenario digest. It reports:

1. The baseline Pareto report.
2. A complete Pareto report for every scenario.
3. Per-candidate counts for how often it was rankable, frontier, dominated, incomplete, blocked, or indeterminate.
4. A **robust frontier** containing only candidates that are complete in every scenario and are not robustly dominated.
5. Pairwise robust dominance only when one candidate is no worse on every objective in every required scenario, and is strictly better on at least one objective in at least one scenario.

Scenarios are sorted by ID, candidates by candidate ID, and objective definitions by key. Objective keys, directions, and units must agree. Missing values are not replaced by baseline values. Existing hard-gate behavior remains non-compensatory.

## Exact semantics and limits

- A scenario digest is supplied by the caller. This module records it; it does not calculate or authenticate the digest, inspect the assumptions, or prove source truth.
- Scenario probabilities are not inferred. Every declared scenario is treated as required for robust dominance; do not call the result a probability of success.
- V1 gates are candidate-level and assumed invariant across the scenario set. A scenario that changes recipe identity, allergen status, food-safety conditions, or other gate-relevant facts must use a separately identified candidate and re-evaluated gates. It must not reuse a passing gate receipt for a changed recipe.
- Objectives need comparable definitions and explicit units. Matching unit strings is a schema check, not a conversion or dimensional-analysis engine.
- The report is decision support, not clinical advice, food-safety certification, validated life-cycle assessment, or a claim that a meal will be liked.
- There is no weighted average over scenario frontiers. The robust relation is intentionally conservative and is distinct from “usually best,” expected utility, regret minimization, and chance-constrained optimization. Those would require explicit, justified assumptions and separate acceptance contracts.

## Preregistered synthetic acceptance corpus

All fixtures should be versioned before runtime tuning, with hand-computed expected outputs.

| Fixture family | Required assertion |
|---|---|
| Baseline/shock rank reversal | Baseline and shocked frontiers differ when a price shock reverses dominance; neither alternative is falsely called a universal winner. |
| Stable robust dominance | A candidate no worse in every metric/scenario and strictly better somewhere robustly dominates its comparator. |
| Shared scenario assumption | Candidates are evaluated against the exact same scenario IDs and digest set; per-candidate scenario sets and per-candidate metric digests cannot silently drift. |
| Missing scenario value | The candidate is incomplete in that scenario and is ineligible for robust dominance; no baseline carry-forward occurs. |
| Gate failure / unknown | Blocked or indeterminate candidates never enter any scenario frontier or the robust frontier. |
| Schema mismatch | Empty/duplicate scenario IDs, duplicate scenario records, missing scenarios, unknown scenarios, inconsistent objective keys/directions/units, non-finite values, and missing provenance for present values fail closed. |
| Ties and incomparable candidates | Ties do not dominate; genuine trade-offs remain as alternatives. |
| Replay invariance | Permuting scenario, candidate, and objective declaration order yields byte-equivalent canonical JSON after the documented serializer settings are applied. |
| Deterministic oracle | 100% exact agreement with independently hand-computed scenario frontiers, robust dominance relations, and rank-instability counts. |

## Proposed performance and utility gates

These are future targets, **not measured results**.

- **Semantic correctness:** 100% exact agreement with the preregistered oracle corpus.
- **Hard-gate safety:** zero blocked or indeterminate candidates on any frontier.
- **Evidence completeness:** 100% of candidates with missing scenario metrics explicitly reported; no silent imputation.
- **Robust eligibility:** zero candidate receives robust-dominance status unless complete and feasible in every required scenario.
- **Reproducibility:** 100% report invariance under candidate/scenario/objective input permutation.
- **Performance:** first record a release-mode baseline before setting the final budget. Measure 100, 1,000, and 10,000 candidates; 2, 8, and 16 objectives; 1, 4, and 12 scenarios; both high-dominance and all/mostly-non-dominated distributions. Include p50/p95, memory, scenario count, objective count, hardware, compiler/profile, and exact commit SHA. Only then establish a regression budget.
- **Decision utility:** no measured household benefit claim until a preregistered, real-context evaluation compares plans to transparent baselines and measures basket cost, feasibility, preparation time, edible/avoidable waste, nutrient criteria, and acceptability separately.

The direct implementation is pairwise in candidates for each scenario, with a second pairwise pass for robust dominance; complexity is approximately (O(SN^2M)), where (S) is scenario count, (N) candidates, and (M) objectives. The robust relation also performs key-based metric lookup, so benchmarks must measure the implementation as written before optimization claims.

## Data-validation path after synthetic correctness

Do not start by connecting every food API and assuming the combined data is ground truth. Start with a small evidence ledger and one defined geographic market:

1. **Nutrients:** USDA FoodData Central provides downloadable, versioned nutrient datasets and a documented API. Pin the release, food record IDs, nutrient identifiers, per-100-g/serving basis, preparation state, and conversions. The newest available release does not automatically match the local food/product or the recipe preparation state.
2. **Staple-market price context:** FAO's Food Price Monitoring and Analysis (FPMA) tool exposes price series and supports market/commodity/season comparison. Verify the country, market, commodity, period, and retail-vs-wholesale coverage before using it for household basket costs.
3. **Packaged-product cross-checks:** Open Food Facts provides community-contributed product ingredients and nutrition fields. Treat each entry as an attributed observation with freshness and completeness metadata, not authoritative allergen or safety proof.
4. **Actual basket costs:** for the intended pilot market, collect date-stamped, unit-normalized retailer prices or consented receipts and retain item identity, package size, currency, and availability. Report unobserved items as unavailable/unknown instead of backfilling with a market average labelled as a current price.
5. **Scenario provenance:** record the joint assumption set and source snapshot behind every scenario. Example scenario families: observed basket; staple price shock; seasonal produce substitution; transport disruption; alternate but defensible environmental allocation; and nutrient values within a justified data-quality range. Scenario ranges must come from measured history, source uncertainty, or explicitly labelled stress tests—not invented probabilities.

Potential sources:
- [USDA FoodData Central API guide](https://fdc.nal.usda.gov/api-guide/)
- [USDA FoodData Central downloadable datasets](https://fdc.nal.usda.gov/download-datasets/)
- [FAO FPMA tool](https://www.fao.org/giews/food-prices/price-tool/en/)
- [Open Food Facts API documentation](https://openfoodfacts.github.io/openfoodfacts-server/api/)

## Qualification boundary

The implementation and authored unit tests are not evidence of compilation, test passage, runtime performance, truthful ingredient data, or household impact. Before qualification, run formatting, unit tests, Clippy as appropriate, and exact-head CI. Then independently review the oracle fixtures and source-data receipts. A source-data adapter should be a separate increment after the decision semantics are qualified.
