# Plasma pathway selection engine v0.1

**Status:** Executable synthetic-only decision prototype.  
**Owner target:** `PLASMA-PROCESS-PATHWAY-SELECTION-001` in `data/research/plasma_energy_reference_cases_v0_1.json`.  
**Implementation:** `xtask/src/plasma_pathway_decision.rs`, compiled through the isolated `xtask/plasma-audit-harness`.

## Research proposition

Select the best *qualified route to one accepted industrial service*, not the most impressive plasma metric. A route is comparable only when the candidate set shares:
- functional unit and accepted product/service contract;
- feedstock/impurity profile;
- complete grid/fuel-to-accepted-product boundary;
- currency and cost basis;
- coproduct-credit policy.

Plasma and non-plasma incumbents must be present in each chemistry group. Hard product constraints are non-compensatory: a route that misses purity or recovery does not become acceptable because it is cheap or energy-efficient.

## v0.1 algorithm

1. **Eligibility:** validate matched service, feedstock, boundary, functional unit, currency, metric units, common uncertainty pairing and a mandatory `NO_CREDIT` baseline.
2. **Paired uncertainty:** use a pinned seed and at least 10,000 draws. `COMMON_EXOGENOUS` bands share the same deterministic random percentile across candidate routes; `ROUTE_SPECIFIC` bands use route-keyed draws. This is a common-random-number variance-reduction technique, not a claim that all uncertainties are perfectly correlated in the physical system.
3. **Hard feasibility:** evaluate every declared minimum/maximum product constraint in each draw. A route is robustly feasible only when its estimated feasibility probability reaches the frozen threshold.
4. **Multi-objective analysis:** report probability of Pareto membership, pairwise probability of dominance, and mean / 95th-percentile regret independently for cost, grid energy and GHG intensity. Do not create a weighted aggregate or a single winner.
5. **Abstention:** unmatched service contracts return `ABSTAIN_NOT_COMPARABLE`; fewer than two robustly feasible routes return `ABSTAIN_INSUFFICIENT_FEASIBLE_ROUTES`; missing or inconsistent evidence returns `BLOCKED`.

## Real source register versus synthetic algorithm fixture

The current three literature candidate groups (NH3 cracking, methane pyrolysis and biogas-to-hydrogen) are still marked `NOT_YET_MATCHED`. The CLI checks that each contains plasma and non-plasma candidates, then returns `ABSTAIN_ALL_GROUPS_NOT_COMPARABLE`. It does not execute a performance ranking over those literature summaries.

The 10,000-draw fixture is deliberately synthetic. It proves that the algorithm executes, is deterministic for a fixed seed, enforces paired inputs and hard constraints, and emits Pareto / regret outputs without scalarization. Its numbers are not derived from the cited papers, and no output is a physical or economic result.

## Evidence and statistical limitations

- The triangular low/mode/high distributions in the fixture are invented software-test values, not fitted distributions from measured plasma campaigns.
- A shared pairing key induces a common percentile/rank coupling; it is not a general covariance model. Real use needs source-derived correlation structure, scenario provenance and sensitivity analysis.
- 10,000 Monte Carlo draws reduce numerical sampling noise for the declared input distributions; they do not validate those distributions, prove calibration or erase model uncertainty.
- `false_feasible_rate` is not estimated by v0.1. It requires an independent labelled prospective validation set and explicit decision/acceptance outcomes.
- Value-of-information measurement selection is not implemented yet.
- Baselines (best-qualified-incumbent, fixed-weighted-sum, TOPSIS, NSGA-II/equivalent, random/space-filling and uncertainty-only selection) are not yet run head-to-head.
- The prototype has no operating authority and cannot accept physical evidence as qualified merely because it parses.

## Next benchmark tranche

1. Add source-resolved probability distributions with empirical provenance and a documented copula/correlation model.
2. Implement best-qualified-incumbent, fixed-weighted-sum, TOPSIS and NSGA-II baselines as comparisons only; preserve Pareto outputs as the decision surface.
3. Add prospective sealed-holdout evaluation for calibration, false-feasible rate, decision regret and abstention quality.
4. Add expected value-of-information for selecting the next lab measurement under a fixed experiment budget.
5. Add full-chain plant economics and dynamic renewable/load-following service once product, power, heat-recovery and durability boundaries are source-matched.

## Verification

Nine authored Rust tests cover:
- 10,000-paired-draw execution and Pareto/dominance output;
- fixed-seed determinism;
- abstention for the current unmatched source groups;
- retention of non-plasma incumbents;
- mismatched common-random pairing;
- non-compensatory product-quality constraints;
- mandatory no-credit policy;
- mismatched functional units;
- minimum draw count.

All tests run in the isolated Rust 1.96 workflow. The synthetic fixture is a software test only; it is not a physical process benchmark.


## v0.2 implementation increment — Monte Carlo confidence and baseline suite

The v0.2 branch `research/plasma-pathway-decision-v0-2` adds decision-report diagnostics without changing the v0.1 feasibility or abstention policy.

### Sampling uncertainty

Each route reports:
- feasibility probability with successes/trials;
- Monte Carlo standard error;
- 95% Wilson score interval;
- probability of robust-Pareto membership with its own interval;
- mean regret plus Monte Carlo standard error and p95 regret.

Each pairwise dominance estimate reports the joint-feasibility denominator, Monte Carlo standard error and Wilson interval. Pairwise intervals are conditional on draws in which both routes satisfy their hard constraints. Pareto-membership events use all configured draws, so the estimate includes both feasibility and non-domination.

These intervals quantify numerical Monte Carlo sampling uncertainty under the declared synthetic distribution and pseudorandom draw construction. They do **not** validate triangular input bands, model uncertainty, physical calibration or transfer to unseen devices/campaigns. More draws narrow Monte Carlo intervals but do not repair a bad input distribution.

### Diagnostic baseline suite

For the same sampled route outcomes and robustly feasible candidate set, the report now includes four diagnostic rankings:

1. `BEST_QUALIFIED_NON_PLASMA_INCUMBENT_BY_MEAN_COST` — the best robustly feasible non-plasma incumbent by conditional mean cost.
2. `FIXED_EQUAL_WEIGHT_NORMALIZED_SUM` — a deliberately simple equal-weight normalized diagnostic, not an approved preference model.
3. `TOPSIS_EQUAL_METRIC_DIRECTION` — equal-direction TOPSIS over cost, grid energy and GHG.
4. `CENTRAL_MEAN_NONDOMINATED_SORT` — deterministic non-dominated sorting on conditional mean metrics. This is not NSGA-II and is not a stochastic optimizer.

Every baseline is marked `diagnostic_only=true`; none can override missing service comparability, a failed hard constraint, the robust Pareto output or an abstention disposition. The central-mean baseline is explicitly named so it cannot be misreported as NSGA-II.

### v0.2 tests added

Six tests check Wilson interval behavior, interval narrowing with more draws, all-success/all-failure limits, pairwise joint-feasible denominators, baseline metadata, and preservation of abstention under failed hard constraints.

### Remaining v0.2 work

Not yet implemented:
- empirical/joint input distributions and dependence models;
- independent seed/draw-count sensitivity reports;
- TOPSIS/weighted/NSGA-II head-to-head regret on a sealed validation corpus;
- empirical calibration, false-feasible estimates and prospective holdouts;
- value-of-information experiment selection.

The current 10,000-draw pathway fixture remains synthetic software-test input. The real NH3, methane and biogas-to-hydrogen source groups remain `NOT_YET_MATCHED` and correctly abstain. Do not interpret the synthetic baselines or probability intervals as a physical route ranking.
