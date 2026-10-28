# Aviation Network Robustness Under a Frozen Scenario Set (ANR-001)

**Status:** research implementation, not yet compiled or benchmarked.  
**Implementation:** `crates/domains/symthaea-operations-research/src/aviation_network_robustness.rs`  
**Companion:** [AVIATION_NETWORK_EXTERNALITY_BENCHMARK.md](./AVIATION_NETWORK_EXTERNALITY_BENCHMARK.md)

## Research question

Can a network-level aviation recovery policy deliver useful average benefit while respecting per-actor harm limits and containing severe scenario regressions across a predeclared stress ensemble—not just on one nominal scenario?

The central methodological shift is from selecting on one point estimate to **screening on an explicit finite scenario set**, while retaining distinct measures of mean benefit, worst-case loss, tail regression and worst actor impact.

This is not a distribution-free guarantee. It guarantees only what the checker establishes over the supplied finite scenarios, subject to the truth of their inputs, modeled costs and feasibility labels.

## Why this benchmark is needed

EUROCONTROL's September 2026 CWBO report describes explicit network triggers for severe weather, weather combined with capacity reductions, coordinated capacity shortages and emerging network instability. It reports around 3.1 million delay minutes saved from 1 May through 31 August 2026, €400 million estimated operational cost savings using €130 per ATFM delay minute, and 14% higher network stability on challenging days. These are program-level reported results, not a directly comparable score for this prototype: https://www.eurocontrol.int/publication/how-capacity-and-weather-based-operations-cwbo-are-building-network-resilience-capacity.

A newly indexed 2026 *Journal of Air Transport Management* paper proposes collaborative decision-making for airline irregular-flight recovery with a network manager. Airlines submit alternative recovery schedules, the network manager selects/adjusts them under updated airport capacity, and a fairness mechanism limits reversed flight-pair counts. It reports experiments on 32 scenarios derived from realistic Chinese airline datasets under airport-capacity reduction. This is a particularly relevant design comparator; reproduction requires compatible data and objective definitions: https://www.sciencedirect.com/science/article/pii/S1366554526003169.

Research on robust airline scheduling with turnaround uncertainty also warns that robustness has a cost and can become over-conservative when its extra resource costs are not modeled. Scenario design, model choice and evaluation metrics materially affect conclusions: https://doi.org/10.1016/j.tre.2025.104440 and https://doi.org/10.1016/j.cor.2018.04.011.

## Implemented surface

The pure-std Rust module exports:

- `evaluate_robust_network_candidate`: joins baseline and candidate outcomes by unique scenario IDs (not array position), requires equal scenario sets, requires the same actor universe and actor domains, and delegates every paired outcome to the network-externality ledger.
- `RobustNetworkReport`: reports mean benefit, worst-scenario benefit, upper-tail mean of positive system regressions, worst actor regression and its scenario ID, count of regressing/improving/unchanged scenarios, and sorted per-scenario outcomes.
- `assess_robust_network_candidate`: applies a per-actor regression ceiling in every supplied scenario, a minimum mean benefit, a minimum worst-case benefit and a maximum empirical upper-tail regression. It rejects a mismatched tail fraction so the evaluated risk measure cannot silently differ from the configured gate.
- Malformed scenario sets, duplicate IDs, actor-universe drift, domain changes, invalid tail fractions, and rejected/indeterminate feasibility status are errors, not penalties that can be traded away for lower cost.

The module evaluates declared outcomes; it does not generate plans, independently check aircraft/dispatch feasibility, validate cost calibration, learn scenario probabilities, or certify generalization to unseen disruptions.

## Frozen benchmark design

Create an immutable scenario corpus with scenario ID bound to an input digest and a frozen objective version. Each policy receives the same information, feasible action set and compute budget.

Policy set:
1. incumbent/do-nothing;
2. rule-based trigger policy reimplemented independently from the public operational protocol;
3. airline-local optimizer;
4. network-total-cost optimizer;
5. network-cost optimizer with per-actor harm ceilings;
6. finite-set robust selector;
7. conventional MILP/CP-SAT or decomposition when constraints/data are comparable;
8. hindsight oracle, explicitly non-deployable.

Scenario families:
- severe weather and capacity reductions;
- emerging instability and correlated airport/airspace shocks;
- gate/stand bottlenecks and cascading turnarounds;
- aircraft grounding, qualified-part shortage and maintenance-window conflicts;
- alternate-airport/rerouting choices;
- missing, stale or contradictory evidence;
- compound events and held-out disruption families;
- scenarios in which the incumbent or do-nothing option is best.

## Metrics

Report separate ledgers, not one blended score:

- mean and median system cost change;
- worst-case system benefit/regression;
- upper-tail mean positive regression and CVaR-style measures at predeclared levels;
- worst individual actor regression and which actor/scenario causes it;
- actor-level gain/burden and externality ratio;
- cancellations, non-cancelled delay minutes, passenger outcome measures and kg CO2e;
- replan count, plan churn, runtime, memory and timeout/abstention rate;
- constraint failures, stale-evidence detections, indeterminate outcomes, and exclusions;
- paired effect sizes and cluster-aware confidence intervals over independent disruption episodes.

Cancellation and non-cancelled-delay measures must remain mutually exclusive. CO2e remains a separate physical ledger unless a sourcing and calibration protocol defines a versioned conversion. Any harm thresholds, minimum network benefit and tail threshold must be selected and preregistered with stakeholder input before final test results are inspected.

## Proposed acceptance gates (not achieved results)

- 100% agreement with an independent evaluator on hand-computable cases.
- Zero silent mismatch between baseline/candidate scenario IDs, actor sets, actor domains or objective versions.
- Zero accepted candidate violating a predeclared actor-harm cap in any scenario in the frozen set.
- A paired improvement on the preregistered primary system metric over the strongest feasible baseline, with the 95% interval excluding no improvement.
- No unreported deterioration in declared tail-risk, subgroup or actor-burden measures.
- Ablations distinguish gains from broader action sets, more compute, the scenario set, and the robust gate itself.
- Publish the finite-scenario boundary explicitly; passing does not establish safety, causal savings, airworthiness, dispatch authority, maintenance release or regulator approval.

## Implementation test status

Twelve Rust unit tests are defined in the module, including:
- average improvement cannot hide a worst-case regression;
- per-actor harm caps are applied across all scenarios;
- exact scenario-set matching and duplicate-ID rejection;
- actor-set/domain stability;
- rejected/indeterminate plans are not scored as feasible;
- deterministic report ordering;
- consistent tail-fraction semantics.

No Rust compile/test is claimed in the current environment because `cargo` and `rustc` are unavailable. PR-related workflow records seen for the research branch are skipped, not passed.
