# Culinary decision quality and value-of-information benchmark v1

**Status:** benchmark contract proposal, stacked on the draft decision-bundle validator PR. No runtime implementation, held-out result, household study, or measured benefit is claimed by this document.

## Why this is a different problem

A correct Pareto frontier is necessary but not sufficient for a useful culinary assistant. A system can correctly compute a frontier from stale prices, overstate evidence, ask for measurements that cannot change a decision, or recommend a high-average plan that breaks an essential constraint for one household group.

The next paradigm shift is **from ranking recipes to choosing the next defensible decision**:

- recommend when evidence and constraints support a stable choice;
- preserve a choice set when alternatives trade off or scenario assumptions reverse the winner;
- ask for a measurement when that measurement could change an action enough to justify its cost;
- stop and escalate when a non-compensatory safety, allergen, rights, or consent gate is failed or unresolved;
- remain explicitly uncertain when evidence is stale, missing, conflicting, revoked, or outside the model's qualified scope.

The benchmark consumes the immutable decision receipt defined by [Culinary cross-system interop v1](../docs/CULINARY_CROSS_SYSTEM_INTEROP_V1.md) and the robust Pareto result. It is **not** a second Pareto solver or value-of-information engine. The existing Symtropy value-of-information / measurement-action work remains the canonical planner; this benchmark evaluates its decisions through a narrow adapter.

## Evidence motivating the design

A 2025 review of 101 food-composition databases spanning 110 countries reported aggregate FAIR criteria scores of 30% for accessibility, 69% for interoperability, and 43% for reusability; the authors highlighted gaps in machine-readable access, update frequency, provenance, and licensing. Nutrition rows therefore cannot safely be treated as interchangeable facts merely because they share a nutrient name. [Brinkley & Gallo-Franco, *Frontiers in Nutrition* (2025)](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1552367/full).

For South African affordability context, PMBEJD publishes a monthly Household Affordability Index and food-basket measures, including a Johannesburg basket. Those are valuable contextual time series, but they are not a live SKU-level quote for a particular shop, package size, stock state, or household. [PMBEJD Household Affordability Index](https://pmbejd.org.za/index.php/household-affordability-index/). FAO's FPMA tool supports historical price comparisons across markets, commodities, and seasons; the source market and price series scope still need to be checked before using it as a household retail price. [FAO FPMA](https://www.fao.org/giews/food-prices/price-tool/en/).

These sources motivate strict source-type and freshness handling. They do not supply measured outcomes for this benchmark.

## Decision dispositions

Every benchmark episode must return exactly one primary disposition:

| Disposition | Meaning |
|---|---|
| `recommend` | Return a feasible candidate or a clearly stated, user-configurable choice; cite decision-relevant evidence and its age. |
| `preserve_choice_set` | Keep multiple non-dominated or assumption-sensitive options visible; do not imply a universal winner. |
| `request_measurement` | Name a feasible measurement, its expected decision impact, the cost/burden estimate, and the source/consent conditions needed to collect it. |
| `human_review` | A required gate is unresolved, evidence conflicts materially, or the case exceeds qualified scope. No model confidence overrides this disposition. |
| `no_feasible_candidate` | All considered candidates fail a hard constraint or no feasible candidate is evidenced. Do not relax constraints silently. |

The system may also include non-exclusive flags such as `stale_evidence`, `missing_objective`, `source_conflict`, `scenario_sensitive`, and `privacy_limited`. Flags do not replace the primary disposition.

## Non-compensatory invariants

1. A failed or unknown applicable food-safety, allergen, consent, rights, or explicitly user-defined minimum-adequacy gate cannot be offset by lower price, taste, lower footprint, or any other score.
2. Missing, stale, conflicting, revoked, or suppressed values are not numeric zeroes, not a default mean, and not current observations.
3. Each price is bound to product identity, pack size, currency, unit conversion, market, date/time, availability state, and source class. Each composition value is bound to food identity, edible/preparation state, basis, component identifier, method/source, and release/version where available.
4. `valid_at` (when the fact applied) and `known_at` (when the system could know it) are distinct. Historical replay must not use evidence published or received after the decision time.
5. A signed, replicated, or attested source claim is provenance, not proof the food was physically present, safe, purchased, consumed, or delivered.
6. Forecast, simulation, planned stock, observed availability, sale, and fulfilled delivery stay separate evidence classes.
7. No household identity, address, receipt history, or precise home location is emitted by default. Spatial analysis receives only the precision/aggregate justified by consent and data policy.
8. No probability is inferred from a scenario list. Expected value of sample information is permitted only when a calibrated predictive distribution, target utility/loss, measurement model, and outcome path are justified. Otherwise report a probability-free interval/sensitivity or worst-case-regret analysis and label it accordingly.
9. A source correction, recall, expiry, or consent revocation must invalidate or downgrade dependent cached claims according to the source contract; a previously generated receipt is not automatically current.
10. The evaluator is separate from the decision producer. Oracle labels and synthetic truths are not available to the candidate policy during a scored episode.

## Evaluation tracks

### Track A — deterministic contract correctness

Use the synthetic fixture corpus in `../fixtures/culinary-decision-quality-v1/corpus.json`. Fixture cases are known before any tuning. Add an independently authored oracle, not expected outputs generated by the implementation under test.

Required reports: per-case expected/actual disposition, hard-gate violations, stale-as-current errors, incorrect source-class upgrades, missing-data imputation, privacy violations, and explanation/provenance completeness.

### Track B — decision robustness and abstention quality

Compare at least these arms on identical candidate/evidence budgets:

1. Point-estimate scalar ranking, with its weights explicitly frozen.
2. Standard Pareto frontier.
3. Scenario-aligned robust Pareto.
4. Robust Pareto plus the canonical Symtropy measurement-action / value-of-information policy, if available through an adapter.
5. Full-information oracle, clearly marked as a non-deployable upper bound.
6. Simple conservative policy (review/retain alternatives where evidence is insufficient).

Report separately:
- regret against a declared task-specific loss function (do not invent one universal scalar for every objective);
- worst-case and per-scenario regret where the scenario set supports it;
- frequency and severity of decision reversals;
- useful-query precision and recall, unnecessary measurement rate, cost/time burden per useful decision change, and missed high-impact queries;
- selective risk vs coverage for any model that may abstain;
- the rate of wrong confident answers versus correct abstentions, and the converse rate of unnecessary abstentions;
- performance by evidence freshness, source class, missingness, market, affordability stratum, and scenario family.

Do not reward an abstain-only policy as successful merely because it makes no false claims. It must also report coverage, avoidable deferrals, and whether its suggested next measurement would resolve the decision.

### Track C — value of information (VOI) qualification

Use the existing Symtropy planner; do not clone its algorithm inside Symthaea.

For synthetic episodes with known hidden truths, the evaluator can compute the oracle-best query for a fixed cost/budget and compare the selected measurement to that oracle. For empirical episodes, do not expose hidden truths to the policy or tune against post-decision outcomes.

For probability-free cases, define the decision loss first and score the reduction in **worst-case regret** over a declared uncertainty set after the candidate measurement. Call this a robust measurement priority, not expected VOI.

For expected VOI, preregister the probability model, calibration cohort, measurement result model, utility/loss function, acquisition cost, and evaluation split. Report calibration and utility separately. If these prerequisites are missing, expected VOI is **not scoreable**.

A query is only feasible when its data source is accessible under policy, collection burden is acceptable, and any required consent is explicit. A mathematically informative query that cannot be lawfully or practically collected is not a valid action.

### Track D — external validity and human outcomes

After the deterministic tracks pass, run a separate consented prospective pilot. Preregister eligibility, privacy rules, comparison arm, outcomes, exclusion rules, stopping conditions, and analysis before outcome collection. Use transparent baselines (for example, the household's usual basket/planning process and a simple rule-based substitution list).

Report independently:
- out-of-pocket cost actually paid, not just predicted basket price;
- constraint satisfaction and nutrient adequacy against a qualified reference, with uncertainty;
- preparation time and equipment/energy needs;
- edible waste and avoidable waste;
- availability / failed substitutions and time spent shopping;
- household acceptability, cultural fit, and effort burden;
- distribution of outcomes across relevant groups, including worst-group performance where adequately sampled.

Do not collapse these into one headline score. Do not claim causality from synthetic fixtures, retrospective replay, or an uncontrolled pilot. Clinical recommendations, allergy safety certification, and measured health outcomes are outside this benchmark's claim boundary.

## Preregistered acceptance targets

These are proposed targets, not achieved measurements:

- **Contract:** 100% agreement with independent expected dispositions on the published deterministic fixtures.
- **Hard gates:** zero cases in which failed/unknown applicable hard gates are marked feasible, ranked above a gate-compliant candidate, or offset by a soft objective.
- **Evidence semantics:** zero stale/missing/conflicted/revoked facts silently promoted to current observations; zero simulation/forecast/planned inventory relabelled as physical observation or fulfillment.
- **Time travel:** zero decision-time replays using facts whose `known_at` is later than the replay cutoff.
- **Privacy:** zero prohibited precise household identifiers in consumer-facing decision receipts and projections.
- **Repeatability:** 100% identical disposition and canonical evidence references under input-order permutations, with exact producer commit/config/fixture digests recorded.
- **VOI:** score query-choice regret, useful-query precision/recall and unnecessary burden against a frozen oracle on hidden synthetic cases; publish all strata. Freeze a minimum utility threshold only after a baseline is measured, before opening the held-out set.
- **Calibration:** if probabilities are emitted, publish proper scoring rules (for example Brier/log loss), reliability plots, and subgroup/cohort limits. Do not treat a scenario count or an HDC similarity score as a probability.
- **Runtime:** first measure release-mode latency, allocations/memory and adapter overhead at 100/1,000/10,000 candidates and increasing evidence/scenario counts. Set regression budgets only after a reproducible baseline on named hardware and the exact commit.
- **Household benefit:** no universal percentage-improvement target until the pilot has a power/precision justification, acceptable burden, and a preregistered outcome model. Report uncertainty and null/negative outcomes.

A threshold change after examining the held-out data requires a new benchmark version and a new held-out cohort.

## Cross-system responsibility split

- **Symthaea Culinary:** candidate-level decisions, hard gates, robust Pareto output, evidence gaps, decision disposition, and an auditable request for a useful next measurement.
- **Symtropy:** canonical scenario/portfolio analysis and measurement-action / VOI planning, when the probability or robust-regret assumptions are explicit.
- **Mycelix:** source rights/consent, product and lot identity, producer claims, canonical correction/recall lineage, and observed order/fulfilment events. A signature or reputation score does not establish physical safety.
- **Sol Atlas:** spatial/temporal context, coverage, access and market-area representation with privacy-preserving granularity; not authority for exact stock or household identity.

The contract stays read-only. A recommendation does not place an order, spend money, disclose a receipt, direct a vehicle, or actuate a kitchen, farm, or storage system.

## Reproduction checklist

Every result must carry the exact Symthaea and adapter commit SHAs, fixture-corpus digest, evaluator version, candidate-policy/config digest, scenario-manifest digest, source snapshot and license references, seed where applicable, hardware/runtime profile, and a complete per-case results artifact. Queued or skipped CI is not a pass. Authored tests are not executed tests; static fixture checks are not Rust conformance evidence.

## References

- Brinkley, S. B. & Gallo-Franco, J. J. (2025). [The state of food composition databases: data attributes and FAIR data harmonization in the era of digital innovation](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1552367/full). *Frontiers in Nutrition* 12, 1552367.
- Pietermaritzburg Economic Justice & Dignity. [Household Affordability Index](https://pmbejd.org.za/index.php/household-affordability-index/).
- Food and Agriculture Organization of the United Nations. [Food Price Monitoring and Analysis (FPMA) Tool](https://www.fao.org/giews/food-prices/price-tool/en/).
- Existing internal interfaces: [Symtropy AGRI-VOI-001 #1661](https://github.com/Luminous-Dynamics/symtropy/issues/1661); [Symthaea robust Pareto PR #7951](https://github.com/Luminous-Dynamics/symthaea/pull/7951); [Symthaea evidence/local basket issue #7952](https://github.com/Luminous-Dynamics/symthaea/issues/7952).
