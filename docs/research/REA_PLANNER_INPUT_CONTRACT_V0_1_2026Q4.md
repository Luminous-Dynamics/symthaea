# REA planner input contract v0.1 — 8 November 2026

**Status:** design contract plus one synthetic mathematical known-answer example. The example is not a real target observation, does not qualify the production planners, and is deliberately tagged `math_only_synthetic`.

## Why this exists

The v0 proof-obligation fixtures (#11110) do not currently carry complete point-EIG, robust-EIG or EVSI planner inputs. The existing Rust planner APIs require independently rooted supporting Evidence, normalized hypothesis priors, complete likelihood rows for every candidate/outcome/hypothesis, scenario weights, and complete terminal decision utilities.

This contract makes those separate inputs explicit and binds them to the claim/target scope.

## Two qualification modes — never conflate them

### 1. `math_only_synthetic`

Permits synthetic evidence records solely to define a finite arithmetic oracle. This mode is for direct mathematical known-answer calculations (Bayes updates, EIG, EVSI, scenario mixtures, regret), not for calling the production planner APIs. Existing planners reject synthetic-only support by design.

### 2. `planner_integration_observed`

Requires exact REA Evidence references rooted in actual authorized observations or derivations from them. The adapter must verify each Evidence ID with the REA Evidence contract and preserve independent lineage. It must not relabel synthetic, model-generated or external-reference-only records as independent observations. Only this mode can exercise the existing planner's actual grounding and eligibility path.

A future test-only override may be useful for unit-testing planner arithmetic, but it must be compiled behind `cfg(test)` or a clearly non-production feature, produce a separate claim ceiling, and never satisfy production grounding requirements.

## Contract mapping to current Rust APIs

| JSON field | Existing Rust API |
|---|---|
| evidence_catalog | `EvidenceProvenance` |
| hypotheses | `Hypothesis` |
| experiments + point_likelihoods | `ExperimentProposal` + `LikelihoodRow` |
| likelihood_scenarios | `LikelihoodModelScenario` + `ScenarioExperimentLikelihood` |
| terminal_decisions | `TerminalDecision` + `HypothesisUtility` |
| planner_configs.point | `PlannerConfig` |
| planner_configs.robust | `RobustPlannerConfig` |
| planner_configs.decision_aware | `DecisionAwareConfig` |

No planner is copied or reimplemented by the schema.

## Cross-field checks required beyond JSON Schema

1. Priors sum to 1 and hypothesis IDs are unique.
2. All support/contradiction IDs exist in evidence_catalog; evidence IDs are unique and derivation parents resolve.
3. `planner_integration_observed` hypotheses have independently rooted evidence according to the existing planner rules; a synthetic-only root must reject.
4. Each point experiment has exactly one likelihood row for every hypothesis; every row repeats the exact `outcome_order` labels and sequence declared by the experiment, and its probability vector has that same length and sums to 1.
5. Every robust scenario has exactly one matrix per experiment, with all rows and outcome lengths aligned; scenario weights sum to 1.
6. Every terminal decision has exactly one finite utility for every hypothesis, with no missing or duplicate entries.
7. Every policy arm binds identical claim, evidence catalog, hypothesis, candidate, scope, resource, scenario and utility digests.
8. Predicted outcomes are never applied to claim readiness. Only an executed, independently admitted observation may change obligation state.



Each `LikelihoodRow` now repeats `outcome_order` in the input contract. The adapter maps `outcome_probabilities[i]` only to `outcome_order[i]`, and the checker requires that list to equal the experiment's declared `outcomes` array exactly for both point and robust-scenario likelihood matrices. This catches same-length reorderings that a vector-length check would miss.

## Synthetic known-answer case

The example has two hypotheses with prior 0.5/0.5 and one two-outcome reproduction probe:
- Nominal likelihoods: P(reproduced|vulnerable)=0.9, P(reproduced|not vulnerable)=0.1.
- Conservative scenario: 0.75 and 0.25 respectively.
- The outcome probabilities are 0.5/0.5 under either symmetric scenario.
- Nominal posterior probabilities are 0.9 after reproduction and 0.1 after non-reproduction.
- Nominal EIG is approximately 0.531004 bits.
- Conservative EIG is approximately 0.188722 bits.
- With scenario weights 0.5/0.5, expected scenario EIG is approximately 0.359863 bits; worst case is approximately 0.188722 bits.
- Terminal decisions use a complete utility matrix: mitigate=(3,-6), no_action=(-20,0). Baseline best action is mitigate with expected utility -1.5. Nominal EVSI is 1.55 utility units; conservative EVSI is 0; expected scenario EVSI is 0.775 under equal scenario weights.

These values are mathematical golden expectations, not planner-run receipts or model results. The utility units are arbitrary synthetic units and do not transfer to real remediation decisions.

## Next implementation sequence

1. Meta-validate the schema and the example.
2. Add semantic contract validation and mutation tests for all cross-field invariants.
3. Test pure mathematical golden values independently.
4. Build the observed-Evidence adapter only after acquiring and verifying genuine REA Evidence roots.
5. Invoke existing point, robust and decision-aware planner APIs on the observed mode.
6. Compare all policy arms with the same frozen digests and a separate hidden DGP split.
