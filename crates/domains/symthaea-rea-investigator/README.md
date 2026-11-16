# symthaea-rea-investigator

Deterministic, dependency-free Rust planner for selecting the next authorized experiment from a caller-supplied hypothesis distribution and REA Evidence provenance catalog.

## Paradigm

Move from "rank the most likely cause" to "identify which permitted observation most reduces uncertainty between competing explanations."

The planner computes expected information gain in bits:

```
EIG(E) = H(Hypotheses) - Σ_outcomes P(outcome) × H(Hypotheses | outcome)
```

It ranks eligible proposals by EIG divided by normalized estimated time, energy and risk cost. Ties are deterministic: utility descending, EIG descending, experiment ID ascending.

## Safety and claim boundaries

- This crate ranks proposals only. It does not execute an experiment, access a device, launch a process or mutate a target.
- Only explicitly authorized, reversible, non-destructive proposals within the configured target-impact and resource budgets are eligible.
- Unknown authorization is denied.
- Evidence origin classifications are caller-supplied and are not cryptographically authenticated.
- Independent derivations qualify only when their parent graph resolves to an independent observation; dangling references and cycles are rejected.
- Model-generated, synthetic and external-reference records do not independently ground hypotheses.
- If any hypothesis is ungrounded, the planner abstains instead of silently dropping that alternative.
- The expected information gain is conditional on caller-supplied priors and likelihoods. It does not prove these models are calibrated.
- Observed entropy reduction is a separate function that requires independently rooted observation Evidence, and still does not authenticate that Evidence.

The output claim ceiling explicitly leaves producer authentication, causal truth, experiment execution, target mutation and likelihood calibration unverified.

## API

Enable the root Symthaea feature `rea-investigator` and call `symthaea::rea_investigator::plan_next_experiment` in-process. MCP is not required.

The public API defines:
- `EvidenceProvenance` and `EvidenceOrigin`;
- `Hypothesis` with normalized prior probability and support/contradiction Evidence IDs;
- `ExperimentProposal` with outcome likelihoods, authorization, reversibility, target impact and time/energy/risk estimates;
- `PlannerConfig` with explicit budgets, weights and minimum EIG;
- `InvestigationPlan` with selected/ranked proposals, rejection reasons and abstention state;
- `measure_observed_information_gain` to separately record posterior entropy reduction after independent observations.

## Validation

The crate is tested through a standalone, locked harness so validation does not rely on the repository's currently unhealthy root workspace glob:

```bash
cargo fmt --manifest-path tools/rea-investigator-harness/Cargo.toml -- --check
cargo test --manifest-path tools/rea-investigator-harness/Cargo.toml --locked
```

The exact upstream-generated Evidence provenance test suite remains separate from this planner. The planner accepts Evidence IDs and provenance classifications; it does not itself verify REA Evidence IDs or producer authenticity.

## Claim ceiling

This is a deterministic decision-support planner over supplied probability models. It is not a causal inference engine, a security decision authority, an autonomous experimentation agent or an evidence authenticator.


## Robust likelihood-model uncertainty (additive API)

The `robust` module is an additive extension; it does not change the point-estimate `plan_next_experiment` contract.

Call `plan_next_experiment_robust` with:
- the same independently rooted REA Evidence provenance catalog and competing hypotheses;
- experiment metadata and safety/resource constraints;
- a shared set of `LikelihoodModelScenario` values. Every scenario must cover every candidate experiment, and each experiment matrix must cover every hypothesis with a normalized outcome distribution;
- `RobustPlannerConfig`, selecting expected EIG, worst-case EIG, or lower-tail CVaR-EIG plus a minimum top-1 rank-stability threshold.

The result reports, per candidate:
- expected EIG across scenarios;
- worst-case EIG;
- lower-tail CVaR-EIG;
- between-scenario EIG variance;
- top-1 and top-3 rank stability;
- each scenario's outcome probabilities and posterior hypothesis distributions.

The default policy ranks by lower-tail CVaR at alpha 0.25 and requires top-1 stability of at least 0.5. If no candidate meets that stability threshold, the planner abstains. Scenario/proposal order does not alter semantic ranking or output order.

### Important uncertainty boundary

Scenario weights and likelihood matrices are still caller-supplied. They are not automatically calibrated, authenticated or guaranteed to contain the true data-generating process. Worst-case or CVaR over an incomplete scenario set is not a formal global worst-case guarantee. The API reports the scenario set used and keeps the claim ceiling false for calibration, scenario-weight authentication, causal truth and experiment execution.

Do not pass ad hoc cell-wise lower/upper probability bounds and renormalize them as though they were a valid joint uncertainty set. If adding interval bounds later, use a mathematically valid simplex-constrained uncertainty set or a reviewed robust-optimization formulation.

The robust tests are additive to the point-planner tests. The dedicated planner workflow runs both through the same isolated locked harness.


## Decision-aware value of information (additive API)

The `decision` module extends EIG and robust EIG without changing either existing API. It answers a different question:

> Which permitted observation is expected to improve the downstream engineering decision, and how fragile is that improvement across plausible likelihood models?

Use `plan_next_experiment_decision_aware` with:
- independently rooted hypothesis Evidence and normalized prior probabilities;
- experiment proposals and the same authorization, reversibility, destructive-operation, target-impact, time, energy and risk constraints used by the point planner;
- shared likelihood scenarios from the robust planner;
- a caller-owned `DecisionUtilityModel` naming its owner, scope, utility units and explicit exclusions;
- terminal decisions with exactly one finite utility for every hypothesis;
- `DecisionPlannerConfig`, selecting expected EVSI, worst-case shared-policy EVSI, lower-tail CVaR shared-policy EVSI, or minimax-regret policy search.

The planner computes:
- the prior Bayes decision and expected utility;
- scenario-specific posterior-optimal actions and scenario-optimal EVSI;
- one common outcome-to-decision policy for expected/worst/CVaR ranking;
- the utility improvement of that shared policy under each scenario;
- expected, worst-case and lower-tail decision value;
- expected and worst-case regret versus the scenario-specific posterior-optimal action;
- probability-weighted decision stability;
- the selected decision ID and impact metadata for each outcome.

### Why this is different from EIG

EIG measures expected hypothesis entropy reduction. EVSI measures expected improvement in the best decision under a declared utility model. A probe can strongly reduce entropy without changing the optimal remediation, so it may have positive EIG but zero EVSI. Conversely, a probe that distinguishes a rare, high-consequence fault can be more valuable than a probe that produces more entropy reduction about benign details.

The MinimaxRegret policy enumerates deterministic outcome-to-decision policies and minimizes worst-case expected regret across the supplied finite scenario set. It uses a caller-configured maximum policy-combination budget and rejects candidates whose exact search exceeds that bound. This is a finite-scenario result, not a guarantee against every possible unknown model.

### Claim and authority boundaries

- The utility model is caller-defined and caller-owned; it is not discovered, authenticated or morally complete merely because it is represented as a typed matrix.
- Utility exclusions are mandatory and travel with the result.
- Scenario probabilities and likelihoods are supplied inputs, not calibrated truth.
- Terminal decision impact is reported as metadata. The planner does not execute that decision.
- Experiments and actions are proposals only. The same authorization and resource gates apply to experiment proposals; terminal action execution requires a separate policy/approval boundary.
- Unobservable or ungrounded hypotheses cause explicit abstention rather than being dropped.
- The output claim ceiling keeps evidence authentication, utility-model review, calibration, causal truth and action execution false.

### Decision-aware tests

The decision module currently includes 10 unit tests with known-answer and adversarial cases for:
- a high-EIG experiment that leaves the optimal action unchanged and therefore has zero EVSI;
- a low-entropy rare-critical-fault probe that has higher decision value than a high-entropy benign-detail probe;
- a shared minimax-regret policy that rejects a scenario-fragile probe;
- explicit abstention when the evidence is only model-generated or synthetic;
- rejection of unknown authorization and invalid utility matrices;
- bounded minimax policy enumeration;
- invariance to scenario ordering;
- correct binary EVSI arithmetic and propagation of terminal decision impact;
- robust worst-case ranking that abstains when one plausible scenario has no decision value.

The dedicated `REA Investigator Planner` workflow validates the full crate and isolated locked harness. The decision-aware extension remains unqualified until its exact-head run passes.
