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


## Decision-aware extension

This crate now supports a separate decision-aware planner in addition to point-EIG and scenario-robust EIG:

- Point-EIG measures expected entropy reduction.
- Robust EIG exposes expected, worst-case and lower-tail information gain across caller-supplied scenarios.
- EVSI evaluates whether an observation is expected to change or improve a declared terminal decision.
- **Minimax regret** performs an exact bounded search over deterministic outcome-to-decision policies and minimizes the maximum expected regret across the supplied finite scenarios.

The minimax policy search is capped by `DecisionAwareConfig.max_policy_evaluations` (default 65,536). If the finite policy space exceeds that cap, the experiment is rejected with `PolicySearchBudgetExceeded`; there is no silent fallback to the scenario-mixture Bayes policy. Policy ties are broken lexicographically by decision IDs in declared outcome order.

The planner still only scores proposed investigations. It does not execute probes or terminal decisions, and its caller-supplied likelihoods, scenario weights and utility matrices are neither authenticated nor calibrated by this crate.

Run the focused harness:

```bash
cargo test --manifest-path tools/rea-investigator-harness/Cargo.toml --locked
```
