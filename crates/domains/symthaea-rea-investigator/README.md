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


## Held-out likelihood calibration (additive API)

The `calibration` module evaluates likelihood predictions against a frozen, labeled calibration cohort. It is separate from both point-EIG and scenario robust-EIG: caller-supplied scenarios are not treated as calibrated merely because they normalize to probability distributions.

Call `evaluate_likelihood_calibration` through `symthaea::rea_investigator` with:
- an immutable model/provider identity and SHA-256;
- named platform, target build, fault family, environment and observation-regime scope;
- training source-group digests;
- held-out observations with unique sample IDs, source-group digest, a full class probability vector and observed class;
- explicit binning/bootstrap configuration and thresholds.

The evaluator:
- rejects training/calibration source-group overlap and duplicate sample IDs;
- requires identical class sets and finite normalized probability vectors;
- computes multiclass Brier score, mean log loss, zero-probability count and top-label ECE;
- emits reliability bins with Wilson 95% accuracy intervals;
- reports deterministic 95% bootstrap intervals for Brier/ECE/log loss;
- uses source-group cluster bootstrap, not row bootstrap, so multiple samples from one source program/binary family are not assumed independent;
- binds the model, target scope, split groups, observation data digest, report digest and receipt digest.

A zero probability assigned to the observed class makes mean log loss explicitly unavailable (None) and increments the zero-probability count; the implementation does not clip the value and silently conceal an infinite log-loss penalty.

The receipt status `EligibleForIndependentReview` means only that the configured sample/group minimums and score thresholds passed. It is **not** a calibration certificate. The receipt is unsigned, and the claim ceiling leaves model/producer authentication, label validity, split audit, scope completeness, stationarity and drift assessment false. Use `verify_calibration_receipt_against_observations` to recompute the digest-bound report against the retained exact observation set.

### Calibration limitations and next research

- Source-group digests are caller supplied; they are not independently proven by this evaluator.
- Group bootstrap reduces obvious within-project pseudoreplication but does not prove source groups are independent.
- Top-label ECE depends on the configured bin count; always report the bin table and proper scoring rules too.
- A calibration result is scoped to the named build, environment, fault family and observation regime; it does not transfer to new phone builds or distribution shifts automatically.
- Drift detection and expiry after model/OS/firmware updates are separate work tracked by [#10207](https://github.com/Luminous-Dynamics/symthaea/issues/10207).

The dedicated `REA Investigator Planner` workflow includes the calibration tests through the same isolated locked Cargo harness.
