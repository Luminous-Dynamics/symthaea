# Connectome experiment design: EIG, EVSI and Fisher information

A standalone, dependency-free Rust 2024 reference crate for three non-interchangeable experiment-design objectives: expected information gain (EIG) over competing hypotheses, expected value of sample information (EVSI) for a declared decision, and local Fisher-information gains for parameter identification.

This crate is a mathematical utility, **not** a neural simulator, causal estimator, continuous control-input optimizer, or biological model. EIG/EVSI callers supply hypotheses, priors, outcome likelihoods, utilities and costs. Fisher-design callers supply a prior precision matrix, local output-sensitivity Jacobian rows, observation variances and costs. Nothing is inferred or calibrated from biological data. The Fisher API is a local linearized parameter-identification comparator; it does not reproduce the continuous waveform optimizer in Ogino et al. (eLife, 2026).

## Run

From the repository root:

```sh
cargo test --locked --manifest-path tools/connectome-experiment-design/Cargo.toml
```

The crate is an isolated Cargo workspace with no third-party dependencies and must not be added to the production workspace dependency graph accidentally.

## Mathematical contract

For a candidate experiment e, hypothesis H, and possible outcome Y:

```
EIG(e) = H(H) - E_Y[H(H | Y, e)] = I(H; Y | e)
```

All entropies and information gains are measured in bits. The crate returns both raw EIG and EIG divided by caller-defined positive cost.

The likelihood matrix is indexed as:

- outer row: hypothesis h;
- inner column: outcome y;
- cell: P(Y = y | H = h, e).

The prior vector must sum to one, and every hypothesis likelihood row must independently sum to one. Invalid probability mass is rejected; the crate never silently renormalizes malformed inputs.

## API

### Hypothesis discrimination (EIG)

- `evaluate_candidate(priors, candidate)`: calculates prior entropy, expected posterior entropy, EIG, and EIG/cost.
- `outcome_probability(priors, candidate, outcome_index)`: calculates the marginal probability of an outcome.
- `posterior_for_observed_outcome(...)`: applies Bayes' rule after an outcome is observed.
- `rank_candidates(...)`: deterministic ranking by EIG/cost, then raw EIG, then ascending experiment ID.
- `selection_regret(...)`: measures opportunity regret against the best candidate in the supplied pool under the EIG/cost objective.

## Synthetic known-answer cases

The unit tests cover:

- uninformative experiments (0 bits);
- perfect discrimination (1 bit for a uniform binary prior);
- a noisy 90%/10% binary discriminator (0.5310044064107188 bits);
- perfect discrimination with an asymmetric prior (0.7219280948873623 bits);
- Bayes posterior updates;
- a cost-aware ranking that selects a lower-raw-EIG but more efficient experiment;
- deterministic tie-breaking and selection regret;
- invalid priors, invalid likelihood mass, NaN, zero cost, duplicate outcomes/IDs, impossible outcomes, and missing selections;
- pairwise-matched hypotheses that differ only under a three-target interaction, plus a state-gated triplet control;
- exact Fisher-information D-/A-optimal known answers, an objective-ranking reversal, variance weighting, non-SPD/asymmetric priors, malformed sensitivities and costs.

The analytical values are independently recorded in `fixtures/known-answers.json`.

## Scientific limitations and safe use

1. EIG is conditional on the candidate hypothesis set, prior, likelihood model, outcome discretization, and cost definition. Misspecified likelihoods can confidently select the wrong experiment.
2. EIG per cost is an information-efficiency objective, not expected behavioural benefit or downstream decision utility. Track decision regret separately.
3. A high EIG score does not establish that a hypothesis is biologically true.
4. Likelihoods should be calibrated on training/validation data and frozen before held-out outcomes are revealed.
5. Compare the policy against random, degree, centrality, largest-predicted-response, and expert-prior baselines under equal experiment budgets.
6. Prospective utility requires running the selected experiment and comparing the realized outcome to the preregistered predictions.

The fixture evidence class is `synthetic_known_answer` and claim ceiling is `SYNTHETIC_ONLY`. No empirical experiment ranking or biological result is claimed.


## Decision value is not information gain

The crate now also exposes a separate, caller-utility-aware objective:

- `DecisionProblem` declares available actions and a utility matrix `U[hypothesis][action]`.
- `evaluate_decision_value(...)` computes expected value of sample information (EVSI): the improvement in the best expected action utility after observing the experiment outcome, versus choosing before the experiment.
- `rank_candidates_by_decision_value(...)` ranks by EVSI/cost, then EVSI, then stable ID.
- `decision_selection_regret(...)` reports the opportunity gap relative to the best candidate in the declared pool under EVSI/cost.

EVSI is reported in caller-defined utility units, not bits. It is not the same as EIG, EIG/cost, a parameter-identification objective, or realized biological utility.

### Parameter identification (Fisher information)

- `FisherInformationExperiment` declares local per-observation parameter sensitivities, independent observation variances and experiment cost.
- `evaluate_fisher_information_candidate(prior_precision, candidate)` constructs `F = Jᵀ R⁻¹ J`, adds it to the prior precision, and reports:
  - D-optimal gain: `0.5 * ln(det(P_post) / det(P_prior))` in nats, corresponding to Gaussian parameter-volume reduction.
  - A-optimal gain: `tr(P_prior⁻¹) - tr(P_post⁻¹)`, the reduction in total marginal parameter variance.
- `rank_fisher_candidates(..., FisherObjective::DOptimal)` and `rank_fisher_candidates(..., FisherObjective::AOptimal)` rank by the selected gain per cost, then raw gain, then stable ID.

The Fisher matrix assumes locally linearized output sensitivities and independent Gaussian observation noise. Correlated observations must be whitened by the caller. The prior precision must be symmetric positive definite. These are design-point approximations; the crate does not propagate nonlinear parameter uncertainty or synthesize continuous stimulation waveforms.

**EIG, EVSI, D-optimal and A-optimal design are four different questions.** EIG asks which experiment best discriminates the declared hypotheses; EVSI asks which experiment most improves a declared decision; D-optimal design shrinks the joint parameter uncertainty volume; A-optimal design reduces the sum of marginal parameter variances. Their rankings can legitimately disagree.

A frozen four-hypothesis control demonstrates the distinction:
- Hypotheses H0/H1 support action A; H2/H3 support action B.
- An experiment that perfectly distinguishes H0 from H1 and H2 from H3 has 1.0 bit EIG but 0 EVSI because it cannot change the best action.
- A noisy experiment that identifies the action-relevant group with 80% accuracy has only about 0.2780719 bits EIG but 0.3 utility units EVSI for the declared 0/1 utility matrix.
- EIG therefore ranks the within-group experiment first, while EVSI ranks the decision-relevant experiment first.

This is a synthetic known-answer demonstration. Utility functions, hypotheses, priors, likelihoods, costs and action feasibility remain caller-supplied and must be calibrated/preregistered for biological use. Fisher information is now implemented as an explicit third objective family; it remains mathematically separate from EIG and EVSI.
