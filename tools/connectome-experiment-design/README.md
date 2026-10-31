# Connectome experiment design: finite-hypothesis EIG

A standalone, dependency-free Rust 2024 reference crate for ranking candidate experiments by expected information gain (EIG) over a finite set of competing hypotheses.

This crate is a small mathematical utility, **not** a neural simulator, causal estimator, continuous control-input optimizer, or biological model. The caller must provide the hypotheses, prior probabilities, conditional outcome likelihoods, and experiment costs. It does not infer or calibrate any of them. In particular, it complements rather than reproduces recent control-theoretic work on designing continuous perturbation waveforms for system identification: this implementation ranks a finite list of discrete candidate experiments under caller-supplied hypothesis likelihoods.

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
- invalid priors, invalid likelihood mass, NaN, zero cost, duplicate outcomes/IDs, impossible outcomes, and missing selections.

The analytical values are independently recorded in `fixtures/known-answers.json`.

## Scientific limitations and safe use

1. EIG is conditional on the candidate hypothesis set, prior, likelihood model, outcome discretization, and cost definition. Misspecified likelihoods can confidently select the wrong experiment.
2. EIG per cost is an information-efficiency objective, not expected behavioural benefit or downstream decision utility. Track decision regret separately.
3. A high EIG score does not establish that a hypothesis is biologically true.
4. Likelihoods should be calibrated on training/validation data and frozen before held-out outcomes are revealed.
5. Compare the policy against random, degree, centrality, largest-predicted-response, and expert-prior baselines under equal experiment budgets.
6. Prospective utility requires running the selected experiment and comparing the realized outcome to the preregistered predictions.

The fixture evidence class is `synthetic_known_answer` and claim ceiling is `SYNTHETIC_ONLY`. No empirical experiment ranking or biological result is claimed.
