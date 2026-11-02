# Connectome experiment design: finite-hypothesis EIG

A standalone, dependency-free Rust 2024 reference crate for finite-hypothesis EIG, Gaussian Fisher D/A-optimal parameter identification and utility-aware EVSI experiment design.

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


## Parameter identification is not model discrimination

The crate now supports a third, independent experiment-design objective: local Gaussian Fisher-information design.

For a parameter prior with covariance `Sigma`, local observation sensitivity/Jacobian `J`, and measurement-noise covariance `R`:

```
F = J^T R^-1 J
Sigma_post = (Sigma^-1 + F)^-1
D_information_bits = 0.5 * log2(det(Sigma) / det(Sigma_post))
A_variance_reduction = trace(Sigma) - trace(Sigma_post)
```

The API is:

- `ParameterPrior { covariance }`: finite, symmetric, strictly positive-definite Gaussian prior covariance.
- `LinearGaussianDesign { id, cost, sensitivities, noise_covariance }`: a finite candidate measurement design. Rows of `sensitivities` are observations; columns are parameters. `noise_covariance` may include correlated measurement noise.
- `evaluate_parameter_design(...)`: returns posterior covariance trace, prior/posterior log determinants, D-optimal information in bits, A-optimal trace reduction and both cost-normalized scores.
- `rank_parameter_candidates(..., ParameterObjective::DOptimal)`: ranks by D-optimal information per cost.
- `rank_parameter_candidates(..., ParameterObjective::AOptimal)`: ranks by A-optimal trace reduction per cost.
- `parameter_selection_regret(...)`: computes opportunity regret for either objective against the same finite candidate pool.

The implementation uses Cholesky factorization and triangular solves; it does not explicitly form `R^-1`. Non-finite, ragged, asymmetric, singular or non-positive-definite prior/noise matrices fail closed. It rejects invalid costs and non-finite cost-normalized scores.

**The assumptions are deliberately narrow.** This is a local linearization with Gaussian parameter prior and Gaussian measurement noise. The caller supplies the sensitivity matrix and noise covariance; the crate does not infer them, estimate biological parameters, optimize continuous waveforms, or guarantee that a local approximation remains valid under large perturbations. The output is conditional on the chosen parameterization, prior covariance, observation model and noise covariance.

### D-optimal versus A-optimal is a real decision, not a naming difference

The frozen two-parameter fixture uses an identity prior covariance and unit observation noise:

- A concentrated observation with sensitivity `[10, 0]` yields **3.3291057414 D-information bits** but only **0.9900990099 trace reduction**.
- A balanced two-observation design with sensitivities `[2, 0]` and `[0, 2]` yields **2.3219280949 D-information bits** but **1.6 trace reduction**.

D-optimal selects the concentrated observation because it most reduces uncertainty volume. A-optimal selects the balanced design because it reduces the sum of marginal parameter variances more. Neither is universally best; the scientific estimand determines which parameter-uncertainty geometry matters.

## Benchmark against current research

The primary recent comparator is Ogino, Sekizawa, Kitazono and Oizumi, *Designing optimal perturbation inputs for system identification in neuroscience*, eLife 15:RP110030 (version of record 13 October 2026): https://doi.org/10.7554/eLife.110030.3

Their linear system-identification analysis relates asymptotic matrix-estimation error to the state covariance through:

```
E[||A_hat - A||_F^2] ~= (1/T) tr(Sigma_noise) tr(Sigma_state^-1)
```

The practical implication is to re-excite weakly observed dynamical modes and make state covariance eigenvalues larger and more uniform, not merely maximize total activity. The paper studies impulse, step, sinusoidal and composite-frequency perturbations, along with iterative input/target refinement. Its central linear formulation assumes a known input matrix B; the paper discusses extensions beyond that setting, and notes the gap between ideal control inputs and experimental feasibility.

**This crate does not claim to reproduce or outperform Ogino et al.** It provides generic local Gaussian parameter-information scores over caller-supplied Jacobian/noise matrices. A fair bake-off must implement the paper's state-covariance objective as a separate comparator, reproduce its published simulations/conditions, and include the D-optimal and A-optimal scores as additional, separately labelled baselines.

### Required experiment-design bake-off

Freeze the candidate experiment pool and compare, at equal measurement duration, input energy, number of stimulation channels, observation count and compute budget:

1. passive/no perturbation;
2. impulse, step and sinusoidal perturbations;
3. uniform-frequency and composite-frequency inputs;
4. the published Ogino-style objective/iterative refinement;
5. random feasible designs;
6. Fisher D-optimal and A-optimal designs;
7. finite-hypothesis EIG designs;
8. utility-aware EVSI designs;
9. degree/centrality/expert-prior and largest-predicted-response heuristics.

Report more than the objective each selector optimized: held-out parameter error, system-matrix Frobenius error, posterior coverage and sharpness, mode/eigenvalue recovery, model-discrimination accuracy, decision regret, robustness under prior/noise/model mismatch, measurement energy, wall time and failure rate. This avoids awarding a method a win for improving its own internal score while the scientific estimate gets worse.

### External connectome challenge sets

**Drosophila walking circuit.** Pugliese et al., *Connectome simulations identify a central pattern generator for fly walking*, Nature (version of record 28 October 2026): https://doi.org/10.1038/s41586-026-11195-y. The study reports a minimal one-inhibitory/two-excitatory interneuron rhythm-generating circuit across four connectome datasets and all six legs, and experimentally tests a predicted DNb08 descending pathway using optogenetics in behaving flies. Its code and figure reproductions are published at https://github.com/smpuglie/Pugliese_2026; simulation output and Anipose data are deposited at https://doi.org/10.5281/zenodo.22260923.

A reproducible Symthaea challenge should pin the exact source/data versions and separately score: DNg100/DNb08 activation responses, rhythm onset/frequency, left-right and inter-leg coordination, three-cell-core necessity/sufficiency, and held-out connectome reconstruction transfer. Do not merge simulated sufficiency, observed behaviour and experimentally verified pathway outcomes into one label.

**C. elegans signal propagation.** Randi et al., *Neural signal propagation atlas of Caenorhabditis elegans*, Nature 623, 406–414 (2023): https://www.nature.com/articles/s41586-023-06683-4. The atlas measures optogenetically evoked propagation over 23,433 neuron pairs with whole-brain calcium imaging. Use it as a perturbation-response/effective-propagation benchmark, not as a direct anatomical synapse catalogue. Separate measured propagation, connectome-predicted routes and inferred latent/shared-input pathways.

### Promotion rules and aspirational targets

These are proposed research gates, not results or universal field standards:

- **Analytic correctness:** all Gaussian fixtures within 1e-10 relative/absolute tolerance; reject singular, asymmetric, malformed and non-finite inputs.
- **Parameter recovery:** beat random, passive, impulse/step/sinusoid and the reproduced published control-theoretic selector on held-out parameter error at matched energy/time.
- **Calibration:** nominal 90%/95% parameter intervals should achieve coverage within preregistered uncertainty bounds, with sharpness reported alongside coverage.
- **Robustness:** publish score degradation under correlated/heteroscedastic noise, latent inputs, observation-kernel mismatch, prior misspecification and nonlinear dynamics.
- **Generalization:** repeat across held-out targets, sessions, animals and connectome reconstructions; never random-split correlated timepoints and call it cross-animal evidence.
- **Efficiency:** publish accuracy/calibration/energy/latency/memory Pareto curves, not a single speed claim.
- **Decision quality:** report realized EIG, posterior parameter error and EVSI regret separately. An information-rich experiment is not necessarily the experiment with the greatest decision value.

The promotion criterion for "better than the current comparator" is a preregistered improvement in the target scientific estimand with a confidence interval clearing a meaningful margin, under matched feasibility and resource budgets. No SOTA claim follows from the synthetic fixture suite alone.

### The three objectives

| Objective | Question | Output unit | Main failure mode |
|---|---|---|---|
| EIG | Which experiment separates the declared hypotheses? | bits | Hypothesis/likelihood misspecification |
| Fisher D/A | Which experiment identifies the declared parameters? | bits / parameter-units squared | Local-linearization or parameterization dependence |
| EVSI | Which experiment improves the declared action? | caller utility | Utility misspecification or actions omit real alternatives |

Do not add these raw scores into one scalar: they have different meanings and units. Compare their Pareto front or choose an objective from a preregistered scientific estimand. A later constrained selector can combine them only after explicit utility normalization and safety/feasibility constraints are declared.

## Decision value is not information gain

The crate now also exposes a separate, caller-utility-aware objective:

- `DecisionProblem` declares available actions and a utility matrix `U[hypothesis][action]`.
- `evaluate_decision_value(...)` computes expected value of sample information (EVSI): the improvement in the best expected action utility after observing the experiment outcome, versus choosing before the experiment.
- `rank_candidates_by_decision_value(...)` ranks by EVSI/cost, then EVSI, then stable ID.
- `decision_selection_regret(...)` reports the opportunity gap relative to the best candidate in the declared pool under EVSI/cost.

EVSI is reported in caller-defined utility units, not bits. It is not the same as EIG, EIG/cost, a parameter-identification objective, or realized biological utility.

A frozen four-hypothesis control demonstrates the distinction:
- Hypotheses H0/H1 support action A; H2/H3 support action B.
- An experiment that perfectly distinguishes H0 from H1 and H2 from H3 has 1.0 bit EIG but 0 EVSI because it cannot change the best action.
- A noisy experiment that identifies the action-relevant group with 80% accuracy has only about 0.2780719 bits EIG but 0.3 utility units EVSI for the declared 0/1 utility matrix.
- EIG therefore ranks the within-group experiment first, while EVSI ranks the decision-relevant experiment first.

This is a synthetic known-answer demonstration. Utility functions, hypotheses, priors, likelihoods, costs and action feasibility remain caller-supplied and must be calibrated/preregistered for biological use. Fisher-information D-/A-optimal parameter identification remains a distinct comparator, not an implementation hidden inside EIG or EVSI.
