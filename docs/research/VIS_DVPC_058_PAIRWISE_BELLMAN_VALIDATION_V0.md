# VIS-DVPC-058 — Pairwise Bellman Validation for OPE model selection v0

**Status:** standalone synthetic finite-MDP selector benchmark. Not a reproduction of the paper's Hopper/Deep OPE experiments, not a finite-sample coverage certificate, and not an SOTA claim.

## Motivation and comparator

van der Laan, Syrgkanis and Kallus, *Model Selection for Off-Policy Evaluation via Pairwise Bellman Validation*, arXiv:2610.35748 (submitted 2026-10-26): https://arxiv.org/abs/2610.35748

The paper identifies a central problem: candidate Q-functions and occupancy-ratio estimates cannot be directly supervised against their unknown target functions using held-out transitions. Squared one-step TD error is observable, but it equals squared Bellman-residual error plus a candidate-dependent conditional-variance term, so minimizing it may select the wrong Q candidate under stochastic rewards/transitions.

The PBV paper proposes unweighted pairwise Bellman validation (PBV), target-aligned PBV (TA-PBV), and adjoint PBV (A-PBV). For Q selection, PBV compares each candidate against all pairwise candidate differences as critics, takes a worst-case score, and subtracts a quadratic penalty for critic approximation under misspecification. It requires no auxiliary critic fitting. Unweighted PBV has guarantees under pairwise coercivity; TA-PBV uses the target discounted occupancy ratio to avoid that coercivity requirement, at the cost of estimating the ratio. The paper reports experiments on Hopper and Deep OPE/D4RL and compares against TD error and LSTD-Tournament.

## Frozen finite MDP

The standalone implementation is `tools/vision-dvpc-058-pairwise-bellman-validation.rs`.

The initial worlds have three states, two actions, a known stochastic transition tensor, Bernoulli rewards, a fixed target policy, a strictly positive offline state-action distribution, an initial-state distribution, and discount factor gamma. The validation design draws independent one-step transition tuples from the registered offline distribution. This is the independent-transition special case; it does not claim to cover arbitrary Markov-chain dependence.

Candidate Q-functions are generated only from a separate training sample: fit an empirical tabular transition/reward model with fixed smoothing, solve its Bellman fixed point, then generate regularization/shrinkage and structured-perturbation candidates. The exact DGP, exact Q function, exact Bellman residual and exact target occupancy are evaluator-only. The selector receives the candidates and held-out validation transitions.

Two scenarios are registered: a reasonably coercive stochastic world and a shifted-overlap world. The benchmark reports the measured finite-candidate pairwise coercivity ratio rather than assuming the condition holds.

## Pairwise Bellman validation

For candidate Q_i and comparator Q_j, define h_ij = Q_i - Q_j and the held-out one-step Bellman residual:

\[
\delta_i(X,R,S')=Q_i(X)-R-\gamma(\pi Q_i)(S').
\]

The unweighted empirical pairwise moment and penalized score follow the paper's Equation (5):

\[
\widehat M_{ij}^{unw}
=\frac1n\sum_{t=1}^n h_{ij}(X_t)\delta_i(X_t,R_t,S'_t)
-\frac14\max\left\{0,
\frac1n\sum_{t=1}^n h_{ij}(X_t)
\left[h_{ij}(X_t)-\gamma(\pi h_{ij})(S'_t)\right]
\right\}.
\]

Select the smallest index minimizing `max_j M_ij`. Deterministic smallest-index tie-breaking is part of the protocol. The unpenalized pairwise selector is retained as a misspecification negative comparator.

TA-PBV multiplies both moment and quadratic-penalty terms by the exact target discounted occupancy ratio `w^pi = d_pi,gamma / nu`. In this benchmark that ratio is computed from the known MDP and labelled **oracle-only**. It isolates the target-aligned score's behavior but does not measure the cost or error of learning `w^pi`. An A-PBV ratio-candidate selector is explicitly deferred.

## Baselines and metrics

Selectors:
- exact target-Q-error oracle (evaluator-only lower bound);
- exact squared Bellman-residual oracle (evaluator-only comparator);
- held-out squared TD error;
- unweighted penalized PBV;
- unweighted unpenalized PBV;
- target-aligned PBV using the exact occupancy ratio (oracle-weight comparator).

Each result reports selection accuracy relative to the best candidate under exact target-occupancy Q error, selected-candidate offline and target Q error, initial-state plug-in value absolute error, exact squared Bellman residual, measured pairwise coercivity, and split-seed roots.

The initial value metric is the plug-in `E_{d0}[pi Q]` diagnostic. It is not a full trajectory DR estimate. No interval coverage or policy promotion claim is made.

## Falsification protocol

- The TD-reversal fixture sets stochastic next-state variation so one Q candidate has lower exact Bellman residual / offline error but substantially higher one-step TD conditional variance. The test requires squared TD selection to prefer the other candidate.
- The exact-realizable population fixture includes Q^pi and verifies that population penalized PBV gives it a zero worst-case score and a positive score to a nonzero candidate under measured coercivity.
- The target-aligned fixture checks that the exact discounted occupancy ratio integrates to one under the offline measure and that TA-PBV's population score selects the exact candidate.
- The shifted-overlap fixture measures candidate-direction coercivity and reports when the unweighted theorem condition is weak or fails.
- The training/validation split has independent seed roots. Candidate fitting accepts only training rows; selectors accept only validation rows. Equal seed roots fail closed.
- Candidate count is fixed at six in v0. K-growth, duplicates, and large outlier tests are follow-on work.

## Acceptance gates

1. Exact Q solves the finite MDP Bellman fixed point to numerical tolerance.
2. Independent population PBV moments match exact finite-MDP formulas.
3. In a coercive realizable fixture, exact Q has zero population PBV worst-case score and nonzero candidates have a positive score.
4. Squared TD demonstrably ranks candidates differently from exact Bellman residual under a registered stochastic-transition fixture.
5. Penalty and unpenalized selector outputs are both checked under candidate misspecification.
6. Target-aligned PBV uses only the declared oracle occupancy ratio and is never described as an estimated-ratio implementation.
7. Training and validation streams remain disjoint, deterministic, and provenance-labelled.
8. Exact-head standalone Rust formatting, tests, deterministic smoke/full TSV schemas, and SHA-256 receipts pass.

## External parity and research limits

The paper's Hopper and Deep OPE/D4RL results are the external benchmark target. v0 is a finite tabular mechanism test and must not be compared numerically to those results. A faithful follow-on should reproduce the paper's candidate-generation protocol, sample splits, normalization, critic penalties, temporal dependence assumptions, and baseline implementations.

Next: A-PBV occupancy-ratio candidate selection; estimated-ratio TA-PBV; Markov validation trajectories with spectral-gap sweeps; LSTD-Tournament; candidate count K in {2,4,8,16,32}; candidate duplicate/scale stress; policy portfolio PBV; and joint model-selection uncertainty for multi-policy OPE.