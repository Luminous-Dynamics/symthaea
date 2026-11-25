# Kinetic State Discrimination Design — complex-response EIG v0.1

This fixture specifies a finite categorical hypothesis set and one candidate intervention. Each hypothesis predicts one registered complex response coefficient. Observation noise is independent, equal-variance Gaussian noise on the real and imaginary quadratures.

The implementation computes:

`EIG(u) = H(H) - E_y[H(H | y,u)] = I(H;Y | u)`

by deterministic 9×9 Gauss–Hermite quadrature. It reports:
- prior entropy in bits;
- expected posterior entropy in bits;
- expected information gain in bits;
- information gain divided by declared evaluation-cost seconds.

The four known-answer cases cover:
1. identical complex predictions (zero information);
2. phase-only separation (same response magnitude, distinct quadrature);
3. high noise suppressing useful information;
4. imbalanced prior probabilities with strong response separation.

The prior is normalized by max scaling to avoid overflow. Hypothesis IDs, priors, complex responses, noise variance and positive evaluation cost are validated. The implementation is bounded to 2–16 hypotheses and a fixed quadrature work budget.

## Scientific limits

This is a finite-hypothesis, single-complex-output, homoscedastic Gaussian observation model. It is not a calibrated likelihood for full kinetic trajectories, correlated QoIs, branch switching or heteroscedastic reference error. A design's EIG is conditional on the supplied hypotheses, prior, noise model and output channel. It is not proof that those hypotheses exhaust the true state space.

The synthetic fixture's noise is not calibrated from plasma data, no Vlasov forward model is run, and no global state equivalence is claimed. Physical KSDD must estimate / bound noise from independent repeats and solver refinement, retain branch labels, and confirm selected probes on independent high-fidelity trajectories.

See #13598 (KSDD), #13503 (state quotients), #13404 (adversarial probe search), #13415 (response spectroscopy) and #12944 (simulation acquisition).

## Matched intervention portfolio

`evaluate_kinetic_discrimination_portfolio` compares a finite set of candidate interventions under a **shared ordered hypothesis set, normalized prior distribution and observation-noise variance**. A portfolio that changes these between candidates is rejected, because its EIG values would not answer the same discrimination question.

The report preserves each candidate's EIG and declared evaluation cost, returns the nondominated Pareto-front indices (maximize information, minimize cost), and separately returns the best EIG-per-cost index. The ratio is a selectable policy, not a replacement for the Pareto frontier.

A synthetic Rust regression fixture includes:
- a low-cost weakly discriminating pulse;
- a high-cost strongly discriminating pulse;
- a lower-information, higher-cost dominated pulse;
- a strong low-cost pulse that dominates the high-cost version.

This selector does not execute a Vlasov solver or choose from physical pulse families. Real acquisition still requires calibrated noise/reference uncertainty, boundary and amplitude admissibility, branch controls, independent replay and confirmation on whole held-out trajectories.
