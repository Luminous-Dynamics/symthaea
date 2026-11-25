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