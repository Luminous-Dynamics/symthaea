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


The fixture also contains a matched four-candidate portfolio. All candidates share identical state hypotheses, prior probabilities and noise variance. The expected Pareto frontier contains the weak/low-cost and strong/low-cost interventions; the strong/high-cost design is dominated by the equal-information strong/low-cost design, and the weak/expensive design is dominated by the weak/low-cost design. The best EIG-per-cost selector is the strong/low-cost option.

The selector rejects portfolios that change hypothesis ordering, normalized priors or the observation-noise model between candidate interventions. It returns the Pareto indices and scalar EIG/cost choice separately, so the scalar decision does not erase the frontier.


## Noise-envelope sensitivity

`evaluate_kinetic_discrimination_noise_envelope` evaluates the same candidate design at the minimum and maximum of a caller-declared positive observation-noise variance interval. Under the stated additive independent Gaussian model, the higher-variance response is a degraded channel (it can be obtained by adding independent Gaussian noise to the lower-variance observation). The data-processing inequality therefore implies non-increasing EIG as variance increases.

The helper checks the endpoint ordering numerically and reports both endpoint EIG and cost-normalized EIG. This is a model-conditional sensitivity check; it does not certify that real instrument noise, numerical reference error or correlated QoI uncertainty lies inside the interval.


## Branch-association gate

Every finite hypothesis carries an explicit non-empty response-branch identity and `branch_association_stable=true`. The schema and Rust evaluator both reject unresolved branch association. The branch IDs are hypothesis metadata; they are not silently inferred from nearest complex-amplitude peaks. If a branch is below floor, created, cancelled or misassociated, the observation must be treated as UNKNOWN / unscorable until a branch-aware likelihood is implemented.

This first KSDD scorer still assumes a stable scalar complex response per hypothesis. It does not model stochastic branch transitions or a mixture of branch-conditioned response distributions.
