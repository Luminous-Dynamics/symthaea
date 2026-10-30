# VIS-DVPC-053 — Estimator-aware logging and cross-fitted DR OPE v0

**Status:** deterministic synthetic benchmark and falsification harness. This is not a real-world OPE result, a calibrated deployment certificate, or a SOTA claim.

## Why this is the next frontier

The preceding VIS-DVPC-049–052 work isolates policy selection, adaptive logging, logger design and finite-budget IPW. The May 2026 paper *Logging Policy Design for Off-Policy Evaluation* explicitly uses IPW to keep logger design separate from reward-model estimation, and says extensions to Direct Method / Doubly Robust estimators are future work because the outcome model used to design the logger may be correlated with the model used in evaluation (Douglas, Persson & Provost, [arXiv:2605.15108, §8](https://arxiv.org/html/2605.15108#S8)).

The same paper derives posterior shrinkage toward a per-context, across-action mean and warns that the covariance used to choose the shrinkage weight must be estimated on data held out from reward-model fitting. Using fitted observations to calibrate the weight inflates covariance and biases shrinkage downward ([§5.3](https://arxiv.org/html/2605.15108#S5.SS3)).

VIS-DVPC-053 turns those caveats into a single end-to-end experiment. It does not assume that shrinkage or cross-fitting always helps.

## Protocol

The standalone Rust source is tools/vision-dvpc-053-crossfit-dr-ope.rs.

### Data roles

For each scenario and replicate, the harness generates three disjoint streams:

1. **Pilot fit:** uniform logging in the source reward world; fit a Beta(1,1)-smoothed two-context/two-action reward table.
2. **Shrinkage calibration:** independently sampled source-world observations. The held-out empirical weight is clamp(1 - Cov(q_hat,R)/Var(q_hat), 0, 1). A deliberately leaky negative control computes the same statistic on the pilot observations used to fit q-hat.
3. **Evaluation:** new observations sampled under the chosen logger, with exact action propensities retained on every row.

The oracle reference weight minimizes source-domain squared prediction error after shrinking each context's two action means toward their common mean. It uses the known synthetic source reward table and is diagnostic only.

### Registered scenarios

- stationary: stationary source/target rewards and balanced contexts.
- noisy_pilot: only 24 pilot observations, making the initial reward model substantially noisier.
- predictor_bias: a registered action-specific perturbation is applied to the fitted pilot predictor.
- reward_drift: two action/context reward means shift between pilot and evaluation.
- context_shift: evaluation context frequencies change from 50/50 to 80/20.

The action floor is 0.02 for all model-derived logger policies. Uniform logging is the reference. Candidate target policies include opposing, balanced and one-sided policies.

### Logger and estimator arms

Logger designs: uniform, raw pilot model, held-out shrinkage, leaky shrinkage, and known-source MSE-oracle shrinkage.

Estimators: IPS with exact propensities; DM with the pilot model; DR with the same pilot/shrunk model used to design the logger; DM with an in-sample evaluation nuisance fit; DR with an in-sample evaluation nuisance fit; and two-fold cross-fitted DR. In cross-fitting, each evaluation fold is scored using a reward model fitted only on the opposite fold.

For a fixed reward model q, the DR score is

\[
\hat V_{DR} = \frac{1}{n}\sum_i\left[
\sum_a\pi(a|X_i)q(X_i,a)
+\frac{\pi(A_i|X_i)}{\mu(A_i|X_i)}
(R_i-q(X_i,A_i))
\right].
\]

With known logging propensities and full target support, its conditional expectation equals the target-policy value for any fixed q. Its variance, finite-sample behavior, and performance under fitted nuisance models remain empirical questions. This benchmark treats in-sample DR separately from cross-fitted DR instead of assuming they are interchangeable.

Score-based normal intervals are reported only for IPS and estimators whose reward-model nuisance fit is external to the evaluation rows (pilot DR/DM and two-fold cross-fitted DR). These are empirical diagnostics, not finite-sample guarantees. The deliberately in-sample DM/DR arms have no reported confidence interval: their nuisance model is fitted on the same evaluation rows, so a standard error calculated from those same scores would not be a defensible interval. Their point-estimate bias, variance across replicates, and MSE remain reported.

## Output

Full and smoke runs emit deterministic TSV. Summary rows contain one row per scenario × sample size × logger × target policy × estimator, with mean shrinkage weight, expected logger reward, mean of replicate-level minimum action propensities, replicate count, bias, across-replicate variance, MSE, empirical 95% interval coverage, mean interval width, identification status, and inference status. In-sample DM/DR use NA for interval coverage and width.

Expected summary rows: 5 scenarios × 2 sample sizes × 5 loggers × 4 target policies × 6 estimators = 1,200. A separate per-replicate trace records each estimate, truth, signed error, squared error, score-dispersion standard error, valid interval endpoints (or NA when same-sample nuisance fitting invalidates score-based inference), and the independent pilot-fit, shrinkage-calibration, and evaluation seeds. That is 153,600 replicate rows for the 128-replicate full run and 14,400 for the 12-replicate smoke run.

All registered loggers preserve a positive action floor, so rows are labelled identified_full_support. The estimator rejects a target action with zero logging support; it does not return an OPE value in that case.

## Falsification and interpretation rules

- Cross-fitting is not presumed to dominate; compare its MSE and coverage against fixed-pilot and in-sample DR across every registered scenario.
- Shrinkage is not presumed to help. Compare held-out and leaky weights with the oracle source-domain prediction-MSE reference and downstream OPE metrics.
- Reward drift can make source-calibrated shrinkage stale. DR's residual correction should be evaluated separately from DM's model-only estimate.
- The benchmark assumes the logging propensities are known exactly. It does not establish robustness to propensity-estimation error.
- Two-fold cross-fitting and score-based normal intervals are transparent baselines, not a claim of optimal nuisance estimation or finite-sample coverage guarantees. In-sample nuisance arms intentionally suppress intervals rather than attach misleading nominal coverage.
- Synthetic results do not establish external validity, commercial readiness, or superiority to SCOPE-RL / Open Bandit Pipeline.

## Industry / SOTA comparison

SCOPE-RL standardizes OPE and policy-selection comparisons using MSE, rank correlation, regret, and Type-I/II errors, and supports direct, importance-sampling and doubly robust estimators. Open Bandit Pipeline supplies real logged bandit feedback with known behavior propensities and a family of OPE methods. Cross-domain OPE/L (ICLR 2025) is a further comparator for source/target-domain transfer.

Useful references:
- [SCOPE-RL documentation](https://scope-rl.readthedocs.io/en/latest/documentation/index.html)
- [Open Bandit Dataset and Pipeline](https://zr-obp.readthedocs.io/en/latest/)
- [Cross-Domain Off-Policy Evaluation and Learning for Contextual Bandits (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/87eb265e8898a7e245d61f01cea4d906-Abstract-Conference.html)
- [Douglas, Persson & Provost (2026), Logging Policy Design for OPE](https://arxiv.org/html/2605.15108)

External-data evaluation is not included in v0. Dataset access terms, preprocessing parity, and target-policy protocol must be checked before claiming a public-data benchmark result.