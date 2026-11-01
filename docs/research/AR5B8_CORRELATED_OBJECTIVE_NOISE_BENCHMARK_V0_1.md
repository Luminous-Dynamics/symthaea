# AR-5B8: Correlated Bounded Objective Noise Benchmark v0.1

**Status:** synthetic generator and tests authored; exact-head validation pending.  
**Stack:** #9429 → #9718 → #9719 → #9721 → #9754 → #9782 → this branch.  
**Research trigger:** Neurocomputing 700 (published 1 November 2026), DOI https://doi.org/10.1016/j.neucom.2026.134400.  
**Claim ceiling:** deterministic bounded synthetic observation generator with a known low-rank cross-objective correlation structure. It is not a GP, does not estimate covariance from data, and does not prove physical sample independence.

## 1. Why this is a first-class benchmark track

The new Neurocomputing paper studies multi-objective Bayesian optimization with intrinsic coregionalization models and distinguishes latent objective correlation from correlated observation noise. Its reported findings are conditional:
- in noisy settings, multi-output GPs can improve optimization efficiency;
- modeling correlated noise can provide additional benefit when objective and noise correlations are structured;
- in noiseless settings, independent GPs can be cheaper with similar optimization outcomes;
- real-world evidence for noise-correlation modeling is mixed.

This argues against a universal multi-output model. Symthaea should measure when covariance modeling changes decisions enough to justify its additional fit and acquisition cost.

Reference: https://doi.org/10.1016/j.neucom.2026.134400

## 2. Implemented bounded factor model

For objective i:

```
epsilon_i = sigma_i * (lambda_i * z_shared
                     + sqrt(1 - lambda_i^2) * z_i)
```

where:
- each z is a Rademacher random variable in {-1,+1};
- the shared factor is common to all objectives in one candidate/replicate;
- each z_i is an objective-specific idiosyncratic factor;
- sigma_i is the marginal noise scale;
- lambda_i is a signed shared loading in [-1,1].

The known covariance and correlation are:

```
Var(epsilon_i) = sigma_i^2
Cov(epsilon_i, epsilon_j) = sigma_i * sigma_j * lambda_i * lambda_j
Corr(epsilon_i, epsilon_j) = lambda_i * lambda_j, i != j
Corr(epsilon_i, epsilon_i) = 1
```

This covariance is positive semidefinite by construction: the correlation matrix is lambda*lambda^T plus a nonnegative diagonal residual matrix. Signed loadings permit positive and negative cross-objective correlations.

The absolute noise bound is:

```
|epsilon_i| <= sigma_i * (|lambda_i| + sqrt(1 - lambda_i^2))
```

This permits conversion into conservative observation bounds in a frozen EventTape noisy-objective contract. The means/objective functions still need their own certified ranges before a complete observation bound can be declared.

## 3. Common-random-number semantics

The counter key binds campaign seed, candidate identity and replicate identity. Policy identity is intentionally excluded. Therefore, two matched policies evaluating the same candidate and replicate receive the same potential observation. This supports paired comparisons and avoids noise schedule differences being mistaken for policy effects.

The stream uses SHA-256 domain separation plus SplitMix64 counter mixing. It is a deterministic synthetic benchmark stream, not evidence of independent physical replicates. Actual independence/stationarity assumptions must be justified by the sampling protocol and evaluated separately. EventTape should retain the model digest, exact observation bytes and resource charge as before.

## 4. Scope and limitations

- One shared latent Rademacher factor plus independent objective-specific factors; it is a rank-one shared-noise model, not a general covariance family.
- Bounded, discrete, non-Gaussian noise; no claim to reproduce Gaussian tails.
- Static covariance and heteroskedastic per-objective scales; no drift or time-varying covariance in v0.1.
- The model generates observations; it does not fit independent or multi-output GPs.
- Tests include analytical covariance, empirical correlation calibration over a deterministic counter stream, deterministic replay, bounds, identity and malformed-contract controls.
- Empirical calibration is a unit-level diagnostic, not a statistical guarantee for physical experiments.

## 5. Matched model comparison

Use one shared EventTape campaign and the same potential outcomes to compare:

1. Independent per-objective surrogate, independent-noise likelihood.
2. Multi-output ICM surrogate, independent-noise likelihood.
3. Multi-output ICM surrogate, correlated-noise likelihood.
4. Low-rank multi-output variants with rank and regularization preregistered.
5. Decision-aware acquisition and sequential policy variants on each surrogate.

Use at least these data-generating regimes:
- noiseless control;
- independent observation noise;
- positive correlated objective signal and positive correlated noise;
- negative correlated objective signal with positive or negative correlated noise;
- objective signal independent but noise correlated;
- heteroskedastic objective scales;
- model bias that makes a richer covariance model unhelpful;
- common-mode drift and time-varying correlation.

For each regime, use identical candidate decisions, objective contracts, common random numbers, evaluation budgets, and independent campaign clusters. Retain every cost including model fitting, acquisition optimization, failed measurements, verification, artifact retrieval, CPU, wall time, accelerator, peak memory and energy.

## 6. Scoreboards

**Optimization:** anytime hypervolume, IGD+, epsilon, target-attainment ECDF, archive coverage and reference sensitivity.

**Decision:** terminal decision loss, Pareto-front decision regret, false dominance, false feasibility, false stopping and calibration.

**Statistical:** covariance estimation error, held-out predictive log score, interval coverage, residual cross-correlation, drift-detection delay and false alarms.

**Economics:** total wall-clock and compute cost per decision-risk reduction; model complexity only earns credit if it changes downstream outcomes at equal all-in budget.

## 7. Proposed gates (not results)

- Generated noise remains inside every declared per-objective bound.
- The analytic correlation matrix is positive semidefinite and matches the declared loadings.
- Empirical counter-stream correlations fall within a preregistered tolerance over fixed deterministic calibration seeds.
- Matched policies receive identical observations for the same candidate/replicate key.
- Every comparison uses exact EventTape objective-contract and raw-evidence replay.
- Multi-output/correlated-noise models are only promoted if they improve held-out decision or optimization quality at equal all-in cost.
- No single synthetic covariance regime supports a general claim of superiority.

## 8. Validation commands

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering correlated_noise_benchmark -- --nocapture
cargo test -p symthaea-architecture-engineering event_tape_noisy -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

No compile, CI, benchmark, empirical SOTA, or physical independence claim is made by this tranche.
