# Predictable conditional-innovation Gaussian e-process v0

**Date:** 2026-11-16  
**Owner:** Symthaea #12442  
**Exact-head qualification:** passed on 37449e8314ceb892482935ea99e98dc3bf631fab; Rust 1.96.0, 32/32 tests, rustfmt and strict Clippy, focused workflow run #48  
**Composes with:** #12375, #12402, PR #12405; Symtropy #3784/#3973

## 1. Objective

Extend the existing predictable Gaussian sequential likelihood-ratio reference from a zero-conditional-mean residual to a known conditional Gaussian innovation process. This makes the evaluator capable of handling a correctly specified AR/state-dependent conditional mean without pretending the observations are marginally independent.

This is a narrow evaluator-side statistical reference. It is not an autoregressive parameter estimator, generic time-series inference engine, experiment selector, measurement authority or physical validation.

## 2. Conditional likelihood-ratio argument

Let F_(t-1) include prior observations, selected actions, and all information committed before observation t. The evaluator freezes/commits:

- null conditional mean mu_t, predictable with respect to F_(t-1);
- positive conditional standard deviation sigma_t, predictable;
- a declared alternative conditional-mean discrepancy d_t, also predictable.

Under the simple conditional null:

    H0: Y_t | F_(t-1), A_t ~ Normal(mu_t, sigma_t^2)

Under the simple alternative:

    H1: Y_t | F_(t-1), A_t ~ Normal(mu_t + d_t, sigma_t^2)

The conditional likelihood-ratio increment is:

    e_t = exp(d_t * (Y_t - mu_t) / sigma_t^2 - d_t^2 / (2 * sigma_t^2))

Because mu_t, sigma_t and d_t are fixed using only pre-observation information:

    E_0[e_t | F_(t-1), A_t] = 1

The product E_t = product_(s<=t) e_s is therefore a nonnegative martingale under the declared simple conditional null. Ville's inequality controls the probability of ever crossing 1/alpha.

This remains valid when observations are serially dependent marginally, provided the conditional Gaussian distribution is correctly specified at each step. An AR(1) example is mu_t = rho * Y_(t-1) with known rho. This is not a blanket allowance for arbitrary serial dependence.

## 3. API composition

Backward-compatible methods:
- commit_prediction(..., d_t) uses mu_t=0 and the process default sigma.
- commit_prediction_with_sigma(..., d_t, sigma_t) uses mu_t=0 and explicit known sigma_t.
- commit_predictions(..., d_vector) and commit_predictions_with_sigma(..., d_vector, sigma_t) do the same for a fixed alternative mixture.

New methods:
- commit_prediction_with_conditional_gaussian(..., d_t, mu_t, sigma_t)
- commit_predictions_with_conditional_gaussian(..., d_vector, mu_t, sigma_t)

The pending prediction stores observation ID, decision digest, all alternative discrepancies, mu_t and sigma_t. No observation update occurs unless the corresponding identity matches. The emitted single/mixture receipt retains the conditional mean and variance for independent replay.

The mixture remains a fixed, predeclared convex mixture. Post-hoc selection of the highest e-value component remains invalid.

## 4. Statistical boundaries

v0 requires:
- a simple conditional Gaussian null;
- correctly specified, known mu_t and sigma_t;
- predictability of mu_t, sigma_t, d_t and the alternative mixture;
- a pre-observation commit and correctly aligned observation;
- known likelihood structure for each conditional step.

It does not cover:
- estimating rho, mu_t or sigma_t from the same adaptive data without a separate valid sequential method;
- a misspecified AR/state transition or sensor measurement model;
- latent state posterior uncertainty not represented in the conditional likelihood;
- heavy-tailed, non-Gaussian, censored or informative missing observations;
- unmodelled regime change or variance drift;
- composite nulls and nuisance parameter estimation;
- arbitrary physical time series because they “look AR-like”.

The guarantee is only as good as the conditional null. A valid e-process under the wrong conditional model is not a valid scientific adequacy claim.

## 5. Executed calibration design

The added binary conditional_innovation_calibration uses a deterministic SplitMix64 / Box-Muller generator for simulation only:

- 20,000 paths per scenario;
- maximum horizon 100;
- nominal alpha 0.05;
- AR(1) rho = 0.65;
- sigma = 0.80;
- declared conditional mean shift d = 0.22.

Scenarios:
1. Null: Y_t = rho * Y_(t-1) + sigma * epsilon_t.
2. Alternative: Y_t = rho * Y_(t-1) + d + sigma * epsilon_t.

The same path is presented to:
- the correctly specified conditional-innovation e-process, which commits mu_t = rho * Y_(t-1);
- a deliberate negative control using the zero-conditional-mean e-process despite the AR(1) null.

Report:
- empirical anytime crossing rate and Wilson 95% interval;
- mean crossing look among crossers;
- mean observations used, counting non-crossers as reaching the full horizon;
- null and alternative results separately.

The focused workflow asserts for this frozen generator:
- correct conditional null crossing <=6%;
- misspecified zero-mean null crossing >10%;
- correct conditional alternative detection >50%.

Exact run #48 (https://github.com/Luminous-Dynamics/symthaea/actions/runs/43825085862) passed all gates. The observed 20,000-path results were:

| Scenario | Correct conditional monitor | Zero-mean misspecified control |
|---|---:|---:|
| AR(1) null crossing | 3.035% (Wilson 95% 2.806–3.282%) | 42.590% (41.906–43.277%) |
| AR(1) conditional alternative crossing | 72.855% (72.234–73.467%) | 98.665% (98.496–98.815%) |

The correct monitor used a mean 98.57 observations on null paths (only 3.04% crossed) and 65.06 observations on alternative paths. The zero-mean negative control is invalid under this AR(1) null; its high crossing rate illustrates why a misspecified conditional baseline must not be treated as valid evidence. The exact source SHA was 37449e8314ceb892482935ea99e98dc3bf631fab. Artifact archive SHA-256: 511f56571a69cceaa35c995315e2abec191968957f1a64b3e04f1613166e6bd5; report JSON SHA-256: 91ab9f3a87a37379ba87f9d79fab4fc2fe291f57b1dc0ec64dc292a37571b8f9.

These are deliberately separated smoke gates and one calibration dataset. They are evidence only for the fixed simulator/seed schedule, not a theorem or cross-family error-rate proof.

## 6. Composition and claim boundary

- Symthaea #12442/#12402 own this conditional-innovation extension and Gaussian sequential semantics.
- Symthaea #12375 owns model-adequacy power and alternative-specific detectability.
- Symthaea #11171/#11060 own the actual blind research-policy integration; evaluator-only mu_t/d_t and results must remain hidden from the candidate.
- Symtropy #3784/#3973 owns physical sensor covariance, parameter identifiability and measurement-model qualification.
- Symtropy #5328 and Mycelix #6031 own sealed challenge custody and evidence provenance.

A green Rust workflow validates implementation and one synthetic AR(1) campaign only. It does not establish general time-uniform validity for estimated conditional models, causal identification, real-world prediction, physical truth or SOTA superiority.
