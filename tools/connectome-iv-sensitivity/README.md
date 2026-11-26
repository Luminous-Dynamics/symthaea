# Single-instrument profiled Anderson–Rubin sensitivity reference

Standalone, dependency-free Rust 2024 implementation of the sample-normalized one-instrument closed form of the profiled Anderson–Rubin (pAR) test.

## Why this exists

The September 2026 pAR preprint combines two concerns that ordinary connectome-informed IV estimates often treat separately:

- weak first stages, where conventional Wald inference can become unreliable;
- bounded direct effects of the instrument, where exact exclusion is not fully credible.

For each candidate structural coefficient, pAR profiles over the complete admissible support of direct instrument effects. It does not pick a small grid of direct-effect values and then pretend the unexamined values do not exist. Under the preprint's maintained conditional Gaussian fixed-design model, it retains a candidate whenever at least one admissible direct effect is consistent with the data.

Reference: Lee (2026), *Profiled Anderson–Rubin Test: Robust Inference Allowing for Direct Effects of Instruments*, https://arxiv.org/abs/2609.18150. This is a recent preprint, not a peer-reviewed validation of this crate.

## Scope

This crate implements only:

- one scalar endogenous treatment;
- one scalar instrument;
- an intercept and no other controls;
- fixed design, conditional Gaussian, homoskedastic errors, independent rows;
- sample-normalized RMS direct-effect support;
- one-candidate pAR evaluation with a caller-supplied F critical value.

It does **not** implement multi-instrument pAR, covariates, dynamic neural state-space models, clustered errors, heteroskedasticity, a continuous confidence-set inversion, or biological estimation.

The pAR preprint explicitly identifies heteroskedastic and clustered extensions as future work. The fixed-design Gaussian validity argument must not be transferred mechanically to repeated neurons, sessions, or animals. A separate proof and evaluator are required before those data regimes can use this statistic as inferential evidence.

## Formula

For candidate structural effect b, regress Y - bX on an intercept and the instrument Z. Let gamma_hat(b) be the resulting instrument coefficient, sigma_hat(b)^2 the residual mean square with n-2 degrees of freedom, and

```
AR_exact(b) = gamma_hat(b)^2 * sum((Z - mean(Z))^2) / sigma_hat(b)^2
```

For the sample-normalized RMS support, the one-instrument pAR statistic is

```
pAR(b; g) = max(sqrt(AR_exact(b)) - sqrt(n) * g / sigma_hat(b), 0)^2
```

The direct-effect support means

```
abs(gamma) * sqrt(sum((Z - mean(Z))^2) / n) <= g
```

where g is in outcome units. A candidate is not rejected when the statistic is no greater than the caller-supplied upper-tail F(1,n-2) critical value.

The function reports:
- first-stage slope and F statistic as continuous diagnostics, with no universal F>10 verdict;
- reduced-form slope and the conventional Wald ratio for comparison only;
- the direct-effect coefficient implied by the tested candidate and its sample RMS;
- exact-exclusion AR F, profiled AR F, the declared critical value, and the resulting non-rejection indicator.

The conventional Wald ratio can be badly misleading when the instrument has a direct path to the outcome. It is shown for comparison, not used for inference.

## Synthetic known-answer results

The included fixture uses a fixed 20-observation, centered one-instrument design and F(1,18) 95% critical value 4.413873419170566. It records:

- strong first stage 2.0, first-stage F 1210.3134, reduced-form slope 6.6 and Wald ratio 3.3 when the synthetic structural coefficient is 3.0 and direct effect is 0.6;
- bounded-direct-effect pAR for candidate 3.0: with RMS bound 0.35, statistic 0 and not rejected; with bound 0.31, statistic approximately 2.74095 and not rejected; with bound 0.10, statistic approximately 128.12744 and rejected;
- weak first stage 0.05, first-stage F approximately 0.75645; the far candidate 10.0 is not rejected under bound 0.35, while the same candidate is strongly rejected with a strong first stage;
- zero first-stage relevance with a bound that covers the reduced-form direct effect, leaving widely separated candidate coefficients unrejected. This is expected non-identification, not a bug to be hidden.

These are synthetic mathematical cases, not a biological result or a calibrated causal effectome.

## Run

From the repository root:

```sh
cargo test --locked --manifest-path tools/connectome-iv-sensitivity/Cargo.toml
```

No third-party dependencies; the crate is an isolated Cargo workspace and uses the repository's AGPL-3.0-only license.

## Scientific safety boundary

The test assumes the stated support g was chosen before observing the held-out outcome. pAR acceptance means only that a candidate is not rejected under the declared support and maintained sampling model. It does not prove the candidate is true or that the exclusion restriction is satisfied.

Instrument randomization/independence, opsin specificity, direct light effects, thermal effects, unobserved cells, extrasynaptic paths, and treatment consistency require independent experimental evidence. This crate does not audit them.

Do not use this implementation for animal/session-clustered or heteroskedastic observations as if it were robust to those designs. Do not convert its synthetic fixture into a biological performance claim.
