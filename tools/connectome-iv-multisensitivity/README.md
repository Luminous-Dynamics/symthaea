# Multi-instrument profiled Anderson–Rubin sensitivity reference

Standalone, dependency-free Rust 2024 reference implementation of the sample-normalized multi-instrument profiled Anderson–Rubin (pAR) statistic.

## Motivation

Optogenetic effectome experiments can use multiple independently controlled laser inputs. Treating each stimulation channel as a separate instrument creates a vector of candidate direct effects, not one scalar nuisance coefficient.

The September 2026 pAR preprint derives a general multi-instrument projection and a closed form when the admissible-set metric is sample-normalized:

- Lee (2026), *Profiled Anderson–Rubin Test: Robust Inference Allowing for Direct Effects of Instruments*: https://arxiv.org/abs/2609.18150.
- Pospisil et al. (2024), *The fly connectome reveals a path to the effectome*: https://doi.org/10.1038/s41586-024-07982-0.

The pAR reference is a recent preprint and this crate is an independently authored narrow implementation. It is not a reproduction of all paper estimators or results.

## Model and sample-normalized support

There is one scalar endogenous treatment X, scalar outcome Y, intercept, and an instrument matrix Z with r columns. After centering, let K = Z'Z. For a candidate structural coefficient b:

```
gamma_hat(b) = K^-1 Z'(Y - bX)
AR_exact(b)  = gamma_hat(b)' K gamma_hat(b) / sigma_hat(b)^2
```

The direct instrument-to-outcome effects are a vector gamma. The sample-normalized ellipsoid restricts their realized direct outcome contribution:

```
gamma' K gamma / n <= g^2
```

This is equivalent to bounding the sample RMS of Z gamma by g. Under this support, the general-r closed form is:

```
pAR(b; g) = max(sqrt(AR_exact(b)) - sqrt(n) * g / sigma_hat(b), 0)^2
```

The candidate is not rejected when:

```
pAR(b; g) / r <= F_critical(r, n - 1 - r)
```

The F critical value is an explicit caller input and must be fixed before held-out outcome unblinding.

**Important:** the sample-normalized metric K/n depends on the realized instrument design. The procedure concerns its associated sample-dependent, design-conditional admissible set. It is not automatically a test of a fixed population RMS bound. The paper separately analyzes the population-support problem; do not silently conflate them.

## Implemented

- Dense n×r instrument rows, one scalar treatment and outcome.
- Intercept removal by centering; no additional covariates.
- Instrument Gram matrix and Cholesky-based positive-definite solves; no explicit matrix inverse.
- Per-column-scale-relative numerical rank guard for singular / numerically indistinguishable instrument columns.
- First-stage coefficient vector and joint partial F statistic, reported continuously without a universal F>10 verdict.
- Reduced-form coefficient vector and scalar 2SLS diagnostic.
- Candidate-compatible vector of direct instrument effects.
- Sample RMS of the implied direct contribution, pAR statistic and caller-supplied F critical comparison.
- Compensated summation for centered moments and dot products.
- 22 authored known-answer and failure-closed unit tests, including extreme instrument-unit rescaling, correlated instrument bases and column-permutation invariance.

## Assumptions and limitations

This reference is conditional Gaussian, fixed-design, homoskedastic, independent-row, and intercept-only. It does not implement cluster-robust or heteroskedastic pAR, multiple endogenous treatments, extra controls, confidence-set inversion, dynamic models, or biological estimation.

The preprint explicitly leaves heteroskedastic and clustered extensions as future work requiring a separate validity argument. Do not treat many cells or time points from a small number of animals as independent rows. Animal/session cluster inference remains out of scope.

The statistic does not validate instrument randomization, instrument independence, exclusion support, actuator specificity, off-target illumination, thermal effects, unobserved source cells, or absence of extrasynaptic routes. The support g must be prespecified and justified. A candidate not rejected under a chosen support is not proven true.


## Negative-control calibration of the direct-effect support

The `control_calibration` module estimates a conservative upper bound on the sample-RMS direct instrument contribution from an independent negative-control cohort.

Accepted calibrator types are:

- **Opsin-free / light-on:** measures direct illumination, heating, and measurement paths that do not require the intended opsin treatment channel.
- **Fluorophore-only / light-on:** a related control for optical and sensor effects.

Opsin-expressing/light-off controls are valuable for dark expression/activity effects but are not, by themselves, a calibration of light-driven direct instrument effects.

For the control regression `Y0 = intercept + Z * gamma + error`, the multivariate regression confidence ellipsoid has radius `sqrt(r * sigma_hat^2 * Fcrit)` in the `K = Z'Z` metric. By the triangle inequality, the control-only sample-RMS upper bound is

```
g_control = sqrt(gamma_hat' K gamma_hat / n)
          + sqrt(r * sigma_hat^2 * Fcrit / n)
```

The caller supplies the nominal confidence level and the matching `F(r, n-1-r)` critical value. The function records both and does not compute distribution quantiles.

The control and target designs must have matching normalized Gram matrices `K/n`, checked numerically and via separately supplied normalized-Gram SHA-256 provenance. A design mismatch fails closed. The calibration output separates:

- `control_only_upper_bound_rms`: statistical upper bound under the stated control regression assumptions;
- `transport_multiplier`: caller-specified sensitivity factor (at least 1);
- `applied_sensitivity_bound_rms`: the product used as a sensitivity value in pAR analysis.

The transport multiplier is **not** learned from the controls and `transportability_proven` remains false. A no-opsin control may not reproduce every state-dependent or opsin-expression-dependent bias in the target cohort. Wavelength, irradiance, pulse timing, geometry, temperature, strain, age, imaging sensor and acquisition pipeline must be matched and documented. If that transfer cannot be defended, report a sensitivity ladder for the multiplier and do not call the target effect identified.

The calibration function is conditional Gaussian, fixed-design, homoskedastic and independent-row only. It is not cluster-robust, does not validate the negative-control outcome assertion, and does not prove the exclusion restriction.

## Synthetic known answers

The two-instrument fixture uses n=8, r=2 and residual degrees of freedom 5. The caller-supplied 95% F(2,5) critical value is 5.786135043349963.

- Gram matrix is diag(8,8), first-stage vector [2.0, 0.5], joint first-stage F 1062.5.
- Synthetic structural coefficient 3.0 and direct-effect vector [0.6,-0.4] produce reduced-form vector [6.6,1.1] and conventional 2SLS ratio 3.2352941176.
- Implied direct-effect sample RMS is sqrt(0.52) ≈ 0.7211103.
- With g=0.8, pAR is zero; the candidate is not rejected.
- With g=0.5, pAR/r ≈ 3.05561; the candidate is not rejected at the supplied critical value.
- With g=0.3, pAR/r ≈ 11.08337; the candidate is rejected.
- The one-instrument fixture matches the scalar pAR known answer for the same n, data, g and F(1,18) critical value.

The fixtures are mathematical known answers only. They are not empirical effectome results.

## Run

From the repository root:

```sh
cargo test --locked --manifest-path tools/connectome-iv-multisensitivity/Cargo.toml
```

The crate has no third-party dependencies, is an isolated Cargo workspace, and uses the repository's AGPL-3.0-only licence.
