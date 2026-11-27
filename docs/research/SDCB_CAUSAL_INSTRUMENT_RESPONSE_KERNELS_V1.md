# SDCB causal instrument-response kernels v1

**Status:** The standalone scalar OU + first-order response analytic reference is authored in `scripts/research/sdcb_exponential_response.rs` with nine Rust tests. The tests are not compiled or executed in this environment. No integration into the existing interval-observation module, high-rate experiment, physical qualification, or SOTA result is claimed.

## 1. Motivation

#13903 models scalar OU covariance over rectangular averaging windows. #13940 composes vector LTI state-space propagation with piecewise interval accumulators and preserves endpoint/observation cross-covariance. These are necessary foundations, but a physical sensor can filter or delay the measurand before the reported averaging window is applied.

A timestamp plus a rectangular reporting interval does not fully describe a channel with sample-line delay, finite response time, analog filtering, digital smoothing, or path-dependent hysteresis. SDCB should represent the complete declared observation chain without pretending every instrument is an instantaneous point sample or a rectangular average.

## 2. Observation operator

For latent state x(t), channel loading c_i, reporting endpoint t_i and causal response kernel h_i(s; theta_i), define the pre-acquisition response

\[
r_i(t_i)=\int_0^{T_i} h_i(s;\theta_i)c_i^T x(t_i-\delta_i-s)\,ds
\]

where delta_i is a separately identified transport / processing delay and T_i is the response-memory horizon (or an explicitly bounded tail).

The recorded observation then applies its acquisition operator (for example an ADC aperture or rectangular reporting average) to r_i and adds its declared electronics/acquisition noise.

Keep these uncertainty and causal layers separate:
1. physical measurand evolution;
2. instrument response dynamics;
3. sample-line / transport / processing delay;
4. digital or analog acquisition aggregation;
5. reference-chain uncertainty;
6. shared environmental and measurement-system causes;
7. device-local drift, hysteresis and residual acquisition noise.

The kernel has explicit units, normalization, support, version, parameter covariance or bounded parameter set, and calibration epoch. A timestamp alone never authenticates the kernel or proves traceability.

## 3. M0 tractable cases

Start with exact linear-Gaussian references:

- identity / zero-lag response;
- pure fixed delay;
- first-order exponential low-pass response;
- first-order response followed by a rectangular acquisition average;
- unequal instrument time constants with overlapping reporting windows;
- shared upstream transport delay plus independent device response;
- response-parameter uncertainty represented as explicit finite model alternatives or caller-supplied bounded parameters.

For a first-order response:

\[
dz_i(t)=\frac{c_i^T x(t-\delta_i)-z_i(t)}{\tau_i}\,dt
\]

and the measured value is then the declared acquisition operator applied to z_i(t). The corresponding causal impulse response is h_i(s)=exp(-s/tau_i)/tau_i for s>=0.

**Delay implementation boundary:** a fixed transport delay is not generally equivalent to increasing the sensor time constant. For a general continuous-time state-space process, evaluating x(t-delta) may require a declared history representation, method-of-steps construction, or an independently verified delayed-state covariance calculation. M0 must only claim exactness for the specific process/delay class covered by its fixtures. Do not quietly fold transport delay into tau_i or use a finite-dimensional Markov augmentation unless equivalence is proved for that case.

Verify augmented-state covariance and endpoint/observation cross-covariance against an independent convolution calculation. #13940's rectangular accumulator remains an explicit reduction/special case.

Do not claim arbitrary nonlinear hysteresis in M0. Hysteresis must be represented by a declared path-dependent state model or rejected as unsupported.


### Authored M0 reference

`scripts/research/sdcb_exponential_response.rs` now provides a standalone two-state analytic reference for
`dx = -lambda x dt + sqrt(2 lambda variance) dW` and
`dz = (x-z)/tau_s dt`.
It computes the stationary covariance, exact 2x2 transition matrix, future/past cross-covariance, filtered-sensor autocovariance, normalized step response, and first-order frequency response. It includes the repeated-root case (equal latent and sensor rates) and uses exp_m1-based rate-gap formulas near that case.

Nine Rust tests are authored for stationary covariance, equal-rate and distinct-rate fixtures, zero lag, near-equal-rate continuity, step response, -3 dB / phase behavior, invalid inputs and negative elapsed time. An independent authoring-runtime check using SciPy matrix exponential and continuous Lyapunov solution reproduced the distinct-rate transition/cross-covariance fixture to below 2e-15 absolute difference. This is an independent numerical calculation, not Rust execution, crate compilation, CI, or physical sensor validation.

## 4. Required fixtures and invariants

- Versioned kernel identity, units, normalization, delay, support horizon, parameter values/uncertainty and calibration epoch.
- Exact zero-lag / zero-delay reduction to the existing #13940 API.
- Analytic OU cross-covariance fixtures for exponential response states.
- Step-up / step-down and settling-time fixtures; asymmetric rise/fall only where an explicit model supports it.
- Sinusoidal magnitude attenuation and phase-lag fixtures from the transfer function.
- Unequal time constants and unequal acquisition windows.
- Duplicate-channel and shared-reference negative controls.
- Preservation of endpoint-state / observation process-noise cross-covariance.
- Explicit bounded response-tail truncation; no silent cutoff.
- Reject invalid units, nonpositive time constants, invalid delays, bad kernel normalization, stale response version, and unsupported path dependence.

## 5. Comparators

Use identical latent trajectories, inputs, sampling, noise and resource budgets:

1. Instantaneous timestamp model (negative control).
2. Rectangular-window-only #13940 model.
3. Augmented-state continuous-time Kalman/state-space filter.
4. Gaussian-process regression with matched convolution / response kernels where applicable.
5. Direct convolution integration on manufactured low-dimensional cases.
6. Explicit deconvolution baselines with noise amplification and instability reported.
7. Hierarchical response-time calibration where repeated device measurements support it.

No benchmark should reward access to latent states or reference measurements not available to the declared method.

## 6. Metrics

- step response: rise/fall time, settling time, over/undershoot and hysteresis;
- frequency response: magnitude attenuation and phase lag by frequency;
- event-time and time-alignment error;
- log predictive density, CRPS, 90%/95% coverage and interval width;
- calibration slope/intercept and held-out device/family/site performance;
- drift and response-time change detection delay and false alarms;
- reference/collocation effort per qualified decision;
- terminal decision loss and exact-oracle policy regret;
- response-tail enclosure / approximation error, latency, memory and hard-budget compliance.

## 7. Data and physical validation boundary

The curated QUANT Zenodo release (DOI 10.5281/zenodo.10775692) is hourly. It is useful for schema, cohort and coarse predictive checks, but cannot identify sub-hourly sensor response dynamics. The raw CEDA QUANT sensor archive has original temporal resolution and richer metadata, but does not itself provide all site reference data. The one-minute reference / five-minute mobile series in Kok & Tabandeh (2026) is paper-specific and must be acquired from the corresponding author as stated in the paper.

A proper instrument-response qualification needs a high-rate controlled step and multi-frequency experiment with a synchronized independent reference, documented sample-line delay, response calibration, device temperature/humidity, and repeated up/down transitions. Do not interpolate hourly records to manufacture fast-response evidence.

## 8. Existing owners

Compose with:
- #13903 scalar OU window covariance;
- #13937 vector continuous-time propagation;
- #13940 interval-integrated observations;
- #13939 covariance-model uncertainty;
- #13942/#13943 model adequacy and conformal calibration;
- #13945/#13951 correlated Gaussian likelihoods;
- #13567 exact sequential policy oracle;
- #13739 experiment-selector integration;
- #13805 evidence-gated belief transition;
- #13974 QUANT data qualification.

Do not create another generic scheduler, decision planner, likelihood ledger, evidence ledger or sensor catalogue.

## 9. Acceptance gates (targets, not achieved)

- Exact zero-lag / rectangular reduction agrees with #13940 fixtures.
- 100% agreement with frozen analytic OU/exponential-kernel fixtures within declared tolerances or validated enclosures.
- Zero silent response-tail truncation, timestamp substitution or deconvolution regularization.
- Zero false-independent-confirmation claims on duplicate/shared-cause controls.
- Nominal 90%/95% coverage on held-out high-rate response tests, with interval width reported.
- At most 2% policy regret on exact small-world response-kernel evaluators.
- At least 10% lower terminal decision loss per equal reference cost than the strongest matched baseline, with paired confidence intervals excluding zero before any superiority claim.
- Unsupported response families or unresolved calibration parameters produce explicit abstention.

## 10. References

- Kok and Tabandeh (2026), short-collocation covariance and process variability, DOI: https://doi.org/10.1002/qre.70319.
- QUANT three-year sensor dataset, DOI: https://doi.org/10.1038/s41597-024-03767-2.
- Sampling/response-time trade-offs in low-cost air-quality sensor nodes, Sensors (2022): https://doi.org/10.3390/s22103964.
- Existing SDCB implementation and ownership: #13903, #13937, #13940, #13974, #13567, #13739, #13805.

All acceptance gates are prospective research targets. This specification does not establish a new sensor law, SOTA, device certification, metrological traceability, physical validation, or production capability.
