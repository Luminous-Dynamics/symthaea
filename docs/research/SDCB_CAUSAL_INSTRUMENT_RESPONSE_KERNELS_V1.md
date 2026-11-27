# SDCB causal instrument-response kernels v1

**Status:** The standalone scalar OU + first-order response analytic reference is authored in `scripts/research/sdcb_exponential_response.rs` with twenty-one Rust tests. The tests are not compiled or executed in this environment. No integration into the existing interval-observation module, high-rate experiment, physical qualification, or SOTA result is claimed.

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

Twenty-one Rust tests are authored for stationary covariance, equal-rate and distinct-rate fixtures, zero lag, near-equal-rate continuity, step response, -3 dB / phase behavior, invalid inputs and negative elapsed time. An independent authoring-runtime check using SciPy matrix exponential and continuous Lyapunov solution reproduced the distinct-rate transition/cross-covariance fixture to below 2e-15 absolute difference. This is an independent numerical calculation, not Rust execution, crate compilation, CI, or physical sensor validation.
An additional independent convolution identity is used to test the state-space covariance calculation. With q = mu*sigma^2/(lambda+mu), for h >= 0 and mu != lambda:
`C_zz(h) = q * (mu*exp(-lambda*h) - lambda*exp(-mu*h)) / (mu-lambda)`.
At equal rates mu=lambda, the continuous limit is
`C_zz(h) = q*exp(-lambda*h)*(1+lambda*h)`.
The implementation uses exp_m1-based factorizations to avoid subtractive cancellation near equal rates. Two tests compare this direct-convolution expression against exp(Ah)P across both rate orderings, equal and near-equal rates, and several elapsed times. A separate Python check exercised 49 parameter/time combinations; the maximum absolute discrepancy was below 5e-12.


## Structural identifiability: physical dynamics versus sensor response

The first-order OU + first-order sensor model has an important exact non-identifiability under passive single-channel observation.

Let the latent OU decay rate be lambda, the instrument response rate be mu, and the latent stationary variance be sigma_x^2. The filtered output autocovariance can be written, for h >= 0, as

[
C_{zz}(h)=q,e^{-r h}left[1+r h,psi(g h)ight],
quad
q=rac{musigma_x^2}{lambda+mu},
quad
r=min(lambda,mu),
quad
g=|lambda-mu|,
]
with (psi(x)=(1-e^{-x})/x) and (psi(0)=1).

The normalized covariance shape is symmetric in lambda and mu. More strongly, the parameter transformation

[
(lambda,mu,sigma_x^2)
mapsto
left(mu,lambda,rac{mu}{lambda}sigma_x^2ight)
]

preserves q and therefore preserves the **entire output autocovariance function**. The two worlds are observationally indistinguishable from passive second-order output statistics, even with infinite data and perfect numerical integration.

A Rust regression now freezes this negative control using:
- World A: latent decay lambda=0.25, sensor response mu=4, process variance 9;
- World B: latent decay lambda=4, sensor response mu=0.25, process variance 144.

Both yield the same output covariance at every lag. This is a structural identifiability limitation, not an optimizer defect or numerical tolerance issue.

### Required modelling response

- Do not report a unique physical-process time constant and a unique sensor-response time constant from this single-channel passive data alone.
- Preserve the observational equivalence class / set of parameter assignments, or report that the physical-vs-instrument attribution is non-identifiable.
- Because the output is Gaussian, equality of its mean and covariance gives identical finite-dimensional output laws. No passive single-channel statistical test can recover the physical-vs-instrument labels in this equivalence class.
- A generic step/frequency excitation is **not automatically sufficient**. For the common unit-DC-gain first-order physical relaxation model `dx/dt=lambda*(u-x)` followed by the first-order sensor `dz/dt=mu*(x-z)`, the command-to-output transfer function is `H(s)=lambda*mu/((s+lambda)*(s+mu))`, which is also symmetric under swapping lambda and mu. The commanded input can reveal the unordered pole pair while still failing to label its physical and instrument poles.
- To break the equivalence, use a traceable step or frequency sweep applied **at the sensor inlet**, bypassing the unknown physical-process dynamics; or use an independently qualified reference channel that observes the latent measurand with known response; or independently characterize one of the two transfer functions. An upstream excitation only qualifies when its physical input-to-state model (including gain and dynamics) is independently known and the resulting combined transfer function demonstrably breaks the pole-swap symmetry.
- Treat prior-driven pole attribution as prior-sensitive, not as empirical identification.
- Benchmark both predictive equivalence and causal attribution separately: identical passive predictions may coexist with different physical explanations.



### Active-calibration caveat

Do not assume "active" automatically means identifiable. With the unit-DC-gain relaxation model above, a known input command still passes through two first-order factors whose transfer-function product is invariant under pole exchange. The benchmark must include an upstream-excitation negative control that remains unresolved, and a sensor-inlet or independent-reference positive control that actually breaks the equivalence.

This identifiability requirement composes with #13691 (active scientific intelligence and hidden mechanism identifiability) and #13567 (exact sequential policy oracle). A calibration action should be selected using the existing decision owner, not by adding a new planner to this response-kernel track.

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


### Dynamic-range numerical hardening

A source audit and 50,000-case high-precision stress sweep found two nontrivial dynamic-range failures and prompted further changes:

- **Transition propagation:** evaluating `mu * exp(-lambda*h) * expm1(-gap*h) / gap` in that order can underflow the intermediate product even when the final transition coefficient is representable. The formula now uses `(mu*h) * ((1-exp(-gap*h))/(gap*h)) * exp(-slow_rate*h)`, with an explicit repeated-root limit for subnormal gap-times-time products.
- **Exponential argument:** the exact product rate×time may be lost when rounded to binary64 before the exponential. The reference now uses `mul_add` to recover the multiplication residual and corrects the exponential by `exp(-residual)`.
- **Stationary covariance:** directly forming `mu/lambda` can underflow to zero while `variance*mu/lambda` remains representable. The low-ratio branch now evaluates `(variance*mu)/lambda` in an order whose intermediate range is bounded for finite binary64 inputs.

New regressions cover a subnormal rate ratio with a large variance, subnormal rate-gap products in both rate orderings, and a high-precision transition fixture. The standalone file now contains eighteen authored Rust tests. An authoring-runtime Python/mpmath sweep of 50,000 logarithmically distributed cases (44,275 returned finite nonzero transition references) found at most 3 ulps absolute representation error on normal-range results and a maximum relative discrepancy of approximately 4.9e-16. Subnormal results are assessed by absolute ulp, not relative error, because relative error is not meaningful at the bottom of binary64. This is author-side numerical verification only; Rust remains uncompiled/unexecuted.


The same 50,000-case sweep also exercised the stationary filtered variance across rates/variances spanning approximately 10^-300 to 10^300. After adding a two-order evaluation for subnormal mu/lambda ratios, a separate 100,000-case mpmath comparison of the stationary covariance scalar found maximum 2 ulps error and maximum relative discrepancy 3.5e-16 on normal-range results. This is independent numerical stress evidence, not Rust execution or formal proof.

The normalized step response now evaluates `1-exp(-mu*t)` through `-expm1(-mu*t)`, adding an FMA-derived correction for the rounded rate-times-time product. This avoids cancellation to zero for small response fractions. An authored regression checks a (10^{-20}) response fraction; an independent Python check confirmed the expm1 result retains that value while direct `1-exp(-x)` rounds to zero.

## Composition with the existing interval-observation engine

The response model now exposes its equivalent continuous-time linear-SDE matrices for state `S=[x,z]`:

[
A=\begin{bmatrix}-\lambda&0\\\mu&-\mu\end{bmatrix},
qquad
Q_c=\begin{bmatrix}2\lambda\sigma_x^2&0\\0&0\end{bmatrix}.
]

A separate bridge `scripts/research/sdcb_response_interval_bridge.rs` imports the response reference and the existing #13940 interval-observation implementation. It constructs observation windows loading the sensor state z, not the latent measurand x. It reuses #13940's piecewise augmented accumulator, Van Loan transition and endpoint/observation cross-covariance; no second interval integrator was added.

Four bridge Rust tests are authored:
1. Overlapping/nested windows [0,2] and [1,2].
2. Adjacent windows [0,1] and [1,2], preserving their nonzero cross-window covariance.
3. Duplicate window IDs and zero-duration rejection.
4. Acquisition/electronics noise added separately after physical response averaging.

For lambda=1, process variance=1 and sensor time constant tau=0.5 (mu=2), the latent-to-sensor covariance is (C_{zz}(h)=\frac{2}{3}(2e^{-h}-e^{-2h})). Independent SciPy double integration gives the nested-window covariance matrix:

[
\begin{bmatrix}
0.5053638855836804 & 0.5053638855836804\\
0.5053638855836804 & 0.6025667487115534
\end{bmatrix}
]

For adjacent unit windows the expected matrix is:

[
\begin{bmatrix}
0.6025667487115534 & 0.4081610224557192\\
0.4081610224557192 & 0.6025667487115534
\end{bmatrix}.
]

The authoring-runtime SciPy `dblquad` checks used absolute/relative tolerances 1e-13. These are independent numerical references, not Rust execution. The bridge adds four authored tests, all uncompiled/unexecuted.