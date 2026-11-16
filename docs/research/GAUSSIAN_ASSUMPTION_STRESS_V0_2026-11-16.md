# Gaussian observation-model assumption stress v0

**Date:** 2026-11-16  
**Tracking issue:** #12540  
**Composition:** #12426, #12472, #12442, #8607  
**Implementation:** gaussian_assumption_stress in research/falsification-power-reference

## Purpose

The fixed-design Gaussian terminal test is mathematically qualified under an independent, known-variance Gaussian observation model. The existing five-seed campaign establishes Monte Carlo stability inside that model. It does not establish robustness when the observation process changes.

This campaign holds the test, alpha, discrepancy, horizon, design sigma and seed protocol fixed while changing only the synthetic null residual process. The objective is to map an assumption-failure surface, not to make the existing Gaussian theorem more general.

## Frozen test contract

- nominal alpha: 0.05;
- fixed horizon: 80;
- alternative discrepancy: 0.22 per look;
- design standard deviation: 0.80 at every look;
- test: existing GaussianTerminalInducedTest, unchanged;
- five unique root seeds;
- 10,000 paired null/alternative paths per seed and regime;
- 50,000 paths per scenario/regime;
- PRNG: SplitMix64 / Box-Muller v1, for simulation only.

Null and alternative for a regime share the same residual realization; the alternative adds the frozen discrepancy to every look. This produces a paired sensitivity comparison.

## Observation-process strata

1. Matched iid Gaussian: independent N(0, sigma^2). This is the only in-contract reference condition.
2. Stationary AR(1): rho = 0.65 and marginal variance sigma^2. The tested statistic continues to assume independent residuals.
3. Standardized Student-t(5): scaled by sqrt((df-2)/df) to preserve unit variance.
4. Smooth heteroscedastic ramp: per-look variance factor from 0.5+(t+0.5)/80; the mean factor over the horizon is exactly one.
5. Abrupt variance shift: first 40% of looks use variance factor 0.25, final 60% use 1.5; aggregate average variance factor is exactly one.
6. Sparse gross contamination: 2% component with variance factor 37.75 and 98% base component with variance factor 0.25; the mixture's marginal variance factor is exactly one.

Each stressed stratum is marked OUT_OF_CONTRACT even if its empirical rejection rate happens to be close to 5%. Only matched iid Gaussian is marked IN_CONTRACT.

## Outputs

For each regime and each frozen root seed, report:
- terminal null rejection count/rate and Wilson 95% interval;
- terminal declared-alternative rejection count/rate and Wilson 95% interval;
- residual mean and marginal variance;
- within-path pooled lag-one autocorrelation.

The pooled record also reports false-alarm excess relative to 0.05, and keeps analytic iid terminal power as a reference. The lag-one and residual-moment fields are evaluator diagnostics, never selector-visible truth.

## Interpretation

The in-contract iid positive control must be near 5% size and its alternative power must agree with the analytic Gaussian terminal power under a predeclared Monte Carlo tolerance.

The other regimes are deliberately outside the theorem's assumptions. Their rejection rates are observations of sensitivity/misspecification, not acceptance targets. The workflow does not demand a 5% false-alarm rate on them, and it must not call them calibrated simply because a particular finite Monte Carlo estimate is near nominal.

An AR(1) model with a known predictable conditional mean can be handled by a distinct, correctly specified conditional-innovation method (#12442). This fixed-independence terminal test does not inherit that method's guarantee. Unknown rho, estimated variance, heavy tails, non-Gaussian innovations and changing variance require separate valid treatment.

## Claim ceiling

A passing workflow validates the generator, deterministic replay, in-contract positive control and the stress-report schema for this finite synthetic campaign. It does not create a robust test, identify the true noise model from blind data, qualify optional stopping under these stressed conditions, prove real sensor behavior, or extend the Gaussian result beyond its stated assumptions.


## 2026-11-16 exact-head result

Exact source head: `b568eb92d8e985b9f5089ed30f6c559cb3e515df`  
Workflow: https://github.com/Luminous-Dynamics/symthaea/actions/runs/43961429461

The focused workflow passed:
- rustfmt;
- 46 Rust tests across the library and four binary test targets;
- existing sequential Gaussian calibration and receipt;
- existing single-seed terminal induction and receipt;
- existing five-seed terminal calibration and receipt;
- six-regime assumption-stress campaign and receipt;
- strict Clippy with warnings denied.

### Pooled empirical results

| Observation process | Contract | Null rejection | 95% Wilson interval | Alternative power | Lag-1 residual correlation |
|---|---|---:|---:|---:|---:|
| Matched iid Gaussian | In contract | 4.984% | 4.797–5.178% | 79.268% | -0.0006 |
| AR(1), rho = 0.65 | Out of contract | **21.820%** | 21.460–22.184% | 64.748% | 0.6502 |
| Student-t(5), variance-standardized | Out of contract | 4.888% | 4.702–5.080% | 79.326% | 0.0000 |
| Smooth heteroscedastic ramp | Out of contract | 4.792% | 4.608–4.983% | 79.322% | 0.0009 |
| Abrupt variance regime shift | Out of contract | 5.112% | 4.922–5.309% | 79.318% | 0.0001 |
| 2% sparse gross contamination | Out of contract | 4.860% | 4.675–5.052% | 81.594% | -0.0002 |

The IID analytic terminal power is 79.2413%; its empirical alternative estimate is 79.268%, with a Wilson interval that contains the analytic value. All six generators had pooled residual variance near the design variance sigma² = 0.64, so the AR(1) false-alarm inflation is not explained by a marginal variance mismatch. The AR(1) generator's measured lag-one correlation was 0.6502, matching the declared 0.65 target.

### Main finding

The independent-Gaussian terminal test rejected the true null in 21.82% of the AR(1) paths, versus the nominal 5%. That is a 16.82 percentage-point excess and about 4.36× nominal rejection. This is a concrete failure when serial dependence is ignored.

The other out-of-contract conditions happened to remain close to nominal size in this particular 80-look design. That does **not** qualify them: a near-nominal finite Monte Carlo estimate is not a theorem, and these rows remain explicitly OUT_OF_CONTRACT. The contamination alternative power also rose to 81.594%, illustrating that departures can change power as well as false-alarm behavior.

The predictable conditional-innovation Gaussian reference in #12442 is the right separate comparator for a *known and correctly specified* AR(1) conditional mean. It must not be conflated with this misspecified independent-observation test. Unknown rho, estimated conditional means/variances and non-Gaussian innovations still require separate treatment.

### Receipt identity

- Stress report JSON SHA-256: `7a548f05a629a33363213afcca6b505edc167d49d4f8a588439fada4ab35579a`
- Uploaded stress artifact ZIP SHA-256: `8c9e748f571513a9150cd104f39a533e3af564166c5197f4a87d08e993864499`
- Multi-seed IID calibration artifact ZIP SHA-256: `7afd00a48cbb262909959ab01c744f2c93132d4b83f5f7097c4f7f496df2b42b`


## 7. Separately qualified known-rho AR(1) conditional comparator

The v1 campaign adds a distinct comparator only on the stationary AR(1) row. This comparator assumes the evaluator-known rho=0.65 and the correctly specified Gaussian conditional law; it does not estimate rho from observations or provide a blind noise classifier.

For raw AR(1) residuals y_t under the null:

    y_0 ~ N(0, sigma^2)
    y_t | y_(t-1) ~ N(rho*y_(t-1), sigma^2*(1-rho^2))

Transform to innovations:

    e_0 = y_0
    e_t = y_t - rho*y_(t-1), t >= 1

For a persistent raw-observation mean discrepancy d at every look, the predictable innovation discrepancies are:

    d_0 = d
    d_t = (1-rho)*d, t >= 1

The innovation standard deviations are:

    sigma_0 = sigma
    sigma_t = sigma*sqrt(1-rho^2), t >= 1

The existing fixed Gaussian terminal-test implementation is applied to those independent known-variance innovations as a distinct simple-vs-simple design. It is reported only as `conditional_known_rho_ar1` on the AR(1) regime and is labelled `IN_CONTRACT_KNOWN_RHO_AR1`. It is not merged with the original IID statistic, and its analytic power is distinct from the IID analytic power.

This is a deliberately strong known-parameter evaluator comparator. It does not show that Symthaea can infer rho, establish the model's correctness from blind observations, handle estimated parameters, or justify an AR(1) model in the physical world. It demonstrates the expected change in operating characteristics when the dependence model is correctly specified rather than ignored.



### Known-rho AR(1) comparator: exact-head results

Exact source head at the comparator run: `27b328b191cc4998d8442dcd604261356312663a`  
Workflow: https://github.com/Luminous-Dynamics/symthaea/actions/runs/43964216088

The same 50,000 AR(1) null paths and 50,000 matched alternatives were passed through two different fixed-horizon tests:

| Test / contract | Null rejection | Wilson 95% interval | Alternative rejection | Analytic alternative power |
|---|---:|---:|---:|---:|
| Naive IID test applied to AR(1) observations (out of contract) | 21.820% | 21.460–22.184% | 64.748% | IID-design reference 79.241% is not applicable |
| Known-rho conditional-innovation test (separately in contract) | **5.026%** | 4.838–5.221% | **31.112%** | **31.348%** |

The known-rho conditional alternative Wilson interval is 30.708–31.519%, containing its analytic 31.348% power. All five root seeds produced 10,000 paths per scenario; their conditional null rates ranged 4.70–5.20%, and conditional alternative rates ranged 30.61–31.63%.

This is a known-parameter conditional Gaussian comparison only. It demonstrates:
1. Ignoring serial dependence can inflate false alarms by more than fourfold even when the marginal variance is correct.
2. Correctly modeling the known AR(1) conditional mean and innovation variance restores the nominal null size.
3. Restoring size costs power in this design: 31.1% versus the naive test's 64.7% (which is invalid due to its false-alarm inflation). The comparison does not justify using the naive power as a valid benchmark.

The known-rho comparator is not a blind noise estimator and does not show how to infer rho with uncertainty. It is distinct from #12442's predictable conditional-innovation e-process: this slice applies the fixed-horizon Gaussian terminal test to evaluator-known transformed innovations, while #12442 owns its separate sequential e-process and assumptions.

### Updated exact-head identity

After adding the comparator, the branch ran all original and new checks successfully:
- 48 Rust tests across the library and four binaries;
- rustfmt;
- sequential and fixed-terminal Gaussian campaigns;
- five-seed IID stability campaign;
- six-regime stress campaign including the known-rho conditional comparator;
- all four receipt uploads;
- strict Clippy with warnings denied.

- Stress report JSON SHA-256: `807a1c6494994d24f813faeb02fd686888ac85cb4ea52f722a94e0f6c37215ee`
- Stress artifact ZIP SHA-256: `92ca5e7f4ccfd4fd740c0d3f6306b01e35e98e328483e9d9d59cd73bf3f143dd`
- Multi-seed IID artifact ZIP SHA-256: `e2ed0bebad799628e02af4697cf3223e2995d93a998eda9263742b5a3cde6b57`

