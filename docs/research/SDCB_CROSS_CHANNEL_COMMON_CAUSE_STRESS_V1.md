# SDCB cross-channel common-cause dependence stress v1

**Status:** standalone Rust implementation and exact-head Rust 1.96 workflow authored. Formatting, seven negative/metamorphic controls and the 24-condition report are pending. No empirical result is claimed until the exact committed source passes and its artifact is inspected.

## Research question

Can conformal block calibration fail when every sensor channel preserves its one-step marginal distribution and within-channel temporal autocorrelation, but cross-channel dependence changes?

This experiment uses two stationary AR(1) channels:

`X_t = rho X_(t-1) + sqrt(1-rho^2) eps_t`

`Y_t = rho Y_(t-1) + sqrt(1-rho^2) (gamma eps_t + sqrt(1-gamma^2) eta_t)`

where `eps_t` and `eta_t` are independent standard normal innovations. The same `gamma` defines the initial stationary cross-channel correlation. Each channel individually has a standard Gaussian AR(1) marginal with the same fixed `rho=0.80`; changing `gamma` changes only cross-channel dependence.

The heavy-tail arm multiplies both channels and every tick in one independent path by a shared positive `sqrt(3/V)`, where `V~chi-square(5)`. Each channel then has a standardized Student-t(5) marginal with unit variance. The shared scale is a common latent factor across the two channels and the whole block.

## Score and experiment

The block score is the maximum absolute residual across both channels and all ticks:

`S = max_{t,c in {1,2}} |X_(c,t)|`

The experiment compares calibration and test values of `gamma`:
- low correlation: `gamma=0.05`;
- high correlation: `gamma=0.95`;
- both shift directions and no-change controls;
- block lengths 1, 8 and 32;
- 39 calibration blocks;
- nominal alarm rate `alpha=0.10`;
- Gaussian and standardized multivariate Student-t(5) common-scale marginals;
- 2,500 trials per condition.

This is 24 conditions × 2,500 trials = 60,000 simulations. The full CSV includes exact seeds, empirical alarm rates and Wilson 95% intervals. A manifest binds the report to the source commit and Rust toolchain.

## Expected direction (hypothesis, not result)

For a maximum across two channels, low cross-channel correlation generally gives more independent opportunities for an extreme value than high correlation. Thus:
- calibration at low `gamma`, test at high `gamma`: expected fewer alarms than nominal (over-conservative);
- calibration at high `gamma`, test at low `gamma`: expected excess alarms (undercoverage).

This direction must be confirmed separately for Gaussian and t5 common-scale marginals. The tests require the Wilson interval to lie below or above nominal in the respective arms; they do not merely assert a point-estimate difference.

## Controls

1. **Gaussian marginal invariance:** 50,000 seeded paths check that each channel's variance remains near one across low/high `gamma`.
2. **Student-t5 marginal invariance:** 50,000 shared-scale paths check unit variance for both channels.
3. **Cross-correlation manipulation:** 50,000 seeded paths per setting check that observed contemporaneous correlation matches the declared `gamma`.
4. **Gaussian regime-shift control:** both shift directions must have Wilson intervals separated from nominal.
5. **Student-t5 common-scale control:** same two directions, separately tested.
6. **Replay control:** identical source, seed and parameters produce identical report structures.
7. **Grid receipt control:** exactly 24 finite rows with complete parameter and uncertainty fields.

## Why this matters

A per-channel diagnostic can pass while a multichannel block score changes distribution. The relevant calibration object is the joint score-generating process, not just each channel's marginal distribution or univariate autocorrelation.

This exposes a concrete common-cause blind spot for sensor networks, distributed systems and multi-agent evidence streams: channels can remain individually well behaved while their joint extremes become more or less likely.

## Limitations and next work

- Two channels, one contemporaneous correlation parameter and stationary Gaussian AR(1) temporal dynamics only.
- The Student-t(5) arm is a shared-scale multivariate-t construction, not arbitrary heavy-tailed dependence.
- No cross-channel covariance rotation, lagged cross-correlation, nonlinear dependence, missingness, drift or adversarial sensor corruption.
- Synthetic simulation does not establish real sensor calibration or a general coverage theorem.
- 2,500 trials per condition are exploratory; confirmatory work needs independent seeds and 50,000–100,000+ trials per condition.
- The next step is a covariance-rotation and latent-factor grid with 3–8 channels, then held-out real multivariate telemetry with source lineage and common-cause annotations.
- Outside the empirically validated envelope, the assessor must return NotQualified or Inconclusive, not infer validity from univariate marginal fit.

The paradigm shift is to qualify the **joint dependence envelope of the decision score**, not just the marginal distribution of each input stream.
