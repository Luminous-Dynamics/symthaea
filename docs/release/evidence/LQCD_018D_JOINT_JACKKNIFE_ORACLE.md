# LQCD-018D independent joint blocked-jackknife oracle

## Subject

`scripts/lqcd-flow-scale-joint-jackknife-oracle.py`

Exact executed-subject SHA-256:

`a33a6dac208d87634eb43f18bfb70944100257c51b03c1b861b72035a2b4266b`

The subject is Python-standard-library only and imports no Symthaea/Rust implementation code.

## Statistical contract

Each retained configuration contributes one complete trajectory across the same ordered flow times:

`X_i = [t_1^2 E_i(t_1), ..., t_n^2 E_i(t_n)]`.

A valid blocked jackknife deletes one contiguous block of **whole trajectory rows** at a time. The same configurations are therefore removed at every flow time in a replicate, preserving the cross-flow covariance induced by evaluating the curve on the same Markov-chain configurations.

The subject refuses partial trailing blocks instead of silently dropping configurations. Choosing a scientifically adequate block size remains external: it must be justified against autocorrelation information rather than inferred by this oracle.

For `B` delete-one-block replicates `theta_b`, the frozen standard-error convention is

`SE_JK = sqrt((B-1)/B * sum_b (theta_b - mean(theta))^2)`.

The nonlinear `t0`-like or `w0`-like scale is recomputed from the complete retained mean curve inside every replicate.

## Frozen synthetic trajectory fixture

The subject uses 12 explicit correlated dimensionless trajectories at six flow times, partitioned into four contiguous blocks of three configurations. No RNG is involved.

Full mean curve:

`[0.12, 0.20, 0.27, 0.35, 0.43, 0.51]` up to ordinary binary floating representation.

For caller targets `t0: 0.30` and `w0: 0.32`:

- full-sample `t0_like = 0.33750000000000002`;
- joint blocked-jackknife `t0` SE = `0.0080544204816276263`;
- full-sample `w0_like = 0.63245553203367588`;
- joint blocked-jackknife `w0` SE = `0.005889670897688516`.

The complete replicate vectors and stdout are frozen in `LQCD_018D_JOINT_JACKKNIFE_RESULT.txt`.

## Deliberately invalid negative control

A second procedure deletes a *different* block in each flow-time column. This destroys the physical same-configuration covariance and is explicitly not a valid uncertainty estimator.

On the frozen fixture it produces:

- invalid `t0` SE = `0.005436721294970426`, only `0.67499844431660228` times the correct joint value;
- invalid `w0` SE = `0.021883689599635547`, `3.7156048240698998` times the correct joint value.

The point is not that covariance always raises or always lowers an error bar. The point is that destroying it can move a nonlinear derived uncertainty strongly in **either direction**. A production scale uncertainty must therefore resample the complete correlated flow trajectory.

## Authority boundary

This qualifies joint delete-one-block resampling semantics on a deterministic synthetic fixture. It does **not** establish:

- that block size 3 is physically adequate for any real ensemble;
- equilibrium or stationarity of a Markov chain;
- a physical `t0` or `w0`;
- a production lattice spacing;
- finite-volume or continuum control;
- bootstrap equivalence;
- bias correction beyond the declared ordinary jackknife convention.

A production implementation must independently reproduce the frozen central values, replicate vectors and standard errors. Real evidence must additionally bind the ensemble manifest, configuration ordering, autocorrelation/block-size justification, energy/flow identities and output digests.
