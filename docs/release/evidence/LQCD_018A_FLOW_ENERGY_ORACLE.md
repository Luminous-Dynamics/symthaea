# LQCD-018A independent flowed-energy-density / scale-extraction oracle

## Subject

`scripts/lqcd-flow-energy-scale-oracle.py`

Exact executed-subject SHA-256:

`35fd8a3c87244077a6e77e316e97848a57f9e444c6fd9040976ec11d1297a1fd`

The subject is Python-standard-library only and imports no Symthaea/Rust implementation code.

## Energy-density convention

The frozen convention writes the Hermitian traceless field-strength matrix as

`F_munu = F_munu^a T^a`, with `T^a=lambda^a/2`

and `Tr(T^a T^b)=delta_ab/2`.

The local Euclidean gauge energy density is represented as

`E = sum_{mu<nu} Tr(F_munu F_munu) = (1/4) F_munu^a F_munu^a`,

where the repeated continuum-style `mu,nu` expression counts both antisymmetric orientations while the explicit lattice sum uses `mu<nu` once.

### Executed normalization/gauge-invariance fixture

Six explicit Hermitian traceless field-strength matrices are constructed from fixed Gell-Mann-basis coefficient vectors. The subject verifies:

- matrix-trace energy equals the coefficient-space analytic value exactly within `2e-16`;
- zero field strength gives exactly zero energy;
- simultaneous SU(3) conjugation of all field strengths preserves the energy to floating precision.

Frozen stdout:

```text
ok
energy_density=0.039850000000000003
analytic_energy_density=0.039850000000000003
gauge_transformed_energy_density=0.03985000000000001
t0_like_synthetic_crossing=0.34999999999999998
w0_like_synthetic_crossing=0.63245553203367588
```

The complete stdout is preserved in `LQCD_018A_FLOW_ENERGY_RESULT.txt`.

## Scale-extraction algebra

This oracle deliberately separates observable semantics from physical scale setting.

For an externally supplied **ensemble-mean** energy curve it forms

`F(t)=t^2 <E(t)>`.

A `t0`-like estimator finds one unique crossing of a caller-supplied target and linearly interpolates across a strict sign-changing bracket. An exact target hit at exactly one sampled flow time is returned directly rather than being double-counted through its two neighboring intervals. Repeated exact hits or multiple strict brackets fail closed as ambiguous.

A `w0`-like estimator uses the centered finite-difference response

`t d/dt [t^2 <E(t)>]`

on interior samples, finds a unique caller-supplied crossing in `t`, and returns `sqrt(t_cross)`.

The synthetic tests are algebra fixtures only:

- a constructed ensemble-mean curve gives `t0_like=0.35`;
- a constructed linear `F(t)=0.02+0.8t` curve with target `0.32` gives `w0_like=sqrt(0.4)`;
- an exact sampled crossing is accepted once;
- a curve with multiple target crossings fails closed instead of silently choosing one branch.

No conventional target such as `0.3` is encoded as universal authority by the implementation.

## Scientific boundary

This evidence does **not** establish:

- a physical `t0` or `w0`;
- a lattice spacing;
- an ensemble mean from Monte Carlo data;
- a production interpolation/fit systematic;
- a continuum extrapolation;
- finite-volume control;
- a particular production flow time or RK3 step size.

Physical scale setting requires qualified equilibrium ensembles, a declared energy-density discretization, covariance/autocorrelation treatment, interpolation/fit-systematic evidence, and cross-lattice continuum analysis.

## Literature boundary

At positive flow time the Wilson/gradient flow provides a smooth gauge field on which local gauge-invariant observables such as the action/energy density are well defined. Standard reference scales are constructed from the ensemble mean `t^2<E(t)>` and, for `w0`, its flow-time derivative. This oracle qualifies the algebraic contract only; literature conventions remain external scientific inputs rather than hard-coded verdicts.
