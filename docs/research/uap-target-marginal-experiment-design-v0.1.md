# Target-marginal experimental design with shared calibration nuisance states

Date: 2026-11-09  
Status: implementation and synthetic qualification in progress  
Parent: [UAP-OED-002 #11235](https://github.com/Luminous-Dynamics/symthaea/issues/11235)  
Foundation: [UAP-OED-001 #11050](https://github.com/Luminous-Dynamics/symthaea/issues/11050)  
Programme: [CTGP PR #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## Why this increment matters

The current budgeted experiment-design kernel chooses actions by marginal expected log-determinant information gain per unit cost. That is appropriate only when the objective is information about the **entire state**.

A real sensor-design problem often has:
- parameters of interest: target position, velocity, acceleration or physical model parameters;
- nuisance parameters: sensor clock offsets/drifts, platform pose, boresight, range bias, common GNSS/INS errors and environmental corrections.

A design can sharply improve nuisance calibration without materially improving target localization. Conversely, a calibration-only observation can improve target localization when a nuisance parameter is correlated with the target state. The objective must distinguish those cases.

This is not a claim that Schur-complement optimal design is new. It is a correction to the current Symthaea objective and its test contract.

## State and objective

Partition the parameter state explicitly:

`theta = [x; eta]`

where `x` contains the leading `target_parameter_count` parameters of interest and `eta` contains nuisance parameters. For a scenario s and selected design A, the full precision is

`Lambda_s(A) = Lambda0_s + J_s(A)^T J_s(A)`

provided the residual rows have been whitened using the correct joint covariance conditional on the explicit state.

Partition the precision:

`Lambda = [[A, B], [B^T, C]]`

The target marginal precision is the Schur complement

`Lambda_target,marginal = A - B C^{-1} B^T`.

The target utility is

`EIG_target = logdet(Lambda_target,marginal,posterior) - logdet(Lambda_target,marginal,prior)`.

The full joint-state utility remains separately visible:

`EIG_joint = logdet(Lambda_posterior) - logdet(Lambda_prior)`.

This is equivalent to computing the reduction in differential entropy of the target marginal Gaussian. It is not the same quantity as joint-state entropy reduction.

### Numerical requirements

- Compute `C^{-1} B^T` using Cholesky triangular solves; never form a dense inverse.
- Do not use a pseudo-inverse to conceal unobservable nuisance directions.
- Reject singular or indefinite nuisance precision blocks.
- Keep target/nuisance cross-covariance; do not replace the prior by a diagonal approximation.
- Report full-state data-only rank and null directions independently of prior-regularized utility.
- If no nuisance parameters are present, target EIG must equal joint EIG.
- Do not claim generic submodular or greedy approximation guarantees for robust/nonlinear/correlated variants without proving the relevant assumptions.

## Correlated measurement boundary

There are two different models:

**Shared latent calibration state.** For example, every sensor reading depends on the same clock offset `delta_t`. Include that one clock parameter once in the joint state and let every relevant measurement row have a derivative with respect to the same state component. Conditional on that state, residual errors may be independent. The current additive information form can represent this case.

**Shared correlated residual noise.** For example, all sensors have a common random timebase error that is not otherwise represented as an estimable state. The residual covariance has off-diagonal terms. Independently whitening each sensor and summing its `J^T J` contribution is incorrect. This requires joint whitening using the selected-set covariance (or an equivalent explicit latent-variable model).

The current API supports the first case only. Arbitrary selected-set covariance and joint whitening remain a separate next step. The implementation must not describe the first as solving the second.

## Current implementation

In `symthaea-uap-observatory::budgeted_design`:

- `DesignScenario.target_parameter_count` explicitly partitions interest and nuisance state blocks.
- `ScenarioDesignScore.information_gain_nats` now means target-marginal EIG.
- `joint_information_gain_nats` exposes full-state EIG.
- `BudgetedDesignReport` reports expected and worst-case target and joint EIG separately.
- A Cholesky-based Schur-complement helper computes target marginal precision; triangular solves are used, not a pseudoinverse.
- The budgeted design example declares target XYZ as the three parameters of interest and clock offset as a nuisance parameter, and prints both utilities.
- The CTGP workflow runs that example and checks that target and joint EIG metrics are emitted.

## Synthetic known-answer controls

The unit-test contract covers:

1. **No nuisance parameters:** target EIG equals joint EIG.
2. **Uncoupled nuisance-only action:** joint EIG is positive while target EIG is zero.
3. **Coupled calibration observation:** adding a clock-calibration observation increases target EIG when target and clock are correlated by the measurement.
4. **Correlated prior:** a nuisance-only observation increases target EIG when the prior precision contains target/nuisance coupling.
5. **Shared-clock negative control:** two sensor observations sharing one clock nuisance must not be modeled as two independent clock nuisance variables. The deliberately mis-specified independent-clock model yields a larger target EIG and demonstrates the direction of overconfidence.
6. **Schur determinant identity:** deterministic SPD blocks verify `logdet(Schur) = logdet(full precision) - logdet(nuisance precision)`.
7. **Singular nuisance block:** fail closed rather than applying an arbitrary pseudoinverse.
8. **Invalid target/nuisance partition:** reject zero or out-of-range target block sizes.

These are mathematical and synthetic controls only. They do not establish field accuracy or real sensor calibration performance.

## Industry and research comparison

Correlated-observation optimal design is established prior art:

- López-Fidalgo & Wong, *Optimal Designs for Correlated Data*, Annual Review of Statistics and Its Application 13 (2026), 245–270, DOI: 10.1146/annurev-statistics-042324-012947.
- Uciński, *A-optimal sensor location subject to correlated observations via a fast exchange algorithm*, Measurement 254 (2025), 117794, DOI: 10.1016/j.measurement.2025.117794.
- Liu et al., *Sensor Selection for Estimation with Correlated Measurement Noise*, IEEE Transactions on Signal Processing 64(13), 3509–3522 (2016), DOI: 10.1109/TSP.2016.2550005.
- *Optimal experimental design for parameter estimation in the presence of observation noise*, Mathematical Biosciences 392 (2026), 109571, DOI: 10.1016/j.mbs.2025.109571.
- Ramon et al., *Multifidelity sensor placement in Bayesian state estimation problems*, Applied Mathematics for Modern Challenges (2026-07-15), DOI: 10.3934/ammc.2026006.

The next fair comparisons are:
- joint-state D-optimality;
- target-marginal D-optimality;
- target-only design that ignores nuisance uncertainty (an intentionally optimistic baseline);
- explicit shared latent nuisance models;
- full correlated residual covariance design on small problems;
- greedy cost-benefit selection, random selection and exhaustive subset oracle.

All methods must receive the same candidate actions, noise information, prior, budgets, scenario weights and mutual-exclusion constraints.

## Next benchmark gates

- Target and joint EIG values agree in the no-nuisance limit.
- Nuisance-only measurements produce zero target EIG when uncoupled and positive target EIG when prior/measurement coupling justifies it.
- Shared clock and pose controls expose overconfidence from incorrect independent-error assumptions.
- The marginal precision agrees with independent dense block-matrix references.
- The selected design never exceeds cost or mutual-exclusion constraints.
- Greedy/exhaustive gaps are reported on small cases.
- Held-out scenarios test sensitivity to target geometry, clock strength, calibration error and correlated priors.
- Empirical target posterior coverage and NEES are reported before any sensor-field claim.

## Claim ceiling

A pass establishes correct target-marginal design calculations for the declared linear-Gaussian model. It does not demonstrate the accuracy of a real UAP track, establish a physical anomaly, or support an exotic propulsion interpretation. If target information is insufficient after nuisance marginalization, the proper output remains unresolved.
