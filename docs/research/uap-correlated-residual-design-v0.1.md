# Selected-set correlated residual covariance design

Date: 2026-11-09  
Status: implementation and synthetic qualification in progress  
Parent: [UAP-OED-003 #11237](https://github.com/Luminous-Dynamics/symthaea/issues/11237)  
Predecessor: [UAP-OED-002 #11235](https://github.com/Luminous-Dynamics/symthaea/issues/11235)  
Programme: [CTGP PR #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## Scientific distinction

There are two mathematically distinct ways to represent shared uncertainty:

1. **Shared latent nuisance state.** A common clock offset, boresight error or platform-pose bias is estimated as part of the state. Conditional on this state, measurement residuals may be independent. The current additive information form can represent this model if each observation Jacobian differentiates against the same shared parameter.
2. **Correlated residual noise.** Common-mode effects remain in the residuals after conditioning on the declared state. Their covariance matrix has off-diagonal terms. The chosen measurement set changes the covariance principal submatrix, so per-sensor pre-whitening and additive JᵀJ is not generally valid.

The code keeps these models separate. [UAP-OED-002 #11235](https://github.com/Luminous-Dynamics/symthaea/issues/11235) implements target-marginal design for explicit nuisance states; this work implements exact selected-set whitening for a declared correlated residual covariance.

## Model

For one scenario, the full candidate pool has a fixed ordering, Jacobian J and positive-definite residual covariance Sigma. A candidate action selects a subset of row indices. For the union I(A) selected by action set A:

`Sigma_A = Sigma[I(A), I(A)]`

`J_A = J[I(A), :]`

Factor the selected covariance:

`Sigma_A = L_A L_A^T`

Jointly whiten:

`J_white = L_A^{-1} J_A`

and compute:

`F_data(A) = J_white^T J_white = J_A^T Sigma_A^{-1} J_A`.

Full posterior precision is `Lambda(A) = Lambda0 + F_data(A)`. The target block is marginalized using the Schur complement from UAP-OED-002. Both target-marginal EIG and full joint-state EIG remain visible, along with data-only rank and weak directions.

The selected covariance is rebuilt for each candidate subset. This is intentionally a small exact reference; it does not yet attempt rank-one, Sherman–Morrison–Woodbury or sparse covariance updates.

## Current implementation

In `symthaea-uap-observatory::correlated_design`:

- `CorrelatedDesignScenario` carries the complete candidate-pool raw Jacobian and full residual covariance.
- `CorrelatedExperimentOption` selects measurement indices per scenario, with cost and mutual-exclusion group.
- The selected principal covariance submatrix is factored with Cholesky; the selected Jacobian is forward-whitened.
- The posterior precision and data-only singular-spectrum diagnostic use the whitened rows.
- The same target-marginal Schur objective, joint EIG and report type as the independent-row method are returned.
- Duplicate observation indices within an option or a selected set are rejected.
- Covariance shape, symmetry, finiteness and positive definiteness are validated.
- Greedy selection and an exact exhaustive subset oracle (up to 20 options) are provided.
- The dense candidate pool is capped at 512 observations.

## Synthetic benchmark controls

The reference example uses two scalar-target observations with identical unit Jacobian rows and a common-mode correlation coefficient rho. The Fisher information is analytically:

`I(rho) = 2 / (1 + rho)`

and, with unit prior precision, target EIG is:

`EIG(rho) = log(1 + 2 / (1 + rho))`.

For rho = 0, this agrees with the independent-noise design exactly. As rho approaches one, repeated measurements increasingly duplicate the same error mode and their marginal information saturates. The independent-noise assumption continues to credit both readings as independent and therefore overstates information.

Unit tests also cover:
- diagonal covariance equivalence with the current independent-row API;
- exact selected principal submatrix behavior;
- measurement-order permutation invariance;
- duplicate selected-observation rejection;
- invalid singular covariance rejection;
- greedy/exhaustive agreement on a small design.

## Current limitations

- The implementation assumes a known covariance matrix. It does not estimate covariance from instrument records.
- The current pool and dense Cholesky reference are deliberately small; no production-scale sparse or low-rank updates are claimed.
- Correlated covariance is assumed positive definite. Singular common-mode models must be represented with a justified latent-state decomposition or handled by a future rank-aware covariance method, not a pseudoinverse.
- Scenario-specific covariance is supported; time-varying or state-dependent covariance is not yet supported.
- This is a local linear-Gaussian design. It does not solve nonlinear target dynamics, visibility, multipath, atmospheric propagation or asynchronous data association.

## Prior art and fair comparison

The method belongs to an established correlated-observation optimal-design literature:
- López-Fidalgo & Wong, *Optimal Designs for Correlated Data*, Annual Review of Statistics and Its Application 13 (2026), 245–270, DOI 10.1146/annurev-statistics-042324-012947.
- Uciński, *A-optimal sensor location subject to correlated observations via a fast exchange algorithm*, Measurement 254 (2025), 117794, DOI 10.1016/j.measurement.2025.117794.
- Liu et al., *Sensor Selection for Estimation with Correlated Measurement Noise*, IEEE Transactions on Signal Processing 64(13), 3509–3522 (2016), DOI 10.1109/TSP.2016.2550005.

The next performance comparison is against the fast-exchange / Sherman–Morrison–Woodbury family on matched small and medium problems. The current implementation is an exact dense reference and a correctness oracle, not a claim of superior throughput. Publish wall time, peak memory, correlation strength, selected-set size and error together.

## Claim ceiling

A synthetic pass validates the declared linear-Gaussian selected-set information calculation. It does not establish real instrument covariance, field accuracy, the identity of a UAP, or an exotic physical mechanism. Field use requires empirical covariance estimation, calibration-chain provenance and held-out coverage evaluation.
