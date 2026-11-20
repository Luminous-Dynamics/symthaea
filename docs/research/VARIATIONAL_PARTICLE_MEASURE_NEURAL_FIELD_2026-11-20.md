# Variational particle-measure neural field operator v0.1

**Date:** 2026-11-20  
**Tracking:** [PLASMA-VARIATIONAL-NEOPIC-001 #12960](https://github.com/Luminous-Dynamics/symthaea/issues/12960)  
**Status:** mathematical benchmark specification; no learned energy functional or plasma simulation implemented

## 1. Motivation

NEOPIC learns electric fields directly from the empirical particle measure. Its paper reports reduced grid-aliasing exposure, but explicitly does not claim exact energy conservation. A high-value alternative is to learn a scalar particle-field energy and derive particle forces by differentiation.

This changes the object being learned:

    direct operator: μ_N -> {E_theta(x_p)}
    variational operator: μ_N -> U_theta(μ_N), with F_p = -∂U_theta/∂x_p

The gradient construction constrains the learned force to be conservative in particle configuration space wherever U_theta is differentiable. It does not alone make a discrete time integrator energy-exact.

## 2. Mathematical contract

Represent a particle state as X={(x_p,q_p,m_p,w_p)}. A learned scalar energy functional U_theta(X) must respect the declared physical symmetries and conventions.

For charged particles, under the pinned energy convention:

    F_p = -∇_{x_p} U_theta(X)
    E_p = F_p / q_p

The code must document sign, species, macro-particle weights and units; the two equations above are not meaningful until those conventions are explicit.

Structural consequences to test:
- conservative force: the configuration-space force Jacobian is symmetric, equivalently its curl vanishes for a smooth scalar U;
- permutation invariance of U implies permutation-equivariance of particle forces;
- common-translation invariance of a closed periodic internal energy implies zero total internal force;
- potential gauge and k=0 handling are fixed by the energy convention, not left to a learned offset.

These results have conditions. External forcing, non-periodic boundaries, non-neutral backgrounds and imposed potentials may break the corresponding symmetries. A neutralising background and particle self-energy/self-force convention must be explicit.

## 3. Time-discrete energy is a separate question

A conservative continuous-time force is not by itself an exactly energy-conserving time-discrete PIC method. Leapfrog/symplectic methods may have bounded or oscillatory energy error but not exact stepwise energy conservation. An exact discrete-energy claim requires a compatible discrete-gradient / average-vector-field / variational time integrator and a registered nonlinear-solver tolerance.

Therefore split the benchmark into:
1. spatial/force structure of U_theta;
2. continuous-time Hamiltonian/trajectory behaviour;
3. time-discrete energy behaviour under each declared integrator;
4. total energy and momentum in the fully coupled particle-field solve.

## 4. Mandatory baselines

- source NEOPIC direct NUFNO;
- scalar energy functional U_theta with gradient-derived forces;
- permutation-invariant pairwise/message-passing Hamiltonian model;
- grid FNO with field from a consistent potential gradient;
- Particle-In-Fourier / Fourier PIC;
- CIC PIC;
- Barnes-Hut / tree method where applicable;
- exact direct-sum reference for small particle counts;
- an energy-/charge-conserving implicit PIC reference for matched regimes.

Do not call the variational method a feature of original NEOPIC. It is a new method.

## 5. Frozen known-answer suite

### Small-N exact tests

1. N=1 self-force with the declared neutralising background and boundary convention.
2. N=2 like-charge interaction and action-reaction pair.
3. N=2 unlike-charge interaction and correct sign.
4. Small periodic N exact/reference energy and force.
5. Particle permutation invariance.
6. Arbitrary common periodic translation invariance.
7. Zero-mode / gauge convention.
8. Singular/coincident-particle handling under a declared softening or finite-shape model; no hidden epsilon.

### Differential structure

- finite-difference ∂F_i/∂x_j against ∂F_j/∂x_i;
- configuration-space curl residual;
- gradient-force agreement under multiple step sizes and precisions;
- directional derivative check for U_theta;
- no accidental detach/stop-gradient in the field-to-force derivative path.

### Kinetic tests

- 1D weak and strong Landau damping;
- 2D two-stream;
- N_train × N_inference particle-count transfer, including N_inference < N_train;
- independent teacher runs, seeds and initialisation families (IID vs quiet-start as separate cohorts);
- full closed-loop trajectories.

## 6. Physics scorecard

**Force / field**
- particlewise E and F errors;
- configuration-space curl and Hessian-symmetry residuals;
- translation/permutation residuals;
- self-force and pair action-reaction residuals;
- exact discrete Poisson residual.

**Conservation**
- total kinetic + potential energy drift;
- momentum / net-internal-force residual;
- particle-field work exchange;
- field energy and total charge;
- separate contributions from force approximation and time integration.

**Kinetic behaviour**
- Landau damping rate and field-energy envelope;
- two-stream growth, saturation time and peak;
- particle position/velocity error and distribution distance;
- OOD/selective risk/abstention.

**Performance**
- energy forward cost;
- reverse-mode force gradient cost;
- higher derivatives/JVP/Hessian cost for implicit solvers;
- full particle push + field/energy + communication;
- memory and scaling with N_p;
- accuracy at matched wall time, accelerator-hours and energy.

## 7. Prospective gates (not results)

1. Reproduce original NEOPIC before extending it.
2. Pass exact small-N tests before kinetic training.
3. Force Jacobian symmetry and conditional translation/permutation identities must satisfy registered numerical tolerances.
4. No self-force in the frozen single-particle negative control.
5. Improve energy/momentum behaviour over direct NEOPIC at matched field/trajectory accuracy.
6. Do not infer exact discrete energy conservation from a gradient-derived force alone.
7. For exact discrete energy, use a compatible discrete-gradient/AVF/variational integrator and verify energy to the declared nonlinear-solver tolerance.
8. Only claim superiority over PIF or energy-conserving PIC on matched accuracy-cost-physics cohorts.

## 8. Relevant prior art

- NEOPIC (Muralikrishnan et al., arXiv:2610.40528, 2026): direct empirical-measure learned electric field; the source says the learned operator is not exactly energy conserving.
- On Variational Fourier Particle Methods, *Journal of Scientific Computing* (2024), DOI 10.1007/s10915-024-02708-w: PIF is derived from a variational principle; translational invariance is linked to momentum conservation.
- Li, Xu & Yang, *JCP* 495 (2023), article 112529, DOI 10.1016/j.jcp.2023.112529: energy-conserving implicit Fourier PIC for Vlasov-Ampere with exact divergence/curl-free constraints and an asymptotic-preserving preconditioner.
- Kormann & Sonnendrücker, *JCP* (2021), article 109890, DOI 10.1016/j.jcp.2020.109890: compatible finite-element / Poisson-splitting Vlasov-Maxwell PIC with discrete energy and Gauss preservation.

These are distinct formulations and regimes. Their published results are not head-to-head NEOPIC comparisons.

## 9. Admission / claim state

- NEOPIC source: identified; datasets/checkpoints request required.
- Variational Fourier PIC prior art: identified.
- Energy-conserving Vlasov-Ampere PIC: identified.
- Exact source configurations and data: not admitted.
- Variational learned energy model: not implemented.
- Force-gradient tests on a learned candidate: not performed.
- Discrete energy comparison: not performed.
- SOTA superiority: not claimed.

## 10. Claim ceiling

This is a design and benchmark protocol only. It does not implement or train a variational NEOPIC, prove exact discrete energy conservation, or qualify a plasma simulation or controller.
