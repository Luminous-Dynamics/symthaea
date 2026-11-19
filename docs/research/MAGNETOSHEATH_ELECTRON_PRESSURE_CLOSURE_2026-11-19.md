# Magnetosheath electron pressure-tensor closure reproduction and frontier benchmark v0.1

**Date:** 2026-11-19  
**Tracking:** [PLASMA-MAGNETOSHEATH-CLOSURE-001 #12923](https://github.com/Luminous-Dynamics/symthaea/issues/12923)  
**Status:** source review and benchmark specification; no reproduction, data acquisition, training or simulation performed

## 1. Why this benchmark matters

Miloshevich et al., *Electron neural closure for turbulent magnetosheath simulations: Energy channels*, Physics of Plasmas 33, 012901 (2026), DOI [10.1063/5.0300009](https://doi.org/10.1063/5.0300009), moves beyond 1D electrostatic Landau damping to a spatially non-local electron pressure-tensor closure in 2D–3V turbulent PIC data.

The paper's reported result makes the next scientific target unusually clear:
- diagonal pressure components: FCNN R² approximately 0.80–0.82;
- off-diagonal pressure components: FCNN R² approximately 0.37–0.41;
- pointwise MLP off-diagonal performance approximately zero or negative;
- predicted heat-flux components are weak (roughly 0.20–0.30 R²).

A useful next model must improve off-diagonal tensor structure and energy-channel fidelity without sacrificing diagonal accuracy, realizability or stability. An improved six-channel MSE alone is not enough.

## 2. Source-faithful paper reproduction

### Physical task
- 2D spatial / 3D velocity (2D–3V), fully kinetic, energy-conserving semi-implicit PIC (ECsim).
- Run A is a high-particle-count 2048×2048 reference (5000 particles/cell).
- Runs B1–B6 use 256 particles/cell with different turbulent initializations.
- The B-run fields are boxcar filtered over 4×4 cells and downsampled from 2048² to 512².
- The reported plasma parameters are a narrow magnetosheath-like range, not a general plasma-domain qualification.

### Source-reported split
- B3/B4/B5/B6: training.
- B2: validation.
- B1: test.
- A: separate high-fidelity test cohort.

Do not pool these cohorts or select hyperparameters using A or B1. The paper's appendix reports a separate robustness split; preserve that as a second, explicitly labelled experiment, not as an alternate final test chosen after observing results.

### Critical source-artifact gap
The publisher's visible data-availability statement identifies only test datasets A and B1 at Zenodo record [10.5281/zenodo.17882782](https://zenodo.org/records/17882782). The paper's training and validation split also requires B2–B6. Therefore, the paper's full train/validation/test split is not yet admitted from the public data statement alone.

Before an exact reproduction:
1. Inventory the versioned Zenodo record, every filename, exact byte count, per-file digest and record-level license.
2. Determine whether B2–B6, raw/filtered forms, split CSVs, normalization statistics and exact preprocessing artifacts are available.
3. Reconcile every training/validation/test cohort against the article's split table.
4. Request missing data/configuration/checkpoints from the authors if necessary.
5. Mark reproduction as BLOCKED_PENDING_SOURCE_ARTIFACTS where any required source component is missing.

Never use B1 or A as training/validation merely to make a runnable experiment.

## 3. Public code baseline and what is actually pinned

Public repository: [georgemilosh/closure](https://github.com/georgemilosh/closure)

- Pinned revision: `826a6ef687a899f9c5aa4cf2436262862feae84c` (2026-08-06).
- Repository LICENSE at that revision: MIT (license file SHA `ede38ed080aab2e53fe7a65b3b645b2144af4e50`).
- Stack: PyTorch / PyTorch Lightning; Python >=3.10. This is an external reproduction reference, not a Symthaea production dependency.
- The repository README contains the 2026 magnetosheath paper citation.
- The source includes a spatial FCNN and pointwise MLP, preprocessing/data-loading code, evaluation tooling and several different plasma experiment configurations.

### Verified feature-ablation definitions

The pinned `examples/optuna/harris_optuna_sweep.py` and `scripts/scaffold_harris_experiments.py` declare:
- `default`: density, B, electron velocity, E;
- `noE`: density, B, electron velocity;
- `noJ`: density, B, E;
- `noJnoE`: density and B only.

The same scripts declare pressure targets Pxx/Pyy/Pzz/Pxy/Pxz/Pyz with log prescaling on diagonal components and arcsinh on signed off-diagonal components; a second task predicts EPx/EPy/EPz with divP enabled.

**Important non-equivalence:** these files are explicitly named Harris Le2GEM15ppc scaffolding/sweeps, and the repository contains multiple experiment families. Their presence verifies the codebase's feature-ablation implementation, not that the file is the exact configuration used for the Miloshevich magnetosheath paper. The paper's exact run/config/data mapping still needs author- or artifact-level reconciliation.

## 4. Required reproduction matrix

Reproduce, without silently changing source conditions:
1. Non-local FCNN pressure-tensor closure.
2. Pointwise MLP baseline.
3. Symbolic anisotropic closure and CGL baseline.
4. Six pressure components separately, not just aggregate loss.
5. Pressure-strain, pressure-dilation, agyrotropy, spectral and conditional structure diagnostics.
6. All source feature ablations that can be matched to the paper's reported runs.
7. Low-particle-count B train / high-particle-count A test transfer.
8. Galilean/frame transformation test, preserving the paper's stated low-speed convention and velocity regime.

For every metric, record whether it is an exact reproduction, an implementation-level match with unresolved details, or a new extension.

## 5. Paradigm-shift hypotheses

### A. Tensor-equivariant closure, not six unrelated scalar regressors

The output is a symmetric pressure tensor. Compare:
- the source FCNN output head;
- symmetry-tied six-component outputs;
- PSD-preserving parameterizations (e.g. Cholesky-like factors);
- rotation-equivariant spatial features and tensor outputs;
- a conservative hybrid with an unconstrained residual head plus a physically admissible base tensor.

Do not force PSD by adding a large isotropic offset. Report the trade-off between component error, smallest eigenvalue, anisotropy and energy-channel error.

### B. Frame-canonicalized, symmetry-aware learning

The paper reports degraded model performance under boosted inputs and recovery after centering electron flow using the spatial mean. Test:
- exact source baseline;
- mean-flow canonicalization;
- fully declared Galilean transform of all applicable inputs/outputs;
- 2D rotation metamorphic evaluation;
- augmentation-only comparator.

The paper's low-speed electromagnetic field convention is not a general Lorentz transform. Do not conflate Galilean and Lorentz invariance or generalize the result outside v≪c and its stated approximation.

### C. Learn the energy transfer, not just tensor pixels

Pressure-strain is a physical transfer channel. Validate:
- total pressure work;
- isotropic pressure-dilation;
- deviatoric pressure-strain;
- scale-filtered pressure-strain spectra;
- conditional Π:D statistics near current sheets, vorticity, strain and coherent structures;
- energy residual against the source solver's conservation convention.

Use the independently implemented Symtropy [pressure tensor audit #6240 / PR #6243](https://github.com/Luminous-Dynamics/symtropy/pull/6243) as a supplied-output diagnostic. This audit does not prove source authenticity or energy conservation of the original PIC run.

### D. Separate physical signal from PIC sampling noise

Compare matched B low-PPC training against A high-PPC test. Stratify by particle count and noise indicators. Test `default`, `noE`, `noJ`, `noJnoE` only after exact source/run mapping. The paper flags cases where the electric-field input may be less helpful under cross-fidelity noise changes.

### E. Generalize across plasma parameters, not just frames

Hold out entire initializations and parameter regimes. Vary beta_i, beta_e, fluctuation amplitude, guide field, mass ratio, turbulence spectrum, grid and boundary conditions. Keep 2D and 3D as separate task scopes until matched 3D kinetic references are admitted.

## 6. Scorecard

**Tensor prediction:** per-component R², MAE, RMSE, normalized error and bias; off-diagonal channels reported individually; spatial correlation and scale-dependent error.

**Solver-operator impact:** compare the discrete div(P) vector using the exact solver stencil, staggering, halo and boundary semantics; report componentwise MAE/RMSE/relative-L2, error spectra, current-sheet/separatrix stratification and conservation residual. A synthetic shared-face periodic reference operator exists in Symtropy [#6274 / PR #6243](https://github.com/Luminous-Dynamics/symtropy/pull/6243), but it is not an ECsim stencil substitute.

**Physical admissibility:** symmetry residual; PSD/minimum eigenvalue; trace, determinant, anisotropy and agyrotropy; pressure-dilation and pressure-strain error.

**Energy/turbulence:** scale-filtered Π:D spectrum; conditional structure statistics; energy-channel balance residual using the exact ECsim convention.

**Robustness:** frame boost, rotations, particle-noise/fidelity, initial-condition holdout, parameter OOD and resolution transfer.

**Economics:** kinetic training-data generation cost, particle count, reference solver hours, GPU-hours, training memory, preprocessing, inference latency and cost per stable qualified domain-time.

No aggregate score may conceal off-diagonal failure or a physical-admissibility violation.

## 7. Prospective improvement targets (not achieved)

The first honest target is exact reproduction of the FCNN/MLP/source-physics table. After baseline variance and denominators are known, explore:
- off-diagonal R² ≥0.60 for each of Pxy/Pxz/Pyz on the same held-out cohort;
- preserve the source's diagonal pressure quality (do not trade away the existing ~0.8 range);
- lower pressure-strain/scale-spectrum error without violating PSD or symmetry;
- zero-shot frame/rotation consistency on a frozen transform cohort;
- improved high-fidelity transfer and full-parameter-regime holdouts;
- lower total cost per qualified simulation, counting data generation and preprocessing.

The ≥0.60 off-diagonal milestone is a proposed research target, not a measured result or a guarantee.

## 8. Evidence state

| Evidence item | Status |
|---|---|
| Paper and DOI | Identified |
| Source article train/val/test split | Extracted |
| Publisher-stated public A/B1 test data | Identified; record-level admission pending |
| B2–B6 train/validation bytes | Not yet admitted |
| Pinned public code revision | Identified |
| Public code license | MIT observed |
| Exact paper run/config mapping | Not yet established |
| Dataset record license / exact digests | Not yet audited |
| Paper reproduction | Not performed |
| New model or better-than-paper result | Not implemented |
| Physical/experimental validation | Not performed |

## 9. Claim ceiling

This document is a source review and reproduction protocol. It does not claim successful reproduction, broad magnetosheath generalization, a 3D closure, a superior model, or experimental/space-weather validation.
