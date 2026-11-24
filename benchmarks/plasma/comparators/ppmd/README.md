# PPMD source admission — v0.1

## Paper identity

Guo & Xiao, *Parametric Probabilistic Manifold Decomposition for Nonlinear Model Reduction*, SIAM Journal on Scientific Computing 48(6), December 2026, A3567–A3594, DOI [10.1137/26M185758X](https://doi.org/10.1137/26M185758X), published online 23 November 2026.

The paper reports convergence analysis for the linear principal subspace, nonlinear manifold recovery and kernel-ridge latent learning, with demonstrations on flow past a cylinder and backward-facing step. It compares against POD+GPR and reports improved generalization on its stated flow parameter tests. These are paper-reported fluid-flow results, not plasma results.

## Exact public source snapshot inspected

Repository: [apprenticeGuo/Parametric-Probabilistic-Manifold-Decomposition-for-Nonlinear-Model-Reduction](https://github.com/apprenticeGuo/Parametric-Probabilistic-Manifold-Decomposition-for-Nonlinear-Model-Reduction)

- Branch: main
- Commit: `056e4879a033cfecb44d8fb11983125b99a562eb`
- Commit date: 21 December 2025
- Tree: `e4049c40e37409709ead86a44f30ce4afd0e90af`
- Files present: `PPMD.ipynb`, `PPMD.py`, `README.md` only.

The inspected `PPMD.py` hard-codes Windows dataset roots:
- `D:\\parametric PMD\\Flow around a cylinder`
- `D:\\parametric PMD\\backward facing step`

It expects VTU input files not included in the inspected tree. The tree has no environment lock / dependency manifest. Imports include NumPy, pandas, meshio, PyTorch, TensorFlow v1 compatibility, scikit-learn, pydiffmap and Matplotlib.

## Admission disposition

**NOT_REPRODUCED.** This is a source-admission audit, not an accusation that the paper is irreproducible. The paper reports a SIAM reproducibility badge, and the current public GitHub snapshot may not be the exact badge-linked archive or supplementary package. We must obtain that exact archive, data, environment and configurations before replaying results.

Do not:
- treat the paper's flow results as plasma accuracy;
- claim the repository snapshot independently reproduces the paper;
- quote a speedup not measured under matched hardware and full-cost accounting;
- use PPMD as an already-qualified plasma baseline.

## Required reproduction sequence

1. Resolve the exact paper / badge-linked source archive and pin commit + file digests.
2. Obtain VTU data and checksums; record meshes and parameter values.
3. Freeze Python package versions, CUDA/runtime/hardware and deterministic settings.
4. Reproduce the paper's train/test parameter split, window lengths, POD dimensions, Isomap/manifold settings and KRR hyperparameters.
5. Reproduce POD+GPR and PPMD on the original cylinder and backward-facing-step cases.
6. Record interpolation and extrapolation separately; hold out whole parameter values, not random snapshots.
7. Add independent uncertainty calibration and active acquisition ablations.
8. Only then port a matched probabilistic-manifold baseline to collisionless 1D1V Vlasov–Poisson; use separate branches for trapping/echo regimes and a separate cohort for collisional high-field VPFP.

## Plasma research hypothesis

The Symthaea extension is **Response-Conditioned Probabilistic Kinetic Manifold Decomposition (RPKMD)** (#13501). The manifold must be calibrated against future intervention-response error and branch identity, not just latent reconstruction variance. KIRD must count the decoder, covariance, chart index, branch identity and residual memory. Near a branch switch, an uncalibrated or multimodal posterior must yield UNKNOWN rather than smooth interpolation across physically distinct futures.
