# Independent validation assets

The Rust crate has no runtime or development dependencies. Files in this directory are optional evidence generators executed outside Cargo.

- `generate_v0_4_references.py` / `v0_4_reference_results.json` cover multivariate models, distributions, power, and sequential evidence.
- `generate_v0_5_references.py` / `v0_5_reference_results.json` cover predictive discrimination, Poisson and Cox regression, HC3 covariance, PCA, survival, meta-analysis, and autoregression.
- `generate_v0_6_references.py` / `v0_6_reference_results.json` cover QR/ridge models, influence diagnostics, cluster/HAC covariance, isotonic and conformal calibration, AIPW/DiD, exact randomization, KDE/DKW, residual diagnostics, and BCa bootstrap intervals.

To regenerate the v0.6 reference set in an environment with NumPy, SciPy, scikit-learn, and statsmodels:

```sh
./scripts/verify_v0_6_references.sh
```

A changed result is evidence to investigate, not a snapshot to update automatically. The current reference set is v0.8 and is verified separately with `../scripts/verify_v0_8_references.sh`.

## v0.7

Run `../scripts/verify_v0_7_references.sh` to regenerate and compare the retained v0.7 SciPy, statsmodels, and scikit-learn reference snapshot.

## v0.8

Run `../scripts/verify_v0_8_references.sh` to regenerate and compare the retained v0.8 NumPy, SciPy, and scikit-learn snapshot covering categorical probability, reliability/agreement, circular statistics, local-level state estimation, extreme-value analysis, and Hotelling inference.

## v1.0

Run `../scripts/verify_v1_0_references.sh` to regenerate the retained NumPy, SciPy, ArviZ, and statsmodels snapshot covering exact discrete inference, compositional geometry, Deming regression, MCMC diagnostics, multivariate-normal conditioning, EDF goodness-of-fit statistics, and Qn scale.

## v1.1 references

`generate_v1_1_references.py` records independent landmarks for distance
statistics, Gaussian-kernel MMD, PERMANOVA, Kruskal-Wallis, generalized-DL
meta-regression, OAS covariance shrinkage, normal-inverse-gamma updating,
multiclass scoring, and multiplicity corrections.
