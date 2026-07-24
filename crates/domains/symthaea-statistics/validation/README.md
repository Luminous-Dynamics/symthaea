# Independent validation assets

The Rust crate has no runtime or development dependencies. Files in this directory are optional evidence generators executed outside Cargo.

- `generate_v0_4_references.py` / `v0_4_reference_results.json` cover multivariate models, distributions, power, and sequential evidence.
- `generate_v0_5_references.py` / `v0_5_reference_results.json` cover predictive discrimination, Poisson and Cox regression, HC3 covariance, PCA, survival, meta-analysis, and autoregression.

To regenerate v0.5 references in an environment with NumPy, SciPy, scikit-learn, and statsmodels:

```sh
./scripts/verify_v0_5_references.sh
```

A changed result is evidence to investigate, not a snapshot to update automatically.
