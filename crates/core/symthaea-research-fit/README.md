# symthaea-research-fit

Content-addressed provenance for **what was allowed to influence a fitted artifact**.

A perfect train/evaluation split can still leak if preprocessing or tuning sees evaluation data. Examples include:

- computing global mean/std across all scenes;
- PCA or dimensionality reduction fit on all samples;
- feature selection using held-out labels;
- HDC codebook/prototype construction using evaluation scenes;
- model training on evaluation examples;
- probability calibration on evaluation outcomes;
- threshold/model/hyperparameter selection against evaluation scores;
- learned cloud/change/ROI rules tuned on the final test set.

This crate makes that influence set explicit and binds it to a frozen `symthaea-research-split` manifest.

## Fit vs apply

The central distinction is:

```text
fit artifact
  <- may learn only from roles allowed by FitRolePolicy

apply frozen artifact
  -> may transform Training, Calibration, or Evaluation samples
```

Evaluation data may therefore pass through a standardizer, PCA basis, feature encoder, or trained model **after the artifact is frozen**. It may not influence how that artifact was fitted.

`TransformReceipt` records the application phase separately so applying a frozen artifact to Evaluation cannot be confused with fitting on Evaluation.

## Policies

`FitRolePolicy::TrainingOnly`

- allows Training;
- rejects Calibration;
- rejects Evaluation.

`FitRolePolicy::TrainingAndCalibration`

- allows Training;
- allows Calibration;
- still rejects Evaluation.

The latter is useful for explicitly preregistered calibration or threshold-selection stages. It should not be used merely to blur the distinction between model fitting and model selection.

## Fit stages

The v1 manifest can label an artifact as:

- preprocessing;
- feature selection;
- representation learning;
- model training;
- calibration;
- threshold selection;
- other.

The stage is descriptive provenance; the role policy is the enforced access rule.

## Lineage

A `FitArtifactManifest` binds:

- artifact identity;
- fit stage and role policy;
- exact frozen split-manifest digest;
- implementation digest;
- hyperparameter/configuration digest;
- every sample id allowed to influence fitting;
- each influence sample's content digest and split role;
- fitted output artifact digest;
- fit time;
- content-addressed manifest digest.

Persisted manifests must be revalidated against the exact `ResearchSplitManifest` before use. Internal digest validity alone cannot prove that the referenced split is the authoritative split for a study.

## Sentinel / Planetary Perception use

For a real Wetland Watch contest, typical fitted artifacts might include:

```text
radiometric/statistical normalizer
feature scaling parameters
PCA / spectral basis
HDC prototypes or learned encoder state
statistical model weights
probability calibrator
operating threshold
semantic ROI scheduler parameters
```

Each should declare which frozen Training/Calibration Sentinel units influenced it.

A particularly important rule is that evaluation rasters must not influence global normalization statistics, learned representations, threshold selection, or HDC prototypes before the final evaluation forecast/score is committed.

## What this does not prove

The crate does not prove that:

- the fitting algorithm is scientifically appropriate;
- a model generalizes;
- hyperparameters were selected optimally;
- a Training/Calibration split is adequate;
- implementation digests correspond to trusted software;
- evaluation payloads were inaccessible through some unrelated ambient channel.

It records and enforces the declared fit influence boundary. Process isolation, custody, software attestation, and scientific validity remain separate concerns.

## Required gates

```bash
cargo fmt --all -- --check
cargo check -p symthaea-research-fit --all-targets
cargo test -p symthaea-research-fit
cargo clippy -p symthaea-research-fit --all-targets -- -D warnings
```
