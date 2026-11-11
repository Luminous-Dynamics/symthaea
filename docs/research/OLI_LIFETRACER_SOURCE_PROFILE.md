# OLI LifeTracer source profile v0

**Status:** source code and metadata pinned; external data not materialized; reproduction not run  
**Related:** [OLI-MOLECULAR-001 #11487](https://github.com/Luminous-Dynamics/symthaea/issues/11487)

## Why this source profile matters

LifeTracer is a useful real-data starting point because the paper, archived code release, dataset records and processed feature exports are public. The paper reports more than 87% accuracy for classifying the 18-sample panel as meteorite or terrestrial using GC×GC-HRTOF-MS-derived features.

That task is narrower than biological life detection. Its labels are sample-provenance proxies: 8 meteorites versus 10 terrestrial geological samples. They must not be silently rewritten as universal life/non-life truth.

The paper's exact code release is linked to Zenodo DOI `10.5281/zenodo.17050812` and states MIT licensing. Separate Hugging Face data cards also report MIT metadata for the meteorite, soil and processed-data datasets. Record rights separately per artifact and preserve the paper's attribution.

## Source-pinned profile

Machine-readable manifest: `benchmarks/oli/lifetracer-source-profile-v0.json`

Pinned source commit:
`c0312eb36c27df72a501c33246ac3c70c8aec38f`

The manifest stores blob SHAs for labels, config, notebook, evaluation, AUC evaluation, feature alignment, TII alignment, and source license files. The 18 sample labels and names are copied from the pinned `data/labels.csv` in the source repository.

Source-paper classifier configuration:
- Logistic regression with L2 penalty and liblinear solver.
- Peak settings: lambda1=5, lambda2=100.
- Retention-time thresholds: RT1=50, RT2=0.8.
- C grid: 1e-4 through 1e4 (nine values).
- Final paper-results notebook demonstrates C=0.1.

The evaluation seed law selects ten values from 1..1000 without replacement after `numpy.random.seed(42)`. The resolved seed list is recorded in the manifest.

## Critical split distinction

The repository has two different evaluation implementations:

| Path | Outer split | Inner validation |
|---|---|---|
| Accuracy | shuffled 9-fold KFold | Leave-one-training-sample-out |
| AUC | shuffled 6-fold StratifiedKFold | 5-fold StratifiedKFold |

The outer test fold sizes differ (2 samples for accuracy, 3 for AUC). The score streams are not one common evaluation.

## Critical preprocessing distinction: transductive versus inductive

I inspected the source at the pinned commit.

1. `TII_alignment.align` first collects the retention-time coordinate values from every sample and constructs shared aligned RT1/RT2 axes.
2. `RetentionTimesAlignment.align_retention_times` loads each mass-to-charge peak table, clusters peaks from the full sample list into a common feature vocabulary, and constructs the matrix for all samples.
3. `evaluation.py` and `evaluation_auc.py` load that full feature matrix before creating their outer train/test folds.

This is a **transductive feature-space evaluation**: test labels are held out from classifier fitting, but test-sample measurements contribute to the shared time axes and feature vocabulary before splitting. That is not automatically invalid for a transductive task, but it is not the same as prospective inference on a new sample unavailable during feature construction.

OLI must preserve the source-faithful protocol and add a separate strictly inductive track:

- For each outer fold, fit RT axes, peak clusters, alignment and feature vocabulary using only the training samples.
- Freeze the learned transform before processing the test fold.
- Map test observations onto the frozen vocabulary.
- Retain out-of-vocabulary peak statistics and unsupported-mass reports instead of silently expanding the vocabulary.
- Compare source-transductive performance against inductive performance on exactly the same outer-fold sample identities.

This is a more meaningful generalization challenge than simply changing the classifier.

## Two further audit flags

**Raw scan schema audit.** The paper says seven columns, then names row index plus seven measurements. The Hugging Face viewer exposes seven data columns—Spectrum, RT1, RT2, TOF, m/z, Area and Resolution—and may display the row index only as a table index. That likely explains the count mismatch, but the exact raw CSV header must still be inspected before implementing a parser.

**Small sample power.** Eighteen samples are insufficient for a strong low-error-rate claim. The source score is a reproduction target, not evidence of a sub-1% error rate or broad transfer to other instruments, laboratories, geological families or biospheres.

## Execution order

1. Download only the small processed feature/peak archives first; do not fetch the complete raw corpus for reconnaissance.
2. Verify archive file inventory, per-file SHA-256, feature row order and labels against the pinned source.
3. Reproduce accuracy and AUC separately using exact source versions and seed laws.
4. Recover per-fold predictions, selected C values and confusion matrices rather than relying on a rounded aggregate score.
5. Implement a fold-fitted preprocessing path for the inductive comparison.
6. Add sample-family holdouts and a prospective-new-sample challenge. The source panel cannot provide an instrument holdout because it does not contain multiple instrument families.
7. Preserve all failures, unknowns, and out-of-vocabulary measurements in an evidence receipt.

## Claim ceiling

This benchmark is a source-faithful meteorite-versus-terrestrial sample-provenance classifier reproduction. It does not detect all life, establish the biological origin of every terrestrial organic compound, establish that meteorite organics are all abiotic, or qualify a planetary mission instrument.
