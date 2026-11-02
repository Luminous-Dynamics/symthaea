# Connectome intervention metrics

Standalone, dependency-free Rust 2024 reference evaluator for paired intervention effects, prediction intervals and cluster-bootstrap comparisons.

This is a scoring utility, not a causal estimator or biological model. It assumes the caller supplies valid paired treatment/control measurements and model predictions. It cannot validate randomization, instrument exclusion, actuator specificity, or direct-synapse attribution.

## Run

From the repository root:

```sh
cargo test --locked --manifest-path tools/connectome-intervention-metrics/Cargo.toml
```

The crate is its own Cargo workspace, contains no third-party dependencies, and should not be added to the production workspace dependency graph accidentally.

## Inputs

Each observed record binds:

- a stable `sample_id`;
- an independent `cluster_id` (usually animal or animal-session);
- a measured treatment trajectory;
- its paired measured control trajectory.

Each model prediction is aligned by `sample_id` and provides predicted treatment and control trajectories. The evaluator scores:

```
observed_effect(t)  = observed_treatment(t)  - observed_control(t)
predicted_effect(t) = predicted_treatment(t) - predicted_control(t)
```

It does not compare unpaired group means as if they were paired effects.

## Point metrics

- pooled full-trace effect RMSE and MAE (every measured time point has equal weight);
- cluster-macro effect RMSE and MAE (each independent cluster has equal weight);
- response-sign accuracy with a declared three-class dead zone: positive / negative / approximately null.

Pooled and cluster-macro values are both required because clusters may contribute unequal trial counts. A pooled average alone can let one heavily sampled animal dominate the score.

## Predictive intervals

Intervals are declared for treatment-minus-control effects. The evaluator reports pointwise and cluster-macro coverage plus mean width. Coverage without width can be made trivially high by returning uninformatively broad intervals.

## Paired cluster bootstrap

The comparator and candidate are scored on the same paired outcomes. The evaluator computes per-cluster RMSE differences and resamples whole clusters with a deterministic seed. Positive improvement means the candidate has lower RMSE than the comparator.

The percentile interval is not a p-value. Two clusters are only the mathematical minimum; meaningful biological inference requires enough independent animals/sessions and a design appropriate to the sampling population. Repeated trials or neurons inside one animal do not increase the number of independent clusters.

## Fail-closed behavior

Rejects empty or duplicate sample IDs, empty cluster IDs, missing/unexpected/duplicate predictions, mismatched horizons, NaN/Inf, invalid tolerance, reversed/non-finite intervals, zero bootstrap seed, fewer than 100 bootstrap replicates, and bootstrap comparisons with fewer than two independent clusters.

## Synthetic known answers

`fixtures/known-answers.json` pins small analytically checkable cases:

- exact paired effect prediction: RMSE 0;
- paired trajectory effect error: RMSE 1 and MAE 1;
- unequal cluster sizes: pooled and cluster-macro RMSE differ;
- fully covering interval with width 1;
- exact candidate vs comparator with RMSE improvement 1 and a degenerate bootstrap interval.

Fixture claim ceiling is `SYNTHETIC_ONLY`. These cases qualify metric semantics only. They do not validate a biological connectome, causal assumption, model or experimental result.


## Three-way functional evidence labels

The crate also scores source-adjudicated pair labels with three separate reference classes:

1. `FunctionallyConnected`;
2. `EquivalentToNullWithinBound` — evidence that the response lies inside a predeclared equivalence bound, not a claim that no anatomical edge exists;
3. `InsufficientEvidence` — the available data do not establish either of the first two classes.

A model may separately predict `Abstain`. Abstention is a model decision, not a fourth biological evidence state. The confusion matrix has three reference rows and four prediction columns (the three classes plus abstain). It reports overall accuracy (abstentions count as incorrect), selective accuracy among non-abstained predictions, abstention rate, and macro-F1 over reference classes present in the scored set.

The metric function does not infer reference labels or choose the equivalence bound. Those must come from the dataset's published statistical protocol, multiplicity correction and source-level adjudication. In particular, a conventional non-significant test must not be silently relabeled as `EquivalentToNullWithinBound`.

## Scientific limitation

The Randi et al. C. elegans atlas distinguishes functionally connected, functionally non-connected within a predeclared equivalence bound, and insufficient evidence. A non-significant test must not be recoded as a verified zero. Label adjudication is upstream and dataset-specific; this crate does not create mechanism labels. See https://doi.org/10.1038/s41586-023-06683-4 and OSF data DOI https://doi.org/10.17605/OSF.IO/E2SYT.
