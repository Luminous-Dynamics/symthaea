# CASTING-001B — Evidence-Bound Benchmark Splits

**Status:** Research implementation on a draft branch stacked on CASTING-001A  
**Scope:** Deterministic train/calibration/holdout identity and leakage checks  
**Claim ceiling:** Software contract only; it does not authenticate source records or prove physical independence.

## Why this tranche exists

A benchmark score is only interpretable if it is bound to the exact model, training values, interval-calibration values, process population, and holdout records used to compute it. A digest attached to a prediction is insufficient when the digest can describe a different split from the one that fitted or calibrated the model.

The Rust module `physics::casting_evidence` introduces:
- typed evidence classes: synthetic, simulated, historical replay, measured physical, and independently replicated;
- observation records with stable IDs, process-population digest, geometry digest, and originating-record digest;
- a deterministic, ordered split manifest over training, conformal calibration, and final holdout partitions;
- overlap rejection for IDs and originating source-record digests;
- process-population compatibility checks and homogeneous, matching evidence classes for the calibration and holdout partitions;
- direct binding to the exact fitted Chvorinov-model digest, training-value digest, and conformal-calibration-value digest;
- independent manifest reconstruction to detect any changed input after the manifest was created;
- an independent aggregate evaluator that replays point predictions and conformal intervals, computes error/coverage metrics, and retains a point-level receipt without calling the model's aggregate holdout-report implementation.

## Key contract

The manifest cannot be created from unbound observation lists alone. It receives the fitted model and the conformal calibrator and verifies:

1. Ordered numeric training observations exactly reproduce the training-data digest retained by the fitted model.
2. The calibrator is bound to the same model identity.
3. Ordered numeric calibration observations exactly reproduce the calibration-data digest retained by the calibrator.
4. The calibration record count matches the calibrator's sample count.
5. All partitions are non-empty and meet their declared sample-count constraints.
6. Observation IDs and source-record digests are unique across the combined split.
7. All records have the same declared process-population digest. Exact geometry digests remain per-record and may differ for geometry holdout testing.
8. Calibration and final holdout each use one evidence class, and those classes match.
9. The final manifest digest binds the schema version, target miscoverage, model identity, ordered numerical data digests, process population, partition sizes, ordered record IDs, full ordered record digests, and evidence-class counts.
10. A verifier reconstructs the manifest from the supplied model, calibrator, and records and refuses if the digest differs.

The split is deliberately ordered: changing the order of training or calibration observations changes the bound data digest. The underlying model/calibrator must be fitted/calibrated in exactly the same order used to declare those partitions.

The benchmark evaluator reports MAE/RMSE and mean/median absolute relative error only over supported predictions, plus prediction coverage over the full holdout. It reports empirical interval coverage among generated intervals, interval provision fraction over the full holdout, interval width, nominal coverage target, model digest, and manifest digest. An out-of-domain point remains in the total denominator but has no fabricated point estimate or interval.

This evaluator independently recomputes aggregation and evidence checks, but it deliberately reuses the model's point-prediction and conformal-interval APIs. It is not an independent numerical implementation of the casting physics; a future verifier should independently reconstruct the formula and interval order statistic from frozen parameters and fixtures.

## What this does not prove

A digest is a content binding, not a signature or source-authenticity guarantee. A party could falsely label a simulation as measured, fabricate a source record, or reuse statistically dependent trials under different IDs. A trustworthy physical benchmark still requires external evidence custody, instrument calibration, sample selection protocols, and independent review.

The split validator also cannot infer statistical exchangeability from numeric rows. The conformal coverage statement remains conditional on the calibration and future in-domain observations being exchangeable. Foundry process drift, changed alloy/mold populations, operators, sensor state, and correlated castings must be addressed by the experiment design and prospective validation.

Evidence class and process-population identity remain separate. A measured holdout is not enough to establish physical validation if the calibration data, model lineage, measurement method, or test population is inappropriate. A synthetic pass establishes software semantics only.

## Required benchmark protocol

- Fit the predictor using only the training partition.
- Calibrate a requested prediction interval using only the disjoint conformal-calibration partition.
- Freeze the final holdout before model/interval selection and do not tune against it.
- Keep physical runs, historical replay, simulation, and synthetic fixtures in separate denominators and reports.
- Use IDs tied to the original observation records and immutable artifact digests. Do not mint new IDs to hide reuse.
- For geometry generalization, state whether geometry digests are fully held out; do not imply a new process population was tested unless its declared population and applicable evidence justify that claim.
- Report conditional prediction errors beside prediction coverage and interval coverage/width.
- Report nominal conformal coverage separately from empirically measured holdout coverage.
- Count physical experiments separately from solver runs and synthetic samples.
- For claims of yield, scrap, energy, production cost, or part acceptance, require actual measured outcomes and a frozen comparable baseline.

## Adversarial fixtures

The new tests cover deterministic replay, duplicate IDs within/across partitions, duplicate source-record reuse under different IDs, incompatible process populations, mixed/mismatched evidence classes, insufficient training/calibration samples, invalid measurements and placeholder digests, ordering/coverage changes, changed records after manifest construction, training-data/model mismatch, calibration-data/calibrator mismatch, and calibrator/model mismatch.

They are authored but not yet executed in this environment. Static source inspection is not a compile/test pass. The parent CASTING-001A PR and this follow-up must remain draft until exact-head formatting, compilation, unit tests, and independent review complete.

## Research context and comparison

The casting literature is advancing beyond isolated regressors. A 2025 squeeze-casting study combined revised Latin-hypercube sampling, Bayesian optimization, and phase-field simulation, reporting improvements after six experimental iterations from an initial 25-sample dataset: [Pei et al., *npj Computational Materials* (2025)](https://www.nature.com/articles/s41524-025-01524-6). Those numerical results are not transferable targets for an unrelated casting process; the relevant benchmark is experiment efficiency under a matched process and quality target.

A 2026 review identifies cross-alloy generalization, uncertainty quantification, benchmark reproducibility, sensor integration, and industrial validation as open gaps in foundry AI/digital-twin work: [Essam et al., *Discover Materials* (2026)](https://link.springer.com/article/10.1007/s43939-026-00685-5). NIST's 2026 manufacturing digital-twin workshops similarly prioritize interoperability, VVUQ, cybersecurity, and workforce readiness: [NISTIR 8620](https://www.nist.gov/publications/digital-twins-workshops-summary-report).

The targeted differentiation is not another model leaderboard. It is the ability to reproduce exactly which evidence generated each reported result, refuse split leakage and incompatible populations, and compare models at declared prediction coverage before claiming improved physical manufacturing outcomes.

## Ownership

This is a casting benchmark/evidence adapter, not a second process registry or observation ontology. Continue composing with MFG-PROC #5686 and the current qualified observation/metrology and Mycelix manufacturing-event owners. Where an upstream contract is not qualified, retain an explicit unresolved dependency rather than claiming end-to-end production qualification.

## Next steps

1. Execute exact-head Rust formatting and tests.
2. Review the conformal quantile implementation and numerical edge cases independently.
3. Add an independent evaluator that recomputes manifest digests and error/coverage metrics without calling the candidate-model report path.
4. Add frozen synthetic fixtures and known-answer numerical vectors in a separately versioned benchmark corpus.
5. Define an actual, low-risk coupon study with appropriate metrology and independent holdout selection.
6. Only after measured calibration/generalization results should the project consider multi-fidelity experiment selection or machine integration.
