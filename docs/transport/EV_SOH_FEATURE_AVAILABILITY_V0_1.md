# SOH Feature-Time Availability and Reference Lineage v0.1

**Status:** Rust-first feature-provenance and benchmark audit. Not an SOH predictor.
**Owner:** [EV-SOH-FEATURE-AVAILABILITY-001 #11419](https://github.com/Luminous-Dynamics/symthaea/issues/11419).
**Parent benchmark contract:** [SOH benchmark manifest PR #11342](https://github.com/Luminous-Dynamics/symthaea/pull/11342).

## Research anchor

Peng, Wang & Zhang, “Deployable Self-Referenced Features for Cross-Cell Battery State-of-Health Estimation: Randomized Pairing Controls and Benchmarking,” *Energies* 19(20), 4805, published 12 October 2026: https://doi.org/10.3390/en19204805

The paper reports that changing feature normalization/reference conventions changed RMSE by as much as about 108% across its 11 evaluation units. It explicitly distinguishes offline full-lifetime normalization (not available to an operating cell) from strict-online feature construction. While the offline protocol found gains for self-referenced features, the corrected strict-online test did not show an overall accuracy gain: median RMSE changed by +4.3% for gradient boosting and +10.4% for its frozen tabular foundation model. The authors also use randomized mispairing and affine negative controls to test whether the chosen reference is responsible for apparent feature gains. These are study-specific results, not a general claim for or against self-referencing.

## Paradigm

A feature is online only if its entire dependency graph was observable at prediction time. Source-file existence is not enough: measurement timestamp, receive/ingestion timestamp, reference availability, transform fit scope and feature-computation time must all precede the decision timestamp.

Every prediction row binds:
- physical asset and sample identity;
- decision timestamp;
- exact feature IDs and value digests;
- latest source measurement and receive timestamps;
- feature-computation timestamp;
- reference type and physical identity/population;
- reference measurement and receive timestamps;
- target/future-label dependency flags;
- availability evidence digest.

Every feature definition binds its schema, implementation, transform and fit scope, fit-asset identities and reference policy.

## Strict-online rules

- Every feature declared by the model must appear exactly once in every prediction receipt.
- Measurement, receive, reference and feature-computation times must be no later than decision time.
- Same-asset prefix references must bind the exact prediction asset identity.
- Peer references must use a distinct development/training asset, never a held-out test asset.
- Training-population transforms must be fit only on development assets.
- Fixed transforms must be evidenced before split freeze.
- Full-trajectory, test/future, test-label and full-lifetime-extrema dependencies are rejected in strict-online mode.
- Peer-reference experiments require a preregistered assignment, at least 100 deterministic derangements, a true-pair arm and an affine negative control. The schedule digest and freeze timestamp are bound before test access.
- Full-lifetime normalization is retained only as an OFFLINE_ORACLE_ONLY comparison; it cannot be included in strict-online leaderboard results.

The audit reports an explicit claim ceiling: STRICT_ONLINE_ELIGIBLE or OFFLINE_ORACLE_ONLY. Acceptance means the declared time/provenance contract is internally consistent. It does not authenticate source data or prove that the source timestamps are truthful.

## Frozen synthetic corpus

Fourteen cases: three accepted and eleven rejected. Accepted cases cover a valid same-cell prefix, an offline full-history oracle explicitly labelled offline-only, and a preregistered peer reference with randomized derangements. Negative controls cover full-life normalization online, delayed receipt, held-out peer references, test-label dependence, future references, test-asset-fitted transforms, late fixed predeclaration, missing feature receipts, missing peer randomization, feature computation after decision, and mismatched same-cell identity.

These cases are synthetic software controls, not physical battery evidence.

## Benchmark tournament

Compare raw, train-only normalized, prefix-only normalized, self-referenced, peer-referenced and fixed external-reference features with identical model, labels, split, compute and data budgets. Keep full-lifetime normalization in an offline-oracle table only.

Report:
- strict-online feature availability and first usable prediction time;
- per-cell/per-pack and worst-domain MAE/RMSE;
- feature missingness and reference-identity mismatch rates;
- online latency, memory and sensor/data burden;
- paired per-asset differences, randomized derangement nulls and affine negative controls;
- predictive uncertainty and support-conditioned calibration from PR #11390.

An accuracy gain is not assumed. Feature computability and feature predictive value are separate benchmark outcomes.

## Limits

This audit does not establish the electrochemical meaning of a self-reference, prove causal effects on ageing, train a feature-selection algorithm or certify BMS deployment.