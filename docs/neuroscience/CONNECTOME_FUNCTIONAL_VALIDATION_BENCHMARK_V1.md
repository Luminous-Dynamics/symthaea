# Connectome structure-to-function validation benchmark V1

**Status:** preregistered research protocol; no Symthaea functional-prediction result is claimed.

**Tracking:** [SCI-CONNECTOMICS-027 / #9501](https://github.com/Luminous-Dynamics/symthaea/issues/9501)

## 1. Research objective

Test whether independently verified improvements to an EM connectome provide incremental predictive value for independently measured neural function, beyond spatial opportunity, morphology, cell class, stimulus tuning and other available covariates.

The benchmark uses paired reconstruction versions and functionally registered cells where suitable data and version lineage are available.

The primary estimand is not raw neural-response prediction accuracy. It is the **paired incremental out-of-sample predictive value attributable to a verified reconstruction change**, under a frozen split, model family, stimulus protocol and cost budget.

This benchmark is downstream validation. Functional similarity is not synaptic ground truth, and predictive association does not establish causal synaptic influence.

## 2. Scientific boundaries

Four evidence planes remain independent:

1. **Local EM evidence:** image-level synapse presence, polarity, segmentation and partner labels.
2. **Biological reconstruction integrity:** cable, branch, cell identity and synapse preservation.
3. **Graph/query reliability:** edge weights, motifs, connectivity queries and their uncertainty.
4. **Structure-to-function predictive validity:** whether a frozen reconstruction adds predictive value for held-out neural activity.

A model must not compensate for worse EM correctness or biological damage by achieving a better functional prediction score.

The MICrONS functional connectomics paper describes calcium imaging of approximately 75,000 neurons co-registered with EM containing more than 200,000 cells and approximately half a billion synapses. The data are from one mouse and a subset of the anatomical population is functionally registered. This is a uniquely valuable paired dataset, but it is not a multi-animal confirmatory cohort.

Primary sources:
- MICrONS Consortium, *Functional connectomics spanning multiple areas of mouse visual cortex*, Nature (2025): https://www.nature.com/articles/s41586-025-08790-w
- *Functional connectomics reveals general wiring rule in mouse visual cortex*, Nature (2025): https://www.nature.com/articles/s41586-025-08840-3
- *Foundation model of neural activity predicts response to new stimulus types*, Nature (2025; author correction published 2026): https://www.nature.com/articles/s41586-025-08829-y
- MICrONS Explorer: https://www.microns-explorer.org/
- CAVE versioning: https://www.nature.com/articles/s41592-024-02426-z

## 3. Data tiers and eligibility

### Tier A — MICrONS functional connectomics

Use the large MICrONS volume only after exact materialization, segmentation, annotation, synapse-table, co-registration and functional-session revisions are recorded.

Eligibility requirements:
- Exact CAVE materialization/version or timestamp.
- Exact EM image/segmentation and annotation revisions.
- Exact synapse table and proofreading state.
- Exact co-registration table and matching protocol.
- Exact stimulus, trial, session and behavioural metadata.
- Stable cell identity mapping across all compared reconstruction versions.
- Every included cell and observation carries an immutable lineage record.

Do not assume all anatomical cells have functional traces, or that the functionally matched subset is representative. Report excluded cells, missing sessions, registration confidence and coverage by area, depth, cell type and morphology.

### Tier B — MICrONS Layer 2/3 Phase 1

The Phase 1 Explorer page describes a 250 × 140 × 90 µm volume imaged at approximately 3.58 × 3.58 × 40 nm, dense segmentation, proofreading of axons and dendrites for 364 excitatory neurons, and an automated table containing approximately 3.2 million putative synapses. The v185 release notes describe a QCed set of 1,961 synapses between 334 excitatory cells for the soma subgraph, while warning that objects outside that subgraph may retain false merges and splits.

Sources:
- Dataset/access/version page: https://www.microns-explorer.org/phase1
- v185 release notes: https://www.microns-explorer.org/v185-release-notes

Use this tier for local structural and synapse validation, not as a substitute for the larger functionally paired benchmark. The 3.2-million-row putative synapse table is not a ground-truth connectivity graph. The QCed soma-subgraph synapse list has a narrower population and a different estimand.

### Versioning rule

CAVE supports versioned annotation/materialization and updates to cell identities after proofreading. Every experimental arm must pin its exact materialization. Never join a synapse table produced under one segmentation revision to cell identities from another without an explicit, verified mapping.

A version change is an experimental intervention only when the actual changed biological objects and their mapping are enumerated. A version label by itself is not an edit-impact record.

## 4. Experimental design

### 4.1 Paired reconstruction versions

Identify a pair or sequence of frozen reconstruction snapshots:
- Baseline segmentation / annotation version.
- Later independently proofread version.
- Where possible, intermediate versions to measure response changes as edits accumulate.

Build an exact cross-version identity map. Retain:
- Stable cell identity where verified.
- Changed root/segment identifiers.
- Split, merge, remap and unresolved identity events.
- Affected synapses, partner assignments, meshes, skeletons and derived features.
- No-edit controls and query-preserving changes.

Exclude unresolvable identity mappings from paired-effect estimates; report their count and reasons. Do not interpret a root-ID change as a biological cell change by default.

### 4.2 Frozen outcome tasks

Preregister separate tasks; do not combine them into one score.

**F1 — Single-cell stimulus-response prediction**
- Predict responses to held-out trials or stimuli using the declared stimulus and behavioural context.
- Use an appropriate response likelihood and report held-out log likelihood/deviance, explained variance and calibration.
- State whether trial-level repeated measures are clustered by cell/session.

**F2 — Pairwise functional similarity**
- Predict conditional functional similarity or response-noise association after controlling for shared stimulus tuning and behavioural state.
- Use continuous rank/error metrics unless a scientifically defensible binary reference exists.
- Do not call functional similarity a direct synaptic connection.

**F3 — Population/cohort prediction**
- Predict held-out population responses or population-level response statistics.
- Keep the unit of generalization explicit: held-out cells, sessions, stimulus families or cortical areas.

**F4 — Domain transfer**
- Test held-out stimuli, cortical areas or sessions only when the available support permits.
- A single mouse cannot establish cross-animal generalization. Return NOT_ESTIMABLE where independent domain support is absent.

### 4.3 Structural feature ablations

All models use identical train/test partitions and a preregistered capacity/resource budget:

1. Stimulus-only baseline.
2. Functional-history baseline, when admissible for the declared prediction time.
3. Spatial distance and axon–dendrite proximity only.
4. Cell type and morphology only.
5. Unproofread baseline segmentation/synapses.
6. Proofread morphology with synapses held constant.
7. Proofread synapses/partners with morphology held constant.
8. Full proofread reconstruction.
9. Graph/motif features with proximity-conditioned nulls.
10. Matched-capacity graph model.
11. Function-only diagnostic upper bound, clearly excluded from the fair structural comparison.
12. Oracle/reference structural slice only where an independently adjudicated reference exists.

This factorization helps identify whether a gain comes from better segmentation, synapse detection, partner assignment, morphology, or simply leakage/extra capacity.

## 5. Split and leakage contract

- Split by independent biological units: neuron/root lineage, spatial region, session and stimulus family as appropriate to the claim.
- Do not treat time bins or repeated stimuli from one neuron as independent biological replicates.
- Prevent duplicate/near-duplicate trials and shared stimulus segments from leaking across train and test when evaluating new-stimulus generalization.
- Keep stimulus identity, behavioural state, area, session and registration confidence explicit.
- Separate within-cell future-trial prediction from held-out-cell prediction and held-out-area prediction.
- Freeze model selection on development data; do not tune on confirmatory outcomes.
- Separate feature construction from evaluator-only functional targets.
- A model used to derive structural features must not receive the held-out functional labels or post-hoc corrected outcomes.
- Report every split's numbers of cells, lineages, sessions, areas, stimulus conditions and effective independent clusters.

## 6. Primary metrics and estimands

For each task and reconstruction version:

- Held-out log likelihood / Poisson deviance where the declared response model is appropriate.
- Noise-corrected explained variance with the exact estimator and ceiling definition.
- Correlation metrics only as secondary metrics, with confidence intervals and per-cell support.
- Paired predictive-score difference for the same eligible cells/trials across reconstruction versions.
- Calibration and selective risk/coverage where probabilistic predictions are emitted.
- Stratified performance by area, cell type, depth, morphology regime, registration confidence and stimulus domain.
- Query/feature sensitivity to alternative plausible reconstructions.

Primary structural estimand:

`Δ = Score(corrected reconstruction) − Score(baseline reconstruction)`

The comparison is paired on the exact same eligible cells, held-out trials and prediction task. Report the distribution of per-cell/session deltas, not only the pooled mean.

Use cluster-aware bootstrap or a valid hierarchical design at neuron/root/session level. Do not bootstrap time bins or synapses as IID samples. Where the number of independent clusters is too small, return NOT_ESTIMABLE rather than a falsely precise interval.

## 7. Negative controls and adversarial tests

- Spatial-only model outperforms graph features: no claim of graph-specific predictive value.
- Shared stimulus tuning explains pairwise similarity: remove/condition the stimulus component.
- Proximity-conditioned null reproduces the apparent graph effect: no motif-specific conclusion.
- Shuffle partners within matched spatial/morphology strata.
- Shuffle functional labels within valid session/area blocks.
- Change root IDs without changing biological identity; paired mapping must preserve identity.
- Change one verified synapse/partner and verify only dependent features/queries change.
- Add a false synapse or false join; biological and functional gates must expose the damage.
- Exclude uncertain co-registrations and rerun the paired analysis.
- Remove all graph features while preserving other model capacity.
- Permute stimulus IDs and trial ordering to test leakage.
- Test a query whose functional score is unchanged under a genuinely irrelevant edit.
- Test a query whose score changes only after a registered structural correction.
- Ensure unproofread/incomplete regions are not treated as complete negative examples.

## 8. Proximity and motif controls

The 2025 MICrONS wiring analysis explicitly compares functional similarity of connected pairs with unconnected bystanders whose axons and dendrites had opportunity to connect. The analysis considers axon–dendrite proximity and co-travel, which are necessary controls for interpreting apparent structural/functional associations.

Therefore:
- Match connected and unconnected pairs on spatial opportunity.
- Retain axon–dendrite proximity, co-travel distance, cell type and cortical area as explicit controls.
- Preserve the exact null graph generation procedure, randomization unit, constraints and seed.
- Do not infer functional significance from raw motif counts.
- Do not use the same functional outcomes to tune the graph null and then claim confirmatory significance.

Related protocol: [#9156 — spatially conditioned circuit motif null models](https://github.com/Luminous-Dynamics/symthaea/issues/9156).

## 9. Promotion gates (goals, not results)

- Zero cross-version cell identity/materialization mismatches in exhaustive synthetic tests.
- Zero functional target leakage on enumerated adversarial fixtures.
- Positive paired out-of-sample predictive improvement from independently verified structural corrections over spatial + morphology controls on at least one preregistered supported task, with a cluster-aware interval excluding zero.
- Improvement survives proximity-conditioned nulls, stimulus/cell-type controls and removal of uncertain co-registrations.
- No degradation of local EM correctness or biological-harm gates.
- No causal synaptic interpretation from predictive association alone.
- Cross-animal or species transfer requires independent animals/species; the one-mouse MICrONS experiment cannot qualify it.
- Prospective workflow or expert-time claims require a separate randomized protocol; retrospective replay is insufficient.

## 10. Integration

- #9472 — matched synapse presence, polarity, partner identity and edge construction.
- #9345 — payload integrity and paired modality cohorts.
- #9344 / #9379 — reconstruction and pipeline uncertainty.
- #9156 — proximity-conditioned graph nulls.
- #9250 — biological-harm-constrained correction portfolios.
- #9255 — query-scoped certificates and invalidation.
- #9462 — finite-population residual-risk assurance.
- #9242 — prospective proofreading workflow evaluation.

## 11. Claim ceiling

A positive result means only that a named reconstruction version, feature family and model improve a specified held-out functional prediction task on the declared registered cohort. It does not establish a direct synaptic causal mechanism, full connectome correctness, representativeness of the matched cohort, cross-animal generalization, or SOTA.