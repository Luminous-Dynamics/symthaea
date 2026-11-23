# Mycelial Posterior Model Adequacy and Open-World Residuals
**Date:** 2026-11-23  
**Status:** Research protocol / synthetic-first contract; no real-data adequacy score or model-performance claim

## 1. Objective

The feasible graph posterior v1 enumerates candidate subsets exactly under a conditional-independent Bernoulli prior conditioned on explicit pairwise incompatibilities. That is a useful bounded inference contract, but exact normalization does not establish that the candidate inventory contains the true graph or that the supplied model family adequately explains the observations.

The question for this protocol is:

> Can the system detect when its own candidate set or generative assumptions are insufficient to explain the observed mycelial evidence?

This is distinct from asking whether the posterior is concentrated. A narrow, normalized posterior can be confidently wrong under candidate-generation failure or shared acquisition artifacts.

Related implementation and benchmark tracks:
- #13369 — feasible graph hypothesis posterior
- #13379 — decision-value active sensing
- #13367/#13368 — functional observation and biological-replicate split contracts
- #13352 — measured transport and tracer validation
- #13409 — source–sink flow under depletion and hydraulic resistance
- #13419 — model adequacy and open-world residuals (this protocol)

## 2. Claim boundaries

Keep these quantities separate:

1. **Candidate probability:** classifier or annotator-derived support for a specific candidate under a declared calibration model.
2. **Graph posterior:** probability over feasible combinations within the declared candidate inventory and prior.
3. **Candidate coverage:** evidence that the generator considered the relevant observed contact/branch domain.
4. **Model adequacy:** whether the model family predicts independent evidence with acceptable residuals.
5. **Biological truth:** an independently adjudicated or experimentally supported connection.

None implies the next automatically.

A residual is not an edge. A highly probable candidate is not biological acceptance. A graph score is not a flow measurement. A connectivity claim is not transport or viability.

## 3. Open-world observation coverage ledger

Create a source-bound coverage ledger for every observation region eligible for reconstruction.

Each observation/contact record must be assigned exactly one state:
- examined and candidate generated;
- examined and candidate rejected;
- examined and candidate unresolved;
- examined, no candidate generated;
- unavailable or unmeasurable;
- outside the declared acquisition scope.

The ledger binds:
- raw image/volume content digest and source identity;
- sample, biological replicate, acquisition session and timepoint;
- objective, pixel/voxel scale, z-step and imaging modality where known;
- coordinate-frame and spatial-transform digests;
- calibration and annotation protocol digests;
- candidate-generator identity, version, configuration and checkpoint digest;
- explicit observation IDs and any generated candidate IDs.

Unmatched observations must retain their original identity and geometry. Do not silently discard them or promote them to synthetic graph edges.

### Residual classes

Represent at least:
- apparent crossing with unresolved depth ordering;
- possible contact absent from the candidate inventory;
- branch/junction residual;
- segmentation or centerline discontinuity;
- occluded/out-of-focus region;
- field-of-view boundary;
- calibration/registration conflict;
- temporal mismatch;
- unknown/unclassified residual.

Each class must carry a bounded explanation and source evidence. The system may conclude that evidence is insufficient; it must not invent missing physical geometry.

## 4. Model adequacy and predictive checks

Define observation signatures predicted by a graph hypothesis and acquisition model, then compare them against evidence not used to select that hypothesis.

Possible signature families:
- focal-plane-specific crossing and z-order observations;
- local filament continuity, tangent and diameter profiles, only where measured;
- contact-group counts and conflict patterns;
- temporal branch/fusion persistence;
- edge-localized cytoplasmic-flow direction and speed;
- tracer arrival, recovery and source-to-sink outcomes;
- microbial dispersal under measured saturation/liquid movement.

Use independent biological-replicate holdouts. Do not split frames, crops, trajectories, sessions or fruiting bodies from the same cultivation box/replicate across train/test.

Use proper predictive scores appropriate to the observation type:
- held-out log predictive score / expected log predictive density;
- Brier score for declared binary events;
- calibration intercept, slope and reliability curves;
- CRPS or another declared proper score for continuous outcomes;
- predictive-interval coverage and width;
- residual checks stratified by organism, device, modality, focal depth, treatment and acquisition group.

Posterior predictive checks are diagnostics, not proof of model truth. Prefer a held-out predictive check when the same data would otherwise be used both to fit and to assess the model.

Reference: Moran, Blei and Ranganath, “Holdout predictive checks for Bayesian model criticism,” Journal of the Royal Statistical Society: Series B (2024), https://doi.org/10.1093/jrsssb/qkad105. Their holdout predictive check compares posterior-predictive draws with an independent held-out population sample, addressing the data-reuse problem of ordinary posterior predictive checks.

## 5. Common-cause and dependent evidence

Candidate decisions are often not independent. Several contacts can share one acquisition artifact or one learned failure mode.

Represent explicit evidence/acquisition groups for:
- z-registration and focal-plane calibration;
- illumination, contrast, stain and sample preparation;
- objective, detector and acquisition device;
- segmentation/checkpoint/model version;
- local occlusion, saturation or projection artifacts;
- annotator, annotation protocol and review session.

Compare, using adjudicated multi-contact groups:
1. v1 conditional-independent Bernoulli posterior;
2. grouped calibration / joint-label baseline;
3. constrained pairwise or log-linear factor model;
4. explicit latent common-cause model, only where the latent cause is identifiable from independent evidence.

Do not double-count evidence. A calibrated classifier posterior is not automatically a likelihood that can be multiplied by the same evidence-derived prior. Every probability input must declare whether it is a prior, likelihood, posterior score or calibrated marginal and which evidence it conditions on.

Exact inference on bounded groups should remain the oracle for validating approximate inference. Any new approximate posterior must report approximation error, effective sample size, convergence diagnostics, and failure/abstention conditions.

## 6. Calibrated abstention

Model adequacy output must distinguish:
- adequate under the tested evidence scope;
- unresolved / insufficient evidence;
- candidate-coverage failure;
- predictive/model-family inadequacy;
- unsupported coordinate or calibration state;
- evidence dependency not modelled;
- numerical/resource-bound refusal.

“Not detected” is not “absent”. “Not measured” is not zero. “No tracer arrival” must be distinguishable from right-censored, below-detection, failed-assay and missing observations.

Any acceptance threshold must be preregistered against the cost of false adequacy and false abstention. Confidence or posterior concentration alone must never choose the adequacy status.

## 7. Synthetic misspecification benchmark

Use frozen seeds, generator version, source digests and exact ground truth for controlled cases:

1. true fusion omitted from the candidate inventory;
2. branch or junction omitted by segmentation;
3. two candidate contacts share a focal-plane registration error;
4. a common illumination/contrast artifact biases several candidates;
5. projected crossing is mistaken for contact because depth is unavailable;
6. several candidates share a segmentation checkpoint failure;
7. contradictory annotations / unresolved depth;
8. observation outside the training organism, stain, modality or device;
9. censoring and unavailable regions that must not be treated as negative labels;
10. adequate-model controls with independent acquisition groups and known feasible graph truth.

A separate independent reference implementation should generate truth and verify exact small cases. Do not reuse the production posterior routine as its own oracle.

## 8. Functional scientific benchmark

Functional validation is downstream and independent of graph construction.

### Evidence families

- **Cytoplasmic flow:** edge-localized direction, speed, trajectory count, time and measurement method.
- **Stable-isotope tracer:** source stratum, sink/fruiting-body identity, tracer and isotope calibration, injection/recovery quantities, colonization time, flush, treatment and censoring.
- **Hyphal highway dispersal:** organism, hypha/bacteria identity, source/destination regions, cell counts, arrival time, liquid saturation and device geometry.

### Baselines

At equal data access and the same biological-replicate splits compare:
- image-only model without reconstructed graph;
- Euclidean distance-only model;
- unweighted topology/reachability;
- shortest path / weighted geodesic;
- MST and Delaunay structural controls where meaningful;
- degree-preserving rewired graphs;
- geometric resistance model only with measured radii/geometry and declared physical assumptions;
- local source–sink demand/depletion model;
- measured-flow/tracer-constrained model.

Do not infer conductance from segment length, treat graph cycles as flow capacity, or turn a shortest path into a biological transport label.

### Outcomes

- flow-direction accuracy and speed MAE/RMSE;
- calibrated edge-flow likelihood;
- tracer arrival-time and source-attribution error;
- recovered quantity prediction and interval coverage;
- per-sink source contributions;
- bacterial dispersal/arrival probability and censored time-to-arrival;
- held-out intervention/perturbation prediction;
- false-delivery, missed-delivery and decision-abstention rates.

Use biological replicate or cultivation box as the split and resampling unit. Keep samples, fruiting bodies, frames, crops, trajectories and repeated assays from a shared biological unit together.

## 9. Initial research targets

These are hypotheses for preregistered gates, not achieved results or claims of superiority:

- 100% agreement with an independent exact enumerator on every bounded feasible-posterior fixture.
- Zero posterior mass on explicitly infeasible graph assignments.
- At least 95% detection of the predefined severe synthetic misspecification cases at no more than 5% false adequacy, with uncertainty intervals and prevalence reported.
- No more than 5% false abstention on matched adequate-model synthetic controls.
- Nominal 90% held-out graph-event predictive coverage within a preregistered tolerance at biological-replicate level.
- On dependent-evidence cohorts, a common-cause model improves held-out proper score over the v1 independence baseline without degrading calibration on independent-evidence controls.
- A functional model must beat the strongest matched non-graph baseline on a held-out biological replicate cohort before a graph-specific predictive benefit is claimed.
- A transport improvement target of at least 10% relative error reduction versus the strongest topology-only baseline is an aspirational gate from #13352, not a present result.

Report denominators, missingness, assay failure, confidence intervals, effect sizes and subgroup performance. Do not report only aggregate F1 or a single scalar score.

## 10. External comparison and interoperability position

### Reconstruction and microscopy

- Sten et al. (2024), 135 fungal microscopy samples, reported filament-detection F1 0.88–0.94 and crossing/overlap F1 0.89: https://doi.org/10.1016/j.ecoinf.2024.102670.
- Sten et al. (2025), 3D z-stack reconstruction reported overlap-detection F1 0.91–0.92 on six manually annotated real fungal images: https://doi.org/10.1016/j.ecoinf.2025.103317. This is a narrow overlap result, not a universal graph-reconstruction SOTA.
- Fiji SNT provides established tracing and analysis, including non-neuronal filament use, and should be an interoperability partner rather than a target to replace.
- Imaris and ZEISS arivis provide commercial 3D analysis/visualization and AI workflow capabilities. Symthaea should consume or reference their traces, preserve foreign object IDs and add evidence qualification, alternatives, counterfactuals and functional validation—not claim to replace mature image viewers.

### Independent biology

- The 2025 plant–fungal trade study linked ~100,000 cytoplasmic-flow trajectories to a growing network with >500,000 tracked nodes: https://doi.org/10.1038/s41586-025-08614-x. Its Petri-plate setup is an important dynamic benchmark, not full 3D soil truth.
- The 2026 AM-fungal study reports ~25,000 flow trajectories from 5,000 videos; its data are described as available upon request, so ingestion requires author permission: https://doi.org/10.1016/j.funbio.2026.101775.
- The 2026 Agaricus stable-isotope study provides a distinct source-to-sink resource-transfer benchmark conditioned on colonization time and network architecture: https://doi.org/10.1111/1462-2920.70345.
- The September 2026 Hyphal Highways Device study provides a controlled bacterial dispersal assay along Pythium hyphae and explicitly varies liquid movement/unsaturated conditions: https://doi.org/10.1093/femsml/uqag035.

### Interchange

Maintain the canonical graph sidecar independently from SWC. SNT's TRACES format can preserve some 4D/5D and metadata that SWC does not. OME-NGFF spatial transforms should be bound by digest; vendor object IDs remain foreign identity mappings. Do not claim a vendor round trip until an actual import/export has been executed against pinned versions and source files.

## 11. Implementation sequence

1. Build a source-bound observation-coverage and residual contract.
2. Add deterministic synthetic fixtures for open-world omissions and common-cause artifacts.
3. Add a separate model-adequacy report; do not alter v1 posterior semantics.
4. Add held-out predictive signatures and calibration metrics.
5. Implement a joint/common-cause baseline only after evidence grouping and double-counting semantics are specified.
6. Connect model adequacy outcomes to the active-sensing evaluator as a reason to acquire or review additional evidence.
7. Ingest licensed real data only after file-level rights, provenance and replicate identity audits.
8. Validate functional prediction using independent measured flow, tracer or dispersal outcomes.

## 12. Acceptance conditions

- Versioned Rust contracts, bounded resources and deterministic canonical digests.
- Original source, acquisition model, calibration, replicate and residual identities are committed into reports.
- Exact small-case oracle is independent from production inference.
- Adversarial tests cover missing candidate, shared acquisition artifact, evidence double-counting, censored observations, tampered digests and unavailable geometry.
- Model adequacy is reported separately from posterior concentration.
- No microscope actuation, automatic biological acceptance, physical transport claim or SOTA claim is inferred from this protocol alone.
