# Connectome-constrained perturbation dynamics benchmark V1

**Status:** research protocol / draft; no Symthaea biological result is claimed.

**Tracking:** [SCI-CONNECTOMICS-029 / #9522](https://github.com/Luminous-Dynamics/symthaea/issues/9522)

**Relation:** extends (but does not merge estimands with) [#9501](https://github.com/Luminous-Dynamics/symthaea/issues/9501) and [PR #9503](https://github.com/Luminous-Dynamics/symthaea/pull/9503).

## 1. Research objective

Test whether a versioned anatomical connectome provides incremental predictive value for neural responses to controlled perturbations, and whether the resulting dynamical model improves prediction of sensory-to-motor/behavioral outcomes.

This is a different question from predicting activity under observed stimuli. It is also not a claim that anatomical edges directly encode physiological causal weights.

The core distinctions are:

1. **Structural route:** anatomy supports a path under a declared reconstruction.
2. **Observed perturbation response:** measured response to a controlled intervention over a finite horizon.
3. **Mechanistic attribution:** a stronger claim requiring an intervention/mediation design that identifies a direct mechanism.

A response at a neuron with no direct edge from the stimulated neuron may arise from multi-hop propagation. A structural edge may be silent in a particular context or horizon. Neither observation should be converted into a false direct-edge claim.

## 2. Why this is the next benchmark

The current structure-to-function protocol tests paired prediction gains from independently verified reconstruction changes against held-out neural activity. A further step is to evaluate whether anatomy constrains responses to controlled perturbations rather than merely describing observational co-variation.

A useful external baseline exists in C. elegans. Creamer, Leifer and Pillow's 2025 bioRxiv v3 preprint fits connectome-constrained dynamics to whole-brain recordings under single-neuron optogenetic perturbation. It describes 156 head neurons recorded across 110 animals, an 80-animal training set and 30-animal test set, and reports that the model captured 92% of measured causal interaction structure. This is a serious reproduction/falsification target, not a result from Symthaea.

The zebrafish resources offer complementary tiers:
- Fish1.5 aligns EM and two-photon calcium imaging at single-cell resolution in the same specimen.
- Fish Fire&Wire is a controlled-access draft same-animal whole-brain wiring/activity/behavior resource.
- ZAPBench is an activity forecasting benchmark; structural conditioning requires verified matching anatomy and registration artifacts.

These tiers must remain separate because they support different estimands.

## 3. Dataset tiers and evidence eligibility

### Tier A — C. elegans optogenetic perturbation

Primary reference: Creamer, Leifer and Pillow, *Bridging the gap between the connectome and whole-brain activity in C. elegans*, bioRxiv v3 (2025), https://doi.org/10.1101/2024.09.22.614271

Use this as an external reproduction/falsification benchmark for connectome-constrained response prediction. The recording cohort spans animals while the anatomical connectome is a species-level constraint; it is not an individually reconstructed connectome for every recorded animal. Therefore this tier may support cross-animal anatomical-template prediction, but it cannot estimate the effect of proofreading a particular animal's connectome on that same animal's recorded responses.

Required pins:
- exact preprint version, dataset and source artifact revisions;
- connectome source/revision and cell identity mapping;
- optogenetic target, dose, timing and response-horizon definitions;
- animal-level train/development/confirmatory partitions;
- preprocessing, cell inclusion, missingness and model/checkpoint digests.

If the source data, code, checkpoint or split cannot be obtained, report NOT_REPRODUCIBLE. Do not silently substitute a reimplementation with different inputs or partitions and call it an exact reproduction.

### Tier B — Fish1.5 same-specimen structure/function

Official release: https://jboulanger91.github.io/fish1.5-release/

The release describes a 7-day larval zebrafish with EM and two-photon calcium imaging registered at single-cell resolution; 62,745 cell bodies, 15,017 functionally imaged cells and four visual stimulus conditions. This supports same-specimen structural/functional analysis with a limited stimulus panel. It is not an optogenetic perturbation dataset merely because it contains functional imaging.

The companion Boulanger-Weill et al. 2025 manuscript describes a biophysically realistic circuit-constrained model and experimental tests of model predictions. Reproduce the relevant published task before claiming an improvement.

### Tier C — Fish Fire&Wire (controlled access)

Official project: https://www.janelia.org/fish-firewire

Janelia describes a draft whole-brain single-synapse-resolution connectome of a 6-day larval zebrafish with approximately 140,000 neurons, same-animal neuron-level activity during multiple sensory, motor, cognitive and autonomic behaviors, and sensory structures plus motor outputs.

This is a potential sensory-to-action validation resource, not an unrestricted public benchmark. The official project states that snapshots are versioned during proofreading, access is under project terms, and early-access papers/preprints may not be submitted before 1 December 2027. Use only after explicit access and collaboration terms are satisfied. Do not scrape, redistribute, or claim an evaluation on data not actually accessed.

### Tier D — ZAPBench activity forecasting

Official benchmark: https://google-research.github.io/zapbench/

ZAPBench provides whole-brain zebrafish activity forecasting data for more than 70,000 neurons. Use its published task as an activity-only baseline. The original benchmark's reference to ongoing synaptic mapping does not itself authorize structural conditioning. Such an arm requires the exact compatible connectome snapshot, cell correspondence, access rights and versioned registration.

### Dataset identity and version rule

A version identifier is meaningful only within its named datastack and source product. The MICrONS tutorial currently lists materialization version 1926 for the minnie65_public datastack (timestamp 2026-09-28T22:44:42.598563Z). This must not be transplanted to Fish1.5, Phase 1 or Fish Fire&Wire. Every adapter must pin its own dataset/datastack, segmentation, annotation, registration, activity and transformation revisions.

## 4. Formal prediction contract

For a declared biological system and context, define a finite-horizon dynamical model:

    x[t+1] = f(x[t], u[t], G, theta, c[t])

where:
- x[t] is the observed or latent neural state under a pinned feature definition;
- u[t] is a stimulus or controlled intervention;
- G is the pinned, versioned anatomical constraint;
- theta contains fitted dynamical parameters, delays and cell-intrinsic terms;
- c[t] contains declared behavioral/context covariates.

The model report must identify which terms are constrained by anatomy and which are estimated from activity. If non-edges prohibit direct parameters, that rule must be explicit. Non-edges do not prohibit indirect multi-hop effects.

Synapse counts are not synaptic conductances. Anatomical polarity is not automatically physiological sign. A structural edge is not proof of transmission in every context. Keep synapse, cell identity, structural path, perturbation response and model parameter evidence as distinct typed objects.

## 5. Frozen evaluation tasks

### P1 — Perturbation impulse-response prediction

Predict the time-resolved response of eligible recipient neurons following a declared single-cell intervention. Evaluate amplitude, sign where meaningful, latency, temporal profile and recovery over a fixed finite horizon.

### P2 — Held-out intervention-target generalization

Hold out stimulation targets, not merely time bins. Separately report held-out target neurons, recipient neurons, target-recipient pairs, intervention strengths and durations where the data support them.

### P3 — Multi-hop propagation

Test whether the model predicts responses in anatomically non-adjacent recipients through explicit multi-hop paths. Compare one-hop-only, path-aware and unconstrained dynamical models. A response in a non-adjacent neuron is not evidence for a direct synapse.

### P4 — Cross-animal generalization

Split at the animal level for cross-animal claims. Repeated cells, time points and trials within an animal are not independent biological replicates. When a canonical connectome is applied across animals, describe the result as anatomical-template transfer, not same-animal reconstruction validation.

### P5 — Sensory-to-motor/behavior prediction

Where inputs, neural state, anatomy and motor/behavior outputs are aligned in the same animal, predict held-out behavioral variables from sensory and neural history, with and without anatomical constraints. Keep behavior labels and future movement out of structural feature construction. Report neural response and behavior prediction as separate outcomes.

### P6 — Reconstruction-change value

Only for same-specimen, same-cell paired cohorts, compare baseline and independently proofread reconstruction versions using identical eligible cells, intervention trials, model capacity, train/test partitions and training budgets. When the connectome is a canonical template from another animal, this paired edit estimand is NOT_ESTIMABLE.

## 6. Baseline matrix

All comparable arms receive identical eligible observations, split manifests, training budgets and evaluation targets.

1. Exact published connectome-constrained dynamical baseline.
2. Activity-history-only autoregressive/state-space baseline.
3. Stimulus/intervention-only baseline.
4. Cell-type and intrinsic-property baseline.
5. Spatial-distance-only and morphology-only controls.
6. Dense unconstrained dynamical model with matched parameter and compute budgets.
7. Anatomically constrained model with matched capacity.
8. Degree-, cell-type- and spatial-opportunity-preserving shuffled/rewired connectomes.
9. Edge ablation/addition and synapse-confidence sensitivity controls.
10. One-hop-only versus multi-hop/path-aware models.
11. Linear/state-space versus nonlinear recurrent and neural ODE/CfC/LTC candidate models.
12. Repeatability/noise ceiling and evaluator-only oracle diagnostics, excluded from deployable comparisons.

A candidate may win one task and lose another. Do not pool species, measurement modalities, intervention designs or response variables into one leaderboard score.

## 7. Leakage and statistical contract

- Split by animal for cross-animal claims; by intervention target/session for held-out-target claims; keep related trials and cell lineages within their declared partitions.
- Pin preprocessing, registration, cell identity, connectome version, intervention protocol and exclusions in immutable manifests.
- Freeze model selection on development data before confirmatory/lockbox outcomes are opened.
- Report animal, intervention-target, recipient, trial and effective independent-cluster counts.
- Use uncertainty at the actual randomization/generalization unit; never bootstrap neuron-time bins as IID samples.
- Compare structural models with shuffled and spatially constrained graphs, not only with an empty-graph baseline.
- Use negative-control interventions and label/time shuffles where supported by the experimental design.
- Separate measurement noise, registration uncertainty, model uncertainty and animal-to-animal variation.
- Unsupported strata, inaccessible artifacts, insufficient independent clusters and unidentifiable effects return NOT_ESTIMABLE.
- Direct-edge, necessity, sufficiency or synaptic-mechanism claims require a separate design that identifies that estimand.

## 8. Adversarial fixtures

At minimum, include:
- strong observational correlation with no measured intervention response;
- a direct structural edge with no detectable effect in the declared context/horizon;
- a non-adjacent recipient with a genuine multi-hop response;
- a response explained by shared input or hidden state rather than the proposed path;
- degree-, cell-type- and spatial-opportunity-preserving graph shuffles;
- correct topology with incorrect synaptic polarity or partner assignment;
- a high-capacity dense model that wins through parameter count or leakage;
- held-out intervention targets absent from training;
- target/recipient cells from one animal leaking across the animal split;
- cross-animal identity mapping errors and one-to-many correspondences;
- version-mixed anatomy/function, stale root IDs and changed transforms;
- average-response fit that misses sign, latency or recovery;
- strong neural-response prediction but failed motor/behavior prediction;
- missing modalities or intervention dose silently treated as valid evidence.

## 9. Primary metrics

- Held-out perturbation-response negative log likelihood where the observation model is appropriate.
- Noise-normalized trajectory error/explained variance with an explicit noise-ceiling estimator.
- Response-kernel correlation and integrated absolute/signed response error.
- Peak response amplitude and latency error.
- Pairwise intervention-effect matrix correlation and calibrated error.
- Multi-hop response prediction, kept separate from direct-edge attribution.
- Calibration and selective risk/coverage by animal, target, cell class and context.
- Held-out behavior prediction on supported tasks.
- Query/decision value and cost per verified predictive improvement.
- Runtime, memory, energy and data-access/annotation cost under matched hardware and policies.

## 10. Promotion gates (research goals, not results)

- Reproduce the published C. elegans baseline on its exact versioned data, split and metric before claiming improvement.
- Demonstrate structural value over activity-only, spatial-only and matched-capacity unconstrained models on preregistered held-out targets/animals, with cluster-aware intervals and a predeclared practical-effect threshold.
- Preserve any advantage over degree-/spatial-/cell-type-preserving shuffled-connectome controls.
- Show calibration and finite-horizon trajectory fidelity, not only pooled correlation.
- Zero cross-animal, intervention-target, lineage and future-label leakage on exhaustive synthetic mutation fixtures.
- No direct-synaptic or mechanistic claim from total system response alone.
- No biological SOTA claim from synthetic fixtures, one-species reproduction or controlled-access unpublished data.
- Reconstruction-change claims require same-specimen pairing and independent anatomical adjudication.

## 11. Implementation sequence

1. Define a versioned dataset/task eligibility manifest and run-record schema.
2. Build exact synthetic fixtures for direct, delayed, indirect, recurrent and confounded perturbation effects.
3. Independently reproduce the C. elegans study as an external falsification baseline.
4. Establish the Fish1.5 same-specimen adapter and reproduce one published structure/function analysis.
5. Add matched-capacity graph-constrained, graph-shuffled and activity-only comparisons.
6. Integrate observation/perturbation results with #9501, #9344, #9379 and #9255; retain #9462 as a separate stopping-assurance contract.
7. Pursue Fish Fire&Wire only through its official collaboration process and obey snapshot, access, authorship and publication restrictions.
8. After retrospective reproduction, preregister a prospective perturbation/behavior evaluation where technically and ethically feasible.

## 12. Integration and claim ceiling

Related work:
- #9501 / PR #9503 — held-out observational structure-to-function validation.
- #9472 — synapse presence, polarity, partner identity and edge construction.
- #9504 — CAVE identity remapping and derived-annotation freshness.
- #9344 / #9379 — reconstruction and analysis-pipeline multiverse.
- #9255 — query-scoped certificates and invalidation.
- #9462 — query-specific residual-risk stopping assurance.
- #9417 / #9403 — adaptive evidence acquisition, calibration, domain shift and abstention.
- #3322 — conceptual separation of structural route, observational association and intervention response; biological adapters require their own evidence semantics.

This protocol does not claim access to Fish Fire&Wire, reproduction of the C. elegans preprint, a new causal model, or any Symthaea biological performance improvement. It does not treat the connectome as a complete dynamical specification or equate anatomical weights with physiological synaptic efficacy.
