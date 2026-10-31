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

A useful external baseline exists in C. elegans, but its public version records contain a material discrepancy. The bioRxiv v4 page posted 18 May 2026 reports 82% capture of measured causal interaction structure relative to the reproducibility of the perturbation data, while the PubMed/PMC preprint record and indexed full text still display 92%. The currently available records do not, by themselves, establish the exact version mapping of the two figures. Do not label them definitively as v3-versus-v4 without checking archived PDFs/supplements; preserve 82% and 92% as separate source-version claims pending reconciliation. The public code is Nondairy-Creamer/Creamer_LDS_2026, latest located commit bba43302d50a4947804d98b01779856e648237cc. Its README describes an LDS whose learned W matrix is masked by an anatomical connectome, with stimulation filters, and says the connectome was assembled from four NemaNode files downloaded 1 November 2025. It points to the Randi et al. OSF recordings, reports an 80-animal training set and 154 neurons for its fit, and estimates approximately 40 hours for a local fit. This is an external reproduction/falsification target, not a result from Symthaea. Pin the exact paper PDF/version, source commit, four connectome CSV digests, OSF file digests, cell mapping, train/test partitions, STAM calculation and reproducibility-ceiling calculation before scoring.

The zebrafish resources offer complementary tiers:
- Fish1.5 aligns EM and two-photon calcium imaging at single-cell resolution in the same specimen.
- Fish Fire&Wire is a controlled-access draft same-animal whole-brain wiring/activity/behavior resource.
- ZAPBench is an activity forecasting benchmark; structural conditioning requires verified matching anatomy and registration artifacts.

These tiers must remain separate because they support different estimands.

## 3. Direct anatomical connectivity is not effective connectivity

### Required peer-reviewed effective-connectivity baseline

Randi et al., *Neural signal propagation atlas of Caenorhabditis elegans*, Nature 623, 406–414 (1 November 2023): https://www.nature.com/articles/s41586-023-06683-4

The study optogenetically activated single neurons while imaging whole-brain calcium responses. It surveyed 23,433 neuron pairs across 113 animals and 186 of 188 head neurons. Of those measured pairs, 1,310 (6%) met the study's stringent functionally-connected criterion. 11% of those significant functional connections were inhibitory. The analysis distinguishes evidence for a functional connection, evidence for functional non-connection within a prespecified equivalence bound, and insufficient evidence. The paper's term "functional connection" means an **effective connection over all possible paths**, not necessarily a direct monosynaptic connection.

The paper reports rapid dense-core-vesicle/neuropeptide signalling that is not visible in the anatomical connectome and better spontaneous-activity prediction from the measured signal-propagation atlas than from anatomy-only predictions in its experimental setting. These findings are a strong reason to include an explicit extrasynaptic/latent channel in dynamical model comparisons.

Limitations remain important: the animals were immobilized and awake under a transgenic study configuration, the pairwise map is conditioned on the stimulation/recording sampling and statistical detection procedure, and it is not a perfect physiological census. Pin the source artifacts and do not generalize the result beyond its experimental scope.



A crucial peer-reviewed C. elegans reference must be part of the benchmark:

Randi et al., *Neural signal propagation atlas of Caenorhabditis elegans*, Nature 623, 406–414 (1 November 2023): https://www.nature.com/articles/s41586-023-06683-4

The study directly optogenetically stimulated individual neurons while recording whole-brain calcium responses. It surveyed 23,433 neuron pairs across 113 animals, covering 186 of 188 head neurons. The authors reported 1,310 pairs (6%) meeting their stringent functional-connection criterion and classified 11% of those significant functional connections as inhibitory. Their statistical framework distinguishes:
- a functionally connected pair under its criterion;
- a pair supported as functionally non-connected within a prespecified equivalence bound;
- a pair for which evidence is insufficient to make either determination.

The key semantic point is that the paper's functional connections are **effective connections**: signal propagation over all available paths, not just direct monosynaptic anatomical edges. The study also reports rapid extrasynaptic dense-core-vesicle/neuropeptide signalling that is invisible to a conventional synaptic wiring diagram, and that its measured signal-propagation atlas better predicted spontaneous activity than anatomy-only predictions in the reported setting.

This changes the evaluation contract. Keep at least four distinct evidence objects:
1. **Direct structural edge:** a synapse/gap junction supported by anatomical evidence, with polarity/partner/version uncertainty.
2. **Indirect structural route:** one or more paths through the versioned graph.
3. **Measured effective response:** a finite-horizon population response after a controlled intervention; it aggregates every pathway and may include extrasynaptic signalling.
4. **Mechanistic attribution:** a stronger claim identifying direct synaptic, polysynaptic, extrasynaptic or state-mediated mechanisms; requires a separately identified design.

A structurally absent direct edge does not imply zero effective response. A measured response does not prove a direct synapse. A graph path does not guarantee a measurable response in every state. A statistically unresolved response is not evidence of zero effect.

Model architecture should therefore permit separate, explicitly typed channels for:
- anatomically supported synaptic/gap-junction routes;
- polysynaptic propagation through those routes;
- declared extrasynaptic/neuromodulatory or latent coupling terms;
- context/state-dependent gain and intrinsic dynamics.

Any non-anatomical channel must retain its own evidence class and provenance. It must never be silently materialized as an EM synapse or graph edge. Conversely, a channel learned from perturbation outcomes must not be allowed to leak held-out functional targets into structure reconstruction or confirmatory evaluation.

The Randi et al. atlas has limits that must remain visible: its animals were immobilized and awake, stimulation/recording were performed under the study's transgenic configuration, and its functional map is an effective lower bound conditioned on its sampling and response-detection procedure. It is an external perturbation benchmark, not a perfect census of all physiological interactions.

## 4. Pairwise versus combinatorial perturbation evidence

A single-target perturbation atlas estimates pairwise effective responses under its particular intervention and observation protocol. It does not identify arbitrary nonlinear interactions among simultaneous inputs.

Randi et al. explicitly note that patterned simultaneous stimulation of multiple neurons would be needed to probe nonlinear interactions within groups. The next experiment must therefore distinguish:
- pairwise effective-response kernels;
- two-way factorial non-additivity;
- higher-order interaction terms;
- state/context dependence;
- stimulus/indicator observation nonlinearities.

For outcome Y under interventions A and B in state s and horizon t, define:

    I_AB(s,t) = Y(AB,s,t) - Y(A,s,t) - Y(B,s,t) + Y({},s,t)

For three interventions:

    I_ABC = Y(ABC) - Y(AB) - Y(AC) - Y(BC) + Y(A) + Y(B) + Y(C) - Y({})

These are experimental finite-difference contrasts. A nonzero interaction establishes non-additivity for the measured outcome under the declared protocol; it does not by itself identify a molecular interaction, direct synaptic mechanism or causal mediation path.

The synthetic oracle linked from [#9550](https://github.com/Luminous-Dynamics/symthaea/pull/9550) now contains exact threshold, additive, saturation, third-order and state-gated fixtures. Their purpose is to verify factorial estimators and evaluator semantics before biological multi-target data are available.

### Measurement model and uncertainty boundary

Observed calcium fluorescence is not the latent neural state. The benchmark must specify the observation kernel, filtering/indicator kinetics, sampling interval, missing-neuron handling and measurement noise separately from the latent neural dynamics. Where voltage or electrophysiological measurements exist, report them as a distinct modality rather than pooling units.

For each neuron pair or intervention combination, preserve three evidence states:
- **EFFECT_SUPPORTED:** response is supported under the preregistered effect criterion.
- **EQUIVALENT_TO_NULL_WITHIN_BOUND:** equivalence testing supports a response inside the declared tolerance.
- **UNRESOLVED:** neither positive-effect nor equivalence evidence is sufficient.

Failure to reject a response is not evidence of no response. The equivalence margin, multiplicity procedure and independent experimental unit must be preregistered.


## 5. Dataset tiers and evidence eligibility

### Tier A — C. elegans optogenetic perturbation

Primary reference: Creamer, Leifer and Pillow, *Bridging the gap between the connectome and whole-brain activity in C. elegans*, latest located version v4 (18 May 2026), https://doi.org/10.1101/2024.09.22.614271

Use this as an external reproduction/falsification benchmark for connectome-constrained response prediction. The recording cohort spans animals while the anatomical connectome is a species-level constraint; it is not an individually reconstructed connectome for every recorded animal. Therefore this tier may support cross-animal anatomical-template prediction, but it cannot estimate the effect of proofreading a particular animal's connectome on that same animal's recorded responses.

Required pins:
- exact preprint version, dataset and source artifact revisions;
- connectome source/revision and cell identity mapping;
- optogenetic target, dose, timing and response-horizon definitions;
- animal-level train/development/confirmatory partitions;
- preprocessing, cell inclusion, missingness and model/checkpoint digests.

If the source data, code, checkpoint or split cannot be obtained, report NOT_REPRODUCIBLE. Do not silently substitute a reimplementation with different inputs or partitions and call it an exact reproduction.



### Additional 2026 behavior and circuit-model comparators

- **Lee (bioRxiv, 9 September 2026), *What an atlas-fitted connectome can and cannot do: evolutionary search over interneuron stimulation in a whole-body C. elegans model*** (https://doi.org/10.64898/2026.09.06.749731). The reported 302-neuron OpenWorm c302 system couples dynamics fitted once to a signal-propagation atlas, a literature-derived motor layer and a viscoelastic body. An external agent may inject current only into a small whitelist of interneurons. The abstract reports 91% chemotaxis success over four of five training seeds when stimulating five command/steering interneurons; eight sensory-neuron stimulation targets perform no better than chance. Shuffled wiring reduces performance to roughly one fifth. This is a model-based behavioral result, not an in-vivo intervention result or proof that the connectome alone explains sensory-to-command transformation. Reproduce the exact preregistration, seeds, maze geometry, fitted parameters and body/motor assumptions before comparing.
- **Chung and Kim (Scientific Reports, 21 May 2026), *Optimization of connectome weights for a neural network model generating both forward and backward locomotion in C. elegans*** (https://www.nature.com/articles/s41598-026-54384-5). This work optimizes effective synaptic weights against anatomical weight proportions, muscle activity and SMD/neck-bend phase relationships. Its methods add phenomenological proprioceptive feedback connections that are not anatomically identified synapses. Treat it as a behavior/muscle-fit comparator, but score held-out perturbations and behavior separately from objectives used for weight fitting.
- **An (bioRxiv, 24 September 2026), *How sparse can a fly-brain model be?*** (https://doi.org/10.64898/2026.09.19.752860). The FlyLite study compares six connectome-sparsification operators at matched edge budgets across feeding, bitter suppression and grooming tasks and across silent/active states. It reports that weak-synapse pruning preserves 79% of feeding output with 6% of edges in its setting, but task-aware pruning loses held-out inhibition early and task/state failure modes differ. This motivates a separate task- and state-conditioned connection-criticality benchmark; it does not justify a universal pruning threshold.
- **Liu et al. (bioRxiv, 24 September 2026), *Connectome-based biophysical modeling of a figure-ground discrimination circuit*** (https://doi.org/10.64898/2026.09.17.752514). The authors use Drosophila connectomics and morphologically detailed biophysics to model T4a-vCH interactions and motion/figure-ground discrimination. Treat this as a circuit-specific biophysical comparator, not a whole-brain dynamics result.

These studies imply that model comparisons must factor anatomy, effective physiological coupling, cell-intrinsic dynamics, proprioceptive/body feedback, network state and task. They also require task/condition holdouts that were not used to fit parameters.

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

## 6. Formal prediction contract

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

## 7. Frozen evaluation tasks

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

## 8. Baseline matrix

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

## 9. Leakage and statistical contract

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

## 10. Adversarial fixtures

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

## 11. Primary metrics

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

## 12. Promotion gates (research goals, not results)

- Reproduce the published C. elegans baseline on its exact versioned data, split and metric before claiming improvement.
- Demonstrate structural value over activity-only, spatial-only and matched-capacity unconstrained models on preregistered held-out targets/animals, with cluster-aware intervals and a predeclared practical-effect threshold.
- Preserve any advantage over degree-/spatial-/cell-type-preserving shuffled-connectome controls.
- Show calibration and finite-horizon trajectory fidelity, not only pooled correlation.
- Zero cross-animal, intervention-target, lineage and future-label leakage on exhaustive synthetic mutation fixtures.
- No direct-synaptic or mechanistic claim from total system response alone.
- No biological SOTA claim from synthetic fixtures, one-species reproduction or controlled-access unpublished data.
- Reconstruction-change claims require same-specimen pairing and independent anatomical adjudication.

## 13. Implementation sequence

1. Define a versioned dataset/task eligibility manifest and run-record schema.
2. Build exact synthetic fixtures for direct, delayed, indirect, recurrent and confounded perturbation effects.
3. Independently reproduce the C. elegans study as an external falsification baseline.
4. Establish the Fish1.5 same-specimen adapter and reproduce one published structure/function analysis.
5. Add matched-capacity graph-constrained, graph-shuffled and activity-only comparisons.
6. Integrate observation/perturbation results with #9501, #9344, #9379 and #9255; retain #9462 as a separate stopping-assurance contract.
7. Pursue Fish Fire&Wire only through its official collaboration process and obey snapshot, access, authorship and publication restrictions.
8. After retrospective reproduction, preregister a prospective perturbation/behavior evaluation where technically and ethically feasible.

## 14. Integration and claim ceiling

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
