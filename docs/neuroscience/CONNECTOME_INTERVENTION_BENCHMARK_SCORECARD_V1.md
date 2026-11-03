# Connectome Intervention Benchmark Scorecard V1

**Status:** contract and benchmark protocol; no biological result is claimed by this document.

**Scope:** blinded, intervention-conditioned comparison of competing connectome-constrained and non-connectome neural models. This scorecard complements the synthetic perturbation oracle (#9584), optimal experiment selection (#9589), combinatorial perturbation work (#9575), state-conditioned edge criticality (#9549), and perturbation-grounded dynamics (#9522).

## 1. Scientific question

Does an independently versioned anatomical reconstruction improve prediction of held-out neural and behavioural responses to controlled interventions, beyond activity-only, pairwise, linear, rewired and phenomenological alternatives?

The benchmark is not an in-sample activity fit contest. A model must predict outcomes for intervention targets, combinations, temporal schedules, animals/sessions or network states that were withheld before preprocessing and model selection.

A model may abstain or remain unresolved. A large predictive response does not prove a direct synapse; a measured effective response may include recurrent paths, cell-intrinsic dynamics, extrasynaptic signalling, shared inputs and observation-kernel effects.

## 2. Evidence tiers and claim ceilings

Every run must use exactly one tier:

| Evidence tier | Permitted claim ceiling | What it can establish |
|---|---|---|
| `synthetic_known_answer` | `synthetic_only` | Implementation and metric correctness against declared equations and known answers |
| `observational_structure_function` | `observational_prediction_only` | Held-out prediction under observed activity/stimulus, not causal intervention efficacy |
| `in_silico_intervention` | `model_counterfactual_only` | Consequences inside a declared model and intervention simulator |
| `measured_intervention` | `measured_intervention_prediction` | Prediction against actual measured intervention outcomes in the declared assay and population |

A schema-valid manifest is not a biological validation result. A synthetic oracle never graduates to measured evidence through a passing unit test. An in-silico intervention never becomes an empirical intervention by being compared with another simulator.

The example manifest is deliberately incomplete and contains null digests, no selected biological intervention, no run result and no reproduction claim.

## 3. Data and identity contract

Each run binds:
- source, dataset ID/version, immutable reference, source manifest SHA-256, rights state and operation-scoped permission receipt;
- taxon, connectome/reconstruction version, neuron-ID namespace, row-axis and column-axis digests, canonical identity-map digest and structural edge-manifest digest;
- exact annotation/alias mapping and the semantics of chemical/electrical edges;
- observation modality, sensor/indicator identity, units, sampling interval, preprocessing manifest and observation-kernel parameters.

**Latent neural dynamics and the observation model are distinct objects.** A calcium-like low-pass kernel, indicator nonlinearity, registration uncertainty or sampling process must not be fitted away and silently described as neural dynamics. The observation-kernel identity and parameters must be pinned separately.

Direct structural edges, indirect graph routes, measured effective responses, and mechanistic attribution (direct, polysynaptic, extrasynaptic/latent, shared-input or unresolved) are different evidence channels. Do not add an inferred latent coupling to the anatomical graph.

## 4. Leakage-resistant split protocol

Random timepoint splits are exploratory only. The primary test split must hold out the unit relevant to the intended claim: animal, session, target neuron, target combination, intervention schedule, state/context, site, study or connectome reconstruction.

Freeze train/validation/test membership before any preprocessing, feature selection, hyperparameter search, connectome pruning, alias resolution or model selection that can use outcome values. The run must carry:
- split manifest digest and separate train/validation/test membership digests;
- declared generalization axes and grouping keys;
- a cryptographic overlap-audit artifact;
- explicit booleans for train-only preprocessing, train-only feature selection, blind test model selection and target-identity split before preprocessing.

A custom verifier must reject:
1. identity overlap across partitions on any declared held-out axis;
2. target combinations that leak across partitions when the target combination is the generalization axis;
3. preprocessing, threshold selection or feature selection fitted on test observations;
4. a test set opened during candidate selection;
5. future-time windows that overlap or violate the declared temporal gap.

Within-animal repeated-measures designs may be valid for a specific within-subject estimand, but are not evidence of cross-animal generalization. State the estimand rather than calling all partitions "unseen."

### Pseudonymous split membership and paired-unit manifests

For executable disjointness checks, each independent evaluation unit is represented by a dataset-scoped HMAC-SHA256 token. The HMAC key itself is never included in the run manifest; only a commitment digest is recorded. Tokens use the pinned `UNIT-TOKEN-V1` canonicalization and are safe to compare across the train/validation/test partitions without publishing raw animal, session or target identifiers.

The semantic verifier recomputes each partition digest as:

`SHA256(sorted lowercase token strings, UTF-8, newline-delimited with one trailing newline)`

It rejects duplicate tokens within a partition, any overlap between train/validation/test, membership-count mismatches and a digest that does not match the token list. Paired comparisons carry their own hashed independent-unit tokens and recomputed membership digest. This lets the verifier check the actual paired membership count instead of trusting `n_independent_units` and a digest string independently.

A complete run requires at least two training units, one validation unit and two test units. The example remains incomplete and has empty token arrays and null commitments; no unit identity has been fabricated.

## 5. Competing model and baseline matrix

At minimum, include these model families where data and task semantics make them applicable:

| Family | Purpose |
|---|---|
| Activity-only / stimulus-only | Tests whether anatomy adds value at all |
| Linear LDS / VAR / GLM | Strong low-complexity dynamical baseline |
| Pairwise effective-response kernel | Uses measured perturbation responses without claiming direct synapses |
| Graph-constrained state-space | Constrains latent propagation using structural routes |
| Cell-type-tied mechanistic | Adds declared cell identity and transfer assumptions |
| Anatomy + separate extrasynaptic/latent channel | Tests non-anatomical coupling without rewriting the graph |
| Pairwise-additive predictor | Negative control for higher-order interactions |
| Nonlinear higher-order/state-conditioned | Tests interaction, saturation and context dependence |
| Degree/type/spatial-preserving rewired graph | Tests whether exact topology matters beyond graph statistics |
| Phenomenological CPG / rule controller | Appropriate locomotor engineering comparator |
| Connectome-constrained CPG / hybrid | Candidate mechanistic model |

For the Drosophila walking vertical slice, keep Pugliese et al. (Nature version of record, 28 October 2026) and Sapkal et al. (bioRxiv v2, preprint) as separate, versioned hypotheses. The former reports a source-grounded three-interneuron minimal rhythm core and experimentally supported DNb08 pathway; the latter proposes distributed per-leg CPG modules and central/peripheral coordination motifs. Neither source justifies treating its circuit as universal across taxa. Rhythm production, inter-leg coordination and stable embodied gait are separate outcomes.

For C. elegans, the Randi et al. 2023 optogenetic signal-propagation atlas is an effective-connectivity comparator, not a direct anatomical synapse catalogue. Creamer et al. is a bioRxiv preprint, not a peer-reviewed benchmark result. Its v1 abstract (posted 2024-09-23) reports 92% of experimental train–test reproducibility; v4 (posted 2026-05-18) reports 82%. These figures are versioned reproducibility-relative claims, not absolute causal-edge accuracy. Do not quote either as “92% connectivity accuracy.” Any comparison must pin the exact preprint version, code revision, source connectome, ground-truth/measurement split and scoring definition, then independently reproduce that version. Until both versions are reconciled, record them as separate hypotheses/results, not a single SOTA number.

Every baseline must bind code HEAD, configuration digest, training-data scope, observation model, parameter count and training/evaluation budget. If resources differ, mark the comparison as an explicit budget/performance trade-off; do not call it matched.

## External behavioural benchmark interoperability

Use the upstream `brandoncho369/flybench` benchmark rather than building a duplicate Drosophila behavioural suite. The upstream source snapshot is pinned in the interoperability bridge to commit `498ceb5d8b69d37ed1b5a3b83bc023e0d36234e8` (2026-10-27), package version 1.0.1, its Draft 2020-12 result schema and per-task YAML blobs. FlyBench currently defines 37 cited tasks (five core, 32 hard), supports FlyWire v783 and MaleCNS, requires a simulator adapter returning spike times and row-order neuron IDs, and includes rewired-connectome controls. Its own governance states that the score measures only the cited manipulations, not general biological realism.

A FlyBench score is an external benchmark result, not a substitute for this intervention benchmark. Import it with a provenance receipt containing the raw report digest, canonicalized report digest, upstream commit, result-schema digest, exact task-catalogue digests, simulator/configuration identity and connectome alias/pin metadata. Do not treat the report's `verified: true` as a cryptographically authenticated attestation; it is a source-generated CI claim unless accompanied by an independently verifiable CI receipt.

Historical result files in upstream `results/` are version-sensitive: at least one committed report declares FlyBench 0.2.1 while the current source package at the pinned snapshot declares 1.0.1; those old reports also contain non-standard bare `NaN` tokens. A bridge may canonicalize bare non-finite tokens to JSON null **only in diagnostic fields**, preserve the raw-byte digest and count substitutions, and reject them in scores, checks, confidence intervals or other ranking-critical fields. Reports from a version other than the pinned 1.0.1 must be classified legacy/unranked, never silently compared to current rows.

The FlyBench adapter interface (`Simulator(connectome, params)`, `run(duration_ms, stimuli) -> SimResult`) is the first interoperability target. An adapter must honour deterministic seeds, reset state between runs, exact neuron row ordering, stimulus timing/rates and declared capabilities. A Symthaea wrapper is not yet implemented by this scorecard; passing the importer or a synthetic fixture does not claim that Symthaea has reproduced any FlyBench task.

## 6. Intervention design and active selection

The intervention pool is frozen and hashed before test outcomes are opened. Each selected intervention binds target IDs and namespace, manipulation, waveform, amplitude/units, duration, onset, frequency, duty cycle, protocol digest, randomization block, control role, feasibility and safety audit.

The mandatory selector matrix includes:
- random feasible selection;
- degree and centrality;
- expert-prior selection;
- largest predicted response;
- Fisher-information D-optimal and A-optimal designs for parameter identifiability;
- expected information gain (EIG) for discrete hypothesis discrimination;
- cost-normalized EIG;
- joint Pareto selection over information, predictive value, cost and constraints.

**Parameter identification and model discrimination are not the same objective.** A waveform that best estimates a parameter in one assumed model need not best separate competing models. Report both objectives and include a fixture where they choose different interventions.

Ogino et al. (eLife version of record, 13 October 2026; DOI 10.7554/eLife.110030.3) provide a current optimal-perturbation comparator for system identification. EIG selection (#9589) addresses discrete hypothesis uncertainty. Neither score establishes that a candidate hypothesis is true; only prospective outcomes can update that evidence.

All selectors receive the same intervention and compute budget. Feasibility checks must fail closed; do not silently clip invalid waveform amplitudes or durations.

### Protocol sequences and state-conditioned experiments

Interventions now optionally support ordered multi-step protocols. Each step binds its order index, target IDs, manipulation, waveform, inter-step interval and optional state-context ID. The parent intervention's target list must equal the union of step targets. The semantic verifier checks contiguous order indices, target-set consistency and the state-measurement receipt. Simultaneous combinations remain represented by one intervention with multiple target IDs.

This contract makes it possible to preregister order-sensitive and state-gated perturbations rather than flattening them into an unordered target set. It does not imply that the assay can physically deliver the protocol; feasibility and safety receipts remain mandatory.

The metric catalogue now explicitly accepts EVSI, D-/A-optimal Fisher gains, persistent-excitation rank and condition number. These remain separate metric families and units; do not combine them into one unqualified “information” score.

## 7. Metrics: keep the scorecard vector-valued

Do not collapse the benchmark into one composite scalar. Keep at least the following metric families separate:

| Metric family | Required output |
|---|---|
| Predictive accuracy | Held-out negative log predictive density; trajectory RMSE/normalized RMSE where pointwise comparison is meaningful |
| Parameter identification | Parameter error, posterior coverage, Fisher information / covariance reduction |
| Calibration | Predictive interval coverage and sharpness by held-out stratum |
| Intervention discrimination | Expected and realized information gain in bits; decision/selection regret |
| Mechanistic attribution | Agreement on direct / polysynaptic / extrasynaptic / shared-input / unresolved categories, with evidence ceiling |
| Robustness | Worst-condition, worst-target and worst-animal performance; missingness and state-shift degradation |
| Cost | Training and inference wall time, function/Jacobian evaluations, memory, energy and experiment count |
| Reliability | Numerical failures, non-finite predictions, invalid interventions and protocol violations |

The primary estimand, metric ID, metric version, direction, units and aggregation must be frozen before evaluation. Do not swap a pointwise trajectory metric for an aggregate correlation after seeing outcomes.

For measured intervention results, the independent unit is usually the randomized animal/session/assay block, not each neuron-timepoint. Use cluster-aware uncertainty. Report both pooled and per-stratum results; do not let a large easy stratum hide a failure in a smaller important one.

### Exact scorecard cohort binding

Each scorecard now carries its own `unit_tokens` and `unit_manifest_sha256`. Measured scorecards require at least two pseudonymous tokens. The verifier recomputes the scorecard membership digest, checks that every token belongs to the held-out test partition, and requires candidate, baseline and paired-comparison token sets to be identical before reconciling mean candidate-minus-baseline arithmetic. Equal sample counts alone are not accepted as evidence of matched units.

This prevents a candidate and baseline computed on different animals/sessions/targets from being called paired merely because both report the same `n`.

## 8. Paired comparisons and uncertainty

Every candidate-vs-baseline comparison binds:
- paired-unit membership manifest SHA-256;
- independent unit count;
- signed candidate-minus-baseline mean difference;
- interval method and confidence level;
- minimum meaningful effect and non-inferiority margin;
- stratum ID and explicit outcome.

Allowed outcomes: `outperforms_by_margin`, `non_inferior`, `regresses_beyond_margin`, `uncertain`, `not_assessed`, `incomplete`.

Do not convert missing, failed or abstained scores to zero. Do not claim superiority from a candidate point estimate alone. A claimed absolute threshold must be cleared by the full preregistered confidence interval. If intervals overlap the relevant margin, the outcome is uncertain.

## 9. Proposed promotion targets (aspirational, not measured)

Freeze thresholds only after baseline variance, sample size, reliability and measurement quality are assessed.

| Objective | Initial research target |
|---|---:|
| Held-out predictive value over strongest non-connectome baseline | ≥10% relative improvement on preregistered intervention strata |
| Predictive interval calibration | 90–95% nominal coverage band, evaluated with cluster-aware uncertainty |
| Intervention selection | Better realized information gain and lower decision regret than random/degree/centrality/expert baselines at equal cost |
| Anatomy contribution | Incremental prediction survives degree/type/spatial-preserving rewired controls |
| Higher-order effects | Detect only preregistered interaction contrasts; false-discovery control across searched combinations |
| Robustness | No unexplained collapse on held-out target/animal/state strata |
| Reliability | Zero non-finite predictions, invalid interventions or unrecorded numerical failures |
| Efficiency | Publish complete accuracy/calibration/compute Pareto front, not one speed figure |

These are hypotheses, not results or universal neuroscience acceptance thresholds. A connectome model that fails a valid stratum must retain that result. Complexity is justified only by out-of-sample explanatory or predictive value.

## 10. Mandatory negative controls

- shuffled neuron IDs and row/column permutations without canonical remapping;
- degree-, cell-type- and spatial-preserving graph rewires;
- no-anatomy / activity-only comparator;
- pairwise additive predictor under a known higher-order interaction oracle;
- no-op / sham intervention;
- random and largest-response selector;
- latent/common-input channel disabled/enabled as a separately typed model condition;
- observation-kernel-only mismatch;
- target identity leakage injected across train/test;
- preprocessing fitted on the full dataset;
- intervention selected after opening test outcomes;
- non-finite prediction and solver non-convergence;
- unequal-budget comparator mislabeled as matched.

The verifier must reject the deliberately invalid versions. A benchmark without adversarial leakage and budget controls is not qualified.

## 11. Run completion and claim promotion

A run may be `planned`, `incomplete`, `running`, `complete`, `failed` or `invalid`. A completed run requires immutable source/data, connectome identity, split, model, intervention, environment, outputs and receipt digests. It also requires all declared baselines, scorecard strata, paired comparisons and calibration outputs. Missing evidence is not a pass.

The strict JSON Schema is `CONNECTOME_INTERVENTION_BENCHMARK_MANIFEST.schema.json`. The example `CONNECTOME_INTERVENTION_BENCHMARK_EXAMPLE_V1.json` is intentionally incomplete. Schema validity checks shape; a separate semantic verifier must check cross-field references, split disjointness, baseline coverage, paired arithmetic, interval consistency, resource equality and evidence-tier claim ceilings.

## 12. External references and current baseline context

- Randi et al., *Neural signal propagation atlas of Caenorhabditis elegans*, Nature (2023): https://www.nature.com/articles/s41586-023-06683-4
- Ogino et al., *Designing optimal perturbation inputs for system identification in neuroscience*, eLife (2026-10-13): https://doi.org/10.7554/eLife.110030.3
- Golan, Schütt & Kriegeskorte, *Making models disagree to learn how brains compute*, Nature Reviews Neuroscience (2026-08-28): https://doi.org/10.1038/s41583-026-01070-0
- Pugliese et al., *Connectome simulations identify a central pattern generator for fly walking*, Nature (2026-10-28): https://doi.org/10.1038/s41586-026-11195-y
- Sapkal et al., *Central versus peripheral neural control of a coordinated walking pattern in Drosophila*, bioRxiv v2 (2026-04-29): https://doi.org/10.64898/2026.04.29.721658

External solver implementation and synthetic oracle status must not be confused with biological validation. The benchmark remains source- and assay-specific.

## 14. November 2026 field update: versioned baselines and the next frontier

This section records the versions and claim limits as of 2026-11-03. Preprints, in-silico results and measured intervention outcomes must not share an unqualified leaderboard.

### C. elegans: reproduce the exact Creamer LDS version before claiming to exceed it

The latest surfaced bioRxiv version is **v4**, posted 2026-05-18; its abstract reports that the connectome-constrained model captures causal interactions at **82% of perturbation-data reproducibility**. Earlier v1 (2024-09-23) and v3 (2025-09-26) report **92%**. These are versioned, reproducibility-relative claims, not absolute “82%/92% accuracy.” Do not pool them or compare them without pinning exact code, connectome, data, split and metric versions.

The v3 full-text methods specify 110 animals split into 80 training and 30 held-out test animals; a latent linear dynamical system with non-zero dynamics weights only on chemical/electrical connectome edges (9.4% of all possible pairwise weights); a diagonal single-stimulated-neuron input filter of 45 timepoints (22.5 s at 2 Hz); and Kalman filtering/smoothing plus EM parameter estimation. The reported measured STAM correlation was 0.15, versus a train/test STAM correlation of 0.16, yielding relative correlation 0.92. These raw and normalized values must always be reported separately.

The v3 authors explicitly identify limitations: pairwise perturbations do not measure nonlinear interactions among groups; fixed-point LTI dynamics cannot represent bistability or limit cycles; calcium is only a proxy for neural activity; learning, neuromodulation and state-based processing are outside the model. Extrasynaptic signalling also remains a live alternative for some responses.

**Required reproduction gates:** run v3 and v4 as separate frozen hypotheses; recover their exact train/test protocol, code, connectome and data; reproduce raw and normalized scores; include original-connectome and identity-shuffled controls; report per-animal paired scores and uncertainty. Matching only a relative score is not a reproduction.

The independent 2023 Nature signal-propagation atlas remains a key functional comparator: it directly stimulated neurons and measured 23,433 neuron pairs (66% of possible pairs), covering 186 of 188 head neurons. It demonstrates that extrasynaptic signalling can generate responses not explained by direct anatomical edges alone. This does not make anatomy irrelevant; it means structural connectivity and effective signalling are separate evidence channels.

### Drosophila: evaluate more than rhythm generation

Pugliese et al. (Nature version of record, 2026-10-28) report a three-interneuron minimal rhythm-generating core downstream of DNg100, producing motor rhythms across six legs and four connectome datasets, plus experimental optogenetic support for a separate DNb08 descending pathway. Treat this as a strong comparator for a specific fly walking circuit, not a universal CPG claim.

Benchmark four levels independently:
1. **Circuit sufficiency:** reproduce rhythms across the declared connectome datasets and parameter uncertainty.
2. **Circuit necessity:** preregister ablation/silencing controls and quantify loss or compensation.
3. **Coordination:** predict inter-leg phase-offset distributions, not merely the presence of oscillation.
4. **Embodied behavior:** predict stance/swing transitions, kinematic trajectories, stability, perturbation recovery and task success.

Use FlyBench's cited behavioural tasks and simulator-adapter protocol rather than creating an incompatible duplicate. Passing a CPG rhythm task does not imply closed-loop locomotion competence or success across all FlyBench tasks.

### Critical perturbations: make competing models disagree

Golan, Schütt and Kriegeskorte's August 2026 review motivates selecting inputs where competing models make divergent predictions, tempered by a prior over ecologically valid stimuli. The intervention analogue is a **critical perturbation**: select a feasible target, combination, temporal order, waveform and state context that separates mechanistically distinct models.

Compare four policies under equal cost and safety constraints: random/standard, parameter identification (Fisher information and D-/A-optimality), model discrimination (EIG/model disagreement), and decision-focused design (EVSI/regret). They have different objectives and need not choose the same experiment.

### Next benchmark tier: higher-order, state-conditioned model recovery

The next synthetic and prospective task family should make every pairwise model insufficient by construction:
- Generate pairwise-additive, three-way interaction, state-gated, sequence-dependent and shared-latent-input ground-truth systems.
- Construct at least one fixture where every single-target and pairwise response marginal matches while a three-target response differs.
- Split by held-out target combinations, intervention sequence and state context before preprocessing or model selection.
- Compare pairwise-additive, higher-order, recurrent/state-space, anatomical-only, anatomy-plus-latent and identity/degree/type/spatial-rewired models.
- Run model-recovery confusion matrices and calibration across noise, missingness and observation-kernel regimes.
- Select interventions sequentially and score realized information gain, EVSI, parameter error and decision regret separately.

Predicting unseen single-neuron interventions is pairwise extrapolation, not proof of higher-order causal understanding.

### Aspirational promotion targets for the new tier

| Test | Initial target |
|---|---:|
| Synthetic model-family recovery | ≥90% balanced accuracy in declared identifiable regimes |
| Higher-order interaction detection | ≥80% power at a preregistered minimum effect, with family-wise false-discovery control |
| Pairwise-matched / triplet-different control | Correctly reject the pairwise-additive model without leakage |
| Held-out target-combination prediction | ≥10% relative improvement in a preregistered proper score over the strongest non-connectome baseline |
| Predictive uncertainty | Calibration reported by noise, state and missingness strata |
| Experiment selection | Lower realized decision regret and higher realized information than random/degree/expert baselines at equal cost |
| Model robustness | Publish worst-condition performance, abstentions and failure rates |

These are research hypotheses, not achieved results or universal acceptance thresholds. Promotion remains evidence-tier specific.


## 15. Baseline coverage gate and run-completion verifier

A `complete` run is stricter than an incomplete research plan. It requires:
- at least four pinned model implementations, including a candidate, at least one non-connectome baseline and a negative control;
- an explicit baseline coverage manifest, with at least five required families and a hash of the coverage decision; every required family must have a pinned model or a specific not-applicable reason;
- preregistered selected interventions, a frozen candidate-pool digest and an independently hashed feasibility audit;
- sham, single-target, combined-target, rewired-graph and activity-only controls;
- a qualified, hashed train/validation/test split with positive leakage checks;
- non-empty scorecards, paired comparisons and calibration results;
- at least two independent units for inferential paired comparisons, with a paired-unit manifest digest;
- pinned source, connectome, observation kernel, preprocessing, code, configuration, environment, raw outputs and run receipt.

For measured-intervention evidence, the manifest additionally requires `rights_state = documented_permitted` and a permission-receipt digest. The source payload and operation scope still need independent review; the digest alone is not permission.

The JSON Schema enforces local shape and completion prerequisites. A separate semantic verifier must enforce cross-object references and arithmetic: model IDs resolve in scorecards, metric IDs resolve in the metric catalogue, required baselines map to models or justified exclusions, model comparison uses matched scope/budgets, paired membership resolves to the same units, reported mean differences reconcile, confidence intervals are ordered, strata use the declared taxonomy and missing/abstained results are never converted to zero. Do not promote a run based on JSON Schema validation alone.

The included example remains `incomplete` with `not_run` results and null artifact digests. It is a contract fixture, not a completed run.
