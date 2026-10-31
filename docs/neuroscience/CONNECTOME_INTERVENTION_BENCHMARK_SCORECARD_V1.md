# Connectome intervention benchmark and scorecard (v1)

**Status:** protocol specification; no biological benchmark result is claimed.  
**Related work:** issue #9587; perturbation oracle PR #9584; experimental design issue #9575; task/state criticality issue #9549; Randi/C. elegans programme issue #9522; Creamer reproduction audit issue #9578.

## 1. Benchmark thesis

Evaluate a model by what it predicts under interventions it has not seen, not by how well it reconstructs activity it was trained on. The central question is:

> Does anatomical structure provide reproducible, calibrated, mechanistic predictive value for a held-out perturbation, beyond what can be achieved by matched activity-only, generic dynamical, and topology-null models?

The benchmark therefore separates five questions that must not be collapsed:

1. **Prediction:** does the model forecast the complete response trajectory in held-out conditions?
2. **Causal validity:** does it predict the difference between intervention and paired sham/control, rather than exploit spontaneous correlation?
3. **Mechanism:** can it distinguish direct anatomical, polysynaptic, extrasynaptic/latent, common-input, and measurement explanations?
4. **Uncertainty:** does it know when its prediction or attribution is unreliable?
5. **Experiment value:** does it select the next perturbation that most efficiently distinguishes plausible mechanisms?

A graph that improves in-sample fit but fails on held-out targets, animals, or states has not demonstrated a functional advantage. A model that predicts outcomes but cannot distinguish observational equivalence must report that ambiguity rather than invent a mechanism.

## 2. Current research and industry comparison

| Reference point | Demonstrated capability | What it does not establish | Benchmark implication |
|---|---|---|---|
| Randi et al., *Nature* (2023), C. elegans | Optogenetic activation and whole-brain calcium imaging across 23,433 neuron pairs, 113 animals, and 186/188 head neurons; measures signed, time-dependent effective responses, including extrasynaptic effects. | A measured effective response is not automatically a direct monosynaptic edge or a uniquely identified mechanism. | Use perturbation-grounded outcomes; score direct anatomy separately from effective response and include extrasynaptic / common-input alternatives. |
| Duan, Dong & Fiete, NeurIPS 2025, Drosophila head-direction circuit | Connectome-constrained dynamics with learned cell-type-level parameters; recovers attractor-like computation and uses in-silico ablations. | A focused circuit/task result is not a cross-dataset blind benchmark or proof that the same model generalizes to arbitrary circuits. | Include a cell-type-parameter-tied mechanistic comparator and held-out physiological interventions. |
| MICrONS Consortium, *Nature* (2025) | Large-scale co-registration of dense calcium responses from about 75,000 neurons with an EM reconstruction of over 200,000 cells and roughly 0.5 billion synapses. | Large paired structure/function data are not equivalent to a complete randomized intervention ground truth. | Use as a scale and structure-function generalization tier; label observational stimulus data correctly. |
| CausalBench, *Communications Biology* (2025) | Real-world perturbational single-cell data, distribution-based interventional metrics, and explicit method comparisons. | It is a gene-regulatory inference benchmark, not a neural connectomics leaderboard. | Borrow the emphasis on real interventional outcomes and distributional scoring, not its domain-specific scores. |
| 2026 connectomics review | Identifies missing dynamics, sign/weight uncertainty, neuromodulatory state, plasticity, glia, and behaviour as core limits of static connectomes; advocates closed-loop physiology/behaviour validation. | A review is not a quantitative baseline. | Benchmark the complete evidence chain, including state and observation model, rather than graph fit alone. |

**Positioning:** current leading work already combines connectome topology with biologically grounded parameter learning and in-silico perturbation. Our proposed differentiator is a transparent, multi-tier, blind evaluation contract that rewards out-of-distribution intervention prediction, calibrated uncertainty, mechanism discrimination, and prospective experiment efficiency. This is a target for research, not a claim of superiority today.

## 3. Evidence tiers and claim ceilings

Never pool these tiers into one leaderboard. Report each separately.

| Tier | Evidence | Permitted claim ceiling |
|---|---|---|
| T0 — synthetic known answer | Frozen synthetic fixtures with exactly specified dynamics and outcomes. | Correct implementation / scoring semantics only. No biological validity. |
| T1 — observational structure/function | Anatomy paired with spontaneous or stimulus-evoked activity, without randomized causal interventions for the scored outcome. | Held-out predictive association in the declared dataset and domain. |
| T2 — in-silico intervention | Model-predicted node/edge lesions, stimulation, or rewiring in a simulated circuit. | Internal model consequences under declared assumptions only. |
| T3 — measured biological intervention | Held-out optogenetic, genetic, pharmacological, or other measured intervention with appropriate controls. | Predictive performance for the measured intervention regime; mechanistic claims require independent identification evidence. |
| T4 — prospective experiment selection | Model-selected interventions run after predictions and selection policy are frozen. | Prospective value under the tested budget, organism, task, and protocol. |

A higher tier does not make a lower tier unnecessary: T0 qualifies the evaluator; T1 tests scaling and observational generalization; T2 tests model-consistent counterfactuals; T3 tests biology; T4 tests whether the model improves science per experiment.

## 4. Dataset strata and admissible use

### 4.1 C. elegans perturbation atlas

Randi et al. is the first-line T3 anchor: whole-brain calcium responses after targeted optogenetic activation. Use the individual animal / session and stimulus records where the source data permit; do not score only a thresholded functional-connectivity adjacency matrix if time-resolved traces are available. Preserve three distinct labels: anatomical direct edge, measured effective response, and unresolved pair. A response without a direct anatomical edge is not a false positive by definition; it may reflect a multi-hop or extrasynaptic route.

Use the paper's conservative three-way inference semantics where available:
- **Functionally connected:** the response meets the source's predeclared evidence criterion.
- **Functionally non-connected within an equivalence bound:** the measured response is supported as sufficiently close to the null under the declared equivalence margin and statistical protocol. This is not a claim that an anatomical edge is absent.
- **Insufficient evidence:** the available observations do not establish either of the first two classes.

A conventional non-significant difference test is not, by itself, an equivalence test. Never recode every non-significant pair as functionally non-connected. Preserve source-level multiplicity correction, empirical null construction, equivalence margin, sample count and adjudication metadata. A missing, underpowered, unmeasured or ambiguous observation remains insufficient evidence. Model abstention is a separate prediction outcome and must not be conflated with the reference insufficient-evidence class.

### 4.2 Creamer LDS baseline

Treat the public Creamer LDS implementation as a separate reproduction track. Do not call the no-argument `exp_test.yml` run a connectome-constrained reproduction: the source audit in #9578 found that it selects the `not_synaptic` mask and a small test configuration. The `synaptic` Slurm configuration is only a candidate full-run configuration until the exact manuscript/supplement mapping is verified. Preserve the reported 82% / 92% version discrepancy as unresolved until the source versions and artifact provenance settle it. A score enters the comparison only with materialized and hashed inputs, exact split membership, mask support, metric definition, environment, and output report.

### 4.3 Drosophila circuit and behaviour

Use circuit-specific connectomes where both physiology and independent genetic/optogenetic validation exist. Keep circuit/task identity explicit. Include cell-type-tied parameter models as a strong comparator. Where available, connect the neural prediction to a behavioural or embodied outcome; do not treat rhythmic simulated activity as proof of locomotor function.

### 4.4 MICrONS scale tier

Use the MICrONS co-registered structural-functional resource for large-scale prediction, representation, and data-efficiency experiments. Distinguish visual stimulus-evoked recordings from intervention trials. Any future intervention subset must be listed as such, with its own evidence tier and split manifest. Record materialization/version identifiers because annotations and proofreading can change.

### 4.5 Cross-species use

Never combine species into one metric denominator. Report per-species results first. Cross-species transfer is a separate zero-shot or few-shot task with held-out species, cell-type mapping provenance, and explicit domain-adaptation budget. Similar names or graph motifs do not establish homologous mechanisms.

## 5. Required model comparator matrix

Every primary evaluation must report all applicable families below. Comparators receive the same training observations, preprocessing, split, tuning budget, and prediction horizon. If a model family is genuinely inapplicable, mark it `not_applicable` with a reason; do not silently omit it.

| ID | Model family | Purpose / failure mode controlled |
|---|---|---|
| B0 | Sham / training-set mean / per-target mean | Establishes the no-mechanism floor. |
| B1 | Direct-edge structural propagation | Tests whether simple anatomy and synapse-count weighting explain the response. |
| B2 | Non-graph linear dynamics: VAR / LDS / GLM | Tests whether graph topology adds value beyond generic temporal dynamics. |
| B3 | Activity-only nonlinear state-space model | Tests whether nonlinear temporal capacity, without connectome constraints, is sufficient. |
| B4 | Fixed connectome topology + shared cell-type parameters | Strong connectome-constrained mechanistic comparator; mirrors the parameter-sharing strategy used in recent circuit work. |
| B5 | Connectome-constrained model with the candidate's full dynamics / latent channels | Candidate method. |
| B6 | Capacity- and compute-matched activity-only counterpart | Controls for extra parameters and optimization effort. |
| B7 | Degree-, cell-type-, and edge-count-preserving rewired connectome | Tests whether the specific biological wiring matters beyond coarse graph statistics. |
| B8 | Edge-weight/sign shuffled or uncertainty-resampled connectome | Tests dependence on uncertain synaptic weights and signs. |
| B9 | Common-input / latent-confound model | Tests whether shared drive can explain apparent pairwise response. |

Report parameter count, training steps, optimizer/tuning budget, compute time, memory, and seed count per model. Use identical held-out data and outcome definitions. A control that is intentionally weaker or receives less optimization is not a fair comparator.

## 6. Split protocol: prevent leakage before fitting

Create split manifests before any feature selection, normalization, neuron filtering, hyperparameter selection, or imputation. Every learned preprocessing object must be fitted on training data only, then frozen and applied to validation/test data.

Required outer evaluation regimes:

1. **Leave-animal/session-out:** test transfer to unseen animals or sessions. The independent unit is the animal/session, not a cell or time point.
2. **Leave-target-neuron-out:** test response prediction for stimulation targets absent from model fitting. Target identity must not leak through target-specific embeddings or statistics.
3. **Leave-perturbation-combination-out:** hold out simultaneous or multi-target patterns, including higher-order combinations.
4. **Temporal extrapolation:** train on earlier epochs/trials and predict later response windows; no random frame split.
5. **State/context shift:** train in one declared physiological context and test in another. State labels and state estimation must be available at prediction time.
6. **Dose / duration / pulse-order holdout:** evaluate unseen stimulation strengths, duration, and order. Separate dose extrapolation from interpolation.

Random cell/time-point splits may be reported as exploratory diagnostics but cannot support a generalization claim. If a dataset contains repeated observations for the same pair, all correlated records must remain in the same outer fold unless the stated task explicitly tests within-pair repeatability.

Minimum provenance per split: split strategy, unit of independence, train/validation/test IDs or privacy-preserving stable IDs, split digest, excluded IDs, preprocessing fit scope, and any stratification keys. Bootstrap and confidence intervals must resample the independent unit (usually animal/session), not treat cells as independent animals.

## 7. Intervention protocol

Each causal score requires a declared intervention contrast. Prefer paired treatment and sham/control within the same animal/session and context when experimentally available. Record target, actuator, dose, duration, onset, pulse order, context, washout, off-target checks, and observation window.

Minimum intervention arms for a target-level comparison:

- baseline / sham control;
- single-target perturbation A;
- single-target perturbation B for interaction studies;
- simultaneous A+B;
- sequential A→B and B→A where temporal ordering is a hypothesis;
- matched light/actuator or genotype control where required by the assay.

Compute contrasts on the measured outcome, not from model labels. For two interventions:

`I_AB(t) = Y_AB(t) - Y_A(t) - Y_B(t) + Y_0(t)`

where all terms refer to aligned, appropriately paired conditions. Third-order terms use the full eight-arm factorial contrast. Report the full trace and a predeclared task-relevant summary; do not select the summary window after seeing test results.

Do not collapse interventions, state shifts, or measurement kinetics into one parameter. Preserve latent neural dynamics and the calcium / sensor observation kernel as separate components. A measured fluorescence decay is not automatically neural persistence.

## 8. Scorecard: report a vector, not a magic number

### A. Held-out trajectory prediction (primary)

- Normalized full-trajectory RMSE and MAE per neuron and per trial, with the normalization scale fitted on training data only.
- Continuous ranked probability score (CRPS) or held-out log predictive density when a probabilistic output is provided.
- Baseline-subtracted treatment-minus-sham trajectory error.
- Onset latency error, response-sign accuracy, peak amplitude error, and recovery-time error as secondary diagnostics.
- Report per-target and per-animal distributions; do not publish only pooled means.

Correlation alone is insufficient: a high correlation can coexist with wrong gain, sign, offset, onset, or uncertainty.

### B. Causal intervention prediction

- Error in the full intervention effect trajectory ΔY(t) = Y_treatment(t) − Y_control(t).
- Correct-sign rate and false-response rate under sham / negative controls.
- Two- and three-way interaction trace error for factorial contrasts.
- Generalization by target, animal/session, state, dose, duration, and pulse order.

### C. Mechanism discrimination

Use a multi-class confusion matrix for direct anatomical, polysynaptic, extrasynaptic/latent, common-input, observation-only, and unresolved. Report macro-F1, per-class precision/recall, and an explicit abstention rate. Synthetic fixtures may have exact known answers; biological labels require independent experimental evidence. A model must not be rewarded for forcing a direct-edge label on every effective response.

### D. Calibration and uncertainty

- Empirical coverage for nominal 50%, 80%, and 95% predictive intervals.
- Mean interval width / sharpness at each nominal coverage.
- CRPS or log score, plus calibration curves.
- Calibration and coverage stratified by animal, target, response magnitude, and shifted context.
- Clustered bootstrap confidence intervals and sample counts at the independent-unit level.

Prediction quality without calibration is not reliable intervention guidance. An abstaining model should be scored on both error among non-abstained predictions and coverage / abstention rate.

### E. Incremental value of anatomy

Compare the candidate against the strongest capacity- and compute-matched non-connectome baseline and the degree/type-preserving rewired controls. Report paired per-split differences, relative improvement, cluster-bootstrap confidence intervals, and the full null distribution. Also report whether the advantage survives edge-weight uncertainty and uncertain cell-type assignment.

### F. Active experiment selection

Under a fixed number of available experiments, evaluate:

- expected information gain (EIG) under the declared model posterior;
- realized reduction in posterior predictive uncertainty after the selected experiment;
- decision regret against the best experiment in the available candidate pool;
- number of experiments needed to reach a predeclared predictive/calibration threshold.

Baselines: uniform random, degree, betweenness/centrality, largest predicted response, and expert-prior ranking. Compare on the same experiment pool and budget. EIG must be calculated using predictions available before outcome reveal; evaluate the policy prospectively where feasible.

### G. Robustness and operational cost

Stress missing neurons/cells, registration uncertainty, missing trials, uncertain edge identity/sign/weight, preprocessing perturbations, state shifts, and dataset version changes. Report degradation curves, not only one corruption point. Report training and inference wall time, peak memory, parameter count, data volume, and energy where measurable.

## 9. Aspirational benchmark targets

These are research targets to pre-register and test, not current results, SOTA claims, or universal biological constants. Calibrate confidence intervals to the sample size and state the number of independent animals/sessions.

| Dimension | Initial target | Acceptance interpretation |
|---|---|---|
| Held-out intervention prediction | At least 10% lower normalized full-trajectory RMSE than the strongest matched non-connectome baseline in at least two independent outer regimes (for example animal-held-out and target-held-out). | The paired animal/session-clustered 95% CI for improvement must exclude zero; no material degradation in probabilistic score. |
| Predictive calibration | 90% interval empirical coverage between 85% and 95%, with useful sharpness; also report 50/80/95% coverage. | Coverage alone is insufficient; wide uninformative intervals do not pass. |
| Mechanism discrimination | Macro-F1 ≥ 0.80 on synthetic known-answer fixtures, with unresolved/abstain supported. | This qualifies the evaluator and synthetic mechanism handling only; it does not certify biological mechanisms. |
| Active experiment selection | At least 2× realized information gain per experiment over uniform random and at least 20% lower decision regret than the strongest simple heuristic at equal budget. | Must be confirmed prospectively or on a locked held-out experiment pool; EIG values alone do not count. |
| Sample efficiency | Reach within 5% of the full-data predictive score using no more than 50% of the intervention budget. | Report the complete learning curve and uncertainty; no extrapolation from one seed. |
| Robustness | No more than 10% relative degradation under a pre-registered 10% missing-observation stress test; publish the full missingness curve. | Must not hide failures on particular targets or contexts. |
| Efficiency | Report p50/p95 inference latency, training wall time, peak memory, parameter count, and data requirements. | Set hardware-specific targets only after measuring the baseline stack on the same machine. |

Do not produce a weighted global score until a community review agrees on the weights. The public result is the vector scorecard plus evidence tier and uncertainty.

## 10. Blind evaluation and anti-gaming protocol

1. Register dataset version, data hashes, split-generation code, split digest, outcome windows, metrics, baseline matrix, and model-selection budget before final evaluation.
2. Keep test outcomes inaccessible to model developers until predictions, predictive distributions, and mechanism labels are committed.
3. Store predictions in a deterministic, schema-validated artifact; hash it before unblinding.
4. Run scoring in a separate evaluator process/repository with pinned version and hash. The evaluator must not import candidate model code.
5. Publish all failures, exclusions, abstentions, missing-data reasons, and negative controls.
6. On a benchmark refresh, rotate a hidden test cohort or hold back newly acquired experiments. Reusing the same public test set for iterative tuning turns it into validation data.
7. Treat a skipped CI workflow, incomplete artifact, unknown split, or missing control as `NOT_EVALUABLE`, never as a pass.

Required manifest: `CONNECTOME_INTERVENTION_BENCHMARK_RUN.schema.json`. Example manifest status is deliberately `INCOMPLETE_REQUIRES_ARTIFACTS`. A completed scorecard must additionally satisfy `CONNECTOME_INTERVENTION_BENCHMARK_SCORECARD.schema.json`.

## 11. Immediate execution order

1. Qualify all T0 oracle suites and scoring semantics independently; the synthetic oracle PR #9584 currently has authored tests but no verified Rust compile/test run in the authoring environment.
2. Materialize and hash Randi data, source connectomes, individual animal/session identities, and exact preprocessing outputs.
3. Close the Creamer reproduction gap separately: resolve the 82%/92% version discrepancy and verify the exact paper configuration before comparison.
4. Implement the baseline matrix with identical splits and hyperparameter budgets.
5. Produce T1/T3 scorecards separately, including all null controls and uncertainty.
6. Use the current intervention oracle to qualify EIG/selection metrics synthetically, then preregister a prospective T4 experiment-selection test.

## References

1. Randi, F. et al. (2023). *Neural signal propagation atlas of Caenorhabditis elegans*. Nature 623, 406–414. https://doi.org/10.1038/s41586-023-06683-4
2. Duan, S., Dong, L. L. & Fiete, I. (2025). *From Synapses to Dynamics: Obtaining Function from Structure in a Connectome Constrained Model of the Head Direction Circuit*. NeurIPS 2025. https://doi.org/10.52202/085713-4273
3. The MICrONS Consortium (2025). *Functional connectomics spanning multiple areas of mouse visual cortex*. Nature 640, 435–447. https://doi.org/10.1038/s41586-025-08790-w
4. *A large-scale benchmark for network inference from single-cell perturbation data* (2025). Communications Biology. https://doi.org/10.1038/s42003-025-07764-y
5. Kyllmann, S. & Keesey, I. W. (2026). *Connectomes: a re-emerging hub dataset for neuroscience*. Frontiers in Neuroscience. https://doi.org/10.3389/fnins.2026.1911198