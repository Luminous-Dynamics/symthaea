# NeuroTwin Clinical Translation Programme v0.1

**Status:** research proposal and evidence-gating specification. Not a clinical product, diagnostic system, treatment recommender, or claim of therapeutic efficacy.
**Owner:** Luminous Dynamics research programme
**Primary systems:** Symthaea (modelling and analysis), Symtropy (only where calibrated dynamical simulation is justified), Mycelix Health (consent, provenance and patient-controlled data workflows).

## 1. The opportunity — and the necessary separation

NeuroTwin investigates whether individual-specific, multimodal biological models can improve prediction of neurological and mental-health trajectories and treatment response.

There are two different connectomics problems in the current Symthaea portfolio. They must not be conflated.

1. **Microscopy / EM connectomics:** reconstructing neurons, neurites and synapses from electron-microscopy volumes; identifying split/merge errors; auditing graph topology and scientific queries. The Connectomics Autopilot programme and ConnectomeBench2 work belong here.
2. **Human in-vivo neuroimaging and electrophysiology:** modelling structural connectivity from diffusion MRI, functional connectivity from resting/task fMRI, EEG/MEG dynamics, clinical measures and longitudinal outcomes. This is the route relevant to patient-specific clinical research.

A performance result on EM split/merge classification is not evidence of performance on human functional-connectome prediction. A connectome graph is not itself a complete brain model, and a computational similarity score is not a treatment effect.

## 2. Current-state audit and claims ceiling

The public Symthaea repository already contains valuable engineering foundations:

- A connectome research programme for provenance-bound reconstruction scenarios, pipeline multiverses, query robustness, correction lineage, uncertainty and event-driven invalidation.
- A finite, explicitly unweighted connectome query ensemble audit. It reports ranges and decision stability; it does not claim a posterior probability.
- Neural-validation scaffolding for cortical activation, temporal dynamics, parcellation, EEG and hybrid-substrate experiments.
- HDC representations and continuous-time / liquid dynamics that may be useful components in future data-driven models.

The following limitations must remain visible:

- The current neural cortical benchmark can generate five synthetic fallback stimuli if the local prediction directory is absent. Those runs are software fixtures, not empirical fMRI validation.
- The current loader silently ignores malformed JSON files. A partially broken input folder must not produce a plausible-looking benchmark result.
- The local text-conditioned “TRIBE” function is a synthetic surrogate, not Meta's TRIBE model.
- The current embedded C. elegans network in the Rust source constructs a representative 279-neuron topology with deterministic pseudo-random edges based on broad circuit motifs. It is not an imported empirical connectome edge table. Tests of that fixture are not biological validation.
- Existing sleep / meditation examples and test labels do not, by themselves, establish external clinical validity, reliable diagnosis, treatment response, or a biomarker of consciousness.

Until source data, preprocessing, cohort, split, model checkpoint, metric computation and an independent reproduction are pinned, every such result must be labelled **synthetic**, **internal structural check**, or **unverified external input** as appropriate.

## 3. Paradigm shifts worth testing

### 3.1 From diagnostic labels to individual trajectories

Model measurements and outcomes over time, not just a static class label. Represent a participant as a time-indexed state with measurement provenance, uncertainty, missingness and context. Predict future observations and clinically meaningful change, with prespecified horizons.

### 3.2 From a single connectome to a multimodal, multi-timescale model

Candidate inputs include:

- Structural connectivity: diffusion MRI / tractography, with acquisition and reconstruction uncertainty.
- Functional connectivity: resting-state and task fMRI; dynamic connectivity only with explicit reliability checks.
- Electrophysiology: EEG/MEG features and event-aligned dynamics.
- Clinical state: validated symptom scales, functioning, adverse effects, treatment exposure and clinician-assessed outcomes.
- Physiology and context: sleep, activity, medication exposure, relevant laboratory values and social/environmental factors where lawful, consented and justified.

Do not force every modality into a single embedding. Keep modality-specific encoders and provenance; compare early fusion, late fusion and missing-modality-aware models.

### 3.3 From population averages to calibrated individual deviations

A normative model may estimate how far a person's measurements deviate from an age-, site- and acquisition-aware reference distribution. Such deviations are hypotheses, not diagnoses. Test whether they add information beyond age, symptoms, baseline severity, motion, site/scanner and standard clinical variables.

### 3.4 From prediction to treatment-effect estimation

A prognostic model estimates outcomes under observed care. A treatment-selection model must estimate how outcomes differ under alternative treatments. These are not interchangeable.

Treatment-effect claims require randomized comparisons or a defensible causal design, explicit treatment-by-marker interaction tests, overlap/positivity checks, confounding analysis, calibration of individual treatment contrasts and prospective validation. A high AUROC for diagnosis is not evidence that the model can choose a treatment.

### 3.5 From black-box digital twins to evidence-carrying counterfactuals

Every simulated result should include:
- Exact input observation and provenance identities.
- Model, feature pipeline, parameter and code versions.
- The supported population and intervention range.
- Which parameters are identifiable versus assumed.
- Prediction intervals and sensitivity to plausible alternatives.
- A clear distinction among observed, inferred, simulated and unidentifiable quantities.
- Conditions that invalidate the result.

An in-silico edit to a connectivity matrix is not equivalent to a physically deliverable treatment. A simulation must map to a specified intervention and dose before clinical interpretation is even considered.

### 3.6 From uncertainty scores to value of information

Select the next measurement, specialist review or follow-up interval by expected reduction in a prespecified decision loss, divided by total cost and burden. Validate that the policy improves outcomes or efficiency; do not equate model confidence with value.

### 3.7 From one held-out split to lifecycle qualification

Use participant-, family-, site- and time-aware partitions; nested preprocessing; locked external cohorts; recalibration policies; distribution-shift detection; selective prediction and explicit abstention. Preserve data lineage and make any new source event capable of invalidating dependent evidence.

## 4. First clinical research domain

**Initial focus: major depressive disorder and anxiety treatment-response research.**

This is a pragmatic first domain because longitudinal clinical cohorts and multimodal measurements exist. It is not a claim that depression or anxiety is primarily a connectome disorder, nor that connectomics alone can determine treatment.

A second, separate neurological track may evaluate stroke recovery or epilepsy only after its dataset, endpoints and clinical collaborators are independently established. Do not pool these conditions into one benchmark.

## 5. Dataset strategy

No patient-level data or protected health information belongs in this repository. Every dataset intake needs a version-pinned manifest, licence / data-use agreement, consent scope, ethics determination, source checksum, subject map, acquisition metadata and preprocessing record.

| Candidate | Appropriate first use | Important limitation |
|---|---|---|
| NKI-Rockland Sample | Exploratory normative modelling and cross-sectional / longitudinal brain-behaviour analyses | Not a randomized treatment-response cohort; verify each data release and use terms |
| Human Connectome Project Young Adult | Healthy-reference and acquisition / site robustness research | Healthy volunteer cohort, not a clinical treatment-outcome dataset |
| OpenNeuro BIDS datasets | Reproducible task-specific fMRI / EEG replication; screen each dataset individually | Dataset count is not a quality guarantee; check task, licence, consent, sample size and metadata |
| CAN-BIND-01 | High-value depression research candidate: multimodal neuroimaging, EEG, clinical and biological measures over treatment visits | Controlled Brain-CODE access; confirm the exact cohort, intervention arms, follow-up, endpoint definitions and permitted uses before protocol lock |
| EMBARC | Randomized sertraline / placebo treatment-response research candidate | Controlled NIMH NDA access; confirm which imaging and clinical modalities are available in the requested release; do not infer modality availability from study descriptions |

Use synthetic data only for software development, schema validation and known-answer tests. Synthetic data must never be included in the empirical cohort results table.

## 6. Reproducible experiment design

Before training:

1. Write a protocol with one primary endpoint, horizon, cohort, target population, inclusion/exclusion criteria, estimand and analysis plan.
2. Freeze the dataset version, subject inclusion manifest and source checksums.
3. Define subject-level partitions before any learned preprocessing.
4. Fit feature selection, imputation, scaling, nuisance regression, harmonization and dimensionality reduction only within the training folds.
5. Keep repeated visits and related participants in the same partition unless the estimand specifically tests temporal forecasting.
6. Use nested validation for hyperparameter selection and a locked external site / cohort for final testing.
7. Record missingness and compare identical participant intersections for modality ablations.
8. Report all prespecified outcomes, failed models, subgroup results and deviations.

Minimum split regimes:

- Participant-held-out nested cross-validation for internal development.
- Leave-site/scanner-out validation where multiple sites exist.
- Forward-in-time validation for future outcome prediction.
- A separately governed external cohort that was not used for feature, model or threshold selection.

## 7. Matched baselines and model ladder

Every model must use the same cohort, endpoint, split, feature availability and tuning budget.

1. **Clinical baseline:** regularized logistic / linear model using prespecified clinical covariates and baseline severity.
2. **Imaging baseline:** connectome-based predictive modelling / regularized model on functional connectivity; structural features tested separately.
3. **Graph baseline:** graph neural network or graph transformer on the same connectivity representation.
4. **Multimodal baseline:** late-fusion model with modality-specific encoders and missingness indicators.
5. **Symthaea ablations:** HDC-only; CfC/LTC-only; graph + HDC; graph + CfC/LTC; multimodal hybrid. Each must be genuinely trained and compared, not generated from hand-authored regional profiles.
6. **External reference models:** published and reproducible methods, including connectome-conditioned behavioural prediction where the task and dataset match. Do not compare a task-prediction score with clinical outcome prediction.

Do not select Symthaea merely because it is novel. If it cannot beat a robust clinical-only baseline on external data, retain the simpler model.

## 8. Benchmark metrics

| Task | Primary metrics | Required supporting metrics |
|---|---|---|
| Binary outcome / response | AUROC and AUPRC | Sensitivity / specificity at prespecified thresholds, Brier score, calibration intercept/slope, confidence intervals |
| Continuous symptom change | MAE / RMSE | Concordance correlation, calibration of prediction intervals, baseline-adjusted comparison |
| Time to relapse / event | C-index or time-dependent AUC | Calibration at fixed horizons, competing-risk handling where required |
| Treatment-effect heterogeneity | Prespecified treatment-by-marker interaction and held-out treatment-contrast calibration | Policy value / net benefit in randomized data, overlap and sensitivity analysis |
| Longitudinal state prediction | Horizon-specific MAE / log score | Prediction-interval coverage, calibration, missingness and temporal drift |
| Selective prediction | Risk–coverage curve | Error among accepted cases, abstention rate and subgroup coverage |
| Transportability | External-site performance and calibration | Scanner / protocol shift, subgroup performance and recalibration cost |
| Operational value | Decision-curve net benefit or prospective workflow outcomes | Clinician time, patient burden, cost, adverse events and access effects |

All metrics need uncertainty intervals and sample counts. Bootstrap at the participant or cluster level, not the scan/timepoint level. Do not report accuracy alone. Do not optimize thresholds on the lockbox.

## 9. Qualification gates and target thresholds

Do not invent a universal numeric AUROC target before the endpoint, population and prevalence are fixed. The target is a prespecified, statistically supported improvement over a credible baseline with clinical utility.

- **G0 — data legitimacy:** lawful access, permitted use, complete lineage, consent and ethics review.
- **G1 — pipeline integrity:** deterministic preprocessing, no leakage, independent reproduction and exact cohort manifest.
- **G2 — internal prediction:** improvement over clinical-only baseline with participant-level uncertainty intervals and calibrated probabilities.
- **G3 — transportability:** external-site / external-cohort validation with acceptable calibration and explicit distribution-shift handling.
- **G4 — causal treatment evidence:** randomized-data treatment-effect heterogeneity, not merely prognosis.
- **G5 — prospective utility:** a protocol-approved prospective study shows benefit or non-inferiority on prespecified patient-relevant outcomes, harms, access and workflow burden.
- **G6 — regulated clinical use:** independent clinical, safety, privacy, security and regulatory review appropriate to the intended use.

A claim of superiority requires the lower confidence bound of a paired, preregistered external comparison to exceed the prespecified margin, plus acceptable calibration and meaningful decision benefit. Otherwise report equivalence, uncertainty or failure.

## 10. Symthaea-specific research hypotheses

These are hypotheses to falsify, not established advantages:

- HDC may offer compact representations and compositional binding for multimodal state/context; compare against conventional embeddings with equal information and compute budgets.
- CfC/LTC may help model irregularly sampled trajectories; compare with GRU, TCN, state-space and continuous-time baselines under identical observation schedules.
- The combination may improve selective prediction or missing-modality robustness; measure risk–coverage and calibration, not only average accuracy.
- A connectome-informed prior may help only in tasks where the neural feature is reliably measured and adds value beyond clinical variables. Test a no-connectome ablation.

No assumption that Φ, IIT-inspired indicators or artificial “consciousness” measures are clinical biomarkers. Keep them outside the primary model and clinical endpoint until independently validated for a clearly defined construct.

## 11. Safety, ethics and intended-use boundary

This programme is for research. It does not diagnose, prescribe, select or adjust medication, determine suicide risk, recommend neuromodulation settings, or replace clinician judgment. Mental-health and substance-use data require especially careful governance. Any crisis screening or safety workflow needs qualified clinical ownership and a functioning local escalation pathway; a model confidence score is not a safety plan.

Before human-subject research: obtain appropriate institutional ethics / IRB review, data-use permissions, data protection impact assessment where applicable, clinical governance, preregistration and a documented adverse-event / incident process. Regulatory classification depends on intended use and jurisdiction; do not claim HIPAA, GDPR, FDA or other compliance from technical controls alone.

## 12. Sources to pin in the experiment registry

- Digital Twin Brain (2026): https://pubmed.ncbi.nlm.nih.gov/?term=Digital+Twin+Brain+Generating+Multitask+Behavior+from+Connectomes
- External validation in neuroimaging prediction (Nature Methods, 27 May 2026): https://www.nature.com/search?q=external%20validation%20neuroimaging%20prediction
- Data leakage in connectome-based machine learning: https://www.nature.com/search?q=Data%20leakage%20inflates%20prediction%20performance%20connectome
- PROBAST+AI (BMJ, 2025): https://www.bmj.com/content/388/bmj-2024-082505
- IMDRF Good Machine Learning Practice (2025): https://www.imdrf.org/documents/good-machine-learning-practice-medical-device-development-guiding-principles
- HCP Young Adult: https://www.humanconnectome.org/study/hcp-young-adult
- NKI-Rockland Sample: https://fcon_1000.projects.nitrc.org/indi/enhanced/
- OpenNeuro: https://openneuro.org/
- CAN-BIND: https://canbind.ca/
- EMBARC / NIMH NDA: https://nda.nih.gov/

## 13. Immediate next experiments

1. Correct the synthetic-vs-empirical labeling and fail-closed input parsing in the existing cortical benchmark.
2. Requalify the C. elegans embedded topology as a synthetic fixture, then add a versioned empirical data import with explicit source attribution before making biological claims.
3. Add a BIDS metadata / phenotype intake auditor that emits a cohort manifest only; no clinical features or raw scans are copied into Git.
4. Design a minimal, transparent clinical baseline on an approved dataset after data access and protocol review.
5. Add connectome and multimodal model arms only after baseline reproducibility is established.
6. Create an independent clinical/scientific review group before considering any patient-facing claim.

**Success for v0.1:** the research programme becomes more falsifiable, reproducible and honest about evidence. It does not claim to cure disease.
