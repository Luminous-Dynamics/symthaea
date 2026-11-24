# NeuroTwin Cross-Trial Depression Response Benchmark v0.1

**Status:** preregistration-ready reproduction specification; no patient data are included.
**Primary comparator:** Zhukovsky et al., *Generalizability of Treatment Outcome Prediction Across Antidepressant Treatment Trials in Depression*, JAMA Network Open, 20 March 2025, DOI 10.1001/jamanetworkopen.2025.1310.
**Purpose:** test whether Symthaea contributes incremental, transportable predictive value beyond published clinical and dACC-connectivity baselines.

## 1. Why this is the first benchmark

Within-cohort random splits can overstate generalization. A direct train-on-one-trial / test-on-another experiment is a stronger first challenge because it changes trial, site mix, country, treatment protocol and cohort composition.

The 2025 cross-trial study combined 363 MDD participants (225 EMBARC, 138 CAN-BIND-1). The best reported clinical + dorsal anterior cingulate functional-connectivity models reached cross-trial AUCs of 0.62–0.67 when trained on CAN-BIND and tested on EMBARC stages, and 0.66 when trained on EMBARC stage 1 sertraline and tested on CAN-BIND. Reported cross-trial balanced accuracy for dACC models ranged approximately 0.61–0.71 depending on direction and treatment arm. Early-treatment (week-2) clinical severity was highly informative and reached comparable or better generalization than pretreatment models.

This is a useful state-of-the-art comparator, not an established clinical decision tool. The study included two multisite RCTs and SSRIs; it did not prospectively randomize treatment based on the model.

## 2. Cohort identity — do not conflate data-release and analytic counts

### CAN-BIND-1

The CONP / Brain-CODE v2.0 data release describes 211 MDD participants plus 112 healthy controls (309 total), with clinical, structural/resting/task fMRI, diffusion MRI, EEG and molecular modalities. Access is controlled through Brain-CODE. The cross-trial prediction paper analysed 138 CAN-BIND MDD participants for its two-trial experiment; the 192-person analytic cohort in the earlier escitalopram multimodal paper is a different inclusion set.

### EMBARC

The cross-trial paper analysed 225 EMBARC participants. EMBARC stage 1 randomized participants to sertraline or placebo; the second stage changed treatment for nonresponders and continued treatment for responders. The exact arm, stage, visit and inclusion set must be frozen from the source paper and release.

### Intake controls

- No patient-level or protected data are copied to this repository.
- The exact source release, version, data-use terms, consent scope, ethics authorization and secure analysis environment are recorded in the restricted registry.
- Where a release is provided as BIDS, run the local cohort preflight and the official BIDS Validator. Where data are provided in another structure, use a source-specific intake adapter and preserve the source format; do not silently convert or infer missing fields.
- The 363-participant cross-trial analysis inclusion manifest must be reconstructed and checked against the source paper. Do not substitute all 309 CAN-BIND release subjects or the complete EMBARC cohort.

## 3. Primary estimands and locked evaluation directions

Reproduce each published direction as a separate experiment:

| Experiment | Training data | Locked test data | Published anchor |
|---|---|---|---|
| A | CAN-BIND-1 analytic cohort | EMBARC stage 1 sertraline | AUC approximately 0.62 for best dACC + clinical model |
| B | CAN-BIND-1 analytic cohort | EMBARC stage 2 sertraline | AUC approximately 0.67 for best dACC + clinical model |
| C | EMBARC stage 1 sertraline | CAN-BIND-1 analytic cohort | AUC approximately 0.66 for best dACC + clinical model |

The placebo arm is a separate negative/comparator transport task, not pooled into a single SSRI outcome. If reproducing additional published arms, retain the exact source direction and stage label.

**Primary endpoint:** the binary treatment-response endpoint exactly as defined in the source paper for each trial and stage. Do not harmonize different response definitions post hoc. Keep continuous symptom change as a separate secondary endpoint.

**Separate prediction times:**
- Pretreatment: only data available before treatment assignment / initiation.
- Early treatment: only prespecified week-2 information available at the intended prediction time.

Do not combine week-2 information into a pretreatment claim. Do not compare a week-2 model with a pretreatment baseline as though they had the same information budget.

## 4. Required matched baselines

Implement the following in the exact same inclusion set and direction:

1. **Clinical-only pretreatment:** regularized model using the source paper's clinical/demographic baseline variables.
2. **Clinical + global FC:** reproduce the published global functional-connectivity feature set.
3. **Clinical + dACC FC:** reproduce the published dorsal anterior cingulate connectivity feature definition and extraction.
4. **Clinical + structural imaging:** reproduce the cortical-thickness comparator where source data and feature mapping permit.
5. **Clinical-only week 2:** early-treatment clinical baseline. This is essential because week-2 severity can be highly predictive without expensive neuroimaging.
6. **Clinical + dACC + week 2:** only where the source paper defines the same prediction time and feature availability.

A reproduction report must include the original model's feature-selection / regularization procedure, preprocessing, normalization, missingness handling, cohort counts, treatment arm, outcome definition and resampling scheme. Any unavailable method must be explicitly marked as not reproduced.

## 5. Symthaea model ladder — only after baseline reproduction

Run on the same features and exact train/test directions:

- Elastic net / logistic regression baseline.
- Regularized canonical correlation or another source-paper-aligned model where appropriate.
- Transparent tree / SVM comparator.
- Graph model on the same functional-connectivity matrices.
- Symthaea HDC representation + linear head.
- Symthaea CfC/LTC temporal model only where longitudinal measurements exist at the prediction time.
- HDC + CfC/LTC hybrid.
- Connectome-informed multimodal model.

The model must be trained from the actual source data. Hand-authored cortical activation maps, generated stimuli, mock TRIBE vectors and synthetic topology fixtures cannot be used as empirical evidence.

No-connectome ablation is mandatory. A neural model that does not add value beyond clinical-only variables does not qualify as a clinical-connectomics improvement.

## 6. Leakage and transportability controls

- Source-trial participant IDs, visits and derived scans must remain grouped.
- Fit scaling, imputation, nuisance regression, feature selection, atlas normalization, harmonization and hyperparameter selection on training data only.
- Never use test-trial outcome labels to select features, tune thresholds, calibrate probabilities or select checkpoints.
- Where batch harmonization is used, distinguish harmonization that is possible at inference from methods requiring access to the full target cohort. Report both the exact source-paper protocol and a prospective-safe variant.
- Preserve site/scanner identifiers and report site-stratified results where counts permit.
- Report missingness and modality eligibility by trial, arm and stage. Compare model arms on matched participant intersections.
- Report all failed directions and subgroup uncertainty. Do not publish only the most favorable train/test direction.

## 7. Metrics and statistical comparison

Primary:
- AUC with participant-bootstrap 95% confidence interval.
- Balanced accuracy using the threshold selected only on training data.
- Sensitivity and specificity at that frozen threshold.

Required:
- Brier score and calibration intercept/slope.
- Calibration plots for each held-out trial.
- Precision-recall AUC where event prevalence warrants it.
- Confusion matrix and sample counts.
- Risk–coverage curve and abstention rate.
- Decision-curve/net-benefit analysis only with a prespecified clinical decision and plausible threshold range.
- Subgroup and site-stratified performance with uncertainty intervals.

The primary comparison is the paired difference between Symthaea and the published-reproduction baseline on the same held-out participants. Use participant-level paired bootstrap or a prespecified paired inference method. Do not treat visits, scans or windows from one participant as independent observations.

## 8. Success and superiority gates

### Gate A — faithful reproduction

Reproduce the published direction-specific baseline within uncertainty using the exact inclusion set and endpoint. If the original model cannot be reproduced, publish the mismatch and do not make a superiority claim.

### Gate B — incremental neural value

Show that neural/connectome features improve on the clinical-only model on the locked external trial, not just within-trial cross-validation. The week-2 clinical baseline must be included as a strong comparator.

### Gate C — Symthaea value-add

Require the paired external AUC difference to exceed a preregistered minimum meaningful margin with a confidence interval excluding zero, while preserving or improving calibration, sensitivity at clinically relevant thresholds, subgroup reliability and decision net benefit.

The published upper cross-trial AUC anchor of approximately 0.68 is a reproduction target, not a universal performance ceiling. An AUC of 0.70 on a different cohort, outcome or split does not beat it. A higher AUC with worse calibration or no decision benefit is not sufficient.

**Stretch research target:** achieve AUC ≥0.70 in each of the preregistered cross-trial directions, on the identical outcome and compatible participant set, with the paired confidence interval for improvement over the reproduced published baseline excluding zero. Treat ≥0.70 as an aspiration, not an acceptance threshold or an independently established clinical cutoff. A result that reaches it in one direction but fails in the reverse direction is a directional result, not general transport superiority. Calibration, clinically relevant sensitivity, subgroup reliability and decision utility must remain acceptable.

Do not set a numerical superiority margin after seeing held-out outcomes. The study protocol and clinical collaborators must define the minimum meaningful paired difference before the lockbox is opened, with the uncertainty and sample-size constraints documented.

### Gate D — independent third-trial transport

Before making a broad generalization claim, validate on a third independent trial / institution not used for model design. Two-trial cross-validation is stronger than a random split but remains a limited evidence base.

### Gate E — treatment-selection claim

A prognostic response model is not a treatment-selection model. Treatment-selection claims require prespecified treatment-by-marker interaction tests or individualized treatment-effect estimation in randomized comparisons, followed by prospective biomarker-guided evaluation.

## 9. Planned outputs

- Frozen source inclusion manifests (restricted).
- Public aggregate cohort/feature availability report where permitted.
- Versioned preprocessing and model configuration.
- Reproduction report for all three directions.
- Matched baseline and ablation comparison table.
- Calibration / subgroup / uncertainty report.
- Independent reproduction receipt and exact commit.
- A negative-results section covering failed transfers and model ablations.

## 10. Source references

- Zhukovsky et al. (2025), *JAMA Network Open*: https://doi.org/10.1001/jamanetworkopen.2025.1310
- CAN-BIND-01 v2.0 controlled dataset release: https://portal.conp.ca/dataset?id=projects%2Fbraincode_CAN-BIND_Biomarkers_for_Depression_Baseline_Data_Release
- EMBARC clinical trial registration: https://clinicaltrials.gov/study/NCT01407094
- CAN-BIND-1 multimodal response paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10482706/

**Evidence ceiling:** faithful retrospective cross-trial reproduction establishes predictive transport only within these trials and endpoints. It does not establish a causal treatment effect, clinical utility, prospective benefit, or regulatory readiness.
