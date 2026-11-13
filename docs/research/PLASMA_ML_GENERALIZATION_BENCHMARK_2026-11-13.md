# PLASMA-ML-GENERALIZATION-001 — Group-held-out plasma outcome prediction

**Date:** 13 November 2026  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11839  
**Parent:** PLASMA-VV-001 #7588; application outcome owner PLASMA-SURFACE-001 #11795  
**Status:** benchmark contract + synthetic split-integrity fixture  
**Claim ceiling:** no original dataset, published-model reproduction, plasma prediction or Luminous experiment is included.

## 1. Why this benchmark matters

Belmekki et al., *Predicting Cold Atmospheric Plasma-Induced Wettability Transformations in Biopolymers: A Stacked Ensemble Machine Learning Framework*, Plasma Processes and Polymers 23 (2026), e70246, first published 1 September 2026, compile 534 observations from 53 publications and report an AdaBoost → Random Forest stacked model.

Reported test values for the relative wettability-reduction target are:
- R² = 0.9079
- RMSE = 0.0522
- MAE = 0.0366
- median absolute error = 0.0238

The paper reports absolute-final-WCA test R² = 0.7275. These values are on the paper's processed target scales and are not comparable to a direct MPa adhesion metric or a production outcome.

Source: https://doi.org/10.1002/ppap.70246.

## 2. Research question: can the result generalize to a new laboratory?

The corpus has a nested structure: observations belong to source papers, laboratories, devices, material families/grades and experiment batches. A random observation split can place highly related measurements from the same experimental lineage in both training and test sets.

The paper also states that SMOGN augmentation was applied before the 80/20 split. SMOGN constructs new observations from nearest-neighbour structure. If the augmentation fit used rows later assigned to the test partition, test information could influence synthetic training rows.

This is a **potential leakage question, not an allegation or conclusion**. The text alone is insufficient to determine whether leakage occurred in the authors' exact code. The supplementary DOCX and/or authors' implementation must be checked. The publisher says the underlying data are available from the corresponding author upon reasonable request; no data-rights or access assumption is made here.

The benchmark must first reproduce the published pipeline faithfully, then evaluate stronger transfer tasks.

## 3. Paradigm shift: from row accuracy to transfer and experiment value

A general-purpose plasma process intelligence system must answer:

- Will the process transfer to an unseen source publication, lab, treatment head or material grade?
- Does it know when its training data do not support a prediction?
- Are prediction intervals calibrated on independent groups?
- Does it predict a downstream process result, or only a proxy like contact angle?
- Which next experiment most reduces uncertainty in a decision that matters?
- Does model-guided experimental selection produce a better result under the same laboratory budget?

The objective is not to maximize random-split R². It is to maximize **validated decision quality per experiment, with calibrated uncertainty and honest abstention**.

## 4. Evaluation tracks

1. **Faithful reproduction:** acquire the original data and supplementary materials under appropriate permission; reconstruct curation, target definitions, SMOGN, scaling, split seed, AutoML search and final model.
2. **Leakage-safe baseline:** split original empirical rows first; fit all transformations and augmentation within each training fold.
3. **Leave-one-study-out:** all rows from one paper remain in one fold.
4. **Leave-one-laboratory-out:** hold out a lab / research group where source metadata support it.
5. **Leave-one-device-out:** hold out treatment head or device family.
6. **Leave-one-material-family/grade-out:** test genuinely new polymer / resin grades with explicit OOD status.
7. **Temporal holdout:** train on older sources and test on later sources where dates exist.
8. **Decision-value:** compare equal-budget model-guided design with random, space-filling and standard Bayesian optimization baselines.

Different tracks answer different questions. They must not be collapsed into one accuracy score.

## 5. Data contract

Every original empirical row must preserve, where actually available:
- raw observation ID and specimen / replicate identity;
- source publication, lab/group, device/treatment-head and experiment-batch identity;
- material family, grade, formulation, lot and initial condition;
- plasma regime and source; gas and declared operating conditions with SI units;
- treatment geometry, timing and storage/aging metadata;
- outcome definition, method, calibration, uncertainty, censoring and exclusions;
- dataset license/permission, extraction provenance and source citation.

Missing fields remain missing with an explicit reason. Unknown humidity, standoff, waveform, flow or treatment delay cannot be silently encoded as zero or inferred from typical equipment.

Do not infer adhesion, aging or lifetime from contact angle.

## 6. Leakage-resistant pipeline rules

- Partition raw empirical observations by group before SMOGN, neighbour synthesis, imputation, scaling, feature selection or target transformation.
- All synthetic observations have explicit parent IDs and are restricted to the training fold of the corresponding split.
- All preprocessing state is fit only on training observations and recorded in a fold-specific receipt.
- No specimen, raw-record digest or near-duplicate source observation crosses train/test boundaries without explicit adjudication.
- Source/lab/device identifiers are used as grouping keys; if evaluated as predictive features, that is a separate declared baseline.
- Keep held-out paper/device/material identities sealed during feature/model selection.
- Retain invalid measurements, failed predictions, exclusions and OOD refusals.
- Preserve a frozen external test set outside AutoML and hyperparameter search.
- Exact and near-duplicate checks must preserve paper provenance; identical descriptors across papers are not automatically duplicates, but shared specimen/raw-row identity is a leak.
- The split receipt binds dataset digest, group map, split algorithm/seed, train/test IDs, preprocessing digest and augmentation lineage.

## 7. Baselines and metrics

Baselines:
- global mean / material-only;
- regularized linear and additive models;
- Random Forest, gradient boosting and the published stacked architecture;
- hierarchical model with source/device effects where identifiable;
- calibrated ensemble or conformal prediction;
- uncertainty-aware experiment selection.

Predictive metrics:
- R², MAE, RMSE, median and maximum absolute error;
- worst-study, worst-lab, worst-device and worst-material errors;
- interval coverage and width at nominal 90% and 95%;
- calibration, abstention and coverage-risk curves;
- performance stratified by target range and OOD distance.

Decision metrics:
- experiments to reach a frozen quality threshold;
- normalized regret / hypervolume under equal evaluation budgets;
- useful accepted output per lab-hour and per complete-system energy/cost;
- feasible-candidate yield and hard-constraint violations;
- confident-error frequency on unseen groups.

MAPE should not be primary where the target can approach zero or become negative.

## 8. Proposed targets — hypotheses, not current results

After reproducing the published pipeline and establishing an adequate raw dataset:
- match the source's random-split result within a preregistered reproduction tolerance;
- improve worst-study and leave-one-device-out errors versus the published stack and simple baselines;
- achieve empirical 90% coverage for nominal 90% intervals on grouped holdouts, while reporting per-group miscalibration;
- reduce physical experiment budget by at least 25% to reach the same frozen quality threshold versus equal-budget random / space-filling selection;
- no increase in confident wrong predictions when abstention is enabled;
- provide OOD refusal for unsupported regime/material/geometry combinations.

If data access or group metadata are insufficient, the model-transfer claim remains BLOCKED. Do not fabricate rows to fill gaps.

## 9. S0 synthetic fixture and evaluator

The fixture at `data/research/plasma_ml_generalization_fixture_v0_1.json` is explicitly synthetic. It contains 12 split scenarios exercising:
- study-, lab-, device- and material-family-held-out group separation;
- SMOGN-like augmentation whose parents remain train-only;
- preprocessing fit-lineage restricted to training rows;
- duplicate specimen and raw-record digest detection across folds;
- missing source identity;
- missing augmentation parent;
- synthetic row assigned to a test fold;
- incomplete preprocessing lineage.

The Rust-native `xtask plasma-ml-generalization-audit` validates only these integrity conditions. A PASS means the synthetic split contract is internally consistent; it does not reproduce the published model, establish the existence of leakage in that publication, or predict physical plasma outcomes.

## 10. Ownership

- PLASMA-VV-001 #7588: physical model capability, source-pinned numerical evidence and credibility.
- PLASMA-ML-GENERALIZATION-001 #11839: literature ML reproducibility and grouped transfer evaluation.
- PLASMA-SURFACE-001 #11795: bondability endpoint.
- MICROPLASMA-001 #11780: device and application corpus.
- Existing Symtropy scenario/evidence tools remain the execution/replay substrate; no second provenance, calibration or evidence authority is introduced.

## References

- Belmekki et al. (2026), DOI: https://doi.org/10.1002/ppap.70246.
- Publisher full text / data availability statement: https://onlinelibrary.wiley.com/doi/full/10.1002/ppap.70246.

