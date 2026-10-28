# Benchmark Protocol: HIV Reservoir Discovery v0.1

**Protocol state:** Proposed and preregistration-ready; not executed.  
**Protocol date:** 2026-10-29  
**Primary cohort:** Dryad DOI 10.5061/dryad.k3j9kd5dw (191 ART-suppressed participants described by the source publication).  
**Decision rule:** No SOTA, novelty, mechanism, cure or clinical-efficacy claim unless the corresponding predeclared gate is met.

## 1. Benchmark questions

**RQ1 — Reproduction:** Can we reproduce the source study's reported host-expression associations with total HIV DNA and residual HIV RNA, using the publicly released files and documented methods?

**RQ2 — Incremental prediction:** Does a Symthaea variant predict a prespecified reservoir measurement more accurately than a clinical-covariate-only model and established transcriptomic baselines on held-out participants?

**RQ3 — Generalisation:** Do selected directions or pathway-level hypotheses recur in an independent study with a compatible assay and biological unit?

**RQ4 — Experiment choice:** Can the system select follow-up observations that discriminate between competing explanations, rather than merely restating published associations?

These are four distinct claims. Passing RQ1 does not pass RQ2; passing RQ2 does not prove a mechanism; passing RQ3 does not demonstrate treatment efficacy.

## 2. Population, data unit and endpoints

### Primary cohort

Start with the released bulk RNA-seq, clinical, intact-DNA and protein files from the Dryad dataset. The participant UUID is the joining key. Verify the exact data dictionary and participant overlaps before modelling; do not infer missing fields or reconstruct identifiers.

### Primary endpoint

Predict the log-transformed total HIV DNA reservoir measurement from the available host transcriptome and prespecified clinical covariates. The exact transformation, handling of zero values, units and exclusions must be documented and frozen during M0, before any model comparison.

### Secondary endpoints

- Residual HIV RNA (usRNA), if the released field and definition match the source paper.
- Intact HIV DNA measurements, handled as a separately reported endpoint with explicit treatment of values below detection or assay limits.
- Protein or cytokine measurements only for prespecified orthogonal corroboration where participant matching and measurement provenance are verified.

The source publication reports many undetectable intact-DNA values and no significant differential host-gene expression association with intact DNA. Do not replace censored/undetectable values with zero without a sensitivity analysis, and do not promote total DNA or residual RNA prediction as a surrogate for replication competence.

### Unit separation

The unit for the primary cohort task is the participant, not the gene, cell, read, assay well or replicate. Cell-line studies, single-cell studies, brain tissue cohorts and proviral clone studies are separate tasks unless a defensible common estimand is specified in advance.

## 3. Baselines and ablations

All baselines must use the same eligible participants, outer folds and outcome definition.

1. **Null baseline:** Predict the training-fold mean (and report a robust median baseline as a sensitivity check).
2. **Clinical-only baseline:** Regularised or standard regression using only covariates actually present, with transformations fixed inside training folds. The source paper adjusted for clinical factors and ancestry; if relevant ancestry data are unavailable in the released files, state that limitation instead of inventing values.
3. **Established transcriptomic baseline:** A faithful reproduction of the published statistical pipeline where feasible, including documented normalisation, variance handling, covariate adjustment and false-discovery correction.
4. **Regularised prediction baselines:** Ridge and elastic-net regression, with feature selection and hyperparameter tuning nested inside training folds.
5. **Nonlinear baseline:** One prespecified tree-based model (for example, random forest or gradient boosting), with tuning nested and model complexity limited for the cohort size.
6. **Symthaea representation baseline:** A version using the proposed HDC/semantic representation but no temporal component.
7. **Symthaea temporal variant:** CfC/LTC or other temporal modelling only on genuinely time-resolved data with leakage-safe temporal splits. Do not claim a temporal advantage from a single cross-sectional observation per person.
8. **Ablations:** Remove HDC, remove temporal dynamics, remove evidence retrieval/hypothesis integration and shuffle feature-to-evidence mappings. Report what each component adds and whether any apparent gain survives the negative controls.

The source paper's adjusted associations are a reproduction target, not a novel discovery. Candidate algorithms and hyperparameters must be frozen before inspecting outer-fold test outcomes.

## 4. Split and leakage controls

- Use participant-grouped nested cross-validation for internal prediction. A provisional design is 5 outer folds and 5 inner folds, with fold assignment and random seeds committed before modelling. If cohort size or missingness makes this unstable, change the design during M0 and record the reason before revealing scores.
- Every preprocessing operation that learns from data—including imputation, scaling, variance filtering, feature selection, pathway scoring, representation fitting and tuning—must occur inside the current training fold.
- Never split cells, reads, assays or repeated measures from one donor across train and test.
- Keep the source study's published result labels separate from newly generated hypotheses. Do not describe rediscovery of P3H3, NBL1 or other reported associations as novel.
- Report a random-label/permutation negative control and a feature-to-label shuffle control.
- Where possible, reserve an independent cohort or study for external validation. Do not tune on the external study.
- Different assays are not exchangeable labels. If there is no compatible endpoint, evaluate only the corresponding mechanistic prediction or qualitative direction and label it as such.
- Maintain an audit of every file, code commit, environment, configuration, seed, fold assignment and exclusion rule.

## 5. Metrics

### Primary prediction score

Relative change in outer-fold RMSE versus the clinical-only baseline for the primary endpoint. Define the sign convention and missing-data denominator before scoring. Report raw RMSE alongside the relative change so the result cannot be improved by a misleading ratio.

### Required secondary scores

- MAE and out-of-fold R-squared, including negative values where the model performs worse than the mean baseline.
- Spearman correlation between out-of-fold predictions and measurements.
- Calibration intercept and slope, with uncertainty; a high correlation is not sufficient if predictions are systematically miscalibrated.
- 95% confidence intervals from a participant-level bootstrap or a justified alternative that respects the sampling structure.
- Feature/pathway stability across folds, directions of effects and false-discovery-controlled inferential results.
- Performance and uncertainty across prespecified subgroups only where sample sizes support interpretable estimates; do not make unsupported population-genetic claims.
- Runtime and compute cost, to evaluate whether any accuracy gain is operationally useful.

### Mechanistic and experiment-selection scores

For each hypothesis, record prior evidence, contradictory evidence, novelty status, predicted observation, at least one alternative explanation, and the experiment that could distinguish them. Score whether the prediction was specified before seeing the validation evidence, whether the result replicated, and whether the experiment would change the relative plausibility of competing hypotheses.

For an experiment-selection benchmark, compare Symthaea's proposed next test with random selection and a simple uncertainty-based baseline using the same candidate set and budget. Use information gain only when the underlying probabilistic model is calibrated; otherwise call the score a heuristic, not measured expected information gain.

## 6. Prospective success thresholds

These are **targets to preregister**, not claimed results or guaranteed attainable thresholds.

### Internal predictive improvement

- Symthaea beats the strongest fair baseline on the primary endpoint using identical outer-fold predictions.
- Target at least a 10% reduction in RMSE versus the clinical-only baseline.
- The uncertainty interval for the paired improvement excludes no improvement, and the gain is not explained by one fold, one influential participant or the permutation control.
- Calibration does not deteriorate materially; report any trade-off.

### Independent scientific value

- A selected direction or pathway replicates in an independent cohort/study with compatible assay and unit, using a prespecified test and appropriate multiple-testing correction (target FDR q <= 0.05 for a declared family of inferential tests).
- The result is not merely rediscovery of a published association.
- A mechanistic claim requires convergent independent evidence and an appropriate functional test by qualified researchers; an observational dataset alone is insufficient.
- A cure, safety or treatment-effect claim is outside the scope of this benchmark and cannot be inferred from these scores.

Failure to meet the threshold is a valid outcome. Do not weaken splits, change endpoints after results are seen, remove inconvenient cohorts or relax thresholds retroactively.

## 7. M0 data-quality gate

Before any model is scored, publish a receipt containing:

- Dataset DOI/accession, source page, retrieval timestamp and licence/terms check.
- Exact filenames, file sizes and SHA-256 digests.
- Original data dictionary and file-version identifiers.
- Participant counts per file; duplicate or missing UUID counts; join coverage; duplicate and missing columns.
- Measurement limits, missingness and the exact treatment of censored values.
- Software versions, source commit and immutable configuration.
- Predeclared folds, seeds, primary endpoint, metrics and exclusion rules.

**Fail closed:** a missing checksum, ambiguous outcome, unresolved participant join, leaked feature selection, unpinned fold, or overwritten source file blocks the benchmark claim. Preserve original source data unchanged.

## 8. External datasets and non-comparable tasks

- GSE260588: flow-FISH-separated latent, transcriptionally active and uninfected CD4+ T-cell populations. Candidate task: replicate cell-state-associated transcriptional signatures; confirm sample and donor structure before choosing splits.
- GSE242997 / GSE210146: study-specific primary CD4+ T-cell and 2D10 cell-line latency/latency-reversal data. Use as separate mechanistic tasks; do not pool cells across donors or model systems.
- GSE241723: single-cell transcriptional signatures across two in-vitro latency models. Use as a model-shift stress test, not as an independent patient-level reservoir-size cohort.
- GSE221685 / GSE221686 / GSE221687: ChIP-seq, scATAC-seq and scRNA-seq from the brain myeloid compartment in three male donors. Use only as a small, exploratory tissue-shift/triangulation task; do not claim population-level generalisation from this cohort.
- GSE310195: expression heterogeneity among clones carrying intact latent proviruses. Use for mechanistic triangulation and clone-level state hypotheses, not to inflate participant-level sample size.
- GSE268010: transcription at integration sites of intact latent provirus. Use as a distinct integration-context task if its sample structure and metadata support a prespecified comparison.

## 9. Interpretation and release gate

Results must be labelled as one of: **reproduced**, **predicted internally**, **independently replicated**, **mechanistically supported**, or **experimentally validated**. These labels are not interchangeable.

Publish negative controls, null findings and known limitations alongside successful scores. The system's ability to cite a paper does not establish that it has reproduced the paper's result. A plausible HDC binding, FEP state or CfC trajectory is not biological evidence unless it improves a predeclared task against fair baselines.
