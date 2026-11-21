# CETB multidimensional measurement qualification stack V0

**Status:** Proposed assurance protocol and machine-readable receipt.  
**Date:** 2026-11-21

## 1. Core design principle

Do not compress measurement quality, reliability, construct validity and generalization into one score. They answer different questions:

1. **Measurement sufficiency:** Is the acquired signal/data adequate to resolve the target structure?
2. **Repeatability:** Does the same participant's measurement reproduce under matched conditions?
3. **Construct validity:** Does the observed result support the specific construct and endpoint being claimed?
4. **Transportability:** Does the finding generalize beyond the participants, sites and acquisition used for development?

A dataset can be technically clean but unreliable. A measurement can be repeatable but invalid. A valid within-site effect can fail at a new site. These should remain visible as separate outcomes.

## 2. Relation to current methods

### NSI — measurement sufficiency for precision fMRI

The Network Similarity Index (Buchanan et al., 2026, DOI 10.1016/j.neuron.2026.05.020) estimates whether an individual fMRI dataset expresses recoverable large-scale network structure needed for precision functional mapping. It was designed to formalize expert quality judgements and inform expected benefit from more data. The authors' public [PFM-NSI implementation](https://github.com/cjl2007/pfm-nsi) ships aligned Python and MATLAB implementations, an NSI/usability projection, contextual outputs and an optional reliability projection.

CETB should treat NSI as an imported fMRI measurement-sufficiency method. The public PFM-NSI implementation expects CIFTI `.dtseries.nii` on fsLR-32k by default; optional fsaverage6 input requires Connectome Workbench resampling. The OpenNeuro ds003171 v2.0.1 release currently exposes functional BOLD NIfTI files, not CIFTI `.dtseries.nii`, so NSI must be recorded as not assessed for that release unless a separately validated compatible conversion is produced. It is not a general EEG/MEG quality score, not a directly observed test–retest ICC, not a state-change metric and not a measure of consciousness. Its optional reliability projection is a distinct model output and must not be relabelled as observed test–retest reliability without separate validation. Its use requires matching its acquisition, surface, parcellation and preprocessing assumptions.

### RCAF — participant-level repeatability

RCAF compares a participant's within-condition repeats with between-condition observations. It addresses whether the measured network contrast is larger than matched within-condition variability under the chosen estimator and segmentation protocol.

RCAF does not independently prove data sufficiency or construct validity. Its repeatability estimate is only as defensible as the underlying signals, ROI correspondence, preprocessing and segment design.

### Construct validity — independent evidence

The measured outcome must be named precisely: responsiveness, sensory processing, task relevance, report, memory, physiology, perturbational complexity or another defined endpoint. None automatically entails phenomenal experience. Theory-specific predictions and controls should be frozen before outcome access.

### Transportability — external validation

Rosenblatt et al. (2026), DOI 10.1038/s41592-026-03115-9, distinguish held-out participants from independent-dataset external validation. CETB therefore records same-participant/new-split, held-out-site, independent acquisition and independent-cohort evidence separately.

A held-out site within one release is not automatically an independent cohort. A second publication on the same participants is not independent replication.

## 3. Qualification receipt

The JSON schema \`research/cetb/measurement_qualification/measurement-qualification.schema.json\` captures one claim-scoped receipt. The synthetic fixture shows the format without asserting empirical measurements.

The receipt:
- records modality and endpoint;
- attaches data provenance and source manifest state;
- scores sufficiency, repeatability and construct validity separately;
- records participant-, site- and independent-cohort validation as distinct lanes;
- forbids a composite grade.

Status values distinguish not assessed, adequate, concern, inadequate and inconclusive. Missing evidence is not silently converted into a fail, a pass or a negative consciousness label.

## 4. Benchmark goals

Proposed targets, to be preregistered per modality and claim:
- measurement sufficiency predicts usability on blinded external judgements or a justified objective criterion;
- repeatability is evaluated at the participant level with confidence intervals and duration/gap surfaces;
- construct validity includes positive and negative controls tied to the operational endpoint;
- participant holdout, site holdout and independent-cohort performance are reported separately;
- any status change can be traced to exact input, code, configuration and environment receipts.

No universal numerical threshold is specified. NSI thresholds belong to its validated precision-fMRI use case; RCAF thresholds require estimator- and task-specific calibration.

## 5. Sources

- Buchanan et al. (2026), “Objective quality assessment for precision functional MRI data,” *Neuron*, DOI 10.1016/j.neuron.2026.05.020.
- Rosenblatt et al. (2026), “External validation improves generalizability, replicability and reproducibility in predictive models for neuroimaging,” *Nature Methods*, DOI 10.1038/s41592-026-03115-9.
- CETB's RCAF protocol: \`RCAF_INJECTION_RECOVERY_PROTOCOL_V0.md\`.

This is a proposed CETB protocol, not a regulatory standard and not a consciousness-measurement certification.
