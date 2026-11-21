# ds006623 ISD Reproduction Plan V1

**Status:** Reproduction plan only. No data files have been downloaded or analyzed by this work item.  
**Dataset:** OpenNeuro ds006623, v1.0.0, DOI 10.18112/openneuro.ds006623.v1.0.0.

## 1. Why this dataset first

ds006623 provides 26 healthy volunteers studied during graded propofol sedation, with fMRI, task events, hand-squeeze behavioural measurements and propofol infusion/effect-site information. The dataset is BIDS-organized and provides fMRIPrep, XCP-D and MRIQC derivatives.

The 2026 dataset descriptor reports a decrease in integration–segregation difference (ISD) during loss of responsiveness. It also publishes the processing configuration and points to the analysis code and Zenodo archive. This gives us a reproducible, bounded first target and a direct way to audit how much of a reported effect depends on the original preprocessing choices.

It does not provide a direct, continuous ground-truth label for phenomenal experience. The primary result is about a neural network measure across behavioural/pharmacological states.

## 2. Frozen source inputs

Required before execution:

- OpenNeuro dataset DOI/version: 10.18112/openneuro.ds006623.v1.0.0.
- Dataset-description JSON, participants metadata and complete file manifest.
- SHA-256 digest for every downloaded input; record excluded and unavailable files.
- Data descriptor DOI: 10.1038/s41597-025-06442-2.
- Published ISD analysis repository: https://github.com/janghw4/Anesthesia-fMRI-functional-connectivity-and-balance-calculation
- Published analysis archive DOI: 10.5281/zenodo.17429311.
- MATLAB version, Brain Connectivity Toolbox version, operating system and all script revisions.
- fMRIPrep 23.2.1, XCP-D 0.10.6 and BIDS 1.8.0 provenance for the published derivatives.

Do not follow a moving latest alias during confirmatory reproduction. Pin the dataset version, archived analysis code and each local manifest.

## 3. Published primary analysis to reproduce

The dataset descriptor specifies the following ISD analysis:

1. Use XCP-D without global-signal regression and with band-pass filtering.
2. Build condition-specific functional-connectivity matrices.
3. Use 456 ROIs: 400 Schaefer cortical parcels and 56 Tian subcortical regions.
4. Calculate pairwise Pearson correlations and Fisher-z transform them.
5. Define integration as multi-level global efficiency.
6. Define segregation as multi-level clustering coefficient.
7. Calculate ISD as integration minus segregation.
8. For segregation, subtract the first mode after matrix eigendecomposition to regress the global integrative component.
9. Use 50 logarithmically spaced thresholds from 0.001 to 1.0.
10. Omit the first and last five minutes of the LOR period from the condition-specific functional-connectivity calculation to reduce transition transients.

Reproduce the authors' archived implementation before porting any method to Rust. A new implementation is a separate method version and must be compared against the archived reference on identical inputs.

## 4. Primary reproduction endpoint

Reproduce the published direction and magnitude of the ISD contrast between behavioural loss-of-responsiveness (LOR) and the specified responsive comparison periods.

The exact contrast, subject inclusion list, aggregation rule, statistical test, multiple-comparison correction and tolerance must be extracted from the article supplement and archived scripts before the run is frozen. Do not infer an endpoint from a figure caption.

A successful reproduction must report:

- participant denominator and every exclusion reason;
- the exact windows used per participant and state;
- subject-level effects, group estimate and uncertainty interval;
- preprocessing/ROI/threshold settings;
- statistical test and correction;
- deviation log from the published implementation;
- output digests and independent clean-environment replay.

## 5. Secondary questions (exploratory until separately preregistered)

- Does ISD add predictive information about hand-squeeze responsiveness beyond effect-site propofol concentration, time since infusion change, head motion and signal-quality metrics?
- Does ISD improve held-out participant prediction over integration alone, segregation alone, mean connectivity, modularity and simple regularized baselines?
- Are apparent improvements robust to global-signal regression choice, band-pass settings, motion censoring, window length and reasonable parcellation alternatives?
- Does a model predict onset/recovery timing rather than merely classify windows after the transition is known?

These questions must be evaluated with participant-grouped splits. Given N=26, use a small, predeclared model family and report uncertainty; do not make a flexible model tournament the primary claim.

## 6. Leakage and confounding controls

- Group all runs, windows and task blocks from one participant into one fold.
- Keep all preprocessing, feature selection, normalization and calibration inside training folds.
- Keep effect-site concentration, motion and acquisition time as explicit candidate predictors and confounds.
- Use matched temporal windows and preserve gaps / censored frames.
- Include shuffled labels, shifted events, motion-only, dose-only and signal-quality-only controls.
- Use participant-level or hierarchical uncertainty; do not treat thousands of BOLD frames as thousands of independent participants.
- Retain all participants and all QC flags in the flow diagram, including failed or motion-contaminated data.
- A model that predicts LOR from dose may be useful for state estimation but does not demonstrate that it measures experience.

## 7. Baselines

1. Original archived ISD implementation.
2. Integration-only and segregation-only.
3. Mean connectivity and modularity.
4. Regularized logistic / mixed-effects baseline for responsiveness.
5. Dose-only, motion-only and signal-quality-only controls.
6. HDC-only and CfC/LTC-only only as secondary candidate models after the published baseline has been reproduced.

A Symthaea model only wins if it improves a frozen, subject-held-out endpoint against matched baselines and remains calibrated. This cohort cannot support an unrestricted “general consciousness detector” claim.

## 8. Interpretation boundary

The result can support statements about reproducibility of an ISD measure and its association with defined behavioural/pharmacological states in this cohort. It cannot by itself establish:

- absence or presence of phenomenal experience during unresponsiveness;
- personal identity continuity;
- consciousness outside a biological substrate;
- a universal PCI or ISD threshold;
- clinical utility in patients with disorders of consciousness.

## 9. Execution status

| Step | Status |
|---|---|
| Dataset version / DOI identified | Complete |
| Published analysis description inspected | Complete |
| Authors' code and Zenodo archive identified | Complete |
| Dataset files downloaded and checksummed | Not started |
| Exact supplement-level contrast extracted | Pending |
| Analysis environment frozen | Pending |
| Published analysis reproduced | Not run |
| Secondary grouped prediction tests | Not run |
