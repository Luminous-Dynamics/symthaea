# CETB SOTA and leakage-resistant validation frontier — 21 November 2026

**Purpose:** Current competitive map for CETB/RCAF. This is a methodology comparison, not a leaderboard of consciousness detectors.

## 1. What the field currently does well

### Functional-connectivity estimators

The 2025 *Nature Methods* benchmark (DOI 10.1038/s41592-025-02704-4) compared 239 pairwise statistics for canonical network features: hub mapping, weight–distance trade-offs, structure–function coupling, correspondence to other neurophysiological networks, individual fingerprinting and brain–behaviour prediction. It found substantial quantitative and qualitative variation across estimators; covariance, precision and distance metrics showed multiple desirable properties. This rules out a generic assumption that one connectivity metric dominates for every scientific purpose.

### COGITATE MEG estimator comparison

Bryant and Whyte (2026), DOI 10.1093/nc/niag029, compared 246 retained statistics on COGITATE BIDS MEG, with 35,280 primary candidate-model configurations. Barycenter-derived measures were among the strongest stimulus-type decoders. It is a same-source-cohort method comparison and must not be described as an independent participant replication. CETB's nested estimator-selection protocol is designed to evaluate the model-selection procedure, not only its retrospective peak candidate.

### Precision-fMRI data sufficiency

The 2026 *Neuron* Network Similarity Index (NSI), DOI 10.1016/j.neuron.2026.05.020, provides an objective readout of recoverable large-scale network structure and aligns with blinded expert judgements of precision-fMRI map usability. The authors' public [PFM-NSI tool](https://github.com/cjl2007/pfm-nsi) supplies Python and MATLAB implementations, a usability projection and an optional reliability projection. CETB should reuse it for compatible precision-fMRI data rather than clone it. Its reliability projection remains distinct from directly observed test–retest reliability. NSI addresses data sufficiency and quality at an individual level; it is complementary to, not interchangeable with, RCAF state-contrast repeatability.

### External validation

Rosenblatt et al. (2026), *Nature Methods*, DOI 10.1038/s41592-026-03115-9, argue that external validation on independent datasets is a stricter generalizability test than internal validation on unseen samples from the same dataset. CETB should preserve separate tiers for participant holdout, site/device holdout and independent-cohort confirmation.

### Leakage-resistant multi-site connectome evaluation

Ubaida et al. (2026), *Diagnostics* 16(20), 3312, DOI 10.3390/diagnostics16203312, audited an ABIDE-I connectivity pipeline. It traced an originally reported balanced accuracy of 0.999 to five leakage pathways and reported 0.632 after the five pathways were closed. Under their nested leave-one-site-out protocol, graph-attention AUC was 0.648 versus 0.679 for an SVM on the same connectivity; decision-level fusion did not add detectable performance.

Important scope detail: the paper's locked 359-participant validation holds out three large sites from the same ABIDE-I release, not a wholly independent second cohort. The authors explicitly note the lack of a matched independent cohort. CETB must call this a locked within-release cross-scanner validation, not independent-cohort replication.

The study is a concrete reminder that site correction, feature/threshold selection, fusion, model checkpoint selection and target construction must all be audited. Its numerical performance is for an autism classification endpoint, not a consciousness benchmark or general SOTA ceiling.

### Multiverse and analytical flexibility

Short et al. (2026), *Advances in Methods and Practices in Psychological Science*, DOI 10.1177/25152459261434881, recommend systematic mapping of defensible analytic pipelines, preregistration, and assessment of defensibility before deciding whether alternatives are equivalent. They distinguish the multiverse of defensible workflows from arbitrary combinatorial perturbation. CETB should borrow this discipline: justify each pipeline choice independently, record compatibility, and report the distribution and sources of outcome variation.

## 2. Proposed CETB competitive position

CETB should not compete with BIDS, fMRIPrep, XCP-D, MNE, pyspi, NSI or the clinical PCI literature by duplicating their primary functions. It should connect them through an assurance layer:

1. **Measurement sufficiency:** Is the signal/map usable for this specific inferential target?
2. **Repeatability:** Does the same participant's measurement reproduce under matched conditions?
3. **Construct validity:** Is the measured endpoint actually the construct claimed?
4. **Selection integrity:** Were every feature, model, threshold, preprocessing and estimator choice isolated from holdout data?
5. **Transportability:** Does the claim hold at an unseen site/device and in an independent cohort?
6. **Theory discrimination:** Were competing predictions frozen before the experiment and evaluated with proper predictive scores?
7. **Evidence receipts:** Can an independent analyst reproduce the exact result from versioned source, code, configuration, environment and output digests?

The key research product is not a new one-number "consciousness score". It is a claim-scoped qualification record connecting data quality, repeatability, construct validity, nested selection, external validation and theory-specific prediction.

## 3. New RCAF calibration benchmark

The RCAF Rust kernel now includes a synthetic Gaussian-factor generator with:
- independent same-state segment pairs for a known null;
- a known module-loading change for injection/recovery;
- finite-sample correlation estimation;
- participant-level paired gaps and exact sign testing;
- Monte-Carlo false-positive and power estimates with Wilson intervals.

The current test fixture is intentionally narrow: 24 participants, 12 regions, 160 time points per segment, 5 injected module regions, loading increment 0.40 and 80 Monte-Carlo replicates. The production example defaults to 100 replicates. The smoke-test bounds are loose engineering gates (null rejection estimate ≤0.15, injection recovery ≥0.80 and a separation in median paired gaps); they are not biological performance targets or sufficiently precise estimates of a nominal 5% tail.

**Next simulation surface:** sweep injection magnitude, participant count, segment duration, module size, participant heterogeneity, missing edges, neutral threshold, estimator family, global nuisance and temporal autocorrelation. Report full null and injection-recovery curves with Monte-Carlo uncertainty. Test invalid ROI ordering and insufficient edge support explicitly.

A method that recovers injections but has inflated null rejection is not acceptable. A method that controls the null but cannot detect meaningful changes is not useful. Sensitivity without specificity and repeatability is not a valid superiority claim.

## 4. Benchmarks CETB should target

All thresholds below are proposed engineering targets requiring preregistration and independent evaluation:

| Dimension | Target |
|---|---|
| Synthetic null calibration | Observed false-positive rate consistent with nominal alpha across a broad nuisance grid; report interval and Monte-Carlo uncertainty |
| Injection recovery | Monotonic dose–response over predeclared perturbation strengths; report power and effect-estimation bias |
| Measurement sufficiency | Estimate usable-data probability and expected gain from more data; compare with NSI where modality and purpose match |
| Repeatability | Participant-level within-state reliability across duration, gap and estimator choices |
| Selection integrity | Zero test-participant influence on preprocessing, feature selection, estimator selection, hyperparameters, calibration, thresholds and site correction |
| Site generalization | Nested leave-one-site-out evaluation with all transforms fit on training sites only |
| Independent confirmation | Locked separate cohort or acquisition; no adaptation to its outcomes |
| Multiverse robustness | Predeclare defensible choices; report effect/interval distributions and explain which choices alter conclusions |
| Reproducibility | Exact source, code, environment, configuration, input and output digests |
| Theory prediction | Frozen predictive distributions scored on untouched outcomes with proper scoring rules |

Do not set a universal minimum accuracy. The correct target depends on endpoint, class prevalence, intended use, error costs, measurement quality and data shift. For some research claims, calibrated repeatability and transportability are more valuable than a small within-cohort accuracy gain.

## 5. Immediate experimental priorities

1. Validate the synthetic RCAF null and injection-recovery implementation on its exact focused CI head.
2. Expand to a parameter sweep and nuisance families, separating code calibration from biological simulation.
3. Compare RCAF with Pearson, Spearman, shrinkage covariance/precision and selected pyspi estimators under matched inputs.
4. Download one authorized, fully pinned COGITATE/DS006623 source bundle, reconcile its file manifest and reproduce the published pipeline before changing it.
5. Add fold-local preprocessing and site-holdout sensitivity as separate corrected analyses.
6. Lock one genuinely independent acquisition for later confirmation; never call a different split of the same cohort an independent replication.

## 6. Claim limits

- The ABIDE-I leakage study is not a consciousness benchmark.
- NSI is a precision-fMRI map-quality framework, not a general EEG/MEG reliability metric.
- The COGITATE 246-measure paper is a same-cohort secondary comparison, not an independent cohort.
- RCAF's synthetic injection/recovery only demonstrates behaviour under a declared Gaussian factor model.
- CETB has not yet reproduced the published COGITATE or ds006623 analyses using biological signals.
- No consciousness detection, causal mechanism, clinical readiness or better-than-SOTA claim is established.


### Temporal dependence added to the synthetic world

The RCAF generator now supports stationary AR(1) autocorrelation in the shared factors and region-level residual processes. The default is rho=0.30; the report and JSON schemas carry the selected value. This creates autocorrelated finite time series while preserving the declared cross-sectional factor model.

It is still not a realistic BOLD/EEG/MEG pipeline simulator: no hemodynamic response, motion, physiology, source leakage, sensor noise, nonstationarity or preprocessing artifacts are represented yet. The next surface should vary rho and include nuisance mechanisms separately rather than calling this model "brain-like".


### Matched-input estimator competition

RCAF now compares three estimator semantics on identical simulated matrices: absolute Weighted Jaccard, signed-shift Weighted Jaccard and normalized mean absolute edge error. The polarity-flip known-answer fixture demonstrates why their outputs are not directly interchangeable (1.0, 1/3 and 0.5 respectively for a single +0.5→−0.5 edge).

The matched-input tournament runs null false-positive and injected-change sensitivity for each estimator, preserving a common seed. CETB should extend this tournament to Pearson, Spearman, shrinkage covariance/precision and modality-appropriate pyspi estimators once the data representation and estimand are matched. Do not select a universal winner from synthetic performance alone.


### Direct pipeline benchmark baselines

CETB should explicitly benchmark against the *end-to-end pipeline* literature, not only estimator libraries.

- The 2024 systematic connectomics study compared **768** network-construction pipelines across **13,824** pipeline/criterion/data evaluations. It assessed repeatability, motion sensitivity, subject discrimination and sensitivity to experimental effects across multiple datasets and timescales. Many pipelines failed at least one criterion, and some could report a reproducible effect in the wrong direction. This is a particularly close methodological precedent for CETB's multidimensional qualification envelope. DOI: [10.1038/s41467-024-48781-5](https://doi.org/10.1038/s41467-024-48781-5).
- A 2026 two-site resting-state study compared **256** FC pipelines for eyes-open/eyes-closed classification and reported approximately 90% accuracy for top configurations under cross-site evaluation. This is a useful pipeline-robustness benchmark, but eye-state classification is a different endpoint from propofol-state repeatability or consciousness inference. DOI: [10.1186/s40708-026-00305-1](https://doi.org/10.1186/s40708-026-00305-1).
- The pyspi-based 239-statistic study benchmarked 239 interaction statistics across several FC properties on HCP participants; it establishes that estimator performance depends on the scientific criterion rather than supplying a single universal winner. DOI: [10.1038/s41592-025-02704-4](https://doi.org/10.1038/s41592-025-02704-4).

CETB's opportunity is to turn these precedents into an explicit assurance layer that reports calibration, repeatability, direction-of-effect, artifact-only rejection, selection integrity and external transportability as separate dimensions. We must not compare a two-site eye-state AUC directly with an RCAF repeatability result or describe either as a consciousness score.

### PFM-NSI input compatibility

The public PFM-NSI README describes a default CIFTI `.dtseries.nii` fsLR-32k input and an optional fsaverage6 input path that uses Connectome Workbench for resampling. The ds003171 v2.0.1 BIDS release tree contains BOLD NIfTI `.nii.gz` files and no `.dtseries.nii` files. Therefore NSI is **not directly applicable to the released ds003171 inputs**. It remains a precision-fMRI sufficiency reference, but its status for this cohort is `not_assessed` until a compatible, independently qualified input conversion exists. Do not silently convert NIfTI volumes and treat the resulting NSI values as validated.

## 5. Separate measurement qualification axes

CETB now has a claim-scoped receipt schema at `research/cetb/measurement_qualification/measurement-qualification.schema.json`. It requires separate records for measurement sufficiency, participant-level repeatability, construct validity and three transportability lanes (participant holdout, site holdout, independent cohort). It explicitly forbids a composite score.

The framework treats NSI as an imported fMRI measurement-sufficiency method, RCAF as a participant-paired repeatability method, and external validation as a separate transportability test. None substitutes for the others. The protocol is `CETB_MEASUREMENT_QUALIFICATION_STACK_V0.md`.


### Estimator-by-effect-size comparison

A matched-input estimator × dose response surface is now implemented as a further extension of the first tournament. It compares absolute WJ, signed-shift WJ and normalized mean absolute edge error at each of five perturbation levels, keeping the random seed and input draws aligned across methods. This will help distinguish useful operating ranges rather than comparing three saturated strong-effect results.

The output contains a full dose-response curve and Monte-Carlo confidence intervals for each metric. It deliberately does not rank or average metrics with different construct semantics.


## 6. First executable RCAF estimator-by-dose results

The matched-input three-estimator by five-dose surface passed exact-head focused validation on source commit `335d7430246ff01f013b96a934601de256f3ea23` (workflow `44899300679`). Each cell used 40 synthetic Monte-Carlo replicates; all estimators had null rejection 1/40 (2.5%, 95% Wilson interval 0.44%–12.88%).

At module-loading increment 0.30, injection detection was:
- absolute WJ: 45% (95% Wilson interval 30.7%–60.2%);
- signed-shift WJ: 85% (70.9%–92.9%);
- normalized mean absolute edge error: 92.5% (80.1%–97.4%).

At 0.40, all reached 100% (40/40; Wilson lower bound about 91.2%). This supports a focused hypothesis that polarity-aware WJ and normalized edge error are more sensitive to moderate perturbations in this one factor model, but it is not a universal estimator ranking. Only 40 replicates per cell were used, and the different similarity definitions have different construct semantics. Final simulation calibration should use at least 1,000 replicates per cell and a broader nuisance grid.
