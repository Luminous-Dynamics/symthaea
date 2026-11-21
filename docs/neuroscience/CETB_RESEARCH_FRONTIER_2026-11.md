# CETB Research Frontier and SOTA Map — 2026-11

**Status:** Research landscape review and proposed benchmarks. It reports no new experiment or performance result.

## 1. What has changed in the field

### Adversarial theory testing

The COGITATE collaboration demonstrated the value of theory proponents and an independent consortium agreeing on divergent predictions, design and interpretations before the data are analysed. CETB should generalize this approach into versioned, machine-readable prediction contracts, with independently scored outcomes and immutable preregistration.

Reference: Melloni et al. (2025), Nature, DOI 10.1038/s41586-025-08888-1.

### Selection-aware COGITATE MEG estimator comparisons

Bryant and Whyte (2026), *Neuroscience of Consciousness*, DOI 10.1093/nc/niag029, evaluated 262 pyspi v1.1.0 pairwise statistics on COGITATE BIDS MEG from 100 participants, with 94 source-localizable participants. A declared availability/variance filter retained 246 measures, and the primary aim examined 35,280 model configurations. Barycenter-derived measures were among the strongest stimulus decoders. The authors report substantial cross-validation fold variability and caution that the analysis does not strongly establish whether either theory's mechanisms are necessary or sufficient.

This is the closest direct comparator for our MEG estimator tournament, but it reuses the COGITATE cohort. CETB will therefore separate historical reproduction from nested participant-grouped metric selection and independent confirmation. The statistic family, region pair, event window, classifier, scaling and calibration must all be selected inside inner training folds; outer folds evaluate the whole selection procedure. Cross-task transfer within a participant is not unseen-participant generalization.

The 2026-10-15 correction (DOI 10.1093/nc/niag071) replaces an earlier supplement and corrects the model equation and table presentation. Any replay must use the corrected supplement.

Reference: Bryant & Whyte (2026), DOI 10.1093/nc/niag029; correction DOI 10.1093/nc/niag071.

### Repeatability-calibrated connectomics

A 2026 NeuroImage: Reports article reanalyses OpenNeuro ds006623 using full correlation-matrix Weighted Jaccard (WJ), a within-condition floor and group-averaging controls. It is an important same-dataset method comparator, not an independent replication.

Reference: Harbert (online 2026-10-04), DOI 10.1016/j.ynirp.2026.100417. The author's analysis code is MIT-licensed and pinned in the CETB RCAF intake audit.

### Clinical TMS–EEG

The 2026 international expert review describes PCI for disorders of consciousness as late stage 3 / early stage 4 in its clinical-translation framework. It also identifies a lack of consensus on a complete end-to-end pipeline and notes that the standard multi-target protocol can require 64-channel EEG and several hours. PCI-ST on 19-channel data is promising but needs standalone validation.

Reference: Ziemann et al. (2026), Clinical Neurophysiology, DOI 10.1016/j.clinph.2025.2111487.

A separate 2026 review emphasizes preprocessing artifacts, inter-individual variation, state dependence and limited standardization in TMS–EEG. This is why CETB should measure analysis-choice sensitivity rather than hide it behind one selected pipeline.

Reference: Wang and He et al. (2026), Frontiers in Human Neuroscience, DOI 10.3389/fnhum.2026.1804482.

## 2. Industry / SOTA comparison

| Existing ecosystem | What it already does well | CETB should do instead of competing |
|---|---|---|
| COGITATE | Adversarial theory design and multimodal study practice | Make predictions versioned, scored and replayable across studies |
| OpenNeuro / BIDS | Standardized, shareable neuroimaging datasets | Preserve exact data/version lineage and interoperable manifests |
| fMRIPrep / XCP-D | Reproducible preprocessing and derivatives | Pin, compare and qualify derivatives; do not rebuild these pipelines |
| MNE / EEGLAB / FieldTrip | EEG processing and TMS–EEG analysis workflows | Compare pipeline sensitivity, provenance and output agreement |
| Clinical PCI research | Physical perturbation and clinically motivated complexity measures | Do not imitate physical PCI in simulation; strengthen reproducibility and transportability |
| Conventional ML | Strong task-specific prediction with established metrics | Evaluate participant/site transfer, calibration, abstention and negative controls |
| pyspi / 246-statistic COGITATE MEG analysis | Broad measure search and theory-informed neural-mass comparisons | Add nested feature-family selection, participant/site holdouts, repeatability and immutable selection ledgers |

The strongest near-term differentiation is not a higher accuracy claim on a small cohort. It is a better measurement and evidence contract that exposes when a purported effect is smaller than within-state noise or changes materially under reasonable analysis choices.

## 3. Proposed paradigm shifts

1. **Uncertainty-aware state transitions:** every between-state difference is evaluated against within-state repeatability at matched recording length.
2. **Analysis multiverse as a first-class object:** every preregistered, scientifically defensible preprocessing/estimator variant is reported; no best-pipeline selection after outcomes are visible.
3. **Hierarchical evidence:** subject, session, run, segment, device, site and pipeline variance are kept distinct.
4. **Theory discrimination:** experiment selection optimizes expected reduction in disagreement between theories, not generic information gain.
5. **Causal perturbation contracts:** a physical TMS intervention, a simulated perturbation and an observational network contrast are distinct evidence types.
6. **Transportability before scale:** participant-held-out, then site/device-held-out evaluation precedes flexible model expansion.
7. **Evidence-carrying outputs:** data, code, environment, configuration, participant exclusions and result digests travel with every finding.

## 4. Proposed benchmark targets

These are programme targets, not achieved results:

| Dimension | Target |
|---|---|
| Reproduction parity | Exact metric-kernel parity on frozen arrays; end-to-end outputs within a preregistered tolerance |
| Repeatability | Within-state floors reported by condition, duration and participant |
| Effect robustness | Directional agreement in at least 90% of predeclared defensible variants, with all exceptions reported |
| Statistical unit | Participant-level or explicitly hierarchical inference; never count windows as independent people |
| Leakage | Zero participant leakage and a machine-checked split manifest |
| External validation | At least one independent dataset/site when construct and modality match |
| Calibration | Proper scores and calibration by participant, condition and domain |
| TMS–EEG | Report artifact sensitivity, test–retest reliability, pipeline agreement and acquisition burden |
| Reproducibility | Independent clean-environment replay with matching digests |

No single composite “consciousness score” should combine these targets.

## 5. A better-than-SOTA claim protocol

A candidate must beat the strongest eligible baseline on the same dataset version, cohort, endpoint, preprocessing, split, tuning budget and metric. It must also:
- retain the gain on a frozen held-out participant/site set;
- remain calibrated;
- survive negative controls and declared pipeline perturbations;
- report participant-clustered uncertainty;
- preserve abstention and unsupported cases;
- provide an independently replayable evidence receipt.

If those conditions do not match, classify the comparison as non-comparable, not as a leaderboard victory.

## 6. Immediate execution sequence

1. Reproduce Bryant & Whyte's COGITATE MEG 246-measure study using the corrected supplement, while keeping historical replay separate from nested selection-aware validation.
2. Independently replay the author's ds006623 WJ revision code from the pinned MIT repository.
3. Reproduce the original ds006623 ISD implementation as a separate pipeline.
4. Implement RCAF's small, dependency-free WJ / paired-repeatability metric kernel.
5. Add per-condition floor, guard-gap and segment-duration analyses.
6. Compare subject-level and group-level aggregation.
7. Only after these baselines are stable, test HDC/CfC/LTC predictors of explicit behavioural or report outcomes.
8. Reserve the NEMAR repeated-awakening dataset for a later report-aligned EEG lane after its exact version, event alignment and licence statements are resolved.

The near-term scientific win is a reliable, falsifiable measurement benchmark—not a claim that the system has detected consciousness.
