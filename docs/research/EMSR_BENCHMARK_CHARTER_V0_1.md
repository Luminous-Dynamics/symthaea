# EMSR Benchmark Charter v0.1

**Status:** Proposed evaluation contract; no benchmark results claimed  
**Date:** 2026-11-06  
**Parent:** Symthaea EMSR-001 / issue #10775  
**Implementation tracker:** EMSR-002 / issue #10814  
**Scope:** Defensive, receive-side PNT and communications resilience

## 1. Research thesis

The benchmark should test whether a system can establish the trustworthiness of PNT-related information and preserve safe service continuity when sources degrade—not merely whether a classifier can recognize a labelled RF event.

The target research contribution is **evidence-qualified service integrity**: a decision package that binds receiver observations, time quality, calibration, configuration, independent-source relationships, service consequences, and recovery prerequisites to an exact evidence snapshot.

An anomaly label is not a cause determination. A cause hypothesis is not actor attribution. A verified signature is not physical truth. A predicted safe mode is not operational authorization.

## 2. Current comparative evidence

### 2.1 Statistical anomaly detection

Bhuiyan et al., *Implementation and performance analysis of a Chi-Square Test based GNSS signal anomaly detection*, GPS Solutions (29 June 2026), evaluates a raw-IF histogram chi-square method across Jammertest 2023, TEXBAT, OAKBAT, and FGI-SpoofRepo, using four RF front ends and FGI-GSRx.

The paper reports detection accuracy above 99% with empirical false-alarm rates near zero in its selected conditions. It also reports notable reductions on TEXBAT ds7 and ds8 (approximately 87% and 93%). The authors explicitly state that the current method detects anomalies but does not distinguish jamming from spoofing; type classification is future work.

Source: https://link.springer.com/article/10.1007/s10291-026-02102-z

**Benchmark implication:** Reproduce this as a strong statistical comparator. Do not describe it as a universal detector or as interference-source attribution. Match its raw-IF preprocessing, nominal initialization, threshold and evaluation conditions before comparing results.

### 2.2 Cross-domain machine learning

Khan et al., *GNSS-Spoofing Detection via Machine Learning With In-Domain and Cross-Domain Testing and Open-Access I/Q Data*, IEEE Access (13 May 2026), evaluates four ML models using TEXBAT and OAKBAT repositories. Its abstract reports average in-domain accuracy above 90%, but average cross-testing accuracy of 66% or less.

Source: https://doi.org/10.1109/ACCESS.2026.3693081

**Benchmark implication:** Cross-domain generalization must be a primary track, not a supplementary experiment. Random-window splits are insufficient. Report exact train/test dataset, receiver, campaign, location and scenario families.

### 2.2.1 Meta-learning and cross-testing

Khan et al., *Evaluating ML-Based Algorithms for GNSS Spoofing Detection with Cross-testing Capabilities: A Comparative Study*, 2026 12th Workshop on Satellite Navigation Technologies (NAVITEC), published in IEEE Xplore 22 September 2026, evaluates a conventional CNN and a Model-Agnostic Meta-Learning (MAML) approach across TEXBAT, OAKBAT and FGI-SpoofRepo. Its abstract reports the strongest cross-testing with MAML using post-correlation Doppler features and notes that training on more complex spoofing scenarios (including TEXBAT DS8) improves generalization across datasets.

Source: https://doi.org/10.1109/NAVITEC202669090.2026.11692526

**Benchmark implication:** Add MAML/meta-learning and leave-one-dataset-out evaluation as mandatory research comparators after faithfully reproducing the published protocol. Compare feature extraction stages and scenario diversity under locked event/receiver/dataset splits. Do not treat the abstract's qualitative result as a numerical performance guarantee.

### 2.3 Collaborative integrity

*Multicriteria Collaborative GNSS Integrity Monitoring*, IEEE Journal of Indoor and Seamless Positioning and Navigation (4 May 2026), combines PVT residuals and carrier-to-noise observations across receivers. It reports real jamming data and simulated spoofing data.

Source: https://doi.org/10.1109/JISPIN.2026.3690078

**Benchmark implication:** Compare single-receiver assessment with collaborative assessment, and explicitly test shared failure domains. The number of reports, signatures or agents is not an independence measure.

### 2.4 Explainable multimodal jammer classification

Savaş, *An explainable multi-modal fusion framework for robust detection and classification of GNSS jamming attacks in critical infrastructures*, International Journal of Information Security (19 May 2026), reports 99.33% test accuracy for a fusion of engineered statistical features, raw 1D temporal features and 2D time-frequency features, with SHAP and Grad-CAM explainability. The paper reports mean inference latency of 95.60 ms per segment.

Its authors also state that the data are simulated and that complex real-world multipath, atmospheric effects, moving receivers/interference sources, dynamic environments, and composite interference remain to be evaluated.

Source: https://doi.org/10.1007/s10207-026-01267-5

**Benchmark implication:** Include the model as a multimodal/XAI comparator only when source data, feature preprocessing, class definitions and split policy can be matched. Do not compare its 99.33% simulated-data test accuracy directly to field-campaign cross-domain scores. Reproduce the modality ablations and report latency/throughput on named hardware, then test real-world holdouts separately.

A 28 July 2026 review of multi-source GNSS spoofing detection also identifies heterogeneous corroboration sources such as INS, vision and communications as a major research direction. This supports a later PNT-service integrity track, provided each alternative source has independently documented failure domains and time quality:
https://doi.org/10.3390/electronics15153326

## 2.4 Dataset diversity and rights

- FGI Jammertest 2025 receiver corpus: SBF/RINEX, receiver status, corrections, metadata and ground truth; collected 15–17 September 2025 with a Septentrio PolaRx5. The Etsin listing states CC BY 4.0 and notes that the dataset is cumulative.
  - https://etsin.fairdata.fi/dataset/095782a8-4e31-456e-8719-7021b4c96362
- Fraunhofer Jammertest 2025 raw-IQ corpus: HDF5 recordings from two areas and two receiver/antenna configurations (Innosense E1/E5 module and a 2×2 CRPA array). The record recommends random-within-area, area-transfer, joint-area, receiver-transfer and weak-event filtering evaluations. License: CC BY-NC-SA 4.0.
  - https://zenodo.org/records/21332689
- Jammertest 2024 v3: recorded receiver data; exact source-level terms and permitted use must be reviewed before ingestion.
  - https://zenodo.org/records/15911589
- Jammertest 2023 raw IF corpus used by the chi-square paper: 141.25 GB across three scenarios, released for independent research reproduction.
  - https://doi.org/10.23729/fd-06d27736-45cb-3ca2-aff8-725d42c6caeb

Public discoverability is not permission for commercial use, redistribution, or model training. Rights are reviewed per source, version, file and intended use. The Fraunhofer CC BY-NC-SA corpus must not be treated as commercially cleared. The Stanford GNSS interference monitoring data are also CC BY-NC and require separate permission for commercial use.

## 3. Industry comparison and product boundary

Receiver vendors already provide serious receiver-level capabilities:

- Septentrio AIM+: real-time spectrum monitoring and event flags, detection, adaptive/digital filtering and graduated protection levels.
  - https://www.septentrio.com/en/learn-more/advanced-positioning-technology/gps-gnss-interference
- NovAtel / Hexagon GRIT: receiver-integrated spoofing detection, interference spectrum analysis, time-tagged digitized samples and digital-filter mitigation.
  - https://novatel.com/products/grit

Luminous should not claim to outperform receiver firmware, antenna arrays, CRPA processing or anti-jam hardware without matched hardware testing. The proposed differentiation is a vendor-neutral assurance and continuity layer above those products:

1. Normalize receiver and service observations without discarding vendor-origin evidence.
2. Track identity, calibration, time, configuration and source lineage.
3. Detect common-mode dependencies and prevent false corroboration.
4. Translate PNT/time/communications degradation into downstream service consequences.
5. Bind continuity recommendations and recovery conditions to current evidence.
6. Independently reconstruct the incident and recommendation offline.

NIST IR 8323 Rev. 2 initial public draft (6 May 2026) is a useful PNT risk-management mapping reference aligned to CSF 2.0; it is a draft, not a final normative standard:
https://csrc.nist.gov/pubs/ir/8323/r2/ipd

## 4. Paradigm shifts

### A. Detection -> service integrity

A detector answers whether an observation is anomalous. The assurance layer must answer whether a particular position, time or communications service remains dependable for a declared use.

Keep separate the observed symptom, cause hypothesis, uncertainty, service impact, permitted response and recovery prerequisite.

### B. Sensor count -> failure-domain independence

Represent dependencies among sensors, clocks, power, correction services, firmware, networks, operators and data pipelines. Two signed observations sharing one critical upstream dependency are not independent corroboration.

Expose a conflict graph and provenance. Do not compress it into a universal trust score.

### C. Point prediction -> calibrated hypotheses and abstention

Maintain benign alternatives (multipath, obstruction, receiver/antenna/cabling fault, timing fault, configuration drift, pipeline fault) alongside interference hypotheses. Return INDETERMINATE when the evidence cannot discriminate. Do not infer hostile intent from anomaly scores.

### D. Random splits -> domain-shift evaluation

Freeze separate tests for:
- receiver transfer;
- event/campaign transfer;
- site/area transfer;
- time-period transfer;
- dataset-family transfer;
- scenario-family transfer;
- weak-event and ambiguous-label subsets.

Adjacent windows from the same recording/event stay in one partition. Keep a separately managed qualification set; do not repeatedly tune on it.

### E. Fault disappearance -> recovery requalification

The return of apparently normal measurements does not automatically restore normal service. Require fresh observations, current calibration and configuration, trustworthy timing, current policy/revocation evidence, independent checks where needed, and domain-owner approval.

### F. Model score -> evidence-carrying decision

Every consequential recommendation references exact source bytes, dataset and split digests, model/code/configuration identity, evidence freshness, operating envelope and authorization owner. A model report never mints authority.

### G. Product lock-in -> vendor-neutral assurance

Use receiver-native flags and outputs as first-class evidence. Compare them on compatible hardware. Add value by correlating across systems and services, not by pretending a software layer replaces receiver-level RF protection.

## 5. Benchmark tracks and metrics

Do not collapse the suite into one weighted score. Report each metric by dataset, receiver, event family, site, campaign, and operating context.

| Track | Primary measurements |
|---|---|
| EMSR-DET: detection | Precision-recall AUC, ROC AUC, detection probability at fixed false-alarm budgets, false alarms per receiver-hour, detection latency |
| EMSR-GEN: generalization | Receiver-, campaign-, area-, date- and dataset-held-out performance; worst-domain and macro averages |
| EMSR-EPI: epistemic quality | Brier score, log loss, reliability diagrams, ECE, selective risk/coverage, abstention correctness, false attribution |
| EMSR-COR: source independence | False independent corroboration rate, shared-domain conflict recall, unknown-dependency abstention |
| EMSR-PNT: service integrity | Time to integrity alarm, misleading-solution exposure duration, invalid-solution acceptance after independent ground-truth loss, false exclusion, continuity |
| EMSR-SVC: service continuity | Approved degraded-mode transition success, false degradation, service availability and downstream service impact |
| EMSR-REC: recovery | Premature re-entry count, time to requalify, currentness failures detected, recovery evidence completeness |
| EMSR-AUD: assurance | Exact receipt completeness, independent offline reconstruction, tamper/mismatch detection |
| EMSR-OPS: deployability | P50/P95/P99 inference latency, throughput, CPU/RAM and resource cost on named hardware |

For aviation and other safety-critical domains, time-to-alert, protection levels, alert limits and acceptable continuity thresholds must be selected by the relevant domain authority. No universal false-alarm, latency or availability threshold should be imposed across aviation, maritime, telecom and industrial operations.

## 6. Mandatory baselines

1. Receiver-provided status/alerts and protection features where compatible physical hardware is available.
2. Simple signal/receiver status threshold baseline.
3. The 2026 raw-IF chi-square method, faithfully reproduced where source and data rights permit.
4. Conventional supervised CNN or comparable feature model.
5. Published cross-domain comparator from the 2026 IEEE Access study, subject to faithful reproduction.
6. 2026 NAVITEC CNN versus MAML cross-testing comparator across TEXBAT, OAKBAT and FGI-SpoofRepo, subject to faithful reproduction.
7. Collaborative PVT-residual/C/N0 integrity comparator.
7. No-collaboration ablation.
8. No-provenance / no-common-mode analysis ablations.
9. Symthaea uncertainty-aware assessment.
10. Explainable multimodal statistical + 1D temporal + 2D time-frequency fusion baseline, with per-modality ablations and matched hardware profiling where data conditions permit.
11. Integrated evidence-qualified service-continuity workflow.

Use matched data, splits, receiver inputs, labels, time basis and false-alarm budgets. Reproducing a paper's abstract-level headline is not reproduction.

## 7. Proposed targets (not achieved)

- 100% non-overlap of event and recording groups across train/validation/test partitions.
- At least two materially different receiver/data pipelines in the initial study.
- Beat the reproduced simple and published comparators on held-out domains at the same predeclared false-alarm budget.
- Stretch target after baseline reproduction: improve mean cross-domain balanced accuracy or macro-F1 by at least 5 absolute percentage points over the strongest faithfully reproduced baseline, without a material worst-domain regression. This is an experimental goal, not a forecast.
- Zero false-independent corroboration in the frozen synthetic common-mode challenge.
- Calibrated uncertainty and correct abstention; publish risk/coverage curves, not only a single calibration scalar.
- Zero premature recovery/re-entry in the frozen deterministic service-transition suite.
- 100% exact evidence references for high-impact recommendations, otherwise INDETERMINATE.
- 100% of reported runs independently reconstructable from raw digests, exact split, source version, code/model/configuration, environment and per-case outcomes.

A failed target is retained as a result and regression fixture. Never weaken the frozen holdout or false-alarm budget to obtain a pass.

## 8. Reproducibility and rights protocol

For each source, record dataset ID, immutable version/DOI, exact source terms, intended use, rights reviewer/time, payload file path and SHA-256, receiver/firmware/antenna metadata, calibration, time basis, label provenance and known limitations.

For each split, record exact dataset version and SHA-256, grouping dimensions, records and partition assignments. Event and recording group separation is mandatory; receiver, campaign and location dimensions are explicitly selected by each transfer experiment.

For each run, record exact Git commit, code/model/configuration digests, dataset catalogue/payload/split digests, toolchain, environment, seed, case-level outcomes and an independent verifier identity/report digest.

Frozen design fixtures contain no measurements. Only separate run receipts contain measured results. A valid receipt establishes provenance structure, not detector accuracy, safety, certification or commercial rights.

## 9. Defensive scope boundary

Recorded receive-side data and deterministic synthetic software replay only. No transmitter construction, RF attack waveform design/optimization, live interference, emitter targeting, kinetic cueing, weapons integration or offensive electronic attack workflows. No physical actuation or operational authorization is granted by this benchmark.
