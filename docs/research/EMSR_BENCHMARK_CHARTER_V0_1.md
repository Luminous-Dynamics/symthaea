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

### 2.2.2 Cross-scenario spatio-temporal graph model

Li et al., *Spatio-Temporal Graph Neural Network for GNSS Spoofing Detection: A Cross-Scenario Generalization Study*, ION GNSS+ 2026 (14–18 September 2026), reports a 96.4% detection rate and 2.26% false-alarm rate under a mixed TEXBAT split. Under a cross-scenario protocol excluding ds6 and ds8 from training while retaining ds7, overall detection falls to 58.6%. The paper reports particularly difficult transfer to dynamic matched-power spoofing.

Source: https://doi.org/10.33012/2026.20700

**Benchmark implication:** Include a spatio-temporal graph model as a comparator only with matched scenario labels, graph construction and event splits. Require leave-one-scenario-family-out and progressively harder held-out scenarios. Report per-scenario recall, false alarms and detection delay; do not hide the 58.6% cross-scenario result behind the mixed-split score.

### 2.2.3 RAIM-driven semi-supervised model

Jiang et al., *A RAIM-Driven Latent Space Diffusion Approach for GNSS Spoofing Detection*, Advances in Space Research (available online 30 July 2026), proposes a semi-supervised latent-space diffusion approach with an adaptive GCN-derived RAIM weighting mechanism for shipboard navigation scenarios.

Source: https://doi.org/10.1016/j.asr.2026.07.097

**Benchmark implication:** Treat this as a maritime-specific research comparator after verifying the full method, data, baselines and reproducibility. Compare it with conventional RAIM residual monitoring and receiver-native integrity outputs. A learned RAIM weight matrix must not be assumed to satisfy a certified aviation or maritime integrity risk without independent qualification.

### 2.2.4 Open-set semi-supervised RF spectrum sensing

Cai et al., *Reliability-Gated RF Consistency Learning for Open-Set Spectrum Sensing* (Physical Communication, available online 2 September 2026), proposes RF-RPL. It combines RF-consistency representation learning, an EMA teacher across RF-preserving transformations, confidence/stability/energy-based knownness gating, pseudo-label routing and structure learning for rejected windows.

On RadioML2016.10A/10B and a simulated electromagnetic-monitoring benchmark, the paper reports 72.1% accuracy, 70.3% macro-F1, AUROC 0.894, FPR95 0.286 and 5.9% unknown pseudo-label contamination at 5% label availability.

Source: https://doi.org/10.1016/j.phycom.2026.103352

**Benchmark implication:** Add RF-RPL as a separate open-set signal-awareness comparator. Reproduce its scarce-label and unknown-contamination evaluations before extending it to new datasets. Evaluate confidence-only, transform-stability-only, knownness-only and full-gate ablations. Report unknown pseudo-label contamination and FPR at fixed known-class recall. Do not compare these RadioML/simulated results directly with GNSS campaign results. An unknown signal is not an attribution or malicious-emitter label.

This work is tracked separately in Symthaea #10893. It is adjacent to EMSR/PNT but does not establish navigation integrity.

### 2.2.4 Domain-adversarial cross-scene comparator

Lei et al., *Cross-Domain GNSS Spoofing Detection via Domain-Adversarial Neural Networks*, ION GNSS+ 2026, reports a dual-branch architecture combining physically interpretable signal-quality features (correlation peak shape, power correlation, phase locking and temporal evolution) with learned channel-attention features. In cross-scene simulations, the basic DANN reports 92.80% accuracy and 85.49% detection at a 1.38% false-alarm rate; the improved DANN reports 95.90% detection at the same false-alarm level.

Source: https://doi.org/10.33012/2026.20876

**Benchmark implication:** Add DANN and the physically interpretable feature branch as explicit comparators/ablations. Reproduce the published cross-scene simulation conditions first, then evaluate on independently recorded receiver/campaign holdouts. The reported figures are cross-scene simulation results, not field qualification.

### 2.2.5 Lightweight multi-domain spatio-temporal fusion

Ye and Chen, *Cross-Domain Feature Fusion for GNSS Spoofing Detection with Lightweight Spatio-Temporal Attention Networks*, ION GNSS+ 2026, combines correlation-, observation- and positioning-domain features, temporal per-satellite modeling and cross-satellite attention. Its abstract reports a satellite-wise TPR of 94.59% at 1% FPR, 229,507 trainable parameters and 0.173 ms/sample on its scenario-diverse evaluation dataset.

Source: https://doi.org/10.33012/2026.20831

**Benchmark implication:** This is a particularly relevant efficiency/feature-fusion comparator. Reproduce the same feature availability, masks, satellite/epoch alignment and evaluation protocol before comparing. Report inference latency on matched hardware, and test unseen receiver, campaign and scenario families. The abstract-level latency is not a device-independent real-time guarantee.

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

## 3.2 Edge AI and ESP32 deployment boundary

The Luminous Platform edge-node profile (#300) is the owner for exact-board embedded inference and auxiliary sensor qualification. ESP32-class devices may contribute equipment-health telemetry, low-rate feature processing, buffering and bounded local inference where matched hardware tests justify it. They are not presumed to be wideband GNSS raw-IF/IQ acquisition front ends or PNT integrity authorities.

For any edge model, pin the board/SoC/module revision, sensor and calibration, firmware, SDK/runtime, model digest, quantization/operator map, compiler configuration and power/thermal state. Benchmark the same frozen windows on a host reference and the exact edge board. Report macro/per-class detection, open-set rejection, calibration, false alerts per operating day, sample loss, P50/P95/P99 latency, memory, energy per decision and offline behavior.

The acronym ESP is ambiguous in this programme:
- **ESP32** means Espressif embedded hardware, relevant to the edge-node track.
- **ESP (extrasensory perception)** is a separate unestablished research hypothesis, tracked as anomalous-information acquisition in Symthaea #10843.
- Neither is a synonym for RF signal processing. RF acquisition requires a separately qualified receiver/front end, sampling clock, antenna, calibration and data chain.

No ESP/AIA result may be used as an EMSR sensor input, PNT evidence, safety claim or operational authority. The AIA investigation remains a blinded, preregistered research experiment only.

## 3.1 Ground-truth and oracle independence

A benchmark cannot claim independent detection evidence if its labels, event times, receiver telemetry and adjudication all derive from the same source or campaign assumption.

For each evaluation set, preserve:
- ground-truth issuer and method;
- event-onset time source, time basis and uncertainty;
- receiver under test versus independent reference receiver identity;
- whether the label is directly measured, operator-reported, simulated, inferred or independently adjudicated;
- dependencies shared between label source, detector inputs and evaluation environment;
- adjudication disagreements and unknown truth states.

The evaluator must distinguish physical-event truth, receiver-integrity truth, and service-level consequence. A recorded interference event is not automatically proof that the navigation solution was invalid; a navigation residual is not automatically proof of an interference event.

Where truth provenance is insufficient, report the case as ambiguous/indeterminate or place it in a separate exploratory stratum. Do not silently turn model-derived labels into oracle ground truth. The benchmark receipt should bind the adjudication manifest and its exact digest.

## 3.3 Industry platform comparators

In addition to receiver vendors, compare against existing PNT monitoring/integrity platforms and software libraries:

- **SESAR real-time GNSS monitoring/interference solutions:** area/route-level monitoring, sentinel/receiver data streams, GNSS performance indicators, A-RAIM protection-level estimates, APIs and SWIM interfaces. This is a direct platform-level comparator for monitoring feeds and service APIs: https://sesarju.paddlecms.net/sesar-solutions/real-time-gnss-monitoring-and-interference
- **CISA PNT Integrity Library:** a modular framework for GNSS-based PNT manipulation detection intended for integration by receiver and timing-server OEMs: https://github.com/cisagov/PNT-Integrity
- **PNTMAP:** an event/intelligence API and alert/webhook service. Its published capability status distinguishes currently live ADS-B-derived monitoring from direct-RF sensing/validation that has not yet been demonstrated in its described deployment: https://pntmap.co.uk/api-access
- **Receiver-native vendor status and protection outputs:** remain first-class operational baselines wherever exact hardware is available.

The Luminous differentiation hypothesis is not “we have a map/API too.” It is vendor-neutral evidence qualification across source identity, calibration, time, shared dependencies, oracle independence, downstream service consequence, approved degradation and recovery. This must be demonstrated head-to-head on matched data and scenarios. The cited platforms are capability comparators, not claims that Luminous already surpasses them.

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
2. Receiver-native integrity/authentication outputs as separate operational baselines: RAIM/ARAIM protection-level/status outputs, SBAS integrity status and Galileo OSNMA authentication state where available. These are not interchangeable with ML anomaly scores.
3. Simple signal/receiver status threshold baseline.
4. The 2026 raw-IF chi-square method, faithfully reproduced where source and data rights permit.
5. Conventional supervised CNN or comparable feature model.
6. Published cross-domain comparator from the 2026 IEEE Access study, subject to faithful reproduction.
7. 2026 NAVITEC CNN versus MAML cross-testing comparator across TEXBAT, OAKBAT and FGI-SpoofRepo, subject to faithful reproduction.
8. ION GNSS+ 2026 domain-adversarial neural network (DANN) cross-scene simulation comparator, plus physical-feature-only and learned-feature-only ablations.
9. ION GNSS+ 2026 lightweight multi-domain spatio-temporal feature-fusion comparator, with matched hardware latency and per-domain performance.
10. Spatio-temporal graph model cross-scenario comparator (ION GNSS+ 2026), if faithful data/protocol reproduction is possible.
9. RAIM-driven semi-supervised maritime comparator (Advances in Space Research, 2026), if full-method and data reproduction conditions are available.
10. Collaborative PVT-residual/C/N0 integrity comparator.
11. Explainable multimodal statistical + 1D temporal + 2D time-frequency fusion baseline, with per-modality ablations and matched hardware profiling where data conditions permit.
12. No-collaboration ablation.
13. No-provenance / no-common-mode analysis ablations.
14. Symthaea uncertainty-aware assessment.
15. Integrated evidence-qualified service-continuity workflow.
16. Open-set RF-RPL-style semi-supervised signal-awareness comparator, under the separate #10893 track and only on compatible recorded RF datasets.

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
