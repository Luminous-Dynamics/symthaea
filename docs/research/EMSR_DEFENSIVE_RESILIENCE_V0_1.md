# Defensive Electromagnetic-Spectrum and PNT Resilience v0.1

**Status:** Research specification; not a deployed capability or certification  
**Date:** 2026-11-06  
**Parent issue:** Symthaea #10775 — EMSR-001  
**Projects:** Symthaea, Mycelix, Symtropy, Sol-Atlas

## 1. Purpose and bounded claim

Build a cross-stack research platform for detecting, characterizing, and responding safely to electromagnetic-environment degradation and positioning/navigation/timing (PNT) integrity loss.

The initial claim to test is:

> For a declared dataset, receiver class, asset configuration, service dependency graph, and scenario distribution, the system can identify degraded or uncertain observations with calibrated uncertainty, preserve source provenance, and support a safe, evidence-qualified transition to an approved degraded operating mode.

This is a defensive resilience and operational assurance programme. It is not an offensive electronic-warfare system.

## 2. Scope boundary

### In scope

- Receive-side monitoring using recorded or lawfully collected observations.
- Detection of PNT integrity anomalies and communications/timing service degradation.
- Separation of observed symptoms from causal hypotheses and actor attribution.
- Sensor identity, calibration, time-quality, configuration, and evidence provenance.
- Cross-sensor consistency, uncertainty, abstention, and benign-cause alternatives.
- Service continuity, safe degraded modes, incident reconstruction, and recovery.
- Deterministic simulation and software-level synthetic corruption.
- Interoperability with relevant public safety guidance and E3 assurance standards.

### Out of scope

- Jamming, spoofing, meaconing, or other interference transmitter construction or optimization.
- RF attack waveforms, transmission parameters, or methods to defeat or evade receiver protections.
- Emitter geolocation or tracking for target cueing, target selection, fire control, or kinetic engagement.
- Offensive electronic attack planning, battle-management integration, or weapons interfaces.
- Live over-the-air interference experiments.

The v0.1 corpus is receive-side and effects-disabled. Any later hardware validation would require specific lawful authorization, a shielded laboratory, and separate safety review. It must not create emissions that affect public spectrum users.

## 3. Why now: external evidence and guidance

Public operational and technical sources establish a concrete need:

- EASA published SIB 2022-02 Revision 4 on 3 July 2026 after analysis of recent GNSS interference occurrences. It addresses recognition, operational procedures, reporting, and alternative navigation means.
- EUROCONTROL published GUID-203, edition 1.0, on 5 November 2026. It organizes GNSS degradation response around preparedness, detection, coordination, response, and recovery for ATM/ANS.
- The UK Maritime and Coastguard Agency's MGN 719 (October 2026) recommends dependency reviews, independent navigation cross-checks, drills, and continuity planning in GNSS-denied/degraded environments.
- The Finnish Geospatial Research Institute's Jammertest 2025 receiver dataset was released through the Finnish research data service on 30 September 2026, with receiver observations, ground truth, and multiple interference-event classes.
- A June 2026 *GPS Solutions* paper evaluates raw receiver intermediate-frequency-sample anomaly detection using multiple public datasets and Jammertest data.
- A May 2026 IEEE paper studies collaborative GNSS integrity monitoring using multiple observations, including PVT residuals and carrier-to-noise information.
- The U.S. ASSIST catalogue lists MIL-STD-464 as active with a 9 February 2026 document date and describes its scope as electromagnetic environmental effects (E3) interface requirements and verification criteria for airborne, sea, space, and ground systems.

These publications are external references, not evidence that Luminous has achieved the corresponding performance or compliance.

## 4. Architecture and subsystem ownership

### Symthaea — epistemic assessment

Responsibilities:
- ingest receiver and sensor observations with immutable source references;
- detect anomalies and contradictions;
- maintain competing explanations, including intentional interference, unintentional RFI, multipath/environmental effects, receiver/antenna fault, time-source fault, configuration drift, and data-pipeline failure;
- calibrate uncertainty and abstain when evidence cannot distinguish explanations;
- preserve causal lineage between source observations and derived conclusions.

A Symthaea output is an evidence-qualified assessment, not an authorization to transmit or actuate.

### Mycelix — provenance and federation

Responsibilities:
- authenticate sensor/workload identities and issuer roles;
- preserve payload digests, schema versions, device/calibration/configuration lineage, timestamp quality, key lifecycle, trust policy, and revocation;
- distinguish identity authentication, payload integrity, calibration assurance, physical validity, and permission to disclose;
- support privacy-aware federation and offline evidence reconstruction.

A valid signature proves that a key signed bytes. It does not prove that the described physical measurement is true, calibrated, independent, or current.

### Symtropy — operational resilience simulation

Responsibilities:
- model service-level effects of lost or degraded PNT, timing, communications, or sensors;
- evaluate declared safe degraded modes, fallback availability, recovery, and requalification;
- replay deterministic scenarios and preserve event/effect traces;
- keep domain-specific physical safety authority separate from experimental cognitive metrics.

Symtropy v0.1 does not need an RF propagation or emitter-attack simulator. Do not create a second generic runtime-assurance controller.

### Sol-Atlas — infrastructure dependency and decision twin

Responsibilities:
- map asset/service dependencies on PNT, timing, communications, and monitoring;
- represent site/configuration identity and evidence currentness;
- calculate downstream service-impact scenarios without unsupported source attribution;
- evaluate advisory continuity options against current operating envelopes and decision-reliance requirements;
- produce offline-reconstructable decision packages.

Start with generalized public or synthetic infrastructure profiles. Do not publish precise vulnerabilities of live facilities.

## 5. Data contract

A receive-side observation should preserve at least:

- observation ID, source ID, source payload digest, schema and dataset version;
- receiver/device class, device and firmware identity where available;
- antenna/installation/configuration identity where available and appropriate;
- calibration identity, calibration validity and quality state;
- observation time, record time, clock source, time-quality/uncertainty, and known clock discontinuities;
- raw observation reference and derived-output references;
- receiver status and declared data-quality indicators;
- ground-truth quality and label provenance;
- rights, confidentiality class, permitted recipient/purpose, and retention limits.

Avoid precise location data by default. Retain only the spatial granularity needed for the declared civilian resilience purpose and lawful data use.

Do not silently infer units, receiver identity, time source, calibration, missing values, or label certainty.

## 6. State model

Use typed states rather than one Boolean:

- `NORMAL`
- `DEGRADED`
- `INTEGRITY_UNCERTAIN`
- `UNAVAILABLE`
- `RECOVERING`
- `INDETERMINATE`

Keep distinct:
- observed symptom;
- candidate cause;
- confidence/calibration;
- operational consequence;
- approved response;
- recovery/requalification condition.

No anomaly label alone establishes hostile intent or identifies an actor.

## 7. Benchmark plan

### Public development datasets

1. Finnish Geospatial Research Institute, Jammertest 2025 GNSS interference dataset, released 30 September 2026.
2. Jammertest 2024 receiver dataset, Zenodo DOI 10.5281/zenodo.15911589, version 3.
3. FGI GNSS Spoofing Dataset Repository (FGI-SpoofRepo), subject to source, license, and permitted-use review.
4. Public receiver/anomaly datasets cited in the June 2026 *GPS Solutions* study.
5. Publicly described collaborative integrity-monitoring results from the May 2026 IEEE study.

Before importing any binary data, record license/terms, exact version, file hashes, receiver classes, label semantics, time basis, calibration/ground-truth quality, and any restrictions. Do not assume a dataset is unrestricted merely because it is publicly discoverable.

### Split design

- split by event campaign and receiver/device group;
- hold out complete dates and collection campaigns;
- prevent adjacent windows from one event appearing in train and test;
- report receiver-specific and cross-receiver generalization separately;
- preserve a public development set and a separately managed qualification set;
- do not claim an external holdout when none was used.

### Baselines

At minimum compare:
1. documented receiver-provided status/alert outputs where available;
2. simple statistical threshold baseline;
3. conventional supervised classifier baseline;
4. anomaly model trained without labels;
5. multi-source integrity assessment using independent receiver/time-quality evidence.

The published statistical and collaborative methods are comparators, not implementation requirements. Reproduce methods faithfully where source code/data and licensing permit.

### Metrics

Detection:
- precision-recall AUC and ROC AUC with class prevalence;
- detection probability at a declared false-alarm rate;
- false alarms per receiver-hour;
- detection latency relative to independently timestamped event onset;
- calibration (Brier score, reliability diagrams, ECE);
- abstention and indeterminate rates;
- false attribution rates and confusion across benign vs intentional-cause labels.

PNT/service integrity:
- duration for which invalid/degraded PNT is accepted after independent ground-truth integrity loss;
- time to indicate integrity uncertainty or loss of service;
- false exclusion / unnecessary service loss;
- continuity and availability under each declared scenario;
- recovery time and correct requalification.

Operational resilience:
- safe degraded-mode transition rate;
- premature restoration / unsafe re-entry count;
- fallback availability and dependency independence;
- service impact across dependent assets;
- operator workload and incident evidence completeness;
- independent replay/reconstruction success.

Do not collapse these into a single score. Report metrics by receiver, event family, date, and environment.

## 8. Proposed targets (not yet achieved)

- Establish baselines on at least two distinct receiver/data pipelines.
- Improve held-out detection performance against a simple statistical baseline at the same declared false-alarm rate.
- Demonstrate generalization to held-out receiver and event campaigns.
- Produce calibrated uncertainty and abstain rather than attribute causes unsupported by evidence.
- Require complete source/time/calibration/configuration/currentness references for every high-impact recommendation, else return `INDETERMINATE`.
- Achieve zero decisions that grant operational authority solely from anomaly labels, model confidence, Phi, coherence, or reputation.
- Reproduce all reported results from dataset/version hashes, exact split manifests, code/model digests, and per-case outputs.

Numeric false-alarm, detection-latency, continuity, and service-loss thresholds must be set by domain owners and independent safety engineers. Aviation, maritime, telecom, and industrial operations do not share one acceptable threshold.

## 9. Runner-independent P0 tasks

1. Freeze dataset inventory and lawful-use review.
2. Freeze observation/state schema and label-quality vocabulary.
3. Create an exact split manifest grouped by campaign, receiver, site, and time.
4. Publish baseline commands and expected outputs.
5. Build deterministic, receive-side replay from recorded observations.
6. Create benign-confound scenarios and sensor-health fault cases.
7. Bind evidence to Mycelix source/calibration/time provenance.
8. Define Symtropy safe-degraded-mode transitions and re-entry requirements.
9. Define Sol-Atlas service dependency and currentness requirements.
10. Require offline verifier reconstruction before any end-to-end resilience claim.

## 10. Qualification and claims

A passing result is scoped to the exact datasets, receiver classes, models, split, configuration, time basis, software versions, and tested operational context.

No claim of:
- best-in-class electronic warfare;
- military operational superiority;
- spectrum-regulatory compliance;
- airworthiness or maritime certification;
- real-world navigation safety;
- hostile actor attribution;
- offensive EW capability

is supported by this programme.

The target is defensive, evidence-qualified resilience and safe service continuity.

## 11. Primary references

- EASA, [SIB 2022-02 Revision 4: GNSS Outage and Alterations](https://www.easa.europa.eu/en/newsroom-and-events/news/easa-updates-safety-information-bulletin-gnss-interference), 3 July 2026.
- EUROCONTROL, [GUID-203: Operational Response to GNSS Degradations](https://www.eurocontrol.int/publication/eurocontrol-guidelines-guid-203-operational-response-gnss-degradations), edition 1.0, 5 November 2026.
- UK Maritime and Coastguard Agency, [MGN 719: Operating in GNSS Denied and Degraded Environments](https://www.gov.uk/government/publications/mgn-719-mf-operating-in-gnss-denied-and-degraded-environments), October 2026.
- Finnish Geospatial Research Institute, [Jammertest 2025 GNSS interference dataset](https://etsin.fairdata.fi/dataset/095782a8-4e31-456e-8719-7021b4c96362), released 30 September 2026.
- FGI, [GNSS Jamming and Spoofing Dataset Repository](https://www.maanmittauslaitos.fi/en/research/research/gnss-specialists/fgi-gnss-jamming-and-spoofing-dataset-repository-fgi-jsdr).
- Renaudin et al., [Jammertest 2024 dataset](https://zenodo.org/records/15911589), DOI 10.5281/zenodo.15911589, v3.
- [Chi-square GNSS signal anomaly detection evaluation](https://link.springer.com/article/10.1007/s10291-026-02102-z), *GPS Solutions*, 29 June 2026.
- [Multicriteria Collaborative GNSS Integrity Monitoring](https://ieeexplore.ieee.org/abstract/document/11505703), IEEE Journal of Indoor and Seamless Positioning and Navigation, 4 May 2026.
- U.S. DoD ASSIST, [MIL-STD-464](https://quicksearch.dla.mil/qsdocdetails.aspx?ident_number=35794), active listing, document date 9 February 2026.
- CISA, [Understanding Vulnerabilities of Positioning, Navigation & Timing](https://www.cisa.gov/sites/default/files/2023-04/fs_positioning-navigation-timing-vulnerabilities_508.pdf).
