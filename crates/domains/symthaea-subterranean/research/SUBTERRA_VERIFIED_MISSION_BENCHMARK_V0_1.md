# Subterra Verified Mission Benchmark v0.1

**Status:** research benchmark protocol  
**Date:** 2026-11-17  
**Tracking:** [#12621 — VERIFIED-MISSION-BENCH-001](https://github.com/Luminous-Dynamics/symthaea/issues/12621)  
**Composes:** [#12606 construction](https://github.com/Luminous-Dynamics/symthaea/issues/12606), [#12609 asset observability](https://github.com/Luminous-Dynamics/symthaea/issues/12609), [#12612 robotic loops](https://github.com/Luminous-Dynamics/symthaea/issues/12612)

## 1. Paradigm shift: score accepted physical work, not nominal autonomy

The benchmark's unit of success is an **independently accepted physical state transition**.

A planned operation, command receipt, actuator response, plausible point cloud, detector output, or robot-reported "complete" status is not a successful outcome. The work counts only when the declared postconditions are independently verified against a still-valid work package, and all authority, resource and safety gates remain satisfied.

This changes the optimization problem:

`maximize accepted work under hard safety, authority, integrity and uncertainty constraints`

rather than:

`maximize task completions or machine utilization`

A high-productivity candidate that creates false closeouts, unauthorized mutations, unobserved "clear" regions or unresolved duplicate work is ineligible to win.

## 2. Novel research contributions to test

### 2.1 Task-to-proof compilation

Compile engineering work packages into explicit:
- preconditions and required source revisions;
- allowed physical state deltas and no-contact regions;
- resource envelopes and stop conditions;
- postconditions and spatial coverage obligations;
- evidence types, calibration requirements and verifier independence constraints;
- reconciliation paths for unknown execution outcomes.

This is not an autonomous engineer and must not invent structural design criteria. The design and acceptance criteria remain approved inputs from responsible engineering authorities. The research contribution is compiling approved requirements into machine-checkable execution and evidence obligations.

### 2.2 Physical operations as distributed transactions

Treat each intervention as a **saga-like physical transaction**, not an ACID database transaction. Physical actions may be irreversible, partially completed or unobservable after a communications failure.

The lifecycle is:

`prepare → authorize → command → observe actuation → observe physical delta → independently verify → close or reconcile`

Timeout is not rollback. Retrying is not safe by default. A globally unique idempotency key, attempt identity, work-scope lock and independent reconciliation are necessary to prevent duplicate physical mutation.

### 2.3 Decision-conditioned observability

The next observation should be selected by its expected reduction in consequential decision error, not by point-cloud size or waypoint completion.

Compare uniform coverage, nearest-next-view, highest-risk-first, entropy reduction and decision-conditioned value of information. Match inspection time, energy, exposure and sensor availability. Report the Pareto frontier; do not assume one scalar information score captures all risks.

### 2.4 Evidence diversity, not sensor count

Two sensors are not independent merely because they are different devices. Model common causes across:
- power and communication buses;
- clocks and timestamp sources;
- sensor mounts and coordinate transforms;
- calibration targets and calibration routines;
- firmware, perception code and model weights;
- shared environmental blind spots;
- data preprocessing and evidence storage.

An independence assessment is itself evidence with provenance and an explicit scope. A missing or incomplete dependency map yields insufficient evidence, not independence by assumption.

### 2.5 Unknown-state debt

Track unverified volume/area, age, consequence class, source of uncertainty and the cost of resolving it. Unknown regions should not silently age into "clear" regions. Include the debt in inspection scheduling and maintenance planning.

### 2.6 Assurance-adjusted fleet productivity

Report accepted work per machine-hour and per total mission-hour, alongside verification effort, energy, human intervention, rework and evidence latency. This exposes the difference between a fast robot that exports verification cost to humans and a system that actually produces accepted infrastructure work.

## 3. Matched baseline ladder

Run every candidate on the same work packages, geometry/material models, sensor streams, resource budgets, hidden-truth splits and fault schedules.

A. scripted/open-loop task sequence;  
B. local reactive feedback;  
C. conventional perception plus task planner;  
D. risk-aware model-predictive controller;  
E. robust shared contingent policy evaluated over model disagreement;  
F. verification-unaware active inspection;  
G. verification-aware active inspection;  
H. composable-loop system with bounded intervention, independent verification and evidence/recovery contracts.

Include human inspection/planning under a matched information and time budget where practical. Include vendor/commercial baselines only when their operating domain, machine class, data access and acceptance criteria can be matched. Do not convert vendor press-release productivity figures into benchmark scores.

## 4. Benchmark families and metrics

### B1 — asset observability

Pipe/cable corridors, support systems, drawpoint hang-up volume and occluded underground geometry.

Report per defect class:
- recall at fixed false-alarm rates;
- precision, severity-weighted miss cost and detection latency;
- 3-D location/extent error and volumetric error;
- verified surface/volume coverage and unknown fraction;
- Brier score, ECE, risk-coverage and OOD abstention;
- inspection time, joules, data volume and human review time.

### B2 — construction/intervention

Excavation increment, support installation, lining, sealing and repair.

Report:
- as-built geometry and tolerance-violation volume;
- installation completeness, embedment/alignment and defect status;
- false acceptance, first-pass acceptance and rework;
- no-contact/utility incursions;
- planned/dispensed/installed/recovered/unexplained materials;
- energy, time, tooling, wear and consumables.

### B3 — independent verification

Report:
- false-accept and false-closeout rate;
- required-volume coverage and unobserved required regions;
- verifier agreement with independent metrology/human adjudication;
- common-cause failure sensitivity;
- evidence completeness, lineage and post-action latency;
- verifier cost and resource use.

### B4 — ambiguous execution and recovery

Report:
- correct handling of timeout-before-execution and timeout-after-execution;
- duplicate/reordered receipts and idempotency-key reuse;
- unresolved-scope lock correctness;
- reboot/partition evidence continuity;
- time to reconcile, safely resume or escalate;
- residual work and hazard after partial execution.

### B5 — fleet resilience

Compare single robot, nearest-capable, greedy assignment, charging-aware planner, capability-aware fleet and verification-aware fleet.

Report:
- accepted work per robot-hour and per total mission-hour;
- charging feasibility and return-reserve violations;
- communication outage tolerance and relay recovery;
- task regret, assignment churn and deadlock;
- duplicate-work prevention and evidence survival after handoff.

### B6 — integrated asset lifecycle

Compose inspect → classify condition → engineering disposition → authorize work → intervene → independently verify → update asset record → schedule recurrence monitoring.

Report time and total cost to accepted closeout; inspection and verifier effort; rework; unresolved unknown-state debt; and recurrence detection. Retain all B1–B5 metrics separately.

## 5. Frozen negative-control catalogue

The companion JSON file contains scenario IDs and expected dispositions. Core fixtures include:
- stale work package, wrong design, unit/frame mismatch and expired calibration;
- commanded-but-not-installed and partial/misaligned support;
- occluded inspection and false free-space / false-clear claims;
- shared timebase, power, mount, calibration or perception stack;
- timeout with execution unknown, executed-but-uninspected and not-executed reconciliation;
- duplicate command keys across retry and robot handoff;
- reboot, partition, delayed evidence and reordered receipts;
- robot immobilization, charging shortfall and return-reserve breach;
- OOD geology, unexpected water ingress, tool wear and drift;
- pipe/cable defects and drawpoint blockage with insufficient views;
- rework attempt with mismatched or stale replacement authority.

A negative control is passed only when the system produces the declared fail-closed or insufficient-evidence disposition and no unauthorized state mutation occurs.

## 6. Statistical protocol

- Freeze work-package revisions, input datasets, simulator versions, calibration sets, model families, seeds, fault schedules, baselines and evaluator code before comparison.
- Split by site, asset, geological/material family, sensor campaign and time period. Never randomly split adjacent frames from one inspection into train and test.
- Use paired scenarios and common seeds across candidate baselines; publish all raw per-scenario outcomes and exclusions.
- Report denominators and confidence intervals for defect recall, false acceptance, coverage and recovery.
- For zero failures in independent Bernoulli trials, show the exact confidence bound. The familiar rule-of-three approximation gives a 95% upper bound of about 3/n: around 300 independent failure-free trials corresponds to ~1%, and around 3,000 corresponds to ~0.1%. These are statistical statements about the tested trial distribution, not field safety arguments. Correlated synthetic seeds or shared fault templates do not count as independent trials.
- Keep safety, authority, no-contact and evidence-integrity gates hard. Do not average them into a productivity score.

## 7. Proposed targets — hypotheses, not achieved results

- ≥15% less rework or inspection effort than matched scripted/open-loop construction, at non-inferior physical quality.
- ≥20% less inspection time or energy than matched inspection baselines, at non-inferior defect recall and false-alarm rate.
- Zero false closeouts in the frozen negative-control suite.
- 100% rejection of frozen stale-work-package, design/frame/unit mismatch, expired calibration, duplicate mutation and unauthorized-operation fixtures.
- No new mutation while the previous physical outcome is unresolved.
- Higher accepted work per total mission-hour without increasing false acceptance, unknown-state debt or unaccounted verifier/human effort.
- Gains reproduced on at least two distinct geology/material families and one held-out fault distribution.

All productivity targets are secondary to passing identical hard gates. No finite synthetic suite establishes zero real-world risk.

## 8. Industry and SOTA comparison

The decision-conditioned inspection adapter in [#12628](https://github.com/Luminous-Dynamics/symthaea/issues/12628) reuses the existing finite-horizon Subterra measurement planner and independent robust-policy referee; it does not create a competing EVSI engine.

| Reference | Current demonstrated focus | Correct Subterra comparison |
|---|---|---|
| DARPA SubT / CoSTAR | Extreme-environment exploration, mapping, artifact search and multi-robot autonomy | Preserve exploration baselines; add construction, asset condition and accepted closeout |
| Sandvik AutoMine Aura | Underground equipment automation, 3-D perception and vendor-reported productivity | Do not compete on "AI autonomy" alone; measure independently accepted work and cross-vendor evidence |
| Epiroc Deep Automation | Underground-to-surface automated truck cycles using 3-D LiDAR | Matched haulage/localization baseline; add verified intervention and infrastructure-state records |
| Komatsu / AIM | Commercial autonomous dozers/excavators connected to Smart Construction workflows; adjacent construction domain | Compare work-package interoperability and measured as-built acceptance, not mine-specific capability |
| 2026 charging-aware mine inspection | Task allocation, path planning and charging-feasible multi-robot inspection in simulation | Reproduce its declared task objective first; extend with verification-aware information gain, model shift and recovery |
| UC San Diego Pipe Network Inspection Robot | Self-centering 2–6 inch in-pipe platforms; 6-inch steel tests cover straight runs, elbows and T-joints; autonomous traversal is in a new development phase | Component baseline for confined-pipe mobility, joint recognition and odometry; do not equate it with mine-wide asset-condition qualification |
| 2025 gecko-inspired underground cable robot | Quadruped locomotion concept and vision-based defect recognition for insulation ageing, micro-cracks and contamination | Component baseline for cable access and visual defect detection; independent calibration and site transfer remain separate questions |
| 2026 dual-resolution acoustic in-pipe robot | Acoustic modality for autonomous in-pipe inspection | Compare modality-specific detection and sensing cost; add cross-modal evidence and condition-decision verification |
| 2026 mine infrastructure review | Finds dedicated mine pipe/cable inspection and drawpoint hang-up volumetric assessment comparatively under-addressed | Target mine-specific asset-condition and drawpoint tasks, while explicitly benchmarking against adjacent in-pipe and cable-inspection prototypes |
| 2026 rock-bolt AI review | AI/optimization, monitoring and compliance checks; weak cross-site validation and weak links to as-built records | Make as-built evidence, cross-site robustness and human-governed engineering review central |

### Evidence ledger: what published and commercial systems actually demonstrate

- **UC San Diego pipe robot (project status, updated November 2026):** 2-inch and 4-inch prototypes have been built/tested; a 6-inch robot runs in a real steel test network. On-board functions include joint classification, orientation estimation and visual-inertial odometry. The project describes Phase 2, begun in October 2026, as the step toward autonomous network traversal. Therefore, treat it as a strong small-pipe mobility/joint-perception baseline, but do not describe it as already completing autonomous network traversal. [Project timeline](https://robotics.ucsd.edu/projects/pipe-robot/)
- **Dual-resolution acoustic in-pipe robot (Sun et al., 2026):** reports average localization variation of 8 cm over an 18 m pipe setup and 76% average feature classification accuracy during autonomous inspection. This is a modality-specific experimental baseline; reproduce its exact pipe, feature classes, sensing setup and scoring before comparing. It is not a mine-wide cable/pipe condition-acceptance rate. [University of Bristol record](https://research-information.bris.ac.uk/en/publications/a-dual-resolution-acoustic-sensing-robot-for-autonomous-in-pipe-i/)
- **Gecko-inspired cable inspection robot (Guan & Shakibaei Asli, 2025):** combines quadruped locomotion with binocular vision and an enhanced YOLOv8 detector aimed at insulation ageing, micro-cracks and surface contamination. It is a direct cable-access/vision research comparator; its detection dataset and experimental operating conditions must be matched before transferring its reported performance to mine infrastructure. [Paper](https://doi.org/10.3390/electronics14153142)
- **Charging-aware multi-robot mine inspection (Xu et al., Scientific Reports, 6 August 2026):** reports mean objective cost 6,800.3, about 7.20–11.16% below its selected heuristic comparators, with 6.50 standard deviation and 120 s mean runtime. Its own limitations section states the work uses a 2-D simulated tunnel, has no physical-robot validation, simplifies energy/charging, and uses a decoupled two-stage method that may lose optimality. Reproduce its objective and seed/budget protocol before using it as a fleet-planning comparison; it does not measure defect calibration or independently accepted maintenance. [Paper](https://www.nature.com/articles/s41598-026-65759-z)
- **Sandvik AutoMine Aura (vendor-reported, 26 May 2026):** Sandvik states that 3-D perception/navigation moved over 15% more material at one customer site and says AutoMine has been proven across more than 140 mines. Treat this as an important deployed-industry productivity benchmark, but retain the vendor's own scope and do not compare the material-movement percentage directly with inspection accuracy or accepted repair work. [Sandvik announcement](https://www.mining.sandvik/en/news-and-media/news-archive/2026/05/sandvik-launches-automine-aura-a-first-of-its-kind-automation-platform-for-the-future-of-mining/)
- **Epiroc Deep Automation (vendor demonstration, September 2026):** reports a June 2026 live demonstration of battery-electric truck automation spanning underground and surface operations using 3-D LiDAR for localization, mapping and obstacle detection, including stopping for placed obstacles. This is a strong automated-haulage/localization comparator, not evidence of mine-infrastructure defect inspection. [Epiroc announcement](https://www.epiroc.com/en-pl/newsroom/2026/epiroc-deep-automation-introduces-seamless-underground-to-surface-truck-automation-with-3d-lidar-technology)
- **Mine-infrastructure research gap (Naheem et al., accepted 11 November 2026):** the review concludes that relevant robotic capabilities are demonstrated for re-entry and ground-condition assessment, while dedicated mine pipe/cable inspection and robotic drawpoint hang-up volumetric assessment remain largely unaddressed. This supports prioritizing these task classes, but it is a literature review, not proof that no adjacent research or commercial pipe robot exists. [Review](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2026.2017682/abstract)

The competitive claim should therefore be narrow and testable: Subterra does not lead on generic underground autonomy or small-pipe mobility by assertion. It aims to join task-specific asset observation, decision-valued additional inspection, independently verified physical closeout, and fail-closed recovery into one interoperable evidence loop. Every claimed advantage must be measured against matched baselines using the same hidden truth, sensor/action budget, resource accounting and acceptance rules.

### Public inspection datasets: admitted uses and non-claims

The first external vision/data baselines are tracked separately in [#12652 — SUBTERRA-REAL-DATA-ADMISSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/12652).

| Dataset | Published scope/size | Appropriate benchmark use | Explicit limitation |
|---|---|---|---|
| [Sewer-ML](https://vap.aau.dk/sewer-ml/) | 1.3M images from 75,618 videos, 3 Danish utilities, 9 years; licensed-inspector annotations; 17 official defect classes | Reproduce image-level sewer multi-label classification and source-specific metric | Not a cable-duct, drawpoint or mine-support dataset; check terms before any training or redistribution |
| [CableInspect-AD](https://mila-iqia.github.io/cableinspect-ad/) | 4,798 expert-annotated power-line cable images, 6,023 anomalies, 7 defect types/severity labels, image/pixel/bbox labels | Reproduce anomaly detection, pixel segmentation, thresholding and severity evaluation | CC BY-NC-SA 4.0 (non-commercial); power-line cable imagery is not underground duct condition data |
| [IPD](https://zenodo-rdm.web.cern.ch/records/17398156) | 357 insulated industrial-pipeline images with binary masks | Industrial pipe-insulation anomaly segmentation baseline | Signed licence required; binary masks do not separate hole/open insulation/dent classes |
| VideoPipe QV-Pipe / CCTV-Pipe | Published review describes 9,601 short clips (>55 h) and 575 long videos (87 h) | Candidate video classification and temporal defect localization | Access/licence and original split provenance must be verified before admission |

Sewer-ML's published CVPR benchmark reported F2-CIW 55.11% and F1-Normal 90.94% for its selected method; reproduce the original split and metric before adding new methods. These are dataset-specific classification scores, not mission-level accepted-asset performance. [Official benchmark page](https://vap.aau.dk/sewer-ml/).

Do not randomly split neighbouring frames across training and test. Keep video/asset/site/utility/time groups intact, preserve natural class prevalence, publish per-class denominators, and mark source-domain transfer as transfer—not as an equivalent task. The current public set contains no admitted mine drawpoint hang-up volume ground truth. Until authorized mine-specific truth exists, drawpoint volumetric validation remains synthetic/NOT_RUN.

### Standards boundary

- ISO 17757:2019 is a relevant safety standard for autonomous/semi-autonomous earth-moving and mining machine systems; applicability must be determined for the equipment and operational domain.
- ISO 21423:2026 was published in October 2026. It specifies industrial mobile-robot communications/interoperability, and explicitly excludes safety and navigation requirements. Use as an interface comparator only.

This benchmark is not certification against either standard.

## 9. Sources and reproducibility

1. [DARPA Subterranean Challenge](https://www.darpa.mil/research/programs/darpa-subterranean-challenge)
2. [Sandvik AutoMine Aura, 26 May 2026](https://www.mining.sandvik/en/news-and-media/news-archive/2026/05/sandvik-launches-automine-aura-a-first-of-its-kind-automation-platform-for-the-future-of-mining/)
3. [Epiroc Deep Automation, 3 September 2026](https://www.epiroc.com/en-pl/newsroom/2026/epiroc-deep-automation-introduces-seamless-underground-to-surface-truck-automation-with-3d-lidar-technology)
4. [Komatsu / AIM partnership, 31 July 2026](https://www.komatsu.com/en-us/newsroom/2026/komatsu-aim-enter-strategic-partnership)
5. [Xu et al., charging-aware multi-robot mine inspection, Scientific Reports, 6 August 2026](https://www.nature.com/articles/s41598-026-65759-z)
6. [UC San Diego Pipe Network Inspection Robot](https://robotics.ucsd.edu/projects/pipe-robot/)
7. [Sun et al., dual-resolution acoustic-sensing robot for autonomous in-pipe inspection, February 2026](https://doi.org/10.1016/j.tust.2025.107133)
8. [Naheem et al., Underground Mine of the Future review, 2026](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2026.2017682/abstract)
9. [Ping & Shakimon, AI-driven generative design and optimization in rock bolt support systems, 16 September 2026](https://link.springer.com/article/10.1007/s44290-026-00597-6)
10. [ISO 21423:2026](https://www.iso.org/standard/86749.html)
11. [ISO 17757:2019](https://www.iso.org/standard/76126.html)

## Claim ceiling

A benchmark pass qualifies only the declared synthetic or instrumented test-cell conditions, systems, datasets, model families and acceptance criteria. It does not establish mine safety, geotechnical adequacy, structural integrity, electrical safety, equipment certification, regulatory compliance or field productivity without independent physical trials and specialist engineering authority.
