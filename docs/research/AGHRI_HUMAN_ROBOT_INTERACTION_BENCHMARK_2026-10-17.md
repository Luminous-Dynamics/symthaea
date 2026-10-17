# AGHRI Human–Robot Interaction Benchmark Adapter
## Research and reproducibility contract — 2026-10-17

**Status:** metadata-only research adapter proposal. The dataset has not been downloaded or evaluated by this repository. No safety qualification or benchmark result is claimed.

## 1. Why this is the next high-value step

Agricultural robotics already has a useful public HRI resource and existing baseline ecosystem. Rebuilding a parallel detector/tracker stack would create more code without creating stronger evidence.

The University of Lincoln's AGHRI release is described by Agri-OpenCore as a multimodal dataset for human perception around agricultural/off-road robots:
- 65 sequences, about 1,642.515 seconds (27.4 minutes), about 70 GB;
- 10 participants;
- sequence-level train/validation/test partition of 52 / 7 / 6;
- synchronized ZED RGB-D, three fisheye camera views, and Livox Mid-360 LiDAR;
- 138,741 camera frames, 93,116 annotated camera frames, 154,002 2D human boxes, 13,963 LiDAR frames, 13,799 annotated LiDAR frames, 27,311 3D boxes, 454 camera-specific 2D tracks and 136 3D tracks;
- strawberry polytunnel, vineyard, footpath and farmside settings at the University of Lincoln Riseholme Campus, recorded October–November 2024.

Sources:
- Dataset metadata and licensing: https://agri-opencore.org/software-stacks/datasets/
- Dataset DOI: https://doi.org/10.24385/lincoln.32982638
- Official benchmark code: https://github.com/LCAS/AGHRI-dataset-benchmark
- Official dataset preparation/conversion tools: https://github.com/LCAS/AGHRI-dataset-tools
- Agri-OpenCore released software and licenses: https://agri-opencore.org/software-stacks/

Source snapshot used for planning:
- `LCAS/AGHRI-dataset-benchmark` main commit `565a289e0c9a742916da41bfc73c75126759d05b` (Apache-2.0).
- `LCAS/AGHRI-dataset-tools` main commit `ab70ecc3123e2089c662df5071c1380d8f9f8ecd` (MIT).
- `LCAS/sentor` master commit `93487502fb9b2645c75f79b9b9613c7e2650fd06` (Apache-2.0).
- `LCAS/autonomy_metrics_logger` main commit `1a0e56149b9712a40ec311cbc71ae09f7d3fbad8` (Apache-2.0).

These commit SHAs identify the source snapshot reviewed for this plan; verify the upstream default-branch heads and license files again before implementation or distribution.

## 2. Paradigm: perception ≠ temporal continuity ≠ runtime safety ≠ safe physical effect

Keep four evidence claims distinct:

1. **Perception:** was a person detected/localized in the recorded sensor frame?
2. **Temporal continuity:** did the system preserve the correct track and represent occlusion or lost observations accurately?
3. **Runtime observability:** did the monitor detect stale/missing sensor topics, degraded inference, unexpected latency, or a human-proximity trigger?
4. **Physical risk control:** did one specifically identified and qualified machine/tool enter its required safe state within a validated operating domain?

Recorded AGHRI can support benchmarking of the first two and can be used to test replay/monitoring logic for the third. It cannot alone qualify physical stopping distance, brake response, tool energy, emergency-stop behavior or a machine safety function. A person detection is not a stop guarantee; no collision in a short replay is not evidence of safety.

ISO 18497-1:2024 and ISO 18497-4:2024 are design/verification inputs for the later machine-specific safety case:
- https://www.iso.org/standard/82684.html
- https://www.iso.org/standard/82688.html

Citing a standard does not establish compliance, certification or legal authorization.

## 3. Reuse the existing upstream baseline

Do not implement a second perception leaderboard. Begin by adapting report metadata from:
- upstream 2D detector and 2D MOT/tracking workspaces;
- upstream 3D LiDAR detection and 3D tracking workspaces;
- existing camera-LiDAR early/late fusion evaluation;
- upstream tools for dataset preparation, coordinate/frame conversion and ROS 2 bag generation.

Upstream docs describe these workspaces as distinct task-level components with their own environments, metrics and output formats. The adapter should preserve their native result records and add a source-bound summary; it must not treat their metrics as directly interchangeable.

Dataset archive files are not in the upstream Git repository. Do not add the approximately 70 GB dataset to Symthaea, do not make it a default CI download, and do not invent sequence IDs before reading the authorized release's actual `dataset_summary.csv` and split metadata.

## 4. Licence and data handling

The Agri-OpenCore dataset page describes AGHRI as CC BY 4.0. The upstream benchmark code is Apache-2.0; the preparation tools are MIT-licensed. Keep these license scopes distinct.

The adapter must:
- include dataset attribution, DOI, license identifier and access date in reports;
- retain source archive/file SHA-256 hashes after an authorized download;
- store only metadata/manifests and permitted derived summaries in this repo, not raw sensor frames or point clouds;
- maintain the upstream sequence-level 52/7/6 split; no random frame-level split across a sequence;
- keep the official test split out of training, detector/model selection, threshold selection and calibration;
- use sequence IDs, not guessed filesystem names, as the stable split identity;
- minimize personal data: the first task is anonymous person detection/tracking, not cross-session re-identification. Any re-identification extension requires a separately justified purpose and privacy/retention review;
- record every excluded sequence and reason so selective reporting is visible.

A source-repository license does not override the dataset license. A metadata commit does not prove that the dataset archive was downloaded, uncorrupted or evaluated.

## 5. Run-manifest contract

The companion files in this directory provide:
- `docs/research/aghri-run-manifest.v1.schema.json`: strict JSON Schema for provenance and run metadata;
- `docs/research/aghri-run-manifest.template.json`: metadata-only, unqualified template.

Every real run must bind:
- dataset DOI/version/license/attribution and acquisition status;
- upstream benchmark/tool commit SHAs and licenses;
- environment/container digest, command/config/checkpoint digests and evaluator identity;
- actual selected sequence IDs and their official split membership;
- sensor modalities, timestamp/synchronization expectations, calibration refs and source file SHA-256 values;
- task-specific metrics, evaluator versions, exclusions, denominators and generated report digests;
- declared evidence stage: offline replay, digital twin, supervised field, multi-site or multi-season. Unknown stages must not be upgraded from replay evidence.

For an unacquired template, source file lists and selected sequences remain empty; there is no fake file digest or fabricated result.

## 6. Evaluation matrix

### Perception and localization
- source-declared precision/recall/AP and 3D localization metrics;
- breakdown by sensor, environment, distance/size, activity and occlusion only where source annotations support the slice;
- confidence calibration and abstention under missing, stale, desynchronized or out-of-domain input;
- matched-sequence RGB-only, fisheye, LiDAR-only and fusion ablations where upstream outputs support them.

### Temporal continuity
- HOTA, IDF1, identity switches, fragmentation, track continuity and time-to-reacquire where the upstream evaluator supports them;
- missed-person time and false-track persistence, with exact ground-truth and observable-frame denominators;
- partial, total and edge-of-frame occlusions matched to the actual annotation semantics.

### Runtime observability
- warning delay after stale/missing stream or heartbeat loss;
- missed fault injections, nuisance alarms, recovery time and log completeness;
- dropped frames, synchronization/calibration mismatch, latency spikes and resource saturation under defined replay tests.

### Physical system safety — separate qualification gate
- time from detection/monitor to stop request;
- stop command/acknowledgement separately from measured physical effect;
- physical stopping distance under a named robot/tool, speed/load/surface/slope and operating domain;
- nuisance stops, missed stops, near misses, interventions, collisions and property/crop/soil damage, all with exposure denominators;
- operator supervision and emergency recovery conditions.

AGHRI does not provide every physical-machine variable required by this gate. Missing physical evidence must remain an explicit unmeasured blocker, not a favorable imputation.

## 7. Reproducibility and acceptance gates

1. **Metadata-only gate:** validate the JSON schema, license/source fields, duplicate sequence IDs and official split membership without downloading sensor data.
2. **Integrity gate:** once data are acquired, verify archive/source hashes; preserve raw source data read-only; record per-file/sequence hashes and conversion outputs.
3. **No-leakage gate:** prove sequence-level split preservation, with zero train/validation/test sequence overlap.
4. **Baseline reproduction gate:** on an explicitly selected subset of the official split, rerun an upstream published baseline and explain any difference within a predeclared tolerance.
5. **Adversarial replay gate:** exercise prolonged occlusion, person entry/exit, missing modality, desynchronization, stale heartbeat, latency spike, false positive, false negative and changed model/config identity.
6. **External-validity gate:** report the sample size and environment limits; later data from distinct farms, seasons and regions are needed before transfer claims.
7. **Machine-safety gate:** separate qualified physical tests and independent review before claims about stop response or safe operation.

Every gate has its own evidence class. Synthetic, offline replay, simulation, controlled field and multi-site/multi-season results remain distinct.

## 8. Immediate deliverable and next tranche

Immediate: land the schema, the metadata-only run-manifest template, this plan, and a CI-sized schema/provenance validation. Do not make the 70 GB dataset or GPU frameworks required for ordinary Rust unit tests.

Next:
1. retrieve the authorized dataset summary and official split metadata; choose a minimal sequence subset by documented coverage criteria;
2. reproduce one upstream detection result and one tracking result with fixed config/checkpoint and exact outputs;
3. build a report adapter for source metrics plus exposure-normalized runtime events;
4. add deterministic stale-topic/latency/occlusion replay controls;
5. only then design a machine-specific physical safety test plan with qualified reviewers.


## 10. Operational economics are part of the benchmark

A 2026 open-access study modeled autonomous field-robot costs using empirical data from 61 maize fields in northwestern Germany. It reported average modeled total cost of €73/ha, fieldwork costs of €60/ha, and greater dispersion in logistics cost than fieldwork cost. Field size explained 58% of total-cost variance in that dataset, route distance 11%, and access points 1%. These results are context-specific, but they demonstrate why price-per-robot or detector accuracy cannot stand in for farm-level adoption economics.

Reference: *Structural cost modeling and sensitivity analysis of autonomous field robot operations*, Smart Agricultural Technology 13 (March 2026), https://doi.org/10.1016/j.atech.2026.101913 (open-access article page: https://www.sciencedirect.com/science/article/pii/S2772375526001371).

For later platform comparisons, report cost per completed task / treated hectare / marketable kilogram including setup, transport between plots, charging, supervision, missed missions, recovery, parts, maintenance, seasonal utilization and repair delays. Use local field geometry and route access rather than transferring the German maize cost numbers to a Southern African farm. Model low/base/high utilization and disruptions; publish the measured-data boundary and uncertainty.

## 11. Explicit nonclaims

This adapter does not establish that any robot is safe, that AGHRI covers all agricultural settings, or that UK benchmark performance transfers to Southern Africa. It does not establish crop yield, food safety, return on investment, ISO compliance or stop-function qualification, and it grants no physical actuation authority.
