# Real temporal dataset provenance manifest v0.1

Tracking: #9811  
Implementation branch: `research/edge-realdata-manifest-v0.1`  
Parent bridge: [PR #9808](https://github.com/Luminous-Dynamics/symthaea/pull/9808)

## Purpose

T1–T10 are deterministic synthetic tasks with known oracles. They are essential for exact causal and verifier controls, but they are not substitutes for naturally recorded data. This v0.1 manifest is the first data-ingestion boundary for real temporal workloads.

The manifest is metadata-only. It binds:
- source dataset ID/version/DOI/URL and exact archive SHA-256;
- license class, license URI, intended use and relevant share-alike/noncommercial/no-derivatives acknowledgement;
- timestamp representation and optional nominal rate;
- timestamp/grouping column semantics, sorted channel names and label map;
- source schema and preprocessing digests;
- sorted per-file relative path, byte length, SHA-256 and source-family ID;
- source-family-to-split assignments;
- canonical manifest digest.

No data binaries, participant-level rows or raw recordings belong in the public source repository or normal CI artifacts. Store the archive in a license-controlled data location and publish only the metadata manifest/file digests where permitted.

## License-aware dataset candidates

### HARTH — first primary inertial dataset

- UCI record: https://archive.ics.uci.edu/dataset/779/harth
- DOI: 10.24432/C5NC90
- License: CC BY 4.0.
- 22 subjects; around 90–120 minutes of free-living data per subject; two three-axis accelerometers at thigh and lower back; 50 Hz; professionally annotated activities; separate CSV recordings per subject.
- Initial task: causal activity recognition and transition detection, with subject-held-out evaluation.
- Sensor-subset experiments: thigh only, back only, both. Measure accuracy/false alarms/detection latency against sensor samples/hour and complete-event energy.
- Source family: participant/subject recording, derived from the original per-subject recording inventory before window creation.

The UCI record describes 6,461,328 instances and timestamp, accelerometer and activity-label fields. CC BY 4.0 permits broad reuse with attribution, subject to the exact terms.

### SHD and SSC — event-based audio and temporal SNN comparisons

- Maintainer page: https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
- License: CC BY 4.0.
- SHD: 20 digit classes; the maintained page lists 8,156 train and 2,264 test samples.
- SSC: 35 word classes; page lists 75,466 train, 9,981 validation and 20,382 test samples.
- Spike inputs are generated in 700 channels by an artificial cochlea conversion pipeline. Pin the exact conversion version as part of preprocessing.
- These datasets are useful for timing/event representation, temporal credit assignment and SNN comparisons. They are not equivalent to continuous wake-word listening with hours of negative exposure.

The maintained SHD leaderboard currently lists Sun et al. 2025 parameter-free attention for delay SNNs at 96.26 ± 0.08%, with multiple other recent models around 95–96%. SSC results include recent methods around 82–83%. These are published/site-reported results, not Symthaea reproductions. Pin exact paper/code/task splits and reproduce baselines before declaring a comparison.

### MIMII — industrial acoustic anomaly detection

- Original dataset: https://zenodo.org/records/3384388
- Baseline/readme: https://github.com/MIMII-hitachi/mimii-baseline/
- License: CC BY-SA 4.0.
- Public 1.0 describes valve, pump, fan and slide-rail sounds across machine models; normal and anomalous recordings, factory background noise, eight-channel 16 kHz audio.
- Source families must be grouped by machine model/recording/operating condition before any windowing. Avoid random clips from the same machine appearing in multiple partitions.
- Treat normal-only fitting, threshold calibration and one-shot anomaly confirmatory evaluation as distinct stages.
- CC BY-SA permits commercial use subject to attribution and share-alike requirements. If adapted dataset material is redistributed, downstream license obligations must be reviewed and acknowledged. Do not combine derived dataset redistributions with incompatible terms.

### Restricted datasets not selected for the commercial lane

The 2026 comprehensive full-body IMU dataset from 30 people/17 IMUs and the AICOS-HAR multi-device dataset (106 people, 87 smartphones, 17 wearables) are reported under CC BY-NC-ND 4.0. They can be considered for noncommercial research only after direct license review; they are not admitted to the commercial-product dataset lane and modified derivatives must not be distributed under that license.

## Manifest semantics and fail-closed rules

- Dataset file paths must be relative and cross-platform safe. Absolute paths, empty path segments, dot traversal, Windows drive/UNC/backslash paths and stream/alternate-data syntax are rejected regardless of the machine doing validation.
- Each file has a nonzero byte length, digest and source-family identity.
- Every source family is assigned exactly one of Training, Development, Calibration or Confirmatory.
- Every split must contain at least one source family. Subject/device/machine/speaker/recording grouping is fixed before windowing.
- No source-family can appear in more than one split. Repeated windows and multiple files from one source family remain in the same partition.
- Channel names and label IDs are sorted and unique; timestamp unit and sample-rate metadata are explicit.
- Dataset identity has an expected public license class (HARTH/SHD/SSC CC BY; MIMII CC BY-SA). A mismatch fails closed.
- CC BY-NC-ND data is restricted to noncommercial research and cannot be used to redistribute adapted dataset material. Custom licenses require a pinned review-record digest.
- Hashes are commitments, not proof of acquisition or licensing. The archive, license terms, source inventory and conversion script must be acquired/reviewed separately.

## Evaluation rules

### Splitting

The subject/recording group is the independent split unit. For HARTH, split participants, not windows. For SHD/SSC, group by available speaker/source recording identity. For MIMII, group by machine model/operating session. All windowing and feature extraction runs only after split assignments are frozen.

### Baselines

For HARTH:
- threshold/EMA and hand-engineered features;
- INT8 static CNN/TCN;
- GRU/LSTM and adaptive TinyML;
- HDC-only, CfC-only, HDC-LTC;
- active-sensing policies with the same allowed sensor subset and event-quality target.

For SHD/SSC:
- GRU/TCN/LSTM;
- adaptive LIF/ALIF SNN;
- BrainTrace D-RTRL and pp-prop/e-prop where task-compatible;
- delay/attention SNN references from the maintained leaderboard;
- HDC-only, CfC-only, HDC-LTC.

For MIMII:
- one-class threshold/EMA, statistical anomaly detectors and autoencoder/TCN references;
- GRU-D/continuous-time models when missingness or irregular sampling is present;
- HDC associative anomaly memory and HDC-LTC;
- online drift calibration and machine-model transfer where the split allows it.

Every implementation needs exact source revision/release, dependencies, config, preprocessing, parameter/precision, training/update budget and random seed. Paper score is a literature reference, not a reproduced baseline.

### Report

Publish each task separately:
- quality and worst-group quality;
- false alarms per hour, missed events and detection delay;
- drift/change recovery and prior-regime retention where applicable;
- sensor samples/hour, communication bytes and online-update energy;
- persistent state/peak SRAM/flash, memory traffic and raw latency distribution;
- measured complete-event energy per successful correct on-time event plus expanded uncertainty.

Do not average SHD, HARTH and MIMII into a single score. Their tasks, data types and class distributions differ.

## Current implementation

`research/temporal-bench/src/real_dataset_manifest.rs` defines:
- four known real dataset IDs (HARTH, SHD, SSC, MIMII) plus a custom-license option;
- license policy/usage scope;
- timestamp representation/sample rate;
- source file inventory and source-family split assignment;
- schema/preprocessing/archive digests;
- canonical content-bound receipt.

Six synthetic tests are authored for a valid HARTH manifest, MIMII share-alike acknowledgement, CC BY-NC-ND commercial-use refusal, source-family overlap, file inventory mismatch, path traversal and sample-rate metadata. Tests are not compiled or run. No archive has been downloaded or parsed.

## Claim ceiling

- A manifest proves only what source bytes/metadata were presented to it and bound; it does not authenticate a publisher or validate a license grant.
- CC license classification does not replace legal review or source-specific terms.
- No real dataset benchmark score, model performance, hardware energy or SOTA result has been generated.
- Real-data performance must use subject/device/machine/source-family splits. Window-random splits are invalid for a generalization claim.

## References

- HARTH UCI dataset: https://archive.ics.uci.edu/dataset/779/harth
- HARTH DOI: https://doi.org/10.24432/C5NC90
- SHD/SSC dataset and license: https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
- SHD/SSC benchmark references: https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
- MIMII dataset: https://zenodo.org/records/3384388
- MIMII license/source description: https://github.com/datasets-mila/datasets--mimii/blob/master/README.rst
- 2026 comprehensive IMU dataset: https://doi.org/10.1038/s41597-026-06710-9 (CC BY-NC-ND 4.0)
- AICOS-HAR: https://doi.org/10.5281/zenodo.19452049 (CC BY-NC-ND 4.0)
