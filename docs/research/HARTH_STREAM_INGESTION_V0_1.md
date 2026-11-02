# HARTH streaming ingestion v0.1

Tracking: [#9846](https://github.com/Luminous-Dynamics/symthaea/issues/9846)  
Implementation branch: `research/harth-stream-parser-v0.1`  
Parent: [active-sensing simulator PR #9845](https://github.com/Luminous-Dynamics/symthaea/pull/9845)

## Source and expected schema

The primary source is the [UCI Machine Learning Repository HARTH record](https://archive.ics.uci.edu/dataset/779/harth), DOI [10.24432/C5NC90](https://doi.org/10.24432/C5NC90), CC BY 4.0.

UCI documents 22 participants with right-thigh and lower-back three-axis accelerometers, nominal 50 Hz sampling, participant recordings in separate CSV files, and 6,461,328 total instances. The eight canonical fields are:

`timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label`

The six acceleration values are expressed in g. The documented activity codes are:

| Code | Activity |
|---:|---|
| 1 | Walking |
| 2 | Running |
| 3 | Shuffling |
| 4 | Stairs ascending |
| 5 | Stairs descending |
| 6 | Standing |
| 7 | Sitting |
| 8 | Lying |
| 13 | Cycling (seated) |
| 14 | Cycling (standing) |
| 130 | Raw code only: UCI metadata says cycling (sitting, inactive); the original paper describes transport (sitting) |
| 140 | Raw code only: UCI metadata says cycling (standing, inactive); the original paper describes transport (standing) |

## Parser contract

`research/temporal-bench/src/harth_stream.rs` adds a standard-library-only `BufRead` streaming parser.

- Validates the canonical eight-column header in its exact semantic order.
- Admits one optional leading export-index column only for the recognized empty/`index`/`Unnamed: 0` header forms, and checks that the index starts at zero or one and increases sequentially. This is a compatibility allowance, not a claim that UCI's canonical schema contains an index.
- Handles UTF-8 BOM, CRLF, ordinary quoted fields and escaped quote characters. Multiline CSV fields are rejected.
- Parses timezone-naive `YYYY-MM-DD HH:MM:SS[.fraction]` or `T` timestamps with up to nanosecond fractional precision. It rejects timezone suffixes and does not infer UTC or Johannesburg local time.
- Requires strictly increasing timestamps within each source-family recording.
- Requires finite numeric values for all six acceleration channels and one of the twelve documented labels.
- Streams each `HarthRow` to a caller callback. The parser holds the current line and its fields, not a complete participant recording.
- Returns the row count, first/last timestamp, minimum/maximum sample interval, number of gaps, maximum gap, source-family digest and canonical normalized-row digest.

The nominal sample period is 20 ms. A gap is counted when an observed adjacent-row interval exceeds 30 ms (1.5× nominal), and is reported—not interpolated, filled, or silently resampled. Other downstream definitions of missingness can use the retained row stream and interval data.

## Versioned target-label profiles

The parser exposes numeric source codes, including neutral variants `Label130` and `Label140`. Classification experiments select an explicit, digest-bound profile from `harth_label_profile.rs`:

- `harth.native-12.v1`: all twelve raw source codes remain separate target classes. For contiguous classifier output, the profile maps the sorted raw-code list `[1,2,3,4,5,6,7,8,13,14,130,140]` to target IDs `1..=12`; the raw code is retained separately in each parsed row.
- `harth.paper-merged-9.v1`: reproduces the original paper's nine-class merge: raw labels 3/6/140 map to Standing; 7/130 map to Sitting; labels 1,2,4,5,8,13,14 remain their respective seven classes. Target IDs are contiguous `1..=9`.

The profile identifier and complete raw-to-target mapping are domain-separated and SHA-256 committed. Every training/evaluation manifest must pin both the raw dataset archive digest and the selected profile digest. Do not use the paper's merged-nine profile while reporting results for the native twelve-label task.

## Leakage and temporal protocol

The independent split unit is the participant/source-family recording. Freeze participant identities into Training, Development, Calibration and Confirmatory before:
- making fixed or sliding windows;
- computing normalization statistics;
- extracting features;
- augmenting recordings;
- selecting thresholds or calibrating abstention.

Never concatenate participant files and then randomly split windows. Repeated windows, samples and derived features from one participant stay in one partition.

The parser's normalized time key is only an ordering/interval representation. Since the source timestamp is timezone-naive, it is not an absolute UTC instant. Keep the source bytes and archive SHA-256 in the existing real-data manifest; the parser's digest binds parsed row values and the declared source-family identity, not the publisher or original file bytes. Activity code 130/140 semantics are deliberately left unresolved in the base parser because the current UCI metadata and original article use different descriptions.

## Initial comparison target and current reference point

HARTH is not a solved classification task. Its 12 activity labels are highly imbalanced, rare classes such as cycling-standing and stair transitions are difficult, and a random-window split would overstate generalization. Use subject-held-out evaluation.

### Strong HARTH-specific reference: SelfPAB

The strongest directly comparable HARTH-specific reference found in this audit is **SelfPAB 100k** (Logacjov et al., *Applied Intelligence*, 2024):
- Pretrained by masked reconstruction on up to 100,000 hours of unlabeled dual-accelerometer HUNT4 data using the same thigh/lower-back placement style.
- Evaluated on HARTH v1.2 with 22-subject leave-one-subject-out (LOSO).
- Reports **81.3 ± 1.3% mean F1** for SelfPAB 100k and **74.2% F1** for its best supervised XGBoost baseline.
- Its main HARTH comparison uses ten classes: cycling-sit-inactive is merged with cycling-sit, and cycling-stand-inactive with cycling-stand. It produces 1-second activity predictions with 0.5-second overlap, then unfolds predictions to the original temporal grid.
- The paper's subject-average F1, not raw window accuracy, is its primary quality measure because of severe class imbalance.

These are published figures, not reproduced scores. They must not be compared to a 12-class result or a different window/aggregation protocol as though the tasks were identical. Reproduce SelfPAB's ten-class label merge, 50 Hz six-channel signal pipeline, LOSO grouping, prediction cadence, per-subject macro-F1 aggregation and evaluation conventions before claiming a direct win.

### Cross-dataset pretrained comparator: HART (2026)

The HART repository describes a HAR-specialized pretrained model based on TSPulse:
- Pretrained on 26 public HAR datasets, with 23,930 sessions, 507 subjects and approximately 456.6 million standardized sensor points.
- The released L120/d48 configuration uses 120 samples (4 seconds at 30 Hz), eight encoder blocks and 206,860 trainable parameters before downstream adaptation.
- Its 14-dataset benchmark reports aggregate F1 of 0.6803 on its in-distribution set and 0.5656 on its OOD set. These are **aggregate across different datasets**, not HARTH-specific scores.
- Its repository includes INT8 quantization and Raspberry Pi 5 deployment experiments. The listed 24.45 ms bs=1 inference / 1,255 MB RSS numbers are for WISDM, not HARTH and not an MCU measurement.

HART is a necessary cross-dataset representation and deployment comparator, but its aggregate scores cannot be inserted into the HARTH leaderboard. Before assigning HART to the HARTH ID or OOD lane, pin its exact pretraining dataset inventory and subject overlap. A dataset-family-level OOD claim requires that the target family and its subjects are absent from pretraining.

### Matched experiment matrix

The HARTH evaluation should retain two explicit label tracks:
1. **12-class native track:** all documented HARTH labels remain distinct.
2. **10-class SelfPAB-comparable track:** merge label 130 into 13 and label 140 into 14 exactly as described in SelfPAB.

For each track and each LOSO fold, compare:
- majority/stratified prior, threshold and feature-statistic baselines;
- pinned SVM and XGBoost pipelines;
- compact CNN/TCN, GRU/LSTM and GRU-D where missingness semantics apply;
- SelfPAB 100k under its published compatible pipeline;
- HART only with exact pretraining overlap disclosed and its 30 Hz / 4-second protocol reproduced where possible;
- HDC-only, CfC-only and HDC-LTC, with equal access to sensor channels and label budgets;
- fixed-rate and active-sensing policies only in a separate acquisition-policy track with matched deadlines and sensor access.

Use the same frozen participant splits for every candidate. Fit normalization, feature transforms, resampling filters, calibration and thresholds using training/development/calibration subjects only. Never window then split.

### Metrics and evidence

Primary quality:
- participant-wise macro-F1 averaged across the 22 held-out subjects;
- paired per-subject difference against the best reproduced baseline;
- worst-subject and per-class F1/recall, particularly stairs, shuffling and cycling-standing;
- both 12-class and 10-class results, never merged without explicit mapping.

For dynamic recognition:
- transition-event precision/recall, detection delay and false transitions/hour;
- quality and calibration under missingness, timestamp jitter, sensor dropout and subject-held-out shift;
- label efficiency at fixed annotation budgets (for example 1 minute, 5 minutes, 15 minutes, and full training data per activity where feasible).

For edge deployment:
- peak SRAM/flash and persistent state;
- raw p50/p95/p99 latency;
- online update time and state-write bytes;
- sensor samples/hour and wake count for active policies;
- physical complete-event energy per correct on-time decision with expanded uncertainty, measured on the same target and measurement boundary.

### Aspirational research gates (not achieved results)

- First reproduce SelfPAB's 81.3% ± 1.3% figure within its stated ten-class/LOSO protocol. A non-reproduction is still useful evidence; do not silently tune the test folds.
- Aim for at least **+2 absolute F1 percentage points** over the reproduced SelfPAB comparator, with a paired subject-level 95% confidence interval above zero, on the exact comparable track.
- A second win condition is to match the best comparator within **1 absolute F1 point** while reducing measured complete-event energy by at least **2×** on the same physical target and deadline, without worse worst-subject recall.
- For label efficiency, aim to match the full-label baseline with at most 25% of its annotated subject-time while preserving rare-class recall. This is an aspiration; establish a reproducible label-budget curve first.
- For active sensing, aim for at least 25% fewer sensor samples/hour at a preregistered fixed on-time recall and false-alarm ceiling. Measure real sensor wake and acquisition energy; do not derive battery-life claims from the synthetic simulator.

The first evidence report must publish all 22 subject-fold results, class-wise confusion counts, exact pipeline/config digests and baseline source revisions. With only 22 independent subjects, confidence intervals and paired tests matter more than a tiny difference in the aggregate point estimate.
## Claim ceiling and status

- This parser does not download or redistribute data.
- Synthetic tests do not establish that the actual archive matches every accepted CSV variant; verify against the exact archive SHA before first real run.
- Parser acceptance is not source authenticity, license adjudication, or benchmark performance.
- The optional index compatibility form should be retained only when a specific source export demonstrates its need.
- Tests are authored but have not been compiled or run. No HARTH benchmark score, energy measurement, or SOTA improvement is claimed.

## Tests authored

The test module covers canonical six-channel parsing, optional export-index continuity, quoted timestamps and CRLF, timestamp ordering and calendar validity, timezone rejection, gap accounting, non-finite sensor values, unknown labels, normalized digest replay and source-family binding, complete activity code coverage, and rejection of an unbound source-family identity.
