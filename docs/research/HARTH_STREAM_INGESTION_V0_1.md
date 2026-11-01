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
| 130 | Transport (sitting) |
| 140 | Transport (standing) |

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

## Leakage and temporal protocol

The independent split unit is the participant/source-family recording. Freeze participant identities into Training, Development, Calibration and Confirmatory before:
- making fixed or sliding windows;
- computing normalization statistics;
- extracting features;
- augmenting recordings;
- selecting thresholds or calibrating abstention.

Never concatenate participant files and then randomly split windows. Repeated windows, samples and derived features from one participant stay in one partition.

The parser's normalized time key is only an ordering/interval representation. Since the source timestamp is timezone-naive, it is not an absolute UTC instant. Keep the source bytes and archive SHA-256 in the existing real-data manifest; the parser's digest binds parsed row values and the declared source-family identity, not the publisher or original file bytes.

## Initial comparison target

The original HARTH paper reports two different label protocols. With all **12 annotated labels**, its SVM leave-one-subject-out mean F1 is **0.66 ± 0.32** (Table 2). The frequently cited **0.81 ± 0.18** is the SVM result after the paper merges those 12 labels into **nine physical activities** (Table 3): shuffling, transport-standing and standing are merged into standing; sitting and transport-sitting are merged into sitting. These are different tasks and must never be presented as one baseline. Reproduce both protocols with the exact paper features, participant folds, confusion-matrix aggregation and scoring before making an SOTA comparison.

The next matched comparison should include:
- majority/stratified prior and simple threshold/feature baselines;
- the paper's SVM baseline with pinned preprocessing;
- a compact INT8 CNN/TCN;
- GRU/LSTM and GRU-D where missingness handling is relevant;
- HDC-only, CfC-only and HDC-LTC;
- sensor-subset experiments (back only, thigh only, both);
- fixed-rate and active-sensing policies only when the acquisition semantics are matched.

Report participant-wise macro-F1 and worst-subject F1, per-class recall/precision, transition-event detection, false alarms/hour, detection delay, sensor samples/hour and complete-event energy on physical hardware. Do not report only pooled window accuracy.

## Industry context and qualification targets

Published edge-HAR systems provide useful architecture and system-cost anchors, but scores from different datasets and split protocols are not interchangeable:

| Reference | Reported result | How to use it |
|---|---|---|
| HARTH original paper | SVM mean F1 0.81 ± 0.18 in leave-one-subject-out evaluation | Primary HARTH literature baseline; reproduce the exact split/features and label aggregation before comparison |
| TinyHAR-Net (STM32F446RE) | F1 96.75–99.76% across WISDM, MotionSense and MHEALTH; 30.12 KiB RAM; 43.78–51.30 KiB flash; 41.31–51.57 ms inference | MCU footprint/deployment anchor only; do not compare those cross-dataset scores directly with HARTH |
| STORM (X-HEEP + streaming CGRA) | 19.7k parameters; 0.801 macro-F1 on its own 8-class cross-source benchmark; 67.4 ms at 100 MHz; 632.4 µJ activity-based energy estimate | Cross-source co-design and energy-accounting anchor; its dataset, hardware, task and power methodology differ from HARTH |
| L-HAR (2026 deployment study) | Three IMUs from one participant; quantized TCN Raspberry Pi streaming 9.066 ms E2E / 69.928 Hz and 49.933 mJ software-estimated energy per inference | Streaming methodology and backend-dependence warning, not cross-subject evidence; energy is an estimate rather than a direct physical power-trace result |

References:
- [HARTH original paper](https://doi.org/10.3390/s21237853)
- [TinyHAR-Net](https://doi.org/10.1109/LES.2025.3639944)
- [STORM](https://www.mdpi.com/2079-9292/15/9/1924)
- [L-HAR](https://doi.org/10.3390/s26175678)

### Proposed gates (aspirational, not measured)

1. **Reproducibility:** independently reproduce both HARTH SVM LOSO protocols: (a) the strict 12-label result (reported F1 0.66 ± 0.32) and (b) the explicitly merged nine-activity result (reported F1 0.81 ± 0.18). Publish all 22 participant-fold results, not just pooled windows, and preserve the exact confusion-matrix aggregation.
2. **Quality:** aim for at least +3 percentage points in participant-mean macro-F1 over the reproduced SVM, with a participant-paired confidence interval above zero and no material regression in the worst-subject tail. Treat +3 pp as a research target, not a claim or guaranteed threshold.
3. **Active sensing:** at a preregistered non-inferiority margin of -1 pp macro-F1, target at least 25% fewer sensor samples and report transition-event recall and p95 transition-detection delay. Do not infer these metrics from a randomly windowed classification split.
4. **Edge feasibility:** report full application-event p50/p95/p99 latency, peak SRAM after reserve, flash, sensor/wake/compute/update/communication costs and physical complete-event energy. For 50 Hz streaming, p95 end-to-end processing below the 20 ms sample period is a useful target where the selected board and workload make it meaningful; it is not a universal requirement for lower-rate active sensing.
5. **Energy superiority:** target at least 10% lower measured complete-event energy than the strongest matched baseline at the same preregistered quality and deadline constraints, with uncertainty reported. A modeled energy estimate or model-only inference measurement does not qualify.
6. **Robustness:** report subject-held-out, sensor-subset, missingness, time-jitter, abrupt and gradual shift strata separately. Use participant as the independent statistical unit; multiple windows from one participant do not create independent subjects.

Do not target the 96–99% figures from other datasets as a HARTH requirement. The first hard milestone is a faithful reproduction of both HARTH label protocols under participant-held-out evaluation. Use the nine-class 0.81 ± 0.18 comparator only for the nine-class track; use the 12-class 0.66 ± 0.32 comparator for the full-label track.

## Claim ceiling and status

- This parser does not download or redistribute data.
- Synthetic tests do not establish that the actual archive matches every accepted CSV variant; verify against the exact archive SHA before first real run.
- Parser acceptance is not source authenticity, license adjudication, or benchmark performance.
- The optional index compatibility form should be retained only when a specific source export demonstrates its need.
- Tests are authored but have not been compiled or run. No HARTH benchmark score, energy measurement, or SOTA improvement is claimed.

## Tests authored

The test module covers canonical six-channel parsing, optional export-index continuity, quoted timestamps and CRLF, timestamp ordering and calendar validity, timezone rejection, gap accounting, non-finite sensor values, unknown labels, normalized digest replay and source-family binding, complete activity code coverage, and rejection of an unbound source-family identity.
