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
| 13 | Cycling seated |
| 14 | Cycling standing |
| 130 | Cycling seated, inactive |
| 140 | Cycling standing, inactive |

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

The original HARTH work includes a leave-one-subject-out SVM baseline with reported mean F1 around 0.81 ± 0.18. That number is a literature reference, not a reproduced comparator; reproduce the exact features, label handling, subject split and scoring before using it as an SOTA comparison.

The next matched comparison should include:
- majority/stratified prior and simple threshold/feature baselines;
- the paper's SVM baseline with pinned preprocessing;
- a compact INT8 CNN/TCN;
- GRU/LSTM and GRU-D where missingness handling is relevant;
- HDC-only, CfC-only and HDC-LTC;
- sensor-subset experiments (back only, thigh only, both);
- fixed-rate and active-sensing policies only when the acquisition semantics are matched.

Report participant-wise macro-F1 and worst-subject F1, per-class recall/precision, transition-event detection, false alarms/hour, detection delay, sensor samples/hour and complete-event energy on physical hardware. Do not report only pooled window accuracy.

## Claim ceiling and status

- This parser does not download or redistribute data.
- Synthetic tests do not establish that the actual archive matches every accepted CSV variant; verify against the exact archive SHA before first real run.
- Parser acceptance is not source authenticity, license adjudication, or benchmark performance.
- The optional index compatibility form should be retained only when a specific source export demonstrates its need.
- Tests are authored but have not been compiled or run. No HARTH benchmark score, energy measurement, or SOTA improvement is claimed.

## Tests authored

The test module covers canonical six-channel parsing, optional export-index continuity, quoted timestamps and CRLF, timestamp ordering and calendar validity, timezone rejection, gap accounting, non-finite sensor values, unknown labels, normalized digest replay and source-family binding, complete activity code coverage, and rejection of an unbound source-family identity.
