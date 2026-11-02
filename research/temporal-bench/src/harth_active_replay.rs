//! Label-hidden, causal active-sensing replay over one HARTH participant recording.
//!
//! This is an offline emulated sensor policy. The evaluator owns all row labels
//! and the complete source recording; the candidate sees only the most recent
//! requested back/thigh accelerometer samples. Skipped rows are never provided
//! as measurements, and values are never interpolated or forward-filled.
//!
//! The in-process trait is not a security boundary. Confirmatory runs require
//! the isolated candidate/evaluator runner tracked in issue #9939.

use std::error::Error;
use std::fmt;

use super::digest::CanonicalHasher;
use super::harth_loso::HarthLabelTrack;
use super::harth_stream::HarthRow;

const PROBABILITY_SCALE: u32 = 1_000_000;
const MAX_WAKE_AFTER_ROWS: u32 = 4_096;
const CALIBRATION_BINS: usize = 10;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthActiveReplayError {
    pub field: String,
    pub message: String,
}

impl HarthActiveReplayError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for HarthActiveReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for HarthActiveReplayError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HarthSensorMask {
    None,
    Back,
    Thigh,
    Both,
}

impl HarthSensorMask {
    fn includes_back(self) -> bool {
        matches!(self, Self::Back | Self::Both)
    }

    fn includes_thigh(self) -> bool {
        matches!(self, Self::Thigh | Self::Both)
    }
}

/// One sensor reading visible to the candidate. No activity label is present.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarthPublicSensorSample {
    pub row_index: u64,
    pub timestamp_ns: i128,
    pub acceleration_g: [f32; 3],
}

/// Candidate-facing observation. Each modality has its own last acquired
/// sample so a policy can reason about modality-specific age and staleness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarthPublicObservation {
    pub wake_row_index: u64,
    pub wake_timestamp_ns: i128,
    pub latest_back_sample: Option<HarthPublicSensorSample>,
    pub latest_thigh_sample: Option<HarthPublicSensorSample>,
    pub back_sample_count: u64,
    pub thigh_sample_count: u64,
    pub both_sample_count: u64,
}

/// A prediction is held over the interval until the next wake. None is an
/// explicit abstention and is scored as a false negative for every true row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HarthActiveReplayAction {
    /// A code from the selected HarthLabelTrack, never a raw merged-away code.
    pub prediction_label_code: Option<u16>,
    pub confidence_ppm: u32,
    /// Acquisition mask to apply at the next wake, not at this one.
    pub next_sensor_mask: HarthSensorMask,
    /// Advance in source rows, not fabricated time. Actual timestamp intervals
    /// are retained and reported because HARTH recordings contain gaps.
    pub next_wake_after_rows: u32,
    pub declared_compute_cycles: u64,
    pub declared_transmit_bytes: u64,
    pub declared_memory_write_bytes: u64,
}

pub trait HarthActiveReplayPolicy {
    fn decide(&mut self, observation: &HarthPublicObservation) -> HarthActiveReplayAction;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthReplayCandidateIdentity {
    pub policy_id: String,
    pub source_revision: String,
    pub dependency_lock_sha256: [u8; 32],
    pub artifact_sha256: [u8; 32],
    pub configuration_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthReplayClassMetrics {
    pub label_code: u16,
    pub true_support: u64,
    pub predicted_support: u64,
    pub precision_ppm: Option<u32>,
    pub recall_ppm: Option<u32>,
    pub f1_ppm: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthReplayCalibrationBin {
    /// Bin index 0..9. Confidence 1.0 is included in bin 9.
    pub bin_index: u8,
    pub sample_count: u64,
    pub confidence_sum_ppm: u64,
    pub correct_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthTransitionMetrics {
    pub true_transition_count: u64,
    pub detected_transition_count: u64,
    pub missed_transition_count: u64,
    pub false_transition_alarm_count: u64,
    /// Delays are in source timestamp nanoseconds, not nominal sample periods.
    pub detected_transition_delays_ns: Vec<u64>,
    pub transition_delay_p95_ns: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthActiveReplayReceipt {
    pub schema: String,
    pub version: String,
    pub dataset_manifest_sha256: [u8; 32],
    pub preprocessing_sha256: [u8; 32],
    pub source_family_sha256: [u8; 32],
    pub source_normalized_rows_sha256: [u8; 32],
    pub label_track: HarthLabelTrack,
    pub candidate: HarthReplayCandidateIdentity,
    pub row_count: u64,
    /// Row-major matrix: true class x predicted class, plus one final abstain
    /// column. Abstentions therefore remain false negatives, not exclusions.
    pub confusion_counts: Vec<u64>,
    pub class_metrics: Vec<HarthReplayClassMetrics>,
    pub accuracy_ppm: u32,
    pub macro_f1_ppm: u32,
    pub covered_rows: u64,
    pub abstained_rows: u64,
    pub coverage_ppm: u32,
    pub calibration_bins: Vec<HarthReplayCalibrationBin>,
    pub expected_calibration_error_ppm: u32,
    pub transitions: HarthTransitionMetrics,
    pub back_sample_count: u64,
    pub thigh_sample_count: u64,
    pub both_sample_count: u64,
    pub sample_acquisition_count: u64,
    pub sampled_axis_count: u64,
    pub wake_count: u64,
    pub wake_intervals_ns: Vec<u64>,
    pub wake_interval_p50_ns: Option<u64>,
    pub wake_interval_p95_ns: Option<u64>,
    pub wake_interval_max_ns: Option<u64>,
    /// Age of the oldest last-acquired modality, measured for each scored row.
    pub mean_observation_age_ns: u64,
    pub observation_age_p95_ns: u64,
    pub maximum_observation_age_ns: u64,
    pub declared_compute_cycles: u64,
    pub declared_transmit_bytes: u64,
    pub declared_memory_write_bytes: u64,
    pub candidate_trace_sha256: [u8; 32],
    pub receipt_sha256: [u8; 32],
}

fn digest_is_nonzero(value: &[u8; 32]) -> bool {
    value.iter().any(|byte| *byte != 0)
}

fn p95_nearest_rank(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (95 * sorted.len() + 99) / 100;
    Some(sorted[rank.saturating_sub(1)])
}

fn percentile_nearest_rank(values: &[u64], percentile: usize) -> Option<u64> {
    if values.is_empty() || !(1..=100).contains(&percentile) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (percentile * sorted.len() + 99) / 100;
    Some(sorted[rank.saturating_sub(1)])
}

fn ppm(numerator: u64, denominator: u64) -> u32 {
    if denominator == 0 {
        0
    } else {
        ((u128::from(numerator) * u128::from(PROBABILITY_SCALE)
            / u128::from(denominator))
            .min(u128::from(PROBABILITY_SCALE))) as u32
    }
}

fn validate_rows(
    rows: &[HarthRow],
    track: HarthLabelTrack,
) -> Result<(), HarthActiveReplayError> {
    if rows.is_empty() {
        return Err(HarthActiveReplayError::new("rows", "participant recording is empty"));
    }
    let mut previous_timestamp = None;
    for (index, row) in rows.iter().enumerate() {
        if row.row_index != index as u64 {
            return Err(HarthActiveReplayError::new(
                format!("rows[{index}].row_index"),
                "row indices must start at zero and be contiguous within a participant file",
            ));
        }
        if previous_timestamp.is_some_and(|previous| row.timestamp_ns <= previous) {
            return Err(HarthActiveReplayError::new(
                format!("rows[{index}].timestamp_ns"),
                "timestamps must be strictly increasing",
            ));
        }
        if row.acceleration_g.iter().any(|value| !value.is_finite()) {
            return Err(HarthActiveReplayError::new(
                format!("rows[{index}].acceleration_g"),
                "all six channels must be finite",
            ));
        }
        if track.class_index(row.activity.code()).is_none() {
            return Err(HarthActiveReplayError::new(
                format!("rows[{index}].activity"),
                format!("raw label {} is not represented in selected track", row.activity.code()),
            ));
        }
        previous_timestamp = Some(row.timestamp_ns);
    }
    Ok(())
}

fn sample_sensor(
    row: &HarthRow,
    modality: HarthSensorMask,
) -> (Option<HarthPublicSensorSample>, Option<HarthPublicSensorSample>) {
    let back = modality.includes_back().then_some(HarthPublicSensorSample {
        row_index: row.row_index,
        timestamp_ns: row.timestamp_ns,
        acceleration_g: [row.acceleration_g[0], row.acceleration_g[1], row.acceleration_g[2]],
    });
    let thigh = modality.includes_thigh().then_some(HarthPublicSensorSample {
        row_index: row.row_index,
        timestamp_ns: row.timestamp_ns,
        acceleration_g: [row.acceleration_g[3], row.acceleration_g[4], row.acceleration_g[5]],
    });
    (back, thigh)
}

fn confidence_bin(confidence_ppm: u32) -> usize {
    if confidence_ppm >= PROBABILITY_SCALE {
        CALIBRATION_BINS - 1
    } else {
        (confidence_ppm as usize * CALIBRATION_BINS / PROBABILITY_SCALE as usize)
            .min(CALIBRATION_BINS - 1)
    }
}

fn class_metrics_from_confusion(
    track: HarthLabelTrack,
    confusion: &[u64],
) -> Result<(Vec<HarthReplayClassMetrics>, u32, u32), HarthActiveReplayError> {
    let class_count = track.class_count();
    let expected = class_count.checked_mul(class_count + 1).ok_or_else(|| {
        HarthActiveReplayError::new("confusion_counts", "matrix dimensions overflow")
    })?;
    if confusion.len() != expected {
        return Err(HarthActiveReplayError::new(
            "confusion_counts",
            format!("expected {expected} counts, got {}", confusion.len()),
        ));
    }

    let mut metrics = Vec::with_capacity(class_count);
    let mut f1_sum = 0_u128;
    let mut present_classes = 0_u64;
    let mut diagonal = 0_u64;

    for class_index in 0..class_count {
        let mut true_support = 0_u64;
        let mut predicted_support = 0_u64;
        for predicted in 0..=class_count {
            true_support = true_support.checked_add(confusion[class_index * (class_count + 1) + predicted])
                .ok_or_else(|| HarthActiveReplayError::new("confusion_counts", "row support overflow"))?;
        }
        for truth in 0..class_count {
            predicted_support = predicted_support.checked_add(confusion[truth * (class_count + 1) + class_index])
                .ok_or_else(|| HarthActiveReplayError::new("confusion_counts", "column support overflow"))?;
        }
        let tp = confusion[class_index * (class_count + 1) + class_index];
        diagonal = diagonal.checked_add(tp)
            .ok_or_else(|| HarthActiveReplayError::new("confusion_counts", "diagonal overflow"))?;
        let fp = predicted_support.saturating_sub(tp);
        let fn_count = true_support.saturating_sub(tp);

        let precision = (predicted_support > 0).then(|| ppm(tp, predicted_support));
        let recall = (true_support > 0).then(|| ppm(tp, true_support));
        let denominator = u128::from(tp) * 2 + u128::from(fp) + u128::from(fn_count);
        let f1 = (denominator > 0).then(|| {
            ((u128::from(tp) * 2 * u128::from(PROBABILITY_SCALE) / denominator)
                .min(u128::from(PROBABILITY_SCALE))) as u32
        });
        if let Some(score) = f1 {
            f1_sum += u128::from(score);
            present_classes += 1;
        }
        metrics.push(HarthReplayClassMetrics {
            label_code: track.class_codes()[class_index],
            true_support,
            predicted_support,
            precision_ppm: precision,
            recall_ppm: recall,
            f1_ppm: f1,
        });
    }

    let macro_f1 = if present_classes == 0 {
        0
    } else {
        ((f1_sum + u128::from(present_classes / 2)) / u128::from(present_classes)) as u32
    };
    Ok((metrics, macro_f1, diagonal))
}

fn calibration_error(
    bins: &[HarthReplayCalibrationBin],
    total_rows: u64,
) -> Result<u32, HarthActiveReplayError> {
    if bins.len() != CALIBRATION_BINS {
        return Err(HarthActiveReplayError::new(
            "calibration_bins",
            "exactly ten fixed calibration bins are required",
        ));
    }
    let mut total = 0_u64;
    let mut weighted_error = 0_u128;
    for (index, bin) in bins.iter().enumerate() {
        let lower_confidence = (index as u64) * u64::from(PROBABILITY_SCALE) / CALIBRATION_BINS as u64;
        let upper_confidence = ((index + 1) as u64) * u64::from(PROBABILITY_SCALE) / CALIBRATION_BINS as u64;
        let minimum_sum = lower_confidence.saturating_mul(bin.sample_count);
        let maximum_sum_exclusive = upper_confidence.saturating_mul(bin.sample_count);
        if usize::from(bin.bin_index) != index
            || bin.correct_count > bin.sample_count
            || bin.confidence_sum_ppm > bin.sample_count.saturating_mul(u64::from(PROBABILITY_SCALE))
            || (bin.sample_count > 0 && bin.confidence_sum_ppm < minimum_sum)
            || (bin.sample_count > 0 && index + 1 < CALIBRATION_BINS
                && bin.confidence_sum_ppm >= maximum_sum_exclusive)
        {
            return Err(HarthActiveReplayError::new(
                format!("calibration_bins[{index}]"),
                "bin identity, confidence range or correct-count denominator is invalid",
            ));
        }
        total = total.checked_add(bin.sample_count).ok_or_else(|| {
            HarthActiveReplayError::new("calibration_bins", "sample count overflow")
        })?;
        if bin.sample_count > 0 {
            let mean_confidence = bin.confidence_sum_ppm / bin.sample_count;
            let empirical_accuracy = u64::from(ppm(bin.correct_count, bin.sample_count));
            let absolute_gap = mean_confidence.abs_diff(empirical_accuracy);
            weighted_error = weighted_error.saturating_add(
                u128::from(bin.sample_count) * u128::from(absolute_gap)
            );
        }
    }
    if total != total_rows {
        return Err(HarthActiveReplayError::new(
            "calibration_bins",
            format!("non-abstaining bin samples {total} must equal covered rows {total_rows}"),
        ));
    }
    if total_rows == 0 {
        return Ok(0);
    }
    Ok((weighted_error / u128::from(total_rows)).min(u128::from(PROBABILITY_SCALE)) as u32)
}

fn hash_public_sample(
    hash: &mut CanonicalHasher,
    sample: Option<HarthPublicSensorSample>,
) {
    match sample {
        None => hash.u8(0),
        Some(sample) => {
            hash.u8(1);
            hash.u64(sample.row_index);
            hash.bytes(&sample.timestamp_ns.to_le_bytes());
            for value in sample.acceleration_g {
                hash.u32(value.to_bits());
            }
        }
    }
}

fn result_digest(receipt: &HarthActiveReplayReceipt) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.harth-active-replay-receipt.v1");
    hash.bytes(receipt.schema.as_bytes());
    hash.bytes(receipt.version.as_bytes());
    hash.bytes(&receipt.dataset_manifest_sha256);
    hash.bytes(&receipt.preprocessing_sha256);
    hash.bytes(&receipt.source_family_sha256);
    hash.bytes(&receipt.source_normalized_rows_sha256);
    hash.u8(receipt.label_track as u8);
    hash.bytes(receipt.candidate.policy_id.as_bytes());
    hash.bytes(receipt.candidate.source_revision.as_bytes());
    hash.bytes(&receipt.candidate.dependency_lock_sha256);
    hash.bytes(&receipt.candidate.artifact_sha256);
    hash.bytes(&receipt.candidate.configuration_sha256);
    hash.u64(receipt.row_count);
    hash.usize(receipt.confusion_counts.len());
    for count in &receipt.confusion_counts {
        hash.u64(*count);
    }
    hash.usize(receipt.class_metrics.len());
    for metric in &receipt.class_metrics {
        hash.u16(metric.label_code);
        hash.u64(metric.true_support);
        hash.u64(metric.predicted_support);
        hash.u32(metric.precision_ppm.unwrap_or(u32::MAX));
        hash.u32(metric.recall_ppm.unwrap_or(u32::MAX));
        hash.u32(metric.f1_ppm.unwrap_or(u32::MAX));
    }
    hash.u32(receipt.accuracy_ppm);
    hash.u32(receipt.macro_f1_ppm);
    hash.u64(receipt.covered_rows);
    hash.u64(receipt.abstained_rows);
    hash.u32(receipt.coverage_ppm);
    hash.usize(receipt.calibration_bins.len());
    for bin in &receipt.calibration_bins {
        hash.u8(bin.bin_index);
        hash.u64(bin.sample_count);
        hash.u64(bin.confidence_sum_ppm);
        hash.u64(bin.correct_count);
    }
    hash.u32(receipt.expected_calibration_error_ppm);
    hash.u64(receipt.transitions.true_transition_count);
    hash.u64(receipt.transitions.detected_transition_count);
    hash.u64(receipt.transitions.missed_transition_count);
    hash.u64(receipt.transitions.false_transition_alarm_count);
    hash.usize(receipt.transitions.detected_transition_delays_ns.len());
    for delay in &receipt.transitions.detected_transition_delays_ns {
        hash.u64(*delay);
    }
    hash.u64(receipt.transitions.transition_delay_p95_ns.unwrap_or(u64::MAX));
    for value in [
        receipt.back_sample_count,
        receipt.thigh_sample_count,
        receipt.both_sample_count,
        receipt.sample_acquisition_count,
        receipt.sampled_axis_count,
        receipt.wake_count,
    ] {
        hash.u64(value);
    }
    hash.usize(receipt.wake_intervals_ns.len());
    for interval in &receipt.wake_intervals_ns {
        hash.u64(*interval);
    }
    hash.u64(receipt.wake_interval_p50_ns.unwrap_or(u64::MAX));
    hash.u64(receipt.wake_interval_p95_ns.unwrap_or(u64::MAX));
    hash.u64(receipt.wake_interval_max_ns.unwrap_or(u64::MAX));
    hash.u64(receipt.mean_observation_age_ns);
    hash.u64(receipt.observation_age_p95_ns);
    hash.u64(receipt.maximum_observation_age_ns);
    hash.u64(receipt.declared_compute_cycles);
    hash.u64(receipt.declared_transmit_bytes);
    hash.u64(receipt.declared_memory_write_bytes);
    hash.bytes(&receipt.candidate_trace_sha256);
    hash.finish()
}

/// Run one recording. rows must contain one participant only; callers must
/// freeze participant splits before windowing or joining source families.
pub fn run_harth_active_replay<P: HarthActiveReplayPolicy>(
    rows: &[HarthRow],
    dataset_manifest_sha256: [u8; 32],
    preprocessing_sha256: [u8; 32],
    source_family_sha256: [u8; 32],
    source_normalized_rows_sha256: [u8; 32],
    track: HarthLabelTrack,
    candidate: HarthReplayCandidateIdentity,
    policy: &mut P,
) -> Result<HarthActiveReplayReceipt, HarthActiveReplayError> {
    validate_rows(rows, track)?;
    for (name, digest) in [
        ("dataset_manifest_sha256", &dataset_manifest_sha256),
        ("preprocessing_sha256", &preprocessing_sha256),
        ("source_family_sha256", &source_family_sha256),
        ("source_normalized_rows_sha256", &source_normalized_rows_sha256),
        ("candidate.dependency_lock_sha256", &candidate.dependency_lock_sha256),
        ("candidate.artifact_sha256", &candidate.artifact_sha256),
        ("candidate.configuration_sha256", &candidate.configuration_sha256),
    ] {
        if !digest_is_nonzero(digest) {
            return Err(HarthActiveReplayError::new(name, "digest must be nonzero"));
        }
    }
    if candidate.policy_id.trim().is_empty() || candidate.source_revision.trim().is_empty() {
        return Err(HarthActiveReplayError::new(
            "candidate",
            "policy ID and source revision must be nonempty",
        ));
    }

    // Common bootstrap: every policy receives and pays for back + thigh at row 0.
    let (mut latest_back, mut latest_thigh) = sample_sensor(&rows[0], HarthSensorMask::Both);
    let mut back_sample_count = 1_u64;
    let mut thigh_sample_count = 1_u64;
    let mut both_sample_count = 1_u64;
    let mut sample_acquisition_count = 1_u64;
    let mut wake_count = 0_u64;
    let mut wake_intervals_ns = Vec::new();
    let mut observation_ages_ns = Vec::with_capacity(rows.len());
    let mut predictions: Vec<Option<usize>> = Vec::with_capacity(rows.len());
    let mut confidence_by_row: Vec<u32> = Vec::with_capacity(rows.len());
    let class_count = track.class_count();
    let mut trace = CanonicalHasher::new(b"symthaea.temporal.harth-active-replay-trace.v1");
    trace.bytes(&dataset_manifest_sha256);
    trace.bytes(&preprocessing_sha256);
    trace.bytes(&source_family_sha256);
    trace.bytes(&source_normalized_rows_sha256);
    trace.u8(track as u8);
    trace.bytes(&candidate.artifact_sha256);
    trace.bytes(&candidate.configuration_sha256);

    let mut declared_compute_cycles = 0_u64;
    let mut declared_transmit_bytes = 0_u64;
    let mut declared_memory_write_bytes = 0_u64;
    let mut current_row = 0_usize;

    while current_row < rows.len() {
        let row = &rows[current_row];
        let back_age = latest_back.map(|sample| {
            u64::try_from(row.timestamp_ns - sample.timestamp_ns).unwrap_or(u64::MAX)
        });
        let thigh_age = latest_thigh.map(|sample| {
            u64::try_from(row.timestamp_ns - sample.timestamp_ns).unwrap_or(u64::MAX)
        });
        let public = HarthPublicObservation {
            wake_row_index: row.row_index,
            wake_timestamp_ns: row.timestamp_ns,
            latest_back_sample: latest_back,
            latest_thigh_sample: latest_thigh,
            back_sample_count,
            thigh_sample_count,
            both_sample_count,
        };
        let action = policy.decide(&public);
        if action.next_wake_after_rows == 0 || action.next_wake_after_rows > MAX_WAKE_AFTER_ROWS {
            return Err(HarthActiveReplayError::new(
                "action.next_wake_after_rows",
                format!("wake interval must be in 1..={MAX_WAKE_AFTER_ROWS}"),
            ));
        }
        if action.confidence_ppm > PROBABILITY_SCALE {
            return Err(HarthActiveReplayError::new(
                "action.confidence_ppm",
                "confidence must be in [0, 1,000,000] ppm",
            ));
        }
        let prediction_class = match action.prediction_label_code {
            Some(code) => Some(track.class_index(code).ok_or_else(|| {
                HarthActiveReplayError::new(
                    "action.prediction_label_code",
                    format!("prediction code {code} is not in the selected label track"),
                )
            })?),
            None => {
                if action.confidence_ppm != 0 {
                    return Err(HarthActiveReplayError::new(
                        "action.confidence_ppm",
                        "abstaining actions must use zero confidence",
                    ));
                }
                None
            }
        };

        let next_row = current_row
            .saturating_add(action.next_wake_after_rows as usize)
            .min(rows.len());
        for scored_index in current_row..next_row {
            let scored = &rows[scored_index];
            predictions.push(prediction_class);
            confidence_by_row.push(action.confidence_ppm);
            let mut age = 0_u64;
            if let Some(sample) = latest_back {
                age = age.max(u64::try_from(scored.timestamp_ns - sample.timestamp_ns).unwrap_or(u64::MAX));
            }
            if let Some(sample) = latest_thigh {
                age = age.max(u64::try_from(scored.timestamp_ns - sample.timestamp_ns).unwrap_or(u64::MAX));
            }
            observation_ages_ns.push(age);
        }

        wake_count = wake_count.checked_add(1).ok_or_else(|| {
            HarthActiveReplayError::new("wake_count", "counter overflow")
        })?;
        declared_compute_cycles = declared_compute_cycles.checked_add(action.declared_compute_cycles)
            .ok_or_else(|| HarthActiveReplayError::new("declared_compute_cycles", "counter overflow"))?;
        declared_transmit_bytes = declared_transmit_bytes.checked_add(action.declared_transmit_bytes)
            .ok_or_else(|| HarthActiveReplayError::new("declared_transmit_bytes", "counter overflow"))?;
        declared_memory_write_bytes = declared_memory_write_bytes.checked_add(action.declared_memory_write_bytes)
            .ok_or_else(|| HarthActiveReplayError::new("declared_memory_write_bytes", "counter overflow"))?;

        trace.u64(row.row_index);
        trace.bytes(&row.timestamp_ns.to_le_bytes());
        hash_public_sample(&mut trace, latest_back);
        hash_public_sample(&mut trace, latest_thigh);
        trace.u64(back_age.unwrap_or(u64::MAX));
        trace.u64(thigh_age.unwrap_or(u64::MAX));
        trace.u8(action.prediction_label_code.map(|code| track.class_index(code).unwrap_or(u8::MAX as usize) as u8).unwrap_or(u8::MAX));
        trace.u32(action.confidence_ppm);
        trace.u8(match action.next_sensor_mask {
            HarthSensorMask::None => 0,
            HarthSensorMask::Back => 1,
            HarthSensorMask::Thigh => 2,
            HarthSensorMask::Both => 3,
        });
        trace.u32(action.next_wake_after_rows);
        trace.u64(action.declared_compute_cycles);
        trace.u64(action.declared_transmit_bytes);
        trace.u64(action.declared_memory_write_bytes);

        if next_row < rows.len() {
            let previous_wake = &rows[current_row];
            let next_wake = &rows[next_row];
            let wake_delta = u64::try_from(next_wake.timestamp_ns - previous_wake.timestamp_ns)
                .map_err(|_| HarthActiveReplayError::new("wake_intervals_ns", "timestamp delta is negative or too large"))?;
            if wake_delta == 0 {
                return Err(HarthActiveReplayError::new("wake_intervals_ns", "wake interval must be positive"));
            }
            wake_intervals_ns.push(wake_delta);

            match action.next_sensor_mask {
                HarthSensorMask::None => {}
                mask => {
                    let (back, thigh) = sample_sensor(next_wake, mask);
                    if let Some(sample) = back {
                        latest_back = Some(sample);
                        back_sample_count = back_sample_count.checked_add(1).ok_or_else(|| {
                            HarthActiveReplayError::new("back_sample_count", "counter overflow")
                        })?;
                    }
                    if let Some(sample) = thigh {
                        latest_thigh = Some(sample);
                        thigh_sample_count = thigh_sample_count.checked_add(1).ok_or_else(|| {
                            HarthActiveReplayError::new("thigh_sample_count", "counter overflow")
                        })?;
                    }
                    if mask == HarthSensorMask::Both {
                        both_sample_count = both_sample_count.checked_add(1).ok_or_else(|| {
                            HarthActiveReplayError::new("both_sample_count", "counter overflow")
                        })?;
                    }
                    sample_acquisition_count = sample_acquisition_count.checked_add(1).ok_or_else(|| {
                        HarthActiveReplayError::new("sample_acquisition_count", "counter overflow")
                    })?;
                }
            }
        }
        current_row = next_row;
    }

    let mut confusion_counts = vec![0_u64; class_count * (class_count + 1)];
    let mut calibration_bins: Vec<HarthReplayCalibrationBin> = (0..CALIBRATION_BINS)
        .map(|index| HarthReplayCalibrationBin {
            bin_index: index as u8,
            sample_count: 0,
            confidence_sum_ppm: 0,
            correct_count: 0,
        })
        .collect();
    let mut covered_rows = 0_u64;
    let mut abstained_rows = 0_u64;

    for (index, (prediction, confidence)) in predictions.iter().zip(confidence_by_row.iter()).enumerate() {
        let truth_code = rows[index].activity.code();
        let truth_class = track.class_index(truth_code).ok_or_else(|| {
            HarthActiveReplayError::new("truth_label", "validated label disappeared from selected track")
        })?;
        let prediction_column = prediction.unwrap_or(class_count);
        let matrix_index = truth_class * (class_count + 1) + prediction_column;
        confusion_counts[matrix_index] = confusion_counts[matrix_index].checked_add(1).ok_or_else(|| {
            HarthActiveReplayError::new("confusion_counts", "count overflow")
        })?;
        match prediction {
            None => abstained_rows += 1,
            Some(predicted) => {
                covered_rows += 1;
                let bin = &mut calibration_bins[confidence_bin(*confidence)];
                bin.sample_count += 1;
                bin.confidence_sum_ppm += u64::from(*confidence);
                if *predicted == truth_class {
                    bin.correct_count += 1;
                }
            }
        }
    }

    let (class_metrics, macro_f1_ppm, diagonal) =
        class_metrics_from_confusion(track, &confusion_counts)?;
    let row_count = rows.len() as u64;
    let accuracy_ppm = ppm(diagonal, row_count);
    let coverage_ppm = ppm(covered_rows, row_count);
    let expected_calibration_error_ppm = calibration_error(&calibration_bins, covered_rows)?;

    let mut true_transition_count = 0_u64;
    let mut detected_transition_count = 0_u64;
    let mut missed_transition_count = 0_u64;
    let mut detected_transition_delays_ns = Vec::new();
    let mut false_transition_alarm_count = 0_u64;
    for index in 1..rows.len() {
        let truth_previous = track.class_index(rows[index - 1].activity.code()).unwrap();
        let truth_current = track.class_index(rows[index].activity.code()).unwrap();
        if truth_current != truth_previous {
            true_transition_count += 1;
            let mut segment_end = index + 1;
            while segment_end < rows.len()
                && track.class_index(rows[segment_end].activity.code()).unwrap() == truth_current
            {
                segment_end += 1;
            }
            let detection = (index..segment_end)
                .find(|candidate_index| {
                    predictions[*candidate_index] == Some(truth_current)
                        && predictions[*candidate_index - 1] != Some(truth_current)
                });
            if let Some(detected_at) = detection {
                detected_transition_count += 1;
                let delay = u64::try_from(rows[detected_at].timestamp_ns - rows[index].timestamp_ns)
                    .map_err(|_| HarthActiveReplayError::new("transition_delay", "negative or oversized delay"))?;
                detected_transition_delays_ns.push(delay);
            } else {
                missed_transition_count += 1;
            }
        } else if let Some(current) = predictions[index] {
            if predictions[index - 1] != Some(current) && current != truth_current {
                false_transition_alarm_count += 1;
            }
        }
    }

    let maximum_observation_age_ns = observation_ages_ns.iter().copied().max().unwrap_or(0);
    let mean_observation_age_ns = if observation_ages_ns.is_empty() {
        0
    } else {
        (observation_ages_ns.iter().map(|value| u128::from(*value)).sum::<u128>()
            / observation_ages_ns.len() as u128)
            .min(u128::from(u64::MAX)) as u64
    };
    let observation_age_p95_ns = p95_nearest_rank(&observation_ages_ns).unwrap_or(0);
    let sampled_axis_count = back_sample_count.checked_add(thigh_sample_count)
        .and_then(|count| count.checked_mul(3))
        .ok_or_else(|| HarthActiveReplayError::new("sampled_axis_count", "axis count overflow"))?;

    let mut receipt = HarthActiveReplayReceipt {
        schema: "symthaea.temporal.harth-active-replay".to_owned(),
        version: "0.1.0".to_owned(),
        dataset_manifest_sha256,
        preprocessing_sha256,
        source_family_sha256,
        source_normalized_rows_sha256,
        label_track: track,
        candidate,
        row_count,
        confusion_counts,
        class_metrics,
        accuracy_ppm,
        macro_f1_ppm,
        covered_rows,
        abstained_rows,
        coverage_ppm,
        calibration_bins,
        expected_calibration_error_ppm,
        transitions: HarthTransitionMetrics {
            true_transition_count,
            detected_transition_count,
            missed_transition_count,
            false_transition_alarm_count,
            transition_delay_p95_ns: p95_nearest_rank(&detected_transition_delays_ns),
            detected_transition_delays_ns,
        },
        back_sample_count,
        thigh_sample_count,
        both_sample_count,
        sample_acquisition_count,
        sampled_axis_count,
        wake_count,
        wake_interval_p50_ns: percentile_nearest_rank(&wake_intervals_ns, 50),
        wake_interval_p95_ns: p95_nearest_rank(&wake_intervals_ns),
        wake_interval_max_ns: wake_intervals_ns.iter().copied().max(),
        wake_intervals_ns,
        mean_observation_age_ns,
        observation_age_p95_ns,
        maximum_observation_age_ns,
        declared_compute_cycles,
        declared_transmit_bytes,
        declared_memory_write_bytes,
        candidate_trace_sha256: trace.finish(),
        receipt_sha256: [0; 32],
    };
    receipt.receipt_sha256 = result_digest(&receipt);
    validate_harth_active_replay_receipt(&receipt)?;
    Ok(receipt)
}

/// Recheck principal denominators, derived metrics and the canonical digest.
/// Full participant rows remain necessary for independent raw-data replay.
pub fn validate_harth_active_replay_receipt(
    receipt: &HarthActiveReplayReceipt,
) -> Result<(), HarthActiveReplayError> {
    if receipt.schema != "symthaea.temporal.harth-active-replay" || receipt.version != "0.1.0" {
        return Err(HarthActiveReplayError::new("schema", "unsupported schema or version"));
    }
    for (name, digest) in [
        ("dataset_manifest_sha256", &receipt.dataset_manifest_sha256),
        ("preprocessing_sha256", &receipt.preprocessing_sha256),
        ("source_family_sha256", &receipt.source_family_sha256),
        ("source_normalized_rows_sha256", &receipt.source_normalized_rows_sha256),
        ("candidate.dependency_lock_sha256", &receipt.candidate.dependency_lock_sha256),
        ("candidate.artifact_sha256", &receipt.candidate.artifact_sha256),
        ("candidate.configuration_sha256", &receipt.candidate.configuration_sha256),
        ("candidate_trace_sha256", &receipt.candidate_trace_sha256),
    ] {
        if !digest_is_nonzero(digest) {
            return Err(HarthActiveReplayError::new(name, "digest must be nonzero"));
        }
    }
    if receipt.candidate.policy_id.trim().is_empty() || receipt.candidate.source_revision.trim().is_empty() {
        return Err(HarthActiveReplayError::new("candidate", "identity strings must be nonempty"));
    }
    if receipt.row_count == 0
        || receipt.covered_rows.checked_add(receipt.abstained_rows) != Some(receipt.row_count)
    {
        return Err(HarthActiveReplayError::new("row_count", "row/coverage denominators are inconsistent"));
    }
    let matrix_total = receipt.confusion_counts.iter().try_fold(0_u64, |sum, value| sum.checked_add(*value))
        .ok_or_else(|| HarthActiveReplayError::new("confusion_counts", "matrix total overflow"))?;
    if matrix_total != receipt.row_count {
        return Err(HarthActiveReplayError::new("confusion_counts", "matrix does not sum to row count"));
    }
    let class_count = receipt.label_track.class_count();
    let expected_matrix_len = class_count.checked_mul(class_count + 1).ok_or_else(|| {
        HarthActiveReplayError::new("confusion_counts", "matrix dimensions overflow")
    })?;
    if receipt.confusion_counts.len() != expected_matrix_len
        || receipt.class_metrics.len() != class_count
    {
        return Err(HarthActiveReplayError::new("confusion_counts", "matrix or class metric dimensions do not match label track"));
    }
    let abstention_count = (0..class_count).try_fold(0_u64, |sum, truth| {
        sum.checked_add(receipt.confusion_counts[truth * (class_count + 1) + class_count])
    }).ok_or_else(|| HarthActiveReplayError::new("confusion_counts", "abstention count overflow"))?;
    if abstention_count != receipt.abstained_rows
        || matrix_total.checked_sub(abstention_count) != Some(receipt.covered_rows)
    {
        return Err(HarthActiveReplayError::new("coverage", "abstention/coverage counts do not reproduce from confusion matrix"));
    }
    if receipt.wake_count == 0
        || receipt.wake_count > receipt.row_count
        || receipt.wake_intervals_ns.len() as u64 + 1 != receipt.wake_count
        || receipt.wake_intervals_ns.iter().any(|interval| *interval == 0)
        || receipt.sample_acquisition_count > receipt.wake_count
        || receipt.back_sample_count > receipt.wake_count
        || receipt.thigh_sample_count > receipt.wake_count
        || receipt.both_sample_count > receipt.sample_acquisition_count
    {
        return Err(HarthActiveReplayError::new("wake_count", "wake and acquisition denominators are inconsistent"));
    }
    let modality_sum = receipt.back_sample_count.checked_add(receipt.thigh_sample_count)
        .ok_or_else(|| HarthActiveReplayError::new("sample_acquisition_count", "modality count overflow"))?;
    if receipt.both_sample_count > receipt.back_sample_count
        || receipt.both_sample_count > receipt.thigh_sample_count
        || modality_sum.checked_sub(receipt.both_sample_count) != Some(receipt.sample_acquisition_count)
    {
        return Err(HarthActiveReplayError::new("sample_acquisition_count", "sensor acquisition denominators are inconsistent"));
    }
    let expected_axes = modality_sum.checked_mul(3)
        .ok_or_else(|| HarthActiveReplayError::new("sampled_axis_count", "axis count overflow"))?;
    if expected_axes != receipt.sampled_axis_count {
        return Err(HarthActiveReplayError::new("sampled_axis_count", "axis count does not reproduce from modality counts"));
    }
    let (expected_classes, expected_macro, expected_diagonal) =
        class_metrics_from_confusion(receipt.label_track, &receipt.confusion_counts)?;
    if receipt.class_metrics != expected_classes
        || receipt.macro_f1_ppm != expected_macro
        || receipt.accuracy_ppm != ppm(expected_diagonal, receipt.row_count)
        || receipt.coverage_ppm != ppm(receipt.covered_rows, receipt.row_count)
    {
        return Err(HarthActiveReplayError::new("derived_metrics", "metrics do not reproduce from confusion counts"));
    }
    let expected_ece = calibration_error(&receipt.calibration_bins, receipt.covered_rows)?;
    if expected_ece != receipt.expected_calibration_error_ppm {
        return Err(HarthActiveReplayError::new("expected_calibration_error_ppm", "ECE does not reproduce from calibration bins"));
    }
    if receipt.transitions.detected_transition_count.checked_add(receipt.transitions.missed_transition_count)
        != Some(receipt.transitions.true_transition_count)
        || receipt.transitions.detected_transition_delays_ns.len() as u64
            != receipt.transitions.detected_transition_count
        || receipt.transitions.transition_delay_p95_ns
            != p95_nearest_rank(&receipt.transitions.detected_transition_delays_ns)
    {
        return Err(HarthActiveReplayError::new("transitions", "transition denominators or delay summary are inconsistent"));
    }
    if receipt.wake_interval_p50_ns != percentile_nearest_rank(&receipt.wake_intervals_ns, 50)
        || receipt.wake_interval_p95_ns != p95_nearest_rank(&receipt.wake_intervals_ns)
        || receipt.wake_interval_max_ns != receipt.wake_intervals_ns.iter().copied().max()
    {
        return Err(HarthActiveReplayError::new("wake_intervals_ns", "wake interval summaries do not reproduce"));
    }
    if receipt.observation_age_p95_ns > receipt.maximum_observation_age_ns
        || receipt.mean_observation_age_ns > receipt.maximum_observation_age_ns
    {
        return Err(HarthActiveReplayError::new("observation_age", "age summaries exceed maximum"));
    }
    if receipt.receipt_sha256 != result_digest(receipt) {
        return Err(HarthActiveReplayError::new("receipt_sha256", "canonical receipt digest mismatch"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::harth_stream::HarthActivity;

    struct FixedPolicy {
        prediction: Option<u16>,
        mask: HarthSensorMask,
        interval: u32,
        confidence: u32,
        seen: Vec<HarthPublicObservation>,
    }

    impl HarthActiveReplayPolicy for FixedPolicy {
        fn decide(&mut self, observation: &HarthPublicObservation) -> HarthActiveReplayAction {
            self.seen.push(*observation);
            HarthActiveReplayAction {
                prediction_label_code: self.prediction,
                confidence_ppm: self.confidence,
                next_sensor_mask: self.mask,
                next_wake_after_rows: self.interval,
                declared_compute_cycles: 10,
                declared_transmit_bytes: 2,
                declared_memory_write_bytes: 1,
            }
        }
    }

    fn rows() -> Vec<HarthRow> {
        let labels = [
            HarthActivity::Walking,
            HarthActivity::Walking,
            HarthActivity::Running,
            HarthActivity::Running,
            HarthActivity::Walking,
            HarthActivity::Walking,
        ];
        labels.iter().enumerate().map(|(index, activity)| HarthRow {
            timestamp_ns: 1_000_000_000 + index as i128 * 20_000_000
                + if index >= 3 { 20_000_000 } else { 0 },
            acceleration_g: [
                index as f32,
                0.1,
                1.0,
                -(index as f32),
                0.2,
                0.9,
            ],
            activity: *activity,
            row_index: index as u64,
        }).collect()
    }

    fn candidate() -> HarthReplayCandidateIdentity {
        HarthReplayCandidateIdentity {
            policy_id: "test-policy".to_owned(),
            source_revision: "test-rev".to_owned(),
            dependency_lock_sha256: [7; 32],
            artifact_sha256: [1; 32],
            configuration_sha256: [2; 32],
        }
    }

    fn run(policy: &mut FixedPolicy) -> HarthActiveReplayReceipt {
        run_harth_active_replay(
            &rows(),
            [3; 32],
            [4; 32],
            [5; 32],
            [6; 32],
            HarthLabelTrack::Native12,
            candidate(),
            policy,
        ).unwrap()
    }

    #[test]
    fn deterministic_replay_reproduces_same_receipt() {
        let mut first = FixedPolicy {
            prediction: Some(1), mask: HarthSensorMask::Both,
            interval: 1, confidence: 900_000, seen: Vec::new(),
        };
        let mut second = FixedPolicy {
            prediction: Some(1), mask: HarthSensorMask::Both,
            interval: 1, confidence: 900_000, seen: Vec::new(),
        };
        assert_eq!(run(&mut first), run(&mut second));
    }

    #[test]
    fn back_only_requests_do_not_reveal_new_thigh_rows() {
        let mut policy = FixedPolicy {
            prediction: Some(1), mask: HarthSensorMask::Back,
            interval: 1, confidence: 800_000, seen: Vec::new(),
        };
        let _ = run(&mut policy);
        assert!(policy.seen[0].latest_back_sample.is_some());
        assert!(policy.seen[0].latest_thigh_sample.is_some()); // common bootstrap
        assert!(policy.seen[1].latest_back_sample.is_some());
        assert_eq!(policy.seen[1].latest_thigh_sample.unwrap().row_index, 0);
        assert_eq!(policy.seen[1].latest_back_sample.unwrap().row_index, 1);
        assert_eq!(policy.seen[2].latest_thigh_sample.unwrap().row_index, 0);
    }

    #[test]
    fn sleeping_increases_age_without_revealing_skipped_values() {
        let mut policy = FixedPolicy {
            prediction: Some(1), mask: HarthSensorMask::None,
            interval: 2, confidence: 900_000, seen: Vec::new(),
        };
        let receipt = run(&mut policy);
        assert_eq!(policy.seen[1].latest_back_sample.unwrap().row_index, 0);
        assert!(policy.seen[1].wake_timestamp_ns > policy.seen[1].latest_back_sample.unwrap().timestamp_ns);
        assert_eq!(receipt.back_sample_count, 1);
        assert_eq!(receipt.thigh_sample_count, 1);
        assert_eq!(receipt.sample_acquisition_count, 1);
    }

    #[test]
    fn abstention_is_counted_as_uncovered_and_as_a_false_negative() {
        let mut policy = FixedPolicy {
            prediction: None, mask: HarthSensorMask::None,
            interval: 1, confidence: 0, seen: Vec::new(),
        };
        let receipt = run(&mut policy);
        assert_eq!(receipt.covered_rows, 0);
        assert_eq!(receipt.abstained_rows, receipt.row_count);
        assert_eq!(receipt.coverage_ppm, 0);
        assert_eq!(receipt.accuracy_ppm, 0);
        assert!(receipt.confusion_counts.iter().any(|count| *count > 0));
        validate_harth_active_replay_receipt(&receipt).unwrap();
    }

    #[test]
    fn invalid_prediction_and_wake_bounds_fail_closed() {
        let mut invalid_label = FixedPolicy {
            prediction: Some(999), mask: HarthSensorMask::None,
            interval: 1, confidence: 100, seen: Vec::new(),
        };
        assert_eq!(
            run_harth_active_replay(
                &rows(), [3; 32], [4; 32], [5; 32], [6; 32],
                HarthLabelTrack::Native12, candidate(), &mut invalid_label
            ).unwrap_err().field,
            "action.prediction_label_code"
        );

        let mut invalid_wake = FixedPolicy {
            prediction: Some(1), mask: HarthSensorMask::None,
            interval: 0, confidence: 100, seen: Vec::new(),
        };
        assert_eq!(
            run_harth_active_replay(
                &rows(), [3; 32], [4; 32], [5; 32], [6; 32],
                HarthLabelTrack::Native12, candidate(), &mut invalid_wake
            ).unwrap_err().field,
            "action.next_wake_after_rows"
        );
    }

    #[test]
    fn participant_row_indices_and_timestamps_are_validated() {
        let mut invalid = rows();
        invalid[2].row_index = 7;
        let mut policy = FixedPolicy {
            prediction: Some(1), mask: HarthSensorMask::None,
            interval: 1, confidence: 100, seen: Vec::new(),
        };
        assert_eq!(
            run_harth_active_replay(
                &invalid, [3; 32], [4; 32], [5; 32], [6; 32],
                HarthLabelTrack::Native12, candidate(), &mut policy
            ).unwrap_err().field,
            "rows[2].row_index"
        );
    }

    #[test]
    fn transition_detection_and_false_transition_denominators_are_retained() {
        let mut policy = FixedPolicy {
            prediction: Some(1), mask: HarthSensorMask::None,
            interval: 1, confidence: 900_000, seen: Vec::new(),
        };
        let receipt = run(&mut policy);
        assert_eq!(receipt.transitions.true_transition_count, 2);
        assert_eq!(receipt.transitions.detected_transition_count, 1);
        assert_eq!(receipt.transitions.missed_transition_count, 1);
        assert!(receipt.transitions.false_transition_alarm_count <= receipt.row_count);
    }

    #[test]
    fn timestamp_gaps_are_reflected_in_actual_wake_intervals() {
        let mut policy = FixedPolicy {
            prediction: Some(1), mask: HarthSensorMask::Back,
            interval: 1, confidence: 700_000, seen: Vec::new(),
        };
        let receipt = run(&mut policy);
        assert_eq!(receipt.wake_interval_max_ns, Some(40_000_000));
        assert_eq!(receipt.back_sample_count, receipt.row_count);
    }

    #[test]
    fn result_tampering_is_detected() {
        let mut policy = FixedPolicy {
            prediction: Some(1), mask: HarthSensorMask::Both,
            interval: 1, confidence: 900_000, seen: Vec::new(),
        };
        let mut receipt = run(&mut policy);
        receipt.abstained_rows += 1;
        assert!(validate_harth_active_replay_receipt(&receipt).is_err());
    }
}
