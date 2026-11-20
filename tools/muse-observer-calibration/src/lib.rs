//! Leakage-resistant calibration and selective-risk evaluation for MUSE acoustic evidence.
//!
//! Raw spectral salience is not a probability. This module fits a monotone
//! isotonic mapping on one group-disjoint calibration split, selects a
//! conservative positive-support threshold on a second split, and reports
//! selective metrics on a third untouched split. Unsupported measurements
//! never become negative or positive classifications: they abstain.
//!
//! This tool calibrates a supplied feature and provenance-bound labels. It
//! does not independently replay WAV/MIDI observers; ingestion must first
//! replay upstream #11617 receipts against their exact WAV/MIDI inputs.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;

pub const DATASET_SCHEMA_V1: &str = "muse-observer-calibration-dataset-v1";
pub const CONTRACT_SCHEMA_V1: &str = "muse-observer-calibration-contract-v1";
pub const BUNDLE_SCHEMA_V1: &str = "muse-observer-calibration-bundle-v1";
pub const CALIBRATOR_ID_V1: &str = "muse-isotonic-salience-calibrator";
pub const CALIBRATOR_VERSION_V1: &str = "0.1.0";
const DATASET_DOMAIN_V1: &[u8] = b"luminous-dynamics-muse-observer-calibration-dataset-v1";
const BUNDLE_DOMAIN_V1: &[u8] = b"luminous-dynamics-muse-observer-calibration-bundle-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitV1 {
    Fit,
    ThresholdSelection,
    HeldOut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementStateV1 {
    Measured,
    NoMeasurableSpectralEnergy,
    InsufficientDuration,
    TruncatedByAudio,
    FundamentalAboveNyquist,
    InsufficientFrequencyResolution,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKindV1 {
    SyntheticKnownAnswer,
    MeasuredRendererAdjudicated,
    HumanAdjudicated,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationObservationV1 {
    pub sample_id: String,
    /// Must identify the full work/recording lineage, not a random note excerpt.
    pub source_group_id: String,
    pub split: SplitV1,
    pub observer_id: String,
    pub observer_version: String,
    pub feature_id: String,
    pub measurement_state: MeasurementStateV1,
    /// The raw feature is intentionally not called confidence or probability.
    /// For the first adapter this is fundamental peak-to-sideband salience dB.
    pub raw_score_db: Option<f64>,
    pub event_present: bool,
    /// Required for absent-event cases, e.g. octave_collision, missing_note,
    /// wrong_part, noise_only or cross_part_contamination.
    pub negative_family: Option<String>,
    pub reference_kind: ReferenceKindV1,
    pub source_audio_sha256: String,
    pub source_midi_sha256: String,
    pub observation_receipt_sha256: String,
    pub reference_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationDatasetV1 {
    pub schema: String,
    pub dataset_id: String,
    pub observer_id: String,
    pub observer_version: String,
    pub feature_id: String,
    pub annotation_protocol_sha256: String,
    pub rights_clearance_receipt_sha256: String,
    pub observations: Vec<CalibrationObservationV1>,
    pub dataset_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationContractV1 {
    pub schema: String,
    pub feature_id: String,
    pub minimum_fit_examples_per_class: u32,
    pub minimum_threshold_examples_per_class: u32,
    pub minimum_held_out_examples_per_class: u32,
    /// Positive support may be emitted only under this false-support ceiling.
    pub maximum_false_support_rate: f64,
    pub ece_bins: u16,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IsotonicBlockV1 {
    pub lower_score_db: f64,
    pub upper_score_db: f64,
    pub positive_count: u64,
    pub total_count: u64,
    pub probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IsotonicModelV1 {
    pub schema: String,
    pub calibrator_id: String,
    pub calibrator_version: String,
    pub feature_id: String,
    pub fit_dataset_sha256: String,
    pub fit_example_count: u64,
    pub fit_positive_count: u64,
    pub blocks: Vec<IsotonicBlockV1>,
    pub model_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThresholdSelectionV1 {
    pub target_maximum_false_support_rate: f64,
    pub selected_support_threshold: Option<f64>,
    pub threshold_examples: u64,
    pub threshold_positive_count: u64,
    pub threshold_negative_count: u64,
    pub supported_count: u64,
    pub false_support_count: u64,
    pub false_support_rate: Option<f64>,
    pub true_support_rate: Option<f64>,
    pub selection_status: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeldOutMetricsV1 {
    pub total_examples: u64,
    pub measured_examples: u64,
    pub unmeasured_examples: u64,
    pub positive_examples: u64,
    pub negative_examples: u64,
    pub brier_score: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub roc_auc: Option<f64>,
    pub average_precision: Option<f64>,
    pub supported_examples: u64,
    pub abstained_examples: u64,
    pub true_support_count: u64,
    pub false_support_count: u64,
    pub support_coverage: f64,
    pub measurement_coverage: f64,
    pub abstention_rate: f64,
    pub support_precision: Option<f64>,
    pub false_support_rate_all_negatives: Option<f64>,
    pub false_support_rate_measured_negatives: Option<f64>,
    pub event_recall: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationBundleV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub contract_sha256: String,
    pub model: IsotonicModelV1,
    pub threshold_selection: ThresholdSelectionV1,
    pub held_out: HeldOutMetricsV1,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug)]
struct PooledPoint {
    lower: f64,
    upper: f64,
    positive: u64,
    total: u64,
}

impl PooledPoint {
    fn probability(&self) -> f64 {
        self.positive as f64 / self.total as f64
    }
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, String> {
    fn normalize(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let mut keys: Vec<&String> = object.keys().collect();
                keys.sort_unstable();
                let mut sorted = Map::new();
                for key in keys {
                    let child = object.get(key).expect("key came from this object");
                    sorted.insert(key.clone(), normalize(child));
                }
                Value::Object(sorted)
            }
            Value::Array(items) => Value::Array(items.iter().map(normalize).collect()),
            _ => value.clone(),
        }
    }
    serde_json::to_vec(&normalize(value))
        .map_err(|error| format!("canonical JSON serialization failed: {error}"))
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update([0]);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn dataset_digest(dataset: &CalibrationDatasetV1) -> Result<String, String> {
    let mut value = serde_json::to_value(dataset)
        .map_err(|error| format!("calibration dataset serialization failed: {error}"))?;
    value["dataset_sha256"] = Value::String(String::new());
    Ok(domain_digest(DATASET_DOMAIN_V1, &canonical_json_bytes(&value)?))
}

fn contract_digest(contract: &CalibrationContractV1) -> Result<String, String> {
    let value = serde_json::to_value(contract)
        .map_err(|error| format!("calibration contract serialization failed: {error}"))?;
    Ok(domain_digest(DATASET_DOMAIN_V1, &canonical_json_bytes(&value)?))
}

/// Seal a dataset after the producer has populated its evidence/label records.
/// The seal is tamper-evident, not a signature or proof that upstream receipts
/// were replayed. The calibration workflow must admit only replay-verified rows.
pub fn seal_calibration_dataset_v1(
    dataset: &mut CalibrationDatasetV1,
) -> Result<(), String> {
    dataset.dataset_sha256.clear();
    dataset.dataset_sha256 = dataset_digest(dataset)?;
    validate_dataset_v1(dataset)
}

pub fn validate_dataset_v1(dataset: &CalibrationDatasetV1) -> Result<(), String> {
    if dataset.schema != DATASET_SCHEMA_V1
        || dataset.dataset_id.trim().is_empty()
        || dataset.observer_id.trim().is_empty()
        || dataset.observer_version.trim().is_empty()
        || dataset.feature_id.trim().is_empty()
        || dataset.observations.is_empty()
        || dataset.observations.len() > 2_000_000
    {
        return Err("dataset schema, identity, feature or observation count is invalid".into());
    }
    for (field, value) in [
        ("annotation_protocol_sha256", dataset.annotation_protocol_sha256.as_str()),
        ("rights_clearance_receipt_sha256", dataset.rights_clearance_receipt_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            return Err(format!("{field} is not lowercase SHA-256"));
        }
    }

    let mut sample_ids = std::collections::BTreeSet::new();
    let mut group_splits = std::collections::BTreeMap::new();
    let mut audio_splits = std::collections::BTreeMap::new();
    let mut midi_splits = std::collections::BTreeMap::new();
    for (index, observation) in dataset.observations.iter().enumerate() {
        if observation.sample_id.trim().is_empty()
            || observation.source_group_id.trim().is_empty()
            || observation.observer_id != dataset.observer_id
            || observation.observer_version != dataset.observer_version
            || observation.feature_id != dataset.feature_id
        {
            return Err(format!("observations[{index}] identity does not match the dataset"));
        }
        if !sample_ids.insert(observation.sample_id.as_str()) {
            return Err(format!("duplicate sample_id {}", observation.sample_id));
        }
        for (field, value) in [
            ("source_audio_sha256", observation.source_audio_sha256.as_str()),
            ("source_midi_sha256", observation.source_midi_sha256.as_str()),
            ("observation_receipt_sha256", observation.observation_receipt_sha256.as_str()),
            ("reference_receipt_sha256", observation.reference_receipt_sha256.as_str()),
        ] {
            if !is_sha256(value) {
                return Err(format!("observations[{index}].{field} is not lowercase SHA-256"));
            }
        }
        match observation.measurement_state {
            MeasurementStateV1::Measured => {
                let score = observation.raw_score_db
                    .ok_or_else(|| format!("observations[{index}] measured row has no raw score"))?;
                if !score.is_finite() || !(-300.0..=120.0).contains(&score) {
                    return Err(format!("observations[{index}] raw score is outside [-300, 120] dB"));
                }
            }
            _ if observation.raw_score_db.is_some() => {
                return Err(format!("observations[{index}] non-measured row must not carry a raw score"));
            }
            _ => {}
        }
        if !observation.event_present
            && observation.negative_family.as_deref().is_none_or(|value| value.trim().is_empty())
        {
            return Err(format!("observations[{index}] absent-event label requires negative_family"));
        }
        if observation.event_present && observation.negative_family.is_some() {
            return Err(format!("observations[{index}] present-event label must not carry negative_family"));
        }
        if let Some(split) = group_splits.insert(observation.source_group_id.clone(), observation.split) {
            if split != observation.split {
                return Err(format!(
                    "source_group_id {} leaks across dataset splits",
                    observation.source_group_id
                ));
            }
        }
        for (map, digest, label) in [
            (&mut audio_splits, observation.source_audio_sha256.as_str(), "source_audio"),
            (&mut midi_splits, observation.source_midi_sha256.as_str(), "source_midi"),
        ] {
            if let Some(split) = map.insert(digest.to_string(), observation.split) {
                if split != observation.split {
                    return Err(format!("{label} digest leaks across dataset splits"));
                }
            }
        }
    }
    if !is_sha256(&dataset.dataset_sha256) || dataset_digest(dataset)? != dataset.dataset_sha256 {
        return Err("calibration dataset commitment is invalid".into());
    }
    Ok(())
}

pub fn validate_contract_v1(contract: &CalibrationContractV1) -> Result<(), String> {
    if contract.schema != CONTRACT_SCHEMA_V1
        || contract.feature_id.trim().is_empty()
        || contract.minimum_fit_examples_per_class == 0
        || contract.minimum_threshold_examples_per_class == 0
        || contract.minimum_held_out_examples_per_class == 0
        || !contract.maximum_false_support_rate.is_finite()
        || !(0.0..=0.5).contains(&contract.maximum_false_support_rate)
        || !(2..=100).contains(&contract.ece_bins)
    {
        return Err("calibration contract has unsupported schema or invalid bounds".into());
    }
    Ok(())
}

fn count_class(
    rows: &[&CalibrationObservationV1],
    split: SplitV1,
    present: bool,
) -> usize {
    rows.iter()
        .filter(|row| row.split == split
            && row.measurement_state == MeasurementStateV1::Measured
            && row.event_present == present)
        .count()
}

fn fit_isotonic(rows: &[&CalibrationObservationV1]) -> Result<Vec<IsotonicBlockV1>, String> {
    let mut points = rows
        .iter()
        .filter(|row| row.measurement_state == MeasurementStateV1::Measured)
        .map(|row| {
            (
                row.raw_score_db.expect("validated measured observation"),
                row.event_present,
            )
        })
        .collect::<Vec<_>>();
    points.sort_by(|left, right| left.0.total_cmp(&right.0));
    if points.is_empty() {
        return Err("no measured observations are available for isotonic fitting".into());
    }

    let mut pooled: Vec<PooledPoint> = Vec::new();
    for (score, positive) in points {
        if let Some(last) = pooled.last_mut() {
            if last.upper == score {
                last.total += 1;
                if positive { last.positive += 1; }
                continue;
            }
        }
        pooled.push(PooledPoint {
            lower: score,
            upper: score,
            positive: u64::from(positive),
            total: 1,
        });
        while pooled.len() >= 2 {
            let n = pooled.len();
            if pooled[n - 2].probability() <= pooled[n - 1].probability() {
                break;
            }
            let right = pooled.pop().expect("two points exist");
            let left = pooled.pop().expect("two points exist");
            pooled.push(PooledPoint {
                lower: left.lower,
                upper: right.upper,
                positive: left.positive + right.positive,
                total: left.total + right.total,
            });
        }
    }

    Ok(pooled.into_iter().map(|block| IsotonicBlockV1 {
        lower_score_db: block.lower,
        upper_score_db: block.upper,
        positive_count: block.positive,
        total_count: block.total,
        probability: block.probability(),
    }).collect())
}

fn predict(model: &IsotonicModelV1, raw_score_db: f64) -> f64 {
    model.blocks.iter()
        .find(|block| raw_score_db <= block.upper_score_db)
        .or_else(|| model.blocks.last())
        .map(|block| block.probability)
        .unwrap_or(0.0)
}

fn select_support_threshold(
    model: &IsotonicModelV1,
    rows: &[&CalibrationObservationV1],
    maximum_false_support_rate: f64,
) -> ThresholdSelectionV1 {
    let measured = rows.iter()
        .filter(|row| row.measurement_state == MeasurementStateV1::Measured)
        .collect::<Vec<_>>();
    let threshold_negative_count = measured.iter().filter(|row| !row.event_present).count() as u64;
    let threshold_positive_count = measured.iter().filter(|row| row.event_present).count() as u64;
    let all_negative_count = rows.iter().filter(|row| !row.event_present).count() as u64;
    let mut candidates = measured.iter()
        .map(|row| predict(model, row.raw_score_db.expect("measured row has score")))
        .collect::<Vec<_>>();
    candidates.sort_by(f64::total_cmp);
    candidates.dedup_by(|left, right| *left == *right);

    let mut selected: Option<(f64, u64, u64)> = None;
    for threshold in candidates {
        let supported = measured.iter()
            .filter(|row| predict(model, row.raw_score_db.expect("measured row has score")) >= threshold)
            .collect::<Vec<_>>();
        let false_supports = supported.iter().filter(|row| !row.event_present).count() as u64;
        let false_rate = if all_negative_count == 0 {
            1.0
        } else {
            false_supports as f64 / all_negative_count as f64
        };
        if false_rate <= maximum_false_support_rate {
            selected = Some((threshold, supported.len() as u64, false_supports));
            break;
        }
    }
    match selected {
        Some((threshold, supported_count, false_support_count)) => {
            let true_support_count = supported_count - false_support_count;
            ThresholdSelectionV1 {
                target_maximum_false_support_rate: maximum_false_support_rate,
                selected_support_threshold: Some(threshold),
                threshold_examples: rows.len() as u64,
                threshold_positive_count,
                threshold_negative_count,
                supported_count,
                false_support_count,
                false_support_rate: (all_negative_count > 0).then_some(
                    false_support_count as f64 / all_negative_count as f64
                ),
                true_support_rate: (threshold_positive_count > 0).then_some(
                    true_support_count as f64 / threshold_positive_count as f64
                ),
                selection_status: "threshold_selected".into(),
            }
        }
        None => ThresholdSelectionV1 {
            target_maximum_false_support_rate: maximum_false_support_rate,
            selected_support_threshold: None,
            threshold_examples: rows.len() as u64,
            threshold_positive_count,
            threshold_negative_count,
            supported_count: 0,
            false_support_count: 0,
            false_support_rate: Some(0.0),
            true_support_rate: Some(0.0),
            selection_status: "no_safe_threshold_all_abstain".into(),
        }
    }
}

fn roc_auc(labels: &[bool], scores: &[f64]) -> Option<f64> {
    let positive = labels.iter().filter(|label| **label).count();
    let negative = labels.len().checked_sub(positive)?;
    if positive == 0 || negative == 0 || labels.len() != scores.len() {
        return None;
    }
    let mut pairs = scores.iter().copied().zip(labels.iter().copied()).collect::<Vec<_>>();
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut rank_sum_positive = 0.0;
    let mut index = 0;
    while index < pairs.len() {
        let mut end = index + 1;
        while end < pairs.len() && pairs[end].0 == pairs[index].0 { end += 1; }
        let average_rank = ((index + 1 + end) as f64) * 0.5;
        let positive_in_tie = pairs[index..end].iter().filter(|pair| pair.1).count();
        rank_sum_positive += average_rank * positive_in_tie as f64;
        index = end;
    }
    let p = positive as f64;
    let n = negative as f64;
    Some((rank_sum_positive - p * (p + 1.0) * 0.5) / (p * n))
}

fn average_precision(labels: &[bool], scores: &[f64]) -> Option<f64> {
    let positives = labels.iter().filter(|label| **label).count();
    if positives == 0 || labels.len() != scores.len() {
        return None;
    }
    let mut pairs = scores.iter().copied().zip(labels.iter().copied()).collect::<Vec<_>>();
    pairs.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut true_positives = 0_u64;
    let mut seen = 0_u64;
    let mut sum_precision = 0.0;
    let mut index = 0;
    while index < pairs.len() {
        let mut end = index + 1;
        while end < pairs.len() && pairs[end].0 == pairs[index].0 { end += 1; }
        let positives_in_tie = pairs[index..end].iter().filter(|pair| pair.1).count() as u64;
        seen += (end - index) as u64;
        true_positives += positives_in_tie;
        if positives_in_tie > 0 {
            sum_precision += (true_positives as f64 / seen as f64)
                * (positives_in_tie as f64 / positives as f64);
        }
        index = end;
    }
    Some(sum_precision)
}

fn held_out_metrics(
    model: &IsotonicModelV1,
    threshold: &ThresholdSelectionV1,
    rows: &[&CalibrationObservationV1],
    ece_bins: u16,
) -> HeldOutMetricsV1 {
    let total = rows.len() as u64;
    let measured = rows.iter()
        .filter(|row| row.measurement_state == MeasurementStateV1::Measured)
        .collect::<Vec<_>>();
    let unmeasured = total - measured.len() as u64;
    let positive_total = rows.iter().filter(|row| row.event_present).count() as u64;
    let negative_total = total - positive_total;
    let labels = measured.iter().map(|row| row.event_present).collect::<Vec<_>>();
    let probs = measured.iter()
        .map(|row| predict(model, row.raw_score_db.expect("measured row has score")))
        .collect::<Vec<_>>();
    let brier_score = (!probs.is_empty()).then(|| {
        probs.iter().zip(&labels)
            .map(|(p, label)| (p - if *label { 1.0 } else { 0.0 }).powi(2))
            .sum::<f64>() / probs.len() as f64
    });
    let ece = if probs.is_empty() { None } else {
        let mut weighted = 0.0;
        for bin in 0..ece_bins {
            let lower = f64::from(bin) / f64::from(ece_bins);
            let upper = f64::from(bin + 1) / f64::from(ece_bins);
            let indexes = probs.iter().enumerate().filter(|(_, p)| {
                **p >= lower && (if bin + 1 == ece_bins { **p <= upper } else { **p < upper })
            }).map(|(i, _)| i).collect::<Vec<_>>();
            if indexes.is_empty() { continue; }
            let confidence = indexes.iter().map(|i| probs[*i]).sum::<f64>() / indexes.len() as f64;
            let accuracy = indexes.iter().filter(|i| labels[**i]).count() as f64 / indexes.len() as f64;
            weighted += indexes.len() as f64 / probs.len() as f64 * (confidence - accuracy).abs();
        }
        Some(weighted)
    };
    let threshold_value = threshold.selected_support_threshold;
    let supported = measured.iter().zip(&probs)
        .filter(|(_, probability)| threshold_value.is_some_and(|t| **probability >= t))
        .map(|(row, _)| *row)
        .collect::<Vec<_>>();
    let true_support = supported.iter().filter(|row| row.event_present).count() as u64;
    let false_support = supported.len() as u64 - true_support;
    let measured_negatives = measured.iter().filter(|row| !row.event_present).count() as u64;
    let raw_scores = probs.clone();

    HeldOutMetricsV1 {
        total_examples: total,
        measured_examples: measured.len() as u64,
        unmeasured_examples: unmeasured,
        positive_examples: positive_total,
        negative_examples: negative_total,
        brier_score,
        expected_calibration_error: ece,
        roc_auc: roc_auc(&labels, &raw_scores),
        average_precision: average_precision(&labels, &raw_scores),
        supported_examples: supported.len() as u64,
        abstained_examples: total - supported.len() as u64,
        true_support_count: true_support,
        false_support_count: false_support,
        support_coverage: supported.len() as f64 / total.max(1) as f64,
        measurement_coverage: measured.len() as f64 / total.max(1) as f64,
        abstention_rate: (total - supported.len() as u64) as f64 / total.max(1) as f64,
        support_precision: (!supported.is_empty()).then_some(true_support as f64 / supported.len() as f64),
        false_support_rate_all_negatives: (negative_total > 0).then_some(false_support as f64 / negative_total as f64),
        false_support_rate_measured_negatives: (measured_negatives > 0).then_some(false_support as f64 / measured_negatives as f64),
        event_recall: (positive_total > 0).then_some(true_support as f64 / positive_total as f64),
    }
}

/// Fit the calibrator, choose a conservative support threshold on a separate
/// group-disjoint tuning split, and evaluate once on the untouched held-out
/// split. This emits evidence, not a universal pass threshold.
pub fn run_calibration_v1(
    dataset: &CalibrationDatasetV1,
    contract: &CalibrationContractV1,
) -> Result<CalibrationBundleV1, String> {
    validate_dataset_v1(dataset)?;
    validate_contract_v1(contract)?;
    if dataset.feature_id != contract.feature_id {
        return Err("dataset feature_id does not match calibration contract".into());
    }
    let fit_rows = dataset.observations.iter()
        .filter(|row| row.split == SplitV1::Fit)
        .collect::<Vec<_>>();
    let threshold_rows = dataset.observations.iter()
        .filter(|row| row.split == SplitV1::ThresholdSelection)
        .collect::<Vec<_>>();
    let held_out_rows = dataset.observations.iter()
        .filter(|row| row.split == SplitV1::HeldOut)
        .collect::<Vec<_>>();
    for (split_name, rows, minimum) in [
        ("fit", fit_rows.as_slice(), contract.minimum_fit_examples_per_class),
        ("threshold selection", threshold_rows.as_slice(), contract.minimum_threshold_examples_per_class),
        ("held-out", held_out_rows.as_slice(), contract.minimum_held_out_examples_per_class),
    ] {
        let positive = count_class(rows, match split_name {
            "fit" => SplitV1::Fit,
            "threshold selection" => SplitV1::ThresholdSelection,
            _ => SplitV1::HeldOut,
        }, true);
        let negative = count_class(rows, match split_name {
            "fit" => SplitV1::Fit,
            "threshold selection" => SplitV1::ThresholdSelection,
            _ => SplitV1::HeldOut,
        }, false);
        if positive < minimum as usize || negative < minimum as usize {
            return Err(format!("{split_name} measured rows must contain at least {minimum} examples per class"));
        }
    }
    let blocks = fit_isotonic(&fit_rows)?;
    let fit_positive = count_class(&fit_rows, SplitV1::Fit, true) as u64;
    let model_value = json!({
        "schema": "muse-isotonic-calibration-model-v1",
        "calibrator_id": CALIBRATOR_ID_V1,
        "calibrator_version": CALIBRATOR_VERSION_V1,
        "feature_id": contract.feature_id,
        "fit_dataset_sha256": dataset.dataset_sha256,
        "fit_example_count": fit_rows.len(),
        "fit_positive_count": fit_positive,
        "blocks": blocks
    });
    let model_sha = domain_digest(BUNDLE_DOMAIN_V1, &canonical_json_bytes(&model_value)?);
    let model: IsotonicModelV1 = serde_json::from_value({
        let mut value = model_value;
        value["model_sha256"] = Value::String(model_sha);
        value
    }).map_err(|error| format!("calibration model serialization failed: {error}"))?;
    let threshold = select_support_threshold(
        &model,
        &threshold_rows,
        contract.maximum_false_support_rate,
    );
    let held_out = held_out_metrics(&model, &threshold, &held_out_rows, contract.ece_bins);
    let mut bundle = CalibrationBundleV1 {
        schema: BUNDLE_SCHEMA_V1.into(),
        dataset_sha256: dataset.dataset_sha256.clone(),
        contract_sha256: contract_digest(contract)?,
        model,
        threshold_selection: threshold,
        held_out,
        receipt_sha256: String::new(),
    };
    let value = serde_json::to_value(&bundle)
        .map_err(|error| format!("calibration bundle serialization failed: {error}"))?;
    bundle.receipt_sha256 = domain_digest(BUNDLE_DOMAIN_V1, &canonical_json_bytes(&value)?);
    Ok(bundle)
}

/// Recompute the complete model, threshold and held-out metrics from the exact
/// sealed dataset/contract and compare them with the supplied calibration bundle.
pub fn verify_calibration_bundle_v1(
    supplied: &CalibrationBundleV1,
    dataset: &CalibrationDatasetV1,
    contract: &CalibrationContractV1,
) -> Result<(), String> {
    let expected = run_calibration_v1(dataset, contract)?;
    if &expected != supplied {
        return Err("calibration bundle differs from deterministic dataset/contract replay".into());
    }
    let mut unsealed = supplied.clone();
    unsealed.receipt_sha256.clear();
    let value = serde_json::to_value(&unsealed)
        .map_err(|error| format!("calibration receipt serialization failed: {error}"))?;
    let actual = domain_digest(BUNDLE_DOMAIN_V1, &canonical_json_bytes(&value)?);
    if actual != supplied.receipt_sha256 {
        return Err("calibration bundle receipt seal mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: usize) -> String {
        format!("{seed:064x}")
    }

    fn row(
        index: usize,
        split: SplitV1,
        present: bool,
        score: Option<f64>,
        state: MeasurementStateV1,
    ) -> CalibrationObservationV1 {
        let split_base = match split {
            SplitV1::Fit => 10_000,
            SplitV1::ThresholdSelection => 20_000,
            SplitV1::HeldOut => 30_000,
        };
        let group_number = index / 2;
        CalibrationObservationV1 {
            sample_id: format!("sample-{index}"),
            source_group_id: format!("work-{}-{split:?}", index / 2),
            split,
            observer_id: "muse-score-audio-pitch-observer".into(),
            observer_version: "0.1.0".into(),
            feature_id: "fundamental_salience_db".into(),
            measurement_state: state,
            raw_score_db: score,
            event_present: present,
            negative_family: (!present).then_some("octave_or_missing_fundamental".into()),
            reference_kind: ReferenceKindV1::SyntheticKnownAnswer,
            source_audio_sha256: digest(split_base + group_number),
            source_midi_sha256: digest(40_000 + split_base + group_number),
            observation_receipt_sha256: digest(50_000 + index),
            reference_receipt_sha256: digest(60_000 + split_base + group_number),
        }
    }

    fn dataset() -> CalibrationDatasetV1 {
        let mut rows = Vec::new();
        for (base, split) in [
            (0usize, SplitV1::Fit),
            (20usize, SplitV1::ThresholdSelection),
            (40usize, SplitV1::HeldOut),
        ] {
            for offset in 0..12 {
                let present = offset >= 6;
                let score = if present { 8.0 + offset as f64 } else { -24.0 + offset as f64 };
                rows.push(row(
                    base + offset,
                    split,
                    present,
                    Some(score),
                    MeasurementStateV1::Measured,
                ));
            }
            rows.push(row(
                base + 12,
                split,
                true,
                None,
                MeasurementStateV1::InsufficientFrequencyResolution,
            ));
        }
        let mut dataset = CalibrationDatasetV1 {
            schema: DATASET_SCHEMA_V1.into(),
            dataset_id: "synthetic-isotonic-v1".into(),
            observer_id: "muse-score-audio-pitch-observer".into(),
            observer_version: "0.1.0".into(),
            feature_id: "fundamental_salience_db".into(),
            annotation_protocol_sha256: digest(1),
            rights_clearance_receipt_sha256: digest(2),
            observations: rows,
            dataset_sha256: String::new(),
        };
        seal_calibration_dataset_v1(&mut dataset).unwrap();
        dataset
    }

    fn contract() -> CalibrationContractV1 {
        CalibrationContractV1 {
            schema: CONTRACT_SCHEMA_V1.into(),
            feature_id: "fundamental_salience_db".into(),
            minimum_fit_examples_per_class: 5,
            minimum_threshold_examples_per_class: 5,
            minimum_held_out_examples_per_class: 5,
            maximum_false_support_rate: 0.01,
            ece_bins: 10,
        }
    }

    #[test]
    fn isotonic_model_is_monotone_and_threshold_is_conservative() {
        let dataset = dataset();
        let bundle = run_calibration_v1(&dataset, &contract()).unwrap();
        assert!(bundle.model.blocks.windows(2).all(|pair| pair[0].probability <= pair[1].probability));
        assert_eq!(bundle.threshold_selection.selection_status, "threshold_selected");
        assert!(bundle.held_out.false_support_rate_all_negatives.unwrap() <= 0.01);
        assert!(bundle.held_out.abstention_rate >= 0.0);
    }

    #[test]
    fn unmeasured_observations_are_always_abstentions() {
        let dataset = dataset();
        let bundle = run_calibration_v1(&dataset, &contract()).unwrap();
        assert_eq!(bundle.held_out.unmeasured_examples, 1);
        assert_eq!(bundle.held_out.abstained_examples, 7);
    }

    #[test]
    fn no_safe_threshold_results_in_all_abstain() {
        let mut data = dataset();
        for observation in data.observations.iter_mut().filter(|row| row.split == SplitV1::ThresholdSelection && row.measurement_state == MeasurementStateV1::Measured) {
            observation.raw_score_db = Some(0.0);
        }
        seal_calibration_dataset_v1(&mut data).unwrap();
        let bundle = run_calibration_v1(&data, &contract()).unwrap();
        assert_eq!(bundle.threshold_selection.selection_status, "no_safe_threshold_all_abstain");
        assert_eq!(bundle.held_out.supported_examples, 0);
    }

    #[test]
    fn group_leakage_across_splits_is_rejected() {
        let mut data = dataset();
        data.observations[0].source_group_id = data.observations[20].source_group_id.clone();
        seal_calibration_dataset_v1(&mut data).unwrap_err();
    }

    #[test]
    fn duplicate_case_ids_are_rejected() {
        let mut data = dataset();
        data.observations[1].sample_id = data.observations[0].sample_id.clone();
        seal_calibration_dataset_v1(&mut data).unwrap_err();
    }

    #[test]
    fn measured_rows_require_finite_raw_scores() {
        let mut data = dataset();
        data.observations[0].raw_score_db = Some(f64::NAN);
        data.dataset_sha256.clear();
        data.dataset_sha256 = dataset_digest(&data).unwrap();
        assert!(validate_dataset_v1(&data).is_err());
    }

    #[test]
    fn non_measured_rows_cannot_smuggle_a_score() {
        let mut data = dataset();
        let row = data.observations.iter_mut().find(|row| row.measurement_state != MeasurementStateV1::Measured).unwrap();
        row.raw_score_db = Some(7.0);
        data.dataset_sha256 = dataset_digest(&data).unwrap();
        assert!(validate_dataset_v1(&data).is_err());
    }

    #[test]
    fn exact_receipt_replay_detects_bundle_tampering() {
        let dataset = dataset();
        let mut bundle = run_calibration_v1(&dataset, &contract()).unwrap();
        verify_calibration_bundle_v1(&bundle, &dataset, &contract()).unwrap();
        bundle.held_out.brier_score = Some(0.999);
        assert!(verify_calibration_bundle_v1(&bundle, &dataset, &contract()).is_err());
    }

    #[test]
    fn test_rows_do_not_change_fitted_model() {
        let data = dataset();
        let initial = run_calibration_v1(&data, &contract()).unwrap();
        let mut changed = data.clone();
        let row = changed.observations.iter_mut().find(|row| row.split == SplitV1::HeldOut && row.measurement_state == MeasurementStateV1::Measured).unwrap();
        row.raw_score_db = Some(row.raw_score_db.unwrap() + 0.5);
        seal_calibration_dataset_v1(&mut changed).unwrap();
        let changed_bundle = run_calibration_v1(&changed, &contract()).unwrap();
        assert_eq!(initial.model, changed_bundle.model);
        assert_ne!(initial.held_out, changed_bundle.held_out);
    }
}
