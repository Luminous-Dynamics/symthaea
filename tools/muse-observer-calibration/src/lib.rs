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

pub const DATASET_SCHEMA_V1: &str = "muse-observer-calibration-dataset-v1";
pub const CONTRACT_SCHEMA_V1: &str = "muse-observer-calibration-contract-v1";
pub const BUNDLE_SCHEMA_V1: &str = "muse-observer-calibration-bundle-v1";
pub const CALIBRATOR_ID_V1: &str = "muse-isotonic-salience-calibrator";
pub const CALIBRATOR_VERSION_V1: &str = "0.1.0";
pub const MAX_DATASET_OBSERVATIONS_V1: usize = 2_000_000;
const DATASET_DOMAIN_V1: &[u8] = b"luminous-dynamics-muse-observer-calibration-dataset-v1";
const SPLIT_DOMAIN_V1: &[u8] = b"luminous-dynamics-muse-observer-calibration-split-v1";
const CONTRACT_DOMAIN_V1: &[u8] = b"luminous-dynamics-muse-observer-calibration-contract-v1";
const MODEL_DOMAIN_V1: &[u8] = b"luminous-dynamics-muse-observer-calibration-model-v1";
const BUNDLE_DOMAIN_V1: &[u8] = b"luminous-dynamics-muse-observer-calibration-bundle-v1";
const MAX_BOOTSTRAP_SCORING_ROWS_V1: usize = 100_000_000;

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
    pub minimum_held_out_groups: u16,
    pub bootstrap_replicates: u16,
    pub bootstrap_seed: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsotonicBlockV1 {
    pub lower_score_db: f64,
    pub upper_score_db: f64,
    pub positive_count: u64,
    pub total_count: u64,
    pub probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsotonicModelV1 {
    pub schema: String,
    pub calibrator_id: String,
    pub calibrator_version: String,
    pub feature_id: String,
    pub fit_dataset_sha256: String,
    pub fit_example_count: u64,
    pub fit_unmeasured_count: u64,
    pub fit_positive_count: u64,
    pub blocks: Vec<IsotonicBlockV1>,
    pub model_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdSelectionV1 {
    pub threshold_dataset_sha256: String,
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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapIntervalV1 {
    pub lower_95: Option<f64>,
    pub upper_95: Option<f64>,
    pub valid_replicates: u16,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NegativeFamilyMetricsV1 {
    pub negative_family: String,
    pub total_negative_examples: u64,
    pub measured_negative_examples: u64,
    pub false_support_count: u64,
    pub false_support_rate_all_negatives: Option<f64>,
    pub false_support_rate_measured_negatives: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterBootstrapMetricsV1 {
    pub support_precision: BootstrapIntervalV1,
    pub false_support_rate_all_negatives: BootstrapIntervalV1,
    pub false_support_rate_measured_negatives: BootstrapIntervalV1,
    pub event_recall: BootstrapIntervalV1,
    pub support_coverage: BootstrapIntervalV1,
    pub brier_score: BootstrapIntervalV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeldOutMetricsV1 {
    pub held_out_dataset_sha256: String,
    pub cluster_bootstrap: ClusterBootstrapMetricsV1,
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
    pub negative_family_metrics: Vec<NegativeFamilyMetricsV1>,
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
    Ok(domain_digest(SPLIT_DOMAIN_V1, &canonical_json_bytes(&value)?))
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
        || dataset.observations.len() > MAX_DATASET_OBSERVATIONS_V1
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
    let mut audio_lineage = std::collections::BTreeMap::new();
    let mut midi_lineage = std::collections::BTreeMap::new();
    for (index, observation) in dataset.observations.iter().enumerate() {
        if observation.sample_id.trim().is_empty()
            || observation.sample_id.len() > 256
            || observation.source_group_id.trim().is_empty()
            || observation.source_group_id.len() > 512
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
        if observation.negative_family.as_deref().is_some_and(|value| value.trim().is_empty() || value.len() > 128) {
            return Err(format!("observations[{index}] negative_family is empty or too long"));
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
            (&mut audio_lineage, observation.source_audio_sha256.as_str(), "source_audio"),
            (&mut midi_lineage, observation.source_midi_sha256.as_str(), "source_midi"),
        ] {
            let lineage = (observation.split, observation.source_group_id.clone());
            if let Some(previous) = map.insert(digest.to_string(), lineage.clone()) {
                if previous != lineage {
                    return Err(format!(
                        "{label} digest is assigned to multiple splits or source groups"
                    ));
                }
            }
        }
    }
    if !is_sha256(&dataset.dataset_sha256) || dataset_digest(dataset)? != dataset.dataset_sha256 {
        return Err("calibration dataset commitment is invalid".into());
    }
    Ok(())
}

fn split_digest(dataset: &CalibrationDatasetV1, split: SplitV1) -> Result<String, String> {
    let mut observations = dataset.observations.iter()
        .filter(|row| row.split == split)
        .collect::<Vec<_>>();
    observations.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    let value = json!({
        "schema": "muse-observer-calibration-split-v1",
        "dataset_id": dataset.dataset_id,
        "observer_id": dataset.observer_id,
        "observer_version": dataset.observer_version,
        "feature_id": dataset.feature_id,
        "annotation_protocol_sha256": dataset.annotation_protocol_sha256,
        "rights_clearance_receipt_sha256": dataset.rights_clearance_receipt_sha256,
        "split": split,
        "observations": observations
    });
    Ok(domain_digest(DATASET_DOMAIN_V1, &canonical_json_bytes(&value)?))
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
        || !(3..=10_000).contains(&contract.minimum_held_out_groups)
        || !(100..=5_000).contains(&contract.bootstrap_replicates)
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
    threshold_dataset_sha256: &str,
    maximum_false_support_rate: f64,
) -> ThresholdSelectionV1 {
    let measured = rows.iter()
        .filter(|row| row.measurement_state == MeasurementStateV1::Measured)
        .collect::<Vec<_>>();
    let threshold_negative_count = measured.iter().filter(|row| !row.event_present).count() as u64;
    let threshold_positive_count = measured.iter().filter(|row| row.event_present).count() as u64;
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
        // Threshold selection uses measured negatives only. Unmeasured
        // negatives remain abstentions and must not dilute the false-support
        // denominator.
        let false_rate = if threshold_negative_count == 0 {
            1.0
        } else {
            false_supports as f64 / threshold_negative_count as f64
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
                threshold_dataset_sha256: threshold_dataset_sha256.into(),
                target_maximum_false_support_rate: maximum_false_support_rate,
                selected_support_threshold: Some(threshold),
                threshold_examples: rows.len() as u64,
                threshold_positive_count,
                threshold_negative_count,
                supported_count,
                false_support_count,
                false_support_rate: (threshold_negative_count > 0).then_some(
                    false_support_count as f64 / threshold_negative_count as f64
                ),
                true_support_rate: (threshold_positive_count > 0).then_some(
                    true_support_count as f64 / threshold_positive_count as f64
                ),
                selection_status: "threshold_selected".into(),
            }
        }
        None => ThresholdSelectionV1 {
            threshold_dataset_sha256: threshold_dataset_sha256.into(),
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
    held_out_dataset_sha256: &str,
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
    let mut negative_family_rows = std::collections::BTreeMap::<String, Vec<&CalibrationObservationV1>>::new();
    for row in rows.iter().filter(|row| !row.event_present) {
        let family = row.negative_family.as_deref().unwrap_or("unclassified_negative");
        negative_family_rows.entry(family.to_string()).or_default().push(*row);
    }
    let negative_family_metrics = negative_family_rows.into_iter().map(|(family, family_rows)| {
        let family_measured = family_rows.iter()
            .filter(|row| row.measurement_state == MeasurementStateV1::Measured)
            .collect::<Vec<_>>();
        let family_false_support = family_measured.iter()
            .filter(|row| threshold_value.is_some_and(|t|
                predict(model, row.raw_score_db.expect("measured row has score")) >= t
            ))
            .count() as u64;
        NegativeFamilyMetricsV1 {
            negative_family: family,
            total_negative_examples: family_rows.len() as u64,
            measured_negative_examples: family_measured.len() as u64,
            false_support_count: family_false_support,
            false_support_rate_all_negatives: (!family_rows.is_empty()).then_some(
                family_false_support as f64 / family_rows.len() as f64
            ),
            false_support_rate_measured_negatives: (!family_measured.is_empty()).then_some(
                family_false_support as f64 / family_measured.len() as f64
            ),
        }
    }).collect::<Vec<_>>();
    HeldOutMetricsV1 {
        held_out_dataset_sha256: held_out_dataset_sha256.into(),
        cluster_bootstrap: ClusterBootstrapMetricsV1::default(),
        total_examples: total,
        measured_examples: measured.len() as u64,
        unmeasured_examples: unmeasured,
        positive_examples: positive_total,
        negative_examples: negative_total,
        brier_score,
        expected_calibration_error: ece,
        roc_auc: roc_auc(&labels, &probs),
        average_precision: average_precision(&labels, &probs),
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
        negative_family_metrics,
    }
}

#[derive(Clone, Debug)]
struct SplitMix64V1 {
    state: u64,
}

impl SplitMix64V1 {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }
}

fn percentile(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() { return None; }
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        Some(sorted[lower])
    } else {
        let weight = position - lower as f64;
        Some(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
    }
}

fn bootstrap_interval(mut samples: Vec<f64>) -> BootstrapIntervalV1 {
    if samples.is_empty() {
        return BootstrapIntervalV1::default();
    }
    samples.sort_by(f64::total_cmp);
    BootstrapIntervalV1 {
        lower_95: percentile(&samples, 0.025),
        upper_95: percentile(&samples, 0.975),
        valid_replicates: samples.len().min(u16::MAX as usize) as u16,
    }
}

fn cluster_bootstrap_metrics(
    model: &IsotonicModelV1,
    threshold: &ThresholdSelectionV1,
    rows: &[&CalibrationObservationV1],
    dataset_sha256: &str,
    contract: &CalibrationContractV1,
) -> Result<ClusterBootstrapMetricsV1, String> {
    let mut clusters = std::collections::BTreeMap::<String, Vec<&CalibrationObservationV1>>::new();
    for row in rows {
        clusters.entry(row.source_group_id.clone()).or_default().push(*row);
    }
    if clusters.len() < contract.minimum_held_out_groups as usize {
        return Err(format!(
            "held-out split has {} source groups; contract requires at least {}",
            clusters.len(), contract.minimum_held_out_groups
        ));
    }
    let cluster_rows = clusters.into_values().collect::<Vec<_>>();
    let largest_cluster = cluster_rows.iter().map(Vec::len).max().unwrap_or(0);
    let worst_case_resampled_rows = largest_cluster
        .checked_mul(cluster_rows.len())
        .and_then(|per_replicate| per_replicate.checked_mul(contract.bootstrap_replicates as usize))
        .unwrap_or(usize::MAX);
    if worst_case_resampled_rows > MAX_BOOTSTRAP_SCORING_ROWS_V1 {
        return Err("requested cluster bootstrap exceeds the bounded scoring-work budget".into());
    }
    let seed_material = format!("{}:{}", dataset_sha256, contract.bootstrap_seed);
    let seed_digest = domain_digest(BUNDLE_DOMAIN_V1, seed_material.as_bytes());
    let seed = u64::from_str_radix(&seed_digest[..16], 16)
        .map_err(|error| format!("bootstrap seed parse failed: {error}"))?;
    let mut rng = SplitMix64V1::new(seed);

    let mut support_precision = Vec::new();
    let mut false_support_all = Vec::new();
    let mut false_support_measured = Vec::new();
    let mut event_recall = Vec::new();
    let mut support_coverage = Vec::new();
    let mut brier = Vec::new();

    for _ in 0..contract.bootstrap_replicates {
        let mut sampled = Vec::new();
        for _ in 0..cluster_rows.len() {
            let index = (rng.next_u64() % cluster_rows.len() as u64) as usize;
            sampled.extend(cluster_rows[index].iter().copied());
        }
        let metrics = held_out_metrics(model, threshold, &sampled, dataset_sha256, contract.ece_bins);
        if let Some(value) = metrics.support_precision { support_precision.push(value); }
        if let Some(value) = metrics.false_support_rate_all_negatives { false_support_all.push(value); }
        if let Some(value) = metrics.false_support_rate_measured_negatives { false_support_measured.push(value); }
        if let Some(value) = metrics.event_recall { event_recall.push(value); }
        support_coverage.push(metrics.support_coverage);
        if let Some(value) = metrics.brier_score { brier.push(value); }
    }
    Ok(ClusterBootstrapMetricsV1 {
        support_precision: bootstrap_interval(support_precision),
        false_support_rate_all_negatives: bootstrap_interval(false_support_all),
        false_support_rate_measured_negatives: bootstrap_interval(false_support_measured),
        event_recall: bootstrap_interval(event_recall),
        support_coverage: bootstrap_interval(support_coverage),
        brier_score: bootstrap_interval(brier),
    })
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
    let fit_split_sha256 = split_digest(dataset, SplitV1::Fit)?;
    let threshold_split_sha256 = split_digest(dataset, SplitV1::ThresholdSelection)?;
    let held_out_split_sha256 = split_digest(dataset, SplitV1::HeldOut)?;
    let blocks = fit_isotonic(&fit_rows)?;
    let fit_positive = count_class(&fit_rows, SplitV1::Fit, true) as u64;
    let fit_measured_count = fit_rows.iter()
        .filter(|row| row.measurement_state == MeasurementStateV1::Measured)
        .count() as u64;
    let fit_unmeasured_count = fit_rows.len() as u64 - fit_measured_count;
    let model_value = json!({
        "schema": "muse-isotonic-calibration-model-v1",
        "calibrator_id": CALIBRATOR_ID_V1,
        "calibrator_version": CALIBRATOR_VERSION_V1,
        "feature_id": contract.feature_id,
        "fit_dataset_sha256": fit_split_sha256,
        "fit_example_count": fit_measured_count,
        "fit_unmeasured_count": fit_unmeasured_count,
        "fit_positive_count": fit_positive,
        "blocks": blocks
    });
    let model_sha = domain_digest(MODEL_DOMAIN_V1, &canonical_json_bytes(&model_value)?);
    let model: IsotonicModelV1 = serde_json::from_value({
        let mut value = model_value;
        value["model_sha256"] = Value::String(model_sha);
        value
    }).map_err(|error| format!("calibration model serialization failed: {error}"))?;
    let threshold = select_support_threshold(
        &model,
        &threshold_rows,
        &threshold_split_sha256,
        contract.maximum_false_support_rate,
    );
    let mut held_out = held_out_metrics(
        &model, &threshold, &held_out_rows, &held_out_split_sha256, contract.ece_bins
    );
    held_out.cluster_bootstrap = cluster_bootstrap_metrics(
        &model, &threshold, &held_out_rows, &held_out_split_sha256, contract
    )?;
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
            source_group_id: format!("work-{}", index / 2),
            split,
            observer_id: "muse-score-audio-pitch-observer".into(),
            observer_version: "0.1.0".into(),
            feature_id: "fundamental_salience_db".into(),
            measurement_state: state,
            raw_score_db: score,
            event_present: present,
            negative_family: (!present).then_some("octave_or_missing_fundamental".into()),
            reference_kind: ReferenceKindV1::SyntheticKnownAnswer,
            source_audio_sha256: digest(100_000 + split_base + group_number),
            source_midi_sha256: digest(200_000 + split_base + group_number),
            observation_receipt_sha256: digest(300_000 + index),
            reference_receipt_sha256: digest(400_000 + split_base + group_number),
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
            minimum_held_out_groups: 5,
            bootstrap_replicates: 200,
            bootstrap_seed: 12345,
        }
    }

    #[test]
    fn checked_in_synthetic_fixture_replays_and_keeps_splits_separate() {
        let dataset: CalibrationDatasetV1 = serde_json::from_str(include_str!(
            "../../../docs/research/fixtures/muse-observer-calibration-dataset-v1.synthetic.json"
        )).unwrap();
        let contract: CalibrationContractV1 = serde_json::from_str(include_str!(
            "../../../docs/research/fixtures/muse-observer-calibration-contract-v1.synthetic.json"
        )).unwrap();
        validate_dataset_v1(&dataset).unwrap();
        let bundle = run_calibration_v1(&dataset, &contract).unwrap();
        assert_eq!(bundle.threshold_selection.selection_status, "threshold_selected");
        assert_eq!(bundle.held_out.total_examples, 13);
        assert_eq!(bundle.held_out.unmeasured_examples, 1);
        verify_calibration_bundle_v1(&bundle, &dataset, &contract).unwrap();
    }

    #[test]
    fn isotonic_model_is_monotone_and_threshold_is_conservative() {
        let dataset = dataset();
        let bundle = run_calibration_v1(&dataset, &contract()).unwrap();
        assert!(bundle.model.blocks.windows(2).all(|pair| pair[0].probability <= pair[1].probability));
        assert_eq!(bundle.threshold_selection.selection_status, "threshold_selected");
        assert!(bundle.held_out.false_support_rate_all_negatives.unwrap() <= 0.01);
        assert!(bundle.held_out.abstention_rate >= 0.0);
        assert_eq!(bundle.model.fit_example_count, 12);
        assert_eq!(bundle.model.fit_unmeasured_count, 1);
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
    fn unmeasured_negative_cases_cannot_dilute_the_false_support_budget() {
        let mut data = dataset();
        for offset in 0..1_000 {
            data.observations.push(row(
                10_000 + offset,
                SplitV1::ThresholdSelection,
                false,
                None,
                MeasurementStateV1::InsufficientFrequencyResolution,
            ));
        }
        // All measured tuning rows receive the same calibrated score. The
        // measured false-support rate is therefore 50%, regardless of how
        // many unmeasured negative rows are appended.
        for observation in data.observations.iter_mut().filter(|row|
            row.split == SplitV1::ThresholdSelection
                && row.measurement_state == MeasurementStateV1::Measured
        ) {
            observation.raw_score_db = Some(0.0);
        }
        seal_calibration_dataset_v1(&mut data).unwrap();
        let bundle = run_calibration_v1(&data, &contract()).unwrap();
        assert_eq!(bundle.threshold_selection.selection_status, "no_safe_threshold_all_abstain");
        assert_eq!(bundle.threshold_selection.selected_support_threshold, None);
    }

    #[test]
    fn group_leakage_across_splits_is_rejected() {
        let mut data = dataset();
        data.observations[0].source_group_id = data.observations[20].source_group_id.clone();
        seal_calibration_dataset_v1(&mut data).unwrap_err();
    }

    #[test]
    fn identical_audio_cannot_be_split_into_multiple_source_groups() {
        let mut data = dataset();
        data.observations[0].source_audio_sha256 =
            data.observations[2].source_audio_sha256.clone();
        data.observations[0].source_group_id = "other-work".into();
        assert!(seal_calibration_dataset_v1(&mut data).is_err());
    }

    #[test]
    fn identical_midi_cannot_be_split_into_multiple_source_groups() {
        let mut data = dataset();
        data.observations[0].source_midi_sha256 =
            data.observations[2].source_midi_sha256.clone();
        data.observations[0].source_group_id = "other-work".into();
        assert!(seal_calibration_dataset_v1(&mut data).is_err());
    }

    #[test]
    fn repeated_source_artifacts_within_the_same_group_are_allowed() {
        let mut data = dataset();
        data.observations[1].source_audio_sha256 =
            data.observations[0].source_audio_sha256.clone();
        data.observations[1].source_midi_sha256 =
            data.observations[0].source_midi_sha256.clone();
        assert!(seal_calibration_dataset_v1(&mut data).is_ok());
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
        assert!(seal_calibration_dataset_v1(&mut data).is_err());
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
    fn held_out_false_support_is_reported_by_negative_family() {
        let mut data = dataset();
        let negatives = data.observations.iter_mut().filter(|row|
            row.split == SplitV1::HeldOut && !row.event_present
        ).collect::<Vec<_>>();
        for (index, row) in negatives.iter_mut().enumerate() {
            row.negative_family = Some(if index < 3 {
                "octave_collision".into()
            } else {
                "cross_part_contamination".into()
            });
        }
        // One held-out cross-part negative is deliberately assigned a
        // high-salience feature to exercise family-specific false support.
        negatives[3].raw_score_db = Some(16.0);
        seal_calibration_dataset_v1(&mut data).unwrap();
        let bundle = run_calibration_v1(&data, &contract()).unwrap();
        let metrics = &bundle.held_out.negative_family_metrics;
        assert_eq!(metrics.len(), 2);
        let octave = metrics.iter().find(|m| m.negative_family == "octave_collision").unwrap();
        let cross_part = metrics.iter().find(|m| m.negative_family == "cross_part_contamination").unwrap();
        assert_eq!(octave.false_support_count, 0);
        assert_eq!(cross_part.false_support_count, 1);
        assert_eq!(cross_part.measured_negative_examples, 3);
        assert!((cross_part.false_support_rate_measured_negatives.unwrap() - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn group_cluster_bootstrap_is_reproducible_and_reports_intervals() {
        let data = dataset();
        let first = run_calibration_v1(&data, &contract()).unwrap();
        let second = run_calibration_v1(&data, &contract()).unwrap();
        assert_eq!(first.held_out.cluster_bootstrap, second.held_out.cluster_bootstrap);
        assert!(first.held_out.cluster_bootstrap.support_coverage.lower_95.is_some());
        assert!(first.held_out.cluster_bootstrap.support_coverage.upper_95.is_some());
        assert!(first.held_out.cluster_bootstrap.support_coverage.valid_replicates > 0);
    }

    #[test]
    fn too_few_held_out_source_groups_are_rejected() {
        let mut case = contract();
        case.minimum_held_out_groups = 8;
        let error = run_calibration_v1(&dataset(), &case).unwrap_err();
        assert!(error.contains("source groups"));
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
