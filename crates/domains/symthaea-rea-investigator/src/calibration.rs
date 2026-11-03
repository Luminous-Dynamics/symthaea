//! Deterministic, scope-bound likelihood calibration evaluation.
//!
//! Predictions are evaluated against a held-out observed-outcome cohort. The
//! module computes multiclass Brier score, log loss, top-label ECE, reliability
//! bins, Wilson accuracy intervals and source-group cluster-bootstrap intervals.
//!
//! The resulting receipt is unsigned. It binds the supplied model identity,
//! target scope, source-group split, observations and report, but does not
//! authenticate the producer, labels, source groups or global calibration.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const DIGEST_DOMAIN_DATA: &[u8] = b"LUMINOUS-REA-CALIBRATION-DATA-V1\0";
const DIGEST_DOMAIN_REPORT: &[u8] = b"LUMINOUS-REA-CALIBRATION-REPORT-V1\0";
const DIGEST_DOMAIN_RECEIPT: &[u8] = b"LUMINOUS-REA-CALIBRATION-RECEIPT-V1\0";
pub const CALIBRATION_EVALUATOR_VERSION: &str = "rea-likelihood-calibration-v1";
pub const CALIBRATION_BOOTSTRAP_UNIT: &str = "source-group-cluster";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationErrorKind {
    InvalidModelIdentity,
    InvalidScope,
    InvalidTrainingSplit,
    TrainingCalibrationLeakage,
    InvalidObservation,
    DuplicateSampleId,
    InconsistentClassSet,
    InvalidProbabilityDistribution,
    InvalidConfiguration,
    InvalidThresholds,
    ReceiptDigestMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationError {
    pub kind: CalibrationErrorKind,
    pub message: String,
}

impl CalibrationError {
    fn new(kind: CalibrationErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for CalibrationError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationModelIdentity {
    pub model_id: String,
    pub model_version: String,
    /// SHA-256 of the model/provider implementation or immutable model artifact.
    pub model_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationScope {
    pub platform: String,
    pub target_build: String,
    pub fault_family: String,
    pub environment: String,
    pub observation_regime: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassProbability {
    pub class_id: String,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationObservation {
    pub sample_id: String,
    /// Digest of the source project/binary family. The bootstrap resamples this
    /// group as a cluster, not individual rows, to reduce near-duplicate leakage.
    pub source_group_sha256: String,
    pub predicted_probabilities: Vec<ClassProbability>,
    pub observed_class_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationConfig {
    pub confidence_bins: usize,
    pub bootstrap_iterations: usize,
    pub bootstrap_seed: u64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            confidence_bins: 10,
            bootstrap_iterations: 1_000,
            bootstrap_seed: 0x7265_612d_6361_6c31,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationThresholds {
    pub minimum_observations: usize,
    pub minimum_source_groups: usize,
    pub maximum_brier_score: f64,
    pub maximum_top_label_ece: f64,
    pub maximum_log_loss_nats: Option<f64>,
    pub maximum_zero_probability_count: usize,
}

impl Default for CalibrationThresholds {
    fn default() -> Self {
        Self {
            minimum_observations: 100,
            minimum_source_groups: 10,
            maximum_brier_score: 0.25,
            maximum_top_label_ece: 0.05,
            maximum_log_loss_nats: Some(0.5),
            maximum_zero_probability_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationEligibility {
    InsufficientSupport,
    ThresholdsNotMet,
    EligibleForIndependentReview,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReliabilityBin {
    pub index: usize,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub count: usize,
    pub mean_confidence: Option<f64>,
    pub observed_accuracy: Option<f64>,
    /// Wilson 95% interval for observed top-label accuracy in this bin.
    pub accuracy_wilson_95: Option<CalibrationInterval>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationMetrics {
    pub observation_count: usize,
    pub source_group_count: usize,
    /// Multiclass Brier score: mean sum_k (p_k - y_k)^2, range [0, 2].
    pub brier_score: f64,
    /// Mean negative log likelihood. None when any observed class had zero
    /// predicted probability; zero_probability_count records that event.
    pub mean_log_loss_nats: Option<f64>,
    pub zero_probability_count: usize,
    /// Top-label expected calibration error with uniform confidence bins.
    pub top_label_ece: f64,
    pub reliability_bins: Vec<ReliabilityBin>,
    pub bootstrap_unit: String,
    pub brier_bootstrap_95: CalibrationInterval,
    pub top_label_ece_bootstrap_95: CalibrationInterval,
    pub mean_log_loss_bootstrap_95: Option<CalibrationInterval>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationClaimCeiling {
    pub producer_authenticated: bool,
    pub source_labels_independently_verified: bool,
    pub training_calibration_split_independently_audited: bool,
    pub calibration_certified: bool,
    pub target_scope_completeness_verified: bool,
    pub distribution_stationarity_proven: bool,
    pub drift_checked: bool,
}

impl Default for CalibrationClaimCeiling {
    fn default() -> Self {
        Self {
            producer_authenticated: false,
            source_labels_independently_verified: false,
            training_calibration_split_independently_audited: false,
            calibration_certified: false,
            target_scope_completeness_verified: false,
            distribution_stationarity_proven: false,
            drift_checked: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationReceipt {
    pub schema: String,
    pub evaluator_version: String,
    pub model: CalibrationModelIdentity,
    pub scope: CalibrationScope,
    pub training_source_group_sha256s: Vec<String>,
    pub calibration_source_group_sha256s: Vec<String>,
    pub observations_sha256: String,
    pub report_sha256: String,
    pub receipt_sha256: String,
    pub config: CalibrationConfig,
    pub thresholds: CalibrationThresholds,
    pub metrics: CalibrationMetrics,
    pub eligibility: CalibrationEligibility,
    pub claim_ceiling: CalibrationClaimCeiling,
}

#[derive(Debug, Clone)]
struct NormalizedObservation {
    sample_id: String,
    source_group_sha256: String,
    probabilities: Vec<f64>,
    observed_class_index: usize,
    top_label_confidence: f64,
    top_label_correct: bool,
    brier_score: f64,
    log_loss_nats: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct ScoreSummary {
    brier_score: f64,
    mean_log_loss_nats: Option<f64>,
    zero_probability_count: usize,
    top_label_ece: f64,
}

fn validate_digest(value: &str, context: &str) -> Result<(), CalibrationError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(CalibrationError::new(
            CalibrationErrorKind::InvalidObservation,
            format!("{context} must be a 64-character lowercase SHA-256 digest"),
        ));
    }
    Ok(())
}

fn validate_model(model: &CalibrationModelIdentity) -> Result<(), CalibrationError> {
    if model.model_id.trim().is_empty() || model.model_version.trim().is_empty() {
        return Err(CalibrationError::new(
            CalibrationErrorKind::InvalidModelIdentity,
            "model ID and version must be non-empty",
        ));
    }
    validate_digest(&model.model_sha256, "model_sha256").map_err(|mut error| {
        error.kind = CalibrationErrorKind::InvalidModelIdentity;
        error
    })
}

fn validate_scope(scope: &CalibrationScope) -> Result<(), CalibrationError> {
    let values = [
        &scope.platform,
        &scope.target_build,
        &scope.fault_family,
        &scope.environment,
        &scope.observation_regime,
    ];
    if values.iter().any(|value| value.trim().is_empty()) {
        return Err(CalibrationError::new(
            CalibrationErrorKind::InvalidScope,
            "platform, target build, fault family, environment and observation regime are required",
        ));
    }
    Ok(())
}

fn validate_config(config: CalibrationConfig) -> Result<(), CalibrationError> {
    if !(2..=100).contains(&config.confidence_bins)
        || !(100..=50_000).contains(&config.bootstrap_iterations)
        || config.bootstrap_seed == 0
    {
        return Err(CalibrationError::new(
            CalibrationErrorKind::InvalidConfiguration,
            "confidence bins must be 2..=100, bootstrap iterations 100..=50000, and bootstrap seed non-zero",
        ));
    }
    Ok(())
}

fn validate_thresholds(thresholds: CalibrationThresholds) -> Result<(), CalibrationError> {
    let finite = [thresholds.maximum_brier_score, thresholds.maximum_top_label_ece]
        .into_iter()
        .all(f64::is_finite);
    if thresholds.minimum_observations == 0
        || thresholds.minimum_source_groups == 0
        || !finite
        || !(0.0..=2.0).contains(&thresholds.maximum_brier_score)
        || !(0.0..=1.0).contains(&thresholds.maximum_top_label_ece)
        || thresholds
            .maximum_log_loss_nats
            .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(CalibrationError::new(
            CalibrationErrorKind::InvalidThresholds,
            "calibration thresholds contain an invalid count, range or non-finite limit",
        ));
    }
    Ok(())
}

fn validate_observations(
    training_groups: &[String],
    observations: &[CalibrationObservation],
) -> Result<(Vec<String>, Vec<NormalizedObservation>), CalibrationError> {
    if observations.is_empty() {
        return Err(CalibrationError::new(
            CalibrationErrorKind::InvalidObservation,
            "calibration requires at least one labeled observation",
        ));
    }

    let mut training = BTreeSet::new();
    for group in training_groups {
        validate_digest(group, "training source group digest").map_err(|mut error| {
            error.kind = CalibrationErrorKind::InvalidTrainingSplit;
            error
        })?;
        if !training.insert(group.clone()) {
            return Err(CalibrationError::new(
                CalibrationErrorKind::InvalidTrainingSplit,
                format!("duplicate training source group {group}"),
            ));
        }
    }

    let mut sample_ids = HashSet::new();
    let mut class_set: Option<BTreeSet<String>> = None;
    let mut normalized = Vec::with_capacity(observations.len());

    for observation in observations {
        if observation.sample_id.trim().is_empty() {
            return Err(CalibrationError::new(
                CalibrationErrorKind::InvalidObservation,
                "sample ID must be non-empty",
            ));
        }
        if !sample_ids.insert(observation.sample_id.as_str()) {
            return Err(CalibrationError::new(
                CalibrationErrorKind::DuplicateSampleId,
                format!("duplicate calibration sample ID {}", observation.sample_id),
            ));
        }
        validate_digest(
            &observation.source_group_sha256,
            "calibration source group digest",
        )?;
        if training.contains(&observation.source_group_sha256) {
            return Err(CalibrationError::new(
                CalibrationErrorKind::TrainingCalibrationLeakage,
                format!(
                    "source group {} appears in both training and calibration cohorts",
                    observation.source_group_sha256
                ),
            ));
        }

        if observation.predicted_probabilities.len() < 2 {
            return Err(CalibrationError::new(
                CalibrationErrorKind::InvalidProbabilityDistribution,
                format!(
                    "sample {} must predict at least two classes",
                    observation.sample_id
                ),
            ));
        }

        let mut row = observation.predicted_probabilities.clone();
        row.sort_by(|left, right| left.class_id.cmp(&right.class_id));
        let ids: BTreeSet<String> = row.iter().map(|item| item.class_id.clone()).collect();
        if ids.len() != row.len() || ids.iter().any(|id| id.trim().is_empty()) {
            return Err(CalibrationError::new(
                CalibrationErrorKind::InvalidProbabilityDistribution,
                format!(
                    "sample {} has empty or duplicate predicted class IDs",
                    observation.sample_id
                ),
            ));
        }
        if let Some(expected) = &class_set {
            if expected != &ids {
                return Err(CalibrationError::new(
                    CalibrationErrorKind::InconsistentClassSet,
                    format!("sample {} predicts a different class set", observation.sample_id),
                ));
            }
        } else {
            class_set = Some(ids.clone());
        }

        let observed_class_index = row
            .iter()
            .position(|item| item.class_id == observation.observed_class_id)
            .ok_or_else(|| {
                CalibrationError::new(
                    CalibrationErrorKind::InvalidObservation,
                    format!(
                        "observed class {} is absent from sample {} prediction",
                        observation.observed_class_id, observation.sample_id
                    ),
                )
            })?;

        let mut probabilities = Vec::with_capacity(row.len());
        let mut total = 0.0;
        for item in &row {
            if !item.probability.is_finite() || !(0.0..=1.0).contains(&item.probability) {
                return Err(CalibrationError::new(
                    CalibrationErrorKind::InvalidProbabilityDistribution,
                    format!(
                        "sample {} contains a non-finite or out-of-range probability",
                        observation.sample_id
                    ),
                ));
            }
            probabilities.push(item.probability);
            total += item.probability;
        }
        if (total - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(CalibrationError::new(
                CalibrationErrorKind::InvalidProbabilityDistribution,
                format!(
                    "sample {} probability mass is {total}, expected 1.0",
                    observation.sample_id
                ),
            ));
        }

        // Class IDs are sorted, so an exact confidence tie deterministically
        // chooses the lexicographically smallest class ID.
        let mut top_index = 0;
        for index in 1..probabilities.len() {
            if probabilities[index] > probabilities[top_index] {
                top_index = index;
            }
        }
        let confidence = probabilities[top_index];
        let correct = top_index == observed_class_index;
        let brier_score = probabilities
            .iter()
            .enumerate()
            .map(|(index, probability)| {
                let expected = if index == observed_class_index { 1.0 } else { 0.0 };
                (probability - expected).powi(2)
            })
            .sum::<f64>();
        let observed_probability = probabilities[observed_class_index];
        let log_loss = if observed_probability == 0.0 {
            None
        } else {
            Some(-observed_probability.ln())
        };

        normalized.push(NormalizedObservation {
            sample_id: observation.sample_id.clone(),
            source_group_sha256: observation.source_group_sha256.clone(),
            probabilities,
            observed_class_index,
            top_label_confidence: confidence,
            top_label_correct: correct,
            brier_score,
            log_loss_nats: log_loss,
        });
    }

    normalized.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    Ok((
        class_set.expect("non-empty observations produce a class set").into_iter().collect(),
        normalized,
    ))
}

fn summarize(
    observations: &[NormalizedObservation],
    indexes: &[usize],
    bin_count: usize,
) -> ScoreSummary {
    let mut total_brier = 0.0;
    let mut total_log_loss = 0.0;
    let mut zero_probability_count = 0;
    let mut bins = vec![(0usize, 0.0, 0usize); bin_count];

    for index in indexes {
        let observation = &observations[*index];
        total_brier += observation.brier_score;
        match observation.log_loss_nats {
            Some(value) => total_log_loss += value,
            None => zero_probability_count += 1,
        }

        let bin_index = ((observation.top_label_confidence * bin_count as f64).floor()
            as usize)
            .min(bin_count - 1);
        bins[bin_index].0 += 1;
        bins[bin_index].1 += observation.top_label_confidence;
        bins[bin_index].2 += usize::from(observation.top_label_correct);
    }

    let count = indexes.len() as f64;
    let mut ece = 0.0;
    for (bin_count, confidence_sum, correct_count) in bins {
        if bin_count == 0 {
            continue;
        }
        let fraction = bin_count as f64 / count;
        let mean_confidence = confidence_sum / bin_count as f64;
        let accuracy = correct_count as f64 / bin_count as f64;
        ece += fraction * (mean_confidence - accuracy).abs();
    }

    ScoreSummary {
        brier_score: total_brier / count,
        mean_log_loss_nats: if zero_probability_count == 0 {
            Some(total_log_loss / count)
        } else {
            None
        },
        zero_probability_count,
        top_label_ece: ece,
    }
}

fn wilson_interval(successes: usize, count: usize) -> CalibrationInterval {
    const Z: f64 = 1.959_963_984_540_054;
    let n = count as f64;
    let observed = successes as f64 / n;
    let z2 = Z * Z;
    let denominator = 1.0 + z2 / n;
    let center = (observed + z2 / (2.0 * n)) / denominator;
    let half_width =
        Z * ((observed * (1.0 - observed) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    CalibrationInterval {
        lower: (center - half_width).max(0.0),
        upper: (center + half_width).min(1.0),
    }
}

fn reliability_bins(
    observations: &[NormalizedObservation],
    bin_count: usize,
) -> Vec<ReliabilityBin> {
    let mut counts = vec![0usize; bin_count];
    let mut confidence_sums = vec![0.0; bin_count];
    let mut correct_counts = vec![0usize; bin_count];

    for observation in observations {
        let bin = ((observation.top_label_confidence * bin_count as f64).floor() as usize)
            .min(bin_count - 1);
        counts[bin] += 1;
        confidence_sums[bin] += observation.top_label_confidence;
        correct_counts[bin] += usize::from(observation.top_label_correct);
    }

    counts
        .iter()
        .enumerate()
        .map(|(index, count)| {
            let lower = index as f64 / bin_count as f64;
            let upper = (index + 1) as f64 / bin_count as f64;
            if *count == 0 {
                ReliabilityBin {
                    index,
                    lower_bound: lower,
                    upper_bound: upper,
                    count: 0,
                    mean_confidence: None,
                    observed_accuracy: None,
                    accuracy_wilson_95: None,
                }
            } else {
                ReliabilityBin {
                    index,
                    lower_bound: lower,
                    upper_bound: upper,
                    count: *count,
                    mean_confidence: Some(confidence_sums[index] / *count as f64),
                    observed_accuracy: Some(correct_counts[index] as f64 / *count as f64),
                    accuracy_wilson_95: Some(wilson_interval(correct_counts[index], *count)),
                }
            }
        })
        .collect()
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    fn index(&mut self, length: usize) -> usize {
        ((self.next() as u128 * length as u128) >> 64) as usize
    }
}

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    let index = (probability * (sorted.len().saturating_sub(1)) as f64).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

/// Cluster bootstrap: resample source groups with replacement and include all
/// observations from each selected group. This avoids treating multiple samples
/// from one source program/binary family as independent units.
fn bootstrap_intervals(
    observations: &[NormalizedObservation],
    config: CalibrationConfig,
) -> (
    CalibrationInterval,
    CalibrationInterval,
    Option<CalibrationInterval>,
) {
    let mut groups = BTreeMap::<&str, Vec<usize>>::new();
    for (index, observation) in observations.iter().enumerate() {
        groups
            .entry(observation.source_group_sha256.as_str())
            .or_default()
            .push(index);
    }
    let group_indexes: Vec<&Vec<usize>> = groups.values().collect();
    let mut rng = XorShift64::new(config.bootstrap_seed);
    let mut indexes = Vec::with_capacity(observations.len());
    let mut brier = Vec::with_capacity(config.bootstrap_iterations);
    let mut ece = Vec::with_capacity(config.bootstrap_iterations);
    let all_log_losses_finite = observations.iter().all(|item| item.log_loss_nats.is_some());
    let mut log_loss = Vec::with_capacity(config.bootstrap_iterations);

    for _ in 0..config.bootstrap_iterations {
        indexes.clear();
        for _ in 0..group_indexes.len() {
            let selected_group = group_indexes[rng.index(group_indexes.len())];
            indexes.extend(selected_group.iter().copied());
        }
        let summary = summarize(observations, &indexes, config.confidence_bins);
        brier.push(summary.brier_score);
        ece.push(summary.top_label_ece);
        if let Some(value) = summary.mean_log_loss_nats {
            log_loss.push(value);
        }
    }

    brier.sort_by(f64::total_cmp);
    ece.sort_by(f64::total_cmp);
    log_loss.sort_by(f64::total_cmp);

    let interval = |values: &[f64]| CalibrationInterval {
        lower: percentile(values, 0.025),
        upper: percentile(values, 0.975),
    };
    (
        interval(&brier),
        interval(&ece),
        if all_log_losses_finite {
            Some(interval(&log_loss))
        } else {
            None
        },
    )
}

fn update_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn update_string(hasher: &mut Sha256, value: &str) {
    update_bytes(hasher, value.as_bytes());
}

fn update_usize(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_be_bytes());
}

fn update_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_be_bytes());
}

fn update_f64(hasher: &mut Sha256, value: f64) {
    hasher.update(value.to_bits().to_be_bytes());
}

fn digest_hex(hasher: Sha256) -> String {
    let bytes = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in bytes {
        use fmt::Write as _;
        let _ = write!(&mut hex, "{byte:02x}");
    }
    hex
}

fn data_digest(class_ids: &[String], observations: &[NormalizedObservation]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_DOMAIN_DATA);
    update_usize(&mut hasher, class_ids.len());
    for class_id in class_ids {
        update_string(&mut hasher, class_id);
    }
    update_usize(&mut hasher, observations.len());
    for observation in observations {
        update_string(&mut hasher, &observation.sample_id);
        update_string(&mut hasher, &observation.source_group_sha256);
        update_string(&mut hasher, &class_ids[observation.observed_class_index]);
        update_usize(&mut hasher, observation.probabilities.len());
        for probability in &observation.probabilities {
            update_f64(&mut hasher, *probability);
        }
    }
    digest_hex(hasher)
}

fn hash_optional_f64(hasher: &mut Sha256, value: Option<f64>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            update_f64(hasher, value);
        }
        None => hasher.update([0]),
    }
}

fn hash_interval(hasher: &mut Sha256, interval: Option<CalibrationInterval>) {
    match interval {
        Some(value) => {
            hasher.update([1]);
            update_f64(hasher, value.lower);
            update_f64(hasher, value.upper);
        }
        None => hasher.update([0]),
    }
}

fn report_digest(metrics: &CalibrationMetrics, eligibility: CalibrationEligibility) -> String {
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_DOMAIN_REPORT);
    update_usize(&mut hasher, metrics.observation_count);
    update_usize(&mut hasher, metrics.source_group_count);
    update_f64(&mut hasher, metrics.brier_score);
    hash_optional_f64(&mut hasher, metrics.mean_log_loss_nats);
    update_usize(&mut hasher, metrics.zero_probability_count);
    update_f64(&mut hasher, metrics.top_label_ece);
    update_string(&mut hasher, &metrics.bootstrap_unit);
    hash_interval(&mut hasher, Some(metrics.brier_bootstrap_95));
    hash_interval(&mut hasher, Some(metrics.top_label_ece_bootstrap_95));
    hash_interval(&mut hasher, metrics.mean_log_loss_bootstrap_95);
    update_usize(&mut hasher, metrics.reliability_bins.len());
    for bin in &metrics.reliability_bins {
        update_usize(&mut hasher, bin.index);
        update_f64(&mut hasher, bin.lower_bound);
        update_f64(&mut hasher, bin.upper_bound);
        update_usize(&mut hasher, bin.count);
        hash_optional_f64(&mut hasher, bin.mean_confidence);
        hash_optional_f64(&mut hasher, bin.observed_accuracy);
        hash_interval(&mut hasher, bin.accuracy_wilson_95);
    }
    hasher.update([eligibility as u8]);
    digest_hex(hasher)
}

fn receipt_digest(receipt: &CalibrationReceipt) -> String {
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_DOMAIN_RECEIPT);
    update_string(&mut hasher, &receipt.schema);
    update_string(&mut hasher, &receipt.evaluator_version);
    update_string(&mut hasher, &receipt.model.model_id);
    update_string(&mut hasher, &receipt.model.model_version);
    update_string(&mut hasher, &receipt.model.model_sha256);
    for value in [
        &receipt.scope.platform,
        &receipt.scope.target_build,
        &receipt.scope.fault_family,
        &receipt.scope.environment,
        &receipt.scope.observation_regime,
    ] {
        update_string(&mut hasher, value);
    }
    update_usize(&mut hasher, receipt.training_source_group_sha256s.len());
    for group in &receipt.training_source_group_sha256s {
        update_string(&mut hasher, group);
    }
    update_usize(&mut hasher, receipt.calibration_source_group_sha256s.len());
    for group in &receipt.calibration_source_group_sha256s {
        update_string(&mut hasher, group);
    }
    update_string(&mut hasher, &receipt.observations_sha256);
    update_string(&mut hasher, &receipt.report_sha256);
    update_usize(&mut hasher, receipt.config.confidence_bins);
    update_usize(&mut hasher, receipt.config.bootstrap_iterations);
    update_u64(&mut hasher, receipt.config.bootstrap_seed);
    update_usize(&mut hasher, receipt.thresholds.minimum_observations);
    update_usize(&mut hasher, receipt.thresholds.minimum_source_groups);
    update_f64(&mut hasher, receipt.thresholds.maximum_brier_score);
    update_f64(&mut hasher, receipt.thresholds.maximum_top_label_ece);
    hash_optional_f64(&mut hasher, receipt.thresholds.maximum_log_loss_nats);
    update_usize(
        &mut hasher,
        receipt.thresholds.maximum_zero_probability_count,
    );
    hasher.update([receipt.eligibility as u8]);
    hasher.update([
        receipt.claim_ceiling.producer_authenticated as u8,
        receipt.claim_ceiling.source_labels_independently_verified as u8,
        receipt.claim_ceiling.training_calibration_split_independently_audited as u8,
        receipt.claim_ceiling.calibration_certified as u8,
        receipt.claim_ceiling.target_scope_completeness_verified as u8,
        receipt.claim_ceiling.distribution_stationarity_proven as u8,
        receipt.claim_ceiling.drift_checked as u8,
    ]);
    digest_hex(hasher)
}

/// Evaluate a frozen, held-out calibration cohort and create an unsigned receipt.
///
/// This function computes metrics and binds them to caller-supplied model/scope
/// identities. It does not authenticate the model, dataset, labels or source
/// groups. EligibleForIndependentReview means only that configured support and
/// metric thresholds were met; it is not a calibration certification.
pub fn evaluate_likelihood_calibration(
    model: CalibrationModelIdentity,
    scope: CalibrationScope,
    training_source_group_sha256s: &[String],
    observations: &[CalibrationObservation],
    config: CalibrationConfig,
    thresholds: CalibrationThresholds,
) -> Result<CalibrationReceipt, CalibrationError> {
    validate_model(&model)?;
    validate_scope(&scope)?;
    validate_config(config)?;
    validate_thresholds(thresholds)?;

    let (class_ids, normalized) =
        validate_observations(training_source_group_sha256s, observations)?;
    let mut training_groups = training_source_group_sha256s.to_vec();
    training_groups.sort();
    let calibration_groups: Vec<String> = normalized
        .iter()
        .map(|item| item.source_group_sha256.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let indexes: Vec<usize> = (0..normalized.len()).collect();
    let summary = summarize(&normalized, &indexes, config.confidence_bins);
    let reliability = reliability_bins(&normalized, config.confidence_bins);
    let (brier_interval, ece_interval, log_loss_interval) =
        bootstrap_intervals(&normalized, config);
    let source_group_count = calibration_groups.len();

    let eligibility = if normalized.len() < thresholds.minimum_observations
        || source_group_count < thresholds.minimum_source_groups
    {
        CalibrationEligibility::InsufficientSupport
    } else {
        let log_loss_passes = match thresholds.maximum_log_loss_nats {
            None => true,
            Some(maximum) => summary
                .mean_log_loss_nats
                .is_some_and(|value| value <= maximum),
        };
        if summary.brier_score > thresholds.maximum_brier_score
            || summary.top_label_ece > thresholds.maximum_top_label_ece
            || !log_loss_passes
            || summary.zero_probability_count > thresholds.maximum_zero_probability_count
        {
            CalibrationEligibility::ThresholdsNotMet
        } else {
            CalibrationEligibility::EligibleForIndependentReview
        }
    };

    let metrics = CalibrationMetrics {
        observation_count: normalized.len(),
        source_group_count,
        brier_score: summary.brier_score,
        mean_log_loss_nats: summary.mean_log_loss_nats,
        zero_probability_count: summary.zero_probability_count,
        top_label_ece: summary.top_label_ece,
        reliability_bins: reliability,
        bootstrap_unit: CALIBRATION_BOOTSTRAP_UNIT.to_owned(),
        brier_bootstrap_95: brier_interval,
        top_label_ece_bootstrap_95: ece_interval,
        mean_log_loss_bootstrap_95: log_loss_interval,
    };

    let mut receipt = CalibrationReceipt {
        schema: "luminous.rea-likelihood-calibration-receipt.v1".to_owned(),
        evaluator_version: CALIBRATION_EVALUATOR_VERSION.to_owned(),
        model,
        scope,
        training_source_group_sha256s: training_groups,
        calibration_source_group_sha256s: calibration_groups,
        observations_sha256: data_digest(&class_ids, &normalized),
        report_sha256: report_digest(&metrics, eligibility),
        receipt_sha256: String::new(),
        config,
        thresholds,
        metrics,
        eligibility,
        claim_ceiling: CalibrationClaimCeiling::default(),
    };
    receipt.receipt_sha256 = receipt_digest(&receipt);
    Ok(receipt)
}

/// Verify the internal report/envelope digests and recompute them against the
/// exact supplied calibration observations.
pub fn verify_calibration_receipt_against_observations(
    receipt: &CalibrationReceipt,
    observations: &[CalibrationObservation],
) -> Result<(), CalibrationError> {
    validate_model(&receipt.model)?;
    validate_scope(&receipt.scope)?;
    validate_config(receipt.config)?;
    validate_thresholds(receipt.thresholds)?;
    let (class_ids, normalized) =
        validate_observations(&receipt.training_source_group_sha256s, observations)?;
    let groups: Vec<String> = normalized
        .iter()
        .map(|item| item.source_group_sha256.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if data_digest(&class_ids, &normalized) != receipt.observations_sha256
        || groups != receipt.calibration_source_group_sha256s
        || report_digest(&receipt.metrics, receipt.eligibility) != receipt.report_sha256
        || receipt_digest(receipt) != receipt.receipt_sha256
    {
        return Err(CalibrationError::new(
            CalibrationErrorKind::ReceiptDigestMismatch,
            "calibration receipt does not match the supplied observations or its own report/envelope digests",
        ));
    }
    let expected = evaluate_likelihood_calibration(
        receipt.model.clone(),
        receipt.scope.clone(),
        &receipt.training_source_group_sha256s,
        observations,
        receipt.config,
        receipt.thresholds,
    )?;
    if expected.receipt_sha256 != receipt.receipt_sha256 {
        return Err(CalibrationError::new(
            CalibrationErrorKind::ReceiptDigestMismatch,
            "recomputed calibration receipt differs from the supplied receipt",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CalibrationModelIdentity {
        CalibrationModelIdentity {
            model_id: "symthaea-mobile-diagnosis".to_owned(),
            model_version: "0.4.2".to_owned(),
            model_sha256: "a".repeat(64),
        }
    }

    fn scope() -> CalibrationScope {
        CalibrationScope {
            platform: "android".to_owned(),
            target_build: "pixel8pro-test-2026-11".to_owned(),
            fault_family: "network-parser".to_owned(),
            environment: "cuttlefish-api-36".to_owned(),
            observation_regime: "offline-replay".to_owned(),
        }
    }

    fn config() -> CalibrationConfig {
        CalibrationConfig {
            confidence_bins: 10,
            bootstrap_iterations: 200,
            bootstrap_seed: 0x1234_5678_9abc_def0,
        }
    }

    fn thresholds() -> CalibrationThresholds {
        CalibrationThresholds {
            minimum_observations: 4,
            minimum_source_groups: 2,
            maximum_brier_score: 2.0,
            maximum_top_label_ece: 1.0,
            maximum_log_loss_nats: None,
            maximum_zero_probability_count: 4,
        }
    }

    fn group(label: char) -> String {
        label.to_string().repeat(64)
    }

    fn observation(
        sample: &str,
        source: char,
        p_fault: f64,
        p_healthy: f64,
        outcome: &str,
    ) -> CalibrationObservation {
        CalibrationObservation {
            sample_id: sample.to_owned(),
            source_group_sha256: group(source),
            predicted_probabilities: vec![
                ClassProbability {
                    class_id: "fault".to_owned(),
                    probability: p_fault,
                },
                ClassProbability {
                    class_id: "healthy".to_owned(),
                    probability: p_healthy,
                },
            ],
            observed_class_id: outcome.to_owned(),
        }
    }

    fn evaluate(observations: &[CalibrationObservation]) -> CalibrationReceipt {
        evaluate_likelihood_calibration(
            model(),
            scope(),
            &[group('f')],
            observations,
            config(),
            thresholds(),
        )
        .unwrap()
    }

    #[test]
    fn known_answer_brier_logloss_and_top_label_ece() {
        let receipt = evaluate(&[observation("sample-1", '1', 0.8, 0.2, "fault")]);
        assert!((receipt.metrics.brier_score - 0.08).abs() < 1.0e-12);
        assert!(
            (receipt.metrics.mean_log_loss_nats.unwrap() - (-0.8f64.ln())).abs()
                < 1.0e-12
        );
        assert!((receipt.metrics.top_label_ece - 0.2).abs() < 1.0e-12);
        assert_eq!(receipt.metrics.zero_probability_count, 0);
    }

    #[test]
    fn zero_probability_outcome_is_an_explicit_unbounded_logloss_case() {
        let receipt = evaluate(&[observation("sample-1", '1', 1.0, 0.0, "healthy")]);
        assert_eq!(receipt.metrics.zero_probability_count, 1);
        assert!(receipt.metrics.mean_log_loss_nats.is_none());
        assert!(receipt.metrics.mean_log_loss_bootstrap_95.is_none());
    }

    #[test]
    fn training_and_calibration_source_groups_must_be_disjoint() {
        let result = evaluate_likelihood_calibration(
            model(),
            scope(),
            &[group('1')],
            &[observation("sample-1", '1', 0.8, 0.2, "fault")],
            config(),
            thresholds(),
        );
        assert_eq!(
            result.unwrap_err().kind,
            CalibrationErrorKind::TrainingCalibrationLeakage
        );
    }

    #[test]
    fn duplicate_sample_ids_are_rejected() {
        let item = observation("duplicate", '1', 0.8, 0.2, "fault");
        let result = evaluate(&[item.clone(), item]);
        assert_eq!(
            result.unwrap_err().kind,
            CalibrationErrorKind::DuplicateSampleId
        );
    }

    #[test]
    fn class_and_observation_order_do_not_change_digests() {
        let first = vec![
            observation("sample-a", '1', 0.8, 0.2, "fault"),
            observation("sample-b", '2', 0.2, 0.8, "healthy"),
        ];
        let mut second = first.clone();
        second.reverse();
        for item in &mut second {
            item.predicted_probabilities.reverse();
        }
        let a = evaluate(&first);
        let b = evaluate(&second);
        assert_eq!(a.observations_sha256, b.observations_sha256);
        assert_eq!(a.report_sha256, b.report_sha256);
        assert_eq!(a.receipt_sha256, b.receipt_sha256);
    }

    #[test]
    fn receipt_tampering_is_detected() {
        let mut receipt = evaluate(&[
            observation("sample-a", '1', 0.8, 0.2, "fault"),
            observation("sample-b", '2', 0.2, 0.8, "healthy"),
        ]);
        verify_calibration_receipt_against_observations(
            &receipt,
            &[
                observation("sample-a", '1', 0.8, 0.2, "fault"),
                observation("sample-b", '2', 0.2, 0.8, "healthy"),
            ],
        )
        .unwrap();
        receipt.metrics.top_label_ece = 0.9;
        assert_eq!(
            verify_calibration_receipt_against_observations(
                &receipt,
                &[
                    observation("sample-a", '1', 0.8, 0.2, "fault"),
                    observation("sample-b", '2', 0.2, 0.8, "healthy"),
                ],
            )
            .unwrap_err()
            .kind,
            CalibrationErrorKind::ReceiptDigestMismatch
        );
    }

    #[test]
    fn bootstrap_intervals_are_reproducible() {
        let observations = vec![
            observation("sample-a1", '1', 0.8, 0.2, "fault"),
            observation("sample-a2", '1', 0.8, 0.2, "healthy"),
            observation("sample-b1", '2', 0.6, 0.4, "fault"),
            observation("sample-b2", '2', 0.6, 0.4, "fault"),
        ];
        let first = evaluate(&observations);
        let second = evaluate(&observations);
        assert_eq!(
            first.metrics.brier_bootstrap_95,
            second.metrics.brier_bootstrap_95
        );
        assert_eq!(
            first.metrics.top_label_ece_bootstrap_95,
            second.metrics.top_label_ece_bootstrap_95
        );
        assert_eq!(
            first.metrics.bootstrap_unit,
            CALIBRATION_BOOTSTRAP_UNIT
        );
    }

    #[test]
    fn eligibility_is_only_for_independent_review_not_certification() {
        let observations = vec![
            observation("sample-a", '1', 0.8, 0.2, "fault"),
            observation("sample-b", '2', 0.8, 0.2, "healthy"),
            observation("sample-c", '3', 0.6, 0.4, "fault"),
            observation("sample-d", '4', 0.6, 0.4, "fault"),
        ];
        let receipt = evaluate(&observations);
        assert_eq!(
            receipt.eligibility,
            CalibrationEligibility::EligibleForIndependentReview
        );
        assert!(!receipt.claim_ceiling.calibration_certified);
        assert!(!receipt.claim_ceiling.producer_authenticated);
    }

    #[test]
    fn insufficient_source_group_support_is_not_qualified() {
        let observations = vec![
            observation("sample-a", '1', 0.8, 0.2, "fault"),
            observation("sample-b", '1', 0.8, 0.2, "fault"),
            observation("sample-c", '1', 0.8, 0.2, "fault"),
            observation("sample-d", '1', 0.8, 0.2, "fault"),
        ];
        let receipt = evaluate(&observations);
        assert_eq!(
            receipt.eligibility,
            CalibrationEligibility::InsufficientSupport
        );
        assert_eq!(receipt.metrics.observation_count, 4);
        assert_eq!(receipt.metrics.source_group_count, 1);
    }

    #[test]
    fn non_normalized_predictions_are_rejected() {
        let result = evaluate(&[observation("sample-a", '1', 0.7, 0.2, "fault")]);
        assert_eq!(
            result.unwrap_err().kind,
            CalibrationErrorKind::InvalidProbabilityDistribution
        );
    }
}
