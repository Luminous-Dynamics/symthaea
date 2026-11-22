// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Probability calibration and selective-risk scoring for mycelial reconstruction.
//!
//! This module evaluates confidence quality separately from detection and graph
//! fidelity. It measures seven-way class probabilities for each predicted
//! junction, treating unmatched predicted junctions as the explicit NoObject
//! outcome. Missed ground-truth junctions remain false negatives in the
//! reconstruction scorecard; they are not silently introduced as confidence
//! samples here.
//!
//! Confidence, calibration and abstention never authorize anastomosis.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_reconstruction_metrics::{
    AnnotatedJunctionKind, AnnotatedMycelialSkeleton, MycelialReconstructionScore,
};

pub const MYCELIAL_RECONSTRUCTION_CALIBRATION_SCHEMA_VERSION: u16 = 1;
pub const MAX_RECONSTRUCTION_CALIBRATION_BINS: usize = 100;
pub const MAX_RECONSTRUCTION_CALIBRATION_PREDICTIONS: usize = 8_192;
pub const RECONSTRUCTION_CALIBRATION_CLASS_COUNT: usize = 7;
const CLASS_NAMES: [&str; RECONSTRUCTION_CALIBRATION_CLASS_COUNT] = [
    "origin",
    "branch",
    "tip",
    "anastomosis",
    "projected_crossing",
    "uncertain",
    "no_object",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionCalibrationClass {
    Origin,
    Branch,
    Tip,
    Anastomosis,
    ProjectedCrossing,
    Uncertain,
    NoObject,
}

impl ReconstructionCalibrationClass {
    pub const ALL: [Self; RECONSTRUCTION_CALIBRATION_CLASS_COUNT] = [
        Self::Origin,
        Self::Branch,
        Self::Tip,
        Self::Anastomosis,
        Self::ProjectedCrossing,
        Self::Uncertain,
        Self::NoObject,
    ];

    pub fn index(self) -> usize {
        match self {
            Self::Origin => 0,
            Self::Branch => 1,
            Self::Tip => 2,
            Self::Anastomosis => 3,
            Self::ProjectedCrossing => 4,
            Self::Uncertain => 5,
            Self::NoObject => 6,
        }
    }

    pub fn from_junction_kind(kind: AnnotatedJunctionKind) -> Self {
        match kind {
            AnnotatedJunctionKind::Origin => Self::Origin,
            AnnotatedJunctionKind::Branch => Self::Branch,
            AnnotatedJunctionKind::Tip => Self::Tip,
            AnnotatedJunctionKind::Anastomosis => Self::Anastomosis,
            AnnotatedJunctionKind::ProjectedCrossing => Self::ProjectedCrossing,
            AnnotatedJunctionKind::Uncertain => Self::Uncertain,
        }
    }

    pub fn name(self) -> &'static str {
        CLASS_NAMES[self.index()]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JunctionClassProbabilityPrediction {
    pub predicted_junction_id: String,
    /// Probabilities in ReconstructionCalibrationClass::ALL order.
    /// The final entry is NoObject: the candidate should be abstained/rejected.
    pub probabilities: [f64; RECONSTRUCTION_CALIBRATION_CLASS_COUNT],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JunctionCalibrationObservation {
    pub predicted_junction_id: String,
    pub target_class: ReconstructionCalibrationClass,
    pub probabilities: [f64; RECONSTRUCTION_CALIBRATION_CLASS_COUNT],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReconstructionCalibrationConfig {
    pub bin_count: usize,
    pub probability_sum_tolerance: f64,
    pub negative_log_likelihood_epsilon: f64,
}

impl Default for ReconstructionCalibrationConfig {
    fn default() -> Self {
        Self {
            bin_count: 10,
            probability_sum_tolerance: 1.0e-6,
            negative_log_likelihood_epsilon: 1.0e-12,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionCalibrationBin {
    pub bin_index: usize,
    pub sample_count: usize,
    pub mean_confidence: Option<f64>,
    pub empirical_frequency: Option<f64>,
    pub absolute_calibration_gap: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionClassCalibration {
    pub class: ReconstructionCalibrationClass,
    pub brier_score: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub bins: Vec<ReconstructionCalibrationBin>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionTopLabelCalibration {
    pub brier_score: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub negative_log_likelihood: Option<f64>,
    pub accuracy: Option<f64>,
    pub bins: Vec<ReconstructionCalibrationBin>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionSelectiveRiskPoint {
    /// Predictions with top-label confidence >= this value are retained.
    pub minimum_included_confidence: f64,
    pub retained_prediction_count: usize,
    pub coverage: f64,
    pub classification_error_count: usize,
    pub selective_risk: f64,
    pub predicted_anastomosis_count: usize,
    pub false_anastomosis_prediction_count: usize,
    pub false_anastomosis_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionCalibrationCounts {
    pub prediction_count: usize,
    pub matched_prediction_count: usize,
    pub unmatched_prediction_count: usize,
    pub correct_top_label_count: usize,
    pub top_label_error_count: usize,
    pub predicted_anastomosis_count: usize,
    pub false_anastomosis_prediction_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialReconstructionCalibrationReport {
    pub schema_version: u16,
    pub source_observation_digest: String,
    pub ground_truth_graph_digest: String,
    pub prediction_graph_digest: String,
    pub reconstruction_score_digest: String,
    /// Digest of sorted prediction IDs and exact probability-vector bit patterns.
    pub probability_inventory_digest: String,
    pub matching_policy_id: String,
    pub calibration_policy_id: String,
    pub bin_count: usize,
    pub probability_sum_tolerance: f64,
    pub negative_log_likelihood_epsilon: f64,
    pub counts: ReconstructionCalibrationCounts,
    /// Complete sorted probability-and-target inventory; all aggregate metrics
    /// can be independently recomputed from this evidence.
    pub observations: Vec<JunctionCalibrationObservation>,
    pub class_prediction_counts: Vec<(ReconstructionCalibrationClass, usize)>,
    pub class_calibration: Vec<ReconstructionClassCalibration>,
    pub top_label_calibration: ReconstructionTopLabelCalibration,
    /// One point per distinct confidence threshold; ties are never split.
    pub selective_risk_curve: Vec<ReconstructionSelectiveRiskPoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MycelialReconstructionCalibrationError {
    InvalidConfig,
    InvalidPolicyId,
    TooManyPredictions(usize),
    SourceMismatch,
    ScoreDigestMismatch,
    PredictionInventoryMismatch,
    DuplicatePredictionId(String),
    InvalidProbabilityVector(String),
    InvalidReport(&'static str),
}

impl fmt::Display for MycelialReconstructionCalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "invalid reconstruction calibration configuration"),
            Self::InvalidPolicyId => write!(f, "calibration policy ID must be non-empty and bounded"),
            Self::TooManyPredictions(count) => write!(
                f,
                "calibration prediction count {count} exceeds {MAX_RECONSTRUCTION_CALIBRATION_PREDICTIONS}"
            ),
            Self::SourceMismatch => {
                write!(f, "calibration inputs do not share the reconstruction source identity")
            }
            Self::ScoreDigestMismatch => {
                write!(f, "reconstruction score does not bind the supplied graph pair")
            }
            Self::PredictionInventoryMismatch => {
                write!(f, "probability predictions do not cover the predicted junction inventory")
            }
            Self::DuplicatePredictionId(id) => {
                write!(f, "duplicate probability prediction for junction {id}")
            }
            Self::InvalidProbabilityVector(id) => {
                write!(f, "invalid seven-class probability vector for junction {id}")
            }
            Self::InvalidReport(reason) => {
                write!(f, "invalid reconstruction calibration report: {reason}")
            }
        }
    }
}

impl Error for MycelialReconstructionCalibrationError {}

/// Evaluate probabilities for every predicted junction using the supplied
/// reconstruction identity map.
///
/// For a matched prediction, the target is the corresponding ground-truth
/// junction class. An unmatched predicted junction has NoObject as its target.
/// Ground-truth junctions missed entirely by the detector remain false
/// negatives in the reconstruction score; they are not probability samples.
///
/// The named policy, source/graph digests, probability constraints, calibration
/// bins and selective-risk curve are bound into a canonical digest.
pub fn evaluate_mycelial_reconstruction_calibration(
    ground_truth: &AnnotatedMycelialSkeleton,
    prediction: &AnnotatedMycelialSkeleton,
    reconstruction_score: &MycelialReconstructionScore,
    calibration_policy_id: &str,
    probability_predictions: &[JunctionClassProbabilityPrediction],
    config: ReconstructionCalibrationConfig,
) -> Result<MycelialReconstructionCalibrationReport, MycelialReconstructionCalibrationError> {
    ground_truth
        .validate()
        .map_err(|_| MycelialReconstructionCalibrationError::SourceMismatch)?;
    prediction
        .validate()
        .map_err(|_| MycelialReconstructionCalibrationError::SourceMismatch)?;
    reconstruction_score
        .validate()
        .map_err(|_| MycelialReconstructionCalibrationError::ScoreDigestMismatch)?;
    if ground_truth.frame_id != prediction.frame_id
        || ground_truth.coordinate_frame_id != prediction.coordinate_frame_id
        || ground_truth.calibration_digest != prediction.calibration_digest
        || ground_truth.source_observation_digest != prediction.source_observation_digest
        || reconstruction_score.frame_id != ground_truth.frame_id
        || reconstruction_score.coordinate_frame_id != ground_truth.coordinate_frame_id
        || reconstruction_score.calibration_digest != ground_truth.calibration_digest
        || reconstruction_score.source_observation_digest != ground_truth.source_observation_digest
    {
        return Err(MycelialReconstructionCalibrationError::SourceMismatch);
    }
    let ground_truth_digest = ground_truth
        .canonical_digest_hex()
        .map_err(|_| MycelialReconstructionCalibrationError::SourceMismatch)?;
    let prediction_digest = prediction
        .canonical_digest_hex()
        .map_err(|_| MycelialReconstructionCalibrationError::SourceMismatch)?;
    let score_digest = reconstruction_score
        .canonical_digest_hex()
        .map_err(|_| MycelialReconstructionCalibrationError::ScoreDigestMismatch)?;
    if reconstruction_score.ground_truth_digest != ground_truth_digest
        || reconstruction_score.prediction_digest != prediction_digest
    {
        return Err(MycelialReconstructionCalibrationError::ScoreDigestMismatch);
    }
    if calibration_policy_id.is_empty()
        || calibration_policy_id.len() > 128
        || calibration_policy_id.chars().any(char::is_control)
    {
        return Err(MycelialReconstructionCalibrationError::InvalidPolicyId);
    }
    if config.bin_count == 0
        || config.bin_count > MAX_RECONSTRUCTION_CALIBRATION_BINS
        || !config.probability_sum_tolerance.is_finite()
        || !(1.0e-12..=1.0e-2).contains(&config.probability_sum_tolerance)
        || !config.negative_log_likelihood_epsilon.is_finite()
        || !(1.0e-15..=1.0e-3).contains(&config.negative_log_likelihood_epsilon)
    {
        return Err(MycelialReconstructionCalibrationError::InvalidConfig);
    }
    if probability_predictions.len() > MAX_RECONSTRUCTION_CALIBRATION_PREDICTIONS {
        return Err(MycelialReconstructionCalibrationError::TooManyPredictions(
            probability_predictions.len(),
        ));
    }
    if probability_predictions.len() != prediction.junctions.len() {
        return Err(MycelialReconstructionCalibrationError::PredictionInventoryMismatch);
    }

    let ground_truth_kind: BTreeMap<_, _> = ground_truth
        .junctions
        .iter()
        .map(|junction| (junction.id.as_str(), ReconstructionCalibrationClass::from_junction_kind(junction.kind)))
        .collect();
    let prediction_ids: BTreeSet<_> = prediction
        .junctions
        .iter()
        .map(|junction| junction.id.as_str())
        .collect();
    let mut prediction_target = BTreeMap::<&str, ReconstructionCalibrationClass>::new();
    for matched in &reconstruction_score.identity_matches {
        let target = *ground_truth_kind
            .get(matched.ground_truth_junction_id.as_str())
            .ok_or(MycelialReconstructionCalibrationError::ScoreDigestMismatch)?;
        if prediction_target
            .insert(matched.predicted_junction_id.as_str(), target)
            .is_some()
        {
            return Err(MycelialReconstructionCalibrationError::ScoreDigestMismatch);
        }
    }

    let mut predictions = probability_predictions.to_vec();
    predictions.sort_by(|left, right| {
        left.predicted_junction_id.cmp(&right.predicted_junction_id)
    });
    if predictions.windows(2).any(|pair| {
        pair[0].predicted_junction_id == pair[1].predicted_junction_id
    }) {
        let duplicate = predictions
            .windows(2)
            .find(|pair| pair[0].predicted_junction_id == pair[1].predicted_junction_id)
            .map(|pair| pair[0].predicted_junction_id.clone())
            .unwrap_or_default();
        return Err(MycelialReconstructionCalibrationError::DuplicatePredictionId(
            duplicate,
        ));
    }
    if predictions
        .iter()
        .any(|value| !prediction_ids.contains(value.predicted_junction_id.as_str()))
        || predictions
            .iter()
            .map(|value| value.predicted_junction_id.as_str())
            .collect::<BTreeSet<_>>()
            != prediction_ids
    {
        return Err(MycelialReconstructionCalibrationError::PredictionInventoryMismatch);
    }

    let mut samples = Vec::with_capacity(predictions.len());
    for item in &predictions {
        let sum = item.probabilities.iter().sum::<f64>();
        if item.predicted_junction_id.is_empty()
            || item.predicted_junction_id.len() > 128
            || item.predicted_junction_id.chars().any(char::is_control)
            || item
                .probabilities
                .iter()
                .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
            || !sum.is_finite()
            || (sum - 1.0).abs() > config.probability_sum_tolerance
        {
            return Err(
                MycelialReconstructionCalibrationError::InvalidProbabilityVector(
                    item.predicted_junction_id.clone(),
                ),
            );
        }
        let target = prediction_target
            .get(item.predicted_junction_id.as_str())
            .copied()
            .unwrap_or(ReconstructionCalibrationClass::NoObject);
        let predicted_class = top_class(&item.probabilities);
        let confidence = item.probabilities[predicted_class.index()];
        samples.push(CalibrationSample {
            prediction_id: item.predicted_junction_id.clone(),
            probabilities: item.probabilities,
            target,
            predicted_class,
            confidence,
        });
    }

    let counts = calculate_counts(&samples);
    let class_prediction_counts = ReconstructionCalibrationClass::ALL
        .into_iter()
        .map(|class| {
            (
                class,
                samples
                    .iter()
                    .filter(|sample| sample.predicted_class == class)
                    .count(),
            )
        })
        .collect();
    let class_calibration = ReconstructionCalibrationClass::ALL
        .into_iter()
        .map(|class| class_calibration(class, &samples, config.bin_count))
        .collect();
    let top_label_calibration = top_label_calibration(&samples, config.bin_count, config.negative_log_likelihood_epsilon);
    let selective_risk_curve = selective_risk_curve(&samples);

    let observations: Vec<_> = samples
        .iter()
        .map(|sample| JunctionCalibrationObservation {
            predicted_junction_id: sample.prediction_id.clone(),
            target_class: sample.target,
            probabilities: sample.probabilities,
        })
        .collect();
    let report = MycelialReconstructionCalibrationReport {
        schema_version: MYCELIAL_RECONSTRUCTION_CALIBRATION_SCHEMA_VERSION,
        source_observation_digest: ground_truth.source_observation_digest.clone(),
        ground_truth_graph_digest: ground_truth_digest,
        prediction_graph_digest: prediction_digest,
        reconstruction_score_digest: score_digest,
        probability_inventory_digest: probability_inventory_digest(&observations),
        matching_policy_id: reconstruction_score.matching_policy_id.clone(),
        calibration_policy_id: calibration_policy_id.to_string(),
        bin_count: config.bin_count,
        probability_sum_tolerance: config.probability_sum_tolerance,
        negative_log_likelihood_epsilon: config.negative_log_likelihood_epsilon,
        counts,
        observations,
        class_prediction_counts,
        class_calibration,
        top_label_calibration,
        selective_risk_curve,
    };
    report.validate()?;
    Ok(report)
}

fn top_class(
    probabilities: &[f64; RECONSTRUCTION_CALIBRATION_CLASS_COUNT],
) -> ReconstructionCalibrationClass {
    let mut best = ReconstructionCalibrationClass::ALL[0];
    for class in ReconstructionCalibrationClass::ALL.into_iter().skip(1) {
        if probabilities[class.index()] > probabilities[best.index()] {
            best = class;
        }
    }
    best
}

#[derive(Debug, Clone)]
struct CalibrationSample {
    prediction_id: String,
    probabilities: [f64; RECONSTRUCTION_CALIBRATION_CLASS_COUNT],
    target: ReconstructionCalibrationClass,
    predicted_class: ReconstructionCalibrationClass,
    confidence: f64,
}

fn samples_from_observations(
    observations: &[JunctionCalibrationObservation],
    probability_sum_tolerance: f64,
) -> Result<Vec<CalibrationSample>, MycelialReconstructionCalibrationError> {
    if observations.windows(2).any(|pair| {
        pair[0].predicted_junction_id >= pair[1].predicted_junction_id
    }) {
        return Err(MycelialReconstructionCalibrationError::InvalidReport(
            "calibration observations are not uniquely sorted by prediction ID",
        ));
    }
    let mut samples = Vec::with_capacity(observations.len());
    for observation in observations {
        let sum = observation.probabilities.iter().sum::<f64>();
        if !valid_id(&observation.predicted_junction_id)
            || observation
                .probabilities
                .iter()
                .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
            || !sum.is_finite()
            || (sum - 1.0).abs() > probability_sum_tolerance
        {
            return Err(MycelialReconstructionCalibrationError::InvalidReport(
                "calibration observation has an invalid probability vector",
            ));
        }
        let predicted_class = top_class(&observation.probabilities);
        samples.push(CalibrationSample {
            prediction_id: observation.predicted_junction_id.clone(),
            probabilities: observation.probabilities,
            target: observation.target_class,
            predicted_class,
            confidence: observation.probabilities[predicted_class.index()],
        });
    }
    Ok(samples)
}

fn calculate_counts(samples: &[CalibrationSample]) -> ReconstructionCalibrationCounts {
    let matched_prediction_count = samples
        .iter()
        .filter(|sample| sample.target != ReconstructionCalibrationClass::NoObject)
        .count();
    let correct_top_label_count = samples
        .iter()
        .filter(|sample| sample.predicted_class == sample.target)
        .count();
    let predicted_anastomosis_count = samples
        .iter()
        .filter(|sample| sample.predicted_class == ReconstructionCalibrationClass::Anastomosis)
        .count();
    let false_anastomosis_prediction_count = samples
        .iter()
        .filter(|sample| {
            sample.predicted_class == ReconstructionCalibrationClass::Anastomosis
                && sample.target != ReconstructionCalibrationClass::Anastomosis
        })
        .count();
    ReconstructionCalibrationCounts {
        prediction_count: samples.len(),
        matched_prediction_count,
        unmatched_prediction_count: samples.len().saturating_sub(matched_prediction_count),
        correct_top_label_count,
        top_label_error_count: samples.len().saturating_sub(correct_top_label_count),
        predicted_anastomosis_count,
        false_anastomosis_prediction_count,
    }
}

fn empty_bins(bin_count: usize) -> Vec<ReconstructionCalibrationBin> {
    (0..bin_count)
        .map(|bin_index| ReconstructionCalibrationBin {
            bin_index,
            sample_count: 0,
            mean_confidence: None,
            empirical_frequency: None,
            absolute_calibration_gap: None,
        })
        .collect()
}

fn calibration_bin_index(confidence: f64, bin_count: usize) -> usize {
    ((confidence * bin_count as f64).floor() as usize).min(bin_count - 1)
}

fn class_calibration(
    class: ReconstructionCalibrationClass,
    samples: &[CalibrationSample],
    bin_count: usize,
) -> ReconstructionClassCalibration {
    let mut bins = empty_bins(bin_count);
    let mut squared_error = 0.0;
    for sample in samples {
        let confidence = sample.probabilities[class.index()];
        let outcome = if sample.target == class { 1.0 } else { 0.0 };
        squared_error += (confidence - outcome).powi(2);
        let bin = &mut bins[calibration_bin_index(confidence, bin_count)];
        bin.sample_count += 1;
        bin.mean_confidence = Some(bin.mean_confidence.unwrap_or(0.0) + confidence);
        bin.empirical_frequency = Some(
            bin.empirical_frequency.unwrap_or(0.0) + outcome,
        );
    }
    let mut ece = 0.0;
    for bin in &mut bins {
        if bin.sample_count == 0 {
            continue;
        }
        let mean_confidence = bin.mean_confidence.unwrap() / bin.sample_count as f64;
        let empirical_frequency =
            bin.empirical_frequency.unwrap() / bin.sample_count as f64;
        let gap = (mean_confidence - empirical_frequency).abs();
        bin.mean_confidence = Some(mean_confidence);
        bin.empirical_frequency = Some(empirical_frequency);
        bin.absolute_calibration_gap = Some(gap);
        ece += bin.sample_count as f64 / samples.len() as f64 * gap;
    }
    ReconstructionClassCalibration {
        class,
        brier_score: (!samples.is_empty())
            .then_some(squared_error / samples.len() as f64),
        expected_calibration_error: (!samples.is_empty()).then_some(ece),
        bins,
    }
}

fn top_label_calibration(
    samples: &[CalibrationSample],
    bin_count: usize,
    epsilon: f64,
) -> ReconstructionTopLabelCalibration {
    let mut bins = empty_bins(bin_count);
    let mut brier = 0.0;
    let mut nll = 0.0;
    let mut correct = 0usize;
    for sample in samples {
        let is_correct = sample.predicted_class == sample.target;
        correct += if is_correct { 1 } else { 0 };
        brier += (sample.confidence - if is_correct { 1.0 } else { 0.0 }).powi(2);
        nll -= sample.probabilities[sample.target.index()]
            .max(epsilon)
            .ln();
        let bin = &mut bins[calibration_bin_index(sample.confidence, bin_count)];
        bin.sample_count += 1;
        bin.mean_confidence = Some(bin.mean_confidence.unwrap_or(0.0) + sample.confidence);
        bin.empirical_frequency = Some(
            bin.empirical_frequency.unwrap_or(0.0) + if is_correct { 1.0 } else { 0.0 },
        );
    }
    let mut ece = 0.0;
    for bin in &mut bins {
        if bin.sample_count == 0 {
            continue;
        }
        let mean_confidence = bin.mean_confidence.unwrap() / bin.sample_count as f64;
        let empirical_frequency =
            bin.empirical_frequency.unwrap() / bin.sample_count as f64;
        let gap = (mean_confidence - empirical_frequency).abs();
        bin.mean_confidence = Some(mean_confidence);
        bin.empirical_frequency = Some(empirical_frequency);
        bin.absolute_calibration_gap = Some(gap);
        ece += bin.sample_count as f64 / samples.len() as f64 * gap;
    }
    ReconstructionTopLabelCalibration {
        brier_score: (!samples.is_empty()).then_some(brier / samples.len() as f64),
        expected_calibration_error: (!samples.is_empty()).then_some(ece),
        negative_log_likelihood: (!samples.is_empty()).then_some(nll / samples.len() as f64),
        accuracy: (!samples.is_empty())
            .then_some(correct as f64 / samples.len() as f64),
        bins,
    }
}

fn selective_risk_curve(samples: &[CalibrationSample]) -> Vec<ReconstructionSelectiveRiskPoint> {
    let mut ordered: Vec<_> = samples.iter().collect();
    ordered.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| left.prediction_id.cmp(&right.prediction_id))
    });
    let mut curve = Vec::new();
    let mut cursor = 0usize;
    let mut errors = 0usize;
    let mut predicted_anastomosis = 0usize;
    let mut false_anastomosis = 0usize;
    while cursor < ordered.len() {
        let threshold = ordered[cursor].confidence;
        while cursor < ordered.len() && ordered[cursor].confidence == threshold {
            let sample = ordered[cursor];
            errors += if sample.predicted_class != sample.target { 1 } else { 0 };
            if sample.predicted_class == ReconstructionCalibrationClass::Anastomosis {
                predicted_anastomosis += 1;
                false_anastomosis += if sample.target != ReconstructionCalibrationClass::Anastomosis {
                    1
                } else {
                    0
                };
            }
            cursor += 1;
        }
        let retained = cursor;
        curve.push(ReconstructionSelectiveRiskPoint {
            minimum_included_confidence: threshold,
            retained_prediction_count: retained,
            coverage: retained as f64 / ordered.len() as f64,
            classification_error_count: errors,
            selective_risk: errors as f64 / retained as f64,
            predicted_anastomosis_count: predicted_anastomosis,
            false_anastomosis_prediction_count: false_anastomosis,
            false_anastomosis_rate: (predicted_anastomosis > 0)
                .then_some(false_anastomosis as f64 / predicted_anastomosis as f64),
        });
    }
    curve
}

impl MycelialReconstructionCalibrationReport {
    pub fn validate(&self) -> Result<(), MycelialReconstructionCalibrationError> {
        if self.schema_version != MYCELIAL_RECONSTRUCTION_CALIBRATION_SCHEMA_VERSION
            || !valid_digest(&self.source_observation_digest)
            || !valid_digest(&self.ground_truth_graph_digest)
            || !valid_digest(&self.prediction_graph_digest)
            || !valid_digest(&self.reconstruction_score_digest)
            || !valid_digest(&self.probability_inventory_digest)
            || !valid_id(&self.matching_policy_id)
            || !valid_id(&self.calibration_policy_id)
            || self.bin_count == 0
            || self.bin_count > MAX_RECONSTRUCTION_CALIBRATION_BINS
            || !self.probability_sum_tolerance.is_finite()
            || !(1.0e-12..=1.0e-2).contains(&self.probability_sum_tolerance)
            || !self.negative_log_likelihood_epsilon.is_finite()
            || !(1.0e-15..=1.0e-3).contains(&self.negative_log_likelihood_epsilon)
            || self.counts.matched_prediction_count + self.counts.unmatched_prediction_count
                != self.counts.prediction_count
            || self.counts.correct_top_label_count + self.counts.top_label_error_count
                != self.counts.prediction_count
            || self.counts.false_anastomosis_prediction_count
                > self.counts.predicted_anastomosis_count
            || self.class_prediction_counts.len() != RECONSTRUCTION_CALIBRATION_CLASS_COUNT
            || self
                .class_prediction_counts
                .iter()
                .zip(ReconstructionCalibrationClass::ALL)
                .any(|((class, _), expected)| *class != expected)
            || self.class_prediction_counts.iter().map(|(_, count)| *count).sum::<usize>()
                != self.counts.prediction_count
            || self
                .class_prediction_counts
                .iter()
                .find(|(class, _)| *class == ReconstructionCalibrationClass::Anastomosis)
                .map(|(_, count)| *count)
                != Some(self.counts.predicted_anastomosis_count)
            || self.class_calibration.len() != RECONSTRUCTION_CALIBRATION_CLASS_COUNT
            || self
                .class_calibration
                .iter()
                .zip(ReconstructionCalibrationClass::ALL)
                .any(|(metric, expected)| {
                    metric.class != expected
                        || metric.bins.len() != self.bin_count
                        || !valid_calibration_bins(&metric.bins)
                        || metric.bins.iter().map(|bin| bin.sample_count).sum::<usize>()
                        != self.counts.prediction_count
                        || (self.counts.prediction_count == 0)
                            != metric.brier_score.is_none()
                        || (self.counts.prediction_count == 0)
                            != metric.expected_calibration_error.is_none()
                    || metric.brier_score.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
                        || metric
                            .expected_calibration_error
                            .is_some_and(|value| !value.is_finite() || value < 0.0)
                })
            || self.top_label_calibration.bins.len() != self.bin_count
            || !valid_calibration_bins(&self.top_label_calibration.bins)
            || self
                .top_label_calibration
                .bins
                .iter()
                .map(|bin| bin.sample_count)
                .sum::<usize>()
                != self.counts.prediction_count
            || (self.counts.prediction_count == 0)
                != self.top_label_calibration.brier_score.is_none()
            || (self.counts.prediction_count == 0)
                != self.top_label_calibration.accuracy.is_none()
            || (self.counts.prediction_count == 0)
                != self.top_label_calibration.negative_log_likelihood.is_none()
            || (self.counts.prediction_count == 0)
                != self.top_label_calibration.expected_calibration_error.is_none()
            || self.top_label_calibration.brier_score.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
            || self.top_label_calibration.expected_calibration_error.is_some_and(|value| !value.is_finite() || value < 0.0)
            || self.top_label_calibration.negative_log_likelihood.is_some_and(|value| !value.is_finite() || value < 0.0)
            || self.top_label_calibration.accuracy.is_some_and(|value| !(0.0..=1.0).contains(&value))
        {
            return Err(MycelialReconstructionCalibrationError::InvalidReport(
                "metadata, sample totals or calibration metrics are invalid",
            ));
        }
        if self.observations.len() != self.counts.prediction_count
            || probability_inventory_digest(&self.observations) != self.probability_inventory_digest
        {
            return Err(MycelialReconstructionCalibrationError::InvalidReport(
                "probability inventory does not match its digest or prediction count",
            ));
        }
        let samples = samples_from_observations(
            &self.observations,
            self.probability_sum_tolerance,
        )?;
        let expected_counts = calculate_counts(&samples);
        let expected_class_prediction_counts = ReconstructionCalibrationClass::ALL
            .into_iter()
            .map(|class| {
                (
                    class,
                    samples
                        .iter()
                        .filter(|sample| sample.predicted_class == class)
                        .count(),
                )
            })
            .collect::<Vec<_>>();
        let expected_class_calibration = ReconstructionCalibrationClass::ALL
            .into_iter()
            .map(|class| class_calibration(class, &samples, self.bin_count))
            .collect::<Vec<_>>();
        let expected_top_label_calibration = top_label_calibration(
            &samples,
            self.bin_count,
            self.negative_log_likelihood_epsilon,
        );
        let expected_selective_risk_curve = selective_risk_curve(&samples);
        if self.counts != expected_counts
            || self.class_prediction_counts != expected_class_prediction_counts
            || self.class_calibration != expected_class_calibration
            || self.top_label_calibration != expected_top_label_calibration
            || self.selective_risk_curve != expected_selective_risk_curve
        {
            return Err(MycelialReconstructionCalibrationError::InvalidReport(
                "calibration aggregates cannot be reproduced from the retained observations",
            ));
        }
        let mut previous_coverage = 0.0;
        let mut previous_threshold = f64::INFINITY;
        let mut previous_errors = 0usize;
        let mut previous_false_anastomosis = 0usize;
        for point in &self.selective_risk_curve {
            let expected_coverage = (self.counts.prediction_count > 0).then_some(
                point.retained_prediction_count as f64 / self.counts.prediction_count as f64,
            );
            let expected_risk = (point.retained_prediction_count > 0).then_some(
                point.classification_error_count as f64
                    / point.retained_prediction_count as f64,
            );
            let expected_false_anastomosis_rate = (point.predicted_anastomosis_count > 0)
                .then_some(
                    point.false_anastomosis_prediction_count as f64
                        / point.predicted_anastomosis_count as f64,
                );
            if !point.minimum_included_confidence.is_finite()
                || !(0.0..=1.0).contains(&point.minimum_included_confidence)
                || point.minimum_included_confidence >= previous_threshold
                || point.retained_prediction_count == 0
                || point.retained_prediction_count > self.counts.prediction_count
                || expected_coverage != Some(point.coverage)
                || point.coverage <= previous_coverage
                || !point.selective_risk.is_finite()
                || expected_risk != Some(point.selective_risk)
                || point.classification_error_count < previous_errors
                || point.classification_error_count > point.retained_prediction_count
                || point.false_anastomosis_prediction_count < previous_false_anastomosis
                || point.false_anastomosis_prediction_count
                    > point.predicted_anastomosis_count
                || point.false_anastomosis_rate != expected_false_anastomosis_rate
            {
                return Err(MycelialReconstructionCalibrationError::InvalidReport(
                    "selective-risk curve is not monotone or has invalid rates",
                ));
            }
            previous_coverage = point.coverage;
            previous_threshold = point.minimum_included_confidence;
            previous_errors = point.classification_error_count;
            previous_false_anastomosis = point.false_anastomosis_prediction_count;
        }
        if self.counts.prediction_count == 0 && !self.selective_risk_curve.is_empty()
            || self.counts.prediction_count > 0
                && self
                    .selective_risk_curve
                    .last()
                    .is_none_or(|point| {
                        point.retained_prediction_count != self.counts.prediction_count
                            || point.classification_error_count != self.counts.top_label_error_count
                            || point.predicted_anastomosis_count
                                != self.counts.predicted_anastomosis_count
                            || point.false_anastomosis_prediction_count
                                != self.counts.false_anastomosis_prediction_count
                    })
        {
            return Err(MycelialReconstructionCalibrationError::InvalidReport(
                "selective-risk curve does not terminate at full prediction coverage",
            ));
        }
        Ok(())
    }

    pub fn canonical_digest_hex(
        &self,
    ) -> Result<String, MycelialReconstructionCalibrationError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-reconstruction-calibration-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        for digest in [
            &self.source_observation_digest,
            &self.ground_truth_graph_digest,
            &self.prediction_graph_digest,
            &self.reconstruction_score_digest,
            &self.probability_inventory_digest,
        ] {
            hasher.update(digest.as_bytes());
        }
        hash_str(&mut hasher, &self.matching_policy_id);
        hash_str(&mut hasher, &self.calibration_policy_id);
        hasher.update(&(self.bin_count as u64).to_le_bytes());
        hasher.update(&self.probability_sum_tolerance.to_bits().to_le_bytes());
        hasher.update(&self.negative_log_likelihood_epsilon.to_bits().to_le_bytes());
        for value in [
            self.counts.prediction_count,
            self.counts.matched_prediction_count,
            self.counts.unmatched_prediction_count,
            self.counts.correct_top_label_count,
            self.counts.top_label_error_count,
            self.counts.predicted_anastomosis_count,
            self.counts.false_anastomosis_prediction_count,
        ] {
            hasher.update(&(value as u64).to_le_bytes());
        }
        for (class, count) in &self.class_prediction_counts {
            hasher.update(&[class.index() as u8]);
            hasher.update(&(*count as u64).to_le_bytes());
        }
        for metric in &self.class_calibration {
            hasher.update(&[metric.class.index() as u8]);
            hash_optional_f64(&mut hasher, metric.brier_score);
            hash_optional_f64(&mut hasher, metric.expected_calibration_error);
            hash_bins(&mut hasher, &metric.bins);
        }
        hash_optional_f64(&mut hasher, self.top_label_calibration.brier_score);
        hash_optional_f64(
            &mut hasher,
            self.top_label_calibration.expected_calibration_error,
        );
        hash_optional_f64(
            &mut hasher,
            self.top_label_calibration.negative_log_likelihood,
        );
        hash_optional_f64(&mut hasher, self.top_label_calibration.accuracy);
        hash_bins(&mut hasher, &self.top_label_calibration.bins);
        for point in &self.selective_risk_curve {
            hasher.update(&point.minimum_included_confidence.to_bits().to_le_bytes());
            hasher.update(&(point.retained_prediction_count as u64).to_le_bytes());
            hasher.update(&point.coverage.to_bits().to_le_bytes());
            hasher.update(&(point.classification_error_count as u64).to_le_bytes());
            hasher.update(&point.selective_risk.to_bits().to_le_bytes());
            hasher.update(&(point.predicted_anastomosis_count as u64).to_le_bytes());
            hasher.update(&(point.false_anastomosis_prediction_count as u64).to_le_bytes());
            hash_optional_f64(&mut hasher, point.false_anastomosis_rate);
        }
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

fn probability_inventory_digest(
    observations: &[JunctionCalibrationObservation],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-reconstruction-probability-inventory-v1");
    hasher.update(&(observations.len() as u64).to_le_bytes());
    for observation in observations {
        hash_str(&mut hasher, &observation.predicted_junction_id);
        hasher.update(&[observation.target_class.index() as u8]);
        for probability in observation.probabilities {
            hasher.update(&probability.to_bits().to_le_bytes());
        }
    }
    hex_digest(hasher.finalize().as_bytes())
}

fn valid_calibration_bins(bins: &[ReconstructionCalibrationBin]) -> bool {
    bins.iter().enumerate().all(|(index, bin)| {
        if bin.bin_index != index {
            return false;
        }
        if bin.sample_count == 0 {
            return bin.mean_confidence.is_none()
                && bin.empirical_frequency.is_none()
                && bin.absolute_calibration_gap.is_none();
        }
        let (Some(confidence), Some(frequency), Some(gap)) = (
            bin.mean_confidence,
            bin.empirical_frequency,
            bin.absolute_calibration_gap,
        ) else {
            return false;
        };
        confidence.is_finite()
            && (0.0..=1.0).contains(&confidence)
            && frequency.is_finite()
            && (0.0..=1.0).contains(&frequency)
            && gap.is_finite()
            && (0.0..=1.0).contains(&gap)
            && (gap - (confidence - frequency).abs()).abs() <= 1.0e-12
    })
}

fn hash_bins(hasher: &mut blake3::Hasher, bins: &[ReconstructionCalibrationBin]) {
    for bin in bins {
        hasher.update(&(bin.bin_index as u64).to_le_bytes());
        hasher.update(&(bin.sample_count as u64).to_le_bytes());
        hash_optional_f64(hasher, bin.mean_confidence);
        hash_optional_f64(hasher, bin.empirical_frequency);
        hash_optional_f64(hasher, bin.absolute_calibration_gap);
    }
}

fn valid_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= 128 && !value.chars().any(char::is_control)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_optional_f64(hasher: &mut blake3::Hasher, value: Option<f64>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_bits().to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        use fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_reconstruction_metrics::{
        AnnotatedHyphaSegment, AnnotatedJunction, AnnotatedJunctionKind,
        JunctionIdentityMatch, MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
        ReconstructionDimensionality, ReconstructionEvaluationConfig,
        ReconstructionLengthUnit, ReconstructionPoint3, evaluate_mycelial_reconstruction,
        match_junctions_hungarian,
    };

    fn graph(
        producer: &str,
        nodes: Vec<AnnotatedJunction>,
        segments: Vec<AnnotatedHyphaSegment>,
    ) -> AnnotatedMycelialSkeleton {
        AnnotatedMycelialSkeleton {
            schema_version: MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
            frame_id: "frame-calibration-test-001".to_string(),
            coordinate_frame_id: "microscope-stage-v1".to_string(),
            calibration_digest: "c".repeat(64),
            source_observation_digest: "d".repeat(64),
            dimensionality: ReconstructionDimensionality::TwoD,
            length_unit: ReconstructionLengthUnit::Micrometer,
            producer_id: producer.to_string(),
            producer_version: "fixture-v1".to_string(),
            junctions: nodes,
            segments,
        }
    }

    fn node(id: &str, kind: AnnotatedJunctionKind, x: f64, y: f64) -> AnnotatedJunction {
        AnnotatedJunction {
            id: id.to_string(),
            kind,
            position: ReconstructionPoint3 { x, y, z: 0.0 },
        }
    }

    fn probabilities(
        id: &str,
        target: ReconstructionCalibrationClass,
        confidence: f64,
    ) -> JunctionClassProbabilityPrediction {
        let mut values = [0.0; RECONSTRUCTION_CALIBRATION_CLASS_COUNT];
        let other = 1.0 - confidence;
        values[target.index()] = confidence;
        values[ReconstructionCalibrationClass::Uncertain.index()] = other;
        JunctionClassProbabilityPrediction {
            predicted_junction_id: id.to_string(),
            probabilities: values,
        }
    }

    fn fixture() -> (
        AnnotatedMycelialSkeleton,
        AnnotatedMycelialSkeleton,
        MycelialReconstructionScore,
    ) {
        let truth = graph(
            "human",
            vec![
                node("A", AnnotatedJunctionKind::Branch, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Tip, 10.0, 0.0),
                node("C", AnnotatedJunctionKind::ProjectedCrossing, 0.0, 10.0),
            ],
            vec![
                AnnotatedHyphaSegment {
                    id: "ab".to_string(),
                    start_junction_id: "A".to_string(),
                    end_junction_id: "B".to_string(),
                    length: 10.0,
                },
                AnnotatedHyphaSegment {
                    id: "ac".to_string(),
                    start_junction_id: "A".to_string(),
                    end_junction_id: "C".to_string(),
                    length: 10.0,
                },
            ],
        );
        let prediction = graph(
            "model",
            vec![
                node("pA", AnnotatedJunctionKind::Branch, 0.01, 0.0),
                node("pB", AnnotatedJunctionKind::Tip, 10.01, 0.0),
                node("pC", AnnotatedJunctionKind::Anastomosis, 0.0, 10.01),
            ],
            vec![
                AnnotatedHyphaSegment {
                    id: "pab".to_string(),
                    start_junction_id: "pA".to_string(),
                    end_junction_id: "pB".to_string(),
                    length: 10.0,
                },
                AnnotatedHyphaSegment {
                    id: "pac".to_string(),
                    start_junction_id: "pA".to_string(),
                    end_junction_id: "pC".to_string(),
                    length: 10.0,
                },
                AnnotatedHyphaSegment {
                    id: "false-fusion".to_string(),
                    start_junction_id: "pB".to_string(),
                    end_junction_id: "pC".to_string(),
                    length: 14.14,
                },
            ],
        );
        let matches = match_junctions_hungarian(&truth, &prediction, 0.1).unwrap();
        let score = evaluate_mycelial_reconstruction(
            &truth,
            &prediction,
            "hungarian-v1",
            &matches,
            ReconstructionEvaluationConfig {
                maximum_junction_match_distance: 0.1,
                ..ReconstructionEvaluationConfig::default()
            },
        )
        .unwrap();
        (truth, prediction, score)
    }

    #[test]
    fn calibration_report_separates_false_fusion_from_detection_and_binds_receipt() {
        let (truth, prediction, score) = fixture();
        let probabilities = vec![
            probabilities("pA", ReconstructionCalibrationClass::Branch, 0.95),
            probabilities("pB", ReconstructionCalibrationClass::Tip, 0.95),
            probabilities("pC", ReconstructionCalibrationClass::Anastomosis, 0.99),
        ];
        let report = evaluate_mycelial_reconstruction_calibration(
            &truth,
            &prediction,
            &score,
            "synthetic-calibration-v1",
            &probabilities,
            ReconstructionCalibrationConfig::default(),
        )
        .unwrap();
        assert_eq!(report.counts.prediction_count, 3);
        assert_eq!(report.counts.matched_prediction_count, 3);
        assert_eq!(report.counts.false_anastomosis_prediction_count, 1);
        assert_eq!(report.counts.predicted_anastomosis_count, 1);
        assert!(report.top_label_calibration.negative_log_likelihood.unwrap() > 0.0);
        assert_eq!(report.selective_risk_curve.last().unwrap().coverage, 1.0);
        assert_eq!(
            report.canonical_digest_hex().unwrap(),
            report.canonical_digest_hex().unwrap()
        );
    }

    #[test]
    fn calibrated_correct_predictions_have_lower_brier_than_overconfident_false_fusion() {
        let (truth, prediction, score) = fixture();
        let calibrated = vec![
            probabilities("pA", ReconstructionCalibrationClass::Branch, 0.90),
            probabilities("pB", ReconstructionCalibrationClass::Tip, 0.90),
            probabilities("pC", ReconstructionCalibrationClass::ProjectedCrossing, 0.90),
        ];
        let overconfident = vec![
            probabilities("pA", ReconstructionCalibrationClass::Branch, 0.90),
            probabilities("pB", ReconstructionCalibrationClass::Tip, 0.90),
            probabilities("pC", ReconstructionCalibrationClass::Anastomosis, 0.999),
        ];
        let good = evaluate_mycelial_reconstruction_calibration(
            &truth,
            &prediction,
            &score,
            "test-calibration",
            &calibrated,
            ReconstructionCalibrationConfig::default(),
        )
        .unwrap();
        let bad = evaluate_mycelial_reconstruction_calibration(
            &truth,
            &prediction,
            &score,
            "test-calibration",
            &overconfident,
            ReconstructionCalibrationConfig::default(),
        )
        .unwrap();
        assert!(
            good.top_label_calibration.brier_score.unwrap()
                < bad.top_label_calibration.brier_score.unwrap()
        );
        assert!(
            good.top_label_calibration.negative_log_likelihood.unwrap()
                < bad.top_label_calibration.negative_log_likelihood.unwrap()
        );
        assert_eq!(bad.counts.false_anastomosis_prediction_count, 1);
    }

    #[test]
    fn abstention_reduces_risk_and_ties_are_not_split() {
        let (truth, prediction, score) = fixture();
        let values = vec![
            probabilities("pA", ReconstructionCalibrationClass::Branch, 0.99),
            probabilities("pB", ReconstructionCalibrationClass::Tip, 0.95),
            probabilities("pC", ReconstructionCalibrationClass::Anastomosis, 0.55),
        ];
        let report = evaluate_mycelial_reconstruction_calibration(
            &truth,
            &prediction,
            &score,
            "abstention-policy-v1",
            &values,
            ReconstructionCalibrationConfig::default(),
        )
        .unwrap();
        assert_eq!(report.selective_risk_curve.len(), 3);
        assert_eq!(report.selective_risk_curve[0].retained_prediction_count, 1);
        assert_eq!(report.selective_risk_curve[0].selective_risk, 0.0);
        assert_eq!(report.selective_risk_curve[2].retained_prediction_count, 3);
        assert!(report.selective_risk_curve[2].selective_risk > 0.0);
    }

    #[test]
    fn invalid_probability_inventory_and_sum_fail_closed() {
        let (truth, prediction, score) = fixture();
        let mut values = vec![
            probabilities("pA", ReconstructionCalibrationClass::Branch, 0.9),
            probabilities("pB", ReconstructionCalibrationClass::Tip, 0.9),
            probabilities("pC", ReconstructionCalibrationClass::ProjectedCrossing, 0.9),
        ];
        values[0].probabilities[0] = 0.5;
        assert!(matches!(
            evaluate_mycelial_reconstruction_calibration(
                &truth,
                &prediction,
                &score,
                "invalid-probability",
                &values,
                ReconstructionCalibrationConfig::default(),
            ),
            Err(MycelialReconstructionCalibrationError::InvalidProbabilityVector(_))
        ));
        values.pop();
        assert_eq!(
            evaluate_mycelial_reconstruction_calibration(
                &truth,
                &prediction,
                &score,
                "missing-probability",
                &values,
                ReconstructionCalibrationConfig::default(),
            ),
            Err(MycelialReconstructionCalibrationError::PredictionInventoryMismatch)
        );
    }

    #[test]
    fn prediction_order_permutation_preserves_calibration_receipt_digest() {
        let (truth, prediction, score) = fixture();
        let values = vec![
            probabilities("pA", ReconstructionCalibrationClass::Branch, 0.90),
            probabilities("pB", ReconstructionCalibrationClass::Tip, 0.90),
            probabilities("pC", ReconstructionCalibrationClass::ProjectedCrossing, 0.90),
        ];
        let first = evaluate_mycelial_reconstruction_calibration(
            &truth,
            &prediction,
            &score,
            "order-invariant-v1",
            &values,
            ReconstructionCalibrationConfig::default(),
        )
        .unwrap();
        let mut reversed = values;
        reversed.reverse();
        let second = evaluate_mycelial_reconstruction_calibration(
            &truth,
            &prediction,
            &score,
            "order-invariant-v1",
            &reversed,
            ReconstructionCalibrationConfig::default(),
        )
        .unwrap();
        assert_eq!(
            first.canonical_digest_hex().unwrap(),
            second.canonical_digest_hex().unwrap()
        );
    }
}
