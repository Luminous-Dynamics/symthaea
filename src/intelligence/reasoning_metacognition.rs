// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RQ-006 metacognition qualification primitives.
//!
//! This module measures externally auditable metacognitive behavior. It does not define how a
//! reasoner should generate confidence and does not treat confidence as evidence merely because
//! it lies in `[0, 1]`.
//!
//! The evaluator keeps separate:
//! - calibration of predicted correctness,
//! - discrimination between correct and incorrect episodes,
//! - selective risk / abstention behavior,
//! - detection of weak assumptions,
//! - confidence revision after new evidence, and
//! - whether a confidence report is actually bound to the current episode.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fmt;

pub const METACOGNITION_EVALUATOR_VERSION: &str = "rq-006-metacognition-v6";
pub const CORRECTNESS_FORECAST_SCHEMA_VERSION: u32 = 1;
pub const CORRECTNESS_OUTCOME_SCHEMA_VERSION: u32 = 1;
pub const FROZEN_FORECAST_SET_SCHEMA_VERSION: u32 = 2;
pub const FORECAST_BASELINE_SCHEMA_VERSION: u32 = 1;
pub const FORECAST_BASELINE_COMPARISON_SCHEMA_VERSION: u32 = 1;
pub const FORECAST_OUTCOME_BINDING_REPORT_SCHEMA_VERSION: u32 = 1;
const LOG_LOSS_EPSILON: f64 = 1.0e-15;
const SELECTIVE_RISK_FAMILYWISE_ALPHA: f64 = 0.05;
const SELECTIVE_RISK_BOUND_METHOD: &str = "hoeffding-familywise-95-v1";
const SELECTIVE_RISK_BOUND_ASSUMPTIONS: &str = concat!(
    "IID evaluation episodes; frozen scoring and selection rule; ",
    "supplied threshold set predeclared before correctness outcomes; ",
    "task-family taxonomy and equal-width confidence bins predeclared and outcome-independent; ",
    "simultaneous bounds cover pooled and observed task-family by threshold and bin comparisons; ",
    "no distribution-shift guarantee",
);

/// One pre-outcome prediction of whether the subject's answer is correct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrectnessPrediction {
    /// Episode whose correctness is being evaluated.
    pub episode_id: String,
    /// Stable task-family identifier from a taxonomy frozen before outcomes are inspected.
    /// Do not assign or revise this label using correctness outcomes.
    pub task_family_id: String,
    /// Episode the confidence value claims to describe. A mismatch exposes stale/cross-episode
    /// confidence rather than silently attributing it to the current episode.
    pub confidence_target_episode_id: String,
    /// Predicted probability that an asserted answer would be correct.
    pub confidence: f64,
    /// Ground-truth correctness supplied by the benchmark after the prediction is frozen.
    pub correct: bool,
    /// Whether the subject actually asserted an answer rather than abstaining.
    pub asserted: bool,
}

/// Binary ground truth for whether a materially weak assumption was present and detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeakAssumptionObservation {
    pub episode_id: String,
    pub weak_assumption_present: bool,
    pub weak_assumption_detected: bool,
}

/// Direction in which confidence should move after independently specified new evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceRevisionDirection {
    Increase,
    Decrease,
    Stable,
}

/// Paired confidence observation before and after new evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceRevisionObservation {
    pub pair_id: String,
    pub before_episode_id: String,
    pub after_episode_id: String,
    pub before_confidence: f64,
    pub after_confidence: f64,
    pub expected_direction: ConfidenceRevisionDirection,
    /// Absolute delta considered observationally stable for `Stable` expectations.
    pub stable_tolerance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SelectiveRiskPoint {
    pub threshold: f64,
    /// Fraction of all evaluated episodes on which confidence meets the threshold and the subject
    /// actually asserted an answer.
    pub coverage: f64,
    /// Empirical error rate conditional on selection. `None` when no episode is selected.
    pub risk: Option<f64>,
    /// Conservative one-sided 95% family-wise upper bound over the evaluator's supplied,
    /// predeclared threshold list. Uses Hoeffding + Bonferroni and assumes IID evaluation
    /// episodes and a frozen score/selection rule. `None` when none are selected.
    /// The bound does not cover thresholds searched outside that list and cannot repair shift.
    pub risk_upper_bound_95: Option<f64>,
    pub selected: usize,
}

/// One fixed-width confidence bin with outcome frequency and uncertainty interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBinReport {
    pub bin_index: usize,
    /// Inclusive lower endpoint of the confidence interval represented by this bin.
    pub confidence_lower: f64,
    /// Upper endpoint; exclusive except that the final bin includes confidence 1.0.
    pub confidence_upper: f64,
    pub episodes: usize,
    pub mean_confidence: Option<f64>,
    pub empirical_accuracy: Option<f64>,
    /// Conservative family-wise adjusted two-sided Hoeffding interval for accuracy in this bin.
    /// None when the bin is empty.
    pub accuracy_lower_95: Option<f64>,
    pub accuracy_upper_95: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryDetectionReport {
    pub observations: usize,
    pub true_positive: usize,
    pub false_positive: usize,
    pub true_negative: usize,
    pub false_negative: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub accuracy: Option<f64>,
}

/// Calibration report scoped to one predeclared task family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskFamilyCalibrationReport {
    pub task_family_id: String,
    pub episodes: usize,
    pub empirical_accuracy: Option<f64>,
    pub mean_confidence: Option<f64>,
    pub brier_score: Option<f64>,
    pub log_loss: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub maximum_calibration_error: Option<f64>,
    /// Mean confidence minus accuracy; positive means overconfidence in this family.
    pub confidence_bias: Option<f64>,
    pub correctness_auroc: Option<f64>,
    pub current_episode_binding_rate: Option<f64>,
    pub assertion_rate: Option<f64>,
    pub asserted_risk: Option<f64>,
    /// Group-local coverage and risk; upper bounds share family-wise correction
    /// with pooled results.
    pub selective_risk: Vec<SelectiveRiskPoint>,
    /// Equal-width reliability bins with simultaneous accuracy intervals.
    pub calibration_bins: Vec<CalibrationBinReport>,
}

/// Decomposed RQ-006 measurement report. There is intentionally no single "metacognition score".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetacognitionReport {
    pub evaluator_version: String,
    pub predictions: usize,
    pub empirical_accuracy: Option<f64>,
    pub mean_confidence: Option<f64>,
    pub brier_score: Option<f64>,
    pub log_loss: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub maximum_calibration_error: Option<f64>,
    /// Mean confidence minus empirical accuracy. Positive values indicate overconfidence.
    pub confidence_bias: Option<f64>,
    /// Rank discrimination: probability that a randomly chosen correct episode receives higher
    /// confidence than a randomly chosen incorrect episode, with ties worth 0.5.
    pub correctness_auroc: Option<f64>,
    /// Fraction whose confidence target is the episode currently being evaluated.
    pub current_episode_binding_rate: Option<f64>,
    /// Actual assertion rate, independent of correctness.
    pub assertion_rate: Option<f64>,
    /// Error rate among actually asserted answers.
    pub asserted_risk: Option<f64>,
    /// Fraction of abstentions that would have been correct if answered; useful for distinguishing
    /// prudent abstention from indiscriminate refusal on benchmark tasks with known answers.
    pub abstention_opportunity_cost: Option<f64>,
    /// Machine-readable method identity for simultaneous selective-risk upper bounds.
    pub selective_risk_bound_method: String,
    /// Explicit assumptions/limitations shipped with the report for downstream consumers.
    pub selective_risk_bound_assumptions: String,
    /// Deterministically sorted, group-local calibration results; never averaged into a scalar.
    pub task_family_calibration: Vec<TaskFamilyCalibrationReport>,
    /// Pooled equal-width reliability bins with simultaneous accuracy intervals.
    pub calibration_bins: Vec<CalibrationBinReport>,
    pub selective_risk: Vec<SelectiveRiskPoint>,
    pub weak_assumption_detection: BinaryDetectionReport,
    pub confidence_revisions: usize,
    pub revision_direction_accuracy: Option<f64>,
    /// Mean signed movement in the expected direction. Positive is favorable; stable pairs use
    /// negative excess movement outside tolerance as their signed contribution.
    pub mean_expected_revision_delta: Option<f64>,
    /// Present only when the two-phase forecast/outcome binding API was used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forecast_outcome_binding: Option<ForecastOutcomeBindingReport>,
    /// Present only when baselines were frozen on a separate calibration split.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_comparison: Option<ForecastBaselineComparisonReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetacognitionEvaluationError {
    InvalidBinCount(usize),
    EmptyEpisodeId(&'static str),
    DuplicatePredictionEpisode(String),
    EmptyTaskFamilyId { episode_id: String },
    DuplicateAssumptionEpisode(String),
    DuplicateRevisionPair(String),
    InvalidProbability { field: &'static str, value: f64 },
    InvalidTolerance(f64),
    InvalidThreshold(f64),
    EmptyForecastSet,
    EmptyForecastField { forecast_id: String, field: &'static str },
    EmptyOutcomeField { forecast_id: String, field: &'static str },
    UnsupportedForecastSchemaVersion(u32),
    UnsupportedOutcomeSchemaVersion(u32),
    UnsupportedBaselineSchemaVersion(u32),
    UnsupportedFrozenSetSchemaVersion(u32),
    DuplicateForecastId(String),
    DuplicateOutcomeForecastId(String),
    DuplicateOutcomeReceipt(String),
    OutcomeForUnknownForecast(String),
    MissingOutcomeForForecast(String),
    OutcomeEpisodeMismatch {
        forecast_id: String,
        expected: String,
        found: String,
    },
    OutcomeProfileMismatch {
        forecast_id: String,
        expected: String,
        found: String,
    },
    MixedForecastScope {
        field: &'static str,
        expected: String,
        found: String,
    },
    MissingEvaluationSplit,
    MissingEvaluationManifest,
    EmptyBaselineField { baseline_id: String, field: &'static str },
    InvalidBaselineSampleCount { baseline_id: String },
    DuplicateBaselineId(String),
    DuplicateBaselineMethodForTaskFamily {
        method: ForecastBaselineMethod,
        task_family_id: String,
    },
    MissingBaselineForTaskFamily {
        method: ForecastBaselineMethod,
        task_family_id: String,
    },
    BaselineTrainingSplitEqualsEvaluation {
        baseline_id: String,
        split_id: String,
    },
    BaselineTrainingManifestEqualsEvaluation {
        baseline_id: String,
        manifest_ref: String,
    },
    BaselineTaxonomyMismatch {
        baseline_id: String,
        expected: String,
        found: String,
    },
    BaselineOutcomeProfileMismatch {
        baseline_id: String,
        expected: String,
        found: String,
    },
}

impl fmt::Display for MetacognitionEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBinCount(count) => {
                write!(f, "calibration bin count must be in 1..=100, got {count}")
            }
            Self::EmptyEpisodeId(field) => {
                write!(f, "required episode identifier `{field}` is empty")
            }
            Self::DuplicatePredictionEpisode(id) => {
                write!(f, "episode `{id}` has more than one correctness prediction")
            }
            Self::EmptyTaskFamilyId { episode_id } => {
                write!(f, "episode `{episode_id}` has an empty task-family identifier")
            }
            Self::DuplicateAssumptionEpisode(id) => {
                write!(f, "episode `{id}` has more than one weak-assumption observation")
            }
            Self::DuplicateRevisionPair(id) => {
                write!(f, "revision pair `{id}` appears more than once")
            }
            Self::InvalidProbability { field, value } => {
                write!(f, "probability `{field}` must be finite and within [0, 1], got {value}")
            }
            Self::InvalidTolerance(value) => {
                write!(f, "stable tolerance must be finite and within [0, 1], got {value}")
            }
            Self::InvalidThreshold(value) => {
                write!(f, "selective-risk threshold must be finite and within [0, 1], got {value}")
            }
            Self::EmptyForecastSet => write!(f, "frozen forecast set must not be empty"),
            Self::EmptyForecastField { forecast_id, field } => write!(
                f, "forecast '{forecast_id}' has empty required field '{field}'"
            ),
            Self::EmptyOutcomeField { forecast_id, field } => write!(
                f, "outcome for forecast '{forecast_id}' has empty required field '{field}'"
            ),
            Self::UnsupportedForecastSchemaVersion(v) => {
                write!(f, "unsupported forecast schema version {v}")
            }
            Self::UnsupportedOutcomeSchemaVersion(v) => {
                write!(f, "unsupported outcome schema version {v}")
            }
            Self::UnsupportedBaselineSchemaVersion(v) => {
                write!(f, "unsupported baseline schema version {v}")
            }
            Self::UnsupportedFrozenSetSchemaVersion(v) => {
                write!(f, "unsupported frozen-set schema version {v}")
            }
            Self::DuplicateForecastId(id) => write!(f, "forecast ID '{id}' appears more than once"),
            Self::DuplicateOutcomeForecastId(id) => {
                write!(f, "forecast '{id}' has more than one outcome")
            }
            Self::DuplicateOutcomeReceipt(id) => write!(f, "outcome receipt ID '{id}' is reused"),
            Self::OutcomeForUnknownForecast(id) => {
                write!(f, "outcome references unknown forecast '{id}'")
            }
            Self::MissingOutcomeForForecast(id) => {
                write!(f, "forecast '{id}' has no bound outcome")
            }
            Self::OutcomeEpisodeMismatch { forecast_id, expected, found } => write!(
                f, "outcome for '{forecast_id}' has episode '{found}', expected '{expected}'"
            ),
            Self::OutcomeProfileMismatch { forecast_id, expected, found } => write!(
                f, "outcome for '{forecast_id}' has profile '{found}', expected '{expected}'"
            ),
            Self::MixedForecastScope { field, expected, found } => write!(
                f,
                "forecast batch mixes '{field}': expected '{expected}', found '{found}'"
            ),
            Self::MissingEvaluationSplit => {
                write!(f, "baseline comparison requires an explicit evaluation split ID")
            }
            Self::MissingEvaluationManifest => {
                write!(f, "baseline comparison requires an evaluation corpus manifest reference")
            }
            Self::EmptyBaselineField { baseline_id, field } => {
                write!(f, "baseline '{baseline_id}' has empty required field '{field}'")
            }
            Self::InvalidBaselineSampleCount { baseline_id } => {
                write!(f, "baseline '{baseline_id}' needs at least one calibration sample")
            }
            Self::DuplicateBaselineId(id) => {
                write!(f, "baseline ID '{id}' appears more than once")
            }
            Self::DuplicateBaselineMethodForTaskFamily { method, task_family_id } => {
                write!(
                    f,
                    "baseline method {method:?} appears twice for task family '{task_family_id}'"
                )
            }
            Self::MissingBaselineForTaskFamily { method, task_family_id } => {
                write!(
                    f,
                    "baseline method {method:?} is missing for task family '{task_family_id}'"
                )
            }
            Self::BaselineTrainingSplitEqualsEvaluation { baseline_id, split_id } => {
                write!(
                    f,
                    "baseline '{baseline_id}' training split '{split_id}' equals evaluation split"
                )
            }
            Self::BaselineTrainingManifestEqualsEvaluation {
                baseline_id,
                manifest_ref,
            } => {
                write!(
                    f,
                    "baseline '{baseline_id}' training manifest '{manifest_ref}' equals evaluation manifest"
                )
            }
            Self::BaselineTaxonomyMismatch { baseline_id, expected, found } => {
                write!(
                    f,
                    "baseline '{baseline_id}' taxonomy '{found}' does not match '{expected}'"
                )
            }
            Self::BaselineOutcomeProfileMismatch { baseline_id, expected, found } => {
                write!(
                    f,
                    "baseline '{baseline_id}' outcome '{found}' does not match '{expected}'"
                )
            }
        }
    }
}

impl std::error::Error for MetacognitionEvaluationError {}



/// Prospective forecast record. IDs must refer to frozen identities, not mutable display names.
/// References are opaque; this module checks binding shape, not signatures or content hashes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrectnessForecastV1 {
    pub schema_version: u32,
    pub forecast_id: String,
    pub episode_id: String,
    pub task_family_id: String,
    pub task_taxonomy_id: String,
    /// Exact proposition, e.g. answer-correct-under-policy-v2.
    pub outcome_profile_id: String,
    pub subject_id: String,
    pub model_profile_id: String,
    /// Reference to the input/evidence snapshot visible when the forecast was made.
    pub input_snapshot_ref: String,
    pub predicted_probability: f64,
    pub asserted: bool,
}

/// Oracle/benchmark outcome attached after a forecast has been frozen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrectnessOutcomeV1 {
    pub schema_version: u32,
    pub forecast_id: String,
    pub episode_id: String,
    pub outcome_profile_id: String,
    pub outcome_receipt_id: String,
    pub outcome_evidence_ref: String,
    pub correct: bool,
}

/// Traceability retained in a report produced by the two-phase binding path.
/// This is not a cryptographic proof of provenance or chronology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastOutcomeBindingReport {
    pub schema_version: u32,
    pub binding_method: String,
    pub outcome_profile_id: String,
    pub subject_id: String,
    pub model_profile_id: String,
    pub task_taxonomy_id: String,
    pub forecast_ids: Vec<String>,
    pub outcome_receipt_ids: Vec<String>,
    pub outcome_evidence_refs: Vec<String>,
}

/// Baseline family used for prospective probability comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ForecastBaselineMethod {
    /// Constant probability estimated from a disjoint calibration split's base rate.
    ConstantBaseRate,
    /// Trailing-window empirical correctness rate estimated before the holdout split.
    RecentEmpiricalAccuracy,
}

/// Pre-estimated baseline probability captured using calibration data only.
///
/// The manifest reference must identify the exact calibration corpus and the split ID must
/// differ from the held-out split. This does not prove that those artifacts are disjoint; an
/// external manifest verifier remains responsible for that property.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastBaselineV1 {
    pub schema_version: u32,
    pub baseline_id: String,
    pub method: ForecastBaselineMethod,
    pub task_family_id: String,
    pub task_taxonomy_id: String,
    pub outcome_profile_id: String,
    pub predicted_probability: f64,
    pub training_sample_count: usize,
    pub training_split_id: String,
    pub training_corpus_manifest_ref: String,
}

/// One frozen baseline's metrics on the same held-out outcomes as the candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastBaselineScoreReport {
    pub baseline_id: String,
    pub method: ForecastBaselineMethod,
    pub task_family_id: String,
    pub training_split_id: String,
    pub training_corpus_manifest_ref: String,
    pub training_sample_count: usize,
    pub evaluation_split_id: String,
    pub evaluation_corpus_manifest_ref: String,
    pub evaluation_sample_count: usize,
    pub predicted_probability: f64,
    pub empirical_accuracy: Option<f64>,
    pub brier_score: Option<f64>,
    pub log_loss: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    /// Candidate Brier score minus baseline score; negative favors the candidate.
    pub candidate_brier_delta: Option<f64>,
    /// Candidate log loss minus baseline loss; negative favors the candidate.
    pub candidate_log_loss_delta: Option<f64>,
}

/// Baselines and the candidate scored against the exact same family-local holdout outcomes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskFamilyForecastBaselineComparison {
    pub task_family_id: String,
    pub evaluation_episodes: usize,
    pub candidate_brier_score: Option<f64>,
    pub candidate_log_loss: Option<f64>,
    pub candidate_expected_calibration_error: Option<f64>,
    pub baselines: Vec<ForecastBaselineScoreReport>,
}

/// Split-aware, strictly comparative report. It selects no winner and grants no authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastBaselineComparisonReport {
    pub schema_version: u32,
    pub outcome_profile_id: String,
    pub task_taxonomy_id: String,
    pub evaluation_split_id: String,
    pub evaluation_corpus_manifest_ref: String,
    pub baseline_methods: Vec<ForecastBaselineMethod>,
    pub family_reports: Vec<TaskFamilyForecastBaselineComparison>,
}

/// Forecasts and scoring policy fixed before outcomes are attached. Private fields prevent
/// ordinary Rust callers from mutating the batch or policy through this API. A trusted capture
/// layer must still preserve the serialized value before outcomes are observed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "FrozenCorrectnessForecastSetWire")]
pub struct FrozenCorrectnessForecastSet {
    schema_version: u32,
    forecasts: Vec<CorrectnessForecastV1>,
    calibration_bins: usize,
    selective_thresholds: Vec<f64>,
    evaluation_split_id: Option<String>,
    evaluation_corpus_manifest_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct FrozenCorrectnessForecastSetWire {
    schema_version: u32,
    forecasts: Vec<CorrectnessForecastV1>,
    calibration_bins: usize,
    selective_thresholds: Vec<f64>,
    #[serde(default)]
    evaluation_split_id: Option<String>,
    #[serde(default)]
    evaluation_corpus_manifest_ref: Option<String>,
}

impl FrozenCorrectnessForecastSet {
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn forecasts(&self) -> &[CorrectnessForecastV1] {
        &self.forecasts
    }

    pub fn calibration_bins(&self) -> usize {
        self.calibration_bins
    }

    pub fn selective_thresholds(&self) -> &[f64] {
        &self.selective_thresholds
    }

    pub fn evaluation_split_id(&self) -> Option<&str> {
        self.evaluation_split_id.as_deref()
    }

    pub fn evaluation_corpus_manifest_ref(&self) -> Option<&str> {
        self.evaluation_corpus_manifest_ref.as_deref()
    }
}

impl TryFrom<FrozenCorrectnessForecastSetWire> for FrozenCorrectnessForecastSet {
    type Error = MetacognitionEvaluationError;

    fn try_from(w: FrozenCorrectnessForecastSetWire) -> Result<Self, Self::Error> {
        if w.schema_version != FROZEN_FORECAST_SET_SCHEMA_VERSION {
            return Err(
                MetacognitionEvaluationError::UnsupportedFrozenSetSchemaVersion(
                    w.schema_version,
                ),
            );
        }
        validate_forecast_set(&w.forecasts, w.calibration_bins, &w.selective_thresholds)?;
        validate_evaluation_scope(
            w.evaluation_split_id.as_deref(),
            w.evaluation_corpus_manifest_ref.as_deref(),
        )?;
        Ok(Self {
            schema_version: w.schema_version,
            forecasts: w.forecasts,
            calibration_bins: w.calibration_bins,
            selective_thresholds: w.selective_thresholds,
            evaluation_split_id: w.evaluation_split_id,
            evaluation_corpus_manifest_ref: w.evaluation_corpus_manifest_ref,
        })
    }
}

/// Freeze the forecast records and all bin/threshold choices before outcome collection.
/// A report must cover exactly one subject, model/profile, task taxonomy, and outcome profile
/// so pooled calibration has one semantic meaning.
pub fn freeze_correctness_forecasts(
    forecasts: Vec<CorrectnessForecastV1>,
    calibration_bins: usize,
    selective_thresholds: Vec<f64>,
) -> Result<FrozenCorrectnessForecastSet, MetacognitionEvaluationError> {
    freeze_correctness_forecasts_internal(
        forecasts,
        calibration_bins,
        selective_thresholds,
        None,
        None,
    )
}

/// Freeze a qualification batch explicitly tagged with its held-out evaluation split.
///
/// Use this constructor for all baseline comparisons. Baseline profiles must name a distinct
/// calibration split and manifest reference; matching IDs are rejected, and the manifest
/// verifier must independently establish that the split artifacts are actually disjoint.
pub fn freeze_correctness_forecasts_for_split(
    forecasts: Vec<CorrectnessForecastV1>,
    calibration_bins: usize,
    selective_thresholds: Vec<f64>,
    evaluation_split_id: String,
    evaluation_corpus_manifest_ref: String,
) -> Result<FrozenCorrectnessForecastSet, MetacognitionEvaluationError> {
    if evaluation_split_id.trim().is_empty()
        || evaluation_split_id.trim() != evaluation_split_id
    {
        return Err(MetacognitionEvaluationError::MissingEvaluationSplit);
    }
    if evaluation_corpus_manifest_ref.trim().is_empty()
        || evaluation_corpus_manifest_ref.trim() != evaluation_corpus_manifest_ref
    {
        return Err(MetacognitionEvaluationError::MissingEvaluationManifest);
    }
    freeze_correctness_forecasts_internal(
        forecasts,
        calibration_bins,
        selective_thresholds,
        Some(evaluation_split_id),
        Some(evaluation_corpus_manifest_ref),
    )
}

fn freeze_correctness_forecasts_internal(
    forecasts: Vec<CorrectnessForecastV1>,
    calibration_bins: usize,
    selective_thresholds: Vec<f64>,
    evaluation_split_id: Option<String>,
    evaluation_corpus_manifest_ref: Option<String>,
) -> Result<FrozenCorrectnessForecastSet, MetacognitionEvaluationError> {
    FrozenCorrectnessForecastSet::try_from(FrozenCorrectnessForecastSetWire {
        schema_version: FROZEN_FORECAST_SET_SCHEMA_VERSION,
        forecasts,
        calibration_bins,
        selective_thresholds,
        evaluation_split_id,
        evaluation_corpus_manifest_ref,
    })
}

fn validate_evaluation_scope(
    split_id: Option<&str>,
    manifest_ref: Option<&str>,
) -> Result<(), MetacognitionEvaluationError> {
    match (split_id, manifest_ref) {
        (None, None) => Ok(()),
        (Some(split), Some(manifest))
            if !split.trim().is_empty()
                && split.trim() == split
                && !manifest.trim().is_empty()
                && manifest.trim() == manifest =>
        {
            Ok(())
        }
        (None, Some(_)) => Err(MetacognitionEvaluationError::MissingEvaluationSplit),
        (Some(_), None) => Err(MetacognitionEvaluationError::MissingEvaluationManifest),
        _ => Err(MetacognitionEvaluationError::MissingEvaluationManifest),
    }
}

fn require_forecast_field(
    id: &str,
    field: &'static str,
    value: &str,
) -> Result<(), MetacognitionEvaluationError> {
    if value.trim().is_empty() {
        return Err(MetacognitionEvaluationError::EmptyForecastField {
            forecast_id: id.to_owned(),
            field,
        });
    }
    Ok(())
}

fn validate_forecast_set(
    forecasts: &[CorrectnessForecastV1],
    bins: usize,
    thresholds: &[f64],
) -> Result<(), MetacognitionEvaluationError> {
    if forecasts.is_empty() {
        return Err(MetacognitionEvaluationError::EmptyForecastSet);
    }
    if !(1..=100).contains(&bins) {
        return Err(MetacognitionEvaluationError::InvalidBinCount(bins));
    }

    let first = &forecasts[0];
    let (mut ids, mut episodes) = (HashSet::new(), HashSet::new());
    for p in forecasts {
        if p.schema_version != CORRECTNESS_FORECAST_SCHEMA_VERSION {
            return Err(
                MetacognitionEvaluationError::UnsupportedForecastSchemaVersion(p.schema_version),
            );
        }
        for (field, value) in [
            ("forecast_id", p.forecast_id.as_str()),
            ("episode_id", p.episode_id.as_str()),
            ("task_family_id", p.task_family_id.as_str()),
            ("task_taxonomy_id", p.task_taxonomy_id.as_str()),
            ("outcome_profile_id", p.outcome_profile_id.as_str()),
            ("subject_id", p.subject_id.as_str()),
            ("model_profile_id", p.model_profile_id.as_str()),
            ("input_snapshot_ref", p.input_snapshot_ref.as_str()),
        ] {
            require_forecast_field(&p.forecast_id, field, value)?;
        }
        validate_probability("predicted_probability", p.predicted_probability)?;
        if !ids.insert(p.forecast_id.as_str()) {
            return Err(MetacognitionEvaluationError::DuplicateForecastId(
                p.forecast_id.clone(),
            ));
        }
        if !episodes.insert(p.episode_id.as_str()) {
            return Err(MetacognitionEvaluationError::DuplicatePredictionEpisode(
                p.episode_id.clone(),
            ));
        }
        for (field, expected, found) in [
            (
                "outcome_profile_id",
                first.outcome_profile_id.as_str(),
                p.outcome_profile_id.as_str(),
            ),
            ("subject_id", first.subject_id.as_str(), p.subject_id.as_str()),
            (
                "model_profile_id",
                first.model_profile_id.as_str(),
                p.model_profile_id.as_str(),
            ),
            (
                "task_taxonomy_id",
                first.task_taxonomy_id.as_str(),
                p.task_taxonomy_id.as_str(),
            ),
        ] {
            if expected != found {
                return Err(MetacognitionEvaluationError::MixedForecastScope {
                    field,
                    expected: expected.to_owned(),
                    found: found.to_owned(),
                });
            }
        }
    }
    for &t in thresholds {
        validate_probability("selective_threshold", t)
            .map_err(|_| MetacognitionEvaluationError::InvalidThreshold(t))?;
    }
    Ok(())
}

/// Bind a complete outcome set to a previously frozen forecast set. Binding requires a
/// one-to-one exact match on forecast ID, episode ID, and outcome profile. Missing, duplicate,
/// unknown, or mismatched receipts fail closed. This API cannot prove the caller did not observe
/// outcomes before freezing; establish chronology through an external trusted append-only capture.
pub fn evaluate_frozen_correctness_forecasts(
    frozen: &FrozenCorrectnessForecastSet,
    outcomes: &[CorrectnessOutcomeV1],
    assumptions: &[WeakAssumptionObservation],
    revisions: &[ConfidenceRevisionObservation],
) -> Result<MetacognitionReport, MetacognitionEvaluationError> {
    validate_forecast_set(
        &frozen.forecasts,
        frozen.calibration_bins,
        &frozen.selective_thresholds,
    )?;
    let forecasts: BTreeMap<&str, &CorrectnessForecastV1> = frozen
        .forecasts
        .iter()
        .map(|p| (p.forecast_id.as_str(), p))
        .collect();
    let (mut joined, mut receipt_ids) = (BTreeMap::new(), HashSet::new());
    for o in outcomes {
        if o.schema_version != CORRECTNESS_OUTCOME_SCHEMA_VERSION {
            return Err(
                MetacognitionEvaluationError::UnsupportedOutcomeSchemaVersion(o.schema_version),
            );
        }
        for (field, value) in [
            ("forecast_id", o.forecast_id.as_str()),
            ("episode_id", o.episode_id.as_str()),
            ("outcome_profile_id", o.outcome_profile_id.as_str()),
            ("outcome_receipt_id", o.outcome_receipt_id.as_str()),
            ("outcome_evidence_ref", o.outcome_evidence_ref.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(MetacognitionEvaluationError::EmptyOutcomeField {
                    forecast_id: o.forecast_id.clone(),
                    field,
                });
            }
        }
        if !receipt_ids.insert(o.outcome_receipt_id.as_str()) {
            return Err(MetacognitionEvaluationError::DuplicateOutcomeReceipt(
                o.outcome_receipt_id.clone(),
            ));
        }
        let p = forecasts.get(o.forecast_id.as_str()).ok_or_else(|| {
            MetacognitionEvaluationError::OutcomeForUnknownForecast(o.forecast_id.clone())
        })?;
        if o.episode_id != p.episode_id {
            return Err(MetacognitionEvaluationError::OutcomeEpisodeMismatch {
                forecast_id: o.forecast_id.clone(),
                expected: p.episode_id.clone(),
                found: o.episode_id.clone(),
            });
        }
        if o.outcome_profile_id != p.outcome_profile_id {
            return Err(MetacognitionEvaluationError::OutcomeProfileMismatch {
                forecast_id: o.forecast_id.clone(),
                expected: p.outcome_profile_id.clone(),
                found: o.outcome_profile_id.clone(),
            });
        }
        if joined.insert(o.forecast_id.as_str(), o).is_some() {
            return Err(MetacognitionEvaluationError::DuplicateOutcomeForecastId(
                o.forecast_id.clone(),
            ));
        }
    }

    let (mut predictions, mut ordered_receipts, mut evidence_refs) = (
        Vec::with_capacity(frozen.forecasts.len()),
        Vec::new(),
        Vec::new(),
    );
    for p in &frozen.forecasts {
        let o = joined.get(p.forecast_id.as_str()).ok_or_else(|| {
            MetacognitionEvaluationError::MissingOutcomeForForecast(p.forecast_id.clone())
        })?;
        predictions.push(CorrectnessPrediction {
            episode_id: p.episode_id.clone(),
            task_family_id: p.task_family_id.clone(),
            confidence_target_episode_id: p.episode_id.clone(),
            confidence: p.predicted_probability,
            correct: o.correct,
            asserted: p.asserted,
        });
        ordered_receipts.push(o.outcome_receipt_id.clone());
        evidence_refs.push(o.outcome_evidence_ref.clone());
    }
    let mut report = evaluate_metacognition(
        &predictions,
        assumptions,
        revisions,
        frozen.calibration_bins,
        &frozen.selective_thresholds,
    )?;
    let p = &frozen.forecasts[0];
    report.forecast_outcome_binding = Some(ForecastOutcomeBindingReport {
        schema_version: FORECAST_OUTCOME_BINDING_REPORT_SCHEMA_VERSION,
        binding_method: "exact-forecast-id-episode-target-profile-v1".into(),
        outcome_profile_id: p.outcome_profile_id.clone(),
        subject_id: p.subject_id.clone(),
        model_profile_id: p.model_profile_id.clone(),
        task_taxonomy_id: p.task_taxonomy_id.clone(),
        forecast_ids: frozen
            .forecasts
            .iter()
            .map(|x| x.forecast_id.clone())
            .collect(),
        outcome_receipt_ids: ordered_receipts,
        outcome_evidence_refs: evidence_refs,
    });
    Ok(report)
}


/// Evaluate the candidate and two simple probability baselines on the same held-out outcomes.
///
/// Every task family must have both baseline methods, estimated from a non-matching calibration
/// split. The caller must ensure the training-corpus manifests are disjoint from the holdout;
/// this function checks identity/split separation but cannot prove manifest lineage itself.
pub fn evaluate_frozen_forecasts_with_baselines(
    frozen: &FrozenCorrectnessForecastSet,
    outcomes: &[CorrectnessOutcomeV1],
    assumptions: &[WeakAssumptionObservation],
    revisions: &[ConfidenceRevisionObservation],
    baselines: &[ForecastBaselineV1],
) -> Result<MetacognitionReport, MetacognitionEvaluationError> {
    let evaluation_split_id = frozen
        .evaluation_split_id
        .as_deref()
        .filter(|id| !id.trim().is_empty())
        .ok_or(MetacognitionEvaluationError::MissingEvaluationSplit)?;
    let evaluation_corpus_manifest_ref = frozen
        .evaluation_corpus_manifest_ref
        .as_deref()
        .filter(|reference| !reference.trim().is_empty())
        .ok_or(MetacognitionEvaluationError::MissingEvaluationManifest)?;

    // Reuse the strict forecast/outcome join. No baseline gets to change the candidate's
    // predictions, outcome bindings, confidence thresholds, or reliability-bin policy.
    let mut report =
        evaluate_frozen_correctness_forecasts(frozen, outcomes, assumptions, revisions)?;
    let first = &frozen.forecasts[0];
    let outcomes_by_id: BTreeMap<&str, &CorrectnessOutcomeV1> = outcomes
        .iter()
        .map(|o| (o.forecast_id.as_str(), o))
        .collect();

    let mut by_family: BTreeMap<String, Vec<&CorrectnessForecastV1>> = BTreeMap::new();
    for forecast in &frozen.forecasts {
        by_family
            .entry(forecast.task_family_id.clone())
            .or_default()
            .push(forecast);
    }

    let mut baseline_index: BTreeMap<(String, ForecastBaselineMethod), &ForecastBaselineV1> =
        BTreeMap::new();
    let mut baseline_ids = HashSet::new();
    for baseline in baselines {
        if baseline.schema_version != FORECAST_BASELINE_SCHEMA_VERSION {
            return Err(
                MetacognitionEvaluationError::UnsupportedBaselineSchemaVersion(
                    baseline.schema_version,
                ),
            );
        }
        for (field, value) in [
            ("baseline_id", baseline.baseline_id.as_str()),
            ("task_family_id", baseline.task_family_id.as_str()),
            ("task_taxonomy_id", baseline.task_taxonomy_id.as_str()),
            ("outcome_profile_id", baseline.outcome_profile_id.as_str()),
            ("training_split_id", baseline.training_split_id.as_str()),
            (
                "training_corpus_manifest_ref",
                baseline.training_corpus_manifest_ref.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                return Err(MetacognitionEvaluationError::EmptyBaselineField {
                    baseline_id: baseline.baseline_id.clone(),
                    field,
                });
            }
        }
        if baseline.training_split_id.trim() != baseline.training_split_id {
            return Err(MetacognitionEvaluationError::EmptyBaselineField {
                baseline_id: baseline.baseline_id.clone(),
                field: "training_split_id",
            });
        }
        if baseline.training_sample_count == 0 {
            return Err(MetacognitionEvaluationError::InvalidBaselineSampleCount {
                baseline_id: baseline.baseline_id.clone(),
            });
        }
        validate_probability("baseline_probability", baseline.predicted_probability)?;
        if baseline.training_split_id == evaluation_split_id {
            return Err(
                MetacognitionEvaluationError::BaselineTrainingSplitEqualsEvaluation {
                    baseline_id: baseline.baseline_id.clone(),
                    split_id: evaluation_split_id.to_owned(),
                },
            );
        }
        if baseline.training_corpus_manifest_ref == evaluation_corpus_manifest_ref {
            return Err(
                MetacognitionEvaluationError::BaselineTrainingManifestEqualsEvaluation {
                    baseline_id: baseline.baseline_id.clone(),
                    manifest_ref: evaluation_corpus_manifest_ref.to_owned(),
                },
            );
        }
        if baseline.task_taxonomy_id != first.task_taxonomy_id {
            return Err(MetacognitionEvaluationError::BaselineTaxonomyMismatch {
                baseline_id: baseline.baseline_id.clone(),
                expected: first.task_taxonomy_id.clone(),
                found: baseline.task_taxonomy_id.clone(),
            });
        }
        if baseline.outcome_profile_id != first.outcome_profile_id {
            return Err(
                MetacognitionEvaluationError::BaselineOutcomeProfileMismatch {
                    baseline_id: baseline.baseline_id.clone(),
                    expected: first.outcome_profile_id.clone(),
                    found: baseline.outcome_profile_id.clone(),
                },
            );
        }
        if !by_family.contains_key(&baseline.task_family_id) {
            return Err(MetacognitionEvaluationError::EmptyBaselineField {
                baseline_id: baseline.baseline_id.clone(),
                field: "task_family_id_not_in_evaluation_set",
            });
        }
        if !baseline_ids.insert(baseline.baseline_id.as_str()) {
            return Err(MetacognitionEvaluationError::DuplicateBaselineId(
                baseline.baseline_id.clone(),
            ));
        }
        let key = (baseline.task_family_id.clone(), baseline.method);
        if baseline_index.insert(key, baseline).is_some() {
            return Err(
                MetacognitionEvaluationError::DuplicateBaselineMethodForTaskFamily {
                    method: baseline.method,
                    task_family_id: baseline.task_family_id.clone(),
                },
            );
        }
    }

    for task_family_id in by_family.keys() {
        for method in [
            ForecastBaselineMethod::ConstantBaseRate,
            ForecastBaselineMethod::RecentEmpiricalAccuracy,
        ] {
            if !baseline_index.contains_key(&(task_family_id.clone(), method)) {
                return Err(
                    MetacognitionEvaluationError::MissingBaselineForTaskFamily {
                        method,
                        task_family_id: task_family_id.clone(),
                    },
                );
            }
        }
    }

    let candidate_by_family: BTreeMap<&str, &TaskFamilyCalibrationReport> = report
        .task_family_calibration
        .iter()
        .map(|r| (r.task_family_id.as_str(), r))
        .collect();
    let mut family_reports = Vec::with_capacity(by_family.len());
    for (task_family_id, family_forecasts) in by_family {
        let candidate = candidate_by_family
            .get(task_family_id.as_str())
            .ok_or_else(|| MetacognitionEvaluationError::MissingBaselineForTaskFamily {
                method: ForecastBaselineMethod::ConstantBaseRate,
                task_family_id: task_family_id.clone(),
            })?;
        let observed: Vec<bool> = family_forecasts
            .iter()
            .map(|p| {
                outcomes_by_id
                    .get(p.forecast_id.as_str())
                    .map(|o| o.correct)
                    .ok_or_else(|| {
                        MetacognitionEvaluationError::MissingOutcomeForForecast(
                            p.forecast_id.clone(),
                        )
                    })
            })
            .collect::<Result<_, _>>()?;
        let accuracy = mean_bool(observed.iter().copied());
        let mut family_baselines = Vec::with_capacity(2);
        for method in [
            ForecastBaselineMethod::ConstantBaseRate,
            ForecastBaselineMethod::RecentEmpiricalAccuracy,
        ] {
            let baseline = baseline_index
                .get(&(task_family_id.clone(), method))
                .ok_or_else(|| {
                    MetacognitionEvaluationError::MissingBaselineForTaskFamily {
                        method,
                        task_family_id: task_family_id.clone(),
                    }
                })?;
            let n = observed.len();
            let mut brier = 0.0;
            let mut log_loss = 0.0;
            for &correct in &observed {
                let target = if correct { 1.0 } else { 0.0 };
                brier += (baseline.predicted_probability - target).powi(2);
                let p_correct = if correct {
                    baseline.predicted_probability
                } else {
                    1.0 - baseline.predicted_probability
                };
                log_loss += -p_correct
                    .clamp(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON)
                    .ln();
            }
            let baseline_brier = brier / n as f64;
            let baseline_log_loss = log_loss / n as f64;
            let baseline_ece = accuracy.map(|a| (baseline.predicted_probability - a).abs());
            family_baselines.push(ForecastBaselineScoreReport {
                baseline_id: baseline.baseline_id.clone(),
                method,
                task_family_id: task_family_id.clone(),
                training_split_id: baseline.training_split_id.clone(),
                training_corpus_manifest_ref: baseline.training_corpus_manifest_ref.clone(),
                training_sample_count: baseline.training_sample_count,
                evaluation_split_id: evaluation_split_id.to_owned(),
                evaluation_corpus_manifest_ref: evaluation_corpus_manifest_ref.to_owned(),
                evaluation_sample_count: n,
                predicted_probability: baseline.predicted_probability,
                empirical_accuracy: accuracy,
                brier_score: Some(baseline_brier),
                log_loss: Some(baseline_log_loss),
                expected_calibration_error: baseline_ece,
                candidate_brier_delta: candidate
                    .brier_score
                    .map(|score| score - baseline_brier),
                candidate_log_loss_delta: candidate
                    .log_loss
                    .map(|score| score - baseline_log_loss),
            });
        }
        family_reports.push(TaskFamilyForecastBaselineComparison {
            task_family_id,
            evaluation_episodes: family_forecasts.len(),
            candidate_brier_score: candidate.brier_score,
            candidate_log_loss: candidate.log_loss,
            candidate_expected_calibration_error: candidate.expected_calibration_error,
            baselines: family_baselines,
        });
    }

    report.baseline_comparison = Some(ForecastBaselineComparisonReport {
        schema_version: FORECAST_BASELINE_COMPARISON_SCHEMA_VERSION,
        outcome_profile_id: first.outcome_profile_id.clone(),
        task_taxonomy_id: first.task_taxonomy_id.clone(),
        evaluation_split_id: evaluation_split_id.to_owned(),
        evaluation_corpus_manifest_ref: evaluation_corpus_manifest_ref.to_owned(),
        baseline_methods: vec![
            ForecastBaselineMethod::ConstantBaseRate,
            ForecastBaselineMethod::RecentEmpiricalAccuracy,
        ],
        family_reports,
    });
    Ok(report)
}

/// Evaluate a frozen set of metacognitive observations.
pub fn evaluate_metacognition(
    predictions: &[CorrectnessPrediction],
    assumptions: &[WeakAssumptionObservation],
    revisions: &[ConfidenceRevisionObservation],
    calibration_bins: usize,
    selective_thresholds: &[f64],
) -> Result<MetacognitionReport, MetacognitionEvaluationError> {
    if !(1..=100).contains(&calibration_bins) {
        return Err(MetacognitionEvaluationError::InvalidBinCount(calibration_bins));
    }
    validate_predictions(predictions)?;
    validate_assumptions(assumptions)?;
    validate_revisions(revisions)?;
    for &threshold in selective_thresholds {
        validate_probability("selective_threshold", threshold)
            .map_err(|_| MetacognitionEvaluationError::InvalidThreshold(threshold))?;
    }

    let n = predictions.len();
    let mut task_family_members: BTreeMap<String, Vec<CorrectnessPrediction>> = BTreeMap::new();
    for prediction in predictions {
        task_family_members
            .entry(prediction.task_family_id.clone())
            .or_default()
            .push(prediction.clone());
    }
    // Correct across pooled + every observed task family, for each predeclared threshold.
    // Task-family labels themselves must come from a frozen, outcome-independent taxonomy.
    let familywise_comparison_count = selective_thresholds
        .len()
        .saturating_add(calibration_bins)
        .saturating_mul(task_family_members.len().saturating_add(1));

    let empirical_accuracy = mean_bool(predictions.iter().map(|p| p.correct));
    let mean_confidence = mean_f64(predictions.iter().map(|p| p.confidence));

    let (brier_score, log_loss) = if predictions.is_empty() {
        (None, None)
    } else {
        let mut brier = 0.0;
        let mut log = 0.0;
        for prediction in predictions {
            let target = if prediction.correct { 1.0 } else { 0.0 };
            brier += (prediction.confidence - target).powi(2);
            let p_correct = if prediction.correct {
                prediction.confidence
            } else {
                1.0 - prediction.confidence
            };
            log += -p_correct
                .clamp(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON)
                .ln();
        }
        (Some(brier / n as f64), Some(log / n as f64))
    };

    let (ece, mce) = calibration_errors(predictions, calibration_bins);
    let confidence_bias = match (mean_confidence, empirical_accuracy) {
        (Some(confidence), Some(accuracy)) => Some(confidence - accuracy),
        _ => None,
    };
    let correctness_auroc = auroc(predictions);
    let current_episode_binding_rate = mean_bool(
        predictions
            .iter()
            .map(|p| p.episode_id == p.confidence_target_episode_id),
    );
    let assertion_rate = mean_bool(predictions.iter().map(|p| p.asserted));
    let asserted_risk = conditional_rate(predictions.iter().filter(|p| p.asserted), |p| !p.correct);
    let abstention_opportunity_cost =
        conditional_rate(predictions.iter().filter(|p| !p.asserted), |p| p.correct);

    // Freeze thresholds, equal-width bins, and group taxonomy before inspecting outcomes.
    // Bonferroni covers every pooled/group threshold bound and every reported calibration bin.
    let selective_risk = selective_thresholds
        .iter()
        .map(|&threshold| {
            let selected: Vec<&CorrectnessPrediction> = predictions
                .iter()
                .filter(|p| p.asserted && p.confidence >= threshold)
                .collect();
            let selected_n = selected.len();
            let errors = selected.iter().filter(|p| !p.correct).count();
            SelectiveRiskPoint {
                threshold,
                coverage: if n == 0 {
                    0.0
                } else {
                    selected_n as f64 / n as f64
                },
                risk: ratio(errors, selected_n),
                risk_upper_bound_95: hoeffding_familywise_risk_upper_bound(
                    errors,
                    selected_n,
                    familywise_comparison_count,
                ),
                selected: selected_n,
            }
        })
        .collect();

    let task_family_calibration = task_family_members
        .iter()
        .map(|(task_family_id, members)| {
            task_family_report(
                task_family_id,
                members,
                calibration_bins,
                selective_thresholds,
                familywise_comparison_count,
            )
        })
        .collect();
    let pooled_calibration_bins = calibration_bin_reports(
        predictions,
        calibration_bins,
        familywise_comparison_count,
    );

    let weak_assumption_detection = detection_report(assumptions);
    let (revision_direction_accuracy, mean_expected_revision_delta) = revision_metrics(revisions);

    Ok(MetacognitionReport {
        evaluator_version: METACOGNITION_EVALUATOR_VERSION.into(),
        predictions: n,
        empirical_accuracy,
        mean_confidence,
        brier_score,
        log_loss,
        expected_calibration_error: ece,
        maximum_calibration_error: mce,
        confidence_bias,
        correctness_auroc,
        current_episode_binding_rate,
        assertion_rate,
        asserted_risk,
        abstention_opportunity_cost,
        selective_risk_bound_method: SELECTIVE_RISK_BOUND_METHOD.into(),
        selective_risk_bound_assumptions: SELECTIVE_RISK_BOUND_ASSUMPTIONS.into(),
        task_family_calibration,
        calibration_bins: pooled_calibration_bins,
        selective_risk,
        weak_assumption_detection,
        confidence_revisions: revisions.len(),
        revision_direction_accuracy,
        mean_expected_revision_delta,
        forecast_outcome_binding: None,
        baseline_comparison: None,
    })
}

fn validate_predictions(
    predictions: &[CorrectnessPrediction],
) -> Result<(), MetacognitionEvaluationError> {
    let mut seen = HashSet::new();
    for prediction in predictions {
        if prediction.episode_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyEpisodeId("episode_id"));
        }
        if prediction.confidence_target_episode_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyEpisodeId(
                "confidence_target_episode_id",
            ));
        }
        if prediction.task_family_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyTaskFamilyId {
                episode_id: prediction.episode_id.clone(),
            });
        }
        validate_probability("confidence", prediction.confidence)?;
        if !seen.insert(prediction.episode_id.as_str()) {
            return Err(MetacognitionEvaluationError::DuplicatePredictionEpisode(
                prediction.episode_id.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_assumptions(
    assumptions: &[WeakAssumptionObservation],
) -> Result<(), MetacognitionEvaluationError> {
    let mut seen = HashSet::new();
    for observation in assumptions {
        if observation.episode_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyEpisodeId("episode_id"));
        }
        if !seen.insert(observation.episode_id.as_str()) {
            return Err(MetacognitionEvaluationError::DuplicateAssumptionEpisode(
                observation.episode_id.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_revisions(
    revisions: &[ConfidenceRevisionObservation],
) -> Result<(), MetacognitionEvaluationError> {
    let mut seen = HashSet::new();
    for observation in revisions {
        if observation.pair_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyEpisodeId("pair_id"));
        }
        if observation.before_episode_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyEpisodeId(
                "before_episode_id",
            ));
        }
        if observation.after_episode_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyEpisodeId(
                "after_episode_id",
            ));
        }
        if !seen.insert(observation.pair_id.as_str()) {
            return Err(MetacognitionEvaluationError::DuplicateRevisionPair(
                observation.pair_id.clone(),
            ));
        }
        validate_probability("before_confidence", observation.before_confidence)?;
        validate_probability("after_confidence", observation.after_confidence)?;
        if !observation.stable_tolerance.is_finite()
            || !(0.0..=1.0).contains(&observation.stable_tolerance)
        {
            return Err(MetacognitionEvaluationError::InvalidTolerance(
                observation.stable_tolerance,
            ));
        }
    }
    Ok(())
}

fn validate_probability(
    field: &'static str,
    value: f64,
) -> Result<(), MetacognitionEvaluationError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(MetacognitionEvaluationError::InvalidProbability { field, value })
    }
}

/// Conservative one-sided Hoeffding bound with Bonferroni correction over a frozen threshold
/// family. It assumes independent evaluation episodes under a fixed selection rule; it is not
/// a distribution-shift guarantee.
fn hoeffding_familywise_risk_upper_bound(
    errors: usize,
    selected: usize,
    comparison_count: usize,
) -> Option<f64> {
    if selected == 0 || comparison_count == 0 || errors > selected {
        return None;
    }
    let empirical_risk = errors as f64 / selected as f64;
    let per_comparison_alpha = SELECTIVE_RISK_FAMILYWISE_ALPHA / comparison_count as f64;
    let radius = ((1.0 / per_comparison_alpha).ln() / (2.0 * selected as f64)).sqrt();
    Some((empirical_risk + radius).min(1.0))
}

fn task_family_report(
    task_family_id: &str,
    predictions: &[CorrectnessPrediction],
    calibration_bins: usize,
    selective_thresholds: &[f64],
    familywise_comparison_count: usize,
) -> TaskFamilyCalibrationReport {
    let n = predictions.len();
    let empirical_accuracy = mean_bool(predictions.iter().map(|p| p.correct));
    let mean_confidence = mean_f64(predictions.iter().map(|p| p.confidence));
    let (brier_score, log_loss) = if predictions.is_empty() {
        (None, None)
    } else {
        let mut brier = 0.0;
        let mut log = 0.0;
        for prediction in predictions {
            let target = if prediction.correct { 1.0 } else { 0.0 };
            brier += (prediction.confidence - target).powi(2);
            let p_correct = if prediction.correct {
                prediction.confidence
            } else {
                1.0 - prediction.confidence
            };
            log += -p_correct.clamp(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON).ln();
        }
        (Some(brier / n as f64), Some(log / n as f64))
    };
    let (ece, mce) = calibration_errors(predictions, calibration_bins);
    let confidence_bias = match (mean_confidence, empirical_accuracy) {
        (Some(confidence), Some(accuracy)) => Some(confidence - accuracy),
        _ => None,
    };
    let current_episode_binding_rate = mean_bool(
        predictions.iter().map(|p| p.episode_id == p.confidence_target_episode_id),
    );
    let assertion_rate = mean_bool(predictions.iter().map(|p| p.asserted));
    let asserted_risk = conditional_rate(predictions.iter().filter(|p| p.asserted), |p| !p.correct);
    let selective_risk = selective_thresholds
        .iter()
        .map(|&threshold| {
            let selected: Vec<&CorrectnessPrediction> = predictions
                .iter()
                .filter(|p| p.asserted && p.confidence >= threshold)
                .collect();
            let selected_n = selected.len();
            let errors = selected.iter().filter(|p| !p.correct).count();
            SelectiveRiskPoint {
                threshold,
                coverage: if n == 0 { 0.0 } else { selected_n as f64 / n as f64 },
                risk: ratio(errors, selected_n),
                risk_upper_bound_95: hoeffding_familywise_risk_upper_bound(
                    errors,
                    selected_n,
                    familywise_comparison_count,
                ),
                selected: selected_n,
            }
        })
        .collect();

    TaskFamilyCalibrationReport {
        task_family_id: task_family_id.to_string(),
        episodes: n,
        empirical_accuracy,
        mean_confidence,
        brier_score,
        log_loss,
        expected_calibration_error: ece,
        maximum_calibration_error: mce,
        confidence_bias,
        correctness_auroc: auroc(predictions),
        current_episode_binding_rate,
        assertion_rate,
        asserted_risk,
        selective_risk,
        calibration_bins: calibration_bin_reports(
            predictions,
            calibration_bins,
            familywise_comparison_count,
        ),
    }
}

fn calibration_bin_reports(
    predictions: &[CorrectnessPrediction],
    bins: usize,
    familywise_comparison_count: usize,
) -> Vec<CalibrationBinReport> {
    let mut counts = vec![0usize; bins];
    let mut confidence_sums = vec![0.0_f64; bins];
    let mut correct_counts = vec![0usize; bins];
    for prediction in predictions {
        let index = ((prediction.confidence * bins as f64).floor() as usize).min(bins - 1);
        counts[index] += 1;
        confidence_sums[index] += prediction.confidence;
        correct_counts[index] += usize::from(prediction.correct);
    }

    (0..bins)
        .map(|bin_index| {
            let episodes = counts[bin_index];
            let empirical_accuracy = ratio(correct_counts[bin_index], episodes);
            let mean_confidence =
                (episodes > 0).then(|| confidence_sums[bin_index] / episodes as f64);
            let interval = empirical_accuracy.map(|accuracy| {
                // Two-sided Hoeffding interval at alpha / familywise_comparison_count.
                // The union bound covers every pooled/group bin and selective-risk threshold.
                let per_comparison_alpha =
                    SELECTIVE_RISK_FAMILYWISE_ALPHA / familywise_comparison_count as f64;
                let radius = ((2.0 / per_comparison_alpha).ln()
                    / (2.0 * episodes as f64))
                    .sqrt();
                ((accuracy - radius).max(0.0), (accuracy + radius).min(1.0))
            });
            CalibrationBinReport {
                bin_index,
                confidence_lower: bin_index as f64 / bins as f64,
                confidence_upper: (bin_index + 1) as f64 / bins as f64,
                episodes,
                mean_confidence,
                empirical_accuracy,
                accuracy_lower_95: interval.map(|(lower, _)| lower),
                accuracy_upper_95: interval.map(|(_, upper)| upper),
            }
        })
        .collect()
}

fn calibration_errors(
    predictions: &[CorrectnessPrediction],
    bins: usize,
) -> (Option<f64>, Option<f64>) {
    if predictions.is_empty() {
        return (None, None);
    }
    let mut confidence_sum = vec![0.0; bins];
    let mut correct_sum = vec![0usize; bins];
    let mut count = vec![0usize; bins];
    for prediction in predictions {
        let index = ((prediction.confidence * bins as f64).floor() as usize).min(bins - 1);
        confidence_sum[index] += prediction.confidence;
        correct_sum[index] += usize::from(prediction.correct);
        count[index] += 1;
    }

    let mut ece = 0.0;
    let mut mce: f64 = 0.0;
    for index in 0..bins {
        if count[index] == 0 {
            continue;
        }
        let mean_confidence = confidence_sum[index] / count[index] as f64;
        let accuracy = correct_sum[index] as f64 / count[index] as f64;
        let gap = (mean_confidence - accuracy).abs();
        ece += gap * count[index] as f64 / predictions.len() as f64;
        mce = mce.max(gap);
    }
    (Some(ece), Some(mce))
}

fn auroc(predictions: &[CorrectnessPrediction]) -> Option<f64> {
    let positives: Vec<&CorrectnessPrediction> = predictions.iter().filter(|p| p.correct).collect();
    let negatives: Vec<&CorrectnessPrediction> = predictions
        .iter()
        .filter(|p| !p.correct)
        .collect();
    if positives.is_empty() || negatives.is_empty() {
        return None;
    }
    let mut wins = 0.0;
    let mut pairs = 0usize;
    for positive in &positives {
        for negative in &negatives {
            pairs += 1;
            if positive.confidence > negative.confidence {
                wins += 1.0;
            } else if positive.confidence == negative.confidence {
                wins += 0.5;
            }
        }
    }
    Some(wins / pairs as f64)
}

fn detection_report(observations: &[WeakAssumptionObservation]) -> BinaryDetectionReport {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut tn = 0usize;
    let mut fn_ = 0usize;
    for observation in observations {
        match (
            observation.weak_assumption_present,
            observation.weak_assumption_detected,
        ) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (false, false) => tn += 1,
            (true, false) => fn_ += 1,
        }
    }
    let precision = ratio(tp, tp + fp);
    let recall = ratio(tp, tp + fn_);
    let f1 = match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        _ => None,
    };
    BinaryDetectionReport {
        observations: observations.len(),
        true_positive: tp,
        false_positive: fp,
        true_negative: tn,
        false_negative: fn_,
        precision,
        recall,
        f1,
        accuracy: ratio(tp + tn, observations.len()),
    }
}

fn revision_metrics(revisions: &[ConfidenceRevisionObservation]) -> (Option<f64>, Option<f64>) {
    if revisions.is_empty() {
        return (None, None);
    }
    let mut correct_direction = 0usize;
    let mut signed_total = 0.0;
    for revision in revisions {
        let delta = revision.after_confidence - revision.before_confidence;
        let (correct, signed) = match revision.expected_direction {
            ConfidenceRevisionDirection::Increase => (delta > 0.0, delta),
            ConfidenceRevisionDirection::Decrease => (delta < 0.0, -delta),
            ConfidenceRevisionDirection::Stable => {
                let excess = delta.abs() - revision.stable_tolerance;
                (excess <= 0.0, -excess.max(0.0))
            }
        };
        if correct {
            correct_direction += 1;
        }
        signed_total += signed;
    }
    (
        Some(correct_direction as f64 / revisions.len() as f64),
        Some(signed_total / revisions.len() as f64),
    )
}

fn conditional_rate<'a, I, F>(items: I, predicate: F) -> Option<f64>
where
    I: IntoIterator<Item = &'a CorrectnessPrediction>,
    F: Fn(&CorrectnessPrediction) -> bool,
{
    let mut count = 0usize;
    let mut matched = 0usize;
    for item in items {
        count += 1;
        if predicate(item) {
            matched += 1;
        }
    }
    ratio(matched, count)
}

fn mean_bool<I>(values: I) -> Option<f64>
where
    I: IntoIterator<Item = bool>,
{
    let mut count = 0usize;
    let mut sum = 0usize;
    for value in values {
        count += 1;
        sum += usize::from(value);
    }
    ratio(sum, count)
}

fn mean_f64<I>(values: I) -> Option<f64>
where
    I: IntoIterator<Item = f64>,
{
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count += 1;
        sum += value;
    }
    (count != 0).then(|| sum / count as f64)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prediction(
        id: &str,
        confidence: f64,
        correct: bool,
        asserted: bool,
    ) -> CorrectnessPrediction {
        CorrectnessPrediction {
            episode_id: id.into(),
            task_family_id: "general".into(),
            confidence_target_episode_id: id.into(),
            confidence,
            correct,
            asserted,
        }
    }


    fn prospective_forecast(
        id: &str,
        episode: &str,
        probability: f64,
        family: &str,
    ) -> CorrectnessForecastV1 {
        CorrectnessForecastV1 {
            schema_version: CORRECTNESS_FORECAST_SCHEMA_VERSION,
            forecast_id: id.into(),
            episode_id: episode.into(),
            task_family_id: family.into(),
            task_taxonomy_id: "rq-taxonomy-v1".into(),
            outcome_profile_id: "answer-correct-under-policy-v2".into(),
            subject_id: "symthaea-test-subject".into(),
            model_profile_id: "symthaea-profile-sha256:abc123".into(),
            input_snapshot_ref: format!("input-snapshot:{episode}"),
            predicted_probability: probability,
            asserted: true,
        }
    }

    fn prospective_outcome(
        id: &str,
        episode: &str,
        correct: bool,
    ) -> CorrectnessOutcomeV1 {
        CorrectnessOutcomeV1 {
            schema_version: CORRECTNESS_OUTCOME_SCHEMA_VERSION,
            forecast_id: id.into(),
            episode_id: episode.into(),
            outcome_profile_id: "answer-correct-under-policy-v2".into(),
            outcome_receipt_id: format!("outcome-receipt:{id}"),
            outcome_evidence_ref: format!("benchmark-evidence:{episode}"),
            correct,
        }
    }

    fn prospective_baseline(
        id: &str,
        method: ForecastBaselineMethod,
        family: &str,
        probability: f64,
        training_split: &str,
        training_samples: usize,
    ) -> ForecastBaselineV1 {
        ForecastBaselineV1 {
            schema_version: FORECAST_BASELINE_SCHEMA_VERSION,
            baseline_id: id.into(),
            method,
            task_family_id: family.into(),
            task_taxonomy_id: "rq-taxonomy-v1".into(),
            outcome_profile_id: "answer-correct-under-policy-v2".into(),
            predicted_probability: probability,
            training_sample_count: training_samples,
            training_split_id: training_split.into(),
            training_corpus_manifest_ref: format!("calibration-manifest:{id}"),
        }
    }

    #[test]
    fn holdout_baselines_are_family_scoped_and_scored_only_on_frozen_evaluation_split() {
        let frozen = freeze_correctness_forecasts_for_split(
            vec![
                prospective_forecast("f-1", "episode-1", 0.99, "reasoning"),
                prospective_forecast("f-2", "episode-2", 0.90, "reasoning"),
                prospective_forecast("f-3", "episode-3", 0.40, "retrieval"),
            ],
            5,
            vec![0.5, 0.9],
            "holdout-v1".into(),
            "holdout-manifest-v1".into(),
        )
        .unwrap_or_else(|e| panic!("holdout freeze must succeed: {e}"));
        let baselines = vec![
            prospective_baseline(
                "reasoning-base-rate",
                ForecastBaselineMethod::ConstantBaseRate,
                "reasoning",
                0.20,
                "calibration-v1",
                500,
            ),
            prospective_baseline(
                "reasoning-recent",
                ForecastBaselineMethod::RecentEmpiricalAccuracy,
                "reasoning",
                0.25,
                "calibration-v1",
                50,
            ),
            prospective_baseline(
                "retrieval-base-rate",
                ForecastBaselineMethod::ConstantBaseRate,
                "retrieval",
                0.60,
                "calibration-v1",
                400,
            ),
            prospective_baseline(
                "retrieval-recent",
                ForecastBaselineMethod::RecentEmpiricalAccuracy,
                "retrieval",
                0.55,
                "calibration-v1",
                40,
            ),
        ];
        let report = evaluate_frozen_forecasts_with_baselines(
            &frozen,
            &[
                prospective_outcome("f-1", "episode-1", false),
                prospective_outcome("f-2", "episode-2", false),
                prospective_outcome("f-3", "episode-3", true),
            ],
            &[],
            &[],
            &baselines,
        )
        .unwrap_or_else(|e| panic!("baseline comparison must succeed: {e}"));
        let comparison = report.baseline_comparison.expect("comparison report");
        assert_eq!(comparison.evaluation_split_id, "holdout-v1");
        assert_eq!(
            comparison.evaluation_corpus_manifest_ref,
            "holdout-manifest-v1"
        );
        assert_eq!(comparison.family_reports.len(), 2);
        let reasoning = comparison
            .family_reports
            .iter()
            .find(|r| r.task_family_id == "reasoning")
            .expect("reasoning family");
        assert_eq!(reasoning.evaluation_episodes, 2);
        assert_eq!(reasoning.baselines.len(), 2);
        assert_eq!(
            reasoning.candidate_brier_score,
            Some((0.99_f64.powi(2) + 0.90_f64.powi(2)) / 2.0)
        );
        assert!(reasoning.baselines.iter().all(|b| b.evaluation_split_id == "holdout-v1"));
        assert!(reasoning.baselines.iter().all(|b| {
            b.evaluation_corpus_manifest_ref == "holdout-manifest-v1"
        }));
        assert!(reasoning.baselines.iter().all(|b| b.training_split_id == "calibration-v1"));
        assert!(reasoning.baselines.iter().all(|b| {
            b.training_corpus_manifest_ref
                .starts_with("calibration-manifest:")
        }));
        assert!(reasoning
            .baselines
            .iter()
            .all(|b| b.candidate_brier_delta.unwrap_or(0.0) > 0.0));
        assert_eq!(
            reasoning.baselines.iter().map(|b| b.method).collect::<Vec<_>>(),
            vec![
                ForecastBaselineMethod::ConstantBaseRate,
                ForecastBaselineMethod::RecentEmpiricalAccuracy
            ]
        );
    }

    #[test]
    fn baseline_comparison_rejects_training_split_reuse_and_missing_family_baselines() {
        let frozen = freeze_correctness_forecasts_for_split(
            vec![prospective_forecast("f-1", "episode-1", 0.8, "reasoning")],
            5,
            vec![0.5],
            "holdout-v1".into(),
            "holdout-manifest-v1".into(),
        )
        .unwrap_or_else(|e| panic!("holdout freeze must succeed: {e}"));
        let reused_split = prospective_baseline(
            "leaky-base",
            ForecastBaselineMethod::ConstantBaseRate,
            "reasoning",
            0.5,
            "holdout-v1",
            25,
        );
        assert!(matches!(
            evaluate_frozen_forecasts_with_baselines(
                &frozen,
                &[prospective_outcome("f-1", "episode-1", true)],
                &[],
                &[],
                &[reused_split]
            ),
            Err(MetacognitionEvaluationError::BaselineTrainingSplitEqualsEvaluation { .. })
        ));

        let mut reused_manifest = prospective_baseline(
            "reused-manifest",
            ForecastBaselineMethod::ConstantBaseRate,
            "reasoning",
            0.5,
            "calibration-v1",
            25,
        );
        reused_manifest.training_corpus_manifest_ref = "holdout-manifest-v1".into();
        assert!(matches!(
            evaluate_frozen_forecasts_with_baselines(
                &frozen,
                &[prospective_outcome("f-1", "episode-1", true)],
                &[],
                &[],
                &[reused_manifest]
            ),
            Err(MetacognitionEvaluationError::BaselineTrainingManifestEqualsEvaluation { .. })
        ));

        let one_method = prospective_baseline(
            "reasoning-base-rate",
            ForecastBaselineMethod::ConstantBaseRate,
            "reasoning",
            0.5,
            "calibration-v1",
            25,
        );
        assert!(matches!(
            evaluate_frozen_forecasts_with_baselines(
                &frozen,
                &[prospective_outcome("f-1", "episode-1", true)],
                &[],
                &[],
                &[one_method]
            ),
            Err(MetacognitionEvaluationError::MissingBaselineForTaskFamily {
                method: ForecastBaselineMethod::RecentEmpiricalAccuracy,
                task_family_id
            }) if task_family_id == "reasoning"
        ));
    }

    #[test]
    fn baseline_comparison_requires_explicit_split_identity_and_matching_target() {
        let legacy_frozen = freeze_correctness_forecasts(
            vec![prospective_forecast("f-1", "episode-1", 0.8, "reasoning")],
            5,
            vec![0.5],
        )
        .unwrap_or_else(|e| panic!("legacy freeze must succeed: {e}"));
        let baseline = prospective_baseline(
            "reasoning-base-rate",
            ForecastBaselineMethod::ConstantBaseRate,
            "reasoning",
            0.5,
            "calibration-v1",
            25,
        );
        assert!(matches!(
            evaluate_frozen_forecasts_with_baselines(
                &legacy_frozen,
                &[prospective_outcome("f-1", "episode-1", true)],
                &[],
                &[],
                &[baseline.clone()]
            ),
            Err(MetacognitionEvaluationError::MissingEvaluationSplit)
        ));

        let split_frozen = freeze_correctness_forecasts_for_split(
            vec![prospective_forecast("f-1", "episode-1", 0.8, "reasoning")],
            5,
            vec![0.5],
            "holdout-v1".into(),
            "holdout-manifest-v1".into(),
        )
        .unwrap_or_else(|e| panic!("holdout freeze must succeed: {e}"));
        let mut wrong_target = baseline;
        wrong_target.outcome_profile_id = "self-correction-succeeds".into();
        let recent = prospective_baseline(
            "reasoning-recent",
            ForecastBaselineMethod::RecentEmpiricalAccuracy,
            "reasoning",
            0.6,
            "calibration-v1",
            25,
        );
        assert!(matches!(
            evaluate_frozen_forecasts_with_baselines(
                &split_frozen,
                &[prospective_outcome("f-1", "episode-1", true)],
                &[],
                &[],
                &[wrong_target, recent]
            ),
            Err(MetacognitionEvaluationError::BaselineOutcomeProfileMismatch { .. })
        ));
    }

    #[test]
    fn frozen_forecast_binding_retains_outcome_refs_and_scores_stable_but_wrong_forecasts() {
        let frozen = freeze_correctness_forecasts(
            vec![
                prospective_forecast("f-1", "episode-1", 0.99, "reasoning"),
                prospective_forecast("f-2", "episode-2", 0.25, "retrieval"),
            ],
            5,
            vec![0.5, 0.9],
        )
        .unwrap_or_else(|e| panic!("freeze must succeed: {e}"));
        let prior_probability = frozen.forecasts()[0].predicted_probability;
        let report = evaluate_frozen_correctness_forecasts(
            &frozen,
            &[
                prospective_outcome("f-2", "episode-2", false),
                prospective_outcome("f-1", "episode-1", false),
            ],
            &[],
            &[],
        )
        .unwrap_or_else(|e| panic!("binding must succeed: {e}"));
        assert_eq!(report.predictions, 2);
        assert_eq!(
            report.brier_score,
            Some((0.99_f64.powi(2) + 0.25_f64.powi(2)) / 2.0)
        );
        let binding = report.forecast_outcome_binding.expect("binding report");
        assert_eq!(
            binding.forecast_ids.iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["f-1", "f-2"]
        );
        assert_eq!(
            binding
                .outcome_receipt_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["outcome-receipt:f-1", "outcome-receipt:f-2"]
        );
        assert_eq!(
            binding
                .outcome_evidence_refs
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["benchmark-evidence:episode-1", "benchmark-evidence:episode-2"]
        );
        assert_eq!(
            frozen.forecasts()[0].predicted_probability,
            prior_probability
        );
        assert_eq!(report.current_episode_binding_rate, Some(1.0));
    }

    #[test]
    fn frozen_forecast_set_roundtrips_through_validating_deserialization() {
        let frozen = freeze_correctness_forecasts(
            vec![prospective_forecast("f-1", "episode-1", 0.6, "reasoning")],
            7,
            vec![0.6],
        )
        .unwrap_or_else(|e| panic!("freeze must succeed: {e}"));
        let encoded =
            serde_json::to_string(&frozen).unwrap_or_else(|e| panic!("serialize: {e}"));
        let decoded: FrozenCorrectnessForecastSet = serde_json::from_str(&encoded)
            .unwrap_or_else(|e| panic!("validated restore: {e}"));
        assert_eq!(decoded, frozen);
        assert_eq!(decoded.calibration_bins(), 7);
        assert_eq!(decoded.selective_thresholds(), &[0.6]);

        let mut tampered: serde_json::Value =
            serde_json::from_str(&encoded).unwrap_or_else(|e| panic!("parse JSON: {e}"));
        tampered["calibration_bins"] = serde_json::json!(0);
        assert!(
            serde_json::from_value::<FrozenCorrectnessForecastSet>(tampered.clone()).is_err(),
            "deserialization must reject policy values that could not be frozen"
        );
        tampered["calibration_bins"] = serde_json::json!(7);
        tampered["evaluation_split_id"] = serde_json::json!("holdout-v1");
        assert!(
            serde_json::from_value::<FrozenCorrectnessForecastSet>(tampered).is_err(),
            "split identity without a corpus manifest must fail closed"
        );
    }

    #[test]
    fn forecast_outcome_join_rejects_missing_duplicate_unknown_and_mismatched_receipts() {
        let frozen = freeze_correctness_forecasts(
            vec![
                prospective_forecast("f-1", "episode-1", 0.8, "reasoning"),
                prospective_forecast("f-2", "episode-2", 0.4, "reasoning"),
            ],
            5,
            vec![0.5],
        )
        .unwrap_or_else(|e| panic!("freeze must succeed: {e}"));
        assert!(matches!(
            evaluate_frozen_correctness_forecasts(
                &frozen,
                &[prospective_outcome("f-1", "episode-1", true)],
                &[],
                &[]
            ),
            Err(MetacognitionEvaluationError::MissingOutcomeForForecast(id)) if id == "f-2"
        ));
        let mut duplicate = prospective_outcome("f-1", "episode-1", true);
        duplicate.outcome_receipt_id = "another-receipt".into();
        assert!(matches!(
            evaluate_frozen_correctness_forecasts(
                &frozen,
                &[
                    prospective_outcome("f-1", "episode-1", true),
                    duplicate
                ],
                &[],
                &[]
            ),
            Err(MetacognitionEvaluationError::DuplicateOutcomeForecastId(id)) if id == "f-1"
        ));
        assert!(matches!(
            evaluate_frozen_correctness_forecasts(
                &frozen,
                &[prospective_outcome("unknown", "episode-1", true)],
                &[],
                &[]
            ),
            Err(MetacognitionEvaluationError::OutcomeForUnknownForecast(id)) if id == "unknown"
        ));
        assert!(matches!(
            evaluate_frozen_correctness_forecasts(
                &frozen,
                &[
                    prospective_outcome("f-1", "wrong-episode", true),
                    prospective_outcome("f-2", "episode-2", false)
                ],
                &[],
                &[]
            ),
            Err(MetacognitionEvaluationError::OutcomeEpisodeMismatch { forecast_id, .. })
                if forecast_id == "f-1"
        ));
        let mut wrong_profile = prospective_outcome("f-1", "episode-1", true);
        wrong_profile.outcome_profile_id = "self-correction-succeeds".into();
        assert!(matches!(
            evaluate_frozen_correctness_forecasts(
                &frozen,
                &[
                    wrong_profile,
                    prospective_outcome("f-2", "episode-2", false)
                ],
                &[],
                &[]
            ),
            Err(MetacognitionEvaluationError::OutcomeProfileMismatch { forecast_id, .. })
                if forecast_id == "f-1"
        ));
        let mut reuse = prospective_outcome("f-2", "episode-2", false);
        reuse.outcome_receipt_id = "outcome-receipt:f-1".into();
        assert!(matches!(
            evaluate_frozen_correctness_forecasts(
                &frozen,
                &[
                    prospective_outcome("f-1", "episode-1", true),
                    reuse
                ],
                &[],
                &[]
            ),
            Err(MetacognitionEvaluationError::DuplicateOutcomeReceipt(id))
                if id == "outcome-receipt:f-1"
        ));
    }

    #[test]
    fn forecast_freeze_rejects_mixed_target_and_missing_provenance_refs() {
        let mut different_target =
            prospective_forecast("f-2", "episode-2", 0.4, "reasoning");
        different_target.outcome_profile_id = "self-correction-succeeds".into();
        assert!(matches!(
            freeze_correctness_forecasts(
                vec![
                    prospective_forecast("f-1", "episode-1", 0.8, "reasoning"),
                    different_target
                ],
                5,
                vec![0.5]
            ),
            Err(MetacognitionEvaluationError::MixedForecastScope {
                field: "outcome_profile_id",
                ..
            })
        ));
        let mut missing_snapshot =
            prospective_forecast("f-1", "episode-1", 0.8, "reasoning");
        missing_snapshot.input_snapshot_ref.clear();
        assert!(matches!(
            freeze_correctness_forecasts(vec![missing_snapshot], 5, vec![0.5]),
            Err(MetacognitionEvaluationError::EmptyForecastField {
                field: "input_snapshot_ref",
                ..
            })
        ));
        assert!(matches!(
            freeze_correctness_forecasts(vec![], 5, vec![0.5]),
            Err(MetacognitionEvaluationError::EmptyForecastSet)
        ));
    }


    #[test]
    fn perfect_predictions_are_perfectly_calibrated_and_discriminative() {
        let observations = vec![
            prediction("p1", 1.0, true, true),
            prediction("p2", 1.0, true, true),
            prediction("p3", 0.0, false, false),
            prediction("p4", 0.0, false, false),
        ];
        let report = evaluate_metacognition(&observations, &[], &[], 10, &[0.5, 0.9])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.brier_score, Some(0.0));
        assert_eq!(report.expected_calibration_error, Some(0.0));
        assert_eq!(report.correctness_auroc, Some(1.0));
        assert_eq!(report.current_episode_binding_rate, Some(1.0));
        assert_eq!(report.asserted_risk, Some(0.0));
        assert_eq!(report.abstention_opportunity_cost, Some(0.0));
    }

    #[test]
    fn constant_confidence_has_no_discrimination() {
        let observations = vec![
            prediction("p1", 0.7, true, true),
            prediction("p2", 0.7, false, true),
        ];
        let report = evaluate_metacognition(&observations, &[], &[], 10, &[])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.correctness_auroc, Some(0.5));
        assert!((report.brier_score.unwrap_or_default() - 0.29).abs() < 1.0e-12);
        assert!((report.confidence_bias.unwrap_or_default() - 0.2).abs() < 1.0e-12);
    }

    #[test]
    fn stale_confidence_binding_is_measured_not_hidden() {
        let mut observations = vec![prediction("p1", 0.8, true, true)];
        observations[0].confidence_target_episode_id = "previous-episode".into();
        let report = evaluate_metacognition(&observations, &[], &[], 10, &[])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.current_episode_binding_rate, Some(0.0));
    }

    #[test]
    fn weak_assumption_metrics_keep_false_positives_and_false_negatives_visible() {
        let observations = vec![
            WeakAssumptionObservation {
                episode_id: "a".into(),
                weak_assumption_present: true,
                weak_assumption_detected: true,
            },
            WeakAssumptionObservation {
                episode_id: "b".into(),
                weak_assumption_present: true,
                weak_assumption_detected: false,
            },
            WeakAssumptionObservation {
                episode_id: "c".into(),
                weak_assumption_present: false,
                weak_assumption_detected: true,
            },
            WeakAssumptionObservation {
                episode_id: "d".into(),
                weak_assumption_present: false,
                weak_assumption_detected: false,
            },
        ];
        let report = evaluate_metacognition(&[], &observations, &[], 10, &[])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.weak_assumption_detection.true_positive, 1);
        assert_eq!(report.weak_assumption_detection.false_positive, 1);
        assert_eq!(report.weak_assumption_detection.false_negative, 1);
        assert_eq!(report.weak_assumption_detection.true_negative, 1);
        assert_eq!(report.weak_assumption_detection.accuracy, Some(0.5));
    }

    #[test]
    fn evidence_revision_scores_direction_not_storytelling() {
        let revisions = vec![
            ConfidenceRevisionObservation {
                pair_id: "contradiction".into(),
                before_episode_id: "p1-before".into(),
                after_episode_id: "p1-after".into(),
                before_confidence: 0.9,
                after_confidence: 0.4,
                expected_direction: ConfidenceRevisionDirection::Decrease,
                stable_tolerance: 0.01,
            },
            ConfidenceRevisionObservation {
                pair_id: "confirmation".into(),
                before_episode_id: "p2-before".into(),
                after_episode_id: "p2-after".into(),
                before_confidence: 0.4,
                after_confidence: 0.7,
                expected_direction: ConfidenceRevisionDirection::Increase,
                stable_tolerance: 0.01,
            },
        ];
        let report = evaluate_metacognition(&[], &[], &revisions, 10, &[])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.revision_direction_accuracy, Some(1.0));
        assert!((report.mean_expected_revision_delta.unwrap_or_default() - 0.4).abs() < 1.0e-12);
    }

    #[test]
    fn selective_risk_reports_coverage_and_errors_separately() {
        let observations = vec![
            prediction("p1", 0.95, true, true),
            prediction("p2", 0.90, false, true),
            prediction("p3", 0.60, true, true),
            prediction("p4", 0.40, false, false),
        ];
        let report = evaluate_metacognition(&observations, &[], &[], 10, &[0.5, 0.9])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.selective_risk_bound_method, "hoeffding-familywise-95-v1");
        assert!(report.selective_risk_bound_assumptions.contains("IID evaluation episodes"));
        assert!(report
            .selective_risk_bound_assumptions
            .contains("no distribution-shift guarantee"));
        assert!(report
            .selective_risk_bound_assumptions
            .contains("task-family taxonomy predeclared"));
        assert_eq!(report.selective_risk[0].selected, 3);
        assert_eq!(report.selective_risk[0].coverage, 0.75);
        assert!((report.selective_risk[0].risk.unwrap_or_default() - 1.0 / 3.0).abs() < 1.0e-12);
        assert!(report.selective_risk[0].risk_upper_bound_95.unwrap_or_default()
            >= report.selective_risk[0].risk.unwrap_or_default());
        assert_eq!(report.selective_risk[1].selected, 2);
        assert_eq!(report.selective_risk[1].risk, Some(0.5));
        assert!(report.selective_risk[1].risk_upper_bound_95.unwrap_or_default() >= 0.5);
    }

    #[test]
    fn selective_risk_upper_bound_is_finite_sample_conservative_and_multiplicity_corrected() {
        assert_eq!(hoeffding_familywise_risk_upper_bound(0, 0, 1), None);
        assert_eq!(hoeffding_familywise_risk_upper_bound(1, 0, 1), None);
        assert_eq!(hoeffding_familywise_risk_upper_bound(2, 1, 1), None);
        let expected_one = (20.0_f64.ln() / 200.0).sqrt();
        let one_threshold = hoeffding_familywise_risk_upper_bound(0, 100, 1)
            .expect("non-empty selected sample has a bound");
        let two_thresholds = hoeffding_familywise_risk_upper_bound(0, 100, 2)
            .expect("non-empty selected sample has a bound");
        assert!((one_threshold - expected_one).abs() < 1.0e-12);
        assert!(one_threshold > 0.0);
        assert!(two_thresholds > one_threshold,
            "family-wise correction must account for additional thresholds");
        assert_eq!(hoeffding_familywise_risk_upper_bound(100, 100, 4), Some(1.0));
        assert!((0..=100).all(|errors| {
            hoeffding_familywise_risk_upper_bound(errors, 100, 5)
                .is_some_and(|bound| (errors as f64 / 100.0) <= bound && bound <= 1.0)
        }));
    }

    #[test]
    fn task_family_reports_expose_pooled_calibration_cancellation() {
        let mut observations = Vec::new();
        for index in 0..5 {
            let mut p = prediction(&format!("reasoning-{index}"), 0.9, true, true);
            p.task_family_id = "reasoning".into();
            observations.push(p);
        }
        for index in 0..5 {
            let mut p = prediction(&format!("retrieval-{index}"), 0.9, index != 0, true);
            p.task_family_id = "retrieval".into();
            observations.push(p);
        }

        let report = evaluate_metacognition(&observations, &[], &[], 10, &[0.5, 0.9])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.empirical_accuracy, Some(0.9));
        assert!(
            (report.mean_confidence.unwrap_or_default() - 0.9).abs() < 1.0e-12
        );
        assert!(
            report.expected_calibration_error.unwrap_or_default().abs() < 1.0e-12
        );
        assert_eq!(report.task_family_calibration.len(), 2);
        assert_eq!(report.task_family_calibration[0].task_family_id, "reasoning");
        assert!(
            report.task_family_calibration[0]
                .expected_calibration_error
                .unwrap_or_default()
                .abs()
                < 1.0e-12
        );
        assert_eq!(report.task_family_calibration[1].task_family_id, "retrieval");
        assert!(
            (report.task_family_calibration[1]
                .expected_calibration_error
                .unwrap_or_default()
                - 0.1)
                .abs()
                < 1.0e-12
        );
        assert!(report.task_family_calibration.iter().all(|family| {
            family.selective_risk.iter().all(|point| {
                point.risk_upper_bound_95.unwrap_or_default() >= point.risk.unwrap_or_default()
            })
        }));
        assert!(report.task_family_calibration[0].selective_risk[0].risk_upper_bound_95
            >= report.selective_risk[0].risk_upper_bound_95);

        assert_eq!(report.calibration_bins.len(), 10);
        let pooled_top_bin = report.calibration_bins[9];
        assert_eq!(pooled_top_bin.episodes, 10);
        assert!((pooled_top_bin.mean_confidence.unwrap_or_default() - 0.9).abs() < 1.0e-12);
        assert_eq!(pooled_top_bin.empirical_accuracy, Some(0.9));
        assert!(pooled_top_bin.accuracy_lower_95.unwrap_or_default() <= 0.9);
        assert!(pooled_top_bin.accuracy_upper_95.unwrap_or_default() >= 0.9);
        assert_eq!(report.calibration_bins[0].episodes, 0);
        assert_eq!(report.calibration_bins[0].empirical_accuracy, None);
        assert_eq!(report.calibration_bins[0].accuracy_lower_95, None);
        assert_eq!(report.task_family_calibration[1].calibration_bins[9].episodes, 5);
        assert_eq!(
            report.task_family_calibration[1].calibration_bins[9].empirical_accuracy,
            Some(0.8)
        );
        let pooled_width = pooled_top_bin.accuracy_upper_95.unwrap_or_default()
            - pooled_top_bin.accuracy_lower_95.unwrap_or_default();
        let subgroup_bin = report.task_family_calibration[1].calibration_bins[9];
        let subgroup_width = subgroup_bin.accuracy_upper_95.unwrap_or_default()
            - subgroup_bin.accuracy_lower_95.unwrap_or_default();
        assert!(subgroup_width > pooled_width, "smaller subgroup must show greater uncertainty");
    }

    #[test]
    fn empty_task_family_id_fails_closed() {
        let mut observation = prediction("p1", 0.5, true, true);
        observation.task_family_id.clear();
        assert!(matches!(
            evaluate_metacognition(&[observation], &[], &[], 10, &[]),
            Err(MetacognitionEvaluationError::EmptyTaskFamilyId { .. })
        ));
    }

    #[test]
    fn invalid_probabilities_and_duplicate_episode_predictions_fail_closed() {
        let invalid = prediction("p1", f64::NAN, true, true);
        assert!(matches!(
            evaluate_metacognition(&[invalid], &[], &[], 10, &[]),
            Err(MetacognitionEvaluationError::InvalidProbability { .. })
        ));

        let duplicate = prediction("same", 0.5, true, true);
        assert!(matches!(
            evaluate_metacognition(&[duplicate.clone(), duplicate], &[], &[], 10, &[]),
            Err(MetacognitionEvaluationError::DuplicatePredictionEpisode(_))
        ));
    }
}
