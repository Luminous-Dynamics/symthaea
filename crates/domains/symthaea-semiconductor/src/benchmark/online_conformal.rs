// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Online class-conditional conformal calibration with delayed metrology feedback.
//!
//! Predictions are registered before their outcomes are known. Calibration
//! updates consume only graph-validated labels that have matured by the
//! information cutoff. Revisions replace prior feedback; they are never
//! double-counted. Coverage degradation latches the state into abstention.

use chrono::{DateTime, FixedOffset, NaiveDateTime};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::benchmark::conformal_abstention::{
    ConformalCalibrationSampleV1, ConformalClassThresholdV1, ConformalPredictionSetV1,
    ConformalSubjectV1, SelectiveDecisionV1, SUBJECT_SCHEMA_V1,
};
use crate::benchmark::process_event_graph::{
    ClockBasisV1, MetrologyLabelStatusV1, PredictionInformationCutoffV1, ProcessEventGraphV1,
    SNAPSHOT_SCHEMA_V1, build_information_cutoff_snapshot_v1,
};

pub const ONLINE_STATE_SCHEMA_V1: &str = "luminous.semi-bench.online-conformal-state.v1";
pub const ONLINE_PREDICTION_SCHEMA_V1: &str = "luminous.semi-bench.online-conformal-prediction.v1";
pub const ONLINE_FEEDBACK_SCHEMA_V1: &str = "luminous.semi-bench.online-conformal-feedback.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OnlineConformalConfigV1 {
    pub alpha: f64,
    pub residual_window_per_class: usize,
    pub minimum_residual_support_per_class: usize,
    pub coverage_window_per_class: usize,
    pub minimum_coverage_observations_per_class: usize,
    /// Drift is suspected when observed class-conditional miscoverage exceeds
    /// alpha by at least this amount in the rolling monitoring window.
    pub maximum_miscoverage_excess: f64,
}

impl Default for OnlineConformalConfigV1 {
    fn default() -> Self {
        Self {
            alpha: 0.10,
            residual_window_per_class: 500,
            minimum_residual_support_per_class: 20,
            coverage_window_per_class: 100,
            minimum_coverage_observations_per_class: 30,
            maximum_miscoverage_excess: 0.10,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnlineCoverageStateV1 {
    MonitoringInsufficient,
    MonitoringWithinTolerance,
    CoverageDegraded,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OnlineCoverageDiagnosticV1 {
    pub class_positive: bool,
    pub sample_count: usize,
    pub miss_count: usize,
    pub observed_miscoverage: Option<f64>,
    pub allowed_miscoverage: f64,
    pub exceeds_tolerance: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OnlineResidualV1 {
    pub label_id: String,
    pub sample_id: String,
    pub prediction_id: Option<String>,
    pub positive: bool,
    pub nonconformity: f64,
    pub ingested_at: String,
    pub initial_calibration: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OnlinePredictionLedgerV1 {
    pub prediction_id: String,
    pub sample_id: String,
    pub subject_id: String,
    pub target_measurand_id: String,
    pub score: f64,
    pub model_sha256: String,
    pub issued_at: String,
    pub clock_domain_id: String,
    pub clock_basis: ClockBasisV1,
    pub calibration_state_sha256: String,
    pub prediction_set: ConformalPredictionSetV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OnlineCoverageObservationV1 {
    pub prediction_id: String,
    pub label_id: String,
    pub positive: bool,
    pub covered: bool,
    pub ingested_at: String,
    pub sequence: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OnlineFeedbackLedgerV1 {
    pub label_id: String,
    pub prediction_id: String,
    pub sample_id: String,
    pub subject_id: String,
    pub positive: bool,
    pub target_measurand_id: String,
    pub class_semantics_ref: String,
    pub class_semantics_sha256: String,
    pub revision: u64,
    pub supersedes_label_id: Option<String>,
    pub available_at: String,
    pub ingested_at: String,
    pub label_delay_micros: i64,
    pub sequence: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OnlineConformalStateV1 {
    pub schema: String,
    pub state_id: String,
    pub model_sha256: String,
    pub graph_id: String,
    pub graph_sha256: String,
    pub cutoff_id: String,
    pub clock_domain_id: String,
    pub clock_basis: ClockBasisV1,
    pub config: OnlineConformalConfigV1,
    pub residuals: Vec<OnlineResidualV1>,
    pub predictions: Vec<OnlinePredictionLedgerV1>,
    pub feedback_ledger: Vec<OnlineFeedbackLedgerV1>,
    pub active_label_by_prediction: BTreeMap<String, String>,
    pub coverage_history: Vec<OnlineCoverageObservationV1>,
    pub positive_coverage: OnlineCoverageDiagnosticV1,
    pub negative_coverage: OnlineCoverageDiagnosticV1,
    pub coverage_state: OnlineCoverageStateV1,
    pub coverage_degraded_at_sequence: Option<u64>,
    pub update_sequence: u64,
    pub last_ingested_at: String,
    /// Hash of the complete state with this field set to the empty string.
    pub state_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OnlinePredictionInputV1 {
    pub prediction_id: String,
    pub sample_id: String,
    pub subject_id: String,
    pub target_measurand_id: String,
    pub score: f64,
    pub issued_at: String,
    pub clock_domain_id: String,
    pub clock_basis: ClockBasisV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OnlinePredictionReceiptV1 {
    pub schema: String,
    pub state_sha256: String,
    pub model_sha256: String,
    pub calibration_state_sha256: String,
    pub negative_threshold: ConformalClassThresholdV1,
    pub positive_threshold: ConformalClassThresholdV1,
    pub coverage_state: OnlineCoverageStateV1,
    pub positive_coverage: OnlineCoverageDiagnosticV1,
    pub negative_coverage: OnlineCoverageDiagnosticV1,
    pub prediction_set: ConformalPredictionSetV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OnlineMaturedFeedbackV1 {
    pub label_id: String,
    pub prediction_id: String,
    pub sample_id: String,
    pub subject_id: String,
    pub positive: bool,
    pub target_measurand_id: String,
    /// Versioned interpretation that maps the physical measurement to the
    /// benchmark's binary defect class.
    pub class_semantics_ref: String,
    pub class_semantics_sha256: String,
    pub revision: u64,
    pub supersedes_label_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OnlineFeedbackReceiptV1 {
    pub schema: String,
    pub state_sha256: String,
    pub label_id: String,
    pub prediction_id: String,
    pub replaced_label_id: Option<String>,
    pub residual_window_positive: usize,
    pub residual_window_negative: usize,
    pub coverage_window_positive: usize,
    pub coverage_window_negative: usize,
    pub coverage_state: OnlineCoverageStateV1,
    pub positive_coverage: OnlineCoverageDiagnosticV1,
    pub negative_coverage: OnlineCoverageDiagnosticV1,
    pub coverage_degraded_at_sequence: Option<u64>,
    pub available_at: String,
    pub ingested_at: String,
    pub label_delay_micros: i64,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct OnlineConformalError(pub String);

fn require_text(value: &str, field: &str) -> Result<(), OnlineConformalError> {
    if value.trim().is_empty() {
        Err(OnlineConformalError(format!(
            "{field}: expected non-empty string"
        )))
    } else {
        Ok(())
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn timestamp_micros(
    value: &str,
    basis: &ClockBasisV1,
    field: &str,
) -> Result<i64, OnlineConformalError> {
    match basis {
        ClockBasisV1::OffsetAware => {
            let parsed = DateTime::<FixedOffset>::parse_from_rfc3339(value).map_err(|_| {
                OnlineConformalError(format!(
                    "{field}: expected RFC3339 timestamp with an explicit offset"
                ))
            })?;
            Ok(parsed.timestamp_micros())
        }
        ClockBasisV1::SourceLocalNaive => {
            let formats = [
                "%Y-%m-%dT%H:%M:%S%.f",
                "%Y-%m-%d %H:%M:%S%.f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
            ];
            for format in formats {
                if let Ok(parsed) = NaiveDateTime::parse_from_str(value, format) {
                    return Ok(parsed.and_utc().timestamp_micros());
                }
            }
            Err(OnlineConformalError(format!(
                "{field}: invalid source-local naive timestamp"
            )))
        }
    }
}

fn state_digest(state: &OnlineConformalStateV1) -> Result<String, OnlineConformalError> {
    let mut material = state.clone();
    material.state_sha256.clear();
    let bytes = serde_json::to_vec(&material)
        .map_err(|error| OnlineConformalError(format!("state serialization failed: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn verify_state(state: &OnlineConformalStateV1) -> Result<(), OnlineConformalError> {
    if state.schema != ONLINE_STATE_SCHEMA_V1 {
        return Err(OnlineConformalError("state.schema: unsupported".into()));
    }
    if !valid_sha256(&state.model_sha256) || !valid_sha256(&state.graph_sha256) {
        return Err(OnlineConformalError(
            "state model/graph digest is invalid".into(),
        ));
    }
    if state_digest(state)? != state.state_sha256 {
        return Err(OnlineConformalError(
            "state_sha256: state payload integrity mismatch".into(),
        ));
    }
    Ok(())
}

fn validate_config(config: &OnlineConformalConfigV1) -> Result<(), OnlineConformalError> {
    if !config.alpha.is_finite() || !(0.0..1.0).contains(&config.alpha) {
        return Err(OnlineConformalError(
            "config.alpha: expected finite value in (0, 1)".into(),
        ));
    }
    if config.minimum_residual_support_per_class == 0
        || config.residual_window_per_class < config.minimum_residual_support_per_class
    {
        return Err(OnlineConformalError(
            "config residual window must be at least the positive minimum support".into(),
        ));
    }
    if config.minimum_coverage_observations_per_class == 0
        || config.coverage_window_per_class < config.minimum_coverage_observations_per_class
    {
        return Err(OnlineConformalError(
            "config coverage window must be at least the positive minimum support".into(),
        ));
    }
    if !config.maximum_miscoverage_excess.is_finite()
        || !(0.0..=1.0).contains(&config.maximum_miscoverage_excess)
    {
        return Err(OnlineConformalError(
            "config.maximum_miscoverage_excess: expected finite value in [0, 1]".into(),
        ));
    }
    Ok(())
}

fn thresholds(
    residuals: &[OnlineResidualV1],
    config: &OnlineConformalConfigV1,
) -> Result<(ConformalClassThresholdV1, ConformalClassThresholdV1), OnlineConformalError> {
    let mut negative: Vec<f64> = residuals
        .iter()
        .filter(|row| !row.positive)
        .map(|row| row.nonconformity)
        .collect();
    let mut positive: Vec<f64> = residuals
        .iter()
        .filter(|row| row.positive)
        .map(|row| row.nonconformity)
        .collect();
    negative.sort_by(f64::total_cmp);
    positive.sort_by(f64::total_cmp);
    if negative.len() < config.minimum_residual_support_per_class
        || positive.len() < config.minimum_residual_support_per_class
    {
        return Err(OnlineConformalError(
            "calibration support is below the configured per-class minimum".into(),
        ));
    }
    let threshold = |values: &[f64], class_positive: bool| {
        let count = values.len();
        let rank = (((count + 1) as f64) * (1.0 - config.alpha)).ceil() as usize;
        let unbounded = rank > count;
        ConformalClassThresholdV1 {
            class_positive,
            calibration_count: count,
            quantile_rank: rank,
            threshold: if unbounded { None } else { Some(values[rank - 1]) },
            unbounded_threshold: unbounded,
        }
    };
    Ok((threshold(&negative, false), threshold(&positive, true)))
}

fn contains(threshold: &ConformalClassThresholdV1, score: f64, positive: bool) -> bool {
    let nonconformity = if positive { 1.0 - score } else { score };
    threshold
        .threshold
        .map(|value| nonconformity <= value)
        .unwrap_or(true)
}

fn counts(state: &OnlineConformalStateV1) -> (usize, usize, usize, usize) {
    let residual_positive = state.residuals.iter().filter(|row| row.positive).count();
    let residual_negative = state.residuals.len() - residual_positive;
    let coverage_positive = state.coverage_history.iter().filter(|row| row.positive).count();
    let coverage_negative = state.coverage_history.len() - coverage_positive;
    (
        residual_positive,
        residual_negative,
        coverage_positive,
        coverage_negative,
    )
}

fn trim_per_class<T>(
    rows: &mut Vec<T>,
    maximum: usize,
    positive_of: impl Fn(&T) -> bool,
) {
    for positive in [false, true] {
        let matching: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter_map(|(index, row)| (positive_of(row) == positive).then_some(index))
            .collect();
        if matching.len() > maximum {
            let remove_count = matching.len() - maximum;
            let remove: BTreeSet<usize> = matching.into_iter().take(remove_count).collect();
            *rows = rows
                .drain(..)
                .enumerate()
                .filter_map(|(index, row)| (!remove.contains(&index)).then_some(row))
                .collect();
        }
    }
}

fn update_state_digest(state: &mut OnlineConformalStateV1) -> Result<(), OnlineConformalError> {
    state.state_sha256 = state_digest(state)?;
    Ok(())
}

fn verify_snapshot(
    graph: &ProcessEventGraphV1,
    cutoff: &PredictionInformationCutoffV1,
) -> Result<InformationCutoffSnapshotV1, OnlineConformalError> {
    build_information_cutoff_snapshot_v1(graph, cutoff)
        .map_err(|error| OnlineConformalError(format!("information snapshot rejected: {error}")))
}

/// Initializes a bounded online residual window from a matured class-conditional
/// calibration set. The model SHA is fixed for the lifetime of this state.
pub fn initialize_online_conformal_v1(
    subject: &ConformalSubjectV1,
    graph: &ProcessEventGraphV1,
    cutoff: &PredictionInformationCutoffV1,
    config: OnlineConformalConfigV1,
) -> Result<OnlineConformalStateV1, OnlineConformalError> {
    validate_config(&config)?;
    if subject.schema != SUBJECT_SCHEMA_V1 {
        return Err(OnlineConformalError("subject.schema: unsupported".into()));
    }
    require_text(&subject.case_id, "subject.case_id")?;
    if !valid_sha256(&subject.model_sha256) {
        return Err(OnlineConformalError(
            "subject.model_sha256: expected lowercase SHA-256".into(),
        ));
    }
    if (subject.alpha - config.alpha).abs() > f64::EPSILON {
        return Err(OnlineConformalError(
            "subject/config alpha mismatch".into(),
        ));
    }
    if !subject.prediction_samples.is_empty() {
        return Err(OnlineConformalError(
            "initial calibration subject must not contain prediction samples".into(),
        ));
    }
    let snapshot = verify_snapshot(graph, cutoff)?;
    if subject.training_sample_ids.is_empty() {
        return Err(OnlineConformalError(
            "training_sample_ids: expected non-empty collection".into(),
        ));
    }
    let training_ids: BTreeSet<&str> = subject
        .training_sample_ids
        .iter()
        .map(String::as_str)
        .collect();
    if training_ids.len() != subject.training_sample_ids.len() {
        return Err(OnlineConformalError(
            "training_sample_ids: duplicate identity".into(),
        ));
    }
    let mut calibration_samples = subject.calibration_samples.clone();
    calibration_samples.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    let mut sample_ids = BTreeSet::new();
    let mut label_ids = BTreeSet::new();
    let matured_ids: BTreeSet<&str> = snapshot
        .matured_validated_label_ids
        .iter()
        .map(String::as_str)
        .collect();
    for (index, sample) in calibration_samples.iter().enumerate() {
        require_text(&sample.sample_id, &format!("calibration[{index}].sample_id"))?;
        require_text(&sample.label_id, &format!("calibration[{index}].label_id"))?;
        if !sample.score.is_finite() || !(0.0..=1.0).contains(&sample.score) {
            return Err(OnlineConformalError(format!(
                "calibration[{index}].score: expected finite value in [0, 1]"
            )));
        }
        if !sample_ids.insert(sample.sample_id.as_str())
            || !label_ids.insert(sample.label_id.as_str())
        {
            return Err(OnlineConformalError(format!(
                "calibration[{index}]: duplicate sample or label identity"
            )));
        }
        if training_ids.contains(sample.sample_id.as_str()) {
            return Err(OnlineConformalError(format!(
                "calibration[{index}]: overlaps model training IDs"
            )));
        }
        if !matured_ids.contains(sample.label_id.as_str()) {
            return Err(OnlineConformalError(format!(
                "calibration[{index}]: label is not matured and validated at cutoff"
            )));
        }
    }
    let mut residuals: Vec<OnlineResidualV1> = calibration_samples
        .iter()
        .map(|sample| OnlineResidualV1 {
            label_id: sample.label_id.clone(),
            sample_id: sample.sample_id.clone(),
            prediction_id: None,
            positive: sample.positive,
            nonconformity: if sample.positive {
                1.0 - sample.score
            } else {
                sample.score
            },
            ingested_at: cutoff.cutoff_time.clone(),
            initial_calibration: true,
        })
        .collect();
    trim_per_class(
        &mut residuals,
        config.residual_window_per_class,
        |row| row.positive,
    );
    let positive_count = residuals.iter().filter(|row| row.positive).count();
    let negative_count = residuals.len() - positive_count;
    if positive_count < config.minimum_residual_support_per_class
        || negative_count < config.minimum_residual_support_per_class
    {
        return Err(OnlineConformalError(
            "initial calibration does not meet configured per-class support".into(),
        ));
    }

    let positive_coverage = coverage_diagnostic(&[], true, &config);
    let negative_coverage = coverage_diagnostic(&[], false, &config);
    let mut state = OnlineConformalStateV1 {
        schema: ONLINE_STATE_SCHEMA_V1.into(),
        state_id: subject.case_id.clone(),
        model_sha256: subject.model_sha256.clone(),
        graph_id: graph.graph_id.clone(),
        graph_sha256: snapshot.graph_sha256,
        cutoff_id: cutoff.cutoff_id.clone(),
        clock_domain_id: cutoff.clock_domain_id.clone(),
        clock_basis: cutoff.clock_basis.clone(),
        config,
        residuals,
        predictions: Vec::new(),
        feedback_ledger: Vec::new(),
        active_label_by_prediction: BTreeMap::new(),
        coverage_history: Vec::new(),
        positive_coverage,
        negative_coverage,
        coverage_state: OnlineCoverageStateV1::MonitoringInsufficient,
        coverage_degraded_at_sequence: None,
        update_sequence: 0,
        last_ingested_at: cutoff.cutoff_time.clone(),
        state_sha256: String::new(),
    };
    let _ = thresholds(&state.residuals, &state.config)?;
    update_state_digest(&mut state)?;
    Ok(state)
}

/// Registers a prediction and returns the class-conditional set under the
/// current residual window. Predictions are label-free and fixed to one model
/// version. Once coverage is degraded, the system abstains until reinitialized.
pub fn predict_online_conformal_v1(
    state: &mut OnlineConformalStateV1,
    input: &OnlinePredictionInputV1,
) -> Result<OnlinePredictionReceiptV1, OnlineConformalError> {
    verify_state(state)?;
    require_text(&input.prediction_id, "prediction_id")?;
    require_text(&input.sample_id, "sample_id")?;
    require_text(&input.subject_id, "subject_id")?;
    require_text(&input.target_measurand_id, "target_measurand_id")?;
    if !input.score.is_finite() || !(0.0..=1.0).contains(&input.score) {
        return Err(OnlineConformalError(
            "prediction.score: expected finite value in [0, 1]".into(),
        ));
    }
    if input.clock_domain_id != state.clock_domain_id || input.clock_basis != state.clock_basis {
        return Err(OnlineConformalError(
            "prediction clock is not aligned with calibration state".into(),
        ));
    }
    let issued = timestamp_micros(&input.issued_at, &input.clock_basis, "prediction.issued_at")?;
    let last = timestamp_micros(
        &state.last_ingested_at,
        &state.clock_basis,
        "state.last_ingested_at",
    )?;
    if issued < last {
        return Err(OnlineConformalError(
            "prediction.issued_at precedes current information cutoff".into(),
        ));
    }
    if state
        .predictions
        .iter()
        .any(|row| row.prediction_id == input.prediction_id || row.sample_id == input.sample_id)
    {
        return Err(OnlineConformalError(
            "prediction_id and sample_id must be unique within model epoch".into(),
        ));
    }
    let calibration_state_sha256 = state.state_sha256.clone();
    let (negative_threshold, positive_threshold) = thresholds(&state.residuals, &state.config)?;
    let mut negative_in_set = contains(&negative_threshold, input.score, false);
    let mut positive_in_set = contains(&positive_threshold, input.score, true);
    let mut decision = match (negative_in_set, positive_in_set) {
        (true, false) => SelectiveDecisionV1::PredictNegative,
        (false, true) => SelectiveDecisionV1::PredictPositive,
        _ => SelectiveDecisionV1::Abstain,
    };
    if matches!(&state.coverage_state, OnlineCoverageStateV1::CoverageDegraded) {
        negative_in_set = true;
        positive_in_set = true;
        decision = SelectiveDecisionV1::Abstain;
    }
    let prediction_set = ConformalPredictionSetV1 {
        sample_id: input.sample_id.clone(),
        score: input.score,
        negative_in_set,
        positive_in_set,
        set_size: (if negative_in_set { 1 } else { 0 })
            + (if positive_in_set { 1 } else { 0 }),
        decision,
    };
    state.predictions.push(OnlinePredictionLedgerV1 {
        prediction_id: input.prediction_id.clone(),
        sample_id: input.sample_id.clone(),
        subject_id: input.subject_id.clone(),
        target_measurand_id: input.target_measurand_id.clone(),
        score: input.score,
        model_sha256: state.model_sha256.clone(),
        issued_at: input.issued_at.clone(),
        clock_domain_id: input.clock_domain_id.clone(),
        clock_basis: input.clock_basis.clone(),
        calibration_state_sha256: calibration_state_sha256.clone(),
        prediction_set: prediction_set.clone(),
    });
    update_state_digest(state)?;
    Ok(OnlinePredictionReceiptV1 {
        schema: ONLINE_PREDICTION_SCHEMA_V1.into(),
        state_sha256: state.state_sha256.clone(),
        model_sha256: state.model_sha256.clone(),
        calibration_state_sha256,
        negative_threshold,
        positive_threshold,
        coverage_state: state.coverage_state.clone(),
        positive_coverage: state.positive_coverage.clone(),
        negative_coverage: state.negative_coverage.clone(),
        prediction_set,
    })
}

/// Consumes one matured, validated metrology result. The physical graph is
/// revalidated and the cutoff snapshot is rebuilt here; callers cannot make an
/// immature label trainable by fabricating a snapshot object.
pub fn ingest_online_metrology_feedback_v1(
    state: &mut OnlineConformalStateV1,
    graph: &ProcessEventGraphV1,
    cutoff: &PredictionInformationCutoffV1,
    feedback: &OnlineMaturedFeedbackV1,
) -> Result<OnlineFeedbackReceiptV1, OnlineConformalError> {
    verify_state(state)?;
    require_text(&feedback.label_id, "feedback.label_id")?;
    require_text(&feedback.prediction_id, "feedback.prediction_id")?;
    require_text(&feedback.sample_id, "feedback.sample_id")?;
    require_text(&feedback.subject_id, "feedback.subject_id")?;
    require_text(&feedback.target_measurand_id, "feedback.target_measurand_id")?;
    require_text(&feedback.class_semantics_ref, "feedback.class_semantics_ref")?;
    if !valid_sha256(&feedback.class_semantics_sha256) {
        return Err(OnlineConformalError(
            "feedback.class_semantics_sha256: expected lowercase SHA-256".into(),
        ));
    }
    if feedback.revision == 0 {
        return Err(OnlineConformalError(
            "feedback.revision: revisions start at one".into(),
        ));
    }
    if state.graph_id != graph.graph_id || graph.schema != crate::benchmark::process_event_graph::GRAPH_SCHEMA_V1 {
        return Err(OnlineConformalError(
            "feedback graph identity/schema does not match state".into(),
        ));
    }
    if cutoff.clock_domain_id != state.clock_domain_id || cutoff.clock_basis != state.clock_basis {
        return Err(OnlineConformalError(
            "feedback cutoff clock does not match online state".into(),
        ));
    }
    if cutoff.cutoff_id == state.cutoff_id {
        return Err(OnlineConformalError(
            "feedback cutoff must advance beyond the previous cutoff".into(),
        ));
    }
    let ingested_micros = timestamp_micros(&cutoff.cutoff_time, &cutoff.clock_basis, "cutoff")?;
    let last_micros = timestamp_micros(
        &state.last_ingested_at,
        &state.clock_basis,
        "state.last_ingested_at",
    )?;
    if ingested_micros < last_micros {
        return Err(OnlineConformalError(
            "feedback cutoff moves backwards in information time".into(),
        ));
    }
    let snapshot = verify_snapshot(graph, cutoff)?;
    if snapshot.schema != SNAPSHOT_SCHEMA_V1
        || !snapshot
            .matured_validated_label_ids
            .iter()
            .any(|id| id == &feedback.label_id)
    {
        return Err(OnlineConformalError(
            "feedback label is not matured and validated at the supplied cutoff".into(),
        ));
    }
    let graph_label = graph
        .metrology_labels
        .iter()
        .find(|label| label.label_id == feedback.label_id)
        .ok_or_else(|| OnlineConformalError("feedback label does not exist in graph".into()))?;
    if !matches!(&graph_label.status, MetrologyLabelStatusV1::Validated)
        || graph_label.subject_id != feedback.subject_id
        || graph_label.measurand_id != feedback.target_measurand_id
        || graph_label.revision != feedback.revision
        || graph_label.supersedes_label_id != feedback.supersedes_label_id
        || graph_label.measurement_clock_domain_id != state.clock_domain_id
        || graph_label.measurement_clock_basis != state.clock_basis
        || graph_label.availability_clock_domain_id != state.clock_domain_id
        || graph_label.availability_clock_basis != state.clock_basis
    {
        return Err(OnlineConformalError(
            "feedback interpretation does not match validated graph label".into(),
        ));
    }
    require_text(&graph_label.authority_ref, "graph_label.authority_ref")?;
    let available_at = timestamp_micros(
        &graph_label.available_at,
        &graph_label.availability_clock_basis,
        "graph_label.available_at",
    )?;
    if available_at > ingested_micros {
        return Err(OnlineConformalError(
            "feedback result is not yet available at ingestion cutoff".into(),
        ));
    }
    let measurement_at = timestamp_micros(
        &graph_label.measurement_time,
        &graph_label.measurement_clock_basis,
        "graph_label.measurement_time",
    )?;
    if measurement_at > available_at {
        return Err(OnlineConformalError(
            "measurement time is after result availability".into(),
        ));
    }
    let label_delay_micros = ingested_micros - available_at;
    if label_delay_micros < 0 {
        return Err(OnlineConformalError(
            "negative label delay is impossible at ingestion cutoff".into(),
        ));
    }
    let prediction = state
        .predictions
        .iter()
        .find(|row| row.prediction_id == feedback.prediction_id)
        .ok_or_else(|| {
            OnlineConformalError("feedback does not refer to a registered prediction".into())
        })?;
    if prediction.sample_id != feedback.sample_id
        || prediction.subject_id != feedback.subject_id
        || prediction.target_measurand_id != feedback.target_measurand_id
        || prediction.model_sha256 != state.model_sha256
    {
        return Err(OnlineConformalError(
            "feedback sample/subject/model does not match prediction ledger".into(),
        ));
    }
    if state
        .feedback_ledger
        .iter()
        .any(|row| row.label_id == feedback.label_id)
    {
        return Err(OnlineConformalError(
            "feedback label_id has already been ingested".into(),
        ));
    }
    let previous_label_id = state
        .active_label_by_prediction
        .get(&feedback.prediction_id)
        .cloned();
    match (&previous_label_id, &feedback.supersedes_label_id) {
        (None, None) if feedback.revision == 1 => {}
        (Some(previous), Some(supersedes)) if previous == supersedes => {
            let previous_row = state
                .feedback_ledger
                .iter()
                .find(|row| &row.label_id == previous)
                .ok_or_else(|| {
                    OnlineConformalError("active feedback ledger entry is missing".into())
                })?;
            if previous_row.prediction_id != feedback.prediction_id
                || previous_row.sample_id != feedback.sample_id
                || previous_row.subject_id != feedback.subject_id
                || previous_row.target_measurand_id != feedback.target_measurand_id
                || previous_row.class_semantics_sha256 != feedback.class_semantics_sha256
                || previous_row.revision >= feedback.revision
            {
                return Err(OnlineConformalError(
                    "superseding label must preserve identity and increase revision".into(),
                ));
            }
        }
        _ => {
            return Err(OnlineConformalError(
                "feedback revision/supersession does not match active label".into(),
            ));
        }
    }

    let covered = if feedback.positive {
        prediction.prediction_set.positive_in_set
    } else {
        prediction.prediction_set.negative_in_set
    };
    let next_sequence = state.update_sequence.saturating_add(1);
    let residual = OnlineResidualV1 {
        label_id: feedback.label_id.clone(),
        sample_id: feedback.sample_id.clone(),
        prediction_id: Some(feedback.prediction_id.clone()),
        positive: feedback.positive,
        nonconformity: if feedback.positive {
            1.0 - prediction.score
        } else {
            prediction.score
        },
        ingested_at: cutoff.cutoff_time.clone(),
        initial_calibration: false,
    };
    if let Some(previous) = previous_label_id.as_deref() {
        state.residuals.retain(|row| row.label_id != previous);
        state
            .coverage_history
            .retain(|row| row.prediction_id != feedback.prediction_id);
        state
            .active_label_by_prediction
            .remove(&feedback.prediction_id);
    }
    state.residuals.push(residual);
    trim_per_class(
        &mut state.residuals,
        state.config.residual_window_per_class,
        |row| row.positive,
    );
    state.coverage_history.push(OnlineCoverageObservationV1 {
        prediction_id: feedback.prediction_id.clone(),
        label_id: feedback.label_id.clone(),
        positive: feedback.positive,
        covered,
        ingested_at: cutoff.cutoff_time.clone(),
        sequence: next_sequence,
    });
    trim_per_class(
        &mut state.coverage_history,
        state.config.coverage_window_per_class,
        |row| row.positive,
    );
    state.feedback_ledger.push(OnlineFeedbackLedgerV1 {
        label_id: feedback.label_id.clone(),
        prediction_id: feedback.prediction_id.clone(),
        sample_id: feedback.sample_id.clone(),
        subject_id: feedback.subject_id.clone(),
        positive: feedback.positive,
        target_measurand_id: feedback.target_measurand_id.clone(),
        class_semantics_ref: feedback.class_semantics_ref.clone(),
        class_semantics_sha256: feedback.class_semantics_sha256.clone(),
        revision: feedback.revision,
        supersedes_label_id: feedback.supersedes_label_id.clone(),
        available_at: graph_label.available_at.clone(),
        ingested_at: cutoff.cutoff_time.clone(),
        label_delay_micros,
        sequence: next_sequence,
    });
    state.active_label_by_prediction.insert(
        feedback.prediction_id.clone(),
        feedback.label_id.clone(),
    );
    state.update_sequence = next_sequence;
    state.last_ingested_at = cutoff.cutoff_time.clone();
    state.graph_sha256 = snapshot.graph_sha256;
    state.cutoff_id = cutoff.cutoff_id.clone();

    state.positive_coverage =
        coverage_diagnostic(&state.coverage_history, true, &state.config);
    state.negative_coverage =
        coverage_diagnostic(&state.coverage_history, false, &state.config);
    if state.positive_coverage.exceeds_tolerance || state.negative_coverage.exceeds_tolerance {
        state.coverage_state = OnlineCoverageStateV1::CoverageDegraded;
        state.coverage_degraded_at_sequence.get_or_insert(next_sequence);
    } else if !matches!(
        &state.coverage_state,
        OnlineCoverageStateV1::CoverageDegraded
    ) {
        let monitoring_has_support = state.positive_coverage.sample_count
            >= state.config.minimum_coverage_observations_per_class
            && state.negative_coverage.sample_count
                >= state.config.minimum_coverage_observations_per_class;
        state.coverage_state = if monitoring_has_support {
            OnlineCoverageStateV1::MonitoringWithinTolerance
        } else {
            OnlineCoverageStateV1::MonitoringInsufficient
        };
    }
    update_state_digest(state)?;
    let (residual_positive, residual_negative, coverage_positive, coverage_negative) =
        counts(state);
    Ok(OnlineFeedbackReceiptV1 {
        schema: ONLINE_FEEDBACK_SCHEMA_V1.into(),
        state_sha256: state.state_sha256.clone(),
        label_id: feedback.label_id.clone(),
        prediction_id: feedback.prediction_id.clone(),
        replaced_label_id: previous_label_id,
        residual_window_positive: residual_positive,
        residual_window_negative: residual_negative,
        coverage_window_positive: coverage_positive,
        coverage_window_negative: coverage_negative,
        coverage_state: state.coverage_state.clone(),
        positive_coverage: state.positive_coverage.clone(),
        negative_coverage: state.negative_coverage.clone(),
        coverage_degraded_at_sequence: state.coverage_degraded_at_sequence,
        available_at: graph_label.available_at.clone(),
        ingested_at: cutoff.cutoff_time.clone(),
        label_delay_micros,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::process_event_graph::{
        ProcessEventEdgeV1, ProcessEventKindV1, ProcessEventV1, ProcessEventGraphV1,
    };

    const MODEL_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SOURCE_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const TRANSFORM_SHA: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn graph(label_ids: &[(&str, &str, u64, Option<&str>)]) -> ProcessEventGraphV1 {
        let events = vec![ProcessEventV1 {
            event_id: "met-event".into(),
            event_kind: ProcessEventKindV1::Metrology,
            subject_id: Some("wafer-100".into()),
            wafer_id: Some("wafer-100".into()),
            lot_id: Some("lot-9".into()),
            carrier_id: None,
            tool_id: Some("metrology-1".into()),
            chamber_id: None,
            module_id: None,
            process_step_id: None,
            recipe_id: None,
            recipe_revision: None,
            event_time: "2026-11-22T08:00:00".into(),
            event_clock_domain_id: "fab-local".into(),
            event_clock_basis: ClockBasisV1::SourceLocalNaive,
            available_at: "2026-11-22T08:00:01".into(),
            availability_clock_domain_id: "fab-local".into(),
            availability_clock_basis: ClockBasisV1::SourceLocalNaive,
            source_sequence: Some(1),
            source_artifact_sha256: SOURCE_SHA.into(),
            transform_sha256: TRANSFORM_SHA.into(),
            authority_ref: "metrology-source".into(),
        }];
        let mut labels = Vec::new();
        for (label_id, available_at, revision, supersedes) in label_ids {
            let label = crate::benchmark::process_event_graph::MetrologyLabelV1 {
                label_id: (*label_id).into(),
                measurement_event_id: "met-event".into(),
                subject_id: "wafer-100".into(),
                measurand_id: "defect-rate".into(),
                value: Some(0.25),
                unit: "fraction".into(),
                standard_uncertainty: Some(0.01),
                censoring_lower_bound: None,
                censoring_upper_bound: None,
                censoring_basis: None,
                status: MetrologyLabelStatusV1::Validated,
                measurement_time: "2026-11-22T08:00:00".into(),
                measurement_clock_domain_id: "fab-local".into(),
                measurement_clock_basis: ClockBasisV1::SourceLocalNaive,
                available_at: (*available_at).into(),
                availability_clock_domain_id: "fab-local".into(),
                availability_clock_basis: ClockBasisV1::SourceLocalNaive,
                instrument_id: Some("metrology-1".into()),
                calibration_ref: Some("cal-1".into()),
                revision: *revision,
                supersedes_label_id: supersedes.map(str::to_string),
                source_artifact_sha256: SOURCE_SHA.into(),
                authority_ref: "lims-validated".into(),
            };
            labels.push(label);
        }
        // The graph only needs one metrology event; all labels are revisions/
        // repeat validations of this subject's measured target.
        ProcessEventGraphV1 {
            schema: crate::benchmark::process_event_graph::GRAPH_SCHEMA_V1.into(),
            graph_id: "online-graph-100".into(),
            source_artifact_sha256: SOURCE_SHA.into(),
            transform_sha256: TRANSFORM_SHA.into(),
            events,
            edges: Vec::<ProcessEventEdgeV1>::new(),
            metrology_labels: labels,
        }
    }

    fn cutoff(id: &str, time: &str) -> PredictionInformationCutoffV1 {
        PredictionInformationCutoffV1 {
            cutoff_id: id.into(),
            cutoff_time: time.into(),
            clock_domain_id: "fab-local".into(),
            clock_basis: ClockBasisV1::SourceLocalNaive,
        }
    }

    fn subject() -> ConformalSubjectV1 {
        let calibration_samples: Vec<_> = (0..40)
            .map(|index| ConformalCalibrationSampleV1 {
                sample_id: format!("cal-sample-{index:02}"),
                label_id: format!("cal-label-{index:02}"),
                positive: index < 20,
                score: if index < 20 { 0.85 } else { 0.15 },
            })
            .collect();
        ConformalSubjectV1 {
            schema: SUBJECT_SCHEMA_V1.into(),
            case_id: "online-test".into(),
            model_sha256: MODEL_SHA.into(),
            alpha: 0.1,
            training_sample_ids: (0..20).map(|index| format!("train-{index:02}")).collect(),
            calibration_samples,
            prediction_samples: Vec::new(),
        }
    }

    fn graph_with_initial_calibration() -> ProcessEventGraphV1 {
        let mut input = graph(&[]);
        input.metrology_labels = (0..40)
            .map(|index| crate::benchmark::process_event_graph::MetrologyLabelV1 {
                label_id: format!("cal-label-{index:02}"),
                measurement_event_id: "met-event".into(),
                subject_id: "wafer-100".into(),
                measurand_id: "defect-rate".into(),
                value: Some(0.2),
                unit: "fraction".into(),
                standard_uncertainty: Some(0.01),
                censoring_lower_bound: None,
                censoring_upper_bound: None,
                censoring_basis: None,
                status: MetrologyLabelStatusV1::Validated,
                measurement_time: "2026-11-22T08:00:00".into(),
                measurement_clock_domain_id: "fab-local".into(),
                measurement_clock_basis: ClockBasisV1::SourceLocalNaive,
                available_at: "2026-11-22T09:00:00".into(),
                availability_clock_domain_id: "fab-local".into(),
                availability_clock_basis: ClockBasisV1::SourceLocalNaive,
                instrument_id: Some("metrology-1".into()),
                calibration_ref: Some("cal-1".into()),
                revision: 1,
                supersedes_label_id: None,
                source_artifact_sha256: SOURCE_SHA.into(),
                authority_ref: "lims-validated".into(),
            })
            .collect();
        input
    }

    fn config() -> OnlineConformalConfigV1 {
        OnlineConformalConfigV1 {
            alpha: 0.1,
            residual_window_per_class: 30,
            minimum_residual_support_per_class: 3,
            coverage_window_per_class: 10,
            minimum_coverage_observations_per_class: 3,
            maximum_miscoverage_excess: 0.0,
        }
    }

    fn initialise() -> OnlineConformalStateV1 {
        initialize_online_conformal_v1(
            &subject(),
            &graph_with_initial_calibration(),
            &cutoff("calibration-cutoff", "2026-11-22T10:00:00"),
            config(),
        )
        .unwrap()
    }

    #[test]
    fn prediction_is_label_free_bound_to_model_and_calibration_digest() {
        let mut state = initialise();
        let receipt = predict_online_conformal_v1(
            &mut state,
            &OnlinePredictionInputV1 {
                prediction_id: "prediction-1".into(),
                sample_id: "test-sample-1".into(),
                subject_id: "wafer-100".into(),
                target_measurand_id: "defect-rate".into(),
                score: 0.9,
                issued_at: "2026-11-22T10:05:00".into(),
                clock_domain_id: "fab-local".into(),
                clock_basis: ClockBasisV1::SourceLocalNaive,
            },
        )
        .unwrap();
        assert_eq!(receipt.model_sha256, MODEL_SHA);
        assert!(valid_sha256(&receipt.calibration_state_sha256));
        assert!(matches!(
            &receipt.coverage_state,
            OnlineCoverageStateV1::MonitoringInsufficient
        ));
        assert!(verify_state(&state).is_ok());
    }

    #[test]
    fn immature_label_cannot_update_online_calibration() {
        let mut state = initialise();
        predict_online_conformal_v1(
            &mut state,
            &OnlinePredictionInputV1 {
                prediction_id: "prediction-1".into(),
                sample_id: "test-sample-1".into(),
                subject_id: "wafer-100".into(),
                target_measurand_id: "defect-rate".into(),
                score: 0.9,
                issued_at: "2026-11-22T10:05:00".into(),
                clock_domain_id: "fab-local".into(),
                clock_basis: ClockBasisV1::SourceLocalNaive,
            },
        )
        .unwrap();
        let graph = graph(&[("label-new", "2026-11-22T12:00:00", 1, None)]);
        let result = ingest_online_metrology_feedback_v1(
            &mut state,
            &graph,
            &cutoff("cutoff-too-early", "2026-11-22T11:00:00"),
            &OnlineMaturedFeedbackV1 {
                label_id: "label-new".into(),
                prediction_id: "prediction-1".into(),
                sample_id: "test-sample-1".into(),
                subject_id: "wafer-100".into(),
                positive: true,
                target_measurand_id: "defect-rate".into(),
                class_semantics_ref: "defect-threshold-v1".into(),
                class_semantics_sha256: TRANSFORM_SHA.into(),
                revision: 1,
                supersedes_label_id: None,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn matured_feedback_replaces_no_labels_and_updates_one_residual() {
        let mut state = initialise();
        predict_online_conformal_v1(
            &mut state,
            &OnlinePredictionInputV1 {
                prediction_id: "prediction-1".into(),
                sample_id: "wafer-100".into(),
                subject_id: "wafer-100".into(),
                target_measurand_id: "defect-rate".into(),
                score: 0.9,
                issued_at: "2026-11-22T10:05:00".into(),
                clock_domain_id: "fab-local".into(),
                clock_basis: ClockBasisV1::SourceLocalNaive,
            },
        )
        .unwrap();
        let graph = graph(&[("label-new", "2026-11-22T10:30:00", 1, None)]);
        let result = ingest_online_metrology_feedback_v1(
            &mut state,
            &graph,
            &cutoff("cutoff-after-label", "2026-11-22T10:45:00"),
            &OnlineMaturedFeedbackV1 {
                label_id: "label-new".into(),
                prediction_id: "prediction-1".into(),
                sample_id: "wafer-100".into(),
                subject_id: "wafer-100".into(),
                positive: true,
                target_measurand_id: "defect-rate".into(),
                class_semantics_ref: "defect-threshold-v1".into(),
                class_semantics_sha256: TRANSFORM_SHA.into(),
                revision: 1,
                supersedes_label_id: None,
            },
        )
        .unwrap();
        assert_eq!(result.label_id, "label-new");
        assert_eq!(result.label_delay_micros, 900_000_000);
        assert_eq!(state.active_label_by_prediction.get("prediction-1").unwrap(), "label-new");
        assert!(state.residuals.iter().any(|row| row.label_id == "label-new"));
        assert!(verify_state(&state).is_ok());
    }

    #[test]
    fn label_revision_replaces_previous_residual_instead_of_double_counting() {
        let mut state = initialise();
        predict_online_conformal_v1(
            &mut state,
            &OnlinePredictionInputV1 {
                prediction_id: "prediction-1".into(),
                sample_id: "wafer-100".into(),
                subject_id: "wafer-100".into(),
                score: 0.9,
                issued_at: "2026-11-22T10:05:00".into(),
                clock_domain_id: "fab-local".into(),
                clock_basis: ClockBasisV1::SourceLocalNaive,
            },
        )
        .unwrap();
        let first_graph = graph(&[("label-v1", "2026-11-22T10:30:00", 1, None)]);
        ingest_online_metrology_feedback_v1(
            &mut state,
            &first_graph,
            &cutoff("cutoff-v1", "2026-11-22T10:45:00"),
            &OnlineMaturedFeedbackV1 {
                label_id: "label-v1".into(),
                prediction_id: "prediction-1".into(),
                sample_id: "wafer-100".into(),
                subject_id: "wafer-100".into(),
                positive: true,
                target_measurand_id: "defect-rate".into(),
                class_semantics_ref: "defect-threshold-v1".into(),
                class_semantics_sha256: TRANSFORM_SHA.into(),
                revision: 1,
                supersedes_label_id: None,
            },
        )
        .unwrap();
        let second_graph = graph(&[
            ("label-v1", "2026-11-22T10:30:00", 1, None),
            ("label-v2", "2026-11-22T11:00:00", 2, Some("label-v1")),
        ]);
        let result = ingest_online_metrology_feedback_v1(
            &mut state,
            &second_graph,
            &cutoff("cutoff-v2", "2026-11-22T11:15:00"),
            &OnlineMaturedFeedbackV1 {
                label_id: "label-v2".into(),
                prediction_id: "prediction-1".into(),
                sample_id: "wafer-100".into(),
                subject_id: "wafer-100".into(),
                positive: false,
                target_measurand_id: "defect-rate".into(),
                class_semantics_ref: "defect-threshold-v1".into(),
                revision: 2,
                supersedes_label_id: Some("label-v1".into()),
            },
        )
        .unwrap();
        assert_eq!(result.replaced_label_id.as_deref(), Some("label-v1"));
        assert!(!state.residuals.iter().any(|row| row.label_id == "label-v1"));
        assert!(state.residuals.iter().any(|row| row.label_id == "label-v2"));
        assert_eq!(
            state.coverage_history.iter().filter(|row| row.prediction_id == "prediction-1").count(),
            1
        );
    }

    #[test]
    fn coverage_degradation_latches_abstention() {
        let mut state = initialise();
        for index in 0..3 {
            predict_online_conformal_v1(
                &mut state,
                &OnlinePredictionInputV1 {
                    prediction_id: format!("prediction-{index}"),
                    sample_id: format!("sample-{index}"),
                    subject_id: format!("wafer-{index}"),
                    target_measurand_id: "defect-rate".into(),
                    score: 0.01,
                    issued_at: format!("2026-11-22T10:{:02}:00", 5 + index),
                    clock_domain_id: "fab-local".into(),
                    clock_basis: ClockBasisV1::SourceLocalNaive,
                },
            )
            .unwrap();
        }
        let mut labels = Vec::new();
        for index in 0..3 {
            labels.push(crate::benchmark::process_event_graph::MetrologyLabelV1 {
                label_id: format!("label-{index}"),
                measurement_event_id: "met-event".into(),
                subject_id: format!("wafer-{index}"),
                measurand_id: "defect-rate".into(),
                value: Some(0.2),
                unit: "fraction".into(),
                standard_uncertainty: Some(0.01),
                censoring_lower_bound: None,
                censoring_upper_bound: None,
                censoring_basis: None,
                status: MetrologyLabelStatusV1::Validated,
                measurement_time: "2026-11-22T08:00:00".into(),
                measurement_clock_domain_id: "fab-local".into(),
                measurement_clock_basis: ClockBasisV1::SourceLocalNaive,
                available_at: "2026-11-22T10:30:00".into(),
                availability_clock_domain_id: "fab-local".into(),
                availability_clock_basis: ClockBasisV1::SourceLocalNaive,
                instrument_id: None,
                calibration_ref: None,
                revision: 1,
                supersedes_label_id: None,
                source_artifact_sha256: SOURCE_SHA.into(),
                authority_ref: "lims-validated".into(),
            });
        }
        let mut test_graph = graph(&[]);
        test_graph.events[0].subject_id = Some("wafer-0".into());
        test_graph.metrology_labels = labels;
        // Use separate events per wafer so source identity remains explicit.
        test_graph.events = (0..3)
            .map(|index| ProcessEventV1 {
                event_id: format!("met-event-{index}"),
                subject_id: Some(format!("wafer-{index}")),
                wafer_id: Some(format!("wafer-{index}")),
                event_kind: ProcessEventKindV1::Metrology,
                lot_id: Some("lot-9".into()),
                carrier_id: None,
                tool_id: Some("metrology-1".into()),
                chamber_id: None,
                module_id: None,
                process_step_id: None,
                recipe_id: None,
                recipe_revision: None,
                event_time: "2026-11-22T08:00:00".into(),
                event_clock_domain_id: "fab-local".into(),
                event_clock_basis: ClockBasisV1::SourceLocalNaive,
                available_at: "2026-11-22T08:00:01".into(),
                availability_clock_domain_id: "fab-local".into(),
                availability_clock_basis: ClockBasisV1::SourceLocalNaive,
                source_sequence: Some(index + 1),
                source_artifact_sha256: SOURCE_SHA.into(),
                transform_sha256: TRANSFORM_SHA.into(),
                authority_ref: "metrology-source".into(),
            })
            .collect();
        for (index, label) in test_graph.metrology_labels.iter_mut().enumerate() {
            label.measurement_event_id = format!("met-event-{index}");
        }
        for index in 0..3 {
            let cutoff = cutoff(
                &format!("feedback-cutoff-{index}"),
                &format!("2026-11-22T10:{:02}:00", 30 + index),
            );
            ingest_online_metrology_feedback_v1(
                &mut state,
                &test_graph,
                &cutoff,
                &OnlineMaturedFeedbackV1 {
                    label_id: format!("label-{index}"),
                    prediction_id: format!("prediction-{index}"),
                    sample_id: format!("sample-{index}"),
                    subject_id: format!("wafer-{index}"),
                    positive: true,
                    target_measurand_id: "defect-rate".into(),
                    class_semantics_ref: "defect-threshold-v1".into(),
                    class_semantics_sha256: TRANSFORM_SHA.into(),
                    revision: 1,
                    supersedes_label_id: None,
                },
            )
            .unwrap();
        }
        assert!(matches!(&state.coverage_state, OnlineCoverageStateV1::CoverageDegraded));
        let receipt = predict_online_conformal_v1(
            &mut state,
            &OnlinePredictionInputV1 {
                prediction_id: "prediction-after-drift".into(),
                sample_id: "sample-after-drift".into(),
                subject_id: "wafer-after-drift".into(),
                target_measurand_id: "defect-rate".into(),
                score: 0.9,
                issued_at: "2026-11-22T10:40:00".into(),
                clock_domain_id: "fab-local".into(),
                clock_basis: ClockBasisV1::SourceLocalNaive,
            },
        )
        .unwrap();
        assert!(matches!(&receipt.coverage_state, OnlineCoverageStateV1::CoverageDegraded));
        assert!(matches!(&receipt.prediction_set.decision, SelectiveDecisionV1::Abstain));
        assert_eq!(receipt.prediction_set.set_size, 2);
    }
}
