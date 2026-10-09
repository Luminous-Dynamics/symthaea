// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned decision-cohort scoring for prospective metacognition.
//!
//! Unlike the v1 correctness-only scorer, this API represents both asserted answers and
//! abstentions while keeping correctness calibration restricted to explicitly scored answers.
//! Caller-provided identities and evidence references are still not proof of authenticity,
//! chronology, or corpus separation; those properties require an independent verifier.

use super::reasoning_metacognition::{
    evaluate_metacognition, ConfidenceRevisionObservation, CorrectnessPrediction,
    ForecastBaselineMethod, ForecastBaselineV1, MetacognitionEvaluationError, MetacognitionReport,
    WeakAssumptionObservation, FORECAST_BASELINE_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;

pub const DECISION_FORECAST_SCHEMA_VERSION: u32 = 2;
pub const DECISION_OUTCOME_SCHEMA_VERSION: u32 = 1;
pub const FROZEN_DECISION_COHORT_SCHEMA_VERSION: u32 = 2;
pub const DECISION_COHORT_REPORT_SCHEMA_VERSION: u32 = 2;
pub const DECISION_COHORT_EVALUATOR_VERSION: &str = "rq-006-decision-cohort-v2";
const LOG_LOSS_EPSILON: f64 = 1.0e-15;

/// One pre-outcome decision and its correctness probability.
///
/// An asserted answer requires answer_ref. An abstention must not have answer_ref, but may retain
/// a separately identified counterfactual_answer_ref if the subject actually generated a candidate
/// answer before choosing to abstain. Such a reference is not counterfactual evidence by itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionForecastV2 {
    pub schema_version: u32,
    pub forecast_id: String,
    pub episode_id: String,
    pub task_family_id: String,
    pub task_taxonomy_id: String,
    pub outcome_profile_id: String,
    pub subject_id: String,
    pub model_profile_id: String,
    pub input_snapshot_ref: String,
    /// Probability that the referenced asserted/counterfactual answer is correct.
    /// Absent for an abstention with no frozen counterfactual answer.
    pub predicted_probability: Option<f64>,
    pub asserted: bool,
    pub answer_ref: Option<String>,
    pub counterfactual_answer_ref: Option<String>,
}

/// Independent outcome receipt for a decision.
///
/// asserted_answer_correct must be present exactly for an asserted answer. It must be absent for
/// an abstention. Counterfactual correctness is admissible only when the pre-outcome forecast
/// retained a counterfactual answer reference and the outcome carries separate evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionOutcomeV1 {
    pub schema_version: u32,
    pub forecast_id: String,
    pub episode_id: String,
    pub outcome_profile_id: String,
    pub outcome_receipt_id: String,
    pub outcome_evidence_ref: String,
    pub asserted_answer_correct: Option<bool>,
    pub counterfactual_answer_correct: Option<bool>,
    pub counterfactual_evidence_ref: Option<String>,
}

/// One threshold's risk over the full decision cohort, not just the asserted-answer subset.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DecisionSelectiveRiskPoint {
    pub threshold: f64,
    pub total_decisions: usize,
    pub selected_assertions: usize,
    /// Selected asserted answers divided by all decisions in this cohort.
    pub coverage_all_decisions: f64,
    /// Error rate over selected asserted answers; None when no answers are selected.
    pub risk: Option<f64>,
    /// One-sided 95% family-wise upper bound; None for an empty selected set.
    pub risk_upper_bound_95: Option<f64>,
}

/// Calibration of predeclared candidate answers that were not asserted.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DecisionCounterfactualForecastMetricsV2 {
    pub adjudicated_answers: usize,
    pub mean_confidence: Option<f64>,
    pub empirical_accuracy: Option<f64>,
    pub brier_score: Option<f64>,
    pub log_loss: Option<f64>,
}

/// Baseline comparison on the same scoreable asserted-answer cohort as the candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionBaselineScoreV2 {
    pub baseline_id: String,
    pub method: ForecastBaselineMethod,
    pub task_family_id: String,
    pub training_split_id: String,
    pub training_corpus_manifest_ref: String,
    pub training_sample_count: usize,
    pub evaluation_split_id: String,
    pub evaluation_corpus_manifest_ref: String,
    pub evaluation_decisions: usize,
    pub candidate_scored_assertions: usize,
    pub baseline_probability: f64,
    pub empirical_accuracy: Option<f64>,
    pub baseline_brier_score: Option<f64>,
    pub baseline_log_loss: Option<f64>,
    /// Candidate score minus baseline score; negative favors the candidate.
    pub candidate_brier_delta: Option<f64>,
    pub candidate_log_loss_delta: Option<f64>,
    pub candidate_brier_delta_lower_95: Option<f64>,
    pub candidate_brier_delta_upper_95: Option<f64>,
}

/// Family report includes families that have zero asserted answers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionCohortFamilyReportV2 {
    pub task_family_id: String,
    pub total_decisions: usize,
    pub asserted_decisions: usize,
    pub abstained_decisions: usize,
    pub scoreable_assertions: usize,
    pub decision_coverage: f64,
    /// Full calibration/discrimination report scored only on the family’s asserted answers.
    /// A zero-assertion family remains present and has empty correctness metrics.
    pub correctness_metrics: MetacognitionReport,
    /// Threshold coverage always uses total_decisions as its denominator.
    pub selective_risk: Vec<DecisionSelectiveRiskPoint>,
    pub counterfactual_forecast_metrics: DecisionCounterfactualForecastMetricsV2,
    pub baselines: Vec<DecisionBaselineScoreV2>,
}

/// Receipt linkage retained in deterministic forecast order. External verification is still needed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionCohortBindingV1 {
    pub schema_version: u32,
    pub forecast_ids: Vec<String>,
    pub outcome_receipt_ids: Vec<String>,
    pub outcome_evidence_refs: Vec<String>,
    pub counterfactual_evidence_refs: Vec<Option<String>>,
}

/// Decomposed score: all decisions are accounted for, but correctness is calibrated only on answers
/// that were asserted and independently scored. No scalar metacognition score is defined.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionCohortReportV2 {
    pub schema_version: u32,
    pub evaluator_version: String,
    pub outcome_profile_id: String,
    pub task_taxonomy_id: String,
    pub subject_id: String,
    pub model_profile_id: String,
    pub evaluation_split_id: String,
    pub evaluation_corpus_manifest_ref: String,
    pub total_decisions: usize,
    pub asserted_decisions: usize,
    pub abstained_decisions: usize,
    pub scoreable_assertions: usize,
    pub decision_coverage: f64,
    /// Calibration/discrimination metrics over scoreable asserted answers only.
    pub correctness_metrics: MetacognitionReport,
    /// Selective risk and coverage over the full decision cohort.
    pub selective_risk: Vec<DecisionSelectiveRiskPoint>,
    pub counterfactual_forecast_metrics: DecisionCounterfactualForecastMetricsV2,
    pub family_reports: Vec<DecisionCohortFamilyReportV2>,
    pub binding: DecisionCohortBindingV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionCohortError {
    pub message: String,
}

impl DecisionCohortError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for DecisionCohortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DecisionCohortError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "FrozenDecisionCohortWireV2")]
pub struct FrozenDecisionCohortV2 {
    schema_version: u32,
    forecasts: Vec<DecisionForecastV2>,
    baselines: Vec<ForecastBaselineV1>,
    calibration_bins: usize,
    selective_thresholds: Vec<f64>,
    evaluation_split_id: String,
    evaluation_corpus_manifest_ref: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct FrozenDecisionCohortWireV2 {
    schema_version: u32,
    forecasts: Vec<DecisionForecastV2>,
    baselines: Vec<ForecastBaselineV1>,
    calibration_bins: usize,
    selective_thresholds: Vec<f64>,
    evaluation_split_id: String,
    evaluation_corpus_manifest_ref: String,
}

impl TryFrom<FrozenDecisionCohortWireV2> for FrozenDecisionCohortV2 {
    type Error = DecisionCohortError;

    fn try_from(w: FrozenDecisionCohortWireV2) -> Result<Self, Self::Error> {
        if w.schema_version != FROZEN_DECISION_COHORT_SCHEMA_VERSION {
            return Err(DecisionCohortError::new(format!(
                "unsupported decision-cohort schema version {}",
                w.schema_version
            )));
        }
        validate_forecasts(&w.forecasts, w.calibration_bins, &w.selective_thresholds)?;
        require_nonempty("evaluation_split_id", &w.evaluation_split_id)?;
        require_nonempty("evaluation_corpus_manifest_ref", &w.evaluation_corpus_manifest_ref)?;
        validate_frozen_baselines(
            &w.baselines,
            &w.forecasts,
            &w.evaluation_split_id,
            &w.evaluation_corpus_manifest_ref,
        )?;
        if w.forecasts.windows(2).any(|pair| pair[0].forecast_id > pair[1].forecast_id) {
            return Err(DecisionCohortError::new("frozen forecasts are not in canonical order"));
        }
        if w.baselines.windows(2).any(|pair| {
            (&pair[0].task_family_id, pair[0].method)
                > (&pair[1].task_family_id, pair[1].method)
        }) {
            return Err(DecisionCohortError::new("frozen baselines are not in canonical order"));
        }
        Ok(Self {
            schema_version: w.schema_version,
            forecasts: w.forecasts,
            baselines: w.baselines,
            calibration_bins: w.calibration_bins,
            selective_thresholds: w.selective_thresholds,
            evaluation_split_id: w.evaluation_split_id,
            evaluation_corpus_manifest_ref: w.evaluation_corpus_manifest_ref,
        })
    }
}

impl FrozenDecisionCohortV2 {
    pub fn forecasts(&self) -> &[DecisionForecastV2] {
        &self.forecasts
    }

    pub fn baselines(&self) -> &[ForecastBaselineV1] {
        &self.baselines
    }

    pub fn calibration_bins(&self) -> usize {
        self.calibration_bins
    }

    pub fn selective_thresholds(&self) -> &[f64] {
        &self.selective_thresholds
    }

    pub fn evaluation_split_id(&self) -> &str {
        &self.evaluation_split_id
    }

    pub fn evaluation_corpus_manifest_ref(&self) -> &str {
        &self.evaluation_corpus_manifest_ref
    }
}

/// Freeze a mixed decision cohort before any correctness outcomes are exposed.
pub fn freeze_decision_cohort_for_split(
    mut forecasts: Vec<DecisionForecastV2>,
    mut baselines: Vec<ForecastBaselineV1>,
    calibration_bins: usize,
    mut selective_thresholds: Vec<f64>,
    evaluation_split_id: String,
    evaluation_corpus_manifest_ref: String,
) -> Result<FrozenDecisionCohortV2, DecisionCohortError> {
    // Canonicalize threshold ordering and reject duplicates so the policy is deterministic.
    for threshold in &selective_thresholds {
        validate_probability("selective threshold", *threshold)?;
    }
    selective_thresholds.sort_by(f64::total_cmp);
    if selective_thresholds.windows(2).any(|w| w[0] == w[1]) {
        return Err(DecisionCohortError::new("duplicate selective-risk threshold"));
    }
    forecasts.sort_by(|a, b| a.forecast_id.cmp(&b.forecast_id));
    baselines.sort_by(|a, b| {
        (&a.task_family_id, a.method).cmp(&(&b.task_family_id, b.method))
    });
    FrozenDecisionCohortV2::try_from(FrozenDecisionCohortWireV2 {
        schema_version: FROZEN_DECISION_COHORT_SCHEMA_VERSION,
        forecasts,
        baselines,
        calibration_bins,
        selective_thresholds,
        evaluation_split_id,
        evaluation_corpus_manifest_ref,
    })
}

fn require_nonempty(field: &str, value: &str) -> Result<(), DecisionCohortError> {
    if value.trim().is_empty() || value.trim() != value {
        Err(DecisionCohortError::new(format!("required field {field} is empty or not canonical")))
    } else {
        Ok(())
    }
}

fn validate_probability(field: &str, value: f64) -> Result<(), DecisionCohortError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(DecisionCohortError::new(format!(
            "{field} must be finite and within [0, 1], got {value}"
        )))
    }
}

fn validate_optional_ref(field: &str, value: Option<&str>) -> Result<(), DecisionCohortError> {
    if let Some(value) = value {
        require_nonempty(field, value)?;
    }
    Ok(())
}

fn validate_frozen_baselines(
    baselines: &[ForecastBaselineV1],
    forecasts: &[DecisionForecastV2],
    evaluation_split_id: &str,
    evaluation_manifest: &str,
) -> Result<(), DecisionCohortError> {
    let first = forecasts
        .first()
        .ok_or_else(|| DecisionCohortError::new("baseline validation requires forecasts"))?;
    let families: BTreeSet<&str> = forecasts
        .iter()
        .map(|forecast| forecast.task_family_id.as_str())
        .collect();
    let mut baseline_ids = HashSet::new();
    let mut family_methods = BTreeSet::new();

    for baseline in baselines {
        if baseline.schema_version != FORECAST_BASELINE_SCHEMA_VERSION {
            return Err(DecisionCohortError::new(format!(
                "unsupported baseline schema version {}",
                baseline.schema_version
            )));
        }
        for (name, value) in [
            ("baseline_id", baseline.baseline_id.as_str()),
            ("task_family_id", baseline.task_family_id.as_str()),
            ("task_taxonomy_id", baseline.task_taxonomy_id.as_str()),
            ("outcome_profile_id", baseline.outcome_profile_id.as_str()),
            ("training_split_id", baseline.training_split_id.as_str()),
            ("training_corpus_manifest_ref", baseline.training_corpus_manifest_ref.as_str()),
        ] {
            require_nonempty(name, value)?;
        }
        if baseline.training_sample_count == 0 {
            return Err(DecisionCohortError::new(format!(
                "baseline {} has no calibration samples",
                baseline.baseline_id
            )));
        }
        validate_probability("baseline probability", baseline.predicted_probability)?;
        if baseline.training_split_id == evaluation_split_id
            || baseline.training_corpus_manifest_ref == evaluation_manifest
        {
            return Err(DecisionCohortError::new(format!(
                "baseline {} reuses evaluation split or manifest",
                baseline.baseline_id
            )));
        }
        if baseline.task_taxonomy_id != first.task_taxonomy_id
            || baseline.outcome_profile_id != first.outcome_profile_id
        {
            return Err(DecisionCohortError::new(format!(
                "baseline {} has a mismatched taxonomy or outcome profile",
                baseline.baseline_id
            )));
        }
        if !families.contains(baseline.task_family_id.as_str()) {
            return Err(DecisionCohortError::new(format!(
                "baseline {} belongs to an unknown evaluation family",
                baseline.baseline_id
            )));
        }
        if !baseline_ids.insert(baseline.baseline_id.as_str()) {
            return Err(DecisionCohortError::new(format!(
                "duplicate baseline id {}",
                baseline.baseline_id
            )));
        }
        if !family_methods.insert((baseline.task_family_id.as_str(), baseline.method)) {
            return Err(DecisionCohortError::new(format!(
                "duplicate baseline method {:?} for {}",
                baseline.method,
                baseline.task_family_id
            )));
        }
    }

    for family in &families {
        for method in [
            ForecastBaselineMethod::ConstantBaseRate,
            ForecastBaselineMethod::RecentEmpiricalAccuracy,
        ] {
            if !family_methods.contains(&(*family, method)) {
                return Err(DecisionCohortError::new(format!(
                    "missing baseline {:?} for family {}",
                    method,
                    family
                )));
            }
        }
    }
    Ok(())
}

fn validate_forecasts(
    forecasts: &[DecisionForecastV2],
    bins: usize,
    thresholds: &[f64],
) -> Result<(), DecisionCohortError> {
    if forecasts.is_empty() {
        return Err(DecisionCohortError::new("decision cohort must not be empty"));
    }
    if !(1..=100).contains(&bins) {
        return Err(DecisionCohortError::new(format!(
            "calibration bin count must be 1..=100, got {bins}"
        )));
    }
    if thresholds.is_empty() {
        return Err(DecisionCohortError::new("at least one selective-risk threshold is required"));
    }
    let first = &forecasts[0];
    let mut forecast_ids = HashSet::new();
    let mut episode_ids = HashSet::new();
    let mut previous_threshold = None;
    for threshold in thresholds {
        validate_probability("selective threshold", *threshold)?;
        if previous_threshold.is_some_and(|previous| *threshold <= previous) {
            return Err(DecisionCohortError::new(
                "selective thresholds must be strictly increasing",
            ));
        }
        previous_threshold = Some(*threshold);
    }
    for f in forecasts {
        if f.schema_version != DECISION_FORECAST_SCHEMA_VERSION {
            return Err(DecisionCohortError::new(format!(
                "unsupported decision-forecast schema version {}",
                f.schema_version
            )));
        }
        for (name, value) in [
            ("forecast_id", f.forecast_id.as_str()),
            ("episode_id", f.episode_id.as_str()),
            ("task_family_id", f.task_family_id.as_str()),
            ("task_taxonomy_id", f.task_taxonomy_id.as_str()),
            ("outcome_profile_id", f.outcome_profile_id.as_str()),
            ("subject_id", f.subject_id.as_str()),
            ("model_profile_id", f.model_profile_id.as_str()),
            ("input_snapshot_ref", f.input_snapshot_ref.as_str()),
        ] {
            require_nonempty(name, value)?;
        }
        validate_optional_ref("answer_ref", f.answer_ref.as_deref())?;
        validate_optional_ref("counterfactual_answer_ref", f.counterfactual_answer_ref.as_deref())?;
        match (f.asserted || f.counterfactual_answer_ref.is_some(), f.predicted_probability) {
            (true, Some(probability)) => validate_probability("predicted_probability", probability)?,
            (true, None) => return Err(DecisionCohortError::new(format!(
                "forecast {} references an answer but has no correctness probability",
                f.forecast_id
            ))),
            (false, None) => {}
            (false, Some(_)) => return Err(DecisionCohortError::new(format!(
                "abstention {} has a probability but no frozen answer target",
                f.forecast_id
            ))),
        }
        if !forecast_ids.insert(f.forecast_id.as_str()) {
            return Err(DecisionCohortError::new(format!(
                "duplicate forecast id {}",
                f.forecast_id
            )));
        }
        if !episode_ids.insert(f.episode_id.as_str()) {
            return Err(DecisionCohortError::new(format!("duplicate episode id {}", f.episode_id)));
        }
        if f.asserted {
            if f.answer_ref.is_none() || f.counterfactual_answer_ref.is_some() {
                return Err(DecisionCohortError::new(format!(
                    "asserted forecast {} requires answer_ref and forbids counterfactual_answer_ref",
                    f.forecast_id
                )));
            }
        } else if f.answer_ref.is_some() {
            return Err(DecisionCohortError::new(format!(
                "abstained forecast {} must not contain answer_ref",
                f.forecast_id
            )));
        }
        for (field, expected, found) in [
            ("task_taxonomy_id", first.task_taxonomy_id.as_str(), f.task_taxonomy_id.as_str()),
            (
                "outcome_profile_id",
                first.outcome_profile_id.as_str(),
                f.outcome_profile_id.as_str(),
            ),
            ("subject_id", first.subject_id.as_str(), f.subject_id.as_str()),
            ("model_profile_id", first.model_profile_id.as_str(), f.model_profile_id.as_str()),
        ] {
            if expected != found {
                return Err(DecisionCohortError::new(format!(
                    "mixed {field}: expected {expected}, found {found}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_outcome_for_forecast(
    f: &DecisionForecastV2,
    o: &DecisionOutcomeV1,
) -> Result<(), DecisionCohortError> {
    if o.schema_version != DECISION_OUTCOME_SCHEMA_VERSION {
        return Err(DecisionCohortError::new(format!(
            "unsupported decision-outcome schema version {}",
            o.schema_version
        )));
    }
    for (name, value) in [
        ("forecast_id", o.forecast_id.as_str()),
        ("episode_id", o.episode_id.as_str()),
        ("outcome_profile_id", o.outcome_profile_id.as_str()),
        ("outcome_receipt_id", o.outcome_receipt_id.as_str()),
        ("outcome_evidence_ref", o.outcome_evidence_ref.as_str()),
    ] {
        require_nonempty(name, value)?;
    }
    if o.forecast_id != f.forecast_id
        || o.episode_id != f.episode_id
        || o.outcome_profile_id != f.outcome_profile_id
    {
        return Err(DecisionCohortError::new(format!(
            "outcome binding mismatch for forecast {}",
            f.forecast_id
        )));
    }
    validate_optional_ref("counterfactual_evidence_ref", o.counterfactual_evidence_ref.as_deref())?;
    if f.asserted {
        if o.asserted_answer_correct.is_none()
            || o.counterfactual_answer_correct.is_some()
            || o.counterfactual_evidence_ref.is_some()
        {
            return Err(DecisionCohortError::new(format!(
                "asserted forecast {} needs only asserted_answer_correct",
                f.forecast_id
            )));
        }
    } else {
        if o.asserted_answer_correct.is_some() {
            return Err(DecisionCohortError::new(format!(
                "abstention {} cannot carry asserted-answer correctness",
                f.forecast_id
            )));
        }
        match (
            f.counterfactual_answer_ref.as_ref(),
            o.counterfactual_answer_correct,
            o.counterfactual_evidence_ref.as_ref(),
        ) {
            (Some(_), Some(_), Some(_)) => {}
            (Some(_), _, _) => return Err(DecisionCohortError::new(format!(
                "counterfactual answer {} requires correctness and separate evidence",
                f.forecast_id
            ))),
            (None, None, None) => {}
            (None, _, _) => return Err(DecisionCohortError::new(format!(
                "outcome {} contains counterfactual evidence not declared before outcome access",
                f.forecast_id
            ))),
        }
    }
    Ok(())
}

fn risk_point(
    threshold: f64,
    rows: &[(&DecisionForecastV2, &DecisionOutcomeV1)],
    total_decisions: usize,
    familywise_comparison_count: usize,
) -> DecisionSelectiveRiskPoint {
    let selected: Vec<_> = rows
        .iter()
        .filter(|(f, _)| f.asserted && f.predicted_probability.is_some_and(|probability| probability >= threshold))
        .collect();
    let n = selected.len();
    let errors = selected.iter().filter(|(_, o)| o.asserted_answer_correct == Some(false)).count();
    let risk = (n > 0).then(|| errors as f64 / n as f64);
    let bound = if n == 0 || familywise_comparison_count == 0 {
        None
    } else {
        let alpha = 0.05 / familywise_comparison_count as f64;
        let radius = ((1.0 / alpha).ln() / (2.0 * n as f64)).sqrt();
        Some((errors as f64 / n as f64 + radius).min(1.0))
    };
    DecisionSelectiveRiskPoint {
        threshold,
        total_decisions,
        selected_assertions: n,
        coverage_all_decisions: if total_decisions == 0 {
            0.0
        } else {
            n as f64 / total_decisions as f64
        },
        risk,
        risk_upper_bound_95: bound,
    }
}

fn score_predictions(
    rows: &[(&DecisionForecastV2, &DecisionOutcomeV1)],
) -> Vec<CorrectnessPrediction> {
    rows.iter()
        .filter_map(|(f, o)| {
            o.asserted_answer_correct
                .zip(f.predicted_probability)
                .map(|(correct, confidence)| CorrectnessPrediction {
                    episode_id: f.episode_id.clone(),
                    task_family_id: f.task_family_id.clone(),
                    confidence_target_episode_id: f.episode_id.clone(),
                    confidence,
                    correct,
                    asserted: true,
                })
        })
        .collect()
}

fn counterfactual_forecast_metrics(
    rows: &[(&DecisionForecastV2, &DecisionOutcomeV1)],
) -> DecisionCounterfactualForecastMetricsV2 {
    let observed: Vec<(f64, bool)> = rows
        .iter()
        .filter_map(|(forecast, outcome)| {
            if forecast.asserted {
                return None;
            }
            match (
                forecast.counterfactual_answer_ref.as_ref(),
                forecast.predicted_probability,
                outcome.counterfactual_answer_correct,
            ) {
                (Some(_), Some(probability), Some(correct)) => Some((probability, correct)),
                _ => None,
            }
        })
        .collect();
    let n = observed.len();
    if n == 0 {
        return DecisionCounterfactualForecastMetricsV2 {
            adjudicated_answers: 0,
            mean_confidence: None,
            empirical_accuracy: None,
            brier_score: None,
            log_loss: None,
        };
    }
    let correct_count = observed.iter().filter(|entry| entry.1).count();
    let mean_confidence =
        observed.iter().map(|(probability, _)| *probability).sum::<f64>() / n as f64;
    let empirical_accuracy = correct_count as f64 / n as f64;
    let brier_score = observed
        .iter()
        .map(|(probability, correct)| {
            (*probability - if *correct { 1.0 } else { 0.0 }).powi(2)
        })
        .sum::<f64>()
        / n as f64;
    let log_loss = observed
        .iter()
        .map(|(probability, correct)| {
            let target_probability = if *correct { *probability } else { 1.0 - *probability };
            -target_probability
                .clamp(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON)
                .ln()
        })
        .sum::<f64>()
        / n as f64;
    DecisionCounterfactualForecastMetricsV2 {
        adjudicated_answers: n,
        mean_confidence: Some(mean_confidence),
        empirical_accuracy: Some(empirical_accuracy),
        brier_score: Some(brier_score),
        log_loss: Some(log_loss),
    }
}

fn paired_brier_delta_interval(
    delta: Option<f64>,
    n: usize,
    comparison_count: usize,
) -> (Option<f64>, Option<f64>) {
    let Some(delta) = delta.filter(|_| n > 0 && comparison_count > 0) else {
        return (None, None);
    };
    let alpha = 0.05 / comparison_count as f64;
    let radius = (2.0 * (2.0 / alpha).ln() / n as f64).sqrt();
    (Some((delta - radius).max(-1.0)), Some((delta + radius).min(1.0)))
}

fn baseline_score(
    baseline: &ForecastBaselineV1,
    rows: &[(&DecisionForecastV2, &DecisionOutcomeV1)],
    evaluation_split_id: &str,
    evaluation_manifest: &str,
    familywise_comparison_count: usize,
) -> DecisionBaselineScoreV2 {
    let scored: Vec<_> = rows
        .iter()
        .filter(|(f, o)| f.asserted && o.asserted_answer_correct.is_some())
        .collect();
    let n = scored.len();
    let (brier, log_loss, accuracy) = if n == 0 {
        (None, None, None)
    } else {
        let mut brier_sum = 0.0;
        let mut loss_sum = 0.0;
        let mut correct_count = 0usize;
        for (_, outcome) in &scored {
            let correct = outcome.asserted_answer_correct.unwrap_or(false);
            let target = if correct { 1.0 } else { 0.0 };
            brier_sum += (baseline.predicted_probability - target).powi(2);
            let p_target = if correct {
                baseline.predicted_probability
            } else {
                1.0 - baseline.predicted_probability
            };
            loss_sum += -p_target.clamp(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON).ln();
            correct_count += usize::from(correct);
        }
        (
            Some(brier_sum / n as f64),
            Some(loss_sum / n as f64),
            Some(correct_count as f64 / n as f64),
        )
    };
    let predictions = score_predictions(rows);
    let candidate_brier = if predictions.is_empty() {
        None
    } else {
        Some(
            predictions
                .iter()
                .map(|p| (p.confidence - if p.correct { 1.0 } else { 0.0 }).powi(2))
                .sum::<f64>()
                / predictions.len() as f64,
        )
    };
    let candidate_log_loss = if predictions.is_empty() {
        None
    } else {
        Some(predictions.iter().map(|p| {
            let p_target = if p.correct { p.confidence } else { 1.0 - p.confidence };
            -p_target.clamp(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON).ln()
        }).sum::<f64>() / predictions.len() as f64)
    };
    let delta = candidate_brier.zip(brier).map(|(c, b)| c - b);
    let (lower, upper) = paired_brier_delta_interval(delta, n, familywise_comparison_count);
    DecisionBaselineScoreV2 {
        baseline_id: baseline.baseline_id.clone(),
        method: baseline.method,
        task_family_id: baseline.task_family_id.clone(),
        training_split_id: baseline.training_split_id.clone(),
        training_corpus_manifest_ref: baseline.training_corpus_manifest_ref.clone(),
        training_sample_count: baseline.training_sample_count,
        evaluation_split_id: evaluation_split_id.into(),
        evaluation_corpus_manifest_ref: evaluation_manifest.into(),
        evaluation_decisions: rows.len(),
        candidate_scored_assertions: n,
        baseline_probability: baseline.predicted_probability,
        empirical_accuracy: accuracy,
        baseline_brier_score: brier,
        baseline_log_loss: log_loss,
        candidate_brier_delta: delta,
        candidate_log_loss_delta: candidate_log_loss.zip(log_loss).map(|(c, b)| c - b),
        candidate_brier_delta_lower_95: lower,
        candidate_brier_delta_upper_95: upper,
    }
}

/// Score a complete mixed decision cohort. Every decision needs exactly one outcome receipt.
/// Correctness calibration and baseline comparison use asserted answers only, while threshold
/// coverage divides by all decisions. Baselines must be frozen from a separately verified corpus.
pub fn evaluate_decision_cohort(
    frozen: &FrozenDecisionCohortV2,
    outcomes: &[DecisionOutcomeV1],
    assumptions: &[WeakAssumptionObservation],
    revisions: &[ConfidenceRevisionObservation],
) -> Result<DecisionCohortReportV2, DecisionCohortError> {
    validate_forecasts(&frozen.forecasts, frozen.calibration_bins, &frozen.selective_thresholds)?;
    let forecast_by_id: BTreeMap<&str, &DecisionForecastV2> = frozen
        .forecasts
        .iter()
        .map(|f| (f.forecast_id.as_str(), f))
        .collect();
    let mut outcome_by_id: BTreeMap<&str, &DecisionOutcomeV1> = BTreeMap::new();
    let mut receipt_ids = HashSet::new();
    for o in outcomes {
        let f = forecast_by_id
            .get(o.forecast_id.as_str())
            .ok_or_else(|| {
                DecisionCohortError::new(format!(
                    "outcome references unknown forecast {}",
                    o.forecast_id
                ))
            })?;
        validate_outcome_for_forecast(f, o)?;
        if !receipt_ids.insert(o.outcome_receipt_id.as_str()) {
            return Err(DecisionCohortError::new(format!(
                "duplicate outcome receipt {}",
                o.outcome_receipt_id
            )));
        }
        if outcome_by_id.insert(o.forecast_id.as_str(), o).is_some() {
            return Err(DecisionCohortError::new(format!(
                "duplicate outcome for forecast {}",
                o.forecast_id
            )));
        }
    }
    if outcome_by_id.len() != frozen.forecasts.len() {
        let missing = frozen
            .forecasts
            .iter()
            .find(|f| !outcome_by_id.contains_key(f.forecast_id.as_str()));
        return Err(DecisionCohortError::new(format!(
            "outcome set must bind every forecast exactly once; missing {}",
            missing.map(|f| f.forecast_id.as_str()).unwrap_or("unknown forecast")
        )));
    }
    let rows: Vec<(&DecisionForecastV2, &DecisionOutcomeV1)> = frozen.forecasts.iter().map(|f| {
        let o = outcome_by_id
            .get(f.forecast_id.as_str())
            .copied()
            .ok_or_else(|| DecisionCohortError::new(format!("missing outcome {}", f.forecast_id)))?;
        Ok((f, o))
    }).collect::<Result<_, DecisionCohortError>>()?;
    let mut family_rows: BTreeMap<
        String,
        Vec<(&DecisionForecastV2, &DecisionOutcomeV1)>,
    > = BTreeMap::new();
    for (f, o) in &rows {
        family_rows.entry(f.task_family_id.clone()).or_default().push((*f, *o));
    }

    validate_frozen_baselines(
        &frozen.baselines,
        &frozen.forecasts,
        &frozen.evaluation_split_id,
        &frozen.evaluation_corpus_manifest_ref,
    )?;
    let baseline_index: BTreeMap<(String, ForecastBaselineMethod), &ForecastBaselineV1> = frozen
        .baselines
        .iter()
        .map(|baseline| {
            (
                (baseline.task_family_id.clone(), baseline.method),
                baseline,
            )
        })
        .collect();

    let scored = score_predictions(&rows);
    let pooled_metrics = evaluate_metacognition(
        &scored,
        assumptions,
        revisions,
        frozen.calibration_bins,
        &frozen.selective_thresholds,
    )
        .map_err(|e: MetacognitionEvaluationError| DecisionCohortError::new(e.to_string()))?;
    let familywise_risk_comparisons = frozen
        .selective_thresholds
        .len()
        .saturating_mul(family_rows.len().saturating_add(1));
    let familywise_baseline_comparisons = family_rows.len().saturating_mul(2);
    let pooled_risk: Vec<DecisionSelectiveRiskPoint> = frozen
        .selective_thresholds
        .iter()
        .map(|t| risk_point(*t, &rows, rows.len(), familywise_risk_comparisons))
        .collect();

    let mut family_reports = Vec::with_capacity(family_rows.len());
    for (family, frows) in family_rows {
        let family_predictions = score_predictions(&frows);
        let family_episode_ids: HashSet<&str> = frows
            .iter()
            .map(|(f, _)| f.episode_id.as_str())
            .collect();
        let family_assumptions: Vec<_> = assumptions
            .iter()
            .filter(|a| family_episode_ids.contains(a.episode_id.as_str()))
            .cloned()
            .collect();
        let family_revisions: Vec<_> = revisions
            .iter()
            .filter(|r| {
                family_episode_ids.contains(r.before_episode_id.as_str())
                    && family_episode_ids.contains(r.after_episode_id.as_str())
            })
            .cloned()
            .collect();
        let metrics = evaluate_metacognition(
            &family_predictions,
            &family_assumptions,
            &family_revisions,
            frozen.calibration_bins,
            &frozen.selective_thresholds,
        )
            .map_err(|e| DecisionCohortError::new(e.to_string()))?;
        let asserted = frows.iter().filter(|(f, _)| f.asserted).count();
        let abstained = frows.len() - asserted;
        let counterfactual_metrics = counterfactual_forecast_metrics(&frows);
        let selective_risk = frozen
            .selective_thresholds
            .iter()
            .map(|t| risk_point(*t, &frows, frows.len(), familywise_risk_comparisons))
            .collect();
        let mut family_baselines = Vec::new();
        for method in [
            ForecastBaselineMethod::ConstantBaseRate,
            ForecastBaselineMethod::RecentEmpiricalAccuracy,
        ] {
            let baseline = baseline_index
                .get(&(family.clone(), method))
                .ok_or_else(|| DecisionCohortError::new(format!("missing baseline for {family}")))?;
            family_baselines.push(baseline_score(
                baseline,
                &frows,
                &frozen.evaluation_split_id,
                &frozen.evaluation_corpus_manifest_ref,
                familywise_baseline_comparisons,
            ));
        }
        family_reports.push(DecisionCohortFamilyReportV2 {
            task_family_id: family,
            total_decisions: frows.len(),
            asserted_decisions: asserted,
            abstained_decisions: abstained,
            scoreable_assertions: family_predictions.len(),
            decision_coverage: asserted as f64 / frows.len() as f64,
            correctness_metrics: metrics,
            selective_risk,
            counterfactual_forecast_metrics: counterfactual_metrics,
            baselines: family_baselines,
        });
    }
    let total = rows.len();
    let asserted = rows.iter().filter(|(f, _)| f.asserted).count();
    let counterfactual_metrics = counterfactual_forecast_metrics(&rows);
    Ok(DecisionCohortReportV2 {
        schema_version: DECISION_COHORT_REPORT_SCHEMA_VERSION,
        evaluator_version: DECISION_COHORT_EVALUATOR_VERSION.into(),
        outcome_profile_id: first.outcome_profile_id.clone(),
        task_taxonomy_id: first.task_taxonomy_id.clone(),
        subject_id: first.subject_id.clone(),
        model_profile_id: first.model_profile_id.clone(),
        evaluation_split_id: frozen.evaluation_split_id.clone(),
        evaluation_corpus_manifest_ref: frozen.evaluation_corpus_manifest_ref.clone(),
        total_decisions: total,
        asserted_decisions: asserted,
        abstained_decisions: total - asserted,
        scoreable_assertions: scored.len(),
        decision_coverage: asserted as f64 / total as f64,
        correctness_metrics: pooled_metrics,
        selective_risk: pooled_risk,
        counterfactual_forecast_metrics: counterfactual_metrics,
        family_reports,
        binding: DecisionCohortBindingV1 {
            schema_version: 1,
            forecast_ids: rows.iter().map(|(f, _)| f.forecast_id.clone()).collect(),
            outcome_receipt_ids: rows.iter().map(|(_, o)| o.outcome_receipt_id.clone()).collect(),
            outcome_evidence_refs: rows
                .iter()
                .map(|(_, o)| o.outcome_evidence_ref.clone())
                .collect(),
            counterfactual_evidence_refs: rows
                .iter()
                .map(|(_, o)| o.counterfactual_evidence_ref.clone())
                .collect(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn forecast(
        id: &str,
        episode: &str,
        family: &str,
        probability: f64,
        asserted: bool,
        cf: Option<&str>,
    ) -> DecisionForecastV2 {
        DecisionForecastV2 {
            schema_version: DECISION_FORECAST_SCHEMA_VERSION,
            forecast_id: id.into(),
            episode_id: episode.into(),
            task_family_id: family.into(),
            task_taxonomy_id: "taxonomy-v1".into(),
            outcome_profile_id: "answer-correct-v1".into(),
            subject_id: "subject-sha:test".into(),
            model_profile_id: "profile-sha:test".into(),
            input_snapshot_ref: format!("input:{episode}"),
            predicted_probability: (asserted || cf.is_some()).then_some(probability),
            asserted,
            answer_ref: asserted.then(|| format!("answer:{episode}")),
            counterfactual_answer_ref: cf.map(str::to_owned),
        }
    }

    fn outcome(
        f: &DecisionForecastV2,
        correct: Option<bool>,
        cf_correct: Option<bool>,
    ) -> DecisionOutcomeV1 {
        DecisionOutcomeV1 {
            schema_version: DECISION_OUTCOME_SCHEMA_VERSION,
            forecast_id: f.forecast_id.clone(),
            episode_id: f.episode_id.clone(),
            outcome_profile_id: f.outcome_profile_id.clone(),
            outcome_receipt_id: format!("receipt:{}", f.forecast_id),
            outcome_evidence_ref: format!("outcome-evidence:{}", f.episode_id),
            asserted_answer_correct: correct,
            counterfactual_answer_correct: cf_correct,
            counterfactual_evidence_ref: cf_correct
                .map(|_| format!("counterfactual-evidence:{}", f.episode_id)),
        }
    }

    fn baseline(
        id: &str,
        method: ForecastBaselineMethod,
        family: &str,
        p: f64,
    ) -> ForecastBaselineV1 {
        ForecastBaselineV1 {
            schema_version: FORECAST_BASELINE_SCHEMA_VERSION,
            baseline_id: id.into(),
            method,
            task_family_id: family.into(),
            task_taxonomy_id: "taxonomy-v1".into(),
            outcome_profile_id: "answer-correct-v1".into(),
            predicted_probability: p,
            training_sample_count: 100,
            training_split_id: "calibration-v1".into(),
            training_corpus_manifest_ref: format!("calibration-manifest:{family}"),
        }
    }

    fn family_baselines(family: &str) -> Vec<ForecastBaselineV1> {
        vec![
            baseline(
                &format!("{family}-constant"),
                ForecastBaselineMethod::ConstantBaseRate,
                family,
                0.6,
            ),
            baseline(
                &format!("{family}-recent"),
                ForecastBaselineMethod::RecentEmpiricalAccuracy,
                family,
                0.5,
            ),
        ]
    }

    #[test]
    fn mixed_decision_cohort_separates_calibration_coverage_and_counterfactual_cost() {
        let fs = vec![
            forecast("f1", "e1", "reasoning", 0.8, true, None),
            forecast("f2", "e2", "reasoning", 0.9, false, Some("hidden-answer:e2")),
            forecast("f3", "e3", "retrieval", 0.7, false, None),
        ];
        let baselines = vec![
            baseline("r-base", ForecastBaselineMethod::ConstantBaseRate, "reasoning", 0.6),
            baseline("r-recent", ForecastBaselineMethod::RecentEmpiricalAccuracy, "reasoning", 0.5),
            baseline("q-base", ForecastBaselineMethod::ConstantBaseRate, "retrieval", 0.4),
            baseline("q-recent", ForecastBaselineMethod::RecentEmpiricalAccuracy, "retrieval", 0.3),
        ];
        let frozen = freeze_decision_cohort_for_split(
            fs.clone(),
            baselines,
            5,
            vec![0.9, 0.5],
            "holdout-v1".into(),
            "holdout-manifest-v1".into(),
        )
        .unwrap_or_else(|e| panic!("freeze: {e}"));
        let outcomes = vec![
            outcome(&fs[0], Some(true), None),
            outcome(&fs[1], None, Some(true)),
            outcome(&fs[2], None, None),
        ];
        let report = evaluate_decision_cohort(&frozen, &outcomes, &[], &[])
            .unwrap_or_else(|e| panic!("evaluate: {e}"));
        assert_eq!(report.total_decisions, 3);
        assert_eq!(report.asserted_decisions, 1);
        assert_eq!(report.abstained_decisions, 2);
        assert_eq!(report.decision_coverage, 1.0 / 3.0);
        assert_eq!(report.correctness_metrics.predictions, 1);
        assert!(
            (report.correctness_metrics.brier_score.unwrap_or(f64::NAN) - 0.04).abs()
                < 1.0e-12
        );
        assert_eq!(report.selective_risk[0].threshold, 0.5);
        assert_eq!(report.selective_risk[0].coverage_all_decisions, 1.0 / 3.0);
        assert_eq!(report.counterfactual_forecast_metrics.adjudicated_answers, 1);
        assert_eq!(report.counterfactual_forecast_metrics.empirical_accuracy, Some(1.0));
        assert!(
            (report.counterfactual_forecast_metrics.brier_score.unwrap_or(f64::NAN) - 0.01)
                .abs()
                < 1.0e-12
        );
        assert_eq!(report.family_reports.len(), 2);
        let reasoning = report
            .family_reports
            .iter()
            .find(|r| r.task_family_id == "reasoning")
            .expect("reasoning family retained");
        assert_eq!(reasoning.total_decisions, 2);
        assert_eq!(reasoning.scoreable_assertions, 1);
        assert_eq!(reasoning.baselines[0].candidate_scored_assertions, 1);
        assert!(
            (reasoning.baselines[0].baseline_brier_score.unwrap_or(f64::NAN) - 0.16).abs()
                < 1.0e-12
        );
        assert_eq!(reasoning.baselines[0].evaluation_decisions, 2);
        let retrieval = report
            .family_reports
            .iter()
            .find(|r| r.task_family_id == "retrieval")
            .expect("abstain-only family retained");
        assert_eq!(retrieval.total_decisions, 1);
        assert_eq!(retrieval.scoreable_assertions, 0);
        assert_eq!(retrieval.correctness_metrics.brier_score, None);
        assert_eq!(retrieval.baselines.len(), 2);
        assert_eq!(retrieval.baselines[0].candidate_scored_assertions, 0);
        assert_eq!(retrieval.baselines[0].baseline_brier_score, None);
        assert_eq!(retrieval.counterfactual_forecast_metrics.adjudicated_answers, 0);
        assert_eq!(retrieval.counterfactual_forecast_metrics.brier_score, None);
    }

    #[test]
    fn cohort_freeze_requires_both_baseline_methods_per_family() {
        let forecast = forecast("f1", "e1", "reasoning", 0.5, true, None);
        let only_one_baseline = vec![baseline(
            "reasoning-constant",
            ForecastBaselineMethod::ConstantBaseRate,
            "reasoning",
            0.6,
        )];
        assert!(
            freeze_decision_cohort_for_split(
                vec![forecast],
                only_one_baseline,
                5,
                vec![0.5],
                "holdout".into(),
                "manifest".into(),
            )
            .is_err()
        );
    }

    #[test]
    fn decision_outcomes_must_bind_exactly_once() {
        let forecast = forecast("f1", "e1", "reasoning", 0.5, true, None);
        let frozen = freeze_decision_cohort_for_split(
            vec![forecast.clone()],
            family_baselines("reasoning"),
            5,
            vec![0.5],
            "holdout".into(),
            "manifest".into(),
        )
        .unwrap_or_else(|e| panic!("freeze: {e}"));
        let receipt = outcome(&forecast, Some(true), None);
        assert!(evaluate_decision_cohort(
            &frozen,
            &[receipt.clone(), receipt],
            &[],
            &[],
        ).is_err());
        assert!(evaluate_decision_cohort(&frozen, &[], &[], &[]).is_err());
    }

    #[test]
    fn abstention_cannot_be_given_asserted_answer_correctness() {
        let f = forecast("f1", "e1", "reasoning", 0.5, false, None);
        let mut o = outcome(&f, None, None);
        o.asserted_answer_correct = Some(true);
        let frozen = freeze_decision_cohort_for_split(
            vec![f],
            family_baselines("reasoning"),
            5,
            vec![0.5],
            "holdout".into(),
            "manifest".into(),
        )
            .unwrap_or_else(|e| panic!("freeze: {e}"));
        assert!(evaluate_decision_cohort(&frozen, &[o], &[], &[]).is_err());
    }

    #[test]
    fn asserted_answer_requires_a_scored_outcome() {
        let f = forecast("f1", "e1", "reasoning", 0.5, true, None);
        let o = outcome(&f, None, None);
        let frozen = freeze_decision_cohort_for_split(
            vec![f],
            family_baselines("reasoning"),
            5,
            vec![0.5],
            "holdout".into(),
            "manifest".into(),
        )
            .unwrap_or_else(|e| panic!("freeze: {e}"));
        assert!(evaluate_decision_cohort(&frozen, &[o], &[], &[]).is_err());
    }

    #[test]
    fn counterfactual_correctness_requires_a_predeclared_answer_and_separate_evidence() {
        let f = forecast("f1", "e1", "reasoning", 0.5, false, Some("hidden-answer:e1"));
        let mut o = outcome(&f, None, None);
        o.counterfactual_answer_correct = Some(true);
        // Deliberately omit counterfactual_evidence_ref.
        let frozen = freeze_decision_cohort_for_split(
            vec![f],
            family_baselines("reasoning"),
            5,
            vec![0.5],
            "holdout".into(),
            "manifest".into(),
        )
            .unwrap_or_else(|e| panic!("freeze: {e}"));
        assert!(evaluate_decision_cohort(&frozen, &[o], &[], &[]).is_err());
    }

    #[test]
    fn abstention_without_candidate_answer_must_not_claim_correctness_probability() {
        let mut f = forecast("f1", "e1", "reasoning", 0.5, false, None);
        f.predicted_probability = Some(0.5);
        assert!(freeze_decision_cohort_for_split(
            vec![f],
            family_baselines("reasoning"),
            5,
            vec![0.5],
            "holdout".into(),
            "manifest".into(),
        )
        .is_err());
    }

    #[test]
    fn frozen_cohort_deserialization_rejects_tampered_policy() {
        let f = forecast("f1", "e1", "reasoning", 0.5, true, None);
        let frozen = freeze_decision_cohort_for_split(
            vec![f],
            family_baselines("reasoning"),
            5,
            vec![0.5],
            "holdout".into(),
            "manifest".into(),
        )
            .unwrap_or_else(|e| panic!("freeze: {e}"));
        let encoded = serde_json::to_string(&frozen).unwrap_or_else(|e| panic!("serialize: {e}"));
        let restored: FrozenDecisionCohortV2 =
            serde_json::from_str(&encoded).unwrap_or_else(|e| panic!("restore: {e}"));
        assert_eq!(restored, frozen);
        assert_eq!(restored.baselines(), frozen.baselines());

        let mut old_schema: serde_json::Value =
            serde_json::from_str(&encoded).unwrap_or_else(|e| panic!("parse: {e}"));
        old_schema["schema_version"] = serde_json::json!(1);
        assert!(serde_json::from_value::<FrozenDecisionCohortV2>(old_schema).is_err());

        let mut baseline_swap: serde_json::Value =
            serde_json::from_str(&encoded).unwrap_or_else(|e| panic!("parse: {e}"));
        baseline_swap["baselines"][0]["training_split_id"] = serde_json::json!("holdout");
        assert!(
            serde_json::from_value::<FrozenDecisionCohortV2>(baseline_swap).is_err(),
            "baseline profiles may not be rebound to the evaluation split"
        );

        let mut tampered: serde_json::Value =
            serde_json::from_str(&encoded).unwrap_or_else(|e| panic!("parse: {e}"));
        tampered["calibration_bins"] = serde_json::json!(0);
        assert!(serde_json::from_value::<FrozenDecisionCohortV2>(tampered).is_err());
    }
}
