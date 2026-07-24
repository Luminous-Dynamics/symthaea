// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen prequential evaluation for adaptive musical-outcome calibration.
//!
//! Each record captures a prediction made *before* its observed outcome was
//! admitted to the model. The evaluator compares the calibrated prediction
//! with the unchanged hand-authored prior, rejects evidence-count leakage, and
//! applies practical error thresholds without claiming statistical significance.

use crate::adaptive_prediction::{PredictionCalibrationEvidence, PredictionContext};
use crate::cognitive_bridge::{
    ObservedMusicalOutcome, PredictedMusicalOutcome, default_predicted_outcome,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Minimum independent holdout observations required by the frozen V5 gate.
pub const MIN_ADAPTIVE_HOLDOUT_RECORDS: usize = 16;
/// Minimum reduction in mean absolute prediction error.
pub const MIN_ADAPTIVE_MAE_IMPROVEMENT: f32 = 0.02;
/// No individual outcome channel may regress by more than this amount.
pub const MAX_CHANNEL_MAE_REGRESSION: f32 = 0.01;

/// One prequential prediction made before learning from its own observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveHoldoutRecord {
    pub trial_id: String,
    pub frozen_input_sha256: String,
    pub evaluation_order: u64,
    /// Total completed training observations visible when this prediction was made.
    pub training_observations_before: u64,
    pub context: PredictionContext,
    pub calibration: PredictionCalibrationEvidence,
    pub observed: ObservedMusicalOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptiveExperimentIssue {
    NoRecords,
    TooFewRecords { found: usize, required: usize },
    EmptyTrialId { evaluation_order: u64 },
    DuplicateTrialId { trial_id: String },
    DuplicateEvaluationOrder { evaluation_order: u64 },
    InvalidFrozenDigest { trial_id: String },
    EmptyModelVersion { trial_id: String },
    ContextMismatch { trial_id: String },
    HandAuthoredPriorMismatch { trial_id: String },
    EvidenceCountExceedsTrainingHistory { trial_id: String },
    NonFiniteMeasurement { trial_id: String, field: String },
}

/// Error summary retained separately for the original prior and calibration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct ChannelMae {
    pub tension: f32,
    pub density: f32,
    pub familiarity: f32,
    pub tonal_displacement: f32,
    pub overall: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveExperimentSummary {
    pub records: usize,
    pub distinct_contexts: usize,
    pub prior_mae: ChannelMae,
    pub calibrated_mae: ChannelMae,
    pub overall_improvement: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveExperimentConclusion {
    pub success: bool,
    pub sample_gate_passed: bool,
    pub leakage_gate_passed: bool,
    pub practical_improvement_gate_passed: bool,
    pub no_material_channel_regression: bool,
    pub issues: Vec<AdaptiveExperimentIssue>,
    pub summary: Option<AdaptiveExperimentSummary>,
    pub rationale: Vec<String>,
}

pub fn validate_adaptive_holdout(
    records: &[AdaptiveHoldoutRecord],
) -> Vec<AdaptiveExperimentIssue> {
    let mut issues = Vec::new();
    if records.is_empty() {
        issues.push(AdaptiveExperimentIssue::NoRecords);
        return issues;
    }
    if records.len() < MIN_ADAPTIVE_HOLDOUT_RECORDS {
        issues.push(AdaptiveExperimentIssue::TooFewRecords {
            found: records.len(),
            required: MIN_ADAPTIVE_HOLDOUT_RECORDS,
        });
    }

    let mut ids = BTreeSet::new();
    let mut orders = BTreeSet::new();
    for record in records {
        if record.trial_id.trim().is_empty() {
            issues.push(AdaptiveExperimentIssue::EmptyTrialId {
                evaluation_order: record.evaluation_order,
            });
        } else if !ids.insert(record.trial_id.clone()) {
            issues.push(AdaptiveExperimentIssue::DuplicateTrialId {
                trial_id: record.trial_id.clone(),
            });
        }
        if !orders.insert(record.evaluation_order) {
            issues.push(AdaptiveExperimentIssue::DuplicateEvaluationOrder {
                evaluation_order: record.evaluation_order,
            });
        }
        if !is_sha256(&record.frozen_input_sha256) {
            issues.push(AdaptiveExperimentIssue::InvalidFrozenDigest {
                trial_id: record.trial_id.clone(),
            });
        }
        if record.calibration.model_version.trim().is_empty() {
            issues.push(AdaptiveExperimentIssue::EmptyModelVersion {
                trial_id: record.trial_id.clone(),
            });
        }
        if record.calibration.context != record.context {
            issues.push(AdaptiveExperimentIssue::ContextMismatch {
                trial_id: record.trial_id.clone(),
            });
        }
        if record.calibration.prior != default_predicted_outcome(record.context.action) {
            issues.push(AdaptiveExperimentIssue::HandAuthoredPriorMismatch {
                trial_id: record.trial_id.clone(),
            });
        }
        if record.calibration.exact_context_samples > record.training_observations_before
            || record.calibration.action_fallback_samples > record.training_observations_before
        {
            issues.push(
                AdaptiveExperimentIssue::EvidenceCountExceedsTrainingHistory {
                    trial_id: record.trial_id.clone(),
                },
            );
        }
        for (field, value) in prediction_values(record.calibration.prior)
            .into_iter()
            .chain(prediction_values(record.calibration.calibrated))
            .chain(observed_values(record.observed))
        {
            if !value.is_finite() {
                issues.push(AdaptiveExperimentIssue::NonFiniteMeasurement {
                    trial_id: record.trial_id.clone(),
                    field: field.into(),
                });
            }
        }
    }
    issues
}

pub fn summarize_adaptive_holdout(
    records: &[AdaptiveHoldoutRecord],
) -> Option<AdaptiveExperimentSummary> {
    if records.is_empty() {
        return None;
    }
    let mut prior = ChannelMae::default();
    let mut calibrated = ChannelMae::default();
    let contexts: BTreeSet<String> = records
        .iter()
        .map(|record| context_identity(&record.context))
        .collect();
    for record in records {
        add_error(&mut prior, record.calibration.prior, record.observed);
        add_error(
            &mut calibrated,
            record.calibration.calibrated,
            record.observed,
        );
    }
    divide(&mut prior, records.len() as f32);
    divide(&mut calibrated, records.len() as f32);
    Some(AdaptiveExperimentSummary {
        records: records.len(),
        distinct_contexts: contexts.len(),
        prior_mae: prior,
        calibrated_mae: calibrated,
        overall_improvement: prior.overall - calibrated.overall,
    })
}

/// Apply the frozen V5 descriptive gate.
///
/// Passing means only that calibrated predictions reduced held-out symbolic
/// error by a practical margin without materially worsening any channel.
/// Listener benefit, compositional quality, and inferential significance remain
/// separate questions.
pub fn conclude_adaptive_holdout(
    records: &[AdaptiveHoldoutRecord],
) -> AdaptiveExperimentConclusion {
    let issues = validate_adaptive_holdout(records);
    let sample_gate_passed = records.len() >= MIN_ADAPTIVE_HOLDOUT_RECORDS;
    let leakage_gate_passed = !issues.iter().any(|issue| {
        matches!(
            issue,
            AdaptiveExperimentIssue::DuplicateTrialId { .. }
                | AdaptiveExperimentIssue::DuplicateEvaluationOrder { .. }
                | AdaptiveExperimentIssue::ContextMismatch { .. }
                | AdaptiveExperimentIssue::HandAuthoredPriorMismatch { .. }
                | AdaptiveExperimentIssue::EvidenceCountExceedsTrainingHistory { .. }
                | AdaptiveExperimentIssue::NonFiniteMeasurement { .. }
        )
    });
    let summary = summarize_adaptive_holdout(records);
    let practical_improvement_gate_passed = summary
        .as_ref()
        .is_some_and(|summary| summary.overall_improvement >= MIN_ADAPTIVE_MAE_IMPROVEMENT);
    let no_material_channel_regression = summary.as_ref().is_some_and(|summary| {
        channel_regression(summary.prior_mae.tension, summary.calibrated_mae.tension)
            <= MAX_CHANNEL_MAE_REGRESSION
            && channel_regression(summary.prior_mae.density, summary.calibrated_mae.density)
                <= MAX_CHANNEL_MAE_REGRESSION
            && channel_regression(
                summary.prior_mae.familiarity,
                summary.calibrated_mae.familiarity,
            ) <= MAX_CHANNEL_MAE_REGRESSION
            && channel_regression(
                summary.prior_mae.tonal_displacement,
                summary.calibrated_mae.tonal_displacement,
            ) <= MAX_CHANNEL_MAE_REGRESSION
    });
    let success = sample_gate_passed
        && leakage_gate_passed
        && practical_improvement_gate_passed
        && no_material_channel_regression
        && issues.is_empty();

    let mut rationale = vec![format!(
        "sample gate {} ({} frozen prequential observations required)",
        pass_fail(sample_gate_passed),
        MIN_ADAPTIVE_HOLDOUT_RECORDS
    )];
    rationale.push(format!(
        "prequential leakage gate {}",
        pass_fail(leakage_gate_passed)
    ));
    if let Some(summary) = &summary {
        rationale.push(format!(
            "held-out mean absolute error changed from {:.4} to {:.4} ({:+.4})",
            summary.prior_mae.overall,
            summary.calibrated_mae.overall,
            summary.calibrated_mae.overall - summary.prior_mae.overall
        ));
    }
    rationale.push(format!(
        "practical improvement gate {} and per-channel non-regression gate {}",
        pass_fail(practical_improvement_gate_passed),
        pass_fail(no_material_channel_regression)
    ));
    rationale.push(
        "a pass is calibration evidence only; it is not evidence of better music or listener preference"
            .into(),
    );

    AdaptiveExperimentConclusion {
        success,
        sample_gate_passed,
        leakage_gate_passed,
        practical_improvement_gate_passed,
        no_material_channel_regression,
        issues,
        summary,
        rationale,
    }
}

fn add_error(
    total: &mut ChannelMae,
    predicted: PredictedMusicalOutcome,
    observed: ObservedMusicalOutcome,
) {
    let error = predicted.error(observed);
    total.tension += error.tension_error.abs();
    total.density += error.density_error.abs();
    total.familiarity += error.familiarity_error.abs();
    total.tonal_displacement += error.tonal_displacement_error.abs();
    total.overall += error.mean_absolute_error;
}

fn divide(value: &mut ChannelMae, denominator: f32) {
    value.tension /= denominator;
    value.density /= denominator;
    value.familiarity /= denominator;
    value.tonal_displacement /= denominator;
    value.overall /= denominator;
}

fn channel_regression(prior: f32, calibrated: f32) -> f32 {
    calibrated - prior
}

fn prediction_values(value: PredictedMusicalOutcome) -> [(&'static str, f32); 4] {
    [
        ("prediction.tension", value.tension_delta),
        ("prediction.density", value.density_delta),
        ("prediction.familiarity", value.familiarity_delta),
        (
            "prediction.tonal_displacement",
            value.tonal_displacement_delta,
        ),
    ]
}

fn observed_values(value: ObservedMusicalOutcome) -> [(&'static str, f32); 4] {
    [
        ("observed.tension", value.tension_delta),
        ("observed.density", value.density_delta),
        ("observed.familiarity", value.familiarity_delta),
        (
            "observed.tonal_displacement",
            value.tonal_displacement_delta,
        ),
    ]
}

fn context_identity(context: &PredictionContext) -> String {
    format!(
        "{:?}|{:?}|{}|{}|{}|{:?}",
        context.action,
        context.section,
        context.style_name,
        context.form_name,
        context.meter,
        context.texture_band
    )
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn pass_fail(value: bool) -> &'static str {
    if value { "passed" } else { "failed" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_prediction::{AdaptiveOutcomeModel, PredictionEvidenceSource, TextureBand};
    use crate::cognitive_bridge::{CognitiveSection, SymbolicAction};

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn record(index: u64, calibrated_density: f32) -> AdaptiveHoldoutRecord {
        let context = PredictionContext::new(
            SymbolicAction::IncreaseDensity,
            CognitiveSection::Development,
            "Sonata",
            "Sonata",
            4,
            TextureBand::Chamber,
        );
        let model = AdaptiveOutcomeModel::default();
        let mut calibration = model.predict(&context);
        calibration.source = PredictionEvidenceSource::ActionFallback;
        calibration.action_fallback_samples = index;
        calibration.calibrated.density_delta = calibrated_density;
        AdaptiveHoldoutRecord {
            trial_id: format!("trial-{index}"),
            frozen_input_sha256: DIGEST.into(),
            evaluation_order: index,
            training_observations_before: index,
            context,
            calibration,
            observed: ObservedMusicalOutcome {
                tension_delta: 0.15,
                density_delta: 0.15,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
        }
    }

    #[test]
    fn calibrated_holdout_can_clear_the_practical_gate() {
        let records: Vec<_> = (0..MIN_ADAPTIVE_HOLDOUT_RECORDS as u64)
            .map(|index| record(index, 0.15))
            .collect();
        let conclusion = conclude_adaptive_holdout(&records);
        assert!(conclusion.sample_gate_passed);
        assert!(conclusion.leakage_gate_passed);
        assert!(conclusion.practical_improvement_gate_passed);
        assert!(conclusion.no_material_channel_regression);
        assert!(conclusion.success);
    }

    #[test]
    fn holdout_cannot_replace_the_frozen_hand_authored_prior() {
        let mut records: Vec<_> = (0..MIN_ADAPTIVE_HOLDOUT_RECORDS as u64)
            .map(|index| record(index, 0.15))
            .collect();
        records[0].calibration.prior.density_delta = 0.15;
        let conclusion = conclude_adaptive_holdout(&records);
        assert!(!conclusion.leakage_gate_passed);
        assert!(!conclusion.success);
        assert!(conclusion.issues.iter().any(|issue| matches!(
            issue,
            AdaptiveExperimentIssue::HandAuthoredPriorMismatch { .. }
        )));
    }

    #[test]
    fn evidence_cannot_claim_more_samples_than_existed() {
        let mut records: Vec<_> = (0..MIN_ADAPTIVE_HOLDOUT_RECORDS as u64)
            .map(|index| record(index, 0.15))
            .collect();
        records[0].calibration.action_fallback_samples = 1;
        let conclusion = conclude_adaptive_holdout(&records);
        assert!(!conclusion.leakage_gate_passed);
        assert!(!conclusion.success);
        assert!(conclusion.issues.iter().any(|issue| matches!(
            issue,
            AdaptiveExperimentIssue::EvidenceCountExceedsTrainingHistory { .. }
        )));
    }

    #[test]
    fn tiny_or_channel_regressive_changes_do_not_pass() {
        let mut records: Vec<_> = (0..MIN_ADAPTIVE_HOLDOUT_RECORDS as u64)
            .map(|index| record(index, 0.34))
            .collect();
        for record in &mut records {
            record.calibration.calibrated.tension_delta = -0.5;
        }
        let conclusion = conclude_adaptive_holdout(&records);
        assert!(
            !conclusion.practical_improvement_gate_passed
                || !conclusion.no_material_channel_regression
        );
        assert!(!conclusion.success);
    }
}
