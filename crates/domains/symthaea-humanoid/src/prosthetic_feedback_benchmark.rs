// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Per-participant/session evaluation for prosthetic sensory cue recognition.
//!
//! This benchmark layer evaluates whether a participant correctly recognizes
//! an expected semantic event under one fixed modality, codebook and session.
//! It is intentionally separate from hand mechanics, task success, clinical
//! outcomes and the physical haptic renderer.
//!
//! Never pool different participants, sessions, modalities or mappings through
//! this evaluator. Call it once per stratum and preserve the returned identity.
//! Synthetic/simulated evidence is retained as such and cannot establish human
//! sensory performance.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::prosthetic_sensory::{HapticCueSemantic, SensoryEvidenceClass};

/// Feedback condition used during one recognition trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackModalityCondition {
    NoFeedback,
    VisualOnly,
    Auditory,
    Vibrotactile,
    Multimodal,
}

/// One forced-choice or signal-detection recognition observation.
///
/// `expected` is the reference event. `reported` is the participant's
/// response; None means no cue was reported. Response latency is optional
/// because some protocols record accuracy without reaction time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackRecognitionTrialV1 {
    pub trial_id: String,
    /// Pseudonymous study identifier; never put direct personal identifiers here.
    pub participant_ref: String,
    pub session_ref: String,
    pub condition: FeedbackModalityCondition,
    pub mapping_id: String,
    pub evidence_class: SensoryEvidenceClass,
    pub expected: Option<HapticCueSemantic>,
    pub reported: Option<HapticCueSemantic>,
    pub response_latency_ms: Option<f64>,
}

/// Binary one-vs-rest measures for one semantic class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CueClassMetrics {
    pub semantic: HapticCueSemantic,
    pub true_positive: u64,
    pub false_positive: u64,
    pub false_negative: u64,
    pub true_negative: u64,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub specificity: Option<f64>,
    pub f1: Option<f64>,
}

/// Metrics for exactly one participant × session × modality × mapping stratum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackRecognitionSummaryV1 {
    pub participant_ref: String,
    pub session_ref: String,
    pub condition: FeedbackModalityCondition,
    pub mapping_id: String,
    pub evidence_class: SensoryEvidenceClass,
    pub trial_count: u64,
    pub exact_matches: u64,
    pub exact_match_rate: Option<f64>,
    pub correct_cue_responses: u64,
    pub missed_expected_cues: u64,
    pub false_alarm_trials: u64,
    pub cue_substitutions: u64,
    pub median_correct_response_latency_ms: Option<f64>,
    pub macro_f1: Option<f64>,
    pub per_semantic_class: Vec<CueClassMetrics>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeedbackBenchmarkError {
    EmptyTrials,
    MissingTrialIdentity,
    DuplicateTrialId { trial_id: String },
    MixedParticipantSessionModalityOrMapping,
    InvalidResponseLatency { trial_id: String },
    LatencyWithoutResponse { trial_id: String },
}

/// Evaluate one immutable stratum. Callers must group data by participant,
/// session, condition and mapping before invoking this function.
pub fn evaluate_feedback_recognition(
    trials: &[FeedbackRecognitionTrialV1],
) -> Result<FeedbackRecognitionSummaryV1, FeedbackBenchmarkError> {
    let first = trials.first().ok_or(FeedbackBenchmarkError::EmptyTrials)?;

    if first.participant_ref.trim().is_empty()
        || first.session_ref.trim().is_empty()
        || first.mapping_id.trim().is_empty()
    {
        return Err(FeedbackBenchmarkError::MissingTrialIdentity);
    }

    let mut trial_ids = HashSet::new();
    for trial in trials {
        if trial.trial_id.trim().is_empty()
            || trial.participant_ref.trim().is_empty()
            || trial.session_ref.trim().is_empty()
            || trial.mapping_id.trim().is_empty()
        {
            return Err(FeedbackBenchmarkError::MissingTrialIdentity);
        }
        if !trial_ids.insert(trial.trial_id.as_str()) {
            return Err(FeedbackBenchmarkError::DuplicateTrialId {
                trial_id: trial.trial_id.clone(),
            });
        }
        if trial.participant_ref != first.participant_ref
            || trial.session_ref != first.session_ref
            || trial.condition != first.condition
            || trial.mapping_id != first.mapping_id
            || trial.evidence_class != first.evidence_class
        {
            return Err(FeedbackBenchmarkError::MixedParticipantSessionModalityOrMapping);
        }
        if let Some(latency) = trial.response_latency_ms {
            if !latency.is_finite() || latency < 0.0 {
                return Err(FeedbackBenchmarkError::InvalidResponseLatency {
                    trial_id: trial.trial_id.clone(),
                });
            }
            if trial.reported.is_none() {
                return Err(FeedbackBenchmarkError::LatencyWithoutResponse {
                    trial_id: trial.trial_id.clone(),
                });
            }
        }
    }

    let semantic_classes = all_semantic_classes();
    let mut per_semantic_class = Vec::with_capacity(semantic_classes.len());

    for semantic in semantic_classes {
        let mut tp = 0u64;
        let mut fp = 0u64;
        let mut fn_count = 0u64;
        let mut tn = 0u64;

        for trial in trials {
            let expected_positive = trial.expected == Some(semantic);
            let reported_positive = trial.reported == Some(semantic);
            match (expected_positive, reported_positive) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_count += 1,
                (false, false) => tn += 1,
            }
        }

        per_semantic_class.push(CueClassMetrics {
            semantic,
            true_positive: tp,
            false_positive: fp,
            false_negative: fn_count,
            true_negative: tn,
            precision: ratio(tp, tp + fp),
            recall: ratio(tp, tp + fn_count),
            specificity: ratio(tn, tn + fp),
            f1: ratio(2 * tp, 2 * tp + fp + fn_count),
        });
    }

    let exact_matches = trials
        .iter()
        .filter(|trial| trial.expected == trial.reported)
        .count() as u64;
    let correct_cue_responses = trials
        .iter()
        .filter(|trial| trial.expected.is_some() && trial.expected == trial.reported)
        .count() as u64;
    let missed_expected_cues = trials
        .iter()
        .filter(|trial| trial.expected.is_some() && trial.expected != trial.reported)
        .count() as u64;
    let false_alarm_trials = trials
        .iter()
        .filter(|trial| trial.expected.is_none() && trial.reported.is_some())
        .count() as u64;
    let cue_substitutions = trials
        .iter()
        .filter(|trial| {
            trial.expected.is_some()
                && trial.reported.is_some()
                && trial.expected != trial.reported
        })
        .count() as u64;

    let mut correct_latencies: Vec<f64> = trials
        .iter()
        .filter(|trial| trial.expected.is_some() && trial.expected == trial.reported)
        .filter_map(|trial| trial.response_latency_ms)
        .collect();
    let median_correct_response_latency_ms = median(&mut correct_latencies);

    let f1_values: Vec<f64> = per_semantic_class
        .iter()
        .filter_map(|metrics| metrics.f1)
        .collect();
    let macro_f1 = if f1_values.is_empty() {
        None
    } else {
        Some(f1_values.iter().sum::<f64>() / f1_values.len() as f64)
    };

    Ok(FeedbackRecognitionSummaryV1 {
        participant_ref: first.participant_ref.clone(),
        session_ref: first.session_ref.clone(),
        condition: first.condition,
        mapping_id: first.mapping_id.clone(),
        evidence_class: first.evidence_class,
        trial_count: trials.len() as u64,
        exact_matches,
        exact_match_rate: ratio(exact_matches, trials.len() as u64),
        correct_cue_responses,
        missed_expected_cues,
        false_alarm_trials,
        cue_substitutions,
        median_correct_response_latency_ms,
        macro_f1,
        per_semantic_class,
    })
}

fn all_semantic_classes() -> Vec<HapticCueSemantic> {
    use crate::prosthetic_sensory::{GripBand, HandDigit};

    vec![
        HapticCueSemantic::Contact(HandDigit::Thumb),
        HapticCueSemantic::Contact(HandDigit::Index),
        HapticCueSemantic::Contact(HandDigit::Middle),
        HapticCueSemantic::Contact(HandDigit::Ring),
        HapticCueSemantic::Contact(HandDigit::Little),
        HapticCueSemantic::Grip(GripBand::WithinPreferred),
        HapticCueSemantic::Grip(GripBand::NearDeclaredLimit),
        HapticCueSemantic::Grip(GripBand::AboveDeclaredLimit),
        HapticCueSemantic::SlipRisk,
        HapticCueSemantic::DataUnavailable,
    ]
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[middle - 1] + values[middle]) / 2.0)
    } else {
        Some(values[middle])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(
        trial_id: &str,
        expected: Option<HapticCueSemantic>,
        reported: Option<HapticCueSemantic>,
        latency: Option<f64>,
    ) -> FeedbackRecognitionTrialV1 {
        FeedbackRecognitionTrialV1 {
            trial_id: trial_id.into(),
            participant_ref: "participant:p1".into(),
            session_ref: "session:s1".into(),
            condition: FeedbackModalityCondition::Vibrotactile,
            mapping_id: "mapping:v1".into(),
            evidence_class: SensoryEvidenceClass::HumanObserved,
            expected,
            reported,
            response_latency_ms: latency,
        }
    }

    #[test]
    fn perfect_recognition_reports_full_accuracy_and_latency() {
        let contact = HapticCueSemantic::Contact(
            crate::prosthetic_sensory::HandDigit::Thumb,
        );
        let trials = vec![
            trial("t1", Some(contact), Some(contact), Some(100.0)),
            trial("t2", Some(contact), Some(contact), Some(200.0)),
            trial("t3", None, None, None),
        ];

        let summary = evaluate_feedback_recognition(&trials).unwrap();
        assert_eq!(summary.trial_count, 3);
        assert_eq!(summary.exact_matches, 3);
        assert_eq!(summary.exact_match_rate, Some(1.0));
        assert_eq!(summary.correct_cue_responses, 2);
        assert_eq!(summary.false_alarm_trials, 0);
        assert_eq!(summary.missed_expected_cues, 0);
        assert_eq!(summary.median_correct_response_latency_ms, Some(150.0));
    }

    #[test]
    fn misses_false_alarms_and_substitutions_are_separate() {
        use crate::prosthetic_sensory::HandDigit;
        let thumb = HapticCueSemantic::Contact(HandDigit::Thumb);
        let index = HapticCueSemantic::Contact(HandDigit::Index);
        let trials = vec![
            trial("t1", Some(thumb), None, None),
            trial("t2", None, Some(index), Some(150.0)),
            trial("t3", Some(thumb), Some(index), Some(200.0)),
        ];

        let summary = evaluate_feedback_recognition(&trials).unwrap();
        assert_eq!(summary.missed_expected_cues, 2);
        assert_eq!(summary.false_alarm_trials, 1);
        assert_eq!(summary.cue_substitutions, 1);
        assert_eq!(summary.exact_matches, 0);

        let thumb_metrics = summary
            .per_semantic_class
            .iter()
            .find(|metrics| metrics.semantic == thumb)
            .unwrap();
        assert_eq!(thumb_metrics.false_negative, 2);

        let index_metrics = summary
            .per_semantic_class
            .iter()
            .find(|metrics| metrics.semantic == index)
            .unwrap();
        assert_eq!(index_metrics.false_positive, 2);
    }

    #[test]
    fn valid_strata_cannot_mix_participants_or_sessions() {
        let contact = HapticCueSemantic::Contact(
            crate::prosthetic_sensory::HandDigit::Thumb,
        );
        let mut trials = vec![trial("t1", Some(contact), Some(contact), Some(90.0))];
        let mut second = trial("t2", Some(contact), Some(contact), Some(95.0));
        second.participant_ref = "participant:p2".into();
        trials.push(second);

        assert_eq!(
            evaluate_feedback_recognition(&trials),
            Err(FeedbackBenchmarkError::MixedParticipantSessionModalityOrMapping)
        );
    }

    #[test]
    fn duplicate_trial_id_is_rejected() {
        let trials = vec![
            trial("same", None, None, None),
            trial("same", None, None, None),
        ];
        assert_eq!(
            evaluate_feedback_recognition(&trials),
            Err(FeedbackBenchmarkError::DuplicateTrialId {
                trial_id: "same".into()
            })
        );
    }

    #[test]
    fn response_latency_must_be_finite_and_nonnegative() {
        let trials = vec![trial("t1", None, Some(HapticCueSemantic::SlipRisk), Some(f64::NAN))];
        assert_eq!(
            evaluate_feedback_recognition(&trials),
            Err(FeedbackBenchmarkError::InvalidResponseLatency {
                trial_id: "t1".into()
            })
        );

        let trials = vec![trial("t2", None, None, Some(20.0))];
        assert_eq!(
            evaluate_feedback_recognition(&trials),
            Err(FeedbackBenchmarkError::LatencyWithoutResponse {
                trial_id: "t2".into()
            })
        );
    }

    #[test]
    fn evidence_class_and_mapping_are_preserved_per_stratum() {
        let trials = vec![trial("t1", None, None, None)];
        let summary = evaluate_feedback_recognition(&trials).unwrap();
        assert_eq!(summary.evidence_class, SensoryEvidenceClass::HumanObserved);
        assert_eq!(summary.mapping_id, "mapping:v1");
        assert_eq!(summary.condition, FeedbackModalityCondition::Vibrotactile);
        assert_eq!(summary.participant_ref, "participant:p1");
        assert_eq!(summary.session_ref, "session:s1");
    }
}
