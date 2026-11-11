// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Per-participant/session evaluation for prosthetic sensory cue recognition.
//!
//! This benchmark layer evaluates whether a participant correctly recognizes
//! an expected semantic event under one fixed modality, information source,
//! codebook and session. Modality (visual/audio/vibration) is distinct from the
//! information conveyed (EMG intent, requested control, measured contact/force,
//! or an estimated physical outcome). Never pool those source classes.
//!
//! It is intentionally separate from hand mechanics, task success, clinical
//! outcomes and the physical haptic renderer. Synthetic/simulated evidence is
//! retained as such and cannot establish human sensory performance.

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

/// Physical or inferred variable encoded by the feedback, independent of the
/// channel used to present it. In particular, EMG/intent is not measured
/// contact force, and a requested command is not an observed outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackInformationSource {
    /// No information is presented in the no-feedback reference condition.
    NoInformation,
    /// Residual-muscle / EMG signal representing user intent or effort.
    UserIntentEmg,
    /// Requested or projected control command; not proof of physical execution.
    RequestedControlState,
    /// Sensor measurement indicating physical contact.
    MeasuredContact,
    /// Calibrated sensor measurement of grip force.
    MeasuredGripForce,
    /// Estimated slip state; not a direct measurement unless separately established.
    EstimatedSlip,
    /// Measured prosthetic joint position.
    MeasuredJointPosition,
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
    /// Information source encoded by the feedback channel.
    pub information_source: FeedbackInformationSource,
    pub mapping_id: String,
    /// Shared stimulus/trial identity used only for paired crossover analysis.
    pub paired_trial_ref: Option<String>,
    pub evidence_class: SensoryEvidenceClass,
    pub expected: Option<HapticCueSemantic>,
    pub reported: Option<HapticCueSemantic>,
    pub response_latency_ms: Option<f64>,
}

/// One cell in the full confusion matrix, including no-cue responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackResponseClass {
    NoCue,
    Cue(HapticCueSemantic),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CueConfusionCell {
    pub expected: FeedbackResponseClass,
    pub reported: FeedbackResponseClass,
    pub count: u64,
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

/// Metrics for exactly one participant × session × modality × information-source × mapping stratum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackRecognitionSummaryV1 {
    pub participant_ref: String,
    pub session_ref: String,
    pub condition: FeedbackModalityCondition,
    pub information_source: FeedbackInformationSource,
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
    pub confusion_matrix: Vec<CueConfusionCell>,
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
    MissingPairIdentity { trial_id: String },
    DuplicatePairIdentity { pair_ref: String },
    PairSetsDoNotMatch,
    PairedExpectedCueMismatch { pair_ref: String },
    SameModalityCondition,
    InformationSourceConditionMismatch,
    DifferentParticipantOrSession,
    InvalidReferenceStratum,
    InvalidCandidateStratum,
}

/// Outcome for one identical stimulus presented under both conditions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedFeedbackPairResult {
    pub paired_trial_ref: String,
    pub expected: Option<HapticCueSemantic>,
    pub reference_reported: Option<HapticCueSemantic>,
    pub candidate_reported: Option<HapticCueSemantic>,
    pub reference_correct: bool,
    pub candidate_correct: bool,
    /// Candidate minus reference latency when both answers are correct and measured.
    pub candidate_minus_reference_latency_ms: Option<f64>,
}

/// Descriptive within-participant crossover comparison. This function does not
/// estimate a population treatment effect or statistical significance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedFeedbackComparisonV1 {
    pub participant_ref: String,
    pub session_ref: String,
    pub reference_condition: FeedbackModalityCondition,
    pub candidate_condition: FeedbackModalityCondition,
    pub reference_information_source: FeedbackInformationSource,
    pub candidate_information_source: FeedbackInformationSource,
    pub reference_mapping_id: String,
    pub candidate_mapping_id: String,
    pub pair_count: u64,
    pub both_correct: u64,
    pub candidate_only_correct: u64,
    pub reference_only_correct: u64,
    pub both_incorrect: u64,
    /// (candidate correct - reference correct) / pair_count.
    pub paired_accuracy_difference: Option<f64>,
    /// Negative values mean the candidate was faster.
    pub median_candidate_minus_reference_latency_ms: Option<f64>,
    pub pairs: Vec<PairedFeedbackPairResult>,
}

/// Compare two conditions using exactly matched stimuli for one participant
/// and session. Conditions may differ by modality, information source, or
/// mapping; all identities are retained in the output. The paired reference
/// label must still represent the same preregistered event in both conditions.
pub fn compare_paired_feedback_conditions(
    reference: &[FeedbackRecognitionTrialV1],
    candidate: &[FeedbackRecognitionTrialV1],
) -> Result<PairedFeedbackComparisonV1, FeedbackBenchmarkError> {
    let reference_summary = evaluate_feedback_recognition(reference)
        .map_err(|_| FeedbackBenchmarkError::InvalidReferenceStratum)?;
    let candidate_summary = evaluate_feedback_recognition(candidate)
        .map_err(|_| FeedbackBenchmarkError::InvalidCandidateStratum)?;

    if reference_summary.participant_ref != candidate_summary.participant_ref
        || reference_summary.session_ref != candidate_summary.session_ref
        || reference_summary.evidence_class != candidate_summary.evidence_class
    {
        return Err(FeedbackBenchmarkError::DifferentParticipantOrSession);
    }
    if reference_summary.condition == candidate_summary.condition
        && reference_summary.information_source == candidate_summary.information_source
        && reference_summary.mapping_id == candidate_summary.mapping_id
    {
        return Err(FeedbackBenchmarkError::SameModalityCondition);
    }

    let mut candidate_by_pair = std::collections::HashMap::new();
    for trial in candidate {
        let pair_ref = trial
            .paired_trial_ref
            .as_deref()
            .filter(|reference| !reference.trim().is_empty())
            .ok_or_else(|| FeedbackBenchmarkError::MissingPairIdentity {
                trial_id: trial.trial_id.clone(),
            })?;
        if candidate_by_pair.insert(pair_ref, trial).is_some() {
            return Err(FeedbackBenchmarkError::DuplicatePairIdentity {
                pair_ref: pair_ref.to_string(),
            });
        }
    }

    let mut seen_reference_pairs = HashSet::new();
    let mut pairs = Vec::with_capacity(reference.len());
    let mut both_correct = 0u64;
    let mut candidate_only_correct = 0u64;
    let mut reference_only_correct = 0u64;
    let mut both_incorrect = 0u64;
    let mut latency_differences = Vec::new();

    for reference_trial in reference {
        let pair_ref = reference_trial
            .paired_trial_ref
            .as_deref()
            .filter(|reference| !reference.trim().is_empty())
            .ok_or_else(|| FeedbackBenchmarkError::MissingPairIdentity {
                trial_id: reference_trial.trial_id.clone(),
            })?;
        if !seen_reference_pairs.insert(pair_ref) {
            return Err(FeedbackBenchmarkError::DuplicatePairIdentity {
                pair_ref: pair_ref.to_string(),
            });
        }
        let candidate_trial = candidate_by_pair
            .get(pair_ref)
            .ok_or(FeedbackBenchmarkError::PairSetsDoNotMatch)?;

        if reference_trial.expected != candidate_trial.expected {
            return Err(FeedbackBenchmarkError::PairedExpectedCueMismatch {
                pair_ref: pair_ref.to_string(),
            });
        }

        let reference_correct = reference_trial.expected == reference_trial.reported;
        let candidate_correct = candidate_trial.expected == candidate_trial.reported;
        match (reference_correct, candidate_correct) {
            (true, true) => {
                both_correct += 1;
                if let (Some(reference_latency), Some(candidate_latency)) = (
                    reference_trial.response_latency_ms,
                    candidate_trial.response_latency_ms,
                ) {
                    latency_differences.push(candidate_latency - reference_latency);
                }
            }
            (false, true) => candidate_only_correct += 1,
            (true, false) => reference_only_correct += 1,
            (false, false) => both_incorrect += 1,
        }

        let candidate_minus_reference_latency_ms =
            if reference_correct && candidate_correct {
                match (
                    reference_trial.response_latency_ms,
                    candidate_trial.response_latency_ms,
                ) {
                    (Some(reference_latency), Some(candidate_latency)) => {
                        Some(candidate_latency - reference_latency)
                    }
                    _ => None,
                }
            } else {
                None
            };

        pairs.push(PairedFeedbackPairResult {
            paired_trial_ref: pair_ref.to_string(),
            expected: reference_trial.expected,
            reference_reported: reference_trial.reported,
            candidate_reported: candidate_trial.reported,
            reference_correct,
            candidate_correct,
            candidate_minus_reference_latency_ms,
        });
    }

    if seen_reference_pairs.len() != candidate_by_pair.len() {
        return Err(FeedbackBenchmarkError::PairSetsDoNotMatch);
    }

    let pair_count = pairs.len() as u64;
    let accuracy_difference = if pair_count == 0 {
        None
    } else {
        Some(
            (candidate_only_correct as f64 - reference_only_correct as f64)
                / pair_count as f64,
        )
    };

    Ok(PairedFeedbackComparisonV1 {
        participant_ref: reference_summary.participant_ref,
        session_ref: reference_summary.session_ref,
        reference_condition: reference_summary.condition,
        candidate_condition: candidate_summary.condition,
        reference_information_source: reference_summary.information_source,
        candidate_information_source: candidate_summary.information_source,
        reference_mapping_id: reference_summary.mapping_id,
        candidate_mapping_id: candidate_summary.mapping_id,
        pair_count,
        both_correct,
        candidate_only_correct,
        reference_only_correct,
        both_incorrect,
        paired_accuracy_difference: accuracy_difference,
        median_candidate_minus_reference_latency_ms: median(&mut latency_differences),
        pairs,
    })
}

/// Evaluate one immutable stratum. Callers must group data by participant,
/// session, condition, information source and mapping before invoking this function.
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
        let source_matches_condition = match trial.condition {
            FeedbackModalityCondition::NoFeedback => {
                trial.information_source == FeedbackInformationSource::NoInformation
            }
            FeedbackModalityCondition::VisualOnly
            | FeedbackModalityCondition::Auditory
            | FeedbackModalityCondition::Vibrotactile
            | FeedbackModalityCondition::Multimodal => {
                trial.information_source != FeedbackInformationSource::NoInformation
            }
        };
        if !source_matches_condition {
            return Err(FeedbackBenchmarkError::InformationSourceConditionMismatch);
        }
        if trial.participant_ref != first.participant_ref
            || trial.session_ref != first.session_ref
            || trial.condition != first.condition
            || trial.information_source != first.information_source
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

    // Preserve the full cue-to-cue substitution structure. One-vs-rest
    // metrics alone reveal false positives/negatives but not which two
    // semantic codes the participant confuses.
    let mut response_classes = vec![FeedbackResponseClass::NoCue];
    response_classes.extend(
        all_semantic_classes()
            .into_iter()
            .map(FeedbackResponseClass::Cue),
    );
    let mut confusion_matrix = Vec::with_capacity(
        response_classes.len() * response_classes.len(),
    );
    for expected in &response_classes {
        for reported in &response_classes {
            let count = trials
                .iter()
                .filter(|trial| response_class(trial.expected) == *expected
                    && response_class(trial.reported) == *reported)
                .count() as u64;
            confusion_matrix.push(CueConfusionCell {
                expected: *expected,
                reported: *reported,
                count,
            });
        }
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
        information_source: first.information_source,
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
        confusion_matrix,
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

fn response_class(value: Option<HapticCueSemantic>) -> FeedbackResponseClass {
    value
        .map(FeedbackResponseClass::Cue)
        .unwrap_or(FeedbackResponseClass::NoCue)
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
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
            information_source: FeedbackInformationSource::MeasuredContact,
            mapping_id: "mapping:v1".into(),
            paired_trial_ref: None,
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

        let confusion = |expected, reported| {
            summary
                .confusion_matrix
                .iter()
                .find(|cell| cell.expected == expected && cell.reported == reported)
                .map(|cell| cell.count)
                .unwrap()
        };
        assert_eq!(
            confusion(
                FeedbackResponseClass::Cue(thumb),
                FeedbackResponseClass::NoCue
            ),
            1
        );
        assert_eq!(
            confusion(
                FeedbackResponseClass::NoCue,
                FeedbackResponseClass::Cue(index)
            ),
            1
        );
        assert_eq!(
            confusion(
                FeedbackResponseClass::Cue(thumb),
                FeedbackResponseClass::Cue(index)
            ),
            1
        );
        assert_eq!(summary.confusion_matrix.len(), 121);
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
    fn paired_modality_comparison_reports_improved_and_degraded_trials() {
        use crate::prosthetic_sensory::HandDigit;
        let thumb = HapticCueSemantic::Contact(HandDigit::Thumb);
        let index = HapticCueSemantic::Contact(HandDigit::Index);

        let mut reference = vec![
            trial("r1", Some(thumb), None, None),
            trial("r2", Some(thumb), Some(thumb), Some(220.0)),
            trial("r3", Some(thumb), Some(index), Some(300.0)),
            trial("r4", Some(thumb), Some(thumb), Some(180.0)),
        ];
        let mut candidate = vec![
            trial("c1", Some(thumb), Some(thumb), Some(160.0)),
            trial("c2", Some(thumb), Some(thumb), Some(180.0)),
            trial("c3", Some(thumb), Some(thumb), Some(210.0)),
            trial("c4", Some(thumb), None, None),
        ];

        for (index, pair_ref) in ["pair-1", "pair-2", "pair-3", "pair-4"]
            .into_iter()
            .enumerate()
        {
            reference[index].paired_trial_ref = Some(pair_ref.into());
            candidate[index].paired_trial_ref = Some(pair_ref.into());
        }
        reference[0].condition = FeedbackModalityCondition::NoFeedback;
        reference[0].mapping_id = "no-feedback".into();
        for trial in &mut reference {
            trial.condition = FeedbackModalityCondition::NoFeedback;
            trial.information_source = FeedbackInformationSource::NoInformation;
            trial.mapping_id = "no-feedback".into();
        }
        for trial in &mut candidate {
            trial.condition = FeedbackModalityCondition::Vibrotactile;
            trial.mapping_id = "user-haptic-map-v2".into();
        }

        let comparison = compare_paired_feedback_conditions(&reference, &candidate).unwrap();
        assert_eq!(comparison.pair_count, 4);
        assert_eq!(comparison.both_correct, 1);
        assert_eq!(comparison.candidate_only_correct, 2);
        assert_eq!(comparison.reference_only_correct, 1);
        assert_eq!(comparison.both_incorrect, 0);
        assert_eq!(comparison.paired_accuracy_difference, Some(0.25));
        // Both-correct pair 2: 180 ms candidate vs 220 ms reference.
        // Pair 4 is a candidate miss, so it is excluded from correct-response latency.
        assert_eq!(
            comparison.median_candidate_minus_reference_latency_ms,
            Some(-40.0)
        );
        assert_eq!(comparison.reference_mapping_id, "no-feedback");
        assert_eq!(comparison.candidate_mapping_id, "user-haptic-map-v2");
    }

    #[test]
    fn paired_comparison_rejects_unmatched_or_different_expected_stimuli() {
        let contact = HapticCueSemantic::Contact(
            crate::prosthetic_sensory::HandDigit::Thumb,
        );
        let mut reference = vec![trial("r1", Some(contact), Some(contact), Some(100.0))];
        let mut candidate = vec![trial("c1", Some(contact), Some(contact), Some(90.0))];
        reference[0].paired_trial_ref = Some("stimulus-1".into());
        candidate[0].paired_trial_ref = Some("stimulus-2".into());
        candidate[0].condition = FeedbackModalityCondition::Auditory;
        assert_eq!(
            compare_paired_feedback_conditions(&reference, &candidate),
            Err(FeedbackBenchmarkError::PairSetsDoNotMatch)
        );

        candidate[0].paired_trial_ref = Some("stimulus-1".into());
        candidate[0].expected = None;
        assert_eq!(
            compare_paired_feedback_conditions(&reference, &candidate),
            Err(FeedbackBenchmarkError::PairedExpectedCueMismatch {
                pair_ref: "stimulus-1".into()
            })
        );
    }

    #[test]
    fn paired_comparison_rejects_different_participant_and_modality_identity() {
        let contact = HapticCueSemantic::Contact(
            crate::prosthetic_sensory::HandDigit::Thumb,
        );
        let mut reference = vec![trial("r1", Some(contact), Some(contact), Some(100.0))];
        let mut candidate = vec![trial("c1", Some(contact), Some(contact), Some(90.0))];
        reference[0].paired_trial_ref = Some("stimulus-1".into());
        candidate[0].paired_trial_ref = Some("stimulus-1".into());
        candidate[0].condition = FeedbackModalityCondition::Auditory;
        candidate[0].participant_ref = "participant:p2".into();
        assert_eq!(
            compare_paired_feedback_conditions(&reference, &candidate),
            Err(FeedbackBenchmarkError::DifferentParticipantOrSession)
        );

        candidate[0].participant_ref = "participant:p1".into();
        candidate[0].condition = FeedbackModalityCondition::Vibrotactile;
        candidate[0].session_ref = "session:s2".into();
        assert_eq!(
            compare_paired_feedback_conditions(&reference, &candidate),
            Err(FeedbackBenchmarkError::DifferentParticipantOrSession)
        );
    }

    #[test]
    fn information_sources_are_separate_strata() {
        let contact = HapticCueSemantic::Contact(
            crate::prosthetic_sensory::HandDigit::Index,
        );
        let mut trials = vec![trial("t1", Some(contact), Some(contact), Some(100.0))];
        let mut emg = trial("t2", Some(contact), Some(contact), Some(110.0));
        emg.information_source = FeedbackInformationSource::UserIntentEmg;
        trials.push(emg);

        assert_eq!(
            evaluate_feedback_recognition(&trials),
            Err(FeedbackBenchmarkError::MixedParticipantSessionModalityOrMapping)
        );
    }

    #[test]
    fn rejects_no_feedback_with_an_information_source() {
        let mut sample = trial("t1", None, None, None);
        sample.condition = FeedbackModalityCondition::NoFeedback;
        assert_eq!(
            evaluate_feedback_recognition(&[sample]),
            Err(FeedbackBenchmarkError::InformationSourceConditionMismatch)
        );
    }

    #[test]
    fn paired_same_modality_comparison_can_isolate_information_source() {
        let contact = HapticCueSemantic::Contact(
            crate::prosthetic_sensory::HandDigit::Index,
        );
        let mut intent = vec![trial("i1", Some(contact), Some(contact), Some(120.0))];
        let mut measured_force = vec![trial("f1", Some(contact), Some(contact), Some(100.0))];
        intent[0].information_source = FeedbackInformationSource::UserIntentEmg;
        measured_force[0].information_source = FeedbackInformationSource::MeasuredGripForce;
        intent[0].paired_trial_ref = Some("shared-stimulus-1".into());
        measured_force[0].paired_trial_ref = Some("shared-stimulus-1".into());

        let comparison =
            compare_paired_feedback_conditions(&intent, &measured_force).unwrap();
        assert_eq!(
            comparison.reference_information_source,
            FeedbackInformationSource::UserIntentEmg
        );
        assert_eq!(
            comparison.candidate_information_source,
            FeedbackInformationSource::MeasuredGripForce
        );
        assert_eq!(comparison.reference_condition, FeedbackModalityCondition::Vibrotactile);
        assert_eq!(comparison.candidate_condition, FeedbackModalityCondition::Vibrotactile);
        assert_eq!(comparison.paired_accuracy_difference, Some(0.0));
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
