// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired audio-versus-symbolic music perception evaluation.
//!
//! A model that answers from MIDI is not thereby an audio listener. This
//! evaluator binds two modality-specific inputs to the same source-score case,
//! scores each modality independently, and keeps abstentions/missing outputs in
//! the all-cases denominator. It evaluates perception; it does not claim that
//! any particular front-end or model has passed the benchmark.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const PERCEPTION_PARITY_SCHEMA_V1: &str = "mel-perception-parity-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PerceptionTaskV1 {
    SyncopationScoring,
    TranspositionDetection,
    ChordQualityIdentification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PerceptionModalityV1 {
    Audio,
    SymbolicMidi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkSplitV1 {
    Development,
    Validation,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChordQualityV1 {
    Major,
    Minor,
    Dominant,
    Diminished,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "task", rename_all = "snake_case")]
pub enum PerceptionAnswerV1 {
    SyncopationLevel { level: u8 },
    TranspositionDetected { detected: bool },
    ChordQuality { quality: ChordQualityV1 },
}

impl PerceptionAnswerV1 {
    fn matches_task(&self, task: PerceptionTaskV1) -> bool {
        matches!(
            (self, task),
            (
                Self::SyncopationLevel { .. },
                PerceptionTaskV1::SyncopationScoring
            ) | (
                Self::TranspositionDetected { .. },
                PerceptionTaskV1::TranspositionDetection
            ) | (
                Self::ChordQuality { .. },
                PerceptionTaskV1::ChordQualityIdentification
            )
        )
    }
}

/// The same case ID and source-score digest make the audio and MIDI arms a
/// paired comparison; input digests remain modality-specific.
/// One underlying musical case with separate, content-addressed inputs for
/// audio and symbolic MIDI. Both inputs derive from the same source score and
/// share the same ground truth.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptionCaseV1 {
    pub case_id: String,
    pub task: PerceptionTaskV1,
    pub split: BenchmarkSplitV1,
    pub source_score_sha256: String,
    pub audio_input_sha256: String,
    pub symbolic_input_sha256: String,
    pub renderer_recipe_sha256: String,
    pub ground_truth: PerceptionAnswerV1,
}

/// Immutable identity for one evaluated system/configuration. Opaque cloud
/// systems may use provider revision digests; locally hosted systems should use
/// the exact model artifact digest. Missing model identity is not a valid run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptionModelIdentityV1 {
    pub system_id: String,
    pub system_version: String,
    pub model_identity_sha256: String,
    pub preprocessing_identity_sha256: String,
    pub prompt_or_policy_sha256: String,
}

/// A single prediction. A missing answer is an explicit abstention, not a
/// missing row. Missing rows and abstentions both remain in the denominator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptionPredictionV1 {
    pub case_id: String,
    pub modality: PerceptionModalityV1,
    pub input_sha256: String,
    pub answer: Option<PerceptionAnswerV1>,
    /// Required for an answer; absent for an explicit abstention.
    pub confidence: Option<f32>,
}

/// One model's complete or partial execution against a frozen suite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptionRunV1 {
    pub schema: String,
    pub run_id: String,
    pub suite_id: String,
    pub model: PerceptionModelIdentityV1,
    pub predictions: Vec<PerceptionPredictionV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptionParitySuiteV1 {
    pub schema: String,
    pub suite_id: String,
    pub benchmark_version: String,
    pub cases: Vec<PerceptionCaseV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptionParityErrorV1 {
    InvalidSchema,
    EmptySuite,
    EmptyIdentity {
        field: String,
    },
    InvalidDigest {
        field: String,
    },
    DuplicateCaseId {
        case_id: String,
    },
    MissingTaskCases {
        task: PerceptionTaskV1,
    },
    RunSuiteMismatch,
    UnknownCaseId {
        case_id: String,
    },
    DuplicatePrediction {
        case_id: String,
        modality: PerceptionModalityV1,
    },
    InputDigestMismatch {
        case_id: String,
        modality: PerceptionModalityV1,
    },
    WrongAnswerTask {
        case_id: String,
        task: PerceptionTaskV1,
    },
    InvalidSyncopationLevel {
        case_id: String,
        level: u8,
    },
    InvalidConfidence {
        case_id: String,
    },
    ConfidenceMissingForAnswer {
        case_id: String,
    },
    ConfidencePresentForAbstention {
        case_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalityParityMetricsV1 {
    pub case_count: usize,
    pub answered_count: usize,
    pub abstained_count: usize,
    pub missing_count: usize,
    pub correct_count: usize,
    /// Correct answers divided by every case. Abstentions and missing answers
    /// are incorrect here, preventing silent denominator reduction.
    pub accuracy_all_cases: f64,
    /// Correct answers divided by answered cases; None if every item was
    /// abstained or omitted in this modality/task stratum.
    pub accuracy_answered_only: Option<f64>,
    pub coverage: f64,
    /// Defined only for ordinal syncopation levels, over answered rows only.
    pub mean_absolute_error_answered: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptionTaskParityMetricsV1 {
    pub task: PerceptionTaskV1,
    pub audio: ModalityParityMetricsV1,
    pub symbolic_midi: ModalityParityMetricsV1,
    /// Audio accuracy minus symbolic-MIDI accuracy, both using all cases.
    /// Negative values expose an audio-perception gap.
    pub audio_minus_symbolic_accuracy: f64,
    /// Agreement of audio and MIDI answers where both modalities answered.
    pub paired_answer_agreement: Option<f64>,
    pub paired_answer_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptionParityReportV1 {
    pub schema: String,
    pub suite_id: String,
    pub run_id: String,
    pub model: PerceptionModelIdentityV1,
    pub total_cases: usize,
    pub task_metrics: Vec<PerceptionTaskParityMetricsV1>,
}

/// Evaluate one model run. Every suite case is scored under both modalities;
/// predictions may be absent or abstain, but neither is removed from coverage
/// or all-cases accuracy.
pub fn evaluate_perception_parity(
    suite: &PerceptionParitySuiteV1,
    run: &PerceptionRunV1,
) -> Result<PerceptionParityReportV1, PerceptionParityErrorV1> {
    validate_suite(suite)?;
    validate_run(suite, run)?;

    let mut task_metrics = Vec::with_capacity(3);
    for task in [
        PerceptionTaskV1::SyncopationScoring,
        PerceptionTaskV1::TranspositionDetection,
        PerceptionTaskV1::ChordQualityIdentification,
    ] {
        let task_cases: Vec<_> = suite
            .cases
            .iter()
            .filter(|case| case.task == task)
            .collect();
        let audio = modality_metrics(
            &task_cases,
            &run.predictions,
            PerceptionModalityV1::Audio,
            task,
        );
        let symbolic_midi = modality_metrics(
            &task_cases,
            &run.predictions,
            PerceptionModalityV1::SymbolicMidi,
            task,
        );
        let paired: Vec<_> = task_cases
            .iter()
            .filter_map(|case| {
                let audio_prediction =
                    prediction_for(&run.predictions, &case.case_id, PerceptionModalityV1::Audio)?
                        .answer
                        .as_ref()?;
                let symbolic_prediction = prediction_for(
                    &run.predictions,
                    &case.case_id,
                    PerceptionModalityV1::SymbolicMidi,
                )?
                .answer
                .as_ref()?;
                Some((audio_prediction, symbolic_prediction))
            })
            .collect();
        let paired_agreement = if paired.is_empty() {
            None
        } else {
            let agreements = paired
                .iter()
                .filter(|(audio, symbolic)| audio == symbolic)
                .count();
            Some(agreements as f64 / paired.len() as f64)
        };
        task_metrics.push(PerceptionTaskParityMetricsV1 {
            task,
            audio_minus_symbolic_accuracy: audio.accuracy_all_cases
                - symbolic_midi.accuracy_all_cases,
            audio,
            symbolic_midi,
            paired_answer_agreement: paired_agreement,
            paired_answer_count: paired.len(),
        });
    }

    Ok(PerceptionParityReportV1 {
        schema: PERCEPTION_PARITY_SCHEMA_V1.to_owned(),
        suite_id: suite.suite_id.clone(),
        run_id: run.run_id.clone(),
        model: run.model.clone(),
        total_cases: suite.cases.len(),
        task_metrics,
    })
}

fn modality_metrics(
    cases: &[&PerceptionCaseV1],
    predictions: &[PerceptionPredictionV1],
    modality: PerceptionModalityV1,
    task: PerceptionTaskV1,
) -> ModalityParityMetricsV1 {
    let total = cases.len();
    let mut answered = 0usize;
    let mut abstained = 0usize;
    let mut missing = 0usize;
    let mut correct = 0usize;
    let mut absolute_error = 0u64;
    let mut ordinal_answered = 0usize;

    for case in cases {
        let Some(prediction) = prediction_for(predictions, &case.case_id, modality) else {
            missing += 1;
            continue;
        };
        let Some(answer) = prediction.answer.as_ref() else {
            abstained += 1;
            continue;
        };
        answered += 1;
        if answer == &case.ground_truth {
            correct += 1;
        }
        if let (
            PerceptionTaskV1::SyncopationScoring,
            PerceptionAnswerV1::SyncopationLevel { level: expected },
            PerceptionAnswerV1::SyncopationLevel { level: actual },
        ) = (task, &case.ground_truth, answer)
        {
            absolute_error += u64::from(expected.abs_diff(*actual));
            ordinal_answered += 1;
        }
    }

    ModalityParityMetricsV1 {
        case_count: total,
        answered_count: answered,
        abstained_count: abstained,
        missing_count: missing,
        correct_count: correct,
        accuracy_all_cases: correct as f64 / total as f64,
        accuracy_answered_only: (answered > 0).then_some(correct as f64 / answered as f64),
        coverage: answered as f64 / total as f64,
        mean_absolute_error_answered: (ordinal_answered > 0)
            .then_some(absolute_error as f64 / ordinal_answered as f64),
    }
}

fn prediction_for<'a>(
    predictions: &'a [PerceptionPredictionV1],
    case_id: &str,
    modality: PerceptionModalityV1,
) -> Option<&'a PerceptionPredictionV1> {
    predictions
        .iter()
        .find(|prediction| prediction.case_id == case_id && prediction.modality == modality)
}

fn validate_suite(suite: &PerceptionParitySuiteV1) -> Result<(), PerceptionParityErrorV1> {
    if suite.schema != PERCEPTION_PARITY_SCHEMA_V1
        || suite.suite_id.trim().is_empty()
        || suite.benchmark_version.trim().is_empty()
    {
        return Err(PerceptionParityErrorV1::InvalidSchema);
    }
    if suite.cases.is_empty() {
        return Err(PerceptionParityErrorV1::EmptySuite);
    }

    let mut case_ids = HashSet::with_capacity(suite.cases.len());
    for case in &suite.cases {
        if case.case_id.trim().is_empty() {
            return Err(PerceptionParityErrorV1::EmptyIdentity {
                field: "case_id".into(),
            });
        }
        if !case_ids.insert(case.case_id.as_str()) {
            return Err(PerceptionParityErrorV1::DuplicateCaseId {
                case_id: case.case_id.clone(),
            });
        }
        validate_digest("source_score_sha256", &case.source_score_sha256)?;
        validate_digest("audio_input_sha256", &case.audio_input_sha256)?;
        validate_digest("symbolic_input_sha256", &case.symbolic_input_sha256)?;
        validate_digest("renderer_recipe_sha256", &case.renderer_recipe_sha256)?;
        validate_answer(&case.case_id, case.task, &case.ground_truth)?;
    }

    for task in [
        PerceptionTaskV1::SyncopationScoring,
        PerceptionTaskV1::TranspositionDetection,
        PerceptionTaskV1::ChordQualityIdentification,
    ] {
        if !suite.cases.iter().any(|case| case.task == task) {
            return Err(PerceptionParityErrorV1::MissingTaskCases { task });
        }
    }
    Ok(())
}

fn validate_run(
    suite: &PerceptionParitySuiteV1,
    run: &PerceptionRunV1,
) -> Result<(), PerceptionParityErrorV1> {
    if run.schema != PERCEPTION_PARITY_SCHEMA_V1
        || run.run_id.trim().is_empty()
        || run.suite_id != suite.suite_id
    {
        return Err(PerceptionParityErrorV1::RunSuiteMismatch);
    }
    if run.model.system_id.trim().is_empty() || run.model.system_version.trim().is_empty() {
        return Err(PerceptionParityErrorV1::EmptyIdentity {
            field: "model.system_id/system_version".into(),
        });
    }
    validate_digest("model_identity_sha256", &run.model.model_identity_sha256)?;
    validate_digest(
        "preprocessing_identity_sha256",
        &run.model.preprocessing_identity_sha256,
    )?;
    validate_digest(
        "prompt_or_policy_sha256",
        &run.model.prompt_or_policy_sha256,
    )?;

    let mut prediction_keys = HashSet::with_capacity(run.predictions.len());
    for prediction in &run.predictions {
        let Some(case) = suite
            .cases
            .iter()
            .find(|case| case.case_id == prediction.case_id)
        else {
            return Err(PerceptionParityErrorV1::UnknownCaseId {
                case_id: prediction.case_id.clone(),
            });
        };
        if !prediction_keys.insert((prediction.case_id.as_str(), prediction.modality)) {
            return Err(PerceptionParityErrorV1::DuplicatePrediction {
                case_id: prediction.case_id.clone(),
                modality: prediction.modality,
            });
        }
        let expected_digest = match prediction.modality {
            PerceptionModalityV1::Audio => &case.audio_input_sha256,
            PerceptionModalityV1::SymbolicMidi => &case.symbolic_input_sha256,
        };
        if &prediction.input_sha256 != expected_digest {
            return Err(PerceptionParityErrorV1::InputDigestMismatch {
                case_id: prediction.case_id.clone(),
                modality: prediction.modality,
            });
        }
        match (&prediction.answer, prediction.confidence) {
            (Some(answer), Some(confidence)) => {
                if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                    return Err(PerceptionParityErrorV1::InvalidConfidence {
                        case_id: prediction.case_id.clone(),
                    });
                }
                validate_answer(&prediction.case_id, case.task, answer)?;
            }
            (Some(_), None) => {
                return Err(PerceptionParityErrorV1::ConfidenceMissingForAnswer {
                    case_id: prediction.case_id.clone(),
                });
            }
            (None, Some(_)) => {
                return Err(PerceptionParityErrorV1::ConfidencePresentForAbstention {
                    case_id: prediction.case_id.clone(),
                });
            }
            (None, None) => {}
        }
    }
    Ok(())
}

fn validate_answer(
    case_id: &str,
    task: PerceptionTaskV1,
    answer: &PerceptionAnswerV1,
) -> Result<(), PerceptionParityErrorV1> {
    if !answer.matches_task(task) {
        return Err(PerceptionParityErrorV1::WrongAnswerTask {
            case_id: case_id.to_owned(),
            task,
        });
    }
    if let PerceptionAnswerV1::SyncopationLevel { level } = answer {
        if *level > 4 {
            return Err(PerceptionParityErrorV1::InvalidSyncopationLevel {
                case_id: case_id.to_owned(),
                level: *level,
            });
        }
    }
    Ok(())
}

fn validate_digest(field: &str, value: &str) -> Result<(), PerceptionParityErrorV1> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PerceptionParityErrorV1::InvalidDigest {
            field: field.to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> String {
        format!("{value:02x}{}", "a".repeat(62))
    }

    fn case(
        id: &str,
        task: PerceptionTaskV1,
        truth: PerceptionAnswerV1,
        seed: u8,
    ) -> PerceptionCaseV1 {
        PerceptionCaseV1 {
            case_id: id.into(),
            task,
            split: BenchmarkSplitV1::Test,
            source_score_sha256: digest(seed),
            audio_input_sha256: digest(seed + 1),
            symbolic_input_sha256: digest(seed + 2),
            renderer_recipe_sha256: digest(seed + 3),
            ground_truth: truth,
        }
    }

    fn suite() -> PerceptionParitySuiteV1 {
        PerceptionParitySuiteV1 {
            schema: PERCEPTION_PARITY_SCHEMA_V1.into(),
            suite_id: "muse-perception-parity-fixture-v1".into(),
            benchmark_version: "fixture-1".into(),
            cases: vec![
                case(
                    "sync-0",
                    PerceptionTaskV1::SyncopationScoring,
                    PerceptionAnswerV1::SyncopationLevel { level: 0 },
                    1,
                ),
                case(
                    "sync-3",
                    PerceptionTaskV1::SyncopationScoring,
                    PerceptionAnswerV1::SyncopationLevel { level: 3 },
                    5,
                ),
                case(
                    "transpose-yes",
                    PerceptionTaskV1::TranspositionDetection,
                    PerceptionAnswerV1::TranspositionDetected { detected: true },
                    9,
                ),
                case(
                    "transpose-no",
                    PerceptionTaskV1::TranspositionDetection,
                    PerceptionAnswerV1::TranspositionDetected { detected: false },
                    13,
                ),
                case(
                    "chord-major",
                    PerceptionTaskV1::ChordQualityIdentification,
                    PerceptionAnswerV1::ChordQuality {
                        quality: ChordQualityV1::Major,
                    },
                    17,
                ),
                case(
                    "chord-diminished",
                    PerceptionTaskV1::ChordQualityIdentification,
                    PerceptionAnswerV1::ChordQuality {
                        quality: ChordQualityV1::Diminished,
                    },
                    21,
                ),
            ],
        }
    }

    fn identity() -> PerceptionModelIdentityV1 {
        PerceptionModelIdentityV1 {
            system_id: "fixture-perceiver".into(),
            system_version: "0.1.0".into(),
            model_identity_sha256: digest(31),
            preprocessing_identity_sha256: digest(32),
            prompt_or_policy_sha256: digest(33),
        }
    }

    fn complete_run(suite: &PerceptionParitySuiteV1) -> PerceptionRunV1 {
        let mut predictions = Vec::new();
        for case in &suite.cases {
            for modality in [
                PerceptionModalityV1::Audio,
                PerceptionModalityV1::SymbolicMidi,
            ] {
                let input_sha256 = match modality {
                    PerceptionModalityV1::Audio => case.audio_input_sha256.clone(),
                    PerceptionModalityV1::SymbolicMidi => case.symbolic_input_sha256.clone(),
                };
                predictions.push(PerceptionPredictionV1 {
                    case_id: case.case_id.clone(),
                    modality,
                    input_sha256,
                    answer: Some(case.ground_truth.clone()),
                    confidence: Some(0.9),
                });
            }
        }
        PerceptionRunV1 {
            schema: PERCEPTION_PARITY_SCHEMA_V1.into(),
            run_id: "fixture-run-1".into(),
            suite_id: suite.suite_id.clone(),
            model: identity(),
            predictions,
        }
    }

    #[test]
    fn perfect_paired_predictions_have_zero_modality_gap() {
        let suite = suite();
        let run = complete_run(&suite);
        let report = evaluate_perception_parity(&suite, &run).unwrap();
        assert_eq!(report.total_cases, 6);
        assert_eq!(report.task_metrics.len(), 3);
        for metric in &report.task_metrics {
            assert_eq!(metric.audio.accuracy_all_cases, 1.0);
            assert_eq!(metric.symbolic_midi.accuracy_all_cases, 1.0);
            assert_eq!(metric.audio_minus_symbolic_accuracy, 0.0);
            assert_eq!(metric.paired_answer_agreement, Some(1.0));
        }
    }

    #[test]
    fn missing_and_abstained_predictions_stay_in_all_case_denominator() {
        let suite = suite();
        let mut run = complete_run(&suite);
        run.predictions.retain(|prediction| {
            !(prediction.case_id == "sync-0" && prediction.modality == PerceptionModalityV1::Audio)
        });
        let abstain = run
            .predictions
            .iter_mut()
            .find(|prediction| {
                prediction.case_id == "sync-3" && prediction.modality == PerceptionModalityV1::Audio
            })
            .unwrap();
        abstain.answer = None;
        abstain.confidence = None;

        let report = evaluate_perception_parity(&suite, &run).unwrap();
        let sync = report
            .task_metrics
            .iter()
            .find(|metric| metric.task == PerceptionTaskV1::SyncopationScoring)
            .unwrap();
        assert_eq!(sync.audio.case_count, 2);
        assert_eq!(sync.audio.missing_count, 1);
        assert_eq!(sync.audio.abstained_count, 1);
        assert_eq!(sync.audio.answered_count, 0);
        assert_eq!(sync.audio.accuracy_all_cases, 0.0);
        assert_eq!(sync.audio.accuracy_answered_only, None);
        assert_eq!(sync.audio.coverage, 0.0);
        assert_eq!(sync.symbolic_midi.accuracy_all_cases, 1.0);
        assert_eq!(sync.audio_minus_symbolic_accuracy, -1.0);
    }

    #[test]
    fn syncopation_reports_ordinal_error_separately_from_exact_accuracy() {
        let suite = suite();
        let mut run = complete_run(&suite);
        let prediction = run
            .predictions
            .iter_mut()
            .find(|prediction| {
                prediction.case_id == "sync-3" && prediction.modality == PerceptionModalityV1::Audio
            })
            .unwrap();
        prediction.answer = Some(PerceptionAnswerV1::SyncopationLevel { level: 2 });
        let report = evaluate_perception_parity(&suite, &run).unwrap();
        let sync = report
            .task_metrics
            .iter()
            .find(|metric| metric.task == PerceptionTaskV1::SyncopationScoring)
            .unwrap();
        assert_eq!(sync.audio.accuracy_all_cases, 0.5);
        assert_eq!(sync.audio.mean_absolute_error_answered, Some(0.5));
    }

    #[test]
    fn duplicate_case_modality_predictions_are_rejected() {
        let suite = suite();
        let mut run = complete_run(&suite);
        run.predictions.push(run.predictions[0].clone());
        assert!(matches!(
            evaluate_perception_parity(&suite, &run),
            Err(PerceptionParityErrorV1::DuplicatePrediction { .. })
        ));
    }

    #[test]
    fn wrong_modality_input_digest_is_rejected() {
        let suite = suite();
        let mut run = complete_run(&suite);
        run.predictions[0].input_sha256 = digest(99);
        assert!(matches!(
            evaluate_perception_parity(&suite, &run),
            Err(PerceptionParityErrorV1::InputDigestMismatch { .. })
        ));
    }

    #[test]
    fn task_mismatched_answers_and_invalid_confidence_are_rejected() {
        let suite = suite();
        let mut wrong_answer = complete_run(&suite);
        wrong_answer.predictions[0].answer =
            Some(PerceptionAnswerV1::TranspositionDetected { detected: true });
        assert!(matches!(
            evaluate_perception_parity(&suite, &wrong_answer),
            Err(PerceptionParityErrorV1::WrongAnswerTask { .. })
        ));

        let mut invalid_confidence = complete_run(&suite);
        invalid_confidence.predictions[0].confidence = Some(f32::NAN);
        assert!(matches!(
            evaluate_perception_parity(&suite, &invalid_confidence),
            Err(PerceptionParityErrorV1::InvalidConfidence { .. })
        ));
    }

    #[test]
    fn invalid_syncopation_level_is_rejected() {
        let mut suite = suite();
        suite.cases[0].ground_truth = PerceptionAnswerV1::SyncopationLevel { level: 5 };
        assert!(matches!(
            evaluate_perception_parity(&suite, &complete_run(&suite)),
            Err(PerceptionParityErrorV1::InvalidSyncopationLevel { .. })
        ));
    }
}
