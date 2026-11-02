// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Held-out few-shot cultural-transfer metrics inspired by MEDAL-ADR.
//!
//! The score separates performance while an expert is present, performance
//! after the expert leaves, and independent learner performance. This module
//! implements the published score and adds strict task/family leakage checks.
//!
//! It is a metrics adapter, not a port of GoalCycle3D or MEDAL-ADR.

use core::fmt;
use std::collections::BTreeSet;

/// Raw scores for one held-out task instance.
#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutTaskScore {
    pub task_id: String,
    pub task_family_id: String,
    pub expert_score: f64,
    pub agent_with_expert_full: f64,
    pub agent_with_expert_half: f64,
    pub agent_solo: f64,
    /// Optional action-level fidelity; diagnostic only, not retained skill.
    pub action_fidelity: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaskTransferMetrics {
    pub following_gain: f64,
    pub half_exposure_gain: f64,
    /// Agent score in the half-exposure condition divided by expert score.
    /// Values above 1 indicate performance beyond the expert's reference score.
    pub half_exposure_normalized_score: f64,
    pub cultural_transmission: f64,
    /// Fraction of the full-expert gain attained in the half-exposure condition.
    /// This includes the first half when the expert is present; it is not a
    /// pure post-dropout-only reward measure.
    pub half_exposure_gain_fraction: Option<f64>,
    pub action_fidelity: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferError {
    EmptyTaskId,
    EmptyTaskFamilyId,
    DuplicateTrainingTaskId { task_id: String },
    DuplicateEvaluationTaskId { task_id: String },
    TrainingEvaluationTaskLeakage { task_id: String },
    EvaluationTaskOutsideHeldOutFamily { task_id: String, family_id: String },
    NoEvaluationTasks,
    NonFiniteScore { task_id: String, field: &'static str },
    NonPositiveExpertScore { task_id: String, score_bits: u64 },
    InvalidActionFidelity { task_id: String, value_bits: u64 },
}

impl fmt::Display for TransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTaskId => write!(f, "task ID must not be empty"),
            Self::EmptyTaskFamilyId => write!(f, "task-family ID must not be empty"),
            Self::DuplicateTrainingTaskId { task_id } => write!(f, "duplicate training task ID: {task_id}"),
            Self::DuplicateEvaluationTaskId { task_id } => write!(f, "duplicate evaluation task ID: {task_id}"),
            Self::TrainingEvaluationTaskLeakage { task_id } => write!(f, "training/evaluation task leakage: {task_id}"),
            Self::EvaluationTaskOutsideHeldOutFamily { task_id, family_id } => {
                write!(f, "evaluation task {task_id} belongs to non-held-out family {family_id}")
            }
            Self::NoEvaluationTasks => write!(f, "at least one held-out evaluation task is required"),
            Self::NonFiniteScore { task_id, field } => write!(f, "task {task_id} has non-finite {field}"),
            Self::NonPositiveExpertScore { task_id, score_bits } => {
                write!(f, "task {task_id} has non-positive expert score {}", f64::from_bits(*score_bits))
            }
            Self::InvalidActionFidelity { task_id, value_bits } => {
                write!(f, "task {task_id} has action fidelity outside [0,1]: {}", f64::from_bits(*value_bits))
            }
        }
    }
}
impl std::error::Error for TransferError {}

/// MEDAL-ADR-inspired score for one held-out task:
/// CT = 0.5 * ((A_full - A_solo)/E + (A_half - A_solo)/E).
///
/// E must be positive: otherwise the published normalization is undefined or
/// reverses the meaning of improvement.
pub fn evaluate_task(task: &HeldOutTaskScore) -> Result<TaskTransferMetrics, TransferError> {
    if task.task_id.trim().is_empty() {
        return Err(TransferError::EmptyTaskId);
    }
    if task.task_family_id.trim().is_empty() {
        return Err(TransferError::EmptyTaskFamilyId);
    }
    for (field, value) in [
        ("expert_score", task.expert_score),
        ("agent_with_expert_full", task.agent_with_expert_full),
        ("agent_with_expert_half", task.agent_with_expert_half),
        ("agent_solo", task.agent_solo),
    ] {
        if !value.is_finite() {
            return Err(TransferError::NonFiniteScore {
                task_id: task.task_id.clone(),
                field,
            });
        }
    }
    if task.expert_score <= 0.0 {
        return Err(TransferError::NonPositiveExpertScore {
            task_id: task.task_id.clone(),
            score_bits: task.expert_score.to_bits(),
        });
    }
    if let Some(value) = task.action_fidelity {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(TransferError::InvalidActionFidelity {
                task_id: task.task_id.clone(),
                value_bits: value.to_bits(),
            });
        }
    }

    let following_gain =
        (task.agent_with_expert_full - task.agent_solo) / task.expert_score;
    let half_exposure_gain =
        (task.agent_with_expert_half - task.agent_solo) / task.expert_score;
    let half_exposure_normalized_score = task.agent_with_expert_half / task.expert_score;
    let cultural_transmission = 0.5 * (following_gain + half_exposure_gain);
    let half_exposure_gain_fraction = if following_gain > 0.0 {
        Some(half_exposure_gain / following_gain)
    } else {
        None
    };

    Ok(TaskTransferMetrics {
        following_gain,
        half_exposure_gain,
        half_exposure_normalized_score,
        cultural_transmission,
        half_exposure_gain_fraction,
        action_fidelity: task.action_fidelity,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferSummary {
    pub task_count: usize,
    pub mean_following_gain: f64,
    pub mean_half_exposure_gain: f64,
    pub mean_half_exposure_normalized_score: f64,
    pub mean_cultural_transmission: f64,
    pub mean_half_exposure_gain_fraction: Option<f64>,
    pub mean_action_fidelity: Option<f64>,
    pub cultural_transmission_sample_sd: Option<f64>,
    pub cultural_transmission_ci95_half_width: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransferEvaluation {
    pub per_task: Vec<(String, TaskTransferMetrics)>,
    pub summary: TransferSummary,
}

/// Macro-average one vote per task. When family_holdout is required, each
/// evaluation task must belong to the predeclared held-out family set.
pub fn evaluate_held_out(
    training_task_ids: &[String],
    held_out_task_family_ids: &[String],
    evaluation_tasks: &[HeldOutTaskScore],
    require_family_holdout: bool,
) -> Result<TransferEvaluation, TransferError> {
    if evaluation_tasks.is_empty() {
        return Err(TransferError::NoEvaluationTasks);
    }

    let mut training = BTreeSet::new();
    for id in training_task_ids {
        if id.trim().is_empty() {
            return Err(TransferError::EmptyTaskId);
        }
        if !training.insert(id.clone()) {
            return Err(TransferError::DuplicateTrainingTaskId { task_id: id.clone() });
        }
    }

    let mut held_families = BTreeSet::new();
    for id in held_out_task_family_ids {
        if id.trim().is_empty() {
            return Err(TransferError::EmptyTaskFamilyId);
        }
        held_families.insert(id.clone());
    }

    let mut seen_evaluation = BTreeSet::new();
    let mut per_task = Vec::with_capacity(evaluation_tasks.len());
    let mut following = Vec::with_capacity(evaluation_tasks.len());
    let mut half_exposure = Vec::with_capacity(evaluation_tasks.len());
    let mut half_exposure_normalized = Vec::with_capacity(evaluation_tasks.len());
    let mut transmission = Vec::with_capacity(evaluation_tasks.len());
    let mut retention = Vec::new();
    let mut fidelity = Vec::new();

    for task in evaluation_tasks {
        if !seen_evaluation.insert(task.task_id.clone()) {
            return Err(TransferError::DuplicateEvaluationTaskId {
                task_id: task.task_id.clone(),
            });
        }
        if training.contains(&task.task_id) {
            return Err(TransferError::TrainingEvaluationTaskLeakage {
                task_id: task.task_id.clone(),
            });
        }
        if require_family_holdout && !held_families.contains(&task.task_family_id) {
            return Err(TransferError::EvaluationTaskOutsideHeldOutFamily {
                task_id: task.task_id.clone(),
                family_id: task.task_family_id.clone(),
            });
        }

        let metrics = evaluate_task(task)?;
        following.push(metrics.following_gain);
        half_exposure.push(metrics.half_exposure_gain);
        half_exposure_normalized.push(metrics.half_exposure_normalized_score);
        transmission.push(metrics.cultural_transmission);
        if let Some(value) = metrics.half_exposure_gain_fraction {
            retention.push(value);
        }
        if let Some(value) = metrics.action_fidelity {
            fidelity.push(value);
        }
        per_task.push((task.task_id.clone(), metrics));
    }

    let mean = |values: &[f64]| -> Option<f64> {
        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        }
    };
    let mean_transmission = mean(&transmission).expect("evaluation tasks are nonempty");
    let sample_sd = if transmission.len() < 2 {
        None
    } else {
        let center = mean_transmission;
        let variance = transmission.iter()
            .map(|value| (value - center).powi(2))
            .sum::<f64>() / (transmission.len() - 1) as f64;
        Some(variance.sqrt())
    };
    let ci95 = sample_sd.map(|sd| 1.96 * sd / (transmission.len() as f64).sqrt());

    Ok(TransferEvaluation {
        per_task,
        summary: TransferSummary {
            task_count: evaluation_tasks.len(),
            mean_following_gain: mean(&following).expect("nonempty"),
            mean_half_exposure_gain: mean(&half_exposure).expect("nonempty"),
            mean_half_exposure_normalized_score: mean(&half_exposure_normalized).expect("nonempty"),
            mean_cultural_transmission: mean_transmission,
            mean_half_exposure_gain_fraction: mean(&retention),
            mean_action_fidelity: mean(&fidelity),
            cultural_transmission_sample_sd: sample_sd,
            cultural_transmission_ci95_half_width: ci95,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: &str, family: &str, expert: f64, full: f64, half: f64, solo: f64) -> HeldOutTaskScore {
        HeldOutTaskScore {
            task_id: id.to_owned(),
            task_family_id: family.to_owned(),
            expert_score: expert,
            agent_with_expert_full: full,
            agent_with_expert_half: half,
            agent_solo: solo,
            action_fidelity: Some(0.9),
        }
    }

    #[test]
    fn independent_agent_has_zero_cultural_transmission() {
        let m = evaluate_task(&task("t1", "family-a", 100.0, 20.0, 20.0, 20.0)).unwrap();
        assert_eq!(m.cultural_transmission, 0.0);
        assert_eq!(m.following_gain, 0.0);
        assert_eq!(m.half_exposure_gain, 0.0);
        assert_eq!(m.half_exposure_gain_fraction, None);
    }

    #[test]
    fn expert_following_without_retention_scores_three_quarters() {
        // The learner matches the expert while present, but earns only the
        // first-half expert reward in the half-exposure condition.
        let m = evaluate_task(&task("t2", "family-a", 100.0, 100.0, 50.0, 0.0)).unwrap();
        assert_eq!(m.following_gain, 1.0);
        assert_eq!(m.half_exposure_gain, 0.5);
        assert_eq!(m.half_exposure_normalized_score, 0.5);
        assert_eq!(m.cultural_transmission, 0.75);
        assert_eq!(m.half_exposure_gain_fraction, Some(0.5));
    }

    #[test]
    fn full_transmission_and_retention_reaches_one() {
        let m = evaluate_task(&task("t3", "family-a", 100.0, 100.0, 100.0, 0.0)).unwrap();
        assert_eq!(m.cultural_transmission, 1.0);
        assert_eq!(m.half_exposure_normalized_score, 1.0);
        assert_eq!(m.half_exposure_gain_fraction, Some(1.0));
    }

    #[test]
    fn family_holdout_and_task_id_leakage_are_checked() {
        let tasks = [task("held-1", "new-family", 100.0, 90.0, 80.0, 10.0)];
        let ok = evaluate_held_out(
            &["train-1".to_owned()],
            &["new-family".to_owned()],
            &tasks,
            true,
        ).unwrap();
        assert_eq!(ok.summary.task_count, 1);

        assert_eq!(
            evaluate_held_out(
                &["held-1".to_owned()],
                &["new-family".to_owned()],
                &tasks,
                true,
            ),
            Err(TransferError::TrainingEvaluationTaskLeakage {
                task_id: "held-1".to_owned(),
            })
        );
    }

    #[test]
    fn non_held_out_family_is_rejected_when_family_holdout_required() {
        let tasks = [task("held-1", "seen-family", 100.0, 90.0, 80.0, 10.0)];
        assert!(matches!(
            evaluate_held_out(&[], &["new-family".to_owned()], &tasks, true),
            Err(TransferError::EvaluationTaskOutsideHeldOutFamily { .. })
        ));
    }

    #[test]
    fn nonpositive_expert_score_is_rejected() {
        assert!(matches!(
            evaluate_task(&task("bad", "family-a", 0.0, 1.0, 1.0, 0.0)),
            Err(TransferError::NonPositiveExpertScore { .. })
        ));
    }

    #[test]
    fn aggregate_is_macro_averaged_and_has_interval() {
        let tasks = [
            task("a", "held", 100.0, 100.0, 100.0, 0.0),
            task("b", "held", 100.0, 50.0, 25.0, 0.0),
        ];
        let result = evaluate_held_out(&[], &["held".to_owned()], &tasks, true).unwrap();
        assert_eq!(result.summary.task_count, 2);
        assert!((result.summary.mean_cultural_transmission - 0.6875).abs() < 1e-12);
        assert!(result.summary.cultural_transmission_ci95_half_width.is_some());
    }
}
