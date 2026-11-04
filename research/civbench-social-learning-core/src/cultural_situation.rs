// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Cross-cultural situational evaluation metrics inspired by LiveCultureBench.
//!
//! This is an adapter-neutral metric layer. It does not implement the official
//! LiveCultureBench simulator, infer a person's culture from demographic traits,
//! or claim to reproduce ACL 2026. Context IDs and norm sets are explicit
//! scenario metadata supplied by a benchmark. Norm violation is a safety metric
//! where lower is better; it is never collapsed into task completion.

use core::fmt;
use std::collections::BTreeMap;

/// A declared cultural/environment context from the task manifest.
/// It must not be inferred from an agent's sensitive traits.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContextId(String);

impl ContextId {
    pub fn new(value: impl Into<String>) -> Result<Self, SituationError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(SituationError::EmptyContextId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Five dimensions reported separately by the LiveCultureBench paper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SituationDimension {
    GoalCompletion,
    NormViolation,
    ProfileFaithfulness,
    ContextualAwareness,
    Coherence,
}

impl SituationDimension {
    pub const ALL: [Self; 5] = [
        Self::GoalCompletion,
        Self::NormViolation,
        Self::ProfileFaithfulness,
        Self::ContextualAwareness,
        Self::Coherence,
    ];

    /// Whether larger values are preferred for this metric.
    pub fn higher_is_better(self) -> bool {
        !matches!(self, Self::NormViolation)
    }
}

/// Per-episode scores produced by a task-specific evaluator/verifier.
/// Each value must be finite and normalized to [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SituationScores {
    pub goal_completion: f64,
    /// Fraction of applicable local norms judged to have been violated.
    pub norm_violation_rate: f64,
    pub profile_faithfulness: f64,
    pub contextual_awareness: f64,
    pub coherence: f64,
}

impl SituationScores {
    pub fn validate(self) -> Result<(), SituationError> {
        for (name, value) in [
            ("goal_completion", self.goal_completion),
            ("norm_violation_rate", self.norm_violation_rate),
            ("profile_faithfulness", self.profile_faithfulness),
            ("contextual_awareness", self.contextual_awareness),
            ("coherence", self.coherence),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(SituationError::InvalidScore { name });
            }
        }
        Ok(())
    }

    pub fn value(self, dimension: SituationDimension) -> f64 {
        match dimension {
            SituationDimension::GoalCompletion => self.goal_completion,
            SituationDimension::NormViolation => self.norm_violation_rate,
            SituationDimension::ProfileFaithfulness => self.profile_faithfulness,
            SituationDimension::ContextualAwareness => self.contextual_awareness,
            SituationDimension::Coherence => self.coherence,
        }
    }
}

/// An episode record with an explicit, declared context label.
#[derive(Debug, Clone, PartialEq)]
pub struct SituationEpisode {
    pub task_id: String,
    pub context_id: ContextId,
    pub seed: u64,
    pub scores: SituationScores,
}

/// Mean and sample count; None means no observations, not zero performance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanMetric {
    pub n: usize,
    pub mean: Option<f64>,
}

impl MeanMetric {
    fn from_sum(n: usize, sum: f64) -> Self {
        Self {
            n,
            mean: if n == 0 { None } else { Some(sum / n as f64) },
        }
    }
}

/// Aggregate per declared context and across the whole evaluated population.
#[derive(Debug, Clone, PartialEq)]
pub struct SituationAggregate {
    pub episodes: usize,
    pub goal_completion: MeanMetric,
    pub norm_violation_rate: MeanMetric,
    pub profile_faithfulness: MeanMetric,
    pub contextual_awareness: MeanMetric,
    pub coherence: MeanMetric,
}

impl SituationAggregate {
    pub fn from_episodes(episodes: &[SituationEpisode]) -> Result<Self, SituationError> {
        let mut sums = [0.0_f64; 5];
        for episode in episodes {
            episode.scores.validate()?;
            for (index, dimension) in SituationDimension::ALL.iter().enumerate() {
                sums[index] += episode.scores.value(*dimension);
            }
        }
        let n = episodes.len();
        Ok(Self {
            episodes: n,
            goal_completion: MeanMetric::from_sum(n, sums[0]),
            norm_violation_rate: MeanMetric::from_sum(n, sums[1]),
            profile_faithfulness: MeanMetric::from_sum(n, sums[2]),
            contextual_awareness: MeanMetric::from_sum(n, sums[3]),
            coherence: MeanMetric::from_sum(n, sums[4]),
        })
    }

    pub fn metric(&self, dimension: SituationDimension) -> MeanMetric {
        match dimension {
            SituationDimension::GoalCompletion => self.goal_completion,
            SituationDimension::NormViolation => self.norm_violation_rate,
            SituationDimension::ProfileFaithfulness => self.profile_faithfulness,
            SituationDimension::ContextualAwareness => self.contextual_awareness,
            SituationDimension::Coherence => self.coherence,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextSituationAggregate {
    pub context_id: ContextId,
    pub aggregate: SituationAggregate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SituationReport {
    pub overall: SituationAggregate,
    pub by_context: Vec<ContextSituationAggregate>,
}

pub fn evaluate_situations(episodes: &[SituationEpisode]) -> Result<SituationReport, SituationError> {
    let overall = SituationAggregate::from_episodes(episodes)?;
    let mut grouped = BTreeMap::<ContextId, Vec<SituationEpisode>>::new();
    for episode in episodes {
        grouped.entry(episode.context_id.clone()).or_default().push(episode.clone());
    }
    let mut by_context = Vec::with_capacity(grouped.len());
    for (context_id, context_episodes) in grouped {
        by_context.push(ContextSituationAggregate {
            context_id,
            aggregate: SituationAggregate::from_episodes(&context_episodes)?,
        });
    }
    Ok(SituationReport { overall, by_context })
}

/// A verifier prediction. A missing prediction is an explicit abstention;
/// a missing human label means not audited and is never treated as negative.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifierJudgment {
    pub dimension: SituationDimension,
    pub context_id: ContextId,
    /// Probability that the binary event is present (e.g. a norm violation).
    pub probability: f64,
    pub prediction: Option<bool>,
    pub human_label: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SituationError {
    EmptyContextId,
    InvalidScore { name: &'static str },
    InvalidVerifierProbability,
    InvalidCalibrationBinCount,
}

impl fmt::Display for SituationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyContextId => write!(f, "context_id must be non-empty"),
            Self::InvalidScore { name } => write!(f, "{name} must be finite and in [0, 1]"),
            Self::InvalidVerifierProbability => write!(f, "verifier probability must be finite and in [0, 1]"),
            Self::InvalidCalibrationBinCount => write!(f, "calibration bin count must be in 1..=20"),
        }
    }
}
impl std::error::Error for SituationError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfusionMetrics {
    pub true_positive: usize,
    pub false_positive: usize,
    pub true_negative: usize,
    pub false_negative: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifierCalibration {
    pub dimension: SituationDimension,
    pub context_id: ContextId,
    pub total_predictions: usize,
    pub human_audited_predictions: usize,
    pub abstentions: usize,
    pub audit_coverage: Option<f64>,
    pub abstention_rate: Option<f64>,
    pub brier_score: Option<f64>,
    pub log_loss: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub confusion: ConfusionMetrics,
}

#[derive(Debug, Clone, Copy, Default)]
struct CalibrationSums {
    n: usize,
    brier: f64,
    log_loss: f64,
    bin_confidence: [f64; 20],
    bin_positive_rate: [f64; 20],
    bin_count: [usize; 20],
    tp: usize,
    fp: usize,
    tn: usize,
    fn_: usize,
}

impl CalibrationSums {
    fn observe(&mut self, prediction: &VerifierJudgment, bins: usize) {
        let Some(label) = prediction.human_label else {
            return;
        };
        self.n += 1;
        let target = if label { 1.0 } else { 0.0 };
        let probability = prediction.probability.clamp(1e-15, 1.0 - 1e-15);
        self.brier += (probability - target).powi(2);
        self.log_loss += -(target * probability.ln() + (1.0 - target) * (1.0 - probability).ln());

        let bin = ((prediction.probability * bins as f64).floor() as usize).min(bins - 1);
        self.bin_confidence[bin] += prediction.probability;
        self.bin_positive_rate[bin] += target;
        self.bin_count[bin] += 1;

        match (prediction.prediction, label) {
            (Some(true), true) => self.tp += 1,
            (Some(true), false) => self.fp += 1,
            (Some(false), false) => self.tn += 1,
            (Some(false), true) => self.fn_ += 1,
            (None, _) => {}
        }
    }

    fn calibration_error(&self, bins: usize) -> Option<f64> {
        if self.n == 0 {
            return None;
        }
        let mut error = 0.0;
        for index in 0..bins {
            let count = self.bin_count[index];
            if count > 0 {
                let confidence = self.bin_confidence[index] / count as f64;
                let empirical = self.bin_positive_rate[index] / count as f64;
                error += count as f64 / self.n as f64 * (confidence - empirical).abs();
            }
        }
        Some(error)
    }

    fn confusion(&self) -> ConfusionMetrics {
        let precision = ratio(self.tp, self.tp + self.fp);
        let recall = ratio(self.tp, self.tp + self.fn_);
        let f1 = match (precision, recall) {
            (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
            (Some(_), Some(_)) => Some(0.0),
            _ => None,
        };
        ConfusionMetrics {
            true_positive: self.tp,
            false_positive: self.fp,
            true_negative: self.tn,
            false_negative: self.fn_,
            precision,
            recall,
            f1,
        }
    }
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 { None } else { Some(numerator as f64 / denominator as f64) }
}

/// Score verifier quality by task dimension and declared cultural context.
/// ECE uses fixed-width bins. It is descriptive and sample-sensitive; reports
/// should also include audit sampling design and confidence intervals.
pub fn evaluate_verifier(
    judgments: &[VerifierJudgment],
    calibration_bins: usize,
) -> Result<Vec<VerifierCalibration>, SituationError> {
    if calibration_bins == 0 || calibration_bins > 20 {
        return Err(SituationError::InvalidCalibrationBinCount);
    }
    let mut grouped = BTreeMap::<(SituationDimension, ContextId), CalibrationSums>::new();
    let mut totals = BTreeMap::<(SituationDimension, ContextId), (usize, usize)>::new();

    for judgment in judgments {
        if !judgment.probability.is_finite() || !(0.0..=1.0).contains(&judgment.probability) {
            return Err(SituationError::InvalidVerifierProbability);
        }
        let key = (judgment.dimension, judgment.context_id.clone());
        let total = totals.entry(key.clone()).or_insert((0, 0));
        total.0 += 1;
        if judgment.prediction.is_none() {
            total.1 += 1;
        }
        grouped.entry(key).or_default().observe(judgment, calibration_bins);
    }

    let mut report = Vec::with_capacity(grouped.len());
    for ((dimension, context_id), sums) in grouped {
        let (total_predictions, abstentions) = totals
            .get(&(dimension, context_id.clone()))
            .copied()
            .unwrap_or_default();
        report.push(VerifierCalibration {
            dimension,
            context_id,
            total_predictions,
            human_audited_predictions: sums.n,
            abstentions,
            audit_coverage: ratio(sums.n, total_predictions),
            abstention_rate: ratio(abstentions, total_predictions),
            brier_score: ratio_float(sums.brier, sums.n),
            log_loss: ratio_float(sums.log_loss, sums.n),
            expected_calibration_error: sums.calibration_error(calibration_bins),
            confusion: sums.confusion(),
        });
    }
    Ok(report)
}

fn ratio_float(numerator: f64, denominator: usize) -> Option<f64> {
    if denominator == 0 { None } else { Some(numerator / denominator as f64) }
}

/// Pareto dominance: left is at least as good on every dimension and
/// strictly better on at least one. Norm violation is minimized; all other
/// dimensions are maximized. No scalar composite hides task/norm trade-offs.
pub fn pareto_dominates(left: SituationScores, right: SituationScores) -> Result<bool, SituationError> {
    left.validate()?;
    right.validate()?;

    let comparisons = [
        (left.goal_completion, right.goal_completion, true),
        (left.norm_violation_rate, right.norm_violation_rate, false),
        (left.profile_faithfulness, right.profile_faithfulness, true),
        (left.contextual_awareness, right.contextual_awareness, true),
        (left.coherence, right.coherence, true),
    ];
    let mut strictly_better = false;
    for (left_value, right_value, higher_is_better) in comparisons {
        if higher_is_better {
            if left_value < right_value { return Ok(false); }
            strictly_better |= left_value > right_value;
        } else {
            if left_value > right_value { return Ok(false); }
            strictly_better |= left_value < right_value;
        }
    }
    Ok(strictly_better)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(value: &str) -> ContextId {
        ContextId::new(value).unwrap()
    }

    fn scores(goal: f64, norm: f64, profile: f64, context: f64, coherence: f64) -> SituationScores {
        SituationScores {
            goal_completion: goal,
            norm_violation_rate: norm,
            profile_faithfulness: profile,
            contextual_awareness: context,
            coherence,
        }
    }

    #[test]
    fn episode_metrics_are_aggregated_by_explicit_context() {
        let episodes = vec![
            SituationEpisode {
                task_id: "task-1".to_owned(),
                context_id: context("declared-context-a"),
                seed: 1,
                scores: scores(1.0, 0.0, 0.8, 1.0, 0.9),
            },
            SituationEpisode {
                task_id: "task-2".to_owned(),
                context_id: context("declared-context-a"),
                seed: 2,
                scores: scores(0.5, 0.2, 0.6, 0.5, 0.7),
            },
            SituationEpisode {
                task_id: "task-3".to_owned(),
                context_id: context("declared-context-b"),
                seed: 3,
                scores: scores(0.0, 1.0, 0.4, 0.0, 0.3),
            },
        ];
        let report = evaluate_situations(&episodes).unwrap();
        assert_eq!(report.overall.episodes, 3);
        assert_eq!(report.overall.goal_completion.mean, Some(0.5));
        assert_eq!(report.by_context.len(), 2);
        assert_eq!(report.by_context[0].context_id.as_str(), "declared-context-a");
        assert_eq!(report.by_context[0].aggregate.episodes, 2);
    }

    #[test]
    fn norm_violation_is_not_collapsed_into_goal_completion() {
        let high_task_high_violation = scores(1.0, 0.9, 0.8, 1.0, 1.0);
        let lower_task_low_violation = scores(0.8, 0.1, 0.8, 1.0, 1.0);
        assert!(!pareto_dominates(high_task_high_violation, lower_task_low_violation).unwrap());
        assert!(!pareto_dominates(lower_task_low_violation, high_task_high_violation).unwrap());
    }

    #[test]
    fn pareto_dominance_minimizes_norm_violations_and_maximizes_other_metrics() {
        let better = scores(0.9, 0.1, 0.9, 0.9, 0.9);
        let worse = scores(0.8, 0.2, 0.8, 0.8, 0.8);
        assert!(pareto_dominates(better, worse).unwrap());
        assert!(!pareto_dominates(worse, better).unwrap());
        assert!(!pareto_dominates(better, better).unwrap());
    }

    #[test]
    fn verifier_reports_per_context_calibration_and_abstention() {
        let judgments = vec![
            VerifierJudgment {
                dimension: SituationDimension::NormViolation,
                context_id: context("context-a"),
                probability: 0.9,
                prediction: Some(true),
                human_label: Some(true),
            },
            VerifierJudgment {
                dimension: SituationDimension::NormViolation,
                context_id: context("context-a"),
                probability: 0.1,
                prediction: Some(false),
                human_label: Some(false),
            },
            VerifierJudgment {
                dimension: SituationDimension::NormViolation,
                context_id: context("context-a"),
                probability: 0.8,
                prediction: None,
                human_label: Some(true),
            },
            VerifierJudgment {
                dimension: SituationDimension::NormViolation,
                context_id: context("context-b"),
                probability: 0.9,
                prediction: Some(true),
                human_label: Some(false),
            },
            VerifierJudgment {
                dimension: SituationDimension::NormViolation,
                context_id: context("context-b"),
                probability: 0.9,
                prediction: Some(true),
                human_label: None,
            },
        ];
        let report = evaluate_verifier(&judgments, 10).unwrap();
        assert_eq!(report.len(), 2);
        let a = &report[0];
        assert_eq!(a.human_audited_predictions, 3);
        assert_eq!(a.abstentions, 1);
        assert_eq!(a.audit_coverage, Some(1.0));
        assert_eq!(a.confusion.true_positive, 1);
        assert_eq!(a.confusion.true_negative, 1);

        let b = &report[1];
        assert_eq!(b.total_predictions, 2);
        assert_eq!(b.human_audited_predictions, 1);
        assert_eq!(b.audit_coverage, Some(0.5));
        assert_eq!(b.confusion.false_positive, 1);
    }

    #[test]
    fn verifier_invalid_probability_fails_closed() {
        let judgments = vec![VerifierJudgment {
            dimension: SituationDimension::GoalCompletion,
            context_id: context("context-a"),
            probability: f64::NAN,
            prediction: Some(true),
            human_label: Some(true),
        }];
        assert_eq!(
            evaluate_verifier(&judgments, 10),
            Err(SituationError::InvalidVerifierProbability)
        );
    }

    #[test]
    fn invalid_context_and_episode_scores_are_rejected() {
        assert_eq!(ContextId::new(" "), Err(SituationError::EmptyContextId));
        let invalid = SituationScores {
            goal_completion: 1.1,
            norm_violation_rate: 0.0,
            profile_faithfulness: 1.0,
            contextual_awareness: 1.0,
            coherence: 1.0,
        };
        assert_eq!(invalid.validate(), Err(SituationError::InvalidScore { name: "goal_completion" }));
    }

    #[test]
    fn empty_data_is_not_reported_as_zero_performance() {
        let aggregate = SituationAggregate::from_episodes(&[]).unwrap();
        assert_eq!(aggregate.episodes, 0);
        assert_eq!(aggregate.goal_completion.mean, None);
        assert_eq!(aggregate.norm_violation_rate.mean, None);
        assert!(evaluate_verifier(&[], 10).unwrap().is_empty());
    }
}
