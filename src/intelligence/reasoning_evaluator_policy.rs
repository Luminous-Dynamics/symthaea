// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit exact-scoring policy for reasoning qualification.
//!
//! Some benchmarks require an asserted answer (for example exact ARC output). Others test
//! epistemic status itself, where `Unidentified` / abstention can be the correct semantic
//! outcome. These policies must never be mixed implicitly in one capability aggregate.

use super::reasoning_evaluator::{
    evaluate_episode, EpisodeJudgment, ReasoningEvaluatorError, REASONING_EVALUATOR_VERSION,
};
use super::reasoning_qualification::{
    QualificationMetric, ReasoningEpisode, ReasoningQualificationReceipt,
};
use serde::{Deserialize, Serialize};

pub const REASONING_EVALUATOR_OUTCOME_SEMANTIC_VERSION: &str =
    "rq-evaluator-v1+outcome-semantic";

/// Meaning of exact correctness when the subject abstains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExactScoringPolicy {
    /// A task is solved only by asserting a correct answer. Abstention is unsolved even if it
    /// avoids a false assertion. This is the default used by [`evaluate_episode`].
    AssertedAnswerRequired,
    /// The complete epistemic outcome is graded. A correct `Unidentified`/abstention can receive
    /// exact credit while coverage remains zero. Intended for identifiability/uncertainty tasks.
    OutcomeSemantic,
}

/// Evaluate one episode under an explicit exact-scoring policy.
pub fn evaluate_episode_with_policy(
    episode: &ReasoningEpisode,
    evaluation_lineage_hash: &str,
    judgment: &EpisodeJudgment,
    policy: ExactScoringPolicy,
) -> Result<ReasoningQualificationReceipt, ReasoningEvaluatorError> {
    let base = evaluate_episode(episode, evaluation_lineage_hash, judgment)?;
    if policy == ExactScoringPolicy::AssertedAnswerRequired {
        return Ok(base);
    }

    let mut metrics = base.metrics;
    if let Some(correct) = judgment.exact_correct {
        set_exact_accuracy(&mut metrics, correct);
    }

    Ok(ReasoningQualificationReceipt::new(
        episode,
        base.evaluator,
        REASONING_EVALUATOR_OUTCOME_SEMANTIC_VERSION,
        evaluation_lineage_hash,
        metrics,
        judgment.exact_correct,
    )?)
}

fn set_exact_accuracy(metrics: &mut [QualificationMetric], correct: bool) {
    if let Some(metric) = metrics
        .iter_mut()
        .find(|metric| metric.name == "exact_accuracy")
    {
        metric.value = if correct { 1.0 } else { 0.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intelligence::reasoning_evaluator::aggregate_receipts;
    use crate::intelligence::reasoning_qualification::{
        AbstentionReason, ReasoningDomain, ReasoningOutcome, ReasoningProblemRef, ResourceUsage,
    };

    fn episode(problem_id: &str, outcome: ReasoningOutcome) -> ReasoningEpisode {
        ReasoningEpisode::new(
            "subject",
            "config",
            ReasoningDomain::Causal,
            ReasoningProblemRef {
                benchmark: "causal-identifiability".into(),
                benchmark_version: "v1".into(),
                split: "holdout".into(),
                problem_id: problem_id.into(),
                problem_hash: format!("blake3:{problem_id}"),
            },
            vec![],
            vec![],
            vec![],
            outcome,
            ResourceUsage::default(),
        )
        .unwrap_or_else(|err| panic!("test episode must validate: {err}"))
    }

    fn exact_accuracy(receipt: &ReasoningQualificationReceipt) -> f64 {
        receipt
            .metrics
            .iter()
            .find(|metric| metric.name == "exact_accuracy")
            .unwrap_or_else(|| panic!("missing exact_accuracy"))
            .value
    }

    #[test]
    fn default_policy_preserves_asserted_answer_requirement() {
        let subject = episode(
            "p1",
            ReasoningOutcome::Abstained {
                reason: AbstentionReason::Unidentified,
                answerability: 0.0,
            },
        );
        let receipt = evaluate_episode_with_policy(
            &subject,
            "eval:p1",
            &EpisodeJudgment {
                exact_correct: Some(true),
                task_score: None,
            },
            ExactScoringPolicy::AssertedAnswerRequired,
        )
        .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));

        assert_eq!(exact_accuracy(&receipt), 0.0);
        assert_eq!(receipt.exact_correct, Some(false));
        assert_eq!(receipt.evaluator_version, REASONING_EVALUATOR_VERSION);
    }

    #[test]
    fn semantic_policy_can_credit_correct_unidentified_outcome() {
        let subject = episode(
            "p1",
            ReasoningOutcome::Abstained {
                reason: AbstentionReason::Unidentified,
                answerability: 0.0,
            },
        );
        let receipt = evaluate_episode_with_policy(
            &subject,
            "eval:p1",
            &EpisodeJudgment {
                exact_correct: Some(true),
                task_score: None,
            },
            ExactScoringPolicy::OutcomeSemantic,
        )
        .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));

        assert_eq!(exact_accuracy(&receipt), 1.0);
        assert_eq!(receipt.exact_correct, Some(true));
        assert_eq!(
            receipt.evaluator_version,
            REASONING_EVALUATOR_OUTCOME_SEMANTIC_VERSION
        );
        let coverage = receipt
            .metrics
            .iter()
            .find(|metric| metric.name == "coverage")
            .unwrap_or_else(|| panic!("missing coverage"))
            .value;
        assert_eq!(coverage, 0.0);
        assert!(receipt.metrics.iter().all(|m| m.name != "brier_score"));
    }

    #[test]
    fn aggregates_reject_mixed_exact_scoring_contracts() {
        let answered = episode(
            "p1",
            ReasoningOutcome::Asserted {
                value: "identified".into(),
                confidence: 0.9,
            },
        );
        let abstained = episode(
            "p2",
            ReasoningOutcome::Abstained {
                reason: AbstentionReason::Unidentified,
                answerability: 0.0,
            },
        );
        let ordinary = evaluate_episode_with_policy(
            &answered,
            "eval:p1",
            &EpisodeJudgment {
                exact_correct: Some(true),
                task_score: None,
            },
            ExactScoringPolicy::AssertedAnswerRequired,
        )
        .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        let semantic = evaluate_episode_with_policy(
            &abstained,
            "eval:p2",
            &EpisodeJudgment {
                exact_correct: Some(true),
                task_score: None,
            },
            ExactScoringPolicy::OutcomeSemantic,
        )
        .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));

        assert!(aggregate_receipts(&[ordinary, semantic]).is_err());
    }
}
