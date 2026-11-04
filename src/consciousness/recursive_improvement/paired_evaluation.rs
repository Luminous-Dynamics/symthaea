// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! In-memory paired-analysis adapter for recursive-improvement candidates.
//!
//! This module is not a persisted receipt format. It projects task-level rows
//! from the shared benchmark-run receipt contract into a deterministic,
//! matched per-case comparison. Integration must first validate the shared
//! receipt and its attestation/regrade using the common benchmark evidence
//! pipeline (see Symthaea issues #10314, #10167, and #10370).
//!
//! Candidate summaries are derived from sorted per-case rows; callers cannot
//! supply an unrelated aggregate score. The archive's receipt digest should
//! identify the canonical shared benchmark receipt, not this in-memory view.
//!
//! The paired sign test is a diagnostic for directional consistency. It is
//! not a promotion rule, does not establish practical significance, and does
//! not correct for repeated candidate selection or multiple metrics.
//!
//! The sign-test p-value assumes independent non-tied case-pair signs. When
//! benchmark cases cluster by repository, task family, project, or run, case-level
//! p-values are descriptive only; confirmation must use independent groups or a
//! cluster-aware analysis in the qualification layer.
//!
//! This adapter does not authenticate input rows, evaluator identity, or
//! benchmark isolation. A digest is an identity commitment, not a signature
//! or proof.

use super::candidate_archive::{
    CandidateArchiveError, CandidateRecord, CapabilityVector, EvaluationContext,
};
use std::collections::BTreeMap;
use std::fmt;

/// One candidate's normalized capability scores for one benchmark case.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseScore {
    case_id: String,
    scores: CapabilityVector,
}

impl CaseScore {
    pub fn new(
        case_id: impl Into<String>,
        scores: CapabilityVector,
    ) -> Result<Self, PairedEvaluationError> {
        let case_id = case_id.into();
        if case_id.trim().is_empty() || case_id.chars().any(char::is_control) {
            return Err(PairedEvaluationError::InvalidCaseId);
        }
        Ok(Self { case_id, scores })
    }

    pub fn case_id(&self) -> &str {
        &self.case_id
    }

    pub fn scores(&self) -> &CapabilityVector {
        &self.scores
    }
}

/// Errors while constructing or comparing paired evaluation receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairedEvaluationError {
    InvalidCaseId,
    EmptyCaseSet,
    DuplicateCase(String),
    MetricSchemaMismatch,
    ContextMismatch,
    CaseSetMismatch,
    ParentLineageMismatch,
    SameCandidate,
    InvalidTieEpsilon,
    CandidateArchive(CandidateArchiveError),
}

impl fmt::Display for PairedEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCaseId => write!(f, "case ID is empty or contains control characters"),
            Self::EmptyCaseSet => write!(f, "evaluation case set must not be empty"),
            Self::DuplicateCase(id) => write!(f, "duplicate evaluation case ID: {id}"),
            Self::MetricSchemaMismatch => {
                write!(f, "case metric dimensions do not match the frozen metric schema")
            }
            Self::ContextMismatch => {
                write!(f, "candidate receipts do not share an identical evaluation context")
            }
            Self::CaseSetMismatch => {
                write!(f, "candidate identity is inconsistent with its receipt")
            }
            Self::ParentLineageMismatch => {
                write!(f, "successor receipt does not identify the evaluated parent")
            }
            Self::SameCandidate => write!(f, "candidate cannot be compared against itself"),
            Self::InvalidTieEpsilon => {
                write!(f, "tie epsilon must be finite and within [0, 1]")
            }
            Self::CandidateArchive(error) => write!(f, "candidate record: {error}"),
        }
    }
}

impl std::error::Error for PairedEvaluationError {}

impl From<CandidateArchiveError> for PairedEvaluationError {
    fn from(value: CandidateArchiveError) -> Self {
        Self::CandidateArchive(value)
    }
}

/// In-memory projection of one candidate's task-level outcomes.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateEvaluationView {
    context: EvaluationContext,
    candidate: CandidateRecord,
    cases: BTreeMap<String, CaseScore>,
}

impl CandidateEvaluationView {
    /// Construct an in-memory view from task-level rows that have already
    /// passed the shared benchmark-receipt and attestation/regrade validators.
    ///
    /// The supplied digest must identify that shared canonical run receipt.
    /// This constructor does not verify it, authenticate it, or prove that the
    /// input rows came from the referenced artifact.
    pub fn new(
        context: EvaluationContext,
        candidate_id: impl Into<String>,
        parent_ids: Vec<String>,
        source_tree_digest: impl Into<String>,
        receipt_digest: impl Into<String>,
        cases: impl IntoIterator<Item = CaseScore>,
    ) -> Result<Self, PairedEvaluationError> {
        let mut indexed = BTreeMap::new();
        let mut expected_metrics: Option<Vec<String>> = None;

        for case in cases {
            let case_id = case.case_id.clone();
            let metric_ids: Vec<String> = case
                .scores
                .metric_ids()
                .map(str::to_owned)
                .collect();

            if let Some(expected) = &expected_metrics {
                if expected != &metric_ids {
                    return Err(PairedEvaluationError::MetricSchemaMismatch);
                }
            } else {
                expected_metrics = Some(metric_ids);
            }

            if indexed.insert(case_id.clone(), case).is_some() {
                return Err(PairedEvaluationError::DuplicateCase(case_id));
            }
        }

        if indexed.is_empty() {
            return Err(PairedEvaluationError::EmptyCaseSet);
        }

        let metric_ids = expected_metrics.ok_or(PairedEvaluationError::EmptyCaseSet)?;
        let case_count = indexed.len() as f64;
        let mut aggregate = Vec::with_capacity(metric_ids.len());

        for metric_id in metric_ids {
            let sum = indexed
                .values()
                .map(|case| {
                    case.scores
                        .get(&metric_id)
                        .expect("case metric schema was checked during insertion")
                })
                .sum::<f64>();
            aggregate.push((metric_id, sum / case_count));
        }

        let capabilities = CapabilityVector::new(aggregate)?;
        let candidate = CandidateRecord::new(
            candidate_id,
            parent_ids,
            source_tree_digest,
            receipt_digest,
            capabilities,
        )?;

        Ok(Self {
            context,
            candidate,
            cases: indexed,
        })
    }

    pub fn candidate(&self) -> &CandidateRecord {
        &self.candidate
    }

    pub fn context(&self) -> &EvaluationContext {
        &self.context
    }

    pub fn case_count(&self) -> usize {
        self.cases.len()
    }

    pub fn case_ids(&self) -> impl Iterator<Item = &str> {
        self.cases.keys().map(String::as_str)
    }

    pub fn case(&self, case_id: &str) -> Option<&CaseScore> {
        self.cases.get(case_id)
    }
}

/// Per-metric paired comparison. A positive delta means the successor scored
/// higher than its parent. Ties are excluded from the sign-test denominator.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricPairedResult {
    metric_id: String,
    wins: usize,
    losses: usize,
    ties: usize,
    mean_delta: f64,
    sign_test_p_value: Option<f64>,
}

impl MetricPairedResult {
    pub fn metric_id(&self) -> &str {
        &self.metric_id
    }

    pub fn wins(&self) -> usize {
        self.wins
    }

    pub fn losses(&self) -> usize {
        self.losses
    }

    pub fn ties(&self) -> usize {
        self.ties
    }

    pub fn mean_delta(&self) -> f64 {
        self.mean_delta
    }

    /// Exact one-sided sign-test tail probability under independent pair signs; None if every pair ties.
    pub fn sign_test_p_value(&self) -> Option<f64> {
        self.sign_test_p_value
    }
}

/// Paired comparison between a candidate and its declared direct parent.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedComparison {
    parent_candidate_id: String,
    candidate_id: String,
    case_count: usize,
    metrics: BTreeMap<String, MetricPairedResult>,
}

impl PairedComparison {
    pub fn compare(
        parent: &CandidateEvaluationView,
        candidate: &CandidateEvaluationView,
        tie_epsilon: f64,
    ) -> Result<Self, PairedEvaluationError> {
        if !tie_epsilon.is_finite() || !(0.0..=1.0).contains(&tie_epsilon) {
            return Err(PairedEvaluationError::InvalidTieEpsilon);
        }
        if parent.context != candidate.context {
            return Err(PairedEvaluationError::ContextMismatch);
        }

        let parent_id = parent.candidate.candidate_id();
        let candidate_id = candidate.candidate.candidate_id();

        if parent_id == candidate_id {
            return Err(PairedEvaluationError::SameCandidate);
        }
        if !candidate.candidate.has_parent(parent_id) {
            return Err(PairedEvaluationError::ParentLineageMismatch);
        }

        if parent.cases.len() != candidate.cases.len()
            || parent.cases.keys().ne(candidate.cases.keys())
        {
            return Err(PairedEvaluationError::CaseSetMismatch);
        }

        let parent_metrics: Vec<&str> =
            parent.candidate.capabilities().metric_ids().collect();
        let candidate_metrics: Vec<&str> =
            candidate.candidate.capabilities().metric_ids().collect();

        if parent_metrics != candidate_metrics {
            return Err(PairedEvaluationError::MetricSchemaMismatch);
        }

        let case_count = parent.cases.len();
        let mut metrics = BTreeMap::new();

        for metric_id in parent_metrics {
            let mut wins = 0usize;
            let mut losses = 0usize;
            let mut ties = 0usize;
            let mut delta_sum = 0.0f64;

            for (case_id, parent_case) in &parent.cases {
                let candidate_case = candidate
                    .cases
                    .get(case_id)
                    .expect("case set equality was checked before metric comparison");
                let parent_score = parent_case
                    .scores
                    .get(metric_id)
                    .expect("receipt enforces metric schema equality");
                let candidate_score = candidate_case
                    .scores
                    .get(metric_id)
                    .expect("receipt enforces metric schema equality");
                let delta = candidate_score - parent_score;
                delta_sum += delta;

                if delta > tie_epsilon {
                    wins += 1;
                } else if delta < -tie_epsilon {
                    losses += 1;
                } else {
                    ties += 1;
                }
            }

            let sign_test_p_value = exact_sign_test_upper_tail(wins, losses);
            metrics.insert(
                metric_id.to_owned(),
                MetricPairedResult {
                    metric_id: metric_id.to_owned(),
                    wins,
                    losses,
                    ties,
                    mean_delta: delta_sum / case_count as f64,
                    sign_test_p_value,
                },
            );
        }

        Ok(Self {
            parent_candidate_id: parent_id.to_owned(),
            candidate_id: candidate_id.to_owned(),
            case_count,
            metrics,
        })
    }

    pub fn parent_candidate_id(&self) -> &str {
        &self.parent_candidate_id
    }

    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn case_count(&self) -> usize {
        self.case_count
    }

    pub fn metric(&self, metric_id: &str) -> Option<&MetricPairedResult> {
        self.metrics.get(metric_id)
    }

    pub fn metrics(&self) -> impl Iterator<Item = (&str, &MetricPairedResult)> {
        self.metrics
            .iter()
            .map(|(metric_id, result)| (metric_id.as_str(), result))
    }
}

/// Exact one-sided binomial tail P[X >= wins], X ~ Binomial(wins + losses, 0.5).
/// Ties are excluded. This assumes independent non-tied case-pair signs; clustered
/// cases require a group-level or cluster-aware confirmation outside this helper.
/// The recurrence is evaluated relative to its boundary
/// term, avoiding direct integer binomial coefficients and their overflow.
fn exact_sign_test_upper_tail(wins: usize, losses: usize) -> Option<f64> {
    let n = wins.checked_add(losses)?;
    if n == 0 {
        return None;
    }
    if wins == 0 {
        return Some(1.0);
    }

    let log_probability_at = |k: usize| -> f64 {
        let terms = k.min(n - k);
        let mut log_choose = 0.0f64;
        for i in 1..=terms {
            log_choose += ((n - terms + i) as f64).ln() - (i as f64).ln();
        }
        log_choose - (n as f64) * std::f64::consts::LN_2
    };

    if wins > n / 2 {
        // Sum the upper tail from P(X = wins), then scale by its log probability.
        let mut relative_term = 1.0f64;
        let mut relative_sum = 1.0f64;
        for k in wins..n {
            relative_term *= (n - k) as f64 / (k + 1) as f64;
            relative_sum += relative_term;
        }
        Some((log_probability_at(wins).exp() * relative_sum).clamp(0.0, 1.0))
    } else {
        // The requested tail is at least one half. Compute its complement
        // from P(X = wins - 1), which is in the lower half of the distribution.
        let boundary = wins - 1;
        let mut relative_term = 1.0f64;
        let mut relative_sum = 1.0f64;
        for k in (1..=boundary).rev() {
            relative_term *= k as f64 / (n - k + 1) as f64;
            relative_sum += relative_term;
        }
        let lower_tail = log_probability_at(boundary).exp() * relative_sum;
        Some((1.0 - lower_tail).clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::candidate_archive::EvaluationPartition;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn context(partition: EvaluationPartition, split: &str) -> EvaluationContext {
        EvaluationContext::new(
            "sym-rsi-paired-v1",
            "v1",
            partition,
            split,
            digest('a'),
            digest('b'),
            digest('c'),
            digest('d'),
            digest('e'),
        )
        .unwrap()
    }

    fn score(case_id: &str, coding: f64, retention: f64) -> CaseScore {
        CaseScore::new(
            case_id,
            CapabilityVector::new(vec![
                ("coding".into(), coding),
                ("retention".into(), retention),
            ])
            .unwrap(),
        )
        .unwrap()
    }

    fn receipt(
        context: EvaluationContext,
        candidate_id: &str,
        parent_id: Option<&str>,
        source_byte: char,
        receipt_byte: char,
        cases: Vec<CaseScore>,
    ) -> CandidateEvaluationView {
        CandidateEvaluationView::new(
            context,
            candidate_id,
            parent_id.map(str::to_owned).into_iter().collect(),
            digest(source_byte),
            digest(receipt_byte),
            cases,
        )
        .unwrap()
    }

    #[test]
    fn derives_candidate_summary_and_compares_matched_cases() {
        let ctx = context(EvaluationPartition::Validation, "split-a");
        let parent = receipt(
            ctx.clone(),
            "parent",
            None,
            '1',
            'a',
            vec![
                score("case-1", 0.5, 0.9),
                score("case-2", 0.6, 0.8),
                score("case-3", 0.7, 0.9),
                score("case-4", 0.8, 0.8),
                score("case-5", 0.6, 0.9),
            ],
        );
        let candidate = receipt(
            ctx,
            "candidate",
            Some("parent"),
            '2',
            'b',
            vec![
                score("case-1", 0.6, 0.9),
                score("case-2", 0.7, 0.8),
                score("case-3", 0.8, 0.9),
                score("case-4", 0.9, 0.8),
                score("case-5", 0.7, 0.9),
            ],
        );

        assert_eq!(parent.candidate().capabilities().get("coding"), Some(0.64));
        let comparison = PairedComparison::compare(&parent, &candidate, 0.0).unwrap();
        assert_eq!(comparison.case_count(), 5);
        let coding = comparison.metric("coding").unwrap();
        assert_eq!(coding.wins(), 5);
        assert_eq!(coding.losses(), 0);
        assert_eq!(coding.ties(), 0);
        assert!((coding.mean_delta() - 0.1).abs() < 1e-12);
        assert_eq!(coding.sign_test_p_value(), Some(0.03125));
        let retention = comparison.metric("retention").unwrap();
        assert_eq!(retention.wins(), 0);
        assert_eq!(retention.losses(), 0);
        assert_eq!(retention.ties(), 5);
        assert_eq!(retention.sign_test_p_value(), None);
    }

    #[test]
    fn sign_test_known_fixtures() {
        assert_eq!(exact_sign_test_upper_tail(5, 0), Some(0.03125));
        assert_eq!(exact_sign_test_upper_tail(4, 1), Some(0.1875));
        assert_eq!(exact_sign_test_upper_tail(3, 2), Some(0.5));
        assert_eq!(exact_sign_test_upper_tail(0, 5), Some(1.0));
        assert_eq!(exact_sign_test_upper_tail(0, 0), None);
    }

    #[test]
    fn rejects_duplicate_and_empty_case_sets() {
        let ctx = context(EvaluationPartition::Development, "dev");
        assert_eq!(
            CandidateEvaluationView::new(
                ctx.clone(),
                "candidate",
                None,
                digest('1'),
                digest('a'),
                vec![score("same", 0.5, 0.5), score("same", 0.6, 0.6)],
            )
            .unwrap_err(),
            PairedEvaluationError::DuplicateCase("same".into())
        );
        assert_eq!(
            CandidateEvaluationView::new(
                ctx,
                "candidate",
                None,
                digest('1'),
                digest('a'),
                Vec::new(),
            )
            .unwrap_err(),
            PairedEvaluationError::EmptyCaseSet
        );
    }

    #[test]
    fn rejects_metric_schema_drift_within_one_receipt() {
        let ctx = context(EvaluationPartition::Development, "dev");
        let result = CandidateEvaluationView::new(
            ctx,
            "candidate",
            None,
            digest('1'),
            digest('a'),
            vec![
                score("case-a", 0.5, 0.5),
                CaseScore::new(
                    "case-b",
                    CapabilityVector::new(vec![("coding".into(), 0.6)]).unwrap(),
                )
                .unwrap(),
            ],
        );
        assert_eq!(result.unwrap_err(), PairedEvaluationError::MetricSchemaMismatch);
    }

    #[test]
    fn rejects_missing_or_extra_case_ids() {
        let ctx = context(EvaluationPartition::Validation, "split");
        let parent = receipt(
            ctx.clone(),
            "parent",
            None,
            '1',
            'a',
            vec![score("a", 0.5, 0.5), score("b", 0.5, 0.5)],
        );
        let child = receipt(
            ctx,
            "child",
            Some("parent"),
            '2',
            'b',
            vec![score("a", 0.6, 0.5), score("c", 0.6, 0.5)],
        );

        assert_eq!(
            PairedComparison::compare(&parent, &child, 0.0),
            Err(PairedEvaluationError::CaseSetMismatch)
        );
    }

    #[test]
    fn rejects_context_and_parent_mismatch() {
        let parent = receipt(
            context(EvaluationPartition::Validation, "split-a"),
            "parent",
            None,
            '1',
            'a',
            vec![score("a", 0.5, 0.5)],
        );
        let wrong_context = receipt(
            context(EvaluationPartition::Validation, "split-b"),
            "child",
            Some("parent"),
            '2',
            'b',
            vec![score("a", 0.6, 0.5)],
        );
        assert_eq!(
            PairedComparison::compare(&parent, &wrong_context, 0.0),
            Err(PairedEvaluationError::ContextMismatch)
        );

        let wrong_parent = receipt(
            parent.context().clone(),
            "child",
            Some("other"),
            '3',
            'c',
            vec![score("a", 0.6, 0.5)],
        );
        assert_eq!(
            PairedComparison::compare(&parent, &wrong_parent, 0.0),
            Err(PairedEvaluationError::ParentLineageMismatch)
        );
    }

    #[test]
    fn rejects_metric_schema_and_tie_epsilon_mismatch() {
        let ctx = context(EvaluationPartition::Validation, "split");
        let parent = receipt(
            ctx.clone(),
            "parent",
            None,
            '1',
            'a',
            vec![score("a", 0.5, 0.5)],
        );
        let child = CandidateEvaluationView::new(
            ctx,
            "child",
            Some("parent".into()),
            digest('2'),
            digest('b'),
            vec![CaseScore::new(
                "a",
                CapabilityVector::new(vec![("coding".into(), 0.6)]).unwrap(),
            )
            .unwrap()],
        )
        .unwrap();
        assert_eq!(
            PairedComparison::compare(&parent, &child, 0.0),
            Err(PairedEvaluationError::MetricSchemaMismatch)
        );
        assert_eq!(
            PairedComparison::compare(&parent, &parent, f64::NAN),
            Err(PairedEvaluationError::InvalidTieEpsilon)
        );
    }

    #[test]
    fn compares_multi_parent_hybrid_against_each_declared_parent() {
        let ctx = context(EvaluationPartition::Validation, "hybrid-split");
        let parent_left = receipt(
            ctx.clone(),
            "parent-left",
            None,
            '1',
            'a',
            vec![score("case-1", 0.5, 0.8), score("case-2", 0.6, 0.8)],
        );
        let parent_right = receipt(
            ctx.clone(),
            "parent-right",
            None,
            '2',
            'b',
            vec![score("case-1", 0.7, 0.9), score("case-2", 0.7, 0.9)],
        );
        let hybrid = CandidateEvaluationView::new(
            ctx,
            "hybrid",
            vec!["parent-left".into(), "parent-right".into()],
            digest('3'),
            digest('c'),
            vec![score("case-1", 0.8, 0.9), score("case-2", 0.9, 0.9)],
        )
        .unwrap();

        let left_comparison = PairedComparison::compare(&parent_left, &hybrid, 0.0).unwrap();
        let right_comparison = PairedComparison::compare(&parent_right, &hybrid, 0.0).unwrap();
        assert_eq!(left_comparison.parent_candidate_id(), "parent-left");
        assert_eq!(right_comparison.parent_candidate_id(), "parent-right");
        assert_eq!(left_comparison.metric("coding").unwrap().wins(), 2);
        assert_eq!(right_comparison.metric("coding").unwrap().wins(), 2);

        let unrelated = receipt(
            hybrid.context().clone(),
            "unrelated",
            None,
            '4',
            'd',
            vec![score("case-1", 0.8, 0.9), score("case-2", 0.9, 0.9)],
        );
        assert_eq!(
            PairedComparison::compare(&unrelated, &hybrid, 0.0),
            Err(PairedEvaluationError::ParentLineageMismatch)
        );
    }

    #[test]
    fn rejects_invalid_case_ids_and_non_finite_scores() {
        assert_eq!(
            CaseScore::new(
                "",
                CapabilityVector::new(vec![("coding".into(), 0.5)]).unwrap()
            ),
            Err(PairedEvaluationError::InvalidCaseId)
        );
        assert!(CapabilityVector::new(vec![("coding".into(), f64::INFINITY)]).is_err());
    }
}
