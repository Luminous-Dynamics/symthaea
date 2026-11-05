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
use super::paired_sign_test::exact_sign_test_upper_tail;
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


/// Errors from the cluster-aware comparison view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusteredEvaluationError {
    Paired(PairedEvaluationError),
    ClusterAssignmentMismatch,
    InvalidClusterId,
}

impl fmt::Display for ClusteredEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Paired(error) => write!(f, "paired comparison: {error}"),
            Self::ClusterAssignmentMismatch => {
                write!(f, "cluster assignments must cover exactly the matched case set")
            }
            Self::InvalidClusterId => {
                write!(f, "cluster IDs must be non-empty and contain no control characters")
            }
        }
    }
}

impl std::error::Error for ClusteredEvaluationError {}

impl From<PairedEvaluationError> for ClusteredEvaluationError {
    fn from(value: PairedEvaluationError) -> Self {
        Self::Paired(value)
    }
}

/// Per-metric comparison after aggregating cases into declared independent
/// clusters. Cluster means receive equal weight; the case-weighted mean is
/// retained separately so the estimand cannot be mistaken.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusteredMetricPairedResult {
    metric_id: String,
    case_count: usize,
    cluster_count: usize,
    cluster_wins: usize,
    cluster_losses: usize,
    cluster_ties: usize,
    case_weighted_mean_delta: f64,
    equal_cluster_mean_delta: f64,
    cluster_sign_test_p_value: Option<f64>,
    holm_adjusted_p_value: Option<f64>,
}

impl ClusteredMetricPairedResult {
    pub fn metric_id(&self) -> &str {
        &self.metric_id
    }

    pub fn case_count(&self) -> usize {
        self.case_count
    }

    pub fn cluster_count(&self) -> usize {
        self.cluster_count
    }

    pub fn cluster_wins(&self) -> usize {
        self.cluster_wins
    }

    pub fn cluster_losses(&self) -> usize {
        self.cluster_losses
    }

    pub fn cluster_ties(&self) -> usize {
        self.cluster_ties
    }

    /// Pooled per-case mean; larger clusters contribute more cases.
    pub fn case_weighted_mean_delta(&self) -> f64 {
        self.case_weighted_mean_delta
    }

    /// Mean of cluster means; each declared independent cluster has equal weight.
    pub fn equal_cluster_mean_delta(&self) -> f64 {
        self.equal_cluster_mean_delta
    }

    /// Exact one-sided sign-test diagnostic across non-tied cluster means.
    /// None is returned when fewer than two non-tied clusters remain.
    pub fn cluster_sign_test_p_value(&self) -> Option<f64> {
        self.cluster_sign_test_p_value
    }

    /// Holm step-down adjusted p-value across the non-empty metric family
    /// in this one comparison. This does not correct across candidates,
    /// generations, repeated runs, or adaptive selection.
    pub fn holm_adjusted_p_value(&self) -> Option<f64> {
        self.holm_adjusted_p_value
    }
}

/// Cluster-aware matched comparison. This is an in-memory statistical view;
/// it is not a receipt format, evaluator, authenticator, or promotion rule.
///
/// The caller-supplied cluster map must be derived from the frozen benchmark
/// manifest and be bound by the shared corpus/receipt identity. This type
/// validates exact coverage and ID shape, but cannot authenticate the map or
/// establish that the declared clusters are genuinely independent.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusteredPairedComparison {
    parent_candidate_id: String,
    candidate_id: String,
    case_count: usize,
    cluster_count: usize,
    metrics: BTreeMap<String, ClusteredMetricPairedResult>,
}

impl ClusteredPairedComparison {
    /// Compare matched cases after aggregating deltas by a preregistered
    /// independent unit such as repository, project, task family, or run.
    ///
    /// Cases within a cluster are first averaged; each cluster then contributes
    /// one sign to the exact sign test. The map must cover the compared case
    /// set exactly. A p-value is withheld when fewer than two non-tied clusters
    /// remain. Holm correction is applied across the metric family in this
    /// comparison only.
    pub fn compare(
        parent: &CandidateEvaluationView,
        candidate: &CandidateEvaluationView,
        cluster_by_case: &BTreeMap<String, String>,
        tie_epsilon: f64,
    ) -> Result<Self, ClusteredEvaluationError> {
        if !tie_epsilon.is_finite() || !(0.0..=1.0).contains(&tie_epsilon) {
            return Err(PairedEvaluationError::InvalidTieEpsilon.into());
        }
        if parent.context != candidate.context {
            return Err(PairedEvaluationError::ContextMismatch.into());
        }

        let parent_id = parent.candidate.candidate_id();
        let candidate_id = candidate.candidate.candidate_id();
        if parent_id == candidate_id {
            return Err(PairedEvaluationError::SameCandidate.into());
        }
        if !candidate.candidate.has_parent(parent_id) {
            return Err(PairedEvaluationError::ParentLineageMismatch.into());
        }
        if parent.cases.len() != candidate.cases.len()
            || parent.cases.keys().ne(candidate.cases.keys())
        {
            return Err(PairedEvaluationError::CaseSetMismatch.into());
        }
        if cluster_by_case.len() != parent.cases.len()
            || parent
                .cases
                .keys()
                .any(|case_id| !cluster_by_case.contains_key(case_id))
            || cluster_by_case
                .keys()
                .any(|case_id| !parent.cases.contains_key(case_id))
        {
            return Err(ClusteredEvaluationError::ClusterAssignmentMismatch);
        }
        if cluster_by_case.values().any(|cluster_id| {
            cluster_id.trim().is_empty() || cluster_id.chars().any(char::is_control)
        }) {
            return Err(ClusteredEvaluationError::InvalidClusterId);
        }

        let parent_metrics: Vec<&str> =
            parent.candidate.capabilities().metric_ids().collect();
        let candidate_metrics: Vec<&str> =
            candidate.candidate.capabilities().metric_ids().collect();
        if parent_metrics != candidate_metrics {
            return Err(PairedEvaluationError::MetricSchemaMismatch.into());
        }

        let mut metrics = BTreeMap::new();
        for metric_id in parent_metrics {
            let mut case_delta_sum = 0.0_f64;
            let mut cluster_delta_sums: BTreeMap<&str, (f64, usize)> = BTreeMap::new();

            for (case_id, parent_case) in &parent.cases {
                let candidate_case = candidate
                    .cases
                    .get(case_id)
                    .expect("case-set equality was checked before metric comparison");
                let parent_score = parent_case
                    .scores
                    .get(metric_id)
                    .expect("receipt enforces metric schema equality");
                let candidate_score = candidate_case
                    .scores
                    .get(metric_id)
                    .expect("receipt enforces metric schema equality");
                let delta = candidate_score - parent_score;
                case_delta_sum += delta;

                let cluster_id = cluster_by_case
                    .get(case_id)
                    .expect("cluster-map equality was checked before metric comparison");
                let entry = cluster_delta_sums
                    .entry(cluster_id.as_str())
                    .or_insert((0.0, 0));
                entry.0 += delta;
                entry.1 += 1;
            }

            let cluster_count = cluster_delta_sums.len();
            let mut cluster_wins = 0usize;
            let mut cluster_losses = 0usize;
            let mut cluster_ties = 0usize;
            let mut equal_cluster_delta_sum = 0.0_f64;

            for (delta_sum, count) in cluster_delta_sums.values() {
                let cluster_mean_delta = *delta_sum / *count as f64;
                equal_cluster_delta_sum += cluster_mean_delta;
                if cluster_mean_delta > tie_epsilon {
                    cluster_wins += 1;
                } else if cluster_mean_delta < -tie_epsilon {
                    cluster_losses += 1;
                } else {
                    cluster_ties += 1;
                }
            }

            let non_tied_clusters = cluster_wins + cluster_losses;
            let cluster_sign_test_p_value = if non_tied_clusters >= 2 {
                exact_sign_test_upper_tail(cluster_wins, cluster_losses)
            } else {
                None
            };

            metrics.insert(
                metric_id.to_owned(),
                ClusteredMetricPairedResult {
                    metric_id: metric_id.to_owned(),
                    case_count: parent.cases.len(),
                    cluster_count,
                    cluster_wins,
                    cluster_losses,
                    cluster_ties,
                    case_weighted_mean_delta: case_delta_sum / parent.cases.len() as f64,
                    equal_cluster_mean_delta: equal_cluster_delta_sum / cluster_count as f64,
                    cluster_sign_test_p_value,
                    holm_adjusted_p_value: None,
                },
            );
        }

        let raw_p_values: BTreeMap<String, f64> = metrics
            .iter()
            .filter_map(|(metric_id, result)| {
                result
                    .cluster_sign_test_p_value
                    .map(|p_value| (metric_id.clone(), p_value))
            })
            .collect();
        // The family size is the full frozen metric schema, not only metrics
        // with enough non-tied clusters to produce a p-value. Unavailable tests
        // therefore count conservatively as p=1 for the step-down family size.
        let adjusted = holm_adjust_p_values(&raw_p_values, metrics.len());
        for (metric_id, result) in &mut metrics {
            result.holm_adjusted_p_value = adjusted.get(metric_id).copied();
        }

        let cluster_count = cluster_by_case
            .values()
            .collect::<std::collections::BTreeSet<_>>()
            .len();

        Ok(Self {
            parent_candidate_id: parent_id.to_owned(),
            candidate_id: candidate_id.to_owned(),
            case_count: parent.cases.len(),
            cluster_count,
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

    pub fn cluster_count(&self) -> usize {
        self.cluster_count
    }

    pub fn metric(&self, metric_id: &str) -> Option<&ClusteredMetricPairedResult> {
        self.metrics.get(metric_id)
    }

    pub fn metrics(
        &self,
    ) -> impl Iterator<Item = (&str, &ClusteredMetricPairedResult)> {
        self.metrics
            .iter()
            .map(|(metric_id, result)| (metric_id.as_str(), result))
    }
}

/// Holm step-down correction for one predeclared metric family. The supplied
/// family_size is the full metric-schema size, including metrics without a
/// testable p-value (which conservatively occupy the p=1 end of the family).
/// Sorting ties by metric ID makes the output deterministic.
fn holm_adjust_p_values(
    p_values: &BTreeMap<String, f64>,
    family_size: usize,
) -> BTreeMap<String, f64> {
    let mut ordered: Vec<(String, f64)> = p_values
        .iter()
        .map(|(metric_id, p_value)| (metric_id.clone(), *p_value))
        .collect();
    ordered.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    debug_assert!(family_size >= ordered.len());
    let mut previous_adjusted = 0.0_f64;
    let mut adjusted = BTreeMap::new();

    for (rank, (metric_id, p_value)) in ordered.into_iter().enumerate() {
        let multiplier = (family_size - rank) as f64;
        let adjusted_p_value = (p_value * multiplier)
            .max(previous_adjusted)
            .clamp(0.0, 1.0);
        previous_adjusted = adjusted_p_value;
        adjusted.insert(metric_id, adjusted_p_value);
    }

    adjusted
}

#[cfg(test)]
mod clustered_tests {
    use super::*;
    use crate::consciousness::recursive_improvement::candidate_archive::EvaluationPartition;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn context() -> EvaluationContext {
        EvaluationContext::new(
            "sym-rsi-clustered-v1",
            "v1",
            EvaluationPartition::Validation,
            "cluster-confirmation",
            digest('a'),
            digest('b'),
            digest('c'),
            digest('d'),
            digest('e'),
        )
        .unwrap()
    }

    fn view(
        context: EvaluationContext,
        candidate_id: &str,
        parent_ids: Vec<String>,
        source: char,
        receipt: char,
        cases: Vec<(String, f64)>,
    ) -> CandidateEvaluationView {
        let scores = cases
            .into_iter()
            .map(|(case_id, coding)| {
                CaseScore::new(
                    case_id,
                    CapabilityVector::new(vec![("coding".to_owned(), coding)]).unwrap(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        CandidateEvaluationView::new(
            context,
            candidate_id,
            parent_ids,
            digest(source),
            digest(receipt),
            scores,
        )
        .unwrap()
    }

    fn assignments(groups: &[(&str, &[&str])]) -> BTreeMap<String, String> {
        let mut result = BTreeMap::new();
        for (cluster, cases) in groups {
            for case_id in *cases {
                result.insert((*case_id).to_owned(), (*cluster).to_owned());
            }
        }
        result
    }

    #[test]
    fn unequal_cluster_sizes_do_not_masquerade_as_independent_replication() {
        let ctx = context();
        let parent_cases: Vec<(String, f64)> = (0..10)
            .map(|index| (format!("case-{index}"), 0.5))
            .collect();
        let candidate_cases: Vec<(String, f64)> = (0..10)
            .map(|index| {
                let score = if index < 9 { 0.6 } else { 0.4 };
                (format!("case-{index}"), score)
            })
            .collect();
        let parent = view(ctx.clone(), "parent", vec![], '1', 'a', parent_cases);
        let candidate = view(
            ctx,
            "candidate",
            vec!["parent".to_owned()],
            '2',
            'b',
            candidate_cases,
        );
        let large_cases: Vec<String> = (0..9).map(|index| format!("case-{index}")).collect();
        let mut cluster_map = large_cases
            .iter()
            .map(|case_id| (case_id.clone(), "repository-large".to_owned()))
            .collect::<BTreeMap<_, _>>();
        cluster_map.insert("case-9".to_owned(), "repository-small".to_owned());

        let comparison =
            ClusteredPairedComparison::compare(&parent, &candidate, &cluster_map, 0.0).unwrap();
        let coding = comparison.metric("coding").unwrap();
        assert_eq!(comparison.case_count(), 10);
        assert_eq!(comparison.cluster_count(), 2);
        assert_eq!(coding.cluster_wins(), 1);
        assert_eq!(coding.cluster_losses(), 1);
        assert_eq!(coding.cluster_ties(), 0);
        assert!((coding.case_weighted_mean_delta() - 0.08).abs() < 1e-12);
        assert!((coding.equal_cluster_mean_delta() - 0.0).abs() < 1e-12);
        assert_eq!(coding.cluster_sign_test_p_value(), Some(0.75));
        assert_eq!(coding.holm_adjusted_p_value(), Some(0.75));
    }

    #[test]
    fn known_cluster_sign_test_uses_independent_units() {
        let ctx = context();
        let parent = view(
            ctx.clone(),
            "parent",
            vec![],
            '1',
            'a',
            (0..5).map(|index| (format!("case-{index}"), 0.5)).collect(),
        );
        let candidate = view(
            ctx,
            "candidate",
            vec!["parent".to_owned()],
            '2',
            'b',
            (0..5).map(|index| (format!("case-{index}"), 0.6)).collect(),
        );
        let map = (0..5)
            .map(|index| (format!("case-{index}"), format!("project-{index}")))
            .collect::<BTreeMap<_, _>>();
        let result = ClusteredPairedComparison::compare(&parent, &candidate, &map, 0.0)
            .unwrap()
            .metric("coding")
            .unwrap()
            .clone();
        assert_eq!(result.cluster_wins(), 5);
        assert_eq!(result.cluster_losses(), 0);
        assert_eq!(result.cluster_sign_test_p_value(), Some(0.03125));
        assert_eq!(result.holm_adjusted_p_value(), Some(0.03125));
    }

    #[test]
    fn one_non_tied_cluster_withholds_inferential_p_value() {
        let ctx = context();
        let parent = view(
            ctx.clone(),
            "parent",
            vec![],
            '1',
            'a',
            vec![("case-a".into(), 0.5), ("case-b".into(), 0.5)],
        );
        let candidate = view(
            ctx,
            "candidate",
            vec!["parent".to_owned()],
            '2',
            'b',
            vec![("case-a".into(), 0.6), ("case-b".into(), 0.5)],
        );
        let map = assignments(&[("project-a", &["case-a"]), ("project-b", &["case-b"])]);
        let result = ClusteredPairedComparison::compare(&parent, &candidate, &map, 0.0)
            .unwrap();
        let coding = result.metric("coding").unwrap();
        assert_eq!(coding.cluster_wins(), 1);
        assert_eq!(coding.cluster_ties(), 1);
        assert_eq!(coding.cluster_sign_test_p_value(), None);
        assert_eq!(coding.holm_adjusted_p_value(), None);
    }

    #[test]
    fn rejects_missing_extra_and_malformed_cluster_assignments() {
        let ctx = context();
        let parent = view(
            ctx.clone(),
            "parent",
            vec![],
            '1',
            'a',
            vec![("a".into(), 0.5), ("b".into(), 0.5)],
        );
        let candidate = view(
            ctx,
            "candidate",
            vec!["parent".to_owned()],
            '2',
            'b',
            vec![("a".into(), 0.6), ("b".into(), 0.6)],
        );

        assert_eq!(
            ClusteredPairedComparison::compare(
                &parent,
                &candidate,
                &BTreeMap::from([("a".to_owned(), "repo".to_owned())]),
                0.0,
            ),
            Err(ClusteredEvaluationError::ClusterAssignmentMismatch)
        );
        assert_eq!(
            ClusteredPairedComparison::compare(
                &parent,
                &candidate,
                &BTreeMap::from([
                    ("a".to_owned(), "repo".to_owned()),
                    ("b".to_owned(), "repo".to_owned()),
                    ("extra".to_owned(), "repo".to_owned()),
                ]),
                0.0,
            ),
            Err(ClusteredEvaluationError::ClusterAssignmentMismatch)
        );
        assert_eq!(
            ClusteredPairedComparison::compare(
                &parent,
                &candidate,
                &BTreeMap::from([
                    ("a".to_owned(), "repo".to_owned()),
                    ("b".to_owned(), "   ".to_owned()),
                ]),
                0.0,
            ),
            Err(ClusteredEvaluationError::InvalidClusterId)
        );
    }

    #[test]
    fn holm_correction_is_deterministic_and_monotone() {
        let raw = BTreeMap::from([
            ("metric-a".to_owned(), 0.01),
            ("metric-b".to_owned(), 0.04),
            ("metric-c".to_owned(), 0.03),
        ]);
        let adjusted = holm_adjust_p_values(&raw, 3);
        assert!((adjusted["metric-a"] - 0.03).abs() < 1e-12);
        assert!((adjusted["metric-c"] - 0.06).abs() < 1e-12);
        assert!((adjusted["metric-b"] - 0.06).abs() < 1e-12);
    }

    #[test]
    fn unavailable_metric_still_counts_in_declared_holm_family() {
        let raw = BTreeMap::from([
            ("metric-a".to_owned(), 0.01),
            ("metric-b".to_owned(), 0.02),
        ]);
        // A third preregistered metric has too few independent clusters and
        // therefore no raw p-value; it still belongs to the family as p=1.
        let adjusted = holm_adjust_p_values(&raw, 3);
        assert!((adjusted["metric-a"] - 0.03).abs() < 1e-12);
        assert!((adjusted["metric-b"] - 0.04).abs() < 1e-12);
    }

    #[test]
    fn all_metric_ties_produce_no_p_value_or_adjustment() {
        let ctx = context();
        let parent = view(
            ctx.clone(),
            "parent",
            vec![],
            '1',
            'a',
            vec![("a".into(), 0.5), ("b".into(), 0.5)],
        );
        let candidate = view(
            ctx,
            "candidate",
            vec!["parent".to_owned()],
            '2',
            'b',
            vec![("a".into(), 0.5), ("b".into(), 0.5)],
        );
        let map = assignments(&[("repo-a", &["a"]), ("repo-b", &["b"])]);
        let coding = ClusteredPairedComparison::compare(&parent, &candidate, &map, 0.0)
            .unwrap()
            .metric("coding")
            .unwrap()
            .clone();
        assert_eq!(coding.cluster_ties(), 2);
        assert_eq!(coding.cluster_sign_test_p_value(), None);
        assert_eq!(coding.holm_adjusted_p_value(), None);
    }
}
