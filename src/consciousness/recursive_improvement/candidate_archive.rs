// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-context-bound Pareto archive for recursive-improvement candidates.
//!
//! The archive retains every candidate for lineage and exposes the non-dominated
//! frontier for a single, exactly matched evaluation context. It deliberately
//! does not collapse capability dimensions into a weighted scalar or authorize
//! promotion. A separate promotion protocol must decide whether a frontier member
//! is safe and useful to deploy.
//!
//! Deserialization is intentionally not implemented for these types. Persisted
//! records must be reconstructed through the validating constructors below.

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// The evaluation partition used to score an improvement candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationPartition {
    /// Public or mutable experiments used to develop a candidate.
    Development,
    /// A fixed partition used to select among candidate variants.
    Validation,
    /// A separately controlled partition reserved for final confirmation.
    SealedHoldout,
}

/// Errors returned when constructing or extending a candidate archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateArchiveError {
    InvalidIdentifier(&'static str),
    InvalidDigest(&'static str),
    EmptyCapabilityVector,
    InvalidMetricId(String),
    DuplicateMetric(String),
    InvalidScore(String),
    DuplicateCandidate(String),
    MissingParent(String),
    SelfParent(String),
    MetricSchemaMismatch,
}

impl fmt::Display for CandidateArchiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier(field) => write!(f, "invalid or empty {field}"),
            Self::InvalidDigest(field) => {
                write!(f, "{field} must be a canonical 64-character lowercase hex digest")
            }
            Self::EmptyCapabilityVector => write!(f, "capability vector must not be empty"),
            Self::InvalidMetricId(id) => write!(f, "invalid capability metric id: {id}"),
            Self::DuplicateMetric(id) => write!(f, "duplicate capability metric id: {id}"),
            Self::InvalidScore(id) => {
                write!(f, "score for {id} must be finite and within [0, 1]")
            }
            Self::DuplicateCandidate(id) => write!(f, "candidate already exists: {id}"),
            Self::MissingParent(id) => write!(f, "candidate parent does not exist: {id}"),
            Self::SelfParent(id) => write!(f, "candidate cannot be its own parent: {id}"),
            Self::MetricSchemaMismatch => {
                write!(f, "candidate capability dimensions do not match this archive")
            }
        }
    }
}

impl std::error::Error for CandidateArchiveError {}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), CandidateArchiveError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(CandidateArchiveError::InvalidIdentifier(field));
    }
    Ok(())
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), CandidateArchiveError> {
    let is_canonical_hex = value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !is_canonical_hex {
        return Err(CandidateArchiveError::InvalidDigest(field));
    }
    Ok(())
}

/// Immutable evaluation identity. Candidates are comparable only when every
/// field is identical; a different split, evaluator, corpus, budget or seed
/// manifest requires a different archive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvaluationContext {
    benchmark_id: String,
    benchmark_version: String,
    partition: EvaluationPartition,
    split_id: String,
    evaluator_digest: String,
    corpus_digest: String,
    metric_schema_digest: String,
    budget_digest: String,
    seed_manifest_digest: String,
}

impl EvaluationContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        benchmark_id: impl Into<String>,
        benchmark_version: impl Into<String>,
        partition: EvaluationPartition,
        split_id: impl Into<String>,
        evaluator_digest: impl Into<String>,
        corpus_digest: impl Into<String>,
        metric_schema_digest: impl Into<String>,
        budget_digest: impl Into<String>,
        seed_manifest_digest: impl Into<String>,
    ) -> Result<Self, CandidateArchiveError> {
        let context = Self {
            benchmark_id: benchmark_id.into(),
            benchmark_version: benchmark_version.into(),
            partition,
            split_id: split_id.into(),
            evaluator_digest: evaluator_digest.into(),
            corpus_digest: corpus_digest.into(),
            metric_schema_digest: metric_schema_digest.into(),
            budget_digest: budget_digest.into(),
            seed_manifest_digest: seed_manifest_digest.into(),
        };

        validate_identifier(&context.benchmark_id, "benchmark_id")?;
        validate_identifier(&context.benchmark_version, "benchmark_version")?;
        validate_identifier(&context.split_id, "split_id")?;
        validate_digest(&context.evaluator_digest, "evaluator_digest")?;
        validate_digest(&context.corpus_digest, "corpus_digest")?;
        validate_digest(&context.metric_schema_digest, "metric_schema_digest")?;
        validate_digest(&context.budget_digest, "budget_digest")?;
        validate_digest(&context.seed_manifest_digest, "seed_manifest_digest")?;
        Ok(context)
    }

    pub fn benchmark_id(&self) -> &str {
        &self.benchmark_id
    }

    pub fn benchmark_version(&self) -> &str {
        &self.benchmark_version
    }

    pub fn partition(&self) -> EvaluationPartition {
        self.partition
    }

    pub fn split_id(&self) -> &str {
        &self.split_id
    }

    pub fn is_sealed_holdout(&self) -> bool {
        self.partition == EvaluationPartition::SealedHoldout
    }
}

/// Capability scores normalized so higher is always better and each dimension
/// is in [0, 1]. Resource efficiency must be normalized by the frozen metric
/// schema; raw cost and latency should not be mixed with capability scores.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CapabilityVector {
    scores: BTreeMap<String, f64>,
}

impl CapabilityVector {
    pub fn new(
        scores: impl IntoIterator<Item = (String, f64)>,
    ) -> Result<Self, CandidateArchiveError> {
        let mut normalized = BTreeMap::new();
        for (metric_id, score) in scores {
            validate_identifier(&metric_id, "metric_id")
                .map_err(|_| CandidateArchiveError::InvalidMetricId(metric_id.clone()))?;
            if !score.is_finite() || !(0.0..=1.0).contains(&score) {
                return Err(CandidateArchiveError::InvalidScore(metric_id));
            }
            if normalized.insert(metric_id.clone(), score).is_some() {
                return Err(CandidateArchiveError::DuplicateMetric(metric_id));
            }
        }

        if normalized.is_empty() {
            return Err(CandidateArchiveError::EmptyCapabilityVector);
        }

        Ok(Self { scores: normalized })
    }

    pub fn get(&self, metric_id: &str) -> Option<f64> {
        self.scores.get(metric_id).copied()
    }

    pub fn metric_ids(&self) -> impl Iterator<Item = &str> {
        self.scores.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.scores.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }
}

/// A candidate version plus its exact source and evaluation evidence identities.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CandidateRecord {
    candidate_id: String,
    parent_id: Option<String>,
    source_tree_digest: String,
    receipt_digest: String,
    capabilities: CapabilityVector,
}

impl CandidateRecord {
    pub fn new(
        candidate_id: impl Into<String>,
        parent_id: Option<String>,
        source_tree_digest: impl Into<String>,
        receipt_digest: impl Into<String>,
        capabilities: CapabilityVector,
    ) -> Result<Self, CandidateArchiveError> {
        let candidate_id = candidate_id.into();
        let source_tree_digest = source_tree_digest.into();
        let receipt_digest = receipt_digest.into();

        validate_identifier(&candidate_id, "candidate_id")?;
        validate_digest(&source_tree_digest, "source_tree_digest")?;
        validate_digest(&receipt_digest, "receipt_digest")?;

        if let Some(parent) = parent_id.as_deref() {
            validate_identifier(parent, "parent_id")?;
            if parent == candidate_id {
                return Err(CandidateArchiveError::SelfParent(candidate_id));
            }
        }

        Ok(Self {
            candidate_id,
            parent_id,
            source_tree_digest,
            receipt_digest,
            capabilities,
        })
    }

    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn parent_id(&self) -> Option<&str> {
        self.parent_id.as_deref()
    }

    pub fn source_tree_digest(&self) -> &str {
        &self.source_tree_digest
    }

    pub fn receipt_digest(&self) -> &str {
        &self.receipt_digest
    }

    pub fn capabilities(&self) -> &CapabilityVector {
        &self.capabilities
    }
}

/// Retains all candidate records while exposing a Pareto frontier over one
/// frozen evaluation context. Dominated candidates remain stored so their
/// lineage and negative results are not erased.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CandidateArchive {
    context: EvaluationContext,
    candidates: BTreeMap<String, CandidateRecord>,
    frontier: BTreeSet<String>,
    metric_ids: Option<BTreeSet<String>>,
}

impl CandidateArchive {
    pub fn new(context: EvaluationContext) -> Self {
        Self {
            context,
            candidates: BTreeMap::new(),
            frontier: BTreeSet::new(),
            metric_ids: None,
        }
    }

    /// Insert a validated candidate. Parents must already exist, which makes
    /// the recorded lineage acyclic by construction.
    pub fn insert(
        &mut self,
        candidate: CandidateRecord,
    ) -> Result<(), CandidateArchiveError> {
        let candidate_id = candidate.candidate_id.clone();
        if self.candidates.contains_key(&candidate_id) {
            return Err(CandidateArchiveError::DuplicateCandidate(candidate_id));
        }

        if let Some(parent_id) = candidate.parent_id.as_deref() {
            if !self.candidates.contains_key(parent_id) {
                return Err(CandidateArchiveError::MissingParent(parent_id.to_owned()));
            }
        }

        let candidate_metric_ids: BTreeSet<String> =
            candidate.capabilities.metric_ids().map(str::to_owned).collect();
        if let Some(expected) = &self.metric_ids {
            if expected != &candidate_metric_ids {
                return Err(CandidateArchiveError::MetricSchemaMismatch);
            }
        } else {
            self.metric_ids = Some(candidate_metric_ids);
        }

        self.candidates.insert(candidate_id, candidate);
        self.recompute_frontier();
        Ok(())
    }

    pub fn candidate(&self, candidate_id: &str) -> Option<&CandidateRecord> {
        self.candidates.get(candidate_id)
    }

    /// IDs in deterministic order. This is a multi-objective frontier, not a
    /// scalar ranking; callers must not infer a unique winner from this list.
    pub fn pareto_frontier_ids(&self) -> Vec<&str> {
        self.frontier.iter().map(String::as_str).collect()
    }

    pub fn evaluation_context(&self) -> &EvaluationContext {
        &self.context
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    fn recompute_frontier(&mut self) {
        self.frontier.clear();
        let candidate_ids: Vec<String> = self.candidates.keys().cloned().collect();

        for candidate_id in &candidate_ids {
            let candidate = self
                .candidates
                .get(candidate_id)
                .expect("candidate id came from the archive key set");
            let dominated = candidate_ids.iter().any(|other_id| {
                if candidate_id == other_id {
                    return false;
                }
                let other = self
                    .candidates
                    .get(other_id)
                    .expect("candidate id came from the archive key set");
                dominates(other, candidate)
            });

            if !dominated {
                self.frontier.insert(candidate_id.clone());
            }
        }
    }
}

/// Returns true only when left is at least as good on every frozen metric
/// and strictly better on at least one. Exact ties remain distinct archive
/// members rather than being silently discarded.
fn dominates(left: &CandidateRecord, right: &CandidateRecord) -> bool {
    let mut strictly_better = false;

    for metric_id in right.capabilities.metric_ids() {
        let left_score = left
            .capabilities
            .get(metric_id)
            .expect("archive enforces an identical metric schema");
        let right_score = right
            .capabilities
            .get(metric_id)
            .expect("metric id came from the right candidate");

        if left_score < right_score {
            return false;
        }
        if left_score > right_score {
            strictly_better = true;
        }
    }

    strictly_better
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn context(partition: EvaluationPartition) -> EvaluationContext {
        EvaluationContext::new(
            "sym-rsi-candidate-archive",
            "v1",
            partition,
            "held-out-2026-11",
            digest('a'),
            digest('b'),
            digest('c'),
            digest('d'),
            digest('e'),
        )
        .unwrap()
    }

    fn candidate(
        id: &str,
        parent: Option<&str>,
        source_byte: char,
        receipt_byte: char,
        scores: &[(&str, f64)],
    ) -> CandidateRecord {
        let capabilities = CapabilityVector::new(
            scores
                .iter()
                .map(|(metric, score)| ((*metric).to_owned(), *score)),
        )
        .unwrap();
        CandidateRecord::new(
            id,
            parent.map(str::to_owned),
            digest(source_byte),
            digest(receipt_byte),
            capabilities,
        )
        .unwrap()
    }

    #[test]
    fn retains_all_candidates_and_exposes_only_non_dominated_frontier() {
        let mut archive = CandidateArchive::new(context(EvaluationPartition::SealedHoldout));

        archive
            .insert(candidate(
                "baseline",
                None,
                '1',
                'a',
                &[("coding", 0.70), ("generalization", 0.70), ("retention", 0.90)],
            ))
            .unwrap();
        archive
            .insert(candidate(
                "coding-specialist",
                Some("baseline"),
                '2',
                'b',
                &[("coding", 0.92), ("generalization", 0.70), ("retention", 0.90)],
            ))
            .unwrap();
        archive
            .insert(candidate(
                "generalist",
                Some("baseline"),
                '3',
                'c',
                &[("coding", 0.82), ("generalization", 0.91), ("retention", 0.90)],
            ))
            .unwrap();
        archive
            .insert(candidate(
                "regressed",
                Some("coding-specialist"),
                '4',
                'd',
                &[("coding", 0.60), ("generalization", 0.60), ("retention", 0.80)],
            ))
            .unwrap();

        assert_eq!(archive.len(), 4);
        assert_eq!(
            archive.pareto_frontier_ids(),
            vec!["coding-specialist", "generalist"]
        );
        assert!(archive.candidate("baseline").is_some());
        assert!(archive.candidate("regressed").is_some());
    }

    #[test]
    fn preserves_tradeoffs_instead_of_scalarizing_them() {
        let mut archive = CandidateArchive::new(context(EvaluationPartition::Validation));
        archive
            .insert(candidate(
                "coding",
                None,
                '1',
                'a',
                &[("coding", 0.95), ("safety", 0.80)],
            ))
            .unwrap();
        archive
            .insert(candidate(
                "safety",
                Some("coding"),
                '2',
                'b',
                &[("coding", 0.85), ("safety", 0.98)],
            ))
            .unwrap();

        assert_eq!(archive.pareto_frontier_ids(), vec!["coding", "safety"]);
    }

    #[test]
    fn rejects_missing_parent_without_mutating_archive() {
        let mut archive = CandidateArchive::new(context(EvaluationPartition::Development));
        let orphan = candidate("orphan", Some("unknown"), '1', 'a', &[("coding", 0.5)]);

        assert_eq!(
            archive.insert(orphan),
            Err(CandidateArchiveError::MissingParent("unknown".into()))
        );
        assert!(archive.is_empty());

        archive
            .insert(candidate("root", None, '2', 'b', &[("coding", 0.5)]))
            .unwrap();
        archive
            .insert(candidate("child", Some("root"), '3', 'c', &[("coding", 0.6)]))
            .unwrap();
        assert_eq!(archive.len(), 2);
    }

    #[test]
    fn rejects_duplicate_candidate_and_metric_schema_drift() {
        let mut archive = CandidateArchive::new(context(EvaluationPartition::Development));
        archive
            .insert(candidate(
                "first",
                None,
                '1',
                'a',
                &[("coding", 0.5), ("retention", 0.8)],
            ))
            .unwrap();

        assert_eq!(
            archive.insert(candidate(
                "first",
                None,
                '2',
                'b',
                &[("coding", 0.6), ("retention", 0.9)],
            )),
            Err(CandidateArchiveError::DuplicateCandidate("first".into()))
        );

        assert_eq!(
            archive.insert(candidate(
                "wrong-schema",
                Some("first"),
                '3',
                'c',
                &[("coding", 0.6)],
            )),
            Err(CandidateArchiveError::MetricSchemaMismatch)
        );
        assert_eq!(archive.len(), 1);
    }

    #[test]
    fn rejects_invalid_scores_and_noncanonical_digests() {
        assert_eq!(
            CapabilityVector::new(vec![("coding".into(), f64::NAN)]),
            Err(CandidateArchiveError::InvalidScore("coding".into()))
        );
        assert_eq!(
            CapabilityVector::new(vec![("coding".into(), 1.01)]),
            Err(CandidateArchiveError::InvalidScore("coding".into()))
        );
        assert_eq!(
            EvaluationContext::new(
                "suite",
                "v1",
                EvaluationPartition::Development,
                "split",
                "not-a-digest",
                digest('b'),
                digest('c'),
                digest('d'),
                digest('e'),
            ),
            Err(CandidateArchiveError::InvalidDigest("evaluator_digest"))
        );
    }

    #[test]
    fn sealed_holdout_is_explicitly_marked_but_not_promotion_authority() {
        let holdout = context(EvaluationPartition::SealedHoldout);
        let development = context(EvaluationPartition::Development);

        assert!(holdout.is_sealed_holdout());
        assert!(!development.is_sealed_holdout());
        // The archive exposes evidence and a Pareto frontier only. It has no
        // promotion or deployment API.
        let archive = CandidateArchive::new(holdout);
        assert!(archive.is_empty());
    }
}
