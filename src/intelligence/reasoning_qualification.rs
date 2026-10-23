// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence substrate for reasoning qualification.
//!
//! This module stores externally auditable facts about a reasoning episode. It deliberately
//! does not require or persist hidden natural-language chain of thought. The qualification
//! surface is instead made from immutable subject/problem identity, evidence provenance,
//! declared assumptions, executed operation summaries, answer/abstention, resource use, and
//! evaluator receipts.
//!
//! Capability maturity is intentionally non-equivalent:
//!
//! `implemented != measured != qualified != replicated != established`.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Schema version for the canonical reasoning evidence record.
pub const REASONING_EPISODE_SCHEMA_VERSION: u32 = 1;

/// Identifier derived deterministically from canonical episode contents.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReasoningEpisodeId(pub String);

/// Broad domain used to slice qualification results without inventing one intelligence scalar.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReasoningDomain {
    Abstraction,
    Logic,
    Mathematics,
    Causal,
    Scientific,
    Coding,
    Planning,
    Commonsense,
    Social,
    Epistemic,
    ToolUse,
    General,
    Custom(String),
}

/// Immutable reference to the problem and evaluation lineage used by an episode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningProblemRef {
    pub benchmark: String,
    pub benchmark_version: String,
    pub split: String,
    pub problem_id: String,
    pub problem_hash: String,
}

/// Reference to evidence consumed during reasoning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub id: String,
    pub content_hash: String,
    pub provenance: String,
    /// Known common-source descendants can share a group so they are not mistaken for
    /// independent corroboration by higher-level evaluators.
    pub independence_group: Option<String>,
}

/// Explicit premise not directly established by supplied evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssumptionRecord {
    pub id: String,
    /// Public auditable statement, not a hidden reasoning transcript.
    pub statement: String,
    pub confidence: f64,
}

/// Public operation-level trace entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningDecisionRecord {
    pub operation: String,
    pub input_refs: Vec<String>,
    pub output_refs: Vec<String>,
    pub verifier: Option<String>,
}

/// Why an episode intentionally returned no asserted answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AbstentionReason {
    InsufficientEvidence,
    Unidentified,
    OutOfDistribution,
    ResourceBudgetExceeded,
    ConflictingEvidence,
    UnsafeToConclude,
    Other(String),
}

/// Subject output of a reasoning episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReasoningOutcome {
    Asserted {
        value: String,
        confidence: f64,
    },
    Abstained {
        reason: AbstentionReason,
        /// Estimated probability that an answer could responsibly be asserted from the
        /// available state. It is not a substitute for correctness calibration.
        answerability: f64,
    },
}

/// Resource usage bound to a reasoning episode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub wall_time_us: u64,
    pub deliberation_steps: u64,
    pub tool_calls: u64,
    pub model_tokens: u64,
}

/// Canonical externally auditable reasoning episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningEpisode {
    pub schema_version: u32,
    pub subject_revision: String,
    pub configuration_id: String,
    pub domain: ReasoningDomain,
    pub problem: ReasoningProblemRef,
    pub evidence: Vec<EvidenceRef>,
    pub assumptions: Vec<AssumptionRecord>,
    pub decisions: Vec<ReasoningDecisionRecord>,
    pub outcome: ReasoningOutcome,
    pub resources: ResourceUsage,
}

/// One evaluator-produced metric. Metrics remain decomposed instead of being hidden behind an
/// opaque composite intelligence score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationMetric {
    pub name: String,
    pub value: f64,
    pub unit: String,
}

/// Immutable evaluator receipt binding a validated episode to qualification measurements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningQualificationReceipt {
    pub receipt_id: String,
    pub episode_id: ReasoningEpisodeId,
    pub evaluator: String,
    pub evaluator_version: String,
    pub evaluation_lineage_hash: String,
    pub metrics: Vec<QualificationMetric>,
    /// Optional exact correctness result when the task admits binary ground truth.
    pub exact_correct: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationValidationError {
    EmptyField(&'static str),
    UnsupportedSchemaVersion(u32),
    InvalidProbability(&'static str),
    InvalidMetric(String),
    DuplicateEvidenceId(String),
    DuplicateAssumptionId(String),
    EmptyDecisionOperation,
    EmptyMetricName,
    DuplicateMetricName(String),
    ReceiptIdMismatch,
}

impl fmt::Display for QualificationValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field `{field}` is empty"),
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported reasoning episode schema version {version}")
            }
            Self::InvalidProbability(field) => {
                write!(f, "probability `{field}` must be finite and within [0, 1]")
            }
            Self::InvalidMetric(name) => write!(f, "metric `{name}` must be finite"),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate evidence id `{id}`"),
            Self::DuplicateAssumptionId(id) => write!(f, "duplicate assumption id `{id}`"),
            Self::EmptyDecisionOperation => write!(f, "decision operation must not be empty"),
            Self::EmptyMetricName => write!(f, "qualification metric name must not be empty"),
            Self::DuplicateMetricName(name) => write!(f, "duplicate metric name `{name}`"),
            Self::ReceiptIdMismatch => write!(f, "qualification receipt identity does not match contents"),
        }
    }
}

impl std::error::Error for QualificationValidationError {}

impl ReasoningEpisode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject_revision: impl Into<String>,
        configuration_id: impl Into<String>,
        domain: ReasoningDomain,
        problem: ReasoningProblemRef,
        evidence: Vec<EvidenceRef>,
        assumptions: Vec<AssumptionRecord>,
        decisions: Vec<ReasoningDecisionRecord>,
        outcome: ReasoningOutcome,
        resources: ResourceUsage,
    ) -> Result<Self, QualificationValidationError> {
        let episode = Self {
            schema_version: REASONING_EPISODE_SCHEMA_VERSION,
            subject_revision: subject_revision.into(),
            configuration_id: configuration_id.into(),
            domain,
            problem,
            evidence,
            assumptions,
            decisions,
            outcome,
            resources,
        };
        episode.validate()?;
        Ok(episode)
    }

    /// Validate persisted or newly created episode state without rewriting it.
    pub fn validate(&self) -> Result<(), QualificationValidationError> {
        if self.schema_version != REASONING_EPISODE_SCHEMA_VERSION {
            return Err(QualificationValidationError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        require_nonempty("subject_revision", &self.subject_revision)?;
        require_nonempty("configuration_id", &self.configuration_id)?;
        self.problem.validate()?;

        let mut evidence_ids = std::collections::HashSet::with_capacity(self.evidence.len());
        for item in &self.evidence {
            item.validate()?;
            if !evidence_ids.insert(item.id.as_str()) {
                return Err(QualificationValidationError::DuplicateEvidenceId(
                    item.id.clone(),
                ));
            }
        }

        let mut assumption_ids = std::collections::HashSet::with_capacity(self.assumptions.len());
        for assumption in &self.assumptions {
            assumption.validate()?;
            if !assumption_ids.insert(assumption.id.as_str()) {
                return Err(QualificationValidationError::DuplicateAssumptionId(
                    assumption.id.clone(),
                ));
            }
        }

        for decision in &self.decisions {
            if decision.operation.trim().is_empty() {
                return Err(QualificationValidationError::EmptyDecisionOperation);
            }
            if let Some(verifier) = &decision.verifier {
                require_nonempty("decision.verifier", verifier)?;
            }
        }

        self.outcome.validate()
    }

    /// Deterministic content identity for a validated episode.
    ///
    /// Length prefixes prevent ambiguous concatenation. Floating-point probabilities are hashed
    /// by their IEEE-754 bit patterns after validation rejects NaN/infinity.
    pub fn id(&self) -> Result<ReasoningEpisodeId, QualificationValidationError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hash_u64(&mut hasher, u64::from(self.schema_version));
        hash_str(&mut hasher, &self.subject_revision);
        hash_str(&mut hasher, &self.configuration_id);
        hash_domain(&mut hasher, &self.domain);
        hash_problem(&mut hasher, &self.problem);

        hash_usize(&mut hasher, self.evidence.len());
        for evidence in &self.evidence {
            hash_str(&mut hasher, &evidence.id);
            hash_str(&mut hasher, &evidence.content_hash);
            hash_str(&mut hasher, &evidence.provenance);
            hash_option_str(&mut hasher, evidence.independence_group.as_deref());
        }

        hash_usize(&mut hasher, self.assumptions.len());
        for assumption in &self.assumptions {
            hash_str(&mut hasher, &assumption.id);
            hash_str(&mut hasher, &assumption.statement);
            hash_u64(&mut hasher, assumption.confidence.to_bits());
        }

        hash_usize(&mut hasher, self.decisions.len());
        for decision in &self.decisions {
            hash_str(&mut hasher, &decision.operation);
            hash_strings(&mut hasher, &decision.input_refs);
            hash_strings(&mut hasher, &decision.output_refs);
            hash_option_str(&mut hasher, decision.verifier.as_deref());
        }

        match &self.outcome {
            ReasoningOutcome::Asserted { value, confidence } => {
                hasher.update(&[0]);
                hash_str(&mut hasher, value);
                hash_u64(&mut hasher, confidence.to_bits());
            }
            ReasoningOutcome::Abstained {
                reason,
                answerability,
            } => {
                hasher.update(&[1]);
                hash_abstention_reason(&mut hasher, reason);
                hash_u64(&mut hasher, answerability.to_bits());
            }
        }

        hash_u64(&mut hasher, self.resources.wall_time_us);
        hash_u64(&mut hasher, self.resources.deliberation_steps);
        hash_u64(&mut hasher, self.resources.tool_calls);
        hash_u64(&mut hasher, self.resources.model_tokens);

        Ok(ReasoningEpisodeId(hasher.finalize().to_hex().to_string()))
    }
}

impl ReasoningProblemRef {
    pub fn validate(&self) -> Result<(), QualificationValidationError> {
        require_nonempty("problem.benchmark", &self.benchmark)?;
        require_nonempty("problem.benchmark_version", &self.benchmark_version)?;
        require_nonempty("problem.split", &self.split)?;
        require_nonempty("problem.problem_id", &self.problem_id)?;
        require_nonempty("problem.problem_hash", &self.problem_hash)
    }
}

impl EvidenceRef {
    pub fn validate(&self) -> Result<(), QualificationValidationError> {
        require_nonempty("evidence.id", &self.id)?;
        require_nonempty("evidence.content_hash", &self.content_hash)?;
        require_nonempty("evidence.provenance", &self.provenance)?;
        if let Some(group) = &self.independence_group {
            require_nonempty("evidence.independence_group", group)?;
        }
        Ok(())
    }
}

impl AssumptionRecord {
    pub fn validate(&self) -> Result<(), QualificationValidationError> {
        require_nonempty("assumption.id", &self.id)?;
        require_nonempty("assumption.statement", &self.statement)?;
        require_probability("assumption.confidence", self.confidence)
    }
}

impl ReasoningOutcome {
    pub fn validate(&self) -> Result<(), QualificationValidationError> {
        match self {
            Self::Asserted { value, confidence } => {
                require_nonempty("outcome.value", value)?;
                require_probability("outcome.confidence", *confidence)
            }
            Self::Abstained { answerability, .. } => {
                require_probability("outcome.answerability", *answerability)
            }
        }
    }
}

impl ReasoningQualificationReceipt {
    pub fn new(
        episode: &ReasoningEpisode,
        evaluator: impl Into<String>,
        evaluator_version: impl Into<String>,
        evaluation_lineage_hash: impl Into<String>,
        metrics: Vec<QualificationMetric>,
        exact_correct: Option<bool>,
    ) -> Result<Self, QualificationValidationError> {
        let episode_id = episode.id()?;
        let evaluator = evaluator.into();
        let evaluator_version = evaluator_version.into();
        let evaluation_lineage_hash = evaluation_lineage_hash.into();
        require_nonempty("receipt.evaluator", &evaluator)?;
        require_nonempty("receipt.evaluator_version", &evaluator_version)?;
        require_nonempty("receipt.evaluation_lineage_hash", &evaluation_lineage_hash)?;
        validate_metrics(&metrics)?;

        let receipt_id = receipt_id(
            &episode_id,
            &evaluator,
            &evaluator_version,
            &evaluation_lineage_hash,
            &metrics,
            exact_correct,
        );
        Ok(Self {
            receipt_id,
            episode_id,
            evaluator,
            evaluator_version,
            evaluation_lineage_hash,
            metrics,
            exact_correct,
        })
    }

    /// Validate both field semantics and the deterministic receipt binding.
    pub fn validate(&self) -> Result<(), QualificationValidationError> {
        require_nonempty("receipt.receipt_id", &self.receipt_id)?;
        require_nonempty("receipt.episode_id", &self.episode_id.0)?;
        require_nonempty("receipt.evaluator", &self.evaluator)?;
        require_nonempty("receipt.evaluator_version", &self.evaluator_version)?;
        require_nonempty(
            "receipt.evaluation_lineage_hash",
            &self.evaluation_lineage_hash,
        )?;
        validate_metrics(&self.metrics)?;

        let expected = receipt_id(
            &self.episode_id,
            &self.evaluator,
            &self.evaluator_version,
            &self.evaluation_lineage_hash,
            &self.metrics,
            self.exact_correct,
        );
        if expected != self.receipt_id {
            return Err(QualificationValidationError::ReceiptIdMismatch);
        }
        Ok(())
    }
}

fn validate_metrics(metrics: &[QualificationMetric]) -> Result<(), QualificationValidationError> {
    let mut names = std::collections::HashSet::with_capacity(metrics.len());
    for metric in metrics {
        if metric.name.trim().is_empty() {
            return Err(QualificationValidationError::EmptyMetricName);
        }
        if !metric.value.is_finite() {
            return Err(QualificationValidationError::InvalidMetric(
                metric.name.clone(),
            ));
        }
        if !names.insert(metric.name.as_str()) {
            return Err(QualificationValidationError::DuplicateMetricName(
                metric.name.clone(),
            ));
        }
    }
    Ok(())
}

fn receipt_id(
    episode_id: &ReasoningEpisodeId,
    evaluator: &str,
    evaluator_version: &str,
    evaluation_lineage_hash: &str,
    metrics: &[QualificationMetric],
    exact_correct: Option<bool>,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_str(&mut hasher, &episode_id.0);
    hash_str(&mut hasher, evaluator);
    hash_str(&mut hasher, evaluator_version);
    hash_str(&mut hasher, evaluation_lineage_hash);
    hash_usize(&mut hasher, metrics.len());
    for metric in metrics {
        hash_str(&mut hasher, &metric.name);
        hash_u64(&mut hasher, metric.value.to_bits());
        hash_str(&mut hasher, &metric.unit);
    }
    match exact_correct {
        None => hasher.update(&[0]),
        Some(false) => hasher.update(&[1]),
        Some(true) => hasher.update(&[2]),
    };
    hasher.finalize().to_hex().to_string()
}

fn require_nonempty(
    field: &'static str,
    value: &str,
) -> Result<(), QualificationValidationError> {
    if value.trim().is_empty() {
        Err(QualificationValidationError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn require_probability(
    field: &'static str,
    value: f64,
) -> Result<(), QualificationValidationError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(QualificationValidationError::InvalidProbability(field))
    }
}

fn hash_problem(hasher: &mut blake3::Hasher, problem: &ReasoningProblemRef) {
    hash_str(hasher, &problem.benchmark);
    hash_str(hasher, &problem.benchmark_version);
    hash_str(hasher, &problem.split);
    hash_str(hasher, &problem.problem_id);
    hash_str(hasher, &problem.problem_hash);
}

fn hash_domain(hasher: &mut blake3::Hasher, domain: &ReasoningDomain) {
    let (tag, custom) = match domain {
        ReasoningDomain::Abstraction => (0_u8, None),
        ReasoningDomain::Logic => (1, None),
        ReasoningDomain::Mathematics => (2, None),
        ReasoningDomain::Causal => (3, None),
        ReasoningDomain::Scientific => (4, None),
        ReasoningDomain::Coding => (5, None),
        ReasoningDomain::Planning => (6, None),
        ReasoningDomain::Commonsense => (7, None),
        ReasoningDomain::Social => (8, None),
        ReasoningDomain::Epistemic => (9, None),
        ReasoningDomain::ToolUse => (10, None),
        ReasoningDomain::General => (11, None),
        ReasoningDomain::Custom(name) => (12, Some(name.as_str())),
    };
    hasher.update(&[tag]);
    if let Some(name) = custom {
        hash_str(hasher, name);
    }
}

fn hash_abstention_reason(hasher: &mut blake3::Hasher, reason: &AbstentionReason) {
    let (tag, custom) = match reason {
        AbstentionReason::InsufficientEvidence => (0_u8, None),
        AbstentionReason::Unidentified => (1, None),
        AbstentionReason::OutOfDistribution => (2, None),
        AbstentionReason::ResourceBudgetExceeded => (3, None),
        AbstentionReason::ConflictingEvidence => (4, None),
        AbstentionReason::UnsafeToConclude => (5, None),
        AbstentionReason::Other(name) => (6, Some(name.as_str())),
    };
    hasher.update(&[tag]);
    if let Some(name) = custom {
        hash_str(hasher, name);
    }
}

fn hash_strings(hasher: &mut blake3::Hasher, values: &[String]) {
    hash_usize(hasher, values.len());
    for value in values {
        hash_str(hasher, value);
    }
}

fn hash_option_str(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        None => {
            hasher.update(&[0]);
        }
        Some(value) => {
            hasher.update(&[1]);
            hash_str(hasher, value);
        }
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hash_usize(hasher, value.len());
    hasher.update(value.as_bytes());
}

fn hash_usize(hasher: &mut blake3::Hasher, value: usize) {
    hash_u64(hasher, u64::try_from(value).unwrap_or(u64::MAX));
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_problem() -> ReasoningProblemRef {
        ReasoningProblemRef {
            benchmark: "unit".into(),
            benchmark_version: "v1".into(),
            split: "holdout".into(),
            problem_id: "p-001".into(),
            problem_hash: "sha256:abc".into(),
        }
    }

    fn sample_episode() -> ReasoningEpisode {
        ReasoningEpisode::new(
            "subject-deadbeef",
            "baseline",
            ReasoningDomain::Logic,
            sample_problem(),
            vec![EvidenceRef {
                id: "e1".into(),
                content_hash: "sha256:e1".into(),
                provenance: "benchmark".into(),
                independence_group: Some("source-a".into()),
            }],
            vec![AssumptionRecord {
                id: "a1".into(),
                statement: "closed-world premise".into(),
                confidence: 0.8,
            }],
            vec![ReasoningDecisionRecord {
                operation: "symbolic-check".into(),
                input_refs: vec!["e1".into()],
                output_refs: vec!["answer".into()],
                verifier: Some("truth-table-v1".into()),
            }],
            ReasoningOutcome::Asserted {
                value: "true".into(),
                confidence: 0.9,
            },
            ResourceUsage {
                wall_time_us: 120,
                deliberation_steps: 3,
                tool_calls: 0,
                model_tokens: 0,
            },
        )
        .unwrap_or_else(|err| panic!("sample episode must validate: {err}"))
    }

    fn sample_receipt() -> ReasoningQualificationReceipt {
        ReasoningQualificationReceipt::new(
            &sample_episode(),
            "rq-evaluator",
            "v1",
            "sha256:evaluation-set",
            vec![QualificationMetric {
                name: "exact_accuracy".into(),
                value: 1.0,
                unit: "fraction".into(),
            }],
            Some(true),
        )
        .unwrap_or_else(|err| panic!("sample receipt must validate: {err}"))
    }

    #[test]
    fn episode_identity_is_deterministic() {
        let episode = sample_episode();
        assert_eq!(episode.id(), episode.id());
    }

    #[test]
    fn episode_identity_changes_with_subject_output() {
        let episode = sample_episode();
        let mut changed = episode.clone();
        changed.outcome = ReasoningOutcome::Asserted {
            value: "false".into(),
            confidence: 0.9,
        };
        assert_ne!(episode.id(), changed.id());
    }

    #[test]
    fn rejects_non_finite_and_out_of_range_probabilities() {
        let mut episode = sample_episode();
        episode.outcome = ReasoningOutcome::Asserted {
            value: "true".into(),
            confidence: f64::NAN,
        };
        assert!(matches!(
            episode.validate(),
            Err(QualificationValidationError::InvalidProbability(
                "outcome.confidence"
            ))
        ));

        episode.outcome = ReasoningOutcome::Abstained {
            reason: AbstentionReason::InsufficientEvidence,
            answerability: 1.01,
        };
        assert!(matches!(
            episode.validate(),
            Err(QualificationValidationError::InvalidProbability(
                "outcome.answerability"
            ))
        ));
    }

    #[test]
    fn rejects_duplicate_evidence_identity() {
        let mut episode = sample_episode();
        episode.evidence.push(episode.evidence[0].clone());
        assert!(matches!(
            episode.validate(),
            Err(QualificationValidationError::DuplicateEvidenceId(ref id)) if id == "e1"
        ));
    }

    #[test]
    fn abstention_is_a_first_class_outcome() {
        let mut episode = sample_episode();
        episode.outcome = ReasoningOutcome::Abstained {
            reason: AbstentionReason::Unidentified,
            answerability: 0.2,
        };
        assert!(episode.validate().is_ok());
        assert!(episode.id().is_ok());
    }

    #[test]
    fn receipt_is_deterministic_and_bound_to_episode() {
        let left = sample_receipt();
        let right = sample_receipt();
        assert_eq!(left, right);
        assert!(left.validate().is_ok());
    }

    #[test]
    fn receipt_detects_post_construction_tampering() {
        let mut receipt = sample_receipt();
        receipt.metrics[0].value = 0.0;
        assert_eq!(
            receipt.validate(),
            Err(QualificationValidationError::ReceiptIdMismatch)
        );
    }

    #[test]
    fn receipt_rejects_duplicate_or_non_finite_metrics() {
        let episode = sample_episode();
        let duplicate = ReasoningQualificationReceipt::new(
            &episode,
            "rq-evaluator",
            "v1",
            "lineage",
            vec![
                QualificationMetric {
                    name: "score".into(),
                    value: 0.5,
                    unit: "fraction".into(),
                },
                QualificationMetric {
                    name: "score".into(),
                    value: 0.6,
                    unit: "fraction".into(),
                },
            ],
            None,
        );
        assert!(matches!(
            duplicate,
            Err(QualificationValidationError::DuplicateMetricName(ref name)) if name == "score"
        ));

        let non_finite = ReasoningQualificationReceipt::new(
            &episode,
            "rq-evaluator",
            "v1",
            "lineage",
            vec![QualificationMetric {
                name: "score".into(),
                value: f64::INFINITY,
                unit: "fraction".into(),
            }],
            None,
        );
        assert!(matches!(
            non_finite,
            Err(QualificationValidationError::InvalidMetric(ref name)) if name == "score"
        ));
    }
}
