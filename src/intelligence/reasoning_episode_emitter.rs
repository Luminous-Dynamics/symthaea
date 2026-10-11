// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Automatic publication of canonical reasoning decisions into RQ evidence episodes.
//!
//! This module is intentionally stateless. A validated V2 decision is already content-bound by the
//! kernel; this layer verifies that commitment, verifies exact evidence/assumption/output parity,
//! adds public operation-level provenance, and constructs the canonical `ReasoningEpisode`.
//!
//! It does not persist hidden natural-language chain of thought.

use super::reasoning_context_competition::ContextResolution;
use super::reasoning_kernel_v2::{
    CanonicalReasoningDecisionV2, CANONICAL_REASONING_KERNEL_V2_VERSION,
};
use super::reasoning_qualification::{
    AssumptionRecord, EvidenceRef, QualificationValidationError, ReasoningDecisionRecord,
    ReasoningDomain, ReasoningEpisode, ReasoningEpisodeId, ReasoningOutcome, ReasoningProblemRef,
    ResourceUsage,
};
use std::collections::HashSet;
use std::fmt;

pub const CANONICAL_RQ_EMITTER_VERSION: &str = "rq-006-episode-emitter-v1";
const RESERVED_OPERATION_PREFIX: &str = "canonical-";

#[derive(Debug, Clone)]
pub struct ReasoningEpisodePublicationInput {
    pub subject_revision: String,
    pub configuration_id: String,
    pub domain: ReasoningDomain,
    pub problem: ReasoningProblemRef,
    pub evidence: Vec<EvidenceRef>,
    /// Every explicit assumption in this list is treated as one assumption flagged by the
    /// pre-outcome canonical decision. Counts must match exactly.
    pub assumptions: Vec<AssumptionRecord>,
    /// Public operation summaries produced after the canonical plan was frozen. Reserved
    /// `canonical-*` operation names are rejected to prevent provenance spoofing.
    pub public_operations: Vec<ReasoningDecisionRecord>,
    /// The subject's answer/abstention. This is known before evaluator ground truth is attached.
    pub outcome: ReasoningOutcome,
    pub resources: ResourceUsage,
}

#[derive(Debug, Clone)]
pub struct EmittedReasoningEpisode {
    pub emitter_version: String,
    pub decision_commitment: String,
    pub episode_id: ReasoningEpisodeId,
    pub episode: ReasoningEpisode,
}

pub fn emit_reasoning_episode(
    decision: &CanonicalReasoningDecisionV2,
    publication: ReasoningEpisodePublicationInput,
) -> Result<EmittedReasoningEpisode, EpisodeEmissionError> {
    if decision.kernel_version != CANONICAL_REASONING_KERNEL_V2_VERSION {
        return Err(EpisodeEmissionError::UnsupportedKernelVersion(
            decision.kernel_version.clone(),
        ));
    }
    if !decision.validate_commitment() {
        return Err(EpisodeEmissionError::InvalidDecisionCommitment);
    }

    if publication.evidence.len() != decision.evidence_items {
        return Err(EpisodeEmissionError::EvidenceCountMismatch {
            expected: decision.evidence_items,
            found: publication.evidence.len(),
        });
    }
    if publication.assumptions.len() != decision.weak_assumptions_flagged {
        return Err(EpisodeEmissionError::AssumptionCountMismatch {
            expected: decision.weak_assumptions_flagged,
            found: publication.assumptions.len(),
        });
    }

    let outcome_abstained = matches!(&publication.outcome, ReasoningOutcome::Abstained { .. });
    if outcome_abstained != decision.abstained {
        return Err(EpisodeEmissionError::AbstentionMismatch {
            decision_abstained: decision.abstained,
            publication_abstained: outcome_abstained,
        });
    }

    let evidence_ids = publication
        .evidence
        .iter()
        .map(|evidence| evidence.id.as_str())
        .collect::<HashSet<_>>();
    for hypothesis in &decision.context_selection.assessment.hypotheses {
        for evidence_ref in &hypothesis.evidence_refs {
            if !evidence_ids.contains(evidence_ref.as_str()) {
                return Err(EpisodeEmissionError::MissingContextEvidenceRef(
                    evidence_ref.clone(),
                ));
            }
        }
    }
    drop(evidence_ids);

    for operation in &publication.public_operations {
        let trimmed = operation.operation.trim();
        if trimmed.starts_with(RESERVED_OPERATION_PREFIX) {
            return Err(EpisodeEmissionError::ReservedOperationName(
                operation.operation.clone(),
            ));
        }
    }

    let mut decisions = canonical_decision_records(decision);
    decisions.extend(publication.public_operations);

    let episode = ReasoningEpisode::new(
        publication.subject_revision,
        publication.configuration_id,
        publication.domain,
        publication.problem,
        publication.evidence,
        publication.assumptions,
        decisions,
        publication.outcome,
        publication.resources,
    )?;
    let episode_id = episode.id()?;

    Ok(EmittedReasoningEpisode {
        emitter_version: CANONICAL_RQ_EMITTER_VERSION.into(),
        decision_commitment: decision.decision_commitment.clone(),
        episode_id,
        episode,
    })
}

fn canonical_decision_records(
    decision: &CanonicalReasoningDecisionV2,
) -> Vec<ReasoningDecisionRecord> {
    let assessment = &decision.context_selection.assessment;
    let mut context_inputs = Vec::new();
    let mut seen_refs = HashSet::new();
    for hypothesis in &assessment.hypotheses {
        for evidence_ref in &hypothesis.evidence_refs {
            if seen_refs.insert(evidence_ref.clone()) {
                context_inputs.push(evidence_ref.clone());
            }
        }
    }

    let mut context_outputs = vec![
        format!("primary-context:{:?}", assessment.primary_context),
        format!(
            "context-resolution:{}",
            match assessment.resolution {
                ContextResolution::Resolved => "resolved",
                ContextResolution::Ambiguous => "ambiguous",
            }
        ),
        format!("top-context-support:{:.17}", assessment.top_support),
    ];
    context_outputs.extend(
        assessment
            .active_contexts
            .iter()
            .map(|context| format!("active-context:{context:?}")),
    );

    let candidate_inputs = decision
        .context_selection
        .evaluations
        .iter()
        .map(|evaluation| {
            format!(
                "candidate:{}:{}",
                evaluation.source_index, evaluation.candidate_name
            )
        })
        .collect::<Vec<_>>();

    vec![
        ReasoningDecisionRecord {
            operation: "canonical-context-competition".into(),
            input_refs: context_inputs,
            output_refs: context_outputs,
            verifier: Some(decision.context_selection.evaluator_version.clone()),
        },
        ReasoningDecisionRecord {
            operation: "canonical-robust-objective-selection".into(),
            input_refs: candidate_inputs,
            output_refs: vec![
                format!(
                    "selected-candidate:{}:{}",
                    decision.context_selection.selected_source_index,
                    decision.context_selection.selected_candidate_name
                ),
                format!(
                    "selected-worst-case-score:{:.17}",
                    decision.context_selection.selected_worst_case_score
                ),
                format!("decision-commitment:{}", decision.decision_commitment),
            ],
            verifier: Some(decision.kernel_version.clone()),
        },
        ReasoningDecisionRecord {
            operation: "canonical-meta-state-freeze".into(),
            input_refs: vec![format!("decision-commitment:{}", decision.decision_commitment)],
            output_refs: vec![
                format!("subject:{}", decision.subject_id),
                format!("logical-episode:{}", decision.episode_id),
                format!("sequence:{}", decision.sequence),
                format!("evidence-items:{}", decision.evidence_items),
                format!(
                    "assumptions-flagged:{}",
                    decision.weak_assumptions_flagged
                ),
                format!("abstained:{}", decision.abstained),
            ],
            verifier: Some(decision.kernel_version.clone()),
        },
    ]
}

#[derive(Debug)]
pub enum EpisodeEmissionError {
    UnsupportedKernelVersion(String),
    InvalidDecisionCommitment,
    EvidenceCountMismatch { expected: usize, found: usize },
    AssumptionCountMismatch { expected: usize, found: usize },
    AbstentionMismatch {
        decision_abstained: bool,
        publication_abstained: bool,
    },
    MissingContextEvidenceRef(String),
    ReservedOperationName(String),
    Qualification(QualificationValidationError),
}

impl fmt::Display for EpisodeEmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedKernelVersion(version) => {
                write!(f, "unsupported reasoning kernel version `{version}`")
            }
            Self::InvalidDecisionCommitment => {
                write!(f, "canonical reasoning decision commitment is invalid")
            }
            Self::EvidenceCountMismatch { expected, found } => write!(
                f,
                "publication evidence count mismatch: decision froze {expected}, publication supplied {found}"
            ),
            Self::AssumptionCountMismatch { expected, found } => write!(
                f,
                "publication assumption count mismatch: decision froze {expected}, publication supplied {found}"
            ),
            Self::AbstentionMismatch {
                decision_abstained,
                publication_abstained,
            } => write!(
                f,
                "publication abstention mismatch: decision froze {decision_abstained}, publication supplied {publication_abstained}"
            ),
            Self::MissingContextEvidenceRef(evidence_ref) => write!(
                f,
                "context hypothesis references evidence `{evidence_ref}` absent from the published episode"
            ),
            Self::ReservedOperationName(name) => write!(
                f,
                "caller operation `{name}` uses reserved canonical provenance namespace"
            ),
            Self::Qualification(err) => write!(f, "reasoning episode validation failed: {err}"),
        }
    }
}

impl std::error::Error for EpisodeEmissionError {}

impl From<QualificationValidationError> for EpisodeEmissionError {
    fn from(value: QualificationValidationError) -> Self {
        Self::Qualification(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::context_aware_evolution::ReasoningContext;
    use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
    use crate::consciousness::primitive_evolution::CandidatePrimitive;
    use crate::hdc::BinaryHV;
    use crate::intelligence::{
        CanonicalReasoningInputV2, CanonicalReasoningKernelV2, ContextHypothesis,
    };
    use symthaea_core::hdc::primitive_system::PrimitiveTier;

    fn candidate(name: &str) -> CandidatePrimitive {
        CandidatePrimitive {
            name: name.into(),
            tier: PrimitiveTier::Physical,
            definition: format!("fixture-{name}"),
            fitness: 0.5,
            encoding: BinaryHV::random(name.len() as u64 + 1200),
            epistemic_coordinate: EpistemicCoordinate::axiom(),
            harmonic_alignment: 0.6,
        }
    }

    fn decision(
        abstained: bool,
        evidence_items: usize,
        assumptions: usize,
    ) -> CanonicalReasoningDecisionV2 {
        let mut kernel = CanonicalReasoningKernelV2::new(8).unwrap();
        kernel
            .reason(CanonicalReasoningInputV2 {
                subject_id: "agent-a".into(),
                episode_id: "logical-0".into(),
                sequence: 0,
                context_hypotheses: vec![ContextHypothesis {
                    context: ReasoningContext::TechnicalImplementation,
                    support: 0.9,
                    source: "fixture-context-adapter-v1".into(),
                    evidence_refs: vec!["query".into()],
                }],
                strategy_support: 0.7,
                abstained,
                evidence_items,
                weak_assumptions_flagged: assumptions,
                candidates: vec![candidate("strategy-a")],
            })
            .unwrap()
    }

    fn evidence() -> EvidenceRef {
        EvidenceRef {
            id: "query".into(),
            content_hash: "blake3:fixture-query".into(),
            provenance: "public-development-fixture".into(),
            independence_group: None,
        }
    }

    fn problem() -> ReasoningProblemRef {
        ReasoningProblemRef {
            benchmark: "rq-006-emitter-fixture".into(),
            benchmark_version: "v1".into(),
            split: "development".into(),
            problem_id: "problem-0".into(),
            problem_hash: "blake3:fixture-problem".into(),
        }
    }

    fn asserted_publication() -> ReasoningEpisodePublicationInput {
        ReasoningEpisodePublicationInput {
            subject_revision: "fixture-revision".into(),
            configuration_id: "fixture-config".into(),
            domain: ReasoningDomain::Coding,
            problem: problem(),
            evidence: vec![evidence()],
            assumptions: Vec::new(),
            public_operations: vec![ReasoningDecisionRecord {
                operation: "execute-selected-strategy".into(),
                input_refs: vec!["query".into()],
                output_refs: vec!["answer".into()],
                verifier: Some("fixture-verifier".into()),
            }],
            outcome: ReasoningOutcome::Asserted {
                value: "fixture-answer".into(),
                confidence: 0.7,
            },
            resources: ResourceUsage {
                wall_time_us: 10,
                deliberation_steps: 2,
                tool_calls: 0,
                model_tokens: 0,
            },
        }
    }

    #[test]
    fn valid_decision_emits_content_bound_reasoning_episode() {
        let decision = decision(false, 1, 0);
        let emitted = emit_reasoning_episode(&decision, asserted_publication()).unwrap();
        emitted.episode.validate().unwrap();
        assert_eq!(emitted.decision_commitment, decision.decision_commitment);
        assert_eq!(
            emitted.episode.decisions[0].operation,
            "canonical-context-competition"
        );
        assert_eq!(
            emitted.episode.decisions[1].operation,
            "canonical-robust-objective-selection"
        );
        assert_eq!(
            emitted.episode.decisions[2].operation,
            "canonical-meta-state-freeze"
        );
        assert_eq!(emitted.episode.id().unwrap(), emitted.episode_id);
    }

    #[test]
    fn tampered_decision_is_rejected() {
        let mut decision = decision(false, 1, 0);
        decision.selected_candidate.name = "tampered".into();
        assert!(matches!(
            emit_reasoning_episode(&decision, asserted_publication()),
            Err(EpisodeEmissionError::InvalidDecisionCommitment)
        ));
    }

    #[test]
    fn missing_context_evidence_is_rejected() {
        let decision = decision(false, 1, 0);
        let mut publication = asserted_publication();
        publication.evidence[0].id = "other".into();
        assert!(matches!(
            emit_reasoning_episode(&decision, publication),
            Err(EpisodeEmissionError::MissingContextEvidenceRef(_))
        ));
    }

    #[test]
    fn evidence_count_must_match_frozen_decision() {
        let decision = decision(false, 2, 0);
        assert!(matches!(
            emit_reasoning_episode(&decision, asserted_publication()),
            Err(EpisodeEmissionError::EvidenceCountMismatch { .. })
        ));
    }

    #[test]
    fn assumption_count_must_match_frozen_decision() {
        let decision = decision(false, 1, 1);
        assert!(matches!(
            emit_reasoning_episode(&decision, asserted_publication()),
            Err(EpisodeEmissionError::AssumptionCountMismatch { .. })
        ));
    }

    #[test]
    fn asserted_output_cannot_publish_abstained_decision() {
        let decision = decision(true, 1, 0);
        assert!(matches!(
            emit_reasoning_episode(&decision, asserted_publication()),
            Err(EpisodeEmissionError::AbstentionMismatch { .. })
        ));
    }

    #[test]
    fn caller_cannot_spoof_canonical_operation_namespace() {
        let decision = decision(false, 1, 0);
        let mut publication = asserted_publication();
        publication.public_operations[0].operation = "canonical-fake-verifier".into();
        assert!(matches!(
            emit_reasoning_episode(&decision, publication),
            Err(EpisodeEmissionError::ReservedOperationName(_))
        ));
    }
}
