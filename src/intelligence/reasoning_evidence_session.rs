// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Append-only evidence acquisition and deterministic V3 replanning.
//!
//! `NeedEvidence` is useful only if later measurements can be attached without erasing how the
//! system reached its earlier uncertainty. This module therefore keeps an immutable genesis state
//! plus a hash-chained event ledger. The current evidence view is always reconstructed by replay.
//!
//! Important transition rules:
//! - an observed objective cannot be silently overwritten;
//! - an observation must be explicitly invalidated before replacement;
//! - context support revisions are append-only events rather than in-place history edits;
//! - context retraction is explicit and evidence-referenced;
//! - every replan uses the replayed current view through the canonical V3 planner.

use super::reasoning_context_competition::{
    ContextCompetitionPolicy, ContextHypothesis,
};
use super::reasoning_evidence_seeking::{
    plan_with_evidence, EvidenceSeekingPlanReport, EvidenceSeekingPlannerError,
};
use super::reasoning_objective_core::ObjectiveKind;
use super::reasoning_objective_evidence::{
    CandidateObjectiveEvidence, ObjectiveEvidence, ObjectiveEvidenceError, ObjectiveEvidenceStatus,
    ObjectiveUnknownReason,
};
use crate::consciousness::context_aware_evolution::ReasoningContext;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

pub const EVIDENCE_ACQUISITION_SESSION_SCHEMA_VERSION: u32 = 1;
const GENESIS_DOMAIN: &[u8] = b"symthaea/reasoning/evidence-session/genesis/v1";
const EVENT_DOMAIN: &[u8] = b"symthaea/reasoning/evidence-session/event/v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvidenceAcquisitionEvent {
    ObjectiveObserved {
        candidate_id: String,
        objective: ObjectiveKind,
        evidence: ObjectiveEvidence,
    },
    ObjectiveInvalidated {
        candidate_id: String,
        objective: ObjectiveKind,
        source: String,
        evidence_refs: Vec<String>,
        rationale: String,
    },
    ContextRevised {
        hypothesis: ContextHypothesis,
    },
    ContextRetracted {
        context: ReasoningContext,
        source: String,
        evidence_refs: Vec<String>,
        rationale: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceAcquisitionEventRecord {
    pub sequence: u64,
    pub previous_digest: String,
    pub event_digest: String,
    pub event: EvidenceAcquisitionEvent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceAcquisitionSnapshot {
    pub session_id: String,
    pub next_sequence: u64,
    pub head_digest: String,
    pub hypotheses: Vec<ContextHypothesis>,
    pub candidates: Vec<CandidateObjectiveEvidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceAcquisitionSession {
    pub schema_version: u32,
    pub session_id: String,
    pub context_policy: ContextCompetitionPolicy,
    pub initial_hypotheses: Vec<ContextHypothesis>,
    pub initial_candidates: Vec<CandidateObjectiveEvidence>,
    pub genesis_digest: String,
    pub events: Vec<EvidenceAcquisitionEventRecord>,
}

impl EvidenceAcquisitionSession {
    pub fn new(
        session_id: impl Into<String>,
        context_policy: ContextCompetitionPolicy,
        initial_hypotheses: Vec<ContextHypothesis>,
        initial_candidates: Vec<CandidateObjectiveEvidence>,
    ) -> Result<Self, EvidenceAcquisitionError> {
        let session_id = session_id.into();
        if session_id.trim().is_empty() {
            return Err(EvidenceAcquisitionError::EmptyField("session_id"));
        }

        // The V3 planner is also the canonical admission validator. Missing/weak evidence may
        // legitimately produce NeedEvidence; malformed evidence still returns an error.
        plan_with_evidence(
            &initial_hypotheses,
            context_policy,
            &initial_candidates,
        )?;

        let mut session = Self {
            schema_version: EVIDENCE_ACQUISITION_SESSION_SCHEMA_VERSION,
            session_id,
            context_policy,
            initial_hypotheses,
            initial_candidates,
            genesis_digest: String::new(),
            events: Vec::new(),
        };
        session.genesis_digest = compute_genesis_digest(&session);
        Ok(session)
    }

    pub fn validate_chain(&self) -> Result<(), EvidenceAcquisitionError> {
        if self.schema_version != EVIDENCE_ACQUISITION_SESSION_SCHEMA_VERSION {
            return Err(EvidenceAcquisitionError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if self.session_id.trim().is_empty() {
            return Err(EvidenceAcquisitionError::EmptyField("session_id"));
        }
        plan_with_evidence(
            &self.initial_hypotheses,
            self.context_policy,
            &self.initial_candidates,
        )?;

        let expected_genesis = compute_genesis_digest(self);
        if self.genesis_digest != expected_genesis {
            return Err(EvidenceAcquisitionError::GenesisDigestMismatch);
        }

        let mut previous_digest = self.genesis_digest.clone();
        let mut expected_sequence = 0u64;
        let mut state = ReplayState {
            hypotheses: self.initial_hypotheses.clone(),
            candidates: self.initial_candidates.clone(),
        };

        for record in &self.events {
            if record.sequence != expected_sequence {
                return Err(EvidenceAcquisitionError::SequenceMismatch {
                    expected: expected_sequence,
                    found: record.sequence,
                });
            }
            if record.previous_digest != previous_digest {
                return Err(EvidenceAcquisitionError::PreviousDigestMismatch {
                    sequence: record.sequence,
                });
            }
            let expected_digest = compute_event_digest(
                &self.session_id,
                record.sequence,
                &record.previous_digest,
                &record.event,
            );
            if record.event_digest != expected_digest {
                return Err(EvidenceAcquisitionError::EventDigestMismatch {
                    sequence: record.sequence,
                });
            }
            apply_event(&mut state, &record.event, self.context_policy)?;
            previous_digest = record.event_digest.clone();
            expected_sequence = expected_sequence
                .checked_add(1)
                .ok_or(EvidenceAcquisitionError::SequenceOverflow)?;
        }
        Ok(())
    }

    pub fn snapshot(&self) -> Result<EvidenceAcquisitionSnapshot, EvidenceAcquisitionError> {
        let state = self.replay()?;
        Ok(EvidenceAcquisitionSnapshot {
            session_id: self.session_id.clone(),
            next_sequence: self.events.len() as u64,
            head_digest: self.head_digest().to_owned(),
            hypotheses: state.hypotheses,
            candidates: state.candidates,
        })
    }

    pub fn plan(&self) -> Result<EvidenceSeekingPlanReport, EvidenceAcquisitionError> {
        let state = self.replay()?;
        Ok(plan_with_evidence(
            &state.hypotheses,
            self.context_policy,
            &state.candidates,
        )?)
    }

    pub fn head_digest(&self) -> &str {
        self.events
            .last()
            .map(|record| record.event_digest.as_str())
            .unwrap_or(self.genesis_digest.as_str())
    }

    pub fn next_sequence(&self) -> u64 {
        self.events.len() as u64
    }

    pub fn append(
        &mut self,
        event: EvidenceAcquisitionEvent,
    ) -> Result<&EvidenceAcquisitionEventRecord, EvidenceAcquisitionError> {
        // Validate the existing ledger before extending it. A caller cannot append onto a
        // corrupted persisted session and thereby create a new apparently-valid suffix.
        self.validate_chain()?;
        let mut state = self.replay_unchecked()?;
        apply_event(&mut state, &event, self.context_policy)?;

        let sequence = u64::try_from(self.events.len())
            .map_err(|_| EvidenceAcquisitionError::SequenceOverflow)?;
        let previous_digest = self.head_digest().to_owned();
        let event_digest = compute_event_digest(
            &self.session_id,
            sequence,
            &previous_digest,
            &event,
        );
        self.events.push(EvidenceAcquisitionEventRecord {
            sequence,
            previous_digest,
            event_digest,
            event,
        });
        Ok(self
            .events
            .last()
            .expect("event was pushed immediately before lookup"))
    }

    pub fn observe_objective(
        &mut self,
        candidate_id: impl Into<String>,
        objective: ObjectiveKind,
        evidence: ObjectiveEvidence,
    ) -> Result<&EvidenceAcquisitionEventRecord, EvidenceAcquisitionError> {
        self.append(EvidenceAcquisitionEvent::ObjectiveObserved {
            candidate_id: candidate_id.into(),
            objective,
            evidence,
        })
    }

    pub fn invalidate_objective(
        &mut self,
        candidate_id: impl Into<String>,
        objective: ObjectiveKind,
        source: impl Into<String>,
        evidence_refs: Vec<String>,
        rationale: impl Into<String>,
    ) -> Result<&EvidenceAcquisitionEventRecord, EvidenceAcquisitionError> {
        self.append(EvidenceAcquisitionEvent::ObjectiveInvalidated {
            candidate_id: candidate_id.into(),
            objective,
            source: source.into(),
            evidence_refs,
            rationale: rationale.into(),
        })
    }

    pub fn revise_context(
        &mut self,
        hypothesis: ContextHypothesis,
    ) -> Result<&EvidenceAcquisitionEventRecord, EvidenceAcquisitionError> {
        self.append(EvidenceAcquisitionEvent::ContextRevised { hypothesis })
    }

    pub fn retract_context(
        &mut self,
        context: ReasoningContext,
        source: impl Into<String>,
        evidence_refs: Vec<String>,
        rationale: impl Into<String>,
    ) -> Result<&EvidenceAcquisitionEventRecord, EvidenceAcquisitionError> {
        self.append(EvidenceAcquisitionEvent::ContextRetracted {
            context,
            source: source.into(),
            evidence_refs,
            rationale: rationale.into(),
        })
    }

    fn replay(&self) -> Result<ReplayState, EvidenceAcquisitionError> {
        self.validate_chain()?;
        self.replay_unchecked()
    }

    fn replay_unchecked(&self) -> Result<ReplayState, EvidenceAcquisitionError> {
        let mut state = ReplayState {
            hypotheses: self.initial_hypotheses.clone(),
            candidates: self.initial_candidates.clone(),
        };
        for record in &self.events {
            apply_event(&mut state, &record.event, self.context_policy)?;
        }
        Ok(state)
    }
}

#[derive(Debug, Clone)]
struct ReplayState {
    hypotheses: Vec<ContextHypothesis>,
    candidates: Vec<CandidateObjectiveEvidence>,
}

fn apply_event(
    state: &mut ReplayState,
    event: &EvidenceAcquisitionEvent,
    context_policy: ContextCompetitionPolicy,
) -> Result<(), EvidenceAcquisitionError> {
    match event {
        EvidenceAcquisitionEvent::ObjectiveObserved {
            candidate_id,
            objective,
            evidence,
        } => {
            require_nonempty("candidate_id", candidate_id)?;
            evidence.validate()?;
            if !matches!(evidence.status, ObjectiveEvidenceStatus::Observed { .. }) {
                return Err(EvidenceAcquisitionError::ObservationMustBeObserved);
            }
            let candidate = candidate_mut(&mut state.candidates, candidate_id)?;
            let current = objective_mut(candidate, *objective);
            if matches!(current.status, ObjectiveEvidenceStatus::Observed { .. }) {
                return Err(EvidenceAcquisitionError::ObjectiveAlreadyObserved {
                    candidate_id: candidate_id.clone(),
                    objective: *objective,
                });
            }
            *current = evidence.clone();
        }
        EvidenceAcquisitionEvent::ObjectiveInvalidated {
            candidate_id,
            objective,
            source,
            evidence_refs,
            rationale,
        } => {
            require_nonempty("candidate_id", candidate_id)?;
            require_nonempty("invalidation.source", source)?;
            require_nonempty("invalidation.rationale", rationale)?;
            validate_references("invalidation", evidence_refs)?;
            let candidate = candidate_mut(&mut state.candidates, candidate_id)?;
            let current = objective_mut(candidate, *objective);
            if !matches!(current.status, ObjectiveEvidenceStatus::Observed { .. }) {
                return Err(EvidenceAcquisitionError::ObjectiveNotObserved {
                    candidate_id: candidate_id.clone(),
                    objective: *objective,
                });
            }
            *current = ObjectiveEvidence::unknown(
                source.clone(),
                evidence_refs.clone(),
                ObjectiveUnknownReason::Invalidated,
            )?;
        }
        EvidenceAcquisitionEvent::ContextRevised { hypothesis } => {
            let mut proposed = state.hypotheses.clone();
            if let Some(existing) = proposed
                .iter_mut()
                .find(|existing| existing.context == hypothesis.context)
            {
                *existing = hypothesis.clone();
            } else {
                proposed.push(hypothesis.clone());
            }
            // V3 accepts low support as NeedEvidence while still validating provenance and bounds.
            plan_with_evidence(&proposed, context_policy, &state.candidates)?;
            state.hypotheses = proposed;
        }
        EvidenceAcquisitionEvent::ContextRetracted {
            context,
            source,
            evidence_refs,
            rationale,
        } => {
            require_nonempty("retraction.source", source)?;
            require_nonempty("retraction.rationale", rationale)?;
            validate_references("retraction", evidence_refs)?;
            let before = state.hypotheses.len();
            state.hypotheses.retain(|hypothesis| hypothesis.context != *context);
            if state.hypotheses.len() == before {
                return Err(EvidenceAcquisitionError::UnknownContext(*context));
            }
            // An empty context set is legitimate and deterministically replans to NeedEvidence.
            plan_with_evidence(&state.hypotheses, context_policy, &state.candidates)?;
        }
    }
    Ok(())
}

fn candidate_mut<'a>(
    candidates: &'a mut [CandidateObjectiveEvidence],
    candidate_id: &str,
) -> Result<&'a mut CandidateObjectiveEvidence, EvidenceAcquisitionError> {
    candidates
        .iter_mut()
        .find(|candidate| candidate.candidate_id == candidate_id)
        .ok_or_else(|| EvidenceAcquisitionError::UnknownCandidate(candidate_id.to_owned()))
}

fn objective_mut(
    candidate: &mut CandidateObjectiveEvidence,
    objective: ObjectiveKind,
) -> &mut ObjectiveEvidence {
    match objective {
        ObjectiveKind::IntegrationProxy => &mut candidate.integration_proxy,
        ObjectiveKind::HarmonicAlignment => &mut candidate.harmonic_alignment,
        ObjectiveKind::EpistemicGrounding => &mut candidate.epistemic_grounding,
    }
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), EvidenceAcquisitionError> {
    if value.trim().is_empty() {
        Err(EvidenceAcquisitionError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_references(
    prefix: &'static str,
    references: &[String],
) -> Result<(), EvidenceAcquisitionError> {
    if references.is_empty() {
        return Err(EvidenceAcquisitionError::MissingEvidenceReferences(prefix));
    }
    let mut seen = HashSet::with_capacity(references.len());
    for reference in references {
        if reference.trim().is_empty() {
            return Err(EvidenceAcquisitionError::EmptyEvidenceReference(prefix));
        }
        if !seen.insert(reference.as_str()) {
            return Err(EvidenceAcquisitionError::DuplicateEvidenceReference(
                reference.clone(),
            ));
        }
    }
    Ok(())
}

fn compute_genesis_digest(session: &EvidenceAcquisitionSession) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, GENESIS_DOMAIN);
    hash_u64(&mut hasher, u64::from(session.schema_version));
    hash_str(&mut hasher, &session.session_id);
    hash_u64(&mut hasher, session.context_policy.minimum_support.to_bits());
    hash_u64(&mut hasher, session.context_policy.ambiguity_band.to_bits());
    hash_u64(&mut hasher, session.initial_hypotheses.len() as u64);
    for hypothesis in &session.initial_hypotheses {
        hash_hypothesis(&mut hasher, hypothesis);
    }
    hash_u64(&mut hasher, session.initial_candidates.len() as u64);
    for candidate in &session.initial_candidates {
        hash_candidate(&mut hasher, candidate);
    }
    hasher.finalize().to_hex().to_string()
}

fn compute_event_digest(
    session_id: &str,
    sequence: u64,
    previous_digest: &str,
    event: &EvidenceAcquisitionEvent,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, EVENT_DOMAIN);
    hash_str(&mut hasher, session_id);
    hash_u64(&mut hasher, sequence);
    hash_str(&mut hasher, previous_digest);
    match event {
        EvidenceAcquisitionEvent::ObjectiveObserved {
            candidate_id,
            objective,
            evidence,
        } => {
            hash_u64(&mut hasher, 0);
            hash_str(&mut hasher, candidate_id);
            hash_objective_kind(&mut hasher, *objective);
            hash_objective_evidence(&mut hasher, evidence);
        }
        EvidenceAcquisitionEvent::ObjectiveInvalidated {
            candidate_id,
            objective,
            source,
            evidence_refs,
            rationale,
        } => {
            hash_u64(&mut hasher, 1);
            hash_str(&mut hasher, candidate_id);
            hash_objective_kind(&mut hasher, *objective);
            hash_str(&mut hasher, source);
            hash_strings(&mut hasher, evidence_refs);
            hash_str(&mut hasher, rationale);
        }
        EvidenceAcquisitionEvent::ContextRevised { hypothesis } => {
            hash_u64(&mut hasher, 2);
            hash_hypothesis(&mut hasher, hypothesis);
        }
        EvidenceAcquisitionEvent::ContextRetracted {
            context,
            source,
            evidence_refs,
            rationale,
        } => {
            hash_u64(&mut hasher, 3);
            hash_context(&mut hasher, *context);
            hash_str(&mut hasher, source);
            hash_strings(&mut hasher, evidence_refs);
            hash_str(&mut hasher, rationale);
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn hash_candidate(hasher: &mut blake3::Hasher, candidate: &CandidateObjectiveEvidence) {
    hash_str(hasher, &candidate.candidate_id);
    hash_str(hasher, &candidate.candidate_label);
    hash_objective_evidence(hasher, &candidate.integration_proxy);
    hash_objective_evidence(hasher, &candidate.harmonic_alignment);
    hash_objective_evidence(hasher, &candidate.epistemic_grounding);
}

fn hash_objective_evidence(hasher: &mut blake3::Hasher, evidence: &ObjectiveEvidence) {
    hash_str(hasher, &evidence.source);
    hash_strings(hasher, &evidence.evidence_refs);
    match evidence.status {
        ObjectiveEvidenceStatus::Observed { value } => {
            hash_u64(hasher, 0);
            hash_u64(hasher, value.to_bits());
        }
        ObjectiveEvidenceStatus::Unknown { reason } => {
            hash_u64(hasher, 1);
            hash_u64(
                hasher,
                match reason {
                    ObjectiveUnknownReason::NotMeasured => 0,
                    ObjectiveUnknownReason::Unavailable => 1,
                    ObjectiveUnknownReason::Invalidated => 2,
                },
            );
        }
    }
}

fn hash_hypothesis(hasher: &mut blake3::Hasher, hypothesis: &ContextHypothesis) {
    hash_context(hasher, hypothesis.context);
    hash_u64(hasher, hypothesis.support.to_bits());
    hash_str(hasher, &hypothesis.source);
    hash_strings(hasher, &hypothesis.evidence_refs);
}

fn hash_context(hasher: &mut blake3::Hasher, context: ReasoningContext) {
    hash_u64(
        hasher,
        match context {
            ReasoningContext::CriticalSafety => 0,
            ReasoningContext::ScientificReasoning => 1,
            ReasoningContext::TechnicalImplementation => 2,
            ReasoningContext::Learning => 3,
            ReasoningContext::SocialInteraction => 4,
            ReasoningContext::PhilosophicalInquiry => 5,
            ReasoningContext::CreativeExploration => 6,
            ReasoningContext::GeneralReasoning => 7,
        },
    );
}

fn hash_objective_kind(hasher: &mut blake3::Hasher, objective: ObjectiveKind) {
    hash_u64(
        hasher,
        match objective {
            ObjectiveKind::IntegrationProxy => 0,
            ObjectiveKind::HarmonicAlignment => 1,
            ObjectiveKind::EpistemicGrounding => 2,
        },
    );
}

fn hash_strings(hasher: &mut blake3::Hasher, values: &[String]) {
    hash_u64(hasher, values.len() as u64);
    for value in values {
        hash_str(hasher, value);
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hash_bytes(hasher, value.as_bytes());
}

fn hash_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value);
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

#[derive(Debug)]
pub enum EvidenceAcquisitionError {
    EmptyField(&'static str),
    UnsupportedSchemaVersion(u32),
    GenesisDigestMismatch,
    SequenceMismatch { expected: u64, found: u64 },
    SequenceOverflow,
    PreviousDigestMismatch { sequence: u64 },
    EventDigestMismatch { sequence: u64 },
    UnknownCandidate(String),
    UnknownContext(ReasoningContext),
    ObservationMustBeObserved,
    ObjectiveAlreadyObserved {
        candidate_id: String,
        objective: ObjectiveKind,
    },
    ObjectiveNotObserved {
        candidate_id: String,
        objective: ObjectiveKind,
    },
    MissingEvidenceReferences(&'static str),
    EmptyEvidenceReference(&'static str),
    DuplicateEvidenceReference(String),
    Objective(ObjectiveEvidenceError),
    Planner(EvidenceSeekingPlannerError),
}

impl fmt::Display for EvidenceAcquisitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field `{field}` is empty"),
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported evidence-acquisition session schema version {version}")
            }
            Self::GenesisDigestMismatch => write!(f, "evidence-acquisition genesis digest mismatch"),
            Self::SequenceMismatch { expected, found } => {
                write!(f, "evidence event sequence mismatch: expected {expected}, found {found}")
            }
            Self::SequenceOverflow => write!(f, "evidence event sequence overflow"),
            Self::PreviousDigestMismatch { sequence } => {
                write!(f, "evidence event {sequence} previous digest mismatch")
            }
            Self::EventDigestMismatch { sequence } => {
                write!(f, "evidence event {sequence} content digest mismatch")
            }
            Self::UnknownCandidate(id) => write!(f, "unknown evidence candidate `{id}`"),
            Self::UnknownContext(context) => write!(f, "cannot retract absent context {context:?}"),
            Self::ObservationMustBeObserved => {
                write!(f, "objective observation event must contain observed evidence")
            }
            Self::ObjectiveAlreadyObserved {
                candidate_id,
                objective,
            } => write!(
                f,
                "candidate `{candidate_id}` objective {objective:?} is already observed; invalidate it before replacement"
            ),
            Self::ObjectiveNotObserved {
                candidate_id,
                objective,
            } => write!(
                f,
                "candidate `{candidate_id}` objective {objective:?} is not currently observed"
            ),
            Self::MissingEvidenceReferences(prefix) => {
                write!(f, "{prefix} event requires at least one evidence reference")
            }
            Self::EmptyEvidenceReference(prefix) => {
                write!(f, "{prefix} event contains an empty evidence reference")
            }
            Self::DuplicateEvidenceReference(reference) => {
                write!(f, "evidence reference `{reference}` is duplicated")
            }
            Self::Objective(err) => write!(f, "objective evidence error: {err}"),
            Self::Planner(err) => write!(f, "evidence replanning error: {err}"),
        }
    }
}

impl std::error::Error for EvidenceAcquisitionError {}

impl From<ObjectiveEvidenceError> for EvidenceAcquisitionError {
    fn from(value: ObjectiveEvidenceError) -> Self {
        Self::Objective(value)
    }
}

impl From<EvidenceSeekingPlannerError> for EvidenceAcquisitionError {
    fn from(value: EvidenceSeekingPlannerError) -> Self {
        Self::Planner(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::reasoning_evidence_seeking::{
        EvidenceRequestKind, EvidenceSeekingOutcome,
    };

    fn context(support: f64) -> ContextHypothesis {
        ContextHypothesis {
            context: ReasoningContext::GeneralReasoning,
            support,
            source: "fixture-context".into(),
            evidence_refs: vec!["query".into()],
        }
    }

    fn unknown(id: &str) -> CandidateObjectiveEvidence {
        let unknown_axis = || {
            ObjectiveEvidence::unknown(
                "fixture-objective",
                Vec::new(),
                ObjectiveUnknownReason::NotMeasured,
            )
            .unwrap()
        };
        CandidateObjectiveEvidence {
            candidate_id: id.into(),
            candidate_label: id.into(),
            integration_proxy: unknown_axis(),
            harmonic_alignment: unknown_axis(),
            epistemic_grounding: unknown_axis(),
        }
    }

    fn observed(value: f64, id: &str) -> ObjectiveEvidence {
        ObjectiveEvidence::observed("fixture-measurement", vec![id.into()], value).unwrap()
    }

    fn session() -> EvidenceAcquisitionSession {
        EvidenceAcquisitionSession::new(
            "fixture-session",
            ContextCompetitionPolicy::development_v1(),
            vec![context(0.9)],
            vec![unknown("a"), unknown("b")],
        )
        .unwrap()
    }

    fn observe_all(session: &mut EvidenceAcquisitionSession, candidate: &str, value: f64) {
        for (objective, suffix) in [
            (ObjectiveKind::IntegrationProxy, "i"),
            (ObjectiveKind::HarmonicAlignment, "h"),
            (ObjectiveKind::EpistemicGrounding, "e"),
        ] {
            session
                .observe_objective(
                    candidate,
                    objective,
                    observed(value, &format!("{candidate}-{suffix}")),
                )
                .unwrap();
        }
    }

    #[test]
    fn need_evidence_can_progress_to_identified_selection() {
        let mut session = session();
        assert!(matches!(
            session.plan().unwrap().outcome,
            EvidenceSeekingOutcome::NeedEvidence { .. }
        ));

        observe_all(&mut session, "a", 0.9);
        observe_all(&mut session, "b", 0.1);

        assert!(matches!(
            session.plan().unwrap().outcome,
            EvidenceSeekingOutcome::Selected { ref candidate_id, .. } if candidate_id == "a"
        ));
        assert_eq!(session.next_sequence(), 6);
        session.validate_chain().unwrap();
    }

    #[test]
    fn observed_value_cannot_be_silently_overwritten() {
        let mut session = session();
        session
            .observe_objective(
                "a",
                ObjectiveKind::IntegrationProxy,
                observed(0.7, "a-i-1"),
            )
            .unwrap();
        let err = session
            .observe_objective(
                "a",
                ObjectiveKind::IntegrationProxy,
                observed(0.8, "a-i-2"),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            EvidenceAcquisitionError::ObjectiveAlreadyObserved { .. }
        ));
        assert_eq!(session.next_sequence(), 1);
    }

    #[test]
    fn explicit_invalidation_allows_later_remeasurement() {
        let mut session = session();
        session
            .observe_objective(
                "a",
                ObjectiveKind::IntegrationProxy,
                observed(0.7, "a-i-1"),
            )
            .unwrap();
        session
            .invalidate_objective(
                "a",
                ObjectiveKind::IntegrationProxy,
                "audit-v1",
                vec!["audit-evidence".into()],
                "measurement calibration was invalid",
            )
            .unwrap();
        let snapshot = session.snapshot().unwrap();
        assert!(matches!(
            snapshot.candidates[0].integration_proxy.status,
            ObjectiveEvidenceStatus::Unknown {
                reason: ObjectiveUnknownReason::Invalidated
            }
        ));
        session
            .observe_objective(
                "a",
                ObjectiveKind::IntegrationProxy,
                observed(0.8, "a-i-2"),
            )
            .unwrap();
        assert_eq!(session.next_sequence(), 3);
        session.validate_chain().unwrap();
    }

    #[test]
    fn weak_context_can_be_revised_without_erasing_history() {
        let mut session = EvidenceAcquisitionSession::new(
            "context-session",
            ContextCompetitionPolicy::development_v1(),
            vec![context(0.4)],
            vec![CandidateObjectiveEvidence {
                candidate_id: "only".into(),
                candidate_label: "only".into(),
                integration_proxy: observed(0.8, "only-i"),
                harmonic_alignment: observed(0.8, "only-h"),
                epistemic_grounding: observed(0.8, "only-e"),
            }],
        )
        .unwrap();
        let first = session.plan().unwrap();
        assert!(matches!(
            first.outcome,
            EvidenceSeekingOutcome::NeedEvidence { ref requests, .. }
                if matches!(requests[0].kind, EvidenceRequestKind::ContextSupport { .. })
        ));

        session.revise_context(context(0.9)).unwrap();
        assert!(matches!(
            session.plan().unwrap().outcome,
            EvidenceSeekingOutcome::Selected { .. }
        ));
        assert_eq!(session.events.len(), 1);
        assert_eq!(session.initial_hypotheses[0].support, 0.4);
    }

    #[test]
    fn tampered_event_is_detected_before_replanning() {
        let mut session = session();
        session
            .observe_objective(
                "a",
                ObjectiveKind::IntegrationProxy,
                observed(0.7, "a-i"),
            )
            .unwrap();
        if let EvidenceAcquisitionEvent::ObjectiveObserved { evidence, .. } =
            &mut session.events[0].event
        {
            evidence.status = ObjectiveEvidenceStatus::Observed { value: 0.99 };
        }
        assert!(matches!(
            session.validate_chain(),
            Err(EvidenceAcquisitionError::EventDigestMismatch { sequence: 0 })
        ));
        assert!(session.plan().is_err());
    }

    #[test]
    fn identical_event_streams_have_identical_heads() {
        let mut left = session();
        let mut right = session();
        for target in [&mut left, &mut right] {
            target
                .observe_objective(
                    "a",
                    ObjectiveKind::IntegrationProxy,
                    observed(0.7, "a-i"),
                )
                .unwrap();
            target
                .invalidate_objective(
                    "a",
                    ObjectiveKind::IntegrationProxy,
                    "audit-v1",
                    vec!["audit-evidence".into()],
                    "invalid calibration",
                )
                .unwrap();
        }
        assert_eq!(left.genesis_digest, right.genesis_digest);
        assert_eq!(left.head_digest(), right.head_digest());
        assert_eq!(left.events, right.events);
    }
}
