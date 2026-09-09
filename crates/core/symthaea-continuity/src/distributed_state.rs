// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact distributed-state transaction context and untrusted participant claims.
//!
//! This module binds a proposed participant transition set to one exact distributed
//! change-budget world and gives verifier adapters a transport claim for participant
//! state. Raw claims are never trusted evidence: they carry no evidence class and
//! cannot directly become a distributed transition witness.
//!
//! Core theorem:
//!
//! `DistributedStateContextV1 != ParticipantStateClaimV1 != AuthenticatedParticipantStateEvidence != DistributedTransitionWitness`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed::{DistributedChangeBudgetId, ValidatedDistributedChangeBudgetV1};
use crate::scope::ContinuitySubjectId;
use crate::verifier::{VerificationAdmissionError, VerifierProfileId, VerifierProfileV1};

pub const DISTRIBUTED_STATE_CONTEXT_SCHEMA_V1: &str =
    "symthaea-continuity-distributed-state-context-v1";
pub const PARTICIPANT_STATE_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-participant-state-claim-v1";

const PARTICIPANT_SET_DOMAIN: &[u8] = b"symthaea.continuity.distributed-participant-set.v1\0";
const CONTEXT_DOMAIN: &[u8] = b"symthaea.continuity.distributed-state-context.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.participant-state-claim.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-participant-state.v1\0";
const MAX_CANDIDATES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ParticipantSetDigest([u8; 32]);

impl ParticipantSetDigest {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DistributedStateContextId([u8; 32]);

impl DistributedStateContextId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ParticipantStateClaimId([u8; 32]);

impl ParticipantStateClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthenticatedParticipantStateEvidenceId([u8; 32]);

impl AuthenticatedParticipantStateEvidenceId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Exact proposed change world for one distributed-state verification transaction.
///
/// The transaction challenge is replay-resistant context, not authorization. The
/// context is serializable configuration/reference material and must be rebound to
/// the exact validated budget before verifier evidence is admitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistributedStateContextV1 {
    schema_version: String,
    aggregate_subject_id: ContinuitySubjectId,
    budget_id: DistributedChangeBudgetId,
    budget_generation: u64,
    participant_set_digest: ParticipantSetDigest,
    candidate_subject_ids: Vec<ContinuitySubjectId>,
    transaction_challenge: [u8; 32],
    context_id: DistributedStateContextId,
}

impl DistributedStateContextV1 {
    pub fn new(
        budget: &ValidatedDistributedChangeBudgetV1,
        mut candidate_subject_ids: Vec<ContinuitySubjectId>,
        transaction_challenge: [u8; 32],
    ) -> Result<Self, DistributedStateError> {
        if transaction_challenge == [0; 32] {
            return Err(DistributedStateError::ZeroTransactionChallenge);
        }
        canonicalize_candidates(&mut candidate_subject_ids)?;
        validate_candidates_against_budget(&candidate_subject_ids, budget)?;

        let aggregate_subject_id = budget.aggregate_subject_id();
        let budget_id = budget.id();
        let budget_generation = budget.generation();
        let participant_set_digest = participant_set_digest(budget.participant_subject_ids());
        let context_id = DistributedStateContextId(hash_context(
            aggregate_subject_id,
            budget_id,
            budget_generation,
            participant_set_digest,
            &candidate_subject_ids,
            transaction_challenge,
        ));

        Ok(Self {
            schema_version: DISTRIBUTED_STATE_CONTEXT_SCHEMA_V1.to_owned(),
            aggregate_subject_id,
            budget_id,
            budget_generation,
            participant_set_digest,
            candidate_subject_ids,
            transaction_challenge,
            context_id,
        })
    }

    pub fn validate(&self) -> Result<(), DistributedStateError> {
        if self.schema_version != DISTRIBUTED_STATE_CONTEXT_SCHEMA_V1 {
            return Err(DistributedStateError::UnsupportedContextSchema(
                self.schema_version.clone(),
            ));
        }
        if self.budget_generation == 0 {
            return Err(DistributedStateError::ZeroBudgetGeneration);
        }
        if self.transaction_challenge == [0; 32] {
            return Err(DistributedStateError::ZeroTransactionChallenge);
        }
        validate_canonical_candidates(&self.candidate_subject_ids)?;

        let expected = DistributedStateContextId(hash_context(
            self.aggregate_subject_id,
            self.budget_id,
            self.budget_generation,
            self.participant_set_digest,
            &self.candidate_subject_ids,
            self.transaction_challenge,
        ));
        if expected != self.context_id {
            return Err(DistributedStateError::ContextIdentityMismatch);
        }
        Ok(())
    }

    pub fn validate_against_budget(
        &self,
        budget: &ValidatedDistributedChangeBudgetV1,
    ) -> Result<ValidatedDistributedStateContextV1, DistributedStateError> {
        self.validate()?;
        if self.aggregate_subject_id != budget.aggregate_subject_id() {
            return Err(DistributedStateError::AggregateSubjectMismatch);
        }
        if self.budget_id != budget.id() {
            return Err(DistributedStateError::BudgetIdentityMismatch);
        }
        if self.budget_generation != budget.generation() {
            return Err(DistributedStateError::BudgetGenerationMismatch);
        }
        if self.participant_set_digest
            != participant_set_digest(budget.participant_subject_ids())
        {
            return Err(DistributedStateError::ParticipantSetMismatch);
        }
        validate_candidates_against_budget(&self.candidate_subject_ids, budget)?;

        Ok(ValidatedDistributedStateContextV1 {
            budget: budget.clone(),
            inner: self.clone(),
        })
    }

    pub fn id(&self) -> DistributedStateContextId {
        self.context_id
    }

    pub fn aggregate_subject_id(&self) -> ContinuitySubjectId {
        self.aggregate_subject_id
    }

    pub fn budget_id(&self) -> DistributedChangeBudgetId {
        self.budget_id
    }

    pub fn budget_generation(&self) -> u64 {
        self.budget_generation
    }

    pub fn participant_set_digest(&self) -> ParticipantSetDigest {
        self.participant_set_digest
    }

    pub fn candidate_subject_ids(&self) -> &[ContinuitySubjectId] {
        &self.candidate_subject_ids
    }

    pub fn transaction_challenge(&self) -> [u8; 32] {
        self.transaction_challenge
    }
}

/// Non-Serde state transaction context rebound to one exact validated budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedDistributedStateContextV1 {
    budget: ValidatedDistributedChangeBudgetV1,
    inner: DistributedStateContextV1,
}

impl ValidatedDistributedStateContextV1 {
    pub fn id(&self) -> DistributedStateContextId {
        self.inner.id()
    }

    pub fn budget(&self) -> &ValidatedDistributedChangeBudgetV1 {
        &self.budget
    }

    pub fn aggregate_subject_id(&self) -> ContinuitySubjectId {
        self.inner.aggregate_subject_id()
    }

    pub fn budget_id(&self) -> DistributedChangeBudgetId {
        self.inner.budget_id()
    }

    pub fn budget_generation(&self) -> u64 {
        self.inner.budget_generation()
    }

    pub fn participant_set_digest(&self) -> ParticipantSetDigest {
        self.inner.participant_set_digest()
    }

    pub fn candidate_subject_ids(&self) -> &[ContinuitySubjectId] {
        self.inner.candidate_subject_ids()
    }

    pub fn transaction_challenge(&self) -> [u8; 32] {
        self.inner.transaction_challenge()
    }

    pub fn as_raw(&self) -> &DistributedStateContextV1 {
        &self.inner
    }
}

/// Protocol-neutral coarse state reported by a verifier adapter.
///
/// The opaque protocol-state digest remains separate so a dedicated adapter can
/// bind exact Raft/database/storage/network state without teaching the generic
/// continuity kernel those protocol semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParticipantOperationalStateV1 {
    Healthy,
    Unhealthy,
    Unknown,
    Transitioning,
}

impl ParticipantOperationalStateV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Healthy => 1,
            Self::Unhealthy => 2,
            Self::Unknown => 3,
            Self::Transitioning => 4,
        }
    }
}

/// Transportable, untrusted participant-state claim.
///
/// Raw callers can construct and serialize this value, so its `state` is only a
/// claim. Evidence strength is deliberately absent. A provisioned verifier profile
/// owns evidence strength, and a later authentication adapter must authenticate the
/// exact policy-checked claim before it can enter distributed witness composition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticipantStateClaimV1 {
    schema_version: String,
    context_id: DistributedStateContextId,
    participant_subject_id: ContinuitySubjectId,
    verifier_profile_id: VerifierProfileId,
    observed_at_unix_ms: u64,
    state: ParticipantOperationalStateV1,
    protocol_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
    claim_id: ParticipantStateClaimId,
}

impl ParticipantStateClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &ValidatedDistributedStateContextV1,
        participant_subject_id: ContinuitySubjectId,
        verifier_profile_id: VerifierProfileId,
        observed_at_unix_ms: u64,
        state: ParticipantOperationalStateV1,
        protocol_state_digest: Option<[u8; 32]>,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, DistributedStateError> {
        if context
            .budget()
            .participant_subject_ids()
            .binary_search(&participant_subject_id)
            .is_err()
        {
            return Err(DistributedStateError::ParticipantOutsideBudget {
                participant: participant_subject_id,
            });
        }
        validate_claim_material(
            observed_at_unix_ms,
            state,
            protocol_state_digest,
            raw_evidence_digest,
        )?;

        let context_id = context.id();
        let claim_id = ParticipantStateClaimId(hash_claim(
            context_id,
            participant_subject_id,
            verifier_profile_id,
            observed_at_unix_ms,
            state,
            protocol_state_digest,
            raw_evidence_digest,
        ));

        Ok(Self {
            schema_version: PARTICIPANT_STATE_CLAIM_SCHEMA_V1.to_owned(),
            context_id,
            participant_subject_id,
            verifier_profile_id,
            observed_at_unix_ms,
            state,
            protocol_state_digest,
            raw_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), DistributedStateError> {
        if self.schema_version != PARTICIPANT_STATE_CLAIM_SCHEMA_V1 {
            return Err(DistributedStateError::UnsupportedClaimSchema(
                self.schema_version.clone(),
            ));
        }
        validate_claim_material(
            self.observed_at_unix_ms,
            self.state,
            self.protocol_state_digest,
            self.raw_evidence_digest,
        )?;
        let expected = ParticipantStateClaimId(hash_claim(
            self.context_id,
            self.participant_subject_id,
            self.verifier_profile_id,
            self.observed_at_unix_ms,
            self.state,
            self.protocol_state_digest,
            self.raw_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(DistributedStateError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ParticipantStateClaimId {
        self.claim_id
    }

    pub fn context_id(&self) -> DistributedStateContextId {
        self.context_id
    }

    pub fn participant_subject_id(&self) -> ContinuitySubjectId {
        self.participant_subject_id
    }

    pub fn verifier_profile_id(&self) -> VerifierProfileId {
        self.verifier_profile_id
    }

    pub fn observed_at_unix_ms(&self) -> u64 {
        self.observed_at_unix_ms
    }

    pub fn state(&self) -> ParticipantOperationalStateV1 {
        self.state
    }

    pub fn protocol_state_digest(&self) -> Option<[u8; 32]> {
        self.protocol_state_digest
    }

    pub fn raw_evidence_digest(&self) -> [u8; 32] {
        self.raw_evidence_digest
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PolicyCheckedParticipantStateEvidenceV1 {
    claim: ParticipantStateClaimV1,
    profile: VerifierProfileV1,
}

/// Exact budget/context/profile admission. This proves no signature and performs
/// no freshness decision beyond structural non-zero observation time.
pub(crate) fn policy_check_participant_state_claim(
    context: &ValidatedDistributedStateContextV1,
    profile: &VerifierProfileV1,
    claim: ParticipantStateClaimV1,
) -> Result<PolicyCheckedParticipantStateEvidenceV1, DistributedStateError> {
    profile.validate()?;
    context.as_raw().validate()?;
    claim.validate()?;
    if claim.context_id() != context.id() {
        return Err(DistributedStateError::ClaimContextMismatch);
    }
    if claim.verifier_profile_id() != profile.id() {
        return Err(DistributedStateError::VerifierProfileClaimMismatch);
    }
    if context
        .budget()
        .participant_subject_ids()
        .binary_search(&claim.participant_subject_id())
        .is_err()
    {
        return Err(DistributedStateError::ParticipantOutsideBudget {
            participant: claim.participant_subject_id(),
        });
    }

    Ok(PolicyCheckedParticipantStateEvidenceV1 {
        claim,
        profile: profile.clone(),
    })
}

/// Authenticated participant-state evidence has no production constructor yet.
/// A future Xenia/crypto adapter must authenticate the exact checked claim under
/// the captured verifier root. Tests may construct it through the cfg(test) gate.
#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedParticipantStateEvidenceV1 {
    checked: PolicyCheckedParticipantStateEvidenceV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedParticipantStateEvidenceId,
}

impl AuthenticatedParticipantStateEvidenceV1 {
    pub(crate) fn id(&self) -> AuthenticatedParticipantStateEvidenceId {
        self.evidence_id
    }

    pub(crate) fn context_id(&self) -> DistributedStateContextId {
        self.checked.claim.context_id()
    }

    pub(crate) fn participant_subject_id(&self) -> ContinuitySubjectId {
        self.checked.claim.participant_subject_id()
    }

    pub(crate) fn profile_id(&self) -> VerifierProfileId {
        self.checked.profile.id()
    }

    pub(crate) fn root_epoch(&self) -> u64 {
        self.checked.profile.root_epoch()
    }

    pub(crate) fn observed_at_unix_ms(&self) -> u64 {
        self.checked.claim.observed_at_unix_ms()
    }

    pub(crate) fn state(&self) -> ParticipantOperationalStateV1 {
        self.checked.claim.state()
    }

    pub(crate) fn protocol_state_digest(&self) -> Option<[u8; 32]> {
        self.checked.claim.protocol_state_digest()
    }

    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        checked: PolicyCheckedParticipantStateEvidenceV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, DistributedStateError> {
        if authentication_evidence_digest == [0; 32] {
            return Err(DistributedStateError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedParticipantStateEvidenceId(hash_authenticated(
            checked.claim.id(),
            checked.profile.id(),
            authentication_evidence_digest,
        ));
        Ok(Self {
            checked,
            authentication_evidence_digest,
            evidence_id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DistributedStateError {
    #[error(transparent)]
    Verification(#[from] VerificationAdmissionError),
    #[error("unsupported distributed-state context schema: {0}")]
    UnsupportedContextSchema(String),
    #[error("unsupported participant-state claim schema: {0}")]
    UnsupportedClaimSchema(String),
    #[error("distributed-state budget generation must be non-zero")]
    ZeroBudgetGeneration,
    #[error("distributed-state transaction challenge must be non-zero")]
    ZeroTransactionChallenge,
    #[error("distributed-state context must contain at least one transition candidate")]
    NoCandidates,
    #[error("distributed-state context exceeds the candidate bound")]
    TooManyCandidates,
    #[error("candidate subject identities must be in canonical sorted unique order")]
    NonCanonicalCandidates,
    #[error("transition candidate is outside the distributed budget: {candidate:?}")]
    CandidateOutsideBudget { candidate: ContinuitySubjectId },
    #[error("candidate count {requested} exceeds max_concurrent_unavailable {allowed}")]
    CandidateSetExceedsBudget { requested: u32, allowed: u32 },
    #[error("candidate set violates a distributed mutual-exclusion set")]
    CandidateMutualExclusionViolation,
    #[error("distributed-state aggregate subject does not match validated budget")]
    AggregateSubjectMismatch,
    #[error("distributed-state context references a different distributed budget")]
    BudgetIdentityMismatch,
    #[error("distributed-state context references a different budget generation")]
    BudgetGenerationMismatch,
    #[error("distributed-state participant-set digest does not match validated budget")]
    ParticipantSetMismatch,
    #[error("stored distributed-state context identity does not match canonical fields")]
    ContextIdentityMismatch,
    #[error("participant-state observation time must be non-zero")]
    ZeroObservationTime,
    #[error("participant-state raw evidence digest must be non-zero")]
    ZeroRawEvidenceDigest,
    #[error("participant-state protocol digest must be non-zero when present")]
    ZeroProtocolStateDigest,
    #[error("healthy/unhealthy/transitioning participant state requires protocol-state digest")]
    MissingProtocolStateDigest,
    #[error("stored participant-state claim identity does not match canonical fields")]
    ClaimIdentityMismatch,
    #[error("participant-state claim belongs to a different distributed transaction context")]
    ClaimContextMismatch,
    #[error("participant-state claim binds a different verifier profile")]
    VerifierProfileClaimMismatch,
    #[error("participant-state claim references a subject outside the distributed budget: {participant:?}")]
    ParticipantOutsideBudget { participant: ContinuitySubjectId },
    #[error("authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
}

fn canonicalize_candidates(
    candidates: &mut Vec<ContinuitySubjectId>,
) -> Result<(), DistributedStateError> {
    if candidates.is_empty() {
        return Err(DistributedStateError::NoCandidates);
    }
    if candidates.len() > MAX_CANDIDATES {
        return Err(DistributedStateError::TooManyCandidates);
    }
    candidates.sort();
    candidates.dedup();
    Ok(())
}

fn validate_canonical_candidates(
    candidates: &[ContinuitySubjectId],
) -> Result<(), DistributedStateError> {
    if candidates.is_empty() {
        return Err(DistributedStateError::NoCandidates);
    }
    if candidates.len() > MAX_CANDIDATES {
        return Err(DistributedStateError::TooManyCandidates);
    }
    if candidates.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(DistributedStateError::NonCanonicalCandidates);
    }
    Ok(())
}

fn validate_candidates_against_budget(
    candidates: &[ContinuitySubjectId],
    budget: &ValidatedDistributedChangeBudgetV1,
) -> Result<(), DistributedStateError> {
    for candidate in candidates {
        if budget.participant_subject_ids().binary_search(candidate).is_err() {
            return Err(DistributedStateError::CandidateOutsideBudget {
                candidate: *candidate,
            });
        }
    }
    if candidates.len() as u32 > budget.max_concurrent_unavailable() {
        return Err(DistributedStateError::CandidateSetExceedsBudget {
            requested: candidates.len() as u32,
            allowed: budget.max_concurrent_unavailable(),
        });
    }
    for exclusion in budget.mutual_exclusion_sets() {
        let selected = exclusion
            .members()
            .iter()
            .filter(|member| candidates.binary_search(member).is_ok())
            .count();
        if selected > 1 {
            return Err(DistributedStateError::CandidateMutualExclusionViolation);
        }
    }
    Ok(())
}

fn validate_claim_material(
    observed_at_unix_ms: u64,
    state: ParticipantOperationalStateV1,
    protocol_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> Result<(), DistributedStateError> {
    if observed_at_unix_ms == 0 {
        return Err(DistributedStateError::ZeroObservationTime);
    }
    if raw_evidence_digest == [0; 32] {
        return Err(DistributedStateError::ZeroRawEvidenceDigest);
    }
    if matches!(protocol_state_digest, Some(digest) if digest == [0; 32]) {
        return Err(DistributedStateError::ZeroProtocolStateDigest);
    }
    if !matches!(state, ParticipantOperationalStateV1::Unknown) && protocol_state_digest.is_none() {
        return Err(DistributedStateError::MissingProtocolStateDigest);
    }
    Ok(())
}

fn participant_set_digest(participants: &[ContinuitySubjectId]) -> ParticipantSetDigest {
    let mut bytes = Vec::with_capacity(8 + participants.len() * 32);
    put_len(&mut bytes, participants.len());
    for participant in participants {
        bytes.extend_from_slice(participant.as_bytes());
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(PARTICIPANT_SET_DOMAIN);
    hasher.update(&bytes);
    ParticipantSetDigest(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn hash_context(
    aggregate_subject_id: ContinuitySubjectId,
    budget_id: DistributedChangeBudgetId,
    budget_generation: u64,
    participant_set_digest: ParticipantSetDigest,
    candidate_subject_ids: &[ContinuitySubjectId],
    transaction_challenge: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(160 + candidate_subject_ids.len() * 32);
    bytes.extend_from_slice(aggregate_subject_id.as_bytes());
    bytes.extend_from_slice(budget_id.as_bytes());
    bytes.extend_from_slice(&budget_generation.to_le_bytes());
    bytes.extend_from_slice(participant_set_digest.as_bytes());
    put_len(&mut bytes, candidate_subject_ids.len());
    for candidate in candidate_subject_ids {
        bytes.extend_from_slice(candidate.as_bytes());
    }
    bytes.extend_from_slice(&transaction_challenge);
    domain_hash(CONTEXT_DOMAIN, &bytes)
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    context_id: DistributedStateContextId,
    participant_subject_id: ContinuitySubjectId,
    verifier_profile_id: VerifierProfileId,
    observed_at_unix_ms: u64,
    state: ParticipantOperationalStateV1,
    protocol_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(192);
    bytes.extend_from_slice(context_id.as_bytes());
    bytes.extend_from_slice(participant_subject_id.as_bytes());
    bytes.extend_from_slice(verifier_profile_id.as_bytes());
    bytes.extend_from_slice(&observed_at_unix_ms.to_le_bytes());
    bytes.push(state.tag());
    match protocol_state_digest {
        Some(digest) => {
            bytes.push(1);
            bytes.extend_from_slice(&digest);
        }
        None => bytes.push(0),
    }
    bytes.extend_from_slice(&raw_evidence_digest);
    domain_hash(CLAIM_DOMAIN, &bytes)
}

fn hash_authenticated(
    claim_id: ParticipantStateClaimId,
    profile_id: VerifierProfileId,
    authentication_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(96);
    bytes.extend_from_slice(claim_id.as_bytes());
    bytes.extend_from_slice(profile_id.as_bytes());
    bytes.extend_from_slice(&authentication_evidence_digest);
    domain_hash(AUTH_DOMAIN, &bytes)
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{
        DistributedChangeBudgetV1, MutualExclusionSetV1, RecoveryPathClassV1,
    };
    use crate::scope::{ContinuityScopeV1, ContinuitySubjectV1};
    use crate::witness::EvidenceClass;

    fn subject(logical_id: &str, scope: ContinuityScopeV1) -> ContinuitySubjectV1 {
        ContinuitySubjectV1::new("org.example", logical_id, scope, None).unwrap()
    }

    fn fixture_budget(
        max_unavailable: u32,
        exclusions: Vec<MutualExclusionSetV1>,
    ) -> (
        ContinuitySubjectV1,
        Vec<ContinuitySubjectV1>,
        ValidatedDistributedChangeBudgetV1,
    ) {
        let aggregate = subject("cluster-a", ContinuityScopeV1::Cluster);
        let participants = vec![
            subject("node-a", ContinuityScopeV1::Machine),
            subject("node-b", ContinuityScopeV1::Machine),
            subject("node-c", ContinuityScopeV1::Machine),
        ];
        let raw = DistributedChangeBudgetV1::new(
            &aggregate,
            7,
            participants.iter().map(ContinuitySubjectV1::id).collect(),
            max_unavailable,
            3 - max_unavailable,
            exclusions,
            vec![RecoveryPathClassV1::OutOfBandManagement],
        )
        .unwrap();
        let validated = raw.validate_against_subject(&aggregate).unwrap();
        (aggregate, participants, validated)
    }

    fn profile(seed: u8) -> VerifierProfileV1 {
        VerifierProfileV1::new(
            format!("distributed-state-{seed}"),
            [seed; 32],
            1,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
    }

    #[test]
    fn candidate_order_is_canonical_and_context_is_budget_bound() {
        let (_, participants, budget) = fixture_budget(2, vec![]);
        let raw = DistributedStateContextV1::new(
            &budget,
            vec![participants[1].id(), participants[0].id()],
            [9; 32],
        )
        .unwrap();
        assert!(raw.candidate_subject_ids()[0] < raw.candidate_subject_ids()[1]);
        raw.validate_against_budget(&budget).unwrap();
    }

    #[test]
    fn candidate_outside_budget_fails_closed() {
        let (_, participants, budget) = fixture_budget(1, vec![]);
        let outsider = subject("node-x", ContinuityScopeV1::Machine);
        assert!(matches!(
            DistributedStateContextV1::new(
                &budget,
                vec![participants[0].id(), outsider.id()],
                [9; 32],
            ),
            Err(DistributedStateError::CandidateOutsideBudget { .. })
                | Err(DistributedStateError::CandidateSetExceedsBudget { .. })
        ));
    }

    #[test]
    fn candidate_count_cannot_exceed_static_change_budget() {
        let (_, participants, budget) = fixture_budget(1, vec![]);
        assert_eq!(
            DistributedStateContextV1::new(
                &budget,
                vec![participants[0].id(), participants[1].id()],
                [9; 32],
            )
            .unwrap_err(),
            DistributedStateError::CandidateSetExceedsBudget {
                requested: 2,
                allowed: 1,
            }
        );
    }

    #[test]
    fn candidate_set_respects_static_mutual_exclusion() {
        let a = subject("node-a", ContinuityScopeV1::Machine);
        let b = subject("node-b", ContinuityScopeV1::Machine);
        let exclusion = MutualExclusionSetV1::new(vec![a.id(), b.id()]).unwrap();
        let aggregate = subject("cluster-a", ContinuityScopeV1::Cluster);
        let c = subject("node-c", ContinuityScopeV1::Machine);
        let raw_budget = DistributedChangeBudgetV1::new(
            &aggregate,
            7,
            vec![a.id(), b.id(), c.id()],
            2,
            1,
            vec![exclusion],
            vec![RecoveryPathClassV1::OutOfBandManagement],
        )
        .unwrap();
        let budget = raw_budget.validate_against_subject(&aggregate).unwrap();
        assert_eq!(
            DistributedStateContextV1::new(&budget, vec![a.id(), b.id()], [9; 32])
                .unwrap_err(),
            DistributedStateError::CandidateMutualExclusionViolation
        );
    }

    #[test]
    fn challenge_changes_transaction_context_identity() {
        let (_, participants, budget) = fixture_budget(1, vec![]);
        let a = DistributedStateContextV1::new(&budget, vec![participants[0].id()], [8; 32])
            .unwrap();
        let b = DistributedStateContextV1::new(&budget, vec![participants[0].id()], [9; 32])
            .unwrap();
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn healthy_claim_requires_protocol_state_digest() {
        let (_, participants, budget) = fixture_budget(1, vec![]);
        let context = DistributedStateContextV1::new(&budget, vec![participants[0].id()], [9; 32])
            .unwrap()
            .validate_against_budget(&budget)
            .unwrap();
        let verifier = profile(7);
        assert_eq!(
            ParticipantStateClaimV1::new(
                &context,
                participants[0].id(),
                verifier.id(),
                1_700_000_000_000,
                ParticipantOperationalStateV1::Healthy,
                None,
                [4; 32],
            )
            .unwrap_err(),
            DistributedStateError::MissingProtocolStateDigest
        );
    }

    #[test]
    fn unknown_claim_may_preserve_missing_protocol_state_explicitly() {
        let (_, participants, budget) = fixture_budget(1, vec![]);
        let context = DistributedStateContextV1::new(&budget, vec![participants[0].id()], [9; 32])
            .unwrap()
            .validate_against_budget(&budget)
            .unwrap();
        let verifier = profile(7);
        let claim = ParticipantStateClaimV1::new(
            &context,
            participants[1].id(),
            verifier.id(),
            1_700_000_000_000,
            ParticipantOperationalStateV1::Unknown,
            None,
            [4; 32],
        )
        .unwrap();
        claim.validate().unwrap();
    }

    #[test]
    fn raw_claim_is_only_promoted_after_exact_profile_and_context_check() {
        let (_, participants, budget) = fixture_budget(1, vec![]);
        let context = DistributedStateContextV1::new(&budget, vec![participants[0].id()], [9; 32])
            .unwrap()
            .validate_against_budget(&budget)
            .unwrap();
        let verifier = profile(7);
        let claim = ParticipantStateClaimV1::new(
            &context,
            participants[1].id(),
            verifier.id(),
            1_700_000_000_000,
            ParticipantOperationalStateV1::Healthy,
            Some([5; 32]),
            [4; 32],
        )
        .unwrap();
        let checked = policy_check_participant_state_claim(&context, &verifier, claim).unwrap();
        let authenticated =
            AuthenticatedParticipantStateEvidenceV1::authenticate_for_test(checked, [6; 32])
                .unwrap();
        assert_eq!(authenticated.context_id(), context.id());
        assert_eq!(authenticated.participant_subject_id(), participants[1].id());
        assert_eq!(authenticated.state(), ParticipantOperationalStateV1::Healthy);
    }

    #[test]
    fn claim_from_another_transaction_context_is_rejected() {
        let (_, participants, budget) = fixture_budget(1, vec![]);
        let context_a = DistributedStateContextV1::new(&budget, vec![participants[0].id()], [8; 32])
            .unwrap()
            .validate_against_budget(&budget)
            .unwrap();
        let context_b = DistributedStateContextV1::new(&budget, vec![participants[0].id()], [9; 32])
            .unwrap()
            .validate_against_budget(&budget)
            .unwrap();
        let verifier = profile(7);
        let claim = ParticipantStateClaimV1::new(
            &context_a,
            participants[1].id(),
            verifier.id(),
            1_700_000_000_000,
            ParticipantOperationalStateV1::Healthy,
            Some([5; 32]),
            [4; 32],
        )
        .unwrap();
        assert_eq!(
            policy_check_participant_state_claim(&context_b, &verifier, claim).unwrap_err(),
            DistributedStateError::ClaimContextMismatch
        );
    }

    #[test]
    fn participant_set_change_breaks_context_rebinding() {
        let (aggregate, participants, budget) = fixture_budget(1, vec![]);
        let raw_context = DistributedStateContextV1::new(
            &budget,
            vec![participants[0].id()],
            [9; 32],
        )
        .unwrap();
        let replacement = subject("node-d", ContinuityScopeV1::Machine);
        let changed_raw = DistributedChangeBudgetV1::new(
            &aggregate,
            7,
            vec![participants[0].id(), participants[1].id(), replacement.id()],
            1,
            2,
            vec![],
            vec![RecoveryPathClassV1::OutOfBandManagement],
        )
        .unwrap();
        let changed = changed_raw.validate_against_subject(&aggregate).unwrap();
        assert!(matches!(
            raw_context.validate_against_budget(&changed),
            Err(DistributedStateError::BudgetIdentityMismatch)
                | Err(DistributedStateError::ParticipantSetMismatch)
        ));
    }
}
