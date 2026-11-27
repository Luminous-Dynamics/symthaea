// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Claim-relative revalidation after a source lifecycle transition.
//!
//! A source mutation does not prove that every claim derived from that source
//! became false. This policy contract permits one claim to be re-evaluated
//! against an exact ledger transition, but only through an injected independent
//! evidence verifier. A passing claim never revives an old projection: it only
//! permits a fresh rebuild with a new projection ID and current dependencies.
//!
//! This module does not implement a source checkout, test runner, signature
//! verifier, or production evidence authority.

use sha2::{Digest, Sha256};

use crate::streaming_session::{
    StreamingInvalidationCauseV1, StreamingSessionLedgerV1,
    StreamingSessionMutationReceiptV1,
};
use crate::temporal_projection::{
    DerivedProjectionIndexV1, DerivedProjectionSpecV1, ProjectionRegisterStatusV1,
    TemporalProjectionRejectV1,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimRevalidationOutcomeV1 {
    Pass,
    Fail,
    NotRun,
    Skipped,
    InfrastructureFailure,
    Inconclusive,
}

impl ClaimRevalidationOutcomeV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Pass => 1,
            Self::Fail => 2,
            Self::NotRun => 3,
            Self::Skipped => 4,
            Self::InfrastructureFailure => 5,
            Self::Inconclusive => 6,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimRevalidationDispositionV1 {
    RevalidationRequired,
    Invalidated,
    PreservedNeedsRebuild,
    Quarantined,
    Revoked,
    ErasureRequired,
}

impl ClaimRevalidationDispositionV1 {
    fn tag(self) -> u8 {
        match self {
            Self::RevalidationRequired => 1,
            Self::Invalidated => 2,
            Self::PreservedNeedsRebuild => 3,
            Self::Quarantined => 4,
            Self::Revoked => 5,
            Self::ErasureRequired => 6,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimRevalidationRejectV1 {
    InvalidClaimId,
    InvalidSourceEventId,
    InvalidDigest,
    UnrecordedSourceTransition,
    TransitionHasNoInvalidationCause,
    SourceEventNotAffectedByTransition,
    TransitionBindingMismatch,
    EvidenceBindingMismatch,
    InvalidEvidence,
    InvalidVerifierReceipt,
    InvalidAssessment,
    RebuildNotPermitted,
    AssessmentBindingMismatch,
    ProjectionDoesNotBindCorrectedEvent,
    ProjectionRegistrationRejected(TemporalProjectionRejectV1),
}

/// Exact binding for one claim-relative check after one source transition.
/// All digest fields are lowercase SHA-256 hex commitments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaimRevalidationBindingV1 {
    pub claim_id: String,
    pub claim_sha256: String,
    pub source_event_id: String,
    pub source_transition_receipt_sha256: String,
    pub source_revision_before_sha256: String,
    pub source_revision_after_sha256: String,
    pub test_artifact_sha256: String,
    pub verifier_policy_sha256: String,
}

impl ClaimRevalidationBindingV1 {
    pub fn binding_sha256(&self) -> String {
        binding_digest(self)
    }
}

/// Raw output of a claim-specific check. This record is not trusted merely
/// because it exists; an injected independent verifier must validate it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaimRevalidationEvidenceV1 {
    pub binding_sha256: String,
    pub outcome: ClaimRevalidationOutcomeV1,
    pub result_sha256: String,
    pub execution_receipt_sha256: String,
    pub evaluator_id: String,
}

impl ClaimRevalidationEvidenceV1 {
    pub fn evidence_sha256(&self) -> String {
        evidence_digest(self)
    }
}

/// Receipt from an independently configured verifier. The cryptographic and
/// provenance checks occur inside the injected verifier implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndependentClaimRevalidationReceiptV1 {
    pub verifier_id: String,
    pub verification_receipt_sha256: String,
}

pub trait ClaimRevalidationEvidenceVerifierV1 {
    /// Return Some only after independently checking the exact claim/source
    /// binding, test artifact, evaluator, execution receipt and result digest.
    /// Return None when evidence is self-reported, unavailable, stale or
    /// otherwise insufficient. The verifier must not trust candidate claims
    /// that their own output is independently verified.
    fn verify_independent_evidence(
        &self,
        binding: &ClaimRevalidationBindingV1,
        evidence: &ClaimRevalidationEvidenceV1,
    ) -> Option<IndependentClaimRevalidationReceiptV1>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaimRevalidationAssessmentV1 {
    pub claim_id: String,
    pub claim_sha256: String,
    pub source_event_id: String,
    pub source_transition_receipt_sha256: String,
    pub disposition: ClaimRevalidationDispositionV1,
    pub binding_sha256: String,
    pub source_transition_event_id: String,
    pub source_transition_sequence: u64,
    /// Always false: a prior derived projection is never reactivated.
    pub old_projection_may_be_reactivated: bool,
    /// True only for an independently verified Pass after a Correction.
    pub new_projection_may_be_built: bool,
    pub evidence_sha256: Option<String>,
    pub verifier_id: Option<String>,
    pub verification_receipt_sha256: Option<String>,
    pub assessment_sha256: String,
}

impl ClaimRevalidationAssessmentV1 {
    pub fn verify_integrity(&self) -> bool {
        !self.old_projection_may_be_reactivated
            && valid_atom(&self.claim_id, 512)
            && valid_atom(&self.source_event_id, 512)
            && valid_atom(&self.source_transition_event_id, 512)
            && validate_digest(&self.claim_sha256).is_ok()
            && validate_digest(&self.source_transition_receipt_sha256).is_ok()
            && validate_digest(&self.binding_sha256).is_ok()
            && self.assessment_sha256 == assessment_digest(
                &self.claim_id,
                &self.claim_sha256,
                &self.source_event_id,
                &self.source_transition_receipt_sha256,
                self.disposition,
                &self.binding_sha256,
                &self.source_transition_event_id,
                self.source_transition_sequence,
                self.old_projection_may_be_reactivated,
                self.new_projection_may_be_built,
                self.evidence_sha256.as_deref(),
                self.verifier_id.as_deref(),
                self.verification_receipt_sha256.as_deref(),
            )
    }
}

/// Register a new derived projection only after a claim has independently
/// passed revalidation following a correction.
///
/// The current ledger must still be at the exact transition sequence, and the
/// new projection must depend on the correction event (the new active source
/// revision), never on the superseded source event. The projection index must
/// already have consumed the same transition; otherwise its sequence gate
/// rejects registration. This function cannot revive an old projection ID.
pub fn register_rebuilt_projection_after_claim_revalidation_v1(
    ledger: &StreamingSessionLedgerV1,
    index: &mut DerivedProjectionIndexV1,
    transition: &StreamingSessionMutationReceiptV1,
    binding: &ClaimRevalidationBindingV1,
    assessment: &ClaimRevalidationAssessmentV1,
    spec: DerivedProjectionSpecV1,
) -> Result<ProjectionRegisterStatusV1, ClaimRevalidationRejectV1> {
    if !assessment.verify_integrity() {
        return Err(ClaimRevalidationRejectV1::InvalidAssessment);
    }
    if assessment.disposition != ClaimRevalidationDispositionV1::PreservedNeedsRebuild
        || !assessment.new_projection_may_be_built
        || assessment.old_projection_may_be_reactivated
    {
        return Err(ClaimRevalidationRejectV1::RebuildNotPermitted);
    }
    if assessment.claim_id != binding.claim_id
        || assessment.claim_sha256 != binding.claim_sha256
        || assessment.source_event_id != binding.source_event_id
        || assessment.source_transition_receipt_sha256 != binding.source_transition_receipt_sha256
        || assessment.binding_sha256 != binding.binding_sha256()
    {
        return Err(ClaimRevalidationRejectV1::AssessmentBindingMismatch);
    }
    if !ledger.verify_recorded_receipt(transition) {
        return Err(ClaimRevalidationRejectV1::UnrecordedSourceTransition);
    }
    if transition.invalidation_cause != Some(StreamingInvalidationCauseV1::Correction)
        || assessment.source_transition_event_id != transition.event_id
        || assessment.source_transition_sequence != transition.sequence
        || assessment.source_transition_receipt_sha256 != transition.receipt_sha256
        || !transition.affected_event_ids.iter().any(|id| id == &binding.source_event_id)
    {
        return Err(ClaimRevalidationRejectV1::TransitionBindingMismatch);
    }
    if ledger.last_sequence() != transition.sequence
        || spec.built_at_sequence != transition.sequence
        || !spec.source_event_ids.iter().any(|id| id == &transition.event_id)
        || spec.source_event_ids.iter().any(|id| id == &binding.source_event_id)
    {
        return Err(ClaimRevalidationRejectV1::ProjectionDoesNotBindCorrectedEvent);
    }
    index.register_projection(ledger, spec)
        .map_err(ClaimRevalidationRejectV1::ProjectionRegistrationRejected)
}

/// Assess one claim against one exact, ledger-recorded transition.
///
/// Quarantine, Revocation and RightsErasure are hard governance overrides.
/// Only Correction can produce PreservedNeedsRebuild or Invalidated. Even a
/// passing revalidation never marks the old projection current.
pub fn assess_claim_revalidation_v1(
    ledger: &StreamingSessionLedgerV1,
    transition: &StreamingSessionMutationReceiptV1,
    binding: &ClaimRevalidationBindingV1,
    evidence: Option<&ClaimRevalidationEvidenceV1>,
    verifier: Option<&dyn ClaimRevalidationEvidenceVerifierV1>,
) -> Result<ClaimRevalidationAssessmentV1, ClaimRevalidationRejectV1> {
    if !valid_atom(&binding.claim_id, 512) {
        return Err(ClaimRevalidationRejectV1::InvalidClaimId);
    }
    if !valid_atom(&binding.source_event_id, 512) {
        return Err(ClaimRevalidationRejectV1::InvalidSourceEventId);
    }
    validate_digest(&binding.claim_sha256)?;
    validate_digest(&binding.source_transition_receipt_sha256)?;
    validate_digest(&binding.source_revision_before_sha256)?;
    validate_digest(&binding.source_revision_after_sha256)?;
    validate_digest(&binding.test_artifact_sha256)?;
    validate_digest(&binding.verifier_policy_sha256)?;

    if !ledger.verify_recorded_receipt(transition) {
        return Err(ClaimRevalidationRejectV1::UnrecordedSourceTransition);
    }
    let Some(cause) = transition.invalidation_cause else {
        return Err(ClaimRevalidationRejectV1::TransitionHasNoInvalidationCause);
    };
    if transition.receipt_sha256 != binding.source_transition_receipt_sha256 {
        return Err(ClaimRevalidationRejectV1::TransitionBindingMismatch);
    }
    if !transition.affected_event_ids.iter().any(|id| id == &binding.source_event_id) {
        return Err(ClaimRevalidationRejectV1::SourceEventNotAffectedByTransition);
    }

    // Lifecycle and rights controls cannot be overridden by functional tests.
    match cause {
        StreamingInvalidationCauseV1::Quarantine => {
            return Ok(build_assessment(
                binding, transition, ClaimRevalidationDispositionV1::Quarantined,
                None, None, None, false,
            ));
        }
        StreamingInvalidationCauseV1::Revocation => {
            return Ok(build_assessment(
                binding, transition, ClaimRevalidationDispositionV1::Revoked,
                None, None, None, false,
            ));
        }
        StreamingInvalidationCauseV1::RightsErasure => {
            return Ok(build_assessment(
                binding, transition, ClaimRevalidationDispositionV1::ErasureRequired,
                None, None, None, false,
            ));
        }
        StreamingInvalidationCauseV1::Correction => {}
    }

    let Some(evidence) = evidence else {
        return Ok(build_assessment(
            binding, transition, ClaimRevalidationDispositionV1::RevalidationRequired,
            None, None, None, false,
        ));
    };
    validate_evidence(binding, evidence)?;

    let Some(verifier) = verifier else {
        return Ok(build_assessment(
            binding, transition, ClaimRevalidationDispositionV1::RevalidationRequired,
            Some(evidence.evidence_sha256()), None, None, false,
        ));
    };
    let Some(attestation) = verifier.verify_independent_evidence(binding, evidence) else {
        return Ok(build_assessment(
            binding, transition, ClaimRevalidationDispositionV1::RevalidationRequired,
            Some(evidence.evidence_sha256()), None, None, false,
        ));
    };
    if !valid_atom(&attestation.verifier_id, 512) {
        return Err(ClaimRevalidationRejectV1::InvalidVerifierReceipt);
    }
    validate_digest(&attestation.verification_receipt_sha256)
        .map_err(|_| ClaimRevalidationRejectV1::InvalidVerifierReceipt)?;

    let disposition = match evidence.outcome {
        ClaimRevalidationOutcomeV1::Pass => ClaimRevalidationDispositionV1::PreservedNeedsRebuild,
        ClaimRevalidationOutcomeV1::Fail => ClaimRevalidationDispositionV1::Invalidated,
        ClaimRevalidationOutcomeV1::NotRun
        | ClaimRevalidationOutcomeV1::Skipped
        | ClaimRevalidationOutcomeV1::InfrastructureFailure
        | ClaimRevalidationOutcomeV1::Inconclusive => {
            ClaimRevalidationDispositionV1::RevalidationRequired
        }
    };
    let may_build = disposition == ClaimRevalidationDispositionV1::PreservedNeedsRebuild;
    Ok(build_assessment(
        binding,
        transition,
        disposition,
        Some(evidence.evidence_sha256()),
        Some(attestation.verifier_id),
        Some(attestation.verification_receipt_sha256),
        may_build,
    ))
}

fn validate_evidence(
    binding: &ClaimRevalidationBindingV1,
    evidence: &ClaimRevalidationEvidenceV1,
) -> Result<(), ClaimRevalidationRejectV1> {
    validate_digest(&evidence.binding_sha256)?;
    validate_digest(&evidence.result_sha256)?;
    validate_digest(&evidence.execution_receipt_sha256)?;
    if !valid_atom(&evidence.evaluator_id, 512) {
        return Err(ClaimRevalidationRejectV1::InvalidEvidence);
    }
    if evidence.binding_sha256 != binding.binding_sha256() {
        return Err(ClaimRevalidationRejectV1::EvidenceBindingMismatch);
    }
    Ok(())
}

fn build_assessment(
    binding: &ClaimRevalidationBindingV1,
    transition: &StreamingSessionMutationReceiptV1,
    disposition: ClaimRevalidationDispositionV1,
    evidence_sha256: Option<String>,
    verifier_id: Option<String>,
    verification_receipt_sha256: Option<String>,
    new_projection_may_be_built: bool,
) -> ClaimRevalidationAssessmentV1 {
    let binding_sha256 = binding.binding_sha256();
    let old_projection_may_be_reactivated = false;
    let assessment_sha256 = assessment_digest(
        &binding.claim_id,
        &binding.claim_sha256,
        &binding.source_event_id,
        &transition.receipt_sha256,
        disposition,
        &binding_sha256,
        &transition.event_id,
        transition.sequence,
        old_projection_may_be_reactivated,
        new_projection_may_be_built,
        evidence_sha256.as_deref(),
        verifier_id.as_deref(),
        verification_receipt_sha256.as_deref(),
    );
    ClaimRevalidationAssessmentV1 {
        claim_id: binding.claim_id.clone(),
        claim_sha256: binding.claim_sha256.clone(),
        source_event_id: binding.source_event_id.clone(),
        source_transition_receipt_sha256: transition.receipt_sha256.clone(),
        disposition,
        binding_sha256,
        source_transition_event_id: transition.event_id.clone(),
        source_transition_sequence: transition.sequence,
        old_projection_may_be_reactivated,
        new_projection_may_be_built,
        evidence_sha256,
        verifier_id,
        verification_receipt_sha256,
        assessment_sha256,
    }
}

fn binding_digest(binding: &ClaimRevalidationBindingV1) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:claim-revalidation-binding:v1\0");
    hash_field(&mut hasher, &binding.claim_id);
    hash_field(&mut hasher, &binding.claim_sha256);
    hash_field(&mut hasher, &binding.source_event_id);
    hash_field(&mut hasher, &binding.source_transition_receipt_sha256);
    hash_field(&mut hasher, &binding.source_revision_before_sha256);
    hash_field(&mut hasher, &binding.source_revision_after_sha256);
    hash_field(&mut hasher, &binding.test_artifact_sha256);
    hash_field(&mut hasher, &binding.verifier_policy_sha256);
    hex_digest(hasher.finalize().as_slice())
}

fn evidence_digest(evidence: &ClaimRevalidationEvidenceV1) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:claim-revalidation-evidence:v1\0");
    hash_field(&mut hasher, &evidence.binding_sha256);
    hasher.update([evidence.outcome.tag()]);
    hash_field(&mut hasher, &evidence.result_sha256);
    hash_field(&mut hasher, &evidence.execution_receipt_sha256);
    hash_field(&mut hasher, &evidence.evaluator_id);
    hex_digest(hasher.finalize().as_slice())
}

#[allow(clippy::too_many_arguments)]
fn assessment_digest(
    claim_id: &str,
    claim_sha256: &str,
    source_event_id: &str,
    source_transition_receipt_sha256: &str,
    disposition: ClaimRevalidationDispositionV1,
    binding_sha256: &str,
    transition_event_id: &str,
    transition_sequence: u64,
    old_projection_may_be_reactivated: bool,
    new_projection_may_be_built: bool,
    evidence_sha256: Option<&str>,
    verifier_id: Option<&str>,
    verification_receipt_sha256: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:claim-revalidation-assessment:v1\0");
    hash_field(&mut hasher, claim_id);
    hash_field(&mut hasher, claim_sha256);
    hash_field(&mut hasher, source_event_id);
    hash_field(&mut hasher, source_transition_receipt_sha256);
    hasher.update([disposition.tag()]);
    hash_field(&mut hasher, binding_sha256);
    hash_field(&mut hasher, transition_event_id);
    hasher.update(transition_sequence.to_be_bytes());
    hasher.update([u8::from(old_projection_may_be_reactivated)]);
    hasher.update([u8::from(new_projection_may_be_built)]);
    hash_optional(&mut hasher, evidence_sha256);
    hash_optional(&mut hasher, verifier_id);
    hash_optional(&mut hasher, verification_receipt_sha256);
    hex_digest(hasher.finalize().as_slice())
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hash_optional(hasher: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            hash_field(hasher, value);
        }
        None => hasher.update([0]),
    }
}

fn valid_atom(value: &str, maximum: usize) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= maximum
        && !value.chars().any(char::is_control)
}

fn validate_digest(value: &str) -> Result<(), ClaimRevalidationRejectV1> {
    if value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(ClaimRevalidationRejectV1::InvalidDigest)
    }
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes.as_ref().iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_session::RawSessionMessageV1;
    use crate::streaming_session::{
        StreamingMutationKindV1, StreamingSessionMutationV1,
    };
    use crate::Role;

    fn message(turn_id: &str, timestamp_ms: i64, text: &str) -> RawSessionMessageV1 {
        RawSessionMessageV1 {
            turn_id: turn_id.to_owned(),
            role: Role::User,
            timestamp_ms,
            text: text.to_owned(),
            raw_line_sha256: "a".repeat(64),
        }
    }

    fn ledger() -> StreamingSessionLedgerV1 {
        let mut ledger = StreamingSessionLedgerV1::new("user-1", "session-1").unwrap();
        let append = StreamingSessionMutationV1 {
            event_id: "source-event-1".into(),
            sequence: 1,
            effective_at_ms: 100,
            kind: StreamingMutationKindV1::Append {
                message: message("turn-1", 100, "A property has value four."),
            },
        };
        ledger.apply(append, 1_000).unwrap();
        ledger
    }

    fn transition(
        cause: StreamingInvalidationCauseV1,
    ) -> (StreamingSessionLedgerV1, StreamingSessionMutationReceiptV1) {
        let mut ledger = ledger();
        let kind = match cause {
            StreamingInvalidationCauseV1::Correction => StreamingMutationKindV1::Correct {
                target_event_id: "source-event-1".into(),
                message: message("turn-1", 100, "A property has value five."),
            },
            StreamingInvalidationCauseV1::Quarantine => StreamingMutationKindV1::Quarantine {
                target_event_id: "source-event-1".into(),
                reason_sha256: "b".repeat(64),
            },
            StreamingInvalidationCauseV1::Revocation => StreamingMutationKindV1::Revoke {
                target_event_id: "source-event-1".into(),
                revocation_receipt_sha256: "c".repeat(64),
            },
            StreamingInvalidationCauseV1::RightsErasure => StreamingMutationKindV1::ErasureRequest {
                target_event_id: "source-event-1".into(),
                rights_receipt_sha256: "d".repeat(64),
            },
        };
        let receipt = ledger.apply(StreamingSessionMutationV1 {
            event_id: "source-transition-2".into(),
            sequence: 2,
            effective_at_ms: 200,
            kind,
        }, 1_100).unwrap().receipt;
        (ledger, receipt)
    }

    fn binding(transition: &StreamingSessionMutationReceiptV1) -> ClaimRevalidationBindingV1 {
        ClaimRevalidationBindingV1 {
            claim_id: "claim-property-four".into(),
            claim_sha256: "1".repeat(64),
            source_event_id: "source-event-1".into(),
            source_transition_receipt_sha256: transition.receipt_sha256.clone(),
            source_revision_before_sha256: "2".repeat(64),
            source_revision_after_sha256: "3".repeat(64),
            test_artifact_sha256: "4".repeat(64),
            verifier_policy_sha256: "5".repeat(64),
        }
    }

    fn evidence(
        binding: &ClaimRevalidationBindingV1,
        outcome: ClaimRevalidationOutcomeV1,
    ) -> ClaimRevalidationEvidenceV1 {
        ClaimRevalidationEvidenceV1 {
            binding_sha256: binding.binding_sha256(),
            outcome,
            result_sha256: "6".repeat(64),
            execution_receipt_sha256: "7".repeat(64),
            evaluator_id: "claim-test-runner-v1".into(),
        }
    }

    struct FixtureIndependentVerifier;

    impl ClaimRevalidationEvidenceVerifierV1 for FixtureIndependentVerifier {
        fn verify_independent_evidence(
            &self,
            binding: &ClaimRevalidationBindingV1,
            evidence: &ClaimRevalidationEvidenceV1,
        ) -> Option<IndependentClaimRevalidationReceiptV1> {
            if evidence.binding_sha256 != binding.binding_sha256() {
                return None;
            }
            Some(IndependentClaimRevalidationReceiptV1 {
                verifier_id: "fixture-independent-verifier".into(),
                verification_receipt_sha256: "8".repeat(64),
            })
        }
    }

    fn projection_spec(
        projection_id: &str,
        sequence: u64,
        source_event_id: &str,
    ) -> DerivedProjectionSpecV1 {
        DerivedProjectionSpecV1 {
            projection_id: projection_id.to_owned(),
            kind: crate::temporal_projection::DerivedProjectionKindV1::Summary,
            content_sha256: "b".repeat(64),
            derivation_receipt_sha256: "c".repeat(64),
            source_event_ids: vec![source_event_id.to_owned()],
            parent_projection_ids: Vec::new(),
            built_at_sequence: sequence,
        }
    }

    #[test]
    fn passing_claim_revalidation_registers_new_projection_not_old_projection() {
        let mut ledger = ledger();
        let mut index = DerivedProjectionIndexV1::new("user-1", "session-1").unwrap();
        index.register_projection(&ledger, projection_spec(
            "summary-old", 1, "source-event-1",
        )).unwrap();

        let receipt = ledger.apply(StreamingSessionMutationV1 {
            event_id: "source-transition-2".into(),
            sequence: 2,
            effective_at_ms: 100,
            kind: StreamingMutationKindV1::Correct {
                target_event_id: "source-event-1".into(),
                message: message("turn-1", 100, "A property has value five."),
            },
        }, 1_100).unwrap().receipt;
        index.apply_transition(&ledger, &receipt).unwrap();

        let binding = binding(&receipt);
        let evidence = evidence(&binding, ClaimRevalidationOutcomeV1::Pass);
        let assessment = assess_claim_revalidation_v1(
            &ledger, &receipt, &binding, Some(&evidence), Some(&FixtureIndependentVerifier),
        ).unwrap();

        let status = register_rebuilt_projection_after_claim_revalidation_v1(
            &ledger, &mut index, &receipt, &binding, &assessment,
            projection_spec("summary-new", 2, "source-transition-2"),
        ).unwrap();
        assert_eq!(status, ProjectionRegisterStatusV1::Created);
        assert_eq!(
            index.projection("summary-old").unwrap().status(),
            crate::temporal_projection::DerivedProjectionStatusV1::Stale
        );
        assert_eq!(
            index.projection("summary-new").unwrap().status(),
            crate::temporal_projection::DerivedProjectionStatusV1::Current
        );
        assert_eq!(
            index.projection("summary-new").unwrap().source_event_ids(),
            &["source-transition-2".to_owned()]
        );
    }

    #[test]
    fn a_passed_claim_cannot_rebuild_from_the_superseded_source_event() {
        let mut ledger = ledger();
        let mut index = DerivedProjectionIndexV1::new("user-1", "session-1").unwrap();
        index.register_projection(&ledger, projection_spec(
            "summary-old", 1, "source-event-1",
        )).unwrap();
        let receipt = ledger.apply(StreamingSessionMutationV1 {
            event_id: "source-transition-2".into(),
            sequence: 2,
            effective_at_ms: 100,
            kind: StreamingMutationKindV1::Correct {
                target_event_id: "source-event-1".into(),
                message: message("turn-1", 100, "Corrected value."),
            },
        }, 1_100).unwrap().receipt;
        index.apply_transition(&ledger, &receipt).unwrap();
        let binding = binding(&receipt);
        let evidence = evidence(&binding, ClaimRevalidationOutcomeV1::Pass);
        let assessment = assess_claim_revalidation_v1(
            &ledger, &receipt, &binding, Some(&evidence), Some(&FixtureIndependentVerifier),
        ).unwrap();
        assert_eq!(
            register_rebuilt_projection_after_claim_revalidation_v1(
                &ledger, &mut index, &receipt, &binding, &assessment,
                projection_spec("summary-bad", 2, "source-event-1"),
            ),
            Err(ClaimRevalidationRejectV1::ProjectionDoesNotBindCorrectedEvent)
        );
    }

    #[test]
    fn revalidation_cannot_register_projection_before_index_consumes_transition() {
        let mut ledger = ledger();
        let mut index = DerivedProjectionIndexV1::new("user-1", "session-1").unwrap();
        index.register_projection(&ledger, projection_spec(
            "summary-old", 1, "source-event-1",
        )).unwrap();
        let receipt = ledger.apply(StreamingSessionMutationV1 {
            event_id: "source-transition-2".into(),
            sequence: 2,
            effective_at_ms: 100,
            kind: StreamingMutationKindV1::Correct {
                target_event_id: "source-event-1".into(),
                message: message("turn-1", 100, "Corrected value."),
            },
        }, 1_100).unwrap().receipt;
        let binding = binding(&receipt);
        let evidence = evidence(&binding, ClaimRevalidationOutcomeV1::Pass);
        let assessment = assess_claim_revalidation_v1(
            &ledger, &receipt, &binding, Some(&evidence), Some(&FixtureIndependentVerifier),
        ).unwrap();

        assert!(matches!(
            register_rebuilt_projection_after_claim_revalidation_v1(
                &ledger, &mut index, &receipt, &binding, &assessment,
                projection_spec("summary-new", 2, "source-transition-2"),
            ),
            Err(ClaimRevalidationRejectV1::ProjectionRegistrationRejected(
                TemporalProjectionRejectV1::SequenceGap { expected: 1, actual: 2 }
            ))
        ));
    }

    #[test]
    fn failed_or_unverified_claim_never_reaches_projection_registration() {
        let mut ledger = ledger();
        let mut index = DerivedProjectionIndexV1::new("user-1", "session-1").unwrap();
        index.register_projection(&ledger, projection_spec(
            "summary-old", 1, "source-event-1",
        )).unwrap();
        let receipt = ledger.apply(StreamingSessionMutationV1 {
            event_id: "source-transition-2".into(),
            sequence: 2,
            effective_at_ms: 100,
            kind: StreamingMutationKindV1::Correct {
                target_event_id: "source-event-1".into(),
                message: message("turn-1", 100, "Corrected value."),
            },
        }, 1_100).unwrap().receipt;
        index.apply_transition(&ledger, &receipt).unwrap();
        let binding = binding(&receipt);
        let evidence = evidence(&binding, ClaimRevalidationOutcomeV1::Fail);
        let assessment = assess_claim_revalidation_v1(
            &ledger, &receipt, &binding, Some(&evidence), Some(&FixtureIndependentVerifier),
        ).unwrap();
        assert_eq!(
            register_rebuilt_projection_after_claim_revalidation_v1(
                &ledger, &mut index, &receipt, &binding, &assessment,
                projection_spec("summary-new", 2, "source-transition-2"),
            ),
            Err(ClaimRevalidationRejectV1::RebuildNotPermitted)
        );
    }

    #[test]
    fn assessment_commits_to_claim_source_and_transition_identity() {
        let (ledger, receipt) = transition(StreamingInvalidationCauseV1::Correction);
        let binding = binding(&receipt);
        let evidence = evidence(&binding, ClaimRevalidationOutcomeV1::Pass);
        let assessment = assess_claim_revalidation_v1(
            &ledger, &receipt, &binding, Some(&evidence), Some(&FixtureIndependentVerifier),
        ).unwrap();
        assert!(assessment.verify_integrity());

        let mut altered_claim = assessment.clone();
        altered_claim.claim_sha256 = "9".repeat(64);
        assert!(!altered_claim.verify_integrity());

        let mut altered_source = assessment.clone();
        altered_source.source_event_id = "other-source".into();
        assert!(!altered_source.verify_integrity());

        let mut altered_transition = assessment;
        altered_transition.source_transition_receipt_sha256 = "8".repeat(64);
        assert!(!altered_transition.verify_integrity());
    }

    #[test]
    fn independent_pass_preserves_claim_but_requires_new_projection() {
        let (ledger, transition) = transition(StreamingInvalidationCauseV1::Correction);
        let binding = binding(&transition);
        let evidence = evidence(&binding, ClaimRevalidationOutcomeV1::Pass);
        let assessment = assess_claim_revalidation_v1(
            &ledger, &transition, &binding, Some(&evidence), Some(&FixtureIndependentVerifier),
        ).unwrap();
        assert_eq!(assessment.disposition, ClaimRevalidationDispositionV1::PreservedNeedsRebuild);
        assert!(assessment.new_projection_may_be_built);
        assert!(!assessment.old_projection_may_be_reactivated);
        assert!(assessment.verify_integrity());
    }

    #[test]
    fn independent_failure_invalidates_only_the_bound_claim() {
        let (ledger, transition) = transition(StreamingInvalidationCauseV1::Correction);
        let binding = binding(&transition);
        let evidence = evidence(&binding, ClaimRevalidationOutcomeV1::Fail);
        let assessment = assess_claim_revalidation_v1(
            &ledger, &transition, &binding, Some(&evidence), Some(&FixtureIndependentVerifier),
        ).unwrap();
        assert_eq!(assessment.disposition, ClaimRevalidationDispositionV1::Invalidated);
        assert!(!assessment.new_projection_may_be_built);
        assert!(!assessment.old_projection_may_be_reactivated);
    }

    #[test]
    fn missing_or_untrusted_evidence_requires_revalidation() {
        let (ledger, transition) = transition(StreamingInvalidationCauseV1::Correction);
        let binding = binding(&transition);
        let missing = assess_claim_revalidation_v1(&ledger, &transition, &binding, None, None).unwrap();
        assert_eq!(missing.disposition, ClaimRevalidationDispositionV1::RevalidationRequired);

        let evidence = evidence(&binding, ClaimRevalidationOutcomeV1::Pass);
        let unverified = assess_claim_revalidation_v1(
            &ledger, &transition, &binding, Some(&evidence), None,
        ).unwrap();
        assert_eq!(unverified.disposition, ClaimRevalidationDispositionV1::RevalidationRequired);
        assert!(!unverified.new_projection_may_be_built);
    }

    #[test]
    fn skipped_infrastructure_failure_and_inconclusive_results_do_not_preserve_claim() {
        let (ledger, transition) = transition(StreamingInvalidationCauseV1::Correction);
        let binding = binding(&transition);
        for outcome in [
            ClaimRevalidationOutcomeV1::NotRun,
            ClaimRevalidationOutcomeV1::Skipped,
            ClaimRevalidationOutcomeV1::InfrastructureFailure,
            ClaimRevalidationOutcomeV1::Inconclusive,
        ] {
            let evidence = evidence(&binding, outcome);
            let assessment = assess_claim_revalidation_v1(
                &ledger, &transition, &binding, Some(&evidence), Some(&FixtureIndependentVerifier),
            ).unwrap();
            assert_eq!(assessment.disposition, ClaimRevalidationDispositionV1::RevalidationRequired);
            assert!(!assessment.new_projection_may_be_built);
        }
    }

    #[test]
    fn quarantine_revocation_and_erasure_override_a_passing_check() {
        let verifier = FixtureIndependentVerifier;
        for (cause, expected) in [
            (StreamingInvalidationCauseV1::Quarantine, ClaimRevalidationDispositionV1::Quarantined),
            (StreamingInvalidationCauseV1::Revocation, ClaimRevalidationDispositionV1::Revoked),
            (StreamingInvalidationCauseV1::RightsErasure, ClaimRevalidationDispositionV1::ErasureRequired),
        ] {
            let (ledger, transition) = transition(cause);
            let binding = binding(&transition);
            let evidence = evidence(&binding, ClaimRevalidationOutcomeV1::Pass);
            let assessment = assess_claim_revalidation_v1(
                &ledger, &transition, &binding, Some(&evidence), Some(&verifier),
            ).unwrap();
            assert_eq!(assessment.disposition, expected);
            assert!(!assessment.new_projection_may_be_built);
            assert!(!assessment.old_projection_may_be_reactivated);
        }
    }

    #[test]
    fn mismatched_source_and_transition_bindings_are_rejected() {
        let (ledger, transition) = transition(StreamingInvalidationCauseV1::Correction);
        let mut binding = binding(&transition);
        binding.source_event_id = "some-other-source".into();
        assert_eq!(
            assess_claim_revalidation_v1(&ledger, &transition, &binding, None, None),
            Err(ClaimRevalidationRejectV1::SourceEventNotAffectedByTransition)
        );

        let mut binding = binding(&transition);
        binding.source_transition_receipt_sha256 = "9".repeat(64);
        assert_eq!(
            assess_claim_revalidation_v1(&ledger, &transition, &binding, None, None),
            Err(ClaimRevalidationRejectV1::TransitionBindingMismatch)
        );
    }

    #[test]
    fn evidence_for_another_claim_cannot_be_replayed() {
        let (ledger, transition) = transition(StreamingInvalidationCauseV1::Correction);
        let binding = binding(&transition);
        let mut evidence = evidence(&binding, ClaimRevalidationOutcomeV1::Pass);
        evidence.binding_sha256 = "a".repeat(64);
        assert_eq!(
            assess_claim_revalidation_v1(
                &ledger, &transition, &binding, Some(&evidence), Some(&FixtureIndependentVerifier),
            ),
            Err(ClaimRevalidationRejectV1::EvidenceBindingMismatch)
        );
    }

    #[test]
    fn unrecorded_transition_from_another_ledger_is_rejected() {
        let (ledger, transition) = transition(StreamingInvalidationCauseV1::Correction);
        let mut other = StreamingSessionLedgerV1::new("user-1", "session-1").unwrap();
        let append = StreamingSessionMutationV1 {
            event_id: "source-event-1".into(),
            sequence: 1,
            effective_at_ms: 100,
            kind: StreamingMutationKindV1::Append {
                message: message("turn-1", 100, "A different source."),
            },
        };
        other.apply(append, 1_000).unwrap();
        let foreign = other.apply(StreamingSessionMutationV1 {
            event_id: "source-transition-2".into(),
            sequence: 2,
            effective_at_ms: 200,
            kind: StreamingMutationKindV1::Correct {
                target_event_id: "source-event-1".into(),
                message: message("turn-1", 100, "Corrected different source."),
            },
        }, 1_100).unwrap().receipt;
        let binding = binding(&transition);
        assert!(foreign.verify_integrity());
        assert_eq!(
            assess_claim_revalidation_v1(&ledger, &foreign, &binding, None, None),
            Err(ClaimRevalidationRejectV1::UnrecordedSourceTransition)
        );
    }

    #[test]
    fn transition_without_invalidation_cause_cannot_revalidate_claim() {
        let mut ledger = ledger();
        let append = StreamingSessionMutationV1 {
            event_id: "event-2".into(),
            sequence: 2,
            effective_at_ms: 150,
            kind: StreamingMutationKindV1::Append {
                message: message("turn-2", 150, "An unrelated fact."),
            },
        };
        let transition = ledger.apply(append, 1_100).unwrap().receipt;
        let mut binding = binding(&transition);
        binding.source_transition_receipt_sha256 = transition.receipt_sha256.clone();
        assert_eq!(
            assess_claim_revalidation_v1(&ledger, &transition, &binding, None, None),
            Err(ClaimRevalidationRejectV1::TransitionHasNoInvalidationCause)
        );
    }
}
