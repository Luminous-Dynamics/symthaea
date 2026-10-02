// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure transition commitments built from semantic state commitments.
//!
//! A state digest answers "what state is this?"; a transition commitment answers
//! "which before/after state pair and semantic operation are being claimed?".
//! It is not a signature and does not establish who performed the transition.

use crate::semantic_admission::SemanticAdmissionState;
use crate::semantic_digest::{semantic_digest, SemanticDigestError};

pub const ALGORITHM: &str = "BLAKE3-256";
pub const VERSION: u16 = 1;
pub const DOMAIN: &[u8] = b"symthaea-swarm/semantic-transition";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransitionKind {
    Admission,
    Replay,
    LifecycleRetirement,
}

impl TransitionKind {
    fn tag(self) -> u8 {
        match self {
            Self::Admission => 1,
            Self::Replay => 2,
            Self::LifecycleRetirement => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransitionCommitment([u8; 32]);

impl TransitionCommitment {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum TransitionCommitmentError {
    #[error("semantic state digest failed: {0}")]
    Digest(#[from] SemanticDigestError),
}

pub fn transition_commitment(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    kind: TransitionKind,
) -> Result<TransitionCommitment, TransitionCommitmentError> {
    let before = semantic_digest(before)?;
    let after = semantic_digest(after)?;

    let mut hasher = blake3::Hasher::new();
    hasher.update(&(DOMAIN.len() as u64).to_be_bytes());
    hasher.update(DOMAIN);
    hasher.update(&VERSION.to_be_bytes());
    hasher.update(&[kind.tag()]);
    hasher.update(before.as_bytes());
    hasher.update(after.as_bytes());
    Ok(TransitionCommitment(*hasher.finalize().as_bytes()))
}


/// A commitment to the semantic operation inputs in addition to the state edge.
///
/// TransitionCommitment identifies a (before, after, kind) edge. This
/// commitment additionally binds the logical evaluation time, admission
/// policy, operation inputs, and semantic result. It therefore prevents two
/// distinct operation invocations that happen to produce the same state edge
/// from being indistinguishable at the evidence layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransitionClaimCommitment([u8; 32]);

impl TransitionClaimCommitment {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

pub const CLAIM_DOMAIN: &[u8] = b"symthaea-swarm/semantic-transition-claim";
pub const CLAIM_VERSION: u16 = 1;

pub fn admission_claim_commitment(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    delivery: &crate::semantic_admission::DeliveryContract,
    observation: &crate::semantic_admission::ObservationRecord,
    policy: crate::semantic_admission::AdmissionPolicy,
    now_ms: u64,
    result: &crate::semantic_admission::SemanticResult,
) -> Result<TransitionClaimCommitment, TransitionCommitmentError> {
    let before_digest = semantic_digest(before)?;
    let after_digest = semantic_digest(after)?;
    let mut hasher = blake3::Hasher::new();
    write_claim_header(
        &mut hasher,
        TransitionKind::Admission,
        &before_digest,
        &after_digest,
        policy,
        now_ms,
    );
    hasher.update(delivery.logical_delivery_id.as_bytes());
    hasher.update(&delivery.schema_version.to_be_bytes());
    hasher.update(&delivery.expires_at_ms.to_be_bytes());
    write_bytes(&mut hasher, &delivery.payload);
    hasher.update(observation.key.observation_id.as_bytes());
    write_bytes(&mut hasher, observation.key.namespace.as_bytes());
    hasher.update(observation.source_id.as_bytes());
    hasher.update(&observation.observed_at_ms.to_be_bytes());
    write_bytes(&mut hasher, &observation.payload);
    hasher.update(result.logical_delivery_id.as_bytes());
    hasher.update(result.observation.observation_id.as_bytes());
    write_bytes(&mut hasher, result.observation.namespace.as_bytes());
    Ok(TransitionClaimCommitment(*hasher.finalize().as_bytes()))
}

pub fn replay_claim_commitment(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    delivery: &crate::semantic_admission::DeliveryContract,
    observation: &crate::semantic_admission::ObservationRecord,
    policy: crate::semantic_admission::AdmissionPolicy,
    now_ms: u64,
    result: &crate::semantic_admission::SemanticResult,
) -> Result<TransitionClaimCommitment, TransitionCommitmentError> {
    let before_digest = semantic_digest(before)?;
    let after_digest = semantic_digest(after)?;
    let mut hasher = blake3::Hasher::new();
    write_claim_header(
        &mut hasher,
        TransitionKind::Replay,
        &before_digest,
        &after_digest,
        policy,
        now_ms,
    );
    hasher.update(delivery.logical_delivery_id.as_bytes());
    hasher.update(&delivery.schema_version.to_be_bytes());
    hasher.update(&delivery.expires_at_ms.to_be_bytes());
    write_bytes(&mut hasher, &delivery.payload);
    hasher.update(observation.key.observation_id.as_bytes());
    write_bytes(&mut hasher, observation.key.namespace.as_bytes());
    hasher.update(observation.source_id.as_bytes());
    hasher.update(&observation.observed_at_ms.to_be_bytes());
    write_bytes(&mut hasher, &observation.payload);
    hasher.update(result.logical_delivery_id.as_bytes());
    hasher.update(result.observation.observation_id.as_bytes());
    write_bytes(&mut hasher, result.observation.namespace.as_bytes());
    Ok(TransitionClaimCommitment(*hasher.finalize().as_bytes()))
}

pub fn lifecycle_claim_commitment(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    policy: crate::semantic_admission::AdmissionPolicy,
    now_ms: u64,
) -> Result<TransitionClaimCommitment, TransitionCommitmentError> {
    let before_digest = semantic_digest(before)?;
    let after_digest = semantic_digest(after)?;
    let mut hasher = blake3::Hasher::new();
    write_claim_header(
        &mut hasher,
        TransitionKind::LifecycleRetirement,
        &before_digest,
        &after_digest,
        policy,
        now_ms,
    );
    Ok(TransitionClaimCommitment(*hasher.finalize().as_bytes()))
}

fn write_claim_header(
    hasher: &mut blake3::Hasher,
    kind: TransitionKind,
    before: &crate::semantic_digest::SemanticDigest,
    after: &crate::semantic_digest::SemanticDigest,
    policy: crate::semantic_admission::AdmissionPolicy,
    now_ms: u64,
) {
    hasher.update(&(CLAIM_DOMAIN.len() as u64).to_be_bytes());
    hasher.update(CLAIM_DOMAIN);
    hasher.update(&CLAIM_VERSION.to_be_bytes());
    hasher.update(&[kind.tag()]);
    hasher.update(before.as_bytes());
    hasher.update(after.as_bytes());
    hasher.update(&now_ms.to_be_bytes());
    hasher.update(&[policy.allow_new_observation as u8]);
    hasher.update(&(policy.max_deliveries as u64).to_be_bytes());
    hasher.update(&(policy.max_observations as u64).to_be_bytes());
    hasher.update(&policy.retention_ms.to_be_bytes());
    hasher.update(&policy.tombstone_retention_ms.to_be_bytes());
}

fn write_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

/// Typed, owned description of a semantic operation claim.
///
/// This groups complete operation inputs into one value so callers can retain,
/// transport, and verify a claim without mixing fields from different calls.
/// It is evidence data, not an assertion of authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionClaim {
    Admission {
        delivery: crate::semantic_admission::DeliveryContract,
        observation: crate::semantic_admission::ObservationRecord,
        policy: crate::semantic_admission::AdmissionPolicy,
        now_ms: u64,
        result: crate::semantic_admission::SemanticResult,
    },
    Replay {
        delivery: crate::semantic_admission::DeliveryContract,
        observation: crate::semantic_admission::ObservationRecord,
        policy: crate::semantic_admission::AdmissionPolicy,
        now_ms: u64,
        result: crate::semantic_admission::SemanticResult,
    },
    LifecycleRetirement {
        policy: crate::semantic_admission::AdmissionPolicy,
        now_ms: u64,
    },
}

/// Commit a typed operation claim. Operation-specific helpers remain available
/// for compatibility; new callers should prefer this typed dispatcher.
pub fn transition_claim_commitment(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    claim: &TransitionClaim,
) -> Result<TransitionClaimCommitment, TransitionCommitmentError> {
    match claim {
        TransitionClaim::Admission { delivery, observation, policy, now_ms, result } =>
            admission_claim_commitment(before, after, delivery, observation, *policy, *now_ms, result),
        TransitionClaim::Replay { delivery, observation, policy, now_ms, result } =>
            replay_claim_commitment(before, after, delivery, observation, *policy, *now_ms, result),
        TransitionClaim::LifecycleRetirement { policy, now_ms } =>
            lifecycle_claim_commitment(before, after, *policy, *now_ms),
    }
}

/// Version of the self-contained replay evidence envelope.
pub const EVIDENCE_VERSION: u16 = 1;

/// Self-contained semantic transition evidence.
///
/// This bundle is deliberately unsigned. It establishes deterministic
/// semantic validity, while authentication, authorization, and
/// non-repudiation remain separate layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionEvidence {
    pub version: u16,
    pub kind: TransitionKind,
    pub before_digest: crate::semantic_digest::SemanticDigest,
    pub after_digest: crate::semantic_digest::SemanticDigest,
    pub transition_commitment: TransitionCommitment,
    pub claim_commitment: TransitionClaimCommitment,
    pub claim: TransitionClaim,
}

impl TransitionEvidence {
    pub fn kind(&self) -> TransitionKind {
        self.kind
    }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum TransitionEvidenceError {
    #[error("unsupported transition evidence version: {0}")]
    UnsupportedVersion(u16),
    #[error("evidence kind does not match typed claim")]
    KindMismatch,
    #[error("before-state digest does not match evidence")]
    BeforeDigestMismatch,
    #[error("after-state digest does not match evidence")]
    AfterDigestMismatch,
    #[error("transition verification failed: {0}")]
    Verification(#[from] TransitionVerificationError),
    #[error("transition commitment could not be reconstructed: {0}")]
    Commitment(#[from] TransitionCommitmentError),
    #[error("semantic state digest could not be reconstructed: {0}")]
    Digest(#[from] SemanticDigestError),
}

/// Build replayable evidence from a semantic state edge and its complete
/// typed operation claim.
pub fn build_transition_evidence(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    claim: TransitionClaim,
) -> Result<TransitionEvidence, TransitionEvidenceError> {
    let kind = match &claim {
        TransitionClaim::Admission { .. } => TransitionKind::Admission,
        TransitionClaim::Replay { .. } => TransitionKind::Replay,
        TransitionClaim::LifecycleRetirement { .. } => TransitionKind::LifecycleRetirement,
    };
    let before_digest = semantic_digest(before)?;
    let after_digest = semantic_digest(after)?;
    let transition_commitment = transition_commitment(before, after, kind)?;
    let claim_commitment = transition_claim_commitment(before, after, &claim)?;

    Ok(TransitionEvidence {
        version: EVIDENCE_VERSION,
        kind,
        before_digest,
        after_digest,
        transition_commitment,
        claim_commitment,
        claim,
    })
}

/// Verify a complete evidence envelope against the actual semantic state edge.
///
/// Verification checks the envelope version/kind and state digests, then
/// replays the operation-specific pure oracle and verifies both commitments.
pub fn verify_transition_evidence(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    evidence: &TransitionEvidence,
) -> Result<(), TransitionEvidenceError> {
    if evidence.version != EVIDENCE_VERSION {
        return Err(TransitionEvidenceError::UnsupportedVersion(evidence.version));
    }

    let expected_kind = match &evidence.claim {
        TransitionClaim::Admission { .. } => TransitionKind::Admission,
        TransitionClaim::Replay { .. } => TransitionKind::Replay,
        TransitionClaim::LifecycleRetirement { .. } => TransitionKind::LifecycleRetirement,
    };
    if evidence.kind != expected_kind {
        return Err(TransitionEvidenceError::KindMismatch);
    }

    if semantic_digest(before)? != evidence.before_digest {
        return Err(TransitionEvidenceError::BeforeDigestMismatch);
    }
    if semantic_digest(after)? != evidence.after_digest {
        return Err(TransitionEvidenceError::AfterDigestMismatch);
    }

    match &evidence.claim {
        TransitionClaim::Admission {
            delivery, observation, policy, now_ms, result,
        } => verify_admission_transition_claim(
            before, after, delivery, observation, *policy, *now_ms, result,
            &evidence.transition_commitment, &evidence.claim_commitment,
        )?,
        TransitionClaim::Replay {
            delivery, observation, policy, now_ms, result,
        } => verify_replay_transition_claim(
            before, after, delivery, observation, *policy, *now_ms, result,
            &evidence.transition_commitment, &evidence.claim_commitment,
        )?,
        TransitionClaim::LifecycleRetirement { policy, now_ms } =>
            verify_lifecycle_retirement_claim(
                before, after, *policy, *now_ms,
                &evidence.transition_commitment, &evidence.claim_commitment,
            )?,
    }

    Ok(())
}

/// Failure returned when a claimed transition cannot be reconstructed from the
/// semantic transition oracle and its committed state edge.
///
/// Verification is deliberately stronger than checking hashes: the verifier
/// re-executes the same pure semantic rule against the claimed inputs and then
/// checks that the resulting state, semantic result, and transition commitment
/// all agree. A successful verification therefore establishes replayable
/// transition validity, not signer identity or non-repudiation.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum TransitionVerificationError {
    #[error("before state violates semantic invariants: {0:?}")]
    InvalidBeforeState(crate::semantic_admission::StateInvariant),
    #[error("after state violates semantic invariants: {0:?}")]
    InvalidAfterState(crate::semantic_admission::StateInvariant),
    #[error("admission oracle did not produce an admission")]
    AdmissionNotAdmitted,
    #[error("replay oracle did not produce an exact replay")]
    ReplayNotReplayed,
    #[error("lifecycle oracle produced no semantic state change")]
    NoStateChange,
    #[error("claimed after state differs from oracle output")]
    AfterStateMismatch,
    #[error("claimed semantic result differs from oracle output")]
    ResultMismatch,
    #[error("claimed transition commitment does not match the reconstructed transition")]
    CommitmentMismatch,
    #[error("claimed operation claim commitment does not match the reconstructed claim")]
    ClaimCommitmentMismatch,
    #[error("transition commitment could not be reconstructed: {0}")]
    Commitment(#[from] TransitionCommitmentError),
}

/// Verify an admission transition by replaying the pure admission oracle.
///
/// The caller supplies all semantic inputs explicitly. No clock, storage,
/// transport metadata, or mutable state is consulted by this verifier.
pub fn verify_admission_transition(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    delivery: &crate::semantic_admission::DeliveryContract,
    observation: &crate::semantic_admission::ObservationRecord,
    policy: crate::semantic_admission::AdmissionPolicy,
    now_ms: u64,
    claimed_result: &crate::semantic_admission::SemanticResult,
    claimed_commitment: &TransitionCommitment,
) -> Result<(), TransitionVerificationError> {
    use crate::semantic_admission::{decide, validate_state, AdmissionOutcome};

    validate_state(before).map_err(TransitionVerificationError::InvalidBeforeState)?;
    validate_state(after).map_err(TransitionVerificationError::InvalidAfterState)?;

    let outcome = decide(before, delivery, observation, policy, now_ms);
    let AdmissionOutcome::Admitted { next_state, result } = outcome else {
        return Err(TransitionVerificationError::AdmissionNotAdmitted);
    };

    if next_state != *after {
        return Err(TransitionVerificationError::AfterStateMismatch);
    }
    if result != *claimed_result {
        return Err(TransitionVerificationError::ResultMismatch);
    }

    verify_commitment(
        before,
        after,
        TransitionKind::Admission,
        claimed_commitment,
    )
}

/// Verify an admission transition and its operation-level claim commitment.
pub fn verify_admission_transition_claim(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    delivery: &crate::semantic_admission::DeliveryContract,
    observation: &crate::semantic_admission::ObservationRecord,
    policy: crate::semantic_admission::AdmissionPolicy,
    now_ms: u64,
    claimed_result: &crate::semantic_admission::SemanticResult,
    claimed_commitment: &TransitionCommitment,
    claimed_claim: &TransitionClaimCommitment,
) -> Result<(), TransitionVerificationError> {
    verify_admission_transition(
        before,
        after,
        delivery,
        observation,
        policy,
        now_ms,
        claimed_result,
        claimed_commitment,
    )?;
    let expected = admission_claim_commitment(
        before,
        after,
        delivery,
        observation,
        policy,
        now_ms,
        claimed_result,
    )?;
    if expected != *claimed_claim {
        return Err(TransitionVerificationError::ClaimCommitmentMismatch);
    }
    Ok(())
}

/// Verify an exact semantic replay.
///
/// A replay is intentionally a no-op at the semantic-state layer. Transport
/// retries may differ while this verifier requires the same semantic result and
/// an unchanged before/after state pair.
pub fn verify_replay_transition(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    delivery: &crate::semantic_admission::DeliveryContract,
    observation: &crate::semantic_admission::ObservationRecord,
    policy: crate::semantic_admission::AdmissionPolicy,
    now_ms: u64,
    claimed_result: &crate::semantic_admission::SemanticResult,
    claimed_commitment: &TransitionCommitment,
) -> Result<(), TransitionVerificationError> {
    use crate::semantic_admission::{decide, validate_state, AdmissionOutcome};

    validate_state(before).map_err(TransitionVerificationError::InvalidBeforeState)?;
    validate_state(after).map_err(TransitionVerificationError::InvalidAfterState)?;

    let outcome = decide(before, delivery, observation, policy, now_ms);
    let AdmissionOutcome::Replay { existing_result } = outcome else {
        return Err(TransitionVerificationError::ReplayNotReplayed);
    };

    if before != after {
        return Err(TransitionVerificationError::AfterStateMismatch);
    }
    if existing_result != *claimed_result {
        return Err(TransitionVerificationError::ResultMismatch);
    }

    verify_commitment(
        before,
        after,
        TransitionKind::Replay,
        claimed_commitment,
    )
}

/// Verify a replay transition and its operation-level claim commitment.
pub fn verify_replay_transition_claim(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    delivery: &crate::semantic_admission::DeliveryContract,
    observation: &crate::semantic_admission::ObservationRecord,
    policy: crate::semantic_admission::AdmissionPolicy,
    now_ms: u64,
    claimed_result: &crate::semantic_admission::SemanticResult,
    claimed_commitment: &TransitionCommitment,
    claimed_claim: &TransitionClaimCommitment,
) -> Result<(), TransitionVerificationError> {
    verify_replay_transition(
        before,
        after,
        delivery,
        observation,
        policy,
        now_ms,
        claimed_result,
        claimed_commitment,
    )?;
    let expected = replay_claim_commitment(
        before,
        after,
        delivery,
        observation,
        policy,
        now_ms,
        claimed_result,
    )?;
    if expected != *claimed_claim {
        return Err(TransitionVerificationError::ClaimCommitmentMismatch);
    }
    Ok(())
}

/// Verify a lifecycle-retirement transition by replaying the pure GC oracle.
///
/// Lifecycle retirement must actually change semantic state. A no-op lifecycle
/// check is policy evaluation, not a transition, and therefore has no valid
/// retirement commitment.
pub fn verify_lifecycle_retirement(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    policy: crate::semantic_admission::AdmissionPolicy,
    now_ms: u64,
    claimed_commitment: &TransitionCommitment,
) -> Result<(), TransitionVerificationError> {
    use crate::semantic_admission::{retire_expired, validate_state};

    validate_state(before).map_err(TransitionVerificationError::InvalidBeforeState)?;
    validate_state(after).map_err(TransitionVerificationError::InvalidAfterState)?;

    let next_state = retire_expired(before, policy, now_ms)
        .map_err(TransitionVerificationError::InvalidBeforeState)?;

    if next_state == *before {
        return Err(TransitionVerificationError::NoStateChange);
    }
    if next_state != *after {
        return Err(TransitionVerificationError::AfterStateMismatch);
    }

    verify_commitment(
        before,
        after,
        TransitionKind::LifecycleRetirement,
        claimed_commitment,
    )
}

/// Verify a lifecycle transition and its operation-level claim commitment.
pub fn verify_lifecycle_retirement_claim(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    policy: crate::semantic_admission::AdmissionPolicy,
    now_ms: u64,
    claimed_commitment: &TransitionCommitment,
    claimed_claim: &TransitionClaimCommitment,
) -> Result<(), TransitionVerificationError> {
    verify_lifecycle_retirement(before, after, policy, now_ms, claimed_commitment)?;
    let expected = lifecycle_claim_commitment(before, after, policy, now_ms)?;
    if expected != *claimed_claim {
        return Err(TransitionVerificationError::ClaimCommitmentMismatch);
    }
    Ok(())
}

fn verify_commitment(
    before: &SemanticAdmissionState,
    after: &SemanticAdmissionState,
    kind: TransitionKind,
    claimed_commitment: &TransitionCommitment,
) -> Result<(), TransitionVerificationError> {
    let expected = transition_commitment(before, after, kind)?;
    if expected != *claimed_commitment {
        return Err(TransitionVerificationError::CommitmentMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_admission::{
        decide, AdmissionOutcome, AdmissionPolicy, DeliveryContract, ObservationKey,
        ObservationRecord,
    };
    use uuid::Uuid;

    #[test]
    fn typed_claim_dispatch_matches_legacy_encoder_and_separates_kinds() {
        let (before, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        let policy = AdmissionPolicy { allow_new_observation: true, ..AdmissionPolicy::default() };
        let result = crate::semantic_admission::SemanticResult {
            logical_delivery_id: delivery.logical_delivery_id,
            observation: observation.key.clone(),
        };
        let admission = TransitionClaim::Admission {
            delivery: delivery.clone(), observation: observation.clone(), policy, now_ms: 10, result: result.clone(),
        };
        let replay = TransitionClaim::Replay { delivery: delivery.clone(), observation: observation.clone(), policy, now_ms: 10, result: result.clone() };
        assert_eq!(
            transition_claim_commitment(&before, &after, &admission).unwrap(),
            admission_claim_commitment(&before, &after, &delivery, &observation, policy, 10, &result).unwrap(),
        );
        assert_ne!(
            transition_claim_commitment(&before, &after, &admission).unwrap(),
            transition_claim_commitment(&before, &after, &replay).unwrap(),
        );
    }

    #[test]
    fn admission_evidence_round_trips_through_verifier() {
        let (before, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let result = match decide(&before, &delivery, &observation, policy, 10) {
            AdmissionOutcome::Admitted { result, .. } => result,
            other => panic!("fixture admission failed: {other:?}"),
        };
        let evidence = build_transition_evidence(
            &before,
            &after,
            TransitionClaim::Admission {
                delivery,
                observation,
                policy,
                now_ms: 10,
                result,
            },
        ).unwrap();

        assert_eq!(evidence.version, EVIDENCE_VERSION);
        assert_eq!(evidence.kind(), TransitionKind::Admission);
        assert_eq!(verify_transition_evidence(&before, &after, &evidence), Ok(()));
    }

    #[test]
    fn replay_evidence_round_trips_as_semantic_noop() {
        let (_, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let result = match decide(&after, &delivery, &observation, policy, 11) {
            AdmissionOutcome::Replay { existing_result } => existing_result,
            other => panic!("fixture replay failed: {other:?}"),
        };
        let evidence = build_transition_evidence(
            &after,
            &after,
            TransitionClaim::Replay {
                delivery,
                observation,
                policy,
                now_ms: 11,
                result,
            },
        ).unwrap();

        assert_eq!(evidence.kind(), TransitionKind::Replay);
        assert_eq!(verify_transition_evidence(&after, &after, &evidence), Ok(()));
    }

    #[test]
    fn lifecycle_evidence_round_trips_retirement() {
        use crate::semantic_admission::retire_expired;

        let (mut before, _) = fixture();
        let observation = before.observations.values().next().unwrap().clone();
        before.observations.get_mut(&observation.key).unwrap().observed_at_ms = 0;
        let policy = AdmissionPolicy {
            retention_ms: 10,
            tombstone_retention_ms: 100,
            ..AdmissionPolicy::default()
        };
        let after = retire_expired(&before, policy, 11).unwrap();
        let evidence = build_transition_evidence(
            &before,
            &after,
            TransitionClaim::LifecycleRetirement {
                policy,
                now_ms: 11,
            },
        ).unwrap();

        assert_eq!(evidence.kind(), TransitionKind::LifecycleRetirement);
        assert_eq!(verify_transition_evidence(&before, &after, &evidence), Ok(()));
    }

    #[test]
    fn evidence_rejects_tampered_state_digest() {
        let (before, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let result = match decide(&before, &delivery, &observation, policy, 10) {
            AdmissionOutcome::Admitted { result, .. } => result,
            other => panic!("fixture admission failed: {other:?}"),
        };
        let mut evidence = build_transition_evidence(
            &before,
            &after,
            TransitionClaim::Admission {
                delivery,
                observation,
                policy,
                now_ms: 10,
                result,
            },
        ).unwrap();

        evidence.before_digest = evidence.after_digest;
        assert_eq!(
            verify_transition_evidence(&before, &after, &evidence),
            Err(TransitionEvidenceError::BeforeDigestMismatch)
        );
    }

    #[test]
    fn evidence_rejects_mismatched_claim_kind() {
        let (before, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let result = crate::semantic_admission::SemanticResult {
            logical_delivery_id: delivery.logical_delivery_id,
            observation: observation.key.clone(),
        };
        let mut evidence = build_transition_evidence(
            &before,
            &after,
            TransitionClaim::Admission {
                delivery,
                observation,
                policy,
                now_ms: 10,
                result,
            },
        ).unwrap();

        evidence.kind = TransitionKind::Replay;
        assert_eq!(
            verify_transition_evidence(&before, &after, &evidence),
            Err(TransitionEvidenceError::KindMismatch)
        );
    }

    fn fixture() -> (SemanticAdmissionState, SemanticAdmissionState) {
        let delivery = DeliveryContract {
            logical_delivery_id: Uuid::from_u128(1),
            schema_version: 1,
            expires_at_ms: 1_000,
            payload: b"delivery".to_vec(),
        };
        let observation = ObservationRecord {
            key: ObservationKey {
                namespace: "source".into(),
                observation_id: Uuid::from_u128(2),
            },
            source_id: Uuid::from_u128(3),
            observed_at_ms: 10,
            payload: b"observation".to_vec(),
        };
        let before = SemanticAdmissionState::default();
        let after = match decide(
            &before,
            &delivery,
            &observation,
            &AdmissionPolicy {
                allow_new_observation: true,
                ..AdmissionPolicy::default()
            },
            10,
        ) {
            AdmissionOutcome::Admitted { next_state, .. } => next_state,
            other => panic!("fixture admission failed: {other:?}"),
        };
        (before, after)
    }

    #[test]
    fn transition_commitment_is_deterministic() {
        let (before, after) = fixture();
        assert_eq!(
            transition_commitment(&before, &after, TransitionKind::Admission).unwrap(),
            transition_commitment(&before, &after, TransitionKind::Admission).unwrap()
        );
    }

    #[test]
    fn transition_kind_is_committed() {
        let (before, after) = fixture();
        assert_ne!(
            transition_commitment(&before, &after, TransitionKind::Admission).unwrap(),
            transition_commitment(&before, &after, TransitionKind::LifecycleRetirement).unwrap()
        );
    }

    #[test]
    fn direction_is_committed() {
        let (before, after) = fixture();
        assert_ne!(
            transition_commitment(&before, &after, TransitionKind::Admission).unwrap(),
            transition_commitment(&after, &before, TransitionKind::Admission).unwrap()
        );
    }

    #[test]
    fn same_state_pair_has_same_commitment_regardless_of_construction_history() {
        let (before, after) = fixture();
        let mut rebuilt = SemanticAdmissionState::default();
        for (id, delivery) in &after.deliveries {
            rebuilt.deliveries.insert(*id, delivery.clone());
        }
        for (key, observation) in &after.observations {
            rebuilt.observations.insert(key.clone(), observation.clone());
        }
        for (key, result) in &after.results {
            rebuilt.results.insert(key.clone(), result.clone());
        }
        for (id, tombstone) in &after.delivery_tombstones {
            rebuilt.delivery_tombstones.insert(*id, tombstone.clone());
        }
        for (key, tombstone) in &after.observation_tombstones {
            rebuilt.observation_tombstones.insert(key.clone(), tombstone.clone());
        }

        assert_eq!(
            transition_commitment(&before, &after, TransitionKind::Admission).unwrap(),
            transition_commitment(&before, &rebuilt, TransitionKind::Admission).unwrap()
        );
    }

    #[test]
    fn transition_commitment_is_not_an_authenticator() {
        // This test documents the security boundary: the commitment contains
        // no signer identity. Authentication belongs to a separate layer.
        let (before, after) = fixture();
        let commitment = transition_commitment(&before, &after, TransitionKind::Admission).unwrap();
        assert_eq!(commitment.as_bytes().len(), 32);
    }

    #[test]
    fn valid_admission_transition_is_replayably_verifiable() {
        use crate::semantic_admission::{decide, AdmissionOutcome, AdmissionPolicy};

        let (before, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };

        let outcome = decide(&before, &delivery, &observation, policy, 10);
        let AdmissionOutcome::Admitted { result, .. } = outcome else {
            panic!("fixture admission should succeed");
        };
        let commitment =
            transition_commitment(&before, &after, TransitionKind::Admission).unwrap();

        assert_eq!(
            verify_admission_transition(
                &before,
                &after,
                &delivery,
                &observation,
                policy,
                10,
                &result,
                &commitment,
            ),
            Ok(())
        );
    }

    #[test]
    fn admission_verification_rejects_mutated_after_state() {
        use crate::semantic_admission::{decide, AdmissionOutcome, AdmissionPolicy};

        let (before, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let AdmissionOutcome::Admitted { result, .. } =
            decide(&before, &delivery, &observation, policy, 10)
        else {
            panic!("fixture admission should succeed");
        };
        let commitment =
            transition_commitment(&before, &after, TransitionKind::Admission).unwrap();

        let mut mutated = after.clone();
        mutated
            .observations
            .get_mut(&observation.key)
            .unwrap()
            .payload = b"tampered".to_vec();

        assert_eq!(
            verify_admission_transition(
                &before,
                &mutated,
                &delivery,
                &observation,
                policy,
                10,
                &result,
                &commitment,
            ),
            Err(TransitionVerificationError::AfterStateMismatch)
        );
    }

    #[test]
    fn admission_verification_rejects_wrong_input_even_when_after_is_valid() {
        use crate::semantic_admission::{decide, AdmissionOutcome, AdmissionPolicy};

        let (before, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let mut wrong_observation = after.observations.values().next().unwrap().clone();
        wrong_observation.payload = b"wrong".to_vec();
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let AdmissionOutcome::Admitted { result, .. } =
            decide(&before, &delivery, &after.observations.values().next().unwrap().clone(), policy, 10)
        else {
            panic!("fixture admission should succeed");
        };
        let commitment =
            transition_commitment(&before, &after, TransitionKind::Admission).unwrap();

        assert_eq!(
            verify_admission_transition(
                &before,
                &after,
                &delivery,
                &wrong_observation,
                policy,
                10,
                &result,
                &commitment,
            ),
            Err(TransitionVerificationError::AdmissionNotAdmitted)
        );
    }

    #[test]
    fn valid_replay_transition_requires_unchanged_state() {
        use crate::semantic_admission::{decide, AdmissionOutcome, AdmissionPolicy};

        let (before, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let AdmissionOutcome::Replay { existing_result } =
            decide(&after, &delivery, &observation, policy, 11)
        else {
            panic!("fixture should replay");
        };
        let commitment =
            transition_commitment(&after, &after, TransitionKind::Replay).unwrap();

        assert_eq!(
            verify_replay_transition(
                &after,
                &after,
                &delivery,
                &observation,
                policy,
                11,
                &existing_result,
                &commitment,
            ),
            Ok(())
        );
    }

    #[test]
    fn replay_verification_rejects_a_state_change() {
        use crate::semantic_admission::{decide, AdmissionOutcome, AdmissionPolicy};

        let (before, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let AdmissionOutcome::Replay { existing_result } =
            decide(&after, &delivery, &observation, policy, 11)
        else {
            panic!("fixture should replay");
        };
        let commitment =
            transition_commitment(&after, &after, TransitionKind::Replay).unwrap();

        assert_eq!(
            verify_replay_transition(
                &after,
                &before,
                &delivery,
                &observation,
                policy,
                11,
                &existing_result,
                &commitment,
            ),
            Err(TransitionVerificationError::AfterStateMismatch)
        );
    }

    #[test]
    fn valid_lifecycle_retirement_is_replayably_verifiable() {
        use crate::semantic_admission::{retire_expired, AdmissionPolicy};

        let (mut before, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        before
            .observations
            .get_mut(&observation.key)
            .unwrap()
            .observed_at_ms = 0;

        let policy = AdmissionPolicy {
            retention_ms: 10,
            tombstone_retention_ms: 100,
            ..AdmissionPolicy::default()
        };
        let after = retire_expired(&before, policy, 11).unwrap();
        let commitment =
            transition_commitment(&before, &after, TransitionKind::LifecycleRetirement).unwrap();

        assert_eq!(
            verify_lifecycle_retirement(&before, &after, policy, 11, &commitment),
            Ok(())
        );
        assert!(!after.observations.contains_key(&observation.key));
        assert!(after.deliveries.contains_key(&delivery.logical_delivery_id));
    }

    #[test]
    fn lifecycle_verification_rejects_noop_retirement() {
        let (before, after) = fixture();
        let commitment =
            transition_commitment(&before, &before, TransitionKind::LifecycleRetirement).unwrap();

        assert_eq!(
            verify_lifecycle_retirement(
                &before,
                &after,
                AdmissionPolicy::default(),
                10,
                &commitment,
            ),
            Err(TransitionVerificationError::NoStateChange)
        );
    }

    #[test]
    fn verification_rejects_invalid_before_state_before_hashing() {
        let (before, after) = fixture();
        let mut invalid = before.clone();
        let delivery_id = *after.deliveries.keys().next().unwrap();
        invalid.deliveries.insert(
            delivery_id,
            crate::semantic_admission::DeliveryContract {
                logical_delivery_id: Uuid::from_u128(999),
                schema_version: 1,
                expires_at_ms: 1_000,
                payload: b"bad".to_vec(),
            },
        );
        let commitment =
            transition_commitment(&before, &after, TransitionKind::Admission).unwrap();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        let result = crate::semantic_admission::SemanticResult {
            logical_delivery_id: delivery.logical_delivery_id,
            observation: observation.key.clone(),
        };

        assert!(matches!(
            verify_admission_transition(
                &invalid,
                &after,
                &delivery,
                &observation,
                AdmissionPolicy::default(),
                10,
                &result,
                &commitment,
            ),
            Err(TransitionVerificationError::InvalidBeforeState(_))
        ));
    }

    #[test]
    fn admission_claim_binds_inputs_beyond_state_edge() {
        let (before, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let result = crate::semantic_admission::SemanticResult {
            logical_delivery_id: delivery.logical_delivery_id,
            observation: observation.key.clone(),
        };

        let original = admission_claim_commitment(
            &before,
            &after,
            &delivery,
            &observation,
            policy,
            10,
            &result,
        )
        .unwrap();

        let mut changed_observation = observation.clone();
        changed_observation.payload = b"different".to_vec();
        let changed = admission_claim_commitment(
            &before,
            &after,
            &delivery,
            &changed_observation,
            policy,
            10,
            &result,
        )
        .unwrap();

        assert_ne!(original, changed);
    }

    #[test]
    fn admission_claim_binds_policy_and_logical_time() {
        let (before, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        let result = SemanticResult {
            logical_delivery_id: delivery.logical_delivery_id,
            observation: observation.key.clone(),
        };
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };

        let base =
            admission_claim_commitment(&before, &after, &delivery, &observation, policy, 10, &result)
                .unwrap();
        let later =
            admission_claim_commitment(&before, &after, &delivery, &observation, policy, 11, &result)
                .unwrap();
        let stricter = admission_claim_commitment(
            &before,
            &after,
            &delivery,
            &observation,
            AdmissionPolicy {
                max_observations: policy.max_observations - 1,
                ..policy
            },
            10,
            &result,
        )
        .unwrap();

        assert_ne!(base, later);
        assert_ne!(base, stricter);
    }

    #[test]
    fn admission_claim_verification_rejects_tampered_claim() {
        let (before, after) = fixture();
        let delivery = after.deliveries.values().next().unwrap().clone();
        let observation = after.observations.values().next().unwrap().clone();
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let result = match decide(&before, &delivery, &observation, policy, 10) {
            AdmissionOutcome::Admitted { result, .. } => result,
            other => panic!("fixture admission failed: {other:?}"),
        };
        let transition =
            transition_commitment(&before, &after, TransitionKind::Admission).unwrap();
        let claim =
            admission_claim_commitment(&before, &after, &delivery, &observation, policy, 10, &result)
                .unwrap();
        let mut tampered = claim.as_bytes().to_owned();
        tampered[0] ^= 1;
        let tampered = TransitionClaimCommitment(tampered);

        assert_eq!(
            verify_admission_transition_claim(
                &before,
                &after,
                &delivery,
                &observation,
                policy,
                10,
                &result,
                &transition,
                &tampered,
            ),
            Err(TransitionVerificationError::ClaimCommitmentMismatch)
        );
    }

    #[test]
    fn lifecycle_claim_binds_policy_and_time() {
        let (before, _) = fixture();
        let policy = AdmissionPolicy::default();
        let same = lifecycle_claim_commitment(&before, &before, policy, 10).unwrap();
        let later = lifecycle_claim_commitment(&before, &before, policy, 11).unwrap();
        assert_ne!(same, later);
    }

}
