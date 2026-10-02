// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic digest for replayable semantic transition evidence.
//!
//! EvidenceDigest identifies the complete canonical evidence envelope. It is
//! deliberately distinct from the semantic state digest, transition
//! commitment, and claim commitment so callers can archive or index one
//! self-contained evidence object without conflating those layers.

use crate::semantic_transition::{
    canonical_evidence_bytes, TransitionEvidence, TransitionEvidenceEncodingError,
};

pub const ALGORITHM: &str = "BLAKE3-256";
pub const VERSION: u16 = 1;
pub const DOMAIN: &[u8] = b"symthaea-swarm/semantic-transition-evidence-digest";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EvidenceDigest([u8; 32]);

impl EvidenceDigest {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum EvidenceDigestError {
    #[error("canonical evidence encoding failed: {0}")]
    Encoding(#[from] TransitionEvidenceEncodingError),
}

/// Hash the complete canonical evidence envelope with an independent,
/// versioned, domain-separated digest namespace.
pub fn evidence_digest(
    evidence: &TransitionEvidence,
) -> Result<EvidenceDigest, EvidenceDigestError> {
    let canonical = canonical_evidence_bytes(evidence)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(DOMAIN.len() as u64).to_be_bytes());
    hasher.update(DOMAIN);
    hasher.update(&VERSION.to_be_bytes());
    hasher.update(&canonical);
    Ok(EvidenceDigest(*hasher.finalize().as_bytes()))
}

impl TransitionEvidence {
    /// Compute the digest of this complete canonical evidence envelope.
    pub fn evidence_digest(&self) -> Result<EvidenceDigest, EvidenceDigestError> {
        evidence_digest(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_admission::{
        decide, AdmissionOutcome, AdmissionPolicy, DeliveryContract, ObservationKey,
        ObservationRecord, SemanticAdmissionState, SemanticResult,
    };
    use crate::semantic_transition::{
        build_transition_evidence, TransitionClaim, TransitionKind,
    };
    use uuid::Uuid;

    fn evidence() -> TransitionEvidence {
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
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let outcome = decide(&before, &delivery, &observation, policy, 10);
        let AdmissionOutcome::Admitted { next_state, result } = outcome else {
            panic!("fixture admission should succeed");
        };
        build_transition_evidence(
            &before,
            &next_state,
            TransitionClaim::Admission {
                delivery,
                observation,
                policy,
                now_ms: 10,
                result,
            },
        )
        .unwrap()
    }

    #[test]
    fn evidence_digest_is_deterministic() {
        let first = evidence_digest(&evidence()).unwrap();
        let second = evidence_digest(&evidence()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.as_bytes().len(), 32);
        assert_eq!(evidence().evidence_digest().unwrap(), first);
    }

    #[test]
    fn evidence_digest_changes_when_state_digest_changes() {
        let original = evidence();
        let mut changed = original.clone();
        changed.before_digest = changed.after_digest;
        assert_ne!(
            evidence_digest(&original).unwrap(),
            evidence_digest(&changed).unwrap()
        );
    }

    #[test]
    fn evidence_digest_changes_when_claim_changes() {
        let original = evidence();
        let mut changed = original.clone();
        let TransitionClaim::Admission {
            delivery,
            observation,
            policy,
            now_ms,
            result,
        } = changed.claim.clone()
        else {
            panic!("fixture must be admission evidence");
        };
        let mut changed_observation = observation;
        changed_observation.payload = b"changed".to_vec();
        changed.claim = TransitionClaim::Admission {
            delivery,
            observation: changed_observation,
            policy,
            now_ms,
            result,
        };

        assert_ne!(
            evidence_digest(&original).unwrap(),
            evidence_digest(&changed).unwrap()
        );
    }

    #[test]
    fn evidence_digest_changes_when_commitment_changes() {
        let original = evidence();
        let mut changed = original.clone();
        let mut bytes = *changed.claim_commitment.as_bytes();
        bytes[0] ^= 1;
        changed.claim_commitment =
            crate::semantic_transition::TransitionClaimCommitment(bytes);

        assert_ne!(
            evidence_digest(&original).unwrap(),
            evidence_digest(&changed).unwrap()
        );
    }

    #[test]
    fn evidence_digest_rejects_noncanonical_envelope() {
        let mut invalid = evidence();
        invalid.kind = TransitionKind::Replay;
        assert!(matches!(
            evidence_digest(&invalid),
            Err(EvidenceDigestError::Encoding(
                TransitionEvidenceEncodingError::KindMismatch
            ))
        ));
    }
}
