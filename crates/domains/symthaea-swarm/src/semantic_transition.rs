// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure transition commitments built from semantic state commitments.
//!
//! A state digest answers "what state is this?"; a transition commitment answers
//! "which before/after state pair and semantic operation are being claimed?".
//! It is not a signature and does not establish who performed the transition.

use crate::semantic_admission::SemanticAdmissionState;
use crate::semantic_digest::{semantic_digest, SemanticDigest, SemanticDigestError};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_admission::{
        decide, AdmissionOutcome, AdmissionPolicy, DeliveryContract, ObservationKey,
        ObservationRecord,
    };
    use uuid::Uuid;

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
        // no signer identity or secret material. Authentication belongs to a
        // separate signature/attestation layer.
        let (before, after) = fixture();
        let commitment = transition_commitment(&before, &after, TransitionKind::Admission).unwrap();
        assert_eq!(commitment.as_bytes().len(), 32);
    }
}
