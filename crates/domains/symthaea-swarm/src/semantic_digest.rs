// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cryptographic commitment to the canonical semantic-admission witness.
//!
//! This module deliberately hashes the output of semantic_canonical rather
//! than introducing a second serializer. The canonical witness owns semantic
//! framing; this module owns only the cryptographic commitment.

use crate::semantic_admission::SemanticAdmissionState;
use crate::semantic_canonical::{canonical_witness, CanonicalStateError};

pub const ALGORITHM: &str = "BLAKE3-256";
pub const VERSION: u16 = 1;
pub const DOMAIN: &[u8] = b"symthaea-swarm/semantic-admission-digest";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SemanticDigest([u8; 32]);

impl SemanticDigest {
    pub const BYTES: usize = 32;
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum SemanticDigestError {
    #[error("canonical state witness failed: {0}")]
    Canonical(#[from] CanonicalStateError),
}

/// Compute the versioned cryptographic commitment for a semantic state.
///
/// The digest input is explicitly framed as:
/// length-prefixed DOMAIN || VERSION || canonical_witness(state).
pub fn semantic_digest(
    state: &SemanticAdmissionState,
) -> Result<SemanticDigest, SemanticDigestError> {
    let witness = canonical_witness(state)?;

    let mut hasher = blake3::Hasher::new();
    hasher.update(&(DOMAIN.len() as u64).to_be_bytes());
    hasher.update(DOMAIN);
    hasher.update(&VERSION.to_be_bytes());
    hasher.update(&witness);

    Ok(SemanticDigest(*hasher.finalize().as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_admission::{
        AdmissionPolicy, DeliveryContract, ObservationKey, ObservationRecord,
    };
    use uuid::Uuid;

    fn state() -> SemanticAdmissionState {
        let delivery_id = Uuid::from_u128(1);
        let observation_id = Uuid::from_u128(2);
        let delivery = DeliveryContract {
            logical_delivery_id: delivery_id,
            schema_version: 1,
            expires_at_ms: 1000,
            payload: b"payload".to_vec(),
        };
        let observation = ObservationRecord {
            key: ObservationKey {
                namespace: "test".into(),
                observation_id,
            },
            source_id: Uuid::from_u128(3),
            observed_at_ms: 10,
            payload: b"observation".to_vec(),
        };
        match crate::semantic_admission::decide(
            &SemanticAdmissionState::default(),
            &delivery,
            &observation,
            &AdmissionPolicy {
                allow_new_observation: true,
                ..AdmissionPolicy::default()
            },
            10,
        ) {
            crate::semantic_admission::AdmissionOutcome::Admitted { next_state, .. } => next_state,
            other => panic!("fixture admission failed: {other:?}"),
        }
    }

    #[test]
    fn digest_is_deterministic() {
        assert_eq!(
            semantic_digest(&state()).unwrap(),
            semantic_digest(&state()).unwrap()
        );
    }

    #[test]
    fn digest_changes_when_semantic_state_changes() {
        let original = state();
        let mut changed = original.clone();
        changed.deliveries.values_mut().next().unwrap().payload = b"changed".to_vec();
        assert_ne!(
            semantic_digest(&original).unwrap(),
            semantic_digest(&changed).unwrap()
        );
    }

    #[test]
    fn digest_matches_direct_hash_of_explicit_commitment_input() {
        let state = state();
        let witness = canonical_witness(&state).unwrap();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(DOMAIN.len() as u64).to_be_bytes());
        hasher.update(DOMAIN);
        hasher.update(&VERSION.to_be_bytes());
        hasher.update(&witness);
        assert_eq!(
            semantic_digest(&state).unwrap().as_bytes(),
            hasher.finalize().as_bytes()
        );
    }

    #[test]
    fn invalid_state_has_no_digest() {
        let mut state = state();
        let delivery_id = *state.deliveries.keys().next().unwrap();
        state.deliveries.get_mut(&delivery_id).unwrap().logical_delivery_id = Uuid::from_u128(99);
        assert!(matches!(
            semantic_digest(&state),
            Err(SemanticDigestError::Canonical(
                CanonicalStateError::InvalidState(_)
            ))
        ));
    }

    #[test]
    fn digest_is_insertion_order_independent() {
        let first = state();
        let delivery = first.deliveries.values().next().unwrap().clone();
        let observation = first.observations.values().next().unwrap().clone();
        let second = match crate::semantic_admission::decide(
            &SemanticAdmissionState::default(),
            &delivery,
            &observation,
            &AdmissionPolicy {
                allow_new_observation: true,
                ..AdmissionPolicy::default()
            },
            10,
        ) {
            crate::semantic_admission::AdmissionOutcome::Admitted { next_state, .. } => next_state,
            other => panic!("fixture admission failed: {other:?}"),
        };
        assert_eq!(
            semantic_digest(&first).unwrap(),
            semantic_digest(&second).unwrap()
        );
    }
}
