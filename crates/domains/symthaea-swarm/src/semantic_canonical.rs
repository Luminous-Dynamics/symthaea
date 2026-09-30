//! Deterministic canonical representation of semantic admission state.
//!
//! Canonicalization is deliberately separate from hashing. These bytes are a
//! complete witness of semantic state; a future digest layer may hash them
//! without changing their meaning.

use crate::semantic_admission::{
    validate_state, IdentityTombstone, ObservationKey, SemanticAdmissionState,
};
use std::collections::BTreeMap;
use thiserror::Error;
use uuid::Uuid;

const DOMAIN: &[u8] = b"symthaea-swarm/semantic-admission-state";
const VERSION: u16 = 1;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CanonicalStateError {
    #[error("semantic state violates invariant: {0:?}")]
    InvalidState(crate::semantic_admission::StateInvariant),
    #[error("canonical field is too large to encode")]
    FieldTooLarge,
}

/// Produce a unique, insertion-order-independent byte representation of valid
/// semantic state. Collections are explicitly sorted before encoding.
pub fn canonical_state_bytes(
    state: &SemanticAdmissionState,
) -> Result<Vec<u8>, CanonicalStateError> {
    validate_state(state).map_err(CanonicalStateError::InvalidState)?;

    let mut out = Vec::new();
    put_bytes(&mut out, DOMAIN)?;
    put_u16(&mut out, VERSION);

    let deliveries = sorted_deliveries(state);
    put_u64(&mut out, deliveries.len());
    for (id, delivery) in deliveries {
        put_uuid(&mut out, id);
        put_u16(&mut out, delivery.schema_version);
        put_u64(&mut out, delivery.expires_at_ms);
        put_bytes(&mut out, &delivery.payload)?;
    }

    let observations = sorted_observations(state);
    put_u64(&mut out, observations.len());
    for (key, observation) in observations {
        put_string(&mut out, &key.namespace)?;
        put_uuid(&mut out, key.observation_id);
        put_uuid(&mut out, observation.source_id);
        put_u64(&mut out, observation.observed_at_ms);
        put_bytes(&mut out, &observation.payload)?;
    }

    let results = sorted_results(state);
    put_u64(&mut out, results.len());
    for (key, result) in results {
        put_string(&mut out, &key.namespace)?;
        put_uuid(&mut out, key.observation_id);
        put_uuid(&mut out, result.logical_delivery_id);
        put_string(&mut out, &result.observation.namespace)?;
        put_uuid(&mut out, result.observation.observation_id);
    }

    let delivery_tombstones = sorted_delivery_tombstones(state);
    put_u64(&mut out, delivery_tombstones.len());
    for (id, tombstone) in delivery_tombstones {
        put_uuid(&mut out, id);
        put_tombstone(&mut out, tombstone);
    }

    let observation_tombstones = sorted_observation_tombstones(state);
    put_u64(&mut out, observation_tombstones.len());
    for (key, tombstone) in observation_tombstones {
        put_string(&mut out, &key.namespace)?;
        put_uuid(&mut out, key.observation_id);
        put_tombstone(&mut out, tombstone);
    }

    Ok(out)
}

/// Alias emphasizing that the canonical bytes are an auditable witness rather
/// than a cryptographic digest.
pub fn canonical_witness(
    state: &SemanticAdmissionState,
) -> Result<Vec<u8>, CanonicalStateError> {
    canonical_state_bytes(state)
}

fn sorted_deliveries(
    state: &SemanticAdmissionState,
) -> BTreeMap<Uuid, &crate::semantic_admission::DeliveryContract> {
    state.deliveries.iter().map(|(id, value)| (*id, value)).collect()
}

fn sorted_observations(
    state: &SemanticAdmissionState,
) -> BTreeMap<ObservationKey, &crate::semantic_admission::ObservationRecord> {
    state
        .observations
        .iter()
        .map(|(key, value)| (key.clone(), value))
        .collect()
}

fn sorted_results(
    state: &SemanticAdmissionState,
) -> BTreeMap<ObservationKey, &crate::semantic_admission::SemanticResult> {
    state
        .results
        .iter()
        .map(|(key, value)| (key.clone(), value))
        .collect()
}

fn sorted_delivery_tombstones(
    state: &SemanticAdmissionState,
) -> BTreeMap<Uuid, &IdentityTombstone> {
    state
        .delivery_tombstones
        .iter()
        .map(|(id, value)| (*id, value))
        .collect()
}

fn sorted_observation_tombstones(
    state: &SemanticAdmissionState,
) -> BTreeMap<ObservationKey, &IdentityTombstone> {
    state
        .observation_tombstones
        .iter()
        .map(|(key, value)| (key.clone(), value))
        .collect()
}

fn put_tombstone(out: &mut Vec<u8>, tombstone: &IdentityTombstone) {
    put_u64(out, tombstone.retired_at_ms);
    put_u64(out, tombstone.reusable_at_ms);
}

fn put_uuid(out: &mut Vec<u8>, value: Uuid) {
    out.extend_from_slice(value.as_bytes());
}

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) -> Result<(), CanonicalStateError> {
    let len = u64::try_from(value.len()).map_err(|_| CanonicalStateError::FieldTooLarge)?;
    put_u64(out, len);
    out.extend_from_slice(value);
    Ok(())
}

fn put_string(out: &mut Vec<u8>, value: &str) -> Result<(), CanonicalStateError> {
    put_bytes(out, value.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_admission::{
        retire_expired, AdmissionOutcome, AdmissionPolicy, DeliveryContract, ObservationRecord,
        SemanticResult,
    };

    fn fixture() -> (DeliveryContract, ObservationRecord) {
        (
            DeliveryContract {
                logical_delivery_id: Uuid::from_u128(1),
                schema_version: 7,
                expires_at_ms: 1_000,
                payload: b"delivery".to_vec(),
            },
            ObservationRecord {
                key: ObservationKey {
                    namespace: "test".into(),
                    observation_id: Uuid::from_u128(2),
                },
                source_id: Uuid::from_u128(3),
                observed_at_ms: 100,
                payload: b"observation".to_vec(),
            },
        )
    }

    fn admitted_state() -> SemanticAdmissionState {
        let (delivery, observation) = fixture();
        let AdmissionOutcome::Admitted { next_state, .. } = crate::semantic_admission::decide(
            &SemanticAdmissionState::default(),
            &delivery,
            &observation,
            AdmissionPolicy::default(),
            100,
        ) else {
            panic!("fixture admission should succeed");
        };
        next_state
    }

    #[test]
    fn canonicalization_is_insertion_order_independent() {
        let state = admitted_state();
        let (delivery, observation) = fixture();
        let mut reordered = SemanticAdmissionState::default();

        // Insert the same semantic records in a deliberately different order.
        reordered.observations.insert(observation.key.clone(), observation.clone());
        reordered.results.insert(
            observation.key.clone(),
            SemanticResult {
                logical_delivery_id: delivery.logical_delivery_id,
                observation: observation.key.clone(),
            },
        );
        reordered.deliveries.insert(delivery.logical_delivery_id, delivery);

        assert_eq!(
            canonical_state_bytes(&state).unwrap(),
            canonical_state_bytes(&reordered).unwrap()
        );
    }

    #[test]
    fn canonicalization_changes_when_semantic_state_changes() {
        let state = admitted_state();
        let key = ObservationKey {
            namespace: "test".into(),
            observation_id: Uuid::from_u128(2),
        };
        let mut changed = state.clone();
        changed.observations.get_mut(&key).unwrap().payload = b"changed".to_vec();

        assert_ne!(
            canonical_state_bytes(&state).unwrap(),
            canonical_state_bytes(&changed).unwrap()
        );
    }

    #[test]
    fn tombstone_lifecycle_is_part_of_canonical_state() {
        let state = admitted_state();
        let (_, observation) = fixture();
        let gc = retire_expired(
            &state,
            AdmissionPolicy {
                retention_ms: 10,
                tombstone_retention_ms: 100,
                ..AdmissionPolicy::default()
            },
            111,
        )
        .unwrap();

        assert_ne!(
            canonical_state_bytes(&state).unwrap(),
            canonical_state_bytes(&gc).unwrap()
        );
        assert!(gc.observation_tombstones.contains_key(&observation.key));
    }

    #[test]
    fn invalid_state_has_no_canonical_witness() {
        let mut invalid = SemanticAdmissionState::default();
        invalid.deliveries.insert(
            Uuid::from_u128(1),
            DeliveryContract {
                logical_delivery_id: Uuid::from_u128(1),
                schema_version: 1,
                expires_at_ms: 10,
                payload: Vec::new(),
            },
        );

        assert!(matches!(
            canonical_state_bytes(&invalid),
            Err(CanonicalStateError::InvalidState(_))
        ));
    }

    #[test]
    fn variable_length_framing_is_explicit() {
        let state = admitted_state();
        let mut changed = state.clone();
        changed.observations.values_mut().next().unwrap().payload = b"".to_vec();

        assert_ne!(
            canonical_state_bytes(&state).unwrap(),
            canonical_state_bytes(&changed).unwrap()
        );
    }
}
