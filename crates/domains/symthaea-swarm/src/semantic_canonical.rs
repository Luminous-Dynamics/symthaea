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
    put_u64(&mut out, u64::try_from(deliveries.len()).map_err(|_| CanonicalStateError::FieldTooLarge)?);
    for (id, delivery) in deliveries {
        put_uuid(&mut out, id);
        put_u16(&mut out, delivery.schema_version);
        put_u64(&mut out, delivery.expires_at_ms);
        put_bytes(&mut out, &delivery.payload)?;
    }

    let observations = sorted_observations(state);
    put_u64(&mut out, u64::try_from(observations.len()).map_err(|_| CanonicalStateError::FieldTooLarge)?);
    for (key, observation) in observations {
        put_string(&mut out, &key.namespace)?;
        put_uuid(&mut out, key.observation_id);
        put_uuid(&mut out, observation.source_id);
        put_u64(&mut out, observation.observed_at_ms);
        put_bytes(&mut out, &observation.payload)?;
    }

    let results = sorted_results(state);
    put_u64(&mut out, u64::try_from(results.len()).map_err(|_| CanonicalStateError::FieldTooLarge)?);
    for (key, result) in results {
        put_string(&mut out, &key.namespace)?;
        put_uuid(&mut out, key.observation_id);
        put_uuid(&mut out, result.logical_delivery_id);
        put_string(&mut out, &result.observation.namespace)?;
        put_uuid(&mut out, result.observation.observation_id);
    }

    let delivery_tombstones = sorted_delivery_tombstones(state);
    put_u64(&mut out, u64::try_from(delivery_tombstones.len()).map_err(|_| CanonicalStateError::FieldTooLarge)?);
    for (id, tombstone) in delivery_tombstones {
        put_uuid(&mut out, id);
        put_tombstone(&mut out, tombstone);
    }

    let observation_tombstones = sorted_observation_tombstones(state);
    put_u64(&mut out, u64::try_from(observation_tombstones.len()).map_err(|_| CanonicalStateError::FieldTooLarge)?);
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
    fn canonical_witness_is_exact_alias_for_canonical_state_bytes() {
        let state = admitted_state();
        assert_eq!(
            canonical_witness(&state).unwrap(),
            canonical_state_bytes(&state).unwrap()
        );
    }

    #[test]
    fn every_delivery_field_is_witnessed() {
        let state = admitted_state();
        let base = canonical_state_bytes(&state).unwrap();

        let mut changed = state.clone();
        changed.deliveries.values_mut().next().unwrap().schema_version += 1;
        assert_ne!(base, canonical_state_bytes(&changed).unwrap());

        let mut changed = state.clone();
        changed.deliveries.values_mut().next().unwrap().expires_at_ms += 1;
        assert_ne!(base, canonical_state_bytes(&changed).unwrap());

        let mut changed = state.clone();
        changed.deliveries.values_mut().next().unwrap().payload.push(0);
        assert_ne!(base, canonical_state_bytes(&changed).unwrap());
    }

    #[test]
    fn every_observation_field_is_witnessed() {
        let state = admitted_state();
        let base = canonical_state_bytes(&state).unwrap();

        let mut changed = state.clone();
        changed.observations.values_mut().next().unwrap().source_id = Uuid::from_u128(99);
        assert_ne!(base, canonical_state_bytes(&changed).unwrap());

        let mut changed = state.clone();
        changed.observations.values_mut().next().unwrap().observed_at_ms += 1;
        assert_ne!(base, canonical_state_bytes(&changed).unwrap());

        let mut changed = state.clone();
        changed.observations.values_mut().next().unwrap().payload.push(0);
        assert_ne!(base, canonical_state_bytes(&changed).unwrap());
    }

    #[test]
    fn every_result_field_is_witnessed() {
        let state = admitted_state();
        let (delivery, observation) = fixture();
        let mut second_delivery = delivery.clone();
        second_delivery.logical_delivery_id = Uuid::from_u128(99);
        second_delivery.payload = b"second-delivery".to_vec();
        let mut second_observation = observation.clone();
        second_observation.key = ObservationKey {
            namespace: "other".into(),
            observation_id: Uuid::from_u128(98),
        };
        let AdmissionOutcome::Admitted { next_state, .. } = crate::semantic_admission::decide(
            &state,
            &second_delivery,
            &second_observation,
            AdmissionPolicy {
                allow_new_observation: true,
                ..AdmissionPolicy::default()
            },
            100,
        ) else {
            panic!("second admission should succeed");
        };
        let state = next_state;
        let base = canonical_state_bytes(&state).unwrap();
        let key = observation.key;

        let mut changed = state.clone();
        changed.results.get_mut(&key).unwrap().logical_delivery_id =
            second_delivery.logical_delivery_id;
        assert_ne!(base, canonical_state_bytes(&changed).unwrap());

        let mut changed = state.clone();
        let result = changed.results.remove(&key).unwrap();
        changed.results.insert(
            second_observation.key.clone(),
            SemanticResult {
                logical_delivery_id: result.logical_delivery_id,
                observation: second_observation.key.clone(),
            },
        );
        assert_ne!(base, canonical_state_bytes(&changed).unwrap());
    }

    #[test]
    fn tombstone_fields_and_keys_are_witnessed() {
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
        let base = canonical_state_bytes(&gc).unwrap();

        let mut changed = gc.clone();
        changed.observation_tombstones.get_mut(&observation.key).unwrap().retired_at_ms += 1;
        assert_ne!(base, canonical_state_bytes(&changed).unwrap());

        let mut changed = gc.clone();
        changed.observation_tombstones.get_mut(&observation.key).unwrap().reusable_at_ms += 1;
        assert_ne!(base, canonical_state_bytes(&changed).unwrap());

        let mut changed = gc.clone();
        let tombstone = changed.observation_tombstones.remove(&observation.key).unwrap();
        let different_key = ObservationKey {
            namespace: "other".into(),
            observation_id: observation.key.observation_id,
        };
        changed.observation_tombstones.insert(different_key, tombstone);
        assert_ne!(base, canonical_state_bytes(&changed).unwrap());
    }

    #[test]
    fn collection_membership_and_order_are_witnessed() {
        let state = admitted_state();
        let (delivery, observation) = fixture();
        let mut expanded = state.clone();
        let mut second = observation.clone();
        second.key.observation_id = Uuid::from_u128(22);
        second.payload = b"second".to_vec();
        expanded.observations.insert(second.key.clone(), second.clone());
        expanded.results.insert(
            second.key.clone(),
            SemanticResult {
                logical_delivery_id: delivery.logical_delivery_id,
                observation: second.key.clone(),
            },
        );
        let base = canonical_state_bytes(&expanded).unwrap();

        let mut reordered = SemanticAdmissionState::default();
        for (key, value) in expanded.observations.iter().rev() {
            reordered.observations.insert(key.clone(), value.clone());
        }
        for (key, value) in expanded.results.iter().rev() {
            reordered.results.insert(key.clone(), value.clone());
        }
        for (key, value) in expanded.deliveries.iter().rev() {
            reordered.deliveries.insert(*key, value.clone());
        }
        assert_eq!(base, canonical_state_bytes(&reordered).unwrap());

        reordered.results.remove(&second.key);
        assert_ne!(base, canonical_state_bytes(&reordered).unwrap());
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
