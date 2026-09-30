//! Pure semantic admission transition oracle.
//!
//! This module intentionally does not know about transport envelopes, clocks,
//! storage, hashing, or network state. It models the semantic identity boundary
//! described by TEMPORAL-STATE-008 so production mutation can be wired to an
//! explicit commit primitive later.

use std::collections::HashMap;
use uuid::Uuid;

/// Identity scope for an observation. IDs are only unique within this scope.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObservationKey {
    pub namespace: String,
    pub observation_id: Uuid,
}

/// Immutable semantic delivery contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveryContract {
    pub logical_delivery_id: Uuid,
    pub schema_version: u16,
    pub expires_at_ms: u64,
    pub payload: Vec<u8>,
}

/// Immutable source observation record.
///
/// Attempt and transport-envelope identifiers deliberately do not belong here:
/// retries may change transport identity without creating a new observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationRecord {
    pub key: ObservationKey,
    pub source_id: Uuid,
    pub observed_at_ms: u64,
    pub payload: Vec<u8>,
}

/// Result produced by a semantic admission. It is retained verbatim so replay
/// returns the original result rather than recomputing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticResult {
    pub logical_delivery_id: Uuid,
    pub observation: ObservationKey,
}

/// State owned by the semantic admission layer.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SemanticAdmissionState {
    pub deliveries: HashMap<Uuid, DeliveryContract>,
    pub observations: HashMap<ObservationKey, ObservationRecord>,
    pub results: HashMap<ObservationKey, SemanticResult>,
}

/// Explicit policy inputs. All time is supplied by the caller; the oracle never
/// reads a wall clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionPolicy {
    /// Whether an already-known delivery may introduce a distinct observation.
    pub allow_new_observation: bool,
    pub max_deliveries: usize,
    pub max_observations: usize,
    /// Logical retention horizon for deduplication records.
    pub retention_ms: u64,
}

impl Default for AdmissionPolicy {
    fn default() -> Self {
        Self {
            allow_new_observation: false,
            max_deliveries: 16_384,
            max_observations: 16_384,
            retention_ms: 24 * 60 * 60 * 1_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictKind {
    DeliveryContract,
    ObservationRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpiryBoundary {
    DeliveryDeadline,
    Retention,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    InvalidDeliveryId,
    InvalidObservationId,
    InvalidObservationNamespace,
    ObservationTimestampInFuture,
    InvalidState(StateInvariant),
    Expired(ExpiryBoundary),
    NewObservationForbidden,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateInvariant {
    DeliveryMissingResult { logical_delivery_id: Uuid },
    ResultMissingDelivery { logical_delivery_id: Uuid },
    ResultMissingObservation { logical_delivery_id: Uuid },
    ResultObservationMismatch { logical_delivery_id: Uuid },
    ObservationKeyMismatch { observation_id: Uuid },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionOutcome {
    Admitted {
        next_state: SemanticAdmissionState,
        result: SemanticResult,
    },
    Replay {
        existing_result: SemanticResult,
    },
    Rejected {
        reason: RejectReason,
    },
    Conflict {
        identity_kind: ConflictKind,
    },
    Expired {
        boundary: ExpiryBoundary,
    },
    CapacityExceeded {
        bound: CapacityBound,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityBound {
    Deliveries,
    Observations,
}

/// Validate the semantic state's internal referential invariants.
///
/// Every admitted delivery has at least one retained result, and every result points
/// to an existing delivery and observation. Observations may remain indexed
/// independently so retention/GC can be implemented without changing
/// delivery identity semantics.
pub fn validate_state(state: &SemanticAdmissionState) -> Result<(), StateInvariant> {
    for logical_delivery_id in state.deliveries.keys() {
        let has_result = state
            .results
            .values()
            .any(|result| result.logical_delivery_id == *logical_delivery_id);
        if !has_result {
            return Err(StateInvariant::DeliveryMissingResult {
                logical_delivery_id: *logical_delivery_id,
            });
        }
    }

    for (observation_key, observation) in &state.observations {
        if observation.key != *observation_key {
            return Err(StateInvariant::ObservationKeyMismatch {
                observation_id: observation.key.observation_id,
            });
        }
    }

    for (observation_key, result) in &state.results {
        if !state.deliveries.contains_key(&result.logical_delivery_id) {
            return Err(StateInvariant::ResultMissingDelivery {
                logical_delivery_id: result.logical_delivery_id,
            });
        }
        if !state.observations.contains_key(observation_key) {
            return Err(StateInvariant::ResultMissingObservation {
                logical_delivery_id: result.logical_delivery_id,
            });
        }
        if result.observation != *observation_key {
            return Err(StateInvariant::ResultObservationMismatch {
                logical_delivery_id: result.logical_delivery_id,
            });
        }
    }

    Ok(())
}

/// Pure transition oracle.
///
/// Precedence is deliberately explicit:
/// validation -> delivery identity/contract -> observation identity/record ->
/// expiry/retention -> candidate successor. No input is mutated.
pub fn decide(
    state: &SemanticAdmissionState,
    delivery: &DeliveryContract,
    observation: &ObservationRecord,
    policy: AdmissionPolicy,
    now_ms: u64,
) -> AdmissionOutcome {
    if let Err(invariant) = validate_state(state) {
        return AdmissionOutcome::Rejected {
            reason: RejectReason::InvalidState(invariant),
        };
    }

    if delivery.logical_delivery_id.is_nil() {
        return AdmissionOutcome::Rejected {
            reason: RejectReason::InvalidDeliveryId,
        };
    }
    if observation.key.observation_id.is_nil() {
        return AdmissionOutcome::Rejected {
            reason: RejectReason::InvalidObservationId,
        };
    }
    if observation.key.namespace.is_empty() {
        return AdmissionOutcome::Rejected {
            reason: RejectReason::InvalidObservationNamespace,
        };
    }

    if let Some(existing) = state.deliveries.get(&delivery.logical_delivery_id) {
        if existing != delivery {
            return AdmissionOutcome::Conflict {
                identity_kind: ConflictKind::DeliveryContract,
            };
        }

        if let Some(result) = state.results.get(&observation.key) {
            if result.logical_delivery_id != delivery.logical_delivery_id {
                return AdmissionOutcome::Conflict {
                    identity_kind: ConflictKind::ObservationRecord,
                };
            }
            if state.observations.get(&observation.key) == Some(observation) {
                return AdmissionOutcome::Replay {
                    existing_result: result.clone(),
                };
            }
        } else if !policy.allow_new_observation {
            return AdmissionOutcome::Rejected {
                reason: RejectReason::NewObservationForbidden,
            };
        }
    }

    if now_ms >= delivery.expires_at_ms {
        return AdmissionOutcome::Expired {
            boundary: ExpiryBoundary::DeliveryDeadline,
        };
    }

    if let Some(existing) = state.observations.get(&observation.key) {
        if existing != observation {
            return AdmissionOutcome::Conflict {
                identity_kind: ConflictKind::ObservationRecord,
            };
        }
    }

    if let Some(existing) = state.results.get(&observation.key) {
        if existing.logical_delivery_id == delivery.logical_delivery_id {
            return AdmissionOutcome::Replay {
                existing_result: existing.clone(),
            };
        }
        return AdmissionOutcome::Conflict {
            identity_kind: ConflictKind::ObservationRecord,
        };
    }

    if observation.observed_at_ms > now_ms {
        return AdmissionOutcome::Rejected {
            reason: RejectReason::ObservationTimestampInFuture,
        };
    }

    if now_ms.saturating_sub(observation.observed_at_ms) > policy.retention_ms {
        return AdmissionOutcome::Expired {
            boundary: ExpiryBoundary::Retention,
        };
    }

    let is_new_delivery = !state.deliveries.contains_key(&delivery.logical_delivery_id);
    let is_new_observation = !state.observations.contains_key(&observation.key);

    if is_new_delivery && state.deliveries.len() >= policy.max_deliveries {
        return AdmissionOutcome::CapacityExceeded {
            bound: CapacityBound::Deliveries,
        };
    }
    if is_new_observation && state.observations.len() >= policy.max_observations {
        return AdmissionOutcome::CapacityExceeded {
            bound: CapacityBound::Observations,
        };
    }

    let mut next = state.clone();
    next.deliveries
        .entry(delivery.logical_delivery_id)
        .or_insert_with(|| delivery.clone());
    next.observations
        .entry(observation.key.clone())
        .or_insert_with(|| observation.clone());

    let result = SemanticResult {
        logical_delivery_id: delivery.logical_delivery_id,
        observation: observation.key.clone(),
    };
    next.results
        .insert(observation.key.clone(), result.clone());

    AdmissionOutcome::Admitted {
        next_state: next,
        result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: u128) -> (Uuid, Uuid) {
        (Uuid::from_u128(n), Uuid::from_u128(n + 1_000))
    }

    fn fixture() -> (DeliveryContract, ObservationRecord) {
        let (delivery_id, observation_id) = ids(1);
        let delivery = DeliveryContract {
            logical_delivery_id: delivery_id,
            schema_version: 1,
            expires_at_ms: 1_000,
            payload: b"alpha".to_vec(),
        };
        let observation = ObservationRecord {
            key: ObservationKey {
                namespace: "source-a/stream-1".into(),
                observation_id,
            },
            source_id: Uuid::from_u128(77),
            observed_at_ms: 900,
            payload: b"alpha".to_vec(),
        };
        (delivery, observation)
    }

    fn admitted_state() -> (SemanticAdmissionState, DeliveryContract, ObservationRecord) {
        let (delivery, observation) = fixture();
        let outcome = decide(
            &SemanticAdmissionState::default(),
            &delivery,
            &observation,
            AdmissionPolicy::default(),
            100,
        );
        let AdmissionOutcome::Admitted { next_state, .. } = outcome else {
            panic!("fixture must admit");
        };
        (next_state, delivery, observation)
    }

    #[test]
    fn valid_admitted_state_satisfies_invariants() {
        let (state, _, _) = admitted_state();
        assert_eq!(validate_state(&state), Ok(()));
    }

    #[test]
    fn delivery_without_result_is_rejected_without_mutation() {
        let (delivery, observation) = fixture();
        let mut state = SemanticAdmissionState::default();
        state.deliveries.insert(delivery.logical_delivery_id, delivery.clone());
        let before = state.clone();
        assert_eq!(
            decide(&state, &delivery, &observation, AdmissionPolicy::default(), 100),
            AdmissionOutcome::Rejected {
                reason: RejectReason::InvalidState(
                    StateInvariant::DeliveryMissingResult {
                        logical_delivery_id: delivery.logical_delivery_id
                    }
                )
            }
        );
        assert_eq!(state, before);
    }

    #[test]
    fn second_observation_for_existing_delivery_is_retained_independently() {
        let (state, delivery, mut observation) = admitted_state();
        observation.key.observation_id = Uuid::from_u128(9_999);
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let AdmissionOutcome::Admitted { next_state, .. } =
            decide(&state, &delivery, &observation, policy, 100)
        else {
            panic!("second observation should admit when policy allows it");
        };
        assert_eq!(next_state.deliveries.len(), 1);
        assert_eq!(next_state.observations.len(), 2);
        assert_eq!(next_state.results.len(), 2);
        assert_eq!(validate_state(&next_state), Ok(()));
    }

    #[test]
    fn observation_record_key_mismatch_is_rejected() {
        let (delivery, observation) = fixture();
        let mut state = SemanticAdmissionState::default();
        state.deliveries.insert(delivery.logical_delivery_id, delivery.clone());
        let mut stored = observation.clone();
        stored.key.observation_id = Uuid::from_u128(55);
        state.observations.insert(observation.key.clone(), stored);
        state.results.insert(
            observation.key.clone(),
            SemanticResult {
                logical_delivery_id: delivery.logical_delivery_id,
                observation: observation.key.clone(),
            },
        );
        assert_eq!(
            validate_state(&state),
            Err(StateInvariant::ObservationKeyMismatch { observation_id: 55 })
        );
    }

    #[test]
    fn same_observation_key_cannot_be_owned_by_another_delivery() {
        let (state, delivery, observation) = admitted_state();
        let mut other_delivery = delivery.clone();
        other_delivery.logical_delivery_id = Uuid::from_u128(44);
        other_delivery.payload = b"other".to_vec();
        let outcome = decide(
            &state,
            &other_delivery,
            &observation,
            AdmissionPolicy::default(),
            100,
        );
        assert_eq!(
            outcome,
            AdmissionOutcome::Conflict {
                identity_kind: ConflictKind::DeliveryContract
            }
        );
    }

    #[test]
    fn result_pointing_to_missing_observation_is_rejected() {
        let (delivery, observation) = fixture();
        let mut state = SemanticAdmissionState::default();
        state.deliveries.insert(delivery.logical_delivery_id, delivery.clone());
        state.results.insert(
            observation.key.clone(),
            SemanticResult {
                logical_delivery_id: delivery.logical_delivery_id,
                observation: observation.key.clone(),
            },
        );
        assert_eq!(
            validate_state(&state),
            Err(StateInvariant::ResultMissingObservation {
                logical_delivery_id: delivery.logical_delivery_id
            })
        );
    }

    #[test]
    fn fresh_delivery_and_observation_admit_once() {
        let (delivery, observation) = fixture();
        assert!(matches!(
            decide(
                &SemanticAdmissionState::default(),
                &delivery,
                &observation,
                AdmissionPolicy::default(),
                100
            ),
            AdmissionOutcome::Admitted { .. }
        ));
    }

    #[test]
    fn exact_retry_is_replay_even_when_transport_identity_changes() {
        let (state, delivery, observation) = admitted_state();
        assert!(matches!(
            decide(&state, &delivery, &observation, AdmissionPolicy::default(), 100),
            AdmissionOutcome::Replay { .. }
        ));
    }

    #[test]
    fn second_observation_replays_its_own_original_result() {
        let (state, delivery, mut observation) = admitted_state();
        observation.key.observation_id = Uuid::from_u128(9_999);
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let AdmissionOutcome::Admitted {
            next_state,
            result: admitted_result,
        } = decide(&state, &delivery, &observation, policy, 100)
        else {
            panic!("second observation should admit");
        };

        assert_eq!(
            decide(&next_state, &delivery, &observation, policy, 100),
            AdmissionOutcome::Replay {
                existing_result: admitted_result,
            }
        );
    }

    #[test]
    fn changed_delivery_contract_is_conflict() {
        let (state, delivery, observation) = admitted_state();
        let mut changed = delivery.clone();
        changed.payload = b"changed".to_vec();
        assert_eq!(
            decide(&state, &changed, &observation, AdmissionPolicy::default(), 100),
            AdmissionOutcome::Conflict {
                identity_kind: ConflictKind::DeliveryContract
            }
        );
    }

    #[test]
    fn same_observation_id_with_changed_record_is_conflict() {
        let (state, delivery, observation) = admitted_state();
        let mut changed = observation.clone();
        changed.payload = b"changed".to_vec();
        assert_eq!(
            decide(&state, &delivery, &changed, AdmissionPolicy::default(), 100),
            AdmissionOutcome::Conflict {
                identity_kind: ConflictKind::ObservationRecord
            }
        );
    }

    #[test]
    fn identical_payload_with_distinct_observation_ids_is_distinct_when_allowed() {
        let (state, delivery, mut observation) = admitted_state();
        observation.key.observation_id = Uuid::from_u128(9999);
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        assert!(matches!(
            decide(&state, &delivery, &observation, policy, 100),
            AdmissionOutcome::Admitted { .. }
        ));
    }

    #[test]
    fn new_observation_is_rejected_when_policy_forbids_it() {
        let (state, delivery, mut observation) = admitted_state();
        observation.key.observation_id = Uuid::from_u128(9999);
        assert_eq!(
            decide(&state, &delivery, &observation, AdmissionPolicy::default(), 100),
            AdmissionOutcome::Rejected {
                reason: RejectReason::NewObservationForbidden
            }
        );
    }

    #[test]
    fn rejection_does_not_mutate_input_state() {
        let (state, delivery, mut observation) = admitted_state();
        observation.key.observation_id = Uuid::from_u128(9999);
        let before = state.clone();
        let _ = decide(&state, &delivery, &observation, AdmissionPolicy::default(), 100);
        assert_eq!(state, before);
    }

    #[test]
    fn exact_delivery_expiry_is_expired() {
        let (delivery, observation) = fixture();
        assert_eq!(
            decide(
                &SemanticAdmissionState::default(),
                &delivery,
                &observation,
                AdmissionPolicy::default(),
                delivery.expires_at_ms
            ),
            AdmissionOutcome::Expired {
                boundary: ExpiryBoundary::DeliveryDeadline
            }
        );
    }

    #[test]
    fn retention_expiry_is_distinct() {
        let (mut delivery, mut observation) = fixture();
        delivery.expires_at_ms = 10_000;
        observation.observed_at_ms = 0;
        assert_eq!(
            decide(
                &SemanticAdmissionState::default(),
                &delivery,
                &observation,
                AdmissionPolicy {
                    retention_ms: 10,
                    ..AdmissionPolicy::default()
                },
                100
            ),
            AdmissionOutcome::Expired {
                boundary: ExpiryBoundary::Retention
            }
        );
    }

    #[test]
    fn future_observation_timestamp_is_rejected() {
        let (delivery, mut observation) = fixture();
        observation.observed_at_ms = 101;
        assert_eq!(
            decide(
                &SemanticAdmissionState::default(),
                &delivery,
                &observation,
                AdmissionPolicy::default(),
                100,
            ),
            AdmissionOutcome::Rejected {
                reason: RejectReason::ObservationTimestampInFuture
            }
        );
    }

    #[test]
    fn capacity_failure_carries_no_candidate_state() {
        let (delivery, observation) = fixture();
        assert_eq!(
            decide(
                &SemanticAdmissionState::default(),
                &delivery,
                &observation,
                AdmissionPolicy {
                    max_deliveries: 0,
                    ..AdmissionPolicy::default()
                },
                100
            ),
            AdmissionOutcome::CapacityExceeded {
                bound: CapacityBound::Deliveries
            }
        );
    }

    #[test]
    fn malformed_input_stutters_semantic_state() {
        let (state, delivery, mut observation) = admitted_state();
        observation.key.observation_id = Uuid::nil();
        let before = state.clone();
        let outcome = decide(
            &state,
            &delivery,
            &observation,
            AdmissionPolicy::default(),
            100,
        );
        assert_eq!(state, before);
        assert_eq!(
            outcome,
            AdmissionOutcome::Rejected {
                reason: RejectReason::InvalidObservationId
            }
        );
    }

    #[test]
    fn decision_does_not_depend_on_wall_clock() {
        let (delivery, observation) = fixture();
        let a = decide(
            &SemanticAdmissionState::default(),
            &delivery,
            &observation,
            AdmissionPolicy::default(),
            100,
        );
        let b = decide(
            &SemanticAdmissionState::default(),
            &delivery,
            &observation,
            AdmissionPolicy::default(),
            100,
        );
        assert_eq!(a, b);
    }
}
