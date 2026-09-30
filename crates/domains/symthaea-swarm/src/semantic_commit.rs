//! Optimistic atomic commit boundary for semantic admission.
//!
//! The transition oracle decides without mutation. This module adds the
//! publication boundary: load a versioned semantic state, decide, then publish
//! the candidate only if the version is unchanged. A failed CAS causes a reload
//! and a fresh decision rather than overwriting a concurrent admission.

use crate::semantic_admission::{
    decide, AdmissionOutcome, AdmissionPolicy, DeliveryContract, ObservationRecord,
    SemanticAdmissionState, SemanticResult,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionedState {
    pub version: u64,
    pub state: SemanticAdmissionState,
}

pub trait SemanticCommitStore {
    type Error;
    fn load(&mut self) -> Result<VersionedState, Self::Error>;
    fn compare_and_swap(
        &mut self,
        expected_version: u64,
        next: SemanticAdmissionState,
    ) -> Result<bool, Self::Error>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtomicAdmissionOutcome<E> {
    Admitted { result: SemanticResult, committed_version: u64 },
    Replay { existing_result: SemanticResult, observed_version: u64 },
    Rejected { outcome: AdmissionOutcome, observed_version: u64 },
    Conflict { outcome: AdmissionOutcome, observed_version: u64 },
    Expired { outcome: AdmissionOutcome, observed_version: u64 },
    CapacityExceeded { outcome: AdmissionOutcome, observed_version: u64 },
    StorageError(E),
}

fn classify_non_admit<E>(outcome: AdmissionOutcome, version: u64) -> AtomicAdmissionOutcome<E> {
    match outcome {
        AdmissionOutcome::Replay { existing_result } =>
            AtomicAdmissionOutcome::Replay { existing_result, observed_version: version },
        AdmissionOutcome::Rejected { .. } =>
            AtomicAdmissionOutcome::Rejected { outcome, observed_version: version },
        AdmissionOutcome::Conflict { .. } =>
            AtomicAdmissionOutcome::Conflict { outcome, observed_version: version },
        AdmissionOutcome::Expired { .. } =>
            AtomicAdmissionOutcome::Expired { outcome, observed_version: version },
        AdmissionOutcome::CapacityExceeded { .. } =>
            AtomicAdmissionOutcome::CapacityExceeded { outcome, observed_version: version },
        AdmissionOutcome::Admitted { .. } => unreachable!("admission handled by caller"),
    }
}

/// Linearizable optimistic admission.
///
/// Every CAS collision reloads the complete state and re-runs the pure oracle,
/// so a loser cannot overwrite a concurrent admission or return a stale
/// decision.
pub fn admit_atomically<S: SemanticCommitStore>(
    store: &mut S,
    delivery: &DeliveryContract,
    observation: &ObservationRecord,
    policy: AdmissionPolicy,
    now_ms: u64,
    max_retries: usize,
) -> AtomicAdmissionOutcome<S::Error> {
    for _attempt in 0..=max_retries {
        let loaded = match store.load() {
            Ok(value) => value,
            Err(error) => return AtomicAdmissionOutcome::StorageError(error),
        };

        let outcome = decide(&loaded.state, delivery, observation, policy, now_ms);

        match outcome {
            AdmissionOutcome::Admitted { next_state, result } => {
                match store.compare_and_swap(loaded.version, next_state) {
                    Ok(true) => return AtomicAdmissionOutcome::Admitted {
                        result,
                        committed_version: loaded.version.saturating_add(1),
                    },
                    Ok(false) => continue,
                    Err(error) => return AtomicAdmissionOutcome::StorageError(error),
                }
            }
            other => return classify_non_admit(other, loaded.version),
        }
    }

    AtomicAdmissionOutcome::Conflict {
        outcome: AdmissionOutcome::Conflict {
            identity_kind: crate::semantic_admission::ConflictKind::DeliveryContract,
        },
        observed_version: u64::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_admission::{ConflictKind, ObservationKey};
    use std::collections::VecDeque;
    use uuid::Uuid;

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum StoreError { Injected }

    #[derive(Debug, Default)]
    struct TestStore {
        version: u64,
        state: SemanticAdmissionState,
        cas_results: VecDeque<Result<bool, StoreError>>,
        loads: usize,
    }

    impl SemanticCommitStore for TestStore {
        type Error = StoreError;

        fn load(&mut self) -> Result<VersionedState, Self::Error> {
            self.loads += 1;
            Ok(VersionedState { version: self.version, state: self.state.clone() })
        }

        fn compare_and_swap(
            &mut self,
            expected_version: u64,
            next: SemanticAdmissionState,
        ) -> Result<bool, Self::Error> {
            if let Some(result) = self.cas_results.pop_front() {
                match result {
                    Ok(true) => {
                        if expected_version != self.version { return Ok(false); }
                        self.state = next;
                        self.version += 1;
                        Ok(true)
                    }
                    Ok(false) => {
                        self.version += 1;
                        Ok(false)
                    }
                    Err(error) => Err(error),
                }
            } else if expected_version == self.version {
                self.state = next;
                self.version += 1;
                Ok(true)
            } else {
                Ok(false)
            }
        }
    }

    fn fixture() -> (DeliveryContract, ObservationRecord) {
        let delivery_id = Uuid::from_u128(1);
        let observation_id = Uuid::from_u128(2);
        (
            DeliveryContract {
                logical_delivery_id: delivery_id,
                schema_version: 1,
                expires_at_ms: 1_000,
                payload: b"alpha".to_vec(),
            },
            ObservationRecord {
                key: ObservationKey {
                    namespace: "source-a/stream-1".into(),
                    observation_id,
                },
                source_id: Uuid::from_u128(3),
                observed_at_ms: 900,
                payload: b"alpha".to_vec(),
            },
        )
    }

    #[test]
    fn successful_admission_publishes_exactly_once() {
        let (delivery, observation) = fixture();
        let mut store = TestStore::default();
        let result = admit_atomically(
            &mut store, &delivery, &observation, AdmissionPolicy::default(), 100, 2
        );
        assert!(matches!(
            result,
            AtomicAdmissionOutcome::Admitted { committed_version: 1, .. }
        ));
        assert_eq!(store.version, 1);
        assert_eq!(store.state.deliveries.len(), 1);
        assert_eq!(store.state.observations.len(), 1);
    }

    #[test]
    fn exact_retry_reads_replay_without_second_commit() {
        let (delivery, observation) = fixture();
        let mut store = TestStore::default();
        let first = admit_atomically(
            &mut store, &delivery, &observation, AdmissionPolicy::default(), 100, 2
        );
        assert!(matches!(first, AtomicAdmissionOutcome::Admitted { .. }));

        let second = admit_atomically(
            &mut store, &delivery, &observation, AdmissionPolicy::default(), 100, 2
        );
        assert!(matches!(
            second,
            AtomicAdmissionOutcome::Replay { observed_version: 1, .. }
        ));
        assert_eq!(store.version, 1);
    }

    #[test]
    fn cas_collision_reloads_and_then_admits_current_state() {
        let (delivery, observation) = fixture();
        let mut store = TestStore {
            cas_results: VecDeque::from([Ok(false), Ok(true)]),
            ..Default::default()
        };
        let result = admit_atomically(
            &mut store, &delivery, &observation, AdmissionPolicy::default(), 100, 2
        );
        assert!(matches!(
            result,
            AtomicAdmissionOutcome::Admitted { committed_version: 2, .. }
        ));
        assert_eq!(store.loads, 2);
        assert_eq!(store.version, 2);
    }

    #[test]
    fn commit_error_does_not_report_admission() {
        let (delivery, observation) = fixture();
        let mut store = TestStore {
            cas_results: VecDeque::from([Err(StoreError::Injected)]),
            ..Default::default()
        };
        let result = admit_atomically(
            &mut store, &delivery, &observation, AdmissionPolicy::default(), 100, 2
        );
        assert_eq!(result, AtomicAdmissionOutcome::StorageError(StoreError::Injected));
        assert!(store.state.deliveries.is_empty());
        assert!(store.state.observations.is_empty());
        assert_eq!(store.version, 0);
    }

    #[test]
    fn semantic_rejection_does_not_call_commit() {
        let (delivery, mut observation) = fixture();
        observation.key.observation_id = Uuid::nil();
        let mut store = TestStore::default();
        let result = admit_atomically(
            &mut store, &delivery, &observation, AdmissionPolicy::default(), 100, 2
        );
        assert!(matches!(
            result,
            AtomicAdmissionOutcome::Rejected { observed_version: 0, .. }
        ));
        assert_eq!(store.version, 0);
    }

    #[test]
    fn conflict_precedes_capacity() {
        let (delivery, observation) = fixture();
        let mut state = SemanticAdmissionState::default();
        let mut changed = delivery.clone();
        changed.payload = b"different".to_vec();
        state.deliveries.insert(delivery.logical_delivery_id, changed);

        let mut store = TestStore { state, ..Default::default() };
        let result = admit_atomically(
            &mut store,
            &delivery,
            &observation,
            AdmissionPolicy { max_deliveries: 0, ..AdmissionPolicy::default() },
            100,
            2,
        );
        assert!(matches!(
            result,
            AtomicAdmissionOutcome::Conflict {
                outcome: AdmissionOutcome::Conflict {
                    identity_kind: ConflictKind::DeliveryContract
                },
                ..
            }
        ));
        assert_eq!(store.version, 0);
    }
}
