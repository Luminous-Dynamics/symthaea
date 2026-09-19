// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::{
    perceptual_enrollment_coordinator::{
        seal_coordinator_state, DurableEnrollmentCoordinatorErrorV1,
        DurableEnrollmentCoordinatorStoreV1, DurablePerceptualEnrollmentCoordinatorStateV1,
        PERCEPTUAL_ENROLLMENT_COORDINATOR_STATE_VERSION,
    },
    perceptual_enrollment_lifecycle::FrozenPerceptualEnrollmentAllocationLedgerV1,
    perceptual_enrollment_witness::FrozenPerceptualEnrollmentWitnessBundleV1,
};

fn dummy_coordinator_state(marker: char) -> DurablePerceptualEnrollmentCoordinatorStateV1 {
    let marker_digest = marker.to_string().repeat(64);
    let ledger = FrozenPerceptualEnrollmentAllocationLedgerV1 {
        ledger_version: "test-ledger-v1".into(),
        enrollment_policy_sha256: marker_digest.clone(),
        token_generation_receipt_sha256: "b".repeat(64),
        participant_schedule_sha256: "c".repeat(64),
        allocations: Vec::new(),
        final_allocation_head_sha256: "0".repeat(64),
        ledger_sha256: "d".repeat(64),
    };
    let witness = FrozenPerceptualEnrollmentWitnessBundleV1 {
        bundle_version: "test-witness-bundle-v1".into(),
        witness_policy_sha256: "e".repeat(64),
        entries: Vec::new(),
        final_witness_head_sha256: "0".repeat(64),
        bundle_sha256: "f".repeat(64),
    };
    let mut state = DurablePerceptualEnrollmentCoordinatorStateV1 {
        state_version: PERCEPTUAL_ENROLLMENT_COORDINATOR_STATE_VERSION.into(),
        enrollment_policy_sha256: marker_digest,
        enrollment_witness_policy_sha256: "e".repeat(64),
        confirmed_enrollment_ledger: ledger,
        confirmed_witness_bundle: witness,
        coordinator_state_sha256: String::new(),
    };
    seal_coordinator_state(&mut state).unwrap();
    state
}

#[test]
fn coordinator_commitment_binds_nested_confirmed_evidence() {
    let left = dummy_coordinator_state('a');
    let mut right = left.clone();
    right.confirmed_witness_bundle.bundle_sha256 = "9".repeat(64);
    seal_coordinator_state(&mut right).unwrap();
    assert_ne!(left.coordinator_state_sha256, right.coordinator_state_sha256);
}

#[cfg(target_os = "linux")]
#[test]
fn durable_coordinator_store_cas_rejects_stale_predecessor() {
    let unique = format!(
        "symthaea-mel003-coordinator-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let root = std::env::temp_dir().join(unique);
    let store = DurableEnrollmentCoordinatorStoreV1::new(&root);
    let first = dummy_coordinator_state('a');
    let initialized = store.initialize(&first).unwrap();
    assert_eq!(initialized, first);

    let stale_head = first.coordinator_state_sha256.clone();
    let output = store
        .transition(&stale_head, |current| {
            let mut next = current.clone();
            next.enrollment_policy_sha256 = "8".repeat(64);
            seal_coordinator_state(&mut next)
                .map_err(|_| DurableEnrollmentCoordinatorErrorV1::Serialization)?;
            Ok((next, 7u8))
        })
        .unwrap();
    assert_eq!(output, 7);

    let current = store.inspect().unwrap();
    assert_ne!(current.coordinator_state_sha256, stale_head);
    let stale_attempt = store.transition::<(), _>(&stale_head, |_| {
        Err(DurableEnrollmentCoordinatorErrorV1::StateMalformed)
    });
    assert!(matches!(
        stale_attempt,
        Err(DurableEnrollmentCoordinatorErrorV1::ExpectedCoordinatorHeadMismatch { .. })
    ));

    let _ = std::fs::remove_dir_all(root);
}
