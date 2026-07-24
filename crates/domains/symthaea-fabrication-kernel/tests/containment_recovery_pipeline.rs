// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use symthaea_fabrication_kernel::{
    ContainmentReplayMismatch, FabricationContainmentState, KeyLifecycleStatus, KeyTrustRecord,
    KeyUsage, Sha256Digest, SignatureAlgorithm, TrustSnapshot, build_containment_replay_contract,
    verify_containment_replay_contract, verify_containment_state_successor,
};

fn trust_snapshot() -> TrustSnapshot {
    TrustSnapshot::new(
        1,
        100,
        10_000,
        vec![KeyTrustRecord {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "containment-authority-a".into(),
            not_before_unix_s: 100,
            not_after_unix_s: Some(10_000),
            status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([
                KeyUsage::ThresholdCeremony,
                KeyUsage::SignerCompromise,
                KeyUsage::PostRollbackRequalification,
            ]),
        }],
    )
    .unwrap()
}

#[test]
fn containment_state_and_replay_are_exact_and_successor_bound() {
    let genesis = FabricationContainmentState::genesis(7, Sha256Digest([1; 32])).unwrap();
    let next = genesis.successor(8, Sha256Digest([2; 32])).unwrap();
    assert_eq!(verify_containment_state_successor(&genesis, &next), Ok(()));

    let trust = trust_snapshot();
    let contract = build_containment_replay_contract(
        Sha256Digest([3; 32]),
        Sha256Digest([4; 32]),
        &trust,
        &next,
        500,
    )
    .unwrap();
    let report = verify_containment_replay_contract(
        &contract,
        Sha256Digest([3; 32]),
        Sha256Digest([4; 32]),
        &trust,
        &next,
    )
    .unwrap();
    assert!(report.exact());

    let drift = verify_containment_replay_contract(
        &contract,
        Sha256Digest([9; 32]),
        Sha256Digest([4; 32]),
        &trust,
        &next,
    )
    .unwrap();
    assert_eq!(
        drift.mismatches,
        vec![ContainmentReplayMismatch::SourceTree]
    );
}

#[test]
fn containment_successor_rejects_resilience_state_substitution() {
    let genesis = FabricationContainmentState::genesis(7, Sha256Digest([1; 32])).unwrap();
    assert!(genesis.successor(7, Sha256Digest([2; 32])).is_err());
}
