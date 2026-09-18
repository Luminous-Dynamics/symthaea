// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea::knowledge::belief_mutation_seal_persistence::BeliefMutationEvidenceSealCapsuleV1;
use symthaea::knowledge::belief_mutation_seal_wire::BeliefMutationSealWireV1;
use symthaea::knowledge::belief_mutation_seal_wire_validation::{
    BeliefMutationSealValidationError, BeliefMutationSealWireValidator,
};
use symthaea::knowledge::{
    BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistory,
    BeliefRevisionHistoryCapsuleV1, BeliefRevisionSchemaHistoryCapsuleV1,
    BeliefRevisionSchemaHistoryV1, EpistemicLedger, EpistemicLedgerInventoryV1,
    EpistemicRestartCapsuleV1, EpistemicRestartCapsuleV2, EpistemicRestartWireV2,
    EpistemicSupportStore,
};

fn empty_restart_and_seals() -> (
    symthaea::knowledge::EpistemicRestartWireSnapshotV2,
    symthaea::knowledge::belief_mutation_seal_wire::BeliefMutationSealWireSnapshotV1,
) {
    let ledger = EpistemicLedger::new();
    let store = EpistemicSupportStore::new();
    let mutations = BeliefMutationPersistenceCapsuleV1::capture(&store, &[], 1).unwrap();
    let history = BeliefRevisionHistory::new();
    let revisions = BeliefRevisionHistoryCapsuleV1::capture(&history, &mutations, 1).unwrap();
    let inventory = EpistemicLedgerInventoryV1::new(vec![], vec![], vec![]).unwrap();
    let base = EpistemicRestartCapsuleV1::capture(
        &ledger,
        &inventory,
        &mutations,
        &revisions,
        1,
    )
    .unwrap();

    let schema_history = BeliefRevisionSchemaHistoryV1::new();
    let schemas = BeliefRevisionSchemaHistoryCapsuleV1::capture(
        &schema_history,
        &history,
        &revisions,
        1,
    )
    .unwrap();
    let v2 = EpistemicRestartCapsuleV2::capture(&base, &schemas).unwrap();
    let base_wire = EpistemicRestartWireV2::encode(&v2).unwrap();
    let base_snapshot = EpistemicRestartWireV2::decode(&base_wire).unwrap();

    let seals = BeliefMutationEvidenceSealCapsuleV1::capture(&[], &mutations, &ledger, 1)
        .unwrap();
    let seal_wire = BeliefMutationSealWireV1::encode(&seals).unwrap();
    let seal_snapshot = BeliefMutationSealWireV1::decode(&seal_wire).unwrap();

    (base_snapshot, seal_snapshot)
}

#[test]
fn consistent_sidecar_is_valid_but_does_not_claim_independent_census_completeness() {
    let (base, seals) = empty_restart_and_seals();
    let report = BeliefMutationSealWireValidator::validate(&base, &seals).unwrap();

    assert!(report.cross_component_consistent());
    assert!(!report.historical_census_completeness_independently_proven());
    assert!(!report.capsule_construction_authorized());
    assert!(!report.hydration_authorized());
    assert!(!report.activation_authorized());
    assert_eq!(report.mutation_count(), 0);
    assert_eq!(report.sealed_evidence_count(), 0);
}

#[test]
fn sidecar_capsule_digest_is_rederived_instead_of_trusted() {
    let (base, mut seals) = empty_restart_and_seals();
    seals.claimed_capsule_digest[0] ^= 0x01;

    assert!(matches!(
        BeliefMutationSealWireValidator::validate(&base, &seals),
        Err(BeliefMutationSealValidationError::CapsuleDigestMismatch)
    ));
}

#[test]
fn sidecar_must_bind_the_same_mutation_capture_epoch() {
    let (base, mut seals) = empty_restart_and_seals();
    seals.linked_mutation_capture_cycle = 2;

    assert!(matches!(
        BeliefMutationSealWireValidator::validate(&base, &seals),
        Err(BeliefMutationSealValidationError::MutationCaptureCycleMismatch { .. })
    ));
}
