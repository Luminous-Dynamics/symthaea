// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![cfg(target_os = "linux")]

use super::{
    perceptual_enrollment_e2e_fixture::Mel003EnrollmentE2eQualificationFixtureV1,
    perceptual_enrollment_orchestrator::{
        EnrollmentWitnessAdapterFailureKindV1, EnrollmentWitnessProviderStageV1,
        PerceptualEnrollmentOrchestratorErrorV1, PerceptualEnrollmentRecoveryOutcomeV1,
    },
    perceptual_enrollment_witness_provider_fixture::QualificationProviderFaultPointV1,
};

#[test]
fn canonical_fixture_clean_enrollment_confirms_one_witnessed_slot() {
    let fixture = Mel003EnrollmentE2eQualificationFixtureV1::new_valid().unwrap();
    let initial = fixture.coordinator_state().unwrap();
    assert!(initial.confirmed_enrollment_ledger.allocations.is_empty());
    assert!(initial.confirmed_witness_bundle.entries.is_empty());

    let gate = fixture.valid_next_eligibility_gate(0).unwrap();
    let mut provider = fixture.provider().unwrap();
    let authority = fixture
        .orchestrator()
        .allocate_and_witness_one(&gate, &mut provider)
        .unwrap();

    assert_eq!(authority.receipt().sequence, 0);
    let final_coordinator = fixture.coordinator_state().unwrap();
    assert_eq!(final_coordinator.confirmed_enrollment_ledger.allocations.len(), 1);
    assert_eq!(final_coordinator.confirmed_witness_bundle.entries.len(), 1);
    assert_eq!(
        &final_coordinator.confirmed_witness_bundle.entries[0],
        authority.receipt()
    );
    assert_eq!(
        final_coordinator.confirmed_enrollment_ledger.ledger_sha256,
        authority.current_enrollment_ledger_sha256()
    );
    let allocator = fixture.current_allocator_state().unwrap();
    assert_eq!(allocator.ledger.allocations.len(), 1);
    assert_eq!(
        allocator.ledger.ledger_sha256,
        final_coordinator.confirmed_enrollment_ledger.ledger_sha256
    );
}

#[test]
fn collection_signer_post_persist_uncertainty_recovers_after_adapter_restart() {
    let fixture = Mel003EnrollmentE2eQualificationFixtureV1::new_valid().unwrap();
    let initial_coordinator = fixture.coordinator_state().unwrap();
    let gate = fixture.valid_next_eligibility_gate(0).unwrap();
    let mut provider = fixture.provider().unwrap();
    provider.inject_collection_fault(QualificationProviderFaultPointV1::AfterPersistUncertain);

    let error = fixture
        .orchestrator()
        .allocate_and_witness_one(&gate, &mut provider)
        .unwrap_err();
    assert!(matches!(
        error,
        PerceptualEnrollmentOrchestratorErrorV1::ProviderAdapter {
            stage: EnrollmentWitnessProviderStageV1::CollectionSigner,
            failure,
        } if failure.kind == EnrollmentWitnessAdapterFailureKindV1::Uncertain
    ));

    let pending_allocator = fixture.current_allocator_state().unwrap();
    assert_eq!(pending_allocator.ledger.allocations.len(), 1);
    let unchanged_coordinator = fixture.coordinator_state().unwrap();
    assert_eq!(
        unchanged_coordinator.coordinator_state_sha256,
        initial_coordinator.coordinator_state_sha256
    );
    assert!(unchanged_coordinator.confirmed_enrollment_ledger.allocations.is_empty());

    drop(provider);
    let mut restarted_provider = fixture.provider().unwrap();
    let recovered = fixture
        .orchestrator()
        .recover_pending(&mut restarted_provider)
        .unwrap();
    let authority = match recovered {
        PerceptualEnrollmentRecoveryOutcomeV1::Recovered(authority) => authority,
        PerceptualEnrollmentRecoveryOutcomeV1::Synchronized => {
            panic!("pending durable allocation unexpectedly disappeared")
        }
    };
    assert_eq!(authority.receipt().sequence, 0);
    let final_coordinator = fixture.coordinator_state().unwrap();
    assert_eq!(final_coordinator.confirmed_enrollment_ledger.allocations.len(), 1);
    assert_eq!(final_coordinator.confirmed_witness_bundle.entries.len(), 1);
    assert_eq!(
        &final_coordinator.confirmed_witness_bundle.entries[0],
        authority.receipt()
    );
}

#[test]
fn independent_witness_post_persist_uncertainty_recovers_after_adapter_restart() {
    let fixture = Mel003EnrollmentE2eQualificationFixtureV1::new_valid().unwrap();
    let gate = fixture.valid_next_eligibility_gate(0).unwrap();
    let mut provider = fixture.provider().unwrap();
    provider.inject_witness_fault(QualificationProviderFaultPointV1::AfterPersistUncertain);

    let error = fixture
        .orchestrator()
        .allocate_and_witness_one(&gate, &mut provider)
        .unwrap_err();
    assert!(matches!(
        error,
        PerceptualEnrollmentOrchestratorErrorV1::ProviderAdapter {
            stage: EnrollmentWitnessProviderStageV1::IndependentWitness,
            failure,
        } if failure.kind == EnrollmentWitnessAdapterFailureKindV1::Uncertain
    ));
    assert_eq!(fixture.current_allocator_state().unwrap().ledger.allocations.len(), 1);
    assert!(fixture
        .coordinator_state()
        .unwrap()
        .confirmed_enrollment_ledger
        .allocations
        .is_empty());

    drop(provider);
    let mut restarted_provider = fixture.provider().unwrap();
    let recovered = fixture
        .orchestrator()
        .recover_pending(&mut restarted_provider)
        .unwrap();
    let authority = match recovered {
        PerceptualEnrollmentRecoveryOutcomeV1::Recovered(authority) => authority,
        PerceptualEnrollmentRecoveryOutcomeV1::Synchronized => {
            panic!("pending witness operation unexpectedly disappeared")
        }
    };
    let final_coordinator = fixture.coordinator_state().unwrap();
    assert_eq!(final_coordinator.confirmed_witness_bundle.entries.len(), 1);
    assert_eq!(
        &final_coordinator.confirmed_witness_bundle.entries[0],
        authority.receipt()
    );
    assert!(authority
        .receipt()
        .witness_anchor_reference
        .starts_with("mel003-local-qualification-anchor-v1:"));
}

#[test]
fn pending_enrollment_blocks_a_second_gate_until_recovery_finishes() {
    let fixture = Mel003EnrollmentE2eQualificationFixtureV1::new_valid().unwrap();
    let first_gate = fixture.valid_next_eligibility_gate(0).unwrap();
    let second_gate = fixture.valid_next_eligibility_gate(1).unwrap();
    let mut provider = fixture.provider().unwrap();
    provider.inject_collection_fault(QualificationProviderFaultPointV1::AfterPersistUncertain);

    let first_error = fixture
        .orchestrator()
        .allocate_and_witness_one(&first_gate, &mut provider)
        .unwrap_err();
    assert!(matches!(
        first_error,
        PerceptualEnrollmentOrchestratorErrorV1::ProviderAdapter { .. }
    ));

    let mut restarted_provider = fixture.provider().unwrap();
    let second_error = fixture
        .orchestrator()
        .allocate_and_witness_one(&second_gate, &mut restarted_provider)
        .unwrap_err();
    assert!(matches!(
        second_error,
        PerceptualEnrollmentOrchestratorErrorV1::PendingRecoveryRequired { .. }
    ));
    assert_eq!(fixture.current_allocator_state().unwrap().ledger.allocations.len(), 1);
    assert!(fixture
        .coordinator_state()
        .unwrap()
        .confirmed_enrollment_ledger
        .allocations
        .is_empty());

    let recovered = fixture
        .orchestrator()
        .recover_pending(&mut restarted_provider)
        .unwrap();
    assert!(matches!(
        recovered,
        PerceptualEnrollmentRecoveryOutcomeV1::Recovered(_)
    ));
}
