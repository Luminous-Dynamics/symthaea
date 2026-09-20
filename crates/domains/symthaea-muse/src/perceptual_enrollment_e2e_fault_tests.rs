// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![cfg(target_os = "linux")]

use super::{
    perceptual_enrollment_coordinator::DurableEnrollmentCoordinatorErrorV1,
    perceptual_enrollment_e2e_fixture::Mel003EnrollmentE2eQualificationFixtureV1,
    perceptual_enrollment_orchestrator::{
        EnrollmentWitnessAdapterFailureKindV1, EnrollmentWitnessAdapterFailureV1,
        EnrollmentWitnessProviderAdapterV1, EnrollmentWitnessProviderStageV1,
        PerceptualEnrollmentOrchestratorErrorV1, PerceptualEnrollmentRecoveryOutcomeV1,
    },
    perceptual_enrollment_witness_provider::{
        ExternalEnrollmentCollectionSignatureV1, ExternalEnrollmentCollectionSigningRequestV1,
        ExternalEnrollmentWitnessServiceRequestV1, ExternalEnrollmentWitnessServiceResponseV1,
    },
    perceptual_enrollment_witness_provider_fixture::{
        DurableQualificationEnrollmentWitnessProviderV1, QualificationProviderFaultPointV1,
    },
};
use std::sync::{Arc, Barrier};

struct WrongCollectionIdentityProviderV1 {
    inner: DurableQualificationEnrollmentWitnessProviderV1,
}

impl EnrollmentWitnessProviderAdapterV1 for WrongCollectionIdentityProviderV1 {
    fn sign_collection(
        &mut self,
        request: &ExternalEnrollmentCollectionSigningRequestV1,
    ) -> Result<ExternalEnrollmentCollectionSignatureV1, EnrollmentWitnessAdapterFailureV1> {
        let mut response = self.inner.sign_collection(request)?;
        response.signer_id = "wrong-qualification-signer".into();
        Ok(response)
    }

    fn witness_enrollment(
        &mut self,
        request: &ExternalEnrollmentWitnessServiceRequestV1,
    ) -> Result<ExternalEnrollmentWitnessServiceResponseV1, EnrollmentWitnessAdapterFailureV1> {
        self.inner.witness_enrollment(request)
    }
}

struct CollectionBarrierProviderV1 {
    inner: DurableQualificationEnrollmentWitnessProviderV1,
    barrier: Arc<Barrier>,
    barrier_used: bool,
}

impl EnrollmentWitnessProviderAdapterV1 for CollectionBarrierProviderV1 {
    fn sign_collection(
        &mut self,
        request: &ExternalEnrollmentCollectionSigningRequestV1,
    ) -> Result<ExternalEnrollmentCollectionSignatureV1, EnrollmentWitnessAdapterFailureV1> {
        let result = self.inner.sign_collection(request);
        if !self.barrier_used {
            self.barrier_used = true;
            self.barrier.wait();
        }
        result
    }

    fn witness_enrollment(
        &mut self,
        request: &ExternalEnrollmentWitnessServiceRequestV1,
    ) -> Result<ExternalEnrollmentWitnessServiceResponseV1, EnrollmentWitnessAdapterFailureV1> {
        self.inner.witness_enrollment(request)
    }
}

#[test]
fn before_persist_unavailability_leaves_one_recoverable_pending_allocation() {
    let fixture = Mel003EnrollmentE2eQualificationFixtureV1::new_valid().unwrap();
    let gate = fixture.valid_next_eligibility_gate(0).unwrap();
    let initial_coordinator = fixture.coordinator_state().unwrap();
    let mut provider = fixture.provider().unwrap();
    provider.inject_collection_fault(QualificationProviderFaultPointV1::BeforePersistUnavailable);

    let error = fixture
        .orchestrator()
        .allocate_and_witness_one(&gate, &mut provider)
        .unwrap_err();
    assert!(matches!(
        error,
        PerceptualEnrollmentOrchestratorErrorV1::ProviderAdapter {
            stage: EnrollmentWitnessProviderStageV1::CollectionSigner,
            failure,
        } if failure.kind == EnrollmentWitnessAdapterFailureKindV1::Unavailable
    ));
    assert_eq!(fixture.current_allocator_state().unwrap().ledger.allocations.len(), 1);
    assert_eq!(
        fixture.coordinator_state().unwrap().coordinator_state_sha256,
        initial_coordinator.coordinator_state_sha256
    );

    let mut restarted_provider = fixture.provider().unwrap();
    let recovered = fixture
        .orchestrator()
        .recover_pending(&mut restarted_provider)
        .unwrap();
    assert!(matches!(
        recovered,
        PerceptualEnrollmentRecoveryOutcomeV1::Recovered(_)
    ));
    assert_eq!(
        fixture
            .coordinator_state()
            .unwrap()
            .confirmed_enrollment_ledger
            .allocations
            .len(),
        1
    );
}

#[test]
fn wrong_collection_signer_identity_is_rejected_before_confirmation() {
    let fixture = Mel003EnrollmentE2eQualificationFixtureV1::new_valid().unwrap();
    let gate = fixture.valid_next_eligibility_gate(0).unwrap();
    let initial_coordinator = fixture.coordinator_state().unwrap();
    let mut provider = WrongCollectionIdentityProviderV1 {
        inner: fixture.provider().unwrap(),
    };

    let error = fixture
        .orchestrator()
        .allocate_and_witness_one(&gate, &mut provider)
        .unwrap_err();
    assert!(matches!(
        error,
        PerceptualEnrollmentOrchestratorErrorV1::ProviderProtocol(_)
    ));
    assert_eq!(fixture.current_allocator_state().unwrap().ledger.allocations.len(), 1);
    assert_eq!(
        fixture.coordinator_state().unwrap().coordinator_state_sha256,
        initial_coordinator.coordinator_state_sha256
    );

    // The underlying provider durably stored the valid response before the
    // wrapper altered its returned presentation. A clean adapter can therefore
    // replay the accepted semantic operation and recover normally.
    let mut clean_provider = fixture.provider().unwrap();
    let recovered = fixture
        .orchestrator()
        .recover_pending(&mut clean_provider)
        .unwrap();
    assert!(matches!(
        recovered,
        PerceptualEnrollmentRecoveryOutcomeV1::Recovered(_)
    ));
}

#[test]
fn concurrent_recovery_callers_share_one_provider_operation_but_only_one_cas_wins() {
    let fixture = Mel003EnrollmentE2eQualificationFixtureV1::new_valid().unwrap();
    let gate = fixture.valid_next_eligibility_gate(0).unwrap();
    let mut unavailable = fixture.provider().unwrap();
    unavailable.inject_collection_fault(QualificationProviderFaultPointV1::BeforePersistUnavailable);
    let initial = fixture
        .orchestrator()
        .allocate_and_witness_one(&gate, &mut unavailable)
        .unwrap_err();
    assert!(matches!(
        initial,
        PerceptualEnrollmentOrchestratorErrorV1::ProviderAdapter { .. }
    ));
    assert_eq!(fixture.current_allocator_state().unwrap().ledger.allocations.len(), 1);

    let barrier = Arc::new(Barrier::new(2));
    let provider_a = CollectionBarrierProviderV1 {
        inner: fixture.provider().unwrap(),
        barrier: Arc::clone(&barrier),
        barrier_used: false,
    };
    let provider_b = CollectionBarrierProviderV1 {
        inner: fixture.provider().unwrap(),
        barrier,
        barrier_used: false,
    };

    let fixture_ref = &fixture;
    let (left, right) = std::thread::scope(|scope| {
        let left_fixture = fixture_ref;
        let right_fixture = fixture_ref;
        let left = scope.spawn(move || {
            let mut provider = provider_a;
            left_fixture
                .orchestrator()
                .recover_pending(&mut provider)
                .map(classify_race_result)
                .unwrap_or_else(classify_race_error)
        });
        let right = scope.spawn(move || {
            let mut provider = provider_b;
            right_fixture
                .orchestrator()
                .recover_pending(&mut provider)
                .map(classify_race_result)
                .unwrap_or_else(classify_race_error)
        });
        (left.join().unwrap(), right.join().unwrap())
    });

    let recovered = (left == 1) as usize + (right == 1) as usize;
    let stale_cas = (left == 2) as usize + (right == 2) as usize;
    assert_eq!(recovered, 1, "exactly one concurrent caller may return authority");
    assert_eq!(stale_cas, 1, "the other caller must lose exact coordinator CAS");
    assert_eq!(
        fixture
            .coordinator_state()
            .unwrap()
            .confirmed_enrollment_ledger
            .allocations
            .len(),
        1
    );
}

fn classify_race_result(outcome: PerceptualEnrollmentRecoveryOutcomeV1) -> u8 {
    match outcome {
        PerceptualEnrollmentRecoveryOutcomeV1::Recovered(_) => 1,
        PerceptualEnrollmentRecoveryOutcomeV1::Synchronized => 3,
    }
}

fn classify_race_error(error: PerceptualEnrollmentOrchestratorErrorV1) -> u8 {
    match error {
        PerceptualEnrollmentOrchestratorErrorV1::Coordinator(
            DurableEnrollmentCoordinatorErrorV1::ExpectedCoordinatorHeadMismatch { .. },
        ) => 2,
        _ => 4,
    }
}
