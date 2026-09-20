// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![cfg(target_os = "linux")]

use super::{
    perceptual_enrollment_e2e_fixture::Mel003EnrollmentE2eQualificationFixtureV1,
    perceptual_enrollment_orchestrator::{
        EnrollmentWitnessAdapterFailureKindV1, EnrollmentWitnessAdapterFailureV1,
        EnrollmentWitnessProviderAdapterV1, EnrollmentWitnessProviderStageV1,
        PerceptualEnrollmentOrchestratorErrorV1, PerceptualEnrollmentRecoveryOutcomeV1,
    },
    perceptual_enrollment_witness_provider::{
        collection_signing_request_commitment, ExternalEnrollmentCollectionSignatureV1,
        ExternalEnrollmentCollectionSigningRequestV1, ExternalEnrollmentWitnessServiceRequestV1,
        ExternalEnrollmentWitnessServiceResponseV1,
    },
    perceptual_enrollment_witness_provider_fixture::{
        DurableQualificationEnrollmentWitnessProviderV1, QualificationProviderFaultPointV1,
    },
};
use std::path::PathBuf;

struct WrongWitnessIdentityProviderV1 {
    inner: DurableQualificationEnrollmentWitnessProviderV1,
}

impl EnrollmentWitnessProviderAdapterV1 for WrongWitnessIdentityProviderV1 {
    fn sign_collection(
        &mut self,
        request: &ExternalEnrollmentCollectionSigningRequestV1,
    ) -> Result<ExternalEnrollmentCollectionSignatureV1, EnrollmentWitnessAdapterFailureV1> {
        self.inner.sign_collection(request)
    }

    fn witness_enrollment(
        &mut self,
        request: &ExternalEnrollmentWitnessServiceRequestV1,
    ) -> Result<ExternalEnrollmentWitnessServiceResponseV1, EnrollmentWitnessAdapterFailureV1> {
        let mut response = self.inner.witness_enrollment(request)?;
        response.witness_signer_id = "wrong-qualification-witness".into();
        Ok(response)
    }
}

struct ChangedCollectionReplayProviderV1 {
    inner: DurableQualificationEnrollmentWitnessProviderV1,
}

impl EnrollmentWitnessProviderAdapterV1 for ChangedCollectionReplayProviderV1 {
    fn sign_collection(
        &mut self,
        request: &ExternalEnrollmentCollectionSigningRequestV1,
    ) -> Result<ExternalEnrollmentCollectionSignatureV1, EnrollmentWitnessAdapterFailureV1> {
        let mut changed = request.clone();
        changed.signing_transcript.push(0x00);
        changed.collection_request_sha256 = collection_signing_request_commitment(&changed)
            .map_err(|_| {
                EnrollmentWitnessAdapterFailureV1::new(
                    EnrollmentWitnessAdapterFailureKindV1::Rejected,
                )
            })?;
        self.inner.sign_collection(&changed)
    }

    fn witness_enrollment(
        &mut self,
        request: &ExternalEnrollmentWitnessServiceRequestV1,
    ) -> Result<ExternalEnrollmentWitnessServiceResponseV1, EnrollmentWitnessAdapterFailureV1> {
        self.inner.witness_enrollment(request)
    }
}

#[test]
fn wrong_independent_witness_identity_is_rejected_before_confirmation() {
    let fixture = Mel003EnrollmentE2eQualificationFixtureV1::new_valid().unwrap();
    let gate = fixture.valid_next_eligibility_gate(0).unwrap();
    let initial_coordinator = fixture.coordinator_state().unwrap();
    let mut provider = WrongWitnessIdentityProviderV1 {
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

    // Both underlying provider roles persisted valid evidence before the wrapper
    // corrupted only the returned witness identity. Clean replay can recover.
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
fn same_request_identity_with_changed_collection_payload_is_rejected_end_to_end() {
    let fixture = Mel003EnrollmentE2eQualificationFixtureV1::new_valid().unwrap();
    let gate = fixture.valid_next_eligibility_gate(0).unwrap();
    let mut provider = fixture.provider().unwrap();
    provider.inject_collection_fault(QualificationProviderFaultPointV1::AfterPersistUncertain);

    let first = fixture
        .orchestrator()
        .allocate_and_witness_one(&gate, &mut provider)
        .unwrap_err();
    assert!(matches!(
        first,
        PerceptualEnrollmentOrchestratorErrorV1::ProviderAdapter {
            stage: EnrollmentWitnessProviderStageV1::CollectionSigner,
            failure,
        } if failure.kind == EnrollmentWitnessAdapterFailureKindV1::Uncertain
    ));

    let mut conflicting = ChangedCollectionReplayProviderV1 {
        inner: fixture.provider().unwrap(),
    };
    let conflict = fixture
        .orchestrator()
        .recover_pending(&mut conflicting)
        .unwrap_err();
    assert!(matches!(
        conflict,
        PerceptualEnrollmentOrchestratorErrorV1::ProviderAdapter {
            stage: EnrollmentWitnessProviderStageV1::CollectionSigner,
            failure,
        } if failure.kind == EnrollmentWitnessAdapterFailureKindV1::Rejected
    ));
    assert_eq!(fixture.current_allocator_state().unwrap().ledger.allocations.len(), 1);
    assert!(fixture
        .coordinator_state()
        .unwrap()
        .confirmed_enrollment_ledger
        .allocations
        .is_empty());

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
fn corrupted_durable_collection_journal_fails_closed_without_confirmation() {
    let fixture = Mel003EnrollmentE2eQualificationFixtureV1::new_valid().unwrap();
    let gate = fixture.valid_next_eligibility_gate(0).unwrap();
    let initial_coordinator = fixture.coordinator_state().unwrap();
    let mut provider = fixture.provider().unwrap();
    provider.inject_collection_fault(QualificationProviderFaultPointV1::AfterPersistUncertain);

    let first = fixture
        .orchestrator()
        .allocate_and_witness_one(&gate, &mut provider)
        .unwrap_err();
    assert!(matches!(
        first,
        PerceptualEnrollmentOrchestratorErrorV1::ProviderAdapter {
            stage: EnrollmentWitnessProviderStageV1::CollectionSigner,
            failure,
        } if failure.kind == EnrollmentWitnessAdapterFailureKindV1::Uncertain
    ));
    drop(provider);

    let accepted = accepted_journal_file(fixture.root().join("provider-collection"));
    std::fs::write(&accepted, b"{corrupt-provider-journal").unwrap();

    let mut restarted_provider = fixture.provider().unwrap();
    let error = fixture
        .orchestrator()
        .recover_pending(&mut restarted_provider)
        .unwrap_err();
    assert!(matches!(
        error,
        PerceptualEnrollmentOrchestratorErrorV1::ProviderAdapter {
            stage: EnrollmentWitnessProviderStageV1::CollectionSigner,
            failure,
        } if failure.kind == EnrollmentWitnessAdapterFailureKindV1::Rejected
    ));
    assert_eq!(fixture.current_allocator_state().unwrap().ledger.allocations.len(), 1);
    assert_eq!(
        fixture.coordinator_state().unwrap().coordinator_state_sha256,
        initial_coordinator.coordinator_state_sha256
    );
}

fn accepted_journal_file(root: PathBuf) -> PathBuf {
    let mut matches = std::fs::read_dir(root)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with(".accepted.json"))
                .unwrap_or(false)
        });
    let path = matches.next().expect("one durable collection acceptance");
    assert!(matches.next().is_none(), "expected exactly one accepted operation");
    path
}
