// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deployment-owned trust binding for verifier-profile adoption persistence.
//!
//! The public registry-store adapter in `profile_adoption_registry_store` is
//! intentionally caller-selectable. Its opaque CAS witness therefore proves only
//! what that selected adapter reported. This module adds a distinct theorem: the
//! CAS must execute through an opaque store handle that was paired with one exact
//! deployment-owned store-authority binding before the operation.
//!
//! Core theorem:
//!
//! `StoreAuthorityBinding != TrustedStoreHandle != StoreObservedCas != StoreAuthorityQualifiedCas != CommittedReceipt != CurrentAuthority`.
//!
//! A binding descriptor is serializable configuration and carries no authority by
//! itself. `TrustedVerifierProfileAdoptionRegistryStoreV1` has no public production
//! constructor in this tranche. A future deployment/provisioning boundary must be
//! the only production source after independently proving that the wrapped store
//! instance is the store named by the binding.

use std::error::Error as StdError;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::profile_adoption_registry_store::{
    StoreObservedVerifierProfileAdoptionCasV1, VerifierProfileAdoptionAtomicRegistryStoreV1,
    VerifierProfileAdoptionRegistryCasError, perform_verifier_profile_adoption_registry_cas,
};
use crate::profile_adoption_registry_write::{
    VerifierProfileAdoptionRegistryWriteClaimV1, VerifierProfileAdoptionRegistryWriteError,
};

pub const VERIFIER_PROFILE_ADOPTION_STORE_AUTHORITY_BINDING_SCHEMA_V1: &str =
    "symthaea-continuity-verifier-profile-adoption-store-authority-binding-v1";
const STORE_AUTHORITY_BINDING_DOMAIN: &[u8] =
    b"symthaea.continuity.verifier-profile-adoption.store-authority-binding.v1\0";
const STORE_AUTHORITY_QUALIFIED_CAS_DOMAIN: &[u8] =
    b"symthaea.continuity.verifier-profile-adoption.store-authority-qualified-cas.v1\0";

/// Content identity of one deployment-owned store-authority binding descriptor.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct VerifierProfileAdoptionStoreAuthorityBindingIdV1([u8; 32]);

impl VerifierProfileAdoptionStoreAuthorityBindingIdV1 {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Serializable description of the persistence authority expected by one
/// deployment.
///
/// This value is deliberately **not** a trusted-store capability. Callers may
/// construct, copy, serialize, and deserialize descriptors without gaining the
/// ability to mint a trusted store handle. The trust-bearing step is the separate
/// pairing of an actual store instance with this exact descriptor by a future
/// deployment-owned provisioning boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionStoreAuthorityBindingV1 {
    schema_version: String,
    deployment_id: String,
    registry_id: String,
    registry_epoch: u64,
    store_id: String,
    store_instance_binding_digest: [u8; 32],
    store_configuration_digest: [u8; 32],
    store_trust_root_digest: [u8; 32],
    provisioning_epoch: u64,
    binding_id: VerifierProfileAdoptionStoreAuthorityBindingIdV1,
}

impl VerifierProfileAdoptionStoreAuthorityBindingV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        deployment_id: impl Into<String>,
        registry_id: impl Into<String>,
        registry_epoch: u64,
        store_id: impl Into<String>,
        store_instance_binding_digest: [u8; 32],
        store_configuration_digest: [u8; 32],
        store_trust_root_digest: [u8; 32],
        provisioning_epoch: u64,
    ) -> Result<Self, VerifierProfileAdoptionStoreAuthorityError> {
        let deployment_id = checked_text("deployment_id", deployment_id.into())?;
        let registry_id = checked_text("registry_id", registry_id.into())?;
        let store_id = checked_text("store_id", store_id.into())?;
        if registry_epoch == 0 {
            return Err(VerifierProfileAdoptionStoreAuthorityError::ZeroRegistryEpoch);
        }
        if store_instance_binding_digest == [0; 32] {
            return Err(
                VerifierProfileAdoptionStoreAuthorityError::ZeroStoreInstanceBindingDigest,
            );
        }
        if store_configuration_digest == [0; 32] {
            return Err(
                VerifierProfileAdoptionStoreAuthorityError::ZeroStoreConfigurationDigest,
            );
        }
        if store_trust_root_digest == [0; 32] {
            return Err(VerifierProfileAdoptionStoreAuthorityError::ZeroStoreTrustRootDigest);
        }
        if provisioning_epoch == 0 {
            return Err(VerifierProfileAdoptionStoreAuthorityError::ZeroProvisioningEpoch);
        }

        let mut binding = Self {
            schema_version: VERIFIER_PROFILE_ADOPTION_STORE_AUTHORITY_BINDING_SCHEMA_V1.to_owned(),
            deployment_id,
            registry_id,
            registry_epoch,
            store_id,
            store_instance_binding_digest,
            store_configuration_digest,
            store_trust_root_digest,
            provisioning_epoch,
            binding_id: VerifierProfileAdoptionStoreAuthorityBindingIdV1([0; 32]),
        };
        binding.binding_id =
            VerifierProfileAdoptionStoreAuthorityBindingIdV1(binding.hash_binding());
        binding.validate()?;
        Ok(binding)
    }

    pub fn validate(&self) -> Result<(), VerifierProfileAdoptionStoreAuthorityError> {
        if self.schema_version != VERIFIER_PROFILE_ADOPTION_STORE_AUTHORITY_BINDING_SCHEMA_V1 {
            return Err(VerifierProfileAdoptionStoreAuthorityError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        checked_text("deployment_id", self.deployment_id.clone())?;
        checked_text("registry_id", self.registry_id.clone())?;
        checked_text("store_id", self.store_id.clone())?;
        if self.registry_epoch == 0 {
            return Err(VerifierProfileAdoptionStoreAuthorityError::ZeroRegistryEpoch);
        }
        if self.store_instance_binding_digest == [0; 32] {
            return Err(
                VerifierProfileAdoptionStoreAuthorityError::ZeroStoreInstanceBindingDigest,
            );
        }
        if self.store_configuration_digest == [0; 32] {
            return Err(
                VerifierProfileAdoptionStoreAuthorityError::ZeroStoreConfigurationDigest,
            );
        }
        if self.store_trust_root_digest == [0; 32] {
            return Err(VerifierProfileAdoptionStoreAuthorityError::ZeroStoreTrustRootDigest);
        }
        if self.provisioning_epoch == 0 {
            return Err(VerifierProfileAdoptionStoreAuthorityError::ZeroProvisioningEpoch);
        }
        let expected = VerifierProfileAdoptionStoreAuthorityBindingIdV1(self.hash_binding());
        if expected != self.binding_id {
            return Err(VerifierProfileAdoptionStoreAuthorityError::BindingIdentityMismatch);
        }
        Ok(())
    }

    /// Validate that this binding governs the exact registry generation named by
    /// the write claim. This is structural compatibility only; it does not turn
    /// the descriptor itself into a trusted store capability.
    pub fn validate_against_claim(
        &self,
        claim: &VerifierProfileAdoptionRegistryWriteClaimV1,
    ) -> Result<(), VerifierProfileAdoptionStoreAuthorityError> {
        self.validate()?;
        claim.validate()?;
        if self.registry_id != claim.registry_id() {
            return Err(VerifierProfileAdoptionStoreAuthorityError::RegistryIdMismatch);
        }
        if self.registry_epoch != claim.registry_epoch() {
            return Err(VerifierProfileAdoptionStoreAuthorityError::RegistryEpochMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> VerifierProfileAdoptionStoreAuthorityBindingIdV1 {
        self.binding_id
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }

    pub fn registry_id(&self) -> &str {
        &self.registry_id
    }

    pub fn registry_epoch(&self) -> u64 {
        self.registry_epoch
    }

    pub fn store_id(&self) -> &str {
        &self.store_id
    }

    pub fn store_instance_binding_digest(&self) -> [u8; 32] {
        self.store_instance_binding_digest
    }

    pub fn store_configuration_digest(&self) -> [u8; 32] {
        self.store_configuration_digest
    }

    pub fn store_trust_root_digest(&self) -> [u8; 32] {
        self.store_trust_root_digest
    }

    pub fn provisioning_epoch(&self) -> u64 {
        self.provisioning_epoch
    }

    fn hash_binding(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        put_str(
            &mut bytes,
            VERIFIER_PROFILE_ADOPTION_STORE_AUTHORITY_BINDING_SCHEMA_V1,
        );
        put_str(&mut bytes, &self.deployment_id);
        put_str(&mut bytes, &self.registry_id);
        bytes.extend_from_slice(&self.registry_epoch.to_le_bytes());
        put_str(&mut bytes, &self.store_id);
        bytes.extend_from_slice(&self.store_instance_binding_digest);
        bytes.extend_from_slice(&self.store_configuration_digest);
        bytes.extend_from_slice(&self.store_trust_root_digest);
        bytes.extend_from_slice(&self.provisioning_epoch.to_le_bytes());
        domain_hash(STORE_AUTHORITY_BINDING_DOMAIN, &bytes)
    }
}

/// Opaque pairing of one exact store instance with one exact deployment-owned
/// store-authority binding.
///
/// There is intentionally no public production constructor. In particular, a
/// caller-selected `VerifierProfileAdoptionAtomicRegistryStoreV1` implementation
/// cannot wrap itself and thereby manufacture persistence authority. A future
/// deployment/provisioning module must establish the independent pairing theorem
/// before it may add a crate-private production constructor.
pub struct TrustedVerifierProfileAdoptionRegistryStoreV1<S> {
    binding: VerifierProfileAdoptionStoreAuthorityBindingV1,
    store: S,
}

impl<S> TrustedVerifierProfileAdoptionRegistryStoreV1<S> {
    pub fn binding(&self) -> &VerifierProfileAdoptionStoreAuthorityBindingV1 {
        &self.binding
    }

    pub fn binding_id(&self) -> VerifierProfileAdoptionStoreAuthorityBindingIdV1 {
        self.binding.id()
    }

    #[cfg(test)]
    pub(crate) fn provision_for_test(
        binding: VerifierProfileAdoptionStoreAuthorityBindingV1,
        store: S,
    ) -> Result<Self, VerifierProfileAdoptionStoreAuthorityError> {
        binding.validate()?;
        Ok(Self { binding, store })
    }
}

/// Content identity of one store-authority-qualified fresh CAS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StoreAuthorityQualifiedVerifierProfileAdoptionCasIdV1([u8; 32]);

impl StoreAuthorityQualifiedVerifierProfileAdoptionCasIdV1 {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Historical CAS evidence produced only by executing through a provisioned
/// trusted-store handle.
///
/// This value intentionally implements neither `Serialize`, `Deserialize`, nor
/// `Clone`. It is not a committed adoption receipt and does not prove the adoption
/// remains current.
#[derive(Debug, PartialEq, Eq)]
pub struct StoreAuthorityQualifiedVerifierProfileAdoptionCasV1 {
    observed_cas: StoreObservedVerifierProfileAdoptionCasV1,
    store_authority_binding: VerifierProfileAdoptionStoreAuthorityBindingV1,
    qualified_id: StoreAuthorityQualifiedVerifierProfileAdoptionCasIdV1,
}

impl StoreAuthorityQualifiedVerifierProfileAdoptionCasV1 {
    pub fn id(&self) -> StoreAuthorityQualifiedVerifierProfileAdoptionCasIdV1 {
        self.qualified_id
    }

    pub fn observed_cas(&self) -> &StoreObservedVerifierProfileAdoptionCasV1 {
        &self.observed_cas
    }

    pub fn store_authority_binding(&self) -> &VerifierProfileAdoptionStoreAuthorityBindingV1 {
        &self.store_authority_binding
    }

    pub fn store_authority_binding_id(&self) -> VerifierProfileAdoptionStoreAuthorityBindingIdV1 {
        self.store_authority_binding.id()
    }
}

/// Execute one exact verifier-adoption CAS through the already provisioned trusted
/// store handle and bind the resulting historical observation to that store
/// authority.
///
/// Registry identity/epoch compatibility is checked before store I/O. The lower
/// store layer then independently revalidates the claim and every returned CAS
/// observation field. Possessing only a serializable binding descriptor or only a
/// caller-selected store adapter is insufficient to call this function.
pub fn perform_trusted_verifier_profile_adoption_registry_cas<S>(
    trusted_store: &mut TrustedVerifierProfileAdoptionRegistryStoreV1<S>,
    claim: &VerifierProfileAdoptionRegistryWriteClaimV1,
) -> Result<
    StoreAuthorityQualifiedVerifierProfileAdoptionCasV1,
    TrustedVerifierProfileAdoptionRegistryCasError<S::Error>,
>
where
    S: VerifierProfileAdoptionAtomicRegistryStoreV1,
{
    trusted_store.binding.validate_against_claim(claim)?;
    let observed_cas = perform_verifier_profile_adoption_registry_cas(&mut trusted_store.store, claim)?;
    let qualified_id = StoreAuthorityQualifiedVerifierProfileAdoptionCasIdV1(domain_hash(
        STORE_AUTHORITY_QUALIFIED_CAS_DOMAIN,
        &[
            observed_cas.id().as_bytes().as_slice(),
            trusted_store.binding.id().as_bytes().as_slice(),
        ]
        .concat(),
    ));

    Ok(StoreAuthorityQualifiedVerifierProfileAdoptionCasV1 {
        observed_cas,
        store_authority_binding: trusted_store.binding.clone(),
        qualified_id,
    })
}

#[derive(Debug, Error)]
pub enum TrustedVerifierProfileAdoptionRegistryCasError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Authority(#[from] VerifierProfileAdoptionStoreAuthorityError),
    #[error(transparent)]
    Cas(#[from] VerifierProfileAdoptionRegistryCasError<E>),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileAdoptionStoreAuthorityError {
    #[error("unsupported verifier-profile adoption store-authority schema: {0}")]
    UnsupportedSchema(String),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("registry epoch must be non-zero")]
    ZeroRegistryEpoch,
    #[error("store instance binding digest must be non-zero")]
    ZeroStoreInstanceBindingDigest,
    #[error("store configuration digest must be non-zero")]
    ZeroStoreConfigurationDigest,
    #[error("store trust-root digest must be non-zero")]
    ZeroStoreTrustRootDigest,
    #[error("store provisioning epoch must be non-zero")]
    ZeroProvisioningEpoch,
    #[error("stored store-authority binding identity does not match canonical fields")]
    BindingIdentityMismatch,
    #[error("store-authority binding names a different registry")]
    RegistryIdMismatch,
    #[error("store-authority binding names a different registry epoch")]
    RegistryEpochMismatch,
    #[error(transparent)]
    RegistryWrite(#[from] VerifierProfileAdoptionRegistryWriteError),
}

fn checked_text(
    field: &'static str,
    value: String,
) -> Result<String, VerifierProfileAdoptionStoreAuthorityError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(VerifierProfileAdoptionStoreAuthorityError::BlankText { field });
    }
    if trimmed.len() > 1024 {
        return Err(VerifierProfileAdoptionStoreAuthorityError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(VerifierProfileAdoptionStoreAuthorityError::ControlCharacters {
            field,
        });
    }
    Ok(trimmed.to_owned())
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn put_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::{
        EvidenceClass, GrantBoundVerifierProfileAdoptionCommitPreconditionsV1,
        RootBoundPolicyCheckedVerifierProfileAdoptionV1,
        TimeBoundVerifierProfileAdoptionCommitPreconditionsV1,
        VerifierAdoptionAuthorityGrantV1, VerifierAdoptionScopeV1,
        VerifierProfileAdoptionAdmissionPolicyV1,
        VerifierProfileAdoptionAuthorityRootSnapshotV1,
        VerifierProfileAdoptionClockObservationV1,
        VerifierProfileAdoptionCommitEvidenceV1, VerifierProfileAdoptionCommitPreconditionsV1,
        VerifierProfileAdoptionHeadV1, VerifierProfileAdoptionRegistryCasObservationV1,
        VerifierProfileAdoptionRegistrySlotIdV1, VerifierProfileAdoptionRegistryWriteClaimIdV1,
        VerifierProfileAdoptionSubjectV1, VerifierProfileAdoptionTransitionDigest,
        VerifierProfileAdoptionTransitionV1, VerifierProfileV1,
        bind_root_bound_adoption_to_authority_grant,
    };
    use crate::profile_adoption_commit_evidence::VerifierProfileAdoptionCommitEvidenceIdV1;

    #[derive(Debug, Clone)]
    struct TestObservation {
        registry_id: String,
        registry_epoch: u64,
        slot_id: VerifierProfileAdoptionRegistrySlotIdV1,
        write_claim_id: VerifierProfileAdoptionRegistryWriteClaimIdV1,
        commit_evidence_id: VerifierProfileAdoptionCommitEvidenceIdV1,
        predecessor_transition_digest: Option<VerifierProfileAdoptionTransitionDigest>,
        predecessor_commit_evidence_id: Option<VerifierProfileAdoptionCommitEvidenceIdV1>,
        candidate_transition_digest: VerifierProfileAdoptionTransitionDigest,
        candidate_generation: u64,
        store_revision_before: u64,
        store_revision_after: u64,
        transaction_challenge: [u8; 32],
        store_transaction_evidence_digest: [u8; 32],
    }

    impl TestObservation {
        fn exact(claim: &VerifierProfileAdoptionRegistryWriteClaimV1, evidence: u8) -> Self {
            Self {
                registry_id: claim.registry_id().to_owned(),
                registry_epoch: claim.registry_epoch(),
                slot_id: claim.slot_id(),
                write_claim_id: claim.id(),
                commit_evidence_id: claim.commit_evidence_id(),
                predecessor_transition_digest: claim.predecessor_transition_digest(),
                predecessor_commit_evidence_id: claim.predecessor_commit_evidence_id(),
                candidate_transition_digest: claim.candidate_transition_digest(),
                candidate_generation: claim.candidate_generation(),
                store_revision_before: claim.store_revision_before(),
                store_revision_after: claim.store_revision_after(),
                transaction_challenge: claim.transaction_challenge(),
                store_transaction_evidence_digest: [evidence; 32],
            }
        }
    }

    impl VerifierProfileAdoptionRegistryCasObservationV1 for TestObservation {
        fn registry_id(&self) -> &str {
            &self.registry_id
        }
        fn registry_epoch(&self) -> u64 {
            self.registry_epoch
        }
        fn slot_id(&self) -> VerifierProfileAdoptionRegistrySlotIdV1 {
            self.slot_id
        }
        fn write_claim_id(&self) -> VerifierProfileAdoptionRegistryWriteClaimIdV1 {
            self.write_claim_id
        }
        fn commit_evidence_id(&self) -> VerifierProfileAdoptionCommitEvidenceIdV1 {
            self.commit_evidence_id
        }
        fn predecessor_transition_digest(&self) -> Option<VerifierProfileAdoptionTransitionDigest> {
            self.predecessor_transition_digest
        }
        fn predecessor_commit_evidence_id(
            &self,
        ) -> Option<VerifierProfileAdoptionCommitEvidenceIdV1> {
            self.predecessor_commit_evidence_id
        }
        fn candidate_transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest {
            self.candidate_transition_digest
        }
        fn candidate_generation(&self) -> u64 {
            self.candidate_generation
        }
        fn store_revision_before(&self) -> u64 {
            self.store_revision_before
        }
        fn store_revision_after(&self) -> u64 {
            self.store_revision_after
        }
        fn transaction_challenge(&self) -> [u8; 32] {
            self.transaction_challenge
        }
        fn store_transaction_evidence_digest(&self) -> [u8; 32] {
            self.store_transaction_evidence_digest
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Error)]
    #[error("test store failure")]
    struct TestStoreError;

    struct TestStore {
        observation: Result<TestObservation, TestStoreError>,
        calls: Arc<AtomicUsize>,
    }

    impl VerifierProfileAdoptionAtomicRegistryStoreV1 for TestStore {
        type Observation = TestObservation;
        type Error = TestStoreError;

        fn compare_and_swap(
            &mut self,
            _claim: &VerifierProfileAdoptionRegistryWriteClaimV1,
            _canonical_claim_bytes: &[u8],
        ) -> Result<Self::Observation, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.observation.clone()
        }
    }

    fn profile() -> VerifierProfileV1 {
        VerifierProfileV1::new(
            "hardware-verifier-v1",
            [9; 32],
            7,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
    }

    fn write_claim() -> VerifierProfileAdoptionRegistryWriteClaimV1 {
        let profile = profile();
        let root = VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            9,
        )
        .unwrap();
        let subject = VerifierProfileAdoptionSubjectV1::new(
            "adopt-1",
            root.authority_subject(),
            root.authority_root_id(),
            root.authority_root_digest(),
            &profile,
            1,
            1_000,
            2_000,
            EvidenceClass::DifferentiallyVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let checked = VerifierProfileAdoptionAdmissionPolicyV1::new(
            root.authority_subject(),
            root.authority_root_id(),
            root.authority_root_digest(),
            profile.profile_name(),
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap()
        .check(1_500, &transition, &profile, None)
        .unwrap();
        let root_bound =
            RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(checked, root.clone()).unwrap();
        let grant = VerifierAdoptionAuthorityGrantV1::new(
            "grant-slot-1",
            root,
            profile.profile_name(),
            3,
            EvidenceClass::DifferentiallyVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();
        let grant_bound =
            bind_root_bound_adoption_to_authority_grant(root_bound.clone(), &grant, None).unwrap();
        let commit =
            VerifierProfileAdoptionCommitPreconditionsV1::from_root_bound(root_bound).unwrap();
        let clock = VerifierProfileAdoptionClockObservationV1::new(
            "trusted-clock-1",
            4,
            1_490,
            1_510,
        )
        .unwrap();
        let time_bound =
            TimeBoundVerifierProfileAdoptionCommitPreconditionsV1::new(commit, &clock, 50).unwrap();
        let preconditions = GrantBoundVerifierProfileAdoptionCommitPreconditionsV1::new(
            grant_bound,
            time_bound,
        )
        .unwrap();
        let evidence = VerifierProfileAdoptionCommitEvidenceV1::project_from_commit_preconditions(
            &preconditions,
            None,
        )
        .unwrap();
        VerifierProfileAdoptionRegistryWriteClaimV1::new(
            "continuity-registry-1",
            5,
            &evidence,
            None,
            40,
            41,
            [0x77; 32],
        )
        .unwrap()
    }

    fn binding(
        registry_id: &str,
        registry_epoch: u64,
        trust_root: u8,
        provisioning_epoch: u64,
    ) -> VerifierProfileAdoptionStoreAuthorityBindingV1 {
        VerifierProfileAdoptionStoreAuthorityBindingV1::new(
            "deployment:test",
            registry_id,
            registry_epoch,
            "store:primary",
            [0x31; 32],
            [0x32; 32],
            [trust_root; 32],
            provisioning_epoch,
        )
        .unwrap()
    }

    fn store_for(
        claim: &VerifierProfileAdoptionRegistryWriteClaimV1,
    ) -> (TestStore, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            TestStore {
                observation: Ok(TestObservation::exact(claim, 0x91)),
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }

    #[test]
    fn exact_provisioned_store_path_qualifies_one_historical_cas() {
        let claim = write_claim();
        let descriptor = binding(claim.registry_id(), claim.registry_epoch(), 0x33, 7);
        let expected_binding_id = descriptor.id();
        let (store, calls) = store_for(&claim);
        let mut trusted =
            TrustedVerifierProfileAdoptionRegistryStoreV1::provision_for_test(descriptor, store)
                .unwrap();
        let qualified =
            perform_trusted_verifier_profile_adoption_registry_cas(&mut trusted, &claim).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(qualified.store_authority_binding_id(), expected_binding_id);
        assert_eq!(qualified.observed_cas().write_claim_id(), claim.id());
        assert_eq!(
            qualified.observed_cas().store_transaction_evidence_digest(),
            [0x91; 32]
        );
    }

    #[test]
    fn registry_mismatch_fails_before_store_io() {
        let claim = write_claim();
        let descriptor = binding("other-registry", claim.registry_epoch(), 0x33, 7);
        let (store, calls) = store_for(&claim);
        let mut trusted =
            TrustedVerifierProfileAdoptionRegistryStoreV1::provision_for_test(descriptor, store)
                .unwrap();

        assert!(matches!(
            perform_trusted_verifier_profile_adoption_registry_cas(&mut trusted, &claim),
            Err(TrustedVerifierProfileAdoptionRegistryCasError::Authority(
                VerifierProfileAdoptionStoreAuthorityError::RegistryIdMismatch
            ))
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn registry_epoch_mismatch_fails_before_store_io() {
        let claim = write_claim();
        let descriptor = binding(claim.registry_id(), claim.registry_epoch() + 1, 0x33, 7);
        let (store, calls) = store_for(&claim);
        let mut trusted =
            TrustedVerifierProfileAdoptionRegistryStoreV1::provision_for_test(descriptor, store)
                .unwrap();

        assert!(matches!(
            perform_trusted_verifier_profile_adoption_registry_cas(&mut trusted, &claim),
            Err(TrustedVerifierProfileAdoptionRegistryCasError::Authority(
                VerifierProfileAdoptionStoreAuthorityError::RegistryEpochMismatch
            ))
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn trust_root_and_provisioning_rotation_change_binding_identity() {
        let a = binding("continuity-registry-1", 5, 0x33, 7);
        let changed_root = binding("continuity-registry-1", 5, 0x34, 7);
        let changed_provisioning = binding("continuity-registry-1", 5, 0x33, 8);

        assert_ne!(a.id(), changed_root.id());
        assert_ne!(a.id(), changed_provisioning.id());
    }

    #[test]
    fn zero_store_trust_material_fails_closed() {
        assert_eq!(
            VerifierProfileAdoptionStoreAuthorityBindingV1::new(
                "deployment:test",
                "continuity-registry-1",
                5,
                "store:primary",
                [0x31; 32],
                [0x32; 32],
                [0; 32],
                7,
            )
            .unwrap_err(),
            VerifierProfileAdoptionStoreAuthorityError::ZeroStoreTrustRootDigest
        );
    }
}
