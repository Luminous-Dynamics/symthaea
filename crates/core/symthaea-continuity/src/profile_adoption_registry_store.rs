// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Store-observed atomic persistence for verifier-profile adoption.
//!
//! A registry-write claim describes an intended compare-and-swap. This module
//! crosses exactly one additional boundary: the application-selected persistence
//! adapter is invoked, and core independently checks the adapter's returned
//! observation against every authority-relevant field of the exact claim.
//!
//! Core theorem:
//!
//! `RegistryWriteClaim != StoreObservedCas != CommittedReceipt != CurrentAuthority`.
//!
//! The opaque success witness is historical evidence only. It does not prove the
//! selected store adapter is honest, does not authenticate the verifier adoption,
//! and does not establish that the adoption remains current. Adapter selection and
//! implementation are deployment trust boundaries.

use std::error::Error as StdError;

use thiserror::Error;

use crate::profile_adoption::VerifierProfileAdoptionTransitionDigest;
use crate::profile_adoption_commit_evidence::VerifierProfileAdoptionCommitEvidenceIdV1;
use crate::profile_adoption_registry_write::{
    VerifierProfileAdoptionRegistrySlotIdV1, VerifierProfileAdoptionRegistryWriteClaimIdV1,
    VerifierProfileAdoptionRegistryWriteClaimV1, VerifierProfileAdoptionRegistryWriteError,
};

const STORE_OBSERVED_CAS_DOMAIN: &[u8] =
    b"symthaea.continuity.verifier-profile-adoption.store-observed-cas.v1\0";

/// Observation returned by an application-selected atomic registry adapter after
/// a successful fresh compare-and-swap.
///
/// The trait deliberately exposes facts rather than a self-authenticating `matches`
/// method. Core compares each field independently with the exact write claim.
/// Implementations must return success only after the write is durably accepted by
/// their persistence boundary. A lost/uncertain acknowledgement is not success in
/// v1 and must remain an adapter error until a separate recovery theorem exists.
pub trait VerifierProfileAdoptionRegistryCasObservationV1 {
    fn registry_id(&self) -> &str;
    fn registry_epoch(&self) -> u64;
    fn slot_id(&self) -> VerifierProfileAdoptionRegistrySlotIdV1;
    fn write_claim_id(&self) -> VerifierProfileAdoptionRegistryWriteClaimIdV1;
    fn commit_evidence_id(&self) -> VerifierProfileAdoptionCommitEvidenceIdV1;
    fn predecessor_transition_digest(&self) -> Option<VerifierProfileAdoptionTransitionDigest>;
    fn predecessor_commit_evidence_id(
        &self,
    ) -> Option<VerifierProfileAdoptionCommitEvidenceIdV1>;
    fn candidate_transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest;
    fn candidate_generation(&self) -> u64;
    fn store_revision_before(&self) -> u64;
    fn store_revision_after(&self) -> u64;
    fn transaction_challenge(&self) -> [u8; 32];

    /// Digest naming adapter-owned evidence for the exact successful store
    /// transaction (for example, a durable transaction/journal record).
    ///
    /// Core treats this as opaque evidence identity. It does not infer which
    /// storage technology, durability mechanism, or external attestation produced
    /// the bytes named by this digest.
    fn store_transaction_evidence_digest(&self) -> [u8; 32];
}

/// Application-selected persistence adapter for one exact fresh atomic CAS.
///
/// `canonical_claim_bytes` are computed by this crate from the already validated
/// claim and are supplied so adapters do not need an independent canonical encoder.
/// A successful return means the adapter asserts that it atomically compared the
/// exact predecessor/revision state and durably installed the exact candidate.
/// Core then rechecks the returned observation before minting an opaque witness.
pub trait VerifierProfileAdoptionAtomicRegistryStoreV1 {
    type Observation: VerifierProfileAdoptionRegistryCasObservationV1;
    type Error: StdError + Send + Sync + 'static;

    fn compare_and_swap(
        &mut self,
        claim: &VerifierProfileAdoptionRegistryWriteClaimV1,
        canonical_claim_bytes: &[u8],
    ) -> Result<Self::Observation, Self::Error>;
}

/// Content identity of one store-observed successful fresh CAS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StoreObservedVerifierProfileAdoptionCasIdV1([u8; 32]);

impl StoreObservedVerifierProfileAdoptionCasIdV1 {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Opaque historical witness that the selected store adapter reported a successful
/// fresh CAS whose observed facts exactly matched one canonical registry-write
/// claim.
///
/// This value intentionally implements neither `Serialize` nor `Deserialize` and
/// has no public constructor. The only production source is
/// [`perform_verifier_profile_adoption_registry_cas`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreObservedVerifierProfileAdoptionCasV1 {
    write_claim: VerifierProfileAdoptionRegistryWriteClaimV1,
    store_transaction_evidence_digest: [u8; 32],
    observation_id: StoreObservedVerifierProfileAdoptionCasIdV1,
}

impl StoreObservedVerifierProfileAdoptionCasV1 {
    pub fn id(&self) -> StoreObservedVerifierProfileAdoptionCasIdV1 {
        self.observation_id
    }

    pub fn write_claim(&self) -> &VerifierProfileAdoptionRegistryWriteClaimV1 {
        &self.write_claim
    }

    pub fn write_claim_id(&self) -> VerifierProfileAdoptionRegistryWriteClaimIdV1 {
        self.write_claim.id()
    }

    pub fn commit_evidence_id(&self) -> VerifierProfileAdoptionCommitEvidenceIdV1 {
        self.write_claim.commit_evidence_id()
    }

    pub fn candidate_transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest {
        self.write_claim.candidate_transition_digest()
    }

    pub fn store_transaction_evidence_digest(&self) -> [u8; 32] {
        self.store_transaction_evidence_digest
    }
}

/// Execute one fresh atomic registry CAS through the selected store adapter and
/// retain success only after independently checking the returned observation.
///
/// Malformed/noncanonical claims fail before store I/O. The exact canonical claim
/// bytes supplied to the adapter are the same bytes from which the claim identity
/// is derived. A successful adapter call still cannot bypass field-by-field
/// observation validation.
pub fn perform_verifier_profile_adoption_registry_cas<S>(
    store: &mut S,
    claim: &VerifierProfileAdoptionRegistryWriteClaimV1,
) -> Result<StoreObservedVerifierProfileAdoptionCasV1, VerifierProfileAdoptionRegistryCasError<S::Error>>
where
    S: VerifierProfileAdoptionAtomicRegistryStoreV1,
{
    claim.validate()?;
    let canonical_claim_bytes = claim.canonical_bytes()?;
    let observation = store
        .compare_and_swap(claim, &canonical_claim_bytes)
        .map_err(VerifierProfileAdoptionRegistryCasError::Store)?;
    let store_transaction_evidence_digest =
        validate_store_observation(claim, &observation)?;
    let observation_id = StoreObservedVerifierProfileAdoptionCasIdV1(hash_store_observation(
        claim.id(),
        store_transaction_evidence_digest,
    ));

    Ok(StoreObservedVerifierProfileAdoptionCasV1 {
        write_claim: claim.clone(),
        store_transaction_evidence_digest,
        observation_id,
    })
}

fn validate_store_observation<O>(
    claim: &VerifierProfileAdoptionRegistryWriteClaimV1,
    observation: &O,
) -> Result<[u8; 32], VerifierProfileAdoptionRegistryCasObservationError>
where
    O: VerifierProfileAdoptionRegistryCasObservationV1,
{
    if observation.registry_id() != claim.registry_id() {
        return Err(VerifierProfileAdoptionRegistryCasObservationError::RegistryIdMismatch);
    }
    if observation.registry_epoch() != claim.registry_epoch() {
        return Err(VerifierProfileAdoptionRegistryCasObservationError::RegistryEpochMismatch);
    }
    if observation.slot_id() != claim.slot_id() {
        return Err(VerifierProfileAdoptionRegistryCasObservationError::RegistrySlotMismatch);
    }
    if observation.write_claim_id() != claim.id() {
        return Err(VerifierProfileAdoptionRegistryCasObservationError::WriteClaimMismatch);
    }
    if observation.commit_evidence_id() != claim.commit_evidence_id() {
        return Err(VerifierProfileAdoptionRegistryCasObservationError::CommitEvidenceMismatch);
    }
    if observation.predecessor_transition_digest() != claim.predecessor_transition_digest() {
        return Err(
            VerifierProfileAdoptionRegistryCasObservationError::PredecessorTransitionMismatch,
        );
    }
    if observation.predecessor_commit_evidence_id() != claim.predecessor_commit_evidence_id() {
        return Err(
            VerifierProfileAdoptionRegistryCasObservationError::PredecessorCommitEvidenceMismatch,
        );
    }
    if observation.candidate_transition_digest() != claim.candidate_transition_digest() {
        return Err(
            VerifierProfileAdoptionRegistryCasObservationError::CandidateTransitionMismatch,
        );
    }
    if observation.candidate_generation() != claim.candidate_generation() {
        return Err(VerifierProfileAdoptionRegistryCasObservationError::CandidateGenerationMismatch);
    }
    if observation.store_revision_before() != claim.store_revision_before() {
        return Err(VerifierProfileAdoptionRegistryCasObservationError::StoreRevisionBeforeMismatch);
    }
    if observation.store_revision_after() != claim.store_revision_after() {
        return Err(VerifierProfileAdoptionRegistryCasObservationError::StoreRevisionAfterMismatch);
    }
    if observation.transaction_challenge() != claim.transaction_challenge() {
        return Err(VerifierProfileAdoptionRegistryCasObservationError::TransactionChallengeMismatch);
    }

    let evidence_digest = observation.store_transaction_evidence_digest();
    if evidence_digest == [0; 32] {
        return Err(
            VerifierProfileAdoptionRegistryCasObservationError::ZeroStoreTransactionEvidenceDigest,
        );
    }
    Ok(evidence_digest)
}

fn hash_store_observation(
    write_claim_id: VerifierProfileAdoptionRegistryWriteClaimIdV1,
    store_transaction_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(STORE_OBSERVED_CAS_DOMAIN);
    hasher.update(write_claim_id.as_bytes());
    hasher.update(&store_transaction_evidence_digest);
    *hasher.finalize().as_bytes()
}

#[derive(Debug, Error)]
pub enum VerifierProfileAdoptionRegistryCasError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Claim(#[from] VerifierProfileAdoptionRegistryWriteError),
    #[error("verifier-profile adoption registry CAS failed at the selected store boundary: {0}")]
    Store(#[source] E),
    #[error(transparent)]
    Observation(#[from] VerifierProfileAdoptionRegistryCasObservationError),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileAdoptionRegistryCasObservationError {
    #[error("store observation names a different registry")]
    RegistryIdMismatch,
    #[error("store observation names a different registry epoch")]
    RegistryEpochMismatch,
    #[error("store observation names a different logical registry slot")]
    RegistrySlotMismatch,
    #[error("store observation names a different registry-write claim")]
    WriteClaimMismatch,
    #[error("store observation names different verifier-adoption commit evidence")]
    CommitEvidenceMismatch,
    #[error("store observation names a different predecessor transition")]
    PredecessorTransitionMismatch,
    #[error("store observation names different predecessor commit evidence")]
    PredecessorCommitEvidenceMismatch,
    #[error("store observation names a different candidate transition")]
    CandidateTransitionMismatch,
    #[error("store observation names a different candidate generation")]
    CandidateGenerationMismatch,
    #[error("store observation saw a different pre-CAS store revision")]
    StoreRevisionBeforeMismatch,
    #[error("store observation saw a different post-CAS store revision")]
    StoreRevisionAfterMismatch,
    #[error("store observation belongs to a different transaction challenge")]
    TransactionChallengeMismatch,
    #[error("store transaction evidence digest must be non-zero")]
    ZeroStoreTransactionEvidenceDigest,
}

#[cfg(test)]
mod tests {
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
        VerifierProfileAdoptionHeadV1, VerifierProfileAdoptionSubjectV1,
        VerifierProfileAdoptionTransitionV1, VerifierProfileV1,
        bind_root_bound_adoption_to_authority_grant,
    };

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
        fn registry_id(&self) -> &str { &self.registry_id }
        fn registry_epoch(&self) -> u64 { self.registry_epoch }
        fn slot_id(&self) -> VerifierProfileAdoptionRegistrySlotIdV1 { self.slot_id }
        fn write_claim_id(&self) -> VerifierProfileAdoptionRegistryWriteClaimIdV1 { self.write_claim_id }
        fn commit_evidence_id(&self) -> VerifierProfileAdoptionCommitEvidenceIdV1 { self.commit_evidence_id }
        fn predecessor_transition_digest(&self) -> Option<VerifierProfileAdoptionTransitionDigest> { self.predecessor_transition_digest }
        fn predecessor_commit_evidence_id(&self) -> Option<VerifierProfileAdoptionCommitEvidenceIdV1> { self.predecessor_commit_evidence_id }
        fn candidate_transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest { self.candidate_transition_digest }
        fn candidate_generation(&self) -> u64 { self.candidate_generation }
        fn store_revision_before(&self) -> u64 { self.store_revision_before }
        fn store_revision_after(&self) -> u64 { self.store_revision_after }
        fn transaction_challenge(&self) -> [u8; 32] { self.transaction_challenge }
        fn store_transaction_evidence_digest(&self) -> [u8; 32] { self.store_transaction_evidence_digest }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Error)]
    #[error("test store failure")]
    struct TestStoreError;

    struct TestStore {
        observation: Result<TestObservation, TestStoreError>,
        saw_exact_canonical_claim: bool,
    }

    impl VerifierProfileAdoptionAtomicRegistryStoreV1 for TestStore {
        type Observation = TestObservation;
        type Error = TestStoreError;

        fn compare_and_swap(
            &mut self,
            claim: &VerifierProfileAdoptionRegistryWriteClaimV1,
            canonical_claim_bytes: &[u8],
        ) -> Result<Self::Observation, Self::Error> {
            self.saw_exact_canonical_claim =
                blake3::hash(canonical_claim_bytes).as_bytes() == claim.id().as_bytes();
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

    fn store_for(claim: &VerifierProfileAdoptionRegistryWriteClaimV1, evidence: u8) -> TestStore {
        TestStore {
            observation: Ok(TestObservation::exact(claim, evidence)),
            saw_exact_canonical_claim: false,
        }
    }

    #[test]
    fn exact_fresh_cas_returns_non_authorizing_opaque_witness() {
        let claim = write_claim();
        let mut store = store_for(&claim, 0x91);
        let observed = perform_verifier_profile_adoption_registry_cas(&mut store, &claim).unwrap();
        assert!(store.saw_exact_canonical_claim);
        assert_eq!(observed.write_claim_id(), claim.id());
        assert_eq!(observed.commit_evidence_id(), claim.commit_evidence_id());
        assert_eq!(observed.store_transaction_evidence_digest(), [0x91; 32]);
    }

    #[test]
    fn store_transaction_evidence_changes_observation_identity() {
        let claim = write_claim();
        let a = perform_verifier_profile_adoption_registry_cas(&mut store_for(&claim, 0x91), &claim)
            .unwrap();
        let b = perform_verifier_profile_adoption_registry_cas(&mut store_for(&claim, 0x92), &claim)
            .unwrap();
        assert_ne!(a.id(), b.id());
        assert_eq!(a.write_claim_id(), b.write_claim_id());
    }

    #[test]
    fn changed_challenge_in_store_observation_fails_closed() {
        let claim = write_claim();
        let mut observation = TestObservation::exact(&claim, 0x91);
        observation.transaction_challenge[0] ^= 1;
        let mut store = TestStore { observation: Ok(observation), saw_exact_canonical_claim: false };
        assert!(matches!(
            perform_verifier_profile_adoption_registry_cas(&mut store, &claim),
            Err(VerifierProfileAdoptionRegistryCasError::Observation(
                VerifierProfileAdoptionRegistryCasObservationError::TransactionChallengeMismatch
            ))
        ));
    }

    #[test]
    fn changed_predecessor_state_in_store_observation_fails_closed() {
        let claim = write_claim();
        let mut observation = TestObservation::exact(&claim, 0x91);
        observation.predecessor_commit_evidence_id = Some(claim.commit_evidence_id());
        let mut store = TestStore { observation: Ok(observation), saw_exact_canonical_claim: false };
        assert!(matches!(
            perform_verifier_profile_adoption_registry_cas(&mut store, &claim),
            Err(VerifierProfileAdoptionRegistryCasError::Observation(
                VerifierProfileAdoptionRegistryCasObservationError::PredecessorCommitEvidenceMismatch
            ))
        ));
    }

    #[test]
    fn changed_store_revision_in_observation_fails_closed() {
        let claim = write_claim();
        let mut observation = TestObservation::exact(&claim, 0x91);
        observation.store_revision_after += 1;
        let mut store = TestStore { observation: Ok(observation), saw_exact_canonical_claim: false };
        assert!(matches!(
            perform_verifier_profile_adoption_registry_cas(&mut store, &claim),
            Err(VerifierProfileAdoptionRegistryCasError::Observation(
                VerifierProfileAdoptionRegistryCasObservationError::StoreRevisionAfterMismatch
            ))
        ));
    }

    #[test]
    fn zero_store_transaction_evidence_fails_closed() {
        let claim = write_claim();
        let mut store = store_for(&claim, 0);
        assert!(matches!(
            perform_verifier_profile_adoption_registry_cas(&mut store, &claim),
            Err(VerifierProfileAdoptionRegistryCasError::Observation(
                VerifierProfileAdoptionRegistryCasObservationError::ZeroStoreTransactionEvidenceDigest
            ))
        ));
    }

    #[test]
    fn store_error_cannot_mint_observation_witness() {
        let claim = write_claim();
        let mut store = TestStore {
            observation: Err(TestStoreError),
            saw_exact_canonical_claim: false,
        };
        assert!(matches!(
            perform_verifier_profile_adoption_registry_cas(&mut store, &claim),
            Err(VerifierProfileAdoptionRegistryCasError::Store(TestStoreError))
        ));
    }
}
