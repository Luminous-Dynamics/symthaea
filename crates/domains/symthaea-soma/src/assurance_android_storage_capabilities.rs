// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDSTORAGECAPS-535: qualify Android storage claims before 534.
//!
//! Android exposes useful primitives, but their guarantees are narrower than the
//! two-plane durability theorem needs. `AtomicFile` provides a synced/committed
//! atomic-file pattern but does not provide inter-thread/inter-process locking.
//! Android Keystore can protect signing keys in software, a TEE, or StrongBox.
//! KeyMint rollback resistance protects *deleted rollback-resistant keys* from
//! restoration; it does not turn an arbitrary signed app checkpoint into a
//! monotonic anti-rollback register.
//!
//! This module prevents those distinctions from being collapsed. It qualifies a
//! journal adapter and an anchor adapter separately, then gates entry into
//! QUAL-ANDROIDPOLICYPERSIST-534. Standard app-file and Keystore-signed anchors are
//! never allowed to claim monotonic checkpoint advancement merely because their
//! bytes are atomic or their signatures are hardware-backed.

use core::fmt;

use crate::assurance_android_policy_codec::AndroidTouchPolicyDecodeLimits;
use crate::assurance_android_policy_durability::{
    certify_android_policy_durable_state, AndroidTouchPolicyAnchorObservation,
    AndroidTouchPolicyDurabilityError, AndroidTouchPolicyDurableState,
    AndroidTouchPolicyJournalDurabilityObservation, AndroidTouchPolicyStorageDomainId,
    AndroidTouchPolicyStorageEvidenceId,
};

const JOURNAL_CAP_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-journal-storage-capability\0";
const ANCHOR_CAP_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-anchor-storage-capability\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub [u8; 32]);

        impl $name {
            pub const ZERO: Self = Self([0; 32]);
            pub const fn as_bytes(&self) -> &[u8; 32] {
                &self.0
            }
            pub fn is_zero(&self) -> bool {
                self.0 == [0; 32]
            }
        }
    };
}

digest_id!(AndroidStorageAdapterDefinitionId);
digest_id!(AndroidJournalStorageCapabilityCertificateId);
digest_id!(AndroidAnchorStorageCapabilityCertificateId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AndroidJournalBackendKind {
    FrameworkAtomicFile = 1,
    AndroidXAtomicFile = 2,
    QualifiedCustomAtomicStore = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AndroidAnchorBackendKind {
    /// Ordinary app-controlled file/data store. Never monotonic by itself.
    ApplicationStorage = 1,
    /// Checkpoint authenticated by an Android Keystore key. Signature validity is
    /// not checkpoint monotonicity.
    KeystoreSigned = 2,
    /// Checkpoint authenticated by a StrongBox-backed key. Hardware key protection
    /// is still not checkpoint monotonicity.
    StrongBoxSigned = 3,
    /// A separately qualified witness outside the journal rollback domain.
    ExternalMonotonicWitness = 4,
    /// A separately qualified device primitive that truly provides monotonic state.
    QualifiedHardwareMonotonicStore = 5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AndroidKeySecurityLevel {
    Unknown = 0,
    Software = 1,
    TrustedEnvironment = 2,
    StrongBox = 3,
}

/// Static/qualified capability statement for the journal backend.
///
/// `AtomicFile` does not itself establish `mutual_exclusion_evidence_id`; the
/// adapter must prove whatever single-writer/locking discipline surrounds it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidJournalStorageCapabilityProfile {
    pub adapter_definition_id: AndroidStorageAdapterDefinitionId,
    pub storage_domain_id: AndroidTouchPolicyStorageDomainId,
    pub backend_kind: AndroidJournalBackendKind,
    pub synced_commit_semantics: bool,
    pub atomic_publish_semantics: bool,
    pub metadata_durability_established: bool,
    pub exact_readback_established: bool,
    pub mutual_exclusion_evidence_id: AndroidTouchPolicyStorageEvidenceId,
    pub metadata_durability_evidence_id: AndroidTouchPolicyStorageEvidenceId,
    pub adapter_qualification_evidence_id: AndroidTouchPolicyStorageEvidenceId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidJournalStorageCapabilityCertificate {
    pub certificate_id: AndroidJournalStorageCapabilityCertificateId,
    pub adapter_definition_id: AndroidStorageAdapterDefinitionId,
    pub storage_domain_id: AndroidTouchPolicyStorageDomainId,
    pub backend_kind: AndroidJournalBackendKind,
}

/// Static/qualified capability statement for the anchor backend.
///
/// `keymint_rollback_resistant_key` is deliberately descriptive only. It never
/// discharges `monotonic_checkpoint_advance` for Keystore/StrongBox signing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidAnchorStorageCapabilityProfile {
    pub adapter_definition_id: AndroidStorageAdapterDefinitionId,
    pub anchor_domain_id: AndroidTouchPolicyStorageDomainId,
    pub independent_from_journal_domain_id: AndroidTouchPolicyStorageDomainId,
    pub backend_kind: AndroidAnchorBackendKind,
    pub key_security_level: AndroidKeySecurityLevel,
    pub checkpoint_authenticity_established: bool,
    pub keymint_rollback_resistant_key: bool,
    pub monotonic_checkpoint_advance: bool,
    pub exact_readback_established: bool,
    pub independence_evidence_id: AndroidTouchPolicyStorageEvidenceId,
    pub monotonicity_evidence_id: AndroidTouchPolicyStorageEvidenceId,
    pub adapter_qualification_evidence_id: AndroidTouchPolicyStorageEvidenceId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidAnchorStorageCapabilityCertificate {
    pub certificate_id: AndroidAnchorStorageCapabilityCertificateId,
    pub adapter_definition_id: AndroidStorageAdapterDefinitionId,
    pub anchor_domain_id: AndroidTouchPolicyStorageDomainId,
    pub independent_from_journal_domain_id: AndroidTouchPolicyStorageDomainId,
    pub backend_kind: AndroidAnchorBackendKind,
    pub key_security_level: AndroidKeySecurityLevel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidStorageCapabilityError {
    ZeroAdapterDefinition,
    ZeroStorageDomain,
    SameStorageAndAnchorDomain,
    MissingAdapterQualification,
    MissingMutualExclusionEvidence,
    SyncedCommitNotEstablished,
    AtomicPublishNotEstablished,
    MetadataDurabilityNotEstablished,
    ExactReadbackNotEstablished,
    MissingIndependenceEvidence,
    CheckpointAuthenticityNotEstablished,
    CheckpointMonotonicityNotEstablished,
    MissingMonotonicityEvidence,
    UnsupportedMonotonicAnchorClaim,
    StrongBoxSecurityLevelMismatch,
    JournalCapabilityMismatch,
    AnchorCapabilityMismatch,
    ReusedOperationEvidence,
    Durability(AndroidTouchPolicyDurabilityError),
}

impl fmt::Display for AndroidStorageCapabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AndroidStorageCapabilityError {}

impl From<AndroidTouchPolicyDurabilityError> for AndroidStorageCapabilityError {
    fn from(value: AndroidTouchPolicyDurabilityError) -> Self {
        Self::Durability(value)
    }
}

fn nonzero(
    id: AndroidTouchPolicyStorageEvidenceId,
    error: AndroidStorageCapabilityError,
) -> Result<(), AndroidStorageCapabilityError> {
    if id.is_zero() {
        return Err(error);
    }
    Ok(())
}

fn all_distinct(ids: &[AndroidTouchPolicyStorageEvidenceId]) -> bool {
    for (index, left) in ids.iter().enumerate() {
        for right in ids.iter().skip(index + 1) {
            if left == right {
                return false;
            }
        }
    }
    true
}

impl AndroidJournalStorageCapabilityProfile {
    pub fn certify(
        &self,
    ) -> Result<AndroidJournalStorageCapabilityCertificate, AndroidStorageCapabilityError> {
        if self.adapter_definition_id.is_zero() {
            return Err(AndroidStorageCapabilityError::ZeroAdapterDefinition);
        }
        if self.storage_domain_id.is_zero() {
            return Err(AndroidStorageCapabilityError::ZeroStorageDomain);
        }
        nonzero(
            self.adapter_qualification_evidence_id,
            AndroidStorageCapabilityError::MissingAdapterQualification,
        )?;
        nonzero(
            self.mutual_exclusion_evidence_id,
            AndroidStorageCapabilityError::MissingMutualExclusionEvidence,
        )?;
        if !self.synced_commit_semantics {
            return Err(AndroidStorageCapabilityError::SyncedCommitNotEstablished);
        }
        if !self.atomic_publish_semantics {
            return Err(AndroidStorageCapabilityError::AtomicPublishNotEstablished);
        }
        if !self.metadata_durability_established {
            return Err(AndroidStorageCapabilityError::MetadataDurabilityNotEstablished);
        }
        nonzero(
            self.metadata_durability_evidence_id,
            AndroidStorageCapabilityError::MetadataDurabilityNotEstablished,
        )?;
        if !self.exact_readback_established {
            return Err(AndroidStorageCapabilityError::ExactReadbackNotEstablished);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(JOURNAL_CAP_DOMAIN);
        hasher.update(self.adapter_definition_id.as_bytes());
        hasher.update(self.storage_domain_id.as_bytes());
        hasher.update(&[self.backend_kind as u8]);
        hasher.update(self.mutual_exclusion_evidence_id.as_bytes());
        hasher.update(self.metadata_durability_evidence_id.as_bytes());
        hasher.update(self.adapter_qualification_evidence_id.as_bytes());
        let certificate_id =
            AndroidJournalStorageCapabilityCertificateId(*hasher.finalize().as_bytes());
        Ok(AndroidJournalStorageCapabilityCertificate {
            certificate_id,
            adapter_definition_id: self.adapter_definition_id,
            storage_domain_id: self.storage_domain_id,
            backend_kind: self.backend_kind,
        })
    }
}

impl AndroidAnchorStorageCapabilityProfile {
    pub fn certify(
        &self,
    ) -> Result<AndroidAnchorStorageCapabilityCertificate, AndroidStorageCapabilityError> {
        if self.adapter_definition_id.is_zero() {
            return Err(AndroidStorageCapabilityError::ZeroAdapterDefinition);
        }
        if self.anchor_domain_id.is_zero() || self.independent_from_journal_domain_id.is_zero() {
            return Err(AndroidStorageCapabilityError::ZeroStorageDomain);
        }
        if self.anchor_domain_id == self.independent_from_journal_domain_id {
            return Err(AndroidStorageCapabilityError::SameStorageAndAnchorDomain);
        }
        nonzero(
            self.adapter_qualification_evidence_id,
            AndroidStorageCapabilityError::MissingAdapterQualification,
        )?;
        nonzero(
            self.independence_evidence_id,
            AndroidStorageCapabilityError::MissingIndependenceEvidence,
        )?;
        if !self.checkpoint_authenticity_established {
            return Err(AndroidStorageCapabilityError::CheckpointAuthenticityNotEstablished);
        }
        if !self.exact_readback_established {
            return Err(AndroidStorageCapabilityError::ExactReadbackNotEstablished);
        }

        // Standard Android app storage and Keystore signing do not expose an
        // arbitrary app checkpoint as a hardware monotonic register. KeyMint's
        // rollback-resistance property for deleted keys must not be laundered into
        // checkpoint monotonicity.
        match self.backend_kind {
            AndroidAnchorBackendKind::ApplicationStorage
            | AndroidAnchorBackendKind::KeystoreSigned
            | AndroidAnchorBackendKind::StrongBoxSigned => {
                if self.monotonic_checkpoint_advance {
                    return Err(AndroidStorageCapabilityError::UnsupportedMonotonicAnchorClaim);
                }
                return Err(AndroidStorageCapabilityError::CheckpointMonotonicityNotEstablished);
            }
            AndroidAnchorBackendKind::ExternalMonotonicWitness
            | AndroidAnchorBackendKind::QualifiedHardwareMonotonicStore => {}
        }

        if !self.monotonic_checkpoint_advance {
            return Err(AndroidStorageCapabilityError::CheckpointMonotonicityNotEstablished);
        }
        nonzero(
            self.monotonicity_evidence_id,
            AndroidStorageCapabilityError::MissingMonotonicityEvidence,
        )?;
        if self.backend_kind == AndroidAnchorBackendKind::QualifiedHardwareMonotonicStore
            && self.key_security_level == AndroidKeySecurityLevel::StrongBox
        {
            // StrongBox is acceptable evidence about key protection, but the
            // monotonic-store qualification still comes from the explicit
            // monotonicity evidence above.
        }
        if self.backend_kind == AndroidAnchorBackendKind::StrongBoxSigned
            && self.key_security_level != AndroidKeySecurityLevel::StrongBox
        {
            return Err(AndroidStorageCapabilityError::StrongBoxSecurityLevelMismatch);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(ANCHOR_CAP_DOMAIN);
        hasher.update(self.adapter_definition_id.as_bytes());
        hasher.update(self.anchor_domain_id.as_bytes());
        hasher.update(self.independent_from_journal_domain_id.as_bytes());
        hasher.update(&[self.backend_kind as u8]);
        hasher.update(&[self.key_security_level as u8]);
        hasher.update(&[u8::from(self.keymint_rollback_resistant_key)]);
        hasher.update(self.independence_evidence_id.as_bytes());
        hasher.update(self.monotonicity_evidence_id.as_bytes());
        hasher.update(self.adapter_qualification_evidence_id.as_bytes());
        let certificate_id =
            AndroidAnchorStorageCapabilityCertificateId(*hasher.finalize().as_bytes());
        Ok(AndroidAnchorStorageCapabilityCertificate {
            certificate_id,
            adapter_definition_id: self.adapter_definition_id,
            anchor_domain_id: self.anchor_domain_id,
            independent_from_journal_domain_id: self.independent_from_journal_domain_id,
            backend_kind: self.backend_kind,
            key_security_level: self.key_security_level,
        })
    }
}

fn validate_operation_evidence(
    journal: &AndroidTouchPolicyJournalDurabilityObservation,
    anchor: &AndroidTouchPolicyAnchorObservation,
) -> Result<(), AndroidStorageCapabilityError> {
    let ids = [
        journal.durable_prepare_evidence_id,
        journal.atomic_publish_evidence_id,
        journal.metadata_durability_evidence_id,
        journal.exact_readback_evidence_id,
        anchor.compare_exchange_evidence_id,
        anchor.monotonic_advance_evidence_id,
        anchor.exact_readback_evidence_id,
    ];
    if !all_distinct(&ids) {
        return Err(AndroidStorageCapabilityError::ReusedOperationEvidence);
    }
    Ok(())
}

/// Android-qualified entry point to 534. The generic theorem remains reusable,
/// but Android callers should use this bridge so platform capabilities are not
/// silently assumed from class names such as AtomicFile or StrongBox.
pub fn certify_android_policy_durable_state_with_capabilities(
    previous: Option<&AndroidTouchPolicyDurableState>,
    journal_bytes: &[u8],
    checkpoint_bytes: &[u8],
    decode_limits: AndroidTouchPolicyDecodeLimits,
    journal_capability: &AndroidJournalStorageCapabilityProfile,
    anchor_capability: &AndroidAnchorStorageCapabilityProfile,
    journal_observation: &AndroidTouchPolicyJournalDurabilityObservation,
    anchor_observation: &AndroidTouchPolicyAnchorObservation,
) -> Result<AndroidTouchPolicyDurableState, AndroidStorageCapabilityError> {
    let journal_certificate = journal_capability.certify()?;
    let anchor_certificate = anchor_capability.certify()?;

    if journal_observation.storage_domain_id != journal_certificate.storage_domain_id
        || anchor_certificate.independent_from_journal_domain_id
            != journal_certificate.storage_domain_id
    {
        return Err(AndroidStorageCapabilityError::JournalCapabilityMismatch);
    }
    if anchor_observation.anchor_domain_id != anchor_certificate.anchor_domain_id {
        return Err(AndroidStorageCapabilityError::AnchorCapabilityMismatch);
    }
    validate_operation_evidence(journal_observation, anchor_observation)?;

    Ok(certify_android_policy_durable_state(
        previous,
        journal_bytes,
        checkpoint_bytes,
        decode_limits,
        journal_observation,
        anchor_observation,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(value: u8) -> [u8; 32] {
        [value; 32]
    }
    fn evidence(value: u8) -> AndroidTouchPolicyStorageEvidenceId {
        AndroidTouchPolicyStorageEvidenceId(d(value))
    }
    fn domain(value: u8) -> AndroidTouchPolicyStorageDomainId {
        AndroidTouchPolicyStorageDomainId(d(value))
    }

    fn atomic_file_profile() -> AndroidJournalStorageCapabilityProfile {
        AndroidJournalStorageCapabilityProfile {
            adapter_definition_id: AndroidStorageAdapterDefinitionId(d(1)),
            storage_domain_id: domain(10),
            backend_kind: AndroidJournalBackendKind::FrameworkAtomicFile,
            synced_commit_semantics: true,
            atomic_publish_semantics: true,
            metadata_durability_established: true,
            exact_readback_established: true,
            mutual_exclusion_evidence_id: evidence(20),
            metadata_durability_evidence_id: evidence(21),
            adapter_qualification_evidence_id: evidence(22),
        }
    }

    fn external_anchor() -> AndroidAnchorStorageCapabilityProfile {
        AndroidAnchorStorageCapabilityProfile {
            adapter_definition_id: AndroidStorageAdapterDefinitionId(d(2)),
            anchor_domain_id: domain(11),
            independent_from_journal_domain_id: domain(10),
            backend_kind: AndroidAnchorBackendKind::ExternalMonotonicWitness,
            key_security_level: AndroidKeySecurityLevel::Unknown,
            checkpoint_authenticity_established: true,
            keymint_rollback_resistant_key: false,
            monotonic_checkpoint_advance: true,
            exact_readback_established: true,
            independence_evidence_id: evidence(30),
            monotonicity_evidence_id: evidence(31),
            adapter_qualification_evidence_id: evidence(32),
        }
    }

    #[test]
    fn atomic_file_still_requires_external_mutual_exclusion_evidence() {
        let mut profile = atomic_file_profile();
        profile.mutual_exclusion_evidence_id = AndroidTouchPolicyStorageEvidenceId::ZERO;
        assert_eq!(
            profile.certify(),
            Err(AndroidStorageCapabilityError::MissingMutualExclusionEvidence)
        );
    }

    #[test]
    fn atomic_file_does_not_invent_metadata_durability() {
        let mut profile = atomic_file_profile();
        profile.metadata_durability_established = false;
        assert_eq!(
            profile.certify(),
            Err(AndroidStorageCapabilityError::MetadataDurabilityNotEstablished)
        );
    }

    #[test]
    fn strongbox_signature_is_not_a_monotonic_checkpoint_anchor() {
        let profile = AndroidAnchorStorageCapabilityProfile {
            adapter_definition_id: AndroidStorageAdapterDefinitionId(d(2)),
            anchor_domain_id: domain(11),
            independent_from_journal_domain_id: domain(10),
            backend_kind: AndroidAnchorBackendKind::StrongBoxSigned,
            key_security_level: AndroidKeySecurityLevel::StrongBox,
            checkpoint_authenticity_established: true,
            keymint_rollback_resistant_key: true,
            monotonic_checkpoint_advance: false,
            exact_readback_established: true,
            independence_evidence_id: evidence(30),
            monotonicity_evidence_id: AndroidTouchPolicyStorageEvidenceId::ZERO,
            adapter_qualification_evidence_id: evidence(32),
        };
        assert_eq!(
            profile.certify(),
            Err(AndroidStorageCapabilityError::CheckpointMonotonicityNotEstablished)
        );
    }

    #[test]
    fn relabeling_strongbox_signing_as_monotonic_is_rejected() {
        let mut profile = AndroidAnchorStorageCapabilityProfile {
            adapter_definition_id: AndroidStorageAdapterDefinitionId(d(2)),
            anchor_domain_id: domain(11),
            independent_from_journal_domain_id: domain(10),
            backend_kind: AndroidAnchorBackendKind::StrongBoxSigned,
            key_security_level: AndroidKeySecurityLevel::StrongBox,
            checkpoint_authenticity_established: true,
            keymint_rollback_resistant_key: true,
            monotonic_checkpoint_advance: true,
            exact_readback_established: true,
            independence_evidence_id: evidence(30),
            monotonicity_evidence_id: evidence(31),
            adapter_qualification_evidence_id: evidence(32),
        };
        assert_eq!(
            profile.certify(),
            Err(AndroidStorageCapabilityError::UnsupportedMonotonicAnchorClaim)
        );
        // The KeyMint key-lifecycle property remains descriptive; changing it does
        // not change checkpoint monotonicity semantics.
        profile.keymint_rollback_resistant_key = false;
        assert_eq!(
            profile.certify(),
            Err(AndroidStorageCapabilityError::UnsupportedMonotonicAnchorClaim)
        );
    }

    #[test]
    fn separately_qualified_external_monotonic_witness_can_supply_anchor_capability() {
        let cert = external_anchor().certify().unwrap();
        assert!(!cert.certificate_id.is_zero());
        assert_eq!(cert.anchor_domain_id, domain(11));
        assert_eq!(cert.independent_from_journal_domain_id, domain(10));
    }

    #[test]
    fn operation_evidence_tokens_must_be_distinct() {
        use crate::assurance_android_policy_durability::{
            AndroidTouchPolicyAnchorObservation, AndroidTouchPolicyCheckpointBytesDigest,
            AndroidTouchPolicyJournalBytesDigest, AndroidTouchPolicyJournalDurableCommitId,
            AndroidTouchPolicyJournalDurabilityObservation,
            AndroidTouchPolicyStorageTransactionId,
        };
        use crate::assurance_android_policy_journal::AndroidTouchPolicyCheckpointId;

        let shared = evidence(40);
        let journal = AndroidTouchPolicyJournalDurabilityObservation {
            storage_domain_id: domain(10),
            transaction_id: AndroidTouchPolicyStorageTransactionId(d(50)),
            previous_storage_generation: 0,
            next_storage_generation: 1,
            journal_bytes_digest: AndroidTouchPolicyJournalBytesDigest(d(51)),
            readback_bytes_digest: AndroidTouchPolicyJournalBytesDigest(d(51)),
            durable_prepare_evidence_id: shared,
            atomic_publish_evidence_id: shared,
            metadata_durability_evidence_id: evidence(42),
            exact_readback_evidence_id: evidence(43),
        };
        let anchor = AndroidTouchPolicyAnchorObservation {
            anchor_domain_id: domain(11),
            previous_anchor_generation: 0,
            next_anchor_generation: 1,
            previous_checkpoint_id: AndroidTouchPolicyCheckpointId::ZERO,
            next_checkpoint_id: AndroidTouchPolicyCheckpointId(d(52)),
            checkpoint_bytes_digest: AndroidTouchPolicyCheckpointBytesDigest(d(53)),
            readback_checkpoint_bytes_digest: AndroidTouchPolicyCheckpointBytesDigest(d(53)),
            journal_commit_id: AndroidTouchPolicyJournalDurableCommitId(d(54)),
            compare_exchange_evidence_id: evidence(44),
            monotonic_advance_evidence_id: evidence(45),
            exact_readback_evidence_id: evidence(46),
        };
        assert_eq!(
            validate_operation_evidence(&journal, &anchor),
            Err(AndroidStorageCapabilityError::ReusedOperationEvidence)
        );
    }
}
