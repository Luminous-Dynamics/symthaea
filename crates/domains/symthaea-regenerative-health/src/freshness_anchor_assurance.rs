// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Explicit security semantics for receiver freshness anchors.
//!
//! A recovery snapshot can be authenticated and integrity-protected while still
//! being rollbackable. This module keeps those properties separate so a
//! deployment cannot silently treat ordinary software persistence as an
//! authoritative anti-rollback mechanism.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FreshnessAnchorBacking {
    /// Protected hardware state such as a TPM/NV counter or equivalent.
    HardwareProtected,
    /// A remote authority whose state is outside the receiver's rollback domain.
    RemoteAuthority,
    /// A quorum/replicated authority outside any single receiver's rollback domain.
    ReplicatedQuorum,
    /// Ordinary receiver-local software persistence.
    SoftwareOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessAnchorCapabilities {
    pub integrity_protected: bool,
    pub authenticated: bool,
    pub monotonic: bool,
    pub rollback_resistant: bool,
    pub atomic_update: bool,
    pub crash_persistent: bool,
}

impl FreshnessAnchorCapabilities {
    pub const fn software_only() -> Self {
        Self {
            integrity_protected: false,
            authenticated: false,
            monotonic: false,
            rollback_resistant: false,
            atomic_update: false,
            crash_persistent: false,
        }
    }

    pub const fn authoritative() -> Self {
        Self {
            integrity_protected: true,
            authenticated: true,
            monotonic: true,
            rollback_resistant: true,
            atomic_update: true,
            crash_persistent: true,
        }
    }

    pub const fn is_authoritative(&self) -> bool {
        self.integrity_protected
            && self.authenticated
            && self.monotonic
            && self.rollback_resistant
            && self.atomic_update
            && self.crash_persistent
    }

    pub fn assurance(&self) -> FreshnessAnchorAssurance {
        if self.is_authoritative() {
            FreshnessAnchorAssurance::RollbackResistant
        } else if self.integrity_protected && self.authenticated && self.monotonic {
            FreshnessAnchorAssurance::MonotonicAuthenticated
        } else if self.integrity_protected && self.authenticated {
            FreshnessAnchorAssurance::Authenticated
        } else if self.integrity_protected {
            FreshnessAnchorAssurance::IntegrityOnly
        } else {
            FreshnessAnchorAssurance::Untrusted
        }
    }

    pub fn missing_authoritative_capabilities(&self) -> Vec<FreshnessAnchorCapability> {
        let mut missing = Vec::new();
        if !self.integrity_protected {
            missing.push(FreshnessAnchorCapability::IntegrityProtection);
        }
        if !self.authenticated {
            missing.push(FreshnessAnchorCapability::Authentication);
        }
        if !self.monotonic {
            missing.push(FreshnessAnchorCapability::Monotonicity);
        }
        if !self.rollback_resistant {
            missing.push(FreshnessAnchorCapability::RollbackResistance);
        }
        if !self.atomic_update {
            missing.push(FreshnessAnchorCapability::AtomicUpdate);
        }
        if !self.crash_persistent {
            missing.push(FreshnessAnchorCapability::CrashPersistence);
        }
        missing
    }

    pub fn require_authoritative(&self) -> Result<(), FreshnessAnchorAssuranceError> {
        let missing = self.missing_authoritative_capabilities();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(FreshnessAnchorAssuranceError::InsufficientCapabilities { missing })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessAnchorCapability {
    IntegrityProtection,
    Authentication,
    Monotonicity,
    RollbackResistance,
    AtomicUpdate,
    CrashPersistence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FreshnessAnchorAssurance {
    Untrusted,
    IntegrityOnly,
    Authenticated,
    MonotonicAuthenticated,
    RollbackResistant,
}

impl FreshnessAnchorAssurance {
    pub const fn is_authoritative(self) -> bool {
        matches!(self, Self::RollbackResistant)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessAnchorProfile {
    pub schema_version: String,
    pub backing: FreshnessAnchorBacking,
    pub capabilities: FreshnessAnchorCapabilities,
    pub provenance: String,
}

impl FreshnessAnchorProfile {
    /// Compute the exact content commitment for the profile's security
    /// semantics using explicit, versioned field framing.
    pub fn fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:freshness-anchor-profile:v1\\0");
        hasher.update(&[match self.backing {
            FreshnessAnchorBacking::HardwareProtected => 0,
            FreshnessAnchorBacking::RemoteAuthority => 1,
            FreshnessAnchorBacking::ReplicatedQuorum => 2,
            FreshnessAnchorBacking::SoftwareOnly => 3,
        }]);
        for value in [&self.schema_version, &self.provenance] {
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        for value in [
            self.capabilities.integrity_protected,
            self.capabilities.authenticated,
            self.capabilities.monotonic,
            self.capabilities.rollback_resistant,
            self.capabilities.atomic_update,
            self.capabilities.crash_persistent,
        ] {
            hasher.update(&[u8::from(value)]);
        }
        hasher.finalize().to_hex().to_string()
    }

    pub fn new(
        backing: FreshnessAnchorBacking,
        capabilities: FreshnessAnchorCapabilities,
        provenance: impl Into<String>,
    ) -> Result<Self, &'static str> {
        let provenance = provenance.into();
        if provenance.trim().is_empty() {
            return Err("anchor provenance must not be empty");
        }
        if matches!(backing, FreshnessAnchorBacking::SoftwareOnly)
            && capabilities.is_authoritative()
        {
            return Err("software-only backing cannot declare authoritative capabilities");
        }

        Ok(Self {
            schema_version: "0.1".into(),
            backing,
            capabilities,
            provenance,
        })
    }

    pub fn assurance(&self) -> FreshnessAnchorAssurance {
        self.capabilities.assurance()
    }

    pub fn require_authoritative(&self) -> Result<(), FreshnessAnchorAssuranceError> {
        if self.schema_version != "0.1" || self.provenance.trim().is_empty() {
            return Err(FreshnessAnchorAssuranceError::InvalidProfile);
        }
        self.capabilities.require_authoritative()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessAnchorEvidenceKind {
    HardwareMonotonicCounter {
        backend_identity_digest: String,
        counter_namespace_digest: String,
        observed_counter: u64,
    },
    RemoteMonotonicSequence {
        authority_identity_digest: String,
        authority_namespace_digest: String,
        observed_sequence: u64,
    },
    QuorumMonotonicSequence {
        quorum_policy_digest: String,
        member_set_digest: String,
        threshold: u16,
        observed_sequence: u64,
        certificate_digest: String,
    },
}

impl FreshnessAnchorEvidenceKind {
    pub fn observed_sequence(&self) -> u64 {
        match self {
            Self::HardwareMonotonicCounter { observed_counter, .. } => *observed_counter,
            Self::RemoteMonotonicSequence { observed_sequence, .. } => *observed_sequence,
            Self::QuorumMonotonicSequence { observed_sequence, .. } => *observed_sequence,
        }
    }

    pub fn matches_backing(&self, backing: FreshnessAnchorBacking) -> bool {
        matches!(
            (backing, self),
            (
                FreshnessAnchorBacking::HardwareProtected,
                Self::HardwareMonotonicCounter { .. }
            ) | (
                FreshnessAnchorBacking::RemoteAuthority,
                Self::RemoteMonotonicSequence { .. }
            ) | (
                FreshnessAnchorBacking::ReplicatedQuorum,
                Self::QuorumMonotonicSequence { .. }
            )
        )
    }

    pub fn validate(&self) -> bool {
        let nonempty = |value: &str| !value.trim().is_empty();
        match self {
            Self::HardwareMonotonicCounter {
                backend_identity_digest,
                counter_namespace_digest,
                ..
            } => nonempty(backend_identity_digest) && nonempty(counter_namespace_digest),
            Self::RemoteMonotonicSequence {
                authority_identity_digest,
                authority_namespace_digest,
                ..
            } => nonempty(authority_identity_digest) && nonempty(authority_namespace_digest),
            Self::QuorumMonotonicSequence {
                quorum_policy_digest,
                member_set_digest,
                threshold,
                certificate_digest,
                ..
            } => {
                nonempty(quorum_policy_digest)
                    && nonempty(member_set_digest)
                    && *threshold > 0
                    && nonempty(certificate_digest)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessAnchorVerificationReceipt {
    pub schema_version: String,
    pub profile_fingerprint: String,
    pub receiver_id: String,
    pub generation: u64,
    pub state_fingerprint: String,
    pub verifier_reference: String,
    pub verifier_policy_digest: String,
    pub reference_values_digest: String,
    pub evidence_reference: String,
    pub evidence_digest: String,
    pub evidence_kind: FreshnessAnchorEvidenceKind,
}

pub trait FreshnessAnchorEvidenceVerifier {
    fn verify(
        &self,
        profile: &FreshnessAnchorProfile,
        receipt: &FreshnessAnchorVerificationReceipt,
    ) -> bool;
}

/// Non-serializable capability minted only after an evidence verifier accepts
/// a receipt bound to the exact anchor profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedFreshnessAnchor {
    profile: FreshnessAnchorProfile,
    receipt: FreshnessAnchorVerificationReceipt,
}

impl VerifiedFreshnessAnchor {
    pub fn verify<V: FreshnessAnchorEvidenceVerifier>(
        profile: FreshnessAnchorProfile,
        receipt: FreshnessAnchorVerificationReceipt,
        verifier: &V,
    ) -> Result<Self, FreshnessAnchorAssuranceError> {
        profile.require_authoritative()?;
        if receipt.schema_version != "0.1"
            || receipt.profile_fingerprint.trim().is_empty()
            || receipt.receiver_id.trim().is_empty()
            || receipt.state_fingerprint.trim().is_empty()
            || receipt.verifier_reference.trim().is_empty()
            || receipt.verifier_policy_digest.trim().is_empty()
            || receipt.reference_values_digest.trim().is_empty()
            || receipt.evidence_reference.trim().is_empty()
            || receipt.evidence_digest.trim().is_empty()
            || !receipt.evidence_kind.validate()
            || !receipt.evidence_kind.matches_backing(profile.backing)
        {
            return Err(FreshnessAnchorAssuranceError::InvalidReceipt);
        }
        let profile_fingerprint = profile.fingerprint();
        if receipt.profile_fingerprint != profile_fingerprint {
            return Err(FreshnessAnchorAssuranceError::ProfileBindingMismatch);
        }
        if !verifier.verify(&profile, &receipt) {
            return Err(FreshnessAnchorAssuranceError::EvidenceVerificationFailed);
        }

        Ok(Self { profile, receipt })
    }

    pub fn profile(&self) -> &FreshnessAnchorProfile {
        &self.profile
    }

    pub fn receipt(&self) -> &FreshnessAnchorVerificationReceipt {
        &self.receipt
    }

    pub fn receiver_id(&self) -> &str {
        &self.receipt.receiver_id
    }

    pub fn generation(&self) -> u64 {
        self.receipt.generation
    }

    pub fn state_fingerprint(&self) -> &str {
        &self.receipt.state_fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FreshnessAnchorAssuranceError {
    InvalidProfile,
    InvalidReceipt,
    ProfileBindingMismatch,
    SubjectBindingMismatch,
    EvidenceVerificationFailed,
    InsufficientCapabilities {
        missing: Vec<FreshnessAnchorCapability>,
    },
    CommitRejected(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn software_only_anchor_is_explicitly_non_authoritative() {
        let capabilities = FreshnessAnchorCapabilities::software_only();
        assert_eq!(capabilities.assurance(), FreshnessAnchorAssurance::Untrusted);
        assert!(!capabilities.is_authoritative());
        assert!(capabilities.require_authoritative().is_err());
        assert!(capabilities
            .missing_authoritative_capabilities()
            .contains(&FreshnessAnchorCapability::IntegrityProtection));
    }

    #[test]
    fn authoritative_capabilities_require_all_properties() {
        let capabilities = FreshnessAnchorCapabilities::authoritative();
        assert_eq!(
            capabilities.assurance(),
            FreshnessAnchorAssurance::RollbackResistant
        );
        assert!(capabilities.is_authoritative());
        assert!(capabilities.require_authoritative().is_ok());
    }

    #[test]
    fn authentication_and_monotonicity_without_rollback_resistance_are_not_authoritative() {
        let capabilities = FreshnessAnchorCapabilities {
            integrity_protected: true,
            authenticated: true,
            monotonic: true,
            rollback_resistant: false,
            atomic_update: true,
            crash_persistent: true,
        };
        assert_eq!(
            capabilities.assurance(),
            FreshnessAnchorAssurance::MonotonicAuthenticated
        );
        assert!(!capabilities.is_authoritative());
        let error = capabilities.require_authoritative().unwrap_err();
        assert_eq!(
            error,
            FreshnessAnchorAssuranceError::InsufficientCapabilities {
                missing: vec![FreshnessAnchorCapability::RollbackResistance],
            }
        );
    }

    #[test]
    fn provenance_is_required_for_a_profile() {
        assert!(FreshnessAnchorProfile::new(
            FreshnessAnchorBacking::RemoteAuthority,
            FreshnessAnchorCapabilities::authoritative(),
            " "
        )
        .is_err());
    }

    #[test]
    fn profile_fingerprint_changes_with_security_semantics() {
        let a = FreshnessAnchorProfile::new(
            FreshnessAnchorBacking::RemoteAuthority,
            FreshnessAnchorCapabilities::authoritative(),
            "remote://authority-a",
        )
        .unwrap();
        let mut b = a.clone();
        b.provenance = "remote://authority-b".into();
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn verification_receipt_must_bind_exact_profile() {
        struct Accept;
        impl FreshnessAnchorEvidenceVerifier for Accept {
            fn verify(
                &self,
                _: &FreshnessAnchorProfile,
                _: &FreshnessAnchorVerificationReceipt,
            ) -> bool { true }
        }

        let profile = FreshnessAnchorProfile::new(
            FreshnessAnchorBacking::RemoteAuthority,
            FreshnessAnchorCapabilities::authoritative(),
            "remote://authority-a",
        )
        .unwrap();
        let receipt = FreshnessAnchorVerificationReceipt {
            schema_version: "0.1".into(),
            profile_fingerprint: "wrong".into(),
            receiver_id: "receiver-1".into(),
            generation: 7,
            state_fingerprint: "state-1".into(),
            verifier_reference: "verifier-1".into(),
            verifier_policy_digest: "policy-digest-1".into(),
            reference_values_digest: "reference-values-1".into(),
            evidence_reference: "evidence-1".into(),
            evidence_digest: "digest-1".into(),
            evidence_kind: FreshnessAnchorEvidenceKind::RemoteMonotonicSequence {
                authority_identity_digest: "authority-id-1".into(),
                authority_namespace_digest: "namespace-1".into(),
                observed_sequence: 7,
            },
        };
        let err = VerifiedFreshnessAnchor::verify(profile, receipt, &Accept).unwrap_err();
        assert_eq!(err, FreshnessAnchorAssuranceError::ProfileBindingMismatch);
    }

    #[test]
    fn evidence_verifier_is_required_to_mint_opaque_capability() {
        struct Reject;
        impl FreshnessAnchorEvidenceVerifier for Reject {
            fn verify(
                &self,
                _: &FreshnessAnchorProfile,
                _: &FreshnessAnchorVerificationReceipt,
            ) -> bool { false }
        }

        let profile = FreshnessAnchorProfile::new(
            FreshnessAnchorBacking::RemoteAuthority,
            FreshnessAnchorCapabilities::authoritative(),
            "remote://authority-a",
        )
        .unwrap();
        let receipt = FreshnessAnchorVerificationReceipt {
            schema_version: "0.1".into(),
            profile_fingerprint: profile.fingerprint(),
            receiver_id: "receiver-1".into(),
            generation: 7,
            state_fingerprint: "state-1".into(),
            verifier_reference: "verifier-1".into(),
            evidence_reference: "evidence-1".into(),
            evidence_digest: "digest-1".into(),
        };
        let err = VerifiedFreshnessAnchor::verify(profile, receipt, &Reject).unwrap_err();
        assert_eq!(err, FreshnessAnchorAssuranceError::EvidenceVerificationFailed);
    }

    #[test]
    fn software_only_backing_cannot_declare_authoritative_capabilities() {
        assert!(FreshnessAnchorProfile::new(
            FreshnessAnchorBacking::SoftwareOnly,
            FreshnessAnchorCapabilities::authoritative(),
            "local-file"
        )
        .is_err());
    }

    #[test]
    fn backing_does_not_automatically_claim_assurance() {
        let profile = FreshnessAnchorProfile::new(
            FreshnessAnchorBacking::HardwareProtected,
            FreshnessAnchorCapabilities {
                integrity_protected: true,
                authenticated: true,
                monotonic: true,
                rollback_resistant: false,
                atomic_update: true,
                crash_persistent: true,
            },
            "tpm-nv-index:7",
        )
        .unwrap();

        assert_eq!(
            profile.assurance(),
            FreshnessAnchorAssurance::MonotonicAuthenticated
        );
        assert!(profile.require_authoritative().is_err());
    }
}
