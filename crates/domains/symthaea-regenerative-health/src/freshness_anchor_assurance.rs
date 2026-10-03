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
            integrity_protected: true,
            authenticated: false,
            monotonic: false,
            rollback_resistant: false,
            atomic_update: false,
            crash_persistent: true,
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
    pub fn new(
        backing: FreshnessAnchorBacking,
        capabilities: FreshnessAnchorCapabilities,
        provenance: impl Into<String>,
    ) -> Result<Self, &'static str> {
        let provenance = provenance.into();
        if provenance.trim().is_empty() {
            return Err("anchor provenance must not be empty");
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FreshnessAnchorAssuranceError {
    InvalidProfile,
    InsufficientCapabilities {
        missing: Vec<FreshnessAnchorCapability>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn software_only_anchor_is_explicitly_non_authoritative() {
        let capabilities = FreshnessAnchorCapabilities::software_only();
        assert_eq!(
            capabilities.assurance(),
            FreshnessAnchorAssurance::IntegrityOnly
        );
        assert!(!capabilities.is_authoritative());
        assert!(capabilities.require_authoritative().is_err());
        assert!(capabilities
            .missing_authoritative_capabilities()
            .contains(&FreshnessAnchorCapability::RollbackResistance));
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
