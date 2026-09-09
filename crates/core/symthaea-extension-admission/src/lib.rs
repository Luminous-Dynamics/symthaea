// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed admission contracts for Symthaea extensions.
//!
//! Discovery, technical compatibility, signature validity, signer authorization,
//! capability admission, and point-of-use activation are different facts. This
//! crate models only the host-owned admission decision and its live
//! policy/trust-currentness check. It performs no cryptography and grants no
//! authority by itself.
//!
//! The important boundary is:
//!
//! ```text
//! serialized AdmissionRecord
//!     != ActiveAdmission
//! ```
//!
//! `AdmissionRecord` is immutable issuance evidence. `ActiveAdmission` is a
//! non-serializable point-of-use value created only after the record is checked
//! against the current manifest, admission-policy generation, trust generation,
//! and live revocation state.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_extension_core::{
    CapabilityId, ExtensionId, ExtensionManifest, FilesystemPermission, NetworkPermission,
    PermissionSet,
};

/// Exact SHA-256 commitment used to bind admission to immutable inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Sha256Digest(pub [u8; 32]);

impl Sha256Digest {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Host-local or externally-resolved principal identity.
///
/// This is intentionally opaque. Xenia/Mycelix/local trust infrastructure may
/// supply stronger semantics without this crate duplicating their identity
/// systems. Text is canonical: leading/trailing whitespace is rejected rather
/// than silently normalized.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PrincipalId(String);

impl PrincipalId {
    pub fn new(value: impl Into<String>) -> Result<Self, AdmissionProblem> {
        let value = value.into();
        validate_principal(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Host-assigned authorization floor.
///
/// This is not a quality/evidence score. A more-trusted publisher does not make
/// a solver or model more accurate.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    #[default]
    Untrusted,
    Community,
    Trusted,
    Privileged,
}

/// Immutable host admission decision.
///
/// The record binds the exact manifest bytes, exact executable/package payload,
/// exact admission policy, and trust-snapshot generation that produced the
/// decision. Signature evidence is deliberately external; `signer` records the
/// already-resolved identity, not a claim that this crate verified it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionRecord {
    extension: ExtensionId,
    extension_version: String,
    manifest_sha256: Sha256Digest,
    payload_sha256: Sha256Digest,
    policy_sha256: Sha256Digest,
    issuer: PrincipalId,
    signer: Option<PrincipalId>,
    trust: TrustLevel,
    granted_capabilities: Vec<CapabilityId>,
    granted_permissions: PermissionSet,
    generation: u64,
    trust_generation: u64,
}

impl AdmissionRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        extension: ExtensionId,
        extension_version: impl Into<String>,
        manifest_sha256: Sha256Digest,
        payload_sha256: Sha256Digest,
        policy_sha256: Sha256Digest,
        issuer: PrincipalId,
        signer: Option<PrincipalId>,
        trust: TrustLevel,
        mut granted_capabilities: Vec<CapabilityId>,
        granted_permissions: PermissionSet,
        generation: u64,
        trust_generation: u64,
    ) -> Result<Self, AdmissionProblem> {
        let extension_version = extension_version.into();
        if extension_version.trim().is_empty() {
            return Err(AdmissionProblem::EmptyVersion);
        }
        if extension_version != extension_version.trim() {
            return Err(AdmissionProblem::NonCanonicalVersion);
        }
        if generation == 0 {
            return Err(AdmissionProblem::ZeroGeneration);
        }
        if trust_generation == 0 {
            return Err(AdmissionProblem::ZeroTrustGeneration);
        }
        if granted_capabilities.is_empty() {
            return Err(AdmissionProblem::EmptyCapabilityGrant);
        }

        granted_capabilities.sort();
        let before = granted_capabilities.len();
        granted_capabilities.dedup();
        if granted_capabilities.len() != before {
            return Err(AdmissionProblem::DuplicateCapabilityGrant);
        }

        let record = Self {
            extension,
            extension_version,
            manifest_sha256,
            payload_sha256,
            policy_sha256,
            issuer,
            signer,
            trust,
            granted_capabilities,
            granted_permissions,
            generation,
            trust_generation,
        };
        record.validate()?;
        Ok(record)
    }

    /// Validate the record's own canonical structure after deserialization.
    pub fn validate(&self) -> Result<(), AdmissionProblem> {
        if self.extension_version.trim().is_empty() {
            return Err(AdmissionProblem::EmptyVersion);
        }
        if self.extension_version != self.extension_version.trim() {
            return Err(AdmissionProblem::NonCanonicalVersion);
        }
        if self.generation == 0 {
            return Err(AdmissionProblem::ZeroGeneration);
        }
        if self.trust_generation == 0 {
            return Err(AdmissionProblem::ZeroTrustGeneration);
        }
        validate_principal(self.issuer.as_str())?;
        if let Some(signer) = &self.signer {
            validate_principal(signer.as_str())?;
        }
        if self.granted_capabilities.is_empty() {
            return Err(AdmissionProblem::EmptyCapabilityGrant);
        }
        if !self
            .granted_capabilities
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        {
            return Err(AdmissionProblem::NonCanonicalCapabilityGrant);
        }
        Ok(())
    }

    /// Prove that this admission cannot widen the extension's own declaration.
    pub fn validate_against_manifest(
        &self,
        manifest: &ExtensionManifest,
    ) -> Result<(), AdmissionProblem> {
        self.validate()?;
        manifest
            .validate()
            .map_err(|_| AdmissionProblem::InvalidManifest)?;

        if self.extension != manifest.id {
            return Err(AdmissionProblem::ExtensionMismatch);
        }
        if self.extension_version != manifest.version {
            return Err(AdmissionProblem::VersionMismatch);
        }

        let provided: BTreeSet<_> = manifest
            .provides
            .iter()
            .map(|capability| &capability.id)
            .collect();
        if self
            .granted_capabilities
            .iter()
            .any(|capability| !provided.contains(capability))
        {
            return Err(AdmissionProblem::CapabilityNotDeclared);
        }

        if !permissions_are_subset(&self.granted_permissions, &manifest.permissions) {
            return Err(AdmissionProblem::PermissionExceedsManifest);
        }

        Ok(())
    }

    /// Revalidate immutable issuance against live policy/trust generations and
    /// revocation state, returning a non-serializable point-of-use admission.
    ///
    /// The validated manifest is retained inside the active value. This prevents
    /// an admission activated against manifest A from being replayed against a
    /// different in-process manifest B that happens to reuse the same ID/version.
    pub fn activate(
        &self,
        manifest: &ExtensionManifest,
        context: AdmissionContext,
    ) -> Result<ActiveAdmission, AdmissionProblem> {
        self.validate_against_manifest(manifest)?;
        if context.revoked {
            return Err(AdmissionProblem::Revoked);
        }
        if context.current_generation != self.generation {
            return Err(AdmissionProblem::GenerationMismatch {
                admitted: self.generation,
                current: context.current_generation,
            });
        }
        if context.current_trust_generation != self.trust_generation {
            return Err(AdmissionProblem::TrustGenerationMismatch {
                admitted: self.trust_generation,
                current: context.current_trust_generation,
            });
        }
        Ok(ActiveAdmission {
            record: self.clone(),
            manifest: manifest.clone(),
        })
    }

    pub fn extension(&self) -> &ExtensionId {
        &self.extension
    }

    pub fn extension_version(&self) -> &str {
        &self.extension_version
    }

    pub fn manifest_sha256(&self) -> Sha256Digest {
        self.manifest_sha256
    }

    pub fn payload_sha256(&self) -> Sha256Digest {
        self.payload_sha256
    }

    pub fn policy_sha256(&self) -> Sha256Digest {
        self.policy_sha256
    }

    pub fn issuer(&self) -> &PrincipalId {
        &self.issuer
    }

    pub fn signer(&self) -> Option<&PrincipalId> {
        self.signer.as_ref()
    }

    pub fn trust(&self) -> TrustLevel {
        self.trust
    }

    pub fn granted_capabilities(&self) -> &[CapabilityId] {
        &self.granted_capabilities
    }

    pub fn granted_permissions(&self) -> &PermissionSet {
        &self.granted_permissions
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn trust_generation(&self) -> u64 {
        self.trust_generation
    }
}

/// Live host facts checked at the instant an admission is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionContext {
    pub current_generation: u64,
    pub current_trust_generation: u64,
    pub revoked: bool,
}

impl AdmissionContext {
    pub const fn active(current_generation: u64, current_trust_generation: u64) -> Self {
        Self {
            current_generation,
            current_trust_generation,
            revoked: false,
        }
    }
}

/// Non-serializable proof that one admission record is current for one exact
/// in-process manifest.
///
/// This type intentionally does not implement `Clone` or serde. Obtain a fresh
/// value at point of use after checking current policy/trust generations and
/// revocation state.
#[derive(Debug, PartialEq, Eq)]
pub struct ActiveAdmission {
    record: AdmissionRecord,
    manifest: ExtensionManifest,
}

impl ActiveAdmission {
    pub fn extension(&self) -> &ExtensionId {
        self.record.extension()
    }

    pub fn trust(&self) -> TrustLevel {
        self.record.trust()
    }

    pub fn generation(&self) -> u64 {
        self.record.generation()
    }

    pub fn trust_generation(&self) -> u64 {
        self.record.trust_generation()
    }

    pub fn manifest_sha256(&self) -> Sha256Digest {
        self.record.manifest_sha256()
    }

    pub fn payload_sha256(&self) -> Sha256Digest {
        self.record.payload_sha256()
    }

    pub fn policy_sha256(&self) -> Sha256Digest {
        self.record.policy_sha256()
    }

    /// Exact structural manifest match proven during activation.
    pub fn matches_manifest(&self, manifest: &ExtensionManifest) -> bool {
        &self.manifest == manifest
    }

    pub fn allows_capability(&self, capability: &CapabilityId) -> bool {
        self.record
            .granted_capabilities()
            .binary_search(capability)
            .is_ok()
    }

    pub fn granted_permissions(&self) -> &PermissionSet {
        self.record.granted_permissions()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionProblem {
    InvalidManifest,
    EmptyVersion,
    NonCanonicalVersion,
    ZeroGeneration,
    ZeroTrustGeneration,
    InvalidPrincipal,
    EmptyCapabilityGrant,
    DuplicateCapabilityGrant,
    NonCanonicalCapabilityGrant,
    ExtensionMismatch,
    VersionMismatch,
    CapabilityNotDeclared,
    PermissionExceedsManifest,
    Revoked,
    GenerationMismatch { admitted: u64, current: u64 },
    TrustGenerationMismatch { admitted: u64, current: u64 },
}

fn validate_principal(value: &str) -> Result<(), AdmissionProblem> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(|character| character.is_control())
    {
        return Err(AdmissionProblem::InvalidPrincipal);
    }
    Ok(())
}

fn permissions_are_subset(granted: &PermissionSet, requested: &PermissionSet) -> bool {
    network_is_subset(&granted.network, &requested.network)
        && filesystem_is_subset(&granted.filesystem, &requested.filesystem)
        && (!granted.gpu || requested.gpu)
        && (!granted.wall_clock || requested.wall_clock)
        && (!granted.randomness || requested.randomness)
        && strings_are_subset(&granted.sensors, &requested.sensors)
        && strings_are_subset(&granted.actuators, &requested.actuators)
}

fn network_is_subset(granted: &NetworkPermission, requested: &NetworkPermission) -> bool {
    match (granted, requested) {
        (NetworkPermission::None, _) => true,
        (NetworkPermission::Allowlist(granted), NetworkPermission::Allowlist(requested)) => {
            strings_are_subset(granted, requested)
        }
        (NetworkPermission::Allowlist(_), NetworkPermission::Unrestricted) => true,
        (NetworkPermission::Unrestricted, NetworkPermission::Unrestricted) => true,
        _ => false,
    }
}

fn filesystem_is_subset(
    granted: &FilesystemPermission,
    requested: &FilesystemPermission,
) -> bool {
    // Paths are exact opaque grants here; this crate deliberately performs no
    // parent-directory or symlink interpretation. A concrete host is responsible
    // for mapping a granted path to its sandbox/runtime semantics.
    match (granted, requested) {
        (FilesystemPermission::None, _) => true,
        (FilesystemPermission::ReadOnly(granted), FilesystemPermission::ReadOnly(requested))
        | (FilesystemPermission::ReadOnly(granted), FilesystemPermission::ReadWrite(requested))
        | (FilesystemPermission::ReadWrite(granted), FilesystemPermission::ReadWrite(requested)) => {
            strings_are_subset(granted, requested)
        }
        _ => false,
    }
}

fn strings_are_subset(granted: &[String], requested: &[String]) -> bool {
    granted.iter().all(|item| requested.contains(item))
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_extension_core::{
        AbiVersion, CapabilityDescriptor, EffectClass, ExtensionKind, ResourceBudget, RuntimeKind,
    };

    fn digest(byte: u8) -> Sha256Digest {
        Sha256Digest::new([byte; 32])
    }

    fn manifest() -> ExtensionManifest {
        ExtensionManifest {
            id: ExtensionId::new("org.example.solver"),
            name: "Example solver".into(),
            version: "1.0.0".into(),
            abi: AbiVersion::V1,
            kind: ExtensionKind::Simulation,
            runtime: RuntimeKind::Wasm,
            description: String::new(),
            provides: vec![
                CapabilityDescriptor {
                    id: CapabilityId::new("engineering.simulation.circuit"),
                    description: "circuit simulation".into(),
                    effect: EffectClass::Pure,
                },
                CapabilityDescriptor {
                    id: CapabilityId::new("engineering.simulation.process"),
                    description: "process simulation".into(),
                    effect: EffectClass::Pure,
                },
            ],
            requires: vec![],
            permissions: PermissionSet {
                network: NetworkPermission::Allowlist(vec!["solver.example".into()]),
                filesystem: FilesystemPermission::ReadOnly(vec!["/models".into()]),
                wall_clock: true,
                ..PermissionSet::default()
            },
            resources: ResourceBudget::default(),
        }
    }

    fn record() -> AdmissionRecord {
        AdmissionRecord::issue(
            ExtensionId::new("org.example.solver"),
            "1.0.0",
            digest(1),
            digest(2),
            digest(3),
            PrincipalId::new("local:extension-authority").unwrap(),
            Some(PrincipalId::new("did:example:publisher").unwrap()),
            TrustLevel::Trusted,
            vec![CapabilityId::new("engineering.simulation.circuit")],
            PermissionSet {
                filesystem: FilesystemPermission::ReadOnly(vec!["/models".into()]),
                ..PermissionSet::default()
            },
            7,
            13,
        )
        .unwrap()
    }

    #[test]
    fn active_admission_requires_current_policy_trust_and_exact_manifest() {
        let manifest = manifest();
        let record = record();
        let active = record
            .activate(&manifest, AdmissionContext::active(7, 13))
            .unwrap();
        assert_eq!(active.extension(), &manifest.id);
        assert_eq!(active.trust_generation(), 13);
        assert!(active.matches_manifest(&manifest));
        assert!(active.allows_capability(&CapabilityId::new(
            "engineering.simulation.circuit"
        )));
        assert!(!active.allows_capability(&CapabilityId::new(
            "engineering.simulation.process"
        )));
    }

    #[test]
    fn active_admission_rejects_same_id_version_manifest_substitution() {
        let manifest = manifest();
        let active = record()
            .activate(&manifest, AdmissionContext::active(7, 13))
            .unwrap();
        let mut substituted = manifest.clone();
        substituted.description = "changed after activation".into();
        assert!(!active.matches_manifest(&substituted));
    }

    #[test]
    fn live_revocation_fails_without_rewriting_issuance_record() {
        let manifest = manifest();
        let record = record();
        assert_eq!(record.generation(), 7);
        assert_eq!(
            record.activate(
                &manifest,
                AdmissionContext {
                    current_generation: 7,
                    current_trust_generation: 13,
                    revoked: true,
                }
            ),
            Err(AdmissionProblem::Revoked)
        );
        assert_eq!(record.generation(), 7);
    }

    #[test]
    fn policy_generation_replacement_invalidates_old_admission() {
        assert_eq!(
            record().activate(&manifest(), AdmissionContext::active(8, 13)),
            Err(AdmissionProblem::GenerationMismatch {
                admitted: 7,
                current: 8,
            })
        );
    }

    #[test]
    fn trust_generation_replacement_invalidates_old_admission() {
        assert_eq!(
            record().activate(&manifest(), AdmissionContext::active(7, 14)),
            Err(AdmissionProblem::TrustGenerationMismatch {
                admitted: 13,
                current: 14,
            })
        );
    }

    #[test]
    fn capability_grant_cannot_exceed_manifest() {
        let mut record = record();
        record.granted_capabilities = vec![CapabilityId::new("robotics.motion.command")];
        assert_eq!(
            record.validate_against_manifest(&manifest()),
            Err(AdmissionProblem::CapabilityNotDeclared)
        );
    }

    #[test]
    fn permission_grant_cannot_exceed_manifest() {
        let mut record = record();
        record.granted_permissions.network = NetworkPermission::Unrestricted;
        assert_eq!(
            record.validate_against_manifest(&manifest()),
            Err(AdmissionProblem::PermissionExceedsManifest)
        );
    }

    #[test]
    fn principal_identity_must_be_canonical() {
        assert_eq!(
            PrincipalId::new(" did:example:publisher "),
            Err(AdmissionProblem::InvalidPrincipal)
        );
    }

    #[test]
    fn deserialized_noncanonical_capability_order_fails_closed() {
        let mut value = serde_json::to_value(record()).unwrap();
        value["granted_capabilities"] = serde_json::json!([
            "engineering.simulation.process",
            "engineering.simulation.circuit"
        ]);
        let decoded: AdmissionRecord = serde_json::from_value(value).unwrap();
        assert_eq!(
            decoded.validate(),
            Err(AdmissionProblem::NonCanonicalCapabilityGrant)
        );
    }
}
