// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed admission contracts for Symthaea extensions.
//!
//! Discovery, technical compatibility, signature validity, signer authorization,
//! persisted admission evidence, and point-of-use authority are distinct facts.
//! This crate models the host-owned admission decision and its live
//! policy/trust-currentness check. It performs no cryptography.
//!
//! The load-bearing boundary is:
//!
//! ```text
//! serialized AdmissionRecordEvidence
//!     != AdmissionRecord
//!     != ActiveAdmission
//! ```
//!
//! [`AdmissionRecord`] is an in-process host-issued authority record. It may be
//! serialized for audit evidence, but deliberately does **not** implement
//! `Deserialize`; persisted bytes cannot recreate authority. Parse persisted
//! data as [`AdmissionRecordEvidence`] and rerun the admission pipeline to mint a
//! new `AdmissionRecord` after restart.
//!
//! Currentness is never accepted as a caller-supplied boolean. Activation and
//! point-of-use rechecks query an injected [`AdmissionCurrentnessSource`] using
//! the immutable admission subject (extension, version, issuer, signer).

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt::Debug;
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

/// Host-assigned authorization floor, never a provider-quality score.
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

/// Stable identity facts presented to the host currentness source.
///
/// The source can use the issuer and signer as well as extension identity when
/// resolving current policy and key-authority state. The subject carries no
/// mutable authority state itself.
#[derive(Debug, Clone, Copy)]
pub struct AdmissionSubject<'a> {
    extension: &'a ExtensionId,
    extension_version: &'a str,
    issuer: &'a PrincipalId,
    signer: Option<&'a PrincipalId>,
}

impl<'a> AdmissionSubject<'a> {
    pub fn extension(&self) -> &'a ExtensionId {
        self.extension
    }

    pub const fn extension_version(&self) -> &'a str {
        self.extension_version
    }

    pub fn issuer(&self) -> &'a PrincipalId {
        self.issuer
    }

    pub const fn signer(&self) -> Option<&'a PrincipalId> {
        self.signer
    }
}

/// Live authority facts returned by a host-owned currentness source.
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

    pub const fn revoked(current_generation: u64, current_trust_generation: u64) -> Self {
        Self {
            current_generation,
            current_trust_generation,
            revoked: true,
        }
    }
}

/// Host-owned authority boundary for policy/trust currentness.
///
/// Returning `None` means current authority cannot be established and fails
/// closed. Production callers should pass the host's authoritative policy/trust
/// source, not synthesize snapshots at the call site.
pub trait AdmissionCurrentnessSource: Debug + Send + Sync {
    fn current_context(&self, subject: AdmissionSubject<'_>) -> Option<AdmissionContext>;
}

/// In-process host-issued admission authority.
///
/// This type intentionally implements `Serialize` but not `Deserialize`. Code
/// that only possesses stored bytes can reconstruct evidence, not authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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

/// Deserializable audit/transport representation of an admission decision.
///
/// There is deliberately no API that upgrades this type back into
/// [`AdmissionRecord`] or [`ActiveAdmission`]. A host that restarts must rerun
/// package inspection, signer verification, and admission policy evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionRecordEvidence {
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
        canonicalize_capability_grants(&mut granted_capabilities)?;
        validate_record_fields(
            &extension_version,
            &issuer,
            signer.as_ref(),
            &granted_capabilities,
            generation,
            trust_generation,
        )?;

        Ok(Self {
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
        })
    }

    pub fn validate(&self) -> Result<(), AdmissionProblem> {
        validate_record_fields(
            &self.extension_version,
            &self.issuer,
            self.signer.as_ref(),
            &self.granted_capabilities,
            self.generation,
            self.trust_generation,
        )
    }

    /// Prove that this admission cannot widen the extension's declaration.
    pub fn validate_against_manifest(
        &self,
        manifest: &ExtensionManifest,
    ) -> Result<(), AdmissionProblem> {
        self.validate()?;
        validate_record_against_manifest(
            &self.extension,
            &self.extension_version,
            &self.granted_capabilities,
            &self.granted_permissions,
            manifest,
        )
    }

    /// Activate this in-process record using the host's authoritative currentness
    /// source. Raw caller-supplied generation/revocation snapshots are not an
    /// activation API.
    pub fn activate(
        &self,
        manifest: &ExtensionManifest,
        currentness: &dyn AdmissionCurrentnessSource,
    ) -> Result<ActiveAdmission, AdmissionProblem> {
        self.validate_against_manifest(manifest)?;
        let context = currentness
            .current_context(self.subject())
            .ok_or(AdmissionProblem::CurrentnessUnavailable)?;
        validate_currentness(self.generation, self.trust_generation, context)?;
        Ok(ActiveAdmission {
            record: self.clone(),
            manifest: manifest.clone(),
        })
    }

    /// Produce a persistable evidence-only representation.
    pub fn evidence(&self) -> AdmissionRecordEvidence {
        AdmissionRecordEvidence::from(self)
    }

    pub fn subject(&self) -> AdmissionSubject<'_> {
        AdmissionSubject {
            extension: &self.extension,
            extension_version: &self.extension_version,
            issuer: &self.issuer,
            signer: self.signer.as_ref(),
        }
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

impl From<&AdmissionRecord> for AdmissionRecordEvidence {
    fn from(record: &AdmissionRecord) -> Self {
        Self {
            extension: record.extension.clone(),
            extension_version: record.extension_version.clone(),
            manifest_sha256: record.manifest_sha256,
            payload_sha256: record.payload_sha256,
            policy_sha256: record.policy_sha256,
            issuer: record.issuer.clone(),
            signer: record.signer.clone(),
            trust: record.trust,
            granted_capabilities: record.granted_capabilities.clone(),
            granted_permissions: record.granted_permissions.clone(),
            generation: record.generation,
            trust_generation: record.trust_generation,
        }
    }
}

impl AdmissionRecordEvidence {
    /// Validate canonical transport structure only. This establishes no current
    /// authority and intentionally returns no routable token.
    pub fn validate(&self) -> Result<(), AdmissionProblem> {
        validate_record_fields(
            &self.extension_version,
            &self.issuer,
            self.signer.as_ref(),
            &self.granted_capabilities,
            self.generation,
            self.trust_generation,
        )
    }

    pub fn validate_against_manifest(
        &self,
        manifest: &ExtensionManifest,
    ) -> Result<(), AdmissionProblem> {
        self.validate()?;
        validate_record_against_manifest(
            &self.extension,
            &self.extension_version,
            &self.granted_capabilities,
            &self.granted_permissions,
            manifest,
        )
    }

    pub fn extension(&self) -> &ExtensionId {
        &self.extension
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn trust_generation(&self) -> u64 {
        self.trust_generation
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
}

/// Non-serializable proof that one in-process admission was activated from the
/// host's currentness source for one exact manifest. This type intentionally does
/// not implement `Clone` or serde. Long-lived consumers must still call
/// [`ActiveAdmission::recheck_currentness`] at the actual use boundary.
#[derive(Debug, PartialEq, Eq)]
pub struct ActiveAdmission {
    record: AdmissionRecord,
    manifest: ExtensionManifest,
}

impl ActiveAdmission {
    pub fn extension(&self) -> &ExtensionId {
        self.record.extension()
    }
    pub fn subject(&self) -> AdmissionSubject<'_> {
        self.record.subject()
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

    /// Re-resolve current policy/trust/revocation state at the actual use site.
    pub fn recheck_currentness(
        &self,
        currentness: &dyn AdmissionCurrentnessSource,
    ) -> Result<(), AdmissionProblem> {
        let context = currentness
            .current_context(self.subject())
            .ok_or(AdmissionProblem::CurrentnessUnavailable)?;
        validate_currentness(self.generation(), self.trust_generation(), context)
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
    CurrentnessUnavailable,
    Revoked,
    GenerationMismatch { admitted: u64, current: u64 },
    TrustGenerationMismatch { admitted: u64, current: u64 },
}

fn canonicalize_capability_grants(
    capabilities: &mut Vec<CapabilityId>,
) -> Result<(), AdmissionProblem> {
    if capabilities.is_empty() {
        return Err(AdmissionProblem::EmptyCapabilityGrant);
    }
    capabilities.sort();
    let before = capabilities.len();
    capabilities.dedup();
    if capabilities.len() != before {
        return Err(AdmissionProblem::DuplicateCapabilityGrant);
    }
    Ok(())
}

fn validate_record_fields(
    extension_version: &str,
    issuer: &PrincipalId,
    signer: Option<&PrincipalId>,
    granted_capabilities: &[CapabilityId],
    generation: u64,
    trust_generation: u64,
) -> Result<(), AdmissionProblem> {
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
    validate_principal(issuer.as_str())?;
    if let Some(signer) = signer {
        validate_principal(signer.as_str())?;
    }
    if granted_capabilities.is_empty() {
        return Err(AdmissionProblem::EmptyCapabilityGrant);
    }
    if !granted_capabilities.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(AdmissionProblem::NonCanonicalCapabilityGrant);
    }
    Ok(())
}

fn validate_record_against_manifest(
    extension: &ExtensionId,
    extension_version: &str,
    granted_capabilities: &[CapabilityId],
    granted_permissions: &PermissionSet,
    manifest: &ExtensionManifest,
) -> Result<(), AdmissionProblem> {
    manifest
        .validate()
        .map_err(|_| AdmissionProblem::InvalidManifest)?;
    if extension != &manifest.id {
        return Err(AdmissionProblem::ExtensionMismatch);
    }
    if extension_version != manifest.version {
        return Err(AdmissionProblem::VersionMismatch);
    }
    let provided: BTreeSet<_> = manifest
        .provides
        .iter()
        .map(|capability| &capability.id)
        .collect();
    if granted_capabilities
        .iter()
        .any(|capability| !provided.contains(capability))
    {
        return Err(AdmissionProblem::CapabilityNotDeclared);
    }
    if !permissions_are_subset(granted_permissions, &manifest.permissions) {
        return Err(AdmissionProblem::PermissionExceedsManifest);
    }
    Ok(())
}

fn validate_currentness(
    generation: u64,
    trust_generation: u64,
    context: AdmissionContext,
) -> Result<(), AdmissionProblem> {
    if context.revoked {
        return Err(AdmissionProblem::Revoked);
    }
    if context.current_generation != generation {
        return Err(AdmissionProblem::GenerationMismatch {
            admitted: generation,
            current: context.current_generation,
        });
    }
    if context.current_trust_generation != trust_generation {
        return Err(AdmissionProblem::TrustGenerationMismatch {
            admitted: trust_generation,
            current: context.current_trust_generation,
        });
    }
    Ok(())
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

fn filesystem_is_subset(granted: &FilesystemPermission, requested: &FilesystemPermission) -> bool {
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

    #[derive(Debug, Clone, Copy)]
    struct FixedCurrentness(Option<AdmissionContext>);

    impl AdmissionCurrentnessSource for FixedCurrentness {
        fn current_context(&self, subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
            assert_eq!(subject.extension().as_str(), "org.example.solver");
            assert_eq!(subject.extension_version(), "1.0.0");
            assert_eq!(subject.issuer().as_str(), "local:extension-authority");
            assert_eq!(
                subject.signer().map(PrincipalId::as_str),
                Some("did:example:publisher")
            );
            self.0
        }
    }

    fn current(generation: u64, trust_generation: u64) -> FixedCurrentness {
        FixedCurrentness(Some(AdmissionContext::active(
            generation,
            trust_generation,
        )))
    }

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
    fn active_admission_requires_authoritative_currentness_and_exact_manifest() {
        let manifest = manifest();
        let source = current(7, 13);
        let active = record().activate(&manifest, &source).unwrap();
        assert!(active.matches_manifest(&manifest));
        assert_eq!(active.trust_generation(), 13);
        assert!(active.allows_capability(&CapabilityId::new(
            "engineering.simulation.circuit"
        )));
        assert!(!active.allows_capability(&CapabilityId::new(
            "engineering.simulation.process"
        )));
    }

    #[test]
    fn stored_bytes_round_trip_only_as_evidence() {
        let serialized = serde_json::to_vec(&record()).unwrap();
        let evidence: AdmissionRecordEvidence = serde_json::from_slice(&serialized).unwrap();
        evidence.validate().unwrap();
        evidence.validate_against_manifest(&manifest()).unwrap();
        assert_eq!(evidence.generation(), 7);
        assert_eq!(evidence.trust_generation(), 13);
    }

    #[test]
    fn noncanonical_stored_evidence_fails_closed() {
        let mut value = serde_json::to_value(record().evidence()).unwrap();
        value["granted_capabilities"] = serde_json::json!([
            "engineering.simulation.process",
            "engineering.simulation.circuit"
        ]);
        let evidence: AdmissionRecordEvidence = serde_json::from_value(value).unwrap();
        assert_eq!(
            evidence.validate(),
            Err(AdmissionProblem::NonCanonicalCapabilityGrant)
        );
    }

    #[test]
    fn unavailable_revoked_and_generation_changes_fail_closed() {
        let manifest = manifest();
        let record = record();
        assert_eq!(
            record.activate(&manifest, &FixedCurrentness(None)),
            Err(AdmissionProblem::CurrentnessUnavailable)
        );
        assert_eq!(
            record.activate(
                &manifest,
                &FixedCurrentness(Some(AdmissionContext::revoked(7, 13))),
            ),
            Err(AdmissionProblem::Revoked)
        );
        assert!(matches!(
            record.activate(&manifest, &current(8, 13)),
            Err(AdmissionProblem::GenerationMismatch { .. })
        ));
        assert!(matches!(
            record.activate(&manifest, &current(7, 14)),
            Err(AdmissionProblem::TrustGenerationMismatch { .. })
        ));
    }

    #[test]
    fn active_admission_rechecks_live_source_at_use() {
        let manifest = manifest();
        let active = record().activate(&manifest, &current(7, 13)).unwrap();
        assert!(active.recheck_currentness(&current(7, 13)).is_ok());
        assert_eq!(
            active.recheck_currentness(&FixedCurrentness(None)),
            Err(AdmissionProblem::CurrentnessUnavailable)
        );
        assert_eq!(
            active.recheck_currentness(&FixedCurrentness(Some(AdmissionContext::revoked(7, 13)))),
            Err(AdmissionProblem::Revoked)
        );
        assert!(matches!(
            active.recheck_currentness(&current(8, 13)),
            Err(AdmissionProblem::GenerationMismatch { .. })
        ));
        assert!(matches!(
            active.recheck_currentness(&current(7, 14)),
            Err(AdmissionProblem::TrustGenerationMismatch { .. })
        ));
    }

    #[test]
    fn manifest_and_permission_escalation_fail_closed() {
        let mut substituted = manifest();
        substituted.description = "changed after activation".into();
        let active = record().activate(&manifest(), &current(7, 13)).unwrap();
        assert!(!active.matches_manifest(&substituted));

        let mut overgrant = record();
        overgrant.granted_capabilities = vec![CapabilityId::new("robotics.motion.command")];
        assert_eq!(
            overgrant.validate_against_manifest(&manifest()),
            Err(AdmissionProblem::CapabilityNotDeclared)
        );

        let mut permission_overgrant = record();
        permission_overgrant.granted_permissions.network = NetworkPermission::Unrestricted;
        assert_eq!(
            permission_overgrant.validate_against_manifest(&manifest()),
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
}
