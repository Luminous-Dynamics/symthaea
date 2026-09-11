// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed admission policy evaluation for Symthaea extensions.
//!
//! This crate composes two explicit trust dependencies:
//!
//! - [`TechnicalInspector`] proves the exact candidate passed the configured
//!   runtime/format compatibility boundary;
//! - [`SignerVerifier`] authenticates the exact package commitment and resolves
//!   a signer principal, trust ceiling, and current trust generation.
//!
//! The evaluator then independently re-parses/re-hashes the candidate, applies
//! local attenuation policy, and issues an in-process [`AdmissionRecord`].

#![deny(unsafe_code)]

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Debug;
use symthaea_extension_admission::{
    AdmissionProblem, AdmissionRecord, PrincipalId, Sha256Digest, TrustLevel,
};
use symthaea_extension_core::{
    CapabilityId, ExtensionManifest, FilesystemPermission, NetworkPermission, PermissionSet,
};

const PACKAGE_DOMAIN: &[u8] = b"symthaea.extension.package.v1\0";
const POLICY_DOMAIN: &[u8] = b"symthaea.extension.admission-policy.v1\0";
const MAX_SIGNATURE_BYTES: usize = 1024 * 1024;
const MAX_LABEL_BYTES: usize = 256;

/// Runtime-neutral technical inspection result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TechnicalInspection {
    manifest: ExtensionManifest,
    manifest_sha256: Sha256Digest,
    payload_sha256: Sha256Digest,
}

impl TechnicalInspection {
    pub fn new(
        manifest: ExtensionManifest,
        manifest_sha256: Sha256Digest,
        payload_sha256: Sha256Digest,
    ) -> Self {
        Self {
            manifest,
            manifest_sha256,
            payload_sha256,
        }
    }

    pub fn manifest(&self) -> &ExtensionManifest {
        &self.manifest
    }

    pub fn manifest_sha256(&self) -> Sha256Digest {
        self.manifest_sha256
    }

    pub fn payload_sha256(&self) -> Sha256Digest {
        self.payload_sha256
    }
}

/// Trusted adapter for runtime-/format-specific compatibility inspection.
///
/// Concrete inspectors should reject excessive input size before expensive
/// parsing/compilation. The generic evaluator calls this boundary before its own
/// independent parse/hash pass.
pub trait TechnicalInspector {
    type Error: Debug;

    fn inspect(
        &self,
        manifest_bytes: &[u8],
        payload_bytes: &[u8],
    ) -> Result<TechnicalInspection, Self::Error>;
}

/// Detached package signature presented to the configured verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureEnvelope {
    pub algorithm: String,
    pub key_id: String,
    pub signature: Vec<u8>,
}

impl SignatureEnvelope {
    fn validate(&self) -> Result<(), AdmissionPolicyError> {
        if !valid_label(&self.algorithm)
            || !valid_label(&self.key_id)
            || self.signature.is_empty()
            || self.signature.len() > MAX_SIGNATURE_BYTES
        {
            return Err(AdmissionPolicyError::InvalidSignatureEnvelope);
        }
        Ok(())
    }
}

/// Domain-separated digest authenticated by [`SignerVerifier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackageCommitment([u8; 32]);

impl PackageCommitment {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Signer authority resolved by the configured verifier/trust source.
///
/// Trust is an authorization ceiling, never a provider-quality score. The trust
/// generation must change whenever signer/key authority changes in a way that
/// should invalidate previously issued admissions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedSigner {
    pub principal: PrincipalId,
    pub trust_ceiling: TrustLevel,
    pub trust_generation: u64,
}

/// Trusted signature + signer-currentness boundary.
pub trait SignerVerifier {
    type Error: Debug;

    fn verify(
        &self,
        commitment: PackageCommitment,
        signature: &SignatureEnvelope,
    ) -> Result<VerifiedSigner, Self::Error>;
}

/// Canonical local authority ceiling for one admission generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionPolicy {
    issuer: PrincipalId,
    generation: u64,
    max_trust: TrustLevel,
    allowed_capabilities: BTreeSet<CapabilityId>,
    permission_ceiling: PermissionSet,
}

impl AdmissionPolicy {
    pub fn new(
        issuer: PrincipalId,
        generation: u64,
        max_trust: TrustLevel,
        allowed_capabilities: impl IntoIterator<Item = CapabilityId>,
        mut permission_ceiling: PermissionSet,
    ) -> Result<Self, AdmissionPolicyError> {
        if generation == 0 {
            return Err(AdmissionPolicyError::ZeroPolicyGeneration);
        }
        canonicalize_permissions(&mut permission_ceiling)?;
        let allowed_capabilities: BTreeSet<_> = allowed_capabilities.into_iter().collect();
        if allowed_capabilities.is_empty() {
            return Err(AdmissionPolicyError::EmptyPolicyCapabilitySet);
        }
        Ok(Self {
            issuer,
            generation,
            max_trust,
            allowed_capabilities,
            permission_ceiling,
        })
    }

    pub fn issuer(&self) -> &PrincipalId {
        &self.issuer
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn max_trust(&self) -> TrustLevel {
        self.max_trust
    }

    pub fn policy_sha256(&self) -> Sha256Digest {
        Sha256Digest::new(sha256(&self.canonical_bytes()))
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(POLICY_DOMAIN);
        out.extend_from_slice(&self.generation.to_be_bytes());
        out.push(trust_tag(self.max_trust));
        put_string(&mut out, self.issuer.as_str());
        put_len(&mut out, self.allowed_capabilities.len());
        for capability in &self.allowed_capabilities {
            put_string(&mut out, capability.as_str());
        }
        put_permissions(&mut out, &self.permission_ceiling);
        out
    }
}

/// Requested authority for one admitted package. It may be narrower than both
/// the manifest declaration and local policy ceiling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionGrant {
    pub trust: TrustLevel,
    pub capabilities: Vec<CapabilityId>,
    pub permissions: PermissionSet,
}

#[derive(Debug)]
pub struct AdmissionEvaluator<I, V> {
    inspector: I,
    verifier: V,
}

impl<I, V> AdmissionEvaluator<I, V> {
    pub fn new(inspector: I, verifier: V) -> Self {
        Self { inspector, verifier }
    }
}

impl<I, V> AdmissionEvaluator<I, V>
where
    I: TechnicalInspector,
    V: SignerVerifier,
{
    /// Evaluate exact candidate bytes and mint one in-process admission record.
    ///
    /// The resulting `AdmissionRecord` may be serialized for evidence but cannot
    /// be deserialized back into authority. A restart must rerun this method.
    pub fn evaluate(
        &self,
        manifest_bytes: &[u8],
        payload_bytes: &[u8],
        signature: &SignatureEnvelope,
        policy: &AdmissionPolicy,
        mut grant: AdmissionGrant,
    ) -> Result<AdmissionRecord, AdmissionPolicyError> {
        signature.validate()?;

        // Runtime-specific inspection receives first refusal so its size/format/
        // sandbox limits can reject hostile inputs before this generic layer
        // performs a second parser/hash pass.
        let inspection = self
            .inspector
            .inspect(manifest_bytes, payload_bytes)
            .map_err(|error| AdmissionPolicyError::InspectionFailed(format!("{error:?}")))?;

        // Independently parse and hash exact caller bytes. The inspector is a
        // trust dependency, but it is not allowed to substitute semantic input.
        let manifest: ExtensionManifest = serde_json::from_slice(manifest_bytes)
            .map_err(|error| AdmissionPolicyError::ManifestJson(error.to_string()))?;
        manifest
            .validate()
            .map_err(|_| AdmissionPolicyError::ManifestInvalid)?;
        let manifest_sha = Sha256Digest::new(sha256(manifest_bytes));
        let payload_sha = Sha256Digest::new(sha256(payload_bytes));

        if inspection.manifest() != &manifest {
            return Err(AdmissionPolicyError::InspectionManifestMismatch);
        }
        if inspection.manifest_sha256() != manifest_sha {
            return Err(AdmissionPolicyError::InspectionManifestDigestMismatch);
        }
        if inspection.payload_sha256() != payload_sha {
            return Err(AdmissionPolicyError::InspectionPayloadDigestMismatch);
        }

        let verified = self
            .verifier
            .verify(package_commitment(manifest_sha, payload_sha), signature)
            .map_err(|error| AdmissionPolicyError::SignerVerificationFailed(format!("{error:?}")))?;
        if verified.trust_generation == 0 {
            return Err(AdmissionPolicyError::ZeroTrustGeneration);
        }
        if grant.trust > policy.max_trust || grant.trust > verified.trust_ceiling {
            return Err(AdmissionPolicyError::TrustExceedsCeiling);
        }

        if grant.capabilities.is_empty() {
            return Err(AdmissionPolicyError::EmptyGrantCapabilitySet);
        }
        grant.capabilities.sort();
        let before = grant.capabilities.len();
        grant.capabilities.dedup();
        if grant.capabilities.len() != before {
            return Err(AdmissionPolicyError::DuplicateGrantCapability);
        }
        if grant
            .capabilities
            .iter()
            .any(|capability| !policy.allowed_capabilities.contains(capability))
        {
            return Err(AdmissionPolicyError::CapabilityOutsidePolicy);
        }

        canonicalize_permissions(&mut grant.permissions)?;
        if !permissions_are_subset(&grant.permissions, &policy.permission_ceiling) {
            return Err(AdmissionPolicyError::PermissionOutsidePolicy);
        }

        let record = AdmissionRecord::issue(
            manifest.id.clone(),
            manifest.version.clone(),
            manifest_sha,
            payload_sha,
            policy.policy_sha256(),
            policy.issuer.clone(),
            Some(verified.principal),
            grant.trust,
            grant.capabilities,
            grant.permissions,
            policy.generation,
            verified.trust_generation,
        )
        .map_err(AdmissionPolicyError::AdmissionContract)?;

        record
            .validate_against_manifest(&manifest)
            .map_err(AdmissionPolicyError::AdmissionContract)?;
        Ok(record)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionPolicyError {
    ManifestJson(String),
    ManifestInvalid,
    InvalidSignatureEnvelope,
    InspectionFailed(String),
    InspectionManifestMismatch,
    InspectionManifestDigestMismatch,
    InspectionPayloadDigestMismatch,
    SignerVerificationFailed(String),
    ZeroPolicyGeneration,
    ZeroTrustGeneration,
    EmptyPolicyCapabilitySet,
    InvalidPermissionEntry,
    TrustExceedsCeiling,
    EmptyGrantCapabilitySet,
    DuplicateGrantCapability,
    CapabilityOutsidePolicy,
    PermissionOutsidePolicy,
    AdmissionContract(AdmissionProblem),
}

pub fn package_commitment(
    manifest_sha256: Sha256Digest,
    payload_sha256: Sha256Digest,
) -> PackageCommitment {
    let mut hasher = Sha256::new();
    hasher.update(PACKAGE_DOMAIN);
    hasher.update(manifest_sha256.as_bytes());
    hasher.update(payload_sha256.as_bytes());
    PackageCommitment(hasher.finalize().into())
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn canonicalize_permissions(permissions: &mut PermissionSet) -> Result<(), AdmissionPolicyError> {
    match &mut permissions.network {
        NetworkPermission::None | NetworkPermission::Unrestricted => {}
        NetworkPermission::Allowlist(values) => canonicalize_strings(values)?,
    }
    match &mut permissions.filesystem {
        FilesystemPermission::None => {}
        FilesystemPermission::ReadOnly(values) | FilesystemPermission::ReadWrite(values) => {
            canonicalize_strings(values)?
        }
    }
    canonicalize_strings(&mut permissions.sensors)?;
    canonicalize_strings(&mut permissions.actuators)?;
    Ok(())
}

fn canonicalize_strings(values: &mut Vec<String>) -> Result<(), AdmissionPolicyError> {
    if values.iter().any(|value| !valid_label(value)) {
        return Err(AdmissionPolicyError::InvalidPermissionEntry);
    }
    values.sort();
    values.dedup();
    Ok(())
}

fn permissions_are_subset(granted: &PermissionSet, ceiling: &PermissionSet) -> bool {
    network_is_subset(&granted.network, &ceiling.network)
        && filesystem_is_subset(&granted.filesystem, &ceiling.filesystem)
        && (!granted.gpu || ceiling.gpu)
        && (!granted.wall_clock || ceiling.wall_clock)
        && (!granted.randomness || ceiling.randomness)
        && strings_are_subset(&granted.sensors, &ceiling.sensors)
        && strings_are_subset(&granted.actuators, &ceiling.actuators)
}

fn network_is_subset(granted: &NetworkPermission, ceiling: &NetworkPermission) -> bool {
    match (granted, ceiling) {
        (NetworkPermission::None, _) => true,
        (NetworkPermission::Allowlist(granted), NetworkPermission::Allowlist(ceiling)) => {
            strings_are_subset(granted, ceiling)
        }
        (NetworkPermission::Allowlist(_), NetworkPermission::Unrestricted) => true,
        (NetworkPermission::Unrestricted, NetworkPermission::Unrestricted) => true,
        _ => false,
    }
}

fn filesystem_is_subset(granted: &FilesystemPermission, ceiling: &FilesystemPermission) -> bool {
    match (granted, ceiling) {
        (FilesystemPermission::None, _) => true,
        (FilesystemPermission::ReadOnly(granted), FilesystemPermission::ReadOnly(ceiling))
        | (FilesystemPermission::ReadOnly(granted), FilesystemPermission::ReadWrite(ceiling))
        | (FilesystemPermission::ReadWrite(granted), FilesystemPermission::ReadWrite(ceiling)) => {
            strings_are_subset(granted, ceiling)
        }
        _ => false,
    }
}

fn strings_are_subset(granted: &[String], ceiling: &[String]) -> bool {
    granted.iter().all(|item| ceiling.binary_search(item).is_ok())
}

fn valid_label(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= MAX_LABEL_BYTES
        && !value.chars().any(|character| character.is_control())
}

fn trust_tag(trust: TrustLevel) -> u8 {
    match trust {
        TrustLevel::Untrusted => 0,
        TrustLevel::Community => 1,
        TrustLevel::Trusted => 2,
        TrustLevel::Privileged => 3,
    }
}

fn put_len(out: &mut Vec<u8>, value: usize) {
    let value = u32::try_from(value).expect("policy collections fit in u32");
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

fn put_strings(out: &mut Vec<u8>, values: &[String]) {
    put_len(out, values.len());
    for value in values {
        put_string(out, value);
    }
}

fn put_permissions(out: &mut Vec<u8>, permissions: &PermissionSet) {
    match &permissions.network {
        NetworkPermission::None => out.push(0),
        NetworkPermission::Allowlist(values) => {
            out.push(1);
            put_strings(out, values);
        }
        NetworkPermission::Unrestricted => out.push(2),
    }
    match &permissions.filesystem {
        FilesystemPermission::None => out.push(0),
        FilesystemPermission::ReadOnly(values) => {
            out.push(1);
            put_strings(out, values);
        }
        FilesystemPermission::ReadWrite(values) => {
            out.push(2);
            put_strings(out, values);
        }
    }
    out.push(u8::from(permissions.gpu));
    out.push(u8::from(permissions.wall_clock));
    out.push(u8::from(permissions.randomness));
    put_strings(out, &permissions.sensors);
    put_strings(out, &permissions.actuators);
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_extension_admission::{
        AdmissionContext, AdmissionCurrentnessSource, AdmissionRecordEvidence, AdmissionSubject,
    };
    use symthaea_extension_core::{
        AbiVersion, CapabilityDescriptor, EffectClass, ExtensionId, ExtensionKind, ResourceBudget,
        RuntimeKind,
    };

    #[derive(Debug)]
    struct Inspector {
        corrupt_manifest: bool,
        corrupt_payload: bool,
        fail: bool,
    }

    impl TechnicalInspector for Inspector {
        type Error = &'static str;

        fn inspect(
            &self,
            manifest_bytes: &[u8],
            payload_bytes: &[u8],
        ) -> Result<TechnicalInspection, Self::Error> {
            if self.fail {
                return Err("rejected by runtime inspector");
            }
            let mut manifest: ExtensionManifest =
                serde_json::from_slice(manifest_bytes).map_err(|_| "manifest")?;
            if self.corrupt_manifest {
                manifest.description = "substituted".into();
            }
            Ok(TechnicalInspection::new(
                manifest,
                Sha256Digest::new(sha256(manifest_bytes)),
                if self.corrupt_payload {
                    Sha256Digest::new([9; 32])
                } else {
                    Sha256Digest::new(sha256(payload_bytes))
                },
            ))
        }
    }

    #[derive(Debug)]
    struct Verifier {
        fail: bool,
        ceiling: TrustLevel,
        generation: u64,
    }

    impl SignerVerifier for Verifier {
        type Error = &'static str;

        fn verify(
            &self,
            _commitment: PackageCommitment,
            _signature: &SignatureEnvelope,
        ) -> Result<VerifiedSigner, Self::Error> {
            if self.fail {
                return Err("bad signature");
            }
            Ok(VerifiedSigner {
                principal: PrincipalId::new("did:example:publisher").unwrap(),
                trust_ceiling: self.ceiling,
                trust_generation: self.generation,
            })
        }
    }

    #[derive(Debug)]
    struct Currentness;

    impl AdmissionCurrentnessSource for Currentness {
        fn current_context(&self, subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
            assert_eq!(subject.extension().as_str(), "org.example.solver");
            assert_eq!(subject.extension_version(), "1.0.0");
            assert_eq!(subject.issuer().as_str(), "local:extension-authority");
            assert_eq!(
                subject.signer().map(PrincipalId::as_str),
                Some("did:example:publisher")
            );
            Some(AdmissionContext::active(4, 11))
        }
    }

    fn manifest() -> ExtensionManifest {
        ExtensionManifest {
            id: ExtensionId::new("org.example.solver"),
            name: "Example".into(),
            version: "1.0.0".into(),
            abi: AbiVersion::V1,
            kind: ExtensionKind::Simulation,
            runtime: RuntimeKind::Wasm,
            description: String::new(),
            provides: vec![CapabilityDescriptor {
                id: CapabilityId::new("engineering.simulation.circuit"),
                description: "circuit simulation".into(),
                effect: EffectClass::Pure,
            }],
            requires: vec![],
            permissions: PermissionSet {
                network: NetworkPermission::Allowlist(vec!["solver.example".into()]),
                filesystem: FilesystemPermission::ReadOnly(vec!["/models".into()]),
                ..PermissionSet::default()
            },
            resources: ResourceBudget::default(),
        }
    }

    fn bytes() -> Vec<u8> {
        serde_json::to_vec(&manifest()).unwrap()
    }

    fn signature() -> SignatureEnvelope {
        SignatureEnvelope {
            algorithm: "test-ed25519".into(),
            key_id: "publisher-1".into(),
            signature: vec![1, 2, 3],
        }
    }

    fn policy() -> AdmissionPolicy {
        AdmissionPolicy::new(
            PrincipalId::new("local:extension-authority").unwrap(),
            4,
            TrustLevel::Trusted,
            [CapabilityId::new("engineering.simulation.circuit")],
            PermissionSet {
                network: NetworkPermission::Allowlist(vec!["solver.example".into()]),
                filesystem: FilesystemPermission::ReadOnly(vec!["/models".into()]),
                ..PermissionSet::default()
            },
        )
        .unwrap()
    }

    fn grant() -> AdmissionGrant {
        AdmissionGrant {
            trust: TrustLevel::Community,
            capabilities: vec![CapabilityId::new("engineering.simulation.circuit")],
            permissions: PermissionSet::default(),
        }
    }

    fn evaluator() -> AdmissionEvaluator<Inspector, Verifier> {
        AdmissionEvaluator::new(
            Inspector {
                corrupt_manifest: false,
                corrupt_payload: false,
                fail: false,
            },
            Verifier {
                fail: false,
                ceiling: TrustLevel::Trusted,
                generation: 11,
            },
        )
    }

    #[test]
    fn pipeline_issues_live_record_but_persists_evidence_only() {
        let manifest_bytes = bytes();
        let record = evaluator()
            .evaluate(
                &manifest_bytes,
                b"component",
                &signature(),
                &policy(),
                grant(),
            )
            .unwrap();
        assert_eq!(record.generation(), 4);
        assert_eq!(record.trust_generation(), 11);

        let parsed: ExtensionManifest = serde_json::from_slice(&manifest_bytes).unwrap();
        assert!(record.activate(&parsed, &Currentness).is_ok());

        let stored = serde_json::to_vec(&record).unwrap();
        let evidence: AdmissionRecordEvidence = serde_json::from_slice(&stored).unwrap();
        evidence.validate().unwrap();
        assert_eq!(evidence.generation(), 4);
        assert_eq!(evidence.trust_generation(), 11);
    }

    #[test]
    fn technical_inspection_failure_or_substitution_fails_closed() {
        let rejected = AdmissionEvaluator::new(
            Inspector {
                corrupt_manifest: false,
                corrupt_payload: false,
                fail: true,
            },
            Verifier {
                fail: false,
                ceiling: TrustLevel::Trusted,
                generation: 11,
            },
        );
        assert!(matches!(
            rejected.evaluate(&bytes(), b"payload", &signature(), &policy(), grant()),
            Err(AdmissionPolicyError::InspectionFailed(_))
        ));

        let substituted = AdmissionEvaluator::new(
            Inspector {
                corrupt_manifest: true,
                corrupt_payload: false,
                fail: false,
            },
            Verifier {
                fail: false,
                ceiling: TrustLevel::Trusted,
                generation: 11,
            },
        );
        assert_eq!(
            substituted.evaluate(&bytes(), b"payload", &signature(), &policy(), grant()),
            Err(AdmissionPolicyError::InspectionManifestMismatch)
        );
    }

    #[test]
    fn signer_failure_and_zero_generation_fail_closed() {
        let failed = AdmissionEvaluator::new(
            Inspector {
                corrupt_manifest: false,
                corrupt_payload: false,
                fail: false,
            },
            Verifier {
                fail: true,
                ceiling: TrustLevel::Trusted,
                generation: 11,
            },
        );
        assert!(matches!(
            failed.evaluate(&bytes(), b"payload", &signature(), &policy(), grant()),
            Err(AdmissionPolicyError::SignerVerificationFailed(_))
        ));

        let stale = AdmissionEvaluator::new(
            Inspector {
                corrupt_manifest: false,
                corrupt_payload: false,
                fail: false,
            },
            Verifier {
                fail: false,
                ceiling: TrustLevel::Trusted,
                generation: 0,
            },
        );
        assert_eq!(
            stale.evaluate(&bytes(), b"payload", &signature(), &policy(), grant()),
            Err(AdmissionPolicyError::ZeroTrustGeneration)
        );
    }

    #[test]
    fn trust_capability_and_permission_escalation_fail_closed() {
        let low_trust = AdmissionEvaluator::new(
            Inspector {
                corrupt_manifest: false,
                corrupt_payload: false,
                fail: false,
            },
            Verifier {
                fail: false,
                ceiling: TrustLevel::Community,
                generation: 11,
            },
        );
        let mut high = grant();
        high.trust = TrustLevel::Trusted;
        assert_eq!(
            low_trust.evaluate(&bytes(), b"payload", &signature(), &policy(), high),
            Err(AdmissionPolicyError::TrustExceedsCeiling)
        );

        let mut outside = grant();
        outside.capabilities = vec![CapabilityId::new("robotics.motion.command")];
        assert_eq!(
            evaluator().evaluate(&bytes(), b"payload", &signature(), &policy(), outside),
            Err(AdmissionPolicyError::CapabilityOutsidePolicy)
        );

        let mut elevated = grant();
        elevated.permissions.network = NetworkPermission::Unrestricted;
        assert_eq!(
            evaluator().evaluate(&bytes(), b"payload", &signature(), &policy(), elevated),
            Err(AdmissionPolicyError::PermissionOutsidePolicy)
        );
    }

    #[test]
    fn package_and_policy_commitments_are_stable_and_domain_bound() {
        assert_ne!(
            package_commitment(Sha256Digest::new([1; 32]), Sha256Digest::new([2; 32])),
            package_commitment(Sha256Digest::new([1; 32]), Sha256Digest::new([3; 32]))
        );

        let issuer = PrincipalId::new("local:extension-authority").unwrap();
        let a = AdmissionPolicy::new(
            issuer.clone(),
            1,
            TrustLevel::Trusted,
            [CapabilityId::new("z.cap"), CapabilityId::new("a.cap")],
            PermissionSet {
                network: NetworkPermission::Allowlist(vec![
                    "b.example".into(),
                    "a.example".into(),
                ]),
                ..PermissionSet::default()
            },
        )
        .unwrap();
        let b = AdmissionPolicy::new(
            issuer,
            1,
            TrustLevel::Trusted,
            [CapabilityId::new("a.cap"), CapabilityId::new("z.cap")],
            PermissionSet {
                network: NetworkPermission::Allowlist(vec![
                    "a.example".into(),
                    "b.example".into(),
                ]),
                ..PermissionSet::default()
            },
        )
        .unwrap();
        assert_eq!(a.policy_sha256(), b.policy_sha256());
    }
}
