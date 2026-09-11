// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Non-executable host for declarative Symthaea extension data packs.
//!
//! Data-only extensions share signer/admission machinery with executable
//! extensions, but they never enter the invocation router and never execute
//! payload bytes. This crate provides the separate technical-inspection and
//! point-of-use path for those packages.

#![deny(unsafe_code)]

use sha2::{Digest, Sha256};
use std::fmt::Debug;
use symthaea_extension_admission::{
    ActiveAdmission, AdmissionCurrentnessSource, Sha256Digest,
};
use symthaea_extension_admission_policy::{TechnicalInspection, TechnicalInspector};
use symthaea_extension_authority::{AuthorityScope, ScopedAdmission, ScopedAdmissionError};
use symthaea_extension_core::{CapabilityId, EffectClass, ExtensionManifest, RuntimeKind};
use thiserror::Error;

/// Independent local ceilings for declarative package inspection/loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataHostPolicy {
    pub max_manifest_bytes: usize,
    pub max_payload_bytes: usize,
}

impl Default for DataHostPolicy {
    fn default() -> Self {
        Self {
            max_manifest_bytes: 256 * 1024,
            max_payload_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Capability-specific validation boundary for one declarative payload shape.
pub trait DataPackValidator: Debug + Send + Sync {
    type Error: Debug;

    fn capability(&self) -> &CapabilityId;
    fn validate(&self, payload: &[u8]) -> Result<(), Self::Error>;
}

/// Runtime-neutral data host backed by one capability-specific validator and one
/// process-local verification scope.
#[derive(Debug)]
pub struct DataPackHost<V> {
    policy: DataHostPolicy,
    validator: V,
    authority: AuthorityScope,
}

impl<V> DataPackHost<V> {
    pub fn new(validator: V, authority: AuthorityScope) -> Self {
        Self {
            policy: DataHostPolicy::default(),
            validator,
            authority,
        }
    }

    pub fn with_policy(validator: V, authority: AuthorityScope, policy: DataHostPolicy) -> Self {
        Self {
            policy,
            validator,
            authority,
        }
    }

    pub const fn policy(&self) -> DataHostPolicy {
        self.policy
    }

    pub const fn validator(&self) -> &V {
        &self.validator
    }

    pub const fn authority_scope(&self) -> &AuthorityScope {
        &self.authority
    }
}

/// Non-serializable point-of-use handle over bytes that have passed authority-
/// scope identity, current admission binding and capability-specific validation.
#[derive(Debug)]
pub struct ValidatedDataPack<'a> {
    manifest: ExtensionManifest,
    capability: CapabilityId,
    payload: &'a [u8],
    manifest_sha256: Sha256Digest,
    payload_sha256: Sha256Digest,
    admission_generation: u64,
    trust_generation: u64,
}

impl<'a> ValidatedDataPack<'a> {
    pub fn manifest(&self) -> &ExtensionManifest {
        &self.manifest
    }

    pub fn capability(&self) -> &CapabilityId {
        &self.capability
    }

    pub const fn payload(&self) -> &'a [u8] {
        self.payload
    }

    pub const fn manifest_sha256(&self) -> Sha256Digest {
        self.manifest_sha256
    }

    pub const fn payload_sha256(&self) -> Sha256Digest {
        self.payload_sha256
    }

    pub const fn admission_generation(&self) -> u64 {
        self.admission_generation
    }

    pub const fn trust_generation(&self) -> u64 {
        self.trust_generation
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DataHostError {
    #[error("manifest exceeds data-host size limit")]
    ManifestTooLarge,
    #[error("payload exceeds data-host size limit")]
    PayloadTooLarge,
    #[error("manifest JSON is invalid: {0}")]
    ManifestJson(String),
    #[error("manifest structural validation failed")]
    ManifestInvalid,
    #[error("data host accepts only data_only runtime extensions")]
    RuntimeNotDataOnly,
    #[error("validator capability is not declared by the manifest")]
    CapabilityNotDeclared,
    #[error("data-only capability must be pure")]
    CapabilityNotPure,
    #[error("capability-specific payload validation failed: {0}")]
    PayloadValidation(String),
    #[error("scoped admission rejected: {0}")]
    AdmissionAuthority(ScopedAdmissionError),
    #[error("active admission does not match the exact manifest")]
    AdmissionManifestMismatch,
    #[error("active admission does not grant the data capability")]
    CapabilityNotAdmitted,
    #[error("exact manifest bytes do not match the admission digest")]
    ManifestDigestMismatch,
    #[error("exact payload bytes do not match the admission digest")]
    PayloadDigestMismatch,
}

impl<V> DataPackHost<V>
where
    V: DataPackValidator,
{
    /// Open a declarative pack only after process-local scope identity and live
    /// currentness are proven for this exact operation.
    ///
    /// The raw `ActiveAdmission` borrow exists only inside the higher-ranked
    /// authority callback; it cannot escape this call as a bearer reference.
    pub fn open<'a>(
        &self,
        manifest_bytes: &[u8],
        payload_bytes: &'a [u8],
        admission: &ScopedAdmission,
        currentness: &dyn AdmissionCurrentnessSource,
    ) -> Result<ValidatedDataPack<'a>, DataHostError> {
        self.authority
            .with_rechecked(admission, currentness, |active| {
                self.open_checked(manifest_bytes, payload_bytes, active)
            })
            .map_err(DataHostError::AdmissionAuthority)?
    }

    fn open_checked<'a>(
        &self,
        manifest_bytes: &[u8],
        payload_bytes: &'a [u8],
        admission: &ActiveAdmission,
    ) -> Result<ValidatedDataPack<'a>, DataHostError> {
        let manifest = self.validate_envelope(manifest_bytes, payload_bytes.len())?;

        if !admission.matches_manifest(&manifest) {
            return Err(DataHostError::AdmissionManifestMismatch);
        }
        if !admission.allows_capability(self.validator.capability()) {
            return Err(DataHostError::CapabilityNotAdmitted);
        }

        let manifest_sha256 = Sha256Digest::new(sha256(manifest_bytes));
        if admission.manifest_sha256() != manifest_sha256 {
            return Err(DataHostError::ManifestDigestMismatch);
        }

        let payload_sha256 = Sha256Digest::new(sha256(payload_bytes));
        if admission.payload_sha256() != payload_sha256 {
            return Err(DataHostError::PayloadDigestMismatch);
        }

        self.validator
            .validate(payload_bytes)
            .map_err(|error| DataHostError::PayloadValidation(format!("{error:?}")))?;

        Ok(ValidatedDataPack {
            manifest,
            capability: self.validator.capability().clone(),
            payload: payload_bytes,
            manifest_sha256,
            payload_sha256,
            admission_generation: admission.generation(),
            trust_generation: admission.trust_generation(),
        })
    }

    fn validate_envelope(
        &self,
        manifest_bytes: &[u8],
        payload_len: usize,
    ) -> Result<ExtensionManifest, DataHostError> {
        if manifest_bytes.len() > self.policy.max_manifest_bytes {
            return Err(DataHostError::ManifestTooLarge);
        }
        if payload_len > self.policy.max_payload_bytes {
            return Err(DataHostError::PayloadTooLarge);
        }

        let manifest: ExtensionManifest = serde_json::from_slice(manifest_bytes)
            .map_err(|error| DataHostError::ManifestJson(error.to_string()))?;
        manifest
            .validate()
            .map_err(|_| DataHostError::ManifestInvalid)?;

        if manifest.runtime != RuntimeKind::DataOnly {
            return Err(DataHostError::RuntimeNotDataOnly);
        }

        let capability = manifest
            .provides
            .iter()
            .find(|candidate| &candidate.id == self.validator.capability())
            .ok_or(DataHostError::CapabilityNotDeclared)?;
        if capability.effect != EffectClass::Pure {
            return Err(DataHostError::CapabilityNotPure);
        }

        Ok(manifest)
    }
}

impl<V> TechnicalInspector for DataPackHost<V>
where
    V: DataPackValidator,
{
    type Error = DataHostError;

    fn inspect(
        &self,
        manifest_bytes: &[u8],
        payload_bytes: &[u8],
    ) -> Result<TechnicalInspection, Self::Error> {
        let manifest = self.validate_envelope(manifest_bytes, payload_bytes.len())?;
        self.validator
            .validate(payload_bytes)
            .map_err(|error| DataHostError::PayloadValidation(format!("{error:?}")))?;

        Ok(TechnicalInspection::new(
            manifest,
            Sha256Digest::new(sha256(manifest_bytes)),
            Sha256Digest::new(sha256(payload_bytes)),
        ))
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_extension_admission::{
        AdmissionContext, AdmissionCurrentnessSource, AdmissionProblem, AdmissionRecord,
        AdmissionSubject, PrincipalId, TrustLevel,
    };
    use symthaea_extension_authority::{AdmissionAuthority, AuthorityScopeError};
    use symthaea_extension_core::PermissionSet;

    #[derive(Debug)]
    struct PrefixValidator {
        capability: CapabilityId,
        prefix: &'static [u8],
    }

    impl PrefixValidator {
        fn new(capability: &str, prefix: &'static [u8]) -> Self {
            Self {
                capability: CapabilityId::new(capability),
                prefix,
            }
        }
    }

    impl DataPackValidator for PrefixValidator {
        type Error = &'static str;

        fn capability(&self) -> &CapabilityId {
            &self.capability
        }

        fn validate(&self, payload: &[u8]) -> Result<(), Self::Error> {
            if payload.starts_with(self.prefix) {
                Ok(())
            } else {
                Err("unexpected payload prefix")
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct FixedCurrentness(Option<AdmissionContext>);

    impl AdmissionCurrentnessSource for FixedCurrentness {
        fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
            self.0
        }
    }

    fn current(generation: u64, trust_generation: u64) -> FixedCurrentness {
        FixedCurrentness(Some(AdmissionContext::active(
            generation,
            trust_generation,
        )))
    }

    fn manifest_bytes(capability: &str) -> Vec<u8> {
        format!(
            r#"{{
                "id":"org.example.knowledge",
                "name":"Example Knowledge",
                "version":"1.0.0",
                "kind":"knowledge_pack",
                "runtime":"data_only",
                "provides":[{{
                    "id":"{capability}",
                    "description":"Example validated knowledge",
                    "effect":"pure"
                }}],
                "resources":{{
                    "memory_bytes":0,
                    "fuel":0,
                    "max_wall_time_ms":0,
                    "max_output_bytes":0,
                    "max_concurrency":0
                }}
            }}"#
        )
        .into_bytes()
    }

    fn record(manifest_bytes: &[u8], payload: &[u8], capability: &str) -> AdmissionRecord {
        let manifest: ExtensionManifest = serde_json::from_slice(manifest_bytes).unwrap();
        AdmissionRecord::issue(
            manifest.id.clone(),
            manifest.version.clone(),
            Sha256Digest::new(sha256(manifest_bytes)),
            Sha256Digest::new(sha256(payload)),
            Sha256Digest::new([3; 32]),
            PrincipalId::new("local:test-authority").unwrap(),
            Some(PrincipalId::new("did:example:data-publisher").unwrap()),
            TrustLevel::Community,
            vec![CapabilityId::new(capability)],
            PermissionSet::default(),
            7,
            11,
        )
        .unwrap()
    }

    fn scoped_admission(
        authority: &AdmissionAuthority,
        manifest_bytes: &[u8],
        payload: &[u8],
        capability: &str,
    ) -> ScopedAdmission {
        let manifest: ExtensionManifest = serde_json::from_slice(manifest_bytes).unwrap();
        authority
            .activate(
                &record(manifest_bytes, payload, capability),
                &manifest,
                &current(7, 11),
            )
            .unwrap()
    }

    fn host(
        capability: &str,
        prefix: &'static [u8],
        authority: &AdmissionAuthority,
    ) -> DataPackHost<PrefixValidator> {
        DataPackHost::new(
            PrefixValidator::new(capability, prefix),
            authority.scope(),
        )
    }

    #[test]
    fn technical_inspection_binds_exact_data_bytes() {
        let capability = "knowledge.example.records";
        let manifest = manifest_bytes(capability);
        let payload = b"records:v1\nalpha";
        let authority = AdmissionAuthority::new();
        let host = host(capability, b"records:v1", &authority);

        let inspection = TechnicalInspector::inspect(&host, &manifest, payload).unwrap();
        assert_eq!(
            inspection.manifest_sha256(),
            Sha256Digest::new(sha256(&manifest))
        );
        assert_eq!(
            inspection.payload_sha256(),
            Sha256Digest::new(sha256(payload))
        );
    }

    #[test]
    fn point_of_use_open_requires_exact_admitted_payload() {
        let capability = "knowledge.example.records";
        let manifest = manifest_bytes(capability);
        let admitted_payload = b"records:v1\nalpha";
        let authority = AdmissionAuthority::new();
        let admission = scoped_admission(&authority, &manifest, admitted_payload, capability);
        let host = host(capability, b"records:v1", &authority);

        let error = host
            .open(
                &manifest,
                b"records:v1\nbeta",
                &admission,
                &current(7, 11),
            )
            .unwrap_err();
        assert_eq!(error, DataHostError::PayloadDigestMismatch);
    }

    #[test]
    fn admitted_data_pack_opens_without_execution() {
        let capability = "knowledge.example.records";
        let manifest = manifest_bytes(capability);
        let payload = b"records:v1\nalpha";
        let authority = AdmissionAuthority::new();
        let admission = scoped_admission(&authority, &manifest, payload, capability);
        let host = host(capability, b"records:v1", &authority);

        let opened = host
            .open(&manifest, payload, &admission, &current(7, 11))
            .unwrap();
        assert_eq!(opened.capability(), &CapabilityId::new(capability));
        assert_eq!(opened.payload(), payload);
        assert_eq!(opened.payload_sha256(), Sha256Digest::new(sha256(payload)));
        assert_eq!(opened.admission_generation(), 7);
        assert_eq!(opened.trust_generation(), 11);
    }

    #[test]
    fn foreign_authority_fails_before_payload_validation() {
        let capability = "knowledge.example.records";
        let manifest = manifest_bytes(capability);
        let payload = b"wrong-format";
        let host_authority = AdmissionAuthority::new();
        let foreign_authority = AdmissionAuthority::new();
        let foreign = scoped_admission(&foreign_authority, &manifest, payload, capability);
        let host = host(capability, b"records:v1", &host_authority);

        let error = host
            .open(&manifest, payload, &foreign, &current(7, 11))
            .unwrap_err();
        assert_eq!(
            error,
            DataHostError::AdmissionAuthority(ScopedAdmissionError::Scope(
                AuthorityScopeError::ForeignAuthority
            ))
        );
    }

    #[test]
    fn point_of_use_rechecks_missing_and_revoked_authority() {
        let capability = "knowledge.example.records";
        let manifest = manifest_bytes(capability);
        let payload = b"records:v1\nalpha";
        let authority = AdmissionAuthority::new();
        let admission = scoped_admission(&authority, &manifest, payload, capability);
        let host = host(capability, b"records:v1", &authority);

        let unavailable = host
            .open(&manifest, payload, &admission, &FixedCurrentness(None))
            .unwrap_err();
        assert_eq!(
            unavailable,
            DataHostError::AdmissionAuthority(ScopedAdmissionError::Currentness(
                AdmissionProblem::CurrentnessUnavailable
            ))
        );

        let revoked = FixedCurrentness(Some(AdmissionContext::revoked(7, 11)));
        let error = host
            .open(&manifest, payload, &admission, &revoked)
            .unwrap_err();
        assert_eq!(
            error,
            DataHostError::AdmissionAuthority(ScopedAdmissionError::Currentness(
                AdmissionProblem::Revoked
            ))
        );
    }

    #[test]
    fn point_of_use_rechecks_policy_and_trust_generations() {
        let capability = "knowledge.example.records";
        let manifest = manifest_bytes(capability);
        let payload = b"records:v1\nalpha";
        let authority = AdmissionAuthority::new();
        let admission = scoped_admission(&authority, &manifest, payload, capability);
        let host = host(capability, b"records:v1", &authority);

        let policy_error = host
            .open(&manifest, payload, &admission, &current(8, 11))
            .unwrap_err();
        assert!(matches!(
            policy_error,
            DataHostError::AdmissionAuthority(ScopedAdmissionError::Currentness(
                AdmissionProblem::GenerationMismatch {
                    admitted: 7,
                    current: 8
                }
            ))
        ));

        let trust_error = host
            .open(&manifest, payload, &admission, &current(7, 12))
            .unwrap_err();
        assert!(matches!(
            trust_error,
            DataHostError::AdmissionAuthority(ScopedAdmissionError::Currentness(
                AdmissionProblem::TrustGenerationMismatch {
                    admitted: 11,
                    current: 12
                }
            ))
        ));
    }

    #[test]
    fn validator_failure_is_fail_closed() {
        let capability = "knowledge.example.records";
        let manifest = manifest_bytes(capability);
        let payload = b"wrong-format";
        let authority = AdmissionAuthority::new();
        let host = host(capability, b"records:v1", &authority);

        let error = TechnicalInspector::inspect(&host, &manifest, payload).unwrap_err();
        assert!(matches!(error, DataHostError::PayloadValidation(_)));
    }

    #[test]
    fn data_host_never_accepts_executable_runtime() {
        let capability = "knowledge.example.records";
        let manifest = String::from_utf8(manifest_bytes(capability))
            .unwrap()
            .replace("\"data_only\"", "\"wasm\"")
            .replace("\"memory_bytes\":0", "\"memory_bytes\":1")
            .replace("\"fuel\":0", "\"fuel\":1")
            .replace("\"max_wall_time_ms\":0", "\"max_wall_time_ms\":1")
            .replace("\"max_output_bytes\":0", "\"max_output_bytes\":1")
            .replace("\"max_concurrency\":0", "\"max_concurrency\":1")
            .into_bytes();
        let authority = AdmissionAuthority::new();
        let host = host(capability, b"records:v1", &authority);

        let error = TechnicalInspector::inspect(&host, &manifest, b"records:v1").unwrap_err();
        assert_eq!(error, DataHostError::RuntimeNotDataOnly);
    }
}
