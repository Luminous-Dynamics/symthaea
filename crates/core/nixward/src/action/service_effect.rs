// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Authority-bound identity for one exact Nixward service effect.
//!
//! This module is deliberately independent of authorization, execution, and
//! post-state observation. It defines the immutable contract that all three
//! layers must agree on.

use super::service_domain::{NixServiceOperationKindV1, NixServiceOperationV1};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const SERVICE_EFFECT_CONTEXT_DOMAIN_V1: &[u8] = b"nixward-service-effect-context-v1";
const INVOCATION_ID_HEX_LEN: usize = 32;
const DIGEST_HEX_LEN: usize = 64;
const MAX_STABILITY_WINDOW_US: u64 = 86_400_000_000;
const MAX_STRING_BYTES: usize = 4096;
const MAX_DEFINITION_FILES: usize = 64;
const MAX_DEFINITION_FILE_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixSystemdUnitDefinitionContentFileV1 {
    /// Exact systemd-reported source path whose bytes were captured.
    pub path: String,
    /// Byte length observed while hashing the open file descriptor.
    pub byte_len: u64,
    /// BLAKE3 commitment to the exact bytes read from that descriptor.
    pub content_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixSystemdUnitDefinitionContentEvidenceV1 {
    pub unit: String,
    /// CROSS-067 source identity commitment. Content is intentionally separate.
    pub source_identity_digest: String,
    pub files: Vec<NixSystemdUnitDefinitionContentFileV1>,
    pub captured_at_monotonic_us: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NixVerifiedServiceDefinitionContentV1 {
    evidence: NixSystemdUnitDefinitionContentEvidenceV1,
}

impl NixVerifiedServiceDefinitionContentV1 {
    pub(crate) fn from_observer(
        evidence: NixSystemdUnitDefinitionContentEvidenceV1,
    ) -> Result<Self, NixServiceEffectContextErrorV1> {
        evidence.validate_shape()?;
        Ok(Self { evidence })
    }

    pub(crate) fn as_ref(&self) -> &NixSystemdUnitDefinitionContentEvidenceV1 {
        &self.evidence
    }

    pub(crate) fn digest(&self) -> Result<String, NixServiceEffectContextErrorV1> {
        self.evidence.digest()
    }
}

impl NixSystemdUnitDefinitionContentEvidenceV1 {
    pub fn validate_shape(&self) -> Result<(), NixServiceEffectContextErrorV1> {
        NixServiceOperationV1::new(self.unit.clone(), NixServiceOperationKindV1::Start)
            .map_err(|error| NixServiceEffectContextErrorV1::InvalidServiceUnit(error.to_string()))?;
        if self.files.is_empty() || self.files.len() > MAX_DEFINITION_FILES {
            return Err(NixServiceEffectContextErrorV1::InvalidDefinitionFileSet);
        }
        validate_digest(&self.source_identity_digest, "source identity digest")?;
        let mut seen = std::collections::BTreeSet::new();
        for file in &self.files {
            if file.path.is_empty() || file.path.len() > MAX_STRING_BYTES || !file.path.starts_with('/') {
                return Err(NixServiceEffectContextErrorV1::InvalidDefinitionFilePath);
            }
            if !seen.insert(file.path.clone()) {
                return Err(NixServiceEffectContextErrorV1::DuplicateDefinitionFile);
            }
            if file.byte_len > MAX_DEFINITION_FILE_BYTES {
                return Err(NixServiceEffectContextErrorV1::DefinitionFileTooLarge);
            }
            validate_digest(&file.content_digest, "definition content digest")?;
        }
        if self.captured_at_monotonic_us == 0 {
            return Err(NixServiceEffectContextErrorV1::InvalidCaptureTimestamp);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, NixServiceEffectContextErrorV1> {
        self.validate_shape()?;
        let mut h = Hasher::new();
        h.update(b"nixward-systemd-unit-definition-content-v1");
        put_str(&mut h, &self.unit);
        put_str(&mut h, &self.source_identity_digest);
        put_u64(&mut h, self.files.len() as u64);
        for file in &self.files {
            put_str(&mut h, &file.path);
            put_u64(&mut h, file.byte_len);
            put_str(&mut h, &file.content_digest);
        }
        Ok(h.finalize().to_hex().to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixServiceEffectContextV1 {
    pub operation: NixServiceOperationKindV1,
    pub unit: String,
    pub authorized_generation: u64,
    pub pre_state_digest: String,
    /// CROSS-067 source identity digest (FragmentPath + DropInPaths).
    pub authorized_definition_digest: String,
    /// Byte-level content commitment captured by the observer boundary.
    #[serde(default)]
    pub authorized_definition_content_digest: String,
    pub pre_invocation_id: Option<String>,
    pub required_stability_us: u64,
}

impl NixServiceEffectContextV1 {
    pub fn new(
        operation: NixServiceOperationKindV1,
        unit: impl Into<String>,
        authorized_generation: u64,
        pre_state_digest: impl Into<String>,
        authorized_definition_digest: impl Into<String>,
        authorized_definition_content_digest: impl Into<String>,
        pre_invocation_id: Option<String>,
        required_stability_us: u64,
    ) -> Result<Self, NixServiceEffectContextErrorV1> {
        let value = Self {
            operation,
            unit: unit.into(),
            authorized_generation,
            pre_state_digest: pre_state_digest.into(),
            authorized_definition_digest: authorized_definition_digest.into(),
            authorized_definition_content_digest: authorized_definition_content_digest.into(),
            pre_invocation_id,
            required_stability_us,
        };
        value.validate_shape()?;
        Ok(value)
    }

    pub fn validate_shape(&self) -> Result<(), NixServiceEffectContextErrorV1> {
        NixServiceOperationV1::new(self.unit.clone(), self.operation)
            .map_err(|error| NixServiceEffectContextErrorV1::InvalidServiceUnit(error.to_string()))?;

        if self.unit.len() > MAX_STRING_BYTES {
            return Err(NixServiceEffectContextErrorV1::FieldTooLong("service unit"));
        }
        if self.authorized_generation == 0 {
            return Err(NixServiceEffectContextErrorV1::InvalidGeneration);
        }
        validate_digest(&self.pre_state_digest, "pre-state digest")?;
        validate_digest(
            &self.authorized_definition_digest,
            "authorized definition digest",
        )?;
        validate_digest(
            &self.authorized_definition_content_digest,
            "authorized definition content digest",
        )?;
        validate_invocation_id(self.pre_invocation_id.as_deref())?;
        if self.required_stability_us > MAX_STABILITY_WINDOW_US {
            return Err(NixServiceEffectContextErrorV1::StabilityWindowTooLarge);
        }
        Ok(())
    }

    pub(crate) fn from_verified_definition_content(
        operation: NixServiceOperationKindV1,
        unit: impl Into<String>,
        authorized_generation: u64,
        pre_state_digest: impl Into<String>,
        pre_invocation_id: Option<String>,
        required_stability_us: u64,
        content: &NixVerifiedServiceDefinitionContentV1,
    ) -> Result<Self, NixServiceEffectContextErrorV1> {
        let unit = unit.into();
        let evidence = content.as_ref();
        if evidence.unit != unit {
            return Err(NixServiceEffectContextErrorV1::DefinitionContentUnitMismatch);
        }
        let content_digest = content.digest()?;
        Self::new(
            operation,
            unit,
            authorized_generation,
            pre_state_digest,
            evidence.source_identity_digest.clone(),
            content_digest,
            pre_invocation_id,
            required_stability_us,
        )
    }

    pub fn digest(&self) -> Result<String, NixServiceEffectContextErrorV1> {
        self.validate_shape()?;
        let mut hasher = Hasher::new();
        hasher.update(SERVICE_EFFECT_CONTEXT_DOMAIN_V1);
        put_u8(&mut hasher, operation_tag(self.operation));
        put_str(&mut hasher, &self.unit);
        put_u64(&mut hasher, self.authorized_generation);
        put_str(&mut hasher, &self.pre_state_digest);
        put_str(&mut hasher, &self.authorized_definition_digest);
        put_str(&mut hasher, &self.authorized_definition_content_digest);
        put_opt_str(&mut hasher, self.pre_invocation_id.as_deref());
        put_u64(&mut hasher, self.required_stability_us);
        Ok(hasher.finalize().to_hex().to_string())
    }
}

fn validate_digest(
    value: &str,
    field: &'static str,
) -> Result<(), NixServiceEffectContextErrorV1> {
    if value.len() != DIGEST_HEX_LEN || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(NixServiceEffectContextErrorV1::InvalidDigest(field));
    }
    Ok(())
}

fn validate_invocation_id(
    value: Option<&str>,
) -> Result<(), NixServiceEffectContextErrorV1> {
    if let Some(value) = value {
        if value.len() != INVOCATION_ID_HEX_LEN
            || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(NixServiceEffectContextErrorV1::InvalidInvocationId);
        }
    }
    Ok(())
}

fn operation_tag(value: NixServiceOperationKindV1) -> u8 {
    match value {
        NixServiceOperationKindV1::Start => 0,
        NixServiceOperationKindV1::Stop => 1,
        NixServiceOperationKindV1::Restart => 2,
        NixServiceOperationKindV1::Reload => 3,
        NixServiceOperationKindV1::Enable => 4,
        NixServiceOperationKindV1::Disable => 5,
    }
}

fn put_u8(hasher: &mut Hasher, value: u8) {
    hasher.update(&[value]);
}

fn put_u64(hasher: &mut Hasher, value: u64) {
    hasher.update(&value.to_be_bytes());
}

fn put_str(hasher: &mut Hasher, value: &str) {
    put_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

fn put_opt_str(hasher: &mut Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            put_u8(hasher, 1);
            put_str(hasher, value);
        }
        None => put_u8(hasher, 0),
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NixServiceEffectContextErrorV1 {
    #[error("invalid service unit: {0}")]
    InvalidServiceUnit(String),
    #[error("service unit is too long")]
    FieldTooLong(&'static str),
    #[error("authorized NixOS generation must be non-zero")]
    InvalidGeneration,
    #[error("invalid {0}")]
    InvalidDigest(&'static str),
    #[error("invalid pre-invocation identity")]
    InvalidInvocationId,
    #[error("required stability window is too large")]
    StabilityWindowTooLarge,
    #[error("service definition content file set is invalid")]
    InvalidDefinitionFileSet,
    #[error("invalid service definition content file path")]
    InvalidDefinitionFilePath,
    #[error("duplicate service definition content file")]
    DuplicateDefinitionFile,
    #[error("service definition content file is too large")]
    DefinitionFileTooLarge,
    #[error("invalid content capture timestamp")]
    InvalidCaptureTimestamp,
    #[error("definition content unit does not match the service intent")]
    DefinitionContentUnitMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> NixServiceEffectContextV1 {
        NixServiceEffectContextV1::new(
            NixServiceOperationKindV1::Restart,
            "nginx.service",
            42,
            &"aa".repeat(32),
            &"bb".repeat(32),
            &"dd".repeat(32),
            Some("cc".repeat(16)),
            1_000,
        )
        .unwrap()
    }

    fn content_evidence() -> NixSystemdUnitDefinitionContentEvidenceV1 {
        NixSystemdUnitDefinitionContentEvidenceV1 {
            unit: "nginx.service".into(),
            source_identity_digest: "11".repeat(32),
            files: vec![
                NixSystemdUnitDefinitionContentFileV1 {
                    path: "/nix/store/nginx.service".into(),
                    byte_len: 10,
                    content_digest: "22".repeat(32),
                },
                NixSystemdUnitDefinitionContentFileV1 {
                    path: "/nix/store/nginx-dropin.conf".into(),
                    byte_len: 20,
                    content_digest: "33".repeat(32),
                },
            ],
            captured_at_monotonic_us: 1,
        }
    }

    #[test]
    fn context_is_deterministic() {
        assert_eq!(context().digest().unwrap(), context().digest().unwrap());
    }

    #[test]
    fn context_digest_commits_every_authority_relevant_field() {
        let base = context();
        let baseline = base.digest().unwrap();

        let mutations = [
            NixServiceEffectContextV1 {
                operation: NixServiceOperationKindV1::Stop,
                ..base.clone()
            },
            NixServiceEffectContextV1 {
                unit: "sshd.service".into(),
                ..base.clone()
            },
            NixServiceEffectContextV1 {
                authorized_generation: 43,
                ..base.clone()
            },
            NixServiceEffectContextV1 {
                pre_state_digest: "dd".repeat(32),
                ..base.clone()
            },
            NixServiceEffectContextV1 {
                authorized_definition_digest: "ee".repeat(32),
                ..base.clone()
            },
            NixServiceEffectContextV1 {
                authorized_definition_content_digest: "gg".repeat(32),
                ..base.clone()
            },
            NixServiceEffectContextV1 {
                pre_invocation_id: Some("ff".repeat(16)),
                ..base.clone()
            },
            NixServiceEffectContextV1 {
                required_stability_us: 2_000,
                ..base
            },
        ];

        for mutation in mutations {
            assert_ne!(baseline, mutation.digest().unwrap());
        }
    }

    #[test]
    fn definition_content_evidence_digest_commits_every_file_field() {
        let base = content_evidence();
        let baseline = base.digest().unwrap();

        let mut changed = base.clone();
        changed.files[0].content_digest = "44".repeat(32);
        assert_ne!(baseline, changed.digest().unwrap());

        let mut changed = base.clone();
        changed.files[0].byte_len += 1;
        assert_ne!(baseline, changed.digest().unwrap());

        let mut changed = base;
        changed.files[1].path = "/nix/store/other.conf".into();
        assert_ne!(baseline, changed.digest().unwrap());
    }

    #[test]
    fn malformed_definition_content_is_rejected() {
        let mut evidence = content_evidence();
        evidence.files[0].content_digest = "short".into();
        assert!(matches!(
            evidence.validate_shape().unwrap_err(),
            NixServiceEffectContextErrorV1::InvalidDigest("definition content digest")
        ));
    }

    #[test]
    fn malformed_context_is_rejected() {
        let mut value = context();
        value.authorized_generation = 0;
        assert!(matches!(
            value.validate_shape().unwrap_err(),
            NixServiceEffectContextErrorV1::InvalidGeneration
        ));

        let mut value = context();
        value.pre_state_digest = "not-a-digest".into();
        assert!(matches!(
            value.validate_shape().unwrap_err(),
            NixServiceEffectContextErrorV1::InvalidDigest("pre-state digest")
        ));

        let mut value = context();
        value.pre_invocation_id = Some("short".into());
        assert_eq!(
            value.validate_shape().unwrap_err(),
            NixServiceEffectContextErrorV1::InvalidInvocationId
        );
    }
}
