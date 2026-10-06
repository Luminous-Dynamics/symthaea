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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixServiceEffectContextV1 {
    pub operation: NixServiceOperationKindV1,
    pub unit: String,
    pub authorized_generation: u64,
    pub pre_state_digest: String,
    pub authorized_definition_digest: String,
    /// Digest of the observer-produced exact definition-content commitment.
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
                authorized_definition_content_digest: "ff".repeat(32),
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
