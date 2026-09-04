// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_authority::{Digest32, ResourceRef};

use crate::{DeviceRealityError, valid_id};

/// Current guard-owned device-reality policy schema.
pub const DEVICE_REALITY_POLICY_SCHEMA_VERSION: u16 = 1;
/// The post-reservation challenge itself is capped at five seconds, so accepted device
/// reality must never outlive that causal window.
pub const MAX_DEVICE_REALITY_RESULT_LIFETIME_MS: u64 = 5_000;
/// Bound the number of independently acceptable verifier identities/reference lineages.
pub const MAX_DEVICE_REALITY_POLICY_ITEMS: usize = 256;
/// Conservative bound for the exact physical-device identifier.
const MAX_DEVICE_ID_BYTES: usize = 512;
const DEVICE_REALITY_POLICY_DOMAIN: &[u8] = b"symthaea-iot-device-reality-policy-v1\0";

/// Immutable local policy deciding which challenge-bound device appraisals may enter
/// the privileged physical-actuation TCB.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceRealityPolicyV1 {
    pub schema_version: u16,
    pub device: ResourceRef,
    pub allowed_verifier_ids: BTreeSet<String>,
    pub accepted_reference_values: BTreeSet<Digest32>,
    pub exact_appraisal_policy_digest: Digest32,
    pub max_result_lifetime_ms: u64,
}

impl DeviceRealityPolicyV1 {
    pub fn validate(&self) -> Result<(), DeviceRealityError> {
        if self.schema_version != DEVICE_REALITY_POLICY_SCHEMA_VERSION {
            return Err(DeviceRealityError::UnsupportedPolicySchema);
        }
        if self.device.0.is_empty()
            || self.device.0.len() > MAX_DEVICE_ID_BYTES
            || self.device.0.trim() != self.device.0
            || self.device.0.chars().any(char::is_control)
        {
            return Err(DeviceRealityError::InvalidPolicyDevice);
        }
        if self.allowed_verifier_ids.is_empty()
            || self.allowed_verifier_ids.len() > MAX_DEVICE_REALITY_POLICY_ITEMS
            || self.allowed_verifier_ids.iter().any(|id| !valid_id(id))
        {
            return Err(DeviceRealityError::InvalidPolicyVerifierSurface);
        }
        if self.accepted_reference_values.is_empty()
            || self.accepted_reference_values.len() > MAX_DEVICE_REALITY_POLICY_ITEMS
            || self
                .accepted_reference_values
                .contains(&Digest32([0; 32]))
        {
            return Err(DeviceRealityError::InvalidPolicyReferenceValues);
        }
        if self.exact_appraisal_policy_digest == Digest32([0; 32]) {
            return Err(DeviceRealityError::ZeroAppraisalPolicyDigest);
        }
        if self.max_result_lifetime_ms == 0
            || self.max_result_lifetime_ms > MAX_DEVICE_REALITY_RESULT_LIFETIME_MS
        {
            return Err(DeviceRealityError::InvalidPolicyResultLifetime);
        }
        Ok(())
    }

    /// Domain-separated commitment retained independently by the privileged guard.
    pub fn digest(&self) -> Result<Digest32, DeviceRealityError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(DEVICE_REALITY_POLICY_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        update_string(&mut h, &self.device.0);
        h.update(&(self.allowed_verifier_ids.len() as u32).to_be_bytes());
        for verifier_id in &self.allowed_verifier_ids {
            update_string(&mut h, verifier_id);
        }
        h.update(&(self.accepted_reference_values.len() as u32).to_be_bytes());
        for digest in &self.accepted_reference_values {
            update_digest(&mut h, *digest);
        }
        update_digest(&mut h, self.exact_appraisal_policy_digest);
        h.update(&self.max_result_lifetime_ms.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }
}

fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u32).to_be_bytes());
    h.update(value.as_bytes());
}

fn update_digest(h: &mut blake3::Hasher, Digest32(value): Digest32) {
    h.update(&value);
}
