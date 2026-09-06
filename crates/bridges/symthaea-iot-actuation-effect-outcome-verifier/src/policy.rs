// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use symthaea_authority::{Digest32, Operation, ResourceRef};

use crate::{
    EffectOutcomeClaimKindV1, EffectOutcomeError, MAX_EFFECT_OUTCOME_DEVICE_ID_BYTES,
    MAX_EFFECT_OUTCOME_ID_BYTES, valid_id,
};

pub const EFFECT_OUTCOME_POLICY_SCHEMA_VERSION: u16 = 1;
pub const MAX_EFFECT_OUTCOME_EVIDENCE_LIFETIME_MS: u64 = 5_000;
pub const MAX_EFFECT_OUTCOME_POLICY_ITEMS: usize = 256;

const EFFECT_OUTCOME_POLICY_DOMAIN: &[u8] = b"symthaea-iot-effect-outcome-policy-v1\0";

/// Immutable guard-owned policy defining one exact device-class outcome proof profile.
///
/// `generation` is part of the policy's cryptographic identity. This prevents policy ABA where
/// generation N and N+2 contain byte-identical semantic rules but represent distinct authoritative
/// publication epochs. `exact_outcome_profile_digest` commits the class-specific semantics of
/// execution records, postcondition sensing, non-execution log coverage and anti-tamper assumptions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectOutcomePolicyV1 {
    pub schema_version: u16,
    pub generation: u64,
    pub device: ResourceRef,
    pub operation: Operation,
    pub allowed_verifier_ids: BTreeSet<String>,
    pub allowed_claim_kinds: BTreeSet<EffectOutcomeClaimKindV1>,
    pub accepted_reference_values: BTreeSet<Digest32>,
    pub exact_outcome_profile_digest: Digest32,
    pub exact_appraisal_policy_digest: Digest32,
    pub max_evidence_lifetime_ms: u64,
}

impl EffectOutcomePolicyV1 {
    pub fn validate(&self) -> Result<(), EffectOutcomeError> {
        if self.schema_version != EFFECT_OUTCOME_POLICY_SCHEMA_VERSION {
            return Err(EffectOutcomeError::UnsupportedPolicySchema);
        }
        if self.generation == 0 {
            return Err(EffectOutcomeError::PolicyGenerationZero);
        }
        if !valid_id(&self.device.0, MAX_EFFECT_OUTCOME_DEVICE_ID_BYTES)
            || !valid_id(&self.operation.0, MAX_EFFECT_OUTCOME_ID_BYTES)
        {
            return Err(EffectOutcomeError::InvalidPolicyTarget);
        }
        if self.allowed_verifier_ids.is_empty()
            || self.allowed_verifier_ids.len() > MAX_EFFECT_OUTCOME_POLICY_ITEMS
            || self
                .allowed_verifier_ids
                .iter()
                .any(|id| !valid_id(id, MAX_EFFECT_OUTCOME_ID_BYTES))
        {
            return Err(EffectOutcomeError::InvalidPolicyVerifierSurface);
        }
        if self.allowed_claim_kinds.is_empty() || self.allowed_claim_kinds.len() > 2 {
            return Err(EffectOutcomeError::InvalidPolicyClaimSurface);
        }
        if self.accepted_reference_values.is_empty()
            || self.accepted_reference_values.len() > MAX_EFFECT_OUTCOME_POLICY_ITEMS
            || self.accepted_reference_values.contains(&Digest32([0; 32]))
        {
            return Err(EffectOutcomeError::InvalidPolicyReferenceValues);
        }
        if self.exact_outcome_profile_digest == Digest32([0; 32]) {
            return Err(EffectOutcomeError::ZeroOutcomeProfileDigest);
        }
        if self.exact_appraisal_policy_digest == Digest32([0; 32]) {
            return Err(EffectOutcomeError::ZeroAppraisalPolicyDigest);
        }
        if self.max_evidence_lifetime_ms == 0
            || self.max_evidence_lifetime_ms > MAX_EFFECT_OUTCOME_EVIDENCE_LIFETIME_MS
        {
            return Err(EffectOutcomeError::InvalidPolicyEvidenceLifetime);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, EffectOutcomeError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(EFFECT_OUTCOME_POLICY_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.generation.to_be_bytes());
        update_string(&mut h, &self.device.0);
        update_string(&mut h, &self.operation.0);
        h.update(&(self.allowed_verifier_ids.len() as u32).to_be_bytes());
        for verifier_id in &self.allowed_verifier_ids {
            update_string(&mut h, verifier_id);
        }
        h.update(&(self.allowed_claim_kinds.len() as u32).to_be_bytes());
        for claim_kind in &self.allowed_claim_kinds {
            h.update(&[claim_kind.tag()]);
        }
        h.update(&(self.accepted_reference_values.len() as u32).to_be_bytes());
        for digest in &self.accepted_reference_values {
            update_digest(&mut h, *digest);
        }
        update_digest(&mut h, self.exact_outcome_profile_digest);
        update_digest(&mut h, self.exact_appraisal_policy_digest);
        h.update(&self.max_evidence_lifetime_ms.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }
}

fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u32).to_be_bytes());
    h.update(value.as_bytes());
}

fn update_digest(h: &mut blake3::Hasher, Digest32(bytes): Digest32) {
    h.update(&bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(generation: u64) -> EffectOutcomePolicyV1 {
        EffectOutcomePolicyV1 {
            schema_version: EFFECT_OUTCOME_POLICY_SCHEMA_VERSION,
            generation,
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("qualification.effect".into()),
            allowed_verifier_ids: BTreeSet::from(["verifier:outcome-a".into()]),
            allowed_claim_kinds: BTreeSet::from([
                EffectOutcomeClaimKindV1::ExecutionAndPostcondition,
                EffectOutcomeClaimKindV1::NonExecution,
            ]),
            accepted_reference_values: BTreeSet::from([Digest32([0x44; 32])]),
            exact_outcome_profile_digest: Digest32([0x55; 32]),
            exact_appraisal_policy_digest: Digest32([0x66; 32]),
            max_evidence_lifetime_ms: MAX_EFFECT_OUTCOME_EVIDENCE_LIFETIME_MS,
        }
    }

    #[test]
    fn policy_generation_is_part_of_cryptographic_identity() {
        let first = policy(1);
        let third = policy(3);
        assert_ne!(first.digest().unwrap(), third.digest().unwrap());
    }

    #[test]
    fn policy_generation_zero_is_rejected() {
        assert!(matches!(
            policy(0).validate(),
            Err(EffectOutcomeError::PolicyGenerationZero)
        ));
    }
}
