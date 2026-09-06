// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_authority::{Digest32, Operation, PrincipalId, ResourceRef};

use crate::{
    EFFECT_OUTCOME_ED25519_ALGORITHM, EffectOutcomeError, MAX_EFFECT_OUTCOME_DEVICE_ID_BYTES,
    MAX_EFFECT_OUTCOME_ID_BYTES, valid_id,
};

pub const EFFECT_OUTCOME_EVIDENCE_SCHEMA_VERSION: u16 = 1;

const EVIDENCE_BODY_DOMAIN: &[u8] = b"symthaea-iot-effect-outcome-evidence-body-v1\0";
const EVIDENCE_SIGNATURE_DOMAIN: &[u8] = b"symthaea-iot-effect-outcome-evidence-signature-v1\0";
const EVIDENCE_OBJECT_DOMAIN: &[u8] = b"symthaea-iot-effect-outcome-evidence-object-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EffectOutcomeClaimKindV1 {
    ExecutionAndPostcondition,
    NonExecution,
}

impl EffectOutcomeClaimKindV1 {
    pub const fn tag(self) -> u8 {
        match self {
            Self::ExecutionAndPostcondition => 0,
            Self::NonExecution => 1,
        }
    }
}

/// Signed semantic claim produced by a device-class-specific outcome verifier profile.
///
/// `NonExecution` is deliberately not expressible as "postcondition absent now". It requires a
/// separately committed non-execution proof and a tamper-evident execution-log head with explicit
/// temporal coverage. The guard later requires that coverage to span the complete actuation window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectOutcomeClaimV1 {
    ExecutionAndPostcondition {
        execution_record_digest: Digest32,
        effect_recorded_at_unix_ms: u64,
        postcondition_evidence_digest: Digest32,
        postcondition_observed_at_unix_ms: u64,
    },
    NonExecution {
        non_execution_proof_digest: Digest32,
        execution_log_head_digest: Digest32,
        coverage_from_unix_ms: u64,
        coverage_through_unix_ms: u64,
    },
}

impl EffectOutcomeClaimV1 {
    pub const fn kind(self) -> EffectOutcomeClaimKindV1 {
        match self {
            Self::ExecutionAndPostcondition { .. } => {
                EffectOutcomeClaimKindV1::ExecutionAndPostcondition
            }
            Self::NonExecution { .. } => EffectOutcomeClaimKindV1::NonExecution,
        }
    }

    fn validate_structure(self) -> Result<(), EffectOutcomeError> {
        match self {
            Self::ExecutionAndPostcondition {
                execution_record_digest,
                effect_recorded_at_unix_ms,
                postcondition_evidence_digest,
                postcondition_observed_at_unix_ms,
            } => {
                if execution_record_digest == Digest32([0; 32])
                    || postcondition_evidence_digest == Digest32([0; 32])
                    || effect_recorded_at_unix_ms == 0
                    || postcondition_observed_at_unix_ms < effect_recorded_at_unix_ms
                {
                    return Err(EffectOutcomeError::InvalidClaimStructure);
                }
            }
            Self::NonExecution {
                non_execution_proof_digest,
                execution_log_head_digest,
                coverage_from_unix_ms,
                coverage_through_unix_ms,
            } => {
                if non_execution_proof_digest == Digest32([0; 32])
                    || execution_log_head_digest == Digest32([0; 32])
                    || coverage_from_unix_ms >= coverage_through_unix_ms
                {
                    return Err(EffectOutcomeError::InvalidClaimStructure);
                }
            }
        }
        Ok(())
    }
}

/// Exact body signed by the trusted device-class outcome verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalEffectOutcomeEvidenceBodyV1 {
    pub schema_version: u16,
    pub device: ResourceRef,
    pub operation: Operation,
    pub executor: PrincipalId,
    pub challenge_digest: Digest32,
    pub command_digest: Digest32,
    pub sequence: u64,
    pub outcome_profile_digest: Digest32,
    pub reference_values_digest: Digest32,
    pub appraisal_policy_digest: Digest32,
    pub verifier_id: String,
    pub key_id: String,
    pub algorithm: String,
    pub claim: EffectOutcomeClaimV1,
    pub evidence_issued_at_unix_ms: u64,
    pub evidence_expires_at_unix_ms: u64,
}

impl PhysicalEffectOutcomeEvidenceBodyV1 {
    pub fn validate_structure(&self) -> Result<(), EffectOutcomeError> {
        if self.schema_version != EFFECT_OUTCOME_EVIDENCE_SCHEMA_VERSION {
            return Err(EffectOutcomeError::UnsupportedEvidenceSchema);
        }
        if !valid_id(&self.device.0, MAX_EFFECT_OUTCOME_DEVICE_ID_BYTES)
            || !valid_id(&self.operation.0, MAX_EFFECT_OUTCOME_ID_BYTES)
            || !valid_id(&self.executor.0, MAX_EFFECT_OUTCOME_ID_BYTES)
            || !valid_id(&self.verifier_id, MAX_EFFECT_OUTCOME_ID_BYTES)
            || !valid_id(&self.key_id, MAX_EFFECT_OUTCOME_ID_BYTES)
        {
            return Err(EffectOutcomeError::InvalidEvidenceIdentity);
        }
        if self.algorithm != EFFECT_OUTCOME_ED25519_ALGORITHM {
            return Err(EffectOutcomeError::UnsupportedEvidenceAlgorithm);
        }
        if self.sequence == 0 {
            return Err(EffectOutcomeError::InvalidClaimStructure);
        }
        for digest in [
            self.challenge_digest,
            self.command_digest,
            self.outcome_profile_digest,
            self.reference_values_digest,
            self.appraisal_policy_digest,
        ] {
            if digest == Digest32([0; 32]) {
                return Err(EffectOutcomeError::ZeroEvidenceCommitment);
            }
        }
        if self.evidence_issued_at_unix_ms >= self.evidence_expires_at_unix_ms {
            return Err(EffectOutcomeError::InvalidEvidenceWindow);
        }
        self.claim.validate_structure()?;
        Ok(())
    }

    /// One explicit cross-language signed representation. All integers are big-endian and all
    /// strings are u32-byte-length-prefixed UTF-8.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, EffectOutcomeError> {
        self.validate_structure()?;
        let mut out = Vec::with_capacity(512);
        out.extend_from_slice(&self.schema_version.to_be_bytes());
        push_string(&mut out, &self.device.0)?;
        push_string(&mut out, &self.operation.0)?;
        push_string(&mut out, &self.executor.0)?;
        push_digest(&mut out, self.challenge_digest);
        push_digest(&mut out, self.command_digest);
        push_u64(&mut out, self.sequence);
        push_digest(&mut out, self.outcome_profile_digest);
        push_digest(&mut out, self.reference_values_digest);
        push_digest(&mut out, self.appraisal_policy_digest);
        push_string(&mut out, &self.verifier_id)?;
        push_string(&mut out, &self.key_id)?;
        push_string(&mut out, &self.algorithm)?;
        out.push(self.claim.kind().tag());
        match self.claim {
            EffectOutcomeClaimV1::ExecutionAndPostcondition {
                execution_record_digest,
                effect_recorded_at_unix_ms,
                postcondition_evidence_digest,
                postcondition_observed_at_unix_ms,
            } => {
                push_digest(&mut out, execution_record_digest);
                push_u64(&mut out, effect_recorded_at_unix_ms);
                push_digest(&mut out, postcondition_evidence_digest);
                push_u64(&mut out, postcondition_observed_at_unix_ms);
            }
            EffectOutcomeClaimV1::NonExecution {
                non_execution_proof_digest,
                execution_log_head_digest,
                coverage_from_unix_ms,
                coverage_through_unix_ms,
            } => {
                push_digest(&mut out, non_execution_proof_digest);
                push_digest(&mut out, execution_log_head_digest);
                push_u64(&mut out, coverage_from_unix_ms);
                push_u64(&mut out, coverage_through_unix_ms);
            }
        }
        push_u64(&mut out, self.evidence_issued_at_unix_ms);
        push_u64(&mut out, self.evidence_expires_at_unix_ms);
        Ok(out)
    }

    pub fn digest(&self) -> Result<Digest32, EffectOutcomeError> {
        let bytes = self.canonical_bytes()?;
        Ok(domain_hash(EVIDENCE_BODY_DOMAIN, &bytes))
    }

    pub fn signature_message(&self) -> Result<Vec<u8>, EffectOutcomeError> {
        let bytes = self.canonical_bytes()?;
        let mut message = Vec::with_capacity(
            EVIDENCE_SIGNATURE_DOMAIN.len()
                + std::mem::size_of::<u64>()
                + bytes.len(),
        );
        message.extend_from_slice(EVIDENCE_SIGNATURE_DOMAIN);
        message.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
        message.extend_from_slice(&bytes);
        Ok(message)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalEffectOutcomeEvidenceV1 {
    pub body: PhysicalEffectOutcomeEvidenceBodyV1,
    pub signature: [u8; 64],
}

impl PhysicalEffectOutcomeEvidenceV1 {
    pub fn validate_structure(&self) -> Result<(), EffectOutcomeError> {
        self.body.validate_structure()?;
        if self.signature == [0; 64] {
            return Err(EffectOutcomeError::InvalidEvidenceSignature);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, EffectOutcomeError> {
        self.validate_structure()?;
        let body_digest = self.body.digest()?;
        let mut bytes = Vec::with_capacity(96);
        push_digest(&mut bytes, body_digest);
        bytes.extend_from_slice(&self.signature);
        Ok(domain_hash(EVIDENCE_OBJECT_DOMAIN, &bytes))
    }
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_digest(out: &mut Vec<u8>, Digest32(bytes): Digest32) {
    out.extend_from_slice(&bytes);
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<(), EffectOutcomeError> {
    let len = u32::try_from(value.len()).map_err(|_| EffectOutcomeError::EncodingLengthOverflow)?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> Digest32 {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    h.update(&(bytes.len() as u64).to_be_bytes());
    h.update(bytes);
    Digest32(*h.finalize().as_bytes())
}
