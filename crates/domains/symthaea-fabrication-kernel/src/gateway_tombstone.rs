// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable tombstones for fully decommissioned gateways.
//!
//! Mutable retirement tracking is useful during quarantine and erasure. Once a
//! gateway reaches the terminal state, this module seals the plan, final state,
//! credential revocation, erase evidence, and successor membership into a
//! threshold-authorized record that must never be replaced or removed.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_decommission::AuthorizedGatewayDecommission;
use crate::gateway_decommission_tracker::{GatewayRetirementRecord, GatewayRetirementStage};
use crate::threshold::VerifiedThresholdCeremony;
use serde::{Deserialize, Serialize};

pub const GATEWAY_TOMBSTONE_SCHEMA: &str = "symthaea.fabrication.gateway-tombstone.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayTombstone {
    pub schema_version: String,
    pub tombstone_sequence: u64,
    pub gateway_id: String,
    pub decommission_plan_digest: Sha256Digest,
    pub decommission_ceremony_digest: Sha256Digest,
    pub final_retirement_record_digest: Sha256Digest,
    pub last_gateway_state_digest: Sha256Digest,
    pub credential_revocation_digest: Sha256Digest,
    pub planned_erase_evidence_digest: Sha256Digest,
    pub verified_erase_evidence_digest: Sha256Digest,
    pub successor_membership_digest: Sha256Digest,
    pub decommissioned_at_unix_s: u64,
    pub issued_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayTombstoneError {
    UnsupportedSchema,
    SequenceZero,
    InvalidGatewayId,
    RetirementNotTerminal,
    GatewayMismatch,
    PlanMismatch,
    MissingEraseEvidence,
    InvalidTime,
    EmptyEvidenceDigest,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedGatewayTombstone {
    tombstone: GatewayTombstone,
    tombstone_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedGatewayTombstone {
    pub fn tombstone(&self) -> &GatewayTombstone {
        &self.tombstone
    }
    pub fn tombstone_digest(&self) -> Sha256Digest {
        self.tombstone_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }
}

pub fn build_gateway_tombstone(
    authorization: &AuthorizedGatewayDecommission,
    final_record: &GatewayRetirementRecord,
    tombstone_sequence: u64,
    issued_at_unix_s: u64,
) -> Result<GatewayTombstone, GatewayTombstoneError> {
    if tombstone_sequence == 0 {
        return Err(GatewayTombstoneError::SequenceZero);
    }
    if final_record.stage != GatewayRetirementStage::Decommissioned {
        return Err(GatewayTombstoneError::RetirementNotTerminal);
    }
    if final_record.gateway_id != authorization.plan().gateway_id {
        return Err(GatewayTombstoneError::GatewayMismatch);
    }
    if final_record.plan_digest != authorization.plan_digest() {
        return Err(GatewayTombstoneError::PlanMismatch);
    }
    let Some(verified_erase_evidence_digest) = final_record.erase_verification_digest else {
        return Err(GatewayTombstoneError::MissingEraseEvidence);
    };
    if issued_at_unix_s < final_record.updated_at_unix_s {
        return Err(GatewayTombstoneError::InvalidTime);
    }
    Ok(GatewayTombstone {
        schema_version: GATEWAY_TOMBSTONE_SCHEMA.into(),
        tombstone_sequence,
        gateway_id: authorization.plan().gateway_id.clone(),
        decommission_plan_digest: authorization.plan_digest(),
        decommission_ceremony_digest: authorization.ceremony_digest(),
        final_retirement_record_digest: final_record.final_record_digest,
        last_gateway_state_digest: authorization.plan().last_gateway_state_digest,
        credential_revocation_digest: authorization.plan().credential_revocation_digest,
        planned_erase_evidence_digest: authorization.plan().secure_erase_evidence_digest,
        verified_erase_evidence_digest,
        successor_membership_digest: authorization.plan().successor_membership_digest,
        decommissioned_at_unix_s: final_record.updated_at_unix_s,
        issued_at_unix_s,
    })
}

pub fn digest_gateway_tombstone(
    tombstone: &GatewayTombstone,
) -> Result<Sha256Digest, GatewayTombstoneError> {
    validate_tombstone(tombstone)?;
    let bytes = serde_json::to_vec(tombstone)
        .map_err(|error| GatewayTombstoneError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.gateway-tombstone-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_gateway_tombstone(
    tombstone: GatewayTombstone,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedGatewayTombstone, GatewayTombstoneError> {
    let tombstone_digest = digest_gateway_tombstone(&tombstone)?;
    if ceremony.purpose() != "gateway-decommission-tombstone" {
        return Err(GatewayTombstoneError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != tombstone_digest {
        return Err(GatewayTombstoneError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedGatewayTombstone {
        tombstone,
        tombstone_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

fn validate_tombstone(tombstone: &GatewayTombstone) -> Result<(), GatewayTombstoneError> {
    if tombstone.schema_version != GATEWAY_TOMBSTONE_SCHEMA {
        return Err(GatewayTombstoneError::UnsupportedSchema);
    }
    if tombstone.tombstone_sequence == 0 {
        return Err(GatewayTombstoneError::SequenceZero);
    }
    if tombstone.gateway_id.trim().is_empty()
        || tombstone.gateway_id != tombstone.gateway_id.trim()
        || tombstone.gateway_id.len() > 256
        || tombstone.gateway_id.chars().any(char::is_control)
    {
        return Err(GatewayTombstoneError::InvalidGatewayId);
    }
    if tombstone.decommissioned_at_unix_s == 0
        || tombstone.issued_at_unix_s < tombstone.decommissioned_at_unix_s
    {
        return Err(GatewayTombstoneError::InvalidTime);
    }
    if tombstone.decommission_plan_digest == Sha256Digest([0; 32])
        || tombstone.decommission_ceremony_digest == Sha256Digest([0; 32])
        || tombstone.final_retirement_record_digest == Sha256Digest([0; 32])
        || tombstone.last_gateway_state_digest == Sha256Digest([0; 32])
        || tombstone.credential_revocation_digest == Sha256Digest([0; 32])
        || tombstone.planned_erase_evidence_digest == Sha256Digest([0; 32])
        || tombstone.verified_erase_evidence_digest == Sha256Digest([0; 32])
        || tombstone.successor_membership_digest == Sha256Digest([0; 32])
    {
        return Err(GatewayTombstoneError::EmptyEvidenceDigest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_rejects_pre_decommission_issue_time() {
        let tombstone = GatewayTombstone {
            schema_version: GATEWAY_TOMBSTONE_SCHEMA.into(),
            tombstone_sequence: 1,
            gateway_id: "gateway-a".into(),
            decommission_plan_digest: Sha256Digest([1; 32]),
            decommission_ceremony_digest: Sha256Digest([2; 32]),
            final_retirement_record_digest: Sha256Digest([3; 32]),
            last_gateway_state_digest: Sha256Digest([4; 32]),
            credential_revocation_digest: Sha256Digest([5; 32]),
            planned_erase_evidence_digest: Sha256Digest([6; 32]),
            verified_erase_evidence_digest: Sha256Digest([7; 32]),
            successor_membership_digest: Sha256Digest([8; 32]),
            decommissioned_at_unix_s: 20,
            issued_at_unix_s: 19,
        };
        assert_eq!(
            digest_gateway_tombstone(&tombstone),
            Err(GatewayTombstoneError::InvalidTime)
        );
    }
}
