// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Controlled removal and evidence-preserving decommission of gateways.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_membership::{
    AuthorizedGatewayMembership, GatewayMembership, digest_gateway_membership,
};
use crate::threshold::VerifiedThresholdCeremony;
use serde::{Deserialize, Serialize};

pub const GATEWAY_DECOMMISSION_SCHEMA: &str = "symthaea.fabrication.gateway-decommission.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayDecommissionPlan {
    pub schema_version: String,
    pub gateway_id: String,
    pub current_membership_digest: Sha256Digest,
    pub successor_membership_digest: Sha256Digest,
    pub membership_transition_digest: Sha256Digest,
    pub last_gateway_state_digest: Sha256Digest,
    pub credential_revocation_digest: Sha256Digest,
    pub secure_erase_evidence_digest: Sha256Digest,
    pub quarantined_at_unix_s: u64,
    pub decommission_at_unix_s: u64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayDecommissionPolicy {
    pub minimum_quarantine_duration_s: u64,
    pub maximum_plan_duration_s: u64,
    pub require_nonzero_erase_evidence: bool,
}

impl Default for GatewayDecommissionPolicy {
    fn default() -> Self {
        Self {
            minimum_quarantine_duration_s: 3_600,
            maximum_plan_duration_s: 30 * 24 * 3_600,
            require_nonzero_erase_evidence: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayDecommissionError {
    UnsupportedSchema,
    InvalidPolicy,
    InvalidGatewayId,
    InvalidWindow,
    InvalidReason,
    GatewayNotInCurrentMembership,
    GatewayStillInSuccessorMembership,
    CurrentMembershipMismatch,
    SuccessorMembershipMismatch,
    TransitionMismatch,
    EmptyEvidenceDigest,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    MembershipInvalid(String),
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedGatewayDecommission {
    plan: GatewayDecommissionPlan,
    plan_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedGatewayDecommission {
    pub fn plan(&self) -> &GatewayDecommissionPlan {
        &self.plan
    }
    pub fn plan_digest(&self) -> Sha256Digest {
        self.plan_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_gateway_decommission_plan(
    gateway_id: impl Into<String>,
    current_membership: &GatewayMembership,
    authorized_successor: &AuthorizedGatewayMembership,
    last_gateway_state_digest: Sha256Digest,
    credential_revocation_digest: Sha256Digest,
    secure_erase_evidence_digest: Sha256Digest,
    quarantined_at_unix_s: u64,
    decommission_at_unix_s: u64,
    reason: impl Into<String>,
    policy: &GatewayDecommissionPolicy,
) -> Result<GatewayDecommissionPlan, GatewayDecommissionError> {
    validate_policy(policy)?;
    current_membership
        .validate()
        .map_err(|error| GatewayDecommissionError::MembershipInvalid(format!("{error:?}")))?;
    let gateway_id = gateway_id.into();
    validate_identifier(&gateway_id)?;
    if current_membership.member(&gateway_id).is_none() {
        return Err(GatewayDecommissionError::GatewayNotInCurrentMembership);
    }
    if authorized_successor
        .proposed_membership()
        .member(&gateway_id)
        .is_some()
    {
        return Err(GatewayDecommissionError::GatewayStillInSuccessorMembership);
    }
    if quarantined_at_unix_s >= decommission_at_unix_s
        || decommission_at_unix_s.saturating_sub(quarantined_at_unix_s)
            < policy.minimum_quarantine_duration_s
        || decommission_at_unix_s.saturating_sub(quarantined_at_unix_s)
            > policy.maximum_plan_duration_s
    {
        return Err(GatewayDecommissionError::InvalidWindow);
    }
    if policy.require_nonzero_erase_evidence
        && secure_erase_evidence_digest == Sha256Digest([0; 32])
    {
        return Err(GatewayDecommissionError::EmptyEvidenceDigest);
    }
    if credential_revocation_digest == Sha256Digest([0; 32])
        || last_gateway_state_digest == Sha256Digest([0; 32])
    {
        return Err(GatewayDecommissionError::EmptyEvidenceDigest);
    }
    let reason = reason.into();
    if reason.trim().is_empty()
        || reason != reason.trim()
        || reason.len() > 2_048
        || reason.chars().any(char::is_control)
    {
        return Err(GatewayDecommissionError::InvalidReason);
    }
    let current_digest = digest_gateway_membership(current_membership)
        .map_err(|error| GatewayDecommissionError::MembershipInvalid(format!("{error:?}")))?;
    if authorized_successor.transition().current_membership_digest != current_digest {
        return Err(GatewayDecommissionError::CurrentMembershipMismatch);
    }
    Ok(GatewayDecommissionPlan {
        schema_version: GATEWAY_DECOMMISSION_SCHEMA.into(),
        gateway_id,
        current_membership_digest: current_digest,
        successor_membership_digest: digest_gateway_membership(
            authorized_successor.proposed_membership(),
        )
        .map_err(|error| GatewayDecommissionError::MembershipInvalid(format!("{error:?}")))?,
        membership_transition_digest: authorized_successor.transition_digest(),
        last_gateway_state_digest,
        credential_revocation_digest,
        secure_erase_evidence_digest,
        quarantined_at_unix_s,
        decommission_at_unix_s,
        reason,
    })
}

pub fn digest_gateway_decommission_plan(
    plan: &GatewayDecommissionPlan,
) -> Result<Sha256Digest, GatewayDecommissionError> {
    validate_plan(plan)?;
    let bytes = serde_json::to_vec(plan)
        .map_err(|error| GatewayDecommissionError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.gateway-decommission-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_gateway_decommission(
    plan: GatewayDecommissionPlan,
    current_membership: &GatewayMembership,
    authorized_successor: &AuthorizedGatewayMembership,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedGatewayDecommission, GatewayDecommissionError> {
    validate_plan(&plan)?;
    let current_digest = digest_gateway_membership(current_membership)
        .map_err(|error| GatewayDecommissionError::MembershipInvalid(format!("{error:?}")))?;
    if plan.current_membership_digest != current_digest {
        return Err(GatewayDecommissionError::CurrentMembershipMismatch);
    }
    let successor_digest = digest_gateway_membership(authorized_successor.proposed_membership())
        .map_err(|error| GatewayDecommissionError::MembershipInvalid(format!("{error:?}")))?;
    if plan.successor_membership_digest != successor_digest {
        return Err(GatewayDecommissionError::SuccessorMembershipMismatch);
    }
    if plan.membership_transition_digest != authorized_successor.transition_digest() {
        return Err(GatewayDecommissionError::TransitionMismatch);
    }
    let plan_digest = digest_gateway_decommission_plan(&plan)?;
    if ceremony.purpose() != "gateway-decommission" {
        return Err(GatewayDecommissionError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != plan_digest {
        return Err(GatewayDecommissionError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedGatewayDecommission {
        plan,
        plan_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

fn validate_plan(plan: &GatewayDecommissionPlan) -> Result<(), GatewayDecommissionError> {
    if plan.schema_version != GATEWAY_DECOMMISSION_SCHEMA {
        return Err(GatewayDecommissionError::UnsupportedSchema);
    }
    validate_identifier(&plan.gateway_id)?;
    if plan.quarantined_at_unix_s >= plan.decommission_at_unix_s {
        return Err(GatewayDecommissionError::InvalidWindow);
    }
    if plan.reason.trim().is_empty()
        || plan.reason != plan.reason.trim()
        || plan.reason.len() > 2_048
        || plan.reason.chars().any(char::is_control)
    {
        return Err(GatewayDecommissionError::InvalidReason);
    }
    Ok(())
}

fn validate_policy(policy: &GatewayDecommissionPolicy) -> Result<(), GatewayDecommissionError> {
    if policy.minimum_quarantine_duration_s == 0
        || policy.maximum_plan_duration_s < policy.minimum_quarantine_duration_s
    {
        return Err(GatewayDecommissionError::InvalidPolicy);
    }
    Ok(())
}

fn validate_identifier(value: &str) -> Result<(), GatewayDecommissionError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
    {
        return Err(GatewayDecommissionError::InvalidGatewayId);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_digest_rejects_zero_length_window() {
        let plan = GatewayDecommissionPlan {
            schema_version: GATEWAY_DECOMMISSION_SCHEMA.into(),
            gateway_id: "gateway-a".into(),
            current_membership_digest: Sha256Digest([1; 32]),
            successor_membership_digest: Sha256Digest([2; 32]),
            membership_transition_digest: Sha256Digest([3; 32]),
            last_gateway_state_digest: Sha256Digest([4; 32]),
            credential_revocation_digest: Sha256Digest([5; 32]),
            secure_erase_evidence_digest: Sha256Digest([6; 32]),
            quarantined_at_unix_s: 10,
            decommission_at_unix_s: 10,
            reason: "compromised hardware".into(),
        };
        assert_eq!(
            digest_gateway_decommission_plan(&plan),
            Err(GatewayDecommissionError::InvalidWindow)
        );
    }
}
