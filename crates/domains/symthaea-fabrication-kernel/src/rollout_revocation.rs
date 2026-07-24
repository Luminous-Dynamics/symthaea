// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Threshold-authorized revocation of hardware rollout authority.
//!
//! Promotion and staged rollout grants are positive capabilities. This module
//! provides the corresponding negative authority: a release, phase, or bounded
//! set of machines can be stopped immediately without deleting prior rollout
//! evidence or pretending that an already-issued promotion never existed.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::rollout::{RolloutPhase, RolloutPlan, digest_rollout_plan};
use crate::threshold::VerifiedThresholdCeremony;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const ROLLOUT_REVOCATION_SCHEMA: &str = "symthaea.fabrication.rollout-revocation.v1";
pub const MAX_REVOCATION_INCIDENTS: usize = 256;
pub const MAX_REVOKED_MACHINES: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RolloutRevocationScope {
    EntirePromotion,
    PhaseAndAbove(RolloutPhase),
    Machines(BTreeSet<String>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutRevocationEvidence {
    pub schema_version: String,
    pub revocation_sequence: u64,
    pub promotion_digest: Sha256Digest,
    pub rollout_plan_digest: Sha256Digest,
    pub scope: RolloutRevocationScope,
    pub triggering_incident_digests: Vec<Sha256Digest>,
    pub signer_compromise_tracker_digest: Option<Sha256Digest>,
    pub effective_at_unix_s: u64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RolloutRevocationError {
    UnsupportedSchema,
    SequenceZero,
    PromotionMismatch,
    PlanMismatch,
    InvalidScope,
    TooManyIncidents,
    DuplicateIncident,
    InvalidMachineId,
    InvalidReason,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Rollout(String),
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedRolloutRevocation {
    evidence: RolloutRevocationEvidence,
    revocation_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedRolloutRevocation {
    pub fn evidence(&self) -> &RolloutRevocationEvidence {
        &self.evidence
    }
    pub fn revocation_digest(&self) -> Sha256Digest {
        self.revocation_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_rollout_revocation_evidence(
    plan: &RolloutPlan,
    revocation_sequence: u64,
    promotion_digest: Sha256Digest,
    scope: RolloutRevocationScope,
    triggering_incident_digests: Vec<Sha256Digest>,
    signer_compromise_tracker_digest: Option<Sha256Digest>,
    effective_at_unix_s: u64,
    reason: impl Into<String>,
) -> Result<RolloutRevocationEvidence, RolloutRevocationError> {
    plan.validate()
        .map_err(|error| RolloutRevocationError::Rollout(format!("{error:?}")))?;
    if promotion_digest != plan.promotion_digest {
        return Err(RolloutRevocationError::PromotionMismatch);
    }
    let evidence = RolloutRevocationEvidence {
        schema_version: ROLLOUT_REVOCATION_SCHEMA.into(),
        revocation_sequence,
        promotion_digest,
        rollout_plan_digest: digest_rollout_plan(plan)
            .map_err(|error| RolloutRevocationError::Rollout(format!("{error:?}")))?,
        scope,
        triggering_incident_digests,
        signer_compromise_tracker_digest,
        effective_at_unix_s,
        reason: reason.into(),
    };
    validate_evidence(&evidence)?;
    Ok(evidence)
}

pub fn digest_rollout_revocation(
    evidence: &RolloutRevocationEvidence,
) -> Result<Sha256Digest, RolloutRevocationError> {
    validate_evidence(evidence)?;
    let bytes = serde_json::to_vec(evidence)
        .map_err(|error| RolloutRevocationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.rollout-revocation-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_rollout_revocation(
    evidence: RolloutRevocationEvidence,
    plan: &RolloutPlan,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedRolloutRevocation, RolloutRevocationError> {
    if evidence.promotion_digest != plan.promotion_digest {
        return Err(RolloutRevocationError::PromotionMismatch);
    }
    let plan_digest = digest_rollout_plan(plan)
        .map_err(|error| RolloutRevocationError::Rollout(format!("{error:?}")))?;
    if evidence.rollout_plan_digest != plan_digest {
        return Err(RolloutRevocationError::PlanMismatch);
    }
    let revocation_digest = digest_rollout_revocation(&evidence)?;
    if ceremony.purpose() != "hardware-rollout-revocation" {
        return Err(RolloutRevocationError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != revocation_digest {
        return Err(RolloutRevocationError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedRolloutRevocation {
        evidence,
        revocation_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

impl RolloutRevocationEvidence {
    pub fn revokes(&self, phase: RolloutPhase, machine_id: &str, unix_s: u64) -> bool {
        if unix_s < self.effective_at_unix_s {
            return false;
        }
        match &self.scope {
            RolloutRevocationScope::EntirePromotion => true,
            RolloutRevocationScope::PhaseAndAbove(minimum) => phase >= *minimum,
            RolloutRevocationScope::Machines(machines) => machines.contains(machine_id),
        }
    }
}

fn validate_evidence(evidence: &RolloutRevocationEvidence) -> Result<(), RolloutRevocationError> {
    if evidence.schema_version != ROLLOUT_REVOCATION_SCHEMA {
        return Err(RolloutRevocationError::UnsupportedSchema);
    }
    if evidence.revocation_sequence == 0 {
        return Err(RolloutRevocationError::SequenceZero);
    }
    if evidence.triggering_incident_digests.len() > MAX_REVOCATION_INCIDENTS {
        return Err(RolloutRevocationError::TooManyIncidents);
    }
    let mut incidents = BTreeSet::new();
    for digest in &evidence.triggering_incident_digests {
        if !incidents.insert(*digest) {
            return Err(RolloutRevocationError::DuplicateIncident);
        }
    }
    match &evidence.scope {
        RolloutRevocationScope::EntirePromotion | RolloutRevocationScope::PhaseAndAbove(_) => {}
        RolloutRevocationScope::Machines(machines) => {
            if machines.is_empty() || machines.len() > MAX_REVOKED_MACHINES {
                return Err(RolloutRevocationError::InvalidScope);
            }
            for machine_id in machines {
                if machine_id.trim().is_empty()
                    || machine_id != machine_id.trim()
                    || machine_id.len() > 256
                    || machine_id.chars().any(char::is_control)
                {
                    return Err(RolloutRevocationError::InvalidMachineId);
                }
            }
        }
    }
    if evidence.reason.trim().is_empty()
        || evidence.reason != evidence.reason.trim()
        || evidence.reason.len() > 4096
        || evidence.reason.chars().any(char::is_control)
    {
        return Err(RolloutRevocationError::InvalidReason);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn machine_scope_must_not_be_empty() {
        let evidence = RolloutRevocationEvidence {
            schema_version: ROLLOUT_REVOCATION_SCHEMA.into(),
            revocation_sequence: 1,
            promotion_digest: Sha256Digest([1; 32]),
            rollout_plan_digest: Sha256Digest([2; 32]),
            scope: RolloutRevocationScope::Machines(BTreeSet::new()),
            triggering_incident_digests: Vec::new(),
            signer_compromise_tracker_digest: None,
            effective_at_unix_s: 10,
            reason: "isolate suspect machine class".into(),
        };
        assert_eq!(
            digest_rollout_revocation(&evidence),
            Err(RolloutRevocationError::InvalidScope)
        );
    }
}
