// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finalization authority for a probation-cleared, hardware-reauthorized upgrade.

use crate::automatic_rollback::AutomaticRollbackTrigger;
use crate::clock_continuity::{VerifiedClockContinuity, digest_clock_continuity};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::evidence_retention::{
    AuthorizedEvidenceRetentionPolicy, digest_evidence_retention_policy,
};
use crate::hardware_reauthorization_tracker::{
    HardwareReauthorizationTracker, digest_hardware_reauthorization_tracker,
};
use crate::key_continuity::{VerifiedKeyContinuity, digest_key_continuity};
use crate::threshold::VerifiedThresholdCeremony;
use crate::upgrade_handoff::AuthorizedUpgradeHandoff;
use crate::upgrade_probation::AuthorizedUpgradeProbationClearance;
use crate::upgrade_state::{FabricationUpgradeState, digest_upgrade_state};
use crate::upgrade_tracker::UpgradeStage;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const UPGRADE_FINALIZATION_SCHEMA: &str = "symthaea.fabrication.upgrade-finalization.v1";
pub const MAX_FINALIZATION_MACHINES: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpgradeFinalizationPolicy {
    pub minimum_reauthorized_machines: usize,
    pub maximum_authorization_duration_ms: u64,
}

impl Default for UpgradeFinalizationPolicy {
    fn default() -> Self {
        Self {
            minimum_reauthorized_machines: 1,
            maximum_authorization_duration_ms: 15 * 60 * 1_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeFinalizationEvidence {
    pub schema_version: String,
    pub finalization_sequence: u64,
    pub handoff_digest: Sha256Digest,
    pub successor_state_digest: Sha256Digest,
    pub upgrade_state_digest: Sha256Digest,
    pub probation_clearance_digest: Sha256Digest,
    pub hardware_reauthorization_tracker_digest: Sha256Digest,
    pub required_machine_ids: BTreeSet<String>,
    pub retention_policy_digest: Sha256Digest,
    pub key_continuity_digest: Sha256Digest,
    pub clock_continuity_digest: Sha256Digest,
    pub authorized_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeFinalizationError {
    UnsupportedSchema,
    InvalidPolicy,
    SequenceZero,
    ZeroDigest(&'static str),
    UpgradeNotActivated,
    HandoffMismatch,
    FinalizationDeadlinePassed,
    ProbationExpired,
    InvalidMachineScope,
    MachineNotReauthorized(String),
    RetentionPolicyNotEffective,
    RollbackTriggerPresent,
    InvalidWindow,
    Evidence(String),
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedUpgradeFinalization {
    evidence: UpgradeFinalizationEvidence,
    evidence_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedUpgradeFinalization {
    pub fn evidence(&self) -> &UpgradeFinalizationEvidence {
        &self.evidence
    }
    pub fn evidence_digest(&self) -> Sha256Digest {
        self.evidence_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }

    pub fn permits_finalization(&self, handoff_digest: Sha256Digest, unix_ms: u64) -> bool {
        self.evidence.handoff_digest == handoff_digest
            && unix_ms >= self.evidence.authorized_at_unix_ms
            && unix_ms < self.evidence.expires_at_unix_ms
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_upgrade_finalization_evidence(
    finalization_sequence: u64,
    handoff: &AuthorizedUpgradeHandoff,
    upgrade_state: &FabricationUpgradeState,
    probation: &AuthorizedUpgradeProbationClearance,
    hardware_tracker: &HardwareReauthorizationTracker,
    required_machine_ids: BTreeSet<String>,
    retention_policy: &AuthorizedEvidenceRetentionPolicy,
    key_continuity: &VerifiedKeyContinuity,
    clock_continuity: &VerifiedClockContinuity,
    automatic_rollback: Option<&AutomaticRollbackTrigger>,
    authorized_at_unix_ms: u64,
    expires_at_unix_ms: u64,
    policy: &UpgradeFinalizationPolicy,
) -> Result<UpgradeFinalizationEvidence, UpgradeFinalizationError> {
    validate_policy(policy)?;
    if finalization_sequence == 0 {
        return Err(UpgradeFinalizationError::SequenceZero);
    }
    upgrade_state
        .validate_shape()
        .map_err(|error| UpgradeFinalizationError::Evidence(format!("{error:?}")))?;
    if upgrade_state.active_stage != UpgradeStage::Activated {
        return Err(UpgradeFinalizationError::UpgradeNotActivated);
    }
    if upgrade_state.evidence.handoff_digest != handoff.plan_digest
        || probation.evidence().handoff_digest != handoff.plan_digest
    {
        return Err(UpgradeFinalizationError::HandoffMismatch);
    }
    if authorized_at_unix_ms >= handoff.plan.finalization_deadline_unix_ms {
        return Err(UpgradeFinalizationError::FinalizationDeadlinePassed);
    }
    if !probation.permits_finalization(handoff.plan_digest, authorized_at_unix_ms) {
        return Err(UpgradeFinalizationError::ProbationExpired);
    }
    validate_machine_scope(&required_machine_ids, policy)?;
    let now_unix_s = authorized_at_unix_ms / 1_000;
    for machine_id in &required_machine_ids {
        if !hardware_tracker.permits(machine_id, handoff.plan_digest, now_unix_s) {
            return Err(UpgradeFinalizationError::MachineNotReauthorized(
                machine_id.clone(),
            ));
        }
    }
    if retention_policy.policy().effective_at_unix_s > now_unix_s {
        return Err(UpgradeFinalizationError::RetentionPolicyNotEffective);
    }
    if automatic_rollback.is_some() {
        return Err(UpgradeFinalizationError::RollbackTriggerPresent);
    }
    if authorized_at_unix_ms >= expires_at_unix_ms
        || expires_at_unix_ms.saturating_sub(authorized_at_unix_ms)
            > policy.maximum_authorization_duration_ms
        || expires_at_unix_ms > handoff.plan.finalization_deadline_unix_ms
    {
        return Err(UpgradeFinalizationError::InvalidWindow);
    }
    Ok(UpgradeFinalizationEvidence {
        schema_version: UPGRADE_FINALIZATION_SCHEMA.into(),
        finalization_sequence,
        handoff_digest: handoff.plan_digest,
        successor_state_digest: handoff.plan.successor.durable_state_digest,
        upgrade_state_digest: digest_upgrade_state(upgrade_state)
            .map_err(|error| UpgradeFinalizationError::Evidence(format!("{error:?}")))?,
        probation_clearance_digest: probation.evidence_digest(),
        hardware_reauthorization_tracker_digest: digest_hardware_reauthorization_tracker(
            hardware_tracker,
        )
        .map_err(|error| UpgradeFinalizationError::Evidence(format!("{error:?}")))?,
        required_machine_ids,
        retention_policy_digest: digest_evidence_retention_policy(retention_policy.policy())
            .map_err(|error| UpgradeFinalizationError::Evidence(format!("{error:?}")))?,
        key_continuity_digest: digest_key_continuity(key_continuity)
            .map_err(|error| UpgradeFinalizationError::Evidence(format!("{error:?}")))?,
        clock_continuity_digest: digest_clock_continuity(clock_continuity)
            .map_err(|error| UpgradeFinalizationError::Evidence(format!("{error:?}")))?,
        authorized_at_unix_ms,
        expires_at_unix_ms,
    })
}

pub fn digest_upgrade_finalization_evidence(
    evidence: &UpgradeFinalizationEvidence,
) -> Result<Sha256Digest, UpgradeFinalizationError> {
    validate_evidence(evidence)?;
    let bytes = serde_json::to_vec(evidence)
        .map_err(|error| UpgradeFinalizationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-finalization-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_upgrade_finalization(
    evidence: UpgradeFinalizationEvidence,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedUpgradeFinalization, UpgradeFinalizationError> {
    let evidence_digest = digest_upgrade_finalization_evidence(&evidence)?;
    if ceremony.purpose() != "upgrade-finalization" {
        return Err(UpgradeFinalizationError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != evidence_digest {
        return Err(UpgradeFinalizationError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedUpgradeFinalization {
        evidence,
        evidence_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

fn validate_policy(policy: &UpgradeFinalizationPolicy) -> Result<(), UpgradeFinalizationError> {
    if policy.minimum_reauthorized_machines == 0
        || policy.minimum_reauthorized_machines > MAX_FINALIZATION_MACHINES
        || policy.maximum_authorization_duration_ms == 0
    {
        return Err(UpgradeFinalizationError::InvalidPolicy);
    }
    Ok(())
}

fn validate_machine_scope(
    machine_ids: &BTreeSet<String>,
    policy: &UpgradeFinalizationPolicy,
) -> Result<(), UpgradeFinalizationError> {
    if machine_ids.len() < policy.minimum_reauthorized_machines
        || machine_ids.len() > MAX_FINALIZATION_MACHINES
        || machine_ids.iter().any(|machine_id| {
            machine_id.trim().is_empty()
                || machine_id != machine_id.trim()
                || machine_id.len() > 256
                || machine_id.chars().any(char::is_control)
        })
    {
        return Err(UpgradeFinalizationError::InvalidMachineScope);
    }
    Ok(())
}

fn validate_evidence(
    evidence: &UpgradeFinalizationEvidence,
) -> Result<(), UpgradeFinalizationError> {
    if evidence.schema_version != UPGRADE_FINALIZATION_SCHEMA {
        return Err(UpgradeFinalizationError::UnsupportedSchema);
    }
    if evidence.finalization_sequence == 0 {
        return Err(UpgradeFinalizationError::SequenceZero);
    }
    for (name, digest) in [
        ("handoff_digest", evidence.handoff_digest),
        ("successor_state_digest", evidence.successor_state_digest),
        ("upgrade_state_digest", evidence.upgrade_state_digest),
        (
            "probation_clearance_digest",
            evidence.probation_clearance_digest,
        ),
        (
            "hardware_reauthorization_tracker_digest",
            evidence.hardware_reauthorization_tracker_digest,
        ),
        ("retention_policy_digest", evidence.retention_policy_digest),
        ("key_continuity_digest", evidence.key_continuity_digest),
        ("clock_continuity_digest", evidence.clock_continuity_digest),
    ] {
        if digest.0 == [0; 32] {
            return Err(UpgradeFinalizationError::ZeroDigest(name));
        }
    }
    if evidence.required_machine_ids.is_empty()
        || evidence.required_machine_ids.len() > MAX_FINALIZATION_MACHINES
        || evidence.authorized_at_unix_ms >= evidence.expires_at_unix_ms
    {
        return Err(UpgradeFinalizationError::InvalidWindow);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    #[test]
    fn finalization_digest_binds_machine_scope() {
        let base = UpgradeFinalizationEvidence {
            schema_version: UPGRADE_FINALIZATION_SCHEMA.into(),
            finalization_sequence: 1,
            handoff_digest: sha256(b"handoff"),
            successor_state_digest: sha256(b"successor"),
            upgrade_state_digest: sha256(b"upgrade"),
            probation_clearance_digest: sha256(b"probation"),
            hardware_reauthorization_tracker_digest: sha256(b"hardware"),
            required_machine_ids: ["machine-a".to_string()].into_iter().collect(),
            retention_policy_digest: sha256(b"retention"),
            key_continuity_digest: sha256(b"keys"),
            clock_continuity_digest: sha256(b"clock"),
            authorized_at_unix_ms: 10,
            expires_at_unix_ms: 20,
        };
        let mut changed = base.clone();
        changed.required_machine_ids.insert("machine-b".into());
        assert_ne!(
            digest_upgrade_finalization_evidence(&base).unwrap(),
            digest_upgrade_finalization_evidence(&changed).unwrap()
        );
    }
}
