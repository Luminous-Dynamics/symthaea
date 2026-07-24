// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Portable bounded evidence bundles for post-upgrade operations.

use crate::automatic_rollback::AutomaticRollbackTrigger;
use crate::clock_continuity::VerifiedClockContinuity;
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::evidence_retention::EvidenceRetentionPolicy;
use crate::hardware_reauthorization_tracker::HardwareReauthorizationTracker;
use crate::key_continuity::VerifiedKeyContinuity;
use crate::upgrade_operational_replay::{
    UpgradeOperationalReplayContract, UpgradeOperationalReplayError,
    build_upgrade_operational_replay_contract, digest_upgrade_operational_replay_contract,
    verify_upgrade_operational_replay_contract,
};
use crate::upgrade_operational_state::FabricationUpgradeOperationalState;
use crate::upgrade_probation::UpgradeProbationEvidence;
use crate::upgrade_probation_tracker::UpgradeProbationTracker;
use crate::upgrade_state::FabricationUpgradeState;
use serde::{Deserialize, Serialize};

pub const UPGRADE_OPERATIONAL_BUNDLE_SCHEMA: &str =
    "symthaea.fabrication.upgrade-operational-bundle.v1";
pub const DEFAULT_MAX_UPGRADE_OPERATIONAL_BUNDLE_BYTES: usize = 32 * 1024 * 1024;
pub const HARD_MAX_UPGRADE_OPERATIONAL_BUNDLE_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeOperationalEvidenceBundle {
    pub schema_version: String,
    pub source_tree_digest: Sha256Digest,
    pub handoff_digest: Sha256Digest,
    pub upgrade_state: FabricationUpgradeState,
    pub probation_evidence: Option<UpgradeProbationEvidence>,
    pub probation_tracker: UpgradeProbationTracker,
    pub hardware_reauthorization_tracker: HardwareReauthorizationTracker,
    pub retention_policy: EvidenceRetentionPolicy,
    pub key_continuity: VerifiedKeyContinuity,
    pub clock_continuity: VerifiedClockContinuity,
    pub automatic_rollback: Option<AutomaticRollbackTrigger>,
    pub operational_state: FabricationUpgradeOperationalState,
    pub replay_contract: UpgradeOperationalReplayContract,
    pub replay_contract_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpgradeOperationalBundleLimits {
    pub maximum_bytes: usize,
}

impl Default for UpgradeOperationalBundleLimits {
    fn default() -> Self {
        Self {
            maximum_bytes: DEFAULT_MAX_UPGRADE_OPERATIONAL_BUNDLE_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeOperationalBundleError {
    UnsupportedSchema,
    InvalidLimits,
    BundleTooLarge { actual: usize, maximum: usize },
    ReplayDigestMismatch,
    ReplayMismatch(String),
    Replay(UpgradeOperationalReplayError),
    Encoding(String),
}

#[allow(clippy::too_many_arguments)]
pub fn build_upgrade_operational_evidence_bundle(
    source_tree_digest: Sha256Digest,
    handoff_digest: Sha256Digest,
    upgrade_state: FabricationUpgradeState,
    probation_evidence: Option<UpgradeProbationEvidence>,
    probation_tracker: UpgradeProbationTracker,
    hardware_reauthorization_tracker: HardwareReauthorizationTracker,
    retention_policy: EvidenceRetentionPolicy,
    key_continuity: VerifiedKeyContinuity,
    clock_continuity: VerifiedClockContinuity,
    automatic_rollback: Option<AutomaticRollbackTrigger>,
    operational_state: FabricationUpgradeOperationalState,
) -> Result<UpgradeOperationalEvidenceBundle, UpgradeOperationalBundleError> {
    let replay_contract = build_upgrade_operational_replay_contract(
        source_tree_digest,
        handoff_digest,
        &upgrade_state,
        probation_evidence.as_ref(),
        &probation_tracker,
        &hardware_reauthorization_tracker,
        &retention_policy,
        &key_continuity,
        &clock_continuity,
        automatic_rollback.as_ref(),
        &operational_state,
    )
    .map_err(UpgradeOperationalBundleError::Replay)?;
    let replay_contract_digest = digest_upgrade_operational_replay_contract(&replay_contract)
        .map_err(UpgradeOperationalBundleError::Replay)?;
    Ok(UpgradeOperationalEvidenceBundle {
        schema_version: UPGRADE_OPERATIONAL_BUNDLE_SCHEMA.into(),
        source_tree_digest,
        handoff_digest,
        upgrade_state,
        probation_evidence,
        probation_tracker,
        hardware_reauthorization_tracker,
        retention_policy,
        key_continuity,
        clock_continuity,
        automatic_rollback,
        operational_state,
        replay_contract,
        replay_contract_digest,
    })
}

pub fn verify_upgrade_operational_evidence_bundle(
    bundle: &UpgradeOperationalEvidenceBundle,
    limits: &UpgradeOperationalBundleLimits,
) -> Result<(), UpgradeOperationalBundleError> {
    validate_limits(limits)?;
    if bundle.schema_version != UPGRADE_OPERATIONAL_BUNDLE_SCHEMA {
        return Err(UpgradeOperationalBundleError::UnsupportedSchema);
    }
    let digest = digest_upgrade_operational_replay_contract(&bundle.replay_contract)
        .map_err(UpgradeOperationalBundleError::Replay)?;
    if digest != bundle.replay_contract_digest {
        return Err(UpgradeOperationalBundleError::ReplayDigestMismatch);
    }
    let report = verify_upgrade_operational_replay_contract(
        &bundle.replay_contract,
        bundle.source_tree_digest,
        bundle.handoff_digest,
        &bundle.upgrade_state,
        bundle.probation_evidence.as_ref(),
        &bundle.probation_tracker,
        &bundle.hardware_reauthorization_tracker,
        &bundle.retention_policy,
        &bundle.key_continuity,
        &bundle.clock_continuity,
        bundle.automatic_rollback.as_ref(),
        &bundle.operational_state,
    )
    .map_err(UpgradeOperationalBundleError::Replay)?;
    if !report.exact() {
        return Err(UpgradeOperationalBundleError::ReplayMismatch(format!(
            "{:?}",
            report.mismatches
        )));
    }
    Ok(())
}

pub fn encode_upgrade_operational_evidence_bundle(
    bundle: &UpgradeOperationalEvidenceBundle,
    limits: &UpgradeOperationalBundleLimits,
) -> Result<Vec<u8>, UpgradeOperationalBundleError> {
    verify_upgrade_operational_evidence_bundle(bundle, limits)?;
    let bytes = serde_json::to_vec(bundle)
        .map_err(|error| UpgradeOperationalBundleError::Encoding(error.to_string()))?;
    if bytes.len() > limits.maximum_bytes {
        return Err(UpgradeOperationalBundleError::BundleTooLarge {
            actual: bytes.len(),
            maximum: limits.maximum_bytes,
        });
    }
    Ok(bytes)
}

pub fn decode_upgrade_operational_evidence_bundle(
    bytes: &[u8],
    limits: &UpgradeOperationalBundleLimits,
) -> Result<UpgradeOperationalEvidenceBundle, UpgradeOperationalBundleError> {
    validate_limits(limits)?;
    if bytes.len() > limits.maximum_bytes {
        return Err(UpgradeOperationalBundleError::BundleTooLarge {
            actual: bytes.len(),
            maximum: limits.maximum_bytes,
        });
    }
    let bundle: UpgradeOperationalEvidenceBundle = serde_json::from_slice(bytes)
        .map_err(|error| UpgradeOperationalBundleError::Encoding(error.to_string()))?;
    verify_upgrade_operational_evidence_bundle(&bundle, limits)?;
    Ok(bundle)
}

pub fn digest_upgrade_operational_evidence_bundle(
    bundle: &UpgradeOperationalEvidenceBundle,
    limits: &UpgradeOperationalBundleLimits,
) -> Result<Sha256Digest, UpgradeOperationalBundleError> {
    let bytes = encode_upgrade_operational_evidence_bundle(bundle, limits)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-operational-bundle-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn validate_limits(
    limits: &UpgradeOperationalBundleLimits,
) -> Result<(), UpgradeOperationalBundleError> {
    if limits.maximum_bytes == 0 || limits.maximum_bytes > HARD_MAX_UPGRADE_OPERATIONAL_BUNDLE_BYTES
    {
        return Err(UpgradeOperationalBundleError::InvalidLimits);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_input_is_rejected_before_decode() {
        let limits = UpgradeOperationalBundleLimits { maximum_bytes: 8 };
        assert_eq!(
            decode_upgrade_operational_evidence_bundle(&[0; 9], &limits),
            Err(UpgradeOperationalBundleError::BundleTooLarge {
                actual: 9,
                maximum: 8
            })
        );
    }
}
