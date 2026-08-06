// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authenticated storage-profile and sudden-power-loss evidence contracts.
//!
//! Process termination, VM reset, and physical device power interruption are
//! deliberately separate evidence classes. A process-crash campaign can test
//! recovery logic, but it cannot satisfy a physical-power-loss promotion gate.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

pub const CHECKPOINT_STORAGE_PROFILE_SCHEMA: &str = "symthaea.checkpoint-storage-profile.v1";
pub const CHECKPOINT_STORAGE_PROFILE_ATTESTATION_SCHEMA: &str =
    "symthaea.checkpoint-storage-profile-attestation.v1";
pub const CHECKPOINT_POWER_LOSS_CAMPAIGN_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-campaign.v1";
pub const CHECKPOINT_POWER_LOSS_RESULT_SCHEMA: &str = "symthaea.checkpoint-power-loss-result.v1";
pub const CHECKPOINT_POWER_LOSS_EVIDENCE_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-evidence.v1";

const STORAGE_PROFILE_DIGEST_DOMAIN: &[u8] = b"symthaea-checkpoint-storage-profile-digest-v1\0";
const STORAGE_PROFILE_ATTESTATION_DOMAIN: &[u8] =
    b"symthaea-checkpoint-storage-profile-attestation-v1\0";
const POWER_LOSS_CAMPAIGN_DIGEST_DOMAIN: &[u8] =
    b"symthaea-checkpoint-power-loss-campaign-digest-v1\0";
const POWER_LOSS_RESULT_ATTESTATION_DOMAIN: &[u8] =
    b"symthaea-checkpoint-power-loss-result-attestation-v1\0";

pub const MAX_CHECKPOINT_POWER_LOSS_TRIALS: usize = 4096;
pub const MAX_CHECKPOINT_STORAGE_PROFILES: usize = 64;
pub const MAX_CHECKPOINT_POWER_LOSS_EVIDENCE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckpointFilesystemKind {
    Ext4,
    Xfs,
    Btrfs,
    Zfs,
    F2fs,
    Apfs,
    Ntfs,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckpointWriteCachePolicy {
    Disabled,
    VolatileWithFlush,
    VolatileWithForceUnitAccess,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckpointBarrierPolicy {
    Enabled,
    Disabled,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckpointPowerLossEvidenceClass {
    ProcessCrashSimulation,
    VirtualMachinePowerCut,
    PhysicalDevicePowerCut,
}

impl CheckpointPowerLossEvidenceClass {
    pub fn is_physical(self) -> bool {
        self == Self::PhysicalDevicePowerCut
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckpointDurabilityBoundary {
    AfterDataWriteBeforeFileSync,
    AfterFileSyncBeforePublication,
    AfterPublicationBeforeDirectorySync,
    AfterDirectorySyncBeforeAcknowledgement,
    DuringAtomicReplacement,
    DuringCompactionReplacement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckpointPowerLossRecoveryOutcome {
    CleanRecovery,
    FailClosedIndeterminate,
    DetectedCorruption,
    SilentCorruption,
    Unrecoverable,
}

impl CheckpointPowerLossRecoveryOutcome {
    pub fn promotion_safe(self) -> bool {
        matches!(self, Self::CleanRecovery | Self::FailClosedIndeterminate)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointStorageProfileManifest {
    pub schema: String,
    pub profile_id: [u8; 16],
    pub filesystem_kind: CheckpointFilesystemKind,
    pub filesystem_instance_binding: [u8; 32],
    pub block_device_binding: [u8; 32],
    pub mount_options_digest: [u8; 32],
    pub kernel_release_digest: [u8; 32],
    pub storage_stack_digest: [u8; 32],
    pub logical_sector_bytes: u32,
    pub physical_sector_bytes: u32,
    pub atomic_write_unit_bytes: u32,
    pub write_cache_policy: CheckpointWriteCachePolicy,
    pub barrier_policy: CheckpointBarrierPolicy,
    pub flush_supported: bool,
    pub force_unit_access_supported: bool,
    pub stable_write_supported: bool,
    pub volatile_write_cache_present: bool,
    pub observed_at_unix_seconds: u64,
}

impl CheckpointStorageProfileManifest {
    pub fn validate(&self) -> Result<(), CheckpointStorageEvidenceError> {
        if self.schema != CHECKPOINT_STORAGE_PROFILE_SCHEMA
            || self.profile_id == [0u8; 16]
            || self.filesystem_instance_binding == [0u8; 32]
            || self.block_device_binding == [0u8; 32]
            || self.mount_options_digest == [0u8; 32]
            || self.kernel_release_digest == [0u8; 32]
            || self.storage_stack_digest == [0u8; 32]
            || self.observed_at_unix_seconds == 0
        {
            return Err(CheckpointStorageEvidenceError::InvalidStorageProfile);
        }
        for size in [
            self.logical_sector_bytes,
            self.physical_sector_bytes,
            self.atomic_write_unit_bytes,
        ] {
            if !(512..=1_048_576).contains(&size) || !size.is_power_of_two() {
                return Err(CheckpointStorageEvidenceError::InvalidStorageProfile);
            }
        }
        if self.physical_sector_bytes < self.logical_sector_bytes
            || self.atomic_write_unit_bytes < self.logical_sector_bytes
        {
            return Err(CheckpointStorageEvidenceError::InvalidStorageProfile);
        }
        if self.volatile_write_cache_present
            && !(self.flush_supported || self.force_unit_access_supported)
        {
            return Err(CheckpointStorageEvidenceError::InvalidStorageProfile);
        }
        match self.write_cache_policy {
            CheckpointWriteCachePolicy::Disabled => {
                if self.volatile_write_cache_present {
                    return Err(CheckpointStorageEvidenceError::InvalidStorageProfile);
                }
            }
            CheckpointWriteCachePolicy::VolatileWithFlush => {
                if !self.volatile_write_cache_present || !self.flush_supported {
                    return Err(CheckpointStorageEvidenceError::InvalidStorageProfile);
                }
            }
            CheckpointWriteCachePolicy::VolatileWithForceUnitAccess => {
                if !self.volatile_write_cache_present || !self.force_unit_access_supported {
                    return Err(CheckpointStorageEvidenceError::InvalidStorageProfile);
                }
            }
            CheckpointWriteCachePolicy::Unknown => {}
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<[u8; 32], CheckpointStorageEvidenceError> {
        self.validate()?;
        digest_serialized(STORAGE_PROFILE_DIGEST_DOMAIN, self)
    }

    pub fn has_explicit_durability_semantics(&self) -> bool {
        self.barrier_policy == CheckpointBarrierPolicy::Enabled
            && self.write_cache_policy != CheckpointWriteCachePolicy::Unknown
            && (!self.volatile_write_cache_present
                || self.flush_supported
                || self.force_unit_access_supported)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointStorageProfileAttestationKeyId(pub [u8; 16]);

impl CheckpointStorageProfileAttestationKeyId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointStorageEvidenceError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointStorageEvidenceError::InvalidKey);
        }
        Ok(Self(bytes))
    }
}

pub struct CheckpointStorageProfileAttestationKey {
    id: CheckpointStorageProfileAttestationKeyId,
    bytes: [u8; 32],
}

impl CheckpointStorageProfileAttestationKey {
    pub fn new(
        id: CheckpointStorageProfileAttestationKeyId,
        bytes: [u8; 32],
    ) -> Result<Self, CheckpointStorageEvidenceError> {
        if bytes == [0u8; 32] {
            return Err(CheckpointStorageEvidenceError::InvalidKey);
        }
        Ok(Self { id, bytes })
    }

    pub fn id(&self) -> CheckpointStorageProfileAttestationKeyId {
        self.id
    }
}

impl Drop for CheckpointStorageProfileAttestationKey {
    fn drop(&mut self) {
        self.bytes.zeroize();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointStorageProfileAttestationWire {
    schema: String,
    key_id: CheckpointStorageProfileAttestationKeyId,
    body: Vec<u8>,
    authentication_tag: [u8; 32],
}

pub struct CheckpointStorageProfileAuthority {
    key: CheckpointStorageProfileAttestationKey,
}

impl CheckpointStorageProfileAuthority {
    pub fn new(key: CheckpointStorageProfileAttestationKey) -> Self {
        Self { key }
    }

    pub fn key_id(&self) -> CheckpointStorageProfileAttestationKeyId {
        self.key.id()
    }

    pub fn seal_profile(
        &self,
        profile: &CheckpointStorageProfileManifest,
    ) -> Result<Vec<u8>, CheckpointStorageEvidenceError> {
        profile.validate()?;
        let body =
            postcard::to_stdvec(profile).map_err(|_| CheckpointStorageEvidenceError::Encoding)?;
        let wire = CheckpointStorageProfileAttestationWire {
            schema: CHECKPOINT_STORAGE_PROFILE_ATTESTATION_SCHEMA.to_owned(),
            key_id: self.key.id(),
            authentication_tag: keyed_authenticate(
                STORAGE_PROFILE_ATTESTATION_DOMAIN,
                &body,
                &self.key.bytes,
            ),
            body,
        };
        bounded_encode(&wire)
    }

    pub fn open_profile(
        &self,
        encoded: &[u8],
        expected_key_id: CheckpointStorageProfileAttestationKeyId,
    ) -> Result<CheckpointStorageProfileManifest, CheckpointStorageEvidenceError> {
        let wire: CheckpointStorageProfileAttestationWire = bounded_decode(encoded)?;
        if wire.schema != CHECKPOINT_STORAGE_PROFILE_ATTESTATION_SCHEMA
            || wire.key_id != expected_key_id
            || wire.key_id != self.key.id()
            || !constant_time_equal(
                &wire.authentication_tag,
                &keyed_authenticate(
                    STORAGE_PROFILE_ATTESTATION_DOMAIN,
                    &wire.body,
                    &self.key.bytes,
                ),
            )
        {
            return Err(CheckpointStorageEvidenceError::AuthenticationFailed);
        }
        let profile: CheckpointStorageProfileManifest = postcard::from_bytes(&wire.body)
            .map_err(|_| CheckpointStorageEvidenceError::Encoding)?;
        profile.validate()?;
        Ok(profile)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossTrialPlan {
    pub trial_id: [u8; 16],
    pub storage_profile_digest: [u8; 32],
    pub evidence_class: CheckpointPowerLossEvidenceClass,
    pub durability_boundary: CheckpointDurabilityBoundary,
    pub workload_digest: [u8; 32],
    pub expected_pre_power_loss_digest: [u8; 32],
}

impl CheckpointPowerLossTrialPlan {
    pub fn validate(&self) -> Result<(), CheckpointStorageEvidenceError> {
        if self.trial_id == [0u8; 16]
            || self.storage_profile_digest == [0u8; 32]
            || self.workload_digest == [0u8; 32]
            || self.expected_pre_power_loss_digest == [0u8; 32]
        {
            return Err(CheckpointStorageEvidenceError::InvalidTrial);
        }
        Ok(())
    }
}

/// Identifies the authority key that signs power-loss evidence for a campaign. Only ever
/// referenced by identifier -- no corresponding secret-key type exists anywhere in this
/// crate (unlike `CheckpointStorageProfileAttestationKeyId`, which pairs with a real signing
/// key), so this stays a bare identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossEvidenceKeyId(pub [u8; 16]);

impl CheckpointPowerLossEvidenceKeyId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointStorageEvidenceError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointStorageEvidenceError::InvalidKey);
        }
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossCampaignPlan {
    pub schema: String,
    pub campaign_id: [u8; 16],
    pub storage_profiles: Vec<[u8; 32]>,
    pub storage_profile_authority_key_id: CheckpointStorageProfileAttestationKeyId,
    pub power_loss_evidence_authority_key_id: CheckpointPowerLossEvidenceKeyId,
    /// Must match every participating lab's own
    /// `CheckpointPowerLossLabManifest::test_harness_binding` (see
    /// `checkpoint_power_loss_operations.rs::CheckpointPowerLossOperationsPlan::
    /// validate_against`'s cross-check) -- scopes this campaign to one specific test harness.
    pub test_harness_digest: [u8; 32],
    /// Must match every participating lab's own
    /// `CheckpointPowerLossLabManifest::power_controller_binding` -- scopes this campaign to
    /// one specific power controller.
    pub power_controller_binding: [u8; 32],
    pub power_controller_calibration_digest: [u8; 32],
    pub operator_protocol_digest: [u8; 32],
    pub trials: Vec<CheckpointPowerLossTrialPlan>,
    pub minimum_physical_trials: u32,
    pub require_all_durability_boundaries: bool,
}

impl CheckpointPowerLossCampaignPlan {
    pub fn validate(&self) -> Result<(), CheckpointStorageEvidenceError> {
        if self.schema != CHECKPOINT_POWER_LOSS_CAMPAIGN_SCHEMA
            || self.campaign_id == [0u8; 16]
            || self.storage_profiles.is_empty()
            || self.storage_profiles.len() > MAX_CHECKPOINT_STORAGE_PROFILES
            || self.storage_profile_authority_key_id.0 == [0u8; 16]
            || self.power_loss_evidence_authority_key_id.0 == [0u8; 16]
            || self.trials.is_empty()
            || self.trials.len() > MAX_CHECKPOINT_POWER_LOSS_TRIALS
            || self.test_harness_digest == [0u8; 32]
            || self.power_controller_binding == [0u8; 32]
            || self.power_controller_calibration_digest == [0u8; 32]
            || self.operator_protocol_digest == [0u8; 32]
        {
            return Err(CheckpointStorageEvidenceError::InvalidCampaign);
        }
        let profile_set: HashSet<[u8; 32]> = self.storage_profiles.iter().copied().collect();
        if profile_set.len() != self.storage_profiles.len() || profile_set.contains(&[0u8; 32]) {
            return Err(CheckpointStorageEvidenceError::InvalidCampaign);
        }
        let mut trial_ids = HashSet::with_capacity(self.trials.len());
        let mut trial_profiles = HashSet::with_capacity(self.storage_profiles.len());
        let mut physical_trials = 0usize;
        for trial in &self.trials {
            trial.validate()?;
            if !profile_set.contains(&trial.storage_profile_digest)
                || !trial_ids.insert(trial.trial_id)
            {
                return Err(CheckpointStorageEvidenceError::InvalidCampaign);
            }
            trial_profiles.insert(trial.storage_profile_digest);
            if trial.evidence_class.is_physical() {
                physical_trials += 1;
            }
        }
        if trial_profiles != profile_set {
            return Err(CheckpointStorageEvidenceError::InvalidCampaign);
        }
        if self.minimum_physical_trials as usize > physical_trials {
            return Err(CheckpointStorageEvidenceError::InvalidCampaign);
        }
        if self.require_all_durability_boundaries {
            let observed: HashSet<CheckpointDurabilityBoundary> = self
                .trials
                .iter()
                .map(|trial| trial.durability_boundary)
                .collect();
            if observed.len() != 6 {
                return Err(CheckpointStorageEvidenceError::InvalidCampaign);
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<[u8; 32], CheckpointStorageEvidenceError> {
        self.validate()?;
        digest_serialized(POWER_LOSS_CAMPAIGN_DIGEST_DOMAIN, self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossTrialResult {
    pub schema: String,
    pub campaign_id: [u8; 16],
    pub campaign_digest: [u8; 32],
    pub trial_id: [u8; 16],
    pub storage_profile_digest: [u8; 32],
    pub evidence_class: CheckpointPowerLossEvidenceClass,
    pub durability_boundary: CheckpointDurabilityBoundary,
    pub workload_digest: [u8; 32],
    pub pre_power_loss_digest: [u8; 32],
    pub recovered_state_digest: [u8; 32],
    /// Digest of the evidence proving the power-loss event itself was actually observed
    /// (distinct from `recovered_state_digest`, which is evidence of the RECOVERY). Consumed
    /// by `checkpoint_power_loss_operations.rs`'s execution-journal cross-check (the journal's
    /// own `PowerEventObserved` entry must carry this same digest).
    pub power_event_evidence_digest: [u8; 32],
    pub outcome: CheckpointPowerLossRecoveryOutcome,
    pub filesystem_consistency_verified: bool,
    pub application_consistency_verified: bool,
    pub completed_at_unix_seconds: u64,
}

impl CheckpointPowerLossTrialResult {
    pub fn validate_against(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
    ) -> Result<(), CheckpointStorageEvidenceError> {
        campaign.validate()?;
        let campaign_digest = campaign.digest()?;
        let trial = campaign
            .trials
            .iter()
            .find(|trial| trial.trial_id == self.trial_id)
            .ok_or(CheckpointStorageEvidenceError::UnknownTrial)?;
        if self.schema != CHECKPOINT_POWER_LOSS_RESULT_SCHEMA
            || self.campaign_id != campaign.campaign_id
            || self.campaign_digest != campaign_digest
            || self.storage_profile_digest != trial.storage_profile_digest
            || self.evidence_class != trial.evidence_class
            || self.durability_boundary != trial.durability_boundary
            || self.workload_digest != trial.workload_digest
            || self.pre_power_loss_digest != trial.expected_pre_power_loss_digest
            || self.recovered_state_digest == [0u8; 32]
            || self.power_event_evidence_digest == [0u8; 32]
            || self.completed_at_unix_seconds == 0
        {
            return Err(CheckpointStorageEvidenceError::TrialBindingMismatch);
        }
        if self.outcome == CheckpointPowerLossRecoveryOutcome::CleanRecovery
            && (!self.filesystem_consistency_verified || !self.application_consistency_verified)
        {
            return Err(CheckpointStorageEvidenceError::InvalidTrial);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossCampaignEvidence {
    pub schema: String,
    pub campaign_id: [u8; 16],
    pub campaign_digest: [u8; 32],
    pub results: Vec<CheckpointPowerLossTrialResult>,
}

impl CheckpointPowerLossCampaignEvidence {
    pub fn validate_against(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
    ) -> Result<(), CheckpointStorageEvidenceError> {
        campaign.validate()?;
        if self.schema != CHECKPOINT_POWER_LOSS_EVIDENCE_SCHEMA
            || self.campaign_id != campaign.campaign_id
            || self.campaign_digest != campaign.digest()?
            || self.results.len() > campaign.trials.len()
            || self.results.len() > MAX_CHECKPOINT_POWER_LOSS_TRIALS
        {
            return Err(CheckpointStorageEvidenceError::InvalidCampaign);
        }
        let mut result_ids = HashSet::with_capacity(self.results.len());
        for result in &self.results {
            result.validate_against(campaign)?;
            if !result_ids.insert(result.trial_id) {
                return Err(CheckpointStorageEvidenceError::DuplicateTrial);
            }
        }
        Ok(())
    }

    pub fn summary(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
    ) -> Result<CheckpointPowerLossCampaignSummary, CheckpointStorageEvidenceError> {
        self.validate_against(campaign)?;
        let mut summary = CheckpointPowerLossCampaignSummary {
            planned_trials: campaign.trials.len(),
            completed_trials: self.results.len(),
            ..CheckpointPowerLossCampaignSummary::default()
        };
        let mut boundaries = HashSet::new();
        let mut completed_profiles = HashSet::new();
        for result in &self.results {
            boundaries.insert(result.durability_boundary);
            completed_profiles.insert(result.storage_profile_digest);
            match result.evidence_class {
                CheckpointPowerLossEvidenceClass::ProcessCrashSimulation => {
                    summary.process_crash_trials += 1;
                }
                CheckpointPowerLossEvidenceClass::VirtualMachinePowerCut => {
                    summary.virtual_power_cut_trials += 1;
                }
                CheckpointPowerLossEvidenceClass::PhysicalDevicePowerCut => {
                    summary.physical_power_cut_trials += 1;
                }
            }
            match result.outcome {
                CheckpointPowerLossRecoveryOutcome::CleanRecovery => {
                    summary.clean_recoveries += 1;
                }
                CheckpointPowerLossRecoveryOutcome::FailClosedIndeterminate => {
                    summary.fail_closed_recoveries += 1;
                }
                CheckpointPowerLossRecoveryOutcome::DetectedCorruption => {
                    summary.detected_corruptions += 1;
                }
                CheckpointPowerLossRecoveryOutcome::SilentCorruption => {
                    summary.silent_corruptions += 1;
                }
                CheckpointPowerLossRecoveryOutcome::Unrecoverable => {
                    summary.unrecoverable_trials += 1;
                }
            }
        }
        summary.storage_profiles = completed_profiles.len();
        summary.covered_durability_boundaries = boundaries.len();
        summary.all_completed_results_promotion_safe = self
            .results
            .iter()
            .all(|result| result.outcome.promotion_safe());
        Ok(summary)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossCampaignSummary {
    pub planned_trials: usize,
    pub completed_trials: usize,
    pub storage_profiles: usize,
    pub process_crash_trials: usize,
    pub virtual_power_cut_trials: usize,
    pub physical_power_cut_trials: usize,
    pub clean_recoveries: usize,
    pub fail_closed_recoveries: usize,
    pub detected_corruptions: usize,
    pub silent_corruptions: usize,
    pub unrecoverable_trials: usize,
    pub covered_durability_boundaries: usize,
    pub all_completed_results_promotion_safe: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointPowerLossEvidenceWire {
    schema: String,
    key_id: CheckpointStorageProfileAttestationKeyId,
    body: Vec<u8>,
    authentication_tag: [u8; 32],
}

impl CheckpointStorageProfileAuthority {
    pub fn seal_campaign_evidence(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        evidence: &CheckpointPowerLossCampaignEvidence,
    ) -> Result<Vec<u8>, CheckpointStorageEvidenceError> {
        evidence.validate_against(campaign)?;
        let body =
            postcard::to_stdvec(evidence).map_err(|_| CheckpointStorageEvidenceError::Encoding)?;
        let wire = CheckpointPowerLossEvidenceWire {
            schema: CHECKPOINT_POWER_LOSS_EVIDENCE_SCHEMA.to_owned(),
            key_id: self.key.id(),
            authentication_tag: keyed_authenticate(
                POWER_LOSS_RESULT_ATTESTATION_DOMAIN,
                &body,
                &self.key.bytes,
            ),
            body,
        };
        bounded_encode(&wire)
    }

    pub fn open_campaign_evidence(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        encoded: &[u8],
        expected_key_id: CheckpointStorageProfileAttestationKeyId,
    ) -> Result<CheckpointPowerLossCampaignEvidence, CheckpointStorageEvidenceError> {
        let wire: CheckpointPowerLossEvidenceWire = bounded_decode(encoded)?;
        if wire.schema != CHECKPOINT_POWER_LOSS_EVIDENCE_SCHEMA
            || wire.key_id != expected_key_id
            || wire.key_id != self.key.id()
            || !constant_time_equal(
                &wire.authentication_tag,
                &keyed_authenticate(
                    POWER_LOSS_RESULT_ATTESTATION_DOMAIN,
                    &wire.body,
                    &self.key.bytes,
                ),
            )
        {
            return Err(CheckpointStorageEvidenceError::AuthenticationFailed);
        }
        let evidence: CheckpointPowerLossCampaignEvidence = postcard::from_bytes(&wire.body)
            .map_err(|_| CheckpointStorageEvidenceError::Encoding)?;
        evidence.validate_against(campaign)?;
        Ok(evidence)
    }
}

#[derive(Debug)]
pub enum CheckpointStorageEvidenceError {
    InvalidKey,
    InvalidStorageProfile,
    InvalidCampaign,
    InvalidTrial,
    UnknownTrial,
    DuplicateTrial,
    TrialBindingMismatch,
    AuthenticationFailed,
    Encoding,
    TooLarge,
}

impl std::fmt::Display for CheckpointStorageEvidenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidKey => "invalid storage evidence key",
            Self::InvalidStorageProfile => "invalid checkpoint storage profile",
            Self::InvalidCampaign => "invalid checkpoint power-loss campaign",
            Self::InvalidTrial => "invalid checkpoint power-loss trial",
            Self::UnknownTrial => "power-loss result references an unknown trial",
            Self::DuplicateTrial => "power-loss evidence contains a duplicate trial",
            Self::TrialBindingMismatch => {
                "power-loss result does not match its preregistered trial"
            }
            Self::AuthenticationFailed => "storage evidence authentication failed",
            Self::Encoding => "storage evidence encoding failed",
            Self::TooLarge => "storage evidence exceeds its bounded size",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for CheckpointStorageEvidenceError {}

fn digest_serialized<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<[u8; 32], CheckpointStorageEvidenceError> {
    let body = postcard::to_stdvec(value).map_err(|_| CheckpointStorageEvidenceError::Encoding)?;
    let mut input = Vec::with_capacity(domain.len() + body.len());
    input.extend_from_slice(domain);
    input.extend_from_slice(&body);
    Ok(*blake3::hash(&input).as_bytes())
}

fn keyed_authenticate(domain: &[u8], body: &[u8], key: &[u8; 32]) -> [u8; 32] {
    let mut input = Vec::with_capacity(domain.len() + body.len());
    input.extend_from_slice(domain);
    input.extend_from_slice(body);
    *blake3::keyed_hash(key, &input).as_bytes()
}

fn bounded_encode<T: Serialize>(value: &T) -> Result<Vec<u8>, CheckpointStorageEvidenceError> {
    let encoded =
        postcard::to_stdvec(value).map_err(|_| CheckpointStorageEvidenceError::Encoding)?;
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_POWER_LOSS_EVIDENCE_BYTES {
        return Err(CheckpointStorageEvidenceError::TooLarge);
    }
    Ok(encoded)
}

fn bounded_decode<'a, T: Deserialize<'a>>(
    encoded: &'a [u8],
) -> Result<T, CheckpointStorageEvidenceError> {
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_POWER_LOSS_EVIDENCE_BYTES {
        return Err(CheckpointStorageEvidenceError::TooLarge);
    }
    postcard::from_bytes(encoded).map_err(|_| CheckpointStorageEvidenceError::Encoding)
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |difference, (left, right)| difference | (left ^ right))
        == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> CheckpointStorageProfileManifest {
        CheckpointStorageProfileManifest {
            schema: CHECKPOINT_STORAGE_PROFILE_SCHEMA.to_owned(),
            profile_id: [0x11; 16],
            filesystem_kind: CheckpointFilesystemKind::Ext4,
            filesystem_instance_binding: [0x12; 32],
            block_device_binding: [0x13; 32],
            mount_options_digest: [0x14; 32],
            kernel_release_digest: [0x15; 32],
            storage_stack_digest: [0x16; 32],
            logical_sector_bytes: 512,
            physical_sector_bytes: 4096,
            atomic_write_unit_bytes: 4096,
            write_cache_policy: CheckpointWriteCachePolicy::VolatileWithFlush,
            barrier_policy: CheckpointBarrierPolicy::Enabled,
            flush_supported: true,
            force_unit_access_supported: false,
            stable_write_supported: false,
            volatile_write_cache_present: true,
            observed_at_unix_seconds: 1_900_000_000,
        }
    }

    fn authority() -> CheckpointStorageProfileAuthority {
        CheckpointStorageProfileAuthority::new(
            CheckpointStorageProfileAttestationKey::new(
                CheckpointStorageProfileAttestationKeyId::new([0x21; 16]).unwrap(),
                [0x22; 32],
            )
            .unwrap(),
        )
    }

    fn campaign(profile_digest: [u8; 32]) -> CheckpointPowerLossCampaignPlan {
        let boundaries = [
            CheckpointDurabilityBoundary::AfterDataWriteBeforeFileSync,
            CheckpointDurabilityBoundary::AfterFileSyncBeforePublication,
            CheckpointDurabilityBoundary::AfterPublicationBeforeDirectorySync,
            CheckpointDurabilityBoundary::AfterDirectorySyncBeforeAcknowledgement,
            CheckpointDurabilityBoundary::DuringAtomicReplacement,
            CheckpointDurabilityBoundary::DuringCompactionReplacement,
        ];
        CheckpointPowerLossCampaignPlan {
            schema: CHECKPOINT_POWER_LOSS_CAMPAIGN_SCHEMA.to_owned(),
            campaign_id: [0x31; 16],
            storage_profiles: vec![profile_digest],
            storage_profile_authority_key_id: CheckpointStorageProfileAttestationKeyId::new(
                [0x93; 16],
            )
            .unwrap(),
            power_loss_evidence_authority_key_id: CheckpointPowerLossEvidenceKeyId::new([0x94; 16])
                .unwrap(),
            power_controller_calibration_digest: [0x95; 32],
            operator_protocol_digest: [0x96; 32],
            trials: boundaries
                .into_iter()
                .enumerate()
                .map(
                    |(index, durability_boundary)| CheckpointPowerLossTrialPlan {
                        trial_id: [0x40 + index as u8; 16],
                        storage_profile_digest: profile_digest,
                        evidence_class: if index < 2 {
                            CheckpointPowerLossEvidenceClass::ProcessCrashSimulation
                        } else {
                            CheckpointPowerLossEvidenceClass::PhysicalDevicePowerCut
                        },
                        durability_boundary,
                        workload_digest: [0x51 + index as u8; 32],
                        expected_pre_power_loss_digest: [0x61 + index as u8; 32],
                    },
                )
                .collect(),
            minimum_physical_trials: 4,
            require_all_durability_boundaries: true,
            test_harness_digest: [0x91; 32],
            power_controller_binding: [0x92; 32],
        }
    }

    #[test]
    fn storage_profile_attestation_round_trips() {
        let authority = authority();
        let encoded = authority.seal_profile(&profile()).unwrap();
        let opened = authority
            .open_profile(&encoded, authority.key_id())
            .unwrap();
        assert_eq!(opened, profile());
        assert!(opened.has_explicit_durability_semantics());
    }

    #[test]
    fn volatile_cache_without_flush_or_fua_fails_closed() {
        let mut profile = profile();
        profile.flush_supported = false;
        assert!(profile.validate().is_err());
    }

    #[test]
    fn process_crashes_do_not_count_as_physical_power_loss() {
        assert!(!CheckpointPowerLossEvidenceClass::ProcessCrashSimulation.is_physical());
        assert!(CheckpointPowerLossEvidenceClass::PhysicalDevicePowerCut.is_physical());
    }

    #[test]
    fn campaign_requires_unique_trials_and_complete_boundary_coverage() {
        let digest = profile().digest().unwrap();
        let mut campaign = campaign(digest);
        assert!(campaign.validate().is_ok());
        campaign.trials[1].trial_id = campaign.trials[0].trial_id;
        assert!(campaign.validate().is_err());
    }

    #[test]
    fn every_declared_storage_profile_requires_a_trial() {
        let first = profile().digest().unwrap();
        let mut campaign = campaign(first);
        campaign.storage_profiles.push([0xee; 32]);
        assert!(campaign.validate().is_err());
    }

    #[test]
    fn evidence_is_bound_to_the_preregistered_campaign() {
        let digest = profile().digest().unwrap();
        let campaign = campaign(digest);
        let campaign_digest = campaign.digest().unwrap();
        let first = &campaign.trials[0];
        let evidence = CheckpointPowerLossCampaignEvidence {
            schema: CHECKPOINT_POWER_LOSS_EVIDENCE_SCHEMA.to_owned(),
            campaign_id: campaign.campaign_id,
            campaign_digest,
            results: vec![CheckpointPowerLossTrialResult {
                schema: CHECKPOINT_POWER_LOSS_RESULT_SCHEMA.to_owned(),
                campaign_id: campaign.campaign_id,
                campaign_digest,
                trial_id: first.trial_id,
                storage_profile_digest: first.storage_profile_digest,
                evidence_class: first.evidence_class,
                durability_boundary: first.durability_boundary,
                workload_digest: first.workload_digest,
                pre_power_loss_digest: first.expected_pre_power_loss_digest,
                recovered_state_digest: [0x71; 32],
                power_event_evidence_digest: [0x72; 32],
                outcome: CheckpointPowerLossRecoveryOutcome::CleanRecovery,
                filesystem_consistency_verified: true,
                application_consistency_verified: true,
                completed_at_unix_seconds: 1_900_000_100,
            }],
        };
        let authority = authority();
        let encoded = authority
            .seal_campaign_evidence(&campaign, &evidence)
            .unwrap();
        let opened = authority
            .open_campaign_evidence(&campaign, &encoded, authority.key_id())
            .unwrap();
        assert_eq!(opened, evidence);
    }

    #[test]
    fn clean_recovery_requires_filesystem_and_application_checks() {
        let digest = profile().digest().unwrap();
        let campaign = campaign(digest);
        let campaign_digest = campaign.digest().unwrap();
        let first = &campaign.trials[0];
        let result = CheckpointPowerLossTrialResult {
            schema: CHECKPOINT_POWER_LOSS_RESULT_SCHEMA.to_owned(),
            campaign_id: campaign.campaign_id,
            campaign_digest,
            trial_id: first.trial_id,
            storage_profile_digest: first.storage_profile_digest,
            evidence_class: first.evidence_class,
            durability_boundary: first.durability_boundary,
            workload_digest: first.workload_digest,
            pre_power_loss_digest: first.expected_pre_power_loss_digest,
            recovered_state_digest: [0x81; 32],
            power_event_evidence_digest: [0x82; 32],
            outcome: CheckpointPowerLossRecoveryOutcome::CleanRecovery,
            filesystem_consistency_verified: false,
            application_consistency_verified: true,
            completed_at_unix_seconds: 1_900_000_200,
        };
        assert!(result.validate_against(&campaign).is_err());
    }
}
