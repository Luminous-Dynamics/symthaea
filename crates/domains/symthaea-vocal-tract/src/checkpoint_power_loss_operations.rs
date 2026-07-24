// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authenticated operator workflow for sudden-power-loss campaigns.
//!
//! Storage profiles and recovered outcomes answer what was tested and what was
//! observed. This module answers a different set of questions: who was
//! authorized to execute a preregistered trial, which lab and harness were
//! assigned, whether the physical sequence advanced monotonically, and whether
//! an interrupted operator process can resume without inventing state.

use std::collections::{HashMap, HashSet};

#[cfg(unix)]
use std::fs::{self, File, OpenOptions};
#[cfg(unix)]
use std::io::{Read, Write};
#[cfg(unix)]
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use crate::{
    CheckpointPowerLossCampaignPlan, CheckpointPowerLossCampaignEvidence,
    CheckpointPowerLossEvidenceClass, CheckpointPowerLossTrialResult,
    CheckpointStorageEvidenceError,
};

pub const CHECKPOINT_POWER_LOSS_LAB_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-lab.v1";
pub const CHECKPOINT_POWER_LOSS_OPERATIONS_PLAN_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-operations-plan.v1";
pub const CHECKPOINT_POWER_LOSS_OPERATIONS_AUTHORITY_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-operations-authority.v1";
pub const CHECKPOINT_POWER_LOSS_LEASE_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-trial-lease.v1";
pub const CHECKPOINT_POWER_LOSS_JOURNAL_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-execution-journal.v1";
pub const CHECKPOINT_POWER_LOSS_JOURNAL_ENTRY_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-journal-entry.v1";
pub const CHECKPOINT_POWER_LOSS_EXECUTION_RECEIPT_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-execution-receipt.v1";
pub const CHECKPOINT_POWER_LOSS_OPERATIONS_EVIDENCE_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-operations-evidence.v1";

pub const MAX_CHECKPOINT_POWER_LOSS_LABS: usize = 64;
pub const MAX_CHECKPOINT_POWER_LOSS_JOURNAL_ENTRIES: usize = 64;
pub const MAX_CHECKPOINT_POWER_LOSS_OPERATIONS_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_CHECKPOINT_POWER_LOSS_ATTEMPTS_PER_TRIAL: u16 = 16;
pub const MAX_CHECKPOINT_POWER_LOSS_LEASE_SECONDS: u64 = 7 * 24 * 60 * 60;

const LAB_DIGEST_DOMAIN: &[u8] = b"symthaea-power-loss-lab-digest-v1\0";
const OPERATIONS_PLAN_DIGEST_DOMAIN: &[u8] =
    b"symthaea-power-loss-operations-plan-digest-v1\0";
const OPERATIONS_AUTH_DOMAIN: &[u8] = b"symthaea-power-loss-operations-auth-v1\0";
const LEASE_DIGEST_DOMAIN: &[u8] = b"symthaea-power-loss-lease-digest-v1\0";
const JOURNAL_ENTRY_DIGEST_DOMAIN: &[u8] =
    b"symthaea-power-loss-journal-entry-digest-v1\0";
const JOURNAL_DIGEST_DOMAIN: &[u8] = b"symthaea-power-loss-journal-digest-v1\0";
const JOURNAL_AUTH_DOMAIN: &[u8] = b"symthaea-power-loss-journal-auth-v1\0";
const RESULT_DIGEST_DOMAIN: &[u8] = b"symthaea-power-loss-result-digest-v1\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"symthaea-power-loss-execution-receipt-digest-v1\0";
const OPERATIONS_EVIDENCE_AUTH_DOMAIN: &[u8] = b"symthaea-power-loss-operations-evidence-auth-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointPowerLossLabId(pub [u8; 16]);

impl CheckpointPowerLossLabId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointPowerLossOperationsError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointPowerLossOperationsError::InvalidLab);
        }
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointPowerLossOperationsKeyId(pub [u8; 16]);

impl CheckpointPowerLossOperationsKeyId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointPowerLossOperationsError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointPowerLossOperationsError::InvalidKey);
        }
        Ok(Self(bytes))
    }
}

pub struct CheckpointPowerLossOperationsKey {
    id: CheckpointPowerLossOperationsKeyId,
    bytes: [u8; 32],
}

impl CheckpointPowerLossOperationsKey {
    pub fn new(
        id: CheckpointPowerLossOperationsKeyId,
        bytes: [u8; 32],
    ) -> Result<Self, CheckpointPowerLossOperationsError> {
        if bytes == [0u8; 32] {
            return Err(CheckpointPowerLossOperationsError::InvalidKey);
        }
        Ok(Self { id, bytes })
    }

    pub fn id(&self) -> CheckpointPowerLossOperationsKeyId {
        self.id
    }
}

impl Drop for CheckpointPowerLossOperationsKey {
    fn drop(&mut self) {
        self.bytes.zeroize();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossLabManifest {
    pub schema: String,
    pub lab_id: CheckpointPowerLossLabId,
    pub organization_binding: [u8; 32],
    pub operator_group_binding: [u8; 32],
    pub test_harness_binding: [u8; 32],
    pub power_controller_binding: [u8; 32],
    pub facility_binding: [u8; 32],
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointPowerLossLabManifest {
    pub fn validate(&self) -> Result<(), CheckpointPowerLossOperationsError> {
        if self.schema != CHECKPOINT_POWER_LOSS_LAB_SCHEMA
            || self.lab_id.0 == [0u8; 16]
            || self.organization_binding == [0u8; 32]
            || self.operator_group_binding == [0u8; 32]
            || self.test_harness_binding == [0u8; 32]
            || self.power_controller_binding == [0u8; 32]
            || self.facility_binding == [0u8; 32]
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointPowerLossOperationsError::InvalidLab);
        }
        Ok(())
    }

    pub fn valid_at(&self, unix_seconds: u64) -> bool {
        self.valid_from_unix_seconds <= unix_seconds
            && unix_seconds <= self.valid_until_unix_seconds
    }

    pub fn digest(&self) -> Result<[u8; 32], CheckpointPowerLossOperationsError> {
        self.validate()?;
        digest_serialized(LAB_DIGEST_DOMAIN, self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossOperationsPlan {
    pub schema: String,
    pub campaign_id: [u8; 16],
    pub campaign_digest: [u8; 32],
    pub operations_authority_key_id: CheckpointPowerLossOperationsKeyId,
    pub lab_manifests: Vec<CheckpointPowerLossLabManifest>,
    pub maximum_lease_seconds: u64,
    pub maximum_attempts_per_trial: u16,
    pub require_physical_event_before_recovery: bool,
    pub require_complete_journal: bool,
}

impl CheckpointPowerLossOperationsPlan {
    pub fn validate_against(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
    ) -> Result<(), CheckpointPowerLossOperationsError> {
        campaign.validate().map_err(CheckpointPowerLossOperationsError::StorageEvidence)?;
        if self.schema != CHECKPOINT_POWER_LOSS_OPERATIONS_PLAN_SCHEMA
            || self.campaign_id != campaign.campaign_id
            || self.campaign_digest
                != campaign
                    .digest()
                    .map_err(CheckpointPowerLossOperationsError::StorageEvidence)?
            || self.operations_authority_key_id.0 == [0u8; 16]
            || self.lab_manifests.is_empty()
            || self.lab_manifests.len() > MAX_CHECKPOINT_POWER_LOSS_LABS
            || self.maximum_lease_seconds == 0
            || self.maximum_lease_seconds > MAX_CHECKPOINT_POWER_LOSS_LEASE_SECONDS
            || self.maximum_attempts_per_trial == 0
            || self.maximum_attempts_per_trial > MAX_CHECKPOINT_POWER_LOSS_ATTEMPTS_PER_TRIAL
        {
            return Err(CheckpointPowerLossOperationsError::InvalidOperationsPlan);
        }
        let mut labs = HashSet::with_capacity(self.lab_manifests.len());
        for lab in &self.lab_manifests {
            lab.validate()?;
            if !labs.insert(lab.lab_id) {
                return Err(CheckpointPowerLossOperationsError::DuplicateLab);
            }
            if lab.test_harness_binding != campaign.test_harness_digest
                || lab.power_controller_binding != campaign.power_controller_binding
            {
                return Err(CheckpointPowerLossOperationsError::CampaignBindingMismatch);
            }
        }
        Ok(())
    }

    pub fn lab(
        &self,
        lab_id: CheckpointPowerLossLabId,
    ) -> Option<&CheckpointPowerLossLabManifest> {
        self.lab_manifests.iter().find(|lab| lab.lab_id == lab_id)
    }

    pub fn digest(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
    ) -> Result<[u8; 32], CheckpointPowerLossOperationsError> {
        self.validate_against(campaign)?;
        digest_serialized(OPERATIONS_PLAN_DIGEST_DOMAIN, self)
    }
}

pub struct CheckpointPowerLossOperationsAuthority {
    key: CheckpointPowerLossOperationsKey,
}

impl CheckpointPowerLossOperationsAuthority {
    pub fn new(key: CheckpointPowerLossOperationsKey) -> Self {
        Self { key }
    }

    pub fn key_id(&self) -> CheckpointPowerLossOperationsKeyId {
        self.key.id()
    }

    fn authenticate(&self, domain: &[u8], body: &[u8]) -> [u8; 32] {
        let mut input = Vec::with_capacity(domain.len() + body.len());
        input.extend_from_slice(domain);
        input.extend_from_slice(body);
        *blake3::keyed_hash(&self.key.bytes, &input).as_bytes()
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossTrialLease {
    pub schema: String,
    pub campaign_id: [u8; 16],
    pub campaign_digest: [u8; 32],
    pub operations_plan_digest: [u8; 32],
    pub operations_authority_key_id: CheckpointPowerLossOperationsKeyId,
    pub lease_id: [u8; 16],
    pub trial_id: [u8; 16],
    pub storage_profile_digest: [u8; 32],
    pub lab_id: CheckpointPowerLossLabId,
    pub attempt: u16,
    pub issued_at_unix_seconds: u64,
    pub expires_at_unix_seconds: u64,
}

impl CheckpointPowerLossTrialLease {
    pub fn validate_against(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
    ) -> Result<(), CheckpointPowerLossOperationsError> {
        operations.validate_against(campaign)?;
        let trial = campaign
            .trials
            .iter()
            .find(|trial| trial.trial_id == self.trial_id)
            .ok_or(CheckpointPowerLossOperationsError::UnknownTrial)?;
        let lab = operations
            .lab(self.lab_id)
            .ok_or(CheckpointPowerLossOperationsError::InvalidLab)?;
        if self.schema != CHECKPOINT_POWER_LOSS_LEASE_SCHEMA
            || self.campaign_id != campaign.campaign_id
            || self.campaign_digest
                != campaign
                    .digest()
                    .map_err(CheckpointPowerLossOperationsError::StorageEvidence)?
            || self.operations_plan_digest != operations.digest(campaign)?
            || self.operations_authority_key_id != operations.operations_authority_key_id
            || self.lease_id == [0u8; 16]
            || self.storage_profile_digest != trial.storage_profile_digest
            || self.attempt == 0
            || self.attempt > operations.maximum_attempts_per_trial
            || self.issued_at_unix_seconds == 0
            || self.expires_at_unix_seconds <= self.issued_at_unix_seconds
            || self
                .expires_at_unix_seconds
                .saturating_sub(self.issued_at_unix_seconds)
                > operations.maximum_lease_seconds
            || !lab.valid_at(self.issued_at_unix_seconds)
            || !lab.valid_at(self.expires_at_unix_seconds)
        {
            return Err(CheckpointPowerLossOperationsError::InvalidLease);
        }
        Ok(())
    }

    pub fn validate_at(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        now_unix_seconds: u64,
    ) -> Result<(), CheckpointPowerLossOperationsError> {
        self.validate_against(campaign, operations)?;
        if now_unix_seconds < self.issued_at_unix_seconds {
            return Err(CheckpointPowerLossOperationsError::LeaseNotYetValid);
        }
        if now_unix_seconds > self.expires_at_unix_seconds {
            return Err(CheckpointPowerLossOperationsError::LeaseExpired);
        }
        Ok(())
    }

    pub fn digest(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
    ) -> Result<[u8; 32], CheckpointPowerLossOperationsError> {
        self.validate_against(campaign, operations)?;
        digest_serialized(LEASE_DIGEST_DOMAIN, self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointPowerLossLeaseWire {
    schema: String,
    key_id: CheckpointPowerLossOperationsKeyId,
    body: Vec<u8>,
    authentication_tag: [u8; 32],
}

impl CheckpointPowerLossOperationsAuthority {
    #[allow(clippy::too_many_arguments)]
    pub fn issue_lease(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        lab_id: CheckpointPowerLossLabId,
        trial_id: [u8; 16],
        lease_id: [u8; 16],
        attempt: u16,
        issued_at_unix_seconds: u64,
        expires_at_unix_seconds: u64,
    ) -> Result<Vec<u8>, CheckpointPowerLossOperationsError> {
        if self.key_id() != operations.operations_authority_key_id {
            return Err(CheckpointPowerLossOperationsError::AuthenticationFailed);
        }
        let trial = campaign
            .trials
            .iter()
            .find(|trial| trial.trial_id == trial_id)
            .ok_or(CheckpointPowerLossOperationsError::UnknownTrial)?;
        let lease = CheckpointPowerLossTrialLease {
            schema: CHECKPOINT_POWER_LOSS_LEASE_SCHEMA.to_owned(),
            campaign_id: campaign.campaign_id,
            campaign_digest: campaign
                .digest()
                .map_err(CheckpointPowerLossOperationsError::StorageEvidence)?,
            operations_plan_digest: operations.digest(campaign)?,
            operations_authority_key_id: self.key_id(),
            lease_id,
            trial_id,
            storage_profile_digest: trial.storage_profile_digest,
            lab_id,
            attempt,
            issued_at_unix_seconds,
            expires_at_unix_seconds,
        };
        lease.validate_against(campaign, operations)?;
        let body = bounded_encode(&lease)?;
        let wire = CheckpointPowerLossLeaseWire {
            schema: CHECKPOINT_POWER_LOSS_OPERATIONS_AUTHORITY_SCHEMA.to_owned(),
            key_id: self.key_id(),
            authentication_tag: self.authenticate(OPERATIONS_AUTH_DOMAIN, &body),
            body,
        };
        bounded_encode(&wire)
    }

    pub fn open_lease(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        encoded: &[u8],
    ) -> Result<CheckpointPowerLossTrialLease, CheckpointPowerLossOperationsError> {
        if self.key_id() != operations.operations_authority_key_id {
            return Err(CheckpointPowerLossOperationsError::AuthenticationFailed);
        }
        let wire: CheckpointPowerLossLeaseWire = bounded_decode(encoded)?;
        if wire.schema != CHECKPOINT_POWER_LOSS_OPERATIONS_AUTHORITY_SCHEMA
            || wire.key_id != self.key_id()
            || !constant_time_equal(
                &wire.authentication_tag,
                &self.authenticate(OPERATIONS_AUTH_DOMAIN, &wire.body),
            )
        {
            return Err(CheckpointPowerLossOperationsError::AuthenticationFailed);
        }
        let lease: CheckpointPowerLossTrialLease = bounded_decode(&wire.body)?;
        lease.validate_against(campaign, operations)?;
        Ok(lease)
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointPowerLossExecutionState {
    Claimed,
    Prepared,
    Armed,
    PowerEventObserved,
    RecoveryStarted,
    RecoveryClassified,
    EvidenceSealed,
    Completed,
    Aborted,
    Quarantined,
}

impl CheckpointPowerLossExecutionState {
    pub fn terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Aborted | Self::Quarantined)
    }

    fn permits(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Claimed, Self::Prepared | Self::Aborted)
                | (Self::Prepared, Self::Armed | Self::Aborted)
                | (Self::Armed, Self::PowerEventObserved | Self::Aborted)
                | (Self::PowerEventObserved, Self::RecoveryStarted | Self::Quarantined)
                | (Self::RecoveryStarted, Self::RecoveryClassified | Self::Quarantined)
                | (Self::RecoveryClassified, Self::EvidenceSealed | Self::Quarantined)
                | (Self::EvidenceSealed, Self::Completed | Self::Quarantined)
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossJournalEntry {
    pub schema: String,
    pub campaign_id: [u8; 16],
    pub trial_id: [u8; 16],
    pub lease_digest: [u8; 32],
    pub sequence: u32,
    pub state: CheckpointPowerLossExecutionState,
    pub previous_entry_digest: [u8; 32],
    pub event_evidence_digest: [u8; 32],
    pub operator_session_binding: [u8; 32],
    pub observed_at_unix_seconds: u64,
}

impl CheckpointPowerLossJournalEntry {
    pub fn digest(&self) -> Result<[u8; 32], CheckpointPowerLossOperationsError> {
        if self.schema != CHECKPOINT_POWER_LOSS_JOURNAL_ENTRY_SCHEMA
            || self.campaign_id == [0u8; 16]
            || self.trial_id == [0u8; 16]
            || self.lease_digest == [0u8; 32]
            || self.event_evidence_digest == [0u8; 32]
            || self.operator_session_binding == [0u8; 32]
            || self.observed_at_unix_seconds == 0
        {
            return Err(CheckpointPowerLossOperationsError::InvalidJournal);
        }
        digest_serialized(JOURNAL_ENTRY_DIGEST_DOMAIN, self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossExecutionJournal {
    pub schema: String,
    pub campaign_id: [u8; 16],
    pub campaign_digest: [u8; 32],
    pub operations_plan_digest: [u8; 32],
    pub trial_id: [u8; 16],
    pub lease_digest: [u8; 32],
    pub entries: Vec<CheckpointPowerLossJournalEntry>,
}

impl CheckpointPowerLossExecutionJournal {
    pub fn new(
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        lease: &CheckpointPowerLossTrialLease,
        operator_session_binding: [u8; 32],
        event_evidence_digest: [u8; 32],
        observed_at_unix_seconds: u64,
    ) -> Result<Self, CheckpointPowerLossOperationsError> {
        lease.validate_at(campaign, operations, observed_at_unix_seconds)?;
        let lease_digest = lease.digest(campaign, operations)?;
        let first = CheckpointPowerLossJournalEntry {
            schema: CHECKPOINT_POWER_LOSS_JOURNAL_ENTRY_SCHEMA.to_owned(),
            campaign_id: campaign.campaign_id,
            trial_id: lease.trial_id,
            lease_digest,
            sequence: 0,
            state: CheckpointPowerLossExecutionState::Claimed,
            previous_entry_digest: [0u8; 32],
            event_evidence_digest,
            operator_session_binding,
            observed_at_unix_seconds,
        };
        first.digest()?;
        let journal = Self {
            schema: CHECKPOINT_POWER_LOSS_JOURNAL_SCHEMA.to_owned(),
            campaign_id: campaign.campaign_id,
            campaign_digest: campaign
                .digest()
                .map_err(CheckpointPowerLossOperationsError::StorageEvidence)?,
            operations_plan_digest: operations.digest(campaign)?,
            trial_id: lease.trial_id,
            lease_digest,
            entries: vec![first],
        };
        journal.validate_against(campaign, operations, lease)?;
        Ok(journal)
    }

    pub fn append(
        &mut self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        lease: &CheckpointPowerLossTrialLease,
        next_state: CheckpointPowerLossExecutionState,
        event_evidence_digest: [u8; 32],
        operator_session_binding: [u8; 32],
        observed_at_unix_seconds: u64,
    ) -> Result<[u8; 32], CheckpointPowerLossOperationsError> {
        self.validate_against(campaign, operations, lease)?;
        if observed_at_unix_seconds < lease.issued_at_unix_seconds {
            return Err(CheckpointPowerLossOperationsError::LeaseNotYetValid);
        }
        let power_event_already_observed = self.entries.iter().any(|entry| {
            entry.state == CheckpointPowerLossExecutionState::PowerEventObserved
        });
        if observed_at_unix_seconds > lease.expires_at_unix_seconds
            && !power_event_already_observed
        {
            return Err(CheckpointPowerLossOperationsError::LeaseExpired);
        }
        if self.entries.len() >= MAX_CHECKPOINT_POWER_LOSS_JOURNAL_ENTRIES {
            return Err(CheckpointPowerLossOperationsError::TooLarge);
        }
        let previous = self
            .entries
            .last()
            .ok_or(CheckpointPowerLossOperationsError::InvalidJournal)?;
        if previous.state.terminal()
            || !previous.state.permits(next_state)
            || observed_at_unix_seconds < previous.observed_at_unix_seconds
        {
            return Err(CheckpointPowerLossOperationsError::InvalidTransition);
        }
        let entry = CheckpointPowerLossJournalEntry {
            schema: CHECKPOINT_POWER_LOSS_JOURNAL_ENTRY_SCHEMA.to_owned(),
            campaign_id: self.campaign_id,
            trial_id: self.trial_id,
            lease_digest: self.lease_digest,
            sequence: previous
                .sequence
                .checked_add(1)
                .ok_or(CheckpointPowerLossOperationsError::InvalidJournal)?,
            state: next_state,
            previous_entry_digest: previous.digest()?,
            event_evidence_digest,
            operator_session_binding,
            observed_at_unix_seconds,
        };
        let digest = entry.digest()?;
        self.entries.push(entry);
        self.validate_against(campaign, operations, lease)?;
        Ok(digest)
    }

    pub fn validate_against(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        lease: &CheckpointPowerLossTrialLease,
    ) -> Result<(), CheckpointPowerLossOperationsError> {
        lease.validate_against(campaign, operations)?;
        if self.schema != CHECKPOINT_POWER_LOSS_JOURNAL_SCHEMA
            || self.campaign_id != campaign.campaign_id
            || self.campaign_digest
                != campaign
                    .digest()
                    .map_err(CheckpointPowerLossOperationsError::StorageEvidence)?
            || self.operations_plan_digest != operations.digest(campaign)?
            || self.trial_id != lease.trial_id
            || self.lease_digest != lease.digest(campaign, operations)?
            || self.entries.is_empty()
            || self.entries.len() > MAX_CHECKPOINT_POWER_LOSS_JOURNAL_ENTRIES
        {
            return Err(CheckpointPowerLossOperationsError::InvalidJournal);
        }
        let trial = campaign
            .trials
            .iter()
            .find(|trial| trial.trial_id == self.trial_id)
            .ok_or(CheckpointPowerLossOperationsError::UnknownTrial)?;
        let mut previous: Option<&CheckpointPowerLossJournalEntry> = None;
        let mut saw_power_event = false;
        for (index, entry) in self.entries.iter().enumerate() {
            entry.digest()?;
            if entry.observed_at_unix_seconds < lease.issued_at_unix_seconds
                || (!saw_power_event
                    && entry.observed_at_unix_seconds > lease.expires_at_unix_seconds)
            {
                return Err(CheckpointPowerLossOperationsError::LeaseExpired);
            }
            if entry.campaign_id != self.campaign_id
                || entry.trial_id != self.trial_id
                || entry.lease_digest != self.lease_digest
                || entry.sequence as usize != index
            {
                return Err(CheckpointPowerLossOperationsError::InvalidJournal);
            }
            if let Some(prior) = previous {
                if entry.previous_entry_digest != prior.digest()?
                    || !prior.state.permits(entry.state)
                    || entry.observed_at_unix_seconds < prior.observed_at_unix_seconds
                {
                    return Err(CheckpointPowerLossOperationsError::InvalidTransition);
                }
            } else if entry.state != CheckpointPowerLossExecutionState::Claimed
                || entry.previous_entry_digest != [0u8; 32]
            {
                return Err(CheckpointPowerLossOperationsError::InvalidJournal);
            }
            if entry.state == CheckpointPowerLossExecutionState::PowerEventObserved {
                saw_power_event = true;
            }
            if entry.state == CheckpointPowerLossExecutionState::RecoveryStarted
                && operations.require_physical_event_before_recovery
                && trial.evidence_class == CheckpointPowerLossEvidenceClass::PhysicalDevicePowerCut
                && !saw_power_event
            {
                return Err(CheckpointPowerLossOperationsError::InvalidTransition);
            }
            previous = Some(entry);
        }
        Ok(())
    }

    pub fn digest(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        lease: &CheckpointPowerLossTrialLease,
    ) -> Result<[u8; 32], CheckpointPowerLossOperationsError> {
        self.validate_against(campaign, operations, lease)?;
        digest_serialized(JOURNAL_DIGEST_DOMAIN, self)
    }

    pub fn current_state(&self) -> Option<CheckpointPowerLossExecutionState> {
        self.entries.last().map(|entry| entry.state)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointPowerLossResumeDecision {
    ContinueWith(CheckpointPowerLossExecutionState),
    AwaitPowerEvent,
    SealEvidence,
    Completed,
    StartNewAttempt,
    Quarantined,
    LeaseExpired,
}

pub fn checkpoint_power_loss_resume_decision(
    campaign: &CheckpointPowerLossCampaignPlan,
    operations: &CheckpointPowerLossOperationsPlan,
    lease: &CheckpointPowerLossTrialLease,
    journal: &CheckpointPowerLossExecutionJournal,
    now_unix_seconds: u64,
) -> Result<CheckpointPowerLossResumeDecision, CheckpointPowerLossOperationsError> {
    journal.validate_against(campaign, operations, lease)?;
    let power_event_observed = journal.entries.iter().any(|entry| {
        entry.state == CheckpointPowerLossExecutionState::PowerEventObserved
    });
    if now_unix_seconds > lease.expires_at_unix_seconds
        && !power_event_observed
        && !journal.current_state().is_some_and(CheckpointPowerLossExecutionState::terminal)
    {
        return Ok(CheckpointPowerLossResumeDecision::LeaseExpired);
    }
    Ok(match journal
        .current_state()
        .ok_or(CheckpointPowerLossOperationsError::InvalidJournal)?
    {
        CheckpointPowerLossExecutionState::Claimed =>
            CheckpointPowerLossResumeDecision::ContinueWith(CheckpointPowerLossExecutionState::Prepared),
        CheckpointPowerLossExecutionState::Prepared =>
            CheckpointPowerLossResumeDecision::ContinueWith(CheckpointPowerLossExecutionState::Armed),
        CheckpointPowerLossExecutionState::Armed => CheckpointPowerLossResumeDecision::AwaitPowerEvent,
        CheckpointPowerLossExecutionState::PowerEventObserved =>
            CheckpointPowerLossResumeDecision::ContinueWith(CheckpointPowerLossExecutionState::RecoveryStarted),
        CheckpointPowerLossExecutionState::RecoveryStarted =>
            CheckpointPowerLossResumeDecision::ContinueWith(CheckpointPowerLossExecutionState::RecoveryClassified),
        CheckpointPowerLossExecutionState::RecoveryClassified => CheckpointPowerLossResumeDecision::SealEvidence,
        CheckpointPowerLossExecutionState::EvidenceSealed =>
            CheckpointPowerLossResumeDecision::ContinueWith(CheckpointPowerLossExecutionState::Completed),
        CheckpointPowerLossExecutionState::Completed => CheckpointPowerLossResumeDecision::Completed,
        CheckpointPowerLossExecutionState::Aborted => CheckpointPowerLossResumeDecision::StartNewAttempt,
        CheckpointPowerLossExecutionState::Quarantined => CheckpointPowerLossResumeDecision::Quarantined,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointPowerLossJournalWire {
    schema: String,
    key_id: CheckpointPowerLossOperationsKeyId,
    body: Vec<u8>,
    authentication_tag: [u8; 32],
}

impl CheckpointPowerLossOperationsAuthority {
    pub fn seal_journal(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        lease: &CheckpointPowerLossTrialLease,
        journal: &CheckpointPowerLossExecutionJournal,
    ) -> Result<Vec<u8>, CheckpointPowerLossOperationsError> {
        if self.key_id() != operations.operations_authority_key_id {
            return Err(CheckpointPowerLossOperationsError::AuthenticationFailed);
        }
        journal.validate_against(campaign, operations, lease)?;
        let body = bounded_encode(journal)?;
        bounded_encode(&CheckpointPowerLossJournalWire {
            schema: CHECKPOINT_POWER_LOSS_JOURNAL_SCHEMA.to_owned(),
            key_id: self.key_id(),
            authentication_tag: self.authenticate(JOURNAL_AUTH_DOMAIN, &body),
            body,
        })
    }

    pub fn open_journal(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        lease: &CheckpointPowerLossTrialLease,
        encoded: &[u8],
    ) -> Result<CheckpointPowerLossExecutionJournal, CheckpointPowerLossOperationsError> {
        let wire: CheckpointPowerLossJournalWire = bounded_decode(encoded)?;
        if self.key_id() != operations.operations_authority_key_id
            || wire.schema != CHECKPOINT_POWER_LOSS_JOURNAL_SCHEMA
            || wire.key_id != self.key_id()
            || !constant_time_equal(
                &wire.authentication_tag,
                &self.authenticate(JOURNAL_AUTH_DOMAIN, &wire.body),
            )
        {
            return Err(CheckpointPowerLossOperationsError::AuthenticationFailed);
        }
        let journal: CheckpointPowerLossExecutionJournal = bounded_decode(&wire.body)?;
        journal.validate_against(campaign, operations, lease)?;
        Ok(journal)
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossExecutionReceipt {
    pub schema: String,
    pub campaign_id: [u8; 16],
    pub campaign_digest: [u8; 32],
    pub operations_plan_digest: [u8; 32],
    pub operations_authority_key_id: CheckpointPowerLossOperationsKeyId,
    pub trial_id: [u8; 16],
    pub lease_digest: [u8; 32],
    pub journal_digest: [u8; 32],
    pub result_digest: [u8; 32],
    pub sealed_result_evidence_digest: [u8; 32],
    pub lab_id: CheckpointPowerLossLabId,
    pub attempt: u16,
    pub finalized_at_unix_seconds: u64,
}

impl CheckpointPowerLossExecutionReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        lease: &CheckpointPowerLossTrialLease,
        journal: &CheckpointPowerLossExecutionJournal,
        result: &CheckpointPowerLossTrialResult,
        sealed_result_evidence_digest: [u8; 32],
        finalized_at_unix_seconds: u64,
    ) -> Result<Self, CheckpointPowerLossOperationsError> {
        let receipt = Self {
            schema: CHECKPOINT_POWER_LOSS_EXECUTION_RECEIPT_SCHEMA.to_owned(),
            campaign_id: campaign.campaign_id,
            campaign_digest: campaign
                .digest()
                .map_err(CheckpointPowerLossOperationsError::StorageEvidence)?,
            operations_plan_digest: operations.digest(campaign)?,
            operations_authority_key_id: operations.operations_authority_key_id,
            trial_id: lease.trial_id,
            lease_digest: lease.digest(campaign, operations)?,
            journal_digest: journal.digest(campaign, operations, lease)?,
            result_digest: checkpoint_power_loss_trial_result_digest(result)?,
            sealed_result_evidence_digest,
            lab_id: lease.lab_id,
            attempt: lease.attempt,
            finalized_at_unix_seconds,
        };
        receipt.validate_against(campaign, operations, lease, journal, result)?;
        Ok(receipt)
    }

    pub fn validate_against(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        lease: &CheckpointPowerLossTrialLease,
        journal: &CheckpointPowerLossExecutionJournal,
        result: &CheckpointPowerLossTrialResult,
    ) -> Result<(), CheckpointPowerLossOperationsError> {
        result
            .validate_against(campaign)
            .map_err(CheckpointPowerLossOperationsError::StorageEvidence)?;
        journal.validate_against(campaign, operations, lease)?;
        let power_event = journal.entries.iter().find(|entry| {
            entry.state == CheckpointPowerLossExecutionState::PowerEventObserved
        });
        let recovery_classified = journal.entries.iter().find(|entry| {
            entry.state == CheckpointPowerLossExecutionState::RecoveryClassified
        });
        let evidence_sealed = journal.entries.iter().find(|entry| {
            entry.state == CheckpointPowerLossExecutionState::EvidenceSealed
        });
        let final_entry = journal.entries.last();
        if self.schema != CHECKPOINT_POWER_LOSS_EXECUTION_RECEIPT_SCHEMA
            || self.campaign_id != campaign.campaign_id
            || self.campaign_digest
                != campaign
                    .digest()
                    .map_err(CheckpointPowerLossOperationsError::StorageEvidence)?
            || self.operations_plan_digest != operations.digest(campaign)?
            || self.operations_authority_key_id != operations.operations_authority_key_id
            || self.trial_id != lease.trial_id
            || self.trial_id != result.trial_id
            || self.lease_digest != lease.digest(campaign, operations)?
            || self.journal_digest != journal.digest(campaign, operations, lease)?
            || self.result_digest != checkpoint_power_loss_trial_result_digest(result)?
            || self.sealed_result_evidence_digest == [0u8; 32]
            || self.lab_id != lease.lab_id
            || self.attempt != lease.attempt
            || self.finalized_at_unix_seconds == 0
            || self.finalized_at_unix_seconds < result.completed_at_unix_seconds
            || final_entry
                .is_none_or(|entry| self.finalized_at_unix_seconds < entry.observed_at_unix_seconds)
            || journal.current_state() != Some(CheckpointPowerLossExecutionState::Completed)
            || power_event.map(|entry| entry.event_evidence_digest)
                != Some(result.power_event_evidence_digest)
            || recovery_classified.map(|entry| entry.event_evidence_digest)
                != Some(result.recovered_state_digest)
            || evidence_sealed
                .map(|entry| entry.event_evidence_digest)
                != Some(self.sealed_result_evidence_digest)
        {
            return Err(CheckpointPowerLossOperationsError::InvalidReceipt);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<[u8; 32], CheckpointPowerLossOperationsError> {
        if self.schema != CHECKPOINT_POWER_LOSS_EXECUTION_RECEIPT_SCHEMA
            || self.trial_id == [0u8; 16]
            || self.lease_digest == [0u8; 32]
            || self.journal_digest == [0u8; 32]
            || self.result_digest == [0u8; 32]
            || self.sealed_result_evidence_digest == [0u8; 32]
        {
            return Err(CheckpointPowerLossOperationsError::InvalidReceipt);
        }
        digest_serialized(RECEIPT_DIGEST_DOMAIN, self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossExecutionProof {
    pub lease: CheckpointPowerLossTrialLease,
    pub journal: CheckpointPowerLossExecutionJournal,
    pub receipt: CheckpointPowerLossExecutionReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossOperationsEvidence {
    pub schema: String,
    pub campaign_id: [u8; 16],
    pub campaign_digest: [u8; 32],
    pub operations_plan_digest: [u8; 32],
    pub operations_authority_key_id: CheckpointPowerLossOperationsKeyId,
    pub proofs: Vec<CheckpointPowerLossExecutionProof>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CheckpointPowerLossOperationsSummary {
    pub planned_trials: usize,
    pub completed_proofs: usize,
    pub unique_labs: usize,
    pub resumed_trials: usize,
    pub quarantined_trials: usize,
}

impl CheckpointPowerLossOperationsEvidence {
    pub fn validate_against(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        result_evidence: &CheckpointPowerLossCampaignEvidence,
    ) -> Result<(), CheckpointPowerLossOperationsError> {
        operations.validate_against(campaign)?;
        result_evidence
            .validate_against(campaign)
            .map_err(CheckpointPowerLossOperationsError::StorageEvidence)?;
        if self.schema != CHECKPOINT_POWER_LOSS_OPERATIONS_EVIDENCE_SCHEMA
            || self.campaign_id != campaign.campaign_id
            || self.campaign_digest
                != campaign
                    .digest()
                    .map_err(CheckpointPowerLossOperationsError::StorageEvidence)?
            || self.operations_plan_digest != operations.digest(campaign)?
            || self.operations_authority_key_id != operations.operations_authority_key_id
            || self.proofs.len() > campaign.trials.len()
        {
            return Err(CheckpointPowerLossOperationsError::InvalidReceipt);
        }
        let results: HashMap<[u8; 16], &CheckpointPowerLossTrialResult> = result_evidence
            .results
            .iter()
            .map(|result| (result.trial_id, result))
            .collect();
        let mut proof_ids = HashSet::with_capacity(self.proofs.len());
        let mut attempts: HashMap<[u8; 16], HashSet<u16>> = HashMap::new();
        for proof in &self.proofs {
            let result = results
                .get(&proof.receipt.trial_id)
                .ok_or(CheckpointPowerLossOperationsError::UnknownTrial)?;
            proof.receipt.validate_against(
                campaign,
                operations,
                &proof.lease,
                &proof.journal,
                result,
            )?;
            if !proof_ids.insert(proof.receipt.trial_id)
                || !attempts
                    .entry(proof.receipt.trial_id)
                    .or_default()
                    .insert(proof.receipt.attempt)
            {
                return Err(CheckpointPowerLossOperationsError::DuplicateTrial);
            }
        }
        if proof_ids.len() != results.len()
            || !results.keys().all(|trial_id| proof_ids.contains(trial_id))
        {
            return Err(CheckpointPowerLossOperationsError::InvalidReceipt);
        }
        Ok(())
    }

    pub fn summary(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        result_evidence: &CheckpointPowerLossCampaignEvidence,
    ) -> Result<CheckpointPowerLossOperationsSummary, CheckpointPowerLossOperationsError> {
        self.validate_against(campaign, operations, result_evidence)?;
        let unique_labs = self
            .proofs
            .iter()
            .map(|proof| proof.receipt.lab_id)
            .collect::<HashSet<_>>()
            .len();
        Ok(CheckpointPowerLossOperationsSummary {
            planned_trials: campaign.trials.len(),
            completed_proofs: self.proofs.len(),
            unique_labs,
            resumed_trials: self
                .proofs
                .iter()
                .filter(|proof| {
                    proof
                        .journal
                        .entries
                        .iter()
                        .map(|entry| entry.operator_session_binding)
                        .collect::<HashSet<_>>()
                        .len()
                        > 1
                })
                .count(),
            quarantined_trials: self
                .proofs
                .iter()
                .filter(|proof| {
                    proof.journal.entries.iter().any(|entry| {
                        entry.state == CheckpointPowerLossExecutionState::Quarantined
                    })
                })
                .count(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointPowerLossOperationsEvidenceWire {
    schema: String,
    key_id: CheckpointPowerLossOperationsKeyId,
    body: Vec<u8>,
    authentication_tag: [u8; 32],
}

impl CheckpointPowerLossOperationsAuthority {
    pub fn seal_operations_evidence(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        result_evidence: &CheckpointPowerLossCampaignEvidence,
        evidence: &CheckpointPowerLossOperationsEvidence,
    ) -> Result<Vec<u8>, CheckpointPowerLossOperationsError> {
        evidence.validate_against(campaign, operations, result_evidence)?;
        if self.key_id() != operations.operations_authority_key_id {
            return Err(CheckpointPowerLossOperationsError::AuthenticationFailed);
        }
        let body = bounded_encode(evidence)?;
        bounded_encode(&CheckpointPowerLossOperationsEvidenceWire {
            schema: CHECKPOINT_POWER_LOSS_OPERATIONS_EVIDENCE_SCHEMA.to_owned(),
            key_id: self.key_id(),
            authentication_tag: self.authenticate(OPERATIONS_EVIDENCE_AUTH_DOMAIN, &body),
            body,
        })
    }

    pub fn open_operations_evidence(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        result_evidence: &CheckpointPowerLossCampaignEvidence,
        encoded: &[u8],
    ) -> Result<CheckpointPowerLossOperationsEvidence, CheckpointPowerLossOperationsError> {
        let wire: CheckpointPowerLossOperationsEvidenceWire = bounded_decode(encoded)?;
        if self.key_id() != operations.operations_authority_key_id
            || wire.schema != CHECKPOINT_POWER_LOSS_OPERATIONS_EVIDENCE_SCHEMA
            || wire.key_id != self.key_id()
            || !constant_time_equal(
                &wire.authentication_tag,
                &self.authenticate(OPERATIONS_EVIDENCE_AUTH_DOMAIN, &wire.body),
            )
        {
            return Err(CheckpointPowerLossOperationsError::AuthenticationFailed);
        }
        let evidence: CheckpointPowerLossOperationsEvidence = bounded_decode(&wire.body)?;
        evidence.validate_against(campaign, operations, result_evidence)?;
        Ok(evidence)
    }
}

pub fn checkpoint_power_loss_trial_result_digest(
    result: &CheckpointPowerLossTrialResult,
) -> Result<[u8; 32], CheckpointPowerLossOperationsError> {
    digest_serialized(RESULT_DIGEST_DOMAIN, result)
}


#[cfg(unix)]
pub struct CheckpointPowerLossJournalStore {
    root: PathBuf,
    authority: CheckpointPowerLossOperationsAuthority,
    local_lock: Mutex<()>,
    pinned_root: Mutex<Option<Arc<File>>>,
}

#[cfg(unix)]
impl CheckpointPowerLossJournalStore {
    pub fn new(
        root: impl Into<PathBuf>,
        authority: CheckpointPowerLossOperationsAuthority,
    ) -> Self {
        Self {
            root: root.into(),
            authority,
            local_lock: Mutex::new(()),
            pinned_root: Mutex::new(None),
        }
    }

    pub fn authority(&self) -> &CheckpointPowerLossOperationsAuthority {
        &self.authority
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        sealed_lease: &[u8],
        operator_session_binding: [u8; 32],
        event_evidence_digest: [u8; 32],
        observed_at_unix_seconds: u64,
    ) -> Result<CheckpointPowerLossExecutionJournal, CheckpointPowerLossOperationsError> {
        let lease = self
            .authority
            .open_lease(campaign, operations, sealed_lease)?;
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| CheckpointPowerLossOperationsError::UnsafeFilesystemObject)?;
        let lock = self.open_lock_file(&lease)?;
        let _kernel = crate::lock_exclusive(&lock)?;
        let path = self.journal_path(&lease)?;
        match fs::symlink_metadata(&path) {
            Ok(_) => return Err(CheckpointPowerLossOperationsError::ConcurrentModification),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let journal = CheckpointPowerLossExecutionJournal::new(
            campaign,
            operations,
            &lease,
            operator_session_binding,
            event_evidence_digest,
            observed_at_unix_seconds,
        )?;
        self.write_journal_locked(campaign, operations, &lease, &journal)?;
        Ok(journal)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        sealed_lease: &[u8],
        expected_journal_digest: [u8; 32],
        next_state: CheckpointPowerLossExecutionState,
        event_evidence_digest: [u8; 32],
        operator_session_binding: [u8; 32],
        observed_at_unix_seconds: u64,
    ) -> Result<CheckpointPowerLossExecutionJournal, CheckpointPowerLossOperationsError> {
        let lease = self
            .authority
            .open_lease(campaign, operations, sealed_lease)?;
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| CheckpointPowerLossOperationsError::UnsafeFilesystemObject)?;
        let lock = self.open_lock_file(&lease)?;
        let _kernel = crate::lock_exclusive(&lock)?;
        let mut journal = self.read_journal_locked(campaign, operations, &lease)?;
        if journal.digest(campaign, operations, &lease)? != expected_journal_digest {
            return Err(CheckpointPowerLossOperationsError::ConcurrentModification);
        }
        journal.append(
            campaign,
            operations,
            &lease,
            next_state,
            event_evidence_digest,
            operator_session_binding,
            observed_at_unix_seconds,
        )?;
        self.write_journal_locked(campaign, operations, &lease, &journal)?;
        Ok(journal)
    }

    pub fn load(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        sealed_lease: &[u8],
    ) -> Result<CheckpointPowerLossExecutionJournal, CheckpointPowerLossOperationsError> {
        let lease = self
            .authority
            .open_lease(campaign, operations, sealed_lease)?;
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| CheckpointPowerLossOperationsError::UnsafeFilesystemObject)?;
        let lock = self.open_lock_file(&lease)?;
        let _kernel = crate::lock_exclusive(&lock)?;
        self.read_journal_locked(campaign, operations, &lease)
    }

    pub fn resume_decision(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        sealed_lease: &[u8],
        now_unix_seconds: u64,
    ) -> Result<CheckpointPowerLossResumeDecision, CheckpointPowerLossOperationsError> {
        let lease = self
            .authority
            .open_lease(campaign, operations, sealed_lease)?;
        let journal = self.load(campaign, operations, sealed_lease)?;
        checkpoint_power_loss_resume_decision(
            campaign,
            operations,
            &lease,
            &journal,
            now_unix_seconds,
        )
    }

    fn read_journal_locked(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        lease: &CheckpointPowerLossTrialLease,
    ) -> Result<CheckpointPowerLossExecutionJournal, CheckpointPowerLossOperationsError> {
        let path = self.journal_path(lease)?;
        let mut file = open_private_regular_file(&path, false, false)?;
        let metadata = file.metadata()?;
        if metadata.len() == 0
            || metadata.len() as usize > MAX_CHECKPOINT_POWER_LOSS_OPERATIONS_BYTES
        {
            return Err(CheckpointPowerLossOperationsError::TooLarge);
        }
        let mut encoded = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_CHECKPOINT_POWER_LOSS_OPERATIONS_BYTES as u64 + 1)
            .read_to_end(&mut encoded)?;
        if encoded.len() > MAX_CHECKPOINT_POWER_LOSS_OPERATIONS_BYTES {
            return Err(CheckpointPowerLossOperationsError::TooLarge);
        }
        self.authority
            .open_journal(campaign, operations, lease, &encoded)
    }

    fn write_journal_locked(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        lease: &CheckpointPowerLossTrialLease,
        journal: &CheckpointPowerLossExecutionJournal,
    ) -> Result<(), CheckpointPowerLossOperationsError> {
        let encoded = self
            .authority
            .seal_journal(campaign, operations, lease, journal)?;
        let root = self.ensure_root()?;
        let operation_root = self.operation_root_path()?;
        let target = self.journal_path(lease)?;
        let mut nonce = [0u8; 16];
        getrandom::fill(&mut nonce)
            .map_err(|_| CheckpointPowerLossOperationsError::UnsafeFilesystemObject)?;
        let suffix = nonce
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let temp = operation_root.join(format!(
            ".journal-{}-{suffix}.tmp",
            std::process::id(),
        ));
        let result = (|| {
            let mut file = open_private_regular_file(&temp, true, true)?;
            file.write_all(&encoded)?;
            file.sync_all()?;
            fs::rename(&temp, &target)?;
            root.sync_all()?;
            Ok::<(), CheckpointPowerLossOperationsError>(())
        })();
        let _ = fs::remove_file(&temp);
        result
    }

    fn open_lock_file(
        &self,
        lease: &CheckpointPowerLossTrialLease,
    ) -> Result<File, CheckpointPowerLossOperationsError> {
        let path = self.operation_root_path()?.join(format!(
            ".trial-{}-attempt-{}.lock",
            hex_identifier(&lease.trial_id),
            lease.attempt,
        ));
        open_private_regular_file(&path, true, false).map_err(Into::into)
    }

    fn journal_path(
        &self,
        lease: &CheckpointPowerLossTrialLease,
    ) -> Result<PathBuf, CheckpointPowerLossOperationsError> {
        Ok(self.operation_root_path()?.join(format!(
            "trial-{}-attempt-{}.journal",
            hex_identifier(&lease.trial_id),
            lease.attempt,
        )))
    }

    fn ensure_root(&self) -> Result<Arc<File>, CheckpointPowerLossOperationsError> {
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};

        let mut pinned = self
            .pinned_root
            .lock()
            .map_err(|_| CheckpointPowerLossOperationsError::UnsafeFilesystemObject)?;
        if let Some(root) = pinned.as_ref() {
            return Ok(Arc::clone(root));
        }
        fs::create_dir_all(&self.root)?;
        let initial = fs::symlink_metadata(&self.root)?;
        if initial.file_type().is_symlink()
            || !initial.is_dir()
            || initial.uid() != crate::effective_uid()
        {
            return Err(CheckpointPowerLossOperationsError::UnsafeFilesystemObject);
        }
        if initial.permissions().mode() & 0o077 != 0 {
            fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700))?;
        }
        let mut options = OpenOptions::new();
        options.read(true).custom_flags(
            libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        );
        let root = Arc::new(options.open(&self.root)?);
        let metadata = root.metadata()?;
        if !metadata.is_dir()
            || metadata.uid() != crate::effective_uid()
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(CheckpointPowerLossOperationsError::UnsafeFilesystemObject);
        }
        *pinned = Some(Arc::clone(&root));
        Ok(root)
    }

    fn operation_root_path(&self) -> Result<PathBuf, CheckpointPowerLossOperationsError> {
        let root = self.ensure_root()?;
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let path = PathBuf::from(format!("/proc/self/fd/{}", root.as_raw_fd()));
            if !path.is_dir() {
                return Err(CheckpointPowerLossOperationsError::UnsafeFilesystemObject);
            }
            Ok(path)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = root;
            Ok(self.root.clone())
        }
    }
}

#[cfg(unix)]
fn open_private_regular_file(
    path: &Path,
    create: bool,
    create_new: bool,
) -> std::io::Result<File> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create(create)
        .create_new(create_new)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.uid() != crate::effective_uid()
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "unsafe power-loss journal file",
        ));
    }
    Ok(file)
}

fn hex_identifier(bytes: &[u8; 16]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Debug)]
pub enum CheckpointPowerLossOperationsError {
    InvalidKey,
    InvalidLab,
    DuplicateLab,
    InvalidOperationsPlan,
    InvalidLease,
    LeaseExpired,
    LeaseNotYetValid,
    InvalidTransition,
    InvalidJournal,
    InvalidReceipt,
    DuplicateTrial,
    UnknownTrial,
    CampaignBindingMismatch,
    AuthenticationFailed,
    ConcurrentModification,
    UnsafeFilesystemObject,
    Encoding,
    TooLarge,
    StorageEvidence(CheckpointStorageEvidenceError),
    #[cfg(unix)]
    Io(std::io::Error),
}

impl std::fmt::Display for CheckpointPowerLossOperationsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKey => formatter.write_str("invalid power-loss operations key"),
            Self::InvalidLab => formatter.write_str("invalid power-loss lab manifest"),
            Self::DuplicateLab => formatter.write_str("duplicate power-loss lab identifier"),
            Self::InvalidOperationsPlan => formatter.write_str("invalid power-loss operations plan"),
            Self::InvalidLease => formatter.write_str("invalid power-loss trial lease"),
            Self::LeaseExpired => formatter.write_str("power-loss trial lease expired"),
            Self::LeaseNotYetValid => formatter.write_str("power-loss trial lease is not yet valid"),
            Self::InvalidTransition => formatter.write_str("invalid power-loss journal transition"),
            Self::InvalidJournal => formatter.write_str("invalid power-loss execution journal"),
            Self::InvalidReceipt => formatter.write_str("invalid power-loss execution receipt"),
            Self::DuplicateTrial => formatter.write_str("duplicate power-loss trial evidence"),
            Self::UnknownTrial => formatter.write_str("unknown power-loss trial"),
            Self::CampaignBindingMismatch => formatter.write_str("power-loss campaign binding mismatch"),
            Self::AuthenticationFailed => formatter.write_str("power-loss operations authentication failed"),
            Self::ConcurrentModification => formatter.write_str("power-loss journal changed since the caller last observed it"),
            Self::UnsafeFilesystemObject => formatter.write_str("unsafe power-loss journal filesystem object"),
            Self::Encoding => formatter.write_str("power-loss operations encoding failed"),
            Self::TooLarge => formatter.write_str("power-loss operations artifact exceeds its bound"),
            Self::StorageEvidence(error) => write!(formatter, "storage evidence failed: {error}"),
            #[cfg(unix)]
            Self::Io(error) => write!(formatter, "power-loss operations I/O failed: {error}"),
        }
    }
}

impl std::error::Error for CheckpointPowerLossOperationsError {}

#[cfg(unix)]
impl From<std::io::Error> for CheckpointPowerLossOperationsError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

fn digest_serialized<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<[u8; 32], CheckpointPowerLossOperationsError> {
    let encoded = postcard::to_stdvec(value)
        .map_err(|_| CheckpointPowerLossOperationsError::Encoding)?;
    if encoded.len() > MAX_CHECKPOINT_POWER_LOSS_OPERATIONS_BYTES {
        return Err(CheckpointPowerLossOperationsError::TooLarge);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&encoded);
    Ok(*hasher.finalize().as_bytes())
}


fn bounded_encode<T: Serialize>(value: &T) -> Result<Vec<u8>, CheckpointPowerLossOperationsError> {
    let encoded = postcard::to_stdvec(value)
        .map_err(|_| CheckpointPowerLossOperationsError::Encoding)?;
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_POWER_LOSS_OPERATIONS_BYTES {
        return Err(CheckpointPowerLossOperationsError::TooLarge);
    }
    Ok(encoded)
}

fn bounded_decode<T: for<'de> Deserialize<'de>>(
    encoded: &[u8],
) -> Result<T, CheckpointPowerLossOperationsError> {
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_POWER_LOSS_OPERATIONS_BYTES {
        return Err(CheckpointPowerLossOperationsError::TooLarge);
    }
    postcard::from_bytes(encoded).map_err(|_| CheckpointPowerLossOperationsError::Encoding)
}

fn constant_time_equal(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter()
        .zip(right.iter())
        .fold(0u8, |difference, (a, b)| difference | (a ^ b))
        == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CheckpointDurabilityBoundary, CheckpointPowerLossEvidenceKeyId,
        CheckpointPowerLossTrialPlan, CheckpointStorageProfileAttestationKeyId,
    };

    fn campaign() -> CheckpointPowerLossCampaignPlan {
        CheckpointPowerLossCampaignPlan {
            schema: crate::CHECKPOINT_POWER_LOSS_CAMPAIGN_SCHEMA.to_owned(),
            campaign_id: [1u8; 16],
            storage_profiles: vec![[2u8; 32]],
            storage_profile_authority_key_id:
                CheckpointStorageProfileAttestationKeyId::new([3u8; 16]).unwrap(),
            power_loss_evidence_authority_key_id:
                CheckpointPowerLossEvidenceKeyId::new([4u8; 16]).unwrap(),
            test_harness_digest: [5u8; 32],
            power_controller_binding: [6u8; 32],
            power_controller_calibration_digest: [7u8; 32],
            operator_protocol_digest: [8u8; 32],
            trials: vec![CheckpointPowerLossTrialPlan {
                trial_id: [9u8; 16],
                storage_profile_digest: [2u8; 32],
                evidence_class: CheckpointPowerLossEvidenceClass::PhysicalDevicePowerCut,
                durability_boundary:
                    CheckpointDurabilityBoundary::AfterDataWriteBeforeFileSync,
                workload_digest: [10u8; 32],
                expected_pre_power_loss_digest: [11u8; 32],
            }],
            minimum_physical_trials: 1,
            require_all_durability_boundaries: false,
        }
    }

    fn operations_plan(campaign: &CheckpointPowerLossCampaignPlan) -> CheckpointPowerLossOperationsPlan {
        CheckpointPowerLossOperationsPlan {
            schema: CHECKPOINT_POWER_LOSS_OPERATIONS_PLAN_SCHEMA.to_owned(),
            campaign_id: campaign.campaign_id,
            campaign_digest: campaign.digest().unwrap(),
            operations_authority_key_id:
                CheckpointPowerLossOperationsKeyId::new([12u8; 16]).unwrap(),
            lab_manifests: vec![CheckpointPowerLossLabManifest {
                schema: CHECKPOINT_POWER_LOSS_LAB_SCHEMA.to_owned(),
                lab_id: CheckpointPowerLossLabId::new([13u8; 16]).unwrap(),
                organization_binding: [14u8; 32],
                operator_group_binding: [15u8; 32],
                test_harness_binding: campaign.test_harness_digest,
                power_controller_binding: campaign.power_controller_binding,
                facility_binding: [16u8; 32],
                valid_from_unix_seconds: 100,
                valid_until_unix_seconds: 1_000,
            }],
            maximum_lease_seconds: 300,
            maximum_attempts_per_trial: 2,
            require_physical_event_before_recovery: true,
            require_complete_journal: true,
        }
    }

    #[test]
    fn operations_plan_binds_campaign_harness_and_controller() {
        let campaign = campaign();
        let plan = operations_plan(&campaign);
        plan.validate_against(&campaign).unwrap();
        assert_ne!(plan.digest(&campaign).unwrap(), [0u8; 32]);

        let mut altered = plan.clone();
        altered.lab_manifests[0].test_harness_binding = [99u8; 32];
        assert!(matches!(
            altered.validate_against(&campaign),
            Err(CheckpointPowerLossOperationsError::CampaignBindingMismatch)
        ));
    }

    #[test]
    fn lease_is_bound_to_trial_lab_attempt_and_expiry() {
        let campaign = campaign();
        let plan = operations_plan(&campaign);
        let authority = CheckpointPowerLossOperationsAuthority::new(
            CheckpointPowerLossOperationsKey::new(
                plan.operations_authority_key_id,
                [17u8; 32],
            )
            .unwrap(),
        );
        let sealed = authority
            .issue_lease(
                &campaign,
                &plan,
                plan.lab_manifests[0].lab_id,
                campaign.trials[0].trial_id,
                [18u8; 16],
                1,
                200,
                400,
            )
            .unwrap();
        let lease = authority.open_lease(&campaign, &plan, &sealed).unwrap();
        lease.validate_at(&campaign, &plan, 300).unwrap();
        assert!(matches!(
            lease.validate_at(&campaign, &plan, 401),
            Err(CheckpointPowerLossOperationsError::LeaseExpired)
        ));
    }


    #[test]
    fn journal_requires_monotonic_physical_sequence_and_resumes() {
        let campaign = campaign();
        let plan = operations_plan(&campaign);
        let authority = CheckpointPowerLossOperationsAuthority::new(
            CheckpointPowerLossOperationsKey::new(plan.operations_authority_key_id, [17u8; 32]).unwrap(),
        );
        let sealed = authority.issue_lease(
            &campaign,
            &plan,
            plan.lab_manifests[0].lab_id,
            campaign.trials[0].trial_id,
            [18u8; 16],
            1,
            200,
            900,
        ).unwrap();
        let lease = authority.open_lease(&campaign, &plan, &sealed).unwrap();
        let mut journal = CheckpointPowerLossExecutionJournal::new(
            &campaign, &plan, &lease, [19u8; 32], [20u8; 32], 210,
        ).unwrap();
        journal.append(&campaign, &plan, &lease, CheckpointPowerLossExecutionState::Prepared, [21u8; 32], [19u8; 32], 220).unwrap();
        journal.append(&campaign, &plan, &lease, CheckpointPowerLossExecutionState::Armed, [22u8; 32], [19u8; 32], 230).unwrap();
        assert_eq!(
            checkpoint_power_loss_resume_decision(&campaign, &plan, &lease, &journal, 240).unwrap(),
            CheckpointPowerLossResumeDecision::AwaitPowerEvent,
        );
        assert!(journal.append(
            &campaign, &plan, &lease,
            CheckpointPowerLossExecutionState::RecoveryStarted,
            [23u8; 32], [19u8; 32], 240,
        ).is_err());
        journal.append(&campaign, &plan, &lease, CheckpointPowerLossExecutionState::PowerEventObserved, [24u8; 32], [19u8; 32], 250).unwrap();
        journal.append(&campaign, &plan, &lease, CheckpointPowerLossExecutionState::RecoveryStarted, [25u8; 32], [19u8; 32], 260).unwrap();
        let sealed_journal = authority.seal_journal(&campaign, &plan, &lease, &journal).unwrap();
        let restored = authority.open_journal(&campaign, &plan, &lease, &sealed_journal).unwrap();
        assert_eq!(restored.current_state(), Some(CheckpointPowerLossExecutionState::RecoveryStarted));
    }


    fn completed_proof(
        campaign: &CheckpointPowerLossCampaignPlan,
        plan: &CheckpointPowerLossOperationsPlan,
        lease: CheckpointPowerLossTrialLease,
        result: &CheckpointPowerLossTrialResult,
    ) -> CheckpointPowerLossExecutionProof {
        let mut journal = CheckpointPowerLossExecutionJournal::new(
            campaign, plan, &lease, [30u8; 32], [31u8; 32], 210,
        ).unwrap();
        for (index, (state, event_digest)) in [
            (CheckpointPowerLossExecutionState::Prepared, [40u8; 32]),
            (CheckpointPowerLossExecutionState::Armed, [41u8; 32]),
            (
                CheckpointPowerLossExecutionState::PowerEventObserved,
                result.power_event_evidence_digest,
            ),
            (CheckpointPowerLossExecutionState::RecoveryStarted, [43u8; 32]),
            (
                CheckpointPowerLossExecutionState::RecoveryClassified,
                result.recovered_state_digest,
            ),
        ].into_iter().enumerate() {
            journal.append(campaign, plan, &lease, state, event_digest, [30u8; 32], 220 + index as u64).unwrap();
        }
        let sealed_result_digest = [51u8; 32];
        journal.append(campaign, plan, &lease, CheckpointPowerLossExecutionState::EvidenceSealed, sealed_result_digest, [30u8; 32], 230).unwrap();
        journal.append(campaign, plan, &lease, CheckpointPowerLossExecutionState::Completed, [52u8; 32], [30u8; 32], 240).unwrap();
        let receipt = CheckpointPowerLossExecutionReceipt::new(
            campaign, plan, &lease, &journal, result, sealed_result_digest, 240,
        ).unwrap();
        CheckpointPowerLossExecutionProof { lease, journal, receipt }
    }

    #[test]
    fn operations_evidence_requires_one_complete_proof_per_result() {
        let campaign = campaign();
        let plan = operations_plan(&campaign);
        let authority = CheckpointPowerLossOperationsAuthority::new(
            CheckpointPowerLossOperationsKey::new(plan.operations_authority_key_id, [17u8; 32]).unwrap(),
        );
        let sealed_lease = authority.issue_lease(
            &campaign, &plan, plan.lab_manifests[0].lab_id,
            campaign.trials[0].trial_id, [18u8; 16], 1, 200, 900,
        ).unwrap();
        let lease = authority.open_lease(&campaign, &plan, &sealed_lease).unwrap();
        let result = CheckpointPowerLossTrialResult {
            schema: crate::CHECKPOINT_POWER_LOSS_RESULT_SCHEMA.to_owned(),
            campaign_id: campaign.campaign_id,
            campaign_digest: campaign.digest().unwrap(),
            trial_id: campaign.trials[0].trial_id,
            storage_profile_digest: campaign.trials[0].storage_profile_digest,
            evidence_class: campaign.trials[0].evidence_class,
            durability_boundary: campaign.trials[0].durability_boundary,
            workload_digest: campaign.trials[0].workload_digest,
            pre_power_loss_digest: campaign.trials[0].expected_pre_power_loss_digest,
            recovered_state_digest: [60u8; 32],
            power_event_evidence_digest: [61u8; 32],
            outcome: crate::CheckpointPowerLossRecoveryOutcome::CleanRecovery,
            filesystem_consistency_verified: true,
            application_consistency_verified: true,
            completed_at_unix_seconds: 240,
        };
        let result_evidence = CheckpointPowerLossCampaignEvidence {
            schema: crate::CHECKPOINT_POWER_LOSS_EVIDENCE_SCHEMA.to_owned(),
            campaign_id: campaign.campaign_id,
            campaign_digest: campaign.digest().unwrap(),
            results: vec![result.clone()],
        };
        let evidence = CheckpointPowerLossOperationsEvidence {
            schema: CHECKPOINT_POWER_LOSS_OPERATIONS_EVIDENCE_SCHEMA.to_owned(),
            campaign_id: campaign.campaign_id,
            campaign_digest: campaign.digest().unwrap(),
            operations_plan_digest: plan.digest(&campaign).unwrap(),
            operations_authority_key_id: plan.operations_authority_key_id,
            proofs: vec![completed_proof(&campaign, &plan, lease, &result)],
        };
        evidence.validate_against(&campaign, &plan, &result_evidence).unwrap();
        let sealed = authority.seal_operations_evidence(&campaign, &plan, &result_evidence, &evidence).unwrap();
        authority.open_operations_evidence(&campaign, &plan, &result_evidence, &sealed).unwrap();
    }


    #[cfg(unix)]
    #[test]
    fn durable_store_rejects_stale_journal_updates() {
        let campaign = campaign();
        let plan = operations_plan(&campaign);
        let root = std::env::temp_dir().join(format!(
            "symthaea-power-loss-journal-{}",
            std::process::id(),
        ));
        let _ = std::fs::remove_dir_all(&root);
        let store = CheckpointPowerLossJournalStore::new(
            &root,
            CheckpointPowerLossOperationsAuthority::new(
                CheckpointPowerLossOperationsKey::new(plan.operations_authority_key_id, [17u8; 32]).unwrap(),
            ),
        );
        let sealed_lease = store.authority().issue_lease(
            &campaign, &plan, plan.lab_manifests[0].lab_id,
            campaign.trials[0].trial_id, [18u8; 16], 1, 200, 900,
        ).unwrap();
        let journal = store.create(
            &campaign, &plan, &sealed_lease, [70u8; 32], [71u8; 32], 210,
        ).unwrap();
        let lease = store.authority().open_lease(&campaign, &plan, &sealed_lease).unwrap();
        let observed = journal.digest(&campaign, &plan, &lease).unwrap();
        let advanced = store.append(
            &campaign, &plan, &sealed_lease, observed,
            CheckpointPowerLossExecutionState::Prepared,
            [72u8; 32], [70u8; 32], 220,
        ).unwrap();
        assert!(matches!(
            store.append(
                &campaign, &plan, &sealed_lease, observed,
                CheckpointPowerLossExecutionState::Prepared,
                [73u8; 32], [70u8; 32], 221,
            ),
            Err(CheckpointPowerLossOperationsError::ConcurrentModification)
        ));
        assert_eq!(store.load(&campaign, &plan, &sealed_lease).unwrap(), advanced);
        let _ = std::fs::remove_dir_all(root);
    }


    #[test]
    fn observed_power_event_can_be_recovered_after_lease_expiry() {
        let campaign = campaign();
        let plan = operations_plan(&campaign);
        let authority = CheckpointPowerLossOperationsAuthority::new(
            CheckpointPowerLossOperationsKey::new(plan.operations_authority_key_id, [17u8; 32]).unwrap(),
        );
        let sealed = authority.issue_lease(
            &campaign, &plan, plan.lab_manifests[0].lab_id,
            campaign.trials[0].trial_id, [88u8; 16], 1, 200, 300,
        ).unwrap();
        let lease = authority.open_lease(&campaign, &plan, &sealed).unwrap();
        let mut journal = CheckpointPowerLossExecutionJournal::new(
            &campaign, &plan, &lease, [89u8; 32], [90u8; 32], 210,
        ).unwrap();
        journal.append(&campaign, &plan, &lease, CheckpointPowerLossExecutionState::Prepared, [91u8; 32], [89u8; 32], 220).unwrap();
        journal.append(&campaign, &plan, &lease, CheckpointPowerLossExecutionState::Armed, [92u8; 32], [89u8; 32], 230).unwrap();
        journal.append(&campaign, &plan, &lease, CheckpointPowerLossExecutionState::PowerEventObserved, [93u8; 32], [89u8; 32], 240).unwrap();
        assert_eq!(
            checkpoint_power_loss_resume_decision(&campaign, &plan, &lease, &journal, 400).unwrap(),
            CheckpointPowerLossResumeDecision::ContinueWith(CheckpointPowerLossExecutionState::RecoveryStarted),
        );
        journal.append(&campaign, &plan, &lease, CheckpointPowerLossExecutionState::RecoveryStarted, [94u8; 32], [95u8; 32], 400).unwrap();
    }

}
