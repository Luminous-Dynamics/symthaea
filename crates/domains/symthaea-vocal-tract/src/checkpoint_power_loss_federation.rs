// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Federated authorization and evidence reconciliation for physical power-loss campaigns.
//!
//! Series 17 establishes authenticated per-lab execution histories. This module adds
//! a separate federation authority that freezes the participating lab roster, assigns
//! globally ordered trials, handles revocation, and verifies independently signed lab
//! bundles before existing operations evidence is merged.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use crate::{
    CheckpointPowerLossCampaignEvidence, CheckpointPowerLossCampaignPlan,
    CheckpointPowerLossLabId,
    CheckpointPowerLossOperationsError, CheckpointPowerLossOperationsEvidence,
    CheckpointPowerLossOperationsKeyId, CheckpointPowerLossOperationsPlan,
    merge_checkpoint_power_loss_operations_evidence,
    CheckpointStorageEvidenceError,
};

pub const CHECKPOINT_POWER_LOSS_FEDERATION_MEMBER_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-federation-member.v1";
pub const CHECKPOINT_POWER_LOSS_FEDERATION_PLAN_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-federation-plan.v1";
pub const CHECKPOINT_POWER_LOSS_FEDERATION_AUTHORITY_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-federation-authority.v1";
pub const CHECKPOINT_POWER_LOSS_FEDERATION_ALLOCATION_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-federation-allocation.v1";
pub const CHECKPOINT_POWER_LOSS_FEDERATION_REVOCATION_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-federation-revocations.v1";
pub const CHECKPOINT_POWER_LOSS_FEDERATION_CLOCK_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-federation-clock.v1";
pub const CHECKPOINT_POWER_LOSS_FEDERATED_LAB_EVIDENCE_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-federated-lab-evidence.v1";
pub const CHECKPOINT_POWER_LOSS_FEDERATION_MERGE_SCHEMA: &str =
    "symthaea.checkpoint-power-loss-federation-merge.v1";
pub const MAX_CHECKPOINT_POWER_LOSS_FEDERATION_MEMBERS: usize = 64;
pub const MAX_CHECKPOINT_POWER_LOSS_FEDERATION_ALLOCATIONS: usize = 4096;
pub const MAX_CHECKPOINT_POWER_LOSS_FEDERATION_REVOCATIONS: usize = 256;
pub const MAX_CHECKPOINT_POWER_LOSS_FEDERATION_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_CHECKPOINT_POWER_LOSS_CLOCK_OFFSET_SECONDS: u64 = 24 * 60 * 60;
pub const MAX_CHECKPOINT_POWER_LOSS_CLOCK_UNCERTAINTY_SECONDS: u64 = 60 * 60;

const FEDERATION_PLAN_DIGEST_DOMAIN: &[u8] =
    b"symthaea-power-loss-federation-plan-digest-v1\0";
const FEDERATION_AUTH_DOMAIN: &[u8] = b"symthaea-power-loss-federation-auth-v1\0";
const FEDERATION_ALLOCATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea-power-loss-federation-allocation-digest-v1\0";
const FEDERATION_ALLOCATION_AUTH_DOMAIN: &[u8] =
    b"symthaea-power-loss-federation-allocation-auth-v1\0";
const FEDERATION_REVOCATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea-power-loss-federation-revocation-digest-v1\0";
const FEDERATION_REVOCATION_AUTH_DOMAIN: &[u8] =
    b"symthaea-power-loss-federation-revocation-auth-v1\0";
const FEDERATION_CLOCK_DIGEST_DOMAIN: &[u8] =
    b"symthaea-power-loss-federation-clock-digest-v1\0";
const FEDERATED_LAB_EVIDENCE_AUTH_DOMAIN: &[u8] =
    b"symthaea-power-loss-federated-lab-evidence-auth-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointPowerLossFederationId(pub [u8; 16]);

impl CheckpointPowerLossFederationId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointPowerLossFederationError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointPowerLossFederationError::InvalidFederation);
        }
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointPowerLossFederationKeyId(pub [u8; 16]);

impl CheckpointPowerLossFederationKeyId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointPowerLossFederationError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointPowerLossFederationError::InvalidKey);
        }
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointPowerLossLabEvidenceKeyId(pub [u8; 16]);

impl CheckpointPowerLossLabEvidenceKeyId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointPowerLossFederationError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointPowerLossFederationError::InvalidKey);
        }
        Ok(Self(bytes))
    }
}

pub struct CheckpointPowerLossFederationKey {
    id: CheckpointPowerLossFederationKeyId,
    bytes: [u8; 32],
}

impl CheckpointPowerLossFederationKey {
    pub fn new(
        id: CheckpointPowerLossFederationKeyId,
        bytes: [u8; 32],
    ) -> Result<Self, CheckpointPowerLossFederationError> {
        if bytes == [0u8; 32] {
            return Err(CheckpointPowerLossFederationError::InvalidKey);
        }
        Ok(Self { id, bytes })
    }

    pub fn id(&self) -> CheckpointPowerLossFederationKeyId {
        self.id
    }
}

impl Drop for CheckpointPowerLossFederationKey {
    fn drop(&mut self) {
        self.bytes.zeroize();
    }
}

pub struct CheckpointPowerLossLabEvidenceKey {
    id: CheckpointPowerLossLabEvidenceKeyId,
    bytes: [u8; 32],
}

impl CheckpointPowerLossLabEvidenceKey {
    pub fn new(
        id: CheckpointPowerLossLabEvidenceKeyId,
        bytes: [u8; 32],
    ) -> Result<Self, CheckpointPowerLossFederationError> {
        if bytes == [0u8; 32] {
            return Err(CheckpointPowerLossFederationError::InvalidKey);
        }
        Ok(Self { id, bytes })
    }

    pub fn id(&self) -> CheckpointPowerLossLabEvidenceKeyId {
        self.id
    }
}

impl Drop for CheckpointPowerLossLabEvidenceKey {
    fn drop(&mut self) {
        self.bytes.zeroize();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossFederationMember {
    pub schema: String,
    pub federation_id: CheckpointPowerLossFederationId,
    pub lab_id: CheckpointPowerLossLabId,
    pub lab_evidence_key_id: CheckpointPowerLossLabEvidenceKeyId,
    pub operations_authority_key_id: CheckpointPowerLossOperationsKeyId,
    pub organization_binding: [u8; 32],
    pub administration_binding: [u8; 32],
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointPowerLossFederationMember {
    pub fn validate(&self) -> Result<(), CheckpointPowerLossFederationError> {
        if self.schema != CHECKPOINT_POWER_LOSS_FEDERATION_MEMBER_SCHEMA
            || self.federation_id.0 == [0u8; 16]
            || self.lab_id.0 == [0u8; 16]
            || self.lab_evidence_key_id.0 == [0u8; 16]
            || self.operations_authority_key_id.0 == [0u8; 16]
            || self.organization_binding == [0u8; 32]
            || self.administration_binding == [0u8; 32]
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointPowerLossFederationError::InvalidMember);
        }
        Ok(())
    }

    pub fn valid_at(&self, unix_seconds: u64) -> bool {
        self.valid_from_unix_seconds <= unix_seconds
            && unix_seconds <= self.valid_until_unix_seconds
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossFederationPlan {
    pub schema: String,
    pub federation_id: CheckpointPowerLossFederationId,
    pub campaign_id: [u8; 16],
    pub campaign_digest: [u8; 32],
    pub operations_plan_digest: [u8; 32],
    pub federation_authority_key_id: CheckpointPowerLossFederationKeyId,
    pub epoch: u64,
    pub members: Vec<CheckpointPowerLossFederationMember>,
    pub minimum_member_labs: u16,
    pub maximum_clock_offset_seconds: u64,
    pub maximum_clock_uncertainty_seconds: u64,
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointPowerLossFederationPlan {
    pub fn validate_against(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
    ) -> Result<(), CheckpointPowerLossFederationError> {
        campaign
            .validate()
            .map_err(CheckpointPowerLossFederationError::StorageEvidence)?;
        operations
            .validate_against(campaign)
            .map_err(CheckpointPowerLossFederationError::Operations)?;
        let campaign_digest = campaign
            .digest()
            .map_err(CheckpointPowerLossFederationError::StorageEvidence)?;
        let operations_plan_digest = operations
            .digest(campaign)
            .map_err(CheckpointPowerLossFederationError::Operations)?;
        if self.schema != CHECKPOINT_POWER_LOSS_FEDERATION_PLAN_SCHEMA
            || self.federation_id.0 == [0u8; 16]
            || self.campaign_id != campaign.campaign_id
            || self.campaign_digest != campaign_digest
            || self.operations_plan_digest != operations_plan_digest
            || self.federation_authority_key_id.0 == [0u8; 16]
            || self.epoch == 0
            || self.members.len() < 2
            || self.members.len() > MAX_CHECKPOINT_POWER_LOSS_FEDERATION_MEMBERS
            || self.minimum_member_labs < 2
            || usize::from(self.minimum_member_labs) > self.members.len()
            || self.maximum_clock_offset_seconds == 0
            || self.maximum_clock_offset_seconds > MAX_CHECKPOINT_POWER_LOSS_CLOCK_OFFSET_SECONDS
            || self.maximum_clock_uncertainty_seconds == 0
            || self.maximum_clock_uncertainty_seconds
                > MAX_CHECKPOINT_POWER_LOSS_CLOCK_UNCERTAINTY_SECONDS
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointPowerLossFederationError::InvalidPlan);
        }

        let mut lab_ids = HashSet::with_capacity(self.members.len());
        let mut evidence_keys = HashSet::with_capacity(self.members.len());
        let mut administration_bindings = HashSet::with_capacity(self.members.len());
        for member in &self.members {
            member.validate()?;
            if member.federation_id != self.federation_id
                || member.operations_authority_key_id != operations.operations_authority_key_id
                || member.valid_from_unix_seconds > self.valid_from_unix_seconds
                || member.valid_until_unix_seconds < self.valid_until_unix_seconds
                || operations.lab(member.lab_id).is_none()
            {
                return Err(CheckpointPowerLossFederationError::CampaignBindingMismatch);
            }
            if !lab_ids.insert(member.lab_id)
                || !evidence_keys.insert(member.lab_evidence_key_id)
                || !administration_bindings.insert(member.administration_binding)
            {
                return Err(CheckpointPowerLossFederationError::DuplicateMember);
            }
        }
        if operations
            .lab_manifests
            .iter()
            .any(|lab| !lab_ids.contains(&lab.lab_id))
        {
            return Err(CheckpointPowerLossFederationError::MissingMember);
        }
        Ok(())
    }

    pub fn member(
        &self,
        lab_id: CheckpointPowerLossLabId,
    ) -> Option<&CheckpointPowerLossFederationMember> {
        self.members.iter().find(|member| member.lab_id == lab_id)
    }

    pub fn digest(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
    ) -> Result<[u8; 32], CheckpointPowerLossFederationError> {
        self.validate_against(campaign, operations)?;
        digest_serialized(FEDERATION_PLAN_DIGEST_DOMAIN, self)
    }
}

pub struct CheckpointPowerLossFederationAuthority {
    key: CheckpointPowerLossFederationKey,
}

impl CheckpointPowerLossFederationAuthority {
    pub fn new(key: CheckpointPowerLossFederationKey) -> Self {
        Self { key }
    }

    pub fn key_id(&self) -> CheckpointPowerLossFederationKeyId {
        self.key.id()
    }

    pub fn seal_plan(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        plan: &CheckpointPowerLossFederationPlan,
    ) -> Result<Vec<u8>, CheckpointPowerLossFederationError> {
        plan.validate_against(campaign, operations)?;
        if self.key_id() != plan.federation_authority_key_id {
            return Err(CheckpointPowerLossFederationError::AuthenticationFailed);
        }
        let body = bounded_encode(plan)?;
        bounded_encode(&CheckpointPowerLossFederationWire {
            schema: CHECKPOINT_POWER_LOSS_FEDERATION_AUTHORITY_SCHEMA.to_owned(),
            key_id: self.key_id(),
            authentication_tag: self.authenticate(FEDERATION_AUTH_DOMAIN, &body),
            body,
        })
    }

    pub fn open_plan(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        encoded: &[u8],
    ) -> Result<CheckpointPowerLossFederationPlan, CheckpointPowerLossFederationError> {
        let wire: CheckpointPowerLossFederationWire = bounded_decode(encoded)?;
        if wire.schema != CHECKPOINT_POWER_LOSS_FEDERATION_AUTHORITY_SCHEMA
            || wire.key_id != self.key_id()
            || !constant_time_equal(
                &wire.authentication_tag,
                &self.authenticate(FEDERATION_AUTH_DOMAIN, &wire.body),
            )
        {
            return Err(CheckpointPowerLossFederationError::AuthenticationFailed);
        }
        let plan: CheckpointPowerLossFederationPlan = bounded_decode(&wire.body)?;
        plan.validate_against(campaign, operations)?;
        if plan.federation_authority_key_id != self.key_id() {
            return Err(CheckpointPowerLossFederationError::AuthenticationFailed);
        }
        Ok(plan)
    }

    fn authenticate(&self, domain: &[u8], body: &[u8]) -> [u8; 32] {
        let mut input = Vec::with_capacity(domain.len() + body.len());
        input.extend_from_slice(domain);
        input.extend_from_slice(body);
        *blake3::keyed_hash(&self.key.bytes, &input).as_bytes()
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossFederatedTrialAllocation {
    pub schema: String,
    pub federation_id: CheckpointPowerLossFederationId,
    pub federation_plan_digest: [u8; 32],
    pub campaign_id: [u8; 16],
    pub campaign_digest: [u8; 32],
    pub epoch: u64,
    pub allocation_id: [u8; 16],
    pub allocation_sequence: u64,
    pub trial_id: [u8; 16],
    pub storage_profile_digest: [u8; 32],
    pub lab_id: CheckpointPowerLossLabId,
    pub lab_evidence_key_id: CheckpointPowerLossLabEvidenceKeyId,
    pub attempt: u16,
    pub issued_at_unix_seconds: u64,
    pub not_before_unix_seconds: u64,
    pub expires_at_unix_seconds: u64,
}

impl CheckpointPowerLossFederatedTrialAllocation {
    pub fn validate_against(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        federation: &CheckpointPowerLossFederationPlan,
    ) -> Result<(), CheckpointPowerLossFederationError> {
        federation.validate_against(campaign, operations)?;
        let trial = campaign
            .trials
            .iter()
            .find(|trial| trial.trial_id == self.trial_id)
            .ok_or(CheckpointPowerLossFederationError::UnknownTrial)?;
        let member = federation
            .member(self.lab_id)
            .ok_or(CheckpointPowerLossFederationError::InvalidMember)?;
        if self.schema != CHECKPOINT_POWER_LOSS_FEDERATION_ALLOCATION_SCHEMA
            || self.federation_id != federation.federation_id
            || self.federation_plan_digest != federation.digest(campaign, operations)?
            || self.campaign_id != campaign.campaign_id
            || self.campaign_digest
                != campaign
                    .digest()
                    .map_err(CheckpointPowerLossFederationError::StorageEvidence)?
            || self.epoch != federation.epoch
            || self.allocation_id == [0u8; 16]
            || self.allocation_sequence == 0
            || self.storage_profile_digest != trial.storage_profile_digest
            || self.lab_evidence_key_id != member.lab_evidence_key_id
            || self.attempt == 0
            || self.attempt > operations.maximum_attempts_per_trial
            || self.issued_at_unix_seconds == 0
            || self.not_before_unix_seconds < self.issued_at_unix_seconds
            || self.expires_at_unix_seconds <= self.not_before_unix_seconds
            || !member.valid_at(self.issued_at_unix_seconds)
            || !member.valid_at(self.expires_at_unix_seconds)
            || self.issued_at_unix_seconds < federation.valid_from_unix_seconds
            || self.expires_at_unix_seconds > federation.valid_until_unix_seconds
        {
            return Err(CheckpointPowerLossFederationError::InvalidAllocation);
        }
        Ok(())
    }

    pub fn digest(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        federation: &CheckpointPowerLossFederationPlan,
    ) -> Result<[u8; 32], CheckpointPowerLossFederationError> {
        self.validate_against(campaign, operations, federation)?;
        digest_serialized(FEDERATION_ALLOCATION_DIGEST_DOMAIN, self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointPowerLossFederationRevocationScope {
    FutureAssignments,
    AllEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossFederationRevocation {
    pub lab_id: CheckpointPowerLossLabId,
    pub lab_evidence_key_id: CheckpointPowerLossLabEvidenceKeyId,
    pub effective_at_unix_seconds: u64,
    pub scope: CheckpointPowerLossFederationRevocationScope,
    pub reason_digest: [u8; 32],
}

impl CheckpointPowerLossFederationRevocation {
    fn validate_against(
        &self,
        federation: &CheckpointPowerLossFederationPlan,
    ) -> Result<(), CheckpointPowerLossFederationError> {
        let member = federation
            .member(self.lab_id)
            .ok_or(CheckpointPowerLossFederationError::InvalidMember)?;
        if self.lab_evidence_key_id != member.lab_evidence_key_id
            || self.effective_at_unix_seconds == 0
            || self.reason_digest == [0u8; 32]
        {
            return Err(CheckpointPowerLossFederationError::InvalidRevocation);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossFederationRevocationList {
    pub schema: String,
    pub federation_id: CheckpointPowerLossFederationId,
    pub federation_plan_digest: [u8; 32],
    pub federation_authority_key_id: CheckpointPowerLossFederationKeyId,
    pub epoch: u64,
    pub sequence: u64,
    pub issued_at_unix_seconds: u64,
    pub revocations: Vec<CheckpointPowerLossFederationRevocation>,
}

impl CheckpointPowerLossFederationRevocationList {
    pub fn validate_against(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        federation: &CheckpointPowerLossFederationPlan,
    ) -> Result<(), CheckpointPowerLossFederationError> {
        federation.validate_against(campaign, operations)?;
        if self.schema != CHECKPOINT_POWER_LOSS_FEDERATION_REVOCATION_SCHEMA
            || self.federation_id != federation.federation_id
            || self.federation_plan_digest != federation.digest(campaign, operations)?
            || self.federation_authority_key_id != federation.federation_authority_key_id
            || self.epoch != federation.epoch
            || self.sequence == 0
            || self.issued_at_unix_seconds == 0
            || self.revocations.len() > MAX_CHECKPOINT_POWER_LOSS_FEDERATION_REVOCATIONS
        {
            return Err(CheckpointPowerLossFederationError::InvalidRevocation);
        }
        let mut identities = HashSet::with_capacity(self.revocations.len());
        for revocation in &self.revocations {
            revocation.validate_against(federation)?;
            if !identities.insert((revocation.lab_id, revocation.lab_evidence_key_id)) {
                return Err(CheckpointPowerLossFederationError::DuplicateRevocation);
            }
        }
        Ok(())
    }

    pub fn digest(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        federation: &CheckpointPowerLossFederationPlan,
    ) -> Result<[u8; 32], CheckpointPowerLossFederationError> {
        self.validate_against(campaign, operations, federation)?;
        digest_serialized(FEDERATION_REVOCATION_DIGEST_DOMAIN, self)
    }

    pub fn rejects(
        &self,
        allocation: &CheckpointPowerLossFederatedTrialAllocation,
        finalized_at_unix_seconds: u64,
    ) -> bool {
        self.revocations.iter().any(|revocation| {
            if revocation.lab_id != allocation.lab_id
                || revocation.lab_evidence_key_id != allocation.lab_evidence_key_id
            {
                return false;
            }
            match revocation.scope {
                CheckpointPowerLossFederationRevocationScope::AllEvidence => true,
                CheckpointPowerLossFederationRevocationScope::FutureAssignments => {
                    allocation.issued_at_unix_seconds >= revocation.effective_at_unix_seconds
                        || (finalized_at_unix_seconds >= revocation.effective_at_unix_seconds
                            && allocation.not_before_unix_seconds
                                >= revocation.effective_at_unix_seconds)
                }
            }
        })
    }
}

impl CheckpointPowerLossFederationAuthority {
    pub fn issue_allocation(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        federation: &CheckpointPowerLossFederationPlan,
        allocation: &CheckpointPowerLossFederatedTrialAllocation,
    ) -> Result<Vec<u8>, CheckpointPowerLossFederationError> {
        allocation.validate_against(campaign, operations, federation)?;
        if self.key_id() != federation.federation_authority_key_id {
            return Err(CheckpointPowerLossFederationError::AuthenticationFailed);
        }
        let body = bounded_encode(allocation)?;
        bounded_encode(&CheckpointPowerLossFederationWire {
            schema: CHECKPOINT_POWER_LOSS_FEDERATION_ALLOCATION_SCHEMA.to_owned(),
            key_id: self.key_id(),
            authentication_tag: self.authenticate(FEDERATION_ALLOCATION_AUTH_DOMAIN, &body),
            body,
        })
    }

    pub fn open_allocation(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        federation: &CheckpointPowerLossFederationPlan,
        encoded: &[u8],
    ) -> Result<CheckpointPowerLossFederatedTrialAllocation, CheckpointPowerLossFederationError> {
        let wire: CheckpointPowerLossFederationWire = bounded_decode(encoded)?;
        if wire.schema != CHECKPOINT_POWER_LOSS_FEDERATION_ALLOCATION_SCHEMA
            || wire.key_id != self.key_id()
            || !constant_time_equal(
                &wire.authentication_tag,
                &self.authenticate(FEDERATION_ALLOCATION_AUTH_DOMAIN, &wire.body),
            )
        {
            return Err(CheckpointPowerLossFederationError::AuthenticationFailed);
        }
        let allocation: CheckpointPowerLossFederatedTrialAllocation = bounded_decode(&wire.body)?;
        allocation.validate_against(campaign, operations, federation)?;
        Ok(allocation)
    }

    pub fn seal_revocations(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        federation: &CheckpointPowerLossFederationPlan,
        revocations: &CheckpointPowerLossFederationRevocationList,
    ) -> Result<Vec<u8>, CheckpointPowerLossFederationError> {
        revocations.validate_against(campaign, operations, federation)?;
        if self.key_id() != federation.federation_authority_key_id {
            return Err(CheckpointPowerLossFederationError::AuthenticationFailed);
        }
        let body = bounded_encode(revocations)?;
        bounded_encode(&CheckpointPowerLossFederationWire {
            schema: CHECKPOINT_POWER_LOSS_FEDERATION_REVOCATION_SCHEMA.to_owned(),
            key_id: self.key_id(),
            authentication_tag: self.authenticate(FEDERATION_REVOCATION_AUTH_DOMAIN, &body),
            body,
        })
    }

    pub fn open_revocations(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        federation: &CheckpointPowerLossFederationPlan,
        encoded: &[u8],
    ) -> Result<CheckpointPowerLossFederationRevocationList, CheckpointPowerLossFederationError> {
        let wire: CheckpointPowerLossFederationWire = bounded_decode(encoded)?;
        if wire.schema != CHECKPOINT_POWER_LOSS_FEDERATION_REVOCATION_SCHEMA
            || wire.key_id != self.key_id()
            || !constant_time_equal(
                &wire.authentication_tag,
                &self.authenticate(FEDERATION_REVOCATION_AUTH_DOMAIN, &wire.body),
            )
        {
            return Err(CheckpointPowerLossFederationError::AuthenticationFailed);
        }
        let revocations: CheckpointPowerLossFederationRevocationList = bounded_decode(&wire.body)?;
        revocations.validate_against(campaign, operations, federation)?;
        Ok(revocations)
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossLabClockAttestation {
    pub schema: String,
    pub federation_id: CheckpointPowerLossFederationId,
    pub federation_plan_digest: [u8; 32],
    pub lab_id: CheckpointPowerLossLabId,
    pub lab_evidence_key_id: CheckpointPowerLossLabEvidenceKeyId,
    pub sample_id: [u8; 16],
    pub first_allocation_sequence: u64,
    pub last_allocation_sequence: u64,
    pub lab_unix_seconds: u64,
    pub federation_unix_seconds: u64,
    pub uncertainty_seconds: u64,
    pub observed_at_unix_seconds: u64,
}

impl CheckpointPowerLossLabClockAttestation {
    pub fn validate_against(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        federation: &CheckpointPowerLossFederationPlan,
    ) -> Result<(), CheckpointPowerLossFederationError> {
        federation.validate_against(campaign, operations)?;
        let member = federation
            .member(self.lab_id)
            .ok_or(CheckpointPowerLossFederationError::InvalidMember)?;
        if self.schema != CHECKPOINT_POWER_LOSS_FEDERATION_CLOCK_SCHEMA
            || self.federation_id != federation.federation_id
            || self.federation_plan_digest != federation.digest(campaign, operations)?
            || self.lab_evidence_key_id != member.lab_evidence_key_id
            || self.sample_id == [0u8; 16]
            || self.first_allocation_sequence == 0
            || self.last_allocation_sequence < self.first_allocation_sequence
            || self.lab_unix_seconds == 0
            || self.federation_unix_seconds == 0
            || self.uncertainty_seconds == 0
            || self.uncertainty_seconds > federation.maximum_clock_uncertainty_seconds
            || self.lab_unix_seconds.abs_diff(self.federation_unix_seconds)
                > federation.maximum_clock_offset_seconds
            || !member.valid_at(self.observed_at_unix_seconds)
            || self.observed_at_unix_seconds < federation.valid_from_unix_seconds
            || self.observed_at_unix_seconds > federation.valid_until_unix_seconds
        {
            return Err(CheckpointPowerLossFederationError::InvalidClockAttestation);
        }
        Ok(())
    }

    pub fn digest(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        federation: &CheckpointPowerLossFederationPlan,
    ) -> Result<[u8; 32], CheckpointPowerLossFederationError> {
        self.validate_against(campaign, operations, federation)?;
        digest_serialized(FEDERATION_CLOCK_DIGEST_DOMAIN, self)
    }

    pub fn absolute_offset_seconds(&self) -> u64 {
        self.lab_unix_seconds.abs_diff(self.federation_unix_seconds)
    }
}

pub struct CheckpointPowerLossLabEvidenceAuthority {
    key: CheckpointPowerLossLabEvidenceKey,
}

impl CheckpointPowerLossLabEvidenceAuthority {
    pub fn new(key: CheckpointPowerLossLabEvidenceKey) -> Self {
        Self { key }
    }

    pub fn key_id(&self) -> CheckpointPowerLossLabEvidenceKeyId {
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
pub struct CheckpointPowerLossFederatedLabEvidence {
    pub schema: String,
    pub federation_id: CheckpointPowerLossFederationId,
    pub federation_plan_digest: [u8; 32],
    pub revocation_list_digest: [u8; 32],
    pub epoch: u64,
    pub lab_id: CheckpointPowerLossLabId,
    pub lab_evidence_key_id: CheckpointPowerLossLabEvidenceKeyId,
    pub sealed_result_evidence_digest: [u8; 32],
    pub clock_attestation: CheckpointPowerLossLabClockAttestation,
    pub allocations: Vec<CheckpointPowerLossFederatedTrialAllocation>,
    pub operations_evidence: CheckpointPowerLossOperationsEvidence,
}

impl CheckpointPowerLossFederatedLabEvidence {
    #[allow(clippy::too_many_arguments)]
    pub fn validate_against(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        result_evidence: &CheckpointPowerLossCampaignEvidence,
        federation: &CheckpointPowerLossFederationPlan,
        revocations: &CheckpointPowerLossFederationRevocationList,
    ) -> Result<(), CheckpointPowerLossFederationError> {
        federation.validate_against(campaign, operations)?;
        revocations.validate_against(campaign, operations, federation)?;
        self.operations_evidence
            .validate_partial_against(campaign, operations, result_evidence)
            .map_err(CheckpointPowerLossFederationError::Operations)?;
        let member = federation
            .member(self.lab_id)
            .ok_or(CheckpointPowerLossFederationError::InvalidMember)?;
        if self.schema != CHECKPOINT_POWER_LOSS_FEDERATED_LAB_EVIDENCE_SCHEMA
            || self.federation_id != federation.federation_id
            || self.federation_plan_digest != federation.digest(campaign, operations)?
            || self.revocation_list_digest
                != revocations.digest(campaign, operations, federation)?
            || self.epoch != federation.epoch
            || self.lab_evidence_key_id != member.lab_evidence_key_id
            || self.sealed_result_evidence_digest == [0u8; 32]
            || self.sealed_result_evidence_digest
                != self.operations_evidence.sealed_result_evidence_digest
            || self.allocations.is_empty()
            || self.allocations.len() > MAX_CHECKPOINT_POWER_LOSS_FEDERATION_ALLOCATIONS
            || self.allocations.len() != self.operations_evidence.proofs.len()
            || self.clock_attestation.lab_id != self.lab_id
            || self.clock_attestation.lab_evidence_key_id != self.lab_evidence_key_id
        {
            return Err(CheckpointPowerLossFederationError::InvalidLabEvidence);
        }
        self.clock_attestation
            .validate_against(campaign, operations, federation)?;

        let mut allocation_ids = HashSet::with_capacity(self.allocations.len());
        let mut allocation_sequences = HashSet::with_capacity(self.allocations.len());
        let mut allocation_trials = HashSet::with_capacity(self.allocations.len());
        for allocation in &self.allocations {
            allocation.validate_against(campaign, operations, federation)?;
            if allocation.lab_id != self.lab_id
                || allocation.lab_evidence_key_id != self.lab_evidence_key_id
                || allocation.allocation_sequence
                    < self.clock_attestation.first_allocation_sequence
                || allocation.allocation_sequence
                    > self.clock_attestation.last_allocation_sequence
                || !allocation_ids.insert(allocation.allocation_id)
                || !allocation_sequences.insert(allocation.allocation_sequence)
                || !allocation_trials.insert(allocation.trial_id)
            {
                return Err(CheckpointPowerLossFederationError::InvalidAllocation);
            }
        }

        let allocation_by_trial = self
            .allocations
            .iter()
            .map(|allocation| (allocation.trial_id, allocation))
            .collect::<std::collections::HashMap<_, _>>();
        for proof in &self.operations_evidence.proofs {
            let allocation = allocation_by_trial
                .get(&proof.receipt.trial_id)
                .ok_or(CheckpointPowerLossFederationError::MissingAllocation)?;
            if proof.receipt.lab_id != self.lab_id
                || proof.lease.lab_id != self.lab_id
                || proof.receipt.attempt != allocation.attempt
                || proof.lease.attempt != allocation.attempt
                || proof.lease.storage_profile_digest != allocation.storage_profile_digest
                || proof.lease.issued_at_unix_seconds < allocation.not_before_unix_seconds
                || proof.lease.expires_at_unix_seconds > allocation.expires_at_unix_seconds
                || revocations.rejects(allocation, proof.receipt.finalized_at_unix_seconds)
            {
                return Err(CheckpointPowerLossFederationError::RevokedEvidence);
            }
        }
        if self
            .operations_evidence
            .journal_concurrency_tests
            .iter()
            .any(|test| test.lab_id != self.lab_id)
        {
            return Err(CheckpointPowerLossFederationError::InvalidLabEvidence);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointPowerLossFederatedLabEvidenceWire {
    schema: String,
    key_id: CheckpointPowerLossLabEvidenceKeyId,
    body: Vec<u8>,
    authentication_tag: [u8; 32],
}

impl CheckpointPowerLossLabEvidenceAuthority {
    #[allow(clippy::too_many_arguments)]
    pub fn seal_lab_evidence(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        result_evidence: &CheckpointPowerLossCampaignEvidence,
        federation: &CheckpointPowerLossFederationPlan,
        revocations: &CheckpointPowerLossFederationRevocationList,
        evidence: &CheckpointPowerLossFederatedLabEvidence,
    ) -> Result<Vec<u8>, CheckpointPowerLossFederationError> {
        evidence.validate_against(
            campaign,
            operations,
            result_evidence,
            federation,
            revocations,
        )?;
        if self.key_id() != evidence.lab_evidence_key_id {
            return Err(CheckpointPowerLossFederationError::AuthenticationFailed);
        }
        let body = bounded_encode(evidence)?;
        bounded_encode(&CheckpointPowerLossFederatedLabEvidenceWire {
            schema: CHECKPOINT_POWER_LOSS_FEDERATED_LAB_EVIDENCE_SCHEMA.to_owned(),
            key_id: self.key_id(),
            authentication_tag: self.authenticate(FEDERATED_LAB_EVIDENCE_AUTH_DOMAIN, &body),
            body,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open_lab_evidence(
        &self,
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        result_evidence: &CheckpointPowerLossCampaignEvidence,
        federation: &CheckpointPowerLossFederationPlan,
        revocations: &CheckpointPowerLossFederationRevocationList,
        encoded: &[u8],
    ) -> Result<CheckpointPowerLossFederatedLabEvidence, CheckpointPowerLossFederationError> {
        let wire: CheckpointPowerLossFederatedLabEvidenceWire = bounded_decode(encoded)?;
        if wire.schema != CHECKPOINT_POWER_LOSS_FEDERATED_LAB_EVIDENCE_SCHEMA
            || wire.key_id != self.key_id()
            || !constant_time_equal(
                &wire.authentication_tag,
                &self.authenticate(FEDERATED_LAB_EVIDENCE_AUTH_DOMAIN, &wire.body),
            )
        {
            return Err(CheckpointPowerLossFederationError::AuthenticationFailed);
        }
        let evidence: CheckpointPowerLossFederatedLabEvidence = bounded_decode(&wire.body)?;
        evidence.validate_against(
            campaign,
            operations,
            result_evidence,
            federation,
            revocations,
        )?;
        if evidence.lab_evidence_key_id != self.key_id() {
            return Err(CheckpointPowerLossFederationError::AuthenticationFailed);
        }
        Ok(evidence)
    }
}


#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPowerLossFederationSummary {
    pub verified_allocations: usize,
    pub verified_lab_bundles: usize,
    pub unique_labs: usize,
    pub merged_execution_proofs: usize,
    pub revocation_entries_checked: usize,
    pub clock_attestations_verified: usize,
    pub maximum_clock_offset_seconds: u64,
    pub maximum_clock_uncertainty_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointPowerLossFederationMerge {
    pub schema: String,
    pub federation_plan: CheckpointPowerLossFederationPlan,
    pub revocations: CheckpointPowerLossFederationRevocationList,
    pub allocations: Vec<CheckpointPowerLossFederatedTrialAllocation>,
    pub lab_evidence: Vec<CheckpointPowerLossFederatedLabEvidence>,
    pub merged_operations_evidence: CheckpointPowerLossOperationsEvidence,
    pub summary: CheckpointPowerLossFederationSummary,
}

#[allow(clippy::too_many_arguments)]
pub fn open_and_merge_checkpoint_power_loss_federation_evidence(
    federation_authority: &CheckpointPowerLossFederationAuthority,
    lab_authorities: &[CheckpointPowerLossLabEvidenceAuthority],
    campaign: &CheckpointPowerLossCampaignPlan,
    operations: &CheckpointPowerLossOperationsPlan,
    result_evidence: &CheckpointPowerLossCampaignEvidence,
    sealed_result_evidence_digest: [u8; 32],
    sealed_federation_plan: &[u8],
    sealed_revocations: &[u8],
    sealed_allocations: &[Vec<u8>],
    sealed_lab_evidence: &[Vec<u8>],
    verified_at_unix_seconds: u64,
) -> Result<CheckpointPowerLossFederationMerge, CheckpointPowerLossFederationError> {
    if verified_at_unix_seconds == 0
        || sealed_result_evidence_digest == [0u8; 32]
        || sealed_allocations.is_empty()
        || sealed_allocations.len() > MAX_CHECKPOINT_POWER_LOSS_FEDERATION_ALLOCATIONS
        || sealed_lab_evidence.is_empty()
        || sealed_lab_evidence.len() > MAX_CHECKPOINT_POWER_LOSS_FEDERATION_MEMBERS
    {
        return Err(CheckpointPowerLossFederationError::InvalidMerge);
    }
    let federation = federation_authority.open_plan(
        campaign,
        operations,
        sealed_federation_plan,
    )?;
    let revocations = federation_authority.open_revocations(
        campaign,
        operations,
        &federation,
        sealed_revocations,
    )?;
    if revocations.issued_at_unix_seconds > verified_at_unix_seconds {
        return Err(CheckpointPowerLossFederationError::InvalidRevocation);
    }

    let mut allocations = sealed_allocations
        .iter()
        .map(|encoded| {
            federation_authority.open_allocation(
                campaign,
                operations,
                &federation,
                encoded,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    allocations.sort_by_key(|allocation| allocation.allocation_sequence);
    if allocations.len() != campaign.trials.len() {
        return Err(CheckpointPowerLossFederationError::MissingAllocation);
    }
    let mut allocation_ids = HashSet::with_capacity(allocations.len());
    let mut allocation_trials = HashSet::with_capacity(allocations.len());
    for (index, allocation) in allocations.iter().enumerate() {
        let expected_sequence = u64::try_from(index)
            .map_err(|_| CheckpointPowerLossFederationError::TooLarge)?
            .checked_add(1)
            .ok_or(CheckpointPowerLossFederationError::TooLarge)?;
        if allocation.allocation_sequence != expected_sequence
            || !allocation_ids.insert(allocation.allocation_id)
            || !allocation_trials.insert(allocation.trial_id)
        {
            return Err(CheckpointPowerLossFederationError::InvalidAllocationOrder);
        }
    }
    if campaign
        .trials
        .iter()
        .any(|trial| !allocation_trials.contains(&trial.trial_id))
    {
        return Err(CheckpointPowerLossFederationError::MissingAllocation);
    }

    let mut authorities = HashMap::with_capacity(lab_authorities.len());
    for authority in lab_authorities {
        if authorities.insert(authority.key_id(), authority).is_some() {
            return Err(CheckpointPowerLossFederationError::DuplicateMember);
        }
    }
    let mut lab_evidence = Vec::with_capacity(sealed_lab_evidence.len());
    let mut labs = HashSet::with_capacity(sealed_lab_evidence.len());
    let mut observed_allocation_digests = HashSet::with_capacity(allocations.len());
    for encoded in sealed_lab_evidence {
        let wire: CheckpointPowerLossFederatedLabEvidenceWire = bounded_decode(encoded)?;
        let authority = authorities
            .get(&wire.key_id)
            .ok_or(CheckpointPowerLossFederationError::MissingLabAuthority)?;
        let evidence = authority.open_lab_evidence(
            campaign,
            operations,
            result_evidence,
            &federation,
            &revocations,
            encoded,
        )?;
        if evidence.sealed_result_evidence_digest != sealed_result_evidence_digest
            || evidence.clock_attestation.observed_at_unix_seconds > verified_at_unix_seconds
            || !labs.insert(evidence.lab_id)
        {
            return Err(CheckpointPowerLossFederationError::InvalidLabEvidence);
        }
        for allocation in &evidence.allocations {
            let digest = allocation.digest(campaign, operations, &federation)?;
            if !observed_allocation_digests.insert(digest) {
                return Err(CheckpointPowerLossFederationError::InvalidAllocationOrder);
            }
        }
        lab_evidence.push(evidence);
    }
    if labs.len() < usize::from(federation.minimum_member_labs)
        || observed_allocation_digests.len() != allocations.len()
    {
        return Err(CheckpointPowerLossFederationError::InsufficientLabCoverage);
    }
    for allocation in &allocations {
        let digest = allocation.digest(campaign, operations, &federation)?;
        if !observed_allocation_digests.contains(&digest) {
            return Err(CheckpointPowerLossFederationError::MissingAllocation);
        }
    }

    let merged_operations_evidence = merge_checkpoint_power_loss_operations_evidence(
        campaign,
        operations,
        result_evidence,
        lab_evidence
            .iter()
            .map(|evidence| evidence.operations_evidence.clone()),
    )
    .map_err(CheckpointPowerLossFederationError::Operations)?;
    if merged_operations_evidence.sealed_result_evidence_digest
        != sealed_result_evidence_digest
    {
        return Err(CheckpointPowerLossFederationError::InvalidMerge);
    }
    let summary = CheckpointPowerLossFederationSummary {
        verified_allocations: allocations.len(),
        verified_lab_bundles: lab_evidence.len(),
        unique_labs: labs.len(),
        merged_execution_proofs: merged_operations_evidence.proofs.len(),
        revocation_entries_checked: revocations.revocations.len(),
        clock_attestations_verified: lab_evidence.len(),
        maximum_clock_offset_seconds: lab_evidence
            .iter()
            .map(|evidence| evidence.clock_attestation.absolute_offset_seconds())
            .max()
            .unwrap_or(0),
        maximum_clock_uncertainty_seconds: lab_evidence
            .iter()
            .map(|evidence| evidence.clock_attestation.uncertainty_seconds)
            .max()
            .unwrap_or(0),
    };
    Ok(CheckpointPowerLossFederationMerge {
        schema: CHECKPOINT_POWER_LOSS_FEDERATION_MERGE_SCHEMA.to_owned(),
        federation_plan: federation,
        revocations,
        allocations,
        lab_evidence,
        merged_operations_evidence,
        summary,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointPowerLossFederationWire {
    schema: String,
    key_id: CheckpointPowerLossFederationKeyId,
    body: Vec<u8>,
    authentication_tag: [u8; 32],
}

#[derive(Debug)]
pub enum CheckpointPowerLossFederationError {
    InvalidKey,
    InvalidFederation,
    InvalidMember,
    DuplicateMember,
    MissingMember,
    InvalidPlan,
    InvalidAllocation,
    MissingAllocation,
    InvalidClockAttestation,
    InvalidLabEvidence,
    InvalidMerge,
    InvalidAllocationOrder,
    MissingLabAuthority,
    InsufficientLabCoverage,
    RevokedEvidence,
    InvalidRevocation,
    DuplicateRevocation,
    UnknownTrial,
    CampaignBindingMismatch,
    AuthenticationFailed,
    Encoding,
    TooLarge,
    Operations(CheckpointPowerLossOperationsError),
    StorageEvidence(CheckpointStorageEvidenceError),
}

impl std::fmt::Display for CheckpointPowerLossFederationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKey => formatter.write_str("invalid power-loss federation key"),
            Self::InvalidFederation => formatter.write_str("invalid power-loss federation identifier"),
            Self::InvalidMember => formatter.write_str("invalid power-loss federation member"),
            Self::DuplicateMember => formatter.write_str("duplicate power-loss federation member"),
            Self::MissingMember => formatter.write_str("power-loss operations lab is missing from the federation"),
            Self::InvalidPlan => formatter.write_str("invalid power-loss federation plan"),
            Self::InvalidAllocation => formatter.write_str("invalid federated power-loss trial allocation"),
            Self::MissingAllocation => formatter.write_str("federated power-loss proof is missing its allocation"),
            Self::InvalidClockAttestation => formatter.write_str("invalid federated lab clock attestation"),
            Self::InvalidLabEvidence => formatter.write_str("invalid independently signed lab evidence"),
            Self::InvalidMerge => formatter.write_str("invalid federated power-loss evidence merge"),
            Self::InvalidAllocationOrder => formatter.write_str("invalid or duplicate federation allocation order"),
            Self::MissingLabAuthority => formatter.write_str("missing independent lab evidence authority"),
            Self::InsufficientLabCoverage => formatter.write_str("insufficient independently administered lab coverage"),
            Self::RevokedEvidence => formatter.write_str("federated power-loss evidence was revoked"),
            Self::InvalidRevocation => formatter.write_str("invalid power-loss federation revocation"),
            Self::DuplicateRevocation => formatter.write_str("duplicate power-loss federation revocation"),
            Self::UnknownTrial => formatter.write_str("unknown federated power-loss trial"),
            Self::CampaignBindingMismatch => formatter.write_str("power-loss federation campaign binding mismatch"),
            Self::AuthenticationFailed => formatter.write_str("power-loss federation authentication failed"),
            Self::Encoding => formatter.write_str("power-loss federation encoding failed"),
            Self::TooLarge => formatter.write_str("power-loss federation artifact exceeds its bound"),
            Self::Operations(error) => write!(formatter, "operations evidence failed: {error}"),
            Self::StorageEvidence(error) => write!(formatter, "storage evidence failed: {error}"),
        }
    }
}

impl std::error::Error for CheckpointPowerLossFederationError {}

fn digest_serialized<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<[u8; 32], CheckpointPowerLossFederationError> {
    let encoded = postcard::to_stdvec(value)
        .map_err(|_| CheckpointPowerLossFederationError::Encoding)?;
    if encoded.len() > MAX_CHECKPOINT_POWER_LOSS_FEDERATION_BYTES {
        return Err(CheckpointPowerLossFederationError::TooLarge);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&encoded);
    Ok(*hasher.finalize().as_bytes())
}

fn bounded_encode<T: Serialize>(
    value: &T,
) -> Result<Vec<u8>, CheckpointPowerLossFederationError> {
    let encoded = postcard::to_stdvec(value)
        .map_err(|_| CheckpointPowerLossFederationError::Encoding)?;
    if encoded.len() > MAX_CHECKPOINT_POWER_LOSS_FEDERATION_BYTES {
        return Err(CheckpointPowerLossFederationError::TooLarge);
    }
    Ok(encoded)
}

fn bounded_decode<T: for<'de> Deserialize<'de>>(
    encoded: &[u8],
) -> Result<T, CheckpointPowerLossFederationError> {
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_POWER_LOSS_FEDERATION_BYTES {
        return Err(CheckpointPowerLossFederationError::TooLarge);
    }
    postcard::from_bytes(encoded).map_err(|_| CheckpointPowerLossFederationError::Encoding)
}

fn constant_time_equal(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter()
        .zip(right.iter())
        .fold(0u8, |difference, (left, right)| difference | (left ^ right))
        == 0
}

pub fn encode_checkpoint_power_loss_federation_plan(
    campaign: &CheckpointPowerLossCampaignPlan,
    operations: &CheckpointPowerLossOperationsPlan,
    federation: &CheckpointPowerLossFederationPlan,
) -> Result<Vec<u8>, CheckpointPowerLossFederationError> {
    federation.validate_against(campaign, operations)?;
    bounded_encode(federation)
}

pub fn decode_checkpoint_power_loss_federation_plan(
    campaign: &CheckpointPowerLossCampaignPlan,
    operations: &CheckpointPowerLossOperationsPlan,
    encoded: &[u8],
) -> Result<CheckpointPowerLossFederationPlan, CheckpointPowerLossFederationError> {
    let federation: CheckpointPowerLossFederationPlan = bounded_decode(encoded)?;
    federation.validate_against(campaign, operations)?;
    Ok(federation)
}

pub fn encode_checkpoint_power_loss_federation_allocation(
    campaign: &CheckpointPowerLossCampaignPlan,
    operations: &CheckpointPowerLossOperationsPlan,
    federation: &CheckpointPowerLossFederationPlan,
    allocation: &CheckpointPowerLossFederatedTrialAllocation,
) -> Result<Vec<u8>, CheckpointPowerLossFederationError> {
    allocation.validate_against(campaign, operations, federation)?;
    bounded_encode(allocation)
}

pub fn decode_checkpoint_power_loss_federation_allocation(
    campaign: &CheckpointPowerLossCampaignPlan,
    operations: &CheckpointPowerLossOperationsPlan,
    federation: &CheckpointPowerLossFederationPlan,
    encoded: &[u8],
) -> Result<CheckpointPowerLossFederatedTrialAllocation, CheckpointPowerLossFederationError> {
    let allocation: CheckpointPowerLossFederatedTrialAllocation = bounded_decode(encoded)?;
    allocation.validate_against(campaign, operations, federation)?;
    Ok(allocation)
}

pub fn encode_checkpoint_power_loss_federation_revocations(
    campaign: &CheckpointPowerLossCampaignPlan,
    operations: &CheckpointPowerLossOperationsPlan,
    federation: &CheckpointPowerLossFederationPlan,
    revocations: &CheckpointPowerLossFederationRevocationList,
) -> Result<Vec<u8>, CheckpointPowerLossFederationError> {
    revocations.validate_against(campaign, operations, federation)?;
    bounded_encode(revocations)
}

pub fn decode_checkpoint_power_loss_federation_revocations(
    campaign: &CheckpointPowerLossCampaignPlan,
    operations: &CheckpointPowerLossOperationsPlan,
    federation: &CheckpointPowerLossFederationPlan,
    encoded: &[u8],
) -> Result<CheckpointPowerLossFederationRevocationList, CheckpointPowerLossFederationError> {
    let revocations: CheckpointPowerLossFederationRevocationList = bounded_decode(encoded)?;
    revocations.validate_against(campaign, operations, federation)?;
    Ok(revocations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CHECKPOINT_POWER_LOSS_CAMPAIGN_SCHEMA, CHECKPOINT_POWER_LOSS_LAB_SCHEMA,
        CHECKPOINT_POWER_LOSS_OPERATIONS_PLAN_SCHEMA, CheckpointDurabilityBoundary,
        CheckpointPowerLossEvidenceClass, CheckpointPowerLossEvidenceKeyId,
        CheckpointPowerLossLabManifest, CheckpointPowerLossTrialPlan,
        CheckpointStorageProfileAttestationKeyId,
    };

    fn fixture() -> (
        CheckpointPowerLossCampaignPlan,
        CheckpointPowerLossOperationsPlan,
        CheckpointPowerLossFederationPlan,
    ) {
        let campaign = CheckpointPowerLossCampaignPlan {
            schema: CHECKPOINT_POWER_LOSS_CAMPAIGN_SCHEMA.to_owned(),
            campaign_id: [1; 16],
            storage_profiles: vec![[10; 32], [11; 32]],
            storage_profile_authority_key_id: CheckpointStorageProfileAttestationKeyId([2; 16]),
            power_loss_evidence_authority_key_id: CheckpointPowerLossEvidenceKeyId([3; 16]),
            test_harness_digest: [4; 32],
            power_controller_binding: [5; 32],
            power_controller_calibration_digest: [6; 32],
            operator_protocol_digest: [7; 32],
            trials: vec![
                CheckpointPowerLossTrialPlan {
                    trial_id: [20; 16],
                    storage_profile_digest: [10; 32],
                    evidence_class: CheckpointPowerLossEvidenceClass::PhysicalDevicePowerCut,
                    durability_boundary: CheckpointDurabilityBoundary::AfterFileSyncBeforePublication,
                    workload_digest: [21; 32],
                    expected_pre_power_loss_digest: [22; 32],
                },
                CheckpointPowerLossTrialPlan {
                    trial_id: [30; 16],
                    storage_profile_digest: [11; 32],
                    evidence_class: CheckpointPowerLossEvidenceClass::PhysicalDevicePowerCut,
                    durability_boundary: CheckpointDurabilityBoundary::AfterPublicationBeforeDirectorySync,
                    workload_digest: [31; 32],
                    expected_pre_power_loss_digest: [32; 32],
                },
            ],
            minimum_physical_trials: 2,
            require_all_durability_boundaries: false,
        };
        let operations_key = CheckpointPowerLossOperationsKeyId([40; 16]);
        let operations = CheckpointPowerLossOperationsPlan {
            schema: CHECKPOINT_POWER_LOSS_OPERATIONS_PLAN_SCHEMA.to_owned(),
            campaign_id: campaign.campaign_id,
            campaign_digest: campaign.digest().unwrap(),
            operations_authority_key_id: operations_key,
            lab_manifests: vec![
                CheckpointPowerLossLabManifest {
                    schema: CHECKPOINT_POWER_LOSS_LAB_SCHEMA.to_owned(),
                    lab_id: CheckpointPowerLossLabId([41; 16]),
                    organization_binding: [42; 32],
                    operator_group_binding: [43; 32],
                    test_harness_binding: campaign.test_harness_digest,
                    power_controller_binding: campaign.power_controller_binding,
                    facility_binding: [44; 32],
                    valid_from_unix_seconds: 100,
                    valid_until_unix_seconds: 1_000,
                },
                CheckpointPowerLossLabManifest {
                    schema: CHECKPOINT_POWER_LOSS_LAB_SCHEMA.to_owned(),
                    lab_id: CheckpointPowerLossLabId([51; 16]),
                    organization_binding: [52; 32],
                    operator_group_binding: [53; 32],
                    test_harness_binding: campaign.test_harness_digest,
                    power_controller_binding: campaign.power_controller_binding,
                    facility_binding: [54; 32],
                    valid_from_unix_seconds: 100,
                    valid_until_unix_seconds: 1_000,
                },
            ],
            maximum_lease_seconds: 300,
            maximum_attempts_per_trial: 4,
            require_physical_event_before_recovery: true,
            require_complete_journal: true,
        };
        let federation_id = CheckpointPowerLossFederationId([60; 16]);
        let federation = CheckpointPowerLossFederationPlan {
            schema: CHECKPOINT_POWER_LOSS_FEDERATION_PLAN_SCHEMA.to_owned(),
            federation_id,
            campaign_id: campaign.campaign_id,
            campaign_digest: campaign.digest().unwrap(),
            operations_plan_digest: operations.digest(&campaign).unwrap(),
            federation_authority_key_id: CheckpointPowerLossFederationKeyId([61; 16]),
            epoch: 1,
            members: vec![
                CheckpointPowerLossFederationMember {
                    schema: CHECKPOINT_POWER_LOSS_FEDERATION_MEMBER_SCHEMA.to_owned(),
                    federation_id,
                    lab_id: operations.lab_manifests[0].lab_id,
                    lab_evidence_key_id: CheckpointPowerLossLabEvidenceKeyId([62; 16]),
                    operations_authority_key_id: operations_key,
                    organization_binding: [63; 32],
                    administration_binding: [64; 32],
                    valid_from_unix_seconds: 100,
                    valid_until_unix_seconds: 1_000,
                },
                CheckpointPowerLossFederationMember {
                    schema: CHECKPOINT_POWER_LOSS_FEDERATION_MEMBER_SCHEMA.to_owned(),
                    federation_id,
                    lab_id: operations.lab_manifests[1].lab_id,
                    lab_evidence_key_id: CheckpointPowerLossLabEvidenceKeyId([72; 16]),
                    operations_authority_key_id: operations_key,
                    organization_binding: [73; 32],
                    administration_binding: [74; 32],
                    valid_from_unix_seconds: 100,
                    valid_until_unix_seconds: 1_000,
                },
            ],
            minimum_member_labs: 2,
            maximum_clock_offset_seconds: 30,
            maximum_clock_uncertainty_seconds: 5,
            valid_from_unix_seconds: 100,
            valid_until_unix_seconds: 1_000,
        };
        (campaign, operations, federation)
    }

    fn allocation(
        campaign: &CheckpointPowerLossCampaignPlan,
        operations: &CheckpointPowerLossOperationsPlan,
        federation: &CheckpointPowerLossFederationPlan,
    ) -> CheckpointPowerLossFederatedTrialAllocation {
        CheckpointPowerLossFederatedTrialAllocation {
            schema: CHECKPOINT_POWER_LOSS_FEDERATION_ALLOCATION_SCHEMA.to_owned(),
            federation_id: federation.federation_id,
            federation_plan_digest: federation.digest(campaign, operations).unwrap(),
            campaign_id: campaign.campaign_id,
            campaign_digest: campaign.digest().unwrap(),
            epoch: federation.epoch,
            allocation_id: [80; 16],
            allocation_sequence: 1,
            trial_id: campaign.trials[0].trial_id,
            storage_profile_digest: campaign.trials[0].storage_profile_digest,
            lab_id: federation.members[0].lab_id,
            lab_evidence_key_id: federation.members[0].lab_evidence_key_id,
            attempt: 1,
            issued_at_unix_seconds: 200,
            not_before_unix_seconds: 210,
            expires_at_unix_seconds: 500,
        }
    }

    #[test]
    fn federation_plan_round_trips_under_authority() {
        let (campaign, operations, federation) = fixture();
        federation.validate_against(&campaign, &operations).unwrap();
        let authority = CheckpointPowerLossFederationAuthority::new(
            CheckpointPowerLossFederationKey::new(
                federation.federation_authority_key_id,
                [90; 32],
            )
            .unwrap(),
        );
        let sealed = authority
            .seal_plan(&campaign, &operations, &federation)
            .unwrap();
        assert_eq!(
            authority
                .open_plan(&campaign, &operations, &sealed)
                .unwrap(),
            federation
        );
    }

    #[test]
    fn duplicate_administration_binding_is_rejected() {
        let (campaign, operations, mut federation) = fixture();
        federation.members[1].administration_binding =
            federation.members[0].administration_binding;
        assert!(matches!(
            federation.validate_against(&campaign, &operations),
            Err(CheckpointPowerLossFederationError::DuplicateMember)
        ));
    }

    #[test]
    fn all_evidence_revocation_rejects_prior_allocation() {
        let (campaign, operations, federation) = fixture();
        let allocation = allocation(&campaign, &operations, &federation);
        allocation
            .validate_against(&campaign, &operations, &federation)
            .unwrap();
        let revocations = CheckpointPowerLossFederationRevocationList {
            schema: CHECKPOINT_POWER_LOSS_FEDERATION_REVOCATION_SCHEMA.to_owned(),
            federation_id: federation.federation_id,
            federation_plan_digest: federation.digest(&campaign, &operations).unwrap(),
            federation_authority_key_id: federation.federation_authority_key_id,
            epoch: federation.epoch,
            sequence: 1,
            issued_at_unix_seconds: 400,
            revocations: vec![CheckpointPowerLossFederationRevocation {
                lab_id: allocation.lab_id,
                lab_evidence_key_id: allocation.lab_evidence_key_id,
                effective_at_unix_seconds: 350,
                scope: CheckpointPowerLossFederationRevocationScope::AllEvidence,
                reason_digest: [91; 32],
            }],
        };
        revocations
            .validate_against(&campaign, &operations, &federation)
            .unwrap();
        assert!(revocations.rejects(&allocation, 300));
    }

    #[test]
    fn excessive_clock_offset_is_rejected() {
        let (campaign, operations, federation) = fixture();
        let clock = CheckpointPowerLossLabClockAttestation {
            schema: CHECKPOINT_POWER_LOSS_FEDERATION_CLOCK_SCHEMA.to_owned(),
            federation_id: federation.federation_id,
            federation_plan_digest: federation.digest(&campaign, &operations).unwrap(),
            lab_id: federation.members[0].lab_id,
            lab_evidence_key_id: federation.members[0].lab_evidence_key_id,
            sample_id: [92; 16],
            first_allocation_sequence: 1,
            last_allocation_sequence: 1,
            lab_unix_seconds: 500,
            federation_unix_seconds: 600,
            uncertainty_seconds: 2,
            observed_at_unix_seconds: 600,
        };
        assert!(matches!(
            clock.validate_against(&campaign, &operations, &federation),
            Err(CheckpointPowerLossFederationError::InvalidClockAttestation)
        ));
    }
}
