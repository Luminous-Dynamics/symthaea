// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable gateway quarantine and decommission state tracking.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_decommission::AuthorizedGatewayDecommission;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const GATEWAY_DECOMMISSION_TRACKER_SCHEMA: &str =
    "symthaea.fabrication.gateway-decommission-tracker.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GatewayRetirementStage {
    Quarantined,
    EraseVerified,
    Decommissioned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayRetirementRecord {
    pub gateway_id: String,
    pub plan_digest: Sha256Digest,
    pub stage: GatewayRetirementStage,
    pub updated_at_unix_s: u64,
    pub erase_verification_digest: Option<Sha256Digest>,
    pub final_record_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayDecommissionTracker {
    pub schema_version: String,
    records: BTreeMap<String, GatewayRetirementRecord>,
}

impl Default for GatewayDecommissionTracker {
    fn default() -> Self {
        Self {
            schema_version: GATEWAY_DECOMMISSION_TRACKER_SCHEMA.into(),
            records: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayDecommissionTrackingError {
    UnsupportedSchema,
    AlreadyTracked,
    UnknownGateway,
    PlanMismatch,
    TimeRegressed,
    InvalidTransition,
    EmptyEraseEvidence,
    TooEarly,
    InvalidRecord,
    RecordDigestMismatch,
    Encoding(String),
}

impl GatewayDecommissionTracker {
    pub fn quarantine(
        &mut self,
        authorization: &AuthorizedGatewayDecommission,
        recorded_at_unix_s: u64,
    ) -> Result<Sha256Digest, GatewayDecommissionTrackingError> {
        self.validate()?;
        let gateway_id = authorization.plan().gateway_id.clone();
        if self.records.contains_key(&gateway_id) {
            return Err(GatewayDecommissionTrackingError::AlreadyTracked);
        }
        if recorded_at_unix_s < authorization.plan().quarantined_at_unix_s {
            return Err(GatewayDecommissionTrackingError::TooEarly);
        }
        let digest = record_digest(
            &gateway_id,
            authorization.plan_digest(),
            GatewayRetirementStage::Quarantined,
            recorded_at_unix_s,
            None,
        )?;
        self.records.insert(
            gateway_id.clone(),
            GatewayRetirementRecord {
                gateway_id,
                plan_digest: authorization.plan_digest(),
                stage: GatewayRetirementStage::Quarantined,
                updated_at_unix_s: recorded_at_unix_s,
                erase_verification_digest: None,
                final_record_digest: digest,
            },
        );
        Ok(digest)
    }

    pub fn record_erase_verification(
        &mut self,
        authorization: &AuthorizedGatewayDecommission,
        erase_verification_digest: Sha256Digest,
        recorded_at_unix_s: u64,
    ) -> Result<Sha256Digest, GatewayDecommissionTrackingError> {
        self.validate()?;
        if erase_verification_digest == Sha256Digest([0; 32]) {
            return Err(GatewayDecommissionTrackingError::EmptyEraseEvidence);
        }
        let gateway_id = &authorization.plan().gateway_id;
        let Some(current) = self.records.get(gateway_id).cloned() else {
            return Err(GatewayDecommissionTrackingError::UnknownGateway);
        };
        if current.plan_digest != authorization.plan_digest() {
            return Err(GatewayDecommissionTrackingError::PlanMismatch);
        }
        if current.stage != GatewayRetirementStage::Quarantined {
            return Err(GatewayDecommissionTrackingError::InvalidTransition);
        }
        if recorded_at_unix_s < current.updated_at_unix_s {
            return Err(GatewayDecommissionTrackingError::TimeRegressed);
        }
        let digest = record_digest(
            gateway_id,
            current.plan_digest,
            GatewayRetirementStage::EraseVerified,
            recorded_at_unix_s,
            Some(erase_verification_digest),
        )?;
        self.records.insert(
            gateway_id.clone(),
            GatewayRetirementRecord {
                gateway_id: gateway_id.clone(),
                plan_digest: current.plan_digest,
                stage: GatewayRetirementStage::EraseVerified,
                updated_at_unix_s: recorded_at_unix_s,
                erase_verification_digest: Some(erase_verification_digest),
                final_record_digest: digest,
            },
        );
        Ok(digest)
    }

    pub fn finalize(
        &mut self,
        authorization: &AuthorizedGatewayDecommission,
        recorded_at_unix_s: u64,
    ) -> Result<Sha256Digest, GatewayDecommissionTrackingError> {
        self.validate()?;
        let gateway_id = &authorization.plan().gateway_id;
        let Some(current) = self.records.get(gateway_id).cloned() else {
            return Err(GatewayDecommissionTrackingError::UnknownGateway);
        };
        if current.plan_digest != authorization.plan_digest() {
            return Err(GatewayDecommissionTrackingError::PlanMismatch);
        }
        if current.stage != GatewayRetirementStage::EraseVerified {
            return Err(GatewayDecommissionTrackingError::InvalidTransition);
        }
        if recorded_at_unix_s < authorization.plan().decommission_at_unix_s {
            return Err(GatewayDecommissionTrackingError::TooEarly);
        }
        let erase = current.erase_verification_digest;
        let digest = record_digest(
            gateway_id,
            current.plan_digest,
            GatewayRetirementStage::Decommissioned,
            recorded_at_unix_s,
            erase,
        )?;
        self.records.insert(
            gateway_id.clone(),
            GatewayRetirementRecord {
                gateway_id: gateway_id.clone(),
                plan_digest: current.plan_digest,
                stage: GatewayRetirementStage::Decommissioned,
                updated_at_unix_s: recorded_at_unix_s,
                erase_verification_digest: erase,
                final_record_digest: digest,
            },
        );
        Ok(digest)
    }

    pub fn permits_authority(&self, gateway_id: &str) -> bool {
        !self.records.contains_key(gateway_id)
    }

    pub fn record(&self, gateway_id: &str) -> Option<&GatewayRetirementRecord> {
        self.records.get(gateway_id)
    }

    pub fn verify_successor_of(
        &self,
        previous: &Self,
    ) -> Result<(), GatewayDecommissionTrackingError> {
        self.validate()?;
        previous.validate()?;
        for (gateway_id, old) in &previous.records {
            let Some(new) = self.records.get(gateway_id) else {
                return Err(GatewayDecommissionTrackingError::InvalidTransition);
            };
            if new.plan_digest != old.plan_digest
                || new.updated_at_unix_s < old.updated_at_unix_s
                || stage_rank(new.stage) < stage_rank(old.stage)
            {
                return Err(GatewayDecommissionTrackingError::InvalidTransition);
            }
            if old.erase_verification_digest.is_some()
                && new.erase_verification_digest != old.erase_verification_digest
            {
                return Err(GatewayDecommissionTrackingError::InvalidTransition);
            }
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), GatewayDecommissionTrackingError> {
        if self.schema_version != GATEWAY_DECOMMISSION_TRACKER_SCHEMA {
            return Err(GatewayDecommissionTrackingError::UnsupportedSchema);
        }
        for (gateway_id, record) in &self.records {
            if gateway_id != &record.gateway_id
                || gateway_id.trim().is_empty()
                || gateway_id != gateway_id.trim()
                || gateway_id.len() > 256
                || record.plan_digest == Sha256Digest([0; 32])
                || record.updated_at_unix_s == 0
                || matches!(record.stage, GatewayRetirementStage::Quarantined)
                    && record.erase_verification_digest.is_some()
                || matches!(
                    record.stage,
                    GatewayRetirementStage::EraseVerified | GatewayRetirementStage::Decommissioned
                ) && record.erase_verification_digest.is_none()
            {
                return Err(GatewayDecommissionTrackingError::InvalidRecord);
            }
            let expected = record_digest(
                gateway_id,
                record.plan_digest,
                record.stage,
                record.updated_at_unix_s,
                record.erase_verification_digest,
            )?;
            if expected != record.final_record_digest {
                return Err(GatewayDecommissionTrackingError::RecordDigestMismatch);
            }
        }
        Ok(())
    }
}

pub fn digest_gateway_decommission_tracker(
    tracker: &GatewayDecommissionTracker,
) -> Result<Sha256Digest, GatewayDecommissionTrackingError> {
    tracker.validate()?;
    let bytes = serde_json::to_vec(tracker)
        .map_err(|error| GatewayDecommissionTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.gateway-decommission-tracker-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn stage_rank(stage: GatewayRetirementStage) -> u8 {
    match stage {
        GatewayRetirementStage::Quarantined => 0,
        GatewayRetirementStage::EraseVerified => 1,
        GatewayRetirementStage::Decommissioned => 2,
    }
}

fn record_digest(
    gateway_id: &str,
    plan_digest: Sha256Digest,
    stage: GatewayRetirementStage,
    updated_at_unix_s: u64,
    erase_verification_digest: Option<Sha256Digest>,
) -> Result<Sha256Digest, GatewayDecommissionTrackingError> {
    let bytes = serde_json::to_vec(&(
        gateway_id,
        plan_digest,
        stage,
        updated_at_unix_s,
        erase_verification_digest,
    ))
    .map_err(|error| GatewayDecommissionTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.gateway-decommission-record.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
