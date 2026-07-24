// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Operational governance for flight evidence and exported mission data.
//!
//! This module keeps retention, export, redaction, encryption, authenticity,
//! personal-data consent, and legal-hold decisions machine-evaluable. It does
//! not encode jurisdiction-specific law; deployments must supply approved
//! policy and legal authority.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FlightDataClass {
    Public,
    Operational,
    SafetyCritical,
    Personal,
    Restricted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FlightDataDestination {
    LocalArchive,
    Operator,
    Maintainer,
    Research,
    PublicRelease,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightDataRetentionRule {
    pub class: FlightDataClass,
    pub maximum_age_days: u32,
    pub encryption_required: bool,
    pub authenticity_required: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightDataExportRule {
    pub class: FlightDataClass,
    pub allowed_destinations: Vec<FlightDataDestination>,
    pub encryption_required: bool,
    pub authenticity_required: bool,
    pub consent_required: bool,
    pub remove_personal_identifiers: bool,
    pub location_quantization_m: Option<f64>,
    pub remove_mission_id: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightDataGovernancePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub retention_rules: Vec<FlightDataRetentionRule>,
    pub export_rules: Vec<FlightDataExportRule>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightDataRecordDescriptor {
    pub record_id: String,
    pub class: FlightDataClass,
    pub age_days: u32,
    pub contains_personal_identifiers: bool,
    pub contains_precise_location: bool,
    pub contains_mission_id: bool,
    pub encrypted: bool,
    pub authenticity_reference: Option<String>,
    pub consent_reference: Option<String>,
    pub legal_hold_reference: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlightDataRetentionAction {
    Retain,
    Delete,
    LegalHold,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightDataRetentionDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub record_id: String,
    pub action: FlightDataRetentionAction,
    pub reasons: Vec<FlightDataGovernanceReason>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FlightDataRedaction {
    RemovePersonalIdentifiers,
    QuantizeLocation { grid_m: f64 },
    RemoveMissionId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlightDataExportStatus {
    Allowed,
    AllowedWithRedaction,
    Denied,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FlightDataGovernanceReason {
    DestinationNotAllowed,
    EncryptionMissing,
    AuthenticityMissing,
    ConsentMissing,
    RetentionExpired,
    RetentionRuleMissing,
    ExportRuleMissing,
    LegalHoldActive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightDataExportDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub record_id: String,
    pub destination: FlightDataDestination,
    pub status: FlightDataExportStatus,
    pub redactions: Vec<FlightDataRedaction>,
    pub reasons: Vec<FlightDataGovernanceReason>,
}

impl FlightDataExportDecision {
    pub fn canonical_json(&self) -> Result<Vec<u8>, FlightDataGovernanceError> {
        let mut canonical = self.clone();
        canonical.redactions.sort_by_key(redaction_sort_key);
        canonical.reasons.sort_by_key(reason_sort_key);
        serde_json::to_vec(&canonical).map_err(|_| FlightDataGovernanceError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, FlightDataGovernanceError> {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in self.canonical_json()? {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlightDataGovernanceError {
    InvalidPolicy,
    DuplicateRetentionRule(FlightDataClass),
    DuplicateExportRule(FlightDataClass),
    InvalidRule(FlightDataClass),
    InvalidRecord,
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct FlightDataGovernance {
    policy: FlightDataGovernancePolicy,
    retention_rules: BTreeMap<FlightDataClass, FlightDataRetentionRule>,
    export_rules: BTreeMap<FlightDataClass, FlightDataExportRule>,
}

impl FlightDataGovernance {
    pub fn new(policy: FlightDataGovernancePolicy) -> Result<Self, FlightDataGovernanceError> {
        if policy.schema_version.trim().is_empty() || policy.policy_id.trim().is_empty() {
            return Err(FlightDataGovernanceError::InvalidPolicy);
        }
        let mut retention_rules = BTreeMap::new();
        for rule in &policy.retention_rules {
            if retention_rules.insert(rule.class, rule.clone()).is_some() {
                return Err(FlightDataGovernanceError::DuplicateRetentionRule(
                    rule.class,
                ));
            }
        }
        let mut export_rules = BTreeMap::new();
        for rule in &policy.export_rules {
            let destinations: BTreeSet<_> = rule.allowed_destinations.iter().copied().collect();
            if destinations.len() != rule.allowed_destinations.len()
                || rule
                    .location_quantization_m
                    .is_some_and(|value| !value.is_finite() || value <= 0.0)
            {
                return Err(FlightDataGovernanceError::InvalidRule(rule.class));
            }
            if export_rules.insert(rule.class, rule.clone()).is_some() {
                return Err(FlightDataGovernanceError::DuplicateExportRule(rule.class));
            }
        }
        if retention_rules.is_empty() || export_rules.is_empty() {
            return Err(FlightDataGovernanceError::InvalidPolicy);
        }
        Ok(Self {
            policy,
            retention_rules,
            export_rules,
        })
    }

    pub fn assess_retention(
        &self,
        record: &FlightDataRecordDescriptor,
    ) -> Result<FlightDataRetentionDecision, FlightDataGovernanceError> {
        validate_record(record)?;
        let mut reasons = Vec::new();
        let action = if record
            .legal_hold_reference
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
        {
            reasons.push(FlightDataGovernanceReason::LegalHoldActive);
            FlightDataRetentionAction::LegalHold
        } else if let Some(rule) = self.retention_rules.get(&record.class) {
            if rule.encryption_required && !record.encrypted {
                reasons.push(FlightDataGovernanceReason::EncryptionMissing);
            }
            if rule.authenticity_required
                && record
                    .authenticity_reference
                    .as_deref()
                    .is_none_or(|value| value.trim().is_empty())
            {
                reasons.push(FlightDataGovernanceReason::AuthenticityMissing);
            }
            if record.age_days > rule.maximum_age_days {
                reasons.push(FlightDataGovernanceReason::RetentionExpired);
                FlightDataRetentionAction::Delete
            } else if reasons.is_empty() {
                FlightDataRetentionAction::Retain
            } else {
                FlightDataRetentionAction::Incomplete
            }
        } else {
            reasons.push(FlightDataGovernanceReason::RetentionRuleMissing);
            FlightDataRetentionAction::Incomplete
        };
        Ok(FlightDataRetentionDecision {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            record_id: record.record_id.clone(),
            action,
            reasons,
        })
    }

    pub fn assess_export(
        &self,
        record: &FlightDataRecordDescriptor,
        destination: FlightDataDestination,
    ) -> Result<FlightDataExportDecision, FlightDataGovernanceError> {
        validate_record(record)?;
        let mut reasons = Vec::new();
        let mut redactions = Vec::new();
        let Some(rule) = self.export_rules.get(&record.class) else {
            reasons.push(FlightDataGovernanceReason::ExportRuleMissing);
            return Ok(self.export_decision(
                record,
                destination,
                FlightDataExportStatus::Incomplete,
                redactions,
                reasons,
            ));
        };

        if !rule.allowed_destinations.contains(&destination) {
            reasons.push(FlightDataGovernanceReason::DestinationNotAllowed);
        }
        if rule.encryption_required && !record.encrypted {
            reasons.push(FlightDataGovernanceReason::EncryptionMissing);
        }
        if rule.authenticity_required
            && record
                .authenticity_reference
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
        {
            reasons.push(FlightDataGovernanceReason::AuthenticityMissing);
        }
        if rule.consent_required
            && record
                .consent_reference
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
        {
            reasons.push(FlightDataGovernanceReason::ConsentMissing);
        }
        if rule.remove_personal_identifiers && record.contains_personal_identifiers {
            redactions.push(FlightDataRedaction::RemovePersonalIdentifiers);
        }
        if let Some(grid_m) = rule.location_quantization_m
            && record.contains_precise_location
        {
            redactions.push(FlightDataRedaction::QuantizeLocation { grid_m });
        }
        if rule.remove_mission_id && record.contains_mission_id {
            redactions.push(FlightDataRedaction::RemoveMissionId);
        }

        let denied = reasons.iter().any(|reason| {
            matches!(
                reason,
                FlightDataGovernanceReason::DestinationNotAllowed
                    | FlightDataGovernanceReason::EncryptionMissing
                    | FlightDataGovernanceReason::AuthenticityMissing
                    | FlightDataGovernanceReason::ConsentMissing
            )
        });
        let status = if denied {
            FlightDataExportStatus::Denied
        } else if redactions.is_empty() {
            FlightDataExportStatus::Allowed
        } else {
            FlightDataExportStatus::AllowedWithRedaction
        };
        Ok(self.export_decision(record, destination, status, redactions, reasons))
    }

    fn export_decision(
        &self,
        record: &FlightDataRecordDescriptor,
        destination: FlightDataDestination,
        status: FlightDataExportStatus,
        mut redactions: Vec<FlightDataRedaction>,
        mut reasons: Vec<FlightDataGovernanceReason>,
    ) -> FlightDataExportDecision {
        redactions.sort_by_key(redaction_sort_key);
        reasons.sort_by_key(reason_sort_key);
        FlightDataExportDecision {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            record_id: record.record_id.clone(),
            destination,
            status,
            redactions,
            reasons,
        }
    }
}

fn validate_record(record: &FlightDataRecordDescriptor) -> Result<(), FlightDataGovernanceError> {
    if record.record_id.trim().is_empty() {
        return Err(FlightDataGovernanceError::InvalidRecord);
    }
    Ok(())
}

fn redaction_sort_key(redaction: &FlightDataRedaction) -> String {
    format!("{redaction:?}")
}

fn reason_sort_key(reason: &FlightDataGovernanceReason) -> String {
    format!("{reason:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn governance() -> FlightDataGovernance {
        FlightDataGovernance::new(FlightDataGovernancePolicy {
            schema_version: "symthaea.helicopter.flight-data-governance.v1".into(),
            policy_id: "policy-a".into(),
            retention_rules: vec![FlightDataRetentionRule {
                class: FlightDataClass::SafetyCritical,
                maximum_age_days: 365,
                encryption_required: true,
                authenticity_required: true,
            }],
            export_rules: vec![FlightDataExportRule {
                class: FlightDataClass::SafetyCritical,
                allowed_destinations: vec![
                    FlightDataDestination::Operator,
                    FlightDataDestination::Research,
                ],
                encryption_required: true,
                authenticity_required: true,
                consent_required: false,
                remove_personal_identifiers: true,
                location_quantization_m: Some(1_000.0),
                remove_mission_id: true,
            }],
        })
        .unwrap()
    }

    fn record() -> FlightDataRecordDescriptor {
        FlightDataRecordDescriptor {
            record_id: "record-a".into(),
            class: FlightDataClass::SafetyCritical,
            age_days: 30,
            contains_personal_identifiers: true,
            contains_precise_location: true,
            contains_mission_id: true,
            encrypted: true,
            authenticity_reference: Some("signature:a".into()),
            consent_reference: None,
            legal_hold_reference: None,
        }
    }

    #[test]
    fn research_export_requires_redaction() {
        let decision = governance()
            .assess_export(&record(), FlightDataDestination::Research)
            .unwrap();
        assert_eq!(
            decision.status,
            FlightDataExportStatus::AllowedWithRedaction
        );
        assert_eq!(decision.redactions.len(), 3);
    }

    #[test]
    fn public_export_is_denied() {
        let decision = governance()
            .assess_export(&record(), FlightDataDestination::PublicRelease)
            .unwrap();
        assert_eq!(decision.status, FlightDataExportStatus::Denied);
    }

    #[test]
    fn authenticity_is_fail_closed() {
        let mut unsigned = record();
        unsigned.authenticity_reference = None;
        let decision = governance()
            .assess_export(&unsigned, FlightDataDestination::Operator)
            .unwrap();
        assert_eq!(decision.status, FlightDataExportStatus::Denied);
    }

    #[test]
    fn expired_record_is_deleted_without_hold() {
        let mut expired = record();
        expired.age_days = 400;
        let decision = governance().assess_retention(&expired).unwrap();
        assert_eq!(decision.action, FlightDataRetentionAction::Delete);
    }

    #[test]
    fn legal_hold_overrides_expiry() {
        let mut held = record();
        held.age_days = 400;
        held.legal_hold_reference = Some("hold:case-1".into());
        let decision = governance().assess_retention(&held).unwrap();
        assert_eq!(decision.action, FlightDataRetentionAction::LegalHold);
    }
}
