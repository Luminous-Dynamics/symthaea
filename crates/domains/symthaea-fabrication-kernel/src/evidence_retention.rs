// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic evidence-retention policy with incident-bound legal holds.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::threshold::VerifiedThresholdCeremony;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const EVIDENCE_RETENTION_POLICY_SCHEMA: &str =
    "symthaea.fabrication.evidence-retention-policy.v1";
pub const EVIDENCE_DESCRIPTOR_SCHEMA: &str = "symthaea.fabrication.evidence-descriptor.v1";
pub const EVIDENCE_LEGAL_HOLD_SCHEMA: &str = "symthaea.fabrication.evidence-legal-hold.v1";
pub const MAX_RETENTION_RULES: usize = 64;
pub const MAX_LEGAL_HOLDS: usize = 4_096;
pub const MAX_EVIDENCE_ID_BYTES: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvidenceClass {
    SafetyCritical,
    Audit,
    Governance,
    MachineTelemetry,
    BuildArtifact,
    Diagnostic,
    Temporary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRetentionRule {
    pub class: EvidenceClass,
    pub minimum_hot_duration_s: u64,
    pub minimum_total_retention_s: u64,
    pub compaction_permitted: bool,
    pub deletion_permitted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRetentionPolicy {
    pub schema_version: String,
    pub sequence: u64,
    pub effective_at_unix_s: u64,
    pub rules: Vec<EvidenceRetentionRule>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceDescriptor {
    pub schema_version: String,
    pub evidence_id: String,
    pub class: EvidenceClass,
    pub evidence_digest: Sha256Digest,
    pub created_at_unix_s: u64,
    pub last_referenced_at_unix_s: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLegalHold {
    pub schema_version: String,
    pub hold_id: String,
    pub incident_digest: Sha256Digest,
    pub classes: BTreeSet<EvidenceClass>,
    pub evidence_ids: BTreeSet<String>,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: Option<u64>,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceRetentionAction {
    RetainHot,
    Compact,
    Archive,
    Delete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRetentionDecision {
    pub evidence_digest: Sha256Digest,
    pub policy_digest: Sha256Digest,
    pub evaluated_at_unix_s: u64,
    pub action: EvidenceRetentionAction,
    pub active_hold_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceRetentionError {
    UnsupportedSchema,
    SequenceZero,
    InvalidPolicy,
    DuplicateRule,
    MissingRule(EvidenceClass),
    InvalidIdentifier,
    ZeroDigest(&'static str),
    InvalidTime,
    InvalidDescriptor,
    InvalidHold,
    TooManyHolds,
    DuplicateHold,
    PolicyNotEffective,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedEvidenceRetentionPolicy {
    policy: EvidenceRetentionPolicy,
    policy_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedEvidenceRetentionPolicy {
    pub fn policy(&self) -> &EvidenceRetentionPolicy {
        &self.policy
    }
    pub fn policy_digest(&self) -> Sha256Digest {
        self.policy_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }
}

impl EvidenceRetentionPolicy {
    pub fn validate(&self) -> Result<(), EvidenceRetentionError> {
        if self.schema_version != EVIDENCE_RETENTION_POLICY_SCHEMA {
            return Err(EvidenceRetentionError::UnsupportedSchema);
        }
        if self.sequence == 0 || self.rules.is_empty() || self.rules.len() > MAX_RETENTION_RULES {
            return Err(EvidenceRetentionError::InvalidPolicy);
        }
        let mut seen = BTreeSet::new();
        for rule in &self.rules {
            if !seen.insert(rule.class) {
                return Err(EvidenceRetentionError::DuplicateRule);
            }
            if rule.minimum_total_retention_s < rule.minimum_hot_duration_s
                || (!rule.deletion_permitted && rule.minimum_total_retention_s == 0)
            {
                return Err(EvidenceRetentionError::InvalidPolicy);
            }
        }
        for class in all_classes() {
            if !seen.contains(&class) {
                return Err(EvidenceRetentionError::MissingRule(class));
            }
        }
        Ok(())
    }

    pub fn rule(&self, class: EvidenceClass) -> Option<&EvidenceRetentionRule> {
        self.rules.iter().find(|rule| rule.class == class)
    }
}

impl EvidenceDescriptor {
    pub fn validate(&self) -> Result<(), EvidenceRetentionError> {
        if self.schema_version != EVIDENCE_DESCRIPTOR_SCHEMA {
            return Err(EvidenceRetentionError::UnsupportedSchema);
        }
        validate_id(&self.evidence_id)?;
        if self.evidence_digest.0 == [0; 32] {
            return Err(EvidenceRetentionError::ZeroDigest("evidence_digest"));
        }
        if self.created_at_unix_s > self.last_referenced_at_unix_s || self.byte_len == 0 {
            return Err(EvidenceRetentionError::InvalidDescriptor);
        }
        Ok(())
    }
}

impl EvidenceLegalHold {
    pub fn validate(&self) -> Result<(), EvidenceRetentionError> {
        if self.schema_version != EVIDENCE_LEGAL_HOLD_SCHEMA {
            return Err(EvidenceRetentionError::UnsupportedSchema);
        }
        validate_id(&self.hold_id)?;
        if self.incident_digest.0 == [0; 32] {
            return Err(EvidenceRetentionError::ZeroDigest("incident_digest"));
        }
        if self.classes.is_empty() && self.evidence_ids.is_empty() {
            return Err(EvidenceRetentionError::InvalidHold);
        }
        for id in &self.evidence_ids {
            validate_id(id)?;
        }
        if self
            .expires_at_unix_s
            .is_some_and(|expires| expires <= self.issued_at_unix_s)
            || self.reason.trim().is_empty()
            || self.reason != self.reason.trim()
            || self.reason.len() > 4 * 1024
            || self.reason.chars().any(char::is_control)
        {
            return Err(EvidenceRetentionError::InvalidHold);
        }
        Ok(())
    }

    pub fn active_for(&self, descriptor: &EvidenceDescriptor, unix_s: u64) -> bool {
        unix_s >= self.issued_at_unix_s
            && self
                .expires_at_unix_s
                .is_none_or(|expires| unix_s < expires)
            && (self.classes.contains(&descriptor.class)
                || self.evidence_ids.contains(&descriptor.evidence_id))
    }
}

pub fn digest_evidence_retention_policy(
    policy: &EvidenceRetentionPolicy,
) -> Result<Sha256Digest, EvidenceRetentionError> {
    policy.validate()?;
    let mut canonical = policy.clone();
    canonical.rules.sort_by_key(|rule| rule.class);
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| EvidenceRetentionError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.evidence-retention-policy-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_evidence_retention_policy(
    policy: EvidenceRetentionPolicy,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedEvidenceRetentionPolicy, EvidenceRetentionError> {
    let policy_digest = digest_evidence_retention_policy(&policy)?;
    if ceremony.purpose() != "evidence-retention-policy" {
        return Err(EvidenceRetentionError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != policy_digest {
        return Err(EvidenceRetentionError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedEvidenceRetentionPolicy {
        policy,
        policy_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

pub fn evaluate_evidence_retention(
    descriptor: &EvidenceDescriptor,
    policy: &AuthorizedEvidenceRetentionPolicy,
    holds: &[EvidenceLegalHold],
    evaluated_at_unix_s: u64,
) -> Result<EvidenceRetentionDecision, EvidenceRetentionError> {
    descriptor.validate()?;
    if holds.len() > MAX_LEGAL_HOLDS {
        return Err(EvidenceRetentionError::TooManyHolds);
    }
    if evaluated_at_unix_s < policy.policy.effective_at_unix_s
        || evaluated_at_unix_s < descriptor.created_at_unix_s
    {
        return Err(EvidenceRetentionError::PolicyNotEffective);
    }
    let mut active_hold_ids = Vec::new();
    let mut hold_map = BTreeMap::new();
    for hold in holds {
        hold.validate()?;
        if hold_map.insert(hold.hold_id.clone(), hold).is_some() {
            return Err(EvidenceRetentionError::DuplicateHold);
        }
        if hold.active_for(descriptor, evaluated_at_unix_s) {
            active_hold_ids.push(hold.hold_id.clone());
        }
    }
    active_hold_ids.sort();
    let rule = policy
        .policy
        .rule(descriptor.class)
        .ok_or(EvidenceRetentionError::MissingRule(descriptor.class))?;
    let age = evaluated_at_unix_s.saturating_sub(descriptor.created_at_unix_s);
    let since_reference = evaluated_at_unix_s.saturating_sub(descriptor.last_referenced_at_unix_s);
    let action = if !active_hold_ids.is_empty() || age < rule.minimum_hot_duration_s {
        EvidenceRetentionAction::RetainHot
    } else if age < rule.minimum_total_retention_s {
        if rule.compaction_permitted {
            EvidenceRetentionAction::Compact
        } else {
            EvidenceRetentionAction::Archive
        }
    } else if rule.deletion_permitted && since_reference >= rule.minimum_total_retention_s {
        EvidenceRetentionAction::Delete
    } else {
        EvidenceRetentionAction::Archive
    };
    Ok(EvidenceRetentionDecision {
        evidence_digest: descriptor.evidence_digest,
        policy_digest: policy.policy_digest,
        evaluated_at_unix_s,
        action,
        active_hold_ids,
    })
}

pub fn digest_evidence_retention_decision(
    decision: &EvidenceRetentionDecision,
) -> Result<Sha256Digest, EvidenceRetentionError> {
    if decision.evidence_digest.0 == [0; 32]
        || decision.policy_digest.0 == [0; 32]
        || decision
            .active_hold_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(EvidenceRetentionError::InvalidDescriptor);
    }
    let bytes = serde_json::to_vec(decision)
        .map_err(|error| EvidenceRetentionError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.evidence-retention-decision-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn all_classes() -> [EvidenceClass; 7] {
    [
        EvidenceClass::SafetyCritical,
        EvidenceClass::Audit,
        EvidenceClass::Governance,
        EvidenceClass::MachineTelemetry,
        EvidenceClass::BuildArtifact,
        EvidenceClass::Diagnostic,
        EvidenceClass::Temporary,
    ]
}

fn validate_id(value: &str) -> Result<(), EvidenceRetentionError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_EVIDENCE_ID_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(EvidenceRetentionError::InvalidIdentifier);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    fn policy() -> EvidenceRetentionPolicy {
        EvidenceRetentionPolicy {
            schema_version: EVIDENCE_RETENTION_POLICY_SCHEMA.into(),
            sequence: 1,
            effective_at_unix_s: 10,
            rules: all_classes()
                .into_iter()
                .map(|class| EvidenceRetentionRule {
                    class,
                    minimum_hot_duration_s: 10,
                    minimum_total_retention_s: 100,
                    compaction_permitted: true,
                    deletion_permitted: class == EvidenceClass::Temporary,
                })
                .collect(),
        }
    }

    #[test]
    fn rule_order_does_not_change_policy_digest() {
        let first = policy();
        let mut second = policy();
        second.rules.reverse();
        assert_eq!(
            digest_evidence_retention_policy(&first).unwrap(),
            digest_evidence_retention_policy(&second).unwrap()
        );
    }

    #[test]
    fn descriptor_rejects_time_regression() {
        let descriptor = EvidenceDescriptor {
            schema_version: EVIDENCE_DESCRIPTOR_SCHEMA.into(),
            evidence_id: "evidence-a".into(),
            class: EvidenceClass::Audit,
            evidence_digest: sha256(b"evidence"),
            created_at_unix_s: 20,
            last_referenced_at_unix_s: 19,
            byte_len: 1,
        };
        assert_eq!(
            descriptor.validate(),
            Err(EvidenceRetentionError::InvalidDescriptor)
        );
    }
}
