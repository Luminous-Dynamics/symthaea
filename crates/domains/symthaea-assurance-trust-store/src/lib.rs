// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hardware-agnostic monotonic trust-store assurance for policy-lineage anchors.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_assurance_policy_lineage_anchor::PolicyLineageAnchor;

const CHECKPOINT_DIGEST_SCHEMA: &[u8] = b"symthaea-assurance-trust-store-checkpoint-v1\0";
const BACKUP_DIGEST_SCHEMA: &[u8] = b"symthaea-assurance-trust-store-backup-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustStoreBackendKind {
    HardwareMonotonicCounter,
    SecureElement,
    HardwareSecurityModule,
    ExternalWitnessedLog,
    OtherReviewed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustStoreProfile {
    pub schema_version: String,
    pub store_id: String,
    pub trust_store_ref: String,
    pub backend_kind: TrustStoreBackendKind,
    pub hardware_instance_ref: Option<String>,
    pub monotonic_counter_ref: String,
    pub initial_counter_epoch: String,
    pub minimum_independent_recovery_approvals: usize,
    pub evidence_refs: Vec<String>,
}

impl TrustStoreProfile {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.store_id.trim().is_empty()
            && !self.trust_store_ref.trim().is_empty()
            && !self.monotonic_counter_ref.trim().is_empty()
            && !self.initial_counter_epoch.trim().is_empty()
            && self.minimum_independent_recovery_approvals >= 2
            && self
                .hardware_instance_ref
                .as_ref()
                .is_none_or(|value| !value.trim().is_empty())
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustStoreCheckpoint {
    pub schema_version: String,
    pub checkpoint_id: String,
    pub store_id: String,
    pub store_revision: u64,
    pub counter_epoch: String,
    pub counter_value: u64,
    pub anchor_revision: u64,
    pub anchor_digest: String,
    pub policy_tip_revision: u64,
    pub predecessor_checkpoint_digest: Option<String>,
    pub recorded_at_ms: u64,
    pub attestation_ref: String,
    pub independent_verification_ref: String,
    pub evidence_refs: Vec<String>,
}

impl TrustStoreCheckpoint {
    pub fn validate(&self, profile: &TrustStoreProfile) -> bool {
        profile.validate()
            && !self.schema_version.trim().is_empty()
            && !self.checkpoint_id.trim().is_empty()
            && self.store_id == profile.store_id
            && self.store_revision > 0
            && !self.counter_epoch.trim().is_empty()
            && self.counter_value > 0
            && self.anchor_revision > 0
            && valid_digest(&self.anchor_digest)
            && self.policy_tip_revision > 0
            && !self.attestation_ref.trim().is_empty()
            && !self.independent_verification_ref.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
            && match self.store_revision {
                1 => self.predecessor_checkpoint_digest.is_none(),
                _ => self
                    .predecessor_checkpoint_digest
                    .as_ref()
                    .is_some_and(|value| valid_digest(value)),
            }
    }

    pub fn checkpoint_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(CHECKPOINT_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.checkpoint_id);
        push_field(&mut hasher, &self.store_id);
        push_field(&mut hasher, &self.store_revision.to_string());
        push_field(&mut hasher, &self.counter_epoch);
        push_field(&mut hasher, &self.counter_value.to_string());
        push_field(&mut hasher, &self.anchor_revision.to_string());
        push_field(&mut hasher, &self.anchor_digest);
        push_field(&mut hasher, &self.policy_tip_revision.to_string());
        push_optional(&mut hasher, self.predecessor_checkpoint_digest.as_deref());
        push_field(&mut hasher, &self.recorded_at_ms.to_string());
        push_field(&mut hasher, &self.attestation_ref);
        push_field(&mut hasher, &self.independent_verification_ref);
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub fn binds_anchor(&self, profile: &TrustStoreProfile, anchor: &PolicyLineageAnchor) -> bool {
        self.validate(profile)
            && anchor.validate()
            && anchor.trust_store_ref == profile.trust_store_ref
            && self.anchor_revision == anchor.anchor_revision
            && self.anchor_digest == anchor.anchor_digest()
            && self.policy_tip_revision == anchor.tip_revision
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustStoreChainStatus {
    Valid,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustStoreChainIssue {
    EmptyChain,
    InvalidCheckpoint(u64),
    DuplicateStoreRevision(u64),
    DuplicateCheckpointDigest(String),
    FirstRevisionIsNotOne(u64),
    StoreRevisionGap { previous: u64, next: u64 },
    CounterEpochChangedWithoutRecovery { revision: u64 },
    CounterDidNotIncrease { revision: u64 },
    AnchorRevisionDecreased { revision: u64 },
    PolicyTipRevisionDecreased { revision: u64 },
    PredecessorDigestMismatch { revision: u64 },
    RecordingTimeRegressed { revision: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustStoreChainReport {
    pub status: TrustStoreChainStatus,
    pub tip_store_revision: Option<u64>,
    pub tip_counter_epoch: Option<String>,
    pub tip_counter_value: Option<u64>,
    pub tip_checkpoint_digest: Option<String>,
    pub tip_anchor_digest: Option<String>,
    pub tip_policy_revision: Option<u64>,
    pub issues: Vec<TrustStoreChainIssue>,
}

impl TrustStoreChainReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn assess_checkpoint_chain(
    profile: &TrustStoreProfile,
    checkpoints: &[TrustStoreCheckpoint],
) -> TrustStoreChainReport {
    if checkpoints.is_empty() {
        return TrustStoreChainReport {
            status: TrustStoreChainStatus::Invalid,
            tip_store_revision: None,
            tip_counter_epoch: None,
            tip_counter_value: None,
            tip_checkpoint_digest: None,
            tip_anchor_digest: None,
            tip_policy_revision: None,
            issues: vec![TrustStoreChainIssue::EmptyChain],
        };
    }

    let mut ordered = checkpoints.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|value| value.store_revision);
    let mut issues = Vec::new();
    let mut revisions = BTreeSet::new();
    let mut digests = BTreeSet::new();

    for checkpoint in &ordered {
        if !checkpoint.validate(profile) {
            issues.push(TrustStoreChainIssue::InvalidCheckpoint(checkpoint.store_revision));
        }
        if !revisions.insert(checkpoint.store_revision) {
            issues.push(TrustStoreChainIssue::DuplicateStoreRevision(checkpoint.store_revision));
        }
        let digest = checkpoint.checkpoint_digest();
        if !digests.insert(digest.clone()) {
            issues.push(TrustStoreChainIssue::DuplicateCheckpointDigest(digest));
        }
    }

    if ordered[0].store_revision != 1 {
        issues.push(TrustStoreChainIssue::FirstRevisionIsNotOne(ordered[0].store_revision));
    }

    for pair in ordered.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if current.store_revision != previous.store_revision.saturating_add(1) {
            issues.push(TrustStoreChainIssue::StoreRevisionGap {
                previous: previous.store_revision,
                next: current.store_revision,
            });
        }
        if current.counter_epoch != previous.counter_epoch {
            issues.push(TrustStoreChainIssue::CounterEpochChangedWithoutRecovery {
                revision: current.store_revision,
            });
        } else if current.counter_value <= previous.counter_value {
            issues.push(TrustStoreChainIssue::CounterDidNotIncrease {
                revision: current.store_revision,
            });
        }
        if current.anchor_revision < previous.anchor_revision {
            issues.push(TrustStoreChainIssue::AnchorRevisionDecreased {
                revision: current.store_revision,
            });
        }
        if current.policy_tip_revision < previous.policy_tip_revision {
            issues.push(TrustStoreChainIssue::PolicyTipRevisionDecreased {
                revision: current.store_revision,
            });
        }
        let previous_digest = previous.checkpoint_digest();
        if current.predecessor_checkpoint_digest.as_deref() != Some(previous_digest.as_str()) {
            issues.push(TrustStoreChainIssue::PredecessorDigestMismatch {
                revision: current.store_revision,
            });
        }
        if current.recorded_at_ms < previous.recorded_at_ms {
            issues.push(TrustStoreChainIssue::RecordingTimeRegressed {
                revision: current.store_revision,
            });
        }
    }

    let tip = ordered.last().expect("non-empty checkpoint chain");
    TrustStoreChainReport {
        status: if issues.is_empty() {
            TrustStoreChainStatus::Valid
        } else {
            TrustStoreChainStatus::Invalid
        },
        tip_store_revision: Some(tip.store_revision),
        tip_counter_epoch: Some(tip.counter_epoch.clone()),
        tip_counter_value: Some(tip.counter_value),
        tip_checkpoint_digest: Some(tip.checkpoint_digest()),
        tip_anchor_digest: Some(tip.anchor_digest.clone()),
        tip_policy_revision: Some(tip.policy_tip_revision),
        issues,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustStoreBackup {
    pub schema_version: String,
    pub backup_id: String,
    pub store_id: String,
    pub source_checkpoint_digest: String,
    pub counter_epoch: String,
    pub counter_value: u64,
    pub anchor_digest: String,
    pub policy_tip_revision: u64,
    pub created_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl TrustStoreBackup {
    pub fn validate(&self, profile: &TrustStoreProfile) -> bool {
        profile.validate()
            && !self.schema_version.trim().is_empty()
            && !self.backup_id.trim().is_empty()
            && self.store_id == profile.store_id
            && valid_digest(&self.source_checkpoint_digest)
            && !self.counter_epoch.trim().is_empty()
            && self.counter_value > 0
            && valid_digest(&self.anchor_digest)
            && self.policy_tip_revision > 0
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }

    pub fn from_checkpoint(
        profile: &TrustStoreProfile,
        checkpoint: &TrustStoreCheckpoint,
        backup_id: impl Into<String>,
        created_at_ms: u64,
        evidence_refs: Vec<String>,
    ) -> Option<Self> {
        if !checkpoint.validate(profile)
            || evidence_refs.is_empty()
            || created_at_ms < checkpoint.recorded_at_ms
        {
            return None;
        }
        Some(Self {
            schema_version: "1".into(),
            backup_id: backup_id.into(),
            store_id: profile.store_id.clone(),
            source_checkpoint_digest: checkpoint.checkpoint_digest(),
            counter_epoch: checkpoint.counter_epoch.clone(),
            counter_value: checkpoint.counter_value,
            anchor_digest: checkpoint.anchor_digest.clone(),
            policy_tip_revision: checkpoint.policy_tip_revision,
            created_at_ms,
            evidence_refs,
        })
    }

    pub fn backup_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(BACKUP_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.backup_id);
        push_field(&mut hasher, &self.store_id);
        push_field(&mut hasher, &self.source_checkpoint_digest);
        push_field(&mut hasher, &self.counter_epoch);
        push_field(&mut hasher, &self.counter_value.to_string());
        push_field(&mut hasher, &self.anchor_digest);
        push_field(&mut hasher, &self.policy_tip_revision.to_string());
        push_field(&mut hasher, &self.created_at_ms.to_string());
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustStoreRecoveryAuthorization {
    pub authorization_id: String,
    pub store_id: String,
    pub incident_ref: String,
    pub expected_previous_checkpoint_digest: String,
    pub expected_backup_digest: String,
    pub minimum_counter_value: u64,
    pub minimum_policy_tip_revision: u64,
    pub replacement_trust_store_ref: String,
    pub replacement_counter_epoch: String,
    pub approved_by_refs: Vec<String>,
    pub independent_verification_ref: String,
    pub authorized_at_ms: u64,
    pub expires_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl TrustStoreRecoveryAuthorization {
    pub fn validate(&self, profile: &TrustStoreProfile) -> bool {
        let distinct_approvers = self
            .approved_by_refs
            .iter()
            .filter(|value| !value.trim().is_empty())
            .collect::<BTreeSet<_>>();
        profile.validate()
            && !self.authorization_id.trim().is_empty()
            && self.store_id == profile.store_id
            && !self.incident_ref.trim().is_empty()
            && valid_digest(&self.expected_previous_checkpoint_digest)
            && valid_digest(&self.expected_backup_digest)
            && self.minimum_counter_value > 0
            && self.minimum_policy_tip_revision > 0
            && !self.replacement_trust_store_ref.trim().is_empty()
            && !self.replacement_counter_epoch.trim().is_empty()
            && self.replacement_counter_epoch != profile.initial_counter_epoch
            && distinct_approvers.len() >= profile.minimum_independent_recovery_approvals
            && !self.independent_verification_ref.trim().is_empty()
            && self.expires_at_ms >= self.authorized_at_ms
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustStoreRecoveryStatus {
    Eligible,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustStoreRecoveryIssue {
    InvalidProfile,
    InvalidCurrentTip,
    InvalidBackup,
    InvalidAuthorization,
    AuthorizationExpired,
    BackupIsNotCurrentTip,
    BackupSnapshotMismatch,
    BackupPredatesCheckpoint,
    BackupCounterBelowFloor,
    BackupPolicyRevisionBelowFloor,
    PreviousTipMismatch,
    BackupDigestMismatch,
    ReplacementStoreDidNotChange,
    ReplacementCounterEpochDidNotChange,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustStoreRecoveryReport {
    pub status: TrustStoreRecoveryStatus,
    pub previous_checkpoint_digest: String,
    pub backup_digest: String,
    pub replacement_trust_store_ref: String,
    pub replacement_counter_epoch: String,
    pub issues: Vec<TrustStoreRecoveryIssue>,
}

impl TrustStoreRecoveryReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn assess_recovery(
    profile: &TrustStoreProfile,
    current_tip: &TrustStoreCheckpoint,
    backup: &TrustStoreBackup,
    authorization: &TrustStoreRecoveryAuthorization,
    now_ms: u64,
) -> TrustStoreRecoveryReport {
    let mut issues = Vec::new();
    if !profile.validate() {
        issues.push(TrustStoreRecoveryIssue::InvalidProfile);
    }
    if !current_tip.validate(profile) {
        issues.push(TrustStoreRecoveryIssue::InvalidCurrentTip);
    }
    if !backup.validate(profile) {
        issues.push(TrustStoreRecoveryIssue::InvalidBackup);
    }
    if !authorization.validate(profile) {
        issues.push(TrustStoreRecoveryIssue::InvalidAuthorization);
    }
    if authorization.expires_at_ms < now_ms || authorization.authorized_at_ms > now_ms {
        issues.push(TrustStoreRecoveryIssue::AuthorizationExpired);
    }

    let current_digest = current_tip.checkpoint_digest();
    let backup_digest = backup.backup_digest();
    if backup.source_checkpoint_digest != current_digest {
        issues.push(TrustStoreRecoveryIssue::BackupIsNotCurrentTip);
    }
    if backup.counter_epoch != current_tip.counter_epoch
        || backup.counter_value != current_tip.counter_value
        || backup.anchor_digest != current_tip.anchor_digest
        || backup.policy_tip_revision != current_tip.policy_tip_revision
    {
        issues.push(TrustStoreRecoveryIssue::BackupSnapshotMismatch);
    }
    if backup.created_at_ms < current_tip.recorded_at_ms {
        issues.push(TrustStoreRecoveryIssue::BackupPredatesCheckpoint);
    }
    if backup.counter_value < current_tip.counter_value
        || backup.counter_value < authorization.minimum_counter_value
    {
        issues.push(TrustStoreRecoveryIssue::BackupCounterBelowFloor);
    }
    if backup.policy_tip_revision < current_tip.policy_tip_revision
        || backup.policy_tip_revision < authorization.minimum_policy_tip_revision
    {
        issues.push(TrustStoreRecoveryIssue::BackupPolicyRevisionBelowFloor);
    }
    if authorization.expected_previous_checkpoint_digest != current_digest {
        issues.push(TrustStoreRecoveryIssue::PreviousTipMismatch);
    }
    if authorization.expected_backup_digest != backup_digest {
        issues.push(TrustStoreRecoveryIssue::BackupDigestMismatch);
    }
    if authorization.replacement_trust_store_ref == profile.trust_store_ref {
        issues.push(TrustStoreRecoveryIssue::ReplacementStoreDidNotChange);
    }
    if authorization.replacement_counter_epoch == current_tip.counter_epoch {
        issues.push(TrustStoreRecoveryIssue::ReplacementCounterEpochDidNotChange);
    }

    let invalid = issues.iter().any(|issue| {
        matches!(
            issue,
            TrustStoreRecoveryIssue::InvalidProfile
                | TrustStoreRecoveryIssue::InvalidCurrentTip
                | TrustStoreRecoveryIssue::InvalidBackup
                | TrustStoreRecoveryIssue::InvalidAuthorization
        )
    });
    TrustStoreRecoveryReport {
        status: if invalid {
            TrustStoreRecoveryStatus::Invalid
        } else if issues.is_empty() {
            TrustStoreRecoveryStatus::Eligible
        } else {
            TrustStoreRecoveryStatus::Blocked
        },
        previous_checkpoint_digest: current_digest,
        backup_digest,
        replacement_trust_store_ref: authorization.replacement_trust_store_ref.clone(),
        replacement_counter_epoch: authorization.replacement_counter_epoch.clone(),
        issues,
    }
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(value.trim().as_bytes());
    hasher.update(b"\0");
}

fn push_optional(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(b"some\0");
            push_field(hasher, value);
        }
        None => hasher.update(b"none\0"),
    }
}

fn valid_digest(value: &str) -> bool {
    value
        .trim()
        .split_once(':')
        .is_some_and(|(algorithm, digest)| !algorithm.is_empty() && !digest.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> TrustStoreProfile {
        TrustStoreProfile {
            schema_version: "1".into(),
            store_id: "policy-root".into(),
            trust_store_ref: "trust-store:hardware-a".into(),
            backend_kind: TrustStoreBackendKind::HardwareMonotonicCounter,
            hardware_instance_ref: Some("device:tpm-a".into()),
            monotonic_counter_ref: "counter:nv-1".into(),
            initial_counter_epoch: "epoch:hardware-a".into(),
            minimum_independent_recovery_approvals: 2,
            evidence_refs: vec!["review:trust-store-profile".into()],
        }
    }

    fn checkpoint(revision: u64, counter: u64, previous: Option<String>) -> TrustStoreCheckpoint {
        TrustStoreCheckpoint {
            schema_version: "1".into(),
            checkpoint_id: format!("checkpoint:{revision}"),
            store_id: "policy-root".into(),
            store_revision: revision,
            counter_epoch: "epoch:hardware-a".into(),
            counter_value: counter,
            anchor_revision: revision,
            anchor_digest: format!("blake3:anchor-{revision}"),
            policy_tip_revision: revision,
            predecessor_checkpoint_digest: previous,
            recorded_at_ms: revision * 1_000,
            attestation_ref: format!("attestation:{revision}"),
            independent_verification_ref: format!("verification:{revision}"),
            evidence_refs: vec![format!("audit:{revision}")],
        }
    }

    fn chain() -> Vec<TrustStoreCheckpoint> {
        let c1 = checkpoint(1, 10, None);
        let c2 = checkpoint(2, 11, Some(c1.checkpoint_digest()));
        let c3 = checkpoint(3, 15, Some(c2.checkpoint_digest()));
        vec![c1, c2, c3]
    }

    fn backup(tip: &TrustStoreCheckpoint) -> TrustStoreBackup {
        TrustStoreBackup::from_checkpoint(
            &profile(),
            tip,
            "backup:tip",
            tip.recorded_at_ms + 1,
            vec!["backup:audit".into()],
        )
        .unwrap()
    }

    fn authorization(
        tip: &TrustStoreCheckpoint,
        backup: &TrustStoreBackup,
    ) -> TrustStoreRecoveryAuthorization {
        TrustStoreRecoveryAuthorization {
            authorization_id: "recovery:1".into(),
            store_id: "policy-root".into(),
            incident_ref: "incident:hardware-loss".into(),
            expected_previous_checkpoint_digest: tip.checkpoint_digest(),
            expected_backup_digest: backup.backup_digest(),
            minimum_counter_value: tip.counter_value,
            minimum_policy_tip_revision: tip.policy_tip_revision,
            replacement_trust_store_ref: "trust-store:hardware-b".into(),
            replacement_counter_epoch: "epoch:hardware-b".into(),
            approved_by_refs: vec!["reviewer:a".into(), "reviewer:b".into()],
            independent_verification_ref: "verification:recovery".into(),
            authorized_at_ms: 4_000,
            expires_at_ms: 8_000,
            evidence_refs: vec!["audit:recovery".into()],
        }
    }

    #[test]
    fn monotonic_checkpoint_chain_is_valid() {
        let report = assess_checkpoint_chain(&profile(), &chain());
        assert_eq!(report.status, TrustStoreChainStatus::Valid);
        assert_eq!(report.tip_counter_value, Some(15));
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn counter_regression_fails_closed() {
        let mut values = chain();
        values[2].counter_value = 11;
        let report = assess_checkpoint_chain(&profile(), &values);
        assert_eq!(report.status, TrustStoreChainStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            TrustStoreChainIssue::CounterDidNotIncrease { revision: 3 }
        )));
    }

    #[test]
    fn counter_epoch_change_requires_explicit_recovery_path() {
        let mut values = chain();
        values[2].counter_epoch = "epoch:replacement".into();
        let report = assess_checkpoint_chain(&profile(), &values);
        assert_eq!(report.status, TrustStoreChainStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            TrustStoreChainIssue::CounterEpochChangedWithoutRecovery { revision: 3 }
        )));
    }

    #[test]
    fn latest_backup_with_reviewed_recovery_is_eligible() {
        let values = chain();
        let tip = values.last().unwrap();
        let backup = backup(tip);
        let report = assess_recovery(&profile(), tip, &backup, &authorization(tip, &backup), 5_000);
        assert_eq!(report.status, TrustStoreRecoveryStatus::Eligible);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn older_backup_cannot_roll_policy_state_back() {
        let values = chain();
        let tip = values.last().unwrap();
        let old_backup = backup(&values[1]);
        let mut auth = authorization(tip, &old_backup);
        auth.expected_backup_digest = old_backup.backup_digest();
        let report = assess_recovery(&profile(), tip, &old_backup, &auth, 5_000);
        assert_eq!(report.status, TrustStoreRecoveryStatus::Blocked);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            TrustStoreRecoveryIssue::BackupIsNotCurrentTip
                | TrustStoreRecoveryIssue::BackupSnapshotMismatch
                | TrustStoreRecoveryIssue::BackupCounterBelowFloor
                | TrustStoreRecoveryIssue::BackupPolicyRevisionBelowFloor
        )));
    }

    #[test]
    fn forged_backup_snapshot_cannot_claim_current_tip() {
        let values = chain();
        let tip = values.last().unwrap();
        let mut backup = backup(tip);
        backup.anchor_digest = "blake3:other-anchor".into();
        let auth = authorization(tip, &backup);
        let report = assess_recovery(&profile(), tip, &backup, &auth, 5_000);
        assert_eq!(report.status, TrustStoreRecoveryStatus::Blocked);
        assert!(report
            .issues
            .contains(&TrustStoreRecoveryIssue::BackupSnapshotMismatch));
    }

    #[test]
    fn expired_recovery_authorization_is_blocked() {
        let values = chain();
        let tip = values.last().unwrap();
        let backup = backup(tip);
        let mut auth = authorization(tip, &backup);
        auth.expires_at_ms = 4_500;
        let report = assess_recovery(&profile(), tip, &backup, &auth, 5_000);
        assert_eq!(report.status, TrustStoreRecoveryStatus::Blocked);
        assert!(report.issues.contains(&TrustStoreRecoveryIssue::AuthorizationExpired));
    }

    #[test]
    fn replacement_store_and_counter_epoch_must_change() {
        let values = chain();
        let tip = values.last().unwrap();
        let backup = backup(tip);
        let mut auth = authorization(tip, &backup);
        auth.replacement_trust_store_ref = profile().trust_store_ref;
        auth.replacement_counter_epoch = tip.counter_epoch.clone();
        let report = assess_recovery(&profile(), tip, &backup, &auth, 5_000);
        assert_eq!(report.status, TrustStoreRecoveryStatus::Blocked);
        assert!(report
            .issues
            .contains(&TrustStoreRecoveryIssue::ReplacementStoreDidNotChange));
        assert!(report
            .issues
            .contains(&TrustStoreRecoveryIssue::ReplacementCounterEpochDidNotChange));
    }
}
