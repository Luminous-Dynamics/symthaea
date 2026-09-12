// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-shot acceptance ledger for completed assurance trust-store recoveries.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_assurance_trust_store::{
    TrustStoreBackup, TrustStoreCheckpoint, TrustStoreProfile, TrustStoreRecoveryAuthorization,
};
use symthaea_assurance_trust_store_recovery::{
    TrustStoreContinuityStatus, TrustStoreRecoveryCommit, assess_recovery_continuity,
};

const ACCEPTANCE_DIGEST_SCHEMA: &[u8] = b"symthaea-trust-store-recovery-acceptance-v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryAcceptanceRecord {
    pub schema_version: String,
    pub ledger_id: String,
    pub ledger_revision: u64,
    pub logical_store_id: String,
    pub authorization_id: String,
    pub recovery_commit_id: String,
    pub previous_checkpoint_digest: String,
    pub replacement_trust_store_ref: String,
    pub replacement_counter_epoch: String,
    pub first_replacement_checkpoint_digest: String,
    pub continuity_evidence_ref: String,
    pub continuity_evidence_digest: String,
    pub accepted_at_ms: u64,
    pub predecessor_acceptance_digest: Option<String>,
    pub evidence_refs: Vec<String>,
}

impl RecoveryAcceptanceRecord {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.ledger_id.trim().is_empty()
            && self.ledger_revision > 0
            && !self.logical_store_id.trim().is_empty()
            && !self.authorization_id.trim().is_empty()
            && !self.recovery_commit_id.trim().is_empty()
            && valid_digest(&self.previous_checkpoint_digest)
            && !self.replacement_trust_store_ref.trim().is_empty()
            && !self.replacement_counter_epoch.trim().is_empty()
            && valid_digest(&self.first_replacement_checkpoint_digest)
            && !self.continuity_evidence_ref.trim().is_empty()
            && valid_digest(&self.continuity_evidence_digest)
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
            && match self.ledger_revision {
                1 => self.predecessor_acceptance_digest.is_none(),
                _ => self
                    .predecessor_acceptance_digest
                    .as_ref()
                    .is_some_and(|value| valid_digest(value)),
            }
    }

    pub fn record_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(ACCEPTANCE_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.ledger_id);
        push_field(&mut hasher, &self.ledger_revision.to_string());
        push_field(&mut hasher, &self.logical_store_id);
        push_field(&mut hasher, &self.authorization_id);
        push_field(&mut hasher, &self.recovery_commit_id);
        push_field(&mut hasher, &self.previous_checkpoint_digest);
        push_field(&mut hasher, &self.replacement_trust_store_ref);
        push_field(&mut hasher, &self.replacement_counter_epoch);
        push_field(&mut hasher, &self.first_replacement_checkpoint_digest);
        push_field(&mut hasher, &self.continuity_evidence_ref);
        push_field(&mut hasher, &self.continuity_evidence_digest);
        push_field(&mut hasher, &self.accepted_at_ms.to_string());
        push_optional(&mut hasher, self.predecessor_acceptance_digest.as_deref());
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryAcceptanceError {
    ContinuityDidNotPass,
    InvalidEvidenceBinding,
    AcceptancePredatesRecoveredCheckpoint,
    AcceptanceIsFutureDated,
}

/// Recompute the complete recovery-continuity theorem and, only if it passes,
/// produce a durable one-shot acceptance record.
///
/// A caller cannot manufacture ledger authority by constructing a synthetic
/// `TrustStoreContinuityReport`; the report is recomputed from original inputs.
#[allow(clippy::too_many_arguments)]
pub fn accept_recovery(
    ledger_id: impl Into<String>,
    ledger_revision: u64,
    predecessor_acceptance_digest: Option<String>,
    previous_profile: &TrustStoreProfile,
    replacement_profile: &TrustStoreProfile,
    previous_tip: &TrustStoreCheckpoint,
    backup: &TrustStoreBackup,
    authorization: &TrustStoreRecoveryAuthorization,
    commit: &TrustStoreRecoveryCommit,
    first_replacement_checkpoint: &TrustStoreCheckpoint,
    now_ms: u64,
    continuity_evidence_ref: impl Into<String>,
    continuity_evidence_digest: impl Into<String>,
    accepted_at_ms: u64,
    evidence_refs: Vec<String>,
) -> Result<RecoveryAcceptanceRecord, RecoveryAcceptanceError> {
    let continuity = assess_recovery_continuity(
        previous_profile,
        replacement_profile,
        previous_tip,
        backup,
        authorization,
        commit,
        first_replacement_checkpoint,
        now_ms,
    );
    if continuity.status != TrustStoreContinuityStatus::Valid || !continuity.issues.is_empty() {
        return Err(RecoveryAcceptanceError::ContinuityDidNotPass);
    }
    if accepted_at_ms < first_replacement_checkpoint.recorded_at_ms {
        return Err(RecoveryAcceptanceError::AcceptancePredatesRecoveredCheckpoint);
    }
    if accepted_at_ms > now_ms {
        return Err(RecoveryAcceptanceError::AcceptanceIsFutureDated);
    }

    let continuity_evidence_ref = continuity_evidence_ref.into();
    let continuity_evidence_digest = continuity_evidence_digest.into();
    if continuity_evidence_ref.trim().is_empty()
        || !valid_digest(&continuity_evidence_digest)
        || evidence_refs.is_empty()
    {
        return Err(RecoveryAcceptanceError::InvalidEvidenceBinding);
    }

    let record = RecoveryAcceptanceRecord {
        schema_version: "1".into(),
        ledger_id: ledger_id.into(),
        ledger_revision,
        logical_store_id: commit.logical_store_id.clone(),
        authorization_id: commit.authorization_id.clone(),
        recovery_commit_id: commit.commit_id.clone(),
        previous_checkpoint_digest: commit.previous_checkpoint_digest.clone(),
        replacement_trust_store_ref: commit.replacement_trust_store_ref.clone(),
        replacement_counter_epoch: commit.replacement_counter_epoch.clone(),
        first_replacement_checkpoint_digest: first_replacement_checkpoint.checkpoint_digest(),
        continuity_evidence_ref,
        continuity_evidence_digest,
        accepted_at_ms,
        predecessor_acceptance_digest,
        evidence_refs,
    };
    if record.validate() {
        Ok(record)
    } else {
        Err(RecoveryAcceptanceError::InvalidEvidenceBinding)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryLedgerStatus {
    Valid,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryLedgerIssue {
    EmptyLedger,
    InvalidRecord(u64),
    DuplicateLedgerRevision(u64),
    DuplicateRecordDigest(String),
    FirstRevisionIsNotOne(u64),
    LedgerRevisionGap { previous: u64, next: u64 },
    LedgerIdChanged { revision: u64 },
    LogicalStoreIdChanged { revision: u64 },
    PredecessorDigestMismatch { revision: u64 },
    AcceptanceTimeRegressed { revision: u64 },
    AuthorizationReused(String),
    RecoveryCommitReused(String),
    PreviousCheckpointForked(String),
    ReplacementCounterEpochReused(String),
    FirstReplacementCheckpointReused(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryLedgerReport {
    pub status: RecoveryLedgerStatus,
    pub ledger_id: Option<String>,
    pub logical_store_id: Option<String>,
    pub accepted_recovery_count: usize,
    pub tip_ledger_revision: Option<u64>,
    pub tip_record_digest: Option<String>,
    pub issues: Vec<RecoveryLedgerIssue>,
}

impl RecoveryLedgerReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn assess_recovery_ledger(records: &[RecoveryAcceptanceRecord]) -> RecoveryLedgerReport {
    if records.is_empty() {
        return RecoveryLedgerReport {
            status: RecoveryLedgerStatus::Invalid,
            ledger_id: None,
            logical_store_id: None,
            accepted_recovery_count: 0,
            tip_ledger_revision: None,
            tip_record_digest: None,
            issues: vec![RecoveryLedgerIssue::EmptyLedger],
        };
    }

    let mut ordered = records.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|record| record.ledger_revision);
    let ledger_id = ordered[0].ledger_id.clone();
    let logical_store_id = ordered[0].logical_store_id.clone();
    let mut issues = Vec::new();
    let mut ledger_revisions = BTreeSet::new();
    let mut record_digests = BTreeSet::new();
    let mut authorizations = BTreeSet::new();
    let mut recovery_commits = BTreeSet::new();
    let mut previous_checkpoints = BTreeSet::new();
    let mut replacement_epochs = BTreeSet::new();
    let mut first_checkpoints = BTreeSet::new();

    for record in &ordered {
        if !record.validate() {
            issues.push(RecoveryLedgerIssue::InvalidRecord(record.ledger_revision));
        }
        if !ledger_revisions.insert(record.ledger_revision) {
            issues.push(RecoveryLedgerIssue::DuplicateLedgerRevision(record.ledger_revision));
        }
        let digest = record.record_digest();
        if !record_digests.insert(digest.clone()) {
            issues.push(RecoveryLedgerIssue::DuplicateRecordDigest(digest));
        }
        if record.ledger_id != ledger_id {
            issues.push(RecoveryLedgerIssue::LedgerIdChanged {
                revision: record.ledger_revision,
            });
        }
        if record.logical_store_id != logical_store_id {
            issues.push(RecoveryLedgerIssue::LogicalStoreIdChanged {
                revision: record.ledger_revision,
            });
        }
        if !authorizations.insert(record.authorization_id.clone()) {
            issues.push(RecoveryLedgerIssue::AuthorizationReused(
                record.authorization_id.clone(),
            ));
        }
        if !recovery_commits.insert(record.recovery_commit_id.clone()) {
            issues.push(RecoveryLedgerIssue::RecoveryCommitReused(
                record.recovery_commit_id.clone(),
            ));
        }
        if !previous_checkpoints.insert(record.previous_checkpoint_digest.clone()) {
            issues.push(RecoveryLedgerIssue::PreviousCheckpointForked(
                record.previous_checkpoint_digest.clone(),
            ));
        }
        if !replacement_epochs.insert(record.replacement_counter_epoch.clone()) {
            issues.push(RecoveryLedgerIssue::ReplacementCounterEpochReused(
                record.replacement_counter_epoch.clone(),
            ));
        }
        if !first_checkpoints.insert(record.first_replacement_checkpoint_digest.clone()) {
            issues.push(RecoveryLedgerIssue::FirstReplacementCheckpointReused(
                record.first_replacement_checkpoint_digest.clone(),
            ));
        }
    }

    if ordered[0].ledger_revision != 1 {
        issues.push(RecoveryLedgerIssue::FirstRevisionIsNotOne(
            ordered[0].ledger_revision,
        ));
    }

    for pair in ordered.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if current.ledger_revision != previous.ledger_revision.saturating_add(1) {
            issues.push(RecoveryLedgerIssue::LedgerRevisionGap {
                previous: previous.ledger_revision,
                next: current.ledger_revision,
            });
        }
        let previous_digest = previous.record_digest();
        if current.predecessor_acceptance_digest.as_deref() != Some(previous_digest.as_str()) {
            issues.push(RecoveryLedgerIssue::PredecessorDigestMismatch {
                revision: current.ledger_revision,
            });
        }
        if current.accepted_at_ms < previous.accepted_at_ms {
            issues.push(RecoveryLedgerIssue::AcceptanceTimeRegressed {
                revision: current.ledger_revision,
            });
        }
    }

    let tip = ordered.last().expect("non-empty recovery ledger");
    RecoveryLedgerReport {
        status: if issues.is_empty() {
            RecoveryLedgerStatus::Valid
        } else {
            RecoveryLedgerStatus::Invalid
        },
        ledger_id: Some(ledger_id),
        logical_store_id: Some(logical_store_id),
        accepted_recovery_count: ordered.len(),
        tip_ledger_revision: Some(tip.ledger_revision),
        tip_record_digest: Some(tip.record_digest()),
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
    use symthaea_assurance_trust_store::TrustStoreBackendKind;

    fn record(revision: u64, predecessor: Option<String>) -> RecoveryAcceptanceRecord {
        RecoveryAcceptanceRecord {
            schema_version: "1".into(),
            ledger_id: "recovery-ledger:policy-root".into(),
            ledger_revision: revision,
            logical_store_id: "policy-root".into(),
            authorization_id: format!("authorization:{revision}"),
            recovery_commit_id: format!("commit:{revision}"),
            previous_checkpoint_digest: format!("blake3:old-tip-{revision}"),
            replacement_trust_store_ref: format!("trust-store:replacement-{revision}"),
            replacement_counter_epoch: format!("epoch:replacement-{revision}"),
            first_replacement_checkpoint_digest: format!("blake3:first-{revision}"),
            continuity_evidence_ref: format!("qualification:continuity-{revision}"),
            continuity_evidence_digest: format!("blake3:continuity-{revision}"),
            accepted_at_ms: revision * 1_000,
            predecessor_acceptance_digest: predecessor,
            evidence_refs: vec![format!("audit:acceptance-{revision}")],
        }
    }

    fn valid_ledger() -> Vec<RecoveryAcceptanceRecord> {
        let r1 = record(1, None);
        let r2 = record(2, Some(r1.record_digest()));
        let r3 = record(3, Some(r2.record_digest()));
        vec![r1, r2, r3]
    }

    fn old_profile() -> TrustStoreProfile {
        TrustStoreProfile {
            schema_version: "1".into(),
            store_id: "policy-root".into(),
            trust_store_ref: "trust-store:a".into(),
            backend_kind: TrustStoreBackendKind::HardwareMonotonicCounter,
            hardware_instance_ref: Some("device:a".into()),
            monotonic_counter_ref: "counter:a".into(),
            initial_counter_epoch: "epoch:a".into(),
            minimum_independent_recovery_approvals: 2,
            evidence_refs: vec!["review:old".into()],
        }
    }

    fn new_profile() -> TrustStoreProfile {
        TrustStoreProfile {
            schema_version: "1".into(),
            store_id: "policy-root".into(),
            trust_store_ref: "trust-store:b".into(),
            backend_kind: TrustStoreBackendKind::HardwareMonotonicCounter,
            hardware_instance_ref: Some("device:b".into()),
            monotonic_counter_ref: "counter:b".into(),
            initial_counter_epoch: "epoch:b".into(),
            minimum_independent_recovery_approvals: 2,
            evidence_refs: vec!["review:new".into()],
        }
    }

    fn tip() -> TrustStoreCheckpoint {
        TrustStoreCheckpoint {
            schema_version: "1".into(),
            checkpoint_id: "checkpoint:3".into(),
            store_id: "policy-root".into(),
            store_revision: 3,
            counter_epoch: "epoch:a".into(),
            counter_value: 9,
            anchor_revision: 3,
            anchor_digest: "blake3:anchor-3".into(),
            policy_tip_revision: 3,
            predecessor_checkpoint_digest: Some("blake3:checkpoint-2".into()),
            recorded_at_ms: 3_000,
            attestation_ref: "attestation:old".into(),
            independent_verification_ref: "verification:old".into(),
            evidence_refs: vec!["audit:old".into()],
        }
    }

    fn accepted_fixture() -> (
        TrustStoreProfile,
        TrustStoreProfile,
        TrustStoreCheckpoint,
        TrustStoreBackup,
        TrustStoreRecoveryAuthorization,
        TrustStoreRecoveryCommit,
        TrustStoreCheckpoint,
    ) {
        let old = old_profile();
        let new = new_profile();
        let tip = tip();
        let backup = TrustStoreBackup::from_checkpoint(
            &old,
            &tip,
            "backup:tip",
            3_100,
            vec!["audit:backup".into()],
        )
        .unwrap();
        let auth = TrustStoreRecoveryAuthorization {
            authorization_id: "authorization:1".into(),
            store_id: "policy-root".into(),
            incident_ref: "incident:loss".into(),
            expected_previous_checkpoint_digest: tip.checkpoint_digest(),
            expected_backup_digest: backup.backup_digest(),
            minimum_counter_value: tip.counter_value,
            minimum_policy_tip_revision: tip.policy_tip_revision,
            replacement_trust_store_ref: "trust-store:b".into(),
            replacement_counter_epoch: "epoch:b".into(),
            approved_by_refs: vec!["reviewer:a".into(), "reviewer:b".into()],
            independent_verification_ref: "verification:auth".into(),
            authorized_at_ms: 4_000,
            expires_at_ms: 8_000,
            evidence_refs: vec!["audit:auth".into()],
        };
        let commit = TrustStoreRecoveryCommit {
            schema_version: "1".into(),
            commit_id: "commit:1".into(),
            authorization_id: "authorization:1".into(),
            logical_store_id: "policy-root".into(),
            previous_checkpoint_digest: tip.checkpoint_digest(),
            backup_digest: backup.backup_digest(),
            replacement_trust_store_ref: "trust-store:b".into(),
            replacement_counter_epoch: "epoch:b".into(),
            first_counter_value: 1,
            restored_anchor_revision: tip.anchor_revision,
            restored_anchor_digest: tip.anchor_digest.clone(),
            restored_policy_tip_revision: tip.policy_tip_revision,
            replacement_attestation_ref: "attestation:new".into(),
            independent_verification_ref: "verification:commit".into(),
            committed_at_ms: 5_000,
            evidence_refs: vec!["audit:commit".into()],
        };
        let first = TrustStoreCheckpoint {
            schema_version: "1".into(),
            checkpoint_id: "checkpoint:4".into(),
            store_id: "policy-root".into(),
            store_revision: 4,
            counter_epoch: "epoch:b".into(),
            counter_value: 1,
            anchor_revision: tip.anchor_revision,
            anchor_digest: tip.anchor_digest.clone(),
            policy_tip_revision: tip.policy_tip_revision,
            predecessor_checkpoint_digest: Some(tip.checkpoint_digest()),
            recorded_at_ms: 5_100,
            attestation_ref: "attestation:first".into(),
            independent_verification_ref: "verification:first".into(),
            evidence_refs: vec!["audit:first".into()],
        };
        (old, new, tip, backup, auth, commit, first)
    }

    #[test]
    fn acceptance_recomputes_continuity() {
        let (old, new, tip, backup, auth, commit, first) = accepted_fixture();
        let record = accept_recovery(
            "recovery-ledger:policy-root",
            1,
            None,
            &old,
            &new,
            &tip,
            &backup,
            &auth,
            &commit,
            &first,
            5_300,
            "qualification:continuity",
            "blake3:continuity",
            5_200,
            vec!["audit:acceptance".into()],
        )
        .unwrap();
        assert_eq!(record.authorization_id, "authorization:1");
        assert!(!record.grants_physical_authority());
    }

    #[test]
    fn invalid_continuity_cannot_be_accepted() {
        let (old, new, tip, backup, mut auth, commit, first) = accepted_fixture();
        auth.expires_at_ms = 4_500;
        let result = accept_recovery(
            "recovery-ledger:policy-root",
            1,
            None,
            &old,
            &new,
            &tip,
            &backup,
            &auth,
            &commit,
            &first,
            5_300,
            "qualification:continuity",
            "blake3:continuity",
            5_200,
            vec!["audit:acceptance".into()],
        );
        assert_eq!(result, Err(RecoveryAcceptanceError::ContinuityDidNotPass));
    }

    #[test]
    fn valid_recovery_acceptance_chain_is_valid() {
        let report = assess_recovery_ledger(&valid_ledger());
        assert_eq!(report.status, RecoveryLedgerStatus::Valid);
        assert_eq!(report.accepted_recovery_count, 3);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn authorization_is_single_use() {
        let mut ledger = valid_ledger();
        ledger[2].authorization_id = ledger[1].authorization_id.clone();
        let report = assess_recovery_ledger(&ledger);
        assert_eq!(report.status, RecoveryLedgerStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            RecoveryLedgerIssue::AuthorizationReused(_)
        )));
    }

    #[test]
    fn one_old_checkpoint_cannot_fork_into_two_accepted_recoveries() {
        let mut ledger = valid_ledger();
        ledger[2].previous_checkpoint_digest = ledger[1].previous_checkpoint_digest.clone();
        let report = assess_recovery_ledger(&ledger);
        assert_eq!(report.status, RecoveryLedgerStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            RecoveryLedgerIssue::PreviousCheckpointForked(_)
        )));
    }

    #[test]
    fn counter_epoch_cannot_be_reused_across_recoveries() {
        let mut ledger = valid_ledger();
        ledger[2].replacement_counter_epoch = ledger[1].replacement_counter_epoch.clone();
        let report = assess_recovery_ledger(&ledger);
        assert_eq!(report.status, RecoveryLedgerStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            RecoveryLedgerIssue::ReplacementCounterEpochReused(_)
        )));
    }

    #[test]
    fn deleted_middle_acceptance_record_is_detected() {
        let mut ledger = valid_ledger();
        ledger.remove(1);
        let report = assess_recovery_ledger(&ledger);
        assert_eq!(report.status, RecoveryLedgerStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            RecoveryLedgerIssue::LedgerRevisionGap { previous: 1, next: 3 }
        )));
    }
}
