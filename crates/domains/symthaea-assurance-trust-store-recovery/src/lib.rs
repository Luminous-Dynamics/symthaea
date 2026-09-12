// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Recovery-continuity assurance for replacement monotonic trust stores.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_assurance_trust_store::{
    TrustStoreBackup, TrustStoreCheckpoint, TrustStoreProfile, TrustStoreRecoveryAuthorization,
    TrustStoreRecoveryStatus, assess_recovery,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustStoreRecoveryCommit {
    pub schema_version: String,
    pub commit_id: String,
    pub authorization_id: String,
    pub logical_store_id: String,
    pub previous_checkpoint_digest: String,
    pub backup_digest: String,
    pub replacement_trust_store_ref: String,
    pub replacement_counter_epoch: String,
    pub first_counter_value: u64,
    pub restored_anchor_revision: u64,
    pub restored_anchor_digest: String,
    pub restored_policy_tip_revision: u64,
    pub replacement_attestation_ref: String,
    pub independent_verification_ref: String,
    pub committed_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl TrustStoreRecoveryCommit {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.commit_id.trim().is_empty()
            && !self.authorization_id.trim().is_empty()
            && !self.logical_store_id.trim().is_empty()
            && valid_digest(&self.previous_checkpoint_digest)
            && valid_digest(&self.backup_digest)
            && !self.replacement_trust_store_ref.trim().is_empty()
            && !self.replacement_counter_epoch.trim().is_empty()
            && self.first_counter_value > 0
            && self.restored_anchor_revision > 0
            && valid_digest(&self.restored_anchor_digest)
            && self.restored_policy_tip_revision > 0
            && !self.replacement_attestation_ref.trim().is_empty()
            && !self.independent_verification_ref.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustStoreContinuityStatus {
    Valid,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustStoreContinuityIssue {
    RecoveryWasNotEligible,
    InvalidPreviousProfile,
    InvalidReplacementProfile,
    LogicalStoreIdentityChanged,
    ReplacementStoreReferenceMismatch,
    ReplacementCounterEpochMismatch,
    ReplacementProfileDidNotChangeStore,
    InvalidRecoveryCommit,
    AuthorizationMismatch,
    PreviousCheckpointMismatch,
    BackupMismatch,
    RestoredAnchorMismatch,
    RestoredPolicyRevisionMismatch,
    RecoveryCommittedBeforeAuthorization,
    RecoveryCommittedAfterAuthorizationExpiry,
    RecoveryCommitIsFutureDated,
    FirstReplacementCheckpointInvalid,
    FirstReplacementCheckpointRevisionMismatch,
    FirstReplacementCheckpointPredecessorMismatch,
    FirstReplacementCheckpointEpochMismatch,
    FirstReplacementCheckpointCounterMismatch,
    FirstReplacementCheckpointAnchorMismatch,
    FirstReplacementCheckpointPolicyMismatch,
    FirstReplacementCheckpointPredatesCommit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustStoreContinuityReport {
    pub status: TrustStoreContinuityStatus,
    pub previous_checkpoint_digest: String,
    pub replacement_store_ref: String,
    pub replacement_counter_epoch: String,
    pub restored_anchor_revision: u64,
    pub restored_anchor_digest: String,
    pub restored_policy_tip_revision: u64,
    pub issues: Vec<TrustStoreContinuityIssue>,
}

impl TrustStoreContinuityReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Verify that an authorized recovery was actually completed by a replacement
/// trust store without changing or rolling back the last accepted anchor/policy
/// state.
///
/// Recovery eligibility is recomputed at the recorded recovery-commit time. A
/// caller-supplied `Eligible` flag is therefore never accepted as authority.
/// The first replacement checkpoint must represent the exact restored state;
/// subsequent policy advancement belongs in later checkpoints.
pub fn assess_recovery_continuity(
    previous_profile: &TrustStoreProfile,
    replacement_profile: &TrustStoreProfile,
    previous_tip: &TrustStoreCheckpoint,
    backup: &TrustStoreBackup,
    authorization: &TrustStoreRecoveryAuthorization,
    commit: &TrustStoreRecoveryCommit,
    first_replacement_checkpoint: &TrustStoreCheckpoint,
    now_ms: u64,
) -> TrustStoreContinuityReport {
    let mut issues = Vec::new();

    if !previous_profile.validate() {
        issues.push(TrustStoreContinuityIssue::InvalidPreviousProfile);
    }
    if !replacement_profile.validate() {
        issues.push(TrustStoreContinuityIssue::InvalidReplacementProfile);
    }
    if !commit.validate() {
        issues.push(TrustStoreContinuityIssue::InvalidRecoveryCommit);
    }

    let recomputed_recovery = assess_recovery(
        previous_profile,
        previous_tip,
        backup,
        authorization,
        commit.committed_at_ms,
    );
    if recomputed_recovery.status != TrustStoreRecoveryStatus::Eligible
        || !recomputed_recovery.issues.is_empty()
    {
        issues.push(TrustStoreContinuityIssue::RecoveryWasNotEligible);
    }

    if previous_profile.store_id != replacement_profile.store_id
        || commit.logical_store_id != previous_profile.store_id
    {
        issues.push(TrustStoreContinuityIssue::LogicalStoreIdentityChanged);
    }
    if replacement_profile.trust_store_ref != authorization.replacement_trust_store_ref
        || commit.replacement_trust_store_ref != authorization.replacement_trust_store_ref
    {
        issues.push(TrustStoreContinuityIssue::ReplacementStoreReferenceMismatch);
    }
    if replacement_profile.initial_counter_epoch != authorization.replacement_counter_epoch
        || commit.replacement_counter_epoch != authorization.replacement_counter_epoch
    {
        issues.push(TrustStoreContinuityIssue::ReplacementCounterEpochMismatch);
    }
    if replacement_profile.trust_store_ref == previous_profile.trust_store_ref {
        issues.push(TrustStoreContinuityIssue::ReplacementProfileDidNotChangeStore);
    }
    if commit.authorization_id != authorization.authorization_id {
        issues.push(TrustStoreContinuityIssue::AuthorizationMismatch);
    }

    let previous_digest = previous_tip.checkpoint_digest();
    let backup_digest = backup.backup_digest();
    if commit.previous_checkpoint_digest != previous_digest {
        issues.push(TrustStoreContinuityIssue::PreviousCheckpointMismatch);
    }
    if commit.backup_digest != backup_digest {
        issues.push(TrustStoreContinuityIssue::BackupMismatch);
    }
    if commit.restored_anchor_revision != previous_tip.anchor_revision
        || commit.restored_anchor_digest != previous_tip.anchor_digest
        || commit.restored_anchor_digest != backup.anchor_digest
    {
        issues.push(TrustStoreContinuityIssue::RestoredAnchorMismatch);
    }
    if commit.restored_policy_tip_revision != previous_tip.policy_tip_revision
        || commit.restored_policy_tip_revision != backup.policy_tip_revision
        || commit.restored_policy_tip_revision < authorization.minimum_policy_tip_revision
    {
        issues.push(TrustStoreContinuityIssue::RestoredPolicyRevisionMismatch);
    }
    if commit.committed_at_ms < authorization.authorized_at_ms {
        issues.push(TrustStoreContinuityIssue::RecoveryCommittedBeforeAuthorization);
    }
    if commit.committed_at_ms > authorization.expires_at_ms {
        issues.push(TrustStoreContinuityIssue::RecoveryCommittedAfterAuthorizationExpiry);
    }
    if now_ms < commit.committed_at_ms {
        issues.push(TrustStoreContinuityIssue::RecoveryCommitIsFutureDated);
    }

    if !first_replacement_checkpoint.validate(replacement_profile) {
        issues.push(TrustStoreContinuityIssue::FirstReplacementCheckpointInvalid);
    }
    if first_replacement_checkpoint.store_revision != previous_tip.store_revision.saturating_add(1) {
        issues.push(TrustStoreContinuityIssue::FirstReplacementCheckpointRevisionMismatch);
    }
    if first_replacement_checkpoint.predecessor_checkpoint_digest.as_deref()
        != Some(previous_digest.as_str())
    {
        issues.push(TrustStoreContinuityIssue::FirstReplacementCheckpointPredecessorMismatch);
    }
    if first_replacement_checkpoint.counter_epoch != authorization.replacement_counter_epoch {
        issues.push(TrustStoreContinuityIssue::FirstReplacementCheckpointEpochMismatch);
    }
    if first_replacement_checkpoint.counter_value != commit.first_counter_value {
        issues.push(TrustStoreContinuityIssue::FirstReplacementCheckpointCounterMismatch);
    }
    if first_replacement_checkpoint.anchor_revision != commit.restored_anchor_revision
        || first_replacement_checkpoint.anchor_digest != commit.restored_anchor_digest
    {
        issues.push(TrustStoreContinuityIssue::FirstReplacementCheckpointAnchorMismatch);
    }
    if first_replacement_checkpoint.policy_tip_revision != commit.restored_policy_tip_revision {
        issues.push(TrustStoreContinuityIssue::FirstReplacementCheckpointPolicyMismatch);
    }
    if first_replacement_checkpoint.recorded_at_ms < commit.committed_at_ms {
        issues.push(TrustStoreContinuityIssue::FirstReplacementCheckpointPredatesCommit);
    }

    let invalid = issues.iter().any(|issue| {
        matches!(
            issue,
            TrustStoreContinuityIssue::InvalidPreviousProfile
                | TrustStoreContinuityIssue::InvalidReplacementProfile
                | TrustStoreContinuityIssue::LogicalStoreIdentityChanged
                | TrustStoreContinuityIssue::InvalidRecoveryCommit
                | TrustStoreContinuityIssue::AuthorizationMismatch
                | TrustStoreContinuityIssue::PreviousCheckpointMismatch
                | TrustStoreContinuityIssue::BackupMismatch
                | TrustStoreContinuityIssue::FirstReplacementCheckpointInvalid
        )
    });

    TrustStoreContinuityReport {
        status: if invalid {
            TrustStoreContinuityStatus::Invalid
        } else if issues.is_empty() {
            TrustStoreContinuityStatus::Valid
        } else {
            TrustStoreContinuityStatus::Blocked
        },
        previous_checkpoint_digest: previous_digest,
        replacement_store_ref: replacement_profile.trust_store_ref.clone(),
        replacement_counter_epoch: replacement_profile.initial_counter_epoch.clone(),
        restored_anchor_revision: commit.restored_anchor_revision,
        restored_anchor_digest: commit.restored_anchor_digest.clone(),
        restored_policy_tip_revision: commit.restored_policy_tip_revision,
        issues,
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

    fn previous_profile() -> TrustStoreProfile {
        TrustStoreProfile {
            schema_version: "1".into(),
            store_id: "policy-root".into(),
            trust_store_ref: "trust-store:hardware-a".into(),
            backend_kind: TrustStoreBackendKind::HardwareMonotonicCounter,
            hardware_instance_ref: Some("device:tpm-a".into()),
            monotonic_counter_ref: "counter:a".into(),
            initial_counter_epoch: "epoch:a".into(),
            minimum_independent_recovery_approvals: 2,
            evidence_refs: vec!["review:old-profile".into()],
        }
    }

    fn replacement_profile() -> TrustStoreProfile {
        TrustStoreProfile {
            schema_version: "1".into(),
            store_id: "policy-root".into(),
            trust_store_ref: "trust-store:hardware-b".into(),
            backend_kind: TrustStoreBackendKind::HardwareMonotonicCounter,
            hardware_instance_ref: Some("device:tpm-b".into()),
            monotonic_counter_ref: "counter:b".into(),
            initial_counter_epoch: "epoch:b".into(),
            minimum_independent_recovery_approvals: 2,
            evidence_refs: vec!["review:new-profile".into()],
        }
    }

    fn previous_tip() -> TrustStoreCheckpoint {
        TrustStoreCheckpoint {
            schema_version: "1".into(),
            checkpoint_id: "checkpoint:3".into(),
            store_id: "policy-root".into(),
            store_revision: 3,
            counter_epoch: "epoch:a".into(),
            counter_value: 15,
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

    fn backup(tip: &TrustStoreCheckpoint) -> TrustStoreBackup {
        TrustStoreBackup::from_checkpoint(
            &previous_profile(),
            tip,
            "backup:tip",
            3_100,
            vec!["audit:backup".into()],
        )
        .unwrap()
    }

    fn authorization(tip: &TrustStoreCheckpoint, backup: &TrustStoreBackup) -> TrustStoreRecoveryAuthorization {
        TrustStoreRecoveryAuthorization {
            authorization_id: "recovery:1".into(),
            store_id: "policy-root".into(),
            incident_ref: "incident:store-loss".into(),
            expected_previous_checkpoint_digest: tip.checkpoint_digest(),
            expected_backup_digest: backup.backup_digest(),
            minimum_counter_value: tip.counter_value,
            minimum_policy_tip_revision: tip.policy_tip_revision,
            replacement_trust_store_ref: "trust-store:hardware-b".into(),
            replacement_counter_epoch: "epoch:b".into(),
            approved_by_refs: vec!["reviewer:a".into(), "reviewer:b".into()],
            independent_verification_ref: "verification:authorization".into(),
            authorized_at_ms: 4_000,
            expires_at_ms: 8_000,
            evidence_refs: vec!["audit:authorization".into()],
        }
    }

    fn commit(tip: &TrustStoreCheckpoint, backup: &TrustStoreBackup) -> TrustStoreRecoveryCommit {
        TrustStoreRecoveryCommit {
            schema_version: "1".into(),
            commit_id: "recovery-commit:1".into(),
            authorization_id: "recovery:1".into(),
            logical_store_id: "policy-root".into(),
            previous_checkpoint_digest: tip.checkpoint_digest(),
            backup_digest: backup.backup_digest(),
            replacement_trust_store_ref: "trust-store:hardware-b".into(),
            replacement_counter_epoch: "epoch:b".into(),
            first_counter_value: 1,
            restored_anchor_revision: tip.anchor_revision,
            restored_anchor_digest: tip.anchor_digest.clone(),
            restored_policy_tip_revision: tip.policy_tip_revision,
            replacement_attestation_ref: "attestation:new-hardware".into(),
            independent_verification_ref: "verification:recovery-commit".into(),
            committed_at_ms: 5_000,
            evidence_refs: vec!["audit:recovery-commit".into()],
        }
    }

    fn first_replacement_checkpoint(tip: &TrustStoreCheckpoint) -> TrustStoreCheckpoint {
        TrustStoreCheckpoint {
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
            attestation_ref: "attestation:new-checkpoint".into(),
            independent_verification_ref: "verification:new-checkpoint".into(),
            evidence_refs: vec!["audit:new-checkpoint".into()],
        }
    }

    fn fixture() -> (
        TrustStoreProfile,
        TrustStoreProfile,
        TrustStoreCheckpoint,
        TrustStoreBackup,
        TrustStoreRecoveryAuthorization,
        TrustStoreRecoveryCommit,
        TrustStoreCheckpoint,
    ) {
        let old_profile = previous_profile();
        let new_profile = replacement_profile();
        let tip = previous_tip();
        let backup = backup(&tip);
        let auth = authorization(&tip, &backup);
        let commit = commit(&tip, &backup);
        let first = first_replacement_checkpoint(&tip);
        (old_profile, new_profile, tip, backup, auth, commit, first)
    }

    #[test]
    fn reviewed_replacement_preserves_exact_continuity() {
        let (old_profile, new_profile, tip, backup, auth, commit, first) = fixture();
        let continuity = assess_recovery_continuity(
            &old_profile,
            &new_profile,
            &tip,
            &backup,
            &auth,
            &commit,
            &first,
            5_200,
        );
        assert_eq!(continuity.status, TrustStoreContinuityStatus::Valid);
        assert!(!continuity.grants_physical_authority());
    }

    #[test]
    fn restored_policy_state_must_match_exact_pre_loss_tip() {
        let (old_profile, new_profile, tip, backup, auth, mut commit, first) = fixture();
        commit.restored_policy_tip_revision = 2;
        let continuity = assess_recovery_continuity(
            &old_profile, &new_profile, &tip, &backup, &auth, &commit, &first, 5_200,
        );
        assert_eq!(continuity.status, TrustStoreContinuityStatus::Blocked);
        assert!(continuity
            .issues
            .contains(&TrustStoreContinuityIssue::RestoredPolicyRevisionMismatch));
    }

    #[test]
    fn caller_cannot_forge_eligible_recovery_after_expiry() {
        let (old_profile, new_profile, tip, backup, mut auth, mut commit, first) = fixture();
        auth.expires_at_ms = 4_500;
        commit.committed_at_ms = 5_000;
        let continuity = assess_recovery_continuity(
            &old_profile, &new_profile, &tip, &backup, &auth, &commit, &first, 5_200,
        );
        assert_eq!(continuity.status, TrustStoreContinuityStatus::Blocked);
        assert!(continuity
            .issues
            .contains(&TrustStoreContinuityIssue::RecoveryWasNotEligible));
    }

    #[test]
    fn replacement_checkpoint_must_link_to_exact_old_tip() {
        let (old_profile, new_profile, tip, backup, auth, commit, mut first) = fixture();
        first.predecessor_checkpoint_digest = Some("blake3:other-tip".into());
        let continuity = assess_recovery_continuity(
            &old_profile, &new_profile, &tip, &backup, &auth, &commit, &first, 5_200,
        );
        assert_eq!(continuity.status, TrustStoreContinuityStatus::Blocked);
        assert!(continuity.issues.contains(
            &TrustStoreContinuityIssue::FirstReplacementCheckpointPredecessorMismatch
        ));
    }

    #[test]
    fn replacement_profile_cannot_substitute_another_store() {
        let (old_profile, mut new_profile, tip, backup, auth, commit, first) = fixture();
        new_profile.trust_store_ref = "trust-store:unreviewed".into();
        let continuity = assess_recovery_continuity(
            &old_profile, &new_profile, &tip, &backup, &auth, &commit, &first, 5_200,
        );
        assert_eq!(continuity.status, TrustStoreContinuityStatus::Blocked);
        assert!(continuity
            .issues
            .contains(&TrustStoreContinuityIssue::ReplacementStoreReferenceMismatch));
    }

    #[test]
    fn first_replacement_checkpoint_cannot_skip_restored_state() {
        let (old_profile, new_profile, tip, backup, auth, commit, mut first) = fixture();
        first.policy_tip_revision = 4;
        first.counter_value = 2;
        let continuity = assess_recovery_continuity(
            &old_profile, &new_profile, &tip, &backup, &auth, &commit, &first, 5_200,
        );
        assert_eq!(continuity.status, TrustStoreContinuityStatus::Blocked);
        assert!(continuity
            .issues
            .contains(&TrustStoreContinuityIssue::FirstReplacementCheckpointCounterMismatch));
        assert!(continuity
            .issues
            .contains(&TrustStoreContinuityIssue::FirstReplacementCheckpointPolicyMismatch));
    }
}
