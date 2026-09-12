// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adversarial qualification for the assurance trust-root stack.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_assurance_trust_store::{
    TrustStoreBackup, TrustStoreBackendKind, TrustStoreChainStatus, TrustStoreCheckpoint,
    TrustStoreProfile, TrustStoreRecoveryAuthorization, TrustStoreRecoveryStatus,
    assess_checkpoint_chain, assess_recovery,
};
use symthaea_assurance_trust_store_attestation::CheckpointAttestationVerificationReceipt;
use symthaea_assurance_trust_store_current_state::{
    CurrentTrustStoreStatus, assess_current_trust_store_state,
};
use symthaea_assurance_trust_store_recovery::{
    TrustStoreContinuityStatus, TrustStoreRecoveryCommit, assess_recovery_continuity,
};
use symthaea_assurance_trust_store_recovery_ledger::{
    RecoveryAcceptanceError, RecoveryAcceptanceRecord, RecoveryLedgerIssue, RecoveryLedgerStatus,
    accept_recovery, assess_recovery_ledger,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustRootCrucibleStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustRootCrucibleScenario {
    pub scenario_id: String,
    pub passed: bool,
    pub observation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustRootCrucibleReport {
    pub status: TrustRootCrucibleStatus,
    pub scenarios: Vec<TrustRootCrucibleScenario>,
}

impl TrustRootCrucibleReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

fn scenario(id: &str, passed: bool, observation: impl Into<String>) -> TrustRootCrucibleScenario {
    TrustRootCrucibleScenario {
        scenario_id: id.into(),
        passed,
        observation: observation.into(),
    }
}

fn old_profile() -> TrustStoreProfile {
    TrustStoreProfile {
        schema_version: "1".into(),
        store_id: "policy-root".into(),
        trust_store_ref: "trust-store:a".into(),
        backend_kind: TrustStoreBackendKind::HardwareMonotonicCounter,
        hardware_instance_ref: Some("hardware:a".into()),
        monotonic_counter_ref: "counter:a".into(),
        initial_counter_epoch: "epoch:a".into(),
        minimum_independent_recovery_approvals: 2,
        evidence_refs: vec!["review:old-profile".into()],
    }
}

fn new_profile() -> TrustStoreProfile {
    TrustStoreProfile {
        schema_version: "1".into(),
        store_id: "policy-root".into(),
        trust_store_ref: "trust-store:b".into(),
        backend_kind: TrustStoreBackendKind::HardwareMonotonicCounter,
        hardware_instance_ref: Some("hardware:b".into()),
        monotonic_counter_ref: "counter:b".into(),
        initial_counter_epoch: "epoch:b".into(),
        minimum_independent_recovery_approvals: 2,
        evidence_refs: vec!["review:new-profile".into()],
    }
}

fn checkpoint(
    revision: u64,
    epoch: &str,
    counter: u64,
    anchor_revision: u64,
    policy_revision: u64,
    previous: Option<String>,
) -> TrustStoreCheckpoint {
    TrustStoreCheckpoint {
        schema_version: "1".into(),
        checkpoint_id: format!("checkpoint:{revision}"),
        store_id: "policy-root".into(),
        store_revision: revision,
        counter_epoch: epoch.into(),
        counter_value: counter,
        anchor_revision,
        anchor_digest: format!("blake3:anchor-{anchor_revision}"),
        policy_tip_revision: policy_revision,
        predecessor_checkpoint_digest: previous,
        recorded_at_ms: revision * 1_000,
        attestation_ref: format!("attestation:{revision}"),
        independent_verification_ref: format!("verification:{revision}"),
        evidence_refs: vec![format!("audit:{revision}")],
    }
}

fn initial_chain() -> Vec<TrustStoreCheckpoint> {
    let c1 = checkpoint(1, "epoch:a", 3, 1, 1, None);
    let c2 = checkpoint(2, "epoch:a", 4, 2, 2, Some(c1.checkpoint_digest()));
    let c3 = checkpoint(3, "epoch:a", 7, 3, 3, Some(c2.checkpoint_digest()));
    vec![c1, c2, c3]
}

fn backup(tip: &TrustStoreCheckpoint) -> TrustStoreBackup {
    TrustStoreBackup::from_checkpoint(
        &old_profile(),
        tip,
        "backup:tip",
        tip.recorded_at_ms + 100,
        vec!["audit:backup".into()],
    )
    .unwrap()
}

fn authorization(
    tip: &TrustStoreCheckpoint,
    backup: &TrustStoreBackup,
) -> TrustStoreRecoveryAuthorization {
    TrustStoreRecoveryAuthorization {
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
        independent_verification_ref: "verification:authorization".into(),
        authorized_at_ms: 4_000,
        expires_at_ms: 8_000,
        evidence_refs: vec!["audit:authorization".into()],
    }
}

fn recovery_commit(
    tip: &TrustStoreCheckpoint,
    backup: &TrustStoreBackup,
) -> TrustStoreRecoveryCommit {
    TrustStoreRecoveryCommit {
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
        replacement_attestation_ref: "attestation:replacement".into(),
        independent_verification_ref: "verification:commit".into(),
        committed_at_ms: 5_000,
        evidence_refs: vec!["audit:commit".into()],
    }
}

fn first_replacement(tip: &TrustStoreCheckpoint) -> TrustStoreCheckpoint {
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
        attestation_ref: "attestation:first-replacement".into(),
        independent_verification_ref: "verification:first-replacement".into(),
        evidence_refs: vec!["audit:first-replacement".into()],
    }
}

fn acceptance_record_for_first(first: &TrustStoreCheckpoint) -> RecoveryAcceptanceRecord {
    RecoveryAcceptanceRecord {
        schema_version: "1".into(),
        ledger_id: "recovery-ledger:policy-root".into(),
        ledger_revision: 1,
        logical_store_id: "policy-root".into(),
        authorization_id: "authorization:1".into(),
        recovery_commit_id: "commit:1".into(),
        previous_checkpoint_digest: "blake3:old-tip".into(),
        replacement_trust_store_ref: "trust-store:b".into(),
        replacement_counter_epoch: "epoch:b".into(),
        first_replacement_checkpoint_digest: first.checkpoint_digest(),
        continuity_evidence_ref: "qualification:continuity".into(),
        continuity_evidence_digest: "blake3:continuity".into(),
        accepted_at_ms: 5_200,
        predecessor_acceptance_digest: None,
        evidence_refs: vec!["audit:acceptance".into()],
    }
}

pub fn run_trust_root_crucible() -> TrustRootCrucibleReport {
    let mut scenarios = Vec::new();
    let old = old_profile();
    let new = new_profile();
    let chain = initial_chain();
    let tip = chain.last().expect("fixture chain");
    let backup = backup(tip);
    let auth = authorization(tip, &backup);
    let commit = recovery_commit(tip, &backup);
    let first = first_replacement(tip);

    let chain_report = assess_checkpoint_chain(&old, &chain);
    scenarios.push(scenario(
        "monotonic_checkpoint_chain_positive",
        chain_report.status == TrustStoreChainStatus::Valid,
        format!("status={:?}", chain_report.status),
    ));

    let mut regressed = chain.clone();
    regressed[2].counter_value = regressed[1].counter_value;
    let regressed_report = assess_checkpoint_chain(&old, &regressed);
    scenarios.push(scenario(
        "counter_regression_rejected",
        regressed_report.status == TrustStoreChainStatus::Invalid,
        format!("status={:?}", regressed_report.status),
    ));

    let mut epoch_swap = chain.clone();
    epoch_swap[2].counter_epoch = "epoch:unreviewed".into();
    let epoch_report = assess_checkpoint_chain(&old, &epoch_swap);
    scenarios.push(scenario(
        "epoch_change_without_recovery_rejected",
        epoch_report.status == TrustStoreChainStatus::Invalid,
        format!("status={:?}", epoch_report.status),
    ));

    let old_backup = backup(&chain[1]);
    let old_backup_report = assess_recovery(&old, tip, &old_backup, &authorization(tip, &old_backup), 4_500);
    scenarios.push(scenario(
        "old_backup_restore_blocked",
        old_backup_report.status == TrustStoreRecoveryStatus::Blocked,
        format!("status={:?}", old_backup_report.status),
    ));

    let mut forged_backup = backup.clone();
    forged_backup.anchor_digest = "blake3:forged-anchor".into();
    let forged_auth = authorization(tip, &forged_backup);
    let forged_report = assess_recovery(&old, tip, &forged_backup, &forged_auth, 4_500);
    scenarios.push(scenario(
        "forged_backup_snapshot_blocked",
        forged_report.status == TrustStoreRecoveryStatus::Blocked,
        format!("status={:?}", forged_report.status),
    ));

    let continuity = assess_recovery_continuity(
        &old, &new, tip, &backup, &auth, &commit, &first, 5_200,
    );
    scenarios.push(scenario(
        "reviewed_recovery_continuity_positive",
        continuity.status == TrustStoreContinuityStatus::Valid,
        format!("status={:?}", continuity.status),
    ));

    let mut wrong_first = first.clone();
    wrong_first.predecessor_checkpoint_digest = Some("blake3:other-old-tip".into());
    let wrong_first_report = assess_recovery_continuity(
        &old, &new, tip, &backup, &auth, &commit, &wrong_first, 5_200,
    );
    scenarios.push(scenario(
        "replacement_checkpoint_substitution_blocked",
        wrong_first_report.status != TrustStoreContinuityStatus::Valid,
        format!("status={:?}", wrong_first_report.status),
    ));

    let mut expired_auth = auth.clone();
    expired_auth.expires_at_ms = 4_500;
    let acceptance = accept_recovery(
        "recovery-ledger:policy-root",
        1,
        None,
        &old,
        &new,
        tip,
        &backup,
        &expired_auth,
        &commit,
        &first,
        5_300,
        "qualification:continuity",
        "blake3:continuity",
        5_200,
        vec!["audit:acceptance".into()],
    );
    scenarios.push(scenario(
        "invalid_continuity_cannot_enter_recovery_ledger",
        matches!(acceptance, Err(RecoveryAcceptanceError::ContinuityDidNotPass)),
        format!("result={acceptance:?}"),
    ));

    let r1 = acceptance_record_for_first(&first);
    let mut r2 = r1.clone();
    r2.ledger_revision = 2;
    r2.predecessor_acceptance_digest = Some(r1.record_digest());
    r2.recovery_commit_id = "commit:2".into();
    r2.previous_checkpoint_digest = "blake3:another-old-tip".into();
    r2.replacement_counter_epoch = "epoch:c".into();
    r2.first_replacement_checkpoint_digest = "blake3:first-c".into();
    r2.accepted_at_ms = 6_000;
    let replay_report = assess_recovery_ledger(&[r1.clone(), r2.clone()]);
    scenarios.push(scenario(
        "authorization_replay_rejected",
        replay_report.status == RecoveryLedgerStatus::Invalid
            && replay_report
                .issues
                .iter()
                .any(|issue| matches!(issue, RecoveryLedgerIssue::AuthorizationReused(_))),
        format!("status={:?}", replay_report.status),
    ));

    r2.authorization_id = "authorization:2".into();
    r2.previous_checkpoint_digest = r1.previous_checkpoint_digest.clone();
    let fork_report = assess_recovery_ledger(&[r1.clone(), r2]);
    scenarios.push(scenario(
        "old_checkpoint_recovery_fork_rejected",
        fork_report.status == RecoveryLedgerStatus::Invalid
            && fork_report
                .issues
                .iter()
                .any(|issue| matches!(issue, RecoveryLedgerIssue::PreviousCheckpointForked(_))),
        format!("status={:?}", fork_report.status),
    ));

    let accepted = acceptance_record_for_first(&first);
    let current_report = assess_current_trust_store_state(&new, std::slice::from_ref(&first), &[accepted.clone()]);
    scenarios.push(scenario(
        "accepted_recovered_segment_positive",
        current_report.status == CurrentTrustStoreStatus::Valid,
        format!("status={:?}", current_report.status),
    ));

    let mut wrong_acceptance = accepted;
    wrong_acceptance.first_replacement_checkpoint_digest = "blake3:other-first".into();
    let wrong_current = assess_current_trust_store_state(&new, std::slice::from_ref(&first), &[wrong_acceptance]);
    scenarios.push(scenario(
        "unaccepted_recovered_segment_rejected",
        wrong_current.status == CurrentTrustStoreStatus::Invalid,
        format!("status={:?}", wrong_current.status),
    ));

    let verification = CheckpointAttestationVerificationReceipt {
        receipt_id: "verification-receipt:first".into(),
        checkpoint_digest: first.checkpoint_digest(),
        logical_store_id: new.store_id.clone(),
        trust_store_ref: new.trust_store_ref.clone(),
        monotonic_counter_ref: new.monotonic_counter_ref.clone(),
        counter_epoch: first.counter_epoch.clone(),
        counter_value: first.counter_value,
        attestation_ref: first.attestation_ref.clone(),
        verified_by_ref: "verifier:hardware-attestation".into(),
        verification_ref: "verification:first".into(),
        verified_at_ms: 5_200,
        evidence_refs: vec!["audit:verification".into()],
    };
    scenarios.push(scenario(
        "external_checkpoint_attestation_positive",
        verification.validate_for(&new, &first),
        "exact checkpoint/profile binding",
    ));

    let mut self_verified = verification.clone();
    self_verified.verified_by_ref = new.trust_store_ref.clone();
    scenarios.push(scenario(
        "trust_store_self_verification_rejected",
        !self_verified.validate_for(&new, &first),
        "trust store cannot verify itself",
    ));

    let mut substituted = verification;
    substituted.checkpoint_digest = "blake3:other-checkpoint".into();
    scenarios.push(scenario(
        "checkpoint_attestation_substitution_rejected",
        !substituted.validate_for(&new, &first),
        "verification receipt must bind exact checkpoint digest",
    ));

    let status = if scenarios.iter().all(|scenario| scenario.passed) {
        TrustRootCrucibleStatus::Pass
    } else {
        TrustRootCrucibleStatus::Fail
    };
    TrustRootCrucibleReport { status, scenarios }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn reviewed_trust_root_crucible_passes() {
        let report = run_trust_root_crucible();
        assert_eq!(report.status, TrustRootCrucibleStatus::Pass);
        assert!(report.scenarios.iter().all(|scenario| scenario.passed));
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn scenario_ids_are_unique() {
        let report = run_trust_root_crucible();
        let ids = report
            .scenarios
            .iter()
            .map(|scenario| scenario.scenario_id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), report.scenarios.len());
    }
}
