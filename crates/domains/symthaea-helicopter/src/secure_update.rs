// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed dual-bank software update execution.
//!
//! This module does not implement a signature algorithm.  It defines the update
//! state machine and requires an injected authenticity verifier.  A package may
//! advance only when its deployment identity, compatibility set, qualification
//! evidence, digest, signer, anti-rollback counter, trial health, and commit
//! evidence all satisfy policy.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecureUpdatePackage {
    pub schema_version: String,
    pub package_id: String,
    pub deployment_id: String,
    pub software_version: String,
    pub payload_digest: String,
    pub signer_id: String,
    pub signature: Vec<u8>,
    pub anti_rollback_counter: u64,
    pub compatible_hardware_ids: Vec<String>,
    pub qualification_evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecureUpdatePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub deployment_id: String,
    pub hardware_id: String,
    pub trusted_signers: Vec<String>,
    pub minimum_qualification_evidence: usize,
    pub maximum_trial_boots: u32,
    pub require_distinct_qualification_evidence: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateBank {
    A,
    B,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecureUpdateState {
    Idle,
    Staged,
    Verified,
    PendingReboot,
    Trial,
    Committed,
    RolledBack,
    LockedOut,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecureUpdateIssue {
    InvalidPackageIdentity,
    InvalidDigest,
    DeploymentMismatch,
    HardwareIncompatible,
    UntrustedSigner,
    SignatureRejected,
    CounterRollback { observed: u64, minimum: u64 },
    InsufficientQualificationEvidence { observed: usize, required: usize },
    DuplicateQualificationEvidence(String),
    TrialHealthFailed(String),
    TrialBootLimitExceeded,
    InvalidTransition,
    MissingCommitEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrialBootReport {
    pub boot_id: String,
    pub software_version: String,
    pub startup_self_test_passed: bool,
    pub estimator_healthy: bool,
    pub realtime_healthy: bool,
    pub command_outputs_disarmed: bool,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecureUpdateEvidence {
    pub state: SecureUpdateState,
    pub active_bank: UpdateBank,
    pub candidate_bank: UpdateBank,
    pub package_id: Option<String>,
    pub active_version: String,
    pub candidate_version: Option<String>,
    pub accepted_counter: u64,
    pub trial_boots: u32,
    pub issues: Vec<SecureUpdateIssue>,
}

pub trait UpdateAuthenticityVerifier {
    fn verify(
        &self,
        signer_id: &str,
        payload_digest: &str,
        signature: &[u8],
    ) -> Result<bool, SecureUpdateError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct UnavailableUpdateVerifier;

impl UpdateAuthenticityVerifier for UnavailableUpdateVerifier {
    fn verify(
        &self,
        _signer_id: &str,
        _payload_digest: &str,
        _signature: &[u8],
    ) -> Result<bool, SecureUpdateError> {
        Err(SecureUpdateError::VerifierUnavailable)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecureUpdateError {
    InvalidPolicy,
    InvalidState,
    VerifierUnavailable,
}

#[derive(Debug, Clone)]
pub struct DualBankUpdateManager<V> {
    policy: SecureUpdatePolicy,
    verifier: V,
    state: SecureUpdateState,
    active_bank: UpdateBank,
    active_version: String,
    accepted_counter: u64,
    candidate: Option<SecureUpdatePackage>,
    candidate_bank: UpdateBank,
    trial_boots: u32,
    issues: Vec<SecureUpdateIssue>,
}

impl<V: UpdateAuthenticityVerifier> DualBankUpdateManager<V> {
    pub fn new(
        policy: SecureUpdatePolicy,
        verifier: V,
        active_bank: UpdateBank,
        active_version: String,
        accepted_counter: u64,
    ) -> Result<Self, SecureUpdateError> {
        let signers: BTreeSet<_> = policy.trusted_signers.iter().collect();
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.deployment_id.trim().is_empty()
            || policy.hardware_id.trim().is_empty()
            || policy.trusted_signers.is_empty()
            || signers.len() != policy.trusted_signers.len()
            || policy.maximum_trial_boots == 0
            || active_version.trim().is_empty()
        {
            return Err(SecureUpdateError::InvalidPolicy);
        }
        Ok(Self {
            policy,
            verifier,
            state: SecureUpdateState::Idle,
            active_bank,
            active_version,
            accepted_counter,
            candidate: None,
            candidate_bank: opposite(active_bank),
            trial_boots: 0,
            issues: Vec::new(),
        })
    }

    pub fn stage(&mut self, package: SecureUpdatePackage) -> Result<(), SecureUpdateError> {
        if !matches!(
            self.state,
            SecureUpdateState::Idle | SecureUpdateState::RolledBack | SecureUpdateState::Committed
        ) {
            return Err(SecureUpdateError::InvalidState);
        }
        self.issues.clear();
        self.trial_boots = 0;
        self.candidate_bank = opposite(self.active_bank);
        self.candidate = Some(package);
        self.state = SecureUpdateState::Staged;
        Ok(())
    }

    pub fn verify_staged(&mut self) -> Result<bool, SecureUpdateError> {
        if self.state != SecureUpdateState::Staged {
            return Err(SecureUpdateError::InvalidState);
        }
        let package = self
            .candidate
            .as_ref()
            .ok_or(SecureUpdateError::InvalidState)?;
        self.issues = self.package_issues(package)?;
        if self.issues.is_empty() {
            self.state = SecureUpdateState::Verified;
            Ok(true)
        } else {
            self.state = SecureUpdateState::LockedOut;
            Ok(false)
        }
    }

    pub fn request_activation(&mut self) -> Result<(), SecureUpdateError> {
        if self.state != SecureUpdateState::Verified {
            return Err(SecureUpdateError::InvalidState);
        }
        self.state = SecureUpdateState::PendingReboot;
        Ok(())
    }

    pub fn begin_trial_boot(&mut self) -> Result<(), SecureUpdateError> {
        if !matches!(
            self.state,
            SecureUpdateState::PendingReboot | SecureUpdateState::Trial
        ) {
            return Err(SecureUpdateError::InvalidState);
        }
        self.trial_boots = self.trial_boots.saturating_add(1);
        if self.trial_boots > self.policy.maximum_trial_boots {
            self.issues.push(SecureUpdateIssue::TrialBootLimitExceeded);
            self.rollback_internal();
            return Ok(());
        }
        self.state = SecureUpdateState::Trial;
        Ok(())
    }

    pub fn observe_trial(&mut self, report: &TrialBootReport) -> Result<bool, SecureUpdateError> {
        if self.state != SecureUpdateState::Trial {
            return Err(SecureUpdateError::InvalidState);
        }
        let package = self
            .candidate
            .as_ref()
            .ok_or(SecureUpdateError::InvalidState)?;
        let healthy = !report.boot_id.trim().is_empty()
            && report.software_version == package.software_version
            && report.startup_self_test_passed
            && report.estimator_healthy
            && report.realtime_healthy
            && report.command_outputs_disarmed
            && !report.evidence_ids.is_empty();
        if !healthy {
            self.issues
                .push(SecureUpdateIssue::TrialHealthFailed(report.boot_id.clone()));
            self.rollback_internal();
        }
        Ok(healthy)
    }

    pub fn commit(&mut self, commit_evidence_id: &str) -> Result<(), SecureUpdateError> {
        if self.state != SecureUpdateState::Trial {
            return Err(SecureUpdateError::InvalidState);
        }
        if commit_evidence_id.trim().is_empty() {
            self.issues.push(SecureUpdateIssue::MissingCommitEvidence);
            return Err(SecureUpdateError::InvalidState);
        }
        let package = self
            .candidate
            .take()
            .ok_or(SecureUpdateError::InvalidState)?;
        self.active_bank = self.candidate_bank;
        self.active_version = package.software_version;
        self.accepted_counter = package.anti_rollback_counter;
        self.state = SecureUpdateState::Committed;
        Ok(())
    }

    pub fn rollback(&mut self) -> Result<(), SecureUpdateError> {
        if !matches!(
            self.state,
            SecureUpdateState::Trial
                | SecureUpdateState::PendingReboot
                | SecureUpdateState::Verified
        ) {
            return Err(SecureUpdateError::InvalidState);
        }
        self.rollback_internal();
        Ok(())
    }

    pub fn evidence(&self) -> SecureUpdateEvidence {
        SecureUpdateEvidence {
            state: self.state,
            active_bank: self.active_bank,
            candidate_bank: self.candidate_bank,
            package_id: self
                .candidate
                .as_ref()
                .map(|package| package.package_id.clone()),
            active_version: self.active_version.clone(),
            candidate_version: self
                .candidate
                .as_ref()
                .map(|package| package.software_version.clone()),
            accepted_counter: self.accepted_counter,
            trial_boots: self.trial_boots,
            issues: self.issues.clone(),
        }
    }

    fn package_issues(
        &self,
        package: &SecureUpdatePackage,
    ) -> Result<Vec<SecureUpdateIssue>, SecureUpdateError> {
        let mut issues = Vec::new();
        if package.schema_version.trim().is_empty()
            || package.package_id.trim().is_empty()
            || package.software_version.trim().is_empty()
        {
            issues.push(SecureUpdateIssue::InvalidPackageIdentity);
        }
        if !valid_digest(&package.payload_digest) {
            issues.push(SecureUpdateIssue::InvalidDigest);
        }
        if package.deployment_id != self.policy.deployment_id {
            issues.push(SecureUpdateIssue::DeploymentMismatch);
        }
        if !package
            .compatible_hardware_ids
            .contains(&self.policy.hardware_id)
        {
            issues.push(SecureUpdateIssue::HardwareIncompatible);
        }
        if !self.policy.trusted_signers.contains(&package.signer_id) {
            issues.push(SecureUpdateIssue::UntrustedSigner);
        } else if !self.verifier.verify(
            &package.signer_id,
            &package.payload_digest,
            &package.signature,
        )? {
            issues.push(SecureUpdateIssue::SignatureRejected);
        }
        if package.anti_rollback_counter <= self.accepted_counter {
            issues.push(SecureUpdateIssue::CounterRollback {
                observed: package.anti_rollback_counter,
                minimum: self.accepted_counter.saturating_add(1),
            });
        }
        let evidence: BTreeSet<_> = package.qualification_evidence_ids.iter().collect();
        if self.policy.require_distinct_qualification_evidence
            && evidence.len() != package.qualification_evidence_ids.len()
        {
            let mut seen = BTreeSet::new();
            for id in &package.qualification_evidence_ids {
                if !seen.insert(id) {
                    issues.push(SecureUpdateIssue::DuplicateQualificationEvidence(
                        id.clone(),
                    ));
                }
            }
        }
        if evidence.len() < self.policy.minimum_qualification_evidence {
            issues.push(SecureUpdateIssue::InsufficientQualificationEvidence {
                observed: evidence.len(),
                required: self.policy.minimum_qualification_evidence,
            });
        }
        Ok(issues)
    }

    fn rollback_internal(&mut self) {
        self.state = SecureUpdateState::RolledBack;
        self.candidate = None;
        self.trial_boots = 0;
        self.candidate_bank = opposite(self.active_bank);
    }
}

fn opposite(bank: UpdateBank) -> UpdateBank {
    match bank {
        UpdateBank::A => UpdateBank::B,
        UpdateBank::B => UpdateBank::A,
    }
}

fn valid_digest(digest: &str) -> bool {
    let Some((algorithm, value)) = digest.split_once(':') else {
        return false;
    };
    !algorithm.trim().is_empty()
        && value.len() >= 16
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    struct AcceptVerifier;
    impl UpdateAuthenticityVerifier for AcceptVerifier {
        fn verify(
            &self,
            _signer_id: &str,
            _payload_digest: &str,
            signature: &[u8],
        ) -> Result<bool, SecureUpdateError> {
            Ok(signature == b"valid")
        }
    }

    fn manager() -> DualBankUpdateManager<AcceptVerifier> {
        DualBankUpdateManager::new(
            SecureUpdatePolicy {
                schema_version: "1".into(),
                policy_id: "dual-bank".into(),
                deployment_id: "aircraft-1".into(),
                hardware_id: "fc-v2".into(),
                trusted_signers: vec!["release-key".into()],
                minimum_qualification_evidence: 2,
                maximum_trial_boots: 2,
                require_distinct_qualification_evidence: true,
            },
            AcceptVerifier,
            UpdateBank::A,
            "1.0.0".into(),
            5,
        )
        .unwrap()
    }

    fn package(counter: u64) -> SecureUpdatePackage {
        SecureUpdatePackage {
            schema_version: "1".into(),
            package_id: "pkg-2".into(),
            deployment_id: "aircraft-1".into(),
            software_version: "2.0.0".into(),
            payload_digest: "sha256:0123456789abcdef".into(),
            signer_id: "release-key".into(),
            signature: b"valid".to_vec(),
            anti_rollback_counter: counter,
            compatible_hardware_ids: vec!["fc-v2".into()],
            qualification_evidence_ids: vec!["qual-a".into(), "qual-b".into()],
        }
    }

    #[test]
    fn healthy_trial_commits_new_bank() {
        let mut manager = manager();
        manager.stage(package(6)).unwrap();
        assert!(manager.verify_staged().unwrap());
        manager.request_activation().unwrap();
        manager.begin_trial_boot().unwrap();
        assert!(
            manager
                .observe_trial(&TrialBootReport {
                    boot_id: "boot-1".into(),
                    software_version: "2.0.0".into(),
                    startup_self_test_passed: true,
                    estimator_healthy: true,
                    realtime_healthy: true,
                    command_outputs_disarmed: true,
                    evidence_ids: vec!["self-test".into()],
                })
                .unwrap()
        );
        manager.commit("commit-evidence").unwrap();
        assert_eq!(manager.evidence().active_bank, UpdateBank::B);
        assert_eq!(manager.evidence().accepted_counter, 6);
    }

    #[test]
    fn rollback_counter_is_rejected() {
        let mut manager = manager();
        manager.stage(package(5)).unwrap();
        assert!(!manager.verify_staged().unwrap());
        assert_eq!(manager.evidence().state, SecureUpdateState::LockedOut);
    }

    #[test]
    fn unhealthy_trial_rolls_back() {
        let mut manager = manager();
        manager.stage(package(6)).unwrap();
        manager.verify_staged().unwrap();
        manager.request_activation().unwrap();
        manager.begin_trial_boot().unwrap();
        assert!(
            !manager
                .observe_trial(&TrialBootReport {
                    boot_id: "boot-bad".into(),
                    software_version: "2.0.0".into(),
                    startup_self_test_passed: false,
                    estimator_healthy: true,
                    realtime_healthy: true,
                    command_outputs_disarmed: true,
                    evidence_ids: vec!["self-test".into()],
                })
                .unwrap()
        );
        assert_eq!(manager.evidence().state, SecureUpdateState::RolledBack);
        assert_eq!(manager.evidence().active_version, "1.0.0");
    }
}
