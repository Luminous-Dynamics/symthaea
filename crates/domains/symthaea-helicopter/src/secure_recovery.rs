// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Secure emergency recovery authority.
//!
//! Recovery is intentionally narrower than normal command authority. It is
//! ground-only, short-lived, multi-party, action-scoped, and externally
//! authenticated. This module validates evidence; it does not implement
//! cryptography or physical-presence sensing.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecoveryAction {
    InspectIdentity,
    ExportEvidence,
    RestoreTrustedClock,
    RebindQualifiedIdentity,
    RollbackQualifiedBank,
    ClearVerifiedLockout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecoveryApprovalRole {
    Safety,
    Security,
    Operations,
    IndependentVerifier,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryApproval {
    pub approver_id: String,
    pub organization_id: String,
    pub role: RecoveryApprovalRole,
    pub approved_actions: BTreeSet<RecoveryAction>,
    pub approved_at_ms: u64,
    pub authenticity_reference: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecureRecoveryRequest {
    pub request_id: String,
    pub aircraft_id: String,
    pub deployment_digest: String,
    pub challenge_digest: String,
    pub requested_actions: BTreeSet<RecoveryAction>,
    pub requested_at_ms: u64,
    pub expires_at_ms: u64,
    pub physical_presence_evidence_id: Option<String>,
    pub aircraft_grounded: bool,
    pub outputs_disarmed: bool,
    pub approvals: Vec<RecoveryApproval>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecureRecoveryPolicy {
    pub aircraft_id: String,
    pub qualified_deployment_digests: BTreeSet<String>,
    pub allowed_actions: BTreeSet<RecoveryAction>,
    pub required_roles: BTreeSet<RecoveryApprovalRole>,
    pub minimum_organizations: usize,
    pub maximum_request_age_ms: u64,
    pub maximum_authority_duration_ms: u64,
    pub require_physical_presence: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecureRecoveryStatus {
    Authorized,
    Denied,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecureRecoveryIssue {
    EmptyIdentity,
    AircraftMismatch,
    UnqualifiedDeployment,
    MissingAction,
    ActionNotAllowed(RecoveryAction),
    RequestNotYetValid,
    RequestExpired,
    RequestTooOld,
    AuthorityWindowTooLong,
    AircraftNotGrounded,
    OutputsStillArmed,
    MissingPhysicalPresence,
    DuplicateApprover(String),
    MissingApprovalRole(RecoveryApprovalRole),
    InsufficientOrganizations {
        required: usize,
        observed: usize,
    },
    ApprovalOutsideRequest(String),
    MissingAuthenticity(String),
    ApprovalDoesNotCoverAction {
        approver_id: String,
        action: RecoveryAction,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecureRecoveryDecision {
    pub status: SecureRecoveryStatus,
    pub request_id: String,
    pub authorized_actions: Vec<RecoveryAction>,
    pub valid_until_ms: Option<u64>,
    pub issues: Vec<SecureRecoveryIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecureRecoveryError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct SecureRecoveryGate {
    policy: SecureRecoveryPolicy,
}

impl SecureRecoveryGate {
    pub fn new(policy: SecureRecoveryPolicy) -> Result<Self, SecureRecoveryError> {
        if policy.aircraft_id.trim().is_empty()
            || policy.qualified_deployment_digests.is_empty()
            || policy.allowed_actions.is_empty()
            || policy.required_roles.is_empty()
            || policy.minimum_organizations == 0
            || policy.maximum_request_age_ms == 0
            || policy.maximum_authority_duration_ms == 0
        {
            return Err(SecureRecoveryError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(&self, request: &SecureRecoveryRequest, now_ms: u64) -> SecureRecoveryDecision {
        let mut issues = Vec::new();
        if [
            request.request_id.as_str(),
            request.aircraft_id.as_str(),
            request.deployment_digest.as_str(),
            request.challenge_digest.as_str(),
        ]
        .iter()
        .any(|value| value.trim().is_empty())
        {
            issues.push(SecureRecoveryIssue::EmptyIdentity);
        }
        if request.aircraft_id != self.policy.aircraft_id {
            issues.push(SecureRecoveryIssue::AircraftMismatch);
        }
        if !self
            .policy
            .qualified_deployment_digests
            .contains(&request.deployment_digest)
        {
            issues.push(SecureRecoveryIssue::UnqualifiedDeployment);
        }
        if request.requested_actions.is_empty() {
            issues.push(SecureRecoveryIssue::MissingAction);
        }
        for action in &request.requested_actions {
            if !self.policy.allowed_actions.contains(action) {
                issues.push(SecureRecoveryIssue::ActionNotAllowed(*action));
            }
        }
        if now_ms < request.requested_at_ms {
            issues.push(SecureRecoveryIssue::RequestNotYetValid);
        }
        if now_ms > request.expires_at_ms {
            issues.push(SecureRecoveryIssue::RequestExpired);
        }
        if now_ms.saturating_sub(request.requested_at_ms) > self.policy.maximum_request_age_ms {
            issues.push(SecureRecoveryIssue::RequestTooOld);
        }
        if request
            .expires_at_ms
            .saturating_sub(request.requested_at_ms)
            > self.policy.maximum_authority_duration_ms
        {
            issues.push(SecureRecoveryIssue::AuthorityWindowTooLong);
        }
        if !request.aircraft_grounded {
            issues.push(SecureRecoveryIssue::AircraftNotGrounded);
        }
        if !request.outputs_disarmed {
            issues.push(SecureRecoveryIssue::OutputsStillArmed);
        }
        if self.policy.require_physical_presence
            && request
                .physical_presence_evidence_id
                .as_ref()
                .is_none_or(|id| id.trim().is_empty())
        {
            issues.push(SecureRecoveryIssue::MissingPhysicalPresence);
        }

        let mut approvers = BTreeSet::new();
        let mut roles = BTreeSet::new();
        let mut organizations = BTreeSet::new();
        for approval in &request.approvals {
            if !approvers.insert(approval.approver_id.as_str()) {
                issues.push(SecureRecoveryIssue::DuplicateApprover(
                    approval.approver_id.clone(),
                ));
            }
            roles.insert(approval.role);
            organizations.insert(approval.organization_id.as_str());
            if approval.approved_at_ms < request.requested_at_ms
                || approval.approved_at_ms > request.expires_at_ms
            {
                issues.push(SecureRecoveryIssue::ApprovalOutsideRequest(
                    approval.approver_id.clone(),
                ));
            }
            if approval.authenticity_reference.trim().is_empty() {
                issues.push(SecureRecoveryIssue::MissingAuthenticity(
                    approval.approver_id.clone(),
                ));
            }
            for action in &request.requested_actions {
                if !approval.approved_actions.contains(action) {
                    issues.push(SecureRecoveryIssue::ApprovalDoesNotCoverAction {
                        approver_id: approval.approver_id.clone(),
                        action: *action,
                    });
                }
            }
        }
        for role in &self.policy.required_roles {
            if !roles.contains(role) {
                issues.push(SecureRecoveryIssue::MissingApprovalRole(*role));
            }
        }
        if organizations.len() < self.policy.minimum_organizations {
            issues.push(SecureRecoveryIssue::InsufficientOrganizations {
                required: self.policy.minimum_organizations,
                observed: organizations.len(),
            });
        }

        let status = if issues.iter().any(is_denial) {
            SecureRecoveryStatus::Denied
        } else if issues.is_empty() {
            SecureRecoveryStatus::Authorized
        } else {
            SecureRecoveryStatus::Incomplete
        };
        SecureRecoveryDecision {
            status,
            request_id: request.request_id.clone(),
            authorized_actions: if status == SecureRecoveryStatus::Authorized {
                request.requested_actions.iter().copied().collect()
            } else {
                Vec::new()
            },
            valid_until_ms: (status == SecureRecoveryStatus::Authorized)
                .then_some(request.expires_at_ms),
            issues,
        }
    }
}

fn is_denial(issue: &SecureRecoveryIssue) -> bool {
    matches!(
        issue,
        SecureRecoveryIssue::AircraftMismatch
            | SecureRecoveryIssue::UnqualifiedDeployment
            | SecureRecoveryIssue::ActionNotAllowed(_)
            | SecureRecoveryIssue::RequestExpired
            | SecureRecoveryIssue::RequestTooOld
            | SecureRecoveryIssue::AuthorityWindowTooLong
            | SecureRecoveryIssue::AircraftNotGrounded
            | SecureRecoveryIssue::OutputsStillArmed
            | SecureRecoveryIssue::ApprovalOutsideRequest(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> SecureRecoveryPolicy {
        SecureRecoveryPolicy {
            aircraft_id: "aircraft-1".into(),
            qualified_deployment_digests: BTreeSet::from(["sha256:qualified".into()]),
            allowed_actions: BTreeSet::from([
                RecoveryAction::InspectIdentity,
                RecoveryAction::RollbackQualifiedBank,
            ]),
            required_roles: BTreeSet::from([
                RecoveryApprovalRole::Safety,
                RecoveryApprovalRole::Security,
            ]),
            minimum_organizations: 2,
            maximum_request_age_ms: 1_000,
            maximum_authority_duration_ms: 2_000,
            require_physical_presence: true,
        }
    }

    fn request() -> SecureRecoveryRequest {
        let actions = BTreeSet::from([RecoveryAction::RollbackQualifiedBank]);
        SecureRecoveryRequest {
            request_id: "recovery-1".into(),
            aircraft_id: "aircraft-1".into(),
            deployment_digest: "sha256:qualified".into(),
            challenge_digest: "sha256:challenge".into(),
            requested_actions: actions.clone(),
            requested_at_ms: 1_000,
            expires_at_ms: 2_000,
            physical_presence_evidence_id: Some("presence-1".into()),
            aircraft_grounded: true,
            outputs_disarmed: true,
            approvals: vec![
                RecoveryApproval {
                    approver_id: "safety".into(),
                    organization_id: "operator".into(),
                    role: RecoveryApprovalRole::Safety,
                    approved_actions: actions.clone(),
                    approved_at_ms: 1_100,
                    authenticity_reference: "sig:safety".into(),
                },
                RecoveryApproval {
                    approver_id: "security".into(),
                    organization_id: "lab".into(),
                    role: RecoveryApprovalRole::Security,
                    approved_actions: actions,
                    approved_at_ms: 1_200,
                    authenticity_reference: "sig:security".into(),
                },
            ],
        }
    }

    #[test]
    fn grounded_multi_party_recovery_is_authorized() {
        let decision = SecureRecoveryGate::new(policy())
            .unwrap()
            .assess(&request(), 1_300);
        assert_eq!(decision.status, SecureRecoveryStatus::Authorized);
    }

    #[test]
    fn airborne_recovery_is_denied() {
        let mut request = request();
        request.aircraft_grounded = false;
        let decision = SecureRecoveryGate::new(policy())
            .unwrap()
            .assess(&request, 1_300);
        assert_eq!(decision.status, SecureRecoveryStatus::Denied);
    }
}
