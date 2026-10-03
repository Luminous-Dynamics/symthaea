// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Explicit revocation boundary for authoritative topology lifecycle decisions.
//!
//! Revocation invalidates the current authority decision without inventing a
//! replacement topology. Historical evidence remains preserved.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyRevocationState {
    Revoked,
    Quarantined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyRevocationIssue {
    EmptyIdentity,
    WrongAuthority,
    FutureRevocation,
    TargetResolutionMismatch,
    TargetStatementMismatch,
    InvalidTargetEpoch,
    ConfigurationMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyLifecycleRevocation {
    pub schema_version: String,
    pub asset_id: String,
    pub component_id: String,
    pub target_resolution_epoch: u64,
    pub target_resolution_id: String,
    pub target_authority_statement_digest: String,
    pub revocation_id: String,
    pub authority_id: String,
    pub authority_statement_digest: String,
    pub verification_reference: String,
    pub revoked_at_ms: u64,
    pub reason_code: String,
    pub configuration_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyRevocationPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_asset_id: String,
    pub expected_component_id: String,
    pub expected_configuration_digest: String,
    pub expected_authority_id: String,
    pub expected_current_resolution_epoch: u64,
    pub expected_current_resolution_id: String,
    pub expected_current_authority_statement_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyRevocationDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub state: TopologyRevocationState,
    pub revocation_id: String,
    pub target_resolution_epoch: u64,
    pub target_resolution_id: String,
    pub authority_statement_digest: String,
    pub issues: Vec<TopologyRevocationIssue>,
}

#[derive(Debug, Clone)]
pub struct TopologyRevocationGate {
    policy: TopologyRevocationPolicy,
}

impl TopologyRevocationGate {
    pub fn new(policy: TopologyRevocationPolicy) -> Result<Self, &'static str> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.expected_asset_id.trim().is_empty()
            || policy.expected_component_id.trim().is_empty()
            || policy.expected_configuration_digest.trim().is_empty()
            || policy.expected_authority_id.trim().is_empty()
            || policy.expected_current_resolution_epoch == 0
            || policy.expected_current_resolution_id.trim().is_empty()
            || policy.expected_current_authority_statement_digest.trim().is_empty()
        {
            return Err("invalid topology revocation policy");
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        revocation: &TopologyLifecycleRevocation,
        now_ms: u64,
    ) -> TopologyRevocationDecision {
        let mut issues = Vec::new();

        if revocation.schema_version.trim().is_empty()
            || revocation.asset_id.trim().is_empty()
            || revocation.component_id.trim().is_empty()
            || revocation.target_resolution_id.trim().is_empty()
            || revocation.target_authority_statement_digest.trim().is_empty()
            || revocation.revocation_id.trim().is_empty()
            || revocation.authority_id.trim().is_empty()
            || revocation.authority_statement_digest.trim().is_empty()
            || revocation.verification_reference.trim().is_empty()
            || revocation.reason_code.trim().is_empty()
            || revocation.configuration_digest.trim().is_empty()
        {
            issues.push(TopologyRevocationIssue::EmptyIdentity);
        }

        if revocation.asset_id != self.policy.expected_asset_id
            || revocation.component_id != self.policy.expected_component_id
            || revocation.configuration_digest != self.policy.expected_configuration_digest
        {
            issues.push(TopologyRevocationIssue::ConfigurationMismatch);
        }

        if revocation.authority_id != self.policy.expected_authority_id {
            issues.push(TopologyRevocationIssue::WrongAuthority);
        }

        if revocation.revoked_at_ms == 0 || revocation.revoked_at_ms > now_ms {
            issues.push(TopologyRevocationIssue::FutureRevocation);
        }

        if revocation.target_resolution_epoch == 0 {
            issues.push(TopologyRevocationIssue::InvalidTargetEpoch);
        } else if revocation.target_resolution_epoch
            != self.policy.expected_current_resolution_epoch
        {
            issues.push(TopologyRevocationIssue::TargetResolutionMismatch);
        }

        if revocation.target_resolution_id != self.policy.expected_current_resolution_id {
            issues.push(TopologyRevocationIssue::TargetResolutionMismatch);
        }

        if revocation.target_authority_statement_digest
            != self.policy.expected_current_authority_statement_digest
        {
            issues.push(TopologyRevocationIssue::TargetStatementMismatch);
        }

        let state = if issues.is_empty() {
            TopologyRevocationState::Revoked
        } else {
            TopologyRevocationState::Quarantined
        };

        TopologyRevocationDecision {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            state,
            revocation_id: revocation.revocation_id.clone(),
            target_resolution_epoch: revocation.target_resolution_epoch,
            target_resolution_id: revocation.target_resolution_id.clone(),
            authority_statement_digest: revocation.authority_statement_digest.clone(),
            issues,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> TopologyRevocationGate {
        TopologyRevocationGate::new(TopologyRevocationPolicy {
            schema_version: "0.1".into(),
            policy_id: "topology-revocation-v1".into(),
            expected_asset_id: "vehicle-1".into(),
            expected_component_id: "wing-root".into(),
            expected_configuration_digest: "cfg-1".into(),
            expected_authority_id: "mycelix-topology-authority".into(),
            expected_current_resolution_epoch: 1,
            expected_current_resolution_id: "resolution-2".into(),
            expected_current_authority_statement_digest: "resolution-digest-2".into(),
        })
        .unwrap()
    }

    fn revocation() -> TopologyLifecycleRevocation {
        TopologyLifecycleRevocation {
            schema_version: "0.1".into(),
            asset_id: "vehicle-1".into(),
            component_id: "wing-root".into(),
            target_resolution_epoch: 1,
            target_resolution_id: "resolution-2".into(),
            target_authority_statement_digest: "resolution-digest-2".into(),
            revocation_id: "revocation-2".into(),
            authority_id: "mycelix-topology-authority".into(),
            authority_statement_digest: "revocation-digest-2".into(),
            verification_reference: "revocation-verify-2".into(),
            revoked_at_ms: 3_000,
            reason_code: "authority-correction".into(),
            configuration_digest: "cfg-1".into(),
        }
    }

    #[test]
    fn current_resolution_can_be_explicitly_revoked() {
        let d = gate().assess(&revocation(), 3_001);
        assert_eq!(d.state, TopologyRevocationState::Revoked);
        assert_eq!(d.revocation_id, "revocation-2");
    }

    #[test]
    fn revocation_cannot_target_an_older_resolution() {
        let mut r = revocation();
        r.target_resolution_epoch = 0;
        let d = gate().assess(&r, 3_001);
        assert_eq!(d.state, TopologyRevocationState::Quarantined);
        assert!(d.issues.contains(&TopologyRevocationIssue::InvalidTargetEpoch));
    }

    #[test]
    fn revocation_must_target_exact_current_resolution_identity() {
        let mut r = revocation();
        r.target_resolution_id = "resolution-old".into();
        let d = gate().assess(&r, 3_001);
        assert_eq!(d.state, TopologyRevocationState::Quarantined);
        assert!(d.issues.contains(&TopologyRevocationIssue::TargetResolutionMismatch));
    }

    #[test]
    fn revocation_must_bind_exact_authority_statement() {
        let mut r = revocation();
        r.target_authority_statement_digest = "wrong-digest".into();
        let d = gate().assess(&r, 3_001);
        assert_eq!(d.state, TopologyRevocationState::Quarantined);
        assert!(d.issues.contains(&TopologyRevocationIssue::TargetStatementMismatch));
    }

    #[test]
    fn unexpected_authority_cannot_revoke_current_state() {
        let mut r = revocation();
        r.authority_id = "unexpected-authority".into();
        let d = gate().assess(&r, 3_001);
        assert_eq!(d.state, TopologyRevocationState::Quarantined);
        assert!(d.issues.contains(&TopologyRevocationIssue::WrongAuthority));
    }

    #[test]
    fn future_revocation_cannot_change_current_state() {
        let mut r = revocation();
        r.revoked_at_ms = 3_002;
        let d = gate().assess(&r, 3_001);
        assert_eq!(d.state, TopologyRevocationState::Quarantined);
        assert!(d.issues.contains(&TopologyRevocationIssue::FutureRevocation));
    }
}
