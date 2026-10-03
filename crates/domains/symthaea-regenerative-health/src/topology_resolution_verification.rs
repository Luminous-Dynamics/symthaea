// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Verifier-result binding for topology lifecycle resolutions.
//!
//! The verifier result is kept separate from the authority statement. This
//! mirrors the RATS distinction between an asserted statement and a verifier
//! produced Attestation Result. The gate binds the result to the exact
//! resolution and checks its validity window, but does not establish the
//! verifier's trust anchor.

use serde::{Deserialize, Serialize};

use crate::topology_resolution::TopologyLifecycleResolution;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyResolutionVerificationState {
    Verified,
    InsufficientEvidence,
    Quarantined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyResolutionVerificationIssue {
    EmptyIdentity,
    WrongVerifier,
    ResolutionMismatch,
    StatementMismatch,
    VerificationReferenceMismatch,
    InvalidVerificationWindow,
    FutureVerification,
    StaleVerification,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyResolutionVerificationResult {
    pub schema_version: String,
    pub verifier_id: String,
    pub resolution_id: String,
    pub authority_statement_digest: String,
    pub verification_reference: String,
    pub verified_at_ms: u64,
    pub valid_until_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyResolutionVerificationPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_verifier_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyResolutionVerificationDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub state: TopologyResolutionVerificationState,
    pub verifier_id: Option<String>,
    pub verification_reference: Option<String>,
    pub issues: Vec<TopologyResolutionVerificationIssue>,
}

#[derive(Debug, Clone)]
pub struct TopologyResolutionVerificationGate {
    policy: TopologyResolutionVerificationPolicy,
}

impl TopologyResolutionVerificationGate {
    pub fn new(
        policy: TopologyResolutionVerificationPolicy,
    ) -> Result<Self, &'static str> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.expected_verifier_id.trim().is_empty()
        {
            return Err("invalid topology resolution verification policy");
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        resolution: &TopologyLifecycleResolution,
        verification: Option<&TopologyResolutionVerificationResult>,
        now_ms: u64,
    ) -> TopologyResolutionVerificationDecision {
        let mut issues = Vec::new();
        let Some(verification) = verification else {
            return TopologyResolutionVerificationDecision {
                schema_version: self.policy.schema_version.clone(),
                policy_id: self.policy.policy_id.clone(),
                state: TopologyResolutionVerificationState::InsufficientEvidence,
                verifier_id: None,
                verification_reference: None,
                issues: vec![],
            };
        };

        if verification.schema_version.trim().is_empty()
            || verification.verifier_id.trim().is_empty()
            || verification.resolution_id.trim().is_empty()
            || verification.authority_statement_digest.trim().is_empty()
            || verification.verification_reference.trim().is_empty()
        {
            issues.push(TopologyResolutionVerificationIssue::EmptyIdentity);
        }

        if verification.verifier_id != self.policy.expected_verifier_id {
            issues.push(TopologyResolutionVerificationIssue::WrongVerifier);
        }
        if verification.resolution_id != resolution.resolution_id {
            issues.push(TopologyResolutionVerificationIssue::ResolutionMismatch);
        }
        if verification.authority_statement_digest != resolution.authority_statement_digest {
            issues.push(TopologyResolutionVerificationIssue::StatementMismatch);
        }
        if verification.verification_reference != resolution.verification_reference {
            issues.push(TopologyResolutionVerificationIssue::VerificationReferenceMismatch);
        }
        if verification.verified_at_ms == 0 || verification.verified_at_ms > verification.valid_until_ms {
            issues.push(TopologyResolutionVerificationIssue::InvalidVerificationWindow);
        } else if verification.verified_at_ms > now_ms {
            issues.push(TopologyResolutionVerificationIssue::FutureVerification);
        } else if now_ms > verification.valid_until_ms {
            issues.push(TopologyResolutionVerificationIssue::StaleVerification);
        }

        let state = if issues.is_empty() {
            TopologyResolutionVerificationState::Verified
        } else {
            TopologyResolutionVerificationState::Quarantined
        };

        TopologyResolutionVerificationDecision {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            state,
            verifier_id: Some(verification.verifier_id.clone()),
            verification_reference: Some(verification.verification_reference.clone()),
            issues,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology_resolution::TopologyBranchReference;

    fn branch() -> TopologyBranchReference {
        TopologyBranchReference {
            predecessor_epoch: 1,
            predecessor_topology_digest: "topology-v1".into(),
            successor_epoch: 2,
            successor_topology_digest: "topology-v2a".into(),
            successor_effective_from_ms: 1_500,
        }
    }

    fn resolution() -> TopologyLifecycleResolution {
        TopologyLifecycleResolution {
            schema_version: "0.1".into(),
            asset_id: "vehicle-1".into(),
            component_id: "wing-root".into(),
            predecessor_epoch: 1,
            predecessor_topology_digest: "topology-v1".into(),
            selected_successor_epoch: 2,
            selected_successor_topology_digest: "topology-v2a".into(),
            selected_successor_effective_from_ms: 1_500,
            resolution_epoch: 1,
            predecessor_resolution_id: None,
            predecessor_resolution_digest: None,
            observed_successors: vec![branch()],
            observed_successors_digest: crate::topology_resolution::branch_set_digest(&[branch()]),
            resolution_id: "resolution-2".into(),
            authority_id: "mycelix-topology-authority".into(),
            authority_statement_digest: "resolution-digest-2".into(),
            verification_reference: "resolution-verify-2".into(),
            resolved_at_ms: 2_000,
            configuration_digest: "cfg-1".into(),
        }
    }

    fn result() -> TopologyResolutionVerificationResult {
        TopologyResolutionVerificationResult {
            schema_version: "0.1".into(),
            verifier_id: "mycelix-topology-verifier".into(),
            resolution_id: "resolution-2".into(),
            authority_statement_digest: "resolution-digest-2".into(),
            verification_reference: "resolution-verify-2".into(),
            verified_at_ms: 2_100,
            valid_until_ms: 4_000,
        }
    }

    fn gate() -> TopologyResolutionVerificationGate {
        TopologyResolutionVerificationGate::new(
            TopologyResolutionVerificationPolicy {
                schema_version: "0.1".into(),
                policy_id: "topology-resolution-verification-v1".into(),
                expected_verifier_id: "mycelix-topology-verifier".into(),
            },
        )
        .unwrap()
    }

    #[test]
    fn exact_verifier_result_is_admitted() {
        let d = gate().assess(&resolution(), Some(&result()), 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::Verified);
    }

    #[test]
    fn missing_verifier_result_is_insufficient_evidence() {
        let d = gate().assess(&resolution(), None, 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::InsufficientEvidence);
    }

    #[test]
    fn wrong_verifier_is_quarantined() {
        let mut r = result();
        r.verifier_id = "unexpected".into();
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::WrongVerifier));
    }

    #[test]
    fn verification_must_bind_exact_resolution_identity() {
        let mut r = result();
        r.resolution_id = "resolution-old".into();
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::ResolutionMismatch));
    }

    #[test]
    fn verification_must_bind_exact_statement_digest() {
        let mut r = result();
        r.authority_statement_digest = "wrong-digest".into();
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::StatementMismatch));
    }

    #[test]
    fn stale_verification_cannot_support_current_authority() {
        let mut r = result();
        r.valid_until_ms = 2_999;
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::StaleVerification));
    }

    #[test]
    fn future_verification_cannot_support_current_authority() {
        let mut r = result();
        r.verified_at_ms = 3_001;
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::FutureVerification));
    }
}
