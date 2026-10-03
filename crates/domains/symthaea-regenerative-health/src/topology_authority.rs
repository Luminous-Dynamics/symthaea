// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic composition boundary for topology authority state.
//!
//! Resolution establishes an externally authored topology decision. Revocation
//! can invalidate that decision. This module composes those gates so callers
//! cannot accidentally treat a revoked resolution as current.

use serde::{Deserialize, Serialize};

use crate::topology_lifecycle::TopologyBranchReference;
use crate::topology_resolution::{
    TopologyLifecycleResolution, TopologyResolutionDecision, TopologyResolutionGate,
    TopologyResolutionIssue, TopologyResolutionPolicy, TopologyResolutionState,
};
use crate::topology_revocation::{
    TopologyLifecycleRevocation, TopologyRevocationDecision, TopologyRevocationGate,
    TopologyRevocationIssue, TopologyRevocationPolicy, TopologyRevocationState,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyAuthorityState {
    Current,
    InsufficientEvidence,
    Conflicted,
    Revoked,
    Quarantined,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyAuthorityIssue {
    Resolution(TopologyResolutionIssue),
    Revocation(TopologyRevocationIssue),
    RevocationWithoutResolvedAuthority,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyAuthorityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub resolution_policy: TopologyResolutionPolicy,
    pub revocation_policy: TopologyRevocationPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyAuthorityDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub state: TopologyAuthorityState,
    pub resolution_id: Option<String>,
    pub resolution_epoch: Option<u64>,
    pub preserved_branches: Vec<TopologyBranchReference>,
    pub revocation_id: Option<String>,
    pub issues: Vec<TopologyAuthorityIssue>,
}

#[derive(Debug, Clone)]
pub struct TopologyAuthorityGate {
    policy: TopologyAuthorityPolicy,
    resolution_gate: TopologyResolutionGate,
    revocation_gate: TopologyRevocationGate,
}

impl TopologyAuthorityGate {
    pub fn new(policy: TopologyAuthorityPolicy) -> Result<Self, &'static str> {
        if policy.schema_version.trim().is_empty() || policy.policy_id.trim().is_empty() {
            return Err("invalid topology authority policy");
        }

        Ok(Self {
            resolution_gate: TopologyResolutionGate::new(policy.resolution_policy.clone())?,
            revocation_gate: TopologyRevocationGate::new(policy.revocation_policy.clone())?,
            policy,
        })
    }

    pub fn assess(
        &self,
        resolution: Option<&TopologyLifecycleResolution>,
        revocation: Option<&TopologyLifecycleRevocation>,
        observed_successors: &[TopologyBranchReference],
        now_ms: u64,
    ) -> TopologyAuthorityDecision {
        let resolution_decision =
            self.resolution_gate
                .assess(resolution, observed_successors, now_ms);

        let mut issues = resolution_decision
            .issues
            .iter()
            .cloned()
            .map(TopologyAuthorityIssue::Resolution)
            .collect::<Vec<_>>();

        let base = TopologyAuthorityDecision {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            state: TopologyAuthorityState::Quarantined,
            resolution_id: resolution_decision.resolution_id.clone(),
            resolution_epoch: resolution_decision.resolution_epoch,
            preserved_branches: resolution_decision.preserved_branches.clone(),
            revocation_id: revocation.map(|item| item.revocation_id.clone()),
            issues: Vec::new(),
        };

        match resolution_decision.state {
            TopologyResolutionState::InsufficientEvidence => {
                if revocation.is_some() {
                    issues.push(TopologyAuthorityIssue::RevocationWithoutResolvedAuthority);
                }
                TopologyAuthorityDecision {
                    state: TopologyAuthorityState::InsufficientEvidence,
                    issues,
                    ..base
                }
            }
            TopologyResolutionState::Conflicted => {
                if revocation.is_some() {
                    issues.push(TopologyAuthorityIssue::RevocationWithoutResolvedAuthority);
                }
                TopologyAuthorityDecision {
                    state: TopologyAuthorityState::Conflicted,
                    issues,
                    ..base
                }
            }
            TopologyResolutionState::Quarantined => TopologyAuthorityDecision {
                issues,
                ..base
            },
            TopologyResolutionState::Resolved => {
                let Some(revocation) = revocation else {
                    return TopologyAuthorityDecision {
                        state: TopologyAuthorityState::Current,
                        issues,
                        ..base
                    };
                };

                let revocation_decision = self.revocation_gate.assess(revocation, now_ms);
                issues.extend(
                    revocation_decision
                        .issues
                        .iter()
                        .cloned()
                        .map(TopologyAuthorityIssue::Revocation),
                );

                match revocation_decision.state {
                    TopologyRevocationState::Revoked => TopologyAuthorityDecision {
                        state: TopologyAuthorityState::Revoked,
                        issues,
                        ..base
                    },
                    TopologyRevocationState::Quarantined => TopologyAuthorityDecision {
                        state: TopologyAuthorityState::Quarantined,
                        issues,
                        ..base
                    },
                }
            }
        }
    }

    #[allow(dead_code)]
    fn evaluate_components(
        &self,
        resolution: Option<&TopologyLifecycleResolution>,
        revocation: Option<&TopologyLifecycleRevocation>,
        observed_successors: &[TopologyBranchReference],
        now_ms: u64,
    ) -> (
        TopologyResolutionDecision,
        Option<TopologyRevocationDecision>,
    ) {
        let resolution_decision =
            self.resolution_gate
                .assess(resolution, observed_successors, now_ms);
        let revocation_decision = revocation.map(|item| self.revocation_gate.assess(item, now_ms));
        (resolution_decision, revocation_decision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn branch(digest: &str) -> TopologyBranchReference {
        TopologyBranchReference {
            predecessor_epoch: 1,
            predecessor_topology_digest: "topology-v1".into(),
            successor_epoch: 2,
            successor_topology_digest: digest.into(),
            successor_effective_from_ms: 1_500,
        }
    }

    fn resolution(selected: &str, observed: Vec<TopologyBranchReference>) -> TopologyLifecycleResolution {
        TopologyLifecycleResolution {
            schema_version: "0.1".into(),
            asset_id: "vehicle-1".into(),
            component_id: "wing-root".into(),
            predecessor_epoch: 1,
            predecessor_topology_digest: "topology-v1".into(),
            selected_successor_epoch: 2,
            selected_successor_topology_digest: selected.into(),
            selected_successor_effective_from_ms: 1_500,
            resolution_epoch: 1,
            predecessor_resolution_digest: None,
            observed_successors: observed.clone(),
            observed_successors_digest: crate::topology_resolution::branch_set_digest(&observed),
            resolution_id: "resolution-2".into(),
            authority_id: "mycelix-topology-authority".into(),
            authority_statement_digest: "resolution-digest-2".into(),
            verification_reference: "resolution-verify-2".into(),
            resolved_at_ms: 2_000,
            configuration_digest: "cfg-1".into(),
        }
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

    fn gate() -> TopologyAuthorityGate {
        TopologyAuthorityGate::new(TopologyAuthorityPolicy {
            schema_version: "0.1".into(),
            policy_id: "topology-authority-v1".into(),
            resolution_policy: TopologyResolutionPolicy {
                schema_version: "0.1".into(),
                policy_id: "topology-resolution-v1".into(),
                expected_asset_id: "vehicle-1".into(),
                expected_component_id: "wing-root".into(),
                expected_configuration_digest: "cfg-1".into(),
                expected_authority_id: "mycelix-topology-authority".into(),
                expected_resolution_epoch: 1,
                expected_predecessor_resolution_digest: None,
            },
            revocation_policy: TopologyRevocationPolicy {
                schema_version: "0.1".into(),
                policy_id: "topology-revocation-v1".into(),
                expected_asset_id: "vehicle-1".into(),
                expected_component_id: "wing-root".into(),
                expected_configuration_digest: "cfg-1".into(),
                expected_authority_id: "mycelix-topology-authority".into(),
                expected_current_resolution_epoch: 1,
                expected_current_resolution_id: "resolution-2".into(),
                expected_current_authority_statement_digest: "resolution-digest-2".into(),
            },
        })
        .unwrap()
    }

    #[test]
    fn resolved_without_revocation_is_current() {
        let a = branch("topology-v2a");
        let r = resolution("topology-v2a", vec![a.clone()]);
        let d = gate().assess(Some(&r), None, &[a], 3_000);
        assert_eq!(d.state, TopologyAuthorityState::Current);
    }

    #[test]
    fn valid_revocation_overrides_resolved_authority() {
        let a = branch("topology-v2a");
        let r = resolution("topology-v2a", vec![a.clone()]);
        let d = gate().assess(Some(&r), Some(&revocation()), &[a], 3_001);
        assert_eq!(d.state, TopologyAuthorityState::Revoked);
        assert_eq!(d.revocation_id.as_deref(), Some("revocation-2"));
    }

    #[test]
    fn invalid_revocation_quarantines_current_authority() {
        let a = branch("topology-v2a");
        let r = resolution("topology-v2a", vec![a.clone()]);
        let mut revoke = revocation();
        revoke.target_resolution_id = "resolution-old".into();
        let d = gate().assess(Some(&r), Some(&revoke), &[a], 3_001);
        assert_eq!(d.state, TopologyAuthorityState::Quarantined);
        assert!(d.issues.iter().any(|issue| matches!(
            issue,
            TopologyAuthorityIssue::Revocation(
                TopologyRevocationIssue::TargetResolutionMismatch
            )
        )));
    }

    #[test]
    fn revocation_without_resolved_authority_does_not_create_authority() {
        let a = branch("topology-v2a");
        let d = gate().assess(None, Some(&revocation()), &[a], 3_001);
        assert_eq!(d.state, TopologyAuthorityState::InsufficientEvidence);
        assert!(d
            .issues
            .contains(&TopologyAuthorityIssue::RevocationWithoutResolvedAuthority));
    }

    #[test]
    fn unresolved_fork_remains_conflicted_even_with_a_revocation_claim() {
        let a = branch("topology-v2a");
        let b = branch("topology-v2b");
        let d = gate().assess(None, Some(&revocation()), &[a, b], 3_001);
        assert_eq!(d.state, TopologyAuthorityState::Conflicted);
        assert!(d
            .issues
            .contains(&TopologyAuthorityIssue::RevocationWithoutResolvedAuthority));
    }

    #[test]
    fn invalid_resolution_cannot_be_rescued_by_revocation() {
        let a = branch("topology-v2a");
        let mut r = resolution("topology-v2a", vec![a.clone()]);
        r.authority_id = "unexpected-authority".into();
        let d = gate().assess(Some(&r), Some(&revocation()), &[a], 3_001);
        assert_eq!(d.state, TopologyAuthorityState::Quarantined);
        assert!(d
            .issues
            .iter()
            .any(|issue| matches!(issue, TopologyAuthorityIssue::Resolution(
                TopologyResolutionIssue::WrongAuthority
            ))));
    }
}
