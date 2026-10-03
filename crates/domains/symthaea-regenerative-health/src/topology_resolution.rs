// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic boundary for authoritative topology-lifecycle resolution.
//!
//! A fork is never resolved by local arrival order, timestamp, or majority.
//! An external authority may issue a resolution that selects one observed
//! successor while explicitly retaining the competing branches as history.
//! This module validates that resolution record without claiming to verify
//! its cryptographic authority.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyResolutionState {
    Resolved,
    InsufficientEvidence,
    Conflicted,
    Quarantined,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyResolutionIssue {
    EmptyInput,
    EmptyIdentity,
    WrongAuthority,
    FutureResolution,
    PredecessorMismatch,
    SuccessorNotObserved,
    SuccessorEpochMismatch,
    MissingPreservedBranch,
    DuplicateObservedBranch,
    InconsistentObservedPredecessor,
    ConfigurationMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TopologyBranchReference {
    pub predecessor_epoch: u64,
    pub predecessor_topology_digest: String,
    pub successor_epoch: u64,
    pub successor_topology_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyLifecycleResolution {
    pub schema_version: String,
    pub asset_id: String,
    pub component_id: String,
    pub predecessor_epoch: u64,
    pub predecessor_topology_digest: String,
    pub selected_successor_epoch: u64,
    pub selected_successor_topology_digest: String,
    /// Every competing successor considered by the authority, including the
    /// selected branch. Keeping these references makes the resolution
    /// append-only rather than destructive.
    pub observed_successors: Vec<TopologyBranchReference>,
    /// Stable authority-issued resolution statement identity.
    pub resolution_id: String,
    pub authority_id: String,
    pub authority_statement_digest: String,
    pub verification_reference: String,
    pub resolved_at_ms: u64,
    pub configuration_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyResolutionPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_asset_id: String,
    pub expected_component_id: String,
    pub expected_configuration_digest: String,
    pub expected_authority_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyResolutionDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub state: TopologyResolutionState,
    pub selected_successor: Option<TopologyBranchReference>,
    /// Branches intentionally retained as historical evidence. A resolved
    /// lifecycle does not erase competing statements.
    pub preserved_branches: Vec<TopologyBranchReference>,
    pub resolution_id: Option<String>,
    pub issues: Vec<TopologyResolutionIssue>,
}

#[derive(Debug, Clone)]
pub struct TopologyResolutionGate {
    policy: TopologyResolutionPolicy,
}

impl TopologyResolutionGate {
    pub fn new(policy: TopologyResolutionPolicy) -> Result<Self, &'static str> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.expected_asset_id.trim().is_empty()
            || policy.expected_component_id.trim().is_empty()
            || policy.expected_configuration_digest.trim().is_empty()
            || policy.expected_authority_id.trim().is_empty()
        {
            return Err("invalid topology resolution policy");
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        resolution: Option<&TopologyLifecycleResolution>,
        observed_successors: &[TopologyBranchReference],
        now_ms: u64,
    ) -> TopologyResolutionDecision {
        let mut issues = Vec::new();

        if observed_successors.is_empty() {
            issues.push(TopologyResolutionIssue::EmptyInput);
        }

        let mut unique_observed = BTreeSet::new();
        for branch in observed_successors {
            if !unique_observed.insert(branch.clone()) {
                issues.push(TopologyResolutionIssue::DuplicateObservedBranch);
            }
        }

        let predecessors: BTreeSet<_> = unique_observed
            .iter()
            .map(|branch| (branch.predecessor_epoch, branch.predecessor_topology_digest.as_str()))
            .collect();
        if predecessors.len() > 1 {
            issues.push(TopologyResolutionIssue::InconsistentObservedPredecessor);
        }

        let Some(resolution) = resolution else {
            return TopologyResolutionDecision {
                schema_version: self.policy.schema_version.clone(),
                policy_id: self.policy.policy_id.clone(),
                state: if observed_successors.is_empty() {
                    TopologyResolutionState::InsufficientEvidence
                } else if unique_observed
                    .iter()
                    .map(|branch| {
                        (
                            branch.predecessor_epoch,
                            branch.predecessor_topology_digest.as_str(),
                            branch.successor_topology_digest.as_str(),
                        )
                    })
                    .collect::<BTreeSet<_>>()
                    .len()
                    > 1
                {
                    TopologyResolutionState::Conflicted
                } else {
                    TopologyResolutionState::InsufficientEvidence
                },
                selected_successor: None,
                preserved_branches: unique_observed.into_iter().collect(),
                resolution_id: None,
                issues,
            };
        };

        if resolution.schema_version.trim().is_empty()
            || resolution.asset_id.trim().is_empty()
            || resolution.component_id.trim().is_empty()
            || resolution.predecessor_topology_digest.trim().is_empty()
            || resolution.selected_successor_topology_digest.trim().is_empty()
            || resolution.resolution_id.trim().is_empty()
            || resolution.authority_id.trim().is_empty()
            || resolution.authority_statement_digest.trim().is_empty()
            || resolution.verification_reference.trim().is_empty()
            || resolution.configuration_digest.trim().is_empty()
        {
            issues.push(TopologyResolutionIssue::EmptyIdentity);
        }

        if resolution.asset_id != self.policy.expected_asset_id
            || resolution.component_id != self.policy.expected_component_id
            || resolution.configuration_digest != self.policy.expected_configuration_digest
        {
            issues.push(TopologyResolutionIssue::ConfigurationMismatch);
        }

        if resolution.authority_id != self.policy.expected_authority_id {
            issues.push(TopologyResolutionIssue::WrongAuthority);
        }

        if resolution.resolved_at_ms == 0 || resolution.resolved_at_ms > now_ms {
            issues.push(TopologyResolutionIssue::FutureResolution);
        }

        if resolution.predecessor_epoch == 0
            || resolution.predecessor_topology_digest.trim().is_empty()
            || resolution.selected_successor_epoch
                != resolution.predecessor_epoch.saturating_add(1)
        {
            issues.push(TopologyResolutionIssue::SuccessorEpochMismatch);
        }

        for branch in &unique_observed {
            if branch.predecessor_epoch == resolution.predecessor_epoch
                && branch.predecessor_topology_digest == resolution.predecessor_topology_digest
            {
                continue;
            }
            issues.push(TopologyResolutionIssue::PredecessorMismatch);
            break;
        }

        let selected = TopologyBranchReference {
            predecessor_epoch: resolution.predecessor_epoch,
            predecessor_topology_digest: resolution.predecessor_topology_digest.clone(),
            successor_epoch: resolution.selected_successor_epoch,
            successor_topology_digest: resolution.selected_successor_topology_digest.clone(),
        };

        if !unique_observed.contains(&selected) {
            issues.push(TopologyResolutionIssue::SuccessorNotObserved);
        }

        let declared_preserved: BTreeSet<_> =
            resolution.observed_successors.iter().cloned().collect();
        if !declared_preserved.contains(&selected) {
            issues.push(TopologyResolutionIssue::MissingPreservedBranch);
        }

        for branch in &unique_observed {
            if !declared_preserved.contains(branch) {
                issues.push(TopologyResolutionIssue::MissingPreservedBranch);
                break;
            }
        }

        if issues.is_empty() {
            TopologyResolutionDecision {
                schema_version: self.policy.schema_version.clone(),
                policy_id: self.policy.policy_id.clone(),
                state: TopologyResolutionState::Resolved,
                selected_successor: Some(selected),
                preserved_branches: unique_observed.into_iter().collect(),
                resolution_id: Some(resolution.resolution_id.clone()),
                issues,
            }
        } else {
            TopologyResolutionDecision {
                schema_version: self.policy.schema_version.clone(),
                policy_id: self.policy.policy_id.clone(),
                state: TopologyResolutionState::Quarantined,
                selected_successor: None,
                preserved_branches: unique_observed.into_iter().collect(),
                resolution_id: Some(resolution.resolution_id.clone()),
                issues,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> TopologyResolutionGate {
        TopologyResolutionGate::new(TopologyResolutionPolicy {
            schema_version: "0.1".into(),
            policy_id: "topology-resolution-v1".into(),
            expected_asset_id: "vehicle-1".into(),
            expected_component_id: "wing-root".into(),
            expected_configuration_digest: "cfg-1".into(),
            expected_authority_id: "mycelix-topology-authority".into(),
        })
        .unwrap()
    }

    fn branch(digest: &str) -> TopologyBranchReference {
        TopologyBranchReference {
            predecessor_epoch: 1,
            predecessor_topology_digest: "topology-v1".into(),
            successor_epoch: 2,
            successor_topology_digest: digest.into(),
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
            observed_successors: observed,
            resolution_id: "resolution-2".into(),
            authority_id: "mycelix-topology-authority".into(),
            authority_statement_digest: "resolution-digest-2".into(),
            verification_reference: "resolution-verify-2".into(),
            resolved_at_ms: 2_000,
            configuration_digest: "cfg-1".into(),
        }
    }

    #[test]
    fn authoritative_resolution_can_select_one_observed_branch() {
        let a = branch("topology-v2a");
        let b = branch("topology-v2b");
        let r = resolution("topology-v2a", vec![a.clone(), b.clone()]);
        let d = gate().assess(Some(&r), &[a.clone(), b.clone()], 3_000);
        assert_eq!(d.state, TopologyResolutionState::Resolved);
        assert_eq!(d.selected_successor, Some(a));
        assert_eq!(d.preserved_branches, vec![branch("topology-v2a"), branch("topology-v2b")]);
    }

    #[test]
    fn missing_resolution_keeps_fork_conflicted() {
        let a = branch("topology-v2a");
        let b = branch("topology-v2b");
        let d = gate().assess(None, &[a, b], 3_000);
        assert_eq!(d.state, TopologyResolutionState::Conflicted);
        assert!(d.resolution_id.is_none());
    }

    #[test]
    fn unresolved_single_successor_is_insufficient_evidence() {
        let a = branch("topology-v2a");
        let d = gate().assess(None, &[a], 3_000);
        assert_eq!(d.state, TopologyResolutionState::InsufficientEvidence);
    }

    #[test]
    fn resolution_cannot_select_an_unobserved_branch() {
        let a = branch("topology-v2a");
        let b = branch("topology-v2b");
        let r = resolution("topology-v2c", vec![a.clone(), b.clone()]);
        let d = gate().assess(Some(&r), &[a, b], 3_000);
        assert_eq!(d.state, TopologyResolutionState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionIssue::SuccessorNotObserved));
    }

    #[test]
    fn competing_branch_is_preserved_in_the_resolution_record() {
        let a = branch("topology-v2a");
        let b = branch("topology-v2b");
        let r = resolution("topology-v2b", vec![a.clone(), b.clone()]);
        let d = gate().assess(Some(&r), &[a.clone(), b.clone()], 3_000);
        assert_eq!(d.state, TopologyResolutionState::Resolved);
        assert_eq!(d.preserved_branches, vec![a, b]);
    }

    #[test]
    fn wrong_authority_cannot_resolve_a_fork() {
        let a = branch("topology-v2a");
        let b = branch("topology-v2b");
        let mut r = resolution("topology-v2a", vec![a.clone(), b]);
        r.authority_id = "unexpected-authority".into();
        let d = gate().assess(Some(&r), &[a], 3_000);
        assert_eq!(d.state, TopologyResolutionState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionIssue::WrongAuthority));
    }

    #[test]
    fn future_resolution_cannot_be_admitted() {
        let a = branch("topology-v2a");
        let mut r = resolution("topology-v2a", vec![a.clone()]);
        r.resolved_at_ms = 3_001;
        let d = gate().assess(Some(&r), &[a], 3_000);
        assert_eq!(d.state, TopologyResolutionState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionIssue::FutureResolution));
    }

    #[test]
    fn resolution_must_preserve_all_observed_branches() {
        let a = branch("topology-v2a");
        let b = branch("topology-v2b");
        let r = resolution("topology-v2a", vec![a.clone()]);
        let d = gate().assess(Some(&r), &[a, b], 3_000);
        assert_eq!(d.state, TopologyResolutionState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionIssue::MissingPreservedBranch));
    }
}
