// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic topology lifecycle continuity and fork detection.
//!
//! A lifecycle epoch is an append-only transition, not merely a timestamp.
//! Offline actors may legitimately produce lifecycle statements concurrently.
//! If two distinct successors claim the same predecessor, this contract reports
//! a fork rather than silently selecting a winner.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyLifecycleState {
    Accepted,
    InsufficientEvidence,
    Conflicted,
    Quarantined,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyLifecycleIssue {
    EmptyInput,
    EmptyIdentity,
    InvalidEpoch,
    MissingPredecessor,
    EpochRollback,
    EpochSkip,
    PredecessorDigestMismatch,
    InvalidEffectiveTime,
    DuplicateSuccessor,
    ConcurrentSuccessorFork,
    ConfigurationMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologySuccessorStatement {
    pub schema_version: String,
    pub asset_id: String,
    pub component_id: String,
    pub predecessor_epoch: u64,
    pub predecessor_topology_digest: String,
    pub successor_epoch: u64,
    pub successor_topology_digest: String,
    pub successor_topology_version: String,
    pub effective_from_ms: u64,
    pub lifecycle_event_id: String,
    pub configuration_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyLifecyclePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_asset_id: String,
    pub expected_component_id: String,
    pub expected_configuration_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyLifecycleDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub state: TopologyLifecycleState,
    pub accepted_successors: Vec<(u64, String)>,
    pub issues: Vec<TopologyLifecycleIssue>,
}

#[derive(Debug, Clone)]
pub struct TopologyLifecycleGate {
    policy: TopologyLifecyclePolicy,
}

impl TopologyLifecycleGate {
    pub fn new(policy: TopologyLifecyclePolicy) -> Result<Self, &'static str> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.expected_asset_id.trim().is_empty()
            || policy.expected_component_id.trim().is_empty()
            || policy.expected_configuration_digest.trim().is_empty()
        {
            return Err("invalid topology lifecycle policy");
        }
        Ok(Self { policy })
    }

    pub fn assess(&self, statements: &[TopologySuccessorStatement]) -> TopologyLifecycleDecision {
        let mut issues = Vec::new();
        let mut valid = Vec::new();

        if statements.is_empty() {
            issues.push(TopologyLifecycleIssue::EmptyInput);
        }

        for statement in statements {
            if statement.schema_version.trim().is_empty()
                || statement.asset_id.trim().is_empty()
                || statement.component_id.trim().is_empty()
                || statement.predecessor_topology_digest.trim().is_empty()
                || statement.successor_topology_digest.trim().is_empty()
                || statement.successor_topology_version.trim().is_empty()
                || statement.lifecycle_event_id.trim().is_empty()
                || statement.configuration_digest.trim().is_empty()
            {
                issues.push(TopologyLifecycleIssue::EmptyIdentity);
                continue;
            }

            if statement.asset_id != self.policy.expected_asset_id
                || statement.component_id != self.policy.expected_component_id
                || statement.configuration_digest != self.policy.expected_configuration_digest
            {
                issues.push(TopologyLifecycleIssue::ConfigurationMismatch);
                continue;
            }

            if statement.predecessor_epoch == 0 || statement.successor_epoch == 0 {
                issues.push(TopologyLifecycleIssue::InvalidEpoch);
                continue;
            }

            if statement.successor_epoch <= statement.predecessor_epoch {
                issues.push(TopologyLifecycleIssue::EpochRollback);
                continue;
            }

            if statement.successor_epoch != statement.predecessor_epoch.saturating_add(1) {
                issues.push(TopologyLifecycleIssue::EpochSkip);
                continue;
            }

            if statement.predecessor_topology_digest == statement.successor_topology_digest {
                issues.push(TopologyLifecycleIssue::PredecessorDigestMismatch);
                continue;
            }

            if statement.effective_from_ms == 0 {
                issues.push(TopologyLifecycleIssue::InvalidEffectiveTime);
                continue;
            }

            valid.push(statement);
        }

        let mut successors = BTreeSet::new();
        for statement in &valid {
            successors.insert((
                statement.predecessor_epoch,
                statement.predecessor_topology_digest.clone(),
                statement.successor_epoch,
                statement.successor_topology_digest.clone(),
            ));
        }

        let duplicate_count = valid.len().saturating_sub(successors.len());
        if duplicate_count > 0 {
            issues.push(TopologyLifecycleIssue::DuplicateSuccessor);
        }

        let predecessor_keys: BTreeSet<_> = valid
            .iter()
            .map(|statement| {
                (
                    statement.predecessor_epoch,
                    statement.predecessor_topology_digest.as_str(),
                )
            })
            .collect();

        let forked = predecessor_keys.iter().any(|predecessor| {
            let successor_digests: BTreeSet<_> = valid
                .iter()
                .filter(|statement| {
                    statement.predecessor_epoch == predecessor.0
                        && statement.predecessor_topology_digest == predecessor.1
                })
                .map(|statement| statement.successor_topology_digest.as_str())
                .collect();
            successor_digests.len() > 1
        });

        if forked {
            issues.push(TopologyLifecycleIssue::ConcurrentSuccessorFork);
        }

        let state = if forked {
            TopologyLifecycleState::Conflicted
        } else if issues.iter().any(|issue| {
            matches!(
                issue,
                TopologyLifecycleIssue::EmptyIdentity
                    | TopologyLifecycleIssue::InvalidEpoch
                    | TopologyLifecycleIssue::MissingPredecessor
                    | TopologyLifecycleIssue::EpochRollback
                    | TopologyLifecycleIssue::EpochSkip
                    | TopologyLifecycleIssue::PredecessorDigestMismatch
                    | TopologyLifecycleIssue::InvalidEffectiveTime
                    | TopologyLifecycleIssue::ConfigurationMismatch
            )
        }) {
            TopologyLifecycleState::Quarantined
        } else if valid.is_empty() {
            TopologyLifecycleState::InsufficientEvidence
        } else {
            TopologyLifecycleState::Accepted
        };

        let accepted_successors = if state == TopologyLifecycleState::Accepted {
            successors
                .iter()
                .map(|(_, _, epoch, digest)| (*epoch, digest.clone()))
                .collect()
        } else {
            Vec::new()
        };

        TopologyLifecycleDecision {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            state,
            accepted_successors,
            issues,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> TopologyLifecycleGate {
        TopologyLifecycleGate::new(TopologyLifecyclePolicy {
            schema_version: "0.1".into(),
            policy_id: "topology-lifecycle-v1".into(),
            expected_asset_id: "vehicle-1".into(),
            expected_component_id: "wing-root".into(),
            expected_configuration_digest: "cfg-1".into(),
        })
        .unwrap()
    }

    fn successor(digest: &str) -> TopologySuccessorStatement {
        TopologySuccessorStatement {
            schema_version: "0.1".into(),
            asset_id: "vehicle-1".into(),
            component_id: "wing-root".into(),
            predecessor_epoch: 1,
            predecessor_topology_digest: "topology-v1".into(),
            successor_epoch: 2,
            successor_topology_digest: digest.into(),
            successor_topology_version: "2".into(),
            effective_from_ms: 1_500,
            lifecycle_event_id: "topology-transition-2".into(),
            configuration_digest: "cfg-1".into(),
        }
    }

    #[test]
    fn single_valid_successor_is_accepted() {
        let d = gate().assess(&[successor("topology-v2")]);
        assert_eq!(d.state, TopologyLifecycleState::Accepted);
        assert_eq!(d.accepted_successors, vec![(2, "topology-v2".into())]);
    }

    #[test]
    fn duplicate_statement_does_not_create_a_false_fork() {
        let s = successor("topology-v2");
        let d = gate().assess(&[s.clone(), s]);
        assert_eq!(d.state, TopologyLifecycleState::Accepted);
        assert!(d.issues.contains(&TopologyLifecycleIssue::DuplicateSuccessor));
    }

    #[test]
    fn concurrent_successors_from_same_predecessor_are_conflicted() {
        let d = gate().assess(&[successor("topology-v2a"), successor("topology-v2b")]);
        assert_eq!(d.state, TopologyLifecycleState::Conflicted);
        assert!(d.issues.contains(&TopologyLifecycleIssue::ConcurrentSuccessorFork));
    }

    #[test]
    fn epoch_skip_is_quarantined() {
        let mut s = successor("topology-v3");
        s.successor_epoch = 3;
        let d = gate().assess(&[s]);
        assert_eq!(d.state, TopologyLifecycleState::Quarantined);
        assert!(d.issues.contains(&TopologyLifecycleIssue::EpochSkip));
    }

    #[test]
    fn rollback_is_quarantined() {
        let mut s = successor("topology-v0");
        s.successor_epoch = 1;
        let d = gate().assess(&[s]);
        assert_eq!(d.state, TopologyLifecycleState::Quarantined);
        assert!(d.issues.contains(&TopologyLifecycleIssue::EpochRollback));
    }

    #[test]
    fn successor_must_change_topology_digest() {
        let s = successor("topology-v1");
        let d = gate().assess(&[s]);
        assert_eq!(d.state, TopologyLifecycleState::Quarantined);
        assert!(d.issues.contains(&TopologyLifecycleIssue::PredecessorDigestMismatch));
    }

    #[test]
    fn configuration_mismatch_cannot_advance_lifecycle() {
        let mut s = successor("topology-v2");
        s.configuration_digest = "cfg-attacker".into();
        let d = gate().assess(&[s]);
        assert_eq!(d.state, TopologyLifecycleState::Quarantined);
        assert!(d.issues.contains(&TopologyLifecycleIssue::ConfigurationMismatch));
    }

    #[test]
    fn empty_input_is_insufficient_evidence() {
        let d = gate().assess(&[]);
        assert_eq!(d.state, TopologyLifecycleState::InsufficientEvidence);
        assert!(d.issues.contains(&TopologyLifecycleIssue::EmptyInput));
    }
}
