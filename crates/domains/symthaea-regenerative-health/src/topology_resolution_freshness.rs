// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Composition boundary between stateless topology verification and
//! receiver-side freshness sequencing.
//!
//! The existing topology verification gate remains unchanged and continues to
//! validate the exact resolution, verifier, report, policy, environment, and
//! validity window. This adapter adds receiver-local replay/reordering state.
//! It does not authenticate the freshness marker or establish authority.

use serde::{Deserialize, Serialize};

use crate::freshness_reconciliation::{
    FreshnessAcceptanceIssue, FreshnessAcceptanceOutcome, FreshnessAcceptanceState,
    FreshnessMarker,
};
use crate::topology_resolution::TopologyLifecycleResolution;
use crate::topology_resolution_verification::{
    TopologyResolutionVerificationDecision, TopologyResolutionVerificationGate,
    TopologyResolutionVerificationIssue, TopologyResolutionVerificationState,
    TopologyResolutionFreshnessVerificationResult,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyResolutionFreshnessState {
    Verified,
    InsufficientEvidence,
    Quarantined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyResolutionFreshnessIssue {
    Verification(TopologyResolutionVerificationIssue),
    Freshness(FreshnessAcceptanceIssue),
    FreshnessPolicyFingerprintMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyResolutionFreshnessDecision {
    pub state: TopologyResolutionFreshnessState,
    pub verification: TopologyResolutionVerificationDecision,
    pub freshness: Option<crate::freshness_reconciliation::FreshnessAcceptanceDecision>,
    pub next_freshness_state: FreshnessAcceptanceState,
    pub issues: Vec<TopologyResolutionFreshnessIssue>,
}

#[derive(Debug, Clone)]
pub struct TopologyResolutionFreshnessGate {
    policy: crate::freshness_reconciliation::FreshnessAcceptancePolicy,
}

impl TopologyResolutionFreshnessGate {
    pub fn new(
        policy: crate::freshness_reconciliation::FreshnessAcceptancePolicy,
    ) -> Result<Self, &'static str> {
        policy.validate()?;
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        verification_gate: &TopologyResolutionVerificationGate,
        resolution: &TopologyLifecycleResolution,
        verification: Option<&TopologyResolutionVerificationResult>,
        freshness_state: &FreshnessAcceptanceState,
        now_ms: u64,
    ) -> TopologyResolutionFreshnessDecision {
        let verification_decision = verification_gate.assess(resolution, verification, now_ms);
        let mut issues = verification_decision
            .issues
            .iter()
            .copied()
            .map(TopologyResolutionFreshnessIssue::Verification)
            .collect::<Vec<_>>();

        if verification_decision.state != TopologyResolutionVerificationState::Verified {
            let state = if verification_decision.state
                == TopologyResolutionVerificationState::InsufficientEvidence
            {
                TopologyResolutionFreshnessState::InsufficientEvidence
            } else {
                TopologyResolutionFreshnessState::Quarantined
            };
            return TopologyResolutionFreshnessDecision {
                state,
                verification: verification_decision,
                freshness: None,
                next_freshness_state: freshness_state.clone(),
                issues,
            };
        }

        let Some(verification) = verification else {
            return TopologyResolutionFreshnessDecision {
                state: TopologyResolutionFreshnessState::InsufficientEvidence,
                verification: verification_decision,
                freshness: None,
                next_freshness_state: freshness_state.clone(),
                issues,
            };
        };

        if verification.freshness_policy_fingerprint.as_deref()
            != Some(self.policy.fingerprint().as_str())
        {
            issues.push(TopologyResolutionFreshnessIssue::FreshnessPolicyFingerprintMismatch);
            return TopologyResolutionFreshnessDecision {
                state: TopologyResolutionFreshnessState::Quarantined,
                verification: verification_decision,
                freshness: None,
                next_freshness_state: freshness_state.clone(),
                issues,
            };
        }

        let Some(freshness) = verification.freshness.as_ref() else {
            issues.push(TopologyResolutionFreshnessIssue::Freshness(
                FreshnessAcceptanceIssue::EmptyMarkerIdentity,
            ));
            return TopologyResolutionFreshnessDecision {
                state: TopologyResolutionFreshnessState::Quarantined,
                verification: verification_decision,
                freshness: None,
                next_freshness_state: freshness_state.clone(),
                issues,
            };
        };

        let marker = FreshnessMarker {
            scheme: freshness.scheme.clone(),
            source_id: freshness.source_id.clone(),
            epoch: freshness.epoch,
            marker_digest: freshness.marker_digest.clone(),
        };
        let freshness_decision = freshness_state.apply(&self.policy, &marker);

        issues.extend(
            freshness_decision
                .issues
                .iter()
                .copied()
                .map(TopologyResolutionFreshnessIssue::Freshness),
        );

        let freshness_admitted = matches!(
            freshness_decision.outcome,
            FreshnessAcceptanceOutcome::Advanced
                | FreshnessAcceptanceOutcome::AcceptedWithinWindow
                | FreshnessAcceptanceOutcome::Duplicate
        );

        TopologyResolutionFreshnessDecision {
            state: if freshness_admitted && issues.is_empty() {
                TopologyResolutionFreshnessState::Verified
            } else {
                TopologyResolutionFreshnessState::Quarantined
            },
            verification: verification_decision,
            freshness: Some(freshness_decision.clone()),
            next_freshness_state: freshness_decision.state,
            issues,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::freshness_reconciliation::FreshnessAcceptancePolicy;
    use crate::topology_resolution::TopologyBranchReference;
    use crate::topology_resolution_verification::{
        TopologyResolutionFreshness, TopologyResolutionVerificationPolicy,
        TopologyResolutionVerificationResult,
    };

    fn freshness_policy() -> FreshnessAcceptancePolicy {
        FreshnessAcceptancePolicy {
            schema_version: "0.1".into(),
            policy_id: "freshness-window-v1".into(),
            expected_scheme: "epoch-marker-v1".into(),
            expected_source_id: "epoch-bell-1".into(),
            minimum_accepted_epoch: 7,
            acceptance_window: 2,
        }
    }

    fn verification_gate() -> TopologyResolutionVerificationGate {
        let freshness = freshness_policy();
        TopologyResolutionVerificationGate::new(TopologyResolutionVerificationPolicy {
            schema_version: "0.1".into(),
            policy_id: "topology-resolution-verification-v1".into(),
            expected_verifier_id: "mycelix-topology-verifier".into(),
            required_freshness_scheme: Some(freshness.expected_scheme.clone()),
            required_freshness_source_id: Some(freshness.expected_source_id.clone()),
            minimum_freshness_epoch: Some(freshness.minimum_accepted_epoch),
            expected_freshness_policy_fingerprint: Some(freshness.fingerprint()),
        })
        .unwrap()
    }

    fn resolution() -> TopologyLifecycleResolution {
        let branch = TopologyBranchReference {
            predecessor_epoch: 1,
            predecessor_topology_digest: "topology-v1".into(),
            successor_epoch: 2,
            successor_topology_digest: "topology-v2a".into(),
            successor_effective_from_ms: 1_500,
        };
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
            observed_successors: vec![branch.clone()],
            observed_successors_digest: crate::topology_resolution::branch_set_digest(&[branch]),
            resolution_id: "resolution-2".into(),
            authority_id: "mycelix-topology-authority".into(),
            authority_statement_digest: "resolution-digest-2".into(),
            verification_reference: "resolution-verify-2".into(),
            resolved_at_ms: 2_000,
            configuration_digest: "cfg-1".into(),
        }
    }

    fn verification(epoch: u64, policy_fingerprint: String) -> TopologyResolutionVerificationResult {
        TopologyResolutionVerificationResult {
            schema_version: "0.1".into(),
            verifier_id: "mycelix-topology-verifier".into(),
            resolution_id: "resolution-2".into(),
            authority_statement_digest: "resolution-digest-2".into(),
            verification_reference: "resolution-verify-2".into(),
            verification_report_digest: "report-digest-2".into(),
            policy_fingerprint: "policy-fingerprint-2".into(),
            environment_fingerprint: "environment-fingerprint-2".into(),
            freshness_policy_fingerprint: Some(policy_fingerprint),
            freshness: Some(TopologyResolutionFreshness {
                scheme: "epoch-marker-v1".into(),
                source_id: "epoch-bell-1".into(),
                epoch,
                marker_digest: format!("epoch-marker-{epoch}"),
            }),
            verified_at_ms: 2_100,
            valid_until_ms: 4_000,
        }
    }

    #[test]
    fn verified_marker_advances_receiver_state() {
        let policy = freshness_policy();
        let state = FreshnessAcceptanceState::new(&policy).unwrap();
        let d = TopologyResolutionFreshnessGate::new(policy.clone())
            .unwrap()
            .assess(
                &verification_gate(),
                &resolution(),
                Some(&verification(7, policy.fingerprint())),
                &state,
                3_000,
            );
        assert_eq!(d.state, TopologyResolutionFreshnessState::Verified);
        assert_eq!(d.next_freshness_state.highest_accepted_epoch, Some(7));
    }

    #[test]
    fn rollback_marker_is_quarantined_without_advancing_state() {
        let policy = freshness_policy();
        let state = FreshnessAcceptanceState::new(&policy).unwrap();
        let state = state.apply(&policy, &FreshnessMarker {
            scheme: "epoch-marker-v1".into(),
            source_id: "epoch-bell-1".into(),
            epoch: 9,
            marker_digest: "epoch-marker-9".into(),
        }).state;
        let d = TopologyResolutionFreshnessGate::new(policy.clone())
            .unwrap()
            .assess(
                &verification_gate(),
                &resolution(),
                Some(&verification(7, policy.fingerprint())),
                &state,
                3_000,
            );
        assert_eq!(d.state, TopologyResolutionFreshnessState::Quarantined);
        assert_eq!(d.next_freshness_state, state);
    }

    #[test]
    fn conflicting_marker_latches_receiver_state() {
        let policy = freshness_policy();
        let mut state = FreshnessAcceptanceState::new(&policy).unwrap();
        let first = verification(7, policy.fingerprint());
        let gate = TopologyResolutionFreshnessGate::new(policy.clone()).unwrap();
        state = gate
            .assess(&verification_gate(), &resolution(), Some(&first), &state, 3_000)
            .next_freshness_state;
        let mut second = verification(7, policy.fingerprint());
        second.freshness.as_mut().unwrap().marker_digest = "different".into();
        let d = gate.assess(&verification_gate(), &resolution(), Some(&second), &state, 3_000);
        assert_eq!(d.state, TopologyResolutionFreshnessState::Quarantined);
        assert!(d.next_freshness_state.conflicted);
    }

    #[test]
    fn verifier_failure_does_not_mutate_receiver_cursor() {
        let policy = freshness_policy();
        let state = FreshnessAcceptanceState::new(&policy).unwrap();
        let mut bad = verification(9, policy.fingerprint());
        bad.verifier_id = "unexpected".into();
        let d = TopologyResolutionFreshnessGate::new(policy.clone())
            .unwrap()
            .assess(&verification_gate(), &resolution(), Some(&bad), &state, 3_000);
        assert_eq!(d.state, TopologyResolutionFreshnessState::Quarantined);
        assert_eq!(d.next_freshness_state, state);
    }
}
