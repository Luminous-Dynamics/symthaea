// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed behavior during ground-link and fleet-network partitions.
//!
//! A communication loss must not invent remote authority or disable local
//! safety. This module distinguishes a short authenticated-link grace period
//! from bounded onboard autonomy, mandatory return, and immediate landing.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionMissionCriticality {
    Routine,
    SearchAndRescue,
    LifeCritical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkPartitionPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub mission_id: String,
    pub maximum_link_observation_age_ms: u64,
    pub grace_period_ms: u64,
    pub maximum_local_autonomy_ms: u64,
    pub require_authenticated_ground_link: bool,
    pub allow_local_completion: bool,
    pub require_reconciliation_before_remote_commands: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkLinkObservation {
    pub observation_id: String,
    pub timestamp_ms: u64,
    pub ground_link_available: bool,
    pub ground_link_authenticated: bool,
    pub peer_link_available: bool,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalAutonomyState {
    pub mission_id: String,
    pub mission_criticality: PartitionMissionCriticality,
    pub partition_started_ms: Option<u64>,
    pub onboard_plan_digest: Option<String>,
    pub estimator_healthy: bool,
    pub local_abort_available: bool,
    pub return_route_available: bool,
    pub safe_landing_available: bool,
    pub local_evidence_chain_digest: Option<String>,
    pub last_accepted_remote_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconnectionEvidence {
    pub timestamp_ms: u64,
    pub remote_sequence: u64,
    pub acknowledged_local_evidence_digest: Option<String>,
    pub ground_identity_authenticated: bool,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkPartitionMode {
    Connected,
    Grace,
    LocalAutonomy,
    ReturnToBase,
    LandAsSoonAsPracticable,
    ImmediateLanding,
    ReconciliationRequired,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkPartitionIssue {
    InvalidObservationIdentity,
    MissingEvidence(String),
    FutureObservation,
    StaleObservation {
        age_ms: u64,
        maximum_ms: u64,
    },
    MissionMismatch,
    UnauthenticatedGroundLink,
    MissingPartitionStart,
    FuturePartitionStart,
    MissingOnboardPlan,
    EstimatorUnhealthy,
    LocalAbortUnavailable,
    AutonomyDeadlineExceeded {
        elapsed_ms: u64,
        maximum_ms: u64,
    },
    NoReturnRoute,
    NoSafeLanding,
    MissingLocalEvidenceChain,
    ReconciliationMissing,
    ReconnectionUnauthenticated,
    RemoteSequenceReplay {
        observed: u64,
        minimum_exclusive: u64,
    },
    LocalEvidenceNotAcknowledged,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkPartitionDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub mission_id: String,
    pub assessed_at_ms: u64,
    pub mode: NetworkPartitionMode,
    pub accept_remote_commands: bool,
    pub preserve_local_control: bool,
    pub issues: Vec<NetworkPartitionIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkPartitionError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct NetworkPartitionSupervisor {
    policy: NetworkPartitionPolicy,
}

impl NetworkPartitionSupervisor {
    pub fn new(policy: NetworkPartitionPolicy) -> Result<Self, NetworkPartitionError> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.mission_id.trim().is_empty()
            || policy.maximum_link_observation_age_ms == 0
            || policy.grace_period_ms == 0
            || policy.maximum_local_autonomy_ms < policy.grace_period_ms
        {
            return Err(NetworkPartitionError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        link: &NetworkLinkObservation,
        local: &LocalAutonomyState,
        reconnection: Option<&ReconnectionEvidence>,
        now_ms: u64,
    ) -> NetworkPartitionDecision {
        let mut issues = Vec::new();
        if link.observation_id.trim().is_empty() {
            issues.push(NetworkPartitionIssue::InvalidObservationIdentity);
        }
        if link.evidence_ids.is_empty() || link.evidence_ids.iter().any(|id| id.trim().is_empty()) {
            issues.push(NetworkPartitionIssue::MissingEvidence(
                link.observation_id.clone(),
            ));
        }
        if link.timestamp_ms > now_ms {
            issues.push(NetworkPartitionIssue::FutureObservation);
        } else {
            let age = now_ms.saturating_sub(link.timestamp_ms);
            if age > self.policy.maximum_link_observation_age_ms {
                issues.push(NetworkPartitionIssue::StaleObservation {
                    age_ms: age,
                    maximum_ms: self.policy.maximum_link_observation_age_ms,
                });
            }
        }
        if local.mission_id != self.policy.mission_id {
            issues.push(NetworkPartitionIssue::MissionMismatch);
        }
        if !local.estimator_healthy {
            issues.push(NetworkPartitionIssue::EstimatorUnhealthy);
        }
        if !local.local_abort_available {
            issues.push(NetworkPartitionIssue::LocalAbortUnavailable);
        }

        let authenticated_connected = link.ground_link_available
            && (!self.policy.require_authenticated_ground_link || link.ground_link_authenticated);
        if link.ground_link_available && !authenticated_connected {
            issues.push(NetworkPartitionIssue::UnauthenticatedGroundLink);
        }

        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                NetworkPartitionIssue::InvalidObservationIdentity
                    | NetworkPartitionIssue::MissingEvidence(_)
                    | NetworkPartitionIssue::FutureObservation
                    | NetworkPartitionIssue::StaleObservation { .. }
                    | NetworkPartitionIssue::MissionMismatch
            )
        });
        if incomplete {
            return self.decision(
                now_ms,
                NetworkPartitionMode::Incomplete,
                false,
                true,
                issues,
            );
        }

        if authenticated_connected {
            if self.policy.require_reconciliation_before_remote_commands
                && local.partition_started_ms.is_some()
            {
                match reconnection {
                    None => issues.push(NetworkPartitionIssue::ReconciliationMissing),
                    Some(reconnect) => {
                        if reconnect.evidence_ids.is_empty()
                            || reconnect.evidence_ids.iter().any(|id| id.trim().is_empty())
                        {
                            issues.push(NetworkPartitionIssue::MissingEvidence(
                                "reconnection".into(),
                            ));
                        }
                        if !reconnect.ground_identity_authenticated {
                            issues.push(NetworkPartitionIssue::ReconnectionUnauthenticated);
                        }
                        if reconnect.remote_sequence <= local.last_accepted_remote_sequence {
                            issues.push(NetworkPartitionIssue::RemoteSequenceReplay {
                                observed: reconnect.remote_sequence,
                                minimum_exclusive: local.last_accepted_remote_sequence,
                            });
                        }
                        if reconnect.acknowledged_local_evidence_digest.as_deref()
                            != local.local_evidence_chain_digest.as_deref()
                        {
                            issues.push(NetworkPartitionIssue::LocalEvidenceNotAcknowledged);
                        }
                    }
                }
                if !issues.is_empty() {
                    return self.decision(
                        now_ms,
                        NetworkPartitionMode::ReconciliationRequired,
                        false,
                        true,
                        issues,
                    );
                }
            }
            return self.decision(now_ms, NetworkPartitionMode::Connected, true, true, issues);
        }

        let Some(partition_started_ms) = local.partition_started_ms else {
            issues.push(NetworkPartitionIssue::MissingPartitionStart);
            return self.decision(
                now_ms,
                NetworkPartitionMode::Incomplete,
                false,
                true,
                issues,
            );
        };
        if partition_started_ms > now_ms {
            issues.push(NetworkPartitionIssue::FuturePartitionStart);
            return self.decision(
                now_ms,
                NetworkPartitionMode::Incomplete,
                false,
                true,
                issues,
            );
        }
        let elapsed = now_ms.saturating_sub(partition_started_ms);
        if elapsed <= self.policy.grace_period_ms {
            return self.decision(now_ms, NetworkPartitionMode::Grace, false, true, issues);
        }

        if local
            .onboard_plan_digest
            .as_ref()
            .is_none_or(|digest| !valid_digest(digest))
        {
            issues.push(NetworkPartitionIssue::MissingOnboardPlan);
        }
        if local
            .local_evidence_chain_digest
            .as_ref()
            .is_none_or(|digest| !valid_digest(digest))
        {
            issues.push(NetworkPartitionIssue::MissingLocalEvidenceChain);
        }
        if !local.estimator_healthy || !local.local_abort_available {
            let mode = if local.safe_landing_available {
                NetworkPartitionMode::ImmediateLanding
            } else {
                issues.push(NetworkPartitionIssue::NoSafeLanding);
                NetworkPartitionMode::Incomplete
            };
            return self.decision(now_ms, mode, false, true, issues);
        }
        if elapsed > self.policy.maximum_local_autonomy_ms {
            issues.push(NetworkPartitionIssue::AutonomyDeadlineExceeded {
                elapsed_ms: elapsed,
                maximum_ms: self.policy.maximum_local_autonomy_ms,
            });
            let mode = if local.return_route_available {
                NetworkPartitionMode::ReturnToBase
            } else if local.safe_landing_available {
                issues.push(NetworkPartitionIssue::NoReturnRoute);
                NetworkPartitionMode::LandAsSoonAsPracticable
            } else {
                issues.push(NetworkPartitionIssue::NoReturnRoute);
                issues.push(NetworkPartitionIssue::NoSafeLanding);
                NetworkPartitionMode::Incomplete
            };
            return self.decision(now_ms, mode, false, true, issues);
        }

        let local_completion_allowed = self.policy.allow_local_completion
            && local.mission_criticality != PartitionMissionCriticality::Routine
            && issues.is_empty();
        let mode = if local_completion_allowed {
            NetworkPartitionMode::LocalAutonomy
        } else if local.return_route_available {
            NetworkPartitionMode::ReturnToBase
        } else if local.safe_landing_available {
            issues.push(NetworkPartitionIssue::NoReturnRoute);
            NetworkPartitionMode::LandAsSoonAsPracticable
        } else {
            issues.push(NetworkPartitionIssue::NoReturnRoute);
            issues.push(NetworkPartitionIssue::NoSafeLanding);
            NetworkPartitionMode::Incomplete
        };
        self.decision(now_ms, mode, false, true, issues)
    }

    fn decision(
        &self,
        assessed_at_ms: u64,
        mode: NetworkPartitionMode,
        accept_remote_commands: bool,
        preserve_local_control: bool,
        issues: Vec<NetworkPartitionIssue>,
    ) -> NetworkPartitionDecision {
        NetworkPartitionDecision {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            mission_id: self.policy.mission_id.clone(),
            assessed_at_ms,
            mode,
            accept_remote_commands,
            preserve_local_control,
            issues,
        }
    }
}

fn valid_digest(digest: &str) -> bool {
    let digest = digest.trim();
    digest.starts_with("sha256:") && digest.len() > "sha256:".len()
        || digest.starts_with("fnv1a64:") && digest.len() == "fnv1a64:".len() + 16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn supervisor() -> NetworkPartitionSupervisor {
        NetworkPartitionSupervisor::new(NetworkPartitionPolicy {
            schema_version: "1".into(),
            policy_id: "partition-policy".into(),
            mission_id: "mission-1".into(),
            maximum_link_observation_age_ms: 500,
            grace_period_ms: 5_000,
            maximum_local_autonomy_ms: 60_000,
            require_authenticated_ground_link: true,
            allow_local_completion: true,
            require_reconciliation_before_remote_commands: true,
        })
        .unwrap()
    }

    fn link(available: bool, authenticated: bool, timestamp_ms: u64) -> NetworkLinkObservation {
        NetworkLinkObservation {
            observation_id: "link-1".into(),
            timestamp_ms,
            ground_link_available: available,
            ground_link_authenticated: authenticated,
            peer_link_available: false,
            evidence_ids: vec!["link-evidence".into()],
        }
    }

    fn local(partition_started_ms: Option<u64>) -> LocalAutonomyState {
        LocalAutonomyState {
            mission_id: "mission-1".into(),
            mission_criticality: PartitionMissionCriticality::SearchAndRescue,
            partition_started_ms,
            onboard_plan_digest: Some("sha256:plan".into()),
            estimator_healthy: true,
            local_abort_available: true,
            return_route_available: true,
            safe_landing_available: true,
            local_evidence_chain_digest: Some("sha256:local-chain".into()),
            last_accepted_remote_sequence: 10,
        }
    }

    #[test]
    fn bounded_partition_uses_local_autonomy() {
        let decision = supervisor().assess(
            &link(false, false, 20_000),
            &local(Some(10_000)),
            None,
            20_000,
        );
        assert_eq!(decision.mode, NetworkPartitionMode::LocalAutonomy);
        assert!(!decision.accept_remote_commands);
    }

    #[test]
    fn expired_partition_returns_to_base() {
        let decision = supervisor().assess(
            &link(false, false, 80_000),
            &local(Some(10_000)),
            None,
            80_000,
        );
        assert_eq!(decision.mode, NetworkPartitionMode::ReturnToBase);
    }

    #[test]
    fn reconnect_requires_acknowledgement_and_fresh_sequence() {
        let reconnect = ReconnectionEvidence {
            timestamp_ms: 20_000,
            remote_sequence: 10,
            acknowledged_local_evidence_digest: None,
            ground_identity_authenticated: true,
            evidence_ids: vec!["reconnect".into()],
        };
        let decision = supervisor().assess(
            &link(true, true, 20_000),
            &local(Some(10_000)),
            Some(&reconnect),
            20_000,
        );
        assert_eq!(decision.mode, NetworkPartitionMode::ReconciliationRequired);
        assert!(!decision.accept_remote_commands);
    }

    #[test]
    fn connected_without_prior_partition_accepts_commands() {
        let decision = supervisor().assess(&link(true, true, 20_000), &local(None), None, 20_000);
        assert_eq!(decision.mode, NetworkPartitionMode::Connected);
        assert!(decision.accept_remote_commands);
    }
}
