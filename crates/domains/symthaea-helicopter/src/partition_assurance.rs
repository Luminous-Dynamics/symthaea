// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Software partitioning and freedom-from-interference evidence.
//!
//! The checker is intentionally declarative. It does not prove processor or
//! hypervisor behavior; it verifies that declared partitions, resource budgets,
//! communication paths, and interference tests satisfy a reviewable policy.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PartitionCriticality {
    NonCritical,
    Mission,
    Safety,
    FlightCritical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PartitionResourceKind {
    CpuTime,
    Memory,
    Stack,
    Queue,
    BusBandwidth,
    FileDescriptors,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartitionBudget {
    pub kind: PartitionResourceKind,
    pub limit: f64,
    pub observed_peak: f64,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionChannel {
    pub channel_id: String,
    pub from_partition: String,
    pub to_partition: String,
    pub authenticated: bool,
    pub bounded_queue: bool,
    pub fail_closed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftwarePartition {
    pub partition_id: String,
    pub criticality: PartitionCriticality,
    pub processor_domain: String,
    pub memory_domain: String,
    pub scheduler_domain: String,
    pub restart_domain: String,
    pub budgets: Vec<PartitionBudget>,
    pub allowed_channels: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterferenceTestStatus {
    Passed,
    Failed,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterferenceTestEvidence {
    pub evidence_id: String,
    pub aggressor_partition: String,
    pub victim_partition: String,
    pub stressed_resource: PartitionResourceKind,
    pub status: InterferenceTestStatus,
    pub artifact_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartitionAssurancePolicy {
    pub required_budget_kinds: BTreeSet<PartitionResourceKind>,
    pub require_unique_memory_domain_at_or_above: PartitionCriticality,
    pub require_unique_restart_domain_at_or_above: PartitionCriticality,
    pub require_authenticated_cross_criticality_channels: bool,
    pub require_interference_tests_at_or_above: PartitionCriticality,
    pub maximum_budget_utilization: f64,
}

impl Default for PartitionAssurancePolicy {
    fn default() -> Self {
        Self {
            required_budget_kinds: BTreeSet::from([
                PartitionResourceKind::CpuTime,
                PartitionResourceKind::Memory,
                PartitionResourceKind::Stack,
                PartitionResourceKind::Queue,
            ]),
            require_unique_memory_domain_at_or_above: PartitionCriticality::FlightCritical,
            require_unique_restart_domain_at_or_above: PartitionCriticality::Safety,
            require_authenticated_cross_criticality_channels: true,
            require_interference_tests_at_or_above: PartitionCriticality::Safety,
            maximum_budget_utilization: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionAssuranceStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PartitionAssuranceIssue {
    DuplicatePartition {
        partition_id: String,
    },
    MissingBudget {
        partition_id: String,
        kind: PartitionResourceKind,
    },
    InvalidBudget {
        partition_id: String,
        kind: PartitionResourceKind,
    },
    BudgetExceeded {
        partition_id: String,
        kind: PartitionResourceKind,
        utilization: f64,
    },
    SharedMemoryDomain {
        partition_id: String,
        other_partition_id: String,
        domain: String,
    },
    SharedRestartDomain {
        partition_id: String,
        other_partition_id: String,
        domain: String,
    },
    UnknownChannel {
        partition_id: String,
        channel_id: String,
    },
    ChannelEndpointMismatch {
        channel_id: String,
    },
    UnauthenticatedCrossCriticalityChannel {
        channel_id: String,
    },
    UnboundedCrossCriticalityChannel {
        channel_id: String,
    },
    NonFailClosedCrossCriticalityChannel {
        channel_id: String,
    },
    MissingInterferenceTest {
        aggressor_partition: String,
        victim_partition: String,
    },
    FailedInterferenceTest {
        evidence_id: String,
    },
    MissingInterferenceArtifact {
        evidence_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartitionAssuranceReport {
    pub status: PartitionAssuranceStatus,
    pub partition_count: usize,
    pub channel_count: usize,
    pub issues: Vec<PartitionAssuranceIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionAssuranceError {
    InvalidMaximumUtilization,
    EmptyPartitionId,
    EmptyChannelId,
}

pub struct PartitionAssuranceEvaluator {
    policy: PartitionAssurancePolicy,
}

impl PartitionAssuranceEvaluator {
    pub fn new(policy: PartitionAssurancePolicy) -> Result<Self, PartitionAssuranceError> {
        if !policy.maximum_budget_utilization.is_finite()
            || policy.maximum_budget_utilization <= 0.0
        {
            return Err(PartitionAssuranceError::InvalidMaximumUtilization);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        partitions: &[SoftwarePartition],
        channels: &[PartitionChannel],
        tests: &[InterferenceTestEvidence],
    ) -> Result<PartitionAssuranceReport, PartitionAssuranceError> {
        if partitions.iter().any(|p| p.partition_id.trim().is_empty()) {
            return Err(PartitionAssuranceError::EmptyPartitionId);
        }
        if channels.iter().any(|c| c.channel_id.trim().is_empty()) {
            return Err(PartitionAssuranceError::EmptyChannelId);
        }

        let mut issues = Vec::new();
        let mut partition_by_id = BTreeMap::<&str, &SoftwarePartition>::new();
        for partition in partitions {
            if partition_by_id
                .insert(partition.partition_id.as_str(), partition)
                .is_some()
            {
                issues.push(PartitionAssuranceIssue::DuplicatePartition {
                    partition_id: partition.partition_id.clone(),
                });
            }
            self.check_budgets(partition, &mut issues);
        }

        self.check_domains(partitions, &mut issues);

        let channel_by_id: BTreeMap<&str, &PartitionChannel> = channels
            .iter()
            .map(|channel| (channel.channel_id.as_str(), channel))
            .collect();
        for partition in partitions {
            for channel_id in &partition.allowed_channels {
                if !channel_by_id.contains_key(channel_id.as_str()) {
                    issues.push(PartitionAssuranceIssue::UnknownChannel {
                        partition_id: partition.partition_id.clone(),
                        channel_id: channel_id.clone(),
                    });
                }
            }
        }
        self.check_channels(&partition_by_id, channels, &mut issues);
        self.check_interference(partitions, tests, &mut issues);

        issues.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        let status = if issues.iter().any(issue_is_failure) {
            PartitionAssuranceStatus::Fail
        } else if issues.is_empty() {
            PartitionAssuranceStatus::Pass
        } else {
            PartitionAssuranceStatus::Incomplete
        };
        Ok(PartitionAssuranceReport {
            status,
            partition_count: partitions.len(),
            channel_count: channels.len(),
            issues,
        })
    }

    fn check_budgets(
        &self,
        partition: &SoftwarePartition,
        issues: &mut Vec<PartitionAssuranceIssue>,
    ) {
        let budgets: BTreeMap<PartitionResourceKind, &PartitionBudget> = partition
            .budgets
            .iter()
            .map(|budget| (budget.kind, budget))
            .collect();
        for kind in &self.policy.required_budget_kinds {
            let Some(budget) = budgets.get(kind) else {
                issues.push(PartitionAssuranceIssue::MissingBudget {
                    partition_id: partition.partition_id.clone(),
                    kind: *kind,
                });
                continue;
            };
            if !budget.limit.is_finite()
                || budget.limit <= 0.0
                || !budget.observed_peak.is_finite()
                || budget.observed_peak < 0.0
            {
                issues.push(PartitionAssuranceIssue::InvalidBudget {
                    partition_id: partition.partition_id.clone(),
                    kind: *kind,
                });
                continue;
            }
            let utilization = budget.observed_peak / budget.limit;
            if utilization > self.policy.maximum_budget_utilization {
                issues.push(PartitionAssuranceIssue::BudgetExceeded {
                    partition_id: partition.partition_id.clone(),
                    kind: *kind,
                    utilization,
                });
            }
        }
    }

    fn check_domains(
        &self,
        partitions: &[SoftwarePartition],
        issues: &mut Vec<PartitionAssuranceIssue>,
    ) {
        for (index, left) in partitions.iter().enumerate() {
            for right in partitions.iter().skip(index + 1) {
                if left.criticality >= self.policy.require_unique_memory_domain_at_or_above
                    && right.criticality >= self.policy.require_unique_memory_domain_at_or_above
                    && left.memory_domain == right.memory_domain
                {
                    issues.push(PartitionAssuranceIssue::SharedMemoryDomain {
                        partition_id: left.partition_id.clone(),
                        other_partition_id: right.partition_id.clone(),
                        domain: left.memory_domain.clone(),
                    });
                }
                if left.criticality >= self.policy.require_unique_restart_domain_at_or_above
                    && right.criticality >= self.policy.require_unique_restart_domain_at_or_above
                    && left.restart_domain == right.restart_domain
                {
                    issues.push(PartitionAssuranceIssue::SharedRestartDomain {
                        partition_id: left.partition_id.clone(),
                        other_partition_id: right.partition_id.clone(),
                        domain: left.restart_domain.clone(),
                    });
                }
            }
        }
    }

    fn check_channels(
        &self,
        partitions: &BTreeMap<&str, &SoftwarePartition>,
        channels: &[PartitionChannel],
        issues: &mut Vec<PartitionAssuranceIssue>,
    ) {
        for channel in channels {
            let (Some(from), Some(to)) = (
                partitions.get(channel.from_partition.as_str()),
                partitions.get(channel.to_partition.as_str()),
            ) else {
                issues.push(PartitionAssuranceIssue::ChannelEndpointMismatch {
                    channel_id: channel.channel_id.clone(),
                });
                continue;
            };
            if !from.allowed_channels.contains(&channel.channel_id)
                || !to.allowed_channels.contains(&channel.channel_id)
            {
                issues.push(PartitionAssuranceIssue::ChannelEndpointMismatch {
                    channel_id: channel.channel_id.clone(),
                });
            }
            if from.criticality != to.criticality {
                if self.policy.require_authenticated_cross_criticality_channels
                    && !channel.authenticated
                {
                    issues.push(
                        PartitionAssuranceIssue::UnauthenticatedCrossCriticalityChannel {
                            channel_id: channel.channel_id.clone(),
                        },
                    );
                }
                if !channel.bounded_queue {
                    issues.push(PartitionAssuranceIssue::UnboundedCrossCriticalityChannel {
                        channel_id: channel.channel_id.clone(),
                    });
                }
                if !channel.fail_closed {
                    issues.push(
                        PartitionAssuranceIssue::NonFailClosedCrossCriticalityChannel {
                            channel_id: channel.channel_id.clone(),
                        },
                    );
                }
            }
        }
    }

    fn check_interference(
        &self,
        partitions: &[SoftwarePartition],
        tests: &[InterferenceTestEvidence],
        issues: &mut Vec<PartitionAssuranceIssue>,
    ) {
        for evidence in tests {
            match evidence.status {
                InterferenceTestStatus::Failed => {
                    issues.push(PartitionAssuranceIssue::FailedInterferenceTest {
                        evidence_id: evidence.evidence_id.clone(),
                    })
                }
                InterferenceTestStatus::Passed
                    if evidence.artifact_digest.as_deref().unwrap_or("").is_empty() =>
                {
                    issues.push(PartitionAssuranceIssue::MissingInterferenceArtifact {
                        evidence_id: evidence.evidence_id.clone(),
                    });
                }
                _ => {}
            }
        }

        for victim in partitions.iter().filter(|partition| {
            partition.criticality >= self.policy.require_interference_tests_at_or_above
        }) {
            for aggressor in partitions
                .iter()
                .filter(|partition| partition.partition_id != victim.partition_id)
            {
                let passed = tests.iter().any(|test| {
                    test.aggressor_partition == aggressor.partition_id
                        && test.victim_partition == victim.partition_id
                        && test.status == InterferenceTestStatus::Passed
                });
                if !passed {
                    issues.push(PartitionAssuranceIssue::MissingInterferenceTest {
                        aggressor_partition: aggressor.partition_id.clone(),
                        victim_partition: victim.partition_id.clone(),
                    });
                }
            }
        }
    }
}

fn issue_is_failure(issue: &PartitionAssuranceIssue) -> bool {
    matches!(
        issue,
        PartitionAssuranceIssue::DuplicatePartition { .. }
            | PartitionAssuranceIssue::InvalidBudget { .. }
            | PartitionAssuranceIssue::BudgetExceeded { .. }
            | PartitionAssuranceIssue::SharedMemoryDomain { .. }
            | PartitionAssuranceIssue::SharedRestartDomain { .. }
            | PartitionAssuranceIssue::UnauthenticatedCrossCriticalityChannel { .. }
            | PartitionAssuranceIssue::UnboundedCrossCriticalityChannel { .. }
            | PartitionAssuranceIssue::NonFailClosedCrossCriticalityChannel { .. }
            | PartitionAssuranceIssue::FailedInterferenceTest { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budgets() -> Vec<PartitionBudget> {
        vec![
            PartitionBudget {
                kind: PartitionResourceKind::CpuTime,
                limit: 10.0,
                observed_peak: 5.0,
                unit: "ms".into(),
            },
            PartitionBudget {
                kind: PartitionResourceKind::Memory,
                limit: 100.0,
                observed_peak: 50.0,
                unit: "MiB".into(),
            },
            PartitionBudget {
                kind: PartitionResourceKind::Stack,
                limit: 2.0,
                observed_peak: 1.0,
                unit: "MiB".into(),
            },
            PartitionBudget {
                kind: PartitionResourceKind::Queue,
                limit: 64.0,
                observed_peak: 8.0,
                unit: "messages".into(),
            },
        ]
    }

    fn partition(id: &str, criticality: PartitionCriticality) -> SoftwarePartition {
        SoftwarePartition {
            partition_id: id.into(),
            criticality,
            processor_domain: format!("cpu-{id}"),
            memory_domain: format!("mem-{id}"),
            scheduler_domain: format!("sched-{id}"),
            restart_domain: format!("restart-{id}"),
            budgets: budgets(),
            allowed_channels: vec!["telemetry".into()],
        }
    }

    #[test]
    fn complete_partition_evidence_passes() {
        let evaluator =
            PartitionAssuranceEvaluator::new(PartitionAssurancePolicy::default()).unwrap();
        let partitions = vec![
            partition("flight", PartitionCriticality::FlightCritical),
            partition("mission", PartitionCriticality::Mission),
        ];
        let channels = vec![PartitionChannel {
            channel_id: "telemetry".into(),
            from_partition: "flight".into(),
            to_partition: "mission".into(),
            authenticated: true,
            bounded_queue: true,
            fail_closed: true,
        }];
        let tests = vec![InterferenceTestEvidence {
            evidence_id: "IFT-1".into(),
            aggressor_partition: "mission".into(),
            victim_partition: "flight".into(),
            stressed_resource: PartitionResourceKind::CpuTime,
            status: InterferenceTestStatus::Passed,
            artifact_digest: Some("sha256:test".into()),
        }];
        let report = evaluator.assess(&partitions, &channels, &tests).unwrap();
        assert_eq!(report.status, PartitionAssuranceStatus::Pass);
    }

    #[test]
    fn shared_flight_critical_memory_fails() {
        let evaluator =
            PartitionAssuranceEvaluator::new(PartitionAssurancePolicy::default()).unwrap();
        let mut left = partition("left", PartitionCriticality::FlightCritical);
        let mut right = partition("right", PartitionCriticality::FlightCritical);
        left.memory_domain = "shared".into();
        right.memory_domain = "shared".into();
        left.allowed_channels.clear();
        right.allowed_channels.clear();
        let report = evaluator.assess(&[left, right], &[], &[]).unwrap();
        assert_eq!(report.status, PartitionAssuranceStatus::Fail);
    }

    #[test]
    fn missing_test_is_incomplete() {
        let evaluator =
            PartitionAssuranceEvaluator::new(PartitionAssurancePolicy::default()).unwrap();
        let mut flight = partition("flight", PartitionCriticality::FlightCritical);
        let mut mission = partition("mission", PartitionCriticality::Mission);
        flight.allowed_channels.clear();
        mission.allowed_channels.clear();
        let report = evaluator.assess(&[flight, mission], &[], &[]).unwrap();
        assert_eq!(report.status, PartitionAssuranceStatus::Incomplete);
    }
}
