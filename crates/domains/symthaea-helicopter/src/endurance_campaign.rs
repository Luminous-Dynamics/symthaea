// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Long-duration endurance and soak-test evidence.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EndurancePhase {
    GroundIdle,
    Hover,
    Cruise,
    Maneuver,
    DegradedOperation,
    NetworkPartition,
    Recovery,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnduranceSample {
    pub timestamp_ms: u64,
    pub resident_memory_bytes: u64,
    pub heap_bytes: u64,
    pub queue_depth: u64,
    pub cpu_utilization: f64,
    pub maximum_temperature_c: f64,
    pub cumulative_deadline_misses: u64,
    pub cumulative_watchdog_resets: u64,
    pub evidence_chain_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnduranceRun {
    pub run_id: String,
    pub seed: u64,
    pub aircraft_id: String,
    pub deployment_id: String,
    pub started_at_ms: u64,
    pub ended_at_ms: u64,
    pub phases_exercised: BTreeSet<EndurancePhase>,
    pub samples: Vec<EnduranceSample>,
    pub planned_restart_count: u64,
    pub unplanned_restart_count: u64,
    pub safety_incident_count: u64,
    pub terminal_evidence_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnduranceCampaignPolicy {
    pub required_seeds: BTreeSet<u64>,
    pub required_phases: BTreeSet<EndurancePhase>,
    pub minimum_duration_ms: u64,
    pub maximum_sample_gap_ms: u64,
    pub maximum_unplanned_restarts: u64,
    pub maximum_watchdog_resets: u64,
    pub maximum_deadline_misses: u64,
    pub maximum_memory_growth_bytes_per_hour: f64,
    pub maximum_queue_depth: u64,
    pub maximum_cpu_utilization: f64,
    pub maximum_temperature_c: f64,
}

impl Default for EnduranceCampaignPolicy {
    fn default() -> Self {
        Self {
            required_seeds: BTreeSet::new(),
            required_phases: BTreeSet::from([
                EndurancePhase::GroundIdle,
                EndurancePhase::Hover,
                EndurancePhase::Cruise,
                EndurancePhase::DegradedOperation,
                EndurancePhase::Recovery,
            ]),
            minimum_duration_ms: 24 * 60 * 60 * 1_000,
            maximum_sample_gap_ms: 60_000,
            maximum_unplanned_restarts: 0,
            maximum_watchdog_resets: 0,
            maximum_deadline_misses: 0,
            maximum_memory_growth_bytes_per_hour: 1_048_576.0,
            maximum_queue_depth: 1_024,
            maximum_cpu_utilization: 0.95,
            maximum_temperature_c: 90.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnduranceCampaignStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnduranceCampaignIssue {
    MissingSeed {
        seed: u64,
    },
    DuplicateSeed {
        seed: u64,
    },
    InvalidRunWindow {
        seed: u64,
    },
    InsufficientDuration {
        seed: u64,
        observed_ms: u64,
        required_ms: u64,
    },
    MissingPhase {
        seed: u64,
        phase: EndurancePhase,
    },
    MissingSamples {
        seed: u64,
    },
    NonMonotonicSampleTime {
        seed: u64,
        timestamp_ms: u64,
    },
    SampleGapExceeded {
        seed: u64,
        gap_ms: u64,
    },
    MissingEvidenceDigest {
        seed: u64,
        timestamp_ms: Option<u64>,
    },
    MemoryGrowthExceeded {
        seed: u64,
        bytes_per_hour: f64,
        limit: f64,
    },
    QueueDepthExceeded {
        seed: u64,
        observed: u64,
        limit: u64,
    },
    CpuUtilizationExceeded {
        seed: u64,
        observed: f64,
        limit: f64,
    },
    TemperatureExceeded {
        seed: u64,
        observed_c: f64,
        limit_c: f64,
    },
    DeadlineMissesExceeded {
        seed: u64,
        observed: u64,
        limit: u64,
    },
    WatchdogResetsExceeded {
        seed: u64,
        observed: u64,
        limit: u64,
    },
    UnplannedRestartsExceeded {
        seed: u64,
        observed: u64,
        limit: u64,
    },
    SafetyIncidentObserved {
        seed: u64,
        count: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnduranceRunMetrics {
    pub seed: u64,
    pub duration_ms: u64,
    pub memory_growth_bytes_per_hour: Option<f64>,
    pub maximum_queue_depth: Option<u64>,
    pub maximum_cpu_utilization: Option<f64>,
    pub maximum_temperature_c: Option<f64>,
    pub deadline_misses: Option<u64>,
    pub watchdog_resets: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnduranceCampaignReport {
    pub status: EnduranceCampaignStatus,
    pub run_metrics: Vec<EnduranceRunMetrics>,
    pub issues: Vec<EnduranceCampaignIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnduranceCampaignError {
    InvalidPolicy,
    EmptyRunId,
}

pub struct EnduranceCampaignEvaluator {
    policy: EnduranceCampaignPolicy,
}

impl EnduranceCampaignEvaluator {
    pub fn new(policy: EnduranceCampaignPolicy) -> Result<Self, EnduranceCampaignError> {
        if policy.minimum_duration_ms == 0
            || policy.maximum_sample_gap_ms == 0
            || !policy.maximum_memory_growth_bytes_per_hour.is_finite()
            || policy.maximum_memory_growth_bytes_per_hour < 0.0
            || !policy.maximum_cpu_utilization.is_finite()
            || policy.maximum_cpu_utilization <= 0.0
            || !policy.maximum_temperature_c.is_finite()
        {
            return Err(EnduranceCampaignError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        runs: &[EnduranceRun],
    ) -> Result<EnduranceCampaignReport, EnduranceCampaignError> {
        if runs.iter().any(|run| run.run_id.trim().is_empty()) {
            return Err(EnduranceCampaignError::EmptyRunId);
        }
        let mut issues = Vec::new();
        for seed in &self.policy.required_seeds {
            let count = runs.iter().filter(|run| run.seed == *seed).count();
            if count == 0 {
                issues.push(EnduranceCampaignIssue::MissingSeed { seed: *seed });
            } else if count > 1 {
                issues.push(EnduranceCampaignIssue::DuplicateSeed { seed: *seed });
            }
        }

        let mut metrics = Vec::new();
        for run in runs {
            metrics.push(self.assess_run(run, &mut issues));
        }
        metrics.sort_by_key(|metric| metric.seed);
        issues.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        let status = if issues.iter().any(issue_is_failure) {
            EnduranceCampaignStatus::Fail
        } else if issues.is_empty() {
            EnduranceCampaignStatus::Pass
        } else {
            EnduranceCampaignStatus::Incomplete
        };
        Ok(EnduranceCampaignReport {
            status,
            run_metrics: metrics,
            issues,
        })
    }

    fn assess_run(
        &self,
        run: &EnduranceRun,
        issues: &mut Vec<EnduranceCampaignIssue>,
    ) -> EnduranceRunMetrics {
        let duration_ms = run.ended_at_ms.saturating_sub(run.started_at_ms);
        if run.ended_at_ms <= run.started_at_ms {
            issues.push(EnduranceCampaignIssue::InvalidRunWindow { seed: run.seed });
        } else if duration_ms < self.policy.minimum_duration_ms {
            issues.push(EnduranceCampaignIssue::InsufficientDuration {
                seed: run.seed,
                observed_ms: duration_ms,
                required_ms: self.policy.minimum_duration_ms,
            });
        }
        for phase in &self.policy.required_phases {
            if !run.phases_exercised.contains(phase) {
                issues.push(EnduranceCampaignIssue::MissingPhase {
                    seed: run.seed,
                    phase: *phase,
                });
            }
        }
        if run.unplanned_restart_count > self.policy.maximum_unplanned_restarts {
            issues.push(EnduranceCampaignIssue::UnplannedRestartsExceeded {
                seed: run.seed,
                observed: run.unplanned_restart_count,
                limit: self.policy.maximum_unplanned_restarts,
            });
        }
        if run.safety_incident_count > 0 {
            issues.push(EnduranceCampaignIssue::SafetyIncidentObserved {
                seed: run.seed,
                count: run.safety_incident_count,
            });
        }
        if run
            .terminal_evidence_digest
            .as_deref()
            .unwrap_or("")
            .is_empty()
        {
            issues.push(EnduranceCampaignIssue::MissingEvidenceDigest {
                seed: run.seed,
                timestamp_ms: None,
            });
        }
        if run.samples.is_empty() {
            issues.push(EnduranceCampaignIssue::MissingSamples { seed: run.seed });
            return EnduranceRunMetrics {
                seed: run.seed,
                duration_ms,
                memory_growth_bytes_per_hour: None,
                maximum_queue_depth: None,
                maximum_cpu_utilization: None,
                maximum_temperature_c: None,
                deadline_misses: None,
                watchdog_resets: None,
            };
        }

        let mut samples = run.samples.iter().collect::<Vec<_>>();
        samples.sort_by_key(|sample| sample.timestamp_ms);
        let mut previous = None;
        for sample in &samples {
            if let Some(previous_ms) = previous {
                if sample.timestamp_ms <= previous_ms {
                    issues.push(EnduranceCampaignIssue::NonMonotonicSampleTime {
                        seed: run.seed,
                        timestamp_ms: sample.timestamp_ms,
                    });
                }
                let gap_ms = sample.timestamp_ms.saturating_sub(previous_ms);
                if gap_ms > self.policy.maximum_sample_gap_ms {
                    issues.push(EnduranceCampaignIssue::SampleGapExceeded {
                        seed: run.seed,
                        gap_ms,
                    });
                }
            }
            previous = Some(sample.timestamp_ms);
            if sample
                .evidence_chain_digest
                .as_deref()
                .unwrap_or("")
                .is_empty()
            {
                issues.push(EnduranceCampaignIssue::MissingEvidenceDigest {
                    seed: run.seed,
                    timestamp_ms: Some(sample.timestamp_ms),
                });
            }
        }

        let first = samples.first().copied().unwrap();
        let last = samples.last().copied().unwrap();
        let sample_hours = (last.timestamp_ms.saturating_sub(first.timestamp_ms) as f64
            / 3_600_000.0)
            .max(f64::EPSILON);
        let memory_growth = last
            .resident_memory_bytes
            .saturating_sub(first.resident_memory_bytes) as f64
            / sample_hours;
        let max_queue = samples
            .iter()
            .map(|sample| sample.queue_depth)
            .max()
            .unwrap_or(0);
        let max_cpu = samples
            .iter()
            .map(|sample| sample.cpu_utilization)
            .fold(0.0_f64, f64::max);
        let max_temp = samples
            .iter()
            .map(|sample| sample.maximum_temperature_c)
            .fold(f64::NEG_INFINITY, f64::max);
        let deadline_misses = last
            .cumulative_deadline_misses
            .saturating_sub(first.cumulative_deadline_misses);
        let watchdog_resets = last
            .cumulative_watchdog_resets
            .saturating_sub(first.cumulative_watchdog_resets);

        if memory_growth > self.policy.maximum_memory_growth_bytes_per_hour {
            issues.push(EnduranceCampaignIssue::MemoryGrowthExceeded {
                seed: run.seed,
                bytes_per_hour: memory_growth,
                limit: self.policy.maximum_memory_growth_bytes_per_hour,
            });
        }
        if max_queue > self.policy.maximum_queue_depth {
            issues.push(EnduranceCampaignIssue::QueueDepthExceeded {
                seed: run.seed,
                observed: max_queue,
                limit: self.policy.maximum_queue_depth,
            });
        }
        if max_cpu > self.policy.maximum_cpu_utilization {
            issues.push(EnduranceCampaignIssue::CpuUtilizationExceeded {
                seed: run.seed,
                observed: max_cpu,
                limit: self.policy.maximum_cpu_utilization,
            });
        }
        if max_temp > self.policy.maximum_temperature_c {
            issues.push(EnduranceCampaignIssue::TemperatureExceeded {
                seed: run.seed,
                observed_c: max_temp,
                limit_c: self.policy.maximum_temperature_c,
            });
        }
        if deadline_misses > self.policy.maximum_deadline_misses {
            issues.push(EnduranceCampaignIssue::DeadlineMissesExceeded {
                seed: run.seed,
                observed: deadline_misses,
                limit: self.policy.maximum_deadline_misses,
            });
        }
        if watchdog_resets > self.policy.maximum_watchdog_resets {
            issues.push(EnduranceCampaignIssue::WatchdogResetsExceeded {
                seed: run.seed,
                observed: watchdog_resets,
                limit: self.policy.maximum_watchdog_resets,
            });
        }

        EnduranceRunMetrics {
            seed: run.seed,
            duration_ms,
            memory_growth_bytes_per_hour: Some(memory_growth),
            maximum_queue_depth: Some(max_queue),
            maximum_cpu_utilization: Some(max_cpu),
            maximum_temperature_c: Some(max_temp),
            deadline_misses: Some(deadline_misses),
            watchdog_resets: Some(watchdog_resets),
        }
    }
}

fn issue_is_failure(issue: &EnduranceCampaignIssue) -> bool {
    !matches!(
        issue,
        EnduranceCampaignIssue::MissingSeed { .. }
            | EnduranceCampaignIssue::MissingPhase { .. }
            | EnduranceCampaignIssue::MissingSamples { .. }
            | EnduranceCampaignIssue::MissingEvidenceDigest { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run() -> EnduranceRun {
        let phases = EnduranceCampaignPolicy::default().required_phases;
        EnduranceRun {
            run_id: "soak-1".into(),
            seed: 1,
            aircraft_id: "a1".into(),
            deployment_id: "d1".into(),
            started_at_ms: 0,
            ended_at_ms: 3_600_000,
            phases_exercised: phases,
            samples: vec![
                EnduranceSample {
                    timestamp_ms: 0,
                    resident_memory_bytes: 100,
                    heap_bytes: 50,
                    queue_depth: 1,
                    cpu_utilization: 0.2,
                    maximum_temperature_c: 40.0,
                    cumulative_deadline_misses: 0,
                    cumulative_watchdog_resets: 0,
                    evidence_chain_digest: Some("a".into()),
                },
                EnduranceSample {
                    timestamp_ms: 3_600_000,
                    resident_memory_bytes: 200,
                    heap_bytes: 60,
                    queue_depth: 2,
                    cpu_utilization: 0.3,
                    maximum_temperature_c: 45.0,
                    cumulative_deadline_misses: 0,
                    cumulative_watchdog_resets: 0,
                    evidence_chain_digest: Some("b".into()),
                },
            ],
            planned_restart_count: 0,
            unplanned_restart_count: 0,
            safety_incident_count: 0,
            terminal_evidence_digest: Some("final".into()),
        }
    }

    fn short_policy() -> EnduranceCampaignPolicy {
        EnduranceCampaignPolicy {
            minimum_duration_ms: 3_600_000,
            maximum_sample_gap_ms: 3_600_000,
            ..EnduranceCampaignPolicy::default()
        }
    }

    #[test]
    fn stable_soak_run_passes() {
        let evaluator = EnduranceCampaignEvaluator::new(short_policy()).unwrap();
        let report = evaluator.assess(&[run()]).unwrap();
        assert_eq!(report.status, EnduranceCampaignStatus::Pass);
    }

    #[test]
    fn memory_growth_failure_is_detected() {
        let mut sample = run();
        sample.samples[1].resident_memory_bytes = 10_000_000;
        let mut policy = short_policy();
        policy.maximum_memory_growth_bytes_per_hour = 100.0;
        let evaluator = EnduranceCampaignEvaluator::new(policy).unwrap();
        let report = evaluator.assess(&[sample]).unwrap();
        assert_eq!(report.status, EnduranceCampaignStatus::Fail);
    }

    #[test]
    fn missing_required_seed_is_incomplete() {
        let mut policy = short_policy();
        policy.required_seeds.insert(9);
        let evaluator = EnduranceCampaignEvaluator::new(policy).unwrap();
        let report = evaluator.assess(&[]).unwrap();
        assert_eq!(report.status, EnduranceCampaignStatus::Incomplete);
    }
}
