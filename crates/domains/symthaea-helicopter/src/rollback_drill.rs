// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Executable update rollback drill evidence.
//!
//! A rollback capability is not accepted merely because a rollback function
//! exists. Each declared drill must demonstrate detection, authority transfer,
//! bank selection, state preservation, restart, and restored health within a
//! bounded time.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RollbackDrillStage {
    FaultDetected,
    TrialRejected,
    OutputsDisarmed,
    PreviousBankSelected,
    StateRestored,
    RestartCompleted,
    HealthVerified,
    AuthorityRestored,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackStageEvidence {
    pub stage: RollbackDrillStage,
    pub timestamp_ms: u64,
    pub passed: bool,
    pub evidence_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackDrillObservation {
    pub drill_id: String,
    pub seed: u64,
    pub aircraft_id: String,
    pub deployment_id: String,
    pub failed_version: String,
    pub restored_version: String,
    pub expected_restored_version: String,
    pub pre_drill_anti_rollback_counter: u64,
    pub post_drill_anti_rollback_counter: u64,
    pub state_digest_before: String,
    pub state_digest_after: String,
    pub safety_state_preserved: bool,
    pub flight_outputs_disarmed_during_switch: bool,
    pub stages: Vec<RollbackStageEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackDrillPolicy {
    pub required_seeds: BTreeSet<u64>,
    pub required_stages: BTreeSet<RollbackDrillStage>,
    pub maximum_detection_ms: u64,
    pub maximum_recovery_ms: u64,
    pub require_state_digest_match: bool,
    pub require_counter_non_decrease: bool,
}

impl Default for RollbackDrillPolicy {
    fn default() -> Self {
        Self {
            required_seeds: BTreeSet::new(),
            required_stages: BTreeSet::from([
                RollbackDrillStage::FaultDetected,
                RollbackDrillStage::TrialRejected,
                RollbackDrillStage::OutputsDisarmed,
                RollbackDrillStage::PreviousBankSelected,
                RollbackDrillStage::StateRestored,
                RollbackDrillStage::RestartCompleted,
                RollbackDrillStage::HealthVerified,
                RollbackDrillStage::AuthorityRestored,
            ]),
            maximum_detection_ms: 1_000,
            maximum_recovery_ms: 10_000,
            require_state_digest_match: true,
            require_counter_non_decrease: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RollbackDrillStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RollbackDrillIssue {
    MissingSeed {
        seed: u64,
    },
    DuplicateSeed {
        seed: u64,
    },
    MissingStage {
        seed: u64,
        stage: RollbackDrillStage,
    },
    FailedStage {
        seed: u64,
        stage: RollbackDrillStage,
    },
    MissingStageEvidence {
        seed: u64,
        stage: RollbackDrillStage,
    },
    NonMonotonicStageTime {
        seed: u64,
        stage: RollbackDrillStage,
    },
    DetectionDeadlineMissed {
        seed: u64,
        observed_ms: u64,
        limit_ms: u64,
    },
    RecoveryDeadlineMissed {
        seed: u64,
        observed_ms: u64,
        limit_ms: u64,
    },
    WrongRestoredVersion {
        seed: u64,
        expected: String,
        observed: String,
    },
    AntiRollbackCounterDecreased {
        seed: u64,
        before: u64,
        after: u64,
    },
    StateNotPreserved {
        seed: u64,
    },
    StateDigestMismatch {
        seed: u64,
    },
    OutputsNotDisarmed {
        seed: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackDrillReport {
    pub status: RollbackDrillStatus,
    pub required_drills: usize,
    pub observed_drills: usize,
    pub passing_drills: usize,
    pub maximum_observed_recovery_ms: Option<u64>,
    pub issues: Vec<RollbackDrillIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollbackDrillError {
    InvalidPolicy,
    EmptyDrillId,
}

pub struct RollbackDrillEvaluator {
    policy: RollbackDrillPolicy,
}

impl RollbackDrillEvaluator {
    pub fn new(policy: RollbackDrillPolicy) -> Result<Self, RollbackDrillError> {
        if policy.maximum_detection_ms == 0 || policy.maximum_recovery_ms == 0 {
            return Err(RollbackDrillError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        observations: &[RollbackDrillObservation],
    ) -> Result<RollbackDrillReport, RollbackDrillError> {
        if observations
            .iter()
            .any(|observation| observation.drill_id.trim().is_empty())
        {
            return Err(RollbackDrillError::EmptyDrillId);
        }

        let mut issues = Vec::new();
        let mut by_seed = BTreeMap::<u64, Vec<&RollbackDrillObservation>>::new();
        for observation in observations {
            by_seed
                .entry(observation.seed)
                .or_default()
                .push(observation);
        }
        for seed in &self.policy.required_seeds {
            if !by_seed.contains_key(seed) {
                issues.push(RollbackDrillIssue::MissingSeed { seed: *seed });
            }
        }
        for (seed, matching) in &by_seed {
            if matching.len() > 1 {
                issues.push(RollbackDrillIssue::DuplicateSeed { seed: *seed });
            }
        }

        let mut passing_drills = 0usize;
        let mut maximum_observed_recovery_ms = None;
        for observation in observations {
            let before = issues.len();
            let recovery = self.assess_observation(observation, &mut issues);
            if let Some(recovery_ms) = recovery {
                maximum_observed_recovery_ms = Some(
                    maximum_observed_recovery_ms
                        .map(|current: u64| current.max(recovery_ms))
                        .unwrap_or(recovery_ms),
                );
            }
            if issues.len() == before && by_seed[&observation.seed].len() == 1 {
                passing_drills += 1;
            }
        }

        issues.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        let status = if issues.iter().any(issue_is_failure) {
            RollbackDrillStatus::Fail
        } else if issues.is_empty() {
            RollbackDrillStatus::Pass
        } else {
            RollbackDrillStatus::Incomplete
        };
        Ok(RollbackDrillReport {
            status,
            required_drills: self.policy.required_seeds.len(),
            observed_drills: observations.len(),
            passing_drills,
            maximum_observed_recovery_ms,
            issues,
        })
    }

    fn assess_observation(
        &self,
        observation: &RollbackDrillObservation,
        issues: &mut Vec<RollbackDrillIssue>,
    ) -> Option<u64> {
        if observation.restored_version != observation.expected_restored_version {
            issues.push(RollbackDrillIssue::WrongRestoredVersion {
                seed: observation.seed,
                expected: observation.expected_restored_version.clone(),
                observed: observation.restored_version.clone(),
            });
        }
        if self.policy.require_counter_non_decrease
            && observation.post_drill_anti_rollback_counter
                < observation.pre_drill_anti_rollback_counter
        {
            issues.push(RollbackDrillIssue::AntiRollbackCounterDecreased {
                seed: observation.seed,
                before: observation.pre_drill_anti_rollback_counter,
                after: observation.post_drill_anti_rollback_counter,
            });
        }
        if !observation.safety_state_preserved {
            issues.push(RollbackDrillIssue::StateNotPreserved {
                seed: observation.seed,
            });
        }
        if self.policy.require_state_digest_match
            && observation.state_digest_before != observation.state_digest_after
        {
            issues.push(RollbackDrillIssue::StateDigestMismatch {
                seed: observation.seed,
            });
        }
        if !observation.flight_outputs_disarmed_during_switch {
            issues.push(RollbackDrillIssue::OutputsNotDisarmed {
                seed: observation.seed,
            });
        }

        let mut by_stage = BTreeMap::<RollbackDrillStage, &RollbackStageEvidence>::new();
        for stage in &observation.stages {
            by_stage.insert(stage.stage, stage);
        }
        let mut last_time = None;
        for stage_kind in &self.policy.required_stages {
            let Some(stage) = by_stage.get(stage_kind) else {
                issues.push(RollbackDrillIssue::MissingStage {
                    seed: observation.seed,
                    stage: *stage_kind,
                });
                continue;
            };
            if !stage.passed {
                issues.push(RollbackDrillIssue::FailedStage {
                    seed: observation.seed,
                    stage: *stage_kind,
                });
            }
            if stage.evidence_id.as_deref().unwrap_or("").trim().is_empty() {
                issues.push(RollbackDrillIssue::MissingStageEvidence {
                    seed: observation.seed,
                    stage: *stage_kind,
                });
            }
            if let Some(previous) = last_time {
                if stage.timestamp_ms < previous {
                    issues.push(RollbackDrillIssue::NonMonotonicStageTime {
                        seed: observation.seed,
                        stage: *stage_kind,
                    });
                }
            }
            last_time = Some(stage.timestamp_ms);
        }

        let first = observation
            .stages
            .iter()
            .map(|stage| stage.timestamp_ms)
            .min()?;
        let detection = by_stage
            .get(&RollbackDrillStage::FaultDetected)
            .map(|stage| stage.timestamp_ms)?;
        let authority = by_stage
            .get(&RollbackDrillStage::AuthorityRestored)
            .map(|stage| stage.timestamp_ms)?;
        let detection_ms = detection.saturating_sub(first);
        let recovery_ms = authority.saturating_sub(detection);
        if detection_ms > self.policy.maximum_detection_ms {
            issues.push(RollbackDrillIssue::DetectionDeadlineMissed {
                seed: observation.seed,
                observed_ms: detection_ms,
                limit_ms: self.policy.maximum_detection_ms,
            });
        }
        if recovery_ms > self.policy.maximum_recovery_ms {
            issues.push(RollbackDrillIssue::RecoveryDeadlineMissed {
                seed: observation.seed,
                observed_ms: recovery_ms,
                limit_ms: self.policy.maximum_recovery_ms,
            });
        }
        Some(recovery_ms)
    }
}

fn issue_is_failure(issue: &RollbackDrillIssue) -> bool {
    !matches!(
        issue,
        RollbackDrillIssue::MissingSeed { .. }
            | RollbackDrillIssue::MissingStage { .. }
            | RollbackDrillIssue::MissingStageEvidence { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(seed: u64) -> RollbackDrillObservation {
        let stages = RollbackDrillPolicy::default()
            .required_stages
            .iter()
            .enumerate()
            .map(|(index, stage)| RollbackStageEvidence {
                stage: *stage,
                timestamp_ms: index as u64 * 100,
                passed: true,
                evidence_id: Some(format!("ev-{index}")),
            })
            .collect();
        RollbackDrillObservation {
            drill_id: format!("drill-{seed}"),
            seed,
            aircraft_id: "aircraft-1".into(),
            deployment_id: "deploy-1".into(),
            failed_version: "2".into(),
            restored_version: "1".into(),
            expected_restored_version: "1".into(),
            pre_drill_anti_rollback_counter: 10,
            post_drill_anti_rollback_counter: 11,
            state_digest_before: "state".into(),
            state_digest_after: "state".into(),
            safety_state_preserved: true,
            flight_outputs_disarmed_during_switch: true,
            stages,
        }
    }

    #[test]
    fn complete_drill_passes() {
        let mut policy = RollbackDrillPolicy::default();
        policy.required_seeds.insert(7);
        let evaluator = RollbackDrillEvaluator::new(policy).unwrap();
        let report = evaluator.assess(&[observation(7)]).unwrap();
        assert_eq!(report.status, RollbackDrillStatus::Pass);
    }

    #[test]
    fn wrong_version_fails() {
        let mut sample = observation(7);
        sample.restored_version = "3".into();
        let evaluator = RollbackDrillEvaluator::new(RollbackDrillPolicy::default()).unwrap();
        let report = evaluator.assess(&[sample]).unwrap();
        assert_eq!(report.status, RollbackDrillStatus::Fail);
    }

    #[test]
    fn absent_required_seed_is_incomplete() {
        let mut policy = RollbackDrillPolicy::default();
        policy.required_seeds.insert(9);
        let evaluator = RollbackDrillEvaluator::new(policy).unwrap();
        let report = evaluator.assess(&[]).unwrap();
        assert_eq!(report.status, RollbackDrillStatus::Incomplete);
    }
}
