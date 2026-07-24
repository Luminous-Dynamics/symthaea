// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic fault-recovery qualification campaigns.
//!
//! A recovery claim is evaluated from complete per-seed observations rather
//! than one favorable trace. Detection, reconfiguration, stabilization, safe-
//! state timing, control error, and evidence completeness are independently
//! gated with explicit Pass, Fail, and Incomplete semantics.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecoveryFaultClass {
    EngineFlameout,
    MainRotorDegradation,
    TailRotorLoss,
    CollectiveJam,
    CyclicLongitudinalJam,
    CyclicLateralJam,
    NavigationLoss,
    TimingOverrun,
    SensorBias,
    PowerBusLoss,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultRecoveryScenario {
    pub scenario_id: String,
    pub fault: RecoveryFaultClass,
    pub required_seeds: Vec<u64>,
    pub maximum_detection_latency_s: f64,
    pub maximum_reconfiguration_latency_s: f64,
    pub maximum_stabilization_latency_s: f64,
    pub maximum_safe_state_latency_s: f64,
    pub maximum_tracking_error: f64,
    pub safe_state_required: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultRecoveryObservation {
    pub scenario_id: String,
    pub seed: u64,
    pub fault: RecoveryFaultClass,
    pub completed: bool,
    pub detected: bool,
    pub reconfigured: bool,
    pub stabilized: bool,
    pub safe_state_reached: bool,
    pub detection_latency_s: Option<f64>,
    pub reconfiguration_latency_s: Option<f64>,
    pub stabilization_latency_s: Option<f64>,
    pub safe_state_latency_s: Option<f64>,
    pub peak_tracking_error: f64,
    pub evidence_segment_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaultRecoveryStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FaultRecoveryIssue {
    MissingSeed(u64),
    DuplicateSeed(u64),
    FaultMismatch {
        seed: u64,
    },
    ObservationIncomplete(u64),
    MissingEvidence(u64),
    FaultNotDetected(u64),
    ReconfigurationMissing(u64),
    StabilizationMissing(u64),
    SafeStateMissing(u64),
    DetectionLatencyExceeded {
        seed: u64,
        observed_s: f64,
        maximum_s: f64,
    },
    ReconfigurationLatencyExceeded {
        seed: u64,
        observed_s: f64,
        maximum_s: f64,
    },
    StabilizationLatencyExceeded {
        seed: u64,
        observed_s: f64,
        maximum_s: f64,
    },
    SafeStateLatencyExceeded {
        seed: u64,
        observed_s: f64,
        maximum_s: f64,
    },
    TrackingErrorExceeded {
        seed: u64,
        observed: f64,
        maximum: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultRecoveryScenarioReport {
    pub scenario_id: String,
    pub fault: RecoveryFaultClass,
    pub status: FaultRecoveryStatus,
    pub observed_seeds: Vec<u64>,
    pub worst_detection_latency_s: Option<f64>,
    pub worst_reconfiguration_latency_s: Option<f64>,
    pub worst_stabilization_latency_s: Option<f64>,
    pub worst_safe_state_latency_s: Option<f64>,
    pub worst_tracking_error: f64,
    pub issues: Vec<FaultRecoveryIssue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultRecoveryCampaignReport {
    pub schema_version: String,
    pub campaign_id: String,
    pub status: FaultRecoveryStatus,
    pub scenario_reports: Vec<FaultRecoveryScenarioReport>,
    pub required_faults: Vec<RecoveryFaultClass>,
    pub exercised_faults: Vec<RecoveryFaultClass>,
    pub missing_faults: Vec<RecoveryFaultClass>,
}

impl FaultRecoveryCampaignReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, FaultRecoveryCampaignError> {
        let mut canonical = self.clone();
        canonical
            .scenario_reports
            .sort_by(|a, b| a.scenario_id.cmp(&b.scenario_id));
        for report in &mut canonical.scenario_reports {
            report.observed_seeds.sort_unstable();
            report.issues.sort_by_key(issue_sort_key);
        }
        canonical.required_faults.sort();
        canonical.exercised_faults.sort();
        canonical.missing_faults.sort();
        serde_json::to_vec(&canonical).map_err(|_| FaultRecoveryCampaignError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, FaultRecoveryCampaignError> {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in self.canonical_json()? {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FaultRecoveryCampaignError {
    InvalidCampaign,
    DuplicateScenarioId(String),
    InvalidScenario(String),
    UnknownScenario(String),
    InvalidObservation { scenario_id: String, seed: u64 },
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct FaultRecoveryCampaign {
    schema_version: String,
    campaign_id: String,
    scenarios: Vec<FaultRecoveryScenario>,
    required_faults: Vec<RecoveryFaultClass>,
}

impl FaultRecoveryCampaign {
    pub fn new(
        schema_version: impl Into<String>,
        campaign_id: impl Into<String>,
        scenarios: Vec<FaultRecoveryScenario>,
        required_faults: Vec<RecoveryFaultClass>,
    ) -> Result<Self, FaultRecoveryCampaignError> {
        let schema_version = schema_version.into();
        let campaign_id = campaign_id.into();
        if schema_version.trim().is_empty() || campaign_id.trim().is_empty() || scenarios.is_empty()
        {
            return Err(FaultRecoveryCampaignError::InvalidCampaign);
        }
        let required_set: BTreeSet<_> = required_faults.iter().copied().collect();
        if required_set.len() != required_faults.len() || required_faults.is_empty() {
            return Err(FaultRecoveryCampaignError::InvalidCampaign);
        }
        let mut ids = BTreeSet::new();
        for scenario in &scenarios {
            if !ids.insert(scenario.scenario_id.clone()) {
                return Err(FaultRecoveryCampaignError::DuplicateScenarioId(
                    scenario.scenario_id.clone(),
                ));
            }
            let unique_seeds: BTreeSet<_> = scenario.required_seeds.iter().copied().collect();
            let limits = [
                scenario.maximum_detection_latency_s,
                scenario.maximum_reconfiguration_latency_s,
                scenario.maximum_stabilization_latency_s,
                scenario.maximum_safe_state_latency_s,
                scenario.maximum_tracking_error,
            ];
            if scenario.scenario_id.trim().is_empty()
                || scenario.required_seeds.is_empty()
                || unique_seeds.len() != scenario.required_seeds.len()
                || limits
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(FaultRecoveryCampaignError::InvalidScenario(
                    scenario.scenario_id.clone(),
                ));
            }
        }
        Ok(Self {
            schema_version,
            campaign_id,
            scenarios,
            required_faults,
        })
    }

    pub fn evaluate(
        &self,
        observations: &[FaultRecoveryObservation],
    ) -> Result<FaultRecoveryCampaignReport, FaultRecoveryCampaignError> {
        let scenario_map: BTreeMap<_, _> = self
            .scenarios
            .iter()
            .map(|scenario| (scenario.scenario_id.as_str(), scenario))
            .collect();
        for observation in observations {
            if !scenario_map.contains_key(observation.scenario_id.as_str()) {
                return Err(FaultRecoveryCampaignError::UnknownScenario(
                    observation.scenario_id.clone(),
                ));
            }
            if !observation.peak_tracking_error.is_finite()
                || observation.peak_tracking_error < 0.0
                || [
                    observation.detection_latency_s,
                    observation.reconfiguration_latency_s,
                    observation.stabilization_latency_s,
                    observation.safe_state_latency_s,
                ]
                .into_iter()
                .flatten()
                .any(|value| !value.is_finite() || value < 0.0)
            {
                return Err(FaultRecoveryCampaignError::InvalidObservation {
                    scenario_id: observation.scenario_id.clone(),
                    seed: observation.seed,
                });
            }
        }

        let mut reports = Vec::with_capacity(self.scenarios.len());
        let mut exercised_faults = BTreeSet::new();
        for scenario in &self.scenarios {
            let selected: Vec<_> = observations
                .iter()
                .filter(|observation| observation.scenario_id == scenario.scenario_id)
                .collect();
            if !selected.is_empty() {
                exercised_faults.insert(scenario.fault);
            }
            reports.push(evaluate_scenario(scenario, &selected));
        }

        let required_faults: BTreeSet<_> = self.required_faults.iter().copied().collect();
        let missing_faults: Vec<_> = required_faults
            .difference(&exercised_faults)
            .copied()
            .collect();
        let any_fail = reports
            .iter()
            .any(|report| report.status == FaultRecoveryStatus::Fail);
        let any_incomplete = reports
            .iter()
            .any(|report| report.status == FaultRecoveryStatus::Incomplete)
            || !missing_faults.is_empty();
        let status = if any_fail {
            FaultRecoveryStatus::Fail
        } else if any_incomplete {
            FaultRecoveryStatus::Incomplete
        } else {
            FaultRecoveryStatus::Pass
        };

        Ok(FaultRecoveryCampaignReport {
            schema_version: self.schema_version.clone(),
            campaign_id: self.campaign_id.clone(),
            status,
            scenario_reports: reports,
            required_faults: required_faults.into_iter().collect(),
            exercised_faults: exercised_faults.into_iter().collect(),
            missing_faults,
        })
    }
}

fn evaluate_scenario(
    scenario: &FaultRecoveryScenario,
    observations: &[&FaultRecoveryObservation],
) -> FaultRecoveryScenarioReport {
    let mut by_seed: BTreeMap<u64, Vec<&FaultRecoveryObservation>> = BTreeMap::new();
    for observation in observations {
        by_seed
            .entry(observation.seed)
            .or_default()
            .push(*observation);
    }
    let mut issues = Vec::new();
    let mut observed_seeds = Vec::new();
    let mut worst_detection: Option<f64> = None;
    let mut worst_reconfiguration: Option<f64> = None;
    let mut worst_stabilization: Option<f64> = None;
    let mut worst_safe_state: Option<f64> = None;
    let mut worst_error: f64 = 0.0;

    for seed in &scenario.required_seeds {
        let Some(entries) = by_seed.get(seed) else {
            issues.push(FaultRecoveryIssue::MissingSeed(*seed));
            continue;
        };
        if entries.len() != 1 {
            issues.push(FaultRecoveryIssue::DuplicateSeed(*seed));
            continue;
        }
        let observation = entries[0];
        observed_seeds.push(*seed);
        if observation.fault != scenario.fault {
            issues.push(FaultRecoveryIssue::FaultMismatch { seed: *seed });
        }
        if !observation.completed {
            issues.push(FaultRecoveryIssue::ObservationIncomplete(*seed));
        }
        if observation
            .evidence_segment_id
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
        {
            issues.push(FaultRecoveryIssue::MissingEvidence(*seed));
        }
        check_stage(
            *seed,
            observation.detected,
            observation.detection_latency_s,
            scenario.maximum_detection_latency_s,
            FaultRecoveryIssue::FaultNotDetected,
            |seed, observed_s, maximum_s| FaultRecoveryIssue::DetectionLatencyExceeded {
                seed,
                observed_s,
                maximum_s,
            },
            &mut worst_detection,
            &mut issues,
        );
        check_stage(
            *seed,
            observation.reconfigured,
            observation.reconfiguration_latency_s,
            scenario.maximum_reconfiguration_latency_s,
            FaultRecoveryIssue::ReconfigurationMissing,
            |seed, observed_s, maximum_s| FaultRecoveryIssue::ReconfigurationLatencyExceeded {
                seed,
                observed_s,
                maximum_s,
            },
            &mut worst_reconfiguration,
            &mut issues,
        );
        check_stage(
            *seed,
            observation.stabilized,
            observation.stabilization_latency_s,
            scenario.maximum_stabilization_latency_s,
            FaultRecoveryIssue::StabilizationMissing,
            |seed, observed_s, maximum_s| FaultRecoveryIssue::StabilizationLatencyExceeded {
                seed,
                observed_s,
                maximum_s,
            },
            &mut worst_stabilization,
            &mut issues,
        );
        if scenario.safe_state_required {
            check_stage(
                *seed,
                observation.safe_state_reached,
                observation.safe_state_latency_s,
                scenario.maximum_safe_state_latency_s,
                FaultRecoveryIssue::SafeStateMissing,
                |seed, observed_s, maximum_s| FaultRecoveryIssue::SafeStateLatencyExceeded {
                    seed,
                    observed_s,
                    maximum_s,
                },
                &mut worst_safe_state,
                &mut issues,
            );
        }
        worst_error = worst_error.max(observation.peak_tracking_error);
        if observation.peak_tracking_error > scenario.maximum_tracking_error {
            issues.push(FaultRecoveryIssue::TrackingErrorExceeded {
                seed: *seed,
                observed: observation.peak_tracking_error,
                maximum: scenario.maximum_tracking_error,
            });
        }
    }

    let incomplete = issues.iter().any(|issue| {
        matches!(
            issue,
            FaultRecoveryIssue::MissingSeed(_)
                | FaultRecoveryIssue::DuplicateSeed(_)
                | FaultRecoveryIssue::ObservationIncomplete(_)
                | FaultRecoveryIssue::MissingEvidence(_)
        )
    });
    let failed = issues.iter().any(|issue| {
        !matches!(
            issue,
            FaultRecoveryIssue::MissingSeed(_)
                | FaultRecoveryIssue::DuplicateSeed(_)
                | FaultRecoveryIssue::ObservationIncomplete(_)
                | FaultRecoveryIssue::MissingEvidence(_)
        )
    });
    let status = if failed {
        FaultRecoveryStatus::Fail
    } else if incomplete {
        FaultRecoveryStatus::Incomplete
    } else {
        FaultRecoveryStatus::Pass
    };
    FaultRecoveryScenarioReport {
        scenario_id: scenario.scenario_id.clone(),
        fault: scenario.fault,
        status,
        observed_seeds,
        worst_detection_latency_s: worst_detection,
        worst_reconfiguration_latency_s: worst_reconfiguration,
        worst_stabilization_latency_s: worst_stabilization,
        worst_safe_state_latency_s: worst_safe_state,
        worst_tracking_error: worst_error,
        issues,
    }
}

#[allow(clippy::too_many_arguments)]
fn check_stage<M, L>(
    seed: u64,
    achieved: bool,
    latency: Option<f64>,
    maximum: f64,
    missing: M,
    latency_issue: L,
    worst: &mut Option<f64>,
    issues: &mut Vec<FaultRecoveryIssue>,
) where
    M: Fn(u64) -> FaultRecoveryIssue,
    L: Fn(u64, f64, f64) -> FaultRecoveryIssue,
{
    if !achieved {
        issues.push(missing(seed));
        return;
    }
    let Some(observed) = latency else {
        issues.push(missing(seed));
        return;
    };
    *worst = Some(worst.map_or(observed, |current| current.max(observed)));
    if observed > maximum {
        issues.push(latency_issue(seed, observed, maximum));
    }
}

fn issue_sort_key(issue: &FaultRecoveryIssue) -> String {
    format!("{issue:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario() -> FaultRecoveryScenario {
        FaultRecoveryScenario {
            scenario_id: "tail-loss".into(),
            fault: RecoveryFaultClass::TailRotorLoss,
            required_seeds: vec![1, 2],
            maximum_detection_latency_s: 0.2,
            maximum_reconfiguration_latency_s: 0.5,
            maximum_stabilization_latency_s: 2.0,
            maximum_safe_state_latency_s: 20.0,
            maximum_tracking_error: 3.0,
            safe_state_required: true,
        }
    }

    fn observation(seed: u64) -> FaultRecoveryObservation {
        FaultRecoveryObservation {
            scenario_id: "tail-loss".into(),
            seed,
            fault: RecoveryFaultClass::TailRotorLoss,
            completed: true,
            detected: true,
            reconfigured: true,
            stabilized: true,
            safe_state_reached: true,
            detection_latency_s: Some(0.1),
            reconfiguration_latency_s: Some(0.3),
            stabilization_latency_s: Some(1.0),
            safe_state_latency_s: Some(10.0),
            peak_tracking_error: 2.0,
            evidence_segment_id: Some(format!("segment:{seed}")),
        }
    }

    fn campaign() -> FaultRecoveryCampaign {
        FaultRecoveryCampaign::new(
            "symthaea.helicopter.fault-recovery.v1",
            "campaign-a",
            vec![scenario()],
            vec![RecoveryFaultClass::TailRotorLoss],
        )
        .unwrap()
    }

    #[test]
    fn complete_campaign_passes() {
        let report = campaign()
            .evaluate(&[observation(1), observation(2)])
            .unwrap();
        assert_eq!(report.status, FaultRecoveryStatus::Pass);
    }

    #[test]
    fn missing_seed_is_incomplete() {
        let report = campaign().evaluate(&[observation(1)]).unwrap();
        assert_eq!(report.status, FaultRecoveryStatus::Incomplete);
    }

    #[test]
    fn late_detection_fails() {
        let first = observation(1);
        let mut second = observation(2);
        second.detection_latency_s = Some(0.4);
        let report = campaign().evaluate(&[first, second]).unwrap();
        assert_eq!(report.status, FaultRecoveryStatus::Fail);
    }

    #[test]
    fn report_digest_is_order_stable() {
        let first = campaign()
            .evaluate(&[observation(1), observation(2)])
            .unwrap()
            .digest_fnv1a64()
            .unwrap();
        let second = campaign()
            .evaluate(&[observation(2), observation(1)])
            .unwrap()
            .digest_fnv1a64()
            .unwrap();
        assert_eq!(first, second);
    }
}
