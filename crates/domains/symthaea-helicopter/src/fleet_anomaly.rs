// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Robust fleet-cohort anomaly and common-mode detection.
//!
//! Aircraft are compared only within a declared qualified configuration. A
//! median/MAD score isolates individual outliers, while qualified bounds and a
//! fleet fraction gate expose common-mode drift that cohort-relative methods
//! would otherwise normalize away.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FleetMetricBound {
    pub minimum: f64,
    pub maximum: f64,
    pub direction_higher_is_worse: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FleetAnomalyPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub fleet_id: String,
    pub qualified_configuration_digest: String,
    pub minimum_cohort_size: usize,
    pub quarantine_modified_z: f64,
    pub common_mode_fraction: f64,
    pub maximum_observation_age_ms: u64,
    pub metric_bounds: BTreeMap<String, FleetMetricBound>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FleetAircraftObservation {
    pub observation_id: String,
    pub fleet_id: String,
    pub aircraft_id: String,
    pub configuration_digest: String,
    pub timestamp_ms: u64,
    pub metrics: BTreeMap<String, f64>,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FleetAnomalyStatus {
    Normal,
    QuarantineAircraft,
    FleetHold,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FleetAnomalyIssue {
    DuplicateObservation(String),
    DuplicateAircraft(String),
    FleetMismatch(String),
    ConfigurationMismatch(String),
    InvalidMetric {
        aircraft_id: String,
        metric: String,
    },
    MissingMetric {
        aircraft_id: String,
        metric: String,
    },
    MissingEvidence(String),
    FutureObservation(String),
    StaleObservation {
        aircraft_id: String,
        age_ms: u64,
        maximum_ms: u64,
    },
    InsufficientCohort {
        observed: usize,
        required: usize,
    },
    AircraftOutlier {
        aircraft_id: String,
        metric: String,
        modified_z: f64,
    },
    QualifiedBoundViolation {
        aircraft_id: String,
        metric: String,
        value: f64,
    },
    CommonModeViolation {
        metric: String,
        violating_fraction: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FleetMetricAssessment {
    pub metric: String,
    pub cohort_median: f64,
    pub median_absolute_deviation: f64,
    pub target_value: f64,
    pub target_modified_z: f64,
    pub qualified_bound_violating_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FleetAnomalyReport {
    pub schema_version: String,
    pub policy_id: String,
    pub fleet_id: String,
    pub target_aircraft_id: String,
    pub assessed_at_ms: u64,
    pub status: FleetAnomalyStatus,
    pub cohort_size: usize,
    pub metrics: Vec<FleetMetricAssessment>,
    pub issues: Vec<FleetAnomalyIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FleetAnomalyError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct FleetAnomalyDetector {
    policy: FleetAnomalyPolicy,
}

impl FleetAnomalyDetector {
    pub fn new(policy: FleetAnomalyPolicy) -> Result<Self, FleetAnomalyError> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.fleet_id.trim().is_empty()
            || !valid_digest(&policy.qualified_configuration_digest)
            || policy.minimum_cohort_size < 3
            || !policy.quarantine_modified_z.is_finite()
            || policy.quarantine_modified_z <= 0.0
            || !policy.common_mode_fraction.is_finite()
            || !(0.0..=1.0).contains(&policy.common_mode_fraction)
            || policy.common_mode_fraction <= 0.0
            || policy.maximum_observation_age_ms == 0
            || policy.metric_bounds.is_empty()
            || policy.metric_bounds.iter().any(|(metric, bound)| {
                metric.trim().is_empty()
                    || !bound.minimum.is_finite()
                    || !bound.maximum.is_finite()
                    || bound.minimum >= bound.maximum
            })
        {
            return Err(FleetAnomalyError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        target_aircraft_id: &str,
        observations: &[FleetAircraftObservation],
        now_ms: u64,
    ) -> FleetAnomalyReport {
        let mut issues = Vec::new();
        let mut ids = BTreeSet::new();
        let mut aircraft = BTreeSet::new();
        let mut valid = Vec::new();
        for observation in observations {
            if observation.observation_id.trim().is_empty()
                || !ids.insert(observation.observation_id.clone())
            {
                issues.push(FleetAnomalyIssue::DuplicateObservation(
                    observation.observation_id.clone(),
                ));
            }
            if observation.aircraft_id.trim().is_empty()
                || !aircraft.insert(observation.aircraft_id.clone())
            {
                issues.push(FleetAnomalyIssue::DuplicateAircraft(
                    observation.aircraft_id.clone(),
                ));
            }
            if observation.fleet_id != self.policy.fleet_id {
                issues.push(FleetAnomalyIssue::FleetMismatch(
                    observation.aircraft_id.clone(),
                ));
                continue;
            }
            if observation.configuration_digest != self.policy.qualified_configuration_digest {
                issues.push(FleetAnomalyIssue::ConfigurationMismatch(
                    observation.aircraft_id.clone(),
                ));
                continue;
            }
            if observation.evidence_ids.is_empty()
                || observation
                    .evidence_ids
                    .iter()
                    .any(|id| id.trim().is_empty())
            {
                issues.push(FleetAnomalyIssue::MissingEvidence(
                    observation.aircraft_id.clone(),
                ));
            }
            if observation.timestamp_ms > now_ms {
                issues.push(FleetAnomalyIssue::FutureObservation(
                    observation.aircraft_id.clone(),
                ));
                continue;
            }
            let age = now_ms.saturating_sub(observation.timestamp_ms);
            if age > self.policy.maximum_observation_age_ms {
                issues.push(FleetAnomalyIssue::StaleObservation {
                    aircraft_id: observation.aircraft_id.clone(),
                    age_ms: age,
                    maximum_ms: self.policy.maximum_observation_age_ms,
                });
            }
            let mut metrics_valid = true;
            for metric in self.policy.metric_bounds.keys() {
                match observation.metrics.get(metric) {
                    None => {
                        issues.push(FleetAnomalyIssue::MissingMetric {
                            aircraft_id: observation.aircraft_id.clone(),
                            metric: metric.clone(),
                        });
                        metrics_valid = false;
                    }
                    Some(value) if !value.is_finite() => {
                        issues.push(FleetAnomalyIssue::InvalidMetric {
                            aircraft_id: observation.aircraft_id.clone(),
                            metric: metric.clone(),
                        });
                        metrics_valid = false;
                    }
                    Some(_) => {}
                }
            }
            if metrics_valid {
                valid.push(observation);
            }
        }
        if valid.len() < self.policy.minimum_cohort_size {
            issues.push(FleetAnomalyIssue::InsufficientCohort {
                observed: valid.len(),
                required: self.policy.minimum_cohort_size,
            });
        }
        let target = valid
            .iter()
            .find(|observation| observation.aircraft_id == target_aircraft_id);
        if target.is_none() {
            issues.push(FleetAnomalyIssue::MissingEvidence(
                target_aircraft_id.to_string(),
            ));
        }

        let mut metric_reports = Vec::new();
        if let Some(target) = target {
            for (metric, bound) in &self.policy.metric_bounds {
                let mut values: Vec<_> = valid.iter().map(|entry| entry.metrics[metric]).collect();
                values.sort_by(f64::total_cmp);
                let median_value = median(&values);
                let mut deviations: Vec<_> = values
                    .iter()
                    .map(|value| (value - median_value).abs())
                    .collect();
                deviations.sort_by(f64::total_cmp);
                let mad = median(&deviations);
                let target_value = target.metrics[metric];
                let modified_z = if mad <= f64::EPSILON {
                    if (target_value - median_value).abs() <= f64::EPSILON {
                        0.0
                    } else {
                        f64::INFINITY
                    }
                } else {
                    0.674_489_75 * (target_value - median_value) / mad
                };
                if modified_z.abs() >= self.policy.quarantine_modified_z {
                    issues.push(FleetAnomalyIssue::AircraftOutlier {
                        aircraft_id: target_aircraft_id.to_string(),
                        metric: metric.clone(),
                        modified_z,
                    });
                }
                let violating: Vec<_> = valid
                    .iter()
                    .filter(|entry| outside_bound(entry.metrics[metric], bound))
                    .collect();
                for entry in &violating {
                    issues.push(FleetAnomalyIssue::QualifiedBoundViolation {
                        aircraft_id: entry.aircraft_id.clone(),
                        metric: metric.clone(),
                        value: entry.metrics[metric],
                    });
                }
                let fraction = if valid.is_empty() {
                    0.0
                } else {
                    violating.len() as f64 / valid.len() as f64
                };
                if fraction >= self.policy.common_mode_fraction {
                    issues.push(FleetAnomalyIssue::CommonModeViolation {
                        metric: metric.clone(),
                        violating_fraction: fraction,
                    });
                }
                metric_reports.push(FleetMetricAssessment {
                    metric: metric.clone(),
                    cohort_median: median_value,
                    median_absolute_deviation: mad,
                    target_value,
                    target_modified_z: modified_z,
                    qualified_bound_violating_fraction: fraction,
                });
            }
        }

        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                FleetAnomalyIssue::DuplicateObservation(_)
                    | FleetAnomalyIssue::DuplicateAircraft(_)
                    | FleetAnomalyIssue::FleetMismatch(_)
                    | FleetAnomalyIssue::ConfigurationMismatch(_)
                    | FleetAnomalyIssue::InvalidMetric { .. }
                    | FleetAnomalyIssue::MissingMetric { .. }
                    | FleetAnomalyIssue::MissingEvidence(_)
                    | FleetAnomalyIssue::FutureObservation(_)
                    | FleetAnomalyIssue::StaleObservation { .. }
                    | FleetAnomalyIssue::InsufficientCohort { .. }
            )
        });
        let common_mode = issues
            .iter()
            .any(|issue| matches!(issue, FleetAnomalyIssue::CommonModeViolation { .. }));
        let target_outlier = issues.iter().any(|issue| match issue {
            FleetAnomalyIssue::AircraftOutlier { aircraft_id, .. }
            | FleetAnomalyIssue::QualifiedBoundViolation { aircraft_id, .. } => {
                aircraft_id == target_aircraft_id
            }
            _ => false,
        });
        let status = if incomplete {
            FleetAnomalyStatus::Incomplete
        } else if common_mode {
            FleetAnomalyStatus::FleetHold
        } else if target_outlier {
            FleetAnomalyStatus::QuarantineAircraft
        } else {
            FleetAnomalyStatus::Normal
        };

        FleetAnomalyReport {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            fleet_id: self.policy.fleet_id.clone(),
            target_aircraft_id: target_aircraft_id.to_string(),
            assessed_at_ms: now_ms,
            status,
            cohort_size: valid.len(),
            metrics: metric_reports,
            issues,
        }
    }
}

fn median(values: &[f64]) -> f64 {
    match values.len() {
        0 => 0.0,
        length if length % 2 == 1 => values[length / 2],
        length => (values[length / 2 - 1] + values[length / 2]) / 2.0,
    }
}

fn outside_bound(value: f64, bound: &FleetMetricBound) -> bool {
    value < bound.minimum || value > bound.maximum
}

fn valid_digest(digest: &str) -> bool {
    let digest = digest.trim();
    digest.starts_with("sha256:") && digest.len() > "sha256:".len()
        || digest.starts_with("fnv1a64:") && digest.len() == "fnv1a64:".len() + 16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector() -> FleetAnomalyDetector {
        FleetAnomalyDetector::new(FleetAnomalyPolicy {
            schema_version: "1".into(),
            policy_id: "fleet-anomaly".into(),
            fleet_id: "fleet-1".into(),
            qualified_configuration_digest: "sha256:config".into(),
            minimum_cohort_size: 4,
            quarantine_modified_z: 3.5,
            common_mode_fraction: 0.75,
            maximum_observation_age_ms: 1_000,
            metric_bounds: BTreeMap::from([(
                "vibration".into(),
                FleetMetricBound {
                    minimum: 0.0,
                    maximum: 5.0,
                    direction_higher_is_worse: true,
                },
            )]),
        })
        .unwrap()
    }

    fn observation(id: &str, value: f64) -> FleetAircraftObservation {
        FleetAircraftObservation {
            observation_id: format!("obs-{id}"),
            fleet_id: "fleet-1".into(),
            aircraft_id: id.into(),
            configuration_digest: "sha256:config".into(),
            timestamp_ms: 1_000,
            metrics: BTreeMap::from([("vibration".into(), value)]),
            evidence_ids: vec![format!("evidence-{id}")],
        }
    }

    #[test]
    fn isolated_outlier_is_quarantined() {
        let report = detector().assess(
            "d",
            &[
                observation("a", 1.0),
                observation("b", 1.1),
                observation("c", 0.9),
                observation("d", 4.0),
            ],
            1_000,
        );
        assert_eq!(report.status, FleetAnomalyStatus::QuarantineAircraft);
    }

    #[test]
    fn widespread_bound_violation_holds_fleet() {
        let report = detector().assess(
            "a",
            &[
                observation("a", 6.0),
                observation("b", 6.1),
                observation("c", 6.2),
                observation("d", 1.0),
            ],
            1_000,
        );
        assert_eq!(report.status, FleetAnomalyStatus::FleetHold);
    }

    #[test]
    fn healthy_cohort_is_normal() {
        let report = detector().assess(
            "a",
            &[
                observation("a", 1.0),
                observation("b", 1.1),
                observation("c", 0.9),
                observation("d", 1.05),
            ],
            1_000,
        );
        assert_eq!(report.status, FleetAnomalyStatus::Normal);
    }

    #[test]
    fn configuration_mismatch_is_incomplete() {
        let mut entries = vec![
            observation("a", 1.0),
            observation("b", 1.1),
            observation("c", 0.9),
            observation("d", 1.05),
        ];
        entries[0].configuration_digest = "sha256:other".into();
        let report = detector().assess("a", &entries, 1_000);
        assert_eq!(report.status, FleetAnomalyStatus::Incomplete);
    }
}
