// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bounded maintenance trend monitoring.
//!
//! This module detects sustained degradation and abrupt level shifts. It does
//! not extrapolate remaining useful life or replace approved inspection limits.
//! Sparse, stale, noisy, or unauthenticated data produces `Incomplete`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaintenanceTrendPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub metric_name: String,
    pub minimum_samples: usize,
    pub minimum_span_ms: u64,
    pub maximum_sample_gap_ms: u64,
    pub maximum_sample_age_ms: u64,
    pub watch_slope_per_hour: f64,
    pub inspection_slope_per_hour: f64,
    pub grounding_slope_per_hour: f64,
    pub maximum_residual_stddev: f64,
    pub level_shift_threshold: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaintenanceTrendObservation {
    pub observation_id: String,
    pub component_serial_number: String,
    pub timestamp_ms: u64,
    pub value: f64,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaintenanceTrendDisposition {
    Stable,
    Watch,
    InspectionDue,
    Grounded,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MaintenanceTrendIssue {
    DuplicateObservation(String),
    SerialMismatch {
        expected: String,
        observed: String,
    },
    InvalidObservation(String),
    MissingEvidence(String),
    FutureObservation(String),
    StaleObservation {
        observation_id: String,
        age_ms: u64,
        maximum_ms: u64,
    },
    InsufficientSamples {
        observed: usize,
        required: usize,
    },
    InsufficientSpan {
        observed_ms: u64,
        required_ms: u64,
    },
    ExcessiveSampleGap {
        gap_ms: u64,
        maximum_ms: u64,
    },
    ExcessiveResidualNoise {
        observed: f64,
        maximum: f64,
    },
    WatchSlope {
        slope_per_hour: f64,
    },
    InspectionSlope {
        slope_per_hour: f64,
    },
    GroundingSlope {
        slope_per_hour: f64,
    },
    AbruptLevelShift {
        magnitude: f64,
        threshold: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaintenanceTrendReport {
    pub schema_version: String,
    pub policy_id: String,
    pub component_serial_number: String,
    pub metric_name: String,
    pub assessed_at_ms: u64,
    pub disposition: MaintenanceTrendDisposition,
    pub sample_count: usize,
    pub span_ms: u64,
    pub slope_per_hour: Option<f64>,
    pub intercept: Option<f64>,
    pub residual_stddev: Option<f64>,
    pub level_shift: Option<f64>,
    pub issues: Vec<MaintenanceTrendIssue>,
    pub remaining_useful_life: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaintenanceTrendError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct MaintenanceTrendMonitor {
    policy: MaintenanceTrendPolicy,
}

impl MaintenanceTrendMonitor {
    pub fn new(policy: MaintenanceTrendPolicy) -> Result<Self, MaintenanceTrendError> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.metric_name.trim().is_empty()
            || policy.minimum_samples < 3
            || policy.minimum_span_ms == 0
            || policy.maximum_sample_gap_ms == 0
            || policy.maximum_sample_age_ms == 0
            || !policy.watch_slope_per_hour.is_finite()
            || !policy.inspection_slope_per_hour.is_finite()
            || !policy.grounding_slope_per_hour.is_finite()
            || policy.watch_slope_per_hour < 0.0
            || policy.inspection_slope_per_hour <= policy.watch_slope_per_hour
            || policy.grounding_slope_per_hour <= policy.inspection_slope_per_hour
            || !policy.maximum_residual_stddev.is_finite()
            || policy.maximum_residual_stddev <= 0.0
            || !policy.level_shift_threshold.is_finite()
            || policy.level_shift_threshold <= 0.0
        {
            return Err(MaintenanceTrendError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        component_serial_number: &str,
        observations: &[MaintenanceTrendObservation],
        now_ms: u64,
    ) -> MaintenanceTrendReport {
        let mut issues = Vec::new();
        let mut ids = BTreeSet::new();
        let mut valid = Vec::new();
        for observation in observations {
            if observation.observation_id.trim().is_empty()
                || !ids.insert(observation.observation_id.clone())
            {
                issues.push(MaintenanceTrendIssue::DuplicateObservation(
                    observation.observation_id.clone(),
                ));
            }
            if observation.component_serial_number != component_serial_number {
                issues.push(MaintenanceTrendIssue::SerialMismatch {
                    expected: component_serial_number.to_string(),
                    observed: observation.component_serial_number.clone(),
                });
                continue;
            }
            if !observation.value.is_finite() {
                issues.push(MaintenanceTrendIssue::InvalidObservation(
                    observation.observation_id.clone(),
                ));
                continue;
            }
            if observation.evidence_ids.is_empty()
                || observation
                    .evidence_ids
                    .iter()
                    .any(|id| id.trim().is_empty())
            {
                issues.push(MaintenanceTrendIssue::MissingEvidence(
                    observation.observation_id.clone(),
                ));
            }
            if observation.timestamp_ms > now_ms {
                issues.push(MaintenanceTrendIssue::FutureObservation(
                    observation.observation_id.clone(),
                ));
                continue;
            }
            let age = now_ms.saturating_sub(observation.timestamp_ms);
            if age > self.policy.maximum_sample_age_ms {
                issues.push(MaintenanceTrendIssue::StaleObservation {
                    observation_id: observation.observation_id.clone(),
                    age_ms: age,
                    maximum_ms: self.policy.maximum_sample_age_ms,
                });
            }
            valid.push(observation);
        }
        valid.sort_by(|left, right| {
            left.timestamp_ms
                .cmp(&right.timestamp_ms)
                .then_with(|| left.observation_id.cmp(&right.observation_id))
        });

        if valid.len() < self.policy.minimum_samples {
            issues.push(MaintenanceTrendIssue::InsufficientSamples {
                observed: valid.len(),
                required: self.policy.minimum_samples,
            });
        }
        let span_ms = valid.first().zip(valid.last()).map_or(0, |(first, last)| {
            last.timestamp_ms.saturating_sub(first.timestamp_ms)
        });
        if span_ms < self.policy.minimum_span_ms {
            issues.push(MaintenanceTrendIssue::InsufficientSpan {
                observed_ms: span_ms,
                required_ms: self.policy.minimum_span_ms,
            });
        }
        for pair in valid.windows(2) {
            let gap = pair[1].timestamp_ms.saturating_sub(pair[0].timestamp_ms);
            if gap > self.policy.maximum_sample_gap_ms {
                issues.push(MaintenanceTrendIssue::ExcessiveSampleGap {
                    gap_ms: gap,
                    maximum_ms: self.policy.maximum_sample_gap_ms,
                });
            }
        }

        let regression = if valid.len() >= 2 && span_ms > 0 {
            linear_regression(&valid)
        } else {
            None
        };
        let level_shift = if valid.len() >= 4 {
            let midpoint = valid.len() / 2;
            let first_mean = valid[..midpoint]
                .iter()
                .map(|sample| sample.value)
                .sum::<f64>()
                / midpoint as f64;
            let second_mean = valid[midpoint..]
                .iter()
                .map(|sample| sample.value)
                .sum::<f64>()
                / (valid.len() - midpoint) as f64;
            Some(second_mean - first_mean)
        } else {
            None
        };

        if let Some((slope, _, residual_stddev)) = regression {
            let abs_slope = slope.abs();
            if residual_stddev > self.policy.maximum_residual_stddev {
                issues.push(MaintenanceTrendIssue::ExcessiveResidualNoise {
                    observed: residual_stddev,
                    maximum: self.policy.maximum_residual_stddev,
                });
            }
            if abs_slope >= self.policy.grounding_slope_per_hour {
                issues.push(MaintenanceTrendIssue::GroundingSlope {
                    slope_per_hour: slope,
                });
            } else if abs_slope >= self.policy.inspection_slope_per_hour {
                issues.push(MaintenanceTrendIssue::InspectionSlope {
                    slope_per_hour: slope,
                });
            } else if abs_slope >= self.policy.watch_slope_per_hour {
                issues.push(MaintenanceTrendIssue::WatchSlope {
                    slope_per_hour: slope,
                });
            }
        }
        if let Some(shift) = level_shift {
            if shift.abs() >= self.policy.level_shift_threshold {
                issues.push(MaintenanceTrendIssue::AbruptLevelShift {
                    magnitude: shift,
                    threshold: self.policy.level_shift_threshold,
                });
            }
        }

        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                MaintenanceTrendIssue::DuplicateObservation(_)
                    | MaintenanceTrendIssue::SerialMismatch { .. }
                    | MaintenanceTrendIssue::InvalidObservation(_)
                    | MaintenanceTrendIssue::MissingEvidence(_)
                    | MaintenanceTrendIssue::FutureObservation(_)
                    | MaintenanceTrendIssue::StaleObservation { .. }
                    | MaintenanceTrendIssue::InsufficientSamples { .. }
                    | MaintenanceTrendIssue::InsufficientSpan { .. }
                    | MaintenanceTrendIssue::ExcessiveSampleGap { .. }
                    | MaintenanceTrendIssue::ExcessiveResidualNoise { .. }
            )
        });
        let grounding = issues
            .iter()
            .any(|issue| matches!(issue, MaintenanceTrendIssue::GroundingSlope { .. }));
        let inspection = issues.iter().any(|issue| {
            matches!(
                issue,
                MaintenanceTrendIssue::InspectionSlope { .. }
                    | MaintenanceTrendIssue::AbruptLevelShift { .. }
            )
        });
        let watch = issues
            .iter()
            .any(|issue| matches!(issue, MaintenanceTrendIssue::WatchSlope { .. }));
        let disposition = if incomplete {
            MaintenanceTrendDisposition::Incomplete
        } else if grounding {
            MaintenanceTrendDisposition::Grounded
        } else if inspection {
            MaintenanceTrendDisposition::InspectionDue
        } else if watch {
            MaintenanceTrendDisposition::Watch
        } else {
            MaintenanceTrendDisposition::Stable
        };

        MaintenanceTrendReport {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            component_serial_number: component_serial_number.to_string(),
            metric_name: self.policy.metric_name.clone(),
            assessed_at_ms: now_ms,
            disposition,
            sample_count: valid.len(),
            span_ms,
            slope_per_hour: regression.map(|value| value.0),
            intercept: regression.map(|value| value.1),
            residual_stddev: regression.map(|value| value.2),
            level_shift,
            remaining_useful_life: None,
            issues,
        }
    }
}

fn linear_regression(observations: &[&MaintenanceTrendObservation]) -> Option<(f64, f64, f64)> {
    let origin = observations.first()?.timestamp_ms;
    let points: Vec<_> = observations
        .iter()
        .map(|observation| {
            (
                observation.timestamp_ms.saturating_sub(origin) as f64 / 3_600_000.0,
                observation.value,
            )
        })
        .collect();
    let mean_x = points.iter().map(|point| point.0).sum::<f64>() / points.len() as f64;
    let mean_y = points.iter().map(|point| point.1).sum::<f64>() / points.len() as f64;
    let denominator = points
        .iter()
        .map(|point| (point.0 - mean_x).powi(2))
        .sum::<f64>();
    if denominator <= f64::EPSILON {
        return None;
    }
    let slope = points
        .iter()
        .map(|point| (point.0 - mean_x) * (point.1 - mean_y))
        .sum::<f64>()
        / denominator;
    let intercept = mean_y - slope * mean_x;
    let residual_sum = points
        .iter()
        .map(|point| {
            let residual = point.1 - (intercept + slope * point.0);
            residual * residual
        })
        .sum::<f64>();
    let residual_stddev = (residual_sum / points.len() as f64).sqrt();
    Some((slope, intercept, residual_stddev))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monitor() -> MaintenanceTrendMonitor {
        MaintenanceTrendMonitor::new(MaintenanceTrendPolicy {
            schema_version: "1".into(),
            policy_id: "trend".into(),
            metric_name: "gearbox-vibration".into(),
            minimum_samples: 4,
            minimum_span_ms: 3_600_000,
            maximum_sample_gap_ms: 4_000_000,
            maximum_sample_age_ms: 20_000_000,
            watch_slope_per_hour: 0.5,
            inspection_slope_per_hour: 1.0,
            grounding_slope_per_hour: 2.0,
            maximum_residual_stddev: 0.25,
            level_shift_threshold: 3.0,
        })
        .unwrap()
    }

    fn observation(id: &str, hour: u64, value: f64) -> MaintenanceTrendObservation {
        MaintenanceTrendObservation {
            observation_id: id.into(),
            component_serial_number: "gearbox-1".into(),
            timestamp_ms: hour * 3_600_000,
            value,
            evidence_ids: vec![format!("evidence-{id}")],
        }
    }

    #[test]
    fn stable_series_is_serviceable() {
        let report = monitor().assess(
            "gearbox-1",
            &[
                observation("a", 0, 1.0),
                observation("b", 1, 1.05),
                observation("c", 2, 1.1),
                observation("d", 3, 1.15),
            ],
            3 * 3_600_000,
        );
        assert_eq!(report.disposition, MaintenanceTrendDisposition::Stable);
        assert_eq!(report.remaining_useful_life, None);
    }

    #[test]
    fn sustained_degradation_requires_inspection() {
        let report = monitor().assess(
            "gearbox-1",
            &[
                observation("a", 0, 1.0),
                observation("b", 1, 2.2),
                observation("c", 2, 3.4),
                observation("d", 3, 4.6),
            ],
            3 * 3_600_000,
        );
        assert_eq!(
            report.disposition,
            MaintenanceTrendDisposition::InspectionDue
        );
    }

    #[test]
    fn severe_slope_grounds_component() {
        let report = monitor().assess(
            "gearbox-1",
            &[
                observation("a", 0, 1.0),
                observation("b", 1, 3.2),
                observation("c", 2, 5.4),
                observation("d", 3, 7.6),
            ],
            3 * 3_600_000,
        );
        assert_eq!(report.disposition, MaintenanceTrendDisposition::Grounded);
    }

    #[test]
    fn sparse_series_is_incomplete() {
        let report = monitor().assess("gearbox-1", &[observation("a", 0, 1.0)], 1_000);
        assert_eq!(report.disposition, MaintenanceTrendDisposition::Incomplete);
    }
}
