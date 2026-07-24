// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Certification-oriented deterministic test oracles.
//!
//! A scenario is not considered successful merely because it completed.  This
//! module defines explicit signal and response-deadline oracles with
//! Pass/Fail/Incomplete semantics, deterministic evidence, and bounded missing
//! data.  The implementation is intentionally independent from the simulator so
//! the same oracle definitions can be applied to SIL, HIL, and flight-test data.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OracleTolerance {
    Absolute {
        maximum_error: f64,
    },
    Relative {
        maximum_fraction: f64,
        floor: f64,
    },
    Combined {
        absolute: f64,
        relative: f64,
        floor: f64,
    },
    Interval {
        minimum: f64,
        maximum: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignalOracle {
    pub oracle_id: String,
    pub signal_name: String,
    pub tolerance: OracleTolerance,
    pub minimum_samples: usize,
    pub maximum_missing_samples: usize,
    pub maximum_violations: usize,
    pub evaluation_start_s: f64,
    pub evaluation_end_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleSample {
    pub timestamp_s: f64,
    pub expected: f64,
    pub observed: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseDeadlineOracle {
    pub oracle_id: String,
    pub trigger_name: String,
    pub response_name: String,
    pub maximum_response_time_s: f64,
    pub response_required: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseEvent {
    pub event_name: String,
    pub timestamp_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OracleStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OracleIssue {
    MissingSamples { observed: usize, required: usize },
    ExcessiveMissingSamples { observed: usize, allowed: usize },
    ExcessiveViolations { observed: usize, allowed: usize },
    NonFiniteTimestamp,
    NonMonotonicTimestamp,
    NonFiniteExpected,
    NonFiniteObserved,
    TriggerNotObserved,
    ResponseNotObserved,
    ResponseTooLate { elapsed_s: f64, maximum_s: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleViolation {
    pub timestamp_s: f64,
    pub expected: f64,
    pub observed: f64,
    pub error: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleReport {
    pub oracle_id: String,
    pub status: OracleStatus,
    pub sample_count: usize,
    pub missing_count: usize,
    pub violation_count: usize,
    pub maximum_error: f64,
    pub issues: Vec<OracleIssue>,
    pub violations: Vec<OracleViolation>,
}

impl OracleReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, TestOracleError> {
        serde_json::to_vec(self).map_err(|_| TestOracleError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, TestOracleError> {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in self.canonical_json()? {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestOracleError {
    InvalidOracle,
    DuplicateOracleId(String),
    SerializationFailed,
}

impl SignalOracle {
    pub fn validate(&self) -> Result<(), TestOracleError> {
        if self.oracle_id.trim().is_empty()
            || self.signal_name.trim().is_empty()
            || self.minimum_samples == 0
            || !self.evaluation_start_s.is_finite()
            || !self.evaluation_end_s.is_finite()
            || self.evaluation_end_s < self.evaluation_start_s
            || !valid_tolerance(self.tolerance)
        {
            return Err(TestOracleError::InvalidOracle);
        }
        Ok(())
    }

    pub fn evaluate(&self, samples: &[OracleSample]) -> Result<OracleReport, TestOracleError> {
        self.validate()?;
        let mut issues = Vec::new();
        let mut violations = Vec::new();
        let mut sample_count = 0usize;
        let mut missing_count = 0usize;
        let mut maximum_error = 0.0f64;
        let mut previous_timestamp = None;

        for sample in samples {
            if !sample.timestamp_s.is_finite() {
                issues.push(OracleIssue::NonFiniteTimestamp);
                continue;
            }
            if let Some(previous) = previous_timestamp {
                if sample.timestamp_s < previous {
                    issues.push(OracleIssue::NonMonotonicTimestamp);
                }
            }
            previous_timestamp = Some(sample.timestamp_s);
            if sample.timestamp_s < self.evaluation_start_s
                || sample.timestamp_s > self.evaluation_end_s
            {
                continue;
            }
            sample_count += 1;
            if !sample.expected.is_finite() {
                issues.push(OracleIssue::NonFiniteExpected);
                continue;
            }
            let Some(observed) = sample.observed else {
                missing_count += 1;
                continue;
            };
            if !observed.is_finite() {
                issues.push(OracleIssue::NonFiniteObserved);
                continue;
            }
            let error = (observed - sample.expected).abs();
            maximum_error = maximum_error.max(error);
            if !within_tolerance(self.tolerance, sample.expected, observed) {
                violations.push(OracleViolation {
                    timestamp_s: sample.timestamp_s,
                    expected: sample.expected,
                    observed,
                    error,
                });
            }
        }

        if sample_count < self.minimum_samples {
            issues.push(OracleIssue::MissingSamples {
                observed: sample_count,
                required: self.minimum_samples,
            });
        }
        if missing_count > self.maximum_missing_samples {
            issues.push(OracleIssue::ExcessiveMissingSamples {
                observed: missing_count,
                allowed: self.maximum_missing_samples,
            });
        }
        if violations.len() > self.maximum_violations {
            issues.push(OracleIssue::ExcessiveViolations {
                observed: violations.len(),
                allowed: self.maximum_violations,
            });
        }

        let hard_failure = issues.iter().any(|issue| {
            matches!(
                issue,
                OracleIssue::ExcessiveViolations { .. }
                    | OracleIssue::NonFiniteTimestamp
                    | OracleIssue::NonMonotonicTimestamp
                    | OracleIssue::NonFiniteExpected
                    | OracleIssue::NonFiniteObserved
            )
        });
        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                OracleIssue::MissingSamples { .. } | OracleIssue::ExcessiveMissingSamples { .. }
            )
        });
        let status = if hard_failure {
            OracleStatus::Fail
        } else if incomplete {
            OracleStatus::Incomplete
        } else {
            OracleStatus::Pass
        };

        Ok(OracleReport {
            oracle_id: self.oracle_id.clone(),
            status,
            sample_count,
            missing_count,
            violation_count: violations.len(),
            maximum_error,
            issues,
            violations,
        })
    }
}

impl ResponseDeadlineOracle {
    pub fn evaluate(&self, events: &[ResponseEvent]) -> Result<OracleReport, TestOracleError> {
        if self.oracle_id.trim().is_empty()
            || self.trigger_name.trim().is_empty()
            || self.response_name.trim().is_empty()
            || !self.maximum_response_time_s.is_finite()
            || self.maximum_response_time_s < 0.0
        {
            return Err(TestOracleError::InvalidOracle);
        }
        let trigger = events
            .iter()
            .filter(|event| event.event_name == self.trigger_name && event.timestamp_s.is_finite())
            .map(|event| event.timestamp_s)
            .min_by(f64::total_cmp);
        let Some(trigger_time) = trigger else {
            return Ok(OracleReport {
                oracle_id: self.oracle_id.clone(),
                status: OracleStatus::Incomplete,
                sample_count: events.len(),
                missing_count: 1,
                violation_count: 0,
                maximum_error: 0.0,
                issues: vec![OracleIssue::TriggerNotObserved],
                violations: Vec::new(),
            });
        };
        let response = events
            .iter()
            .filter(|event| {
                event.event_name == self.response_name
                    && event.timestamp_s.is_finite()
                    && event.timestamp_s >= trigger_time
            })
            .map(|event| event.timestamp_s)
            .min_by(f64::total_cmp);
        let Some(response_time) = response else {
            let status = if self.response_required {
                OracleStatus::Fail
            } else {
                OracleStatus::Incomplete
            };
            return Ok(OracleReport {
                oracle_id: self.oracle_id.clone(),
                status,
                sample_count: events.len(),
                missing_count: 1,
                violation_count: if self.response_required { 1 } else { 0 },
                maximum_error: 0.0,
                issues: vec![OracleIssue::ResponseNotObserved],
                violations: Vec::new(),
            });
        };
        let elapsed = response_time - trigger_time;
        let late = elapsed > self.maximum_response_time_s;
        Ok(OracleReport {
            oracle_id: self.oracle_id.clone(),
            status: if late {
                OracleStatus::Fail
            } else {
                OracleStatus::Pass
            },
            sample_count: events.len(),
            missing_count: 0,
            violation_count: if late { 1 } else { 0 },
            maximum_error: (elapsed - self.maximum_response_time_s).max(0.0),
            issues: if late {
                vec![OracleIssue::ResponseTooLate {
                    elapsed_s: elapsed,
                    maximum_s: self.maximum_response_time_s,
                }]
            } else {
                Vec::new()
            },
            violations: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TestOracleSuite {
    pub schema_version: String,
    pub suite_id: String,
    pub signal_oracles: Vec<SignalOracle>,
    pub deadline_oracles: Vec<ResponseDeadlineOracle>,
}

impl TestOracleSuite {
    pub fn validate(&self) -> Result<(), TestOracleError> {
        if self.schema_version.trim().is_empty() || self.suite_id.trim().is_empty() {
            return Err(TestOracleError::InvalidOracle);
        }
        let mut ids = BTreeSet::new();
        for oracle in &self.signal_oracles {
            oracle.validate()?;
            if !ids.insert(oracle.oracle_id.clone()) {
                return Err(TestOracleError::DuplicateOracleId(oracle.oracle_id.clone()));
            }
        }
        for oracle in &self.deadline_oracles {
            if oracle.oracle_id.trim().is_empty() || !ids.insert(oracle.oracle_id.clone()) {
                return Err(TestOracleError::DuplicateOracleId(oracle.oracle_id.clone()));
            }
        }
        Ok(())
    }
}

fn valid_tolerance(tolerance: OracleTolerance) -> bool {
    match tolerance {
        OracleTolerance::Absolute { maximum_error } => {
            maximum_error.is_finite() && maximum_error >= 0.0
        }
        OracleTolerance::Relative {
            maximum_fraction,
            floor,
        } => {
            maximum_fraction.is_finite()
                && maximum_fraction >= 0.0
                && floor.is_finite()
                && floor > 0.0
        }
        OracleTolerance::Combined {
            absolute,
            relative,
            floor,
        } => {
            absolute.is_finite()
                && absolute >= 0.0
                && relative.is_finite()
                && relative >= 0.0
                && floor.is_finite()
                && floor > 0.0
        }
        OracleTolerance::Interval { minimum, maximum } => {
            minimum.is_finite() && maximum.is_finite() && minimum <= maximum
        }
    }
}

fn within_tolerance(tolerance: OracleTolerance, expected: f64, observed: f64) -> bool {
    match tolerance {
        OracleTolerance::Absolute { maximum_error } => (observed - expected).abs() <= maximum_error,
        OracleTolerance::Relative {
            maximum_fraction,
            floor,
        } => (observed - expected).abs() <= maximum_fraction * expected.abs().max(floor),
        OracleTolerance::Combined {
            absolute,
            relative,
            floor,
        } => (observed - expected).abs() <= absolute + relative * expected.abs().max(floor),
        OracleTolerance::Interval { minimum, maximum } => {
            observed >= minimum && observed <= maximum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle() -> SignalOracle {
        SignalOracle {
            oracle_id: "hover-altitude".into(),
            signal_name: "altitude_m".into(),
            tolerance: OracleTolerance::Absolute { maximum_error: 0.2 },
            minimum_samples: 2,
            maximum_missing_samples: 0,
            maximum_violations: 0,
            evaluation_start_s: 0.0,
            evaluation_end_s: 10.0,
        }
    }

    #[test]
    fn signal_oracle_passes_bounded_trace() {
        let report = oracle()
            .evaluate(&[
                OracleSample {
                    timestamp_s: 0.0,
                    expected: 20.0,
                    observed: Some(20.1),
                },
                OracleSample {
                    timestamp_s: 1.0,
                    expected: 20.0,
                    observed: Some(19.9),
                },
            ])
            .unwrap();
        assert_eq!(report.status, OracleStatus::Pass);
    }

    #[test]
    fn signal_oracle_fails_real_violation() {
        let report = oracle()
            .evaluate(&[
                OracleSample {
                    timestamp_s: 0.0,
                    expected: 20.0,
                    observed: Some(20.0),
                },
                OracleSample {
                    timestamp_s: 1.0,
                    expected: 20.0,
                    observed: Some(19.0),
                },
            ])
            .unwrap();
        assert_eq!(report.status, OracleStatus::Fail);
        assert_eq!(report.violation_count, 1);
    }

    #[test]
    fn deadline_oracle_detects_late_response() {
        let oracle = ResponseDeadlineOracle {
            oracle_id: "fault-transfer".into(),
            trigger_name: "critical_fault".into(),
            response_name: "baseline_active".into(),
            maximum_response_time_s: 0.1,
            response_required: true,
        };
        let report = oracle
            .evaluate(&[
                ResponseEvent {
                    event_name: "critical_fault".into(),
                    timestamp_s: 1.0,
                },
                ResponseEvent {
                    event_name: "baseline_active".into(),
                    timestamp_s: 1.2,
                },
            ])
            .unwrap();
        assert_eq!(report.status, OracleStatus::Fail);
    }

    #[test]
    fn suite_rejects_duplicate_ids() {
        let duplicate = oracle();
        let suite = TestOracleSuite {
            schema_version: "1".into(),
            suite_id: "qualification".into(),
            signal_oracles: vec![duplicate.clone(), duplicate],
            deadline_oracles: Vec::new(),
        };
        assert!(matches!(
            suite.validate(),
            Err(TestOracleError::DuplicateOracleId(_))
        ));
    }
}
