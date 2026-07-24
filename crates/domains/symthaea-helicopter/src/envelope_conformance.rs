// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Time-series flight-envelope conformance audit.
//!
//! Envelope protectors issue commands, but release evidence must show that the
//! aircraft actually remained inside the dynamic limits or returned within an
//! allowed response deadline. This module audits observed traces, cumulative
//! excursion dwell, peak exceedance, and recovery timing with deterministic
//! Pass/Fail/Incomplete semantics.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EnvelopeQuantity {
    Airspeed,
    BankAngle,
    PitchAngle,
    ClimbRate,
    DescentRate,
    YawRate,
    RotorRpm,
    LoadFactor,
    Vibration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DynamicEnvelopeLimit {
    pub quantity: EnvelopeQuantity,
    pub minimum: f64,
    pub maximum: f64,
    pub maximum_excursion_duration_s: f64,
    pub maximum_total_excursion_s: f64,
    pub maximum_peak_exceedance: f64,
    pub recovery_deadline_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeTraceSample {
    pub timestamp_s: f64,
    pub values: BTreeMap<EnvelopeQuantity, Option<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvelopeConformanceStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnvelopeConformanceIssue {
    MissingQuantity(EnvelopeQuantity),
    MissingSample(EnvelopeQuantity),
    NonFiniteTimestamp,
    NonMonotonicTimestamp,
    NonFiniteValue(EnvelopeQuantity),
    PeakExceedance {
        quantity: EnvelopeQuantity,
        observed: f64,
        allowed: f64,
    },
    ExcursionDurationExceeded {
        quantity: EnvelopeQuantity,
        observed_s: f64,
        allowed_s: f64,
    },
    TotalExcursionExceeded {
        quantity: EnvelopeQuantity,
        observed_s: f64,
        allowed_s: f64,
    },
    RecoveryDeadlineMissed {
        quantity: EnvelopeQuantity,
        observed_s: f64,
        allowed_s: f64,
    },
    UnrecoveredExcursion(EnvelopeQuantity),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantityConformanceEvidence {
    pub quantity: EnvelopeQuantity,
    pub status: EnvelopeConformanceStatus,
    pub sample_count: usize,
    pub missing_count: usize,
    pub excursion_count: usize,
    pub total_excursion_s: f64,
    pub longest_excursion_s: f64,
    pub peak_exceedance: f64,
    pub maximum_recovery_s: f64,
    pub issues: Vec<EnvelopeConformanceIssue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeConformanceReport {
    pub schema_version: String,
    pub audit_id: String,
    pub status: EnvelopeConformanceStatus,
    pub start_timestamp_s: Option<f64>,
    pub end_timestamp_s: Option<f64>,
    pub quantities: Vec<QuantityConformanceEvidence>,
}

impl EnvelopeConformanceReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, EnvelopeConformanceError> {
        let mut canonical = self.clone();
        canonical
            .quantities
            .sort_by_key(|evidence| evidence.quantity);
        serde_json::to_vec(&canonical).map_err(|_| EnvelopeConformanceError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, EnvelopeConformanceError> {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in self.canonical_json()? {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeConformanceError {
    InvalidConfiguration,
    DuplicateQuantity(EnvelopeQuantity),
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct EnvelopeConformanceAuditor {
    schema_version: String,
    audit_id: String,
    limits: BTreeMap<EnvelopeQuantity, DynamicEnvelopeLimit>,
}

impl EnvelopeConformanceAuditor {
    pub fn new(
        schema_version: String,
        audit_id: String,
        limits: Vec<DynamicEnvelopeLimit>,
    ) -> Result<Self, EnvelopeConformanceError> {
        if schema_version.trim().is_empty() || audit_id.trim().is_empty() || limits.is_empty() {
            return Err(EnvelopeConformanceError::InvalidConfiguration);
        }
        let mut by_quantity = BTreeMap::new();
        for limit in limits {
            if !valid_limit(&limit) {
                return Err(EnvelopeConformanceError::InvalidConfiguration);
            }
            let quantity = limit.quantity;
            if by_quantity.insert(quantity, limit).is_some() {
                return Err(EnvelopeConformanceError::DuplicateQuantity(quantity));
            }
        }
        Ok(Self {
            schema_version,
            audit_id,
            limits: by_quantity,
        })
    }

    pub fn evaluate(
        &self,
        samples: &[EnvelopeTraceSample],
    ) -> Result<EnvelopeConformanceReport, EnvelopeConformanceError> {
        let mut global_issues = Vec::new();
        let mut previous_timestamp = None;
        for sample in samples {
            if !sample.timestamp_s.is_finite() {
                global_issues.push(EnvelopeConformanceIssue::NonFiniteTimestamp);
            }
            if previous_timestamp.is_some_and(|previous| sample.timestamp_s < previous) {
                global_issues.push(EnvelopeConformanceIssue::NonMonotonicTimestamp);
            }
            previous_timestamp = Some(sample.timestamp_s);
        }

        let mut quantities = Vec::new();
        for limit in self.limits.values() {
            quantities.push(evaluate_quantity(limit, samples, &global_issues));
        }
        let status = if quantities
            .iter()
            .any(|evidence| evidence.status == EnvelopeConformanceStatus::Fail)
        {
            EnvelopeConformanceStatus::Fail
        } else if quantities
            .iter()
            .any(|evidence| evidence.status == EnvelopeConformanceStatus::Incomplete)
        {
            EnvelopeConformanceStatus::Incomplete
        } else {
            EnvelopeConformanceStatus::Pass
        };
        Ok(EnvelopeConformanceReport {
            schema_version: self.schema_version.clone(),
            audit_id: self.audit_id.clone(),
            status,
            start_timestamp_s: samples.first().map(|sample| sample.timestamp_s),
            end_timestamp_s: samples.last().map(|sample| sample.timestamp_s),
            quantities,
        })
    }
}

fn evaluate_quantity(
    limit: &DynamicEnvelopeLimit,
    samples: &[EnvelopeTraceSample],
    global_issues: &[EnvelopeConformanceIssue],
) -> QuantityConformanceEvidence {
    let mut issues = global_issues.to_vec();
    let mut sample_count = 0usize;
    let mut missing_count = 0usize;
    let mut excursion_count = 0usize;
    let mut total_excursion_s = 0.0f64;
    let mut longest_excursion_s = 0.0f64;
    let mut peak_exceedance = 0.0f64;
    let mut maximum_recovery_s = 0.0f64;
    let mut excursion_started: Option<f64> = None;
    let mut previous_time: Option<f64> = None;
    let mut observed_quantity = false;

    for sample in samples {
        let Some(value) = sample.values.get(&limit.quantity) else {
            missing_count += 1;
            continue;
        };
        observed_quantity = true;
        sample_count += 1;
        let Some(value) = *value else {
            missing_count += 1;
            issues.push(EnvelopeConformanceIssue::MissingSample(limit.quantity));
            continue;
        };
        if !value.is_finite() || !sample.timestamp_s.is_finite() {
            issues.push(EnvelopeConformanceIssue::NonFiniteValue(limit.quantity));
            continue;
        }
        let exceedance = if value < limit.minimum {
            limit.minimum - value
        } else if value > limit.maximum {
            value - limit.maximum
        } else {
            0.0
        };
        peak_exceedance = peak_exceedance.max(exceedance);
        if exceedance > 0.0 {
            if excursion_started.is_none() {
                excursion_started = Some(sample.timestamp_s);
                excursion_count += 1;
            }
            if let Some(previous_time) = previous_time {
                total_excursion_s += (sample.timestamp_s - previous_time).max(0.0);
            }
        } else if let Some(started) = excursion_started.take() {
            let duration = (sample.timestamp_s - started).max(0.0);
            longest_excursion_s = longest_excursion_s.max(duration);
            maximum_recovery_s = maximum_recovery_s.max(duration);
        }
        previous_time = Some(sample.timestamp_s);
    }

    if !observed_quantity {
        issues.push(EnvelopeConformanceIssue::MissingQuantity(limit.quantity));
    }
    if let Some(started) = excursion_started {
        let end = samples
            .last()
            .map(|sample| sample.timestamp_s)
            .unwrap_or(started);
        let duration = (end - started).max(0.0);
        longest_excursion_s = longest_excursion_s.max(duration);
        maximum_recovery_s = maximum_recovery_s.max(duration);
        issues.push(EnvelopeConformanceIssue::UnrecoveredExcursion(
            limit.quantity,
        ));
    }
    if peak_exceedance > limit.maximum_peak_exceedance {
        issues.push(EnvelopeConformanceIssue::PeakExceedance {
            quantity: limit.quantity,
            observed: peak_exceedance,
            allowed: limit.maximum_peak_exceedance,
        });
    }
    if longest_excursion_s > limit.maximum_excursion_duration_s {
        issues.push(EnvelopeConformanceIssue::ExcursionDurationExceeded {
            quantity: limit.quantity,
            observed_s: longest_excursion_s,
            allowed_s: limit.maximum_excursion_duration_s,
        });
    }
    if total_excursion_s > limit.maximum_total_excursion_s {
        issues.push(EnvelopeConformanceIssue::TotalExcursionExceeded {
            quantity: limit.quantity,
            observed_s: total_excursion_s,
            allowed_s: limit.maximum_total_excursion_s,
        });
    }
    if maximum_recovery_s > limit.recovery_deadline_s {
        issues.push(EnvelopeConformanceIssue::RecoveryDeadlineMissed {
            quantity: limit.quantity,
            observed_s: maximum_recovery_s,
            allowed_s: limit.recovery_deadline_s,
        });
    }

    let incomplete = issues.iter().any(|issue| {
        matches!(
            issue,
            EnvelopeConformanceIssue::MissingQuantity(_)
                | EnvelopeConformanceIssue::MissingSample(_)
        )
    });
    let failed = issues.iter().any(|issue| {
        matches!(
            issue,
            EnvelopeConformanceIssue::NonFiniteTimestamp
                | EnvelopeConformanceIssue::NonMonotonicTimestamp
                | EnvelopeConformanceIssue::NonFiniteValue(_)
                | EnvelopeConformanceIssue::PeakExceedance { .. }
                | EnvelopeConformanceIssue::ExcursionDurationExceeded { .. }
                | EnvelopeConformanceIssue::TotalExcursionExceeded { .. }
                | EnvelopeConformanceIssue::RecoveryDeadlineMissed { .. }
                | EnvelopeConformanceIssue::UnrecoveredExcursion(_)
        )
    });
    let status = if failed {
        EnvelopeConformanceStatus::Fail
    } else if incomplete {
        EnvelopeConformanceStatus::Incomplete
    } else {
        EnvelopeConformanceStatus::Pass
    };
    QuantityConformanceEvidence {
        quantity: limit.quantity,
        status,
        sample_count,
        missing_count,
        excursion_count,
        total_excursion_s,
        longest_excursion_s,
        peak_exceedance,
        maximum_recovery_s,
        issues,
    }
}

fn valid_limit(limit: &DynamicEnvelopeLimit) -> bool {
    limit.minimum.is_finite()
        && limit.maximum.is_finite()
        && limit.minimum <= limit.maximum
        && limit.maximum_excursion_duration_s.is_finite()
        && limit.maximum_excursion_duration_s >= 0.0
        && limit.maximum_total_excursion_s.is_finite()
        && limit.maximum_total_excursion_s >= 0.0
        && limit.maximum_peak_exceedance.is_finite()
        && limit.maximum_peak_exceedance >= 0.0
        && limit.recovery_deadline_s.is_finite()
        && limit.recovery_deadline_s >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auditor() -> EnvelopeConformanceAuditor {
        EnvelopeConformanceAuditor::new(
            "1".into(),
            "flight-1".into(),
            vec![DynamicEnvelopeLimit {
                quantity: EnvelopeQuantity::BankAngle,
                minimum: -30.0,
                maximum: 30.0,
                maximum_excursion_duration_s: 1.0,
                maximum_total_excursion_s: 1.0,
                maximum_peak_exceedance: 5.0,
                recovery_deadline_s: 1.0,
            }],
        )
        .unwrap()
    }

    fn sample(timestamp_s: f64, bank: f64) -> EnvelopeTraceSample {
        EnvelopeTraceSample {
            timestamp_s,
            values: BTreeMap::from([(EnvelopeQuantity::BankAngle, Some(bank))]),
        }
    }

    #[test]
    fn bounded_trace_passes() {
        let report = auditor()
            .evaluate(&[sample(0.0, 0.0), sample(1.0, 20.0)])
            .unwrap();
        assert_eq!(report.status, EnvelopeConformanceStatus::Pass);
    }

    #[test]
    fn excessive_peak_fails() {
        let report = auditor()
            .evaluate(&[sample(0.0, 0.0), sample(0.1, 40.0), sample(0.2, 0.0)])
            .unwrap();
        assert_eq!(report.status, EnvelopeConformanceStatus::Fail);
    }

    #[test]
    fn absent_quantity_is_incomplete() {
        let report = auditor()
            .evaluate(&[EnvelopeTraceSample {
                timestamp_s: 0.0,
                values: BTreeMap::new(),
            }])
            .unwrap();
        assert_eq!(report.status, EnvelopeConformanceStatus::Incomplete);
    }
}
