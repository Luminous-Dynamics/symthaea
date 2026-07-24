// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime divergence monitoring between the qualified model and observed aircraft.
//!
//! A digital twin is evidence only when its residuals remain bounded. This
//! module compares predicted and observed signals using declared uncertainty,
//! persistence, freshness, and evidence requirements. It does not silently
//! retune the model or treat a finite prediction as a valid one.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TwinSignal {
    Position,
    Velocity,
    Attitude,
    AngularRate,
    MainRotorSpeed,
    TailRotorSpeed,
    FuelMass,
    ShaftPower,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwinSignalPolicy {
    pub warning_sigma: f64,
    pub unsafe_sigma: f64,
    pub warning_persistence_samples: usize,
    pub unsafe_persistence_samples: usize,
    pub maximum_sample_age_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DigitalTwinDivergencePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub required_signals: Vec<TwinSignal>,
    pub signal_policies: BTreeMap<TwinSignal, TwinSignalPolicy>,
    pub minimum_samples_per_signal: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwinResidualSample {
    pub sample_id: String,
    pub timestamp_ms: u64,
    pub signal: TwinSignal,
    pub predicted: f64,
    pub observed: f64,
    pub combined_sigma: f64,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DigitalTwinDivergenceStatus {
    Aligned,
    Restricted,
    Unsafe,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DigitalTwinDivergenceIssue {
    MissingRequiredSignal(TwinSignal),
    InsufficientSamples {
        signal: TwinSignal,
        observed: usize,
        required: usize,
    },
    DuplicateSampleId(String),
    InvalidSample(String),
    MissingEvidence(String),
    FutureSample(String),
    StaleSample {
        sample_id: String,
        age_ms: u64,
        maximum_ms: u64,
    },
    WarningPersistence {
        signal: TwinSignal,
        consecutive_samples: usize,
    },
    UnsafePersistence {
        signal: TwinSignal,
        consecutive_samples: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwinSignalDivergence {
    pub signal: TwinSignal,
    pub sample_count: usize,
    pub rms_normalized_residual: f64,
    pub peak_normalized_residual: f64,
    pub final_warning_streak: usize,
    pub final_unsafe_streak: usize,
    pub maximum_warning_streak: usize,
    pub maximum_unsafe_streak: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DigitalTwinDivergenceReport {
    pub schema_version: String,
    pub policy_id: String,
    pub assessed_at_ms: u64,
    pub status: DigitalTwinDivergenceStatus,
    pub signals: Vec<TwinSignalDivergence>,
    pub issues: Vec<DigitalTwinDivergenceIssue>,
}

impl DigitalTwinDivergenceReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, DigitalTwinDivergenceError> {
        let mut canonical = self.clone();
        canonical.signals.sort_by_key(|signal| signal.signal);
        canonical.issues.sort_by_key(issue_sort_key);
        serde_json::to_vec(&canonical).map_err(|_| DigitalTwinDivergenceError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, DigitalTwinDivergenceError> {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in self.canonical_json()? {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DigitalTwinDivergenceError {
    InvalidPolicy,
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct DigitalTwinDivergenceMonitor {
    policy: DigitalTwinDivergencePolicy,
}

impl DigitalTwinDivergenceMonitor {
    pub fn new(policy: DigitalTwinDivergencePolicy) -> Result<Self, DigitalTwinDivergenceError> {
        let required: BTreeSet<_> = policy.required_signals.iter().copied().collect();
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.required_signals.is_empty()
            || required.len() != policy.required_signals.len()
            || policy.minimum_samples_per_signal == 0
            || required.iter().any(|signal| {
                policy
                    .signal_policies
                    .get(signal)
                    .is_none_or(|signal_policy| !valid_signal_policy(signal_policy))
            })
            || policy
                .signal_policies
                .values()
                .any(|value| !valid_signal_policy(value))
        {
            return Err(DigitalTwinDivergenceError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        samples: &[TwinResidualSample],
        now_ms: u64,
    ) -> DigitalTwinDivergenceReport {
        let mut issues = Vec::new();
        let mut ids = BTreeSet::new();
        let mut by_signal = BTreeMap::<TwinSignal, Vec<&TwinResidualSample>>::new();

        for sample in samples {
            if sample.sample_id.trim().is_empty() || !ids.insert(sample.sample_id.clone()) {
                issues.push(DigitalTwinDivergenceIssue::DuplicateSampleId(
                    sample.sample_id.clone(),
                ));
            }
            if !sample.predicted.is_finite()
                || !sample.observed.is_finite()
                || !sample.combined_sigma.is_finite()
                || sample.combined_sigma <= 0.0
            {
                issues.push(DigitalTwinDivergenceIssue::InvalidSample(
                    sample.sample_id.clone(),
                ));
                continue;
            }
            if sample.evidence_ids.is_empty()
                || sample.evidence_ids.iter().any(|id| id.trim().is_empty())
            {
                issues.push(DigitalTwinDivergenceIssue::MissingEvidence(
                    sample.sample_id.clone(),
                ));
            }
            if sample.timestamp_ms > now_ms {
                issues.push(DigitalTwinDivergenceIssue::FutureSample(
                    sample.sample_id.clone(),
                ));
                continue;
            }
            if let Some(signal_policy) = self.policy.signal_policies.get(&sample.signal) {
                let age = now_ms.saturating_sub(sample.timestamp_ms);
                if age > signal_policy.maximum_sample_age_ms {
                    issues.push(DigitalTwinDivergenceIssue::StaleSample {
                        sample_id: sample.sample_id.clone(),
                        age_ms: age,
                        maximum_ms: signal_policy.maximum_sample_age_ms,
                    });
                }
            }
            by_signal.entry(sample.signal).or_default().push(sample);
        }

        let mut signal_reports = Vec::new();
        for signal in &self.policy.required_signals {
            let Some(signal_policy) = self.policy.signal_policies.get(signal) else {
                continue;
            };
            let entries = by_signal.get_mut(signal);
            let count = entries.as_ref().map_or(0, |values| values.len());
            if count == 0 {
                issues.push(DigitalTwinDivergenceIssue::MissingRequiredSignal(*signal));
                continue;
            }
            if count < self.policy.minimum_samples_per_signal {
                issues.push(DigitalTwinDivergenceIssue::InsufficientSamples {
                    signal: *signal,
                    observed: count,
                    required: self.policy.minimum_samples_per_signal,
                });
            }
            let values = entries.expect("present after nonzero count");
            values.sort_by(|left, right| {
                left.timestamp_ms
                    .cmp(&right.timestamp_ms)
                    .then_with(|| left.sample_id.cmp(&right.sample_id))
            });

            let mut sum_squares = 0.0;
            let mut peak = 0.0_f64;
            let mut warning_streak = 0usize;
            let mut unsafe_streak = 0usize;
            let mut maximum_warning_streak = 0usize;
            let mut maximum_unsafe_streak = 0usize;
            for sample in values.iter() {
                let normalized =
                    ((sample.observed - sample.predicted) / sample.combined_sigma).abs();
                sum_squares += normalized * normalized;
                peak = peak.max(normalized);
                if normalized >= signal_policy.warning_sigma {
                    warning_streak = warning_streak.saturating_add(1);
                } else {
                    warning_streak = 0;
                }
                if normalized >= signal_policy.unsafe_sigma {
                    unsafe_streak = unsafe_streak.saturating_add(1);
                } else {
                    unsafe_streak = 0;
                }
                maximum_warning_streak = maximum_warning_streak.max(warning_streak);
                maximum_unsafe_streak = maximum_unsafe_streak.max(unsafe_streak);
            }
            if maximum_unsafe_streak >= signal_policy.unsafe_persistence_samples {
                issues.push(DigitalTwinDivergenceIssue::UnsafePersistence {
                    signal: *signal,
                    consecutive_samples: maximum_unsafe_streak,
                });
            } else if maximum_warning_streak >= signal_policy.warning_persistence_samples {
                issues.push(DigitalTwinDivergenceIssue::WarningPersistence {
                    signal: *signal,
                    consecutive_samples: maximum_warning_streak,
                });
            }
            signal_reports.push(TwinSignalDivergence {
                signal: *signal,
                sample_count: values.len(),
                rms_normalized_residual: (sum_squares / values.len() as f64).sqrt(),
                peak_normalized_residual: peak,
                final_warning_streak: warning_streak,
                final_unsafe_streak: unsafe_streak,
                maximum_warning_streak,
                maximum_unsafe_streak,
            });
        }

        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                DigitalTwinDivergenceIssue::MissingRequiredSignal(_)
                    | DigitalTwinDivergenceIssue::InsufficientSamples { .. }
                    | DigitalTwinDivergenceIssue::DuplicateSampleId(_)
                    | DigitalTwinDivergenceIssue::InvalidSample(_)
                    | DigitalTwinDivergenceIssue::MissingEvidence(_)
                    | DigitalTwinDivergenceIssue::FutureSample(_)
                    | DigitalTwinDivergenceIssue::StaleSample { .. }
            )
        });
        let unsafe_divergence = issues
            .iter()
            .any(|issue| matches!(issue, DigitalTwinDivergenceIssue::UnsafePersistence { .. }));
        let warning = issues
            .iter()
            .any(|issue| matches!(issue, DigitalTwinDivergenceIssue::WarningPersistence { .. }));
        let status = if incomplete {
            DigitalTwinDivergenceStatus::Incomplete
        } else if unsafe_divergence {
            DigitalTwinDivergenceStatus::Unsafe
        } else if warning {
            DigitalTwinDivergenceStatus::Restricted
        } else {
            DigitalTwinDivergenceStatus::Aligned
        };

        DigitalTwinDivergenceReport {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            assessed_at_ms: now_ms,
            status,
            signals: signal_reports,
            issues,
        }
    }
}

fn valid_signal_policy(policy: &TwinSignalPolicy) -> bool {
    policy.warning_sigma.is_finite()
        && policy.unsafe_sigma.is_finite()
        && policy.warning_sigma > 0.0
        && policy.unsafe_sigma > policy.warning_sigma
        && policy.warning_persistence_samples > 0
        && policy.unsafe_persistence_samples > 0
        && policy.maximum_sample_age_ms > 0
}

fn issue_sort_key(issue: &DigitalTwinDivergenceIssue) -> String {
    format!("{issue:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monitor() -> DigitalTwinDivergenceMonitor {
        DigitalTwinDivergenceMonitor::new(DigitalTwinDivergencePolicy {
            schema_version: "1".into(),
            policy_id: "twin-policy".into(),
            required_signals: vec![TwinSignal::MainRotorSpeed],
            signal_policies: BTreeMap::from([(
                TwinSignal::MainRotorSpeed,
                TwinSignalPolicy {
                    warning_sigma: 2.0,
                    unsafe_sigma: 4.0,
                    warning_persistence_samples: 3,
                    unsafe_persistence_samples: 2,
                    maximum_sample_age_ms: 1_000,
                },
            )]),
            minimum_samples_per_signal: 3,
        })
        .unwrap()
    }

    fn sample(id: &str, timestamp_ms: u64, residual_sigma: f64) -> TwinResidualSample {
        TwinResidualSample {
            sample_id: id.into(),
            timestamp_ms,
            signal: TwinSignal::MainRotorSpeed,
            predicted: 100.0,
            observed: 100.0 + residual_sigma,
            combined_sigma: 1.0,
            evidence_ids: vec![format!("evidence-{id}")],
        }
    }

    #[test]
    fn aligned_when_residuals_are_bounded() {
        let report = monitor().assess(
            &[
                sample("a", 800, 0.5),
                sample("b", 900, 1.0),
                sample("c", 1_000, 0.2),
            ],
            1_000,
        );
        assert_eq!(report.status, DigitalTwinDivergenceStatus::Aligned);
    }

    #[test]
    fn persistent_warning_restricts() {
        let report = monitor().assess(
            &[
                sample("a", 800, 2.5),
                sample("b", 900, 2.4),
                sample("c", 1_000, 2.2),
            ],
            1_000,
        );
        assert_eq!(report.status, DigitalTwinDivergenceStatus::Restricted);
    }

    #[test]
    fn persistent_unsafe_residual_is_unsafe() {
        let report = monitor().assess(
            &[
                sample("a", 800, 1.0),
                sample("b", 900, 4.5),
                sample("c", 1_000, 4.2),
            ],
            1_000,
        );
        assert_eq!(report.status, DigitalTwinDivergenceStatus::Unsafe);
    }

    #[test]
    fn missing_required_signal_is_incomplete() {
        let report = monitor().assess(&[], 1_000);
        assert_eq!(report.status, DigitalTwinDivergenceStatus::Incomplete);
    }
}
