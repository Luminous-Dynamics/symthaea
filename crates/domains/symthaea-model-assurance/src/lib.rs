// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic residual-based model assurance for evidence-bearing predictive systems.
//!
//! A model is useful evidence only while its predictions remain compatible with
//! observed reality under declared uncertainty, freshness, persistence, and
//! provenance requirements. This crate intentionally does not retune models,
//! authorize physical actions, or infer why a model diverged.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Stable, open-ended signal identifier. Domain crates choose names such as
/// `main-rotor-speed`, `camera-bearing`, `wind-speed`, or `battery-voltage`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SignalId(pub String);

impl SignalId {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelAssuranceError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ModelAssuranceError::InvalidSignalId);
        }
        Ok(Self(value))
    }

    pub fn validate(&self) -> bool {
        !self.0.trim().is_empty()
    }
}

/// Per-signal residual thresholds and persistence requirements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignalPolicy {
    /// Normalized residual magnitude that begins a restricted/warning streak.
    pub warning_sigma: f64,
    /// Normalized residual magnitude that begins an unsafe streak.
    pub unsafe_sigma: f64,
    pub warning_persistence_samples: usize,
    pub unsafe_persistence_samples: usize,
    pub maximum_sample_age_ms: u64,
}

impl SignalPolicy {
    pub fn validate(&self) -> bool {
        self.warning_sigma.is_finite()
            && self.unsafe_sigma.is_finite()
            && self.warning_sigma > 0.0
            && self.unsafe_sigma > self.warning_sigma
            && self.warning_persistence_samples > 0
            && self.unsafe_persistence_samples > 0
            && self.maximum_sample_age_ms > 0
    }
}

/// Reviewed policy describing what evidence is required before a predictive
/// model may be considered aligned with reality.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelAssurancePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub required_signals: Vec<SignalId>,
    pub signal_policies: BTreeMap<SignalId, SignalPolicy>,
    pub minimum_samples_per_signal: usize,
}

impl ModelAssurancePolicy {
    pub fn validate(&self) -> bool {
        let required = self.required_signals.iter().cloned().collect::<BTreeSet<_>>();
        !self.schema_version.trim().is_empty()
            && !self.policy_id.trim().is_empty()
            && !self.required_signals.is_empty()
            && required.len() == self.required_signals.len()
            && self.required_signals.iter().all(SignalId::validate)
            && self.minimum_samples_per_signal > 0
            && required.iter().all(|signal| {
                self.signal_policies
                    .get(signal)
                    .is_some_and(SignalPolicy::validate)
            })
            && self
                .signal_policies
                .iter()
                .all(|(signal, policy)| signal.validate() && policy.validate())
    }
}

/// One comparison between a qualified model prediction and an observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidualSample {
    pub sample_id: String,
    pub timestamp_ms: u64,
    pub signal: SignalId,
    pub predicted: f64,
    pub observed: f64,
    /// One-sigma uncertainty of the *difference* between prediction and observation.
    /// Callers are responsible for correctly combining independent uncertainty terms.
    pub combined_sigma: f64,
    /// Evidence references for both the model prediction and observation lineage.
    pub evidence_refs: Vec<String>,
}

impl ResidualSample {
    fn numerical_valid(&self) -> bool {
        !self.sample_id.trim().is_empty()
            && self.signal.validate()
            && self.predicted.is_finite()
            && self.observed.is_finite()
            && self.combined_sigma.is_finite()
            && self.combined_sigma > 0.0
    }

    fn has_evidence(&self) -> bool {
        !self.evidence_refs.is_empty() && self.evidence_refs.iter().all(|id| !id.trim().is_empty())
    }

    pub fn normalized_residual(&self) -> Option<f64> {
        self.numerical_valid()
            .then(|| ((self.observed - self.predicted) / self.combined_sigma).abs())
    }
}

/// Capability-facing assurance state. Ordering is not used for safety decisions;
/// downstream adapters must map every variant explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelAssuranceStatus {
    Aligned,
    Restricted,
    Unsafe,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelAssuranceIssue {
    MissingRequiredSignal(SignalId),
    InsufficientSamples {
        signal: SignalId,
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
        signal: SignalId,
        consecutive_samples: usize,
    },
    UnsafePersistence {
        signal: SignalId,
        consecutive_samples: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignalAssurance {
    pub signal: SignalId,
    /// Number of admissible, fresh, evidence-bearing samples used in metrics.
    pub sample_count: usize,
    pub rms_normalized_residual: f64,
    pub peak_normalized_residual: f64,
    pub final_warning_streak: usize,
    pub final_unsafe_streak: usize,
    pub maximum_warning_streak: usize,
    pub maximum_unsafe_streak: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelAssuranceReport {
    pub schema_version: String,
    pub policy_id: String,
    pub assessed_at_ms: u64,
    pub status: ModelAssuranceStatus,
    pub signals: Vec<SignalAssurance>,
    pub issues: Vec<ModelAssuranceIssue>,
}

impl ModelAssuranceReport {
    /// Deterministic serialization for evidence binding. This is not a signature.
    pub fn canonical_json(&self) -> Result<Vec<u8>, ModelAssuranceError> {
        let mut canonical = self.clone();
        canonical.signals.sort_by(|a, b| a.signal.cmp(&b.signal));
        canonical.issues.sort_by_key(issue_sort_key);
        serde_json::to_vec(&canonical).map_err(|_| ModelAssuranceError::SerializationFailed)
    }

    /// Lightweight deterministic receipt digest retained for parity with the
    /// original helicopter monitor. Security-sensitive callers should bind the
    /// canonical bytes with the repository's stronger evidence/signature layer.
    pub fn digest_fnv1a64(&self) -> Result<String, ModelAssuranceError> {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in self.canonical_json()? {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }

    /// Model assurance is evidence about model validity, never physical authority.
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelAssuranceError {
    InvalidPolicy,
    InvalidSignalId,
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct ModelAssuranceMonitor {
    policy: ModelAssurancePolicy,
}

impl ModelAssuranceMonitor {
    pub fn new(policy: ModelAssurancePolicy) -> Result<Self, ModelAssuranceError> {
        if !policy.validate() {
            return Err(ModelAssuranceError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn policy(&self) -> &ModelAssurancePolicy {
        &self.policy
    }

    /// Assess only fresh, unique, numerically valid, evidence-bearing residuals.
    /// Invalid or stale samples remain visible as issues but cannot satisfy the
    /// minimum-sample requirement or improve assurance.
    pub fn assess(&self, samples: &[ResidualSample], now_ms: u64) -> ModelAssuranceReport {
        let mut issues = Vec::new();
        let mut seen_ids = BTreeSet::new();
        let mut admitted = BTreeMap::<SignalId, Vec<&ResidualSample>>::new();

        for sample in samples {
            if sample.sample_id.trim().is_empty() {
                issues.push(ModelAssuranceIssue::InvalidSample(sample.sample_id.clone()));
                continue;
            }
            if !seen_ids.insert(sample.sample_id.clone()) {
                issues.push(ModelAssuranceIssue::DuplicateSampleId(sample.sample_id.clone()));
                continue;
            }
            if !sample.numerical_valid() {
                issues.push(ModelAssuranceIssue::InvalidSample(sample.sample_id.clone()));
                continue;
            }
            if !sample.has_evidence() {
                issues.push(ModelAssuranceIssue::MissingEvidence(sample.sample_id.clone()));
                continue;
            }
            if sample.timestamp_ms > now_ms {
                issues.push(ModelAssuranceIssue::FutureSample(sample.sample_id.clone()));
                continue;
            }
            let Some(signal_policy) = self.policy.signal_policies.get(&sample.signal) else {
                // Optional unconfigured signals are ignored rather than allowed to
                // influence assurance for the reviewed required-signal set.
                continue;
            };
            let age = now_ms.saturating_sub(sample.timestamp_ms);
            if age > signal_policy.maximum_sample_age_ms {
                issues.push(ModelAssuranceIssue::StaleSample {
                    sample_id: sample.sample_id.clone(),
                    age_ms: age,
                    maximum_ms: signal_policy.maximum_sample_age_ms,
                });
                continue;
            }
            admitted.entry(sample.signal.clone()).or_default().push(sample);
        }

        let mut signal_reports = Vec::new();
        for signal in &self.policy.required_signals {
            let signal_policy = self
                .policy
                .signal_policies
                .get(signal)
                .expect("validated policy contains every required signal");
            let mut entries = admitted.remove(signal).unwrap_or_default();
            entries.sort_by(|left, right| {
                left.timestamp_ms
                    .cmp(&right.timestamp_ms)
                    .then_with(|| left.sample_id.cmp(&right.sample_id))
            });

            if entries.is_empty() {
                issues.push(ModelAssuranceIssue::MissingRequiredSignal(signal.clone()));
                continue;
            }
            if entries.len() < self.policy.minimum_samples_per_signal {
                issues.push(ModelAssuranceIssue::InsufficientSamples {
                    signal: signal.clone(),
                    observed: entries.len(),
                    required: self.policy.minimum_samples_per_signal,
                });
            }

            let mut sum_squares = 0.0;
            let mut peak = 0.0_f64;
            let mut warning_streak = 0usize;
            let mut unsafe_streak = 0usize;
            let mut maximum_warning_streak = 0usize;
            let mut maximum_unsafe_streak = 0usize;

            for sample in &entries {
                let normalized = sample
                    .normalized_residual()
                    .expect("admitted samples are numerically valid");
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
                issues.push(ModelAssuranceIssue::UnsafePersistence {
                    signal: signal.clone(),
                    consecutive_samples: maximum_unsafe_streak,
                });
            } else if maximum_warning_streak >= signal_policy.warning_persistence_samples {
                issues.push(ModelAssuranceIssue::WarningPersistence {
                    signal: signal.clone(),
                    consecutive_samples: maximum_warning_streak,
                });
            }

            signal_reports.push(SignalAssurance {
                signal: signal.clone(),
                sample_count: entries.len(),
                rms_normalized_residual: (sum_squares / entries.len() as f64).sqrt(),
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
                ModelAssuranceIssue::MissingRequiredSignal(_)
                    | ModelAssuranceIssue::InsufficientSamples { .. }
                    | ModelAssuranceIssue::DuplicateSampleId(_)
                    | ModelAssuranceIssue::InvalidSample(_)
                    | ModelAssuranceIssue::MissingEvidence(_)
                    | ModelAssuranceIssue::FutureSample(_)
                    | ModelAssuranceIssue::StaleSample { .. }
            )
        });
        let unsafe_divergence = issues
            .iter()
            .any(|issue| matches!(issue, ModelAssuranceIssue::UnsafePersistence { .. }));
        let warning = issues
            .iter()
            .any(|issue| matches!(issue, ModelAssuranceIssue::WarningPersistence { .. }));

        // Missing/invalid evidence takes precedence: an incomplete case cannot
        // be promoted to a confident unsafe/restricted conclusion.
        let status = if incomplete {
            ModelAssuranceStatus::Incomplete
        } else if unsafe_divergence {
            ModelAssuranceStatus::Unsafe
        } else if warning {
            ModelAssuranceStatus::Restricted
        } else {
            ModelAssuranceStatus::Aligned
        };

        ModelAssuranceReport {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            assessed_at_ms: now_ms,
            status,
            signals: signal_reports,
            issues,
        }
    }
}

fn issue_sort_key(issue: &ModelAssuranceIssue) -> String {
    format!("{issue:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal() -> SignalId {
        SignalId::new("camera-bearing").unwrap()
    }

    fn monitor() -> ModelAssuranceMonitor {
        let signal = signal();
        ModelAssuranceMonitor::new(ModelAssurancePolicy {
            schema_version: "1".into(),
            policy_id: "camera-model-v1".into(),
            required_signals: vec![signal.clone()],
            signal_policies: BTreeMap::from([(
                signal,
                SignalPolicy {
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

    fn sample(id: &str, timestamp_ms: u64, residual_sigma: f64) -> ResidualSample {
        ResidualSample {
            sample_id: id.into(),
            timestamp_ms,
            signal: signal(),
            predicted: 100.0,
            observed: 100.0 + residual_sigma,
            combined_sigma: 1.0,
            evidence_refs: vec![format!("prediction:{id}"), format!("observation:{id}")],
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
        assert_eq!(report.status, ModelAssuranceStatus::Aligned);
        assert!(!report.grants_physical_authority());
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
        assert_eq!(report.status, ModelAssuranceStatus::Restricted);
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
        assert_eq!(report.status, ModelAssuranceStatus::Unsafe);
    }

    #[test]
    fn stale_samples_cannot_satisfy_minimum_evidence() {
        let report = monitor().assess(
            &[
                sample("a", 0, 0.1),
                sample("b", 100, 0.1),
                sample("c", 200, 0.1),
            ],
            2_000,
        );
        assert_eq!(report.status, ModelAssuranceStatus::Incomplete);
        assert!(report.signals.is_empty());
    }

    #[test]
    fn missing_provenance_cannot_improve_assurance() {
        let mut bad = sample("a", 800, 0.1);
        bad.evidence_refs.clear();
        let report = monitor().assess(&[bad, sample("b", 900, 0.1), sample("c", 1_000, 0.1)], 1_000);
        assert_eq!(report.status, ModelAssuranceStatus::Incomplete);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            ModelAssuranceIssue::MissingEvidence(id) if id == "a"
        )));
    }

    #[test]
    fn duplicate_sample_cannot_double_count_evidence() {
        let duplicated = sample("a", 800, 0.1);
        let report = monitor().assess(
            &[duplicated.clone(), duplicated, sample("b", 900, 0.1), sample("c", 1_000, 0.1)],
            1_000,
        );
        assert_eq!(report.status, ModelAssuranceStatus::Incomplete);
        assert_eq!(report.signals[0].sample_count, 3);
    }

    #[test]
    fn canonical_receipt_is_stable() {
        let report = monitor().assess(
            &[
                sample("a", 800, 0.5),
                sample("b", 900, 1.0),
                sample("c", 1_000, 0.2),
            ],
            1_000,
        );
        assert_eq!(report.digest_fnv1a64().unwrap(), report.digest_fnv1a64().unwrap());
    }
}
