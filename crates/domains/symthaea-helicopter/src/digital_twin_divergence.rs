// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime divergence monitoring between the qualified model and observed aircraft.
//!
//! The helicopter-facing API is retained for compatibility, but the assurance
//! theorem is delegated to `symthaea-model-assurance`. A digital twin is evidence
//! only while residuals remain bounded under declared uncertainty, persistence,
//! freshness, and provenance requirements.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use symthaea_model_assurance::{
    ModelAssuranceIssue, ModelAssuranceMonitor, ModelAssurancePolicy, ModelAssuranceStatus,
    ResidualSample, SignalAssurance, SignalId, SignalPolicy,
};

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

impl TwinSignal {
    const fn stable_name(self) -> &'static str {
        match self {
            Self::Position => "position",
            Self::Velocity => "velocity",
            Self::Attitude => "attitude",
            Self::AngularRate => "angular-rate",
            Self::MainRotorSpeed => "main-rotor-speed",
            Self::TailRotorSpeed => "tail-rotor-speed",
            Self::FuelMass => "fuel-mass",
            Self::ShaftPower => "shaft-power",
        }
    }

    fn signal_id(self) -> SignalId {
        SignalId(self.stable_name().to_string())
    }

    fn from_signal_id(signal: &SignalId) -> Option<Self> {
        match signal.0.as_str() {
            "position" => Some(Self::Position),
            "velocity" => Some(Self::Velocity),
            "attitude" => Some(Self::Attitude),
            "angular-rate" => Some(Self::AngularRate),
            "main-rotor-speed" => Some(Self::MainRotorSpeed),
            "tail-rotor-speed" => Some(Self::TailRotorSpeed),
            "fuel-mass" => Some(Self::FuelMass),
            "shaft-power" => Some(Self::ShaftPower),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwinSignalPolicy {
    pub warning_sigma: f64,
    pub unsafe_sigma: f64,
    pub warning_persistence_samples: usize,
    pub unsafe_persistence_samples: usize,
    pub maximum_sample_age_ms: u64,
}

impl TwinSignalPolicy {
    fn to_shared(&self) -> SignalPolicy {
        SignalPolicy {
            warning_sigma: self.warning_sigma,
            unsafe_sigma: self.unsafe_sigma,
            warning_persistence_samples: self.warning_persistence_samples,
            unsafe_persistence_samples: self.unsafe_persistence_samples,
            maximum_sample_age_ms: self.maximum_sample_age_ms,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DigitalTwinDivergencePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub required_signals: Vec<TwinSignal>,
    pub signal_policies: BTreeMap<TwinSignal, TwinSignalPolicy>,
    pub minimum_samples_per_signal: usize,
}

impl DigitalTwinDivergencePolicy {
    fn to_shared(&self) -> ModelAssurancePolicy {
        ModelAssurancePolicy {
            schema_version: self.schema_version.clone(),
            policy_id: self.policy_id.clone(),
            required_signals: self
                .required_signals
                .iter()
                .copied()
                .map(TwinSignal::signal_id)
                .collect(),
            signal_policies: self
                .signal_policies
                .iter()
                .map(|(signal, policy)| (signal.signal_id(), policy.to_shared()))
                .collect(),
            minimum_samples_per_signal: self.minimum_samples_per_signal,
        }
    }
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

impl TwinResidualSample {
    fn to_shared(&self) -> ResidualSample {
        ResidualSample {
            sample_id: self.sample_id.clone(),
            timestamp_ms: self.timestamp_ms,
            signal: self.signal.signal_id(),
            predicted: self.predicted,
            observed: self.observed,
            combined_sigma: self.combined_sigma,
            evidence_refs: self.evidence_ids.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DigitalTwinDivergenceStatus {
    Aligned,
    Restricted,
    Unsafe,
    Incomplete,
}

impl From<ModelAssuranceStatus> for DigitalTwinDivergenceStatus {
    fn from(status: ModelAssuranceStatus) -> Self {
        match status {
            ModelAssuranceStatus::Aligned => Self::Aligned,
            ModelAssuranceStatus::Restricted => Self::Restricted,
            ModelAssuranceStatus::Unsafe => Self::Unsafe,
            ModelAssuranceStatus::Incomplete => Self::Incomplete,
        }
    }
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

impl DigitalTwinDivergenceIssue {
    fn from_shared(issue: ModelAssuranceIssue) -> Option<Self> {
        match issue {
            ModelAssuranceIssue::MissingRequiredSignal(signal) => {
                Some(Self::MissingRequiredSignal(TwinSignal::from_signal_id(&signal)?))
            }
            ModelAssuranceIssue::InsufficientSamples {
                signal,
                observed,
                required,
            } => Some(Self::InsufficientSamples {
                signal: TwinSignal::from_signal_id(&signal)?,
                observed,
                required,
            }),
            ModelAssuranceIssue::DuplicateSampleId(id) => Some(Self::DuplicateSampleId(id)),
            ModelAssuranceIssue::InvalidSample(id) => Some(Self::InvalidSample(id)),
            ModelAssuranceIssue::MissingEvidence(id) => Some(Self::MissingEvidence(id)),
            ModelAssuranceIssue::FutureSample(id) => Some(Self::FutureSample(id)),
            ModelAssuranceIssue::StaleSample {
                sample_id,
                age_ms,
                maximum_ms,
            } => Some(Self::StaleSample {
                sample_id,
                age_ms,
                maximum_ms,
            }),
            ModelAssuranceIssue::WarningPersistence {
                signal,
                consecutive_samples,
            } => Some(Self::WarningPersistence {
                signal: TwinSignal::from_signal_id(&signal)?,
                consecutive_samples,
            }),
            ModelAssuranceIssue::UnsafePersistence {
                signal,
                consecutive_samples,
            } => Some(Self::UnsafePersistence {
                signal: TwinSignal::from_signal_id(&signal)?,
                consecutive_samples,
            }),
        }
    }
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

impl TwinSignalDivergence {
    fn from_shared(report: SignalAssurance) -> Option<Self> {
        Some(Self {
            signal: TwinSignal::from_signal_id(&report.signal)?,
            sample_count: report.sample_count,
            rms_normalized_residual: report.rms_normalized_residual,
            peak_normalized_residual: report.peak_normalized_residual,
            final_warning_streak: report.final_warning_streak,
            final_unsafe_streak: report.final_unsafe_streak,
            maximum_warning_streak: report.maximum_warning_streak,
            maximum_unsafe_streak: report.maximum_unsafe_streak,
        })
    }
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
    inner: ModelAssuranceMonitor,
}

impl DigitalTwinDivergenceMonitor {
    pub fn new(policy: DigitalTwinDivergencePolicy) -> Result<Self, DigitalTwinDivergenceError> {
        let inner = ModelAssuranceMonitor::new(policy.to_shared())
            .map_err(|_| DigitalTwinDivergenceError::InvalidPolicy)?;
        Ok(Self { inner })
    }

    pub fn assess(
        &self,
        samples: &[TwinResidualSample],
        now_ms: u64,
    ) -> DigitalTwinDivergenceReport {
        let shared_samples = samples
            .iter()
            .map(TwinResidualSample::to_shared)
            .collect::<Vec<_>>();
        let shared = self.inner.assess(&shared_samples, now_ms);

        DigitalTwinDivergenceReport {
            schema_version: shared.schema_version,
            policy_id: shared.policy_id,
            assessed_at_ms: shared.assessed_at_ms,
            status: shared.status.into(),
            signals: shared
                .signals
                .into_iter()
                .filter_map(TwinSignalDivergence::from_shared)
                .collect(),
            issues: shared
                .issues
                .into_iter()
                .filter_map(DigitalTwinDivergenceIssue::from_shared)
                .collect(),
        }
    }
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

    #[test]
    fn stale_samples_do_not_count_toward_minimum_evidence() {
        let report = monitor().assess(
            &[
                sample("a", 0, 0.1),
                sample("b", 100, 0.1),
                sample("c", 200, 0.1),
            ],
            2_000,
        );
        assert_eq!(report.status, DigitalTwinDivergenceStatus::Incomplete);
        assert!(report.signals.is_empty());
    }

    #[test]
    fn missing_evidence_does_not_enter_residual_metrics() {
        let mut missing = sample("a", 800, 0.1);
        missing.evidence_ids.clear();
        let report = monitor().assess(
            &[missing, sample("b", 900, 0.1), sample("c", 1_000, 0.1)],
            1_000,
        );
        assert_eq!(report.status, DigitalTwinDivergenceStatus::Incomplete);
        assert_eq!(report.signals[0].sample_count, 2);
    }
}