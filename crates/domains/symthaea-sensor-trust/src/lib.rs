// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Replay-resistant observation admission and source-trust supervision.
//!
//! This crate generalizes the defensive ideas already used in Symthaea's
//! subterranean sensor redundancy layer without embedding any domain-specific
//! fusion algorithm. It decides whether observations are admissible evidence;
//! it does not decide identity, intent, risk, or physical action.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, HashSet};

use symthaea_domain_awareness::{
    IntegrityStatus, Modality, ObservationEnvelope, SensorHealth,
};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceTrustPolicy {
    /// Hard cap on observations considered from one admission batch.
    pub max_observations_per_batch: usize,
    /// Sources at or below this reliability are isolated.
    pub source_isolation_threshold: f64,
    /// Maximum reliability recovery per positive independent feedback event.
    pub reliability_recovery_rate: f64,
    /// Maximum reliability penalty per negative independent feedback event.
    pub reliability_penalty_rate: f64,
    /// Minimum distinct physical sensing elements required for a usable batch.
    pub minimum_independent_physical_sources: usize,
    /// Minimum distinct sensor modalities required for a usable batch.
    pub minimum_independent_modalities: usize,
}

impl SourceTrustPolicy {
    pub fn validate(&self) -> bool {
        self.max_observations_per_batch > 0
            && self.source_isolation_threshold.is_finite()
            && (0.0..=1.0).contains(&self.source_isolation_threshold)
            && self.reliability_recovery_rate.is_finite()
            && (0.0..=1.0).contains(&self.reliability_recovery_rate)
            && self.reliability_penalty_rate.is_finite()
            && (0.0..=1.0).contains(&self.reliability_penalty_rate)
            && self.minimum_independent_physical_sources > 0
            && self.minimum_independent_modalities > 0
    }
}

impl Default for SourceTrustPolicy {
    fn default() -> Self {
        Self {
            max_observations_per_batch: 64,
            source_isolation_threshold: 0.20,
            reliability_recovery_rate: 0.02,
            reliability_penalty_rate: 0.15,
            minimum_independent_physical_sources: 1,
            minimum_independent_modalities: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceTrustState {
    pub reliability: f64,
    pub last_sequence: u64,
    pub seen: bool,
    pub accepted_observations: u64,
    pub rejected_observations: u64,
}

impl Default for SourceTrustState {
    fn default() -> Self {
        Self {
            reliability: 1.0,
            last_sequence: 0,
            seen: false,
            accepted_observations: 0,
            rejected_observations: 0,
        }
    }
}

impl SourceTrustState {
    pub fn validate(&self) -> bool {
        self.reliability.is_finite() && (0.0..=1.0).contains(&self.reliability)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionReason {
    BatchLimitExceeded,
    DuplicateObservationId,
    DuplicateSourceInBatch,
    InvalidObservation,
    StaleObservation,
    IntegrityFailed,
    SourceUnavailable,
    ReplayOrReorder,
    SourceIsolated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationRejection {
    pub observation_id: Uuid,
    pub source_id: String,
    pub reason: RejectionReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationAdmissionReport {
    pub accepted_observation_ids: Vec<Uuid>,
    pub rejections: Vec<ObservationRejection>,
    pub independent_physical_sources: usize,
    pub independent_modalities: usize,
    pub isolated_sources: usize,
    pub requires_fail_closed: bool,
}

impl ObservationAdmissionReport {
    pub fn accepted_count(&self) -> usize {
        self.accepted_observation_ids.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustFeedbackError {
    InvalidPolicy,
    UnknownSource,
    InvalidQuality,
}

/// Stateful replay/freshness/source-trust supervisor.
///
/// A source's sequence history is keyed by `ObservationEnvelope::source_id` while
/// physical independence is computed from `EvidenceLineage::physical_source_id`.
/// This deliberately allows two processors to have independent sequence spaces
/// without allowing them to masquerade as two physical witnesses.
#[derive(Debug, Clone)]
pub struct SourceTrustSupervisor {
    policy: SourceTrustPolicy,
    sources: BTreeMap<String, SourceTrustState>,
}

impl SourceTrustSupervisor {
    pub fn new(policy: SourceTrustPolicy) -> Result<Self, TrustFeedbackError> {
        if !policy.validate() {
            return Err(TrustFeedbackError::InvalidPolicy);
        }
        Ok(Self {
            policy,
            sources: BTreeMap::new(),
        })
    }

    pub fn policy(&self) -> SourceTrustPolicy {
        self.policy
    }

    pub fn source_state(&self, source_id: &str) -> Option<SourceTrustState> {
        self.sources.get(source_id).copied()
    }

    pub fn admit_batch(
        &mut self,
        observations: &[ObservationEnvelope],
        now_ms: u64,
    ) -> ObservationAdmissionReport {
        let mut accepted = Vec::new();
        let mut rejections = Vec::new();
        let mut seen_observation_ids = HashSet::new();
        let mut seen_source_ids = HashSet::new();
        let mut physical_sources = HashSet::new();
        let mut modalities = HashSet::new();

        for (index, observation) in observations.iter().enumerate() {
            if index >= self.policy.max_observations_per_batch {
                self.reject(observation, RejectionReason::BatchLimitExceeded, &mut rejections);
                continue;
            }
            if !seen_observation_ids.insert(observation.observation_id) {
                self.reject(
                    observation,
                    RejectionReason::DuplicateObservationId,
                    &mut rejections,
                );
                continue;
            }
            if !seen_source_ids.insert(observation.source_id.clone()) {
                self.reject(
                    observation,
                    RejectionReason::DuplicateSourceInBatch,
                    &mut rejections,
                );
                continue;
            }
            if !observation.validate() {
                self.reject(observation, RejectionReason::InvalidObservation, &mut rejections);
                continue;
            }
            if !observation.is_fresh_at(now_ms) {
                self.reject(observation, RejectionReason::StaleObservation, &mut rejections);
                continue;
            }
            if observation.integrity == IntegrityStatus::Failed {
                self.reject(observation, RejectionReason::IntegrityFailed, &mut rejections);
                continue;
            }
            if observation.sensor_health == SensorHealth::Unavailable {
                self.reject(observation, RejectionReason::SourceUnavailable, &mut rejections);
                continue;
            }

            let state = self.sources.entry(observation.source_id.clone()).or_default();
            if state.seen && observation.sequence <= state.last_sequence {
                state.rejected_observations = state.rejected_observations.saturating_add(1);
                rejections.push(ObservationRejection {
                    observation_id: observation.observation_id,
                    source_id: observation.source_id.clone(),
                    reason: RejectionReason::ReplayOrReorder,
                });
                continue;
            }
            if state.reliability <= self.policy.source_isolation_threshold {
                // Consume a fresh monotonic sequence from an isolated source so
                // replaying it later cannot become admissible after rehabilitation.
                state.seen = true;
                state.last_sequence = observation.sequence;
                state.rejected_observations = state.rejected_observations.saturating_add(1);
                rejections.push(ObservationRejection {
                    observation_id: observation.observation_id,
                    source_id: observation.source_id.clone(),
                    reason: RejectionReason::SourceIsolated,
                });
                continue;
            }

            state.seen = true;
            state.last_sequence = observation.sequence;
            state.accepted_observations = state.accepted_observations.saturating_add(1);
            accepted.push(observation.observation_id);
            physical_sources.insert(observation.lineage.physical_source_id.clone());
            modalities.insert(observation.modality);
        }

        let isolated_sources = self
            .sources
            .values()
            .filter(|state| state.reliability <= self.policy.source_isolation_threshold)
            .count();
        let requires_fail_closed = accepted.is_empty()
            || physical_sources.len() < self.policy.minimum_independent_physical_sources
            || modalities.len() < self.policy.minimum_independent_modalities;

        ObservationAdmissionReport {
            accepted_observation_ids: accepted,
            rejections,
            independent_physical_sources: physical_sources.len(),
            independent_modalities: modalities.len(),
            isolated_sources,
            requires_fail_closed,
        }
    }

    /// Record independently-derived evidence quality for one source.
    ///
    /// `quality` is a normalized verifier result in `[0, 1]`. A low score can
    /// isolate a source; a high score can only recover reliability at the
    /// policy's bounded recovery rate. This function must not be fed the source's
    /// own confidence score, otherwise a source could self-rehabilitate.
    pub fn record_independent_quality_feedback(
        &mut self,
        source_id: &str,
        quality: f64,
    ) -> Result<SourceTrustState, TrustFeedbackError> {
        if !quality.is_finite() || !(0.0..=1.0).contains(&quality) {
            return Err(TrustFeedbackError::InvalidQuality);
        }
        let Some(state) = self.sources.get_mut(source_id) else {
            return Err(TrustFeedbackError::UnknownSource);
        };
        let negative = 1.0 - quality;
        let delta = self.policy.reliability_recovery_rate * quality
            - self.policy.reliability_penalty_rate * negative;
        state.reliability = (state.reliability + delta).clamp(0.0, 1.0);
        Ok(*state)
    }

    /// Explicit maintenance action for commissioning/tests. Runtime code should
    /// prefer independent feedback rather than silently resetting trust history.
    pub fn reset_source_after_review(&mut self, source_id: &str) -> bool {
        let Some(state) = self.sources.get_mut(source_id) else {
            return false;
        };
        *state = SourceTrustState::default();
        true
    }

    fn reject(
        &mut self,
        observation: &ObservationEnvelope,
        reason: RejectionReason,
        rejections: &mut Vec<ObservationRejection>,
    ) {
        if !observation.source_id.trim().is_empty() {
            let state = self.sources.entry(observation.source_id.clone()).or_default();
            state.rejected_observations = state.rejected_observations.saturating_add(1);
        }
        rejections.push(ObservationRejection {
            observation_id: observation.observation_id,
            source_id: observation.source_id.clone(),
            reason,
        });
    }
}

impl Default for SourceTrustSupervisor {
    fn default() -> Self {
        Self::new(SourceTrustPolicy::default()).expect("default sensor trust policy is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_domain_awareness::{
        Domain, EvidenceLineage, Measurement, MeasurementUncertainty, TimeEvidence,
    };

    fn observation(
        source: &str,
        physical: &str,
        modality: Modality,
        sequence: u64,
        observed_at_ms: u64,
    ) -> ObservationEnvelope {
        ObservationEnvelope {
            observation_id: Uuid::new_v4(),
            source_id: source.to_string(),
            sequence,
            domain: Domain::Air,
            modality,
            coordinate_frame: "local-test".to_string(),
            measurement: Measurement::Scalar {
                quantity: "presence".to_string(),
                value: 1.0,
                unit: "bool".to_string(),
            },
            uncertainty: MeasurementUncertainty {
                position_sigma_m: None,
                velocity_sigma_mps: None,
                bearing_sigma_deg: None,
            },
            time: TimeEvidence {
                observed_at_ms,
                received_at_ms: observed_at_ms + 5,
                clock_source: "test-clock".to_string(),
                clock_uncertainty_ms: 1,
                maximum_valid_age_ms: 1_000,
            },
            lineage: EvidenceLineage {
                physical_source_id: physical.to_string(),
                processor_id: format!("processor-{source}"),
                network_path: "local".to_string(),
                clock_domain: "test-clock".to_string(),
            },
            integrity: IntegrityStatus::Verified,
            sensor_health: SensorHealth::Nominal,
            confidence: 0.8,
            evidence_refs: vec![format!("evidence:{source}:{sequence}")],
        }
    }

    #[test]
    fn replay_or_reorder_is_rejected() {
        let mut supervisor = SourceTrustSupervisor::default();
        let first = observation("eo-a", "camera-a", Modality::ElectroOptical, 10, 1_000);
        assert_eq!(supervisor.admit_batch(&[first], 1_100).accepted_count(), 1);
        let replay = observation("eo-a", "camera-a", Modality::ElectroOptical, 10, 1_200);
        let report = supervisor.admit_batch(&[replay], 1_250);
        assert_eq!(report.accepted_count(), 0);
        assert_eq!(report.rejections[0].reason, RejectionReason::ReplayOrReorder);
    }

    #[test]
    fn stale_observation_is_rejected_without_advancing_sequence() {
        let mut supervisor = SourceTrustSupervisor::default();
        let stale = observation("eo-a", "camera-a", Modality::ElectroOptical, 10, 1_000);
        let report = supervisor.admit_batch(&[stale], 3_000);
        assert_eq!(report.rejections[0].reason, RejectionReason::StaleObservation);
        let fresh = observation("eo-a", "camera-a", Modality::ElectroOptical, 10, 3_100);
        assert_eq!(supervisor.admit_batch(&[fresh], 3_150).accepted_count(), 1);
    }

    #[test]
    fn two_processors_on_one_camera_do_not_satisfy_two_source_quorum() {
        let policy = SourceTrustPolicy {
            minimum_independent_physical_sources: 2,
            minimum_independent_modalities: 1,
            ..SourceTrustPolicy::default()
        };
        let mut supervisor = SourceTrustSupervisor::new(policy).unwrap();
        let a = observation("eo-model-a", "camera-1", Modality::ElectroOptical, 1, 1_000);
        let b = observation("eo-model-b", "camera-1", Modality::ElectroOptical, 1, 1_000);
        let report = supervisor.admit_batch(&[a, b], 1_050);
        assert_eq!(report.accepted_count(), 2);
        assert_eq!(report.independent_physical_sources, 1);
        assert!(report.requires_fail_closed);
    }

    #[test]
    fn independent_eo_and_ir_can_satisfy_diverse_quorum() {
        let policy = SourceTrustPolicy {
            minimum_independent_physical_sources: 2,
            minimum_independent_modalities: 2,
            ..SourceTrustPolicy::default()
        };
        let mut supervisor = SourceTrustSupervisor::new(policy).unwrap();
        let eo = observation("eo-a", "camera-1", Modality::ElectroOptical, 1, 1_000);
        let ir = observation("ir-b", "thermal-2", Modality::Infrared, 1, 1_000);
        let report = supervisor.admit_batch(&[eo, ir], 1_050);
        assert_eq!(report.independent_physical_sources, 2);
        assert_eq!(report.independent_modalities, 2);
        assert!(!report.requires_fail_closed);
    }

    #[test]
    fn failed_integrity_never_enters_accepted_evidence() {
        let mut supervisor = SourceTrustSupervisor::default();
        let mut bad = observation("eo-a", "camera-a", Modality::ElectroOptical, 1, 1_000);
        bad.integrity = IntegrityStatus::Failed;
        let report = supervisor.admit_batch(&[bad], 1_050);
        assert_eq!(report.accepted_count(), 0);
        assert_eq!(report.rejections[0].reason, RejectionReason::IntegrityFailed);
    }

    #[test]
    fn independent_negative_feedback_can_isolate_source() {
        let policy = SourceTrustPolicy {
            reliability_penalty_rate: 0.5,
            source_isolation_threshold: 0.2,
            ..SourceTrustPolicy::default()
        };
        let mut supervisor = SourceTrustSupervisor::new(policy).unwrap();
        let first = observation("eo-a", "camera-a", Modality::ElectroOptical, 1, 1_000);
        supervisor.admit_batch(&[first], 1_050);
        supervisor.record_independent_quality_feedback("eo-a", 0.0).unwrap();
        let state = supervisor
            .record_independent_quality_feedback("eo-a", 0.0)
            .unwrap();
        assert!(state.reliability <= 0.2);
        let next = observation("eo-a", "camera-a", Modality::ElectroOptical, 2, 1_100);
        let report = supervisor.admit_batch(&[next], 1_150);
        assert_eq!(report.rejections[0].reason, RejectionReason::SourceIsolated);
    }

    #[test]
    fn positive_feedback_recovers_only_at_bounded_rate() {
        let policy = SourceTrustPolicy {
            reliability_recovery_rate: 0.05,
            reliability_penalty_rate: 0.5,
            ..SourceTrustPolicy::default()
        };
        let mut supervisor = SourceTrustSupervisor::new(policy).unwrap();
        let first = observation("eo-a", "camera-a", Modality::ElectroOptical, 1, 1_000);
        supervisor.admit_batch(&[first], 1_050);
        let penalized = supervisor
            .record_independent_quality_feedback("eo-a", 0.0)
            .unwrap();
        let recovered = supervisor
            .record_independent_quality_feedback("eo-a", 1.0)
            .unwrap();
        assert!(recovered.reliability > penalized.reliability);
        assert!(recovered.reliability - penalized.reliability <= 0.05 + f64::EPSILON);
    }

    #[test]
    fn duplicate_source_in_one_batch_cannot_overweight_quorum() {
        let mut supervisor = SourceTrustSupervisor::default();
        let a = observation("eo-a", "camera-a", Modality::ElectroOptical, 1, 1_000);
        let b = observation("eo-a", "camera-a", Modality::ElectroOptical, 2, 1_001);
        let report = supervisor.admit_batch(&[a, b], 1_050);
        assert_eq!(report.accepted_count(), 1);
        assert!(report
            .rejections
            .iter()
            .any(|rejection| rejection.reason == RejectionReason::DuplicateSourceInBatch));
    }
}
