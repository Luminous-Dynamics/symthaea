// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-only bridge from `symthaea-vision-manifold` into
//! `symthaea-domain-awareness`.
//!
//! The bridge is intentionally conservative:
//! - image/grid tracks are not silently promoted to world-space position;
//! - visual persistence is evidence of a persistent visual track, not identity;
//! - bearing evidence requires an explicit calibrated projection;
//! - camera degradation can only reduce the confidence ceiling;
//! - track assurance measures evidence quality and diversity, never authority.

#![deny(unsafe_code)]

use std::collections::{HashMap, HashSet};

use symthaea_domain_awareness::{
    Domain, EpistemicState, EvidenceLineage, IntegrityStatus, Measurement,
    MeasurementUncertainty, Modality, ObservationEnvelope, SensorHealth, TimeEvidence, Track,
};
use symthaea_vision_manifold::TrackedObject;
use uuid::Uuid;

/// Normalized, non-world-space evidence derived from one visual tracker state.
///
/// Coordinates are in `[0, 1]` across the tracker grid. Velocity is normalized
/// grid fraction per second. These values deliberately do not claim range,
/// bearing, meters, or a world coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualTrackEvidence {
    pub visual_track_id: u64,
    pub frame_index: u64,
    pub u_norm: f64,
    pub v_norm: f64,
    pub du_norm_per_s: f64,
    pub dv_norm_per_s: f64,
    pub track_length_frames: u64,
}

impl VisualTrackEvidence {
    pub fn validate(&self) -> bool {
        self.track_length_frames > 0
            && self.u_norm.is_finite()
            && self.v_norm.is_finite()
            && (0.0..=1.0).contains(&self.u_norm)
            && (0.0..=1.0).contains(&self.v_norm)
            && self.du_norm_per_s.is_finite()
            && self.dv_norm_per_s.is_finite()
    }

    /// Convert a vision-manifold track into normalized image/grid evidence.
    ///
    /// `TrackedObject` velocity is expressed in grid cells per observed frame,
    /// so converting to normalized velocity requires the observed frame rate.
    pub fn from_tracked_object(
        track: &TrackedObject,
        frame_index: u64,
        grid_cols: usize,
        grid_rows: usize,
        observed_fps: f64,
    ) -> Result<Self, VisualBridgeError> {
        if grid_cols == 0 || grid_rows == 0 {
            return Err(VisualBridgeError::InvalidGeometry);
        }
        if !observed_fps.is_finite() || observed_fps <= 0.0 {
            return Err(VisualBridgeError::InvalidFrameRate);
        }
        if track.centroid_col >= grid_cols || track.centroid_row >= grid_rows {
            return Err(VisualBridgeError::TrackOutsideGrid);
        }
        if !track.velocity_col.is_finite() || !track.velocity_row.is_finite() {
            return Err(VisualBridgeError::InvalidTrack);
        }

        let evidence = Self {
            visual_track_id: track.track_id,
            frame_index,
            u_norm: (track.centroid_col as f64 + 0.5) / grid_cols as f64,
            v_norm: (track.centroid_row as f64 + 0.5) / grid_rows as f64,
            du_norm_per_s: track.velocity_col as f64 * observed_fps / grid_cols as f64,
            dv_norm_per_s: track.velocity_row as f64 * observed_fps / grid_rows as f64,
            track_length_frames: track.track_length,
        };
        evidence
            .validate()
            .then_some(evidence)
            .ok_or(VisualBridgeError::InvalidTrack)
    }
}

/// Camera-specific source metadata required to construct a domain-awareness
/// observation without inventing source provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionObservationContext {
    pub source_id: String,
    pub physical_source_id: String,
    pub processor_id: String,
    pub network_path: String,
    pub clock_domain: String,
    pub clock_source: String,
    pub coordinate_frame: String,
    pub domain: Domain,
    pub modality: Modality,
    pub integrity: IntegrityStatus,
    pub maximum_valid_age_ms: u64,
    pub clock_uncertainty_ms: u64,
}

impl VisionObservationContext {
    pub fn validate(&self) -> bool {
        !self.source_id.trim().is_empty()
            && !self.physical_source_id.trim().is_empty()
            && !self.processor_id.trim().is_empty()
            && !self.network_path.trim().is_empty()
            && !self.clock_domain.trim().is_empty()
            && !self.clock_source.trim().is_empty()
            && !self.coordinate_frame.trim().is_empty()
            && self.maximum_valid_age_ms > 0
            && matches!(self.modality, Modality::ElectroOptical | Modality::Infrared)
    }

    /// Publish visual persistence only. This observation intentionally makes no
    /// spatial or identity claim beyond the existence of a persistent visual track.
    pub fn presence_observation(
        &self,
        visual: &VisualTrackEvidence,
        observed_at_ms: u64,
        received_at_ms: u64,
        health: SensorHealth,
        producer_confidence: f64,
        mut evidence_refs: Vec<String>,
    ) -> Result<ObservationEnvelope, VisualBridgeError> {
        self.validate_context_and_sample(visual, producer_confidence)?;
        evidence_refs.push(format!(
            "vision-track:{}:frame:{}",
            visual.visual_track_id, visual.frame_index
        ));
        let observation = self.base_observation(
            visual,
            observed_at_ms,
            received_at_ms,
            health,
            producer_confidence,
            evidence_refs,
            Measurement::Scalar {
                quantity: "visual-track-persistence".to_string(),
                value: visual.track_length_frames as f64,
                unit: "frames".to_string(),
            },
            MeasurementUncertainty {
                position_sigma_m: None,
                velocity_sigma_mps: None,
                bearing_sigma_deg: None,
            },
        );
        observation
            .validate()
            .then_some(observation)
            .ok_or(VisualBridgeError::InvalidObservation)
    }

    /// Publish calibrated bearing evidence. The projection must be supplied by
    /// a separately calibrated camera model; this bridge does not infer a field
    /// of view or fabricate camera intrinsics from tracker coordinates.
    pub fn bearing_observation(
        &self,
        visual: &VisualTrackEvidence,
        projection: &BearingProjection,
        observed_at_ms: u64,
        received_at_ms: u64,
        health: SensorHealth,
        producer_confidence: f64,
        mut evidence_refs: Vec<String>,
    ) -> Result<ObservationEnvelope, VisualBridgeError> {
        self.validate_context_and_sample(visual, producer_confidence)?;
        if !projection.validate() {
            return Err(VisualBridgeError::InvalidCalibration);
        }
        evidence_refs.push(projection.calibration_ref.clone());
        evidence_refs.push(format!(
            "vision-track:{}:frame:{}",
            visual.visual_track_id, visual.frame_index
        ));
        let observation = self.base_observation(
            visual,
            observed_at_ms,
            received_at_ms,
            health,
            producer_confidence,
            evidence_refs,
            Measurement::Bearing {
                azimuth_deg: projection.azimuth_deg,
                elevation_deg: Some(projection.elevation_deg),
            },
            MeasurementUncertainty {
                position_sigma_m: None,
                velocity_sigma_mps: None,
                bearing_sigma_deg: Some(projection.one_sigma_error_deg),
            },
        );
        observation
            .validate()
            .then_some(observation)
            .ok_or(VisualBridgeError::InvalidObservation)
    }

    fn validate_context_and_sample(
        &self,
        visual: &VisualTrackEvidence,
        producer_confidence: f64,
    ) -> Result<(), VisualBridgeError> {
        if !self.validate() {
            return Err(VisualBridgeError::InvalidContext);
        }
        if !visual.validate() {
            return Err(VisualBridgeError::InvalidTrack);
        }
        if !producer_confidence.is_finite() || !(0.0..=1.0).contains(&producer_confidence) {
            return Err(VisualBridgeError::InvalidConfidence);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn base_observation(
        &self,
        visual: &VisualTrackEvidence,
        observed_at_ms: u64,
        received_at_ms: u64,
        health: SensorHealth,
        producer_confidence: f64,
        evidence_refs: Vec<String>,
        measurement: Measurement,
        uncertainty: MeasurementUncertainty,
    ) -> ObservationEnvelope {
        ObservationEnvelope {
            observation_id: Uuid::new_v4(),
            source_id: self.source_id.clone(),
            sequence: visual.frame_index,
            domain: self.domain,
            modality: self.modality,
            coordinate_frame: self.coordinate_frame.clone(),
            measurement,
            uncertainty,
            time: TimeEvidence {
                observed_at_ms,
                received_at_ms,
                clock_source: self.clock_source.clone(),
                clock_uncertainty_ms: self.clock_uncertainty_ms,
                maximum_valid_age_ms: self.maximum_valid_age_ms,
            },
            lineage: EvidenceLineage {
                physical_source_id: self.physical_source_id.clone(),
                processor_id: self.processor_id.clone(),
                network_path: self.network_path.clone(),
                clock_domain: self.clock_domain.clone(),
            },
            integrity: self.integrity,
            sensor_health: health,
            confidence: producer_confidence.min(health.trust_cap()),
            evidence_refs,
        }
    }
}

/// Bearing supplied by an independently calibrated camera model.
#[derive(Debug, Clone, PartialEq)]
pub struct BearingProjection {
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub one_sigma_error_deg: f64,
    pub calibration_ref: String,
}

impl BearingProjection {
    pub fn validate(&self) -> bool {
        self.azimuth_deg.is_finite()
            && (-360.0..=360.0).contains(&self.azimuth_deg)
            && self.elevation_deg.is_finite()
            && (-90.0..=90.0).contains(&self.elevation_deg)
            && self.one_sigma_error_deg.is_finite()
            && self.one_sigma_error_deg >= 0.0
            && !self.calibration_ref.trim().is_empty()
    }
}

/// Raw camera-health evidence. Ratios/quality values are normalized to `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraHealthSample {
    pub frames_advancing: bool,
    pub timing_integrity: bool,
    pub calibration_current: bool,
    pub obstruction_ratio: f64,
    pub saturation_ratio: f64,
    pub focus_quality: f64,
    pub visibility_quality: f64,
}

impl CameraHealthSample {
    pub fn validate(&self) -> bool {
        [
            self.obstruction_ratio,
            self.saturation_ratio,
            self.focus_quality,
            self.visibility_quality,
        ]
        .into_iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }
}

/// Explicit deployment-reviewed thresholds. No defaults are provided so a
/// safety-critical deployment cannot inherit arbitrary image-quality limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraAssurancePolicy {
    pub max_obstruction_ratio_nominal: f64,
    pub max_saturation_ratio_nominal: f64,
    pub min_focus_quality_nominal: f64,
    pub min_visibility_quality_nominal: f64,
}

impl CameraAssurancePolicy {
    pub fn validate(&self) -> bool {
        [
            self.max_obstruction_ratio_nominal,
            self.max_saturation_ratio_nominal,
            self.min_focus_quality_nominal,
            self.min_visibility_quality_nominal,
        ]
        .into_iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }

    pub fn assess(&self, sample: &CameraHealthSample) -> CameraAssuranceReport {
        let mut issues = Vec::new();
        if !self.validate() || !sample.validate() {
            issues.push(CameraAssuranceIssue::InvalidEvidence);
            return CameraAssuranceReport {
                health: SensorHealth::Suspect,
                issues,
            };
        }
        if !sample.frames_advancing {
            issues.push(CameraAssuranceIssue::FramesNotAdvancing);
            return CameraAssuranceReport {
                health: SensorHealth::Unavailable,
                issues,
            };
        }
        if !sample.timing_integrity {
            issues.push(CameraAssuranceIssue::TimingIntegrityLost);
        }
        if !sample.calibration_current {
            issues.push(CameraAssuranceIssue::CalibrationInvalidOrStale);
        }
        if !sample.timing_integrity || !sample.calibration_current {
            return CameraAssuranceReport {
                health: SensorHealth::Suspect,
                issues,
            };
        }
        if sample.obstruction_ratio > self.max_obstruction_ratio_nominal {
            issues.push(CameraAssuranceIssue::Obstructed);
        }
        if sample.saturation_ratio > self.max_saturation_ratio_nominal {
            issues.push(CameraAssuranceIssue::Saturated);
        }
        if sample.focus_quality < self.min_focus_quality_nominal {
            issues.push(CameraAssuranceIssue::BlurredOrDefocused);
        }
        if sample.visibility_quality < self.min_visibility_quality_nominal {
            issues.push(CameraAssuranceIssue::EnvironmentalVisibilityDegraded);
        }
        CameraAssuranceReport {
            health: if issues.is_empty() {
                SensorHealth::Nominal
            } else {
                SensorHealth::Degraded
            },
            issues,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraAssuranceIssue {
    InvalidEvidence,
    FramesNotAdvancing,
    TimingIntegrityLost,
    CalibrationInvalidOrStale,
    Obstructed,
    Saturated,
    BlurredOrDefocused,
    EnvironmentalVisibilityDegraded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraAssuranceReport {
    pub health: SensorHealth,
    pub issues: Vec<CameraAssuranceIssue>,
}

/// Evidence maturity for a domain-awareness track. This is not an engagement
/// state and does not grant physical authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrackAssuranceLevel {
    Unassessed,
    Tentative,
    Persistent,
    Corroborated,
    IdentityEvidenceSupported,
}

/// Deployment-reviewed evidence thresholds for track assurance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackAssurancePolicy {
    pub min_persistent_observations: usize,
    pub min_persistent_span_ms: u64,
    pub min_corroborating_physical_sources: usize,
    pub min_corroborating_modalities: usize,
    pub min_identity_confidence: f64,
    pub min_identity_evidence_observations: usize,
}

impl TrackAssurancePolicy {
    pub fn validate(&self) -> bool {
        self.min_persistent_observations > 0
            && self.min_corroborating_physical_sources > 0
            && self.min_corroborating_modalities > 0
            && self.min_identity_evidence_observations > 0
            && self.min_identity_confidence.is_finite()
            && (0.0..=1.0).contains(&self.min_identity_confidence)
    }

    pub fn assess(
        &self,
        track: &Track,
        observations: &[ObservationEnvelope],
        now_ms: u64,
    ) -> TrackAssuranceReport {
        let mut reasons = Vec::new();
        if !self.validate() || !track.validate() {
            reasons.push(TrackAssuranceReason::InvalidEvidence);
            return TrackAssuranceReport::unassessed(reasons);
        }

        let by_id: HashMap<Uuid, &ObservationEnvelope> = observations
            .iter()
            .map(|observation| (observation.observation_id, observation))
            .collect();
        let accepted: Vec<&ObservationEnvelope> = track
            .observation_ids
            .iter()
            .filter_map(|id| by_id.get(id).copied())
            .filter(|observation| {
                observation.is_fresh_at(now_ms)
                    && observation.integrity != IntegrityStatus::Failed
                    && observation.sensor_health != SensorHealth::Unavailable
                    && observation.confidence > 0.0
            })
            .collect();

        if accepted.is_empty() {
            reasons.push(TrackAssuranceReason::NoFreshUsableEvidence);
            return TrackAssuranceReport::unassessed(reasons);
        }

        let independent_sources = accepted
            .iter()
            .map(|observation| observation.lineage.physical_fault_domain().to_string())
            .collect::<HashSet<_>>()
            .len();
        let modalities = accepted
            .iter()
            .map(|observation| observation.modality)
            .collect::<HashSet<_>>()
            .len();
        let minimum_time = accepted
            .iter()
            .map(|observation| observation.time.observed_at_ms)
            .min()
            .unwrap_or(now_ms);
        let maximum_time = accepted
            .iter()
            .map(|observation| observation.time.observed_at_ms)
            .max()
            .unwrap_or(minimum_time);
        let span_ms = maximum_time.saturating_sub(minimum_time);

        let mut level = TrackAssuranceLevel::Tentative;
        if accepted.len() >= self.min_persistent_observations
            && span_ms >= self.min_persistent_span_ms
        {
            level = TrackAssuranceLevel::Persistent;
        } else {
            reasons.push(TrackAssuranceReason::PersistenceNotEstablished);
        }

        if level >= TrackAssuranceLevel::Persistent
            && independent_sources >= self.min_corroborating_physical_sources
            && modalities >= self.min_corroborating_modalities
        {
            level = TrackAssuranceLevel::Corroborated;
        } else if level >= TrackAssuranceLevel::Persistent {
            reasons.push(TrackAssuranceReason::IndependentCorroborationMissing);
        }

        if level >= TrackAssuranceLevel::Corroborated {
            if matches!(
                track.epistemic_state,
                EpistemicState::ConflictingEvidence | EpistemicState::OutOfDistribution
            ) {
                reasons.push(TrackAssuranceReason::IdentityConflictOrOutOfDistribution);
            } else if track.epistemic_state == EpistemicState::Known {
                let accepted_ids = accepted
                    .iter()
                    .map(|observation| observation.observation_id)
                    .collect::<HashSet<_>>();
                let identity_supported = track.identity_hypotheses.iter().any(|hypothesis| {
                    hypothesis.validate()
                        && hypothesis.confidence >= self.min_identity_confidence
                        && hypothesis
                            .evidence_ids
                            .iter()
                            .filter(|id| accepted_ids.contains(id))
                            .collect::<HashSet<_>>()
                            .len()
                            >= self.min_identity_evidence_observations
                });
                if identity_supported {
                    level = TrackAssuranceLevel::IdentityEvidenceSupported;
                } else {
                    reasons.push(TrackAssuranceReason::IdentityEvidenceInsufficient);
                }
            } else {
                reasons.push(TrackAssuranceReason::IdentityNotEstablished);
            }
        }

        TrackAssuranceReport {
            level,
            fresh_usable_observations: accepted.len(),
            independent_physical_sources: independent_sources,
            independent_modalities: modalities,
            evidence_span_ms: span_ms,
            reasons,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackAssuranceReason {
    InvalidEvidence,
    NoFreshUsableEvidence,
    PersistenceNotEstablished,
    IndependentCorroborationMissing,
    IdentityConflictOrOutOfDistribution,
    IdentityNotEstablished,
    IdentityEvidenceInsufficient,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackAssuranceReport {
    pub level: TrackAssuranceLevel,
    pub fresh_usable_observations: usize,
    pub independent_physical_sources: usize,
    pub independent_modalities: usize,
    pub evidence_span_ms: u64,
    pub reasons: Vec<TrackAssuranceReason>,
}

impl TrackAssuranceReport {
    fn unassessed(reasons: Vec<TrackAssuranceReason>) -> Self {
        Self {
            level: TrackAssuranceLevel::Unassessed,
            fresh_usable_observations: 0,
            independent_physical_sources: 0,
            independent_modalities: 0,
            evidence_span_ms: 0,
            reasons,
        }
    }

    /// Explicit reminder at the API boundary: assurance is not actuation authority.
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualBridgeError {
    InvalidGeometry,
    InvalidFrameRate,
    TrackOutsideGrid,
    InvalidTrack,
    InvalidContext,
    InvalidConfidence,
    InvalidCalibration,
    InvalidObservation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_domain_awareness::{IdentityHypothesis, RiskAssessment};

    fn context(modality: Modality, physical: &str) -> VisionObservationContext {
        VisionObservationContext {
            source_id: format!("source-{physical}"),
            physical_source_id: physical.to_string(),
            processor_id: "vision-manifold-v1".to_string(),
            network_path: "local-memory".to_string(),
            clock_domain: "camera-clock".to_string(),
            clock_source: "monotonic-camera".to_string(),
            coordinate_frame: "camera-optical".to_string(),
            domain: Domain::Air,
            modality,
            integrity: IntegrityStatus::Verified,
            maximum_valid_age_ms: 1_000,
            clock_uncertainty_ms: 5,
        }
    }

    fn visual(track_id: u64, frame: u64, length: u64) -> VisualTrackEvidence {
        VisualTrackEvidence {
            visual_track_id: track_id,
            frame_index: frame,
            u_norm: 0.5,
            v_norm: 0.5,
            du_norm_per_s: 0.01,
            dv_norm_per_s: -0.02,
            track_length_frames: length,
        }
    }

    fn policy() -> TrackAssurancePolicy {
        TrackAssurancePolicy {
            min_persistent_observations: 3,
            min_persistent_span_ms: 200,
            min_corroborating_physical_sources: 2,
            min_corroborating_modalities: 2,
            min_identity_confidence: 0.80,
            min_identity_evidence_observations: 2,
        }
    }

    #[test]
    fn image_track_does_not_claim_world_geometry() {
        let evidence = visual(7, 10, 5);
        assert!(evidence.validate());
        let observation = context(Modality::ElectroOptical, "camera-a")
            .presence_observation(
                &evidence,
                1_000,
                1_010,
                SensorHealth::Nominal,
                0.9,
                vec!["frame-digest:abc".to_string()],
            )
            .unwrap();
        assert!(matches!(observation.measurement, Measurement::Scalar { .. }));
        assert_eq!(observation.modality, Modality::ElectroOptical);
    }

    #[test]
    fn calibrated_bearing_requires_calibration_evidence() {
        let evidence = visual(7, 10, 5);
        let invalid = BearingProjection {
            azimuth_deg: 12.0,
            elevation_deg: 4.0,
            one_sigma_error_deg: 0.5,
            calibration_ref: String::new(),
        };
        assert_eq!(
            context(Modality::ElectroOptical, "camera-a")
                .bearing_observation(
                    &evidence,
                    &invalid,
                    1_000,
                    1_010,
                    SensorHealth::Nominal,
                    0.9,
                    vec![],
                )
                .unwrap_err(),
            VisualBridgeError::InvalidCalibration
        );
    }

    #[test]
    fn degraded_camera_caps_observation_confidence() {
        let evidence = visual(7, 10, 5);
        let observation = context(Modality::Infrared, "thermal-a")
            .presence_observation(
                &evidence,
                1_000,
                1_010,
                SensorHealth::Degraded,
                0.95,
                vec![],
            )
            .unwrap();
        assert_eq!(observation.confidence, SensorHealth::Degraded.trust_cap());
    }

    #[test]
    fn camera_timing_or_calibration_loss_is_suspect() {
        let assurance = CameraAssurancePolicy {
            max_obstruction_ratio_nominal: 0.2,
            max_saturation_ratio_nominal: 0.2,
            min_focus_quality_nominal: 0.7,
            min_visibility_quality_nominal: 0.6,
        };
        let sample = CameraHealthSample {
            frames_advancing: true,
            timing_integrity: false,
            calibration_current: true,
            obstruction_ratio: 0.0,
            saturation_ratio: 0.0,
            focus_quality: 1.0,
            visibility_quality: 1.0,
        };
        let report = assurance.assess(&sample);
        assert_eq!(report.health, SensorHealth::Suspect);
    }

    #[test]
    fn stopped_frames_are_unavailable() {
        let assurance = CameraAssurancePolicy {
            max_obstruction_ratio_nominal: 0.2,
            max_saturation_ratio_nominal: 0.2,
            min_focus_quality_nominal: 0.7,
            min_visibility_quality_nominal: 0.6,
        };
        let sample = CameraHealthSample {
            frames_advancing: false,
            timing_integrity: true,
            calibration_current: true,
            obstruction_ratio: 0.0,
            saturation_ratio: 0.0,
            focus_quality: 1.0,
            visibility_quality: 1.0,
        };
        assert_eq!(assurance.assess(&sample).health, SensorHealth::Unavailable);
    }

    #[test]
    fn one_camera_cannot_self_corroborate() {
        let mut track = Track::new(Domain::Air, 1_000);
        let ctx = context(Modality::ElectroOptical, "camera-a");
        let observations = [1_000_u64, 1_150, 1_300]
            .into_iter()
            .enumerate()
            .map(|(index, time)| {
                ctx.presence_observation(
                    &visual(7, index as u64 + 1, index as u64 + 1),
                    time,
                    time + 5,
                    SensorHealth::Nominal,
                    0.9,
                    vec![],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        for observation in &observations {
            assert!(track.attach_observation(observation, observation.time.received_at_ms));
        }
        let report = policy().assess(&track, &observations, 1_350);
        assert_eq!(report.level, TrackAssuranceLevel::Persistent);
        assert_eq!(report.independent_physical_sources, 1);
    }

    #[test]
    fn independent_eo_and_ir_can_corroborate_physical_existence() {
        let mut track = Track::new(Domain::Air, 1_000);
        let eo = context(Modality::ElectroOptical, "camera-a");
        let ir = context(Modality::Infrared, "thermal-b");
        let observations = vec![
            eo.presence_observation(
                &visual(7, 1, 1), 1_000, 1_005, SensorHealth::Nominal, 0.9, vec![],
            )
            .unwrap(),
            eo.presence_observation(
                &visual(7, 2, 2), 1_150, 1_155, SensorHealth::Nominal, 0.9, vec![],
            )
            .unwrap(),
            ir.presence_observation(
                &visual(3, 3, 3), 1_300, 1_305, SensorHealth::Nominal, 0.8, vec![],
            )
            .unwrap(),
        ];
        for observation in &observations {
            assert!(track.attach_observation(observation, observation.time.received_at_ms));
        }
        let report = policy().assess(&track, &observations, 1_350);
        assert_eq!(report.level, TrackAssuranceLevel::Corroborated);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn corroborated_track_does_not_imply_identity_or_intent() {
        let mut track = Track::new(Domain::Air, 1_000);
        let eo = context(Modality::ElectroOptical, "camera-a");
        let ir = context(Modality::Infrared, "thermal-b");
        let observations = vec![
            eo.presence_observation(
                &visual(7, 1, 1), 1_000, 1_005, SensorHealth::Nominal, 0.9, vec![],
            )
            .unwrap(),
            eo.presence_observation(
                &visual(7, 2, 2), 1_150, 1_155, SensorHealth::Nominal, 0.9, vec![],
            )
            .unwrap(),
            ir.presence_observation(
                &visual(3, 3, 3), 1_300, 1_305, SensorHealth::Nominal, 0.8, vec![],
            )
            .unwrap(),
        ];
        for observation in &observations {
            track.attach_observation(observation, observation.time.received_at_ms);
        }
        let report = policy().assess(&track, &observations, 1_350);
        assert_eq!(report.level, TrackAssuranceLevel::Corroborated);
        assert!(track.identity_hypotheses.is_empty());
        assert_ne!(track.epistemic_state, EpistemicState::Known);
    }

    #[test]
    fn identity_support_requires_fresh_accepted_evidence() {
        let mut track = Track::new(Domain::Air, 1_000);
        let eo = context(Modality::ElectroOptical, "camera-a");
        let ir = context(Modality::Infrared, "thermal-b");
        let observations = vec![
            eo.presence_observation(
                &visual(7, 1, 1), 1_000, 1_005, SensorHealth::Nominal, 0.9, vec![],
            )
            .unwrap(),
            eo.presence_observation(
                &visual(7, 2, 2), 1_150, 1_155, SensorHealth::Nominal, 0.9, vec![],
            )
            .unwrap(),
            ir.presence_observation(
                &visual(3, 3, 3), 1_300, 1_305, SensorHealth::Nominal, 0.8, vec![],
            )
            .unwrap(),
        ];
        for observation in &observations {
            track.attach_observation(observation, observation.time.received_at_ms);
        }
        assert!(track.set_identity_hypotheses(vec![IdentityHypothesis {
            label: "known-aircraft-class".to_string(),
            confidence: 0.9,
            evidence_ids: vec![observations[0].observation_id, observations[2].observation_id],
            evidence_refs: vec!["classifier-release:signed-v1".to_string()],
        }]));
        assert_eq!(track.epistemic_state, EpistemicState::Known);
        let report = policy().assess(&track, &observations, 1_350);
        assert_eq!(report.level, TrackAssuranceLevel::IdentityEvidenceSupported);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn risk_stays_separate_from_track_assurance() {
        let report = TrackAssuranceReport {
            level: TrackAssuranceLevel::IdentityEvidenceSupported,
            fresh_usable_observations: 5,
            independent_physical_sources: 3,
            independent_modalities: 2,
            evidence_span_ms: 500,
            reasons: vec![],
        };
        let risk = RiskAssessment {
            track_id: Uuid::new_v4(),
            harmful_outcome_probability: 0.9,
            consequence_severity: 0.9,
            model_confidence: 0.9,
            epistemic_state: EpistemicState::Known,
            evidence_refs: vec!["twin:test".to_string()],
        };
        assert!(risk.validate());
        assert!(risk.review_priority() > 0.0);
        assert!(!report.grants_physical_authority());
    }
}
