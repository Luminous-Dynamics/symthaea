// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Calibrated camera-geometry adapter for domain-awareness vision evidence.
//!
//! This bridge converts normalized visual-track coordinates into camera-frame
//! bearing evidence only after an explicit calibration succeeds. It does not
//! infer range, world position, identity, intent, or physical authority.

#![deny(unsafe_code)]

use symthaea_camera_geometry::{CameraGeometryError, PinholeCalibration};
use symthaea_domain_awareness::{ObservationEnvelope, SensorHealth};
use symthaea_domain_awareness_vision::{
    BearingProjection, VisualBridgeError, VisualTrackEvidence, VisionObservationContext,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibratedVisionBridgeError {
    CameraGeometry(CameraGeometryError),
    CoordinateFrameMismatch,
    VisionBridge(VisualBridgeError),
}

impl From<CameraGeometryError> for CalibratedVisionBridgeError {
    fn from(value: CameraGeometryError) -> Self {
        Self::CameraGeometry(value)
    }
}

impl From<VisualBridgeError> for CalibratedVisionBridgeError {
    fn from(value: VisualBridgeError) -> Self {
        Self::VisionBridge(value)
    }
}

/// Convert a normalized visual track into calibrated camera-frame bearing evidence.
///
/// The calibration consumes already-undistorted image coordinates. All calibration
/// provenance is propagated into the resulting `ObservationEnvelope`. The exact
/// optical frame bound into the calibration must match the observation context.
#[allow(clippy::too_many_arguments)]
pub fn calibrated_bearing_observation(
    context: &VisionObservationContext,
    visual: &VisualTrackEvidence,
    calibration: &PinholeCalibration,
    one_sigma_localization_error_px: f64,
    observed_at_ms: u64,
    received_at_ms: u64,
    health: SensorHealth,
    producer_confidence: f64,
    mut evidence_refs: Vec<String>,
) -> Result<ObservationEnvelope, CalibratedVisionBridgeError> {
    let bearing = calibration.project_normalized(
        visual.u_norm,
        visual.v_norm,
        one_sigma_localization_error_px,
        observed_at_ms,
    )?;

    if context.coordinate_frame != bearing.camera_frame_id {
        return Err(CalibratedVisionBridgeError::CoordinateFrameMismatch);
    }

    evidence_refs.push(format!("camera-calibration-id:{}", bearing.calibration_id));
    evidence_refs.extend(bearing.evidence_refs.iter().cloned());

    let projection = BearingProjection {
        azimuth_deg: bearing.azimuth_deg,
        elevation_deg: bearing.elevation_deg,
        one_sigma_error_deg: bearing.one_sigma_error_deg,
        calibration_ref: bearing.calibration_ref,
    };

    context
        .bearing_observation(
            visual,
            &projection,
            observed_at_ms,
            received_at_ms,
            health,
            producer_confidence,
            evidence_refs,
        )
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_domain_awareness::{
        Domain, IntegrityStatus, Measurement, Modality, SensorHealth,
    };

    fn calibration() -> PinholeCalibration {
        PinholeCalibration {
            schema_version: "1".into(),
            calibration_id: "camera-1-cal".into(),
            camera_frame_id: "camera-1-optical".into(),
            calibration_ref: "calibration:camera-1:blake3:test".into(),
            image_width_px: 1920,
            image_height_px: 1080,
            fx_px: 1000.0,
            fy_px: 1000.0,
            cx_px: 960.0,
            cy_px: 540.0,
            one_sigma_calibration_error_px: 0.5,
            maximum_normalized_ray_radius: 1.2,
            valid_from_ms: 100,
            valid_until_ms: 10_000,
            evidence_refs: vec!["calibration-report:camera-1".into()],
        }
    }

    fn context() -> VisionObservationContext {
        VisionObservationContext {
            source_id: "camera-1-visible".into(),
            physical_source_id: "camera-1".into(),
            processor_id: "vision-manifold-v1".into(),
            network_path: "local-capture".into(),
            clock_domain: "camera-ptp-domain".into(),
            clock_source: "ptp-camera-1".into(),
            coordinate_frame: "camera-1-optical".into(),
            domain: Domain::Air,
            modality: Modality::ElectroOptical,
            integrity: IntegrityStatus::Verified,
            maximum_valid_age_ms: 500,
            clock_uncertainty_ms: 2,
        }
    }

    fn visual(u: f64, v: f64) -> VisualTrackEvidence {
        VisualTrackEvidence {
            visual_track_id: 7,
            frame_index: 42,
            u_norm: u,
            v_norm: v,
            du_norm_per_s: 0.0,
            dv_norm_per_s: 0.0,
            track_length_frames: 12,
        }
    }

    #[test]
    fn calibrated_track_becomes_bearing_not_world_position() {
        let observation = calibrated_bearing_observation(
            &context(),
            &visual(0.5, 0.5),
            &calibration(),
            0.5,
            1_000,
            1_005,
            SensorHealth::Nominal,
            0.9,
            vec!["frame:42".into()],
        )
        .unwrap();

        match observation.measurement {
            Measurement::Bearing {
                azimuth_deg,
                elevation_deg,
            } => {
                assert!(azimuth_deg.abs() < 1e-12);
                assert!(elevation_deg.unwrap().abs() < 1e-12);
            }
            other => panic!("expected bearing evidence, got {other:?}"),
        }
        assert!(observation.uncertainty.position_sigma_m.is_none());
        assert!(observation.uncertainty.velocity_sigma_mps.is_none());
        assert!(observation.uncertainty.bearing_sigma_deg.is_some());
        assert!(observation
            .evidence_refs
            .iter()
            .any(|value| value == "calibration-report:camera-1"));
    }

    #[test]
    fn stale_calibration_cannot_publish_bearing() {
        let error = calibrated_bearing_observation(
            &context(),
            &visual(0.5, 0.5),
            &calibration(),
            0.5,
            20_000,
            20_005,
            SensorHealth::Nominal,
            0.9,
            vec!["frame:42".into()],
        )
        .unwrap_err();
        assert_eq!(
            error,
            CalibratedVisionBridgeError::CameraGeometry(
                CameraGeometryError::CalibrationNotActive
            )
        );
    }

    #[test]
    fn frame_mismatch_cannot_publish_bearing() {
        let mut wrong_context = context();
        wrong_context.coordinate_frame = "another-camera-frame".into();
        let error = calibrated_bearing_observation(
            &wrong_context,
            &visual(0.5, 0.5),
            &calibration(),
            0.5,
            1_000,
            1_005,
            SensorHealth::Nominal,
            0.9,
            vec!["frame:42".into()],
        )
        .unwrap_err();
        assert_eq!(error, CalibratedVisionBridgeError::CoordinateFrameMismatch);
    }

    #[test]
    fn degraded_camera_caps_confidence_after_geometry_succeeds() {
        let observation = calibrated_bearing_observation(
            &context(),
            &visual(0.5, 0.5),
            &calibration(),
            0.5,
            1_000,
            1_005,
            SensorHealth::Degraded,
            0.95,
            vec!["frame:42".into()],
        )
        .unwrap();
        assert_eq!(observation.confidence, SensorHealth::Degraded.trust_cap());
    }

    #[test]
    fn outside_calibrated_ray_envelope_fails_closed() {
        let mut calibration = calibration();
        calibration.maximum_normalized_ray_radius = 0.05;
        let error = calibrated_bearing_observation(
            &context(),
            &visual(0.75, 0.5),
            &calibration,
            0.5,
            1_000,
            1_005,
            SensorHealth::Nominal,
            0.9,
            vec!["frame:42".into()],
        )
        .unwrap_err();
        assert_eq!(
            error,
            CalibratedVisionBridgeError::CameraGeometry(
                CameraGeometryError::OutsideValidatedRayEnvelope
            )
        );
    }
}
