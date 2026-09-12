// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative calibrated camera geometry.
//!
//! This crate converts already-undistorted normalized image coordinates into
//! camera-frame bearing evidence only. It deliberately does not infer range,
//! world position, identity, intent, or physical authority.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PinholeCalibration {
    pub schema_version: String,
    pub calibration_id: String,
    /// Durable reference to the calibration manifest/evidence object.
    pub calibration_ref: String,
    pub image_width_px: u32,
    pub image_height_px: u32,
    pub fx_px: f64,
    pub fy_px: f64,
    pub cx_px: f64,
    pub cy_px: f64,
    /// One-sigma calibration/reprojection error floor in pixels.
    pub one_sigma_calibration_error_px: f64,
    /// Maximum validated normalized ray radius sqrt(x_n^2 + y_n^2).
    pub maximum_normalized_ray_radius: f64,
    pub valid_from_ms: u64,
    pub valid_until_ms: u64,
    /// Additional provenance for calibration images, solver reports, fixtures, etc.
    pub evidence_refs: Vec<String>,
}

impl PinholeCalibration {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.calibration_id.trim().is_empty()
            && !self.calibration_ref.trim().is_empty()
            && self.image_width_px > 0
            && self.image_height_px > 0
            && self.fx_px.is_finite()
            && self.fx_px > 0.0
            && self.fy_px.is_finite()
            && self.fy_px > 0.0
            && self.cx_px.is_finite()
            && self.cy_px.is_finite()
            && self.cx_px >= 0.0
            && self.cx_px <= self.image_width_px as f64
            && self.cy_px >= 0.0
            && self.cy_px <= self.image_height_px as f64
            && self.one_sigma_calibration_error_px.is_finite()
            && self.one_sigma_calibration_error_px >= 0.0
            && self.maximum_normalized_ray_radius.is_finite()
            && self.maximum_normalized_ray_radius > 0.0
            && self.valid_from_ms <= self.valid_until_ms
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }

    pub fn active_at(&self, timestamp_ms: u64) -> bool {
        self.validate() && (self.valid_from_ms..=self.valid_until_ms).contains(&timestamp_ms)
    }

    /// Project an already-undistorted normalized image coordinate into a
    /// camera-frame bearing with conservative angular uncertainty.
    ///
    /// `u_norm` and `v_norm` use image-boundary coordinates in `[0, 1]`.
    /// `(0,0)` is the upper-left image boundary and positive `v` points down.
    /// Positive azimuth points camera-right; positive elevation points camera-up.
    pub fn project_normalized(
        &self,
        u_norm: f64,
        v_norm: f64,
        one_sigma_localization_error_px: f64,
        observed_at_ms: u64,
    ) -> Result<CalibratedBearing, CameraGeometryError> {
        if !self.validate() {
            return Err(CameraGeometryError::InvalidCalibration);
        }
        if !self.active_at(observed_at_ms) {
            return Err(CameraGeometryError::CalibrationNotActive);
        }
        if !u_norm.is_finite()
            || !v_norm.is_finite()
            || !(0.0..=1.0).contains(&u_norm)
            || !(0.0..=1.0).contains(&v_norm)
        {
            return Err(CameraGeometryError::ImagePointOutsideCalibratedImage);
        }
        if !one_sigma_localization_error_px.is_finite()
            || one_sigma_localization_error_px < 0.0
        {
            return Err(CameraGeometryError::InvalidLocalizationUncertainty);
        }

        let x_px = u_norm * self.image_width_px as f64;
        let y_px = v_norm * self.image_height_px as f64;
        let x_n = (x_px - self.cx_px) / self.fx_px;
        let y_n = (y_px - self.cy_px) / self.fy_px;
        let ray_radius = x_n.hypot(y_n);
        if !ray_radius.is_finite() || ray_radius > self.maximum_normalized_ray_radius {
            return Err(CameraGeometryError::OutsideValidatedRayEnvelope);
        }

        let azimuth_rad = x_n.atan2(1.0);
        let elevation_rad = (-y_n).atan2((1.0 + x_n * x_n).sqrt());

        let combined_pixel_sigma = one_sigma_localization_error_px
            .hypot(self.one_sigma_calibration_error_px);
        let minimum_focal_px = self.fx_px.min(self.fy_px);
        let angular_sigma_rad = combined_pixel_sigma.atan2(minimum_focal_px);

        let bearing = CalibratedBearing {
            azimuth_deg: azimuth_rad.to_degrees(),
            elevation_deg: elevation_rad.to_degrees(),
            one_sigma_error_deg: angular_sigma_rad.to_degrees(),
            calibration_id: self.calibration_id.clone(),
            calibration_ref: self.calibration_ref.clone(),
            evidence_refs: self.evidence_refs.clone(),
            observed_at_ms,
        };
        bearing
            .validate()
            .then_some(bearing)
            .ok_or(CameraGeometryError::InvalidProjection)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibratedBearing {
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub one_sigma_error_deg: f64,
    pub calibration_id: String,
    pub calibration_ref: String,
    pub evidence_refs: Vec<String>,
    pub observed_at_ms: u64,
}

impl CalibratedBearing {
    pub fn validate(&self) -> bool {
        self.azimuth_deg.is_finite()
            && (-180.0..=180.0).contains(&self.azimuth_deg)
            && self.elevation_deg.is_finite()
            && (-90.0..=90.0).contains(&self.elevation_deg)
            && self.one_sigma_error_deg.is_finite()
            && self.one_sigma_error_deg >= 0.0
            && !self.calibration_id.trim().is_empty()
            && !self.calibration_ref.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }

    /// Bearing evidence alone cannot establish range, identity, intent, or authority.
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraGeometryError {
    InvalidCalibration,
    CalibrationNotActive,
    ImagePointOutsideCalibratedImage,
    InvalidLocalizationUncertainty,
    OutsideValidatedRayEnvelope,
    InvalidProjection,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration() -> PinholeCalibration {
        PinholeCalibration {
            schema_version: "1".into(),
            calibration_id: "cam-a-2026-09".into(),
            calibration_ref: "calibration:cam-a:blake3:example".into(),
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
            evidence_refs: vec!["calibration-report:cam-a".into()],
        }
    }

    #[test]
    fn principal_point_projects_to_optical_axis() {
        let bearing = calibration()
            .project_normalized(0.5, 0.5, 0.5, 1_000)
            .unwrap();
        assert!(bearing.azimuth_deg.abs() < 1e-12);
        assert!(bearing.elevation_deg.abs() < 1e-12);
        assert!(bearing.one_sigma_error_deg > 0.0);
        assert!(!bearing.grants_physical_authority());
    }

    #[test]
    fn right_and_up_have_expected_signs() {
        let calibration = calibration();
        let right = calibration
            .project_normalized(0.75, 0.5, 0.25, 1_000)
            .unwrap();
        let up = calibration
            .project_normalized(0.5, 0.25, 0.25, 1_000)
            .unwrap();
        assert!(right.azimuth_deg > 0.0);
        assert!(up.elevation_deg > 0.0);
    }

    #[test]
    fn stale_calibration_fails_closed() {
        assert_eq!(
            calibration().project_normalized(0.5, 0.5, 0.5, 20_000),
            Err(CameraGeometryError::CalibrationNotActive)
        );
    }

    #[test]
    fn image_point_outside_unit_square_is_rejected() {
        assert_eq!(
            calibration().project_normalized(1.1, 0.5, 0.5, 1_000),
            Err(CameraGeometryError::ImagePointOutsideCalibratedImage)
        );
    }

    #[test]
    fn validated_ray_envelope_is_enforced() {
        let mut calibration = calibration();
        calibration.maximum_normalized_ray_radius = 0.1;
        assert_eq!(
            calibration.project_normalized(0.75, 0.5, 0.5, 1_000),
            Err(CameraGeometryError::OutsideValidatedRayEnvelope)
        );
    }

    #[test]
    fn larger_localization_error_never_reduces_angular_uncertainty() {
        let calibration = calibration();
        let precise = calibration
            .project_normalized(0.5, 0.5, 0.1, 1_000)
            .unwrap();
        let noisy = calibration
            .project_normalized(0.5, 0.5, 5.0, 1_000)
            .unwrap();
        assert!(noisy.one_sigma_error_deg >= precise.one_sigma_error_deg);
    }

    #[test]
    fn missing_calibration_evidence_is_invalid() {
        let mut calibration = calibration();
        calibration.evidence_refs.clear();
        assert!(!calibration.validate());
    }
}
