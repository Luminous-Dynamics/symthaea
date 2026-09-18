// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed spatial evidence for visual cognition.
//!
//! This module prevents image, patch-grid, relative-depth, camera-bearing, and future metric/world
//! coordinates from collapsing into interchangeable numeric tuples. The initial contract stops at
//! calibrated camera-frame bearing: it deliberately provides no constructor for metric range,
//! camera-frame XYZ, or world-frame XYZ.

use serde::Serialize;
use std::fmt;

use crate::epistemic::{VisualEvidence, VisualObservationRef, VisualOrigin, VisualStreamRef};

/// Semantic kind of an image plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImagePlaneKind {
    /// Raw sensor plane. Lens distortion may still be present.
    RawSensor,
    /// A separately qualified process produced an undistorted/rectified plane.
    UndistortedRectified,
}

/// Stable identity for one image-coordinate plane definition.
///
/// `plane_namespace` is source-owned and must change when rectification/cropping/resampling
/// semantics change materially. Equality therefore means coordinate comparability, not merely
/// equal raster dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct ImagePlaneRef {
    plane_namespace: u64,
    kind: ImagePlaneKind,
}

impl ImagePlaneRef {
    pub fn new(plane_namespace: u64, kind: ImagePlaneKind) -> Result<Self, SpatialEvidenceError> {
        if plane_namespace == 0 {
            return Err(SpatialEvidenceError::MissingImagePlaneNamespace);
        }
        Ok(Self {
            plane_namespace,
            kind,
        })
    }

    pub const fn plane_namespace(self) -> u64 {
        self.plane_namespace
    }

    pub const fn kind(self) -> ImagePlaneKind {
        self.kind
    }
}

/// Exact image raster and coordinate plane for one concrete observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct ImageFrameRef {
    observation: VisualObservationRef,
    width_px: u32,
    height_px: u32,
    plane: ImagePlaneRef,
}

impl ImageFrameRef {
    pub fn new(
        observation: VisualObservationRef,
        width_px: u32,
        height_px: u32,
        plane: ImagePlaneRef,
    ) -> Result<Self, SpatialEvidenceError> {
        if width_px == 0 || height_px == 0 {
            return Err(SpatialEvidenceError::InvalidImageDimensions);
        }
        Ok(Self {
            observation,
            width_px,
            height_px,
            plane,
        })
    }

    pub const fn observation(self) -> VisualObservationRef {
        self.observation
    }

    pub const fn width_px(self) -> u32 {
        self.width_px
    }

    pub const fn height_px(self) -> u32 {
        self.height_px
    }

    pub const fn plane(self) -> ImagePlaneRef {
        self.plane
    }
}

/// Point-localization evidence in an exact image plane.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ImagePointEvidence {
    frame: ImageFrameRef,
    x_px: f32,
    y_px: f32,
    one_sigma_x_px: f32,
    one_sigma_y_px: f32,
    evidence: VisualEvidence,
}

impl ImagePointEvidence {
    pub fn new(
        frame: ImageFrameRef,
        x_px: f32,
        y_px: f32,
        one_sigma_x_px: f32,
        one_sigma_y_px: f32,
        evidence: VisualEvidence,
    ) -> Result<Self, SpatialEvidenceError> {
        if !x_px.is_finite()
            || !y_px.is_finite()
            || x_px < 0.0
            || y_px < 0.0
            || x_px >= frame.width_px as f32
            || y_px >= frame.height_px as f32
        {
            return Err(SpatialEvidenceError::ImagePointOutOfBounds);
        }
        if !one_sigma_x_px.is_finite()
            || !one_sigma_y_px.is_finite()
            || one_sigma_x_px < 0.0
            || one_sigma_y_px < 0.0
        {
            return Err(SpatialEvidenceError::InvalidSpatialUncertainty);
        }
        validate_derived_support(&evidence, frame.observation)?;
        Ok(Self {
            frame,
            x_px,
            y_px,
            one_sigma_x_px,
            one_sigma_y_px,
            evidence,
        })
    }

    pub const fn frame(&self) -> ImageFrameRef {
        self.frame
    }

    pub const fn x_px(&self) -> f32 {
        self.x_px
    }

    pub const fn y_px(&self) -> f32 {
        self.y_px
    }

    pub const fn one_sigma_px(&self) -> [f32; 2] {
        [self.one_sigma_x_px, self.one_sigma_y_px]
    }

    pub fn evidence(&self) -> &VisualEvidence {
        &self.evidence
    }

    /// Euclidean pixel distance is defined only inside the exact same observation/plane/raster.
    pub fn distance_px(&self, other: &Self) -> Result<f32, SpatialEvidenceError> {
        if self.frame != other.frame {
            return Err(SpatialEvidenceError::IncompatibleImageFrames);
        }
        Ok((self.x_px - other.x_px).hypot(self.y_px - other.y_px))
    }
}

/// Exact patch-grid cell belonging to an exact image frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct PatchGridCellRef {
    frame: ImageFrameRef,
    patch_size_px: u32,
    grid_rows: u32,
    grid_cols: u32,
    row: u32,
    col: u32,
}

impl PatchGridCellRef {
    pub fn new(
        frame: ImageFrameRef,
        patch_size_px: u32,
        row: u32,
        col: u32,
    ) -> Result<Self, SpatialEvidenceError> {
        if patch_size_px == 0 {
            return Err(SpatialEvidenceError::InvalidPatchSize);
        }
        let grid_cols = frame.width_px.div_ceil(patch_size_px);
        let grid_rows = frame.height_px.div_ceil(patch_size_px);
        if row >= grid_rows || col >= grid_cols {
            return Err(SpatialEvidenceError::PatchCellOutOfBounds);
        }
        Ok(Self {
            frame,
            patch_size_px,
            grid_rows,
            grid_cols,
            row,
            col,
        })
    }

    pub const fn frame(self) -> ImageFrameRef {
        self.frame
    }

    pub const fn patch_size_px(self) -> u32 {
        self.patch_size_px
    }

    pub const fn grid_rows(self) -> u32 {
        self.grid_rows
    }

    pub const fn grid_cols(self) -> u32 {
        self.grid_cols
    }

    pub const fn row(self) -> u32 {
        self.row
    }

    pub const fn col(self) -> u32 {
        self.col
    }
}

/// How a normalized relative-depth value was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RelativeDepthSource {
    /// Current patch stereo matching: larger disparity maps toward the near end of the normalized
    /// scale. This still does not establish metric range.
    StereoDisparity,
    /// A monocular model/provider produced relative depth only.
    MonocularRelative,
    /// Another provider with explicitly non-metric `0 = near, 1 = far` semantics.
    ProviderRelative,
}

/// Per-patch relative depth. `near_zero_far_one` is dimensionless and can never be read as meters
/// through this API.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PatchRelativeDepthEvidence {
    patch: PatchGridCellRef,
    near_zero_far_one: f32,
    confidence: f32,
    source: RelativeDepthSource,
    evidence: VisualEvidence,
}

impl PatchRelativeDepthEvidence {
    pub fn new(
        patch: PatchGridCellRef,
        near_zero_far_one: f32,
        confidence: f32,
        source: RelativeDepthSource,
        evidence: VisualEvidence,
    ) -> Result<Self, SpatialEvidenceError> {
        if !near_zero_far_one.is_finite() || !(0.0..=1.0).contains(&near_zero_far_one) {
            return Err(SpatialEvidenceError::InvalidRelativeDepth);
        }
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(SpatialEvidenceError::InvalidSpatialConfidence);
        }
        validate_derived_support(&evidence, patch.frame.observation)?;
        Ok(Self {
            patch,
            near_zero_far_one,
            confidence,
            source,
            evidence,
        })
    }

    pub const fn patch(&self) -> PatchGridCellRef {
        self.patch
    }

    pub const fn near_zero_far_one(&self) -> f32 {
        self.near_zero_far_one
    }

    pub const fn confidence(&self) -> f32 {
        self.confidence
    }

    pub const fn source(&self) -> RelativeDepthSource {
        self.source
    }

    pub fn evidence(&self) -> &VisualEvidence {
        &self.evidence
    }
}

/// Stable optical-frame identity. A frame name/namespace is explicitly scoped to a capture stream
/// so equal numeric frame namespaces from different cameras do not compare equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct CameraOpticalFrameRef {
    stream: VisualStreamRef,
    frame_namespace: u64,
}

impl CameraOpticalFrameRef {
    pub fn new(
        stream: VisualStreamRef,
        frame_namespace: u64,
    ) -> Result<Self, SpatialEvidenceError> {
        if frame_namespace == 0 {
            return Err(SpatialEvidenceError::MissingCameraFrameNamespace);
        }
        Ok(Self {
            stream,
            frame_namespace,
        })
    }

    pub const fn stream(self) -> VisualStreamRef {
        self.stream
    }

    pub const fn frame_namespace(self) -> u64 {
        self.frame_namespace
    }
}

/// Calibrated camera-frame bearing evidence.
///
/// Bearing is angular evidence only. This type deliberately contains no range/XYZ fields and has
/// no conversion to a metric or world-space point.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CameraBearingEvidence {
    observation: VisualObservationRef,
    camera_frame: CameraOpticalFrameRef,
    calibration_namespace: u64,
    azimuth_rad: f64,
    elevation_rad: f64,
    one_sigma_rad: f64,
    evidence: VisualEvidence,
}

impl CameraBearingEvidence {
    pub fn new(
        observation: VisualObservationRef,
        camera_frame: CameraOpticalFrameRef,
        calibration_namespace: u64,
        azimuth_rad: f64,
        elevation_rad: f64,
        one_sigma_rad: f64,
        evidence: VisualEvidence,
    ) -> Result<Self, SpatialEvidenceError> {
        if observation.stream() != camera_frame.stream {
            return Err(SpatialEvidenceError::CameraStreamMismatch);
        }
        if calibration_namespace == 0 {
            return Err(SpatialEvidenceError::MissingCalibrationNamespace);
        }
        if !azimuth_rad.is_finite()
            || !elevation_rad.is_finite()
            || !one_sigma_rad.is_finite()
            || !(-std::f64::consts::PI..=std::f64::consts::PI).contains(&azimuth_rad)
            || !(-std::f64::consts::FRAC_PI_2..=std::f64::consts::FRAC_PI_2)
                .contains(&elevation_rad)
            || one_sigma_rad < 0.0
        {
            return Err(SpatialEvidenceError::InvalidBearing);
        }
        validate_derived_support(&evidence, observation)?;
        Ok(Self {
            observation,
            camera_frame,
            calibration_namespace,
            azimuth_rad,
            elevation_rad,
            one_sigma_rad,
            evidence,
        })
    }

    pub const fn observation(&self) -> VisualObservationRef {
        self.observation
    }

    pub const fn camera_frame(&self) -> CameraOpticalFrameRef {
        self.camera_frame
    }

    pub const fn calibration_namespace(&self) -> u64 {
        self.calibration_namespace
    }

    pub const fn azimuth_rad(&self) -> f64 {
        self.azimuth_rad
    }

    pub const fn elevation_rad(&self) -> f64 {
        self.elevation_rad
    }

    pub const fn one_sigma_rad(&self) -> f64 {
        self.one_sigma_rad
    }

    pub fn evidence(&self) -> &VisualEvidence {
        &self.evidence
    }

    /// This tranche intentionally cannot establish metric range.
    pub const fn has_metric_range(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialEvidenceError {
    MissingImagePlaneNamespace,
    InvalidImageDimensions,
    ImagePointOutOfBounds,
    InvalidSpatialUncertainty,
    IncompatibleImageFrames,
    InvalidPatchSize,
    PatchCellOutOfBounds,
    InvalidRelativeDepth,
    InvalidSpatialConfidence,
    MissingCameraFrameNamespace,
    MissingCalibrationNamespace,
    CameraStreamMismatch,
    InvalidBearing,
    ObservedCannotBeDerivedSpatialClaim,
    GenerativeCannotEnterHistoricalSpatialState,
    MissingSourceObservationLineage,
}

impl fmt::Display for SpatialEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::MissingImagePlaneNamespace => "image plane namespace must be non-zero",
            Self::InvalidImageDimensions => "image dimensions must be non-zero",
            Self::ImagePointOutOfBounds => "image point is non-finite or outside its exact raster",
            Self::InvalidSpatialUncertainty => "spatial uncertainty must be finite and non-negative",
            Self::IncompatibleImageFrames => "spatial comparison requires the exact same image frame",
            Self::InvalidPatchSize => "patch size must be non-zero",
            Self::PatchCellOutOfBounds => "patch-grid cell is outside the exact image grid",
            Self::InvalidRelativeDepth => "relative depth must be finite and in [0, 1]",
            Self::InvalidSpatialConfidence => "spatial confidence must be finite and in [0, 1]",
            Self::MissingCameraFrameNamespace => "camera optical frame namespace must be non-zero",
            Self::MissingCalibrationNamespace => "calibration namespace must be non-zero",
            Self::CameraStreamMismatch => "camera optical frame and observation must belong to the same capture stream",
            Self::InvalidBearing => "camera bearing or angular uncertainty is invalid",
            Self::ObservedCannotBeDerivedSpatialClaim => "localized/derived spatial claims are inferred, not direct observation envelopes",
            Self::GenerativeCannotEnterHistoricalSpatialState => "predicted/simulated/counterfactual evidence cannot enter historical spatial state",
            Self::MissingSourceObservationLineage => "spatial claim evidence must cite its exact source observation",
        };
        f.write_str(message)
    }
}

impl std::error::Error for SpatialEvidenceError {}

fn validate_derived_support(
    evidence: &VisualEvidence,
    observation: VisualObservationRef,
) -> Result<(), SpatialEvidenceError> {
    match evidence.origin() {
        VisualOrigin::Inferred | VisualOrigin::Remembered => {}
        VisualOrigin::Observed => {
            return Err(SpatialEvidenceError::ObservedCannotBeDerivedSpatialClaim);
        }
        VisualOrigin::Predicted | VisualOrigin::Simulated | VisualOrigin::Counterfactual => {
            return Err(SpatialEvidenceError::GenerativeCannotEnterHistoricalSpatialState);
        }
    }
    if !evidence.parent_observations().contains(&observation) {
        return Err(SpatialEvidenceError::MissingSourceObservationLineage);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::{VisualCaptureClock, VisualStreamRef};

    fn observation(source: u64, epoch: u64, frame: u64) -> VisualObservationRef {
        VisualObservationRef::new(
            VisualStreamRef::new(source, epoch).unwrap(),
            frame,
            1_000 + frame,
            VisualCaptureClock::StreamMonotonic,
        )
    }

    fn inferred(observation: VisualObservationRef) -> VisualEvidence {
        VisualEvidence::inferred(vec![observation], 0.8).unwrap()
    }

    fn image_frame(observation: VisualObservationRef) -> ImageFrameRef {
        ImageFrameRef::new(
            observation,
            640,
            480,
            ImagePlaneRef::new(11, ImagePlaneKind::UndistortedRectified).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn same_numeric_pixel_in_different_observations_is_not_comparable() {
        let a_obs = observation(1, 1, 1);
        let b_obs = observation(1, 1, 2);
        let a = ImagePointEvidence::new(image_frame(a_obs), 20.0, 30.0, 1.0, 1.0, inferred(a_obs)).unwrap();
        let b = ImagePointEvidence::new(image_frame(b_obs), 20.0, 30.0, 1.0, 1.0, inferred(b_obs)).unwrap();
        assert_eq!(a.distance_px(&b), Err(SpatialEvidenceError::IncompatibleImageFrames));
    }

    #[test]
    fn patch_relative_depth_is_explicitly_non_metric() {
        let obs = observation(2, 3, 4);
        let frame = image_frame(obs);
        let patch = PatchGridCellRef::new(frame, 16, 2, 3).unwrap();
        let depth = PatchRelativeDepthEvidence::new(
            patch,
            0.25,
            0.9,
            RelativeDepthSource::StereoDisparity,
            inferred(obs),
        )
        .unwrap();
        assert_eq!(depth.near_zero_far_one(), 0.25);
        assert_eq!(depth.source(), RelativeDepthSource::StereoDisparity);
    }

    #[test]
    fn bearing_is_bound_to_stream_and_has_no_metric_range() {
        let obs = observation(7, 8, 9);
        let camera = CameraOpticalFrameRef::new(obs.stream(), 41).unwrap();
        let bearing = CameraBearingEvidence::new(
            obs,
            camera,
            99,
            0.1,
            -0.2,
            0.01,
            inferred(obs),
        )
        .unwrap();
        assert!(!bearing.has_metric_range());
    }

    #[test]
    fn bearing_rejects_camera_from_another_stream() {
        let obs = observation(7, 8, 9);
        let other_stream = VisualStreamRef::new(8, 1).unwrap();
        let camera = CameraOpticalFrameRef::new(other_stream, 41).unwrap();
        assert_eq!(
            CameraBearingEvidence::new(obs, camera, 99, 0.0, 0.0, 0.01, inferred(obs)),
            Err(SpatialEvidenceError::CameraStreamMismatch)
        );
    }

    #[test]
    fn derived_spatial_claim_must_cite_exact_observation() {
        let obs = observation(1, 1, 1);
        let other = observation(1, 1, 2);
        let frame = image_frame(obs);
        let error = ImagePointEvidence::new(frame, 10.0, 10.0, 1.0, 1.0, inferred(other))
            .unwrap_err();
        assert_eq!(error, SpatialEvidenceError::MissingSourceObservationLineage);
    }

    #[test]
    fn prediction_cannot_enter_historical_depth_or_bearing() {
        let obs = observation(1, 1, 1);
        let predicted = VisualEvidence::predicted(vec![obs], 0.95).unwrap();
        let patch = PatchGridCellRef::new(image_frame(obs), 16, 0, 0).unwrap();
        assert_eq!(
            PatchRelativeDepthEvidence::new(
                patch,
                0.5,
                0.95,
                RelativeDepthSource::StereoDisparity,
                predicted.clone(),
            ),
            Err(SpatialEvidenceError::GenerativeCannotEnterHistoricalSpatialState)
        );
        let camera = CameraOpticalFrameRef::new(obs.stream(), 3).unwrap();
        assert_eq!(
            CameraBearingEvidence::new(obs, camera, 4, 0.0, 0.0, 0.01, predicted),
            Err(SpatialEvidenceError::GenerativeCannotEnterHistoricalSpatialState)
        );
    }

    #[test]
    fn observed_envelope_cannot_masquerade_as_localization_inference() {
        let obs = observation(1, 1, 1);
        let observed = VisualEvidence::observed(obs, 1.0).unwrap();
        assert_eq!(
            ImagePointEvidence::new(image_frame(obs), 1.0, 1.0, 0.0, 0.0, observed),
            Err(SpatialEvidenceError::ObservedCannotBeDerivedSpatialClaim)
        );
    }
}
