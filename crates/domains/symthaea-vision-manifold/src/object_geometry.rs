// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Honest object-image geometry for VIS-003.
//!
//! This module intentionally distinguishes geometry that only supports a point claim from
//! patch-cluster extent, detector boxes, segmentation masks, and keypoints. Metric eligibility is
//! encoded explicitly so a centroid tracker cannot accidentally advertise box-IoU/HOTA readiness.

use serde::{Deserialize, Serialize};

use crate::epistemic::{VisualEvidence, VisualOrigin};
use crate::types::{ObjectHypothesis, PatchGrid};

/// A sub-pixel point in image coordinates. Origin is the top-left pixel corner; +x is right and
/// +y is down.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PixelPoint {
    pub x: f32,
    pub y: f32,
}

/// Half-open pixel rectangle `[left, right) × [top, bottom)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PixelRect {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

impl PixelRect {
    pub fn width(self) -> f32 {
        self.right - self.left
    }

    pub fn height(self) -> f32 {
        self.bottom - self.top
    }

    pub fn area(self) -> f32 {
        self.width() * self.height()
    }

    pub fn center(self) -> PixelPoint {
        PixelPoint {
            x: (self.left + self.right) * 0.5,
            y: (self.top + self.bottom) * 0.5,
        }
    }
}

/// One run in a row-major binary segmentation mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaskRun {
    /// Flattened row-major start index.
    pub start: u32,
    /// Number of foreground pixels in this run.
    pub len: u32,
}

/// Compact, exact binary mask using sorted, non-overlapping row-major runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentationMaskRle {
    pub width: u32,
    pub height: u32,
    pub runs: Vec<MaskRun>,
}

impl SegmentationMaskRle {
    pub fn new(width: u32, height: u32, runs: Vec<MaskRun>) -> Result<Self, GeometryError> {
        if width == 0 || height == 0 {
            return Err(GeometryError::InvalidFrameDimensions);
        }
        if runs.is_empty() {
            return Err(GeometryError::EmptyMask);
        }
        let total = width
            .checked_mul(height)
            .ok_or(GeometryError::GeometryOverflow)?;
        let mut previous_end = 0u32;
        for (index, run) in runs.iter().enumerate() {
            if run.len == 0 {
                return Err(GeometryError::ZeroLengthMaskRun { index });
            }
            let end = run
                .start
                .checked_add(run.len)
                .ok_or(GeometryError::GeometryOverflow)?;
            if end > total {
                return Err(GeometryError::MaskRunOutOfBounds { index });
            }
            if index > 0 && run.start < previous_end {
                return Err(GeometryError::OverlappingMaskRuns { index });
            }
            previous_end = end;
        }
        Ok(Self {
            width,
            height,
            runs,
        })
    }

    pub fn foreground_pixels(&self) -> u64 {
        self.runs.iter().map(|run| u64::from(run.len)).sum()
    }

    /// Exact foreground bounding box in pixel coordinates.
    pub fn bounds(&self) -> PixelRect {
        let mut min_x = self.width;
        let mut min_y = self.height;
        let mut max_x_exclusive = 0u32;
        let mut max_y_exclusive = 0u32;

        for run in &self.runs {
            let mut cursor = run.start;
            let mut remaining = run.len;
            while remaining > 0 {
                let y = cursor / self.width;
                let x = cursor % self.width;
                let take = remaining.min(self.width - x);
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x_exclusive = max_x_exclusive.max(x + take);
                max_y_exclusive = max_y_exclusive.max(y + 1);
                cursor += take;
                remaining -= take;
            }
        }

        PixelRect {
            left: min_x as f32,
            top: min_y as f32,
            right: max_x_exclusive as f32,
            bottom: max_y_exclusive as f32,
        }
    }

    /// Foreground centroid in pixel-center coordinates.
    pub fn centroid(&self) -> PixelPoint {
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut count = 0u64;
        for run in &self.runs {
            let end = run.start + run.len;
            for flat in run.start..end {
                let y = flat / self.width;
                let x = flat % self.width;
                sum_x += f64::from(x) + 0.5;
                sum_y += f64::from(y) + 0.5;
                count += 1;
            }
        }
        PixelPoint {
            x: (sum_x / count as f64) as f32,
            y: (sum_y / count as f64) as f32,
        }
    }
}

/// Named object keypoint with its own confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisualKeypoint {
    pub name: String,
    pub point: PixelPoint,
    pub confidence: f32,
}

/// Uncertainty attached to a geometry estimate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeometryUncertainty {
    /// Overall confidence in this geometry estimate.
    pub confidence: f32,
    /// Optional 1σ centroid uncertainty `[x, y]` in pixels.
    pub centroid_std_px: Option<[f32; 2]>,
    /// Optional 1σ extent uncertainty `[width, height]` in pixels.
    pub extent_std_px: Option<[f32; 2]>,
}

impl GeometryUncertainty {
    pub fn new(
        confidence: f32,
        centroid_std_px: Option<[f32; 2]>,
        extent_std_px: Option<[f32; 2]>,
    ) -> Result<Self, GeometryError> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(GeometryError::InvalidConfidence);
        }
        for std in [centroid_std_px, extent_std_px].into_iter().flatten() {
            if std.into_iter().any(|value| !value.is_finite() || value < 0.0) {
                return Err(GeometryError::InvalidStandardDeviation);
            }
        }
        Ok(Self {
            confidence,
            centroid_std_px,
            extent_std_px,
        })
    }

    pub const fn unknown(confidence: f32) -> Result<Self, GeometryError> {
        if !confidence.is_finite() || confidence < 0.0 || confidence > 1.0 {
            return Err(GeometryError::InvalidConfidence);
        }
        Ok(Self {
            confidence,
            centroid_std_px: None,
            extent_std_px: None,
        })
    }
}

/// What spatial support actually exists for the object claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometrySupport {
    /// Legacy point tracker: no area claim exists.
    CentroidPoint,
    /// Envelope of spatially clustered HDC patches. This is not a detector-produced box.
    PatchClusterExtent {
        min_row: usize,
        max_row_exclusive: usize,
        min_col: usize,
        max_col_exclusive: usize,
        patch_indices: Vec<usize>,
        bounds: PixelRect,
    },
    /// A box explicitly emitted by a detector/localizer.
    DetectorBox { bounds: PixelRect },
    /// Exact binary segmentation support.
    SegmentationMask { mask: SegmentationMaskRle },
    /// Named keypoints. The object centroid is represented separately.
    Keypoints { points: Vec<VisualKeypoint> },
}

/// Geometry representation level, ordered by meaning rather than presumed quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryKind {
    CentroidPoint,
    PatchClusterExtent,
    DetectorBox,
    SegmentationMask,
    Keypoints,
}

/// Explicit evaluation eligibility. `false` means the representation cannot honestly support the
/// corresponding metric, not that performance on that metric is poor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryMetricEligibility {
    pub centroid_distance: bool,
    pub patch_overlap: bool,
    pub box_iou: bool,
    pub mask_iou: bool,
    pub keypoint_distance: bool,
}

/// One object geometry estimate in a concrete image coordinate frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisualObjectGeometry {
    pub frame_width: u32,
    pub frame_height: u32,
    pub centroid: PixelPoint,
    pub support: GeometrySupport,
    pub uncertainty: GeometryUncertainty,
}

impl VisualObjectGeometry {
    pub fn centroid_only(
        frame_width: u32,
        frame_height: u32,
        centroid: PixelPoint,
        uncertainty: GeometryUncertainty,
    ) -> Result<Self, GeometryError> {
        validate_point(frame_width, frame_height, centroid)?;
        Ok(Self {
            frame_width,
            frame_height,
            centroid,
            support: GeometrySupport::CentroidPoint,
            uncertainty,
        })
    }

    /// Adapter for the current patch-cluster `ObjectHypothesis` representation.
    ///
    /// The resulting bounds are a patch envelope, not a detector box. Consequently `box_iou`
    /// remains false in metric eligibility.
    pub fn from_patch_hypothesis(
        hypothesis: &ObjectHypothesis,
        grid: &PatchGrid,
        uncertainty: GeometryUncertainty,
    ) -> Result<Self, GeometryError> {
        if grid.frame_width == 0 || grid.frame_height == 0 || grid.num_patches() == 0 {
            return Err(GeometryError::InvalidFrameDimensions);
        }
        if hypothesis.patch_indices.is_empty() {
            return Err(GeometryError::EmptyPatchSupport);
        }
        if hypothesis.centroid_row >= grid.rows || hypothesis.centroid_col >= grid.cols {
            return Err(GeometryError::CentroidPatchOutOfBounds);
        }

        let mut indices = hypothesis.patch_indices.clone();
        indices.sort_unstable();
        for pair in indices.windows(2) {
            if pair[0] == pair[1] {
                return Err(GeometryError::DuplicatePatchIndex { index: pair[0] });
            }
        }

        let mut min_row = usize::MAX;
        let mut min_col = usize::MAX;
        let mut max_row = 0usize;
        let mut max_col = 0usize;
        for &index in &indices {
            if index >= grid.num_patches() {
                return Err(GeometryError::PatchIndexOutOfBounds { index });
            }
            let row = index / grid.cols;
            let col = index % grid.cols;
            min_row = min_row.min(row);
            min_col = min_col.min(col);
            max_row = max_row.max(row);
            max_col = max_col.max(col);
        }

        let max_row_exclusive = max_row + 1;
        let max_col_exclusive = max_col + 1;
        let bounds = PixelRect {
            left: (min_col * grid.patch_size) as f32,
            top: (min_row * grid.patch_size) as f32,
            right: ((max_col_exclusive * grid.patch_size).min(grid.frame_width as usize)) as f32,
            bottom: ((max_row_exclusive * grid.patch_size).min(grid.frame_height as usize)) as f32,
        };
        validate_rect(grid.frame_width, grid.frame_height, bounds)?;

        let centroid = PixelPoint {
            x: (((hypothesis.centroid_col as f32) + 0.5) * grid.patch_size as f32)
                .min(grid.frame_width as f32 - f32::EPSILON),
            y: (((hypothesis.centroid_row as f32) + 0.5) * grid.patch_size as f32)
                .min(grid.frame_height as f32 - f32::EPSILON),
        };
        validate_point(grid.frame_width, grid.frame_height, centroid)?;

        Ok(Self {
            frame_width: grid.frame_width,
            frame_height: grid.frame_height,
            centroid,
            support: GeometrySupport::PatchClusterExtent {
                min_row,
                max_row_exclusive,
                min_col,
                max_col_exclusive,
                patch_indices: indices,
                bounds,
            },
            uncertainty,
        })
    }

    pub fn detector_box(
        frame_width: u32,
        frame_height: u32,
        bounds: PixelRect,
        uncertainty: GeometryUncertainty,
    ) -> Result<Self, GeometryError> {
        validate_rect(frame_width, frame_height, bounds)?;
        Ok(Self {
            frame_width,
            frame_height,
            centroid: bounds.center(),
            support: GeometrySupport::DetectorBox { bounds },
            uncertainty,
        })
    }

    pub fn segmentation_mask(
        frame_width: u32,
        frame_height: u32,
        mask: SegmentationMaskRle,
        uncertainty: GeometryUncertainty,
    ) -> Result<Self, GeometryError> {
        if mask.width != frame_width || mask.height != frame_height {
            return Err(GeometryError::MaskFrameMismatch);
        }
        let centroid = mask.centroid();
        validate_point(frame_width, frame_height, centroid)?;
        Ok(Self {
            frame_width,
            frame_height,
            centroid,
            support: GeometrySupport::SegmentationMask { mask },
            uncertainty,
        })
    }

    pub fn keypoints(
        frame_width: u32,
        frame_height: u32,
        centroid: PixelPoint,
        points: Vec<VisualKeypoint>,
        uncertainty: GeometryUncertainty,
    ) -> Result<Self, GeometryError> {
        validate_point(frame_width, frame_height, centroid)?;
        if points.is_empty() {
            return Err(GeometryError::EmptyKeypoints);
        }
        for point in &points {
            if point.name.trim().is_empty() {
                return Err(GeometryError::EmptyKeypointName);
            }
            validate_point(frame_width, frame_height, point.point)?;
            if !point.confidence.is_finite() || !(0.0..=1.0).contains(&point.confidence) {
                return Err(GeometryError::InvalidConfidence);
            }
        }
        Ok(Self {
            frame_width,
            frame_height,
            centroid,
            support: GeometrySupport::Keypoints { points },
            uncertainty,
        })
    }

    pub const fn kind(&self) -> GeometryKind {
        match self.support {
            GeometrySupport::CentroidPoint => GeometryKind::CentroidPoint,
            GeometrySupport::PatchClusterExtent { .. } => GeometryKind::PatchClusterExtent,
            GeometrySupport::DetectorBox { .. } => GeometryKind::DetectorBox,
            GeometrySupport::SegmentationMask { .. } => GeometryKind::SegmentationMask,
            GeometrySupport::Keypoints { .. } => GeometryKind::Keypoints,
        }
    }

    pub const fn metric_eligibility(&self) -> GeometryMetricEligibility {
        match self.support {
            GeometrySupport::CentroidPoint => GeometryMetricEligibility {
                centroid_distance: true,
                patch_overlap: false,
                box_iou: false,
                mask_iou: false,
                keypoint_distance: false,
            },
            GeometrySupport::PatchClusterExtent { .. } => GeometryMetricEligibility {
                centroid_distance: true,
                patch_overlap: true,
                box_iou: false,
                mask_iou: false,
                keypoint_distance: false,
            },
            GeometrySupport::DetectorBox { .. } => GeometryMetricEligibility {
                centroid_distance: true,
                patch_overlap: false,
                box_iou: true,
                mask_iou: false,
                keypoint_distance: false,
            },
            GeometrySupport::SegmentationMask { .. } => GeometryMetricEligibility {
                centroid_distance: true,
                patch_overlap: false,
                box_iou: false,
                mask_iou: true,
                keypoint_distance: false,
            },
            GeometrySupport::Keypoints { .. } => GeometryMetricEligibility {
                centroid_distance: true,
                patch_overlap: false,
                box_iou: false,
                mask_iou: false,
                keypoint_distance: true,
            },
        }
    }
}

/// Geometry plus the epistemic lineage supporting the estimate.
///
/// VIS-003 deliberately requires `Inferred` evidence: a box, mask, keypoint set, or patch envelope
/// is an interpretation of sensor data, not the raw sensor observation itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferredObjectGeometry {
    pub geometry: VisualObjectGeometry,
    pub evidence: VisualEvidence,
}

impl InferredObjectGeometry {
    pub fn new(
        geometry: VisualObjectGeometry,
        evidence: VisualEvidence,
    ) -> Result<Self, GeometryError> {
        evidence
            .validate()
            .map_err(|_| GeometryError::InvalidVisualEvidence)?;
        if evidence.origin() != VisualOrigin::Inferred {
            return Err(GeometryError::RequiresInferredEvidence);
        }
        Ok(Self { geometry, evidence })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometryError {
    InvalidFrameDimensions,
    GeometryOverflow,
    PointOutOfBounds,
    InvalidRectangle,
    InvalidConfidence,
    InvalidStandardDeviation,
    EmptyPatchSupport,
    PatchIndexOutOfBounds { index: usize },
    DuplicatePatchIndex { index: usize },
    CentroidPatchOutOfBounds,
    EmptyMask,
    ZeroLengthMaskRun { index: usize },
    MaskRunOutOfBounds { index: usize },
    OverlappingMaskRuns { index: usize },
    MaskFrameMismatch,
    EmptyKeypoints,
    EmptyKeypointName,
    InvalidVisualEvidence,
    RequiresInferredEvidence,
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFrameDimensions => f.write_str("visual geometry frame dimensions must be non-zero"),
            Self::GeometryOverflow => f.write_str("visual geometry arithmetic overflow"),
            Self::PointOutOfBounds => f.write_str("visual geometry point lies outside its frame"),
            Self::InvalidRectangle => f.write_str("visual geometry rectangle is empty, non-finite, or outside its frame"),
            Self::InvalidConfidence => f.write_str("visual geometry confidence must be finite and within [0, 1]"),
            Self::InvalidStandardDeviation => f.write_str("visual geometry standard deviations must be finite and non-negative"),
            Self::EmptyPatchSupport => f.write_str("patch-cluster geometry requires at least one supporting patch"),
            Self::PatchIndexOutOfBounds { index } => write!(f, "patch index {index} lies outside the patch grid"),
            Self::DuplicatePatchIndex { index } => write!(f, "patch index {index} appears more than once"),
            Self::CentroidPatchOutOfBounds => f.write_str("object centroid patch lies outside the patch grid"),
            Self::EmptyMask => f.write_str("segmentation mask must contain foreground support"),
            Self::ZeroLengthMaskRun { index } => write!(f, "segmentation mask run {index} has zero length"),
            Self::MaskRunOutOfBounds { index } => write!(f, "segmentation mask run {index} exceeds mask dimensions"),
            Self::OverlappingMaskRuns { index } => write!(f, "segmentation mask run {index} overlaps or is out of order"),
            Self::MaskFrameMismatch => f.write_str("segmentation mask dimensions do not match the image frame"),
            Self::EmptyKeypoints => f.write_str("keypoint geometry requires at least one keypoint"),
            Self::EmptyKeypointName => f.write_str("visual keypoint names must be non-empty"),
            Self::InvalidVisualEvidence => f.write_str("geometry carries invalid visual evidence"),
            Self::RequiresInferredEvidence => f.write_str("object geometry must be carried as inferred visual evidence"),
        }
    }
}

impl std::error::Error for GeometryError {}

fn validate_point(
    frame_width: u32,
    frame_height: u32,
    point: PixelPoint,
) -> Result<(), GeometryError> {
    if frame_width == 0 || frame_height == 0 {
        return Err(GeometryError::InvalidFrameDimensions);
    }
    if !point.x.is_finite()
        || !point.y.is_finite()
        || point.x < 0.0
        || point.y < 0.0
        || point.x >= frame_width as f32
        || point.y >= frame_height as f32
    {
        return Err(GeometryError::PointOutOfBounds);
    }
    Ok(())
}

fn validate_rect(
    frame_width: u32,
    frame_height: u32,
    rect: PixelRect,
) -> Result<(), GeometryError> {
    if frame_width == 0 || frame_height == 0 {
        return Err(GeometryError::InvalidFrameDimensions);
    }
    let values = [rect.left, rect.top, rect.right, rect.bottom];
    if values.into_iter().any(|value| !value.is_finite())
        || rect.left < 0.0
        || rect.top < 0.0
        || rect.right > frame_width as f32
        || rect.bottom > frame_height as f32
        || rect.left >= rect.right
        || rect.top >= rect.bottom
    {
        return Err(GeometryError::InvalidRectangle);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::ContinuousHV;

    use crate::epistemic::{VisualCaptureClock, VisualObservationRef, VisualStreamRef};

    fn inferred_evidence(confidence: f32) -> VisualEvidence {
        let stream = VisualStreamRef::new(11, 5).unwrap();
        let observation = VisualObservationRef::new(
            stream,
            9,
            123_000,
            VisualCaptureClock::StreamMonotonic,
        );
        VisualEvidence::inferred(vec![observation], confidence).unwrap()
    }

    fn hypothesis(indices: Vec<usize>, row: usize, col: usize) -> ObjectHypothesis {
        ObjectHypothesis {
            centroid_row: row,
            centroid_col: col,
            patch_indices: indices,
            saliency: 0.7,
            hv: ContinuousHV::random(128, 7),
        }
    }

    #[test]
    fn centroid_only_never_claims_box_iou() {
        let geometry = VisualObjectGeometry::centroid_only(
            100,
            80,
            PixelPoint { x: 20.0, y: 30.0 },
            GeometryUncertainty::unknown(0.8).unwrap(),
        )
        .unwrap();
        assert_eq!(geometry.kind(), GeometryKind::CentroidPoint);
        let metrics = geometry.metric_eligibility();
        assert!(metrics.centroid_distance);
        assert!(!metrics.box_iou);
        assert!(!metrics.mask_iou);
    }

    #[test]
    fn patch_hypothesis_becomes_patch_extent_not_detector_box() {
        let grid = PatchGrid::new(32, 24, 8);
        let geometry = VisualObjectGeometry::from_patch_hypothesis(
            &hypothesis(vec![0, 1, 4, 5], 0, 0),
            &grid,
            GeometryUncertainty::unknown(0.6).unwrap(),
        )
        .unwrap();
        assert_eq!(geometry.kind(), GeometryKind::PatchClusterExtent);
        let metrics = geometry.metric_eligibility();
        assert!(metrics.patch_overlap);
        assert!(!metrics.box_iou);
        match geometry.support {
            GeometrySupport::PatchClusterExtent { bounds, .. } => {
                assert_eq!(bounds.left, 0.0);
                assert_eq!(bounds.top, 0.0);
                assert_eq!(bounds.right, 16.0);
                assert_eq!(bounds.bottom, 16.0);
            }
            _ => panic!("expected patch-cluster extent"),
        }
    }

    #[test]
    fn duplicate_patch_support_fails_closed() {
        let grid = PatchGrid::new(32, 24, 8);
        let result = VisualObjectGeometry::from_patch_hypothesis(
            &hypothesis(vec![0, 0], 0, 0),
            &grid,
            GeometryUncertainty::unknown(0.6).unwrap(),
        );
        assert_eq!(result, Err(GeometryError::DuplicatePatchIndex { index: 0 }));
    }

    #[test]
    fn detector_box_is_box_iou_eligible() {
        let geometry = VisualObjectGeometry::detector_box(
            100,
            80,
            PixelRect {
                left: 10.0,
                top: 20.0,
                right: 40.0,
                bottom: 60.0,
            },
            GeometryUncertainty::unknown(0.9).unwrap(),
        )
        .unwrap();
        assert_eq!(geometry.kind(), GeometryKind::DetectorBox);
        assert!(geometry.metric_eligibility().box_iou);
    }

    #[test]
    fn mask_rle_validates_and_computes_exact_bounds() {
        // 4x3 mask, foreground: row0 x1..2 and row1 x0..1.
        let mask = SegmentationMaskRle::new(
            4,
            3,
            vec![MaskRun { start: 1, len: 2 }, MaskRun { start: 4, len: 2 }],
        )
        .unwrap();
        assert_eq!(mask.foreground_pixels(), 4);
        assert_eq!(
            mask.bounds(),
            PixelRect {
                left: 0.0,
                top: 0.0,
                right: 3.0,
                bottom: 2.0,
            }
        );
        let geometry = VisualObjectGeometry::segmentation_mask(
            4,
            3,
            mask,
            GeometryUncertainty::unknown(0.75).unwrap(),
        )
        .unwrap();
        assert!(geometry.metric_eligibility().mask_iou);
        assert!(!geometry.metric_eligibility().box_iou);
    }

    #[test]
    fn overlapping_mask_runs_are_rejected() {
        assert_eq!(
            SegmentationMaskRle::new(
                4,
                3,
                vec![MaskRun { start: 1, len: 3 }, MaskRun { start: 3, len: 2 }],
            ),
            Err(GeometryError::OverlappingMaskRuns { index: 1 })
        );
    }

    #[test]
    fn geometry_requires_inferred_epistemic_origin() {
        let geometry = VisualObjectGeometry::centroid_only(
            16,
            16,
            PixelPoint { x: 4.0, y: 5.0 },
            GeometryUncertainty::unknown(1.0).unwrap(),
        )
        .unwrap();
        let inferred = InferredObjectGeometry::new(geometry.clone(), inferred_evidence(1.0));
        assert!(inferred.is_ok());

        let stream = VisualStreamRef::new(11, 5).unwrap();
        let observation = VisualObservationRef::new(
            stream,
            9,
            123_000,
            VisualCaptureClock::StreamMonotonic,
        );
        let observed = VisualEvidence::observed(observation, 1.0).unwrap();
        assert_eq!(
            InferredObjectGeometry::new(geometry, observed),
            Err(GeometryError::RequiresInferredEvidence)
        );
    }
}
