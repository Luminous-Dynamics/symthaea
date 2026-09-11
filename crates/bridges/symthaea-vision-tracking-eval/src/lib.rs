// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adapter from `symthaea-vision-manifold` centroid tracks into the generic
//! point-tracking evaluator.

#![deny(unsafe_code)]

use symthaea_tracking_eval::{EvaluationFrame, GroundTruthPoint, Point2, PredictedPoint};
use symthaea_vision_manifold::TrackedObject;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionTrackingEvalError {
    InvalidGrid,
    TrackOutsideGrid,
    InvalidImageGeometry,
    InvalidBoundingBox,
}

/// Convert one vision-manifold track into a normalized image-plane point.
/// This makes no claim about range, bearing, or world-space position.
pub fn prediction_from_tracked_object(
    track: &TrackedObject,
    grid_cols: usize,
    grid_rows: usize,
) -> Result<PredictedPoint, VisionTrackingEvalError> {
    prediction_from_grid_centroid(
        track.track_id,
        track.centroid_col,
        track.centroid_row,
        grid_cols,
        grid_rows,
    )
}

/// Lower-level conversion that is easy to use in fixtures without constructing
/// an HDC-backed `TrackedObject`.
pub fn prediction_from_grid_centroid(
    track_id: u64,
    centroid_col: usize,
    centroid_row: usize,
    grid_cols: usize,
    grid_rows: usize,
) -> Result<PredictedPoint, VisionTrackingEvalError> {
    if grid_cols == 0 || grid_rows == 0 {
        return Err(VisionTrackingEvalError::InvalidGrid);
    }
    if centroid_col >= grid_cols || centroid_row >= grid_rows {
        return Err(VisionTrackingEvalError::TrackOutsideGrid);
    }
    Ok(PredictedPoint {
        track_id,
        point: Point2 {
            x_norm: (centroid_col as f64 + 0.5) / grid_cols as f64,
            y_norm: (centroid_row as f64 + 0.5) / grid_rows as f64,
        },
    })
}

/// Minimal MOT-style ground-truth box input. Coordinates are in source-image
/// pixels with `(x, y)` at the top-left corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundTruthBox {
    pub object_id: u64,
    pub x_px: f64,
    pub y_px: f64,
    pub width_px: f64,
    pub height_px: f64,
}

/// Convert a ground-truth box to its normalized center point.
///
/// The resulting point supports the current centroid-only tracker. This helper
/// must not be mistaken for box-IoU evaluation.
pub fn ground_truth_center_from_box(
    bbox: GroundTruthBox,
    image_width: u32,
    image_height: u32,
) -> Result<GroundTruthPoint, VisionTrackingEvalError> {
    if image_width == 0 || image_height == 0 {
        return Err(VisionTrackingEvalError::InvalidImageGeometry);
    }
    if !bbox.x_px.is_finite()
        || !bbox.y_px.is_finite()
        || !bbox.width_px.is_finite()
        || !bbox.height_px.is_finite()
        || bbox.width_px <= 0.0
        || bbox.height_px <= 0.0
    {
        return Err(VisionTrackingEvalError::InvalidBoundingBox);
    }

    let center_x = bbox.x_px + bbox.width_px / 2.0;
    let center_y = bbox.y_px + bbox.height_px / 2.0;
    if center_x < 0.0
        || center_y < 0.0
        || center_x > image_width as f64
        || center_y > image_height as f64
    {
        return Err(VisionTrackingEvalError::InvalidBoundingBox);
    }

    Ok(GroundTruthPoint {
        object_id: bbox.object_id,
        point: Point2 {
            x_norm: center_x / image_width as f64,
            y_norm: center_y / image_height as f64,
        },
    })
}

/// Build one explicit evaluation frame from current vision tracks and already
/// normalized ground truth. Empty `tracks` and/or `ground_truth` are preserved.
pub fn frame_from_tracks(
    frame_index: u64,
    ground_truth: Vec<GroundTruthPoint>,
    tracks: &[TrackedObject],
    grid_cols: usize,
    grid_rows: usize,
) -> Result<EvaluationFrame, VisionTrackingEvalError> {
    let predictions = tracks
        .iter()
        .map(|track| prediction_from_tracked_object(track, grid_cols, grid_rows))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(EvaluationFrame {
        frame_index,
        ground_truth,
        predictions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_centroid_normalizes_without_world_space_claims() {
        let prediction = prediction_from_grid_centroid(7, 1, 2, 4, 4).unwrap();
        assert_eq!(prediction.track_id, 7);
        assert!((prediction.point.x_norm - 0.375).abs() < 1e-12);
        assert!((prediction.point.y_norm - 0.625).abs() < 1e-12);
    }

    #[test]
    fn out_of_grid_track_fails_closed() {
        assert_eq!(
            prediction_from_grid_centroid(7, 4, 0, 4, 4),
            Err(VisionTrackingEvalError::TrackOutsideGrid)
        );
    }

    #[test]
    fn mot_style_box_maps_to_center_only() {
        let truth = ground_truth_center_from_box(
            GroundTruthBox {
                object_id: 10,
                x_px: 20.0,
                y_px: 10.0,
                width_px: 20.0,
                height_px: 10.0,
            },
            100,
            50,
        )
        .unwrap();
        assert_eq!(truth.object_id, 10);
        assert!((truth.point.x_norm - 0.30).abs() < 1e-12);
        assert!((truth.point.y_norm - 0.30).abs() < 1e-12);
    }

    #[test]
    fn invalid_box_geometry_is_rejected() {
        assert_eq!(
            ground_truth_center_from_box(
                GroundTruthBox {
                    object_id: 1,
                    x_px: 10.0,
                    y_px: 10.0,
                    width_px: -1.0,
                    height_px: 2.0,
                },
                100,
                100,
            ),
            Err(VisionTrackingEvalError::InvalidBoundingBox)
        );
    }
}
