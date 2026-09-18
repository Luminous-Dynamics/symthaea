// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Raw stereo disparity evidence kept separate from normalized relative depth.
//!
//! Disparity is an image-space measurement in pixels. It does not become metric range until a
//! separately qualified stereo calibration establishes intrinsics, baseline, rectification, units,
//! validity, uncertainty propagation, and temporal/frame compatibility.

use serde::Serialize;
use std::fmt;

use crate::epistemic::{VisualEvidence, VisualOrigin};
use crate::spatial_evidence::PatchGridCellRef;

/// Winning per-patch stereo disparity with explicit pixel units.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PatchStereoDisparityEvidence {
    patch: PatchGridCellRef,
    disparity_px: u32,
    maximum_search_disparity_px: u32,
    match_confidence: f32,
    evidence: VisualEvidence,
}

impl PatchStereoDisparityEvidence {
    pub fn new(
        patch: PatchGridCellRef,
        disparity_px: u32,
        maximum_search_disparity_px: u32,
        match_confidence: f32,
        evidence: VisualEvidence,
    ) -> Result<Self, StereoDisparityEvidenceError> {
        if maximum_search_disparity_px == 0 {
            return Err(StereoDisparityEvidenceError::InvalidSearchRange);
        }
        if disparity_px > maximum_search_disparity_px {
            return Err(StereoDisparityEvidenceError::DisparityOutsideSearchRange);
        }
        if !match_confidence.is_finite() || !(0.0..=1.0).contains(&match_confidence) {
            return Err(StereoDisparityEvidenceError::InvalidConfidence);
        }

        let source_observation = patch.frame().observation();
        match evidence.origin() {
            VisualOrigin::Inferred | VisualOrigin::Remembered => {}
            VisualOrigin::Observed => {
                return Err(StereoDisparityEvidenceError::ObservedCannotBeDisparityInference);
            }
            VisualOrigin::Predicted | VisualOrigin::Simulated | VisualOrigin::Counterfactual => {
                return Err(StereoDisparityEvidenceError::GenerativeEvidenceRejected);
            }
        }
        if !evidence.parent_observations().contains(&source_observation) {
            return Err(StereoDisparityEvidenceError::MissingSourceObservationLineage);
        }

        Ok(Self {
            patch,
            disparity_px,
            maximum_search_disparity_px,
            match_confidence,
            evidence,
        })
    }

    pub const fn patch(&self) -> PatchGridCellRef {
        self.patch
    }

    pub const fn disparity_px(&self) -> u32 {
        self.disparity_px
    }

    pub const fn maximum_search_disparity_px(&self) -> u32 {
        self.maximum_search_disparity_px
    }

    pub const fn match_confidence(&self) -> f32 {
        self.match_confidence
    }

    pub fn evidence(&self) -> &VisualEvidence {
        &self.evidence
    }

    /// Raw disparity alone does not establish metric range.
    pub const fn has_metric_range(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoDisparityEvidenceError {
    InvalidSearchRange,
    DisparityOutsideSearchRange,
    InvalidConfidence,
    ObservedCannotBeDisparityInference,
    GenerativeEvidenceRejected,
    MissingSourceObservationLineage,
}

impl fmt::Display for StereoDisparityEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidSearchRange => "stereo disparity search range must be non-zero",
            Self::DisparityOutsideSearchRange => {
                "winning stereo disparity cannot exceed its declared search range"
            }
            Self::InvalidConfidence => "stereo disparity confidence must be finite and in [0, 1]",
            Self::ObservedCannotBeDisparityInference => {
                "stereo correspondence/disparity is inferred from observations, not raw observation"
            }
            Self::GenerativeEvidenceRejected => {
                "predicted/simulated/counterfactual evidence cannot enter historical disparity state"
            }
            Self::MissingSourceObservationLineage => {
                "stereo disparity evidence must cite the exact source observation"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for StereoDisparityEvidenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::{
        VisualCaptureClock, VisualEvidence, VisualObservationRef, VisualStreamRef,
    };
    use crate::spatial_evidence::{ImageFrameRef, ImagePlaneKind, ImagePlaneRef, PatchGridCellRef};

    fn fixture() -> (PatchGridCellRef, VisualObservationRef) {
        let observation = VisualObservationRef::new(
            VisualStreamRef::new(9, 2).unwrap(),
            7,
            10_000,
            VisualCaptureClock::StreamMonotonic,
        );
        let frame = ImageFrameRef::new(
            observation,
            640,
            480,
            ImagePlaneRef::new(17, ImagePlaneKind::UndistortedRectified).unwrap(),
        )
        .unwrap();
        (PatchGridCellRef::new(frame, 16, 2, 3).unwrap(), observation)
    }

    #[test]
    fn raw_disparity_keeps_pixel_units_and_no_metric_range() {
        let (patch, observation) = fixture();
        let evidence = VisualEvidence::inferred(vec![observation], 0.8).unwrap();
        let disparity = PatchStereoDisparityEvidence::new(patch, 12, 64, 0.9, evidence).unwrap();
        assert_eq!(disparity.disparity_px(), 12);
        assert_eq!(disparity.maximum_search_disparity_px(), 64);
        assert!(!disparity.has_metric_range());
    }

    #[test]
    fn disparity_outside_declared_search_range_fails() {
        let (patch, observation) = fixture();
        let evidence = VisualEvidence::inferred(vec![observation], 0.8).unwrap();
        assert_eq!(
            PatchStereoDisparityEvidence::new(patch, 65, 64, 0.9, evidence),
            Err(StereoDisparityEvidenceError::DisparityOutsideSearchRange)
        );
    }

    #[test]
    fn prediction_cannot_be_historical_disparity() {
        let (patch, observation) = fixture();
        let evidence = VisualEvidence::predicted(vec![observation], 0.9).unwrap();
        assert_eq!(
            PatchStereoDisparityEvidence::new(patch, 12, 64, 0.9, evidence),
            Err(StereoDisparityEvidenceError::GenerativeEvidenceRejected)
        );
    }
}
