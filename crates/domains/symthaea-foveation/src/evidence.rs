// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Structured epistemic evidence over the existing foveation result surface.
//!
//! This adapter is deliberately non-breaking: `FoveationResult` remains unchanged.
//! Callers that know the stable visual source identity can bind an existing ventral result
//! to the source observation that produced it. Because recognition is a semantic inference
//! over observed pixels, the resulting provenance is always `VisualOrigin::Inferred`.

use std::fmt;

use serde::Serialize;
use symthaea_core::hdc::ContinuousHV;
use symthaea_vision_manifold::{
    VisualEvidence, VisualEvidenceError, VisualObservationRef, VisualOrigin,
};

use crate::{FoveationResult, RecognizedContent};

/// Stable non-zero identity for one visual source/stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct VisualSourceId(u64);

impl VisualSourceId {
    pub fn new(value: u64) -> Result<Self, StructuredFoveationEvidenceError> {
        if value == 0 {
            return Err(StructuredFoveationEvidenceError::MissingSourceIdentity);
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Structured, provenance-bearing interpretation of a completed foveation request.
#[derive(Debug, Clone)]
pub struct StructuredFoveationEvidence {
    pub request_id: u64,
    pub source: VisualSourceId,
    pub evidence: VisualEvidence,
    pub semantic_hv: ContinuousHV,
    pub content: RecognizedContent,
    pub grid_row: usize,
    pub grid_col: usize,
    pub source_frame_id: u64,
    pub source_timestamp_us: u64,
    pub processing_time_us: u64,
    pub velocity: [f32; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredFoveationEvidenceError {
    MissingSourceIdentity,
    InvalidVisualEvidence(VisualEvidenceError),
}

impl fmt::Display for StructuredFoveationEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSourceIdentity => {
                f.write_str("structured foveation evidence requires a non-zero visual source id")
            }
            Self::InvalidVisualEvidence(error) => write!(f, "invalid visual evidence: {error}"),
        }
    }
}

impl std::error::Error for StructuredFoveationEvidenceError {}

impl From<VisualEvidenceError> for StructuredFoveationEvidenceError {
    fn from(value: VisualEvidenceError) -> Self {
        Self::InvalidVisualEvidence(value)
    }
}

impl StructuredFoveationEvidence {
    /// Bind a completed ventral result to the exact observed frame from which its crop came.
    ///
    /// The semantic result remains `Inferred` regardless of model confidence. This function
    /// cannot construct direct `Observed` evidence.
    pub fn from_result(
        source: VisualSourceId,
        result: &FoveationResult,
    ) -> Result<Self, StructuredFoveationEvidenceError> {
        let observation = VisualObservationRef::new(
            source.get(),
            result.source_frame_id,
            result.source_timestamp_us,
        );
        let evidence = VisualEvidence::inferred(vec![observation], result.confidence)?;

        Ok(Self {
            request_id: result.request_id,
            source,
            evidence,
            semantic_hv: result.semantic_hv.clone(),
            content: result.content.clone(),
            grid_row: result.grid_row,
            grid_col: result.grid_col,
            source_frame_id: result.source_frame_id,
            source_timestamp_us: result.source_timestamp_us,
            processing_time_us: result.processing_time_us,
            velocity: result.velocity,
        })
    }

    pub const fn origin(&self) -> VisualOrigin {
        self.evidence.origin()
    }

    pub fn source_observation(&self) -> VisualObservationRef {
        self.evidence.parent_observations()[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(confidence: f32) -> FoveationResult {
        FoveationResult {
            request_id: 12,
            semantic_hv: ContinuousHV::zero(8),
            content: RecognizedContent::Text("STOP".to_string()),
            confidence,
            grid_row: 2,
            grid_col: 3,
            source_frame_id: 44,
            source_timestamp_us: 1_234_567,
            processing_time_us: 80_000,
            velocity: [1.5, -0.25],
        }
    }

    #[test]
    fn result_is_bound_as_inference_over_exact_source_frame() {
        let source = VisualSourceId::new(9).unwrap();
        let structured = StructuredFoveationEvidence::from_result(source, &result(0.8)).unwrap();

        assert_eq!(structured.origin(), VisualOrigin::Inferred);
        assert!(!structured.origin().is_observed());
        assert_eq!(
            structured.source_observation(),
            VisualObservationRef::new(9, 44, 1_234_567)
        );
        assert_eq!(structured.request_id, 12);
        assert_eq!(structured.grid_row, 2);
        assert_eq!(structured.grid_col, 3);
    }

    #[test]
    fn zero_source_identity_fails_closed() {
        assert_eq!(
            VisualSourceId::new(0),
            Err(StructuredFoveationEvidenceError::MissingSourceIdentity)
        );
    }

    #[test]
    fn invalid_recognition_confidence_cannot_become_structured_evidence() {
        let source = VisualSourceId::new(3).unwrap();
        assert!(matches!(
            StructuredFoveationEvidence::from_result(source, &result(f32::NAN)),
            Err(StructuredFoveationEvidenceError::InvalidVisualEvidence(
                VisualEvidenceError::InvalidConfidence
            ))
        ));
    }

    #[test]
    fn perfect_model_confidence_still_does_not_become_observation() {
        let source = VisualSourceId::new(5).unwrap();
        let structured = StructuredFoveationEvidence::from_result(source, &result(1.0)).unwrap();
        assert_eq!(structured.origin(), VisualOrigin::Inferred);
    }
}
