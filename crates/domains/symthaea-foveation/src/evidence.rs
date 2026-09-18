// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Structured epistemic evidence over completed foveation results.
//!
//! Provenance must already have entered at the capture/frame boundary. This adapter refuses to
//! invent or accept a late source identity after recognition has happened.

use std::fmt;

use symthaea_core::hdc::ContinuousHV;
use symthaea_vision_manifold::{
    VisualEvidence, VisualEvidenceError, VisualObservationRef, VisualOrigin,
};

use crate::{FoveationResult, RecognizedContent, VentralExecutionReceipt};

#[derive(Debug, Clone)]
pub struct StructuredFoveationEvidence {
    pub request_id: u64,
    pub evidence: VisualEvidence,
    pub semantic_hv: ContinuousHV,
    pub content: RecognizedContent,
    pub grid_row: usize,
    pub grid_col: usize,
    pub source_frame_id: u64,
    pub source_timestamp_us: u64,
    pub processing_time_us: u64,
    pub velocity: [f32; 2],
    pub execution: VentralExecutionReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredFoveationEvidenceError {
    MissingSourceObservation,
    SourceFrameMismatch {
        result_frame_id: u64,
        observation_frame_id: u64,
    },
    SourceTimestampMismatch {
        result_timestamp_us: u64,
        observation_timestamp_us: u64,
    },
    InvalidVisualEvidence(VisualEvidenceError),
}

impl fmt::Display for StructuredFoveationEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSourceObservation => {
                f.write_str("structured foveation evidence requires capture-bound observation provenance")
            }
            Self::SourceFrameMismatch {
                result_frame_id,
                observation_frame_id,
            } => write!(
                f,
                "result frame {result_frame_id} does not match observation frame {observation_frame_id}"
            ),
            Self::SourceTimestampMismatch {
                result_timestamp_us,
                observation_timestamp_us,
            } => write!(
                f,
                "result timestamp {result_timestamp_us} does not match observation timestamp {observation_timestamp_us}"
            ),
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
    /// Convert a completed ventral result into an inference grounded in its exact captured frame.
    ///
    /// Recognition is always `Inferred`, even at confidence 1.0. Missing or contradictory
    /// source provenance fails closed rather than being reconstructed from legacy metadata.
    pub fn from_result(
        result: &FoveationResult,
    ) -> Result<Self, StructuredFoveationEvidenceError> {
        let observation = result
            .source_observation
            .ok_or(StructuredFoveationEvidenceError::MissingSourceObservation)?;

        if observation.frame_id() != result.source_frame_id {
            return Err(StructuredFoveationEvidenceError::SourceFrameMismatch {
                result_frame_id: result.source_frame_id,
                observation_frame_id: observation.frame_id(),
            });
        }
        if observation.captured_at_us() != result.source_timestamp_us {
            return Err(StructuredFoveationEvidenceError::SourceTimestampMismatch {
                result_timestamp_us: result.source_timestamp_us,
                observation_timestamp_us: observation.captured_at_us(),
            });
        }

        let evidence = VisualEvidence::inferred(vec![observation], result.confidence)?;

        Ok(Self {
            request_id: result.request_id,
            evidence,
            semantic_hv: result.semantic_hv.clone(),
            content: result.content.clone(),
            grid_row: result.grid_row,
            grid_col: result.grid_col,
            source_frame_id: result.source_frame_id,
            source_timestamp_us: result.source_timestamp_us,
            processing_time_us: result.processing_time_us,
            velocity: result.velocity,
            execution: result.execution,
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
    use crate::{
        RoutingStrategy, VentralExecutionKind, VentralExecutionReceipt, VentralOperation,
    };
    use symthaea_vision_manifold::{VisualCaptureClock, VisualStreamRef};

    fn observation(frame_id: u64, timestamp_us: u64) -> VisualObservationRef {
        VisualObservationRef::new(
            VisualStreamRef::new(7, 9).unwrap(),
            frame_id,
            timestamp_us,
            VisualCaptureClock::StreamMonotonic,
        )
    }

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
            source_observation: Some(observation(44, 1_234_567)),
            execution: VentralExecutionReceipt::new(
                VentralExecutionKind::HashStubV1,
                RoutingStrategy::AlwaysOcr,
                VentralOperation::Ocr,
            ),
            processing_time_us: 80_000,
            velocity: [1.5, -0.25],
        }
    }

    #[test]
    fn result_is_inference_over_exact_capture_owned_observation() {
        let structured = StructuredFoveationEvidence::from_result(&result(0.8)).unwrap();
        assert_eq!(structured.origin(), VisualOrigin::Inferred);
        assert!(!structured.origin().is_observed());
        assert_eq!(structured.source_observation(), observation(44, 1_234_567));
        assert_eq!(structured.execution.kind, VentralExecutionKind::HashStubV1);
    }

    #[test]
    fn missing_capture_provenance_fails_closed() {
        let mut result = result(0.8);
        result.source_observation = None;
        assert_eq!(
            StructuredFoveationEvidence::from_result(&result).unwrap_err(),
            StructuredFoveationEvidenceError::MissingSourceObservation
        );
    }

    #[test]
    fn contradictory_frame_binding_fails_closed() {
        let mut result = result(0.8);
        result.source_frame_id += 1;
        assert!(matches!(
            StructuredFoveationEvidence::from_result(&result),
            Err(StructuredFoveationEvidenceError::SourceFrameMismatch { .. })
        ));
    }

    #[test]
    fn contradictory_timestamp_binding_fails_closed() {
        let mut result = result(0.8);
        result.source_timestamp_us += 1;
        assert!(matches!(
            StructuredFoveationEvidence::from_result(&result),
            Err(StructuredFoveationEvidenceError::SourceTimestampMismatch { .. })
        ));
    }

    #[test]
    fn invalid_confidence_cannot_become_structured_evidence() {
        assert!(matches!(
            StructuredFoveationEvidence::from_result(&result(f32::NAN)),
            Err(StructuredFoveationEvidenceError::InvalidVisualEvidence(
                VisualEvidenceError::InvalidConfidence
            ))
        ));
    }

    #[test]
    fn perfect_confidence_remains_inference() {
        let structured = StructuredFoveationEvidence::from_result(&result(1.0)).unwrap();
        assert_eq!(structured.origin(), VisualOrigin::Inferred);
    }

    #[test]
    fn execution_receipt_survives_structuring() {
        let structured = StructuredFoveationEvidence::from_result(&result(0.8)).unwrap();
        assert_eq!(structured.execution.requested_routing, RoutingStrategy::AlwaysOcr);
        assert_eq!(structured.execution.operation, VentralOperation::Ocr);
        assert!(!structured.execution.used_learned_model());
    }
}
