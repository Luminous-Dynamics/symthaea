// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Data types for the foveation bridge between dorsal and ventral vision streams.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;
use symthaea_vision_manifold::VisualObservationRef;

/// What the dorsal stream found interesting — a salient region to analyze.
#[derive(Debug, Clone)]
pub struct FoveationRequest {
    pub id: u64,
    pub crop_pixels: Vec<u8>,
    pub crop_width: u32,
    pub crop_height: u32,
    pub channels: usize,
    pub grid_row: usize,
    pub grid_col: usize,
    pub surprise_value: f32,
    pub frame_id: u64,
    pub timestamp_us: u64,
    /// Exact source observation when supplied by the capture owner. `None` means the legacy
    /// unproven frame path was used; downstream structured evidence must fail closed.
    pub source_observation: Option<VisualObservationRef>,
    pub velocity: [f32; 2],
}

/// What the ventral stream recognized from a foveated region.
#[derive(Debug, Clone)]
pub struct FoveationResult {
    pub request_id: u64,
    pub semantic_hv: ContinuousHV,
    pub content: RecognizedContent,
    pub confidence: f32,
    pub grid_row: usize,
    pub grid_col: usize,
    pub source_frame_id: u64,
    pub source_timestamp_us: u64,
    /// Exact source observation when the capture owner supplied one.
    pub source_observation: Option<VisualObservationRef>,
    /// What actually produced the semantic result, kept distinct from requested routing.
    pub execution: VentralExecutionReceipt,
    pub processing_time_us: u64,
    pub velocity: [f32; 2],
}

/// Backend that actually produced a ventral result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VentralExecutionKind {
    /// Built-in deterministic pixel-hash + JL backend.
    HashStubV1,
    /// SemanticVision executed an ONNX SigLIP session, but exact model bytes are not pinned.
    SemanticVisionOnnxUnpinned,
    /// SemanticVision returned its deterministic fallback embedding because no ONNX session ran.
    SemanticVisionDeterministicStub,
    /// Recognition failed and a random low-confidence fallback vector was emitted.
    ErrorFallbackRandom,
    /// Execution identity is unavailable. Consumers must not infer a stronger kind.
    Unknown,
}

/// Semantic operation that actually ran, independently of requested routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VentralOperation {
    Ocr,
    Embedding,
    Caption,
    Fallback,
    Unknown,
}

/// Execution receipt attached to every foveation result.
///
/// `requested_routing` records configuration intent. `operation` and `kind` record execution.
/// They are deliberately separate because current/future backends may degrade or route
/// differently from the requested strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VentralExecutionReceipt {
    pub kind: VentralExecutionKind,
    pub requested_routing: RoutingStrategy,
    pub operation: VentralOperation,
}

impl VentralExecutionReceipt {
    pub const fn new(
        kind: VentralExecutionKind,
        requested_routing: RoutingStrategy,
        operation: VentralOperation,
    ) -> Self {
        Self {
            kind,
            requested_routing,
            operation,
        }
    }

    /// True only when an ONNX learned backend actually executed.
    /// This still does not mean exact model bytes were pinned.
    pub const fn used_learned_model(self) -> bool {
        matches!(self.kind, VentralExecutionKind::SemanticVisionOnnxUnpinned)
    }

    /// v1 receipts intentionally do not establish exact model artifact identity.
    pub const fn exact_model_artifact_pinned(self) -> bool {
        false
    }
}

/// Error when capture provenance disagrees with the frame bytes it is intended to identify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameObservationError {
    FrameIdMismatch {
        frame_id: u64,
        observation_frame_id: u64,
    },
    TimestampMismatch {
        timestamp_us: u64,
        observation_timestamp_us: u64,
    },
}

impl std::fmt::Display for FrameObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FrameIdMismatch {
                frame_id,
                observation_frame_id,
            } => write!(
                f,
                "frame id {frame_id} does not match observation frame id {observation_frame_id}"
            ),
            Self::TimestampMismatch {
                timestamp_us,
                observation_timestamp_us,
            } => write!(
                f,
                "frame timestamp {timestamp_us} does not match observation timestamp {observation_timestamp_us}"
            ),
        }
    }
}

impl std::error::Error for FrameObservationError {}

/// What was recognized in a foveated crop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecognizedContent {
    Text(String),
    Object { label: String, embedding: Vec<f32> },
    Caption(String),
    Unknown,
}

/// Configuration for the foveation bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoveationConfig {
    pub max_concurrent: usize,
    pub channel_depth: usize,
    pub min_surprise_threshold: f32,
    pub cooldown_ms: u64,
    pub max_crop_pixels: usize,
    pub routing: RoutingStrategy,
}

impl Default for FoveationConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 2,
            channel_depth: 4,
            min_surprise_threshold: 0.5,
            cooldown_ms: 50,
            max_crop_pixels: 384 * 384,
            routing: RoutingStrategy::Auto,
        }
    }
}

/// Requested ventral routing strategy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    #[default]
    Auto,
    AlwaysEmbed,
    AlwaysOcr,
    AlwaysCaption,
    Full,
}

/// A salient region identified by the dorsal stream with pixel coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalientRegion {
    pub grid_row: usize,
    pub grid_col: usize,
    pub surprise: f32,
    pub pixel_x: usize,
    pub pixel_y: usize,
    pub pixel_w: usize,
    pub pixel_h: usize,
}

/// Telemetry from the foveation subsystem for CycleMetadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FoveationTelemetry {
    pub pending_count: usize,
    pub in_flight_count: usize,
    pub ready_count: usize,
    pub total_dispatched: u64,
    pub total_completed: u64,
    pub avg_processing_time_us: f32,
    pub last_confidence: f32,
}

/// Wraps a stored full-resolution frame for later cropping.
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub frame_id: u64,
    pub timestamp_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foveation_config_default() {
        let cfg = FoveationConfig::default();
        assert_eq!(cfg.max_concurrent, 2);
        assert_eq!(cfg.channel_depth, 4);
        assert!((cfg.min_surprise_threshold - 0.5).abs() < 1e-6);
        assert_eq!(cfg.cooldown_ms, 50);
        assert_eq!(cfg.max_crop_pixels, 384 * 384);
        assert_eq!(cfg.routing, RoutingStrategy::Auto);
    }

    #[test]
    fn test_routing_strategy_default() {
        assert_eq!(RoutingStrategy::default(), RoutingStrategy::Auto);
    }

    #[test]
    fn test_recognized_content_serde_roundtrip() {
        let content = RecognizedContent::Text("hello".to_string());
        let json = serde_json::to_string(&content).unwrap();
        let back: RecognizedContent = serde_json::from_str(&json).unwrap();
        match back {
            RecognizedContent::Text(s) => assert_eq!(s, "hello"),
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_foveation_config_serde_roundtrip() {
        let cfg = FoveationConfig {
            max_concurrent: 4,
            channel_depth: 8,
            min_surprise_threshold: 0.7,
            cooldown_ms: 100,
            max_crop_pixels: 256 * 256,
            routing: RoutingStrategy::Full,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: FoveationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_concurrent, 4);
        assert_eq!(back.routing, RoutingStrategy::Full);
    }

    #[test]
    fn execution_receipt_separates_request_from_execution() {
        let receipt = VentralExecutionReceipt::new(
            VentralExecutionKind::SemanticVisionOnnxUnpinned,
            RoutingStrategy::AlwaysCaption,
            VentralOperation::Embedding,
        );
        assert_eq!(receipt.requested_routing, RoutingStrategy::AlwaysCaption);
        assert_eq!(receipt.operation, VentralOperation::Embedding);
        assert!(receipt.used_learned_model());
        assert!(!receipt.exact_model_artifact_pinned());
    }

    #[test]
    fn test_frame_buffer_construction() {
        let fb = FrameBuffer {
            pixels: vec![128; 64 * 64 * 3],
            width: 64,
            height: 64,
            channels: 3,
            frame_id: 42,
            timestamp_us: 1_000_000,
        };
        assert_eq!(fb.pixels.len(), 64 * 64 * 3);
        assert_eq!(fb.frame_id, 42);
    }
}
