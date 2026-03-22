// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Data types for the foveation bridge between dorsal and ventral vision streams.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;

/// What the dorsal stream found interesting — a salient region to analyze.
#[derive(Debug, Clone)]
pub struct FoveationRequest {
    /// Unique request identifier.
    pub id: u64,
    /// High-resolution pixel crop (RGB or grayscale bytes).
    pub crop_pixels: Vec<u8>,
    /// Crop width in pixels.
    pub crop_width: u32,
    /// Crop height in pixels.
    pub crop_height: u32,
    /// Number of color channels (1 = grayscale, 3 = RGB).
    pub channels: usize,
    /// Grid row of the patch that triggered this request.
    pub grid_row: usize,
    /// Grid column of the patch that triggered this request.
    pub grid_col: usize,
    /// Surprise value from the SurpriseMap (priority ordering).
    pub surprise_value: f32,
    /// Source frame sequence number for temporal binding.
    pub frame_id: u64,
    /// Timestamp (microseconds) when the saliency was detected.
    pub timestamp_us: u64,
    /// Motion velocity at this patch [dx, dy] in pixels/frame.
    /// Used for predictive binding: when the ventral result arrives 100-200ms
    /// later, the cognitive loop compensates for object motion by projecting
    /// the semantic HV to the predicted current position.
    pub velocity: [f32; 2],
}

/// What the ventral stream recognized from a foveated region.
#[derive(Debug, Clone)]
pub struct FoveationResult {
    /// Corresponding request ID.
    pub request_id: u64,
    /// 16,384-dimensional HDC vector ready for GWT injection.
    pub semantic_hv: ContinuousHV,
    /// What was found in the crop.
    pub content: RecognizedContent,
    /// Recognition confidence (0.0–1.0).
    pub confidence: f32,
    /// Original grid row (spatial binding).
    pub grid_row: usize,
    /// Original grid col (spatial binding).
    pub grid_col: usize,
    /// Source frame sequence number (temporal binding anchor).
    pub source_frame_id: u64,
    /// Source timestamp in microseconds (temporal binding anchor).
    pub source_timestamp_us: u64,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
    /// Motion velocity at the source patch [dx, dy] in pixels/frame.
    /// The cognitive loop uses this to compute predicted current position:
    /// `predicted_pos = (grid_row, grid_col) + velocity * processing_time`.
    pub velocity: [f32; 2],
}

/// What was recognized in a foveated crop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecognizedContent {
    /// OCR result — text found in the region.
    Text(String),
    /// Object classification — label + raw embedding.
    Object { label: String, embedding: Vec<f32> },
    /// Visual question answering caption.
    Caption(String),
    /// Below confidence threshold or stub mode.
    Unknown,
}

/// Configuration for the foveation bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoveationConfig {
    /// Maximum concurrent in-flight foveation requests (default: 2).
    pub max_concurrent: usize,
    /// Bounded channel capacity for backpressure (default: 4).
    pub channel_depth: usize,
    /// Minimum surprise value to trigger foveation (default: 0.5).
    pub min_surprise_threshold: f32,
    /// Minimum time between dispatches in milliseconds (default: 50).
    pub cooldown_ms: u64,
    /// Maximum crop size in pixels before downscaling (default: 384*384).
    pub max_crop_pixels: usize,
    /// How to decide which ventral model to use.
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

/// How the ventral pipeline routes crop analysis.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Heuristic: text-like → OCR, objects → embed, fallback → VQA.
    #[default]
    Auto,
    /// SigLIP embedding only (fastest, ~100ms).
    AlwaysEmbed,
    /// OCR only (text-focused).
    AlwaysOcr,
    /// Moondream VQA only (most descriptive).
    AlwaysCaption,
    /// SigLIP + OCR + VQA cascade (most comprehensive, ~300ms).
    Full,
}

/// A salient region identified by the dorsal stream with pixel coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalientRegion {
    /// Grid row in the PatchGrid.
    pub grid_row: usize,
    /// Grid col in the PatchGrid.
    pub grid_col: usize,
    /// Surprise value (higher = more unexpected).
    pub surprise: f32,
    /// Pixel X coordinate of the region's top-left corner.
    pub pixel_x: usize,
    /// Pixel Y coordinate of the region's top-left corner.
    pub pixel_y: usize,
    /// Width in pixels.
    pub pixel_w: usize,
    /// Height in pixels.
    pub pixel_h: usize,
}

/// Telemetry from the foveation subsystem for CycleMetadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FoveationTelemetry {
    /// Number of requests currently queued.
    pub pending_count: usize,
    /// Number of requests currently being processed.
    pub in_flight_count: usize,
    /// Number of results ready for GWT injection.
    pub ready_count: usize,
    /// Total requests dispatched since startup.
    pub total_dispatched: u64,
    /// Total results received since startup.
    pub total_completed: u64,
    /// Average processing time in microseconds (exponential moving average).
    pub avg_processing_time_us: f32,
    /// Most recent result confidence (or 0.0 if none).
    pub last_confidence: f32,
}

/// Wraps a stored full-resolution frame for later cropping.
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    /// Raw pixel data.
    pub pixels: Vec<u8>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Number of channels.
    pub channels: usize,
    /// Frame sequence number.
    pub frame_id: u64,
    /// Capture timestamp in microseconds.
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
    fn test_recognized_content_variants() {
        let text = RecognizedContent::Text("STOP".to_string());
        let obj = RecognizedContent::Object {
            label: "cat".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
        };
        let caption = RecognizedContent::Caption("A red stop sign".to_string());
        let unknown = RecognizedContent::Unknown;

        // Verify Debug works for all variants
        assert!(!format!("{text:?}").is_empty());
        assert!(!format!("{obj:?}").is_empty());
        assert!(!format!("{caption:?}").is_empty());
        assert!(!format!("{unknown:?}").is_empty());
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
    fn test_salient_region_fields() {
        let region = SalientRegion {
            grid_row: 2,
            grid_col: 3,
            surprise: 0.8,
            pixel_x: 24,
            pixel_y: 16,
            pixel_w: 8,
            pixel_h: 8,
        };
        assert_eq!(region.pixel_x, 24);
        assert_eq!(region.pixel_y, 16);
        assert!((region.surprise - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_foveation_telemetry_default() {
        let tel = FoveationTelemetry::default();
        assert_eq!(tel.pending_count, 0);
        assert_eq!(tel.in_flight_count, 0);
        assert_eq!(tel.ready_count, 0);
        assert_eq!(tel.total_dispatched, 0);
        assert_eq!(tel.total_completed, 0);
        assert!((tel.avg_processing_time_us - 0.0).abs() < 1e-6);
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
