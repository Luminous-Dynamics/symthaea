// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ventral pipeline: processes foveated crops into semantic HDC vectors.
//!
//! Execution receipts distinguish requested routing from the backend and operation that
//! actually produced each result. This matters because SemanticVision can initialize while
//! falling back to deterministic embeddings when ONNX model files are absent.

use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

use crate::types::{
    FoveationRequest, FoveationResult, RecognizedContent, RoutingStrategy, VentralExecutionKind,
    VentralExecutionReceipt, VentralOperation,
};

#[allow(private_interfaces)]
pub enum VentralPipeline {
    Stub {
        jl_projector: StubJLProjector,
        routing: RoutingStrategy,
    },
    #[cfg(feature = "perception")]
    Real {
        inner: real::RealVentralPipeline,
        routing: RoutingStrategy,
    },
}

impl VentralPipeline {
    pub fn new(routing: RoutingStrategy) -> Self {
        #[cfg(feature = "perception")]
        if let Some(pipeline) = Self::try_new_real(routing) {
            return pipeline;
        }

        #[cfg(feature = "perception")]
        tracing::info!("Ventral pipeline: using hash stub (SemanticVision unavailable)");

        #[cfg(not(feature = "perception"))]
        tracing::info!("Ventral pipeline: using hash stub (perception feature disabled)");

        Self::Stub {
            jl_projector: StubJLProjector::new(HDC_DIMENSION, 42_700),
            routing,
        }
    }

    #[cfg(feature = "perception")]
    fn try_new_real(routing: RoutingStrategy) -> Option<Self> {
        match real::RealVentralPipeline::new() {
            Ok(real_pipeline) => {
                tracing::info!(
                    "Ventral pipeline: SemanticVision initialized; per-result receipt records ONNX vs deterministic fallback"
                );
                Some(Self::Real {
                    inner: real_pipeline,
                    routing,
                })
            }
            Err(e) => {
                tracing::warn!("SemanticVision ventral initialization failed: {e}");
                None
            }
        }
    }

    pub fn process(&mut self, request: &FoveationRequest) -> FoveationResult {
        match self {
            Self::Stub {
                jl_projector,
                routing,
            } => process_stub(jl_projector, *routing, request),
            #[cfg(feature = "perception")]
            Self::Real { inner, routing } => inner.process(request, *routing),
        }
    }
}

fn process_stub(
    projector: &StubJLProjector,
    routing: RoutingStrategy,
    request: &FoveationRequest,
) -> FoveationResult {
    let start = std::time::Instant::now();

    let (semantic_hv, content, confidence, operation) = match routing {
        RoutingStrategy::AlwaysOcr => {
            let (hv, content, confidence) = stub_ocr(projector, request);
            (hv, content, confidence, VentralOperation::Ocr)
        }
        RoutingStrategy::AlwaysCaption => {
            let (hv, content, confidence) = stub_caption(projector, request);
            (hv, content, confidence, VentralOperation::Caption)
        }
        RoutingStrategy::AlwaysEmbed => {
            let (hv, content, confidence) = stub_embed(projector, request);
            (hv, content, confidence, VentralOperation::Embedding)
        }
        RoutingStrategy::Full | RoutingStrategy::Auto => stub_auto(projector, request),
    };

    FoveationResult {
        request_id: request.id,
        semantic_hv,
        content,
        confidence,
        grid_row: request.grid_row,
        grid_col: request.grid_col,
        source_frame_id: request.frame_id,
        source_timestamp_us: request.timestamp_us,
        source_observation: request.source_observation,
        execution: VentralExecutionReceipt::new(
            VentralExecutionKind::HashStubV1,
            routing,
            operation,
        ),
        processing_time_us: start.elapsed().as_micros() as u64,
        velocity: request.velocity,
    }
}

fn stub_auto(
    projector: &StubJLProjector,
    request: &FoveationRequest,
) -> (ContinuousHV, RecognizedContent, f32, VentralOperation) {
    if pixel_contrast(&request.crop_pixels) > 100.0 {
        let (hv, content, confidence) = stub_ocr(projector, request);
        (hv, content, confidence, VentralOperation::Ocr)
    } else {
        let (hv, content, confidence) = stub_embed(projector, request);
        (hv, content, confidence, VentralOperation::Embedding)
    }
}

fn stub_ocr(
    projector: &StubJLProjector,
    request: &FoveationRequest,
) -> (ContinuousHV, RecognizedContent, f32) {
    let hash = pixel_hash(&request.crop_pixels);
    let hv = projector.project_hash(hash);
    let content = RecognizedContent::Text(format!("stub_text_{:04x}", hash & 0xFFFF));
    (hv, content, 0.5)
}

fn stub_embed(
    projector: &StubJLProjector,
    request: &FoveationRequest,
) -> (ContinuousHV, RecognizedContent, f32) {
    let hash = pixel_hash(&request.crop_pixels);
    let hv = projector.project_hash(hash);
    let content = RecognizedContent::Object {
        label: format!("stub_object_{:04x}", hash & 0xFFFF),
        embedding: vec![0.0; 8],
    };
    (hv, content, 0.4)
}

fn stub_caption(
    projector: &StubJLProjector,
    request: &FoveationRequest,
) -> (ContinuousHV, RecognizedContent, f32) {
    let hash = pixel_hash(&request.crop_pixels);
    let hv = projector.project_hash(hash);
    let content = RecognizedContent::Caption(format!(
        "A region at ({},{}) with hash {:04x}",
        request.grid_row,
        request.grid_col,
        hash & 0xFFFF
    ));
    (hv, content, 0.3)
}

fn pixel_contrast(pixels: &[u8]) -> f32 {
    if pixels.is_empty() {
        return 0.0;
    }
    let mean = pixels.iter().map(|&v| v as f32).sum::<f32>() / pixels.len() as f32;
    let variance = pixels
        .iter()
        .map(|&v| {
            let d = v as f32 - mean;
            d * d
        })
        .sum::<f32>()
        / pixels.len() as f32;
    variance.sqrt()
}

fn pixel_hash(pixels: &[u8]) -> u64 {
    let mut hash: u64 = 5381;
    for &byte in pixels.iter().step_by(16) {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    hash
}

pub(crate) struct StubJLProjector {
    dim: usize,
    seed: u64,
}

impl StubJLProjector {
    fn new(dim: usize, seed: u64) -> Self {
        Self { dim, seed }
    }

    #[allow(dead_code)]
    pub(crate) fn project_embedding(&self, embedding: &[f32]) -> ContinuousHV {
        if embedding.is_empty() {
            return ContinuousHV::zero(self.dim);
        }
        let emb_len = embedding.len();
        let mut values = Vec::with_capacity(self.dim);
        let inv_sqrt = 1.0 / (emb_len as f32).sqrt();

        for i in 0..self.dim {
            let mut state = self.seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let mut sum = 0.0f32;
            for (j, &e) in embedding.iter().enumerate() {
                state ^= (j as u64).wrapping_mul(0x517CC1B727220A95);
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let sign = if state & 1 == 0 { 1.0f32 } else { -1.0 };
                sum += sign * e;
            }
            values.push(sum * inv_sqrt);
        }

        ContinuousHV::from_vec(values).normalize()
    }

    fn project_hash(&self, hash: u64) -> ContinuousHV {
        let mut state = self.seed ^ hash;
        let mut values = Vec::with_capacity(self.dim);

        for _ in 0..self.dim {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.push((state as f32 / u64::MAX as f32) * 2.0 - 1.0);
        }

        ContinuousHV::from_vec(values).normalize()
    }
}

#[cfg(feature = "perception")]
mod real {
    use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};
    use symthaea_perception::semantic_vision::SemanticVision;

    use crate::types::{
        FoveationRequest, FoveationResult, RecognizedContent, RoutingStrategy,
        VentralExecutionKind, VentralExecutionReceipt, VentralOperation,
    };

    pub struct RealVentralPipeline {
        vision: SemanticVision,
        projector: super::StubJLProjector,
    }

    impl RealVentralPipeline {
        pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
            let mut vision = SemanticVision::new(1000);
            vision.initialize()?;
            Ok(Self {
                vision,
                projector: super::StubJLProjector::new(HDC_DIMENSION, 42_700),
            })
        }

        /// Current archived SemanticVision integration always requests an image embedding.
        /// The receipt therefore reports `Embedding` even when the requested routing asks for
        /// OCR/caption/full. This makes the current implementation gap observable rather than
        /// silently describing configuration intent as execution fact.
        pub fn process(
            &mut self,
            request: &FoveationRequest,
            requested_routing: RoutingStrategy,
        ) -> FoveationResult {
            let start = std::time::Instant::now();
            let using_onnx = self.vision.is_using_onnx();

            let (content, confidence, semantic_hv, kind, operation) =
                if let Some(image) = Self::pixels_to_image(request) {
                    match self.vision.embed_image(&image) {
                        Ok(embedding) => {
                            let raw = embedding.vector.as_slice();
                            let hv = self.projector.project_embedding(raw);
                            let content = RecognizedContent::Object {
                                label: "semantic_embedding".to_string(),
                                embedding: raw.to_vec(),
                            };
                            let kind = if using_onnx {
                                VentralExecutionKind::SemanticVisionOnnxUnpinned
                            } else {
                                VentralExecutionKind::SemanticVisionDeterministicStub
                            };
                            (content, 0.8, hv, kind, VentralOperation::Embedding)
                        }
                        Err(_) => Self::fallback(request),
                    }
                } else {
                    Self::fallback(request)
                };

            FoveationResult {
                request_id: request.id,
                semantic_hv,
                content,
                confidence,
                grid_row: request.grid_row,
                grid_col: request.grid_col,
                source_frame_id: request.frame_id,
                source_timestamp_us: request.timestamp_us,
                source_observation: request.source_observation,
                execution: VentralExecutionReceipt::new(kind, requested_routing, operation),
                processing_time_us: start.elapsed().as_micros() as u64,
                velocity: request.velocity,
            }
        }

        fn pixels_to_image(request: &FoveationRequest) -> Option<image::DynamicImage> {
            if request.crop_width == 0 || request.crop_height == 0 {
                return None;
            }
            let gray = image::GrayImage::from_raw(
                request.crop_width,
                request.crop_height,
                request.crop_pixels.clone(),
            )?;
            Some(image::DynamicImage::ImageLuma8(gray))
        }

        fn fallback(
            request: &FoveationRequest,
        ) -> (
            RecognizedContent,
            f32,
            ContinuousHV,
            VentralExecutionKind,
            VentralOperation,
        ) {
            (
                RecognizedContent::Unknown,
                0.1,
                ContinuousHV::random(HDC_DIMENSION, request.id),
                VentralExecutionKind::ErrorFallbackRandom,
                VentralOperation::Fallback,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: u64, pixels: Vec<u8>) -> FoveationRequest {
        FoveationRequest {
            id,
            crop_pixels: pixels,
            crop_width: 8,
            crop_height: 8,
            channels: 1,
            grid_row: 2,
            grid_col: 3,
            surprise_value: 0.8,
            frame_id: 100,
            timestamp_us: 50_000,
            source_observation: None,
            velocity: [0.0, 0.0],
        }
    }

    #[test]
    fn auto_returns_valid_hv_and_execution_receipt() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::Auto);
        let result = pipeline.process(&make_request(1, vec![128; 64]));
        assert_eq!(result.request_id, 1);
        assert_eq!(result.semantic_hv.dim(), HDC_DIMENSION);
        assert!(result.semantic_hv.norm().is_finite());
        assert_eq!(result.execution.kind, VentralExecutionKind::HashStubV1);
        assert_eq!(result.execution.requested_routing, RoutingStrategy::Auto);
        assert_eq!(result.execution.operation, VentralOperation::Embedding);
    }

    #[test]
    fn requested_ocr_reports_ocr_execution() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::AlwaysOcr);
        let result = pipeline.process(&make_request(2, vec![200; 64]));
        assert_eq!(result.execution.operation, VentralOperation::Ocr);
        assert!(matches!(result.content, RecognizedContent::Text(_)));
    }

    #[test]
    fn requested_caption_reports_caption_execution() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::AlwaysCaption);
        let result = pipeline.process(&make_request(4, vec![50; 64]));
        assert_eq!(result.execution.operation, VentralOperation::Caption);
        assert!(matches!(result.content, RecognizedContent::Caption(_)));
    }

    #[test]
    fn auto_reports_actual_operation() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::Auto);
        let high_contrast: Vec<u8> = (0..64).map(|i| if i % 2 == 0 { 0 } else { 255 }).collect();
        let high = pipeline.process(&make_request(40, high_contrast));
        assert_eq!(high.execution.operation, VentralOperation::Ocr);

        let low = pipeline.process(&make_request(41, vec![128; 64]));
        assert_eq!(low.execution.operation, VentralOperation::Embedding);
    }

    #[test]
    fn source_observation_passthrough_is_exact() {
        use symthaea_vision_manifold::{VisualCaptureClock, VisualObservationRef, VisualStreamRef};
        let mut pipeline = VentralPipeline::new(RoutingStrategy::AlwaysEmbed);
        let mut request = make_request(6, vec![100; 64]);
        let observation = VisualObservationRef::new(
            VisualStreamRef::new(9, 3).unwrap(),
            100,
            50_000,
            VisualCaptureClock::StreamMonotonic,
        );
        request.source_observation = Some(observation);
        let result = pipeline.process(&request);
        assert_eq!(result.source_observation, Some(observation));
    }

    #[test]
    fn deterministic_output_for_same_pixels() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::AlwaysEmbed);
        let pixels = vec![42u8; 64];
        let r1 = pipeline.process(&make_request(10, pixels.clone()));
        let r2 = pipeline.process(&make_request(11, pixels));
        assert!((r1.semantic_hv.similarity(&r2.semantic_hv) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn different_inputs_produce_different_hvs() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::AlwaysEmbed);
        let r1 = pipeline.process(&make_request(20, vec![0u8; 64]));
        let r2 = pipeline.process(&make_request(21, vec![255u8; 64]));
        assert!(r1.semantic_hv.similarity(&r2.semantic_hv) < 0.5);
    }

    #[test]
    fn velocity_passthrough() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::Auto);
        let mut req = make_request(50, vec![128; 64]);
        req.velocity = [2.5, -1.3];
        let result = pipeline.process(&req);
        assert!((result.velocity[0] - 2.5).abs() < 1e-6);
        assert!((result.velocity[1] + 1.3).abs() < 1e-6);
    }

    #[test]
    fn projector_is_normalized_and_deterministic() {
        let proj = StubJLProjector::new(HDC_DIMENSION, 42);
        let hv1 = proj.project_hash(777);
        let hv2 = proj.project_hash(777);
        assert_eq!(hv1.dim(), HDC_DIMENSION);
        assert!((hv1.norm() - 1.0).abs() < 0.01);
        assert!((hv1.similarity(&hv2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn empty_embedding_projects_to_zero() {
        let proj = StubJLProjector::new(HDC_DIMENSION, 42);
        let hv = proj.project_embedding(&[]);
        assert_eq!(hv.dim(), HDC_DIMENSION);
        assert!(hv.norm() < 1e-6);
    }
}
