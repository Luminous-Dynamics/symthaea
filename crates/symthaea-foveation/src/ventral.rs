// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ventral pipeline: processes foveated crops into semantic HDC vectors.
//!
//! Runs inside the background thread. In stub mode (default), produces
//! deterministic hash-based embeddings for testing without model files.
//! When real perception models are available and the `perception` feature
//! is enabled, routes through SigLIP, OCR, and/or Moondream VQA.

use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

use crate::types::{FoveationRequest, FoveationResult, RecognizedContent, RoutingStrategy};

/// Ventral stream pipeline that converts pixel crops to semantic HDC vectors.
///
/// Dispatches to either a stub (hash-based) or real (SigLIP-backed) backend.
#[allow(private_interfaces)]
pub enum VentralPipeline {
    /// Hash-based stub backend (no model files required).
    Stub {
        jl_projector: StubJLProjector,
        routing: RoutingStrategy,
    },
    /// Real SigLIP-backed backend (requires `perception` feature + model files).
    #[cfg(feature = "perception")]
    Real {
        inner: real::RealVentralPipeline,
        routing: RoutingStrategy,
    },
}

impl VentralPipeline {
    /// Create a new ventral pipeline with the given routing strategy.
    ///
    /// When the `perception` feature is active, attempts to initialize the real
    /// SigLIP-backed pipeline first. Falls back to the stub pipeline on failure.
    pub fn new(routing: RoutingStrategy) -> Self {
        #[cfg(feature = "perception")]
        if let Some(pipeline) = Self::try_new_real(routing) {
            return pipeline;
        }

        #[cfg(feature = "perception")]
        tracing::info!("Ventral pipeline: using stub (real backend unavailable)");

        #[cfg(not(feature = "perception"))]
        tracing::info!("Ventral pipeline: using stub (perception feature disabled)");

        Self::Stub {
            jl_projector: StubJLProjector::new(HDC_DIMENSION, 42_700),
            routing,
        }
    }

    /// Attempt to construct the real SigLIP-backed ventral pipeline.
    /// Returns `None` if model initialization fails.
    #[cfg(feature = "perception")]
    fn try_new_real(routing: RoutingStrategy) -> Option<Self> {
        match real::RealVentralPipeline::new() {
            Ok(real_pipeline) => {
                tracing::info!("Ventral pipeline: using real SigLIP backend");
                Some(Self::Real {
                    inner: real_pipeline,
                    routing,
                })
            }
            Err(e) => {
                tracing::warn!("Real ventral pipeline init failed: {e}");
                None
            }
        }
    }

    /// Process a foveation request and return a result.
    ///
    /// In stub mode, produces a deterministic hash-based HDC vector
    /// derived from the crop content. In real mode, dispatches to SigLIP.
    /// The resulting 16,384D ContinuousHV is valid for GWT injection.
    pub fn process(&mut self, request: &FoveationRequest) -> FoveationResult {
        match self {
            Self::Stub {
                jl_projector,
                routing,
            } => process_stub(jl_projector, *routing, request),
            #[cfg(feature = "perception")]
            Self::Real { inner, .. } => inner.process(request),
        }
    }
}

/// Process a foveation request using the stub (hash-based) backend.
fn process_stub(
    projector: &StubJLProjector,
    routing: RoutingStrategy,
    request: &FoveationRequest,
) -> FoveationResult {
    let start = std::time::Instant::now();

    let (semantic_hv, content, confidence) = match routing {
        RoutingStrategy::AlwaysOcr => stub_ocr(projector, request),
        RoutingStrategy::AlwaysCaption => stub_caption(projector, request),
        RoutingStrategy::AlwaysEmbed => stub_embed(projector, request),
        RoutingStrategy::Full => stub_full(projector, request),
        RoutingStrategy::Auto => stub_auto(projector, request),
    };

    let elapsed = start.elapsed();

    FoveationResult {
        request_id: request.id,
        semantic_hv,
        content,
        confidence,
        grid_row: request.grid_row,
        grid_col: request.grid_col,
        source_frame_id: request.frame_id,
        source_timestamp_us: request.timestamp_us,
        processing_time_us: elapsed.as_micros() as u64,
        velocity: request.velocity,
    }
}

/// Stub auto-routing: uses pixel statistics to pick a "route".
fn stub_auto(
    projector: &StubJLProjector,
    request: &FoveationRequest,
) -> (ContinuousHV, RecognizedContent, f32) {
    let contrast = pixel_contrast(&request.crop_pixels);
    if contrast > 100.0 {
        stub_ocr(projector, request)
    } else {
        stub_embed(projector, request)
    }
}

/// Stub OCR: returns text content based on pixel hash.
fn stub_ocr(
    projector: &StubJLProjector,
    request: &FoveationRequest,
) -> (ContinuousHV, RecognizedContent, f32) {
    let hash = pixel_hash(&request.crop_pixels);
    let hv = projector.project_hash(hash);
    let content = RecognizedContent::Text(format!("stub_text_{:04x}", hash & 0xFFFF));
    (hv, content, 0.5)
}

/// Stub embedding: returns object content based on pixel hash.
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

/// Stub caption: returns VQA caption based on pixel hash.
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

/// Stub full cascade: combines all three routes.
fn stub_full(
    projector: &StubJLProjector,
    request: &FoveationRequest,
) -> (ContinuousHV, RecognizedContent, f32) {
    stub_auto(projector, request)
}

/// Simple pixel contrast measure (standard deviation proxy).
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

/// Deterministic hash of pixel data (djb2-like).
fn pixel_hash(pixels: &[u8]) -> u64 {
    let mut hash: u64 = 5381;
    let step = 16;
    for &byte in pixels.iter().step_by(step) {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    hash
}

/// JL projector that generates deterministic HDC vectors from hash seeds
/// or projects real model embeddings via seeded Rademacher matrix.
///
/// For stub mode: projects hash values to pseudo-random HDC vectors.
/// For real mode: projects 768D/1024D model embeddings to 16,384D HDC space
/// using a seeded Rademacher matrix (Johnson–Lindenstrauss lemma).
pub(crate) struct StubJLProjector {
    dim: usize,
    seed: u64,
}

impl StubJLProjector {
    fn new(dim: usize, seed: u64) -> Self {
        Self { dim, seed }
    }

    /// Project a real model embedding (e.g. 768D SigLIP) to an HDC vector.
    ///
    /// Uses a seeded Rademacher matrix (±1 entries via xorshift) to project
    /// the embedding to `self.dim` dimensions. O(dim × emb_len).
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

    /// Project a hash value to a deterministic HDC vector.
    fn project_hash(&self, hash: u64) -> ContinuousHV {
        let combined_seed = self.seed ^ hash;
        let mut state = combined_seed;
        let mut values = Vec::with_capacity(self.dim);

        for _ in 0..self.dim {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
            values.push(v);
        }

        ContinuousHV::from_vec(values).normalize()
    }
}

// ── Real perception backend (feature-gated) ─────────────────────────────
#[cfg(feature = "perception")]
mod real {
    use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};
    use symthaea_perception::semantic_vision::SemanticVision;

    use crate::types::{FoveationRequest, FoveationResult, RecognizedContent};

    /// Real ventral pipeline backed by SigLIP/OCR/Moondream models.
    ///
    /// Uses `SemanticVision` for 768D SigLIP embeddings, then projects to
    /// 16,384D HDC space via a seeded Rademacher JL projector.
    pub struct RealVentralPipeline {
        vision: SemanticVision,
        projector: super::StubJLProjector,
    }

    impl RealVentralPipeline {
        /// Create a new real ventral pipeline. Initializes the SemanticVision model.
        pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
            let mut vision = SemanticVision::new(1000);
            vision.initialize()?;
            Ok(Self {
                vision,
                projector: super::StubJLProjector::new(HDC_DIMENSION, 42_700),
            })
        }

        /// Process a foveation request using real perception models.
        pub fn process(&mut self, request: &FoveationRequest) -> FoveationResult {
            let start = std::time::Instant::now();

            let (content, confidence, semantic_hv) =
                if let Some(image) = Self::pixels_to_image(request) {
                    match self.vision.embed_image(&image) {
                        Ok(embedding) => {
                            let raw = embedding.vector.as_slice();
                            let hv = self.projector.project_embedding(raw);
                            let content = RecognizedContent::Object {
                                label: "real_embed".to_string(),
                                embedding: raw.to_vec(),
                            };
                            (content, 0.8, hv)
                        }
                        Err(_) => Self::fallback(request),
                    }
                } else {
                    Self::fallback(request)
                };

            let elapsed = start.elapsed();

            FoveationResult {
                request_id: request.id,
                semantic_hv,
                content,
                confidence,
                grid_row: request.grid_row,
                grid_col: request.grid_col,
                source_frame_id: request.frame_id,
                source_timestamp_us: request.timestamp_us,
                processing_time_us: elapsed.as_micros() as u64,
                velocity: request.velocity,
            }
        }

        fn pixels_to_image(request: &FoveationRequest) -> Option<image::DynamicImage> {
            if request.crop_width == 0 || request.crop_height == 0 {
                return None;
            }
            let gray = image::GrayImage::from_raw(
                request.crop_width as u32,
                request.crop_height as u32,
                request.crop_pixels.clone(),
            )?;
            Some(image::DynamicImage::ImageLuma8(gray))
        }

        fn fallback(request: &FoveationRequest) -> (RecognizedContent, f32, ContinuousHV) {
            let hv = ContinuousHV::random(HDC_DIMENSION, request.id);
            (RecognizedContent::Unknown, 0.1, hv)
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
            velocity: [0.0, 0.0],
        }
    }

    #[test]
    fn test_ventral_pipeline_auto_returns_valid_hv() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::Auto);
        let req = make_request(1, vec![128; 64]);
        let result = pipeline.process(&req);

        assert_eq!(result.request_id, 1);
        assert_eq!(result.semantic_hv.dim(), HDC_DIMENSION);
        assert!(result.semantic_hv.norm() > 0.0);
        assert!(result.semantic_hv.norm().is_finite());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.grid_row, 2);
        assert_eq!(result.grid_col, 3);
        assert_eq!(result.source_frame_id, 100);
    }

    #[test]
    fn test_ventral_pipeline_ocr_route() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::AlwaysOcr);
        let req = make_request(2, vec![200; 64]);
        let result = pipeline.process(&req);

        match &result.content {
            RecognizedContent::Text(s) => assert!(s.starts_with("stub_text_")),
            other => panic!("Expected Text, got {other:?}"),
        }
        assert_eq!(result.semantic_hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_ventral_pipeline_embed_route() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::AlwaysEmbed);
        let req = make_request(3, vec![100; 64]);
        let result = pipeline.process(&req);

        match &result.content {
            RecognizedContent::Object { label, embedding } => {
                assert!(label.starts_with("stub_object_"));
                assert!(!embedding.is_empty());
            }
            other => panic!("Expected Object, got {other:?}"),
        }
    }

    #[test]
    fn test_ventral_pipeline_caption_route() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::AlwaysCaption);
        let req = make_request(4, vec![50; 64]);
        let result = pipeline.process(&req);

        match &result.content {
            RecognizedContent::Caption(s) => assert!(s.contains("region at")),
            other => panic!("Expected Caption, got {other:?}"),
        }
    }

    #[test]
    fn test_ventral_pipeline_full_route() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::Full);
        let req = make_request(5, vec![128; 64]);
        let result = pipeline.process(&req);

        assert_eq!(result.semantic_hv.dim(), HDC_DIMENSION);
        assert!(result.confidence >= 0.0);
    }

    #[test]
    fn test_deterministic_output() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::AlwaysEmbed);
        let pixels = vec![42u8; 64];
        let req1 = make_request(10, pixels.clone());
        let req2 = make_request(11, pixels);

        let r1 = pipeline.process(&req1);
        let r2 = pipeline.process(&req2);

        let sim = r1.semantic_hv.similarity(&r2.semantic_hv);
        assert!(
            (sim - 1.0).abs() < 1e-4,
            "Same input should produce same HV: sim={sim}"
        );
    }

    #[test]
    fn test_different_inputs_different_hvs() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::AlwaysEmbed);
        let req1 = make_request(20, vec![0u8; 64]);
        let req2 = make_request(21, vec![255u8; 64]);

        let r1 = pipeline.process(&req1);
        let r2 = pipeline.process(&req2);

        let sim = r1.semantic_hv.similarity(&r2.semantic_hv);
        assert!(
            sim < 0.5,
            "Different inputs should produce different HVs: sim={sim}"
        );
    }

    #[test]
    fn test_processing_time_recorded() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::Auto);
        let req = make_request(30, vec![128; 64]);
        let result = pipeline.process(&req);

        assert!(
            result.processing_time_us < 1_000_000,
            "Should complete quickly"
        );
    }

    #[test]
    fn test_auto_routes_high_contrast_to_text() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::Auto);
        let high_contrast: Vec<u8> = (0..64).map(|i| if i % 2 == 0 { 0 } else { 255 }).collect();
        let req = make_request(40, high_contrast);
        let result = pipeline.process(&req);

        match &result.content {
            RecognizedContent::Text(_) => {}
            other => panic!("High contrast should route to Text, got {other:?}"),
        }
    }

    #[test]
    fn test_auto_routes_low_contrast_to_embed() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::Auto);
        let low_contrast = vec![128u8; 64];
        let req = make_request(41, low_contrast);
        let result = pipeline.process(&req);

        match &result.content {
            RecognizedContent::Object { .. } => {}
            other => panic!("Low contrast should route to Object, got {other:?}"),
        }
    }

    #[test]
    fn test_stub_jl_projector_dimension() {
        let proj = StubJLProjector::new(HDC_DIMENSION, 42);
        let hv = proj.project_hash(12345);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_stub_jl_projector_normalized() {
        let proj = StubJLProjector::new(HDC_DIMENSION, 42);
        let hv = proj.project_hash(99999);
        let norm = hv.norm();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Projected HV should be normalized, got norm={norm}"
        );
    }

    #[test]
    fn test_stub_jl_projector_deterministic() {
        let proj = StubJLProjector::new(HDC_DIMENSION, 42);
        let hv1 = proj.project_hash(777);
        let hv2 = proj.project_hash(777);
        let sim = hv1.similarity(&hv2);
        assert!((sim - 1.0).abs() < 1e-6, "Same hash → same HV");
    }

    #[test]
    fn test_pixel_contrast_zero_for_uniform() {
        let c = pixel_contrast(&vec![128; 100]);
        assert!(
            c < 0.01,
            "Uniform pixels should have zero contrast, got {c}"
        );
    }

    #[test]
    fn test_pixel_contrast_high_for_alternating() {
        let pixels: Vec<u8> = (0..100).map(|i| if i % 2 == 0 { 0 } else { 255 }).collect();
        let c = pixel_contrast(&pixels);
        assert!(
            c > 100.0,
            "Alternating 0/255 should have high contrast, got {c}"
        );
    }

    #[test]
    fn test_velocity_passthrough() {
        let mut pipeline = VentralPipeline::new(RoutingStrategy::Auto);
        let mut req = make_request(50, vec![128; 64]);
        req.velocity = [2.5, -1.3];
        let result = pipeline.process(&req);

        assert!((result.velocity[0] - 2.5).abs() < 1e-6);
        assert!((result.velocity[1] - (-1.3)).abs() < 1e-6);
    }

    #[test]
    fn test_pixel_contrast_empty() {
        let c = pixel_contrast(&[]);
        assert!((c - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_project_embedding_dimension() {
        let proj = StubJLProjector::new(HDC_DIMENSION, 42);
        let embedding = vec![0.1f32; 768];
        let hv = proj.project_embedding(&embedding);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_project_embedding_normalized() {
        let proj = StubJLProjector::new(HDC_DIMENSION, 42);
        let embedding: Vec<f32> = (0..768).map(|i| (i as f32 / 768.0) * 2.0 - 1.0).collect();
        let hv = proj.project_embedding(&embedding);
        let norm = hv.norm();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Projected embedding should be normalized, got norm={norm}"
        );
    }

    #[test]
    fn test_project_embedding_deterministic() {
        let proj = StubJLProjector::new(HDC_DIMENSION, 42);
        let embedding = vec![0.5f32; 768];
        let hv1 = proj.project_embedding(&embedding);
        let hv2 = proj.project_embedding(&embedding);
        let sim = hv1.similarity(&hv2);
        assert!((sim - 1.0).abs() < 1e-6, "Same embedding → same HV");
    }

    #[test]
    fn test_project_embedding_different_inputs_diverge() {
        let proj = StubJLProjector::new(HDC_DIMENSION, 42);
        let emb_a = vec![1.0f32; 768];
        let emb_b = vec![-1.0f32; 768];
        let hv_a = proj.project_embedding(&emb_a);
        let hv_b = proj.project_embedding(&emb_b);
        let sim = hv_a.similarity(&hv_b);
        assert!(
            sim < 0.0,
            "Opposite embeddings should produce anti-correlated HVs: sim={sim}"
        );
    }

    #[test]
    fn test_project_embedding_empty_returns_zero() {
        let proj = StubJLProjector::new(HDC_DIMENSION, 42);
        let hv = proj.project_embedding(&[]);
        assert_eq!(hv.dim(), HDC_DIMENSION);
        assert!(hv.norm() < 1e-6);
    }
}
