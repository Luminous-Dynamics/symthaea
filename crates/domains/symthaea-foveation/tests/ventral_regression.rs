// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_foveation::ventral::VentralPipeline;
use symthaea_foveation::{
    FoveationRequest, RecognizedContent, RoutingStrategy, VentralExecutionKind, VentralOperation,
};

fn request(id: u64, pixels: Vec<u8>) -> FoveationRequest {
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
fn all_requested_routes_return_valid_hdc_results() {
    for route in [
        RoutingStrategy::Auto,
        RoutingStrategy::AlwaysEmbed,
        RoutingStrategy::AlwaysOcr,
        RoutingStrategy::AlwaysCaption,
        RoutingStrategy::Full,
    ] {
        let mut pipeline = VentralPipeline::new(route);
        let result = pipeline.process(&request(1, vec![128; 64]));
        assert_eq!(result.semantic_hv.dim(), 16_384, "route={route:?}");
        assert!(result.semantic_hv.norm().is_finite());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.execution.requested_routing, route);
    }
}

#[test]
fn explicit_stub_routes_report_actual_operation() {
    let mut ocr = VentralPipeline::new(RoutingStrategy::AlwaysOcr);
    let ocr_result = ocr.process(&request(2, vec![200; 64]));
    assert!(matches!(ocr_result.content, RecognizedContent::Text(_)));
    assert_eq!(ocr_result.execution.operation, VentralOperation::Ocr);

    let mut embed = VentralPipeline::new(RoutingStrategy::AlwaysEmbed);
    let embed_result = embed.process(&request(3, vec![100; 64]));
    assert!(matches!(embed_result.content, RecognizedContent::Object { .. }));
    assert_eq!(embed_result.execution.operation, VentralOperation::Embedding);

    let mut caption = VentralPipeline::new(RoutingStrategy::AlwaysCaption);
    let caption_result = caption.process(&request(4, vec![50; 64]));
    assert!(matches!(caption_result.content, RecognizedContent::Caption(_)));
    assert_eq!(caption_result.execution.operation, VentralOperation::Caption);
}

#[test]
fn auto_routing_receipt_tracks_actual_branch() {
    let mut pipeline = VentralPipeline::new(RoutingStrategy::Auto);
    let high_contrast: Vec<u8> = (0..64).map(|i| if i % 2 == 0 { 0 } else { 255 }).collect();
    let high = pipeline.process(&request(40, high_contrast));
    assert_eq!(high.execution.operation, VentralOperation::Ocr);

    let low = pipeline.process(&request(41, vec![128; 64]));
    assert_eq!(low.execution.operation, VentralOperation::Embedding);
}

#[test]
fn default_build_reports_hash_stub_not_learned_inference() {
    let mut pipeline = VentralPipeline::new(RoutingStrategy::AlwaysEmbed);
    let result = pipeline.process(&request(5, vec![128; 64]));
    assert_eq!(result.execution.kind, VentralExecutionKind::HashStubV1);
    assert!(!result.execution.used_learned_model());
    assert!(!result.execution.exact_model_artifact_pinned());
}

#[test]
fn equal_pixels_produce_equal_stub_hvs() {
    let mut pipeline = VentralPipeline::new(RoutingStrategy::AlwaysEmbed);
    let pixels = vec![42u8; 64];
    let a = pipeline.process(&request(10, pixels.clone()));
    let b = pipeline.process(&request(11, pixels));
    assert!((a.semantic_hv.similarity(&b.semantic_hv) - 1.0).abs() < 1e-4);
}

#[test]
fn substantially_different_pixels_diverge_in_stub_hdc() {
    let mut pipeline = VentralPipeline::new(RoutingStrategy::AlwaysEmbed);
    let a = pipeline.process(&request(20, vec![0u8; 64]));
    let b = pipeline.process(&request(21, vec![255u8; 64]));
    assert!(a.semantic_hv.similarity(&b.semantic_hv) < 0.5);
}

#[test]
fn processing_time_and_velocity_are_preserved() {
    let mut pipeline = VentralPipeline::new(RoutingStrategy::Auto);
    let mut req = request(30, vec![128; 64]);
    req.velocity = [2.5, -1.3];
    let result = pipeline.process(&req);
    assert!(result.processing_time_us < 1_000_000);
    assert!((result.velocity[0] - 2.5).abs() < 1e-6);
    assert!((result.velocity[1] + 1.3).abs() < 1e-6);
}
