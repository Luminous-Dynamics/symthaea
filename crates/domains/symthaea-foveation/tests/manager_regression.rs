// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_foveation::{
    FoveationConfig, FoveationManager, FoveationResult, FrameBuffer, RecognizedContent,
    RoutingStrategy, VentralExecutionKind, VentralExecutionReceipt, VentralOperation,
};
use symthaea_core::hdc::ContinuousHV;

fn frame(id: u64, timestamp_us: u64, value: u8) -> FrameBuffer {
    FrameBuffer {
        pixels: vec![value; 64 * 64],
        width: 64,
        height: 64,
        channels: 1,
        frame_id: id,
        timestamp_us,
    }
}

#[test]
fn telemetry_starts_empty_and_tracks_pending() {
    let mut manager = FoveationManager::new(FoveationConfig::default(), 8);
    let initial = manager.telemetry();
    assert_eq!(initial.pending_count, 0);
    assert_eq!(initial.in_flight_count, 0);
    assert_eq!(initial.ready_count, 0);
    assert_eq!(initial.total_dispatched, 0);
    assert_eq!(initial.total_completed, 0);

    manager.on_frame(frame(1, 1_000_000, 128));
    manager.on_saliency(&[(1, 1, 0.7, [0.0, 0.0]), (2, 2, 0.8, [0.0, 0.0])]);
    assert_eq!(manager.telemetry().pending_count, 2);
}

#[test]
fn clear_pending_is_observable_through_public_api() {
    let mut manager = FoveationManager::new(FoveationConfig::default(), 8);
    manager.on_frame(frame(1, 1_000_000, 128));
    manager.on_saliency(&[
        (0, 0, 0.6, [0.0, 0.0]),
        (1, 1, 0.8, [0.0, 0.0]),
        (2, 2, 0.9, [0.0, 0.0]),
    ]);
    assert_eq!(manager.pending_count(), 3);
    manager.clear_pending();
    assert_eq!(manager.pending_count(), 0);
}

#[test]
fn new_frame_invalidates_old_pending_saliency() {
    let mut manager = FoveationManager::new(FoveationConfig::default(), 8);
    manager.on_frame(frame(1, 1_000_000, 128));
    manager.on_saliency(&[(1, 1, 0.9, [0.0, 0.0])]);
    assert_eq!(manager.pending_count(), 1);
    manager.on_frame(frame(2, 1_020_000, 129));
    assert_eq!(manager.pending_count(), 0);
}

#[test]
fn neuromodulation_retains_original_directionality() {
    let mut manager = FoveationManager::new(FoveationConfig::default(), 8);
    assert!((manager.effective_surprise_threshold() - 0.5).abs() < 1e-6);
    assert_eq!(manager.effective_max_concurrent(), 2);

    manager.modulate(1.5, 1.0);
    assert!((manager.effective_surprise_threshold() - 0.75).abs() < 1e-4);
    assert_eq!(manager.effective_max_concurrent(), 2);

    manager.modulate(0.5, 2.0);
    assert!((manager.effective_surprise_threshold() - 0.25).abs() < 1e-4);
    assert_eq!(manager.effective_max_concurrent(), 4);

    manager.modulate(1.0, 0.3);
    assert_eq!(manager.effective_max_concurrent(), 1);
}

#[test]
fn neuromodulated_threshold_still_controls_saliency_admission() {
    let mut manager = FoveationManager::new(FoveationConfig::default(), 8);
    manager.on_frame(frame(1, 1_000_000, 128));

    manager.modulate(2.0, 1.0);
    manager.on_saliency(&[(0, 0, 0.9, [0.0, 0.0])]);
    assert_eq!(manager.pending_count(), 0);

    manager.modulate(0.2, 1.0);
    manager.on_saliency(&[(0, 0, 0.3, [0.0, 0.0])]);
    assert_eq!(manager.pending_count(), 1);
}

#[test]
fn full_public_pipeline_produces_semantic_result() {
    let config = FoveationConfig {
        max_concurrent: 2,
        cooldown_ms: 0,
        min_surprise_threshold: 0.3,
        routing: RoutingStrategy::Auto,
        ..FoveationConfig::default()
    };
    let mut manager = FoveationManager::new(config, 8);
    manager.on_frame(frame(1, 1_000_000, 128));
    manager.on_saliency(&[(1, 1, 0.8, [0.25, -0.1])]);
    manager.tick(1_000_000);

    for i in 0..20 {
        if manager.ready_count() > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
        manager.tick(1_010_000 + i * 10_000);
    }

    let results = manager.drain_results();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].semantic_hv.dim(), 16_384);
    assert!(results[0].confidence >= 0.0 && results[0].confidence <= 1.0);
    assert_eq!(results[0].execution.kind, VentralExecutionKind::HashStubV1);
}

#[test]
fn predictive_position_helper_retains_motion_semantics() {
    let result = FoveationResult {
        request_id: 1,
        semantic_hv: ContinuousHV::zero(8),
        content: RecognizedContent::Unknown,
        confidence: 0.1,
        grid_row: 10,
        grid_col: 20,
        source_frame_id: 1,
        source_timestamp_us: 1_000,
        source_observation: None,
        execution: VentralExecutionReceipt::new(
            VentralExecutionKind::HashStubV1,
            RoutingStrategy::Auto,
            VentralOperation::Fallback,
        ),
        processing_time_us: 40_000,
        velocity: [1.0, -0.5],
    };

    let (row, col) = FoveationManager::predict_current_position(&result);
    assert_eq!((row, col), (9, 22));
}
