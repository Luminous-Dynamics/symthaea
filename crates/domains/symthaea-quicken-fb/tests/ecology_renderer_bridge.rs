// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::time::Duration;

use symthaea_boot_protocol::{BootHealth, BootPhase, BootSnapshot};
use symthaea_boot_render_projection::RenderSegment;
use symthaea_quicken_fb::mycelium::MycelialNetwork;
use symthaea_quicken_fb::renderer_bridge::{
    EcologyFallbackReason, EcologyRenderOutcome, EcologyRendererBridge, RenderedEcologyFrame,
};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 54;
const STEP_MS: u32 = 250;

fn snapshot(sequence: u64, phase: BootPhase, health: BootHealth) -> BootSnapshot {
    let mut snapshot = BootSnapshot::new(
        sequence,
        Duration::from_millis(sequence.saturating_mul(100)),
        phase,
    );
    snapshot.health = health;
    snapshot
}

fn rendered(outcome: EcologyRenderOutcome) -> RenderedEcologyFrame {
    match outcome {
        EcologyRenderOutcome::Rendered(frame) => frame,
        EcologyRenderOutcome::Fallback(reason) => {
            panic!("unexpected ecology fallback: {}", reason.as_str())
        }
    }
}

#[test]
fn identical_seed_and_authoritative_trace_replay_identically() {
    let trace = [
        snapshot(1, BootPhase::Kernel, BootHealth::Normal),
        snapshot(2, BootPhase::Services, BootHealth::Degraded),
        snapshot(3, BootPhase::Session, BootHealth::Normal),
        snapshot(4, BootPhase::Ready, BootHealth::Normal),
    ];

    let mut left = EcologyRendererBridge::new(WIDTH, HEIGHT, "semantic-replay-seed").unwrap();
    let mut right = EcologyRendererBridge::new(WIDTH, HEIGHT, "semantic-replay-seed").unwrap();
    let mut left_pixels = vec![0u32; (WIDTH * HEIGHT) as usize];
    let mut right_pixels = vec![0u32; (WIDTH * HEIGHT) as usize];

    for item in &trace {
        let left_frame = rendered(left.render_snapshot(item, STEP_MS, &mut left_pixels));
        let right_frame = rendered(right.render_snapshot(item, STEP_MS, &mut right_pixels));

        assert_eq!(left_frame, right_frame);
        assert_eq!(left_pixels, right_pixels);
    }
}

#[test]
fn presentation_never_reverses_progress_or_smooths_bad_health_to_healthy() {
    let mut bridge = EcologyRendererBridge::new(WIDTH, HEIGHT, "monotonic-seed").unwrap();
    let mut pixels = vec![0u32; (WIDTH * HEIGHT) as usize];
    let mut previous_projection = 0u32;

    let degraded = snapshot(10, BootPhase::Services, BootHealth::Degraded);
    for _ in 0..4 {
        let frame = rendered(bridge.render_snapshot(&degraded, STEP_MS, &mut pixels));
        assert_eq!(frame.health, BootHealth::Degraded);
        assert!(frame.projection.elapsed_ms >= previous_projection);
        previous_projection = frame.projection.elapsed_ms;
    }

    let failed = snapshot(11, BootPhase::Services, BootHealth::Failed);
    for _ in 0..4 {
        let frame = rendered(bridge.render_snapshot(&failed, STEP_MS, &mut pixels));
        assert_eq!(frame.health, BootHealth::Failed);
        assert!(frame.projection.elapsed_ms >= previous_projection);
        previous_projection = frame.projection.elapsed_ms;
    }
}

#[test]
fn lineage_reset_preserves_visual_progress_for_same_or_later_truth() {
    let mut bridge = EcologyRendererBridge::new(WIDTH, HEIGHT, "lineage-continuity-seed").unwrap();
    let mut pixels = vec![0u32; (WIDTH * HEIGHT) as usize];

    let graphics = snapshot(50, BootPhase::Graphics, BootHealth::Normal);
    let mut previous_projection = 0u32;
    for _ in 0..4 {
        previous_projection = rendered(bridge.render_snapshot(&graphics, STEP_MS, &mut pixels))
            .projection
            .elapsed_ms;
    }
    assert!(previous_projection > 0);

    bridge.reset_semantics();

    // A replacement observation lineage starts its sequence over, but the
    // machine has not rebooted and presentation must not rewind.
    let replacement = snapshot(1, BootPhase::Graphics, BootHealth::Normal);
    let frame = rendered(bridge.render_snapshot(&replacement, STEP_MS, &mut pixels));
    assert!(frame.projection.elapsed_ms >= previous_projection);
}

#[test]
fn regressed_replacement_lineage_falls_back_instead_of_rewinding() {
    let mut bridge = EcologyRendererBridge::new(WIDTH, HEIGHT, "lineage-regression-seed").unwrap();
    let mut pixels = vec![0u32; (WIDTH * HEIGHT) as usize];

    let graphics = snapshot(60, BootPhase::Graphics, BootHealth::Normal);
    let previous = rendered(bridge.render_snapshot(&graphics, STEP_MS, &mut pixels));
    assert!(previous.projection.elapsed_ms > 0);

    bridge.reset_semantics();
    let regressed = snapshot(1, BootPhase::Network, BootHealth::Normal);

    assert_eq!(
        bridge.render_snapshot(&regressed, STEP_MS, &mut pixels),
        EcologyRenderOutcome::Fallback(EcologyFallbackReason::PresentationRejected)
    );
}

#[test]
fn ecology_cannot_present_terminal_handoff_before_authoritative_ready() {
    let mut bridge = EcologyRendererBridge::new(WIDTH, HEIGHT, "ready-gate-seed").unwrap();
    let mut pixels = vec![0u32; (WIDTH * HEIGHT) as usize];

    let session = snapshot(20, BootPhase::Session, BootHealth::Normal);
    let mut last_session = None;
    for _ in 0..16 {
        let frame = rendered(bridge.render_snapshot(&session, STEP_MS, &mut pixels));
        assert!(!frame.handoff_ready);
        assert_eq!(frame.projection.segment, RenderSegment::PreHandoff);
        last_session = Some(frame);
    }
    let session_frame = last_session.unwrap();
    assert_eq!(session_frame.projection.elapsed_ms, bridge.layout().pre_handoff_ms);

    let ready = snapshot(21, BootPhase::Ready, BootHealth::Normal);
    let ready_frame = rendered(bridge.render_snapshot(&ready, STEP_MS, &mut pixels));
    assert!(ready_frame.handoff_ready);
    assert_eq!(ready_frame.projection.segment, RenderSegment::Handoff);
}

#[test]
fn bad_frame_buffer_requests_legacy_fallback_without_propagating() {
    let mut bridge = EcologyRendererBridge::new(WIDTH, HEIGHT, "fallback-seed").unwrap();
    let boot = snapshot(30, BootPhase::Graphics, BootHealth::Normal);
    let mut undersized = vec![0u32; (WIDTH * HEIGHT) as usize - 1];

    assert_eq!(
        bridge.render_snapshot(&boot, STEP_MS, &mut undersized),
        EcologyRenderOutcome::Fallback(EcologyFallbackReason::BufferTooSmall)
    );

    // The real fallback path remains independent of ecology state and can render
    // immediately after the ecology frame is rejected. No boot/session authority
    // or renderer error is propagated through this visual substitution.
    let mut legacy = MycelialNetwork::new(WIDTH, HEIGHT, "fallback-seed");
    legacy.grow(0.25, 1.0);
    let mut fallback_pixels = vec![0u32; (WIDTH * HEIGHT) as usize];
    legacy.render(&mut fallback_pixels);
    assert!(fallback_pixels.iter().any(|pixel| *pixel != 0));
}

#[test]
fn invalid_authoritative_snapshot_is_visual_failure_only() {
    let mut bridge = EcologyRendererBridge::new(WIDTH, HEIGHT, "invalid-snapshot-seed").unwrap();
    let mut pixels = vec![0u32; (WIDTH * HEIGHT) as usize];
    let mut invalid = snapshot(40, BootPhase::Services, BootHealth::Normal);
    invalid.protocol_version = u16::MAX;

    assert_eq!(
        bridge.render_snapshot(&invalid, STEP_MS, &mut pixels),
        EcologyRenderOutcome::Fallback(EcologyFallbackReason::SnapshotRejected)
    );
}
