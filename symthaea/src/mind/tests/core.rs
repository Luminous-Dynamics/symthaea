// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::super::*;
use symthaea_core::hdc::ContinuousHV;

#[test]
fn test_mind_creation() {
    let mind = ContinuousMind::default();
    assert_eq!(mind.state.tick, 0);
    assert!(!mind.state.is_active);
}

#[test]
fn test_mind_tick() {
    let mut mind = ContinuousMind::default();
    mind.activate();
    mind.tick();
    assert_eq!(mind.state.tick, 1);
}

#[test]
fn test_perception() {
    let mut mind = ContinuousMind::default();
    mind.perceive(ContinuousHV::random(512, 42));
    mind.tick();
    assert_eq!(mind.working_memory.len(), 1);
}

#[test]
fn test_goal_setting() {
    let mut mind = ContinuousMind::default();
    mind.set_goal("Test goal", ContinuousHV::random(512, 42), 1.0);
    mind.tick();
    assert!(!mind.active_goals().is_empty());
}

#[test]
fn test_consciousness_update() {
    let mut mind = ContinuousMind::default();

    for i in 0..5 {
        mind.perceive(ContinuousHV::random(512, 42 + i as u64));
    }

    for _ in 0..5 {
        mind.tick();
    }

    assert!(mind.state.consciousness_level > 0.0);
}

#[test]
fn test_current_thought_nonzero_after_perception() {
    let mut mind = ContinuousMind::default();
    mind.activate();

    // Before any perception, current_thought is zero
    assert!(
        mind.state.current_thought.norm() < f32::EPSILON,
        "current_thought should start as zero"
    );

    mind.perceive(ContinuousHV::random(512, 42));
    mind.tick();

    // After perception, current_thought should be non-zero
    assert!(
        mind.state.current_thought.norm() > 0.1,
        "current_thought should be non-zero after perception: norm={}",
        mind.state.current_thought.norm()
    );
}

#[test]
fn test_current_thought_evolves_with_ema() {
    let mut mind = ContinuousMind::default();
    mind.activate();

    // First perception: adopt directly
    mind.perceive(ContinuousHV::random(512, 100));
    mind.tick();
    let after_first = mind.state.current_thought.clone();

    // Second perception: EMA blend
    mind.perceive(ContinuousHV::random(512, 200));
    mind.tick();
    let after_second = mind.state.current_thought.clone();

    // Thought should have changed (LiquidHolocell step with dt=0.1 produces
    // small-but-nonzero evolution — similarity can be very high for high-dim vectors)
    let sim = after_first.similarity(&after_second);
    assert!(
        sim < 0.999,
        "current_thought should evolve after new perception: sim={}",
        sim
    );

    // But should retain some of the first thought (70% weight)
    assert!(
        sim > 0.1,
        "current_thought should retain prior context: sim={}",
        sim
    );
}

#[test]
fn test_cycle_metadata_mesh_defaults_zero() {
    let metadata = crate::cognitive_loop::types::CycleMetadata::default();
    assert_eq!(metadata.mesh.mesh_health_score, 0.0);
    assert_eq!(metadata.mesh.mesh_peer_count, 0);
    assert_eq!(metadata.mesh.mesh_bytes_sent, 0);
    assert_eq!(metadata.mesh.mesh_bytes_received, 0);
}
