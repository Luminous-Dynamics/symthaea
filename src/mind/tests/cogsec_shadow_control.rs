// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pre-instrumentation determinism baseline for CogSec shadow qualification.
//!
//! These tests intentionally exercise `process_inputs()` directly rather than
//! `tick()`. The full tick samples wall-clock chronobiology and processing
//! latency; those are not part of the first CogSec non-interference claim.
//!
//! This is the N0 control baseline only: two independently constructed legacy
//! minds receive identical deterministic state/input and must produce identical
//! legacy projections before any `ShadowRuntimeOwner` field or hook exists.

use super::super::*;
use crate::memory::memory_coordinator::MemorySource;
use std::collections::HashMap;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

#[derive(Debug, Clone, PartialEq)]
struct GoalProjection {
    id: String,
    description: String,
    embedding: ContinuousHV,
    priority: f32,
    progress: f32,
    is_active: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct EvictedProjection {
    content: ContinuousHV,
    steps_survived: u64,
    source: MemorySource,
    is_verified: bool,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
struct CoordinatorProjection {
    psi: f64,
    coherence: f64,
    sigma: Option<f64>,
    retrieval_rate: f64,
    step: u64,
    graduations_processed: u64,
    graduations_rejected: u64,
    cross_tier_retrievals: u64,
    signal_updates: u64,
}

#[derive(Debug, Clone, PartialEq)]
struct LegacyBehaviorProjection {
    working_memory: Vec<ContinuousHV>,
    working_memory_ticks: Vec<u64>,
    working_memory_sources: Vec<MemorySource>,
    working_memory_verified: Vec<bool>,
    working_memory_metadata: Vec<HashMap<String, String>>,
    evicted_items: Vec<EvictedProjection>,
    current_thought: ContinuousHV,
    holocell: symthaea_core::hdc::LiquidHolocell,
    emotional_valence: f32,
    arousal: f32,
    thermodynamic_load: f32,
    mood_temperature: f32,
    consciousness_level: f64,
    tick: u64,
    goals: Vec<GoalProjection>,
    input_queue_len: usize,
    inputs_processed: u64,
    outputs_generated: u64,
    goals_completed: u64,
    total_ticks: u64,
    coordinator: CoordinatorProjection,
}

fn project(mind: &ContinuousMind) -> LegacyBehaviorProjection {
    let signals = mind.memory_coordinator.signals();
    let coordinator_stats = &mind.memory_coordinator.stats;

    LegacyBehaviorProjection {
        working_memory: mind.working_memory.clone(),
        working_memory_ticks: mind.working_memory_ticks.clone(),
        working_memory_sources: mind.working_memory_sources.clone(),
        working_memory_verified: mind.working_memory_verified.clone(),
        working_memory_metadata: mind.working_memory_metadata.clone(),
        evicted_items: mind
            .evicted_items
            .iter()
            .map(|item| EvictedProjection {
                content: item.content.clone(),
                steps_survived: item.steps_survived,
                source: item.source,
                is_verified: item.is_verified,
                metadata: item.metadata.clone(),
            })
            .collect(),
        current_thought: mind.state.current_thought.clone(),
        holocell: mind.state.holocell.clone(),
        emotional_valence: mind.state.emotional_valence,
        arousal: mind.state.arousal,
        thermodynamic_load: mind.state.thermodynamic_load,
        mood_temperature: mind.state.mood_temperature,
        consciousness_level: mind.state.consciousness_level,
        tick: mind.state.tick,
        goals: mind
            .goals
            .iter()
            .map(|goal| GoalProjection {
                id: goal.id.clone(),
                description: goal.description.clone(),
                embedding: goal.embedding.clone(),
                priority: goal.priority,
                progress: goal.progress,
                is_active: goal.is_active,
            })
            .collect(),
        input_queue_len: mind.input_queue.len(),
        inputs_processed: mind.stats.inputs_processed,
        outputs_generated: mind.stats.outputs_generated,
        goals_completed: mind.stats.goals_completed,
        total_ticks: mind.stats.total_ticks,
        coordinator: CoordinatorProjection {
            psi: signals.psi,
            coherence: signals.coherence,
            sigma: signals.sigma,
            retrieval_rate: signals.retrieval_rate,
            step: signals.step,
            graduations_processed: coordinator_stats.graduations_processed,
            graduations_rejected: coordinator_stats.graduations_rejected,
            cross_tier_retrievals: coordinator_stats.cross_tier_retrievals,
            signal_updates: coordinator_stats.signal_updates,
        },
    }
}

fn mind_pair(mut config: MindConfig) -> (ContinuousMind, ContinuousMind) {
    // Keep this baseline narrow: no social/learning side systems are needed to
    // exercise the process_inputs owner transitions.
    config.enable_social_coherence = false;
    config.learning_enabled = false;

    let genesis = GenesisSeed::from_phrase("cogsec-shadow-control-determinism-v0");
    let a = ContinuousMind::from_genesis(config.clone(), &genesis, "legacy-control");
    let b = ContinuousMind::from_genesis(config, &genesis, "legacy-control");
    (a, b)
}

fn goal_input(seed: u64, description: &str) -> MindInput {
    let mut input = MindInput::new(InputType::Goal, ContinuousHV::random(512, seed));
    input.priority = 0.9;
    input
        .metadata
        .insert("description".to_string(), description.to_string());
    input
}

fn feedback_input(seed: u64, valence: &str) -> MindInput {
    let mut input = MindInput::new(InputType::Feedback, ContinuousHV::random(512, seed));
    input.priority = 0.7;
    input
        .metadata
        .insert("valence".to_string(), valence.to_string());
    input
}

fn seed_full_working_memory(mind: &mut ContinuousMind) {
    assert_eq!(mind.config.working_memory_capacity, 2);
    mind.state.tick = 10;

    let mut first_meta = HashMap::new();
    first_meta.insert("topic".to_string(), "oldest".to_string());
    let mut second_meta = HashMap::new();
    second_meta.insert("topic".to_string(), "newer".to_string());

    mind.working_memory = vec![
        ContinuousHV::random(512, 1001),
        ContinuousHV::random(512, 1002),
    ];
    mind.working_memory_ticks = vec![2, 5];
    mind.working_memory_sources = vec![MemorySource::UserInteraction, MemorySource::WebResearch];
    mind.working_memory_verified = vec![false, true];
    mind.working_memory_metadata = vec![first_meta, second_meta];
}

fn assert_pair_equal(a: &ContinuousMind, b: &ContinuousMind, scenario: &str) {
    assert_eq!(
        project(a),
        project(b),
        "legacy control diverged before CogSec shadow instrumentation: {scenario}"
    );
}

#[test]
fn cogsec_s0_goal_no_eviction_legacy_control_is_exactly_deterministic() {
    let mut config = MindConfig::default();
    config.dimension = 512;
    config.working_memory_capacity = 2;
    let (mut a, mut b) = mind_pair(config);
    a.state.tick = 10;
    b.state.tick = 10;

    let input = goal_input(2001, "retain explicit confirmation");
    a.input(input.clone());
    b.input(input);

    a.process_inputs();
    b.process_inputs();

    assert_pair_equal(&a, &b, "S0 goal/no eviction");
    assert_eq!(a.working_memory.len(), 1);
    assert!(a.evicted_items.is_empty());
    assert_eq!(a.goals.len(), 1);
    assert_eq!(a.goals[0].description, "retain explicit confirmation");
    assert_eq!(a.input_queue.len(), 0);
    assert_eq!(a.stats.inputs_processed, 1);
}

#[test]
fn cogsec_s1_goal_forced_eviction_legacy_control_is_exactly_deterministic() {
    let mut config = MindConfig::default();
    config.dimension = 512;
    config.working_memory_capacity = 2;
    let (mut a, mut b) = mind_pair(config);
    seed_full_working_memory(&mut a);
    seed_full_working_memory(&mut b);

    let old_second = a.working_memory[1].clone();
    let admitted = ContinuousHV::random(512, 3001);
    let mut input = MindInput::new(InputType::Goal, admitted.clone());
    input.priority = 0.95;
    input
        .metadata
        .insert("description".to_string(), "do not bypass confirmation".to_string());

    a.input(input.clone());
    b.input(input);
    a.process_inputs();
    b.process_inputs();

    assert_pair_equal(&a, &b, "S1 goal/forced eviction");
    assert_eq!(a.working_memory.len(), 2);
    assert_eq!(a.working_memory[0], old_second);
    assert_eq!(a.working_memory[1], admitted);
    assert_eq!(a.evicted_items.len(), 1);
    assert_eq!(a.evicted_items[0].source, MemorySource::UserInteraction);
    assert!(!a.evicted_items[0].is_verified);
    assert_eq!(a.evicted_items[0].steps_survived, 8);
    assert_eq!(
        a.evicted_items[0].metadata.get("topic").map(String::as_str),
        Some("oldest")
    );
    assert_eq!(a.goals.len(), 1);
    assert_eq!(a.stats.inputs_processed, 1);
}

#[test]
fn cogsec_s2_feedback_legacy_control_is_exactly_deterministic() {
    let mut config = MindConfig::default();
    config.dimension = 512;
    config.working_memory_capacity = 2;
    let (mut a, mut b) = mind_pair(config);
    a.state.tick = 10;
    b.state.tick = 10;

    let input = feedback_input(4001, "0.7");
    a.input(input.clone());
    b.input(input);
    a.process_inputs();
    b.process_inputs();

    assert_pair_equal(&a, &b, "S2 feedback");
    assert_eq!(a.working_memory.len(), 1);
    assert!(a.goals.is_empty());
    assert!(a.evicted_items.is_empty());
    assert_eq!(
        a.state.emotional_valence,
        (0.7_f32 * 0.3_f32).clamp(-1.0, 1.0)
    );
    assert_eq!(a.stats.inputs_processed, 1);
}
