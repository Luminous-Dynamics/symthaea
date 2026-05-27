// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test for the memory coordinator pipeline.
//!
//! Exercises the full path: MemoryCoordinator signal updates → graduation
//! queueing → episodic storage, verifying that the cross-tier memory
//! coordination system works end-to-end.
//!
//! Also tests:
//! - Phi-weighted learning rate modulation via SemanticMemory
//! - Reconsolidation during episodic replay
//! - Working memory eviction and graduation routing

use symthaea::memory::memory_coordinator::MemorySource;
use symthaea::memory::{
    CoordinatorConfig, Episode, EpisodicMemory, EpisodicReplayConfig, GraduationEvent,
    MemoryCoordinator, SemanticMemory,
};
use symthaea_core::hdc::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY COORDINATOR: Signal propagation + graduation pipeline
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_coordinator_signal_updates() {
    let mut coordinator = MemoryCoordinator::new(CoordinatorConfig::default());

    // Update signals
    coordinator.update_signals(0.7, 0.8);
    let signals = coordinator.signals();

    assert!(signals.psi > 0.0, "Phi should be set after update");
    assert!(
        signals.coherence > 0.0,
        "Coherence should be set after update"
    );
}

#[test]
fn test_graduation_pipeline_end_to_end() {
    let mut coordinator = MemoryCoordinator::new(CoordinatorConfig::default());
    let mut episodic = EpisodicMemory::new(EpisodicReplayConfig::default());

    // Set consciousness signals
    coordinator.update_signals(0.85, 0.9);

    // Queue a graduation event (simulating a working memory eviction)
    let content = ContinuousHV::random(128, 42);
    coordinator.queue_graduation(GraduationEvent {
        content: content.clone(),
        label: "test_evicted_item".to_string(),
        steps_survived: 10, // Survived 10 decay cycles
        final_activation: 0.6,
        psi_at_graduation: 0.85,
        coherence_at_graduation: 0.9,
        source: MemorySource::Internal,
        is_verified: false,
    });

    // Process graduations
    let graduated = coordinator.process_graduations(&mut episodic);
    assert!(graduated > 0, "Should graduate at least one item");

    // Verify the episode was stored in episodic memory
    let stats = episodic.stats();
    assert!(
        stats.total_stored > 0,
        "Episodic memory should have stored the graduated episode"
    );
}

#[test]
fn test_graduation_rejected_when_too_few_steps() {
    let config = CoordinatorConfig {
        min_wm_steps_for_graduation: 5,
        ..Default::default()
    };
    let mut coordinator = MemoryCoordinator::new(config);
    let mut episodic = EpisodicMemory::new(EpisodicReplayConfig::default());

    coordinator.update_signals(0.8, 0.8);

    // Queue an event with only 2 steps survived (below threshold of 5)
    coordinator.queue_graduation(GraduationEvent {
        content: ContinuousHV::random(128, 99),
        label: "short_lived_item".to_string(),
        steps_survived: 2,
        final_activation: 0.3,
        psi_at_graduation: 0.8,
        coherence_at_graduation: 0.8,
        source: MemorySource::Internal,
        is_verified: false,
    });

    let graduated = coordinator.process_graduations(&mut episodic);
    assert_eq!(
        graduated, 0,
        "Should reject items that didn't survive enough steps"
    );

    assert!(
        coordinator.stats.graduations_rejected > 0,
        "Should track rejected graduations"
    );
}

#[test]
fn test_multiple_graduations_in_one_cycle() {
    let mut coordinator = MemoryCoordinator::new(CoordinatorConfig::default());
    let mut episodic = EpisodicMemory::new(EpisodicReplayConfig::default());

    coordinator.update_signals(0.9, 0.95);

    // Queue multiple graduation events
    for i in 0..5u64 {
        coordinator.queue_graduation(GraduationEvent {
            content: ContinuousHV::random(128, i),
            label: format!("batch_item_{}", i),
            steps_survived: 8 + i,
            final_activation: 0.5 + i as f64 * 0.05,
            psi_at_graduation: 0.9,
            coherence_at_graduation: 0.95,
            source: MemorySource::Internal,
            is_verified: false,
        });
    }

    let graduated = coordinator.process_graduations(&mut episodic);
    assert_eq!(graduated, 5, "All 5 items should graduate");
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHI-WEIGHTED LEARNING RATE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_phi_weighted_lr_modulation() {
    let mut semantic = SemanticMemory::with_threshold(100, 0.3);

    // Store a few entries with known prediction errors
    // SemanticMemory works with Vec<f32>, not ContinuousHV
    let base_vec: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01).sin()).collect();
    for i in 0..5u64 {
        let mut v = base_vec.clone();
        v[0] += i as f32 * 0.1; // Slight variation
        semantic.store_with_timestamp(v, 0.3, None, i);
    }

    // Compute phi-weighted LR factor
    let lr_factor = semantic.compute_lr_factor_phi_weighted(&base_vec, 3, 0.8, 10);

    // Should return a valid positive factor
    assert!(lr_factor > 0.0, "LR factor should be positive");
    assert!(lr_factor.is_finite(), "LR factor should be finite");
}

#[test]
fn test_phi_weighted_lr_increases_with_error() {
    let base_vec: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01).sin()).collect();

    let mut semantic = SemanticMemory::with_threshold(100, 0.3);
    // Store entries with HIGH prediction error
    for i in 0..5u64 {
        let mut v = base_vec.clone();
        v[0] += i as f32 * 0.1;
        semantic.store_with_timestamp(v, 0.9, None, i); // High error
    }

    let high_error_lr = semantic.compute_lr_factor_phi_weighted(&base_vec, 3, 0.8, 10);

    // Reset and store with LOW prediction error
    let mut semantic_low = SemanticMemory::with_threshold(100, 0.3);
    for i in 0..5u64 {
        let mut v = base_vec.clone();
        v[0] += i as f32 * 0.1;
        semantic_low.store_with_timestamp(v, 0.05, None, i); // Low error
    }

    let low_error_lr = semantic_low.compute_lr_factor_phi_weighted(&base_vec, 3, 0.8, 10);

    // High-error context should produce higher LR factor (learn more when surprised)
    assert!(
        high_error_lr >= low_error_lr,
        "Higher prediction error context should produce >= LR factor: {} vs {}",
        high_error_lr,
        low_error_lr
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// RECONSOLIDATION
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_episodic_replay_stores_high_phi_episodes() {
    let config = EpisodicReplayConfig {
        psi_threshold: 0.5,
        ..Default::default()
    };
    let mut episodic = EpisodicMemory::new(config);

    // Create a high-phi episode
    let input = ContinuousHV::random(128, 1);
    let output = ContinuousHV::random(128, 2);
    let episode = Episode::with_metadata(
        input, output, 0.9, // High phi
        1,   // cycle
        0.1, // prediction error
        0.5, // valence
        0.8, // coherence
    );

    let stored = episodic.store_if_significant(episode);
    assert!(stored, "High-phi episode should be stored");
    assert_eq!(episodic.stats().total_stored, 1);
}

#[test]
fn test_episodic_replay_rejects_low_phi() {
    let config = EpisodicReplayConfig {
        psi_threshold: 0.5,
        ..Default::default()
    };
    let mut episodic = EpisodicMemory::new(config);

    let input = ContinuousHV::random(128, 1);
    let output = ContinuousHV::random(128, 2);
    let episode = Episode::with_metadata(
        input, output, 0.1, // Low phi (below threshold)
        1, 0.1, 0.5, 0.3,
    );

    let stored = episodic.store_if_significant(episode);
    assert!(!stored, "Low-phi episode should not be stored");
    assert_eq!(episodic.stats().total_stored, 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// WORKING MEMORY EVICTION (ContinuousMind)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_working_memory_eviction_tracking() {
    use symthaea::mind::{ContinuousMind, MindConfig};

    let config = MindConfig {
        working_memory_capacity: 3,
        ..Default::default()
    };
    let mut mind = ContinuousMind::new(config);
    mind.activate();

    let dim = mind.config().dimension;

    // Push 3 items (no eviction yet)
    for i in 0..3u64 {
        let hv = ContinuousHV::random(dim, i);
        mind.perceive(hv);
        mind.tick();
    }

    let evicted = mind.take_evicted();
    assert!(evicted.is_empty(), "No eviction when under capacity");

    // Push a 4th item (should evict the oldest)
    let hv4 = ContinuousHV::random(dim, 100);
    mind.perceive(hv4);
    mind.tick();

    let evicted = mind.take_evicted();
    assert_eq!(
        evicted.len(),
        1,
        "Should evict exactly one item when capacity exceeded"
    );
    // Verify steps_survived is tracked
    let (_hv, steps_survived, _source, _is_verified) = &evicted[0];
    assert!(
        *steps_survived > 0,
        "Evicted item should have survived at least 1 tick"
    );

    // Verify take_evicted drains the buffer
    let evicted_again = mind.take_evicted();
    assert!(
        evicted_again.is_empty(),
        "Buffer should be empty after take"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// FULL PIPELINE: eviction → coordinator → episodic
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_full_eviction_to_graduation_pipeline() {
    use symthaea::mind::{ContinuousMind, MindConfig};

    let config = MindConfig {
        working_memory_capacity: 3,
        ..Default::default()
    };
    let mut mind = ContinuousMind::new(config);
    mind.activate();

    let mut coordinator = MemoryCoordinator::new(CoordinatorConfig::default());
    let mut episodic = EpisodicMemory::new(EpisodicReplayConfig::default());

    let dim = mind.config().dimension;

    // Fill working memory and trigger eviction
    for i in 0..5u64 {
        let hv = ContinuousHV::random(dim, i);
        mind.perceive(hv);
        mind.tick();
    }

    // Drain evicted items and queue for graduation
    let evicted = mind.take_evicted();
    assert!(!evicted.is_empty(), "Should have evicted items");

    coordinator.update_signals(0.85, 0.9);

    for (i, (hv, steps_survived, source, is_verified)) in evicted.into_iter().enumerate() {
        coordinator.queue_graduation(GraduationEvent {
            content: hv,
            label: format!("evicted_wm_{}", i),
            steps_survived,
            final_activation: 0.5,
            psi_at_graduation: 0.85,
            coherence_at_graduation: 0.9,
            source,
            is_verified,
        });
    }

    // Process graduations into episodic memory
    let graduated = coordinator.process_graduations(&mut episodic);
    assert!(
        graduated > 0,
        "Evicted items should graduate to episodic memory"
    );

    let stats = episodic.stats();
    assert!(
        stats.total_stored > 0,
        "Episodic memory should contain graduated episodes from WM eviction"
    );

    assert!(
        coordinator.stats.graduations_processed > 0,
        "Coordinator should track successful graduations"
    );
}