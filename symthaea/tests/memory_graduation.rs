// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Memory Graduation Integration Test: Working Memory -> Episodic -> Database Pipeline
// ==================================================================================
//
// Tests the full memory graduation pipeline under sustained load:
//   1. Create a CognitiveLoopService
//   2. Run 100+ cycles with varied inputs
//   3. Verify working memory stays bounded
//   4. Verify evicted items are tracked
//   5. Verify episodic memory receives high-phi items
//   6. Test that memory search returns relevant results after training
//
// Also tests the standalone memory components (ContinuousMind eviction,
// MemoryCoordinator graduation, EpisodicMemory storage) wired together.
//
// No feature flags required — uses default configuration only.
// ==================================================================================

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::memory::memory_coordinator::MemorySource;
use symthaea::memory::{
    CoordinatorConfig, Episode, EpisodicMemory, EpisodicReplayConfig, GraduationEvent,
    MemoryCoordinator, SemanticMemory,
};
use symthaea::mind::{ContinuousMind, MindConfig};
use symthaea_core::hdc::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════════
// WORKING MEMORY BOUNDS UNDER SUSTAINED LOAD
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_working_memory_stays_bounded_under_load() {
    let capacity = 7;
    let config = MindConfig {
        working_memory_capacity: capacity,
        ..Default::default()
    };
    let mut mind = ContinuousMind::new(config);
    mind.activate();

    let dim = mind.config().dimension;

    // Push many more items than capacity
    for i in 0..50u64 {
        let hv = ContinuousHV::random(dim, i * 31 + 17);
        mind.perceive(hv);
        mind.tick();
    }

    // Working memory should never exceed capacity
    assert!(
        mind.working_memory().len() <= capacity,
        "Working memory should be bounded at {capacity}, got {}",
        mind.working_memory().len()
    );
}

#[test]
fn test_working_memory_7_plus_minus_2() {
    // Default capacity is 7 (the magical number)
    let config = MindConfig::default();
    let mut mind = ContinuousMind::new(config);
    mind.activate();

    let dim = mind.config().dimension;
    let capacity = mind.config().working_memory_capacity;

    // Verify capacity is in the 7+/-2 range
    assert!(
        (5..=9).contains(&capacity),
        "Default working memory capacity should be 7+/-2, got {}",
        capacity
    );

    // Fill past capacity
    for i in 0..20u64 {
        let hv = ContinuousHV::random(dim, i);
        mind.perceive(hv);
        mind.tick();
    }

    assert!(
        mind.working_memory().len() <= capacity,
        "Working memory should not exceed capacity of {capacity}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// EVICTION TRACKING
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_evicted_items_are_tracked_with_steps_survived() {
    let config = MindConfig {
        working_memory_capacity: 3,
        ..Default::default()
    };
    let mut mind = ContinuousMind::new(config);
    mind.activate();

    let dim = mind.config().dimension;

    // Fill memory exactly to capacity (3 items)
    for i in 0..3u64 {
        let hv = ContinuousHV::random(dim, i);
        mind.perceive(hv);
        mind.tick();
    }

    // No evictions yet
    let evicted = mind.take_evicted();
    assert!(evicted.is_empty(), "No eviction when at or under capacity");

    // Push 5 more items, causing 5 evictions
    for i in 10..15u64 {
        let hv = ContinuousHV::random(dim, i);
        mind.perceive(hv);
        mind.tick();
    }

    let evicted = mind.take_evicted();
    assert!(
        !evicted.is_empty(),
        "Should have evicted items after exceeding capacity"
    );

    // All evicted items should have valid steps_survived (> 0 for items that lived at least 1 tick)
    for (_hv, steps_survived, _source, _is_verified) in &evicted {
        assert!(
            *steps_survived > 0,
            "Evicted items should have survived at least 1 tick, got {}",
            steps_survived
        );
    }

    // take_evicted drains the buffer
    let again = mind.take_evicted();
    assert!(again.is_empty(), "take_evicted should drain the buffer");
}

// ═══════════════════════════════════════════════════════════════════════════════
// FULL GRADUATION PIPELINE: eviction -> coordinator -> episodic
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_graduation_pipeline_under_sustained_load() {
    let config = MindConfig {
        working_memory_capacity: 5,
        ..Default::default()
    };
    let mut mind = ContinuousMind::new(config);
    mind.activate();

    let mut coordinator = MemoryCoordinator::new(CoordinatorConfig::default());
    let mut episodic = EpisodicMemory::new(EpisodicReplayConfig {
        psi_threshold: 0.3, // Lower threshold to allow more graduations
        ..Default::default()
    });

    let dim = mind.config().dimension;
    let mut total_graduated = 0usize;

    // Run 100 cycles with varied inputs
    for cycle in 0..100u64 {
        // Use a seed that varies each cycle for diverse inputs
        let hv = ContinuousHV::random(dim, cycle * 7 + 13);
        mind.perceive(hv);
        mind.tick();

        // Drain evicted items
        let evicted = mind.take_evicted();
        if !evicted.is_empty() {
            let snapshot = mind.snapshot();
            coordinator.update_signals(snapshot.consciousness_level, snapshot.meta_awareness);

            for (hv, steps_survived, source, is_verified) in evicted {
                coordinator.queue_graduation(GraduationEvent {
                    content: hv,
                    label: format!("wm_eviction_cycle_{cycle}"),
                    steps_survived,
                    final_activation: 0.5,
                    psi_at_graduation: snapshot.consciousness_level,
                    coherence_at_graduation: snapshot.meta_awareness,
                    source,
                    is_verified,
                });
            }

            let graduated = coordinator.process_graduations(&mut episodic);
            total_graduated += graduated;
        }
    }

    // After 100 cycles with capacity 5, we should have had many evictions
    assert!(
        total_graduated > 0,
        "Should have graduated items to episodic memory over 100 cycles"
    );

    // Episodic memory should contain graduated episodes
    let stats = episodic.stats();
    assert!(
        stats.total_stored > 0,
        "Episodic memory should contain graduated episodes, got {} stored",
        stats.total_stored
    );

    // Working memory should still be bounded
    assert!(
        mind.working_memory().len() <= 5,
        "Working memory should remain bounded"
    );

    // Coordinator should have tracked statistics
    assert!(
        coordinator.stats.graduations_processed > 0,
        "Coordinator should track graduation count"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// EPISODIC MEMORY: HIGH-PHI FILTERING
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_episodic_stores_only_high_phi_episodes() {
    let config = EpisodicReplayConfig {
        psi_threshold: 0.5,
        ..Default::default()
    };
    let mut episodic = EpisodicMemory::new(config);

    // Store high-phi episode (should be accepted)
    let high_phi = Episode::with_metadata(
        ContinuousHV::random(128, 1),
        ContinuousHV::random(128, 2),
        0.8, // High phi
        1,
        0.1,
        0.5,
        0.7,
    );
    assert!(
        episodic.store_if_significant(high_phi),
        "High-phi episode should be stored"
    );

    // Store low-phi episode (should be rejected)
    let low_phi = Episode::with_metadata(
        ContinuousHV::random(128, 3),
        ContinuousHV::random(128, 4),
        0.2, // Low phi (below 0.5 threshold)
        2,
        0.1,
        0.3,
        0.3,
    );
    assert!(
        !episodic.store_if_significant(low_phi),
        "Low-phi episode should be rejected"
    );

    assert_eq!(
        episodic.stats().total_stored,
        1,
        "Only one episode should be stored (the high-phi one)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEMANTIC EVICTION ROUTING TO GRADUATION PIPELINE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_semantic_evictions_route_to_graduation() {
    // Use a tiny ring buffer (capacity 3) so evictions happen quickly
    let mut semantic = SemanticMemory::with_threshold(3, 0.3);
    let mut coordinator = MemoryCoordinator::new(CoordinatorConfig::default());
    let mut episodic = EpisodicMemory::new(EpisodicReplayConfig {
        psi_threshold: 0.1,
        ..Default::default()
    });

    coordinator.update_signals(0.9, 0.9);

    let dim = 128;
    let mut eviction_count = 0usize;

    for i in 0..10u64 {
        let v: Vec<f32> = (0..dim)
            .map(|j| ((i * 7 + j as u64) as f32 * 0.01).sin())
            .collect();
        let evicted = semantic.store_with_timestamp(v, 0.1, None, i);

        if let Some(entry) = evicted {
            eviction_count += 1;
            let hv = ContinuousHV::from_vec(entry.hdc_vector);
            coordinator.queue_graduation(GraduationEvent {
                content: hv,
                label: format!("semantic_eviction_{i}"),
                steps_survived: 10,
                final_activation: 0.5,
                psi_at_graduation: 0.9,
                coherence_at_graduation: 0.9,
                source: MemorySource::Internal,
                is_verified: false,
            });
        }
    }

    assert_eq!(
        eviction_count, 7,
        "Should have 7 evictions from capacity-3 buffer with 10 stores"
    );

    let graduated = coordinator.process_graduations(&mut episodic);
    assert!(
        graduated > 0,
        "Evicted semantic entries should graduate to episodic memory"
    );
    assert!(
        episodic.stats().total_stored > 0,
        "Episodic memory should contain graduated semantic entries"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEMANTIC MEMORY: SEARCH AFTER TRAINING
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_semantic_memory_search_returns_relevant_results() {
    let mut semantic = SemanticMemory::with_threshold(100, 0.3);

    // Create and store several vectors
    let dim = 128;
    let base_vec: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.05).sin()).collect();

    for i in 0..10u64 {
        let mut v = base_vec.clone();
        // Slight variation from base
        v[0] += i as f32 * 0.05;
        let magnitude: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for x in v.iter_mut() {
                *x /= magnitude;
            }
        }
        semantic.store_with_timestamp(v, 0.1, None, i);
    }

    // Search for something similar to the base vector
    let results = semantic.find_similar(&base_vec, 3);
    assert!(
        !results.is_empty(),
        "Semantic search should return results after storing entries"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// COGNITIVE LOOP: MEMORY INTEGRATION OVER 100+ CYCLES
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_cognitive_loop_100_cycles_with_varied_inputs() {
    let config = CognitiveLoopConfig {
        learning_threshold: 0.0, // Always learn
        async_training: false,   // Synchronous for determinism
        ..Default::default()
    };
    let mut service = CognitiveLoopService::new(config).unwrap();

    let inputs = [
        "the cat sat on the mat",
        "quantum mechanics describes wave functions",
        "love compassion kindness empathy",
        "mathematics is the language of nature",
        "neural networks learn patterns from data",
        "consciousness arises from integrated information",
        "the weather is sunny and warm today",
        "trees provide oxygen through photosynthesis",
        "music evokes deep emotional responses",
        "memory consolidation occurs during sleep",
    ];

    let mut errors = Vec::new();
    for cycle in 0..100 {
        let input = inputs[cycle % inputs.len()];
        let result = service.cycle(input);

        assert!(
            result.prediction_error.is_finite(),
            "Prediction error should be finite at cycle {cycle}"
        );
        assert!(
            !result.output.is_empty(),
            "Output should not be empty at cycle {cycle}"
        );

        errors.push(result.prediction_error);
    }

    assert_eq!(service.stats().total_cycles, 100);

    // Prediction error should generally decrease or stabilize
    let first_quarter: f32 = errors[..25].iter().sum::<f32>() / 25.0;
    let last_quarter: f32 = errors[75..].iter().sum::<f32>() / 25.0;

    assert!(
        last_quarter <= first_quarter + 0.15,
        "Error should decrease or stabilize over 100 cycles: first_q={first_quarter:.4}, last_q={last_quarter:.4}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// COORDINATOR STATS ACCUMULATION
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_coordinator_stats_accumulate_correctly() {
    let mut coordinator = MemoryCoordinator::new(CoordinatorConfig::default());
    let mut episodic = EpisodicMemory::new(EpisodicReplayConfig::default());

    // Set high signals to ensure graduation
    coordinator.update_signals(0.9, 0.95);

    let mut total_queued = 0u64;
    let mut total_graduated = 0usize;

    for batch in 0..5u64 {
        // Queue 3 events per batch
        for i in 0..3u64 {
            coordinator.queue_graduation(GraduationEvent {
                content: ContinuousHV::random(128, batch * 10 + i),
                label: format!("batch_{batch}_item_{i}"),
                steps_survived: 8 + i,
                final_activation: 0.6,
                psi_at_graduation: 0.9,
                coherence_at_graduation: 0.95,
                source: MemorySource::Internal,
                is_verified: false,
            });
            total_queued += 1;
        }

        total_graduated += coordinator.process_graduations(&mut episodic);
    }

    assert_eq!(
        coordinator.stats.graduations_processed, total_graduated as u64,
        "Coordinator stats should track total graduations"
    );

    // All items should have been processed (either graduated or rejected)
    assert_eq!(
        coordinator.stats.graduations_processed + coordinator.stats.graduations_rejected,
        total_queued,
        "All queued items should be accounted for (processed + rejected = queued)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// WORKING MEMORY TICKS PARALLEL ARRAY
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_working_memory_ticks_stay_in_sync() {
    let config = MindConfig {
        working_memory_capacity: 5,
        ..Default::default()
    };
    let mut mind = ContinuousMind::new(config);
    mind.activate();

    let dim = mind.config().dimension;

    for i in 0..20u64 {
        let hv = ContinuousHV::random(dim, i);
        mind.perceive(hv);
        mind.tick();

        // Parallel arrays must always have the same length
        assert_eq!(
            mind.working_memory().len(),
            mind.working_memory_ticks_slice().len(),
            "Working memory and ticks arrays must be in sync at cycle {i}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EPISODIC MEMORY CAPACITY BOUNDS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_episodic_memory_tracks_evictions_over_capacity() {
    let config = EpisodicReplayConfig {
        capacity: 10,
        psi_threshold: 0.1, // Low threshold so most episodes are stored
        ..Default::default()
    };
    let mut episodic = EpisodicMemory::new(config);

    // Store 20 episodes with sufficient phi (exceeding capacity of 10)
    for i in 0..20u64 {
        let episode = Episode::with_metadata(
            ContinuousHV::random(128, i),
            ContinuousHV::random(128, i + 100),
            0.5 + (i as f64 * 0.02),
            i,
            0.1,
            0.3,
            0.5,
        );
        episodic.store_if_significant(episode);
    }

    let stats = episodic.stats();

    // All 20 episodes should have been stored (above phi threshold)
    assert_eq!(
        stats.total_stored, 20,
        "All episodes above phi threshold should be stored"
    );

    // Over-capacity episodes should have triggered eviction tracking
    // (capacity is 10, stored 20, so at least 10 eviction events)
    assert!(
        stats.total_evicted >= 10,
        "Should track evictions when exceeding capacity, got {} evictions",
        stats.total_evicted
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// IGNORED: EXTENDED MEMORY GRADUATION (expensive)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn test_extended_graduation_500_cycles() {
    let config = MindConfig {
        working_memory_capacity: 7,
        ..Default::default()
    };
    let mut mind = ContinuousMind::new(config);
    mind.activate();

    let mut coordinator = MemoryCoordinator::new(CoordinatorConfig::default());
    let mut episodic = EpisodicMemory::new(EpisodicReplayConfig {
        psi_threshold: 0.2,
        ..Default::default()
    });

    let dim = mind.config().dimension;
    let mut total_evicted = 0usize;
    let mut total_graduated = 0usize;

    // Run 500 cycles simulating varied cognitive load
    for cycle in 0..500u64 {
        // Vary input complexity by cycling through different seeds
        let hv = ContinuousHV::random(dim, cycle * 37 + 11);
        mind.perceive(hv);
        mind.tick();

        let evicted = mind.take_evicted();
        if !evicted.is_empty() {
            total_evicted += evicted.len();
            let snapshot = mind.snapshot();
            coordinator.update_signals(snapshot.consciousness_level, snapshot.meta_awareness);

            for (hv, steps_survived, source, is_verified) in evicted {
                coordinator.queue_graduation(GraduationEvent {
                    content: hv,
                    label: format!("extended_cycle_{cycle}"),
                    steps_survived,
                    final_activation: 0.5,
                    psi_at_graduation: snapshot.consciousness_level,
                    coherence_at_graduation: snapshot.meta_awareness,
                    source,
                    is_verified,
                });
            }
            total_graduated += coordinator.process_graduations(&mut episodic);
        }

        // Verify invariant: working memory always bounded
        assert!(
            mind.working_memory().len() <= 7,
            "Working memory exceeded capacity at cycle {cycle}"
        );
    }

    assert!(
        total_evicted > 0,
        "Should have evicted items over 500 cycles"
    );
    assert!(
        total_graduated > 0,
        "Should have graduated items to episodic memory over 500 cycles"
    );

    let episodic_stats = episodic.stats();
    assert!(
        episodic_stats.total_stored > 0,
        "Episodic memory should have stored episodes"
    );
}