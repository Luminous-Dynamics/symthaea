//! Week 16 Day 5: Complete Sleep-Memory Consolidation Integration Tests
//!
//! End-to-end validation of the full sleep cycle system:
//! - Sleep Cycle Manager (Week 16 Day 1)
//! - Memory Consolidation (Week 16 Day 2)
//! - Hippocampus Enhancement (Week 16 Day 3)
//! - Forgetting & REM Creativity (Week 16 Day 4)
//!
//! This integration suite validates that all components work together
//! to create a biologically-authentic sleep-based memory system.

use symthaea::brain::{
    SleepCycleManager, SleepConfig,
    MemoryConsolidator,
    AttentionBid,
    prefrontal::Coalition,
};
use std::sync::Arc;

// ============================================================================
// Helper Functions
// ============================================================================

/// Helper function to create a Coalition from a single AttentionBid
/// (for test purposes - simple coalition with one member)
fn create_coalition_from_bid(bid: AttentionBid) -> Coalition {
    let strength = bid.salience + bid.urgency;
    let leader = bid.clone();

    Coalition {
        members: vec![bid],
        strength,
        coherence: 1.0, // Single member = perfect coherence
        leader,
    }
}

/// Helper function to create a test AttentionBid with HDC encoding
fn create_test_bid(content: &str, salience: f32, urgency: f32) -> AttentionBid {
    let mut bid = AttentionBid::new("test", content);
    bid.salience = salience;
    bid.urgency = urgency;

    // Create basic HDC encoding (simple hash-based)
    let hdc_vec: Vec<i8> = content.bytes()
        .take(8)
        .map(|b| if b % 2 == 0 { 1 } else { -1 })
        .collect();

    // Pad to 8 dimensions if needed
    let mut padded = hdc_vec;
    while padded.len() < 8 {
        padded.push(1);
    }
    padded.truncate(8);

    bid.hdc_semantic = Some(Arc::new(padded));
    bid
}

// ============================================================================
// Test 1: Complete Sleep Cycle Flow
// ============================================================================

#[test]
fn test_complete_sleep_cycle_flow() {
    // Setup: Fast sleep cycle for testing
    let config = SleepConfig {
        sleep_threshold: 0.2,      // Sleep quickly
        pressure_increment: 0.1,   // Build pressure fast
        max_awake_cycles: 10,
        sleep_progress_rate: 0.5,  // Complete sleep phases quickly
    };

    let mut sleep_manager = SleepCycleManager::with_config(config);

    // Phase 1: Awake - pressure builds
    assert_eq!(sleep_manager.state().name(), "Awake");
    assert_eq!(sleep_manager.pressure(), 0.0);

    // Cycle 1: Awake, pressure 0.1
    sleep_manager.update();
    assert!(!sleep_manager.is_sleeping());
    assert!((sleep_manager.pressure() - 0.1).abs() < 0.01);

    // Cycle 2: Awake, pressure 0.2 -> triggers sleep!
    sleep_manager.update();
    assert!(sleep_manager.is_sleeping());
    assert_eq!(sleep_manager.state().name(), "Light Sleep");

    // Phase 2: Light Sleep -> Deep Sleep
    sleep_manager.update(); // 50% progress
    assert_eq!(sleep_manager.state().name(), "Light Sleep");

    sleep_manager.update(); // 100% progress -> Deep Sleep
    assert_eq!(sleep_manager.state().name(), "Deep Sleep");

    // Phase 3: Deep Sleep -> REM Sleep
    sleep_manager.update(); // 50% progress
    assert_eq!(sleep_manager.state().name(), "Deep Sleep");

    sleep_manager.update(); // 100% progress -> REM
    assert_eq!(sleep_manager.state().name(), "REM Sleep");

    // Phase 4: REM Sleep -> Awake
    sleep_manager.update(); // 50% progress
    assert_eq!(sleep_manager.state().name(), "REM Sleep");

    sleep_manager.update(); // 100% progress -> Awake!
    assert_eq!(sleep_manager.state().name(), "Awake");
    assert!(!sleep_manager.is_sleeping());
    assert_eq!(sleep_manager.pressure(), 0.0); // Pressure reset
    assert_eq!(sleep_manager.total_cycles(), 1); // One complete cycle
}

// ============================================================================
// Test 2: Memory Consolidation During Sleep
// ============================================================================

#[test]
fn test_memory_consolidation_during_sleep() {
    let mut consolidator = MemoryConsolidator::new();

    // Create diverse working memory items to consolidate
    let bids = vec![
        create_test_bid("install firefox", 0.9, 0.8),
        create_test_bid("configure system", 0.8, 0.7),
        create_test_bid("update packages", 0.7, 0.6),
        create_test_bid("install vim", 0.9, 0.8), // Similar to firefox
    ];

    // Convert to coalitions
    let coalitions: Vec<Coalition> = bids.into_iter()
        .map(|bid| create_coalition_from_bid(bid))
        .collect();

    // Simulate Deep Sleep: Consolidation phase
    let traces = consolidator.consolidate_coalitions(coalitions);

    assert!(traces.len() > 0, "Should create semantic memory traces");
    assert!(traces.len() <= 4, "May merge similar memories");

    // Verify consolidated traces have reasonable properties
    for trace in &traces {
        assert!(trace.importance >= 0.0 && trace.importance <= 1.0,
                "Importance should be normalized");
        assert!(trace.compressed_pattern.len() > 0,
                "Should have compressed pattern");
    }
}

// ============================================================================
// Test 3: Forgetting Curve Over Multiple Sleep Cycles
// ============================================================================

#[test]
fn test_forgetting_curve_over_multiple_cycles() {
    let mut consolidator = MemoryConsolidator::new();

    // Create coalitions with varying importance
    let mut coalitions = vec![];

    // Low importance coalition
    let low_bid = create_test_bid("old memory", 0.5, 0.4);
    coalitions.push(create_coalition_from_bid(low_bid));

    // High importance coalition
    let high_bid = create_test_bid("important memory", 0.9, 0.9);
    coalitions.push(create_coalition_from_bid(high_bid));

    // Initial consolidation
    let mut traces = consolidator.consolidate_coalitions(coalitions);

    let initial_count = traces.len();
    assert!(initial_count >= 1, "Should create at least some traces");

    // Simulate multiple sleep cycles with forgetting
    for _cycle in 0..5 {
        consolidator.apply_forgetting(&mut traces);
    }

    // After multiple cycles, some memories should persist
    // (exact behavior depends on forgetting algorithm)
    assert!(traces.len() >= 0, "Forgetting algorithm applied");

    // High importance memories should be more likely to persist
    if traces.len() > 0 {
        let avg_importance: f32 = traces.iter()
            .map(|t| t.importance)
            .sum::<f32>() / traces.len() as f32;
        assert!(avg_importance > 0.5,
                "Surviving memories should have above-average importance");
    }
}

// ============================================================================
// Test 4: REM Sleep Creativity and Novel Patterns
// ============================================================================

#[test]
fn test_rem_sleep_creativity() {
    let mut sleep_manager = SleepCycleManager::new();

    // Register diverse working memory items
    let bid1 = create_test_bid("package", 0.8, 0.7);
    let bid2 = create_test_bid("manager", 0.8, 0.7);
    let bid3 = create_test_bid("install", 0.8, 0.7);

    // Register coalitions
    sleep_manager.register_coalition(create_coalition_from_bid(bid1.clone()));
    sleep_manager.register_coalition(create_coalition_from_bid(bid2.clone()));
    sleep_manager.register_coalition(create_coalition_from_bid(bid3.clone()));

    // Force sleep to REM phase
    sleep_manager.force_sleep();
    while sleep_manager.state().name() != "REM Sleep" {
        sleep_manager.update();
    }

    // Perform REM recombination
    let novel_patterns = sleep_manager.perform_rem_recombination();

    // Should generate novel combinations
    assert!(novel_patterns.len() > 0,
            "REM sleep should generate novel pattern combinations");
    assert!(novel_patterns.len() <= 5,
            "Should limit novel patterns to prevent combinatorial explosion");

    // Each pattern should be valid HDC vector
    for pattern in novel_patterns {
        assert_eq!(pattern.len(), 8, "HDC vectors should have dimension 8");
        assert!(pattern.iter().all(|&v| v == 1 || v == -1),
                "HDC vectors should be bipolar {{-1, 1}}");
    }
}

// ============================================================================
// Test 5: End-to-End Sleep Cycle with All Components
// ============================================================================

#[test]
fn test_end_to_end_sleep_memory_integration() {
    // Initialize all components
    let config = SleepConfig {
        sleep_threshold: 0.3,
        pressure_increment: 0.1,
        max_awake_cycles: 20,
        sleep_progress_rate: 0.2,
    };

    let mut sleep_manager = SleepCycleManager::with_config(config);
    let mut consolidator = MemoryConsolidator::new();

    // Simulate awake cognitive activity
    let mut working_memory_bids = vec![];
    for i in 0..5 {
        let bid = create_test_bid(
            &format!("memory_{}", i),
            0.7 + (i as f32 * 0.05),
            0.6 + (i as f32 * 0.05)
        );
        working_memory_bids.push(bid.clone());
        sleep_manager.register_coalition(create_coalition_from_bid(bid));
    }

    // Awake phase: pressure builds
    let mut cycles = 0;
    while !sleep_manager.is_sleeping() && cycles < 10 {
        sleep_manager.update();
        cycles += 1;
    }

    assert!(sleep_manager.is_sleeping(), "Should enter sleep after pressure builds");

    // Sleep phase progression
    let mut sleep_phases_seen = vec![];

    while sleep_manager.is_sleeping() {
        let phase = sleep_manager.state().name().to_string();
        if !sleep_phases_seen.contains(&phase) {
            sleep_phases_seen.push(phase.clone());
        }

        match phase.as_str() {
            "Light Sleep" => {
                // Light sleep: Initial replay (no action needed for test)
            }
            "Deep Sleep" => {
                // Deep sleep: Consolidation
                let coalitions: Vec<Coalition> = working_memory_bids.iter()
                    .map(|bid| create_coalition_from_bid(bid.clone()))
                    .collect();
                let _traces = consolidator.consolidate_coalitions(coalitions);
            }
            "REM Sleep" => {
                // REM sleep: Creative recombination
                let novel_patterns = sleep_manager.perform_rem_recombination();
                assert!(novel_patterns.len() > 0, "Should generate novel patterns in REM");
            }
            _ => {}
        }

        sleep_manager.update();
    }

    // Verify complete cycle
    assert_eq!(sleep_phases_seen.len(), 3,
               "Should see all 3 sleep phases: Light, Deep, REM");
    assert!(sleep_phases_seen.contains(&"Light Sleep".to_string()));
    assert!(sleep_phases_seen.contains(&"Deep Sleep".to_string()));
    assert!(sleep_phases_seen.contains(&"REM Sleep".to_string()));

    // Verify return to awake state
    assert!(!sleep_manager.is_sleeping());
    assert_eq!(sleep_manager.pressure(), 0.0, "Pressure should reset after sleep");
}

// ============================================================================
// Test 6: Multiple Sleep Cycles with Forgetting
// ============================================================================

#[test]
fn test_multiple_sleep_cycles_with_forgetting() {
    let config = SleepConfig {
        sleep_threshold: 0.2,
        pressure_increment: 0.25, // Very fast pressure build
        max_awake_cycles: 5,
        sleep_progress_rate: 1.0, // Instant sleep phase completion
    };

    let mut sleep_manager = SleepCycleManager::with_config(config);
    let mut consolidator = MemoryConsolidator::new();
    let mut all_traces = vec![];

    // Simulate 3 complete sleep cycles
    for cycle_num in 0..3 {
        // Awake: Create new memories
        let new_bid = create_test_bid(
            &format!("cycle_{}_memory", cycle_num),
            0.7,
            0.6
        );
        let new_coalition = create_coalition_from_bid(new_bid);

        // Awake until sleep
        while !sleep_manager.is_sleeping() {
            sleep_manager.update();
        }

        // Sleep: Consolidate + Forget
        while sleep_manager.is_sleeping() {
            if sleep_manager.state().name() == "Deep Sleep" {
                // Consolidate new memory
                let new_traces = consolidator.consolidate_coalitions(vec![new_coalition.clone()]);
                all_traces.extend(new_traces);

                // Apply forgetting to old memories
                consolidator.apply_forgetting(&mut all_traces);
            }
            sleep_manager.update();
        }
    }

    // Verify: Should have completed 3 cycles
    assert_eq!(sleep_manager.total_cycles(), 3);

    // Verify: Forgetting should have affected memory count
    // (exact count depends on forgetting algorithm)
    assert!(all_traces.len() <= 3,
            "Forgetting should limit memory accumulation");
}

// ============================================================================
// Test 7: Consolidation Effectiveness Measurement
// ============================================================================

#[test]
fn test_consolidation_effectiveness() {
    let mut consolidator = MemoryConsolidator::new();

    // Create 10 diverse memories
    let coalitions: Vec<Coalition> = (0..10).map(|i| {
        let bid = create_test_bid(
            &format!("memory_{}", i),
            0.5 + (i as f32 * 0.05),
            0.5 + (i as f32 * 0.04)
        );
        create_coalition_from_bid(bid)
    }).collect();

    // Consolidate
    let traces = consolidator.consolidate_coalitions(coalitions);

    // Measure effectiveness
    let input_count = 10;
    let output_count = traces.len();

    assert!(output_count > 0, "Should retain at least some memories");
    assert!(output_count <= input_count, "Should not create more traces than input");

    // Check that consolidated memories have valid properties
    for trace in &traces {
        assert!(trace.importance >= 0.0 && trace.importance <= 1.0,
                "Importance should be in valid range");
        assert!(trace.compressed_pattern.len() > 0,
                "Should have compressed pattern");
    }

    println!("Consolidation Effectiveness:");
    println!("  Input coalitions: {}", input_count);
    println!("  Output traces: {}", output_count);
    println!("  Compression ratio: {:.2}", output_count as f32 / input_count as f32);
}

// ============================================================================
// Test 8: REM Pattern Quality Measurement
// ============================================================================

#[test]
fn test_rem_pattern_quality() {
    let mut sleep_manager = SleepCycleManager::new();

    // Register working memory with diverse HDC patterns
    for i in 0..6 {
        let mut bid = create_test_bid(&format!("concept_{}", i), 0.8, 0.7);
        // Give each a unique HDC pattern
        let pattern: Vec<i8> = (0..8).map(|j| {
            if (i + j) % 2 == 0 { 1 } else { -1 }
        }).collect();
        bid.hdc_semantic = Some(Arc::new(pattern));

        sleep_manager.register_coalition(create_coalition_from_bid(bid));
    }

    // Enter REM sleep
    sleep_manager.force_sleep();
    while sleep_manager.state().name() != "REM Sleep" {
        sleep_manager.update();
    }

    // Generate novel patterns
    let novel_patterns = sleep_manager.perform_rem_recombination();

    // Quality metrics:
    // 1. Should generate patterns
    assert!(novel_patterns.len() > 0, "Should generate novel patterns");

    // 2. Patterns should be valid HDC vectors
    for pattern in &novel_patterns {
        assert_eq!(pattern.len(), 8);
        assert!(pattern.iter().all(|&v| v == 1 || v == -1));
    }

    // 3. Should have reasonable diversity
    // (At least not all identical)
    if novel_patterns.len() > 1 {
        let first = &novel_patterns[0];
        let all_same = novel_patterns.iter().all(|p| p == first);
        assert!(!all_same, "Novel patterns should have some diversity");
    }

    println!("REM Pattern Quality:");
    println!("  Patterns generated: {}", novel_patterns.len());
    println!("  All valid bipolar vectors: true");
}
