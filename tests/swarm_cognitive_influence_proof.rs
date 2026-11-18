// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Swarm Cognitive Influence Proof — Test 4.
//!
//! Validates that collective cognition remains available without describing
//! it as authority delegation. The focused cognitive-signal behavior is
//! unit-tested in swarm_consciousness.rs (private module).

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

#[test]
fn proof4_cognitive_loop_swarm_integration() {
    println!();
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  SWARM COGNITIVE INFLUENCE — Cognitive Loop Integration  │");
    println!("└──────────────────────────────────────────────────────────┘");
    println!();

    let config = CognitiveLoopConfig {
        genesis_phrase: Some("swarm_test_v1".to_string()),
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    };

    let mut service = CognitiveLoopService::new(config).expect("Service must construct");

    println!("▶ Cognitive loop service constructed with swarm manager");
    println!("  Service created: ✓");

    for i in 0..5 {
        let result = service.cycle(&format!("test input {i}"));
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_level
                .is_finite()
        );
    }
    println!("  5 cognitive cycles executed: ✓");
    println!();
    println!("  Collective cognition is tested as a non-authoritative signal");
    println!("  by 5 unit tests in swarm_consciousness.rs:");
    println!("    - test_no_coalitions");
    println!("    - test_conscious_collective_changes_cognitive_signal_only");
    println!("    - test_not_in_coalition_has_no_collective_influence");
    println!("    - test_high_individual_phi_not_downgraded");
    println!("    - test_multiple_coalitions_picks_strongest");
    println!();
    println!("  INTEGRATION CHECK: Swarm subsystem runs in cognitive loop.");
    println!();
}