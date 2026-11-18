// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Swarm Cognitive Influence smoke test.
//!
//! Confirms that the cognitive loop and swarm subsystem can cycle. This is a
//! wiring smoke test, not proof of collective behavior or effect authorization.
//! The focused collective-signal calculations are unit-tested in swarm_consciousness.rs.

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

#[test]
fn swarm_subsystem_runs_without_authority_claims() {
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
    println!("  Focused collective-signal unit tests live in");
    println!("  src/cognitive_loop/managers/swarm_consciousness.rs.");
    println!("  This smoke test does not establish effect authorization.");
    println!();
    println!("  SMOKE CHECK: Swarm subsystem runs in cognitive loop.");
    println!();
}