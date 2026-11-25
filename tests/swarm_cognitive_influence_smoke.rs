// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Swarm Cognitive Influence smoke test.
//!
//! Confirms that the cognitive loop and swarm subsystem can cycle. This is a
//! wiring smoke test, not proof of collective behavior or effect authorization.
//! Focused collective-signal calculations are unit-tested in the module.

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

#[test]
fn swarm_subsystem_runs_without_authority_claims() {
    let config = CognitiveLoopConfig {
        genesis_phrase: Some("swarm_test_v1".to_string()),
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    };

    let mut service = CognitiveLoopService::new(config).expect("Service must construct");

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
}
