// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pipeline Integration Tests
//!
//! End-to-end tests verifying state flows correctly across the 7 pipeline phases:
//!   Observe → Encode → InferGoal → PlanActions → PhiGate → Execute → Learn
//!
//! These tests use mock system data to exercise each phase boundary without
//! requiring a live NixOS system.

use nixward::encoding::{NixCodebook, ServiceState, SystemStateEncoder, SystemStateSnapshot};
use nixward::mind::NixWorldModel;
use nixward::mind::active_inference::NixActiveInference;
use nixward::mind::causal_graph::NixCausalGraph;
use nixward::mind::episodic_memory::EpisodeOutcome;
use nixward::mind::world_model::ActionCategory;
use nixward::plugin::pipeline_integration::{NixConsciousnessQuadrant, NixPipelineProcessor};
use symthaea_core::hdc::ContinuousHV;

/// Helper: build a mock system snapshot with configurable services.
fn mock_snapshot(services: &[(&str, ServiceState)]) -> SystemStateSnapshot {
    SystemStateSnapshot {
        services: services
            .iter()
            .map(|(name, state)| (name.to_string(), *state))
            .collect(),
        packages: vec!["firefox".into(), "vim".into(), "git".into()],
        generation: Some(42),
        store_size_bytes: Some(10_000_000_000),
        store_path_count: Some(3000),
        config_options: vec![
            ("services.nginx.enable".into(), "true".into()),
            ("services.openssh.enable".into(), "true".into()),
        ],
    }
}

/// Helper: build a healthy system snapshot.
fn healthy_snapshot() -> SystemStateSnapshot {
    mock_snapshot(&[
        ("nginx.service", ServiceState::Running),
        ("sshd.service", ServiceState::Running),
        ("postgresql.service", ServiceState::Running),
    ])
}

/// Helper: build a degraded snapshot with a failed service.
fn degraded_snapshot() -> SystemStateSnapshot {
    mock_snapshot(&[
        ("nginx.service", ServiceState::Failed),
        ("sshd.service", ServiceState::Running),
        ("postgresql.service", ServiceState::Running),
    ])
}

// ─── Test 1: Observe → Encode state flow ───────────────────────────────────

#[test]
fn test_observe_to_encode_state_flow() {
    // Phase 1: Observe — create mock system state
    let snapshot = healthy_snapshot();
    assert_eq!(snapshot.services.len(), 3);
    assert_eq!(snapshot.generation, Some(42));

    // Phase 2: Encode — transform into HDC vectors
    let mut codebook = NixCodebook::new();
    let state_hv = {
        let mut encoder = SystemStateEncoder::new(&mut codebook);
        encoder.encode_snapshot(&snapshot)
    };

    // Verify the encoded vector is valid
    assert!(
        state_hv.dim() > 0,
        "Encoded vector should have positive dimension"
    );
    assert!(
        state_hv.norm() > 0.0,
        "Encoded vector should have non-zero norm"
    );

    // Verify that different system states produce different encodings
    let degraded = degraded_snapshot();
    let degraded_hv = {
        let mut encoder = SystemStateEncoder::new(&mut codebook);
        encoder.encode_snapshot(&degraded)
    };

    let similarity = state_hv.similarity(&degraded_hv);
    assert!(
        similarity < 0.99,
        "Different system states should produce different encodings (sim={:.4})",
        similarity,
    );

    // Verify the world model can consume the encoded state
    let mut world_model = NixWorldModel::default();
    world_model.observe(state_hv.clone());
    let free_energy = world_model.free_energy();
    assert!(free_energy.is_finite(), "Free energy should be finite");
}

// ─── Test 2: Full pipeline no panic ────────────────────────────────────────

#[test]
fn test_full_pipeline_no_panic() {
    // Run all 7 pipeline stages with mock data — verify no panics and valid output

    // Stage 1: Observe
    let snapshot = healthy_snapshot();

    // Stage 2: Encode
    let mut codebook = NixCodebook::new();
    let state_hv = {
        let mut encoder = SystemStateEncoder::new(&mut codebook);
        encoder.encode_snapshot(&snapshot)
    };

    // Stage 3: Feed into active inference engine (world model + goal inference)
    let mut engine = NixActiveInference::new();
    engine.observe_state(state_hv.clone());

    // Stage 4: Plan actions from user input
    let plan = engine.process_input("install nginx");
    assert!(
        !plan.needs_clarification,
        "Clear goal should not need clarification"
    );
    assert!(
        !plan.actions.is_empty(),
        "Plan should have at least one action"
    );

    // Verify EFE ordering
    for window in plan.actions.windows(2) {
        assert!(
            window[0].expected_free_energy <= window[1].expected_free_energy + 1e-10,
            "Actions should be sorted by EFE"
        );
    }

    // Stage 5: Phi Gate — check consciousness quadrant
    let phi = 0.7;
    let confidence = 0.8;
    let quadrant = NixConsciousnessQuadrant::from_metrics(phi, confidence);
    assert_eq!(quadrant, NixConsciousnessQuadrant::Confident);
    assert!(quadrant.allows_execution());

    // Stage 6: Execute (skipped in test — would require live system)

    // Stage 7: Learn from outcome
    let state_after = ContinuousHV::random(state_hv.dim(), 42);
    engine.learn_from_outcome(
        &state_hv,
        ActionCategory::Install,
        &state_after,
        EpisodeOutcome::Success,
        phi,
    );
    assert_eq!(engine.episode_count(), 1);
}

// ─── Test 3: Gate veto skips execute ───────────────────────────────────────

#[test]
fn test_gate_veto_skips_execute() {
    let mut proc = NixPipelineProcessor::new()
        .with_skip_observe(true)
        .with_phi_threshold(0.5);

    // Low Phi + low confidence = Confused quadrant → should veto
    let result = proc.process("install firefox", 0.2, 0.2);
    assert_eq!(result.quadrant, NixConsciousnessQuadrant::Confused);
    assert!(!result.phi_allowed, "Low Phi should block execution");
    assert!(
        result.quadrant.suggests_clarification(),
        "Confused state should suggest clarification"
    );

    // High Phi + low confidence = Curious → should also veto
    let result2 = proc.process("remove nginx", 0.8, 0.3);
    assert_eq!(result2.quadrant, NixConsciousnessQuadrant::Curious);
    assert!(!result2.phi_allowed, "Curious state should block execution");

    // Low Phi + high confidence = Habitual → should veto (Phi below threshold)
    let result3 = proc.process("install vim", 0.3, 0.8);
    assert_eq!(result3.quadrant, NixConsciousnessQuadrant::Habitual);
    assert!(
        !result3.phi_allowed,
        "Phi below threshold should block execution even in Habitual"
    );

    // High Phi + high confidence = Confident → should allow
    let result4 = proc.process("install vim", 0.7, 0.8);
    assert_eq!(result4.quadrant, NixConsciousnessQuadrant::Confident);
    assert!(
        result4.phi_allowed,
        "Confident with high Phi should allow execution"
    );
}

// ─── Test 4: Learn phase updates state ─────────────────────────────────────

#[test]
fn test_learn_phase_updates_state() {
    let dim = symthaea_core::hdc::HDC_DIMENSION;
    let mut engine = NixActiveInference::new();

    // Initial state
    let state_before = ContinuousHV::random(dim, 1);
    engine.observe_state(state_before.clone());

    // Verify the world model hasn't learned Install transitions yet
    assert!(
        !engine.world_model().has_learned(&ActionCategory::Install),
        "World model should not know Install before learning"
    );
    assert_eq!(engine.episode_count(), 0);

    // Learn from several install outcomes
    for i in 0..5 {
        let before = ContinuousHV::random(dim, i * 100 + 1);
        let after = ContinuousHV::random(dim, i * 100 + 2);
        engine.learn_from_outcome(
            &before,
            ActionCategory::Install,
            &after,
            EpisodeOutcome::Success,
            0.7,
        );
    }

    // Verify learning happened
    assert!(
        engine.world_model().has_learned(&ActionCategory::Install),
        "World model should know Install after learning"
    );
    assert_eq!(engine.episode_count(), 5);

    // Verify that predictions change after learning
    let predicted = engine.world_model().predict_state(&ActionCategory::Install);
    assert!(predicted.dim() > 0);
    assert!(
        predicted.norm() > 0.0,
        "Predicted state should be non-zero after learning"
    );

    // Learn from a failure — episodic memory should contain both successes and failures
    let fail_before = ContinuousHV::random(dim, 999);
    let fail_after = ContinuousHV::random(dim, 1000);
    engine.learn_from_outcome(
        &fail_before,
        ActionCategory::Rebuild,
        &fail_after,
        EpisodeOutcome::Failure("build error".into()),
        0.5,
    );
    assert_eq!(engine.episode_count(), 6);
}

// ─── Test 5: Pipeline idempotent on no change ──────────────────────────────

#[test]
fn test_pipeline_idempotent_on_no_change() {
    let snapshot = healthy_snapshot();
    let mut codebook = NixCodebook::new();

    // Encode the same snapshot twice
    let hv1 = {
        let mut encoder = SystemStateEncoder::new(&mut codebook);
        encoder.encode_snapshot(&snapshot)
    };
    let hv2 = {
        let mut encoder = SystemStateEncoder::new(&mut codebook);
        encoder.encode_snapshot(&snapshot)
    };

    // Same input should produce identical output (deterministic encoding)
    let similarity = hv1.similarity(&hv2);
    assert!(
        (similarity - 1.0).abs() < 1e-6,
        "Same input should produce identical encoding (sim={:.6})",
        similarity,
    );

    // Feed same state into pipeline twice — results should be consistent
    let mut proc = NixPipelineProcessor::new().with_skip_observe(true);

    let result1 = proc.process("install firefox", 0.7, 0.8);
    let result2 = proc.process("install firefox", 0.7, 0.8);

    assert_eq!(result1.quadrant, result2.quadrant);
    assert_eq!(result1.phi_allowed, result2.phi_allowed);
    assert_eq!(result1.safety_level, result2.safety_level);
    // Plans should have the same number of actions
    assert_eq!(result1.plan.actions.len(), result2.plan.actions.len());

    // Verify both runs produce actions in the same order
    for (a1, a2) in result1.plan.actions.iter().zip(result2.plan.actions.iter()) {
        assert_eq!(
            format!("{:?}", a1.action),
            format!("{:?}", a2.action),
            "Same input should produce the same action ordering"
        );
    }
}

// ─── Test 6: Observe-Encode-Learn with causal graph ────────────────────────

#[test]
fn test_causal_graph_accumulates_through_pipeline() {
    let mut codebook = NixCodebook::new();
    let mut world_model = NixWorldModel::default();
    let mut causal_graph = NixCausalGraph::new(42);

    // Simulate several cycles with state transitions
    let states = vec![
        mock_snapshot(&[
            ("nginx.service", ServiceState::Running),
            ("postgresql.service", ServiceState::Running),
        ]),
        mock_snapshot(&[
            ("nginx.service", ServiceState::Failed),
            ("postgresql.service", ServiceState::Running),
        ]),
        mock_snapshot(&[
            ("nginx.service", ServiceState::Running),
            ("postgresql.service", ServiceState::Failed),
        ]),
        mock_snapshot(&[
            ("nginx.service", ServiceState::Running),
            ("postgresql.service", ServiceState::Running),
        ]),
    ];

    let mut prev_snap: Option<SystemStateSnapshot> = None;

    for snapshot in &states {
        let state_hv = {
            let mut encoder = SystemStateEncoder::new(&mut codebook);
            encoder.encode_snapshot(snapshot)
        };
        world_model.observe(state_hv);

        // If there was a previous snapshot, detect transitions and update causal graph
        if let Some(prev) = &prev_snap {
            let changed: Vec<String> = snapshot
                .services
                .iter()
                .zip(prev.services.iter())
                .filter(|((_, s1), (_, s2))| s1 != s2)
                .map(|((name, _), _)| name.clone())
                .collect();

            if !changed.is_empty() {
                let refs: Vec<&str> = changed.iter().map(|s| s.as_str()).collect();
                let all_keys: Vec<&str> =
                    snapshot.services.iter().map(|(n, _)| n.as_str()).collect();
                causal_graph.observe_outcome(&changed[0], &refs, &all_keys);
            }
        }
        prev_snap = Some(snapshot.clone());
    }

    // The causal graph should have accumulated some edges
    assert!(
        causal_graph.edge_count() > 0,
        "Causal graph should have accumulated edges from observed transitions"
    );
}

// ─── Test 7: Pipeline processor processes ambiguous input ──────────────────

#[test]
fn test_ambiguous_input_needs_clarification() {
    let mut proc = NixPipelineProcessor::new().with_skip_observe(true);

    let result = proc.process("help", 0.7, 0.8);
    assert!(
        result.plan.needs_clarification,
        "Ambiguous input 'help' should need clarification"
    );
    assert!(
        result.plan.actions.is_empty(),
        "Ambiguous input should produce no actions"
    );
}

// ─── Test 8: Multiple pipeline cycles converge ─────────────────────────────

#[test]
fn test_multiple_cycles_free_energy_finite() {
    let mut codebook = NixCodebook::new();
    let mut world_model = NixWorldModel::default();

    // Run 20 observation cycles
    for i in 0..20 {
        let snapshot = mock_snapshot(&[
            ("nginx.service", ServiceState::Running),
            (
                "sshd.service",
                if i % 5 == 0 {
                    ServiceState::Failed
                } else {
                    ServiceState::Running
                },
            ),
        ]);
        let state_hv = {
            let mut encoder = SystemStateEncoder::new(&mut codebook);
            encoder.encode_snapshot(&snapshot)
        };
        world_model.observe(state_hv);
        let fe = world_model.free_energy();
        assert!(
            fe.is_finite(),
            "Free energy should remain finite at cycle {}",
            i
        );
    }
}
