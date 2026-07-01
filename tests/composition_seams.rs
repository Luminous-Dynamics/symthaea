#![cfg(feature = "reasoning_engine")]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Composition Seam Tests
// ==================================================================================
//
// Verifies that the four subsystems (epistemic conflict, tool gate, temporal
// planning, counterfactual) compose correctly through the unified engine.
//
// Seams tested:
//   1. conflict → Φ_eff → gate (e2e)
//   2. dream → prior → MCTS → better plan
//   3. surgery → different causal → different plan
//   4. self-calibration convergence
//   5. telemetry completeness
//   6. full cycle budget compliance
//
// Run: cargo test --test composition_seams --features reasoning_engine
// ==================================================================================

use std::sync::Arc;
use symthaea::consciousness::counterfactual::{
    CausalDAG, CausalQuery, CausalQueryOutcome, CounterfactualReasoner,
};
use symthaea::consciousness::epistemic_conflict::{
    ConflictDetector, MultiTheoryMetrics, TheoryCalibrator, phi_integration::effective_phi,
};
use symthaea::consciousness::reasoning_engine::{ConsciousReasoningEngine, ReasoningContext};
use symthaea::consciousness::temporal_planning::mcts::{MctsPlanner, evs};
use symthaea::consciousness::temporal_planning::types::{
    BudgetTier, ForkedState, MctsConfig, PlannedAction, ReasoningBudget,
};
use symthaea::consciousness::tool_gate::classifier;
use symthaea::consciousness::tool_gate::types::ToolDescriptor;

// ==================================================================================
// SEAM 1: Conflict → Φ_eff → Gate (end-to-end)
// ==================================================================================

#[test]
fn seam_conflict_to_phi_eff_to_gate() {
    // High-conflict metrics → low R → low Φ_eff → gate blocks risky action
    let mut detector = ConflictDetector::new();
    let calibrator = TheoryCalibrator::new();

    // Theories in severe disagreement
    let disagreeing = MultiTheoryMetrics {
        phi: 0.9,
        gwt: 0.1,
        ast: 0.9,
        pp: 0.1,
        rpt: 0.9,
        embodiment: 0.1,
        unified: 0.5,
    };
    let conflicts = detector.detect(&disagreeing);
    assert!(conflicts.max_magnitude() > 0.5, "Should have high conflict");

    let r = calibrator.reliability(&disagreeing);
    let phi_eff = effective_phi(disagreeing.phi, r, calibrator.gamma());
    assert!(
        phi_eff < disagreeing.phi,
        "Φ_eff should be attenuated by low R"
    );

    // Try to gate a high-risk tool with the attenuated Φ_eff
    let risky_tool = ToolDescriptor::from_command("nixos-rebuild switch")
        .with_domain("nixos")
        .with_rollback("nixos-rebuild switch --rollback")
        .with_calibration_count(100);
    let gate_result = classifier::gate(&risky_tool, phi_eff, 0.8);
    assert!(
        !gate_result.is_allowed(),
        "High-risk tool should be blocked when theories disagree (Φ_eff={:.3})",
        phi_eff,
    );

    // Now with agreeing theories → high R → high Φ_eff → gate allows
    let agreeing = MultiTheoryMetrics {
        phi: 0.9,
        gwt: 0.9,
        ast: 0.9,
        pp: 0.9,
        rpt: 0.9,
        embodiment: 0.9,
        unified: 0.9,
    };
    let _conflicts2 = detector.detect(&agreeing);
    let r2 = calibrator.reliability(&agreeing);
    let phi_eff2 = effective_phi(agreeing.phi, r2, calibrator.gamma());
    assert!(
        phi_eff2 > phi_eff,
        "Agreeing theories should produce higher Φ_eff"
    );

    let gate_result2 = classifier::gate(&risky_tool, phi_eff2, 0.8);
    assert!(
        gate_result2.is_allowed(),
        "High-risk tool should be allowed when theories agree (Φ_eff={:.3})",
        phi_eff2,
    );
}

#[test]
fn seam_conflict_kind_drives_epistemic_action() {
    // Specific conflict kinds should recommend specific epistemic actions
    let mut detector = ConflictDetector::new();

    // PP is very low → UnreliablePrediction → Sense(SensorMeasurement)
    let pp_low = MultiTheoryMetrics {
        phi: 0.8,
        gwt: 0.8,
        ast: 0.8,
        pp: 0.1,
        rpt: 0.8,
        embodiment: 0.8,
        unified: 0.7,
    };
    let conflicts = detector.detect(&pp_low);
    let pp_conflict = conflicts.conflicts.iter().find(|c| {
        c.kind == symthaea::consciousness::epistemic_conflict::ConflictKind::UnreliablePrediction
    });
    assert!(pp_conflict.is_some(), "Should detect UnreliablePrediction");

    // The recommended action should be Sense with a sensor anchor
    let action = pp_conflict.unwrap().recommended_action;
    assert!(
        matches!(
            action,
            symthaea::consciousness::epistemic_conflict::EpistemicAction::Sense(_)
        ),
        "UnreliablePrediction should recommend Sense, got {:?}",
        action,
    );
}

// ==================================================================================
// SEAM 2: EVS → MCTS → Action selection
// ==================================================================================

#[test]
fn seam_evs_gates_mcts() {
    // Low R → EVS = 0 → no simulation
    let evs_low_r = evs(0.5, 0.1, 5, 0.5);
    assert_eq!(evs_low_r, 0.0, "EVS should be 0 when R < R_SIM_MIN");

    // Moderate R + high conflict entropy → positive EVS → simulation runs
    let evs_moderate = evs(0.8, 0.4, 5, 0.5);
    assert!(evs_moderate > 0.0, "EVS should be positive with moderate R");

    // Verify MCTS actually runs when EVS is above threshold
    let state = ForkedState::new(
        Arc::new(vec![1.0; 4]),
        vec![vec![0.0; 4]],
        vec![1.0],
        [42u8; 32],
    );
    let actions = vec![
        PlannedAction {
            id: "explore".into(),
            description: "Explore".into(),
            embedding: vec![0.5; 4],
            prior: 0.6,
            is_epistemic: true,
        },
        PlannedAction {
            id: "execute".into(),
            description: "Execute".into(),
            embedding: vec![1.0; 4],
            prior: 0.4,
            is_epistemic: false,
        },
    ];

    let budget = ReasoningBudget::new(10_000, 0.8);
    let config = MctsConfig::tier1();
    let result = MctsPlanner::plan(&state, &actions, &config, &budget, &Default::default());
    assert!(result.did_plan);
    assert!(result.iterations > 0);
    assert!(result.confidence > 0.0);
}

#[test]
fn seam_dream_priors_bias_mcts() {
    // Actions with higher priors should be selected more often
    let state = ForkedState::new(
        Arc::new(vec![0.0; 4]),
        vec![vec![0.0; 4]],
        vec![1.0],
        [42u8; 32],
    );

    // Action 0 has a very high prior (dream says it's good)
    let actions = vec![
        PlannedAction {
            id: "dream_favored".into(),
            description: "Dream-favored action".into(),
            embedding: vec![0.1; 4],
            prior: 0.9, // high dream prior
            is_epistemic: false,
        },
        PlannedAction {
            id: "unfavored".into(),
            description: "Unfavored action".into(),
            embedding: vec![0.1; 4],
            prior: 0.01, // low dream prior
            is_epistemic: false,
        },
    ];

    let budget = ReasoningBudget::new(20_000, 0.9);
    let config = MctsConfig::tier2();
    let result = MctsPlanner::plan(&state, &actions, &config, &budget, &Default::default());

    // With identical embeddings but different priors, the dream-favored
    // action should win due to prior bonus in UCB1
    assert_eq!(
        result.best_action_idx,
        Some(0),
        "Dream-favored action (prior=0.9) should be selected over unfavored (prior=0.01)"
    );
}

// ==================================================================================
// SEAM 3: Causal analysis → different outcomes
// ==================================================================================

#[test]
fn seam_different_dag_different_outcome() {
    let reasoner = CounterfactualReasoner::new();

    // DAG 1: X → Y (identifiable)
    let dag1 = CausalDAG::new(vec!["X".into(), "Y".into()], vec![(0, 1)]);
    let query = CausalQuery {
        treatment: 0,
        outcome: 1,
        conditioning: vec![],
    };
    let outcome1 = reasoner.query(&dag1, &query);
    assert!(matches!(outcome1, CausalQueryOutcome::Identified { .. }));

    // DAG 2: X  Y (no connection → unidentifiable)
    let dag2 = CausalDAG::new(vec!["X".into(), "Y".into()], vec![]);
    let outcome2 = reasoner.query(&dag2, &query);
    assert!(matches!(outcome2, CausalQueryOutcome::Unidentified { .. }));

    // DAG 3: X → M → Y with X → Y (backdoor-identifiable via M)
    let dag3 = CausalDAG::new(
        vec!["X".into(), "M".into(), "Y".into()],
        vec![(0, 1), (1, 2), (0, 2)],
    );
    let query3 = CausalQuery {
        treatment: 0,
        outcome: 2,
        conditioning: vec![],
    };
    let outcome3 = reasoner.query(&dag3, &query3);
    // Should be identifiable (backdoor through M or empty set)
    assert!(
        matches!(outcome3, CausalQueryOutcome::Identified { .. }),
        "DAG with multiple paths should still be identifiable"
    );
}

// ==================================================================================
// SEAM 4: Self-calibration convergence
// ==================================================================================

#[test]
fn seam_calibration_converges_with_good_data() {
    let mut calibrator = TheoryCalibrator::new();

    // Feed consistently accurate predictions for IIT
    for _ in 0..50 {
        calibrator.update_theory(
            symthaea::consciousness::epistemic_conflict::TheoryId::IIT,
            0.8,  // predicted
            0.82, // actual (close to prediction)
        );
    }

    let iit_cal = calibrator
        .calibrations
        .get(symthaea::consciousness::epistemic_conflict::TheoryId::IIT);

    // After good predictions, reliability should increase from 0.5 default
    assert!(
        iit_cal.reliability > 0.5,
        "Reliability should increase with accurate predictions, got {}",
        iit_cal.reliability,
    );
    assert!(
        iit_cal.brier_score < 0.25, // better than uninformative prior
        "Brier score should improve with accurate predictions, got {}",
        iit_cal.brier_score,
    );
}

#[test]
fn seam_calibration_bounded_under_adversarial_input() {
    // INV-9: Each single recalibration step must be bounded by Δγ ≤ 0.1.
    // Recalibration fires every 50 outcomes after the initial 200,
    // so we track per-step changes.
    let mut calibrator = TheoryCalibrator::new();
    let mut prev_gamma = calibrator.gamma();

    // Feed a streak of "perfect" outcomes to try to lower gamma aggressively
    for i in 0..350 {
        calibrator.record_outcome(true, true); // every gated action was good
        let current_gamma = calibrator.gamma();
        let delta = (current_gamma - prev_gamma).abs();
        assert!(
            delta <= 0.1 + 1e-10,
            "INV-9: single-step Δγ = {} at outcome {}, must be ≤ 0.1",
            delta,
            i,
        );
        prev_gamma = current_gamma;
    }

    // Now feed a streak of bad outcomes
    for i in 0..350 {
        calibrator.record_outcome(true, false); // every gated action was bad
        let current_gamma = calibrator.gamma();
        let delta = (current_gamma - prev_gamma).abs();
        assert!(
            delta <= 0.1 + 1e-10,
            "INV-9: single-step Δγ = {} at bad outcome {}, must be ≤ 0.1",
            delta,
            i,
        );
        prev_gamma = current_gamma;
    }

    // γ should still be in global bounds [1.0, 4.0]
    assert!(calibrator.gamma() >= 1.0 && calibrator.gamma() <= 4.0);
}

// ==================================================================================
// SEAM 5: Telemetry completeness
// ==================================================================================

#[test]
fn seam_telemetry_complete_every_cycle() {
    let mut engine = ConsciousReasoningEngine::new();

    // Run across different tiers
    let budgets = [500, 5_000, 25_000]; // Tier 0, 1, 2
    for (i, &budget_us) in budgets.iter().enumerate() {
        let ctx = ReasoningContext {
            theory_metrics: MultiTheoryMetrics {
                phi: 0.8,
                gwt: 0.8,
                ast: 0.8,
                pp: 0.8,
                rpt: 0.8,
                embodiment: 0.8,
                unified: 0.8,
            },
            phi: 0.8,
            available_budget_us: budget_us,
            available_actions: vec![PlannedAction {
                id: "test".into(),
                description: "Test action".into(),
                embedding: vec![0.5; 4],
                prior: 0.5,
                is_epistemic: true,
            }],
            tool: Some(ToolDescriptor::read_only("nix search")),
            recent_utility: 0.5,
            cycle_id: i as u64,
            neuromod_exploration_mod: 1.0,
            epistemic_quality: 0.5,
        };
        engine.reason(&ctx);
    }

    // Every cycle must have emitted a telemetry event
    assert_eq!(engine.recent_events().len(), 3);

    for event in engine.recent_events() {
        // Required fields must be populated
        assert!(event.phi_raw > 0.0, "phi_raw must be set");
        assert!(event.reliability > 0.0, "reliability must be set");
        assert!(event.phi_eff > 0.0, "phi_eff must be set");
        assert!(event.gamma > 0.0, "gamma must be set");
        assert!(event.wall_time_us > 0, "wall_time must be set");
        assert!(
            !event.tier_selected_reason.is_empty(),
            "tier reason must be set"
        );

        // Theory metrics must have 6 entries
        assert_eq!(event.theory_metrics.len(), 6);
        assert_eq!(event.theory_reliabilities.len(), 6);
    }

    // Check tier distribution
    let tiers: Vec<_> = engine
        .recent_events()
        .iter()
        .map(|e| e.budget_tier)
        .collect();
    assert_eq!(tiers[0], BudgetTier::Tier0);
    // Tier 1 or 2 for the second (5ms budget, high R)
    assert!(tiers[1] == BudgetTier::Tier1 || tiers[1] == BudgetTier::Tier2);
    assert_eq!(tiers[2], BudgetTier::Tier2);
}

#[test]
fn seam_telemetry_includes_gate_info() {
    let mut engine = ConsciousReasoningEngine::new();

    let ctx = ReasoningContext {
        theory_metrics: MultiTheoryMetrics {
            phi: 0.8,
            gwt: 0.8,
            ast: 0.8,
            pp: 0.8,
            rpt: 0.8,
            embodiment: 0.8,
            unified: 0.8,
        },
        phi: 0.8,
        available_budget_us: 25_000,
        available_actions: vec![],
        tool: Some(
            ToolDescriptor::from_command("nixos-rebuild switch")
                .with_domain("nixos")
                .with_rollback("nixos-rebuild switch --rollback")
                .with_calibration_count(100),
        ),
        recent_utility: 0.5,
        cycle_id: 0,
        neuromod_exploration_mod: 1.0,
        epistemic_quality: 0.5,
    };

    engine.reason(&ctx);
    let event = engine.last_event().unwrap();

    // Gate info should be populated
    assert!(event.risk_level.is_some());
    assert!(event.required_phi > 0.0);
    assert!(!event.gate_decision.is_empty());
}

// ==================================================================================
// SEAM 6: Full cycle budget compliance
// ==================================================================================

#[test]
fn seam_tier0_completes_within_budget() {
    let mut engine = ConsciousReasoningEngine::new();

    let ctx = ReasoningContext {
        theory_metrics: MultiTheoryMetrics {
            phi: 0.8,
            gwt: 0.8,
            ast: 0.8,
            pp: 0.8,
            rpt: 0.8,
            embodiment: 0.8,
            unified: 0.8,
        },
        phi: 0.8,
        available_budget_us: 1_000, // 1ms → Tier 0
        available_actions: vec![],
        tool: None,
        recent_utility: 0.5,
        cycle_id: 0,
        neuromod_exploration_mod: 1.0,
        epistemic_quality: 0.5,
    };

    let start = std::time::Instant::now();
    let result = engine.reason(&ctx);
    let elapsed = start.elapsed();

    assert_eq!(result.tier, BudgetTier::Tier0);
    // Tier 0 should be very fast (well under 2ms on any modern hardware)
    assert!(
        elapsed.as_millis() < 10,
        "Tier 0 took {}ms, should be <10ms",
        elapsed.as_millis(),
    );
}

#[test]
fn seam_engine_composition_e2e() {
    // Full end-to-end: engine processes multiple cycles with varying conditions
    let mut engine = ConsciousReasoningEngine::new();

    // Cycle 1: Good conditions, full budget
    let ctx1 = ReasoningContext {
        theory_metrics: MultiTheoryMetrics {
            phi: 0.9,
            gwt: 0.9,
            ast: 0.9,
            pp: 0.9,
            rpt: 0.9,
            embodiment: 0.9,
            unified: 0.9,
        },
        phi: 0.9,
        available_budget_us: 25_000,
        available_actions: vec![PlannedAction {
            id: "a".into(),
            description: "Explore".into(),
            embedding: vec![0.5; 4],
            prior: 0.5,
            is_epistemic: true,
        }],
        tool: Some(ToolDescriptor::read_only("nix search")),
        recent_utility: 0.5,
        cycle_id: 1,
        neuromod_exploration_mod: 1.0,
        epistemic_quality: 0.5,
    };
    let r1 = engine.reason(&ctx1);
    assert!(r1.phi_eff > 0.5, "High consensus should produce high Φ_eff");
    assert!(r1.action_allowed(), "Read-only tool should be allowed");

    // Record posthoc: the action was good
    engine.record_posthoc(true, true, 0.05);

    // Cycle 2: Bad conditions, theories disagree
    let ctx2 = ReasoningContext {
        theory_metrics: MultiTheoryMetrics {
            phi: 0.9,
            gwt: 0.1,
            ast: 0.9,
            pp: 0.1,
            rpt: 0.9,
            embodiment: 0.1,
            unified: 0.5,
        },
        phi: 0.9,
        available_budget_us: 25_000,
        available_actions: vec![],
        tool: Some(
            ToolDescriptor::from_command("nixos-rebuild switch")
                .with_domain("nixos")
                .with_rollback("nixos-rebuild switch --rollback")
                .with_calibration_count(100),
        ),
        recent_utility: 0.3,
        cycle_id: 2,
        neuromod_exploration_mod: 1.0,
        epistemic_quality: 0.5,
    };
    let r2 = engine.reason(&ctx2);
    assert!(
        r2.phi_eff < r1.phi_eff,
        "Disagreeing theories should lower Φ_eff"
    );

    // Verify posthoc was attached
    let events: Vec<_> = engine.recent_events().iter().collect();
    assert!(
        events[1].posthoc_outcome.is_some(),
        "Posthoc should be attached to cycle 2"
    );

    // Stats should reflect both cycles
    let stats = engine.stats();
    assert_eq!(stats.total_cycles, 2);
    assert!(stats.avg_phi_eff > 0.0);
}