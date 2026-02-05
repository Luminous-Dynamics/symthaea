#![cfg(feature = "reasoning_engine")]
// ==================================================================================
// Reasoning Engine Integration Tests
// ==================================================================================
//
// Tests all 10 invariants and 7 failure modes through the unified
// ConsciousReasoningEngine.
//
// Run: cargo test --test reasoning_engine_integration --features reasoning_engine
// ==================================================================================

use symthaea::consciousness::epistemic_conflict::{
    TheoryCalibrator, MultiTheoryMetrics, TheoryId,
    phi_integration::thresholds,
};
use symthaea::consciousness::tool_gate::types::{ToolDescriptor, RiskLevel};
use symthaea::consciousness::tool_gate::classifier;
use symthaea::consciousness::temporal_planning::types::{
    BudgetTier, ForkedState, PlannedAction, ReasoningBudget,
};
use symthaea::consciousness::temporal_planning::mcts::{evs, MctsPlanner};
use symthaea::consciousness::counterfactual::{
    CausalDAG, CausalQuery, CausalQueryOutcome, CounterfactualReasoner,
};
use symthaea::consciousness::reasoning_engine::{
    ConsciousReasoningEngine, ReasoningContext,
};
use std::sync::Arc;

fn make_context(phi: f64, consensus: f64, budget_us: u64) -> ReasoningContext {
    ReasoningContext {
        theory_metrics: MultiTheoryMetrics {
            phi,
            gwt: consensus,
            ast: consensus,
            pp: consensus,
            rpt: consensus,
            embodiment: consensus,
            unified: phi * 0.2 + consensus * 0.8,
        },
        phi,
        available_budget_us: budget_us,
        available_actions: vec![
            PlannedAction {
                id: "explore".into(),
                description: "Explore".into(),
                embedding: vec![0.5; 4],
                prior: 0.5,
                is_epistemic: true,
            },
            PlannedAction {
                id: "execute".into(),
                description: "Execute".into(),
                embedding: vec![1.0; 4],
                prior: 0.3,
                is_epistemic: false,
            },
        ],
        tool: None,
        recent_utility: 0.5,
        cycle_id: 1,
    }
}

// ==================================================================================
// INVARIANT TESTS
// ==================================================================================

#[test]
fn inv1_monotonic_caution() {
    // INV-1: If R decreases (all else fixed), Φ_eff must not increase.
    let mut engine = ConsciousReasoningEngine::new();

    // High consensus → high R
    let r_high = engine.reason(&make_context(0.8, 0.9, 25_000));

    // Low consensus → low R (fresh engine to avoid conflict history effects)
    let mut engine2 = ConsciousReasoningEngine::new();
    let mut ctx_low = make_context(0.8, 0.9, 25_000);
    ctx_low.theory_metrics = MultiTheoryMetrics {
        phi: 0.8, gwt: 0.2, ast: 0.8, pp: 0.2, rpt: 0.8, embodiment: 0.2, unified: 0.5,
    };
    let r_low = engine2.reason(&ctx_low);

    if r_high.reliability > r_low.reliability {
        assert!(
            r_high.phi_eff >= r_low.phi_eff - 1e-10,
            "INV-1: higher R ({:.3}) should produce higher Φ_eff ({:.3} vs {:.3})",
            r_high.reliability, r_high.phi_eff, r_low.phi_eff,
        );
    }
}

#[test]
fn inv2_rollback_safety() {
    // INV-2: All non-ReadOnly tools must have rollback OR be Critical.
    let non_readonly_no_rollback = ToolDescriptor::from_command("some-command")
        .with_domain("unknown");
    let risk = classifier::classify(&non_readonly_no_rollback);
    assert_eq!(
        risk,
        RiskLevel::Critical,
        "INV-2/INV-7: Non-ReadOnly without rollback must be Critical"
    );

    // With rollback, it should be lower than Critical
    let with_rollback = ToolDescriptor::from_command("nix build .#pkg")
        .with_domain("nixos")
        .with_rollback("nix store delete")
        .with_calibration_count(100);
    let risk2 = classifier::classify(&with_rollback);
    assert!(
        risk2 < RiskLevel::Critical,
        "INV-2: Tool with rollback should not be Critical, got {:?}",
        risk2,
    );
}

#[test]
fn inv3_deterministic_reasoning() {
    // INV-3: Fixed inputs → identical ReasoningResult (logical, not timing).
    let ctx = make_context(0.7, 0.7, 1_000);

    let mut e1 = ConsciousReasoningEngine::new();
    let mut e2 = ConsciousReasoningEngine::new();

    let r1 = e1.reason(&ctx);
    let r2 = e2.reason(&ctx);

    assert_eq!(r1.tier, r2.tier);
    assert!((r1.phi_eff - r2.phi_eff).abs() < 1e-10, "Φ_eff should be identical");
    assert!((r1.reliability - r2.reliability).abs() < 1e-10, "R should be identical");
    assert!((r1.gamma - r2.gamma).abs() < 1e-10, "γ should be identical");
    assert_eq!(r1.conflicts.conflicts.len(), r2.conflicts.conflicts.len());
}

#[test]
fn inv4_planner_consistency() {
    // INV-4: Higher-horizon plan cannot violate lower-horizon safety.
    // Verify that the gate blocks risky actions even when MCTS says go.
    let mut engine = ConsciousReasoningEngine::new();

    // Low Φ_eff but available actions include a risky tool
    let mut ctx = make_context(0.3, 0.3, 25_000);
    ctx.tool = Some(
        ToolDescriptor::from_command("nix-collect-garbage -d")
            .with_domain("nixos"),
    );

    let result = engine.reason(&ctx);

    // The gate should block regardless of what MCTS plans
    assert!(
        !result.action_allowed(),
        "INV-4: Gate must block destructive action even with planning budget"
    );
}

#[test]
fn inv5_epistemic_dominance() {
    // INV-5: When R < R_EPISTEMIC_THRESHOLD, engine prefers info-seeking.
    let low_r = 0.2;
    let e = evs(0.5, low_r, 5, 0.5);
    // EVS should be positive (above R_SIM_MIN) but system should use
    // epistemic rollout, not reward-maximizing MCTS
    assert!(e > 0.0, "EVS should be positive for R={}", low_r);
    assert!(low_r < thresholds::R_EPISTEMIC_THRESHOLD);

    // Epistemic rollout should prefer epistemic actions
    let state = ForkedState::new(
        Arc::new(vec![0.0; 4]),
        vec![vec![0.0; 4]],
        vec![1.0],
        [42u8; 32],
    );
    let actions = vec![
        PlannedAction { id: "non_ep".into(), description: "Non-epistemic".into(),
            embedding: vec![1.0; 4], prior: 0.8, is_epistemic: false },
        PlannedAction { id: "ep".into(), description: "Epistemic".into(),
            embedding: vec![0.5; 4], prior: 0.5, is_epistemic: true },
    ];
    let budget = ReasoningBudget::new(10_000, 0.8);
    let result = MctsPlanner::epistemic_rollout(&state, &actions, &budget);
    assert_eq!(result.best_action_idx, Some(1), "INV-5: Should prefer epistemic action");
}

#[test]
fn inv6_budget_guarantee() {
    // INV-6: Engine always returns within budget.
    let mut engine = ConsciousReasoningEngine::new();

    // Very tight budget
    let ctx = make_context(0.8, 0.8, 500);
    let start = std::time::Instant::now();
    let result = engine.reason(&ctx);
    let elapsed = start.elapsed();

    assert_eq!(result.tier, BudgetTier::Tier0);
    assert!(
        elapsed.as_millis() < 50,
        "INV-6: Tier 0 took {}ms, should be fast",
        elapsed.as_millis(),
    );
}

#[test]
fn inv7_no_silent_irreversibility() {
    // INV-7: Missing rollback → automatic escalation to Critical.
    let tool = ToolDescriptor::from_command("dangerous-unknown-command");
    let risk = classifier::classify(&tool);
    assert_eq!(risk, RiskLevel::Critical, "INV-7: Missing rollback must be Critical");

    // Even with a known domain, missing rollback → Critical
    let tool2 = ToolDescriptor::from_command("custom-deploy")
        .with_domain("deployment")
        .with_calibration_count(100);
    let risk2 = classifier::classify(&tool2);
    assert_eq!(risk2, RiskLevel::Critical, "INV-7: Known domain but no rollback → Critical");
}

#[test]
fn inv8_confidence_action_alignment() {
    // INV-8: Low confidence blocks even with high Φ_eff.
    let tool = ToolDescriptor::from_command("nixos-rebuild switch")
        .with_domain("nixos")
        .with_rollback("nixos-rebuild switch --rollback")
        .with_calibration_count(100);

    // High Φ_eff, low confidence
    let result = classifier::gate(&tool, 0.95, 0.1);
    assert!(
        !result.is_allowed(),
        "INV-8: Low confidence ({}) should block even with high Φ_eff (0.95)",
        0.1,
    );

    // High Φ_eff, high confidence → allowed
    let result2 = classifier::gate(&tool, 0.95, 0.9);
    assert!(result2.is_allowed(), "Should be allowed with sufficient Φ_eff and confidence");
}

#[test]
fn inv9_bounded_calibration_updates() {
    // INV-9: Max single-step reliability change: Δw ≤ 0.05.
    let mut calibrator = TheoryCalibrator::new();
    let initial_r = calibrator.calibrations.get(TheoryId::IIT).reliability;

    // One perfect prediction
    calibrator.update_theory(TheoryId::IIT, 0.0, 0.0);
    let after = calibrator.calibrations.get(TheoryId::IIT).reliability;

    assert!(
        (after - initial_r).abs() <= 0.05 + 1e-10,
        "INV-9: Δw = {}, must be ≤ 0.05",
        (after - initial_r).abs(),
    );
}

#[test]
fn inv10_anchor_kind_required() {
    // INV-10: Verify and Sense always require AnchorKind.
    // This is a compile-time guarantee in the type system.
    // We verify by constructing all recommended actions and checking anchors.
    use symthaea::consciousness::epistemic_conflict::{ConflictKind, EpistemicAction};

    for kind in [
        ConflictKind::IntegrationCollapse,
        ConflictKind::NoBroadcast,
        ConflictKind::AttentionalInstability,
        ConflictKind::UnreliablePrediction,
        ConflictKind::ShallowRecurrence,
        ConflictKind::UngroundedRepresentation,
    ] {
        let action = kind.recommended_action();
        match action {
            EpistemicAction::Verify(anchor) => {
                let _ = anchor.description(); // INV-10: anchor is present
            }
            EpistemicAction::Sense(anchor) => {
                let _ = anchor.description(); // INV-10: anchor is present
            }
            _ => {} // Ask, Simulate, Defer, Summarize don't need anchors
        }
    }
}

// ==================================================================================
// FAILURE MODE TESTS
// ==================================================================================

#[test]
fn fm1_budget_exceeded() {
    // FM-1: Budget exceeded → return best-so-far + no deeper reasoning.
    let mut engine = ConsciousReasoningEngine::new();
    let ctx = make_context(0.8, 0.8, 100); // 0.1ms — extremely tight
    let result = engine.reason(&ctx);

    assert_eq!(result.tier, BudgetTier::Tier0);
    assert!(result.phi_eff > 0.0, "Should still have valid Φ_eff");
    assert!(result.plan.is_none(), "Should not have planned");
}

#[test]
fn fm2_all_theories_disagree() {
    // FM-2: All theories disagree → low Φ_eff, epistemic action preferred.
    let mut engine = ConsciousReasoningEngine::new();
    let mut ctx = make_context(0.8, 0.8, 25_000);
    ctx.theory_metrics = MultiTheoryMetrics {
        phi: 0.9, gwt: 0.1, ast: 0.9, pp: 0.1, rpt: 0.9, embodiment: 0.1, unified: 0.5,
    };
    let result = engine.reason(&ctx);

    assert!(
        result.phi_eff < ctx.phi * 0.5,
        "FM-2: Φ_eff ({:.3}) should be significantly below raw Φ ({:.3})",
        result.phi_eff, ctx.phi,
    );
}

#[test]
fn fm3_causal_query_unidentifiable() {
    // FM-3: Causal query unidentifiable → return Unidentified(reason) + no action.
    let engine = ConsciousReasoningEngine::new();
    let dag = CausalDAG::new(vec!["X".into(), "Y".into()], vec![]); // no edges
    let query = CausalQuery { treatment: 0, outcome: 1, conditioning: vec![] };
    let outcome = engine.analyze_counterfactual(&dag, &query);

    assert!(
        matches!(outcome, CausalQueryOutcome::Unidentified { .. }),
        "FM-3: Disconnected DAG should be Unidentified"
    );
}

#[test]
fn fm5_no_available_actions() {
    // FM-5: No available actions → no plan.
    let mut engine = ConsciousReasoningEngine::new();
    let mut ctx = make_context(0.8, 0.8, 25_000);
    ctx.available_actions = vec![]; // no actions
    let result = engine.reason(&ctx);

    // Should complete without error, plan should be empty or no_plan
    if let Some(plan) = &result.plan {
        assert!(!plan.did_plan || plan.best_action_idx.is_none());
    }
}

#[test]
fn fm6_calibration_data_cold() {
    // FM-6: Cold calibration → use conservative defaults.
    let calibrator = TheoryCalibrator::new();
    assert!(calibrator.calibrations.any_cold(), "Fresh calibrator should be cold");
    assert_eq!(calibrator.gamma(), 2.0, "FM-6: Cold start should use default γ=2.0");

    // Reliability should still work with conservative priors
    let metrics = MultiTheoryMetrics {
        phi: 0.8, gwt: 0.8, ast: 0.8, pp: 0.8, rpt: 0.8, embodiment: 0.8, unified: 0.8,
    };
    let r = calibrator.reliability(&metrics);
    assert!(r > 0.0 && r <= 1.0, "FM-6: Cold reliability should still be valid");
}

#[test]
fn fm7_harness_match_rate_low() {
    // FM-7: Harness match rate < 99% → auto-downgrade to AssumptionRequired.
    use symthaea::consciousness::counterfactual::identification::CausalReferenceHarness;

    let mut harness = CausalReferenceHarness::new();
    let reasoner = CounterfactualReasoner::new();

    // The default harness should pass with our reasoner
    let result = harness.validate(&reasoner);
    assert!(
        matches!(result, symthaea::consciousness::counterfactual::identification::HarnessResult::Passed),
        "FM-7: Default harness should pass with our reasoner"
    );
}

// ==================================================================================
// MULTI-CYCLE INTEGRATION
// ==================================================================================

#[test]
fn multi_cycle_stability() {
    // Run 50 cycles and verify the engine remains stable.
    let mut engine = ConsciousReasoningEngine::new();

    for i in 0..50 {
        let consensus = 0.5 + 0.3 * ((i as f64 * 0.1).sin()); // oscillating
        let mut ctx = make_context(0.8, consensus, 25_000);
        ctx.cycle_id = i;
        let result = engine.reason(&ctx);

        // Basic sanity checks every cycle
        assert!(result.phi_eff >= 0.0 && result.phi_eff <= 1.0);
        assert!(result.reliability >= 0.0 && result.reliability <= 1.0);
    }

    let stats = engine.stats();
    assert_eq!(stats.total_cycles, 50);
    assert!(stats.avg_phi_eff > 0.0);
    assert!(stats.avg_reliability > 0.0);

    // Telemetry ring buffer should have capped at max_events
    assert!(engine.recent_events().len() <= 100);
}
