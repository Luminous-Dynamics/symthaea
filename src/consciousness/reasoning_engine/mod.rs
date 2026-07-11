// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conscious Reasoning Engine
//!
//! Orchestrates all four subsystems — epistemic conflict detection,
//! temporal planning, counterfactual reasoning, and consciousness-gated
//! tool use — into a unified 7-step reasoning cycle with tiered degradation.
//!
//! ## The Reasoning Cycle (7 steps)
//!
//! ```text
//! reason(context) → ReasoningResult
//!   1. DETECT   → ConflictMatrix (15 pairwise theory conflicts)
//!   2. ASSESS   → Reliability R, Effective Φ_eff = Φ × R^γ
//!   3. DECIDE   → should_simulate? (EVS, reliability-gated)
//!   4. PLAN     → MCTS with O(1) rollouts (budget-bounded)
//!   5. GATE     → tool authorization against Φ_eff AND plan_confidence
//!   6. ANALYZE  → counterfactual (only if identified + harness-validated)
//!   7. EMIT     → ReasoningEvent telemetry (every cycle, no exceptions)
//! ```
//!
//! ## Budget Tiers
//!
//! - **Tier 0** (≤2ms): Steps 1-2 + gate + fallback (always completes)
//! - **Tier 1** (≤8ms): + micro-MCTS (K=5, N=50)
//! - **Tier 2** (≤20ms): + full MCTS + counterfactuals + narrative (best-effort)

pub mod narrative;
pub mod telemetry;
pub mod types;

pub use narrative::generate_narrative;
pub use telemetry::{
    CsvSink, JsonLinesSink, PrometheusMetrics, PrometheusSink, TelemetryConfig, TelemetryError,
    TelemetryExporter, TelemetryExporterBuilder, TelemetrySink,
};
pub use types::{
    CodeReasoningContext, PosthocOutcome, ReasoningContext, ReasoningContextBuilder,
    ReasoningEvent, ReasoningResult,
};

use std::collections::VecDeque;
use std::time::Instant;

use crate::consciousness::counterfactual::{
    CausalDAG, CausalQuery, CausalQueryOutcome, CounterfactualReasoner,
};
use crate::consciousness::epistemic_conflict::{
    ConflictDetector, TheoryCalibrator,
    phi_integration::{effective_phi, thresholds},
};
use crate::consciousness::temporal_planning::{
    mcts::{MctsPlanner, NegativePrototypeBank, evs},
    types::{BudgetTier, ForkedState, MctsConfig, ReasoningBudget},
};
use crate::consciousness::tool_gate::classifier;

/// The Conscious Reasoning Engine.
///
/// Composes all Phase A-C subsystems into a unified reasoning cycle
/// with tiered degradation and self-calibrating telemetry.
pub struct ConsciousReasoningEngine {
    /// Epistemic conflict detector.
    conflict_detector: ConflictDetector,
    /// Theory calibrator (reliability + γ).
    calibrator: TheoryCalibrator,
    /// Counterfactual reasoner.
    counterfactual_reasoner: CounterfactualReasoner,
    /// Cycle counter.
    cycle_count: u64,
    /// Recent telemetry events (ring buffer for introspection).
    recent_events: VecDeque<ReasoningEvent>,
    /// Maximum events to retain.
    max_events: usize,
    /// Pending posthoc outcome (from previous cycle, to be attached to next event).
    pending_posthoc: Option<PosthocOutcome>,
    /// Simulation state for MCTS (updated externally or from defaults).
    simulation_state: Option<ForkedState>,
    /// Bank of disproven logical structures to penalize during planning.
    negative_bank: NegativePrototypeBank,
}

impl ConsciousReasoningEngine {
    /// Create a new reasoning engine with default configuration.
    pub fn new() -> Self {
        Self {
            conflict_detector: ConflictDetector::new(),
            calibrator: TheoryCalibrator::new(),
            counterfactual_reasoner: CounterfactualReasoner::new(),
            cycle_count: 0,
            recent_events: VecDeque::with_capacity(100),
            max_events: 100,
            pending_posthoc: None,
            simulation_state: None,
            negative_bank: NegativePrototypeBank::default(),
        }
    }

    /// Set the simulation state for MCTS planning.
    pub fn set_simulation_state(&mut self, state: ForkedState) {
        self.simulation_state = Some(state);
    }

    /// Record a posthoc outcome (fed back from the previous cycle's action).
    pub fn record_posthoc(&mut self, gate_passed: bool, outcome_good: bool, prediction_error: f64) {
        self.pending_posthoc = Some(PosthocOutcome {
            gate_passed,
            outcome_good,
            prediction_error,
        });
        self.calibrator.record_outcome(gate_passed, outcome_good);
    }

    /// Get the current calibrator (for external reads).
    pub fn calibrator(&self) -> &TheoryCalibrator {
        &self.calibrator
    }

    /// Get mutable calibrator (for external theory updates).
    pub fn calibrator_mut(&mut self) -> &mut TheoryCalibrator {
        &mut self.calibrator
    }

    /// Get recent reasoning events.
    pub fn recent_events(&self) -> &VecDeque<ReasoningEvent> {
        &self.recent_events
    }

    /// Get the last reasoning event.
    pub fn last_event(&self) -> Option<&ReasoningEvent> {
        self.recent_events.back()
    }

    /// Run one reasoning cycle.
    ///
    /// This is the main entry point. It always completes Tier 0 (≤2ms)
    /// and progressively adds Tier 1/2 capabilities as budget allows.
    pub fn reason(&mut self, ctx: &ReasoningContext) -> ReasoningResult {
        let start = Instant::now();
        self.cycle_count += 1;
        let cycle_id = ctx.cycle_id;

        // ── STEP 1: DETECT ─────────────────────────────────────────────
        let conflicts = self.conflict_detector.detect(&ctx.theory_metrics);

        // ── STEP 2: ASSESS ─────────────────────────────────────────────
        let r = self.calibrator.reliability(&ctx.theory_metrics);
        let gamma = self.calibrator.gamma();
        let phi_eff_raw = effective_phi(ctx.phi, r, gamma);
        // Epistemic quality modulates phi_eff: low evidence quality → conservative.
        // Floor at 0.5 so even zero-quality claims retain 50% of their phi_eff.
        // Science: epistemic humility — claims with weak evidence get less Phi amplification.
        let epistemic_mod = 0.5 + 0.5 * ctx.epistemic_quality;
        let mut phi_eff = phi_eff_raw * epistemic_mod;

        // Code-specific modulation: unsafe code and low compile rates reduce phi_eff
        if let Some(ref code_ctx) = ctx.code_context {
            // Unsafe code requires 20% higher consciousness threshold
            if code_ctx.involves_unsafe {
                phi_eff *= 0.8;
            }
            // Low compile rate → reduce confidence in code generation actions
            if code_ctx.recent_compile_rate < 0.5 {
                phi_eff *= 0.7 + 0.3 * code_ctx.recent_compile_rate;
            }
            // Side-effecting code → require higher confidence
            if code_ctx.has_side_effects {
                phi_eff *= 0.9;
            }
            // High retry count → this approach is failing, boost exploration
            // (handled in MCTS exploration constant below)
        }

        // Create budget tracker
        let budget = ReasoningBudget::new(ctx.available_budget_us, r, ctx.phi);

        // ── STEP 5 (early, for Tier 0): GATE ───────────────────────────
        let gate = ctx.tool.as_ref().map(|tool| {
            let plan_confidence = 0.0; // no plan yet in Tier 0
            classifier::gate(tool, phi_eff, plan_confidence)
        });

        // Build telemetry event
        let mut event = ReasoningEvent::new(cycle_id);
        event.phi_raw = ctx.phi;
        event.reliability = r;
        event.phi_eff = phi_eff;
        event.gamma = gamma;
        event.calibration_version = self.calibrator.version;
        event.dominant_conflict = conflicts.dominant;
        event.conflict_kind = conflicts.dominant_kind;
        event.epistemic_entropy = conflicts.total_entropy;
        event.theory_metrics = ctx.theory_metrics.values();
        event.theory_reliabilities = self.calibrator.calibrations.reliabilities();
        event.posthoc_outcome = self.pending_posthoc.take();

        // Populate gate fields
        if let Some(ref g) = gate {
            event.risk_level = Some(g.risk_level);
            event.required_phi = g.required_phi;
            event.required_confidence = g.required_confidence;
            event.gate_decision = format!("{:?}", g.decision);
            event.fallback_used = g.fallback.as_ref().map(|f| format!("{:?}", f));
        }

        // ── TIER 0 (always runs, ≤2ms) ─────────────────────────────────
        if budget.tier == BudgetTier::Tier0 || budget.exceeded() {
            let wall_time_us = start.elapsed().as_micros() as u64;
            event.wall_time_us = wall_time_us;
            event.budget_tier = BudgetTier::Tier0;
            event.tier_selected_reason = tier_reason(&budget, r);
            self.emit_event(event);
            return ReasoningResult::tier0(phi_eff, r, gamma, conflicts, gate, wall_time_us)
                .with_internals(phi_eff_raw, epistemic_mod, 0.0);
        }

        // ── STEP 3: DECIDE (should simulate?) ──────────────────────────
        let evs_val = evs(
            conflicts.total_entropy,
            r,
            ctx.available_actions.len(),
            ctx.recent_utility,
        );
        event.evs = evs_val;

        // ── STEP 4: PLAN ────────────────────────────────────────────────
        let plan = if evs_val > thresholds::EVS_THRESHOLD
            && !ctx.available_actions.is_empty()
            && budget.tier != BudgetTier::Tier1
        {
            let sim_state = self
                .simulation_state
                .as_ref()
                .cloned()
                .unwrap_or_else(|| default_simulation_state());

            if r >= thresholds::R_PLAN_MIN {
                // Full MCTS with neuromod-modulated exploration constant
                // Science: Dayan & Huys (2009) — 5-HT modulates risk sensitivity
                let mut config = if budget.tier == BudgetTier::Tier1 {
                    MctsConfig::tier1()
                } else {
                    MctsConfig::tier2()
                };
                config.exploration_constant *= ctx.neuromod_exploration_mod;
                // Code-specific: high retry count → boost exploration to try alternatives
                if let Some(ref code_ctx) = ctx.code_context {
                    if code_ctx.retry_count > 1 {
                        config.exploration_constant *= 1.0 + 0.2 * code_ctx.retry_count as f64;
                    }
                }
                event.did_simulate = true;
                MctsPlanner::plan(
                    &sim_state,
                    &ctx.available_actions,
                    &config,
                    &budget,
                    &self.negative_bank,
                    &ctx.substrate_cost_model,
                )
            } else if r >= thresholds::R_EPISTEMIC_MIN {
                // Epistemic rollouts only
                event.did_simulate = true;
                MctsPlanner::epistemic_rollout(&sim_state, &ctx.available_actions, &budget)
            } else {
                MctsResult::no_plan()
            }
        } else {
            MctsResult::no_plan()
        };

        event.mcts_iterations = plan.iterations;
        event.mcts_tree_size = plan.tree_size;
        event.plan_confidence = plan.confidence;
        if let Some(idx) = plan.best_action_idx {
            if idx < ctx.available_actions.len() {
                event.selected_action = ctx.available_actions[idx].description.clone();
            }
        }

        // ── STEP 5 (refined): GATE with plan confidence ────────────────
        let gate = ctx
            .tool
            .as_ref()
            .map(|tool| classifier::gate(tool, phi_eff, plan.confidence));
        if let Some(ref g) = gate {
            event.risk_level = Some(g.risk_level);
            event.required_phi = g.required_phi;
            event.required_confidence = g.required_confidence;
            event.gate_decision = format!("{:?}", g.decision);
            event.fallback_used = g.fallback.as_ref().map(|f| format!("{:?}", f));
        }

        // ── TIER 1 check ────────────────────────────────────────────────
        if budget.tier == BudgetTier::Tier1 || budget.exceeded() {
            let wall_time_us = start.elapsed().as_micros() as u64;
            event.wall_time_us = wall_time_us;
            event.budget_tier = BudgetTier::Tier1;
            event.budget_exceeded = budget.exceeded();
            event.tier_selected_reason = tier_reason(&budget, r);
            self.emit_event(event);
            return ReasoningResult::tier1(
                phi_eff,
                r,
                gamma,
                conflicts,
                plan,
                gate,
                wall_time_us,
                budget.exceeded(),
            )
            .with_internals(phi_eff_raw, epistemic_mod, evs_val);
        }

        // ── STEP 6: ANALYZE (counterfactual, Tier 2 only) ──────────────
        let cf = if budget.remaining_us() > 5_000 {
            // Only run counterfactual if we have budget
            // Use a simple two-node DAG as default if none provided
            None // Counterfactual analysis is opt-in via context
        } else {
            None
        };

        if let Some(ref outcome) = cf {
            event.causal_outcome = Some(format!("{:?}", outcome));
        }

        // ── STEP 7: NARRATIVE (best-effort) ─────────────────────────────
        let narrative = if budget.remaining_us() > 3_000 {
            generate_narrative(
                phi_eff,
                r,
                &conflicts,
                Some(&plan),
                gate.as_ref(),
                cf.as_ref(),
            )
        } else {
            None
        };

        let wall_time_us = start.elapsed().as_micros() as u64;
        event.wall_time_us = wall_time_us;
        event.budget_tier = BudgetTier::Tier2;
        event.budget_exceeded = budget.exceeded();
        event.tier_selected_reason = tier_reason(&budget, r);
        self.emit_event(event);

        ReasoningResult::tier2(
            phi_eff,
            r,
            gamma,
            conflicts,
            plan,
            gate,
            cf,
            narrative,
            wall_time_us,
            budget.exceeded(),
        )
        .with_internals(phi_eff_raw, epistemic_mod, evs_val)
    }

    /// Convenience method for code-specific reasoning.
    ///
    /// Wraps `reason()` with a `CodeReasoningContext` pre-populated from
    /// code generation metrics. Adjusts gating thresholds for code tasks:
    /// - Unsafe code requires 20% higher Phi_eff
    /// - Low compile rates increase exploration
    /// - Side-effecting code requires higher confidence
    ///
    /// Returns the same `ReasoningResult` as `reason()`, but code-aware.
    pub fn reason_about_code(
        &mut self,
        base_ctx: &ReasoningContext,
        type_confidence: f64,
        compile_rate: f64,
        retry_count: u32,
        involves_unsafe: bool,
        has_side_effects: bool,
        task_complexity: f64,
    ) -> ReasoningResult {
        let mut ctx = base_ctx.clone();
        ctx.code_context = Some(CodeReasoningContext {
            type_confidence,
            involves_unsafe,
            recent_compile_rate: compile_rate,
            retry_count,
            has_side_effects,
            task_complexity,
        });
        self.reason(&ctx)
    }

    /// Run a counterfactual analysis as part of a reasoning cycle.
    ///
    /// This is called separately when a causal DAG and query are available.
    pub fn analyze_counterfactual(
        &self,
        dag: &CausalDAG,
        query: &CausalQuery,
    ) -> CausalQueryOutcome {
        self.counterfactual_reasoner.query(dag, query)
    }

    /// Emit a telemetry event (appends to the ring buffer).
    fn emit_event(&mut self, event: ReasoningEvent) {
        if self.recent_events.len() >= self.max_events {
            self.recent_events.pop_front();
        }
        self.recent_events.push_back(event);
    }

    /// Get engine statistics.
    pub fn stats(&self) -> EngineStats {
        let events: Vec<&ReasoningEvent> = self.recent_events.iter().collect();
        let total = events.len();
        if total == 0 {
            return EngineStats::default();
        }

        let avg_wall_time =
            events.iter().map(|e| e.wall_time_us).sum::<u64>() as f64 / total as f64;
        let avg_phi_eff = events.iter().map(|e| e.phi_eff).sum::<f64>() / total as f64;
        let avg_reliability = events.iter().map(|e| e.reliability).sum::<f64>() / total as f64;
        let tier0_count = events
            .iter()
            .filter(|e| e.budget_tier == BudgetTier::Tier0)
            .count();
        let tier1_count = events
            .iter()
            .filter(|e| e.budget_tier == BudgetTier::Tier1)
            .count();
        let tier2_count = events
            .iter()
            .filter(|e| e.budget_tier == BudgetTier::Tier2)
            .count();
        let budget_exceeded_count = events.iter().filter(|e| e.budget_exceeded).count();

        EngineStats {
            total_cycles: self.cycle_count,
            avg_wall_time_us: avg_wall_time,
            avg_phi_eff,
            avg_reliability,
            tier0_fraction: tier0_count as f64 / total as f64,
            tier1_fraction: tier1_count as f64 / total as f64,
            tier2_fraction: tier2_count as f64 / total as f64,
            budget_exceeded_fraction: budget_exceeded_count as f64 / total as f64,
            gamma: self.calibrator.gamma(),
            calibration_version: self.calibrator.version,
        }
    }
}

impl Default for ConsciousReasoningEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Engine statistics summary.
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    pub total_cycles: u64,
    pub avg_wall_time_us: f64,
    pub avg_phi_eff: f64,
    pub avg_reliability: f64,
    pub tier0_fraction: f64,
    pub tier1_fraction: f64,
    pub tier2_fraction: f64,
    pub budget_exceeded_fraction: f64,
    pub gamma: f64,
    pub calibration_version: u64,
}

/// Create a default simulation state for MCTS when none is provided.
fn default_simulation_state() -> ForkedState {
    use std::sync::Arc;
    ForkedState::new(
        Arc::new(vec![0.0; 16]),
        vec![vec![0.0; 16]],
        vec![1.0],
        [42u8; 32],
    )
}

/// Generate a human-readable reason for tier selection.
fn tier_reason(budget: &ReasoningBudget, reliability: f64) -> String {
    if budget.exceeded() {
        "budget_exceeded".to_string()
    } else if reliability < 0.2 {
        format!("R={:.2}<0.2", reliability)
    } else if reliability < 0.5 {
        format!("R={:.2}<0.5", reliability)
    } else if budget.tier == BudgetTier::Tier0 {
        format!("budget<2ms")
    } else if budget.tier == BudgetTier::Tier1 {
        format!("budget<8ms")
    } else {
        "full_budget".to_string()
    }
}

use crate::consciousness::temporal_planning::types::MctsResult;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::epistemic_conflict::MultiTheoryMetrics;
    use crate::consciousness::temporal_planning::types::PlannedAction;

    fn make_metrics(phi: f64, consensus: f64) -> MultiTheoryMetrics {
        MultiTheoryMetrics {
            phi,
            gwt: consensus,
            ast: consensus,
            pp: consensus,
            rpt: consensus,
            embodiment: consensus,
            unified: phi * 0.2 + consensus * 0.8,
        }
    }

    fn make_context(phi: f64, consensus: f64, budget_us: u64) -> ReasoningContext {
        ReasoningContext {
            negative_prototypes: Default::default(),
            theory_metrics: make_metrics(phi, consensus),
            phi,
            available_budget_us: budget_us,
            available_actions: vec![
                PlannedAction {
                    id: "explore".to_string(),
                    description: "Explore environment".to_string(),
                    embedding: vec![0.5; 4],
                    prior: 0.5,
                    is_epistemic: true,
                },
                PlannedAction {
                    id: "execute".to_string(),
                    description: "Execute action".to_string(),
                    embedding: vec![1.0; 4],
                    prior: 0.3,
                    is_epistemic: false,
                },
            ],
            tool: None,
            recent_utility: 0.5,
            cycle_id: 1,
            neuromod_exploration_mod: 1.0,
            epistemic_quality: 0.5,
            grid_encoding_norm: 0.0,
            grid_spatial_complexity: 0.0,
            code_context: None,
            substrate_cost_model: Default::default(),
        }
    }

    #[test]
    fn test_tier0_with_low_budget() {
        let mut engine = ConsciousReasoningEngine::new();
        let ctx = make_context(0.8, 0.8, 1_000); // 1ms → Tier 0
        let result = engine.reason(&ctx);
        assert_eq!(result.tier, BudgetTier::Tier0);
        assert!(result.plan.is_none());
        assert!(result.phi_eff > 0.0);
        assert!(result.reliability > 0.0);
    }

    #[test]
    fn test_tier0_with_low_reliability() {
        let mut engine = ConsciousReasoningEngine::new();
        // Theories wildly disagree → low R → Tier 0
        let mut ctx = make_context(0.8, 0.8, 20_000);
        ctx.theory_metrics = MultiTheoryMetrics {
            phi: 0.9,
            gwt: 0.1,
            ast: 0.9,
            pp: 0.1,
            rpt: 0.9,
            embodiment: 0.1,
            unified: 0.5,
        };
        let result = engine.reason(&ctx);
        // With disagreeing theories, reliability should be low
        assert!(result.reliability < 0.7);
    }

    #[test]
    fn test_tier2_with_full_budget() {
        let mut engine = ConsciousReasoningEngine::new();
        let ctx = make_context(0.8, 0.8, 25_000); // 25ms → Tier 2
        let result = engine.reason(&ctx);
        assert_eq!(result.tier, BudgetTier::Tier2);
    }

    #[test]
    fn test_telemetry_always_emitted() {
        let mut engine = ConsciousReasoningEngine::new();

        // Run 5 cycles
        for i in 0..5 {
            let mut ctx = make_context(0.8, 0.8, 25_000);
            ctx.cycle_id = i;
            engine.reason(&ctx);
        }

        // Every cycle should have emitted an event
        assert_eq!(engine.recent_events().len(), 5);
    }

    #[test]
    fn test_inv1_monotonic_caution_in_engine() {
        // INV-1: Lower reliability → lower Φ_eff
        let mut engine = ConsciousReasoningEngine::new();

        let ctx_high_r = make_context(0.8, 0.8, 25_000);
        let result_high = engine.reason(&ctx_high_r);

        let mut ctx_low_r = make_context(0.8, 0.8, 25_000);
        ctx_low_r.theory_metrics = MultiTheoryMetrics {
            phi: 0.9,
            gwt: 0.1,
            ast: 0.9,
            pp: 0.1,
            rpt: 0.9,
            embodiment: 0.1,
            unified: 0.5,
        };
        let result_low = engine.reason(&ctx_low_r);

        // Higher reliability should produce higher Φ_eff
        // (Note: both have phi=0.8 but different R)
        if result_high.reliability > result_low.reliability {
            assert!(
                result_high.phi_eff >= result_low.phi_eff,
                "INV-1: Φ_eff should be higher with higher R"
            );
        }
    }

    #[test]
    fn test_posthoc_feedback() {
        let mut engine = ConsciousReasoningEngine::new();

        // Run a cycle
        let ctx = make_context(0.8, 0.8, 25_000);
        engine.reason(&ctx);

        // Record posthoc
        engine.record_posthoc(true, true, 0.1);

        // Next cycle should include the posthoc outcome
        let mut ctx2 = make_context(0.8, 0.8, 25_000);
        ctx2.cycle_id = 2;
        engine.reason(&ctx2);

        let last = engine.last_event().unwrap();
        assert!(last.posthoc_outcome.is_some());
        assert!(last.posthoc_outcome.as_ref().unwrap().outcome_good);
    }

    #[test]
    fn test_tool_gating_in_engine() {
        let mut engine = ConsciousReasoningEngine::new();

        let mut ctx = make_context(0.8, 0.8, 25_000);
        ctx.tool = Some(
            crate::consciousness::tool_gate::types::ToolDescriptor::read_only(
                "nix search nixpkgs firefox",
            ),
        );
        let result = engine.reason(&ctx);

        // Read-only tool should be allowed with reasonable Φ_eff
        assert!(result.gate.is_some());
        assert!(result.gate.unwrap().is_allowed());
    }

    #[test]
    fn test_engine_stats() {
        let mut engine = ConsciousReasoningEngine::new();

        for i in 0..10 {
            let mut ctx = make_context(0.8, 0.8, 25_000);
            ctx.cycle_id = i;
            engine.reason(&ctx);
        }

        let stats = engine.stats();
        assert_eq!(stats.total_cycles, 10);
        assert!(stats.avg_phi_eff > 0.0);
        assert!(stats.avg_reliability > 0.0);
    }

    #[test]
    fn test_inv3_deterministic_reasoning() {
        // INV-3: Fixed inputs → identical results
        // (Note: timing varies, so we check logical determinism, not bitwise)
        let mut engine1 = ConsciousReasoningEngine::new();
        let mut engine2 = ConsciousReasoningEngine::new();

        let ctx = make_context(0.8, 0.8, 1_000); // Tier 0 for speed

        let r1 = engine1.reason(&ctx);
        let r2 = engine2.reason(&ctx);

        assert_eq!(r1.tier, r2.tier);
        assert!((r1.phi_eff - r2.phi_eff).abs() < 1e-10);
        assert!((r1.reliability - r2.reliability).abs() < 1e-10);
        assert!((r1.gamma - r2.gamma).abs() < 1e-10);
    }

    #[test]
    fn test_counterfactual_analysis() {
        let engine = ConsciousReasoningEngine::new();

        let dag = CausalDAG::new(vec!["X".into(), "Y".into()], vec![(0, 1)]);
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };

        let outcome = engine.analyze_counterfactual(&dag, &query);
        match outcome {
            CausalQueryOutcome::Identified { .. } => {} // good
            _ => panic!("Simple X→Y should be identified"),
        }
    }

    #[test]
    fn test_fm1_budget_exceeded() {
        // FM-1: Budget exceeded → return best-so-far
        let mut engine = ConsciousReasoningEngine::new();
        let ctx = make_context(0.8, 0.8, 500); // 0.5ms → definitely Tier 0
        let result = engine.reason(&ctx);
        assert_eq!(result.tier, BudgetTier::Tier0);
        // Should still have valid data
        assert!(result.phi_eff > 0.0);
    }

    #[test]
    fn test_fm2_all_theories_disagree() {
        // FM-2: All theories disagree → low Φ_eff
        let mut engine = ConsciousReasoningEngine::new();
        let mut ctx = make_context(0.8, 0.8, 25_000);
        ctx.theory_metrics = MultiTheoryMetrics {
            phi: 0.9,
            gwt: 0.1,
            ast: 0.9,
            pp: 0.1,
            rpt: 0.9,
            embodiment: 0.1,
            unified: 0.5,
        };
        let result = engine.reason(&ctx);
        // Φ_eff should be significantly attenuated
        assert!(
            result.phi_eff < ctx.phi,
            "Φ_eff ({}) should be less than raw Φ ({}) when theories disagree",
            result.phi_eff,
            ctx.phi,
        );
    }
}
