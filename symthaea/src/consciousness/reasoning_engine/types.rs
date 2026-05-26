// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reasoning Engine Types
//!
//! Core types for the ConsciousReasoningEngine: ReasoningContext,
//! ReasoningResult, ReasoningEvent (telemetry), and PosthocOutcome.

use serde::{Deserialize, Serialize};

use crate::consciousness::counterfactual::CausalQueryOutcome;
use crate::consciousness::epistemic_conflict::{
    ConflictKind, ConflictMatrix, MultiTheoryMetrics, TheoryId,
};
use crate::consciousness::temporal_planning::types::{BudgetTier, MctsResult, PlannedAction};
use crate::consciousness::tool_gate::types::{GateResult, RiskLevel, ToolDescriptor};

// ─────────────────────────────────────────────────────────────────────────────
// Reasoning Context (input to reason())
// ─────────────────────────────────────────────────────────────────────────────

/// Input context for a single reasoning cycle.
#[derive(Debug, Clone)]
pub struct ReasoningContext {
    /// Current multi-theory consciousness metrics.
    pub theory_metrics: MultiTheoryMetrics,
    /// Raw Φ (integrated information).
    pub phi: f64,
    /// Available budget in microseconds.
    pub available_budget_us: u64,
    /// Available actions for planning.
    pub available_actions: Vec<PlannedAction>,
    /// The tool being considered for gating (if any).
    pub tool: Option<ToolDescriptor>,
    /// Recent utility of simulation (rolling average).
    pub recent_utility: f64,
    /// Cycle identifier.
    pub cycle_id: u64,
    /// Neuromod modulation of MCTS exploration constant (multiplier, default 1.0).
    /// 5-HT/NE-driven: low 5-HT → explore more, high 5-HT → exploit more.
    pub neuromod_exploration_mod: f64,
    /// Epistemic quality score (0.0-1.0) from E/N/M classification.
    /// Modulates phi_eff: low epistemic quality → conservative reasoning.
    /// Science: epistemic humility — claims with weak evidence get less Phi amplification.
    pub epistemic_quality: f64,
    /// Optional code-specific reasoning context.
    /// When present, enables code-aware conflict detection, type safety gating,
    /// and code-specific MCTS action selection.
    pub code_context: Option<CodeReasoningContext>,
    /// Negative prototypes bank for penalizing disproven approaches in MCTS (INV-12).
    pub negative_prototypes: crate::consciousness::temporal_planning::mcts::NegativePrototypeBank,
}

/// Code-specific reasoning context for the consciousness engine.
///
/// Provides domain-specific signals that modulate reasoning when the
/// engine is processing code generation, modification, or debugging tasks.
#[derive(Debug, Clone, Default)]
pub struct CodeReasoningContext {
    /// Type confidence score (0.0-1.0). Low values indicate unresolved types
    /// or generic type inference, which should increase caution in code generation.
    pub type_confidence: f64,
    /// Whether the code involves unsafe operations (raw pointers, FFI, etc.).
    /// Triggers stricter gating: requires higher Phi_eff for tool authorization.
    pub involves_unsafe: bool,
    /// Compilation success rate from recent attempts (0.0-1.0).
    /// Low values increase exploration (try different approaches).
    pub recent_compile_rate: f64,
    /// Number of auto-fix retries already attempted. Higher values suggest
    /// the current approach is failing and alternatives should be explored.
    pub retry_count: u32,
    /// Whether the code modifies external state (filesystem, network, DB).
    /// Side-effecting code requires stricter consciousness gating.
    pub has_side_effects: bool,
    /// Complexity estimate of the code task (0.0 = trivial, 1.0 = very complex).
    /// Maps to algorithm pattern complexity from the CfC code sequencer.
    pub task_complexity: f64,
}

impl ReasoningContext {
    /// Create a builder for constructing a ReasoningContext.
    pub fn builder() -> ReasoningContextBuilder {
        ReasoningContextBuilder::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReasoningContext Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for ergonomic construction of ReasoningContext.
///
/// Provides sensible defaults:
/// - `phi`: 0.5 (neutral consciousness)
/// - `available_budget_us`: 20_000 (Tier 2 budget)
/// - `available_actions`: empty
/// - `tool`: None
/// - `recent_utility`: 0.5 (neutral prior)
/// - `cycle_id`: 0
///
/// # Example
///
/// ```ignore
/// let ctx = ReasoningContext::builder()
///     .with_phi(0.8)
///     .with_budget_us(8_000)
///     .with_actions(actions)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct ReasoningContextBuilder {
    theory_metrics: Option<MultiTheoryMetrics>,
    phi: f64,
    available_budget_us: u64,
    available_actions: Vec<PlannedAction>,
    tool: Option<ToolDescriptor>,
    recent_utility: f64,
    cycle_id: u64,
    epistemic_quality: f64,
    code_context: Option<CodeReasoningContext>,
    negative_prototypes: crate::consciousness::temporal_planning::mcts::NegativePrototypeBank,
}

impl ReasoningContextBuilder {
    /// Create a new builder with sensible defaults.
    pub fn new() -> Self {
        Self {
            theory_metrics: None,
            phi: 0.5,
            available_budget_us: 20_000, // Tier 2 by default
            available_actions: Vec::new(),
            tool: None,
            recent_utility: 0.5,
            cycle_id: 0,
            epistemic_quality: 0.5,
            code_context: None,
            negative_prototypes: crate::consciousness::temporal_planning::mcts::NegativePrototypeBank::default(),
        }
    }

    /// Set the code-specific reasoning context.
    pub fn with_code_context(mut self, code_ctx: CodeReasoningContext) -> Self {
        self.code_context = Some(code_ctx);
        self
    }

    /// Set the multi-theory consciousness metrics.
    pub fn with_theory_metrics(mut self, metrics: MultiTheoryMetrics) -> Self {
        self.theory_metrics = Some(metrics);
        self
    }

    /// Set the raw Φ value.
    pub fn with_phi(mut self, phi: f64) -> Self {
        self.phi = phi;
        self
    }

    /// Set the available budget in microseconds.
    pub fn with_budget_us(mut self, budget_us: u64) -> Self {
        self.available_budget_us = budget_us;
        self
    }

    /// Set the available actions for planning.
    pub fn with_actions(mut self, actions: Vec<PlannedAction>) -> Self {
        self.available_actions = actions;
        self
    }

    /// Set the tool being considered for gating.
    pub fn with_tool(mut self, tool: ToolDescriptor) -> Self {
        self.tool = Some(tool);
        self
    }

    /// Set the recent utility of simulation.
    pub fn with_recent_utility(mut self, utility: f64) -> Self {
        self.recent_utility = utility;
        self
    }

    /// Set the cycle identifier.
    pub fn with_cycle_id(mut self, cycle_id: u64) -> Self {
        self.cycle_id = cycle_id;
        self
    }

    /// Set the epistemic quality (0.0-1.0).
    pub fn with_epistemic_quality(mut self, quality: f64) -> Self {
        self.epistemic_quality = quality;
        self
    }

    /// Set the negative prototypes bank.
    pub fn with_negative_prototypes(
        mut self,
        negatives: crate::consciousness::temporal_planning::mcts::NegativePrototypeBank,
    ) -> Self {
        self.negative_prototypes = negatives;
        self
    }

    /// Build the ReasoningContext.
    ///
    /// If theory_metrics was not set, creates default metrics based on phi.
    pub fn build(self) -> ReasoningContext {
        let theory_metrics = self.theory_metrics.unwrap_or_else(|| {
            // Create default metrics with phi and consensus at phi level
            MultiTheoryMetrics {
                phi: self.phi,
                gwt: self.phi,
                ast: self.phi,
                pp: self.phi,
                rpt: self.phi,
                embodiment: self.phi,
                unified: self.phi,
            }
        });

        ReasoningContext {
            theory_metrics,
            phi: self.phi,
            available_budget_us: self.available_budget_us,
            available_actions: self.available_actions,
            tool: self.tool,
            recent_utility: self.recent_utility,
            cycle_id: self.cycle_id,
            neuromod_exploration_mod: 1.0,
            epistemic_quality: self.epistemic_quality,
            code_context: self.code_context,
            negative_prototypes: self.negative_prototypes,
        }
    }
}

impl Default for ReasoningContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reasoning Result (output of reason())
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a single reasoning cycle.
///
/// Tiered: Tier 0 always completes; Tier 1/2 add planning, counterfactuals,
/// and narrative as budget allows.
#[derive(Debug, Clone)]
pub struct ReasoningResult {
    /// Budget tier that was used.
    pub tier: BudgetTier,
    /// Effective Φ = Φ × R^γ (after epistemic modulation).
    pub phi_eff: f64,
    /// Raw Φ_eff before epistemic modulation: Φ × R^γ.
    pub phi_eff_raw: f64,
    /// Epistemic quality modulation factor applied (0.5–1.0).
    pub epistemic_mod: f64,
    /// Reliability R.
    pub reliability: f64,
    /// Current γ.
    pub gamma: f64,
    /// Conflict matrix (15 pairwise conflicts).
    pub conflicts: ConflictMatrix,
    /// MCTS planning result (Tier 1+).
    pub plan: Option<MctsResult>,
    /// Tool gate result (if a tool was considered).
    pub gate: Option<GateResult>,
    /// Counterfactual analysis (Tier 2 only).
    pub counterfactual: Option<CausalQueryOutcome>,
    /// Human-readable narrative (Tier 2, best-effort).
    pub narrative: Option<String>,
    /// Wall time in microseconds.
    pub wall_time_us: u64,
    /// Whether budget was exceeded.
    pub budget_exceeded: bool,
    /// Number of tool gate evaluations performed (0, 1, or 2).
    pub gate_checks: u32,
    /// Expected Value of Simulation (0.0 for Tier 0).
    pub evs: f64,
}

impl ReasoningResult {
    /// Create a Tier 0 result (always completes).
    pub fn tier0(
        phi_eff: f64,
        reliability: f64,
        gamma: f64,
        conflicts: ConflictMatrix,
        gate: Option<GateResult>,
        wall_time_us: u64,
    ) -> Self {
        let gate_checks = if gate.is_some() { 1 } else { 0 };
        Self {
            tier: BudgetTier::Tier0,
            phi_eff,
            phi_eff_raw: 0.0, // set by caller via with_internals()
            epistemic_mod: 1.0,
            reliability,
            gamma,
            conflicts,
            plan: None,
            gate,
            counterfactual: None,
            narrative: None,
            wall_time_us,
            budget_exceeded: false,
            gate_checks,
            evs: 0.0,
        }
    }

    /// Create a Tier 1 result (conflict + plan + gate).
    pub fn tier1(
        phi_eff: f64,
        reliability: f64,
        gamma: f64,
        conflicts: ConflictMatrix,
        plan: MctsResult,
        gate: Option<GateResult>,
        wall_time_us: u64,
        budget_exceeded: bool,
    ) -> Self {
        let gate_checks = if gate.is_some() { 2 } else { 0 };
        Self {
            tier: BudgetTier::Tier1,
            phi_eff,
            phi_eff_raw: 0.0,
            epistemic_mod: 1.0,
            reliability,
            gamma,
            conflicts,
            plan: Some(plan),
            gate,
            counterfactual: None,
            narrative: None,
            wall_time_us,
            budget_exceeded,
            gate_checks,
            evs: 0.0,
        }
    }

    /// Create a Tier 2 result (full reasoning).
    pub fn tier2(
        phi_eff: f64,
        reliability: f64,
        gamma: f64,
        conflicts: ConflictMatrix,
        plan: MctsResult,
        gate: Option<GateResult>,
        counterfactual: Option<CausalQueryOutcome>,
        narrative: Option<String>,
        wall_time_us: u64,
        budget_exceeded: bool,
    ) -> Self {
        let gate_checks = if gate.is_some() { 2 } else { 0 };
        Self {
            tier: BudgetTier::Tier2,
            phi_eff,
            phi_eff_raw: 0.0,
            epistemic_mod: 1.0,
            reliability,
            gamma,
            conflicts,
            plan: Some(plan),
            gate,
            counterfactual,
            narrative,
            wall_time_us,
            budget_exceeded,
            gate_checks,
            evs: 0.0,
        }
    }

    /// Set internal diagnostics (phi_eff_raw, epistemic_mod, evs) on a result.
    /// Called by the reasoning engine after construction.
    pub fn with_internals(mut self, phi_eff_raw: f64, epistemic_mod: f64, evs: f64) -> Self {
        self.phi_eff_raw = phi_eff_raw;
        self.epistemic_mod = epistemic_mod;
        self.evs = evs;
        self
    }

    /// Whether the engine had enough budget for full reasoning.
    pub fn is_full(&self) -> bool {
        matches!(self.tier, BudgetTier::Tier2) && !self.budget_exceeded
    }

    /// Whether any action was allowed by the gate.
    pub fn action_allowed(&self) -> bool {
        self.gate.as_ref().map_or(true, |g| g.is_allowed())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reasoning Event (telemetry, emitted every cycle)
// ─────────────────────────────────────────────────────────────────────────────

/// Telemetry event emitted every reasoning cycle, no exceptions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningEvent {
    // Timing
    /// Cycle identifier.
    pub cycle_id: u64,
    /// Wall time in microseconds.
    pub wall_time_us: u64,
    /// Budget tier used.
    pub budget_tier: BudgetTier,
    /// Why this tier was selected.
    pub tier_selected_reason: String,
    /// Whether budget was exceeded.
    pub budget_exceeded: bool,

    // Consciousness
    /// Raw Φ value.
    pub phi_raw: f64,
    /// Reliability R.
    pub reliability: f64,
    /// Effective Φ = Φ × R^γ.
    pub phi_eff: f64,
    /// Current γ.
    pub gamma: f64,
    /// Calibration version (tracks which calibration state was used).
    pub calibration_version: u64,

    // Conflicts
    /// Dominant conflict pair (if any).
    pub dominant_conflict: Option<(TheoryId, TheoryId)>,
    /// Dominant conflict kind.
    pub conflict_kind: Option<ConflictKind>,
    /// Total epistemic entropy.
    pub epistemic_entropy: f64,
    /// Theory metric values [IIT, GWT, AST, PP, RPT, 4E].
    pub theory_metrics: [f64; 6],
    /// Theory reliability weights.
    pub theory_reliabilities: [f64; 6],

    // Planning
    /// Expected value of simulation.
    pub evs: f64,
    /// Whether simulation ran.
    pub did_simulate: bool,
    /// MCTS iterations completed.
    pub mcts_iterations: u32,
    /// MCTS tree size.
    pub mcts_tree_size: u32,
    /// Plan confidence.
    pub plan_confidence: f64,

    // Gating
    /// Selected action description.
    pub selected_action: String,
    /// Risk level classification.
    pub risk_level: Option<RiskLevel>,
    /// Required Φ_eff for this action.
    pub required_phi: f64,
    /// Required confidence for this action.
    pub required_confidence: f64,
    /// Gate decision.
    pub gate_decision: String,
    /// Fallback strategy used (if any).
    pub fallback_used: Option<String>,

    // Causal
    /// Counterfactual outcome description.
    pub causal_outcome: Option<String>,
    /// Harness match rate (if validation ran).
    pub harness_match_rate: Option<f64>,

    // Post-hoc (filled in next cycle)
    /// Post-hoc outcome (from previous cycle).
    pub posthoc_outcome: Option<PosthocOutcome>,
}

/// Post-hoc outcome for calibration feedback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosthocOutcome {
    /// Whether the gate allowed the action.
    pub gate_passed: bool,
    /// Whether the outcome was good.
    pub outcome_good: bool,
    /// Prediction error after action.
    pub prediction_error: f64,
}

impl ReasoningEvent {
    /// Create an empty event template for a given cycle.
    pub fn new(cycle_id: u64) -> Self {
        Self {
            cycle_id,
            wall_time_us: 0,
            budget_tier: BudgetTier::Tier0,
            tier_selected_reason: String::new(),
            budget_exceeded: false,
            phi_raw: 0.0,
            reliability: 0.0,
            phi_eff: 0.0,
            gamma: 2.0,
            calibration_version: 0,
            dominant_conflict: None,
            conflict_kind: None,
            epistemic_entropy: 0.0,
            theory_metrics: [0.0; 6],
            theory_reliabilities: [0.0; 6],
            evs: 0.0,
            did_simulate: false,
            mcts_iterations: 0,
            mcts_tree_size: 0,
            plan_confidence: 0.0,
            selected_action: String::new(),
            risk_level: None,
            required_phi: 0.0,
            required_confidence: 0.0,
            gate_decision: String::new(),
            fallback_used: None,
            causal_outcome: None,
            harness_match_rate: None,
            posthoc_outcome: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::epistemic_conflict::ConflictScore;

    fn make_conflicts() -> ConflictMatrix {
        let conflicts: Vec<ConflictScore> = (0..15)
            .map(|i| {
                let a = TheoryId::ALL[i % 6];
                let b = TheoryId::ALL[(i + 1) % 6];
                ConflictScore::new(
                    a,
                    b,
                    0.3,
                    crate::consciousness::epistemic_conflict::ConflictKind::IntegrationCollapse,
                )
            })
            .collect();
        ConflictMatrix::new(conflicts)
    }

    #[test]
    fn test_tier0_result() {
        let conflicts = make_conflicts();
        let result = ReasoningResult::tier0(0.5, 0.8, 2.0, conflicts, None, 1500);
        assert_eq!(result.tier, BudgetTier::Tier0);
        assert!(result.plan.is_none());
        assert!(result.narrative.is_none());
        assert!(result.action_allowed()); // no gate → allowed
    }

    #[test]
    fn test_reasoning_event_new() {
        let event = ReasoningEvent::new(42);
        assert_eq!(event.cycle_id, 42);
        assert_eq!(event.budget_tier, BudgetTier::Tier0);
        assert!(!event.budget_exceeded);
    }

    #[test]
    fn test_result_is_full() {
        let conflicts = make_conflicts();
        let plan = MctsResult::no_plan();
        let result = ReasoningResult::tier2(
            0.5, 0.8, 2.0, conflicts, plan, None, None, None, 15000, false,
        );
        assert!(result.is_full());
    }

    #[test]
    fn test_reasoning_context_builder_defaults() {
        let ctx = ReasoningContext::builder().build();
        assert_eq!(ctx.phi, 0.5);
        assert_eq!(ctx.available_budget_us, 20_000);
        assert!(ctx.available_actions.is_empty());
        assert!(ctx.tool.is_none());
        assert_eq!(ctx.recent_utility, 0.5);
        assert_eq!(ctx.cycle_id, 0);
        // Default metrics derive from phi
        assert_eq!(ctx.theory_metrics.phi, 0.5);
        assert_eq!(ctx.theory_metrics.gwt, 0.5);
    }

    #[test]
    fn test_reasoning_context_builder_custom() {
        let actions = vec![PlannedAction {
            id: "test".to_string(),
            description: "Test action".to_string(),
            embedding: vec![0.0; 4],
            prior: 1.0,
            is_epistemic: false,
        }];

        let ctx = ReasoningContext::builder()
            .with_phi(0.8)
            .with_budget_us(8_000)
            .with_actions(actions)
            .with_recent_utility(0.7)
            .with_cycle_id(42)
            .build();

        assert_eq!(ctx.phi, 0.8);
        assert_eq!(ctx.available_budget_us, 8_000);
        assert_eq!(ctx.available_actions.len(), 1);
        assert_eq!(ctx.recent_utility, 0.7);
        assert_eq!(ctx.cycle_id, 42);
    }

    #[test]
    fn test_reasoning_context_builder_with_explicit_metrics() {
        let metrics = MultiTheoryMetrics {
            phi: 0.9,
            gwt: 0.85,
            ast: 0.8,
            pp: 0.75,
            rpt: 0.7,
            embodiment: 0.65,
            unified: 0.8,
        };

        let ctx = ReasoningContext::builder()
            .with_theory_metrics(metrics)
            .with_phi(0.9)
            .build();

        assert_eq!(ctx.theory_metrics.gwt, 0.85);
        assert_eq!(ctx.theory_metrics.ast, 0.8);
    }
}
