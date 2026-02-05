//! Reasoning Engine Types
//!
//! Core types for the ConsciousReasoningEngine: ReasoningContext,
//! ReasoningResult, ReasoningEvent (telemetry), and PosthocOutcome.

use serde::{Deserialize, Serialize};

use crate::consciousness::epistemic_conflict::{
    ConflictKind, ConflictMatrix, MultiTheoryMetrics, TheoryId,
};
use crate::consciousness::tool_gate::types::{
    FallbackStrategy, GateDecision, GateResult, RiskLevel, ToolDescriptor,
};
use crate::consciousness::temporal_planning::types::{
    BudgetTier, MctsResult, PlannedAction,
};
use crate::consciousness::counterfactual::CausalQueryOutcome;

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
    /// Effective Φ = Φ × R^γ.
    pub phi_eff: f64,
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
        Self {
            tier: BudgetTier::Tier0,
            phi_eff,
            reliability,
            gamma,
            conflicts,
            plan: None,
            gate,
            counterfactual: None,
            narrative: None,
            wall_time_us,
            budget_exceeded: false,
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
        Self {
            tier: BudgetTier::Tier1,
            phi_eff,
            reliability,
            gamma,
            conflicts,
            plan: Some(plan),
            gate,
            counterfactual: None,
            narrative: None,
            wall_time_us,
            budget_exceeded,
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
        Self {
            tier: BudgetTier::Tier2,
            phi_eff,
            reliability,
            gamma,
            conflicts,
            plan: Some(plan),
            gate,
            counterfactual,
            narrative,
            wall_time_us,
            budget_exceeded,
        }
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
}
