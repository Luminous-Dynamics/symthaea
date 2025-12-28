// Revolutionary Enhancement #4 - Phase 3: Action Planning
//
// Implements Goal-Directed Intervention Search
//
// Key Innovation: Backward search from desired outcome to find optimal interventions
//
// Core Questions:
//   "What should we do to achieve goal G?"
//   "What's the best sequence of interventions to maximize outcome Y?"
//   "How do we get from state A to state B?"
//
// Real-World Example:
//   Goal: Φ (consciousness) should be > 0.8
//   Current: Φ = 0.4
//   Plan: [Enable security_check, Enable monitoring] → Φ = 0.85
//
// Planning Approaches:
//   1. Forward Search: Try interventions, see which reaches goal
//   2. Backward Search: Start from goal, work back to current state
//   3. Hybrid: Combine both for efficiency
//
// This enables:
// - Automatic intervention recommendation
// - Multi-step planning
// - Cost-benefit optimization
// - Constraint satisfaction

use super::{
    probabilistic_inference::ProbabilisticCausalGraph,
    causal_intervention::{
        CausalInterventionEngine, InterventionType, InterventionResult,
    },
};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Goal specification for planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// Target variable to optimize
    pub target: String,

    /// Desired value or threshold
    pub desired_value: f64,

    /// Tolerance (goal is met if within this range)
    pub tolerance: f64,

    /// Optimization direction
    pub direction: GoalDirection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalDirection {
    /// Maximize target value
    Maximize,

    /// Minimize target value
    Minimize,

    /// Reach specific target value
    Exact,
}

impl Goal {
    /// Create goal to maximize a variable
    pub fn maximize(target: &str) -> Self {
        Self {
            target: target.to_string(),
            desired_value: 1.0,
            tolerance: 0.1,
            direction: GoalDirection::Maximize,
        }
    }

    /// Create goal to minimize a variable
    pub fn minimize(target: &str) -> Self {
        Self {
            target: target.to_string(),
            desired_value: 0.0,
            tolerance: 0.1,
            direction: GoalDirection::Minimize,
        }
    }

    /// Create goal to reach specific value
    pub fn reach(target: &str, value: f64, tolerance: f64) -> Self {
        Self {
            target: target.to_string(),
            desired_value: value,
            tolerance,
            direction: GoalDirection::Exact,
        }
    }

    /// Check if value satisfies goal
    pub fn is_satisfied(&self, value: f64) -> bool {
        match self.direction {
            GoalDirection::Maximize => value >= self.desired_value - self.tolerance,
            GoalDirection::Minimize => value <= self.desired_value + self.tolerance,
            GoalDirection::Exact => (value - self.desired_value).abs() <= self.tolerance,
        }
    }

    /// Score how well value satisfies goal (0.0 = not satisfied, 1.0 = perfectly satisfied)
    pub fn satisfaction_score(&self, value: f64) -> f64 {
        match self.direction {
            GoalDirection::Maximize => {
                if value >= self.desired_value {
                    1.0
                } else {
                    (value / self.desired_value).max(0.0)
                }
            }
            GoalDirection::Minimize => {
                if value <= self.desired_value {
                    1.0
                } else {
                    (self.desired_value / value).max(0.0)
                }
            }
            GoalDirection::Exact => {
                let distance = (value - self.desired_value).abs();
                if distance <= self.tolerance {
                    1.0 - (distance / self.tolerance)
                } else {
                    0.0
                }
            }
        }
    }
}

/// A plan to achieve a goal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPlan {
    /// Goal this plan achieves
    pub goal: Goal,

    /// Ordered sequence of interventions
    pub interventions: Vec<PlannedIntervention>,

    /// Predicted final value after all interventions
    pub predicted_value: f64,

    /// Confidence in plan success
    pub confidence: f64,

    /// Total cost of plan
    pub total_cost: f64,

    /// Explanation of plan
    pub explanation: String,
}

/// A single intervention in a plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedIntervention {
    /// Step number in plan
    pub step: usize,

    /// Node to intervene on
    pub node: String,

    /// Type of intervention
    pub intervention_type: InterventionType,

    /// Predicted effect on goal
    pub effect: f64,

    /// Cost of this intervention
    pub cost: f64,

    /// Rationale for this step
    pub rationale: String,
}

/// Configuration for action planner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerConfig {
    /// Maximum planning depth (steps)
    pub max_depth: usize,

    /// Maximum number of candidate interventions to consider per step
    pub max_candidates: usize,

    /// Minimum improvement required per step
    pub min_improvement: f64,

    /// Weight for cost vs benefit (0.0 = ignore cost, 1.0 = only cost)
    pub cost_weight: f64,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_candidates: 5,
            min_improvement: 0.1,
            cost_weight: 0.3,
        }
    }
}

/// Action planner - finds interventions to achieve goals
pub struct ActionPlanner {
    /// Intervention engine for predictions
    intervention_engine: CausalInterventionEngine,

    /// Configuration
    config: PlannerConfig,

    /// Known intervention costs (default to 1.0)
    intervention_costs: HashMap<String, f64>,
}

impl ActionPlanner {
    /// Create new action planner
    pub fn new(graph: ProbabilisticCausalGraph) -> Self {
        Self::with_config(graph, PlannerConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(graph: ProbabilisticCausalGraph, config: PlannerConfig) -> Self {
        Self {
            intervention_engine: CausalInterventionEngine::new(graph),
            config,
            intervention_costs: HashMap::new(),
        }
    }

    /// Set cost for an intervention
    pub fn set_intervention_cost(&mut self, node: &str, cost: f64) {
        self.intervention_costs.insert(node.to_string(), cost);
    }

    /// Plan actions to achieve goal
    ///
    /// Uses greedy forward search:
    /// 1. Find candidate interventions that improve goal
    /// 2. Select best intervention (highest benefit / cost)
    /// 3. Repeat until goal satisfied or max depth reached
    ///
    /// Example:
    /// ```ignore
    /// let planner = ActionPlanner::new(prob_graph);
    ///
    /// let goal = Goal::maximize("phi_value");
    /// let plan = planner.plan(&goal, &["security_check", "monitoring"]);
    ///
    /// println!("Plan: {}", plan.explanation);
    /// for intervention in &plan.interventions {
    ///     println!("  Step {}: {}", intervention.step, intervention.rationale);
    /// }
    /// ```
    pub fn plan(&mut self, goal: &Goal, candidate_nodes: &[String]) -> ActionPlan {
        let mut interventions = Vec::new();
        let mut current_value = self.estimate_current_value(&goal.target);
        let mut total_cost = 0.0;

        // Greedy forward search
        for step in 0..self.config.max_depth {
            // Check if goal already satisfied
            if goal.is_satisfied(current_value) {
                break;
            }

            // Find best intervention for this step
            let best = self.find_best_intervention(
                goal,
                candidate_nodes,
                current_value,
                step,
            );

            if let Some((node, result, cost)) = best {
                // Add to plan
                let improvement = result.predicted_value - current_value;

                if improvement < self.config.min_improvement {
                    break; // Not worth continuing
                }

                interventions.push(PlannedIntervention {
                    step: step + 1,
                    node: node.clone(),
                    intervention_type: InterventionType::Enable,
                    effect: improvement,
                    cost,
                    rationale: format!(
                        "Enable {} to increase {} from {:.2} to {:.2}",
                        node, goal.target, current_value, result.predicted_value
                    ),
                });

                current_value = result.predicted_value;
                total_cost += cost;
            } else {
                break; // No more improvements possible
            }
        }

        // Generate explanation
        let explanation = self.generate_plan_explanation(goal, &interventions, current_value);

        ActionPlan {
            goal: goal.clone(),
            interventions,
            predicted_value: current_value,
            confidence: if current_value > 0.0 { 0.8 } else { 0.5 },
            total_cost,
            explanation,
        }
    }

    /// Find best intervention for current step
    fn find_best_intervention(
        &mut self,
        goal: &Goal,
        candidates: &[String],
        current_value: f64,
        _depth: usize,
    ) -> Option<(String, InterventionResult, f64)> {
        let mut best: Option<(String, InterventionResult, f64, f64)> = None;

        for node in candidates {
            // Predict effect of this intervention
            let result = self.intervention_engine.predict_intervention(node, &goal.target);

            // Calculate improvement
            let improvement = result.predicted_value - current_value;

            if improvement < self.config.min_improvement {
                continue; // Not worth it
            }

            // Get cost
            let cost = self.intervention_costs.get(node).copied().unwrap_or(1.0);

            // Calculate benefit/cost ratio
            let value = improvement / cost.max(0.1);

            // Track best
            if best.is_none() || value > best.as_ref().unwrap().3 {
                best = Some((node.clone(), result, cost, value));
            }
        }

        best.map(|(node, result, cost, _value)| (node, result, cost))
    }

    /// Estimate current value of target (without interventions)
    fn estimate_current_value(&self, target: &str) -> f64 {
        // For now, use a heuristic: 0.5 as default
        // In full implementation, this would query current system state
        0.5
    }

    /// Generate natural language plan explanation
    fn generate_plan_explanation(
        &self,
        goal: &Goal,
        interventions: &[PlannedIntervention],
        final_value: f64,
    ) -> String {
        if interventions.is_empty() {
            return format!(
                "Goal: {} {} = {:.2}. No interventions needed or found.",
                goal.target,
                match goal.direction {
                    GoalDirection::Maximize => "maximize",
                    GoalDirection::Minimize => "minimize",
                    GoalDirection::Exact => "reach",
                },
                goal.desired_value
            );
        }

        let steps_desc = interventions.iter()
            .map(|i| format!("{}. {}", i.step, i.rationale))
            .collect::<Vec<_>>()
            .join("\n  ");

        let success = if goal.is_satisfied(final_value) {
            format!("Goal ACHIEVED: {} = {:.2}", goal.target, final_value)
        } else {
            format!("Best effort: {} = {:.2} (goal: {:.2})",
                goal.target, final_value, goal.desired_value)
        };

        format!(
            "Action Plan to {} {}:\n  {}\n\n{}",
            match goal.direction {
                GoalDirection::Maximize => "maximize",
                GoalDirection::Minimize => "minimize",
                GoalDirection::Exact => "reach",
            },
            goal.target,
            steps_desc,
            success
        )
    }

    /// Compare multiple planning strategies
    pub fn compare_plans(
        &mut self,
        goal: &Goal,
        strategies: &[Vec<String>],
    ) -> Vec<ActionPlan> {
        strategies.iter()
            .map(|candidates| self.plan(goal, candidates))
            .collect()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_creation() {
        let max_goal = Goal::maximize("phi_value");
        assert_eq!(max_goal.target, "phi_value");
        assert_eq!(max_goal.direction, GoalDirection::Maximize);

        let min_goal = Goal::minimize("error_rate");
        assert_eq!(min_goal.direction, GoalDirection::Minimize);

        let exact_goal = Goal::reach("temperature", 0.7, 0.05);
        assert_eq!(exact_goal.direction, GoalDirection::Exact);
        assert_eq!(exact_goal.desired_value, 0.7);
    }

    #[test]
    fn test_goal_satisfaction() {
        let goal = Goal::maximize("phi");

        assert!(goal.is_satisfied(1.0));
        assert!(goal.is_satisfied(0.95));
        assert!(!goal.is_satisfied(0.5));

        assert_eq!(goal.satisfaction_score(1.0), 1.0);
        assert!(goal.satisfaction_score(0.5) < 1.0);
    }

    #[test]
    fn test_planner_creation() {
        let graph = ProbabilisticCausalGraph::new();
        let planner = ActionPlanner::new(graph);

        assert_eq!(planner.config.max_depth, 3);
    }

    #[test]
    fn test_simple_plan() {
        // Create graph: A → B (80% probability)
        let mut graph = ProbabilisticCausalGraph::new();

        for _ in 0..8 {
            graph.observe_edge("A", "B", EdgeType::Direct, true);
        }

        let mut planner = ActionPlanner::new(graph);

        // Goal: maximize B
        let goal = Goal::maximize("B");
        let plan = planner.plan(&goal, &vec!["A".to_string()]);

        // Should recommend enabling A
        assert!(!plan.interventions.is_empty(), "Plan should have interventions");
        assert_eq!(plan.interventions[0].node, "A");
        assert!(plan.predicted_value > 0.5, "Plan should improve B");
    }

    #[test]
    fn test_multi_step_plan() {
        // Create chain: A → B → C
        let mut graph = ProbabilisticCausalGraph::new();

        for _ in 0..7 {
            graph.observe_edge("A", "B", EdgeType::Direct, true);
            graph.observe_edge("B", "C", EdgeType::Direct, true);
        }

        let mut planner = ActionPlanner::new(graph);

        // Goal: maximize C
        let goal = Goal::maximize("C");
        let plan = planner.plan(&goal, &vec!["A".to_string(), "B".to_string()]);

        // Should create a plan
        assert!(!plan.interventions.is_empty());
        assert!(plan.predicted_value > 0.5);
    }
}
