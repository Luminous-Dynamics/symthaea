//! # Predictive Meta-Cognitive Router
//!
//! Uses world model predictions to anticipate future routing needs
//! and proactively allocate resources.
//!
//! ## Key Innovation
//!
//! - Current routing is REACTIVE (based on current Φ)
//! - This router is PREDICTIVE (based on predicted Φ trajectory)
//! - System can prepare for consciousness transitions before they happen
//! - Enables "cognitive preparation" - like taking a deep breath before a hard task
//!
//! ## Research Foundation
//!
//! - Predictive Processing (Friston, 2010)
//! - Proactive Cognitive Control (Braver, 2012)
//! - Preparatory Attention (Kastner & Ungerleider, 2000)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

use super::types::{
    RoutingStrategy, PredictedRoute, RoutingPlan, RoutingOutcome,
    CognitiveResourceType, ConsciousnessAction, LatentConsciousnessState,
    ConsciousnessWorldModel, WorldModelConfig, MetaCognitiveController,
    MetaCognitiveConfig, SelfModel, SelfModelConfig, SubsystemId,
    Router, RouterStats,
};

// Import additional types from core that aren't in types yet
use super::super::core::{
    ConsciousnessTransition, CapabilityDomain,
};

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for predictive router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveRouterConfig {
    /// How many steps ahead to predict
    pub prediction_horizon: usize,
    /// Minimum confidence for predictions
    pub min_confidence: f64,
    /// Enable proactive resource allocation
    pub proactive_allocation: bool,
    /// Transition threshold for warning
    pub transition_threshold: f64,
    /// Number of candidate plans to consider
    pub candidate_plans: usize,
    /// Weight for phi in reward
    pub phi_reward_weight: f64,
    /// Weight for efficiency in reward
    pub efficiency_reward_weight: f64,
}

impl Default for PredictiveRouterConfig {
    fn default() -> Self {
        Self {
            prediction_horizon: 5,
            min_confidence: 0.5,
            proactive_allocation: true,
            transition_threshold: 0.3,
            candidate_plans: 10,
            phi_reward_weight: 0.7,
            efficiency_reward_weight: 0.3,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// STATISTICS
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics for predictive routing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictiveRouterStats {
    /// Total routing decisions
    pub decisions_made: u64,
    /// Accurate predictions (actual matched predicted strategy)
    pub accurate_predictions: u64,
    /// Strategy transitions
    pub transitions: u64,
    /// Proactive allocations made
    pub proactive_allocations: u64,
    /// Average prediction confidence
    pub avg_confidence: f64,
    /// Average prediction error (phi)
    pub avg_phi_error: f64,
}

impl PredictiveRouterStats {
    /// Get prediction accuracy
    pub fn accuracy(&self) -> f64 {
        if self.decisions_made == 0 {
            0.0
        } else {
            self.accurate_predictions as f64 / self.decisions_made as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PREDICTIVE ROUTER
// ═══════════════════════════════════════════════════════════════════════════

/// The Predictive Meta-Cognitive Router
///
/// Uses world model predictions to anticipate future routing needs
/// and proactively allocate resources.
pub struct PredictiveRouter {
    /// World model for predictions
    world_model: ConsciousnessWorldModel,
    /// Meta-cognitive controller for resources
    meta_controller: MetaCognitiveController,
    /// Self-model for behavior prediction
    self_model: SelfModel,
    /// Configuration
    config: PredictiveRouterConfig,
    /// Statistics
    stats: PredictiveRouterStats,
    /// History of actual outcomes for learning
    outcome_history: VecDeque<RoutingOutcome>,
    /// Current routing plan
    current_plan: Option<RoutingPlan>,
}

impl PredictiveRouter {
    /// Create a new predictive router
    pub fn new(config: PredictiveRouterConfig) -> Self {
        Self {
            world_model: ConsciousnessWorldModel::new(WorldModelConfig::default()),
            meta_controller: MetaCognitiveController::new(MetaCognitiveConfig::default()),
            self_model: SelfModel::new(SelfModelConfig::default()),
            config,
            stats: PredictiveRouterStats::default(),
            outcome_history: VecDeque::with_capacity(1000),
            current_plan: None,
        }
    }

    /// Get the current routing strategy based on current state
    pub fn current_strategy(&self, current_phi: f64) -> RoutingStrategy {
        RoutingStrategy::from_phi(current_phi)
    }

    /// Create a predictive routing plan
    pub fn plan_route(&mut self, current_state: &LatentConsciousnessState) -> RoutingPlan {
        let current_phi = current_state.phi;
        let current_strategy = self.current_strategy(current_phi);

        // Get available actions for planning
        let available_actions = ConsciousnessAction::all();

        // Use world model to predict future states
        let mut predictions = Vec::new();
        let mut phi_trajectory = vec![current_phi];

        // Simulate multiple candidate plans
        let mut best_plan_reward = f64::NEG_INFINITY;
        let mut best_predictions = Vec::new();
        let mut best_actions = Vec::new();

        for _ in 0..self.config.candidate_plans {
            // Generate random action sequence
            let mut rng_state = (current_phi * 1000.0) as u64;
            let mut candidate_actions = Vec::new();

            for _ in 0..self.config.prediction_horizon {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let idx = (rng_state as usize) % available_actions.len();
                candidate_actions.push(available_actions[idx]);
            }

            // Simulate trajectory
            let trajectory = self.world_model.simulate_trajectory(current_state, &candidate_actions);
            let cumulative_reward = self.world_model.predict_cumulative_reward(&trajectory, &candidate_actions);

            // Weight reward by phi and efficiency
            let avg_phi = trajectory.iter().map(|s| s.phi).sum::<f64>() / trajectory.len() as f64;
            let efficiency = 1.0 / (candidate_actions.len() as f64);
            let weighted_reward = self.config.phi_reward_weight * avg_phi
                + self.config.efficiency_reward_weight * efficiency
                + cumulative_reward;

            if weighted_reward > best_plan_reward {
                best_plan_reward = weighted_reward;
                best_actions = candidate_actions.clone();
                best_predictions = trajectory.iter().enumerate().map(|(i, state)| {
                    let predicted_phi = state.phi;
                    PredictedRoute {
                        predicted_phi,
                        strategy: RoutingStrategy::from_phi(predicted_phi),
                        confidence: 1.0 / (1.0 + i as f64 * 0.1), // Confidence decays with horizon
                        steps_ahead: i + 1,
                        actions: candidate_actions[..=i.min(candidate_actions.len() - 1)].to_vec(),
                        expected_reward: cumulative_reward / (i + 1) as f64,
                    }
                }).collect();
            }
        }

        predictions = best_predictions;
        let _ = best_actions; // Mark as used

        // Update phi trajectory
        for pred in &predictions {
            phi_trajectory.push(pred.predicted_phi);
        }

        // Detect if major strategy transition is coming
        let transition_warning = predictions.iter().any(|p| {
            let current_factor = current_strategy.resource_factor();
            let predicted_factor = p.strategy.resource_factor();
            (current_factor - predicted_factor).abs() > self.config.transition_threshold
        });

        // Calculate recommended pre-allocation if transition is coming
        let mut recommended_preallocation = HashMap::new();
        if transition_warning && self.config.proactive_allocation {
            // Find max resources needed in prediction horizon
            let max_factor = predictions.iter()
                .map(|p| p.strategy.resource_factor())
                .fold(0.0, f64::max);

            // Pre-allocate resources
            recommended_preallocation.insert(CognitiveResourceType::Computation, max_factor * 0.5);
            recommended_preallocation.insert(CognitiveResourceType::WorkingMemory, max_factor * 0.3);
            recommended_preallocation.insert(CognitiveResourceType::Attention, max_factor * 0.4);

            self.stats.proactive_allocations += 1;
        }

        let plan = RoutingPlan {
            current_phi,
            current_strategy,
            predictions,
            recommended_preallocation,
            transition_warning,
            phi_trajectory,
        };

        self.current_plan = Some(plan.clone());
        plan
    }

    /// Execute routing decision and record outcome
    pub fn execute_route(
        &mut self,
        current_state: &LatentConsciousnessState,
        _action: ConsciousnessAction,
    ) -> RoutingStrategy {
        self.stats.decisions_made += 1;

        let strategy = self.current_strategy(current_state.phi);

        // Allocate resources based on strategy
        let resource_factor = strategy.resource_factor();
        self.meta_controller.request_resources(
            SubsystemId::RecursiveOptimizer,
            CognitiveResourceType::Computation,
            resource_factor * 0.3,
        );

        // Track transition
        if let Some(plan) = &self.current_plan {
            if strategy != plan.current_strategy {
                self.stats.transitions += 1;
            }
        }

        // Update average confidence
        if let Some(plan) = &self.current_plan {
            if let Some(pred) = plan.predictions.first() {
                let n = self.stats.decisions_made as f64;
                self.stats.avg_confidence =
                    (self.stats.avg_confidence * (n - 1.0) + pred.confidence) / n;
            }
        }

        strategy
    }

    /// Record actual outcome for learning
    pub fn record_outcome(&mut self, actual_phi: f64, strategy_used: RoutingStrategy, latency_ms: u32) {
        let predicted_phi = if let Some(plan) = &self.current_plan {
            plan.predictions.first().map(|p| p.predicted_phi).unwrap_or(actual_phi)
        } else {
            actual_phi
        };

        let prediction_accurate = RoutingStrategy::from_phi(predicted_phi) == strategy_used;
        if prediction_accurate {
            self.stats.accurate_predictions += 1;
        }

        // Update average phi error
        let error = (predicted_phi - actual_phi).abs();
        let n = self.stats.decisions_made as f64;
        self.stats.avg_phi_error = (self.stats.avg_phi_error * (n - 1.0) + error) / n;

        let outcome = RoutingOutcome {
            predicted_phi,
            actual_phi,
            strategy_used,
            prediction_accurate,
            resources_consumed: strategy_used.resource_factor(),
            latency_ms,
        };

        self.outcome_history.push_back(outcome);
        while self.outcome_history.len() > 1000 {
            self.outcome_history.pop_front();
        }

        // Create transition for world model learning
        if let Some(plan) = &self.current_plan {
            let from_state = LatentConsciousnessState::from_observables(
                plan.current_phi,
                plan.current_phi,
                plan.current_phi,
                plan.current_phi,
            );
            let to_state = LatentConsciousnessState::from_observables(
                actual_phi,
                actual_phi,
                actual_phi,
                actual_phi,
            );

            // Determine which action was effectively taken based on phi change
            let action = if actual_phi > plan.current_phi {
                ConsciousnessAction::FocusIntegration
            } else if actual_phi < plan.current_phi {
                ConsciousnessAction::Rest
            } else {
                ConsciousnessAction::Consolidate
            };

            let transition = ConsciousnessTransition {
                from_state,
                action,
                to_state,
                reward: actual_phi - plan.current_phi, // Reward = phi improvement
                is_real: true,
            };

            self.world_model.observe_transition(transition);
        }
    }

    /// Analyze counterfactual: what if we had used a different strategy?
    pub fn analyze_counterfactual(
        &self,
        current_state: &LatentConsciousnessState,
        alternative_strategy: RoutingStrategy,
    ) -> CounterfactualAnalysis {
        let current_strategy = self.current_strategy(current_state.phi);

        // Map strategies to actions
        let current_action = self.strategy_to_action(current_strategy);
        let alternative_action = self.strategy_to_action(alternative_strategy);

        // Use world model to predict outcomes
        let current_trajectory = self.world_model.simulate_trajectory(
            current_state,
            &[current_action],
        );
        let alternative_trajectory = self.world_model.simulate_trajectory(
            current_state,
            &[alternative_action],
        );

        let current_phi = current_trajectory.last().map(|s| s.phi).unwrap_or(0.0);
        let alternative_phi = alternative_trajectory.last().map(|s| s.phi).unwrap_or(0.0);

        CounterfactualAnalysis {
            current_strategy,
            alternative_strategy,
            predicted_phi_current: current_phi,
            predicted_phi_alternative: alternative_phi,
            phi_difference: alternative_phi - current_phi,
            recommended_switch: alternative_phi > current_phi + 0.1,
            resource_difference: alternative_strategy.resource_factor() - current_strategy.resource_factor(),
        }
    }

    /// Map routing strategy to consciousness action
    fn strategy_to_action(&self, strategy: RoutingStrategy) -> ConsciousnessAction {
        match strategy {
            RoutingStrategy::FullDeliberation => ConsciousnessAction::EngageLearning,
            RoutingStrategy::StandardProcessing => ConsciousnessAction::FocusIntegration,
            RoutingStrategy::HeuristicGuided => ConsciousnessAction::FocusCoherence,
            RoutingStrategy::FastPatterns => ConsciousnessAction::ExplorePatterns,
            RoutingStrategy::Reflexive => ConsciousnessAction::Rest,
            RoutingStrategy::Ensemble => ConsciousnessAction::ApplyImprovement(0),
            RoutingStrategy::Preparatory => ConsciousnessAction::Consolidate,
        }
    }

    /// Run one cycle of the router
    pub fn cycle(&mut self) {
        // Run meta-cognitive controller cycle
        self.meta_controller.cycle();

        // Update self-model based on routing outcomes
        if let Some(outcome) = self.outcome_history.back() {
            self.self_model.update_capability(
                CapabilityDomain::Metacognition,  // Routing is a meta-cognitive function
                if outcome.prediction_accurate { 1.0 } else { 0.0 },
                0.8,  // High reliability for direct observation
            );
        }
    }

    /// Get router statistics
    pub fn get_stats(&self) -> &PredictiveRouterStats {
        &self.stats
    }

    /// Get router summary
    pub fn summary(&self) -> PredictiveRouterSummary {
        PredictiveRouterSummary {
            decisions_made: self.stats.decisions_made,
            accuracy: self.stats.accuracy(),
            avg_confidence: self.stats.avg_confidence,
            avg_phi_error: self.stats.avg_phi_error,
            transitions: self.stats.transitions,
            proactive_allocations: self.stats.proactive_allocations,
            world_model_ready: self.world_model.is_ready(),
            system_health: self.meta_controller.system_health(),
        }
    }
}

// Implement the Router trait
impl Router for PredictiveRouter {
    fn name(&self) -> &'static str {
        "PredictiveRouter"
    }

    fn current_strategy(&self, phi: f64) -> RoutingStrategy {
        PredictiveRouter::current_strategy(self, phi)
    }

    fn plan(&mut self, state: &LatentConsciousnessState) -> RoutingPlan {
        self.plan_route(state)
    }

    fn execute(&mut self, state: &LatentConsciousnessState, action: ConsciousnessAction) -> RoutingStrategy {
        self.execute_route(state, action)
    }

    fn record_outcome(&mut self, outcome: RoutingOutcome) {
        PredictiveRouter::record_outcome(
            self,
            outcome.actual_phi,
            outcome.strategy_used,
            outcome.latency_ms,
        );
    }

    fn stats(&self) -> RouterStats {
        RouterStats {
            decisions_made: self.stats.decisions_made,
            accurate_predictions: self.stats.accurate_predictions,
            transitions: self.stats.transitions,
            avg_confidence: self.stats.avg_confidence,
            avg_phi_error: self.stats.avg_phi_error,
            custom_metrics: {
                let mut m = HashMap::new();
                m.insert("proactive_allocations".to_string(), self.stats.proactive_allocations as f64);
                m
            },
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SUPPORT TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Counterfactual analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualAnalysis {
    pub current_strategy: RoutingStrategy,
    pub alternative_strategy: RoutingStrategy,
    pub predicted_phi_current: f64,
    pub predicted_phi_alternative: f64,
    pub phi_difference: f64,
    pub recommended_switch: bool,
    pub resource_difference: f64,
}

/// Summary of predictive router state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveRouterSummary {
    pub decisions_made: u64,
    pub accuracy: f64,
    pub avg_confidence: f64,
    pub avg_phi_error: f64,
    pub transitions: u64,
    pub proactive_allocations: u64,
    pub world_model_ready: bool,
    pub system_health: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_router_creation() {
        let router = PredictiveRouter::new(PredictiveRouterConfig::default());
        assert_eq!(router.stats.decisions_made, 0);
    }

    #[test]
    fn test_predictive_router_current_strategy() {
        let router = PredictiveRouter::new(PredictiveRouterConfig::default());
        assert_eq!(router.current_strategy(0.9), RoutingStrategy::FullDeliberation);
        assert_eq!(router.current_strategy(0.1), RoutingStrategy::Reflexive);
    }

    #[test]
    fn test_predictive_router_config_default() {
        let config = PredictiveRouterConfig::default();
        assert_eq!(config.prediction_horizon, 5);
        assert!(config.proactive_allocation);
    }

    #[test]
    fn test_predictive_router_stats_accuracy() {
        let mut stats = PredictiveRouterStats::default();
        assert_eq!(stats.accuracy(), 0.0);

        stats.decisions_made = 10;
        stats.accurate_predictions = 8;
        assert!((stats.accuracy() - 0.8).abs() < 0.001);
    }
}
