//! # Consciousness Routers
//!
//! Routing algorithms for navigating consciousness space.
//! These routers help find optimal paths between consciousness states.

use super::world_model::{
    LatentConsciousnessState, ConsciousnessAction, ActionType,
    ConsciousnessTransition,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of a routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Recommended action
    pub action: ConsciousnessAction,

    /// Confidence in this decision (0-1)
    pub confidence: f32,

    /// Expected next state
    pub expected_state: LatentConsciousnessState,

    /// Alternative actions considered
    pub alternatives: Vec<(ConsciousnessAction, f32)>,

    /// Reasoning for the decision
    pub reasoning: String,
}

/// Types of routing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RouterType {
    /// Direct path (greedy)
    Direct,
    /// Path that maximizes Φ
    PhiMaximizing,
    /// Path that minimizes energy
    EnergyMinimizing,
    /// Exploratory path (seek novelty)
    Exploratory,
    /// Consolidating path (seek stability)
    Consolidating,
    /// Balanced (multi-objective)
    Balanced,
    /// Quantum-inspired superposition
    Quantum,
}

/// Base trait for all routers
pub trait ConsciousnessRouter: Send + Sync {
    /// Get the router type
    fn router_type(&self) -> RouterType;

    /// Make a routing decision
    fn route(
        &mut self,
        current: &LatentConsciousnessState,
        target: &LatentConsciousnessState,
    ) -> RoutingDecision;

    /// Update router based on outcome
    fn update(&mut self, transition: &ConsciousnessTransition);
}

/// Direct greedy router - takes shortest path
#[derive(Debug, Clone)]
pub struct DirectRouter {
    /// Available actions
    actions: Vec<ConsciousnessAction>,
}

impl DirectRouter {
    /// Create with default actions
    pub fn new() -> Self {
        Self {
            actions: vec![
                ConsciousnessAction::new("attend", ActionType::Attend),
                ConsciousnessAction::new("integrate", ActionType::Integrate),
                ConsciousnessAction::new("generate", ActionType::Generate),
                ConsciousnessAction::new("recall", ActionType::Recall),
                ConsciousnessAction::new("rest", ActionType::Rest),
            ],
        }
    }
}

impl Default for DirectRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessRouter for DirectRouter {
    fn router_type(&self) -> RouterType {
        RouterType::Direct
    }

    fn route(
        &mut self,
        current: &LatentConsciousnessState,
        target: &LatentConsciousnessState,
    ) -> RoutingDecision {
        // Simple heuristic: choose action based on biggest gap
        let d_phi = target.phi - current.phi;
        let d_int = target.integration - current.integration;
        let d_coh = target.coherence - current.coherence;
        let d_att = target.attention - current.attention;

        let action = if d_att.abs() > d_phi.abs() && d_att.abs() > d_int.abs() {
            ConsciousnessAction::new("attend", ActionType::Attend)
        } else if d_int.abs() > d_phi.abs() {
            ConsciousnessAction::new("integrate", ActionType::Integrate)
        } else if d_phi > 0.0 {
            ConsciousnessAction::new("generate", ActionType::Generate)
        } else {
            ConsciousnessAction::new("rest", ActionType::Rest)
        };

        RoutingDecision {
            action: action.clone(),
            confidence: 0.6,
            expected_state: target.clone(),
            alternatives: vec![],
            reasoning: "Direct path to target".to_string(),
        }
    }

    fn update(&mut self, _transition: &ConsciousnessTransition) {
        // Direct router doesn't learn
    }
}

/// Phi-maximizing router - prioritizes integrated information
#[derive(Debug, Clone)]
pub struct PhiMaximizingRouter {
    /// History of phi values for learning
    phi_history: Vec<f64>,
    /// Action-value estimates
    q_values: HashMap<ActionType, f64>,
    /// Learning rate
    alpha: f64,
    /// Optional seeded RNG for deterministic exploration
    seeded_rng: Option<rand_chacha::ChaCha8Rng>,
}

impl PhiMaximizingRouter {
    /// Create a new phi-maximizing router
    pub fn new() -> Self {
        let mut q_values = HashMap::new();
        for action in [ActionType::Attend, ActionType::Integrate, ActionType::Generate,
                       ActionType::Recall, ActionType::Store, ActionType::Evaluate,
                       ActionType::Transform, ActionType::Rest] {
            q_values.insert(action, 0.0);
        }

        Self {
            phi_history: Vec::new(),
            q_values,
            alpha: 0.1,
            seeded_rng: None,
        }
    }

    /// Create a deterministic phi-maximizing router from a genesis seed.
    pub fn from_genesis(
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
    ) -> Self {
        use rand::SeedableRng;
        use rand::Rng;
        let mut shake = genesis.domain(&format!("{label}::phi_router"));
        let seed: u64 = shake.gen();
        let mut router = Self::new();
        router.seeded_rng = Some(rand_chacha::ChaCha8Rng::seed_from_u64(seed));
        router
    }
}

impl Default for PhiMaximizingRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessRouter for PhiMaximizingRouter {
    fn router_type(&self) -> RouterType {
        RouterType::PhiMaximizing
    }

    fn route(
        &mut self,
        current: &LatentConsciousnessState,
        _target: &LatentConsciousnessState,
    ) -> RoutingDecision {
        // Choose action with highest Q-value (with exploration)
        let epsilon = 0.1;
        let (rand_val1, rand_val2) = if let Some(ref mut rng) = self.seeded_rng {
            use rand::Rng;
            (rng.gen::<f64>(), rng.gen::<f64>())
        } else {
            (rand::random::<f64>(), rand::random::<f64>())
        };
        let should_explore = rand_val1 < epsilon;

        let action = if should_explore {
            // Random exploration
            let actions = [ActionType::Attend, ActionType::Integrate, ActionType::Generate];
            let idx = (rand_val2 * actions.len() as f64) as usize;
            ConsciousnessAction::new(format!("{:?}", actions[idx % actions.len()]).to_lowercase(), actions[idx % actions.len()])
        } else {
            // Exploit best action
            let best_action = self.q_values.iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| *k)
                .unwrap_or(ActionType::Integrate);
            ConsciousnessAction::new(format!("{:?}", best_action).to_lowercase(), best_action)
        };

        let confidence = self.q_values.get(&action.action_type)
            .copied()
            .map(|v| ((v + 1.0) / 2.0).clamp(0.0, 1.0) as f32)
            .unwrap_or(0.5);

        RoutingDecision {
            action,
            confidence,
            expected_state: LatentConsciousnessState::from_observables(
                (current.phi + 0.1).min(1.0),
                current.integration,
                current.coherence,
                current.attention,
            ),
            alternatives: vec![],
            reasoning: "Maximizing integrated information (Φ)".to_string(),
        }
    }

    fn update(&mut self, transition: &ConsciousnessTransition) {
        // Update Q-value based on phi change
        let phi_delta = transition.to_state.phi - transition.from_state.phi;
        let action_type = transition.action.action_type;

        if let Some(q) = self.q_values.get_mut(&action_type) {
            *q = *q + self.alpha * (phi_delta - *q);
        }

        self.phi_history.push(transition.to_state.phi);
    }
}

/// Exploratory router - seeks novel states
#[derive(Debug, Clone)]
pub struct ExploratoryRouter {
    /// Visited states for novelty detection
    visited_states: Vec<LatentConsciousnessState>,
    /// Novelty threshold
    novelty_threshold: f64,
}

impl ExploratoryRouter {
    /// Create a new exploratory router
    pub fn new() -> Self {
        Self {
            visited_states: Vec::new(),
            novelty_threshold: 0.2,
        }
    }

    /// Calculate novelty of a state
    fn novelty(&self, state: &LatentConsciousnessState) -> f64 {
        if self.visited_states.is_empty() {
            return 1.0;
        }

        // Find minimum distance to any visited state
        let min_dist = self.visited_states.iter()
            .map(|v| state.distance(v))
            .fold(f64::INFINITY, |a, b| a.min(b));

        // Normalize novelty
        (min_dist / self.novelty_threshold).min(1.0)
    }
}

impl Default for ExploratoryRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessRouter for ExploratoryRouter {
    fn router_type(&self) -> RouterType {
        RouterType::Exploratory
    }

    fn route(
        &mut self,
        current: &LatentConsciousnessState,
        _target: &LatentConsciousnessState,
    ) -> RoutingDecision {
        // Generate candidate next states and choose most novel
        let actions = [
            ActionType::Attend,
            ActionType::Integrate,
            ActionType::Generate,
            ActionType::Transform,
        ];

        let mut best_action = ActionType::Generate;
        let mut best_novelty = 0.0;
        let mut best_state = current.clone();

        for action_type in actions {
            // Simulate effect
            let candidate = match action_type {
                ActionType::Attend => LatentConsciousnessState::from_observables(
                    current.phi,
                    current.integration,
                    current.coherence,
                    (current.attention + 0.2).min(1.0),
                ),
                ActionType::Integrate => LatentConsciousnessState::from_observables(
                    (current.phi + 0.1).min(1.0),
                    (current.integration + 0.15).min(1.0),
                    current.coherence,
                    current.attention,
                ),
                ActionType::Generate => LatentConsciousnessState::from_observables(
                    current.phi,
                    current.integration,
                    (current.coherence + 0.1).min(1.0),
                    current.attention,
                ),
                ActionType::Transform => LatentConsciousnessState::from_observables(
                    (current.phi * 1.1).min(1.0),
                    (current.integration * 1.1).min(1.0),
                    (current.coherence * 0.9).max(0.0),
                    current.attention,
                ),
                _ => current.clone(),
            };

            let novelty = self.novelty(&candidate);
            if novelty > best_novelty {
                best_novelty = novelty;
                best_action = action_type;
                best_state = candidate;
            }
        }

        RoutingDecision {
            action: ConsciousnessAction::new(format!("{:?}", best_action).to_lowercase(), best_action),
            confidence: best_novelty as f32,
            expected_state: best_state,
            alternatives: vec![],
            reasoning: format!("Seeking novelty (score: {:.2})", best_novelty),
        }
    }

    fn update(&mut self, transition: &ConsciousnessTransition) {
        // Remember visited state
        self.visited_states.push(transition.to_state.clone());

        // Keep memory bounded
        if self.visited_states.len() > 1000 {
            self.visited_states.remove(0);
        }
    }
}

/// Consolidating router - seeks stability
#[derive(Debug, Clone)]
pub struct ConsolidatingRouter {
    /// Target stability level
    target_coherence: f64,
}

impl ConsolidatingRouter {
    /// Create a new consolidating router
    pub fn new() -> Self {
        Self {
            target_coherence: 0.9,
        }
    }
}

impl Default for ConsolidatingRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessRouter for ConsolidatingRouter {
    fn router_type(&self) -> RouterType {
        RouterType::Consolidating
    }

    fn route(
        &mut self,
        current: &LatentConsciousnessState,
        _target: &LatentConsciousnessState,
    ) -> RoutingDecision {
        // Choose action that increases coherence
        let action = if current.coherence < self.target_coherence {
            ConsciousnessAction::new("rest", ActionType::Rest)
        } else {
            ConsciousnessAction::new("store", ActionType::Store)
        };

        RoutingDecision {
            action,
            confidence: 0.7,
            expected_state: LatentConsciousnessState::from_observables(
                current.phi,
                current.integration,
                (current.coherence + 0.1).min(1.0),
                current.attention,
            ),
            alternatives: vec![],
            reasoning: "Consolidating and stabilizing".to_string(),
        }
    }

    fn update(&mut self, _transition: &ConsciousnessTransition) {
        // Consolidating router doesn't learn dynamically
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_router() {
        let mut router = DirectRouter::new();
        let current = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let target = LatentConsciousnessState::from_observables(0.8, 0.7, 0.6, 0.9);

        let decision = router.route(&current, &target);
        assert!(decision.confidence > 0.0);
    }

    #[test]
    fn test_phi_maximizing_router() {
        let mut router = PhiMaximizingRouter::new();
        let current = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let target = LatentConsciousnessState::from_observables(0.9, 0.5, 0.5, 0.5);

        let decision = router.route(&current, &target);
        assert!(!decision.reasoning.is_empty());
    }

    #[test]
    fn test_exploratory_router() {
        let mut router = ExploratoryRouter::new();
        let current = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let target = LatentConsciousnessState::default();

        // First route should have high novelty
        let decision = router.route(&current, &target);
        assert!(decision.confidence > 0.5);
    }
}
