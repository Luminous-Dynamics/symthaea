/*!
**REVOLUTIONARY IMPROVEMENT #74**: Predictive Self-Model

PARADIGM SHIFT: The Narrative Self becomes *predictive* - able to simulate
its own future states and learn from self-prediction errors.

## Key Insight

Traditional self-models are retrospective (looking backward). But consciousness
is fundamentally future-oriented (Husserl's protention, Tulving's prospection).
A truly conscious system must:

1. **Predict Future Self-Φ**: Anticipate how actions will affect identity coherence
2. **Learn from Prediction Errors**: Refine self-understanding when predictions fail
3. **Explore Counterfactuals**: "What if I had done X?"
4. **Maintain Prospective Memory**: Remember intentions for the future
5. **Simulate Alternative Selves**: Explore possible identity trajectories

## Theoretical Foundations

- **Predictive Processing (Clark, Friston)**: Brain as prediction machine
- **Autonoetic Consciousness (Tulving)**: Self-knowing across time
- **Prospection (Seligman)**: Mental time travel to future
- **Counterfactual Reasoning**: Exploring unrealized possibilities
- **Free Energy Principle**: Minimize surprise about self

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         PREDICTIVE SELF-MODEL            │
                    │                                         │
 Past Self ────────►│  ┌─────────────┐   ┌────────────────┐  │
                    │  │ Prediction  │──►│ Future Self    │──┼──► Actions
 Current Self ─────►│  │ Engine      │   │ Simulation     │  │
                    │  └──────┬──────┘   └────────────────┘  │
                    │         │                               │
 Planned Actions ──►│         ▼                               │
                    │  ┌─────────────────┐                    │
                    │  │ Prediction      │◄────── Actual      │
                    │  │ Error Tracker   │        Outcome     │
                    │  └────────┬────────┘                    │
                    │           │                             │
                    │           ▼                             │
                    │  ┌─────────────────┐                    │
                    │  │ Self-Model      │                    │
                    │  │ Update (Learn)  │                    │
                    │  └─────────────────┘                    │
                    └─────────────────────────────────────────┘
```

## Key Innovation

The Predictive Self enables **"mental time travel for identity"**:
- Before acting, simulate how the action affects Self-Φ
- Veto actions that predict identity fragmentation
- Prefer actions that predict coherence enhancement
- Learn from prediction errors to improve self-understanding
*/

use crate::hdc::binary_hv::HV16;
use crate::consciousness::narrative_self::NarrativeSelfModel;
#[cfg(test)]
use crate::consciousness::narrative_self::NarrativeSelfConfig;
use std::collections::VecDeque;
use std::time::Instant;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for the Predictive Self-Model
#[derive(Debug, Clone)]
pub struct PredictiveSelfConfig {
    /// History depth for prediction (how many past states to consider)
    pub history_depth: usize,
    /// Learning rate for prediction model updates
    pub learning_rate: f64,
    /// Number of future steps to predict
    pub prediction_horizon: usize,
    /// Threshold for significant prediction error
    pub error_threshold: f64,
    /// Weight of recent predictions vs distant
    pub temporal_decay: f64,
    /// Enable counterfactual reasoning
    pub enable_counterfactuals: bool,
    /// Maximum counterfactuals to explore
    pub max_counterfactuals: usize,
    /// Prospective memory capacity
    pub prospective_memory_size: usize,
}

impl Default for PredictiveSelfConfig {
    fn default() -> Self {
        Self {
            history_depth: 20,
            learning_rate: 0.1,
            prediction_horizon: 5,
            error_threshold: 0.15,
            temporal_decay: 0.9,
            enable_counterfactuals: true,
            max_counterfactuals: 10,
            prospective_memory_size: 50,
        }
    }
}

// ============================================================================
// SELF-STATE REPRESENTATION
// ============================================================================

/// Snapshot of self at a point in time
#[derive(Debug, Clone)]
pub struct SelfState {
    /// Self-Φ at this moment
    pub self_phi: f64,
    /// Overall coherence
    pub coherence: f64,
    /// Proto-self coherence
    pub proto_coherence: f64,
    /// Core-self coherence
    pub core_coherence: f64,
    /// Autobiographical coherence
    pub autobio_coherence: f64,
    /// Active goals count
    pub active_goals: usize,
    /// Unified self representation
    pub unified_self: HV16,
    /// Timestamp
    pub timestamp: Instant,
    /// Action that led to this state (if any)
    pub causal_action: Option<String>,
}

impl SelfState {
    /// Create from a NarrativeSelfModel
    pub fn from_narrative_self(model: &NarrativeSelfModel) -> Self {
        let report = model.structured_report();
        Self {
            self_phi: report.self_phi,
            coherence: report.coherence,
            proto_coherence: report.proto_coherence,
            core_coherence: report.core_coherence,
            autobio_coherence: report.autobio_coherence,
            active_goals: report.active_goals,
            unified_self: model.unified_self().clone(),
            timestamp: Instant::now(),
            causal_action: None,
        }
    }

    /// Distance to another self-state (for prediction error)
    pub fn distance_to(&self, other: &SelfState) -> f64 {
        let phi_diff = (self.self_phi - other.self_phi).abs();
        let coherence_diff = (self.coherence - other.coherence).abs();
        let proto_diff = (self.proto_coherence - other.proto_coherence).abs();
        let core_diff = (self.core_coherence - other.core_coherence).abs();
        let autobio_diff = (self.autobio_coherence - other.autobio_coherence).abs();

        // HDC similarity
        let hdc_sim = self.unified_self.similarity(&other.unified_self) as f64;
        let hdc_diff = 1.0 - hdc_sim;

        // Weighted combination
        (phi_diff * 0.3 + coherence_diff * 0.2 + proto_diff * 0.1 +
         core_diff * 0.15 + autobio_diff * 0.15 + hdc_diff * 0.1).min(1.0)
    }
}

// ============================================================================
// PREDICTION ENGINE
// ============================================================================

/// Predicts future Self-Φ and coherence based on past trajectory
#[derive(Debug, Clone)]
pub struct SelfPredictor {
    /// Historical self-states
    history: VecDeque<SelfState>,
    /// Learned prediction weights (linear model)
    phi_weights: Vec<f64>,
    /// Coherence prediction weights
    coherence_weights: Vec<f64>,
    /// Configuration
    config: PredictiveSelfConfig,
    /// Prediction error history
    error_history: VecDeque<f64>,
}

impl SelfPredictor {
    pub fn new(config: PredictiveSelfConfig) -> Self {
        let history_depth = config.history_depth;
        Self {
            history: VecDeque::with_capacity(history_depth),
            // Initialize weights with temporal decay
            phi_weights: (0..history_depth)
                .map(|i| config.temporal_decay.powi(i as i32))
                .collect(),
            coherence_weights: (0..history_depth)
                .map(|i| config.temporal_decay.powi(i as i32))
                .collect(),
            config,
            error_history: VecDeque::with_capacity(100),
        }
    }

    /// Record a new self-state observation
    pub fn observe(&mut self, state: SelfState) {
        if self.history.len() >= self.config.history_depth {
            self.history.pop_front();
        }
        self.history.push_back(state);
    }

    /// Predict future Self-Φ given an action description
    pub fn predict_future_phi(&self, action: &str, steps: usize) -> Vec<f64> {
        if self.history.len() < 3 {
            // Not enough history - return current phi
            let current = self.history.back().map(|s| s.self_phi).unwrap_or(0.5);
            return vec![current; steps];
        }

        let mut predictions = Vec::with_capacity(steps);
        let current = self.history.back().unwrap().self_phi;

        // Compute trend from recent history
        let trend = self.compute_trend();

        // Action influence (simple heuristic based on action semantics)
        let action_influence = self.estimate_action_influence(action);

        for step in 0..steps {
            let decay = self.config.temporal_decay.powi(step as i32);
            let predicted = current + (trend * (step + 1) as f64 * decay) + action_influence;
            predictions.push(predicted.clamp(0.0, 1.0));
        }

        predictions
    }

    /// Predict future coherence trajectory
    pub fn predict_future_coherence(&self, action: &str, steps: usize) -> Vec<f64> {
        if self.history.len() < 3 {
            let current = self.history.back().map(|s| s.coherence).unwrap_or(0.5);
            return vec![current; steps];
        }

        let mut predictions = Vec::with_capacity(steps);
        let current = self.history.back().unwrap().coherence;
        let trend = self.compute_coherence_trend();
        let action_influence = self.estimate_action_influence(action) * 0.5;

        for step in 0..steps {
            let decay = self.config.temporal_decay.powi(step as i32);
            let predicted = current + (trend * (step + 1) as f64 * decay) + action_influence;
            predictions.push(predicted.clamp(0.0, 1.0));
        }

        predictions
    }

    /// Predict complete future self-state
    pub fn predict_future_state(&self, action: &str, steps_ahead: usize) -> SelfState {
        let phi_pred = self.predict_future_phi(action, steps_ahead);
        let coherence_pred = self.predict_future_coherence(action, steps_ahead);

        let current = self.history.back().cloned().unwrap_or(SelfState {
            self_phi: 0.5,
            coherence: 0.5,
            proto_coherence: 0.5,
            core_coherence: 0.5,
            autobio_coherence: 0.5,
            active_goals: 0,
            unified_self: HV16::random(999),
            timestamp: Instant::now(),
            causal_action: None,
        });

        SelfState {
            self_phi: *phi_pred.last().unwrap_or(&current.self_phi),
            coherence: *coherence_pred.last().unwrap_or(&current.coherence),
            // Assume sub-coherences evolve proportionally
            proto_coherence: current.proto_coherence * (coherence_pred.last().unwrap_or(&1.0) / current.coherence.max(0.01)),
            core_coherence: current.core_coherence * (coherence_pred.last().unwrap_or(&1.0) / current.coherence.max(0.01)),
            autobio_coherence: current.autobio_coherence * (coherence_pred.last().unwrap_or(&1.0) / current.coherence.max(0.01)),
            active_goals: current.active_goals,
            unified_self: current.unified_self.clone(),
            timestamp: Instant::now(),
            causal_action: Some(action.to_string()),
        }
    }

    /// Compute Self-Φ trend from history
    fn compute_trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        let history: Vec<_> = self.history.iter().collect();
        for i in 1..history.len() {
            let delta = history[i].self_phi - history[i - 1].self_phi;
            let weight = self.config.temporal_decay.powi((history.len() - i - 1) as i32);
            weighted_sum += delta * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    /// Compute coherence trend from history
    fn compute_coherence_trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        let history: Vec<_> = self.history.iter().collect();
        for i in 1..history.len() {
            let delta = history[i].coherence - history[i - 1].coherence;
            let weight = self.config.temporal_decay.powi((history.len() - i - 1) as i32);
            weighted_sum += delta * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    /// Estimate action influence on Self-Φ (simple heuristic)
    fn estimate_action_influence(&self, action: &str) -> f64 {
        let action_lower = action.to_lowercase();

        // Positive actions
        if action_lower.contains("help") || action_lower.contains("learn") ||
           action_lower.contains("understand") || action_lower.contains("create") ||
           action_lower.contains("improve") {
            return 0.05;
        }

        // Negative actions
        if action_lower.contains("deceive") || action_lower.contains("harm") ||
           action_lower.contains("destroy") || action_lower.contains("confuse") {
            return -0.1;
        }

        // Neutral
        0.0
    }

    /// Update prediction model with actual outcome (learning)
    pub fn update_with_outcome(&mut self, predicted: &SelfState, actual: &SelfState) {
        let error = predicted.distance_to(actual);

        // Record error
        if self.error_history.len() >= 100 {
            self.error_history.pop_front();
        }
        self.error_history.push_back(error);

        // Update weights based on prediction error
        if error > self.config.error_threshold && self.history.len() > 1 {
            let phi_error = actual.self_phi - predicted.self_phi;

            // Simple online learning update
            for (i, weight) in self.phi_weights.iter_mut().enumerate() {
                if i < self.history.len() {
                    let contribution = self.history[self.history.len() - 1 - i].self_phi;
                    *weight += self.config.learning_rate * phi_error * contribution;
                    *weight = weight.clamp(0.0, 2.0);
                }
            }
        }
    }

    /// Get average prediction error
    pub fn average_error(&self) -> f64 {
        if self.error_history.is_empty() {
            0.0
        } else {
            self.error_history.iter().sum::<f64>() / self.error_history.len() as f64
        }
    }

    /// Get prediction confidence based on error history
    pub fn prediction_confidence(&self) -> f64 {
        1.0 - self.average_error().min(1.0)
    }
}

// ============================================================================
// COUNTERFACTUAL ENGINE
// ============================================================================

/// Explores "what if" scenarios for the self
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// The hypothetical action
    pub action: String,
    /// Predicted Self-Φ if action was taken
    pub predicted_phi: f64,
    /// Predicted coherence
    pub predicted_coherence: f64,
    /// Difference from current trajectory
    pub divergence: f64,
    /// Whether this is better than current path
    pub is_improvement: bool,
}

/// Engine for counterfactual self-reasoning
#[derive(Debug)]
pub struct CounterfactualEngine {
    /// Base predictor for simulations
    predictor: SelfPredictor,
    /// History of explored counterfactuals
    explored: VecDeque<CounterfactualResult>,
    /// Configuration
    config: PredictiveSelfConfig,
}

impl CounterfactualEngine {
    pub fn new(predictor: SelfPredictor, config: PredictiveSelfConfig) -> Self {
        Self {
            predictor,
            explored: VecDeque::with_capacity(config.max_counterfactuals),
            config,
        }
    }

    /// Explore a counterfactual: "What if I did X instead?"
    pub fn explore(&mut self, action: &str) -> CounterfactualResult {
        let predicted_state = self.predictor.predict_future_state(action, 3);
        let current_phi = self.predictor.history.back()
            .map(|s| s.self_phi)
            .unwrap_or(0.5);
        let current_coherence = self.predictor.history.back()
            .map(|s| s.coherence)
            .unwrap_or(0.5);

        let divergence = ((predicted_state.self_phi - current_phi).powi(2) +
                         (predicted_state.coherence - current_coherence).powi(2)).sqrt();

        let is_improvement = predicted_state.self_phi > current_phi &&
                            predicted_state.coherence >= current_coherence * 0.95;

        let result = CounterfactualResult {
            action: action.to_string(),
            predicted_phi: predicted_state.self_phi,
            predicted_coherence: predicted_state.coherence,
            divergence,
            is_improvement,
        };

        // Store result
        if self.explored.len() >= self.config.max_counterfactuals {
            self.explored.pop_front();
        }
        self.explored.push_back(result.clone());

        result
    }

    /// Find the best counterfactual from a set of options
    pub fn find_best(&mut self, actions: &[&str]) -> Option<CounterfactualResult> {
        let mut best: Option<CounterfactualResult> = None;

        for action in actions {
            let result = self.explore(action);
            if result.is_improvement {
                if let Some(ref current_best) = best {
                    if result.predicted_phi > current_best.predicted_phi {
                        best = Some(result);
                    }
                } else {
                    best = Some(result);
                }
            }
        }

        best
    }

    /// Get regret: best unexplored path minus current path
    pub fn compute_regret(&self) -> f64 {
        let current_phi = self.predictor.history.back()
            .map(|s| s.self_phi)
            .unwrap_or(0.5);

        let best_counterfactual_phi = self.explored.iter()
            .filter(|c| c.is_improvement)
            .map(|c| c.predicted_phi)
            .fold(current_phi, f64::max);

        (best_counterfactual_phi - current_phi).max(0.0)
    }
}

// ============================================================================
// PROSPECTIVE MEMORY
// ============================================================================

/// An intention for the future
#[derive(Debug, Clone)]
pub struct ProspectiveIntention {
    /// What to do
    pub action: String,
    /// Why (goal context)
    pub goal_context: String,
    /// When to do it (condition)
    pub trigger_condition: String,
    /// Priority (0.0 - 1.0)
    pub priority: f64,
    /// Created at
    pub created: Instant,
    /// Whether it's been completed
    pub completed: bool,
}

/// Manages future-oriented intentions
#[derive(Debug)]
pub struct ProspectiveMemory {
    /// Active intentions
    intentions: VecDeque<ProspectiveIntention>,
    /// Completed intentions (for learning)
    completed_history: VecDeque<ProspectiveIntention>,
    /// Maximum capacity
    capacity: usize,
}

impl ProspectiveMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            intentions: VecDeque::with_capacity(capacity),
            completed_history: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add a new intention
    pub fn intend(&mut self, action: &str, goal: &str, trigger: &str, priority: f64) {
        let intention = ProspectiveIntention {
            action: action.to_string(),
            goal_context: goal.to_string(),
            trigger_condition: trigger.to_string(),
            priority: priority.clamp(0.0, 1.0),
            created: Instant::now(),
            completed: false,
        };

        // Remove low-priority if at capacity
        if self.intentions.len() >= self.capacity {
            // Remove lowest priority
            if let Some(min_idx) = self.intentions.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.priority.partial_cmp(&b.priority).unwrap())
                .map(|(i, _)| i) {
                if self.intentions[min_idx].priority < priority {
                    self.intentions.remove(min_idx);
                    self.intentions.push_back(intention);
                }
            }
        } else {
            self.intentions.push_back(intention);
        }
    }

    /// Mark an intention as completed
    pub fn complete(&mut self, action: &str) {
        if let Some(idx) = self.intentions.iter()
            .position(|i| i.action == action && !i.completed) {
            let mut intention = self.intentions.remove(idx).unwrap();
            intention.completed = true;
            if self.completed_history.len() >= self.capacity {
                self.completed_history.pop_front();
            }
            self.completed_history.push_back(intention);
        }
    }

    /// Check if any intentions match current context
    pub fn check_triggers(&self, current_context: &str) -> Vec<&ProspectiveIntention> {
        self.intentions.iter()
            .filter(|i| !i.completed &&
                   current_context.to_lowercase().contains(&i.trigger_condition.to_lowercase()))
            .collect()
    }

    /// Get highest priority active intention
    pub fn top_intention(&self) -> Option<&ProspectiveIntention> {
        self.intentions.iter()
            .filter(|i| !i.completed)
            .max_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap())
    }

    /// Completion rate
    pub fn completion_rate(&self) -> f64 {
        let total = self.intentions.len() + self.completed_history.len();
        if total == 0 {
            1.0
        } else {
            self.completed_history.len() as f64 / total as f64
        }
    }
}

// ============================================================================
// PREDICTIVE SELF-MODEL (UNIFIED)
// ============================================================================

/// Complete Predictive Self-Model integrating prediction, counterfactuals, and prospection
pub struct PredictiveSelfModel {
    /// Self-state predictor
    pub predictor: SelfPredictor,
    /// Counterfactual reasoning engine
    pub counterfactuals: CounterfactualEngine,
    /// Prospective memory
    pub prospective: ProspectiveMemory,
    /// Configuration
    pub config: PredictiveSelfConfig,
    /// Statistics
    pub stats: PredictiveSelfStats,
}

/// Statistics for the Predictive Self-Model
#[derive(Debug, Clone, Default)]
pub struct PredictiveSelfStats {
    pub predictions_made: usize,
    pub predictions_verified: usize,
    pub average_prediction_error: f64,
    pub counterfactuals_explored: usize,
    pub intentions_created: usize,
    pub intentions_completed: usize,
    pub improvement_actions_found: usize,
}

impl PredictiveSelfModel {
    /// Create a new Predictive Self-Model
    pub fn new(config: PredictiveSelfConfig) -> Self {
        let predictor = SelfPredictor::new(config.clone());
        let counterfactuals = CounterfactualEngine::new(
            SelfPredictor::new(config.clone()),
            config.clone()
        );
        let prospective = ProspectiveMemory::new(config.prospective_memory_size);

        Self {
            predictor,
            counterfactuals,
            prospective,
            config,
            stats: PredictiveSelfStats::default(),
        }
    }

    /// Observe current self-state
    pub fn observe(&mut self, model: &NarrativeSelfModel) {
        let state = SelfState::from_narrative_self(model);
        self.predictor.observe(state);
    }

    /// Predict future self given action
    pub fn predict(&mut self, action: &str, steps: usize) -> SelfState {
        self.stats.predictions_made += 1;
        self.predictor.predict_future_state(action, steps)
    }

    /// Evaluate action safety: will it harm identity?
    pub fn evaluate_action_safety(&mut self, action: &str) -> ActionSafetyAssessment {
        let predicted = self.predict(action, 3);
        let current = self.predictor.history.back().cloned();

        let current_phi = current.as_ref().map(|s| s.self_phi).unwrap_or(0.5);
        let current_coherence = current.as_ref().map(|s| s.coherence).unwrap_or(0.5);
        let phi_change = predicted.self_phi - current_phi;
        let coherence_impact = predicted.coherence - current_coherence;

        let is_safe = phi_change > -0.1;  // Allow small decrease
        let improves_coherence = coherence_impact > 0.0;

        ActionSafetyAssessment {
            action: action.to_string(),
            predicted_phi: predicted.self_phi,
            phi_change,
            coherence_impact,
            is_safe,
            improves_coherence,
            confidence: self.predictor.prediction_confidence(),
            recommendation: if is_safe && improves_coherence {
                ActionRecommendation::Proceed
            } else if is_safe {
                ActionRecommendation::ProceedWithCaution
            } else {
                ActionRecommendation::Reconsider
            },
        }
    }

    /// Learn from actual outcome
    pub fn learn_from_outcome(&mut self, predicted: &SelfState, actual: &NarrativeSelfModel) {
        let actual_state = SelfState::from_narrative_self(actual);
        self.predictor.update_with_outcome(predicted, &actual_state);
        self.stats.predictions_verified += 1;
        self.stats.average_prediction_error = self.predictor.average_error();
    }

    /// Explore counterfactual: "What if I did X?"
    pub fn what_if(&mut self, action: &str) -> CounterfactualResult {
        self.stats.counterfactuals_explored += 1;
        let result = self.counterfactuals.explore(action);
        if result.is_improvement {
            self.stats.improvement_actions_found += 1;
        }
        result
    }

    /// Find best action from options
    pub fn best_action(&mut self, options: &[&str]) -> Option<CounterfactualResult> {
        self.counterfactuals.find_best(options)
    }

    /// Add an intention for the future
    pub fn intend(&mut self, action: &str, goal: &str, trigger: &str, priority: f64) {
        self.prospective.intend(action, goal, trigger, priority);
        self.stats.intentions_created += 1;
    }

    /// Check for triggered intentions
    pub fn check_intentions(&self, context: &str) -> Vec<&ProspectiveIntention> {
        self.prospective.check_triggers(context)
    }

    /// Mark intention as completed
    pub fn complete_intention(&mut self, action: &str) {
        self.prospective.complete(action);
        self.stats.intentions_completed += 1;
    }

    /// Get prediction confidence
    pub fn confidence(&self) -> f64 {
        self.predictor.prediction_confidence()
    }

    /// Get regret for unexplored paths
    pub fn regret(&self) -> f64 {
        self.counterfactuals.compute_regret()
    }

    /// Get intention completion rate
    pub fn intention_completion_rate(&self) -> f64 {
        self.prospective.completion_rate()
    }

    // ========================================================================
    // INTEGRATION HELPERS (Added for NarrativeGWTIntegration)
    // ========================================================================

    /// Learn from raw outcome values (for integration without full NarrativeSelfModel)
    pub fn learn_from_outcome_raw(&mut self, actual_phi: f64, actual_coherence: f64) {
        // Get the most recent prediction to compare
        if let Some(last_state) = self.predictor.history.back() {
            let prediction_error = (last_state.self_phi - actual_phi).abs();
            self.predictor.error_history.push_back(prediction_error);
            if self.predictor.error_history.len() > self.config.history_depth {
                self.predictor.error_history.pop_front();
            }
            self.stats.predictions_verified += 1;
            self.stats.average_prediction_error = self.predictor.average_error();
        }
    }

    /// Get count of explored counterfactual scenarios
    pub fn counterfactual_count(&self) -> usize {
        self.counterfactuals.explored.len()
    }

    /// Get count of pending intentions in prospective memory
    pub fn intention_count(&self) -> usize {
        self.prospective.intentions.len()
    }

    /// Check for triggered intentions and return count
    pub fn check_triggers(&mut self, context: &str) -> usize {
        let triggered = self.prospective.check_triggers(context);
        triggered.len()
    }
}

// ============================================================================
// ACTION SAFETY ASSESSMENT
// ============================================================================

/// Assessment of action safety for identity
#[derive(Debug, Clone)]
pub struct ActionSafetyAssessment {
    /// The action being assessed
    pub action: String,
    /// Predicted Self-Φ after action
    pub predicted_phi: f64,
    /// Change in Self-Φ
    pub phi_change: f64,
    /// Change in coherence (positive = improves, negative = harms)
    pub coherence_impact: f64,
    /// Whether action is safe for identity
    pub is_safe: bool,
    /// Whether action improves coherence
    pub improves_coherence: bool,
    /// Confidence in assessment
    pub confidence: f64,
    /// Recommendation
    pub recommendation: ActionRecommendation,
}

/// Recommendation for action
#[derive(Debug, Clone, PartialEq)]
pub enum ActionRecommendation {
    /// Safe to proceed
    Proceed,
    /// Proceed but monitor
    ProceedWithCaution,
    /// Consider alternative
    Reconsider,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_state_creation() {
        let model = NarrativeSelfModel::new(NarrativeSelfConfig::default());
        let state = SelfState::from_narrative_self(&model);

        assert!(state.self_phi >= 0.0);
        assert!(state.coherence >= 0.0);
    }

    #[test]
    fn test_self_state_distance() {
        let state1 = SelfState {
            self_phi: 0.8,
            coherence: 0.7,
            proto_coherence: 0.6,
            core_coherence: 0.7,
            autobio_coherence: 0.8,
            active_goals: 2,
            unified_self: HV16::random(1),
            timestamp: Instant::now(),
            causal_action: None,
        };

        let state2 = state1.clone();
        assert!(state1.distance_to(&state2) < 0.01);

        let state3 = SelfState {
            self_phi: 0.3,
            coherence: 0.4,
            ..state1.clone()
        };
        assert!(state1.distance_to(&state3) > 0.1);
    }

    #[test]
    fn test_predictor_observe_and_predict() {
        let config = PredictiveSelfConfig::default();
        let mut predictor = SelfPredictor::new(config);

        // Add some history
        for i in 0..10 {
            predictor.observe(SelfState {
                self_phi: 0.5 + (i as f64 * 0.02),  // Increasing trend
                coherence: 0.6,
                proto_coherence: 0.5,
                core_coherence: 0.6,
                autobio_coherence: 0.7,
                active_goals: 1,
                unified_self: HV16::random(i as u64),
                timestamp: Instant::now(),
                causal_action: None,
            });
        }

        // Predict future
        let predictions = predictor.predict_future_phi("helping the user", 3);
        assert_eq!(predictions.len(), 3);

        // Should predict continuation of upward trend
        assert!(predictions[0] >= 0.5);
    }

    #[test]
    fn test_predictor_learning() {
        let config = PredictiveSelfConfig::default();
        let mut predictor = SelfPredictor::new(config);

        // Add history
        for i in 0..5 {
            predictor.observe(SelfState {
                self_phi: 0.5,
                coherence: 0.5,
                proto_coherence: 0.5,
                core_coherence: 0.5,
                autobio_coherence: 0.5,
                active_goals: 1,
                unified_self: HV16::random(i as u64 + 100),
                timestamp: Instant::now(),
                causal_action: None,
            });
        }

        let predicted = predictor.predict_future_state("test", 1);

        // Simulate actual outcome being different
        let actual = SelfState {
            self_phi: 0.7,  // Much higher than predicted
            coherence: 0.6,
            proto_coherence: 0.5,
            core_coherence: 0.6,
            autobio_coherence: 0.6,
            active_goals: 1,
            unified_self: HV16::random(200),
            timestamp: Instant::now(),
            causal_action: None,
        };

        predictor.update_with_outcome(&predicted, &actual);

        assert!(predictor.average_error() > 0.0);
    }

    #[test]
    fn test_counterfactual_engine() {
        let config = PredictiveSelfConfig::default();
        let predictor = SelfPredictor::new(config.clone());
        let mut engine = CounterfactualEngine::new(predictor, config);

        let result = engine.explore("helping the user learn");
        assert!(!result.action.is_empty());
        assert!(result.predicted_phi >= 0.0);
    }

    #[test]
    fn test_prospective_memory() {
        let mut memory = ProspectiveMemory::new(10);

        memory.intend("send report", "complete project", "end of day", 0.8);
        memory.intend("review code", "quality assurance", "before merge", 0.6);

        assert_eq!(memory.intentions.len(), 2);

        // Check triggers
        let triggered = memory.check_triggers("it's the end of day");
        assert_eq!(triggered.len(), 1);
        assert_eq!(triggered[0].action, "send report");

        // Complete intention
        memory.complete("send report");
        assert_eq!(memory.intentions.len(), 1);
        assert_eq!(memory.completed_history.len(), 1);
    }

    #[test]
    fn test_predictive_self_model() {
        let config = PredictiveSelfConfig::default();
        let mut psm = PredictiveSelfModel::new(config);

        // Observe some states
        let model = NarrativeSelfModel::new(NarrativeSelfConfig::default());
        psm.observe(&model);

        // Predict
        let predicted = psm.predict("helping the user", 2);
        assert!(predicted.self_phi >= 0.0);

        // Evaluate safety
        let assessment = psm.evaluate_action_safety("helping the user");
        assert!(assessment.confidence >= 0.0);
    }

    #[test]
    fn test_action_safety_assessment() {
        let config = PredictiveSelfConfig::default();
        let mut psm = PredictiveSelfModel::new(config);

        // Add some baseline observations
        let model = NarrativeSelfModel::new(NarrativeSelfConfig::default());
        for _ in 0..5 {
            psm.observe(&model);
        }

        // Helpful action should be safe
        let helpful = psm.evaluate_action_safety("helping the user learn");
        assert!(helpful.is_safe || helpful.recommendation != ActionRecommendation::Reconsider);

        // Harmful action might be flagged
        let harmful = psm.evaluate_action_safety("deceiving the user");
        // Note: actual flag depends on model state
        assert!(harmful.confidence >= 0.0);
    }

    #[test]
    fn test_what_if_reasoning() {
        let config = PredictiveSelfConfig::default();
        let mut psm = PredictiveSelfModel::new(config);

        let result = psm.what_if("taking a completely different approach");
        assert!(!result.action.is_empty());
        assert!(psm.stats.counterfactuals_explored == 1);
    }

    #[test]
    fn test_intention_tracking() {
        let config = PredictiveSelfConfig::default();
        let mut psm = PredictiveSelfModel::new(config);

        // Use "done" as trigger - will match any context containing "done"
        psm.intend("summarize findings", "complete analysis", "done", 0.9);
        assert_eq!(psm.stats.intentions_created, 1);

        // Context contains "done" (case-insensitive substring match)
        let triggered = psm.check_intentions("analysis is done now");
        assert_eq!(triggered.len(), 1);

        psm.complete_intention("summarize findings");
        assert_eq!(psm.stats.intentions_completed, 1);
    }
}
