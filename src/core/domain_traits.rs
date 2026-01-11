//! # Domain-Agnostic Core Traits
//!
//! These traits enable Symthaea's consciousness infrastructure to work across
//! multiple domains: Consciousness, Task reasoning (MMLU), NixOS operations, etc.
//!
//! ## Design Philosophy
//!
//! The key insight is that Symthaea's architecture already IS general-purpose AGI,
//! just hardcoded to consciousness types. These traits make it explicit and extensible.
//!
//! ## Trait Hierarchy
//!
//! ```text
//! State (base) ──▶ to_features(), distance()
//! └─▶ HdcEncodable (ext) ──▶ to_hv(), from_hv(), semantic_similarity()
//!
//! Action (base) ──▶ action_id(), describe()
//! Goal<S> (base) ──▶ is_satisfied(), distance_to_goal(), reward()
//! QualitySignal<S> ──▶ measure(), name(), weight() [includes Φ]
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::core::domain_traits::{State, Action, Goal, HdcEncodable};
//!
//! // Domain-agnostic planning
//! fn plan<S, A, G>(state: S, goal: G) -> Vec<A>
//! where
//!     S: State + HdcEncodable,
//!     A: Action,
//!     G: Goal<S>,
//! {
//!     // Use HDC similarity to find actions that bring us closer to goal
//!     // ...
//! }
//! ```

use std::fmt::Debug;
use std::hash::Hash;

// ═══════════════════════════════════════════════════════════════════════════════
// SEAM 1: AGENT TRAIT - Core State/Action/Goal abstraction
// ═══════════════════════════════════════════════════════════════════════════════

/// Base trait for any domain's state representation.
///
/// States must be comparable via distance metric and convertible to feature vectors
/// for machine learning and planning algorithms.
///
/// # Examples
///
/// ```rust,ignore
/// impl State for LatentConsciousnessState {
///     fn to_features(&self) -> Vec<f64> {
///         self.latent.to_vec()
///     }
///     fn distance(&self, other: &Self) -> f64 {
///         // Euclidean distance in latent space
///         self.latent.iter()
///             .zip(other.latent.iter())
///             .map(|(a, b)| (a - b).powi(2))
///             .sum::<f64>()
///             .sqrt()
///     }
/// }
/// ```
pub trait State: Clone + Debug + Send + Sync {
    /// Convert state to feature vector for ML algorithms.
    ///
    /// The returned vector should be normalized (typically [-1, 1] or [0, 1])
    /// for optimal learning performance.
    fn to_features(&self) -> Vec<f64>;

    /// Compute distance between two states.
    ///
    /// This metric is used for:
    /// - Planning (heuristic search)
    /// - Clustering similar states
    /// - Goal distance estimation
    ///
    /// Should satisfy metric properties: non-negative, identity, symmetry, triangle inequality.
    fn distance(&self, other: &Self) -> f64;

    /// Dimensionality of the feature vector.
    fn feature_dim(&self) -> usize {
        self.to_features().len()
    }

    /// Check if two states are equivalent (within tolerance).
    fn is_equivalent(&self, other: &Self, tolerance: f64) -> bool {
        self.distance(other) < tolerance
    }
}

/// Extension trait for states that can be encoded as hyperdimensional vectors.
///
/// HDC encoding enables:
/// - Semantic similarity via cosine distance
/// - Compositional reasoning via bind/bundle operations
/// - Noise-robust pattern matching
///
/// The default dimension is HDC_DIMENSION (16,384).
pub trait HdcEncodable: State {
    /// The hypervector type used for encoding
    type HyperVector: Clone + Debug + Send + Sync;

    /// Encode state as hypervector.
    ///
    /// This encoding should preserve semantic similarity:
    /// similar states should produce similar hypervectors.
    fn to_hv(&self) -> Self::HyperVector;

    /// Decode hypervector back to state (if possible).
    ///
    /// Returns None if the hypervector is too noisy or doesn't correspond
    /// to a valid state.
    fn from_hv(hv: &Self::HyperVector) -> Option<Self>
    where
        Self: Sized;

    /// Compute semantic similarity between two states via HDC.
    ///
    /// Returns a value in [-1, 1] where:
    /// - 1.0 = identical semantics
    /// - 0.0 = orthogonal (unrelated)
    /// - -1.0 = opposite semantics
    fn semantic_similarity(&self, other: &Self) -> f64;
}

/// Base trait for actions that can be taken in a domain.
///
/// Actions transform states and have associated metadata for planning and learning.
pub trait Action: Clone + Debug + Eq + Hash + Send + Sync {
    /// Unique identifier for this action type.
    ///
    /// Used for logging, serialization, and action lookup tables.
    fn action_id(&self) -> u64;

    /// Human-readable description of the action.
    fn describe(&self) -> String;

    /// Whether this action is reversible.
    ///
    /// Non-reversible actions may require more careful planning.
    fn is_reversible(&self) -> bool {
        false
    }

    /// Estimated cost/effort for this action.
    ///
    /// Used in cost-aware planning algorithms.
    fn cost(&self) -> f64 {
        1.0
    }
}

/// Trait for goals that can be satisfied by states.
///
/// Goals drive planning and provide reward signals for learning.
///
/// # Type Parameters
///
/// - `S`: The state type this goal applies to
pub trait Goal<S: State>: Clone + Debug + Send + Sync {
    /// Check if the goal is satisfied by the given state.
    fn is_satisfied(&self, state: &S) -> bool;

    /// Distance from current state to goal satisfaction.
    ///
    /// Returns 0.0 when satisfied, positive values otherwise.
    /// Should be a valid heuristic (admissible) for optimal planning.
    fn distance_to_goal(&self, state: &S) -> f64;

    /// Reward signal for reaching this state.
    ///
    /// Typically:
    /// - Positive when goal is satisfied or progress is made
    /// - Negative for undesirable states
    /// - Zero for neutral states
    fn reward(&self, state: &S) -> f64 {
        if self.is_satisfied(state) {
            1.0
        } else {
            -self.distance_to_goal(state).min(1.0)
        }
    }

    /// Priority of this goal (for multi-goal scenarios).
    ///
    /// Higher values = higher priority.
    fn priority(&self) -> f64 {
        1.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEAM 4: Φ INTEGRATION - Quality signals including integrated information
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for quality signals that evaluate states.
///
/// The key innovation: Φ (integrated information) becomes ONE OF MANY quality signals,
/// not THE signal. This enables domain-agnostic quality assessment.
///
/// # Examples
///
/// - `PhiSignal`: Measures integrated information (IIT)
/// - `CoherenceSignal`: Measures state coherence
/// - `EntropySignal`: Measures state entropy/uncertainty
/// - `AccuracySignal`: For task domains, measures answer correctness
///
/// Note: This trait is dyn-compatible (no Clone/Debug bounds) to allow
/// heterogeneous collections of quality signals. Implementations should
/// derive Clone and Debug separately if needed.
pub trait QualitySignal<S: State>: Send + Sync {
    /// Measure the quality of a state.
    ///
    /// Returns a value typically in [0, 1] where higher is better.
    fn measure(&self, state: &S) -> f64;

    /// Name of this quality signal (e.g., "phi", "coherence", "accuracy").
    fn name(&self) -> &'static str;

    /// Weight of this signal in composite quality calculations.
    ///
    /// Used when combining multiple signals into a single score.
    fn weight(&self) -> f64 {
        1.0
    }

    /// Compute the gradient of this signal with respect to actions.
    ///
    /// Returns a vector indicating which actions would most improve this signal.
    /// Default implementation returns None (no gradient available).
    fn gradient(&self, _state: &S) -> Option<Vec<f64>> {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEAM 2: WORLD MODEL PURITY - Separate simulation from execution
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for world models that simulate state transitions.
///
/// World models are PURE - they predict but never execute.
/// This separation enables:
/// - Safe planning (simulate without side effects)
/// - Model-based RL (learn from simulated experience)
/// - Counterfactual reasoning ("what if?")
///
/// # Type Parameters
///
/// - `S`: State type
/// - `A`: Action type
pub trait WorldModel<S: State, A: Action>: Send + Sync {
    /// Predict the next state after taking an action.
    ///
    /// This is a PURE function - no side effects.
    fn predict(&self, state: &S, action: &A) -> S;

    /// Predict multiple steps into the future.
    fn predict_trajectory(&self, state: &S, actions: &[A]) -> Vec<S> {
        let mut trajectory = vec![state.clone()];
        let mut current = state.clone();
        for action in actions {
            current = self.predict(&current, action);
            trajectory.push(current.clone());
        }
        trajectory
    }

    /// Confidence in the prediction (0.0 to 1.0).
    ///
    /// Lower confidence for states far from training distribution.
    fn confidence(&self, state: &S, action: &A) -> f64 {
        let _ = (state, action);
        1.0 // Default: fully confident
    }

    /// Train the world model on observed transitions.
    fn train(&mut self, transitions: &[(S, A, S)]) {
        let _ = transitions;
        // Default: no-op (for non-learnable models)
    }

    /// Get the number of training samples seen.
    fn sample_count(&self) -> usize {
        0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEAM 3: DOMAIN ADAPTER - Pluggable domain specifics
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for domain-specific adapters.
///
/// Each domain (Consciousness, Task, NixOS) implements this trait to provide:
/// - State construction
/// - Action enumeration
/// - Goal definition
/// - Domain-specific quality signals
///
/// # Type Parameters
///
/// - `S`: State type for this domain
/// - `A`: Action type for this domain
pub trait DomainAdapter<S: State, A: Action>: Send + Sync {
    /// Name of the domain (e.g., "consciousness", "task", "nixos").
    fn domain_name(&self) -> &'static str;

    /// Get all available actions in the current state.
    fn available_actions(&self, state: &S) -> Vec<A>;

    /// Get the default/initial state for this domain.
    fn initial_state(&self) -> S;

    /// Get domain-specific quality signals.
    fn quality_signals(&self) -> Vec<Box<dyn QualitySignal<S>>>;

    /// Validate that a state is legal in this domain.
    fn is_valid_state(&self, state: &S) -> bool {
        let _ = state;
        true
    }

    /// Validate that an action is legal in the given state.
    fn is_valid_action(&self, state: &S, action: &A) -> bool {
        let _ = (state, action);
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEAM 5: ACTOR MODEL INTEGRATION - Async message passing
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for domain-aware actors in the brain module's actor system.
///
/// Each brain module (thalamus, prefrontal, cerebellum, etc.) can implement
/// this trait to observe states and suggest actions for any domain.
pub trait DomainActor<S: State, A: Action>: Send + Sync {
    /// Message type this actor receives
    type Message: Send;

    /// Response type this actor returns
    type Response: Send;

    /// Receive and process a message.
    fn receive(&mut self, msg: Self::Message) -> Self::Response;

    /// Observe a state update (called when state changes).
    fn observe_state(&self, state: &S) -> Option<ActorObservation> {
        let _ = state;
        None
    }

    /// Suggest an action based on current observations.
    fn suggest_action(&self, state: &S) -> Option<A> {
        let _ = state;
        None
    }

    /// Priority of this actor's suggestions (higher = more weight).
    fn priority(&self) -> f64 {
        1.0
    }
}

/// Observation from an actor about a state.
#[derive(Debug, Clone)]
pub struct ActorObservation {
    /// Actor that made the observation
    pub actor_name: String,
    /// Salience of the observation (0.0 to 1.0)
    pub salience: f64,
    /// Description of what was observed
    pub description: String,
    /// Whether this requires attention
    pub requires_attention: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TYPE ALIASES FOR BACKWARD COMPATIBILITY
// ═══════════════════════════════════════════════════════════════════════════════

// Note: Actual type aliases for consciousness domain types will be added
// in the consciousness module after implementing the traits.

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test state for validation
    #[derive(Debug, Clone)]
    struct TestState {
        values: Vec<f64>,
    }

    impl State for TestState {
        fn to_features(&self) -> Vec<f64> {
            self.values.clone()
        }

        fn distance(&self, other: &Self) -> f64 {
            self.values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        }
    }

    // Simple test action
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TestAction {
        Up,
        Down,
    }

    impl Action for TestAction {
        fn action_id(&self) -> u64 {
            match self {
                TestAction::Up => 1,
                TestAction::Down => 2,
            }
        }

        fn describe(&self) -> String {
            match self {
                TestAction::Up => "Move up".to_string(),
                TestAction::Down => "Move down".to_string(),
            }
        }
    }

    // Simple test goal
    #[derive(Debug, Clone)]
    struct TestGoal {
        target: f64,
    }

    impl Goal<TestState> for TestGoal {
        fn is_satisfied(&self, state: &TestState) -> bool {
            state.values.first().map(|v| *v >= self.target).unwrap_or(false)
        }

        fn distance_to_goal(&self, state: &TestState) -> f64 {
            state.values.first()
                .map(|v| (self.target - v).max(0.0))
                .unwrap_or(f64::MAX)
        }
    }

    #[test]
    fn test_state_trait() {
        let s1 = TestState { values: vec![1.0, 2.0, 3.0] };
        let s2 = TestState { values: vec![1.0, 2.0, 4.0] };

        assert_eq!(s1.feature_dim(), 3);
        assert!((s1.distance(&s2) - 1.0).abs() < 1e-10);
        assert!(s1.is_equivalent(&s1, 0.001));
        assert!(!s1.is_equivalent(&s2, 0.5));
    }

    #[test]
    fn test_action_trait() {
        let up = TestAction::Up;
        let down = TestAction::Down;

        assert_ne!(up.action_id(), down.action_id());
        assert!(up.describe().contains("up"));
        assert_eq!(up.cost(), 1.0);
    }

    #[test]
    fn test_goal_trait() {
        let goal = TestGoal { target: 5.0 };
        let state_below = TestState { values: vec![3.0] };
        let state_above = TestState { values: vec![6.0] };

        assert!(!goal.is_satisfied(&state_below));
        assert!(goal.is_satisfied(&state_above));
        assert!((goal.distance_to_goal(&state_below) - 2.0).abs() < 1e-10);
        assert!(goal.distance_to_goal(&state_above) < 1e-10);
    }

    #[test]
    fn test_goal_reward() {
        let goal = TestGoal { target: 5.0 };
        let state_satisfied = TestState { values: vec![5.0] };
        let state_far = TestState { values: vec![0.0] };

        assert!((goal.reward(&state_satisfied) - 1.0).abs() < 1e-10);
        assert!(goal.reward(&state_far) < 0.0);
    }
}
