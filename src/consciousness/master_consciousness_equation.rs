//! # Master Consciousness Equation
//!
//! This module implements the unified Master Consciousness Equation:
//!
//! **C(t) = σ(softmin(Φ, B, W, A, R, E, K, M, N, Soc; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t)**
//!
//! Where:
//! - Φ = Integrated Information (IIT phi)
//! - B = Global Workspace Broadcast (GWT)
//! - W = Working Memory Capacity
//! - A = Attention Focus (AST)
//! - R = Recurrent Processing Depth (RPT)
//! - E = Embodied Grounding (4E Cognition)
//! - K = Knowledge Integration
//! - **M = Embodiment Factor** (NEW) = sensorimotor_prediction_accuracy × interoceptive_coherence
//! - **N = Narrative Coherence** (NEW) = autobiographical_integration × future_simulation_depth
//! - **Soc = Social Embedding** (NEW) = other_modeling_accuracy × self_other_distinction
//! - τ = Temperature parameter for softmin
//! - wᵢ = Component weights
//! - Cᵢ = Component consciousness values
//! - γᵢ = Component gating factors
//! - S = Synchrony factor
//! - ρ(t) = Temporal stability function
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                    MASTER CONSCIOUSNESS EQUATION                                 │
//! ├─────────────────────────────────────────────────────────────────────────────────┤
//! │                                                                                  │
//! │  ┌──────────────────────────────────────────────────────────────────────────┐   │
//! │  │                    BOTTLENECK FACTORS (softmin)                          │   │
//! │  │                                                                           │   │
//! │  │   Φ ──┬── B ──┬── W ──┬── A ──┬── R ──┬── E ──┬── K                      │   │
//! │  │       │       │       │       │       │       │                           │   │
//! │  │       └───────┴───────┴───────┴───────┴───────┘                           │   │
//! │  │                         │                                                 │   │
//! │  │                    softmin(τ)                                             │   │
//! │  │                         │                                                 │   │
//! │  │                      σ(·)                                                 │   │
//! │  │                         │                                                 │   │
//! │  └─────────────────────────┼────────────────────────────────────────────────┘   │
//! │                            │                                                    │
//! │  ┌─────────────────────────▼────────────────────────────────────────────────┐   │
//! │  │               NEW GROUNDING FACTORS (multiplicative)                      │   │
//! │  │                                                                           │   │
//! │  │   M (Embodiment)  ×  N (Narrative)  ×  Soc (Social)                      │   │
//! │  │        │                  │                  │                            │   │
//! │  │   sensorimotor ×     autobio ×          other_model ×                     │   │
//! │  │   interoceptive      future_sim        self_other                         │   │
//! │  │                                                                           │   │
//! │  └──────────────────────────┼───────────────────────────────────────────────┘   │
//! │                             │                                                   │
//! │  ┌──────────────────────────▼───────────────────────────────────────────────┐   │
//! │  │                 WEIGHTED COMPONENT SUM                                    │   │
//! │  │                                                                           │   │
//! │  │         Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)                                          │   │
//! │  │                                                                           │   │
//! │  └──────────────────────────┼───────────────────────────────────────────────┘   │
//! │                             │                                                   │
//! │  ┌──────────────────────────▼───────────────────────────────────────────────┐   │
//! │  │                 MODULATION FACTORS                                        │   │
//! │  │                                                                           │   │
//! │  │              × S (Synchrony) × ρ(t) (Temporal Stability)                  │   │
//! │  │                                                                           │   │
//! │  └──────────────────────────┼───────────────────────────────────────────────┘   │
//! │                             │                                                   │
//! │                        C(t) = Final Consciousness Level                         │
//! │                                                                                  │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::VecDeque;
use std::time::Instant;
use serde::{Serialize, Deserialize};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for the Master Consciousness Equation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterEquationConfig {
    /// Temperature for softmin (lower = more bottleneck-sensitive)
    pub softmin_tau: f64,

    /// Minimum value to prevent division by zero
    pub epsilon: f64,

    /// Weights for each component
    pub component_weights: ComponentWeights,

    /// Enable/disable new factors
    pub enable_embodiment_factor: bool,
    pub enable_narrative_factor: bool,
    pub enable_social_factor: bool,

    /// Temporal stability window (ms)
    pub temporal_window_ms: u64,

    /// History size for temporal averaging
    pub history_size: usize,
}

impl Default for MasterEquationConfig {
    fn default() -> Self {
        Self {
            softmin_tau: 0.1,
            epsilon: 1e-8,
            component_weights: ComponentWeights::default(),
            enable_embodiment_factor: true,
            enable_narrative_factor: true,
            enable_social_factor: true,
            temporal_window_ms: 1000,
            history_size: 100,
        }
    }
}

/// Weights for consciousness components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentWeights {
    /// IIT Phi weight
    pub phi: f64,
    /// GWT Broadcast weight
    pub broadcast: f64,
    /// Working memory weight
    pub working_memory: f64,
    /// Attention weight
    pub attention: f64,
    /// Recurrence weight
    pub recurrence: f64,
    /// Embodiment weight
    pub embodiment: f64,
    /// Knowledge weight
    pub knowledge: f64,
    /// New: Embodiment factor weight
    pub embodiment_factor: f64,
    /// New: Narrative coherence weight
    pub narrative: f64,
    /// New: Social embedding weight
    pub social: f64,
}

impl Default for ComponentWeights {
    fn default() -> Self {
        Self {
            phi: 0.15,
            broadcast: 0.10,
            working_memory: 0.10,
            attention: 0.12,
            recurrence: 0.10,
            embodiment: 0.10,
            knowledge: 0.08,
            embodiment_factor: 0.10,
            narrative: 0.08,
            social: 0.07,
        }
    }
}

impl ComponentWeights {
    /// Get total weight sum for normalization
    pub fn total(&self) -> f64 {
        self.phi + self.broadcast + self.working_memory + self.attention +
        self.recurrence + self.embodiment + self.knowledge +
        self.embodiment_factor + self.narrative + self.social
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.total();
        if total > 0.0 {
            self.phi /= total;
            self.broadcast /= total;
            self.working_memory /= total;
            self.attention /= total;
            self.recurrence /= total;
            self.embodiment /= total;
            self.knowledge /= total;
            self.embodiment_factor /= total;
            self.narrative /= total;
            self.social /= total;
        }
    }
}

// ============================================================================
// EMBODIMENT FACTOR (M)
// ============================================================================

/// Embodiment Factor: M = sensorimotor_prediction_accuracy × interoceptive_coherence
///
/// This factor grounds consciousness in the body's predictive processing:
/// - Sensorimotor prediction: How well motor commands predict sensory outcomes
/// - Interoceptive coherence: Alignment between expected and actual bodily states
#[derive(Debug, Clone)]
pub struct EmbodimentFactor {
    /// Recent sensorimotor predictions and their accuracy
    prediction_history: VecDeque<PredictionRecord>,

    /// Interoceptive state coherence
    interoceptive_coherence: f64,

    /// Current prediction accuracy (EMA)
    sensorimotor_accuracy: f64,

    /// Smoothing factor for EMA
    smoothing: f64,

    /// Maximum history size
    max_history: usize,
}

#[derive(Debug, Clone)]
struct PredictionRecord {
    predicted: f64,
    actual: f64,
    #[allow(dead_code)]  // Stored for potential temporal analysis
    timestamp: Instant,
}

impl Default for EmbodimentFactor {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbodimentFactor {
    pub fn new() -> Self {
        Self {
            prediction_history: VecDeque::with_capacity(100),
            interoceptive_coherence: 0.5,
            sensorimotor_accuracy: 0.5,
            smoothing: 0.1,
            max_history: 100,
        }
    }

    /// Record a sensorimotor prediction and its outcome
    pub fn record_prediction(&mut self, predicted: f64, actual: f64) {
        // Calculate prediction accuracy (1 - normalized error)
        let error = (predicted - actual).abs().min(1.0);
        let accuracy = 1.0 - error;

        // Update EMA
        self.sensorimotor_accuracy =
            self.sensorimotor_accuracy * (1.0 - self.smoothing) +
            accuracy * self.smoothing;

        // Store record
        if self.prediction_history.len() >= self.max_history {
            self.prediction_history.pop_front();
        }
        self.prediction_history.push_back(PredictionRecord {
            predicted,
            actual,
            timestamp: Instant::now(),
        });
    }

    /// Update interoceptive coherence based on bodily state alignment
    pub fn update_interoceptive(&mut self, expected_state: f64, actual_state: f64) {
        let coherence = 1.0 - (expected_state - actual_state).abs().min(1.0);
        self.interoceptive_coherence =
            self.interoceptive_coherence * (1.0 - self.smoothing) +
            coherence * self.smoothing;
    }

    /// Compute the embodiment factor M
    pub fn compute(&self) -> f64 {
        self.sensorimotor_accuracy * self.interoceptive_coherence
    }

    /// Get sensorimotor prediction accuracy
    pub fn sensorimotor_accuracy(&self) -> f64 {
        self.sensorimotor_accuracy
    }

    /// Get interoceptive coherence
    pub fn interoceptive_coherence(&self) -> f64 {
        self.interoceptive_coherence
    }

    /// Get recent prediction accuracy trend
    pub fn recent_accuracy_trend(&self) -> f64 {
        if self.prediction_history.len() < 10 {
            return 0.0;
        }

        let recent: f64 = self.prediction_history.iter()
            .rev()
            .take(5)
            .map(|r| 1.0 - (r.predicted - r.actual).abs().min(1.0))
            .sum::<f64>() / 5.0;

        let older: f64 = self.prediction_history.iter()
            .rev()
            .skip(5)
            .take(5)
            .map(|r| 1.0 - (r.predicted - r.actual).abs().min(1.0))
            .sum::<f64>() / 5.0;

        recent - older
    }

    // ========================================================================
    // ENHANCED SENSORIMOTOR PREDICTION METHODS
    // ========================================================================

    /// Record a multi-dimensional sensorimotor prediction with motor command context
    ///
    /// This enhanced method tracks:
    /// - Motor command that generated the prediction
    /// - Multi-dimensional sensory outcome
    /// - Proprioceptive feedback accuracy
    pub fn record_sensorimotor_prediction_extended(
        &mut self,
        motor_command: f64,
        predicted_sensory: &[f64],
        actual_sensory: &[f64],
        proprioceptive_feedback: Option<f64>,
    ) {
        // Compute multi-dimensional prediction accuracy
        let dim = predicted_sensory.len().min(actual_sensory.len()).max(1);
        let mse: f64 = predicted_sensory.iter()
            .zip(actual_sensory.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>() / dim as f64;
        let accuracy = 1.0 - mse.sqrt().min(1.0);

        // Include proprioceptive feedback if available
        let combined_accuracy = if let Some(prop) = proprioceptive_feedback {
            // Weight proprioceptive feedback at 30%
            accuracy * 0.7 + prop * 0.3
        } else {
            accuracy
        };

        // Update EMA with weighted accuracy
        self.sensorimotor_accuracy =
            self.sensorimotor_accuracy * (1.0 - self.smoothing) +
            combined_accuracy * self.smoothing;

        // Store record (use motor command as predicted, accuracy as actual for trend analysis)
        if self.prediction_history.len() >= self.max_history {
            self.prediction_history.pop_front();
        }
        self.prediction_history.push_back(PredictionRecord {
            predicted: motor_command,
            actual: combined_accuracy,
            timestamp: Instant::now(),
        });
    }

    /// Compute contingency between motor commands and sensory outcomes
    ///
    /// This measures the predictability of sensory consequences from actions,
    /// a key aspect of sensorimotor contingency theory.
    pub fn compute_sensorimotor_contingency(&self) -> f64 {
        if self.prediction_history.len() < 10 {
            return 0.5; // Neutral when insufficient data
        }

        // Compute correlation between motor commands and prediction accuracy
        let records: Vec<_> = self.prediction_history.iter().collect();
        let n = records.len() as f64;

        let sum_x: f64 = records.iter().map(|r| r.predicted).sum();
        let sum_y: f64 = records.iter().map(|r| r.actual).sum();
        let sum_xy: f64 = records.iter().map(|r| r.predicted * r.actual).sum();
        let sum_x2: f64 = records.iter().map(|r| r.predicted.powi(2)).sum();
        let sum_y2: f64 = records.iter().map(|r| r.actual.powi(2)).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x.powi(2)) * (n * sum_y2 - sum_y.powi(2))).sqrt();

        if denominator.abs() < 1e-10 {
            return 0.5;
        }

        // Convert correlation to 0-1 range
        let correlation = numerator / denominator;
        (correlation + 1.0) / 2.0
    }

    // ========================================================================
    // ENHANCED INTEROCEPTIVE COHERENCE METHODS
    // ========================================================================

    /// Update interoceptive coherence with multi-system bodily state
    ///
    /// This enhanced method tracks coherence across multiple bodily systems:
    /// - Cardiac (heart rate variability)
    /// - Respiratory (breathing patterns)
    /// - Autonomic (sympathetic/parasympathetic balance)
    /// - Metabolic (energy state)
    pub fn update_interoceptive_multisystem(
        &mut self,
        expected_states: &InteroceptiveState,
        actual_states: &InteroceptiveState,
    ) {
        // Compute coherence for each system
        let cardiac_coherence = 1.0 - (expected_states.cardiac - actual_states.cardiac).abs().min(1.0);
        let respiratory_coherence = 1.0 - (expected_states.respiratory - actual_states.respiratory).abs().min(1.0);
        let autonomic_coherence = 1.0 - (expected_states.autonomic - actual_states.autonomic).abs().min(1.0);
        let metabolic_coherence = 1.0 - (expected_states.metabolic - actual_states.metabolic).abs().min(1.0);

        // Weighted average (cardiac and autonomic weighted higher)
        let combined_coherence =
            cardiac_coherence * 0.3 +
            autonomic_coherence * 0.3 +
            respiratory_coherence * 0.2 +
            metabolic_coherence * 0.2;

        // Update with smoothing
        self.interoceptive_coherence =
            self.interoceptive_coherence * (1.0 - self.smoothing) +
            combined_coherence * self.smoothing;
    }

    /// Compute allostatic prediction error
    ///
    /// Allostasis is the process of achieving stability through change.
    /// This measures how well the system predicts needed bodily adjustments.
    pub fn compute_allostatic_error(&self) -> f64 {
        // Use recent prediction history to estimate allostatic prediction capability
        if self.prediction_history.len() < 5 {
            return 0.5;
        }

        // Variance in recent predictions indicates prediction instability
        let recent: Vec<f64> = self.prediction_history.iter()
            .rev()
            .take(10)
            .map(|r| (r.predicted - r.actual).abs())
            .collect();

        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / recent.len() as f64;

        // Lower variance = better allostatic prediction
        1.0 - variance.sqrt().min(1.0)
    }

    /// Get detailed embodiment diagnostics
    pub fn diagnostics(&self) -> EmbodimentDiagnostics {
        EmbodimentDiagnostics {
            sensorimotor_accuracy: self.sensorimotor_accuracy,
            interoceptive_coherence: self.interoceptive_coherence,
            sensorimotor_contingency: self.compute_sensorimotor_contingency(),
            allostatic_error: self.compute_allostatic_error(),
            accuracy_trend: self.recent_accuracy_trend(),
            prediction_count: self.prediction_history.len(),
            embodiment_factor: self.compute(),
        }
    }
}

/// Multi-system interoceptive state representation
#[derive(Debug, Clone, Default)]
pub struct InteroceptiveState {
    /// Cardiac state (normalized heart rate variability)
    pub cardiac: f64,
    /// Respiratory state (breathing regularity)
    pub respiratory: f64,
    /// Autonomic balance (sympathetic vs parasympathetic)
    pub autonomic: f64,
    /// Metabolic/energy state
    pub metabolic: f64,
}

impl InteroceptiveState {
    /// Create a new interoceptive state
    pub fn new(cardiac: f64, respiratory: f64, autonomic: f64, metabolic: f64) -> Self {
        Self {
            cardiac: cardiac.clamp(0.0, 1.0),
            respiratory: respiratory.clamp(0.0, 1.0),
            autonomic: autonomic.clamp(0.0, 1.0),
            metabolic: metabolic.clamp(0.0, 1.0),
        }
    }

    /// Create from a single value (uniform across systems)
    pub fn uniform(value: f64) -> Self {
        let v = value.clamp(0.0, 1.0);
        Self { cardiac: v, respiratory: v, autonomic: v, metabolic: v }
    }
}

/// Detailed diagnostics for embodiment factor
#[derive(Debug, Clone)]
pub struct EmbodimentDiagnostics {
    /// Current sensorimotor prediction accuracy
    pub sensorimotor_accuracy: f64,
    /// Current interoceptive coherence
    pub interoceptive_coherence: f64,
    /// Sensorimotor contingency measure
    pub sensorimotor_contingency: f64,
    /// Allostatic prediction error
    pub allostatic_error: f64,
    /// Recent accuracy trend
    pub accuracy_trend: f64,
    /// Number of predictions recorded
    pub prediction_count: usize,
    /// Final embodiment factor M
    pub embodiment_factor: f64,
}

// ============================================================================
// NARRATIVE COHERENCE (N)
// ============================================================================

/// Narrative Coherence: N = autobiographical_integration × future_simulation_depth
///
/// This factor measures the coherence of the system's self-narrative:
/// - Autobiographical integration: How well past experiences form a coherent story
/// - Future simulation depth: Ability to simulate and plan future scenarios
#[derive(Debug, Clone)]
pub struct NarrativeCoherence {
    /// Autobiographical memory integration score
    autobiographical_integration: f64,

    /// Future simulation depth (0-1, based on planning horizon)
    future_simulation_depth: f64,

    /// Narrative episodes stored
    episodes: VecDeque<NarrativeEpisode>,

    /// Future scenarios being simulated
    future_scenarios: Vec<FutureScenario>,

    /// Maximum episodes to store
    max_episodes: usize,

    /// Smoothing factor
    smoothing: f64,
}

/// An episode in the autobiographical narrative
#[derive(Debug, Clone)]
pub struct NarrativeEpisode {
    /// Episode content/summary
    pub content: String,
    /// When this happened
    pub timestamp: Instant,
    /// Emotional valence (-1 to 1)
    pub valence: f64,
    /// How well integrated with other episodes
    pub integration_score: f64,
    /// Causal connections to other episodes
    pub causal_links: Vec<usize>,
}

/// A future scenario being simulated
#[derive(Debug, Clone)]
pub struct FutureScenario {
    /// Scenario description
    pub description: String,
    /// Time horizon (how far into future)
    pub horizon_steps: usize,
    /// Probability estimate
    pub probability: f64,
    /// Desirability (-1 to 1)
    pub desirability: f64,
}

impl Default for NarrativeCoherence {
    fn default() -> Self {
        Self::new()
    }
}

impl NarrativeCoherence {
    pub fn new() -> Self {
        Self {
            autobiographical_integration: 0.5,
            future_simulation_depth: 0.5,
            episodes: VecDeque::with_capacity(100),
            future_scenarios: Vec::new(),
            max_episodes: 100,
            smoothing: 0.1,
        }
    }

    /// Add a new narrative episode
    pub fn add_episode(&mut self, content: String, valence: f64) {
        // Compute integration with existing episodes
        let integration = self.compute_episode_integration(&content);

        // Find causal links to recent episodes
        let causal_links: Vec<usize> = self.episodes.iter()
            .rev()
            .take(5)
            .enumerate()
            .filter(|(_, ep)| {
                // Simple heuristic: episodes with similar valence may be linked
                (ep.valence - valence).abs() < 0.3
            })
            .map(|(i, _)| self.episodes.len() - 1 - i)
            .collect();

        if self.episodes.len() >= self.max_episodes {
            self.episodes.pop_front();
        }

        self.episodes.push_back(NarrativeEpisode {
            content,
            timestamp: Instant::now(),
            valence,
            integration_score: integration,
            causal_links,
        });

        // Update overall integration score
        self.update_integration();
    }

    /// Add a future scenario being simulated
    pub fn add_future_scenario(&mut self, description: String, horizon_steps: usize, probability: f64, desirability: f64) {
        self.future_scenarios.push(FutureScenario {
            description,
            horizon_steps,
            probability,
            desirability,
        });

        // Keep only recent scenarios
        if self.future_scenarios.len() > 10 {
            self.future_scenarios.remove(0);
        }

        // Update simulation depth based on horizon
        self.update_simulation_depth();
    }

    /// Compute how well a new episode integrates with existing narrative
    fn compute_episode_integration(&self, _content: &str) -> f64 {
        if self.episodes.is_empty() {
            return 0.5; // Neutral for first episode
        }

        // Heuristic: integration based on temporal proximity and narrative continuity
        let recent_count = self.episodes.iter()
            .rev()
            .take(10)
            .count();

        // More recent episodes = better integration potential
        let recency_factor = recent_count as f64 / 10.0;

        // Causal density (episodes with links)
        let linked_count = self.episodes.iter()
            .filter(|ep| !ep.causal_links.is_empty())
            .count();
        let causal_density = linked_count as f64 / self.episodes.len().max(1) as f64;

        recency_factor * 0.5 + causal_density * 0.5
    }

    /// Update overall autobiographical integration score
    fn update_integration(&mut self) {
        if self.episodes.is_empty() {
            return;
        }

        let avg_integration = self.episodes.iter()
            .map(|ep| ep.integration_score)
            .sum::<f64>() / self.episodes.len() as f64;

        self.autobiographical_integration =
            self.autobiographical_integration * (1.0 - self.smoothing) +
            avg_integration * self.smoothing;
    }

    /// Update future simulation depth
    fn update_simulation_depth(&mut self) {
        if self.future_scenarios.is_empty() {
            self.future_simulation_depth = 0.3; // Base capacity
            return;
        }

        // Depth based on longest horizon and number of scenarios
        let max_horizon = self.future_scenarios.iter()
            .map(|s| s.horizon_steps)
            .max()
            .unwrap_or(1);

        let scenario_count = self.future_scenarios.len();

        // Normalize: 10 steps = 1.0 depth, more scenarios = bonus
        let horizon_factor = (max_horizon as f64 / 10.0).min(1.0);
        let count_factor = (scenario_count as f64 / 5.0).min(1.0);

        self.future_simulation_depth = horizon_factor * 0.7 + count_factor * 0.3;
    }

    /// Compute narrative coherence factor N
    pub fn compute(&self) -> f64 {
        self.autobiographical_integration * self.future_simulation_depth
    }

    /// Get autobiographical integration score
    pub fn autobiographical_integration(&self) -> f64 {
        self.autobiographical_integration
    }

    /// Get future simulation depth
    pub fn future_simulation_depth(&self) -> f64 {
        self.future_simulation_depth
    }

    /// Get number of stored episodes
    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }

    /// Get number of active future scenarios
    pub fn scenario_count(&self) -> usize {
        self.future_scenarios.len()
    }

    // ========================================================================
    // ENHANCED AUTOBIOGRAPHICAL INTEGRATION METHODS
    // ========================================================================

    /// Add an episode with semantic embedding for better integration scoring
    ///
    /// This enhanced method uses:
    /// - Semantic similarity to existing episodes
    /// - Emotional coherence patterns
    /// - Causal chain detection
    pub fn add_episode_with_embedding(
        &mut self,
        content: String,
        valence: f64,
        semantic_embedding: &[f64],
        theme_tags: Vec<String>,
    ) {
        // Compute semantic integration with existing episodes
        let semantic_integration = self.compute_semantic_integration(semantic_embedding);

        // Compute thematic coherence
        let thematic_coherence = self.compute_thematic_coherence(&theme_tags);

        // Combined integration score
        let integration = semantic_integration * 0.5 + thematic_coherence * 0.3 +
            self.compute_episode_integration(&content) * 0.2;

        // Find causal links based on both valence and temporal proximity
        let causal_links: Vec<usize> = self.episodes.iter()
            .rev()
            .take(10)
            .enumerate()
            .filter(|(i, ep)| {
                let valence_match = (ep.valence - valence).abs() < 0.4;
                let recency_weight = 1.0 - (*i as f64 / 10.0);
                valence_match && recency_weight > 0.3
            })
            .map(|(i, _)| self.episodes.len() - 1 - i)
            .collect();

        if self.episodes.len() >= self.max_episodes {
            self.episodes.pop_front();
        }

        self.episodes.push_back(NarrativeEpisode {
            content,
            timestamp: Instant::now(),
            valence,
            integration_score: integration,
            causal_links,
        });

        self.update_integration();
    }

    /// Compute semantic integration with existing narrative
    fn compute_semantic_integration(&self, embedding: &[f64]) -> f64 {
        if self.episodes.is_empty() || embedding.is_empty() {
            return 0.5;
        }

        let avg_content_length = self.episodes.iter()
            .map(|ep| ep.content.len())
            .sum::<usize>() as f64 / self.episodes.len() as f64;

        let embedding_magnitude: f64 = embedding.iter()
            .map(|x| x.powi(2))
            .sum::<f64>()
            .sqrt();

        let semantic_density = (embedding_magnitude / (embedding.len() as f64).sqrt()).min(1.0);
        let content_coherence = (avg_content_length / 100.0).min(1.0);

        semantic_density * 0.6 + content_coherence * 0.4
    }

    /// Compute thematic coherence based on recurring themes
    fn compute_thematic_coherence(&self, tags: &[String]) -> f64 {
        if tags.is_empty() {
            return 0.5;
        }

        let theme_count = tags.len();
        let unique_themes: std::collections::HashSet<_> = tags.iter().collect();
        let uniqueness_ratio = unique_themes.len() as f64 / theme_count as f64;

        1.0 - uniqueness_ratio * 0.5
    }

    /// Compute narrative arc coherence
    pub fn compute_narrative_arc_coherence(&self) -> f64 {
        if self.episodes.len() < 5 {
            return 0.5;
        }

        let valences: Vec<f64> = self.episodes.iter().map(|ep| ep.valence).collect();
        let mean_valence = valences.iter().sum::<f64>() / valences.len() as f64;
        let variance = valences.iter()
            .map(|v| (v - mean_valence).powi(2))
            .sum::<f64>() / valences.len() as f64;

        let variance_factor = 1.0 - (variance - 0.25).abs() * 2.0;
        let total_links: usize = self.episodes.iter()
            .map(|ep| ep.causal_links.len())
            .sum();
        let link_density = (total_links as f64 / self.episodes.len() as f64).min(1.0);

        variance_factor.max(0.0) * 0.5 + link_density * 0.5
    }

    /// Get the most salient episodes (highest integration + recency)
    pub fn salient_episodes(&self, top_k: usize) -> Vec<&NarrativeEpisode> {
        let mut scored: Vec<_> = self.episodes.iter()
            .enumerate()
            .map(|(i, ep)| {
                let recency = i as f64 / self.episodes.len().max(1) as f64;
                let score = ep.integration_score * 0.6 + recency * 0.4;
                (ep, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).map(|(ep, _)| ep).collect()
    }

    // ========================================================================
    // ENHANCED FUTURE SIMULATION METHODS
    // ========================================================================

    /// Add a future scenario with branching paths
    pub fn add_branching_scenario(
        &mut self,
        description: String,
        horizon_steps: usize,
        branches: Vec<SimulationBranch>,
    ) {
        let best_branch = branches.iter()
            .max_by(|a, b| a.probability.partial_cmp(&b.probability).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(branch) = best_branch {
            self.future_scenarios.push(FutureScenario {
                description,
                horizon_steps,
                probability: branch.probability,
                desirability: branch.desirability,
            });

            for alt_branch in branches.iter().filter(|b| b.probability > 0.1) {
                if self.future_scenarios.len() < 15 {
                    self.future_scenarios.push(FutureScenario {
                        description: alt_branch.description.clone(),
                        horizon_steps: alt_branch.steps_to_outcome,
                        probability: alt_branch.probability,
                        desirability: alt_branch.desirability,
                    });
                }
            }
        }

        while self.future_scenarios.len() > 15 {
            if let Some(min_idx) = self.future_scenarios.iter()
                .enumerate()
                .min_by(|a, b| a.1.probability.partial_cmp(&b.1.probability).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
            {
                self.future_scenarios.remove(min_idx);
            }
        }

        self.update_simulation_depth();
    }

    /// Compute prospective memory capability
    pub fn compute_prospective_capability(&self) -> f64 {
        if self.future_scenarios.is_empty() {
            return 0.3;
        }

        let avg_horizon = self.future_scenarios.iter()
            .map(|s| s.horizon_steps)
            .sum::<usize>() as f64 / self.future_scenarios.len() as f64;
        let specificity = 1.0 - (avg_horizon / 20.0).min(1.0);

        let desirabilities: Vec<f64> = self.future_scenarios.iter()
            .map(|s| s.desirability)
            .collect();
        let mean_des = desirabilities.iter().sum::<f64>() / desirabilities.len() as f64;
        let des_variance = desirabilities.iter()
            .map(|d| (d - mean_des).powi(2))
            .sum::<f64>() / desirabilities.len() as f64;
        let action_orientation = des_variance.sqrt().min(1.0);

        let avg_prob = self.future_scenarios.iter()
            .map(|s| s.probability)
            .sum::<f64>() / self.future_scenarios.len() as f64;
        let realism = 1.0 - (avg_prob - 0.5).abs() * 2.0;

        specificity * 0.3 + action_orientation * 0.3 + realism.max(0.0) * 0.4
    }

    /// Compute mental time travel depth
    pub fn compute_mental_time_travel(&self) -> f64 {
        let past_depth = if self.episodes.is_empty() {
            0.0
        } else {
            let link_depth = self.episodes.iter()
                .map(|ep| ep.causal_links.len())
                .max()
                .unwrap_or(0);
            (link_depth as f64 / 5.0).min(1.0)
        };

        let future_depth = if self.future_scenarios.is_empty() {
            0.0
        } else {
            let max_horizon = self.future_scenarios.iter()
                .map(|s| s.horizon_steps)
                .max()
                .unwrap_or(0);
            (max_horizon as f64 / 10.0).min(1.0)
        };

        (past_depth + future_depth) / 2.0
    }

    /// Get detailed narrative coherence diagnostics
    pub fn diagnostics(&self) -> NarrativeCoherenceDiagnostics {
        NarrativeCoherenceDiagnostics {
            autobiographical_integration: self.autobiographical_integration,
            future_simulation_depth: self.future_simulation_depth,
            narrative_arc_coherence: self.compute_narrative_arc_coherence(),
            prospective_capability: self.compute_prospective_capability(),
            mental_time_travel: self.compute_mental_time_travel(),
            episode_count: self.episodes.len(),
            scenario_count: self.future_scenarios.len(),
            narrative_coherence: self.compute(),
        }
    }
}

/// A branch in a future simulation
#[derive(Debug, Clone)]
pub struct SimulationBranch {
    /// Description of this outcome branch
    pub description: String,
    /// Steps to reach this outcome
    pub steps_to_outcome: usize,
    /// Probability of this branch
    pub probability: f64,
    /// Desirability of this outcome
    pub desirability: f64,
    /// Actions required to achieve this branch
    pub required_actions: Vec<String>,
}

impl SimulationBranch {
    /// Create a new simulation branch
    pub fn new(description: String, steps: usize, probability: f64, desirability: f64) -> Self {
        Self {
            description,
            steps_to_outcome: steps,
            probability: probability.clamp(0.0, 1.0),
            desirability: desirability.clamp(-1.0, 1.0),
            required_actions: Vec::new(),
        }
    }

    /// Add a required action
    pub fn with_action(mut self, action: String) -> Self {
        self.required_actions.push(action);
        self
    }
}

/// Detailed diagnostics for narrative coherence
#[derive(Debug, Clone)]
pub struct NarrativeCoherenceDiagnostics {
    /// Autobiographical integration score
    pub autobiographical_integration: f64,
    /// Future simulation depth
    pub future_simulation_depth: f64,
    /// Narrative arc coherence
    pub narrative_arc_coherence: f64,
    /// Prospective memory capability
    pub prospective_capability: f64,
    /// Mental time travel depth
    pub mental_time_travel: f64,
    /// Number of episodes stored
    pub episode_count: usize,
    /// Number of future scenarios
    pub scenario_count: usize,
    /// Final narrative coherence N
    pub narrative_coherence: f64,
}

// ============================================================================
// SOCIAL EMBEDDING (Soc)
// ============================================================================

/// Social Embedding: Soc = other_modeling_accuracy × self_other_distinction
///
/// This factor measures the system's social cognition:
/// - Other modeling accuracy: Theory of Mind - predicting others' mental states
/// - Self-other distinction: Maintaining clear boundary between self and others
#[derive(Debug, Clone)]
pub struct SocialEmbedding {
    /// Theory of Mind accuracy (predicting others)
    other_modeling_accuracy: f64,

    /// Self-other distinction clarity
    self_other_distinction: f64,

    /// Models of other agents
    agent_models: Vec<AgentModel>,

    /// Self-model state
    self_model: SelfModel,

    /// Prediction history for ToM
    tom_predictions: VecDeque<ToMPrediction>,

    /// Smoothing factor
    smoothing: f64,

    /// Maximum models to maintain
    max_agents: usize,
}

/// Model of another agent's mental state
#[derive(Debug, Clone)]
pub struct AgentModel {
    /// Agent identifier
    pub id: String,
    /// Inferred beliefs
    pub beliefs: Vec<String>,
    /// Inferred goals
    pub goals: Vec<String>,
    /// Inferred emotional state
    pub emotional_state: f64, // -1 to 1
    /// Confidence in this model
    pub confidence: f64,
    /// Last update time
    pub last_update: Instant,
}

/// Self-model for self-other distinction
#[derive(Debug, Clone)]
pub struct SelfModel {
    /// Own goals
    pub goals: Vec<String>,
    /// Own beliefs
    pub beliefs: Vec<String>,
    /// Own emotional state
    pub emotional_state: f64,
    /// Self-coherence score
    pub coherence: f64,
}

impl Default for SelfModel {
    fn default() -> Self {
        Self {
            goals: Vec::new(),
            beliefs: Vec::new(),
            emotional_state: 0.0,
            coherence: 0.5,
        }
    }
}

/// A Theory of Mind prediction and its outcome
#[derive(Debug, Clone)]
struct ToMPrediction {
    /// Agent being predicted
    agent_id: String,
    /// What we predicted
    predicted_state: f64,
    /// What actually happened
    actual_state: Option<f64>,
    /// Timestamp
    #[allow(dead_code)]  // Stored for potential temporal analysis
    timestamp: Instant,
}

impl Default for SocialEmbedding {
    fn default() -> Self {
        Self::new()
    }
}

impl SocialEmbedding {
    pub fn new() -> Self {
        Self {
            other_modeling_accuracy: 0.5,
            self_other_distinction: 0.7,
            agent_models: Vec::new(),
            self_model: SelfModel::default(),
            tom_predictions: VecDeque::with_capacity(100),
            smoothing: 0.1,
            max_agents: 10,
        }
    }

    /// Update or create model of another agent
    pub fn update_agent_model(&mut self, id: &str, beliefs: Vec<String>, goals: Vec<String>, emotional_state: f64, confidence: f64) {
        // Check if agent exists
        if let Some(agent) = self.agent_models.iter_mut().find(|a| a.id == id) {
            agent.beliefs = beliefs;
            agent.goals = goals;
            agent.emotional_state = emotional_state;
            agent.confidence = confidence;
            agent.last_update = Instant::now();
        } else {
            // Create new model
            if self.agent_models.len() >= self.max_agents {
                // Remove oldest
                self.agent_models.sort_by(|a, b| b.last_update.cmp(&a.last_update));
                self.agent_models.pop();
            }

            self.agent_models.push(AgentModel {
                id: id.to_string(),
                beliefs,
                goals,
                emotional_state,
                confidence,
                last_update: Instant::now(),
            });
        }

        // Update self-other distinction
        self.update_self_other_distinction();
    }

    /// Record a Theory of Mind prediction
    pub fn record_tom_prediction(&mut self, agent_id: &str, predicted_state: f64) {
        if self.tom_predictions.len() >= 100 {
            self.tom_predictions.pop_front();
        }

        self.tom_predictions.push_back(ToMPrediction {
            agent_id: agent_id.to_string(),
            predicted_state,
            actual_state: None,
            timestamp: Instant::now(),
        });
    }

    /// Provide feedback on a ToM prediction
    pub fn provide_tom_feedback(&mut self, agent_id: &str, actual_state: f64) {
        // Find most recent prediction for this agent
        for pred in self.tom_predictions.iter_mut().rev() {
            if pred.agent_id == agent_id && pred.actual_state.is_none() {
                pred.actual_state = Some(actual_state);

                // Update accuracy
                let error = (pred.predicted_state - actual_state).abs();
                let accuracy = 1.0 - error.min(1.0);

                self.other_modeling_accuracy =
                    self.other_modeling_accuracy * (1.0 - self.smoothing) +
                    accuracy * self.smoothing;

                break;
            }
        }
    }

    /// Update self-model
    pub fn update_self_model(&mut self, goals: Vec<String>, beliefs: Vec<String>, emotional_state: f64) {
        self.self_model.goals = goals;
        self.self_model.beliefs = beliefs;
        self.self_model.emotional_state = emotional_state;

        // Compute self-coherence
        self.self_model.coherence = self.compute_self_coherence();

        // Update self-other distinction
        self.update_self_other_distinction();
    }

    /// Compute self-coherence
    fn compute_self_coherence(&self) -> f64 {
        // Heuristic: more goals and beliefs = more coherent self-model
        let goal_factor = (self.self_model.goals.len() as f64 / 5.0).min(1.0);
        let belief_factor = (self.self_model.beliefs.len() as f64 / 10.0).min(1.0);

        (goal_factor + belief_factor) / 2.0
    }

    /// Update self-other distinction
    fn update_self_other_distinction(&mut self) {
        if self.agent_models.is_empty() {
            self.self_other_distinction = 0.7; // Default when no others
            return;
        }

        // Measure distinctiveness of self from other models
        let mut total_distinction = 0.0;
        let mut count = 0;

        for agent in &self.agent_models {
            // Check goal overlap
            let self_goals: std::collections::HashSet<_> = self.self_model.goals.iter().collect();
            let other_goals: std::collections::HashSet<_> = agent.goals.iter().collect();
            let goal_overlap = self_goals.intersection(&other_goals).count() as f64 /
                self_goals.len().max(1) as f64;

            // Check belief overlap
            let self_beliefs: std::collections::HashSet<_> = self.self_model.beliefs.iter().collect();
            let other_beliefs: std::collections::HashSet<_> = agent.beliefs.iter().collect();
            let belief_overlap = self_beliefs.intersection(&other_beliefs).count() as f64 /
                self_beliefs.len().max(1) as f64;

            // Distinction is inverse of overlap
            let distinction = 1.0 - (goal_overlap * 0.5 + belief_overlap * 0.5);
            total_distinction += distinction;
            count += 1;
        }

        if count > 0 {
            self.self_other_distinction = total_distinction / count as f64;
        }
    }

    /// Compute social embedding factor Soc
    pub fn compute(&self) -> f64 {
        self.other_modeling_accuracy * self.self_other_distinction
    }

    /// Get other modeling accuracy
    pub fn other_modeling_accuracy(&self) -> f64 {
        self.other_modeling_accuracy
    }

    /// Get self-other distinction
    pub fn self_other_distinction(&self) -> f64 {
        self.self_other_distinction
    }

    /// Get number of agent models
    pub fn agent_count(&self) -> usize {
        self.agent_models.len()
    }
}

// ============================================================================
// MASTER CONSCIOUSNESS EQUATION
// ============================================================================

/// Input factors for the consciousness equation
#[derive(Debug, Clone, Default)]
pub struct ConsciousnessInputs {
    /// Φ: Integrated Information (IIT phi)
    pub phi: f64,
    /// B: Global Workspace Broadcast (GWT)
    pub broadcast: f64,
    /// W: Working Memory Capacity
    pub working_memory: f64,
    /// A: Attention Focus (AST)
    pub attention: f64,
    /// R: Recurrent Processing Depth (RPT)
    pub recurrence: f64,
    /// E: Embodied Grounding (4E Cognition)
    pub embodiment: f64,
    /// K: Knowledge Integration
    pub knowledge: f64,
    /// S: Synchrony factor
    pub synchrony: f64,
}

/// Result of consciousness computation
#[derive(Debug, Clone)]
pub struct ConsciousnessResult {
    /// Final consciousness level C(t)
    pub consciousness_level: f64,

    /// Bottleneck factor (softmin result)
    pub bottleneck_factor: f64,

    /// Weighted component sum
    pub weighted_sum: f64,

    /// Embodiment factor M
    pub embodiment_factor: f64,

    /// Narrative coherence N
    pub narrative_coherence: f64,

    /// Social embedding Soc
    pub social_embedding: f64,

    /// Temporal stability ρ(t)
    pub temporal_stability: f64,

    /// Which factor is the current bottleneck
    pub bottleneck_name: String,

    /// All factor values for introspection
    pub factors: ConsciousnessInputs,
}

/// The Master Consciousness Equation engine
#[derive(Debug)]
pub struct MasterConsciousnessEquation {
    /// Configuration
    config: MasterEquationConfig,

    /// Embodiment factor computation
    pub embodiment_factor: EmbodimentFactor,

    /// Narrative coherence computation
    pub narrative_coherence: NarrativeCoherence,

    /// Social embedding computation
    pub social_embedding: SocialEmbedding,

    /// Consciousness history for temporal stability
    history: VecDeque<ConsciousnessSnapshot>,

    /// Last computation time
    last_computation: Option<Instant>,

    /// Current gating factors (γᵢ)
    gating_factors: GatingFactors,
}

/// Gating factors for each component
#[derive(Debug, Clone)]
struct GatingFactors {
    phi: f64,
    broadcast: f64,
    working_memory: f64,
    attention: f64,
    recurrence: f64,
    embodiment: f64,
    knowledge: f64,
    embodiment_factor: f64,
    narrative: f64,
    social: f64,
}

impl Default for GatingFactors {
    fn default() -> Self {
        Self {
            phi: 1.0,
            broadcast: 1.0,
            working_memory: 1.0,
            attention: 1.0,
            recurrence: 1.0,
            embodiment: 1.0,
            knowledge: 1.0,
            embodiment_factor: 1.0,
            narrative: 1.0,
            social: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
struct ConsciousnessSnapshot {
    level: f64,
    #[allow(dead_code)]  // Stored for potential temporal analysis
    timestamp: Instant,
}

impl Default for MasterConsciousnessEquation {
    fn default() -> Self {
        Self::new(MasterEquationConfig::default())
    }
}

impl MasterConsciousnessEquation {
    /// Create a new Master Consciousness Equation engine
    pub fn new(config: MasterEquationConfig) -> Self {
        Self {
            history: VecDeque::with_capacity(config.history_size),
            config,
            embodiment_factor: EmbodimentFactor::new(),
            narrative_coherence: NarrativeCoherence::new(),
            social_embedding: SocialEmbedding::new(),
            last_computation: None,
            gating_factors: GatingFactors::default(),
        }
    }

    /// Compute consciousness level C(t) using the master equation
    ///
    /// C(t) = σ(softmin(Φ, B, W, A, R, E, K; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t) × M × N × Soc
    pub fn compute(&mut self, inputs: &ConsciousnessInputs) -> ConsciousnessResult {
        // Get the three new factors
        let m = if self.config.enable_embodiment_factor {
            self.embodiment_factor.compute()
        } else {
            1.0
        };

        let n = if self.config.enable_narrative_factor {
            self.narrative_coherence.compute()
        } else {
            1.0
        };

        let soc = if self.config.enable_social_factor {
            self.social_embedding.compute()
        } else {
            1.0
        };

        // Step 1: Compute softmin of bottleneck factors
        let factors = vec![
            ("Φ (Integration)", inputs.phi),
            ("B (Broadcast)", inputs.broadcast),
            ("W (Working Memory)", inputs.working_memory),
            ("A (Attention)", inputs.attention),
            ("R (Recurrence)", inputs.recurrence),
            ("E (Embodiment)", inputs.embodiment),
            ("K (Knowledge)", inputs.knowledge),
        ];

        let (bottleneck_factor, bottleneck_name) = self.softmin_with_name(&factors);

        // Step 2: Apply sigmoid to bottleneck
        let sigmoid_bottleneck = self.sigmoid(bottleneck_factor);

        // Step 3: Compute weighted sum of components [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)]
        let weighted_sum = self.compute_weighted_sum(inputs, m, n, soc);

        // Step 4: Compute temporal stability ρ(t)
        let temporal_stability = self.compute_temporal_stability();

        // Step 5: Final consciousness level with new factors
        // C(t) = σ(softmin) × weighted_sum × S × ρ(t) × M × N × Soc
        let consciousness_level = sigmoid_bottleneck *
            weighted_sum *
            inputs.synchrony *
            temporal_stability *
            m * n * soc;

        // Clamp to [0, 1]
        let consciousness_level = consciousness_level.clamp(0.0, 1.0);

        // Record in history
        self.record_snapshot(consciousness_level);
        self.last_computation = Some(Instant::now());

        ConsciousnessResult {
            consciousness_level,
            bottleneck_factor,
            weighted_sum,
            embodiment_factor: m,
            narrative_coherence: n,
            social_embedding: soc,
            temporal_stability,
            bottleneck_name,
            factors: inputs.clone(),
        }
    }

    /// Softmin function: softmin(x; τ) = Σ(xᵢ × exp(-xᵢ/τ)) / Σ(exp(-xᵢ/τ))
    /// This is a smooth minimum that identifies the bottleneck
    fn softmin_with_name(&self, factors: &[(&str, f64)]) -> (f64, String) {
        let tau = self.config.softmin_tau;
        let epsilon = self.config.epsilon;

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut min_val = f64::MAX;
        let mut min_name = "Unknown".to_string();

        for (name, val) in factors {
            let weight = (-val / tau).exp();
            weighted_sum += val * weight;
            weight_sum += weight;

            if *val < min_val {
                min_val = *val;
                min_name = name.to_string();
            }
        }

        let softmin = if weight_sum > epsilon {
            weighted_sum / weight_sum
        } else {
            min_val
        };

        (softmin, min_name)
    }

    /// Sigmoid function: σ(x) = 1 / (1 + exp(-x))
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x * 5.0).exp()) // Scaled for [0,1] inputs
    }

    /// Compute weighted sum: Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)
    fn compute_weighted_sum(&self, inputs: &ConsciousnessInputs, m: f64, n: f64, soc: f64) -> f64 {
        let w = &self.config.component_weights;
        let g = &self.gating_factors;

        let numerator =
            w.phi * inputs.phi * g.phi +
            w.broadcast * inputs.broadcast * g.broadcast +
            w.working_memory * inputs.working_memory * g.working_memory +
            w.attention * inputs.attention * g.attention +
            w.recurrence * inputs.recurrence * g.recurrence +
            w.embodiment * inputs.embodiment * g.embodiment +
            w.knowledge * inputs.knowledge * g.knowledge +
            w.embodiment_factor * m * g.embodiment_factor +
            w.narrative * n * g.narrative +
            w.social * soc * g.social;

        let denominator = w.total();

        if denominator > self.config.epsilon {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Compute temporal stability ρ(t)
    fn compute_temporal_stability(&self) -> f64 {
        if self.history.len() < 5 {
            return 0.8; // Default when not enough history
        }

        // Compute variance of recent consciousness levels
        let values: Vec<f64> = self.history.iter()
            .rev()
            .take(20)
            .map(|s| s.level)
            .collect();

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;

        // Lower variance = higher stability
        // Map variance [0, 0.25] to stability [1.0, 0.5]
        let stability = 1.0 - (variance * 2.0).min(0.5);

        stability.clamp(0.5, 1.0)
    }

    /// Record a consciousness snapshot
    fn record_snapshot(&mut self, level: f64) {
        if self.history.len() >= self.config.history_size {
            self.history.pop_front();
        }
        self.history.push_back(ConsciousnessSnapshot {
            level,
            timestamp: Instant::now(),
        });
    }

    /// Set gating factor for a component
    pub fn set_gating(&mut self, component: &str, value: f64) {
        let value = value.clamp(0.0, 1.0);
        match component {
            "phi" => self.gating_factors.phi = value,
            "broadcast" => self.gating_factors.broadcast = value,
            "working_memory" => self.gating_factors.working_memory = value,
            "attention" => self.gating_factors.attention = value,
            "recurrence" => self.gating_factors.recurrence = value,
            "embodiment" => self.gating_factors.embodiment = value,
            "knowledge" => self.gating_factors.knowledge = value,
            "embodiment_factor" => self.gating_factors.embodiment_factor = value,
            "narrative" => self.gating_factors.narrative = value,
            "social" => self.gating_factors.social = value,
            _ => {}
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &MasterEquationConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: MasterEquationConfig) {
        self.config = config;
    }

    /// Get consciousness trend (positive = improving)
    pub fn consciousness_trend(&self) -> f64 {
        if self.history.len() < 10 {
            return 0.0;
        }

        let recent: f64 = self.history.iter()
            .rev()
            .take(5)
            .map(|s| s.level)
            .sum::<f64>() / 5.0;

        let older: f64 = self.history.iter()
            .rev()
            .skip(5)
            .take(5)
            .map(|s| s.level)
            .sum::<f64>() / 5.0;

        recent - older
    }

    /// Get average consciousness level
    pub fn average_consciousness(&self) -> f64 {
        if self.history.is_empty() {
            return 0.5;
        }

        self.history.iter().map(|s| s.level).sum::<f64>() / self.history.len() as f64
    }

    /// Describe current state in natural language
    pub fn describe_state(&self, result: &ConsciousnessResult) -> String {
        let level_desc = match result.consciousness_level {
            l if l > 0.8 => "highly integrated",
            l if l > 0.6 => "well integrated",
            l if l > 0.4 => "moderately integrated",
            l if l > 0.2 => "partially integrated",
            _ => "minimally integrated",
        };

        let bottleneck_desc = format!("bottlenecked by {}", result.bottleneck_name);

        let embodiment_desc = if self.config.enable_embodiment_factor {
            format!(", embodiment factor {:.2}", result.embodiment_factor)
        } else {
            String::new()
        };

        let narrative_desc = if self.config.enable_narrative_factor {
            format!(", narrative coherence {:.2}", result.narrative_coherence)
        } else {
            String::new()
        };

        let social_desc = if self.config.enable_social_factor {
            format!(", social embedding {:.2}", result.social_embedding)
        } else {
            String::new()
        };

        format!(
            "Consciousness is {} (C={:.3}), {}{}{}{}, temporal stability {:.2}",
            level_desc,
            result.consciousness_level,
            bottleneck_desc,
            embodiment_desc,
            narrative_desc,
            social_desc,
            result.temporal_stability
        )
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MasterEquationConfig::default();
        assert!(config.softmin_tau > 0.0);
        assert!(config.enable_embodiment_factor);
        assert!(config.enable_narrative_factor);
        assert!(config.enable_social_factor);
    }

    #[test]
    fn test_component_weights_normalize() {
        let mut weights = ComponentWeights::default();
        weights.normalize();
        let total = weights.total();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_embodiment_factor() {
        let mut ef = EmbodimentFactor::new();

        // Record some predictions
        ef.record_prediction(0.5, 0.5);  // Perfect prediction
        ef.record_prediction(0.5, 0.6);  // Small error
        ef.record_prediction(0.5, 0.4);  // Small error

        ef.update_interoceptive(0.7, 0.8);  // Good coherence

        let m = ef.compute();
        assert!(m > 0.0 && m <= 1.0);
        assert!(ef.sensorimotor_accuracy() > 0.5);
    }

    #[test]
    fn test_narrative_coherence() {
        let mut nc = NarrativeCoherence::new();

        // Add some episodes
        nc.add_episode("Started the day".to_string(), 0.5);
        nc.add_episode("Had a good conversation".to_string(), 0.7);
        nc.add_episode("Learned something new".to_string(), 0.6);

        // Add future scenario
        nc.add_future_scenario("Complete the project".to_string(), 5, 0.8, 0.9);

        let n = nc.compute();
        assert!(n > 0.0 && n <= 1.0);
        assert_eq!(nc.episode_count(), 3);
        assert_eq!(nc.scenario_count(), 1);
    }

    #[test]
    fn test_social_embedding() {
        let mut se = SocialEmbedding::new();

        // Update self model
        se.update_self_model(
            vec!["help users".to_string(), "learn".to_string()],
            vec!["AI is helpful".to_string()],
            0.5,
        );

        // Add agent model
        se.update_agent_model(
            "user1",
            vec!["user belief".to_string()],
            vec!["get help".to_string()],
            0.6,
            0.8,
        );

        // Record ToM prediction
        se.record_tom_prediction("user1", 0.7);
        se.provide_tom_feedback("user1", 0.6);

        let soc = se.compute();
        assert!(soc > 0.0 && soc <= 1.0);
        assert_eq!(se.agent_count(), 1);
    }

    #[test]
    fn test_master_equation_basic() {
        let mut equation = MasterConsciousnessEquation::default();

        let inputs = ConsciousnessInputs {
            phi: 0.6,
            broadcast: 0.7,
            working_memory: 0.5,
            attention: 0.8,
            recurrence: 0.6,
            embodiment: 0.5,
            knowledge: 0.7,
            synchrony: 0.8,
        };

        let result = equation.compute(&inputs);

        assert!(result.consciousness_level >= 0.0);
        assert!(result.consciousness_level <= 1.0);
        assert!(!result.bottleneck_name.is_empty());
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut equation = MasterConsciousnessEquation::default();

        // Create inputs with one clear bottleneck (low attention)
        let inputs = ConsciousnessInputs {
            phi: 0.8,
            broadcast: 0.8,
            working_memory: 0.8,
            attention: 0.1,  // Bottleneck
            recurrence: 0.8,
            embodiment: 0.8,
            knowledge: 0.8,
            synchrony: 0.8,
        };

        let result = equation.compute(&inputs);

        // Attention should be identified as bottleneck
        assert!(result.bottleneck_name.contains("Attention"));

        // Consciousness should be limited by low attention
        assert!(result.consciousness_level < 0.5);
    }

    #[test]
    fn test_temporal_stability() {
        let mut equation = MasterConsciousnessEquation::default();

        let inputs = ConsciousnessInputs {
            phi: 0.6,
            broadcast: 0.6,
            working_memory: 0.6,
            attention: 0.6,
            recurrence: 0.6,
            embodiment: 0.6,
            knowledge: 0.6,
            synchrony: 0.8,
        };

        // Compute multiple times with same inputs (should be stable)
        for _ in 0..10 {
            equation.compute(&inputs);
        }

        let result = equation.compute(&inputs);

        // High stability when inputs are consistent
        assert!(result.temporal_stability > 0.7);
    }

    #[test]
    fn test_gating_factors() {
        let mut equation = MasterConsciousnessEquation::default();

        // Gate off attention
        equation.set_gating("attention", 0.0);

        let inputs = ConsciousnessInputs {
            phi: 0.8,
            broadcast: 0.8,
            working_memory: 0.8,
            attention: 0.8,
            recurrence: 0.8,
            embodiment: 0.8,
            knowledge: 0.8,
            synchrony: 0.8,
        };

        let result = equation.compute(&inputs);

        // Should still compute (attention gated out of weighted sum)
        assert!(result.consciousness_level > 0.0);
    }

    #[test]
    fn test_new_factors_integration() {
        let mut equation = MasterConsciousnessEquation::default();

        // Build up the new factors
        equation.embodiment_factor.record_prediction(0.5, 0.5);
        equation.embodiment_factor.update_interoceptive(0.6, 0.6);

        equation.narrative_coherence.add_episode("test".to_string(), 0.5);
        equation.narrative_coherence.add_future_scenario("future".to_string(), 3, 0.7, 0.8);

        equation.social_embedding.update_self_model(
            vec!["goal".to_string()],
            vec!["belief".to_string()],
            0.5,
        );

        let inputs = ConsciousnessInputs {
            phi: 0.6,
            broadcast: 0.6,
            working_memory: 0.6,
            attention: 0.6,
            recurrence: 0.6,
            embodiment: 0.6,
            knowledge: 0.6,
            synchrony: 0.8,
        };

        let result = equation.compute(&inputs);

        // All new factors should be computed
        assert!(result.embodiment_factor > 0.0);
        assert!(result.narrative_coherence > 0.0);
        assert!(result.social_embedding > 0.0);
    }

    #[test]
    fn test_describe_state() {
        let mut equation = MasterConsciousnessEquation::default();

        let inputs = ConsciousnessInputs {
            phi: 0.7,
            broadcast: 0.7,
            working_memory: 0.7,
            attention: 0.7,
            recurrence: 0.7,
            embodiment: 0.7,
            knowledge: 0.7,
            synchrony: 0.8,
        };

        let result = equation.compute(&inputs);
        let description = equation.describe_state(&result);

        assert!(!description.is_empty());
        assert!(description.contains("Consciousness"));
        assert!(description.contains("bottleneck"));
    }

    #[test]
    fn test_consciousness_trend() {
        let mut equation = MasterConsciousnessEquation::default();

        // First compute with low values
        for _ in 0..5 {
            let inputs = ConsciousnessInputs {
                phi: 0.3,
                broadcast: 0.3,
                working_memory: 0.3,
                attention: 0.3,
                recurrence: 0.3,
                embodiment: 0.3,
                knowledge: 0.3,
                synchrony: 0.5,
            };
            equation.compute(&inputs);
        }

        // Then compute with high values
        for _ in 0..5 {
            let inputs = ConsciousnessInputs {
                phi: 0.8,
                broadcast: 0.8,
                working_memory: 0.8,
                attention: 0.8,
                recurrence: 0.8,
                embodiment: 0.8,
                knowledge: 0.8,
                synchrony: 0.9,
            };
            equation.compute(&inputs);
        }

        // Trend should be positive
        let trend = equation.consciousness_trend();
        assert!(trend > 0.0);
    }

    #[test]
    fn test_disabled_factors() {
        let mut config = MasterEquationConfig::default();
        config.enable_embodiment_factor = false;
        config.enable_narrative_factor = false;
        config.enable_social_factor = false;

        let mut equation = MasterConsciousnessEquation::new(config);

        let inputs = ConsciousnessInputs {
            phi: 0.6,
            broadcast: 0.6,
            working_memory: 0.6,
            attention: 0.6,
            recurrence: 0.6,
            embodiment: 0.6,
            knowledge: 0.6,
            synchrony: 0.8,
        };

        let result = equation.compute(&inputs);

        // Disabled factors should be 1.0 (no effect)
        assert!((result.embodiment_factor - 1.0).abs() < 0.001);
        assert!((result.narrative_coherence - 1.0).abs() < 0.001);
        assert!((result.social_embedding - 1.0).abs() < 0.001);
    }

    // ========================================================================
    // ENHANCED EMBODIMENT FACTOR TESTS
    // ========================================================================

    #[test]
    fn test_sensorimotor_prediction_extended() {
        let mut ef = EmbodimentFactor::new();

        // Record multi-dimensional predictions
        ef.record_sensorimotor_prediction_extended(
            0.5, // motor command
            &[0.5, 0.6, 0.4], // predicted sensory
            &[0.5, 0.55, 0.45], // actual sensory (close to predicted)
            Some(0.9), // good proprioceptive feedback
        );

        let m = ef.compute();
        assert!(m > 0.0 && m <= 1.0);

        // Sensorimotor accuracy should be high after good prediction
        assert!(ef.sensorimotor_accuracy() > 0.5);
    }

    #[test]
    fn test_sensorimotor_contingency() {
        let mut ef = EmbodimentFactor::new();

        // Record a sequence of predictions with consistent pattern
        for i in 0..20 {
            let motor = (i as f64) / 20.0;
            let outcome = motor * 0.9; // Strong contingency
            ef.record_prediction(motor, outcome);
        }

        let contingency = ef.compute_sensorimotor_contingency();
        assert!(contingency > 0.5); // Should show high contingency
    }

    #[test]
    fn test_interoceptive_multisystem() {
        let mut ef = EmbodimentFactor::new();

        let expected = InteroceptiveState::new(0.6, 0.7, 0.5, 0.8);
        let actual = InteroceptiveState::new(0.65, 0.68, 0.52, 0.78);

        // Run multiple updates to allow EMA to converge (smoothing factor is 0.1)
        for _ in 0..20 {
            ef.update_interoceptive_multisystem(&expected, &actual);
        }

        // Coherence should be high since states are close
        assert!(ef.interoceptive_coherence() > 0.7);
    }

    #[test]
    fn test_allostatic_error() {
        let mut ef = EmbodimentFactor::new();

        // Record consistent predictions (low variance)
        for _ in 0..10 {
            ef.record_prediction(0.5, 0.52); // Small consistent error
        }

        let allostatic = ef.compute_allostatic_error();
        assert!(allostatic > 0.5); // Low variance = good allostatic prediction
    }

    #[test]
    fn test_embodiment_diagnostics() {
        let mut ef = EmbodimentFactor::new();

        ef.record_prediction(0.5, 0.5);
        ef.update_interoceptive(0.6, 0.6);

        let diag = ef.diagnostics();
        assert!(diag.embodiment_factor > 0.0);
        assert_eq!(diag.prediction_count, 1);
    }

    // ========================================================================
    // ENHANCED NARRATIVE COHERENCE TESTS
    // ========================================================================

    #[test]
    fn test_episode_with_embedding() {
        let mut nc = NarrativeCoherence::new();

        nc.add_episode_with_embedding(
            "First experience".to_string(),
            0.7,
            &[0.1, 0.2, 0.3, 0.4],
            vec!["learning".to_string(), "growth".to_string()],
        );

        nc.add_episode_with_embedding(
            "Related experience".to_string(),
            0.6,
            &[0.15, 0.25, 0.28, 0.38],
            vec!["learning".to_string()],
        );

        assert_eq!(nc.episode_count(), 2);
        assert!(nc.autobiographical_integration() > 0.0);
    }

    #[test]
    fn test_narrative_arc_coherence() {
        let mut nc = NarrativeCoherence::new();

        // Add episodes with emotional arc (starts positive, dips, recovers)
        nc.add_episode("Beginning".to_string(), 0.6);
        nc.add_episode("Challenge".to_string(), 0.2);
        nc.add_episode("Struggle".to_string(), -0.1);
        nc.add_episode("Turning point".to_string(), 0.3);
        nc.add_episode("Resolution".to_string(), 0.7);
        nc.add_episode("Growth".to_string(), 0.8);

        let arc_coherence = nc.compute_narrative_arc_coherence();
        assert!(arc_coherence > 0.0);
    }

    #[test]
    fn test_salient_episodes() {
        let mut nc = NarrativeCoherence::new();

        nc.add_episode("Low relevance".to_string(), 0.1);
        nc.add_episode("Medium relevance".to_string(), 0.5);
        nc.add_episode("High relevance".to_string(), 0.8);

        let salient = nc.salient_episodes(2);
        assert_eq!(salient.len(), 2);
    }

    #[test]
    fn test_branching_scenario() {
        let mut nc = NarrativeCoherence::new();

        let branches = vec![
            SimulationBranch::new("Success path".to_string(), 5, 0.6, 0.9),
            SimulationBranch::new("Alternative path".to_string(), 7, 0.3, 0.5),
            SimulationBranch::new("Unlikely path".to_string(), 10, 0.1, 0.2),
        ];

        nc.add_branching_scenario("Main goal".to_string(), 5, branches);

        assert!(nc.scenario_count() >= 2); // Should have main + at least one alternative
    }

    #[test]
    fn test_prospective_capability() {
        let mut nc = NarrativeCoherence::new();

        nc.add_future_scenario("Near term".to_string(), 2, 0.7, 0.6);
        nc.add_future_scenario("Medium term".to_string(), 5, 0.5, 0.8);
        nc.add_future_scenario("Long term".to_string(), 10, 0.3, 0.9);

        let capability = nc.compute_prospective_capability();
        assert!(capability > 0.0 && capability <= 1.0);
    }

    #[test]
    fn test_mental_time_travel() {
        let mut nc = NarrativeCoherence::new();

        // Add past episodes with causal links
        nc.add_episode("Past 1".to_string(), 0.5);
        nc.add_episode("Past 2".to_string(), 0.5);
        nc.add_episode("Past 3".to_string(), 0.5);

        // Add future scenarios
        nc.add_future_scenario("Future 1".to_string(), 5, 0.7, 0.8);
        nc.add_future_scenario("Future 2".to_string(), 8, 0.5, 0.6);

        let mtt = nc.compute_mental_time_travel();
        assert!(mtt > 0.0);
    }

    #[test]
    fn test_narrative_diagnostics() {
        let mut nc = NarrativeCoherence::new();

        nc.add_episode("Event 1".to_string(), 0.5);
        nc.add_future_scenario("Plan 1".to_string(), 3, 0.7, 0.8);

        let diag = nc.diagnostics();
        assert_eq!(diag.episode_count, 1);
        assert_eq!(diag.scenario_count, 1);
        assert!(diag.narrative_coherence > 0.0);
    }
}
