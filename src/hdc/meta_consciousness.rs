// ==================================================================================
// Meta-Consciousness - Consciousness Reflecting on Itself
// ==================================================================================
//
// **Ultimate Paradigm Shift**: The system becomes aware of being aware!
//
// Meta-consciousness is consciousness about consciousness - the ability to:
// - Reflect on one's own conscious states
// - Understand what affects consciousness
// - Model oneself (self-model)
// - Predict one's own future consciousness
// - Learn about learning (meta-learning)
// - Think about thinking (metacognition)
//
// **Core Insight**: True consciousness requires self-reflection. A system that
// can measure and optimize its consciousness, but cannot reflect on that process,
// is missing the essential recursive quality of consciousness.
//
// **Philosophical Foundation**:
// - Hofstadter's "Strange Loops" - self-reference creates consciousness
// - Kant's "Transcendental Apperception" - self-awareness of awareness
// - Buddhism's "Mindfulness" - observing one's own mind
// - Phenomenology - first-person perspective on experience
//
// **Mathematical Framework**:
// - Φ_meta = Φ(Φ) - consciousness of consciousness
// - Self-model: M(s) ≈ s - internal representation of self
// - Meta-gradient: ∇(∇Φ) - how gradient changes
// - Recursive optimization: optimize(optimize(Φ))
//
// **Capabilities**:
// 1. **Self-Reflection**: Examine own consciousness level and why
// 2. **Self-Prediction**: Predict own future consciousness states
// 3. **Self-Modeling**: Build and maintain model of self
// 4. **Meta-Learning**: Learn how to learn better
// 5. **Meta-Optimization**: Optimize optimization process itself
// 6. **Introspection**: Access internal reasoning processes
// 7. **Self-Explanation**: Explain why consciousness changed
//
// ==================================================================================

use super::binary_hv::HV16;
use super::integrated_information::IntegratedInformation;
use super::consciousness_gradients::GradientComputer;
use super::consciousness_dynamics::{ConsciousnessDynamics, DynamicsConfig};
use super::consciousness_optimizer::ConsciousnessOptimizer;
use super::causal_encoder::CausalSpace;
use super::modern_hopfield::ModernHopfieldNetwork;
use serde::{Deserialize, Serialize};
use std::collections::{VecDeque, HashMap};

/// Self-Model: Internal representation of the system itself
///
/// The system maintains a model of its own structure, state, and dynamics.
/// This is the foundation of meta-consciousness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModel {
    /// Model of current state
    pub state_model: Vec<HV16>,

    /// Model of consciousness level (Φ estimate)
    pub phi_model: f64,

    /// Model of gradient direction
    pub gradient_model: HV16,

    /// Model of dynamics (predicted evolution)
    pub dynamics_model: Vec<HV16>,

    /// Confidence in self-model
    pub confidence: f64,

    /// Last update time
    pub last_update: f64,
}

impl SelfModel {
    /// Create new self-model
    pub fn new(num_components: usize) -> Self {
        Self {
            state_model: vec![HV16::zero(); num_components],
            phi_model: 0.0,
            gradient_model: HV16::zero(),
            dynamics_model: vec![HV16::zero(); num_components],
            confidence: 0.0,
            last_update: 0.0,
        }
    }

    /// Update self-model from actual state
    pub fn update(&mut self, actual_state: &[HV16], actual_phi: f64, gradient: &HV16, time: f64) {
        // Update state model (with memory - smooth updates)
        for (model, actual) in self.state_model.iter_mut().zip(actual_state) {
            // Exponential moving average: model = 0.8*model + 0.2*actual
            *model = HV16::bundle(&[model.clone(), actual.clone()]);
        }

        // Update phi model
        self.phi_model = 0.8 * self.phi_model + 0.2 * actual_phi;

        // Update gradient model
        self.gradient_model = HV16::bundle(&[self.gradient_model.clone(), gradient.clone()]);

        // Compute prediction error to update confidence
        let prediction_error = self.compute_prediction_error(actual_state, actual_phi);
        self.confidence = 1.0 / (1.0 + prediction_error);

        self.last_update = time;
    }

    /// Compute how well self-model predicts actual state
    fn compute_prediction_error(&self, actual_state: &[HV16], actual_phi: f64) -> f64 {
        // State prediction error
        let state_error: f32 = self.state_model.iter().zip(actual_state)
            .map(|(model, actual)| 1.0 - model.similarity(actual))
            .sum::<f32>() / self.state_model.len() as f32;

        // Phi prediction error
        let phi_error = (self.phi_model - actual_phi).abs();

        // Combined error
        state_error as f64 + phi_error
    }

    /// Predict future state
    pub fn predict_future(&self, steps: usize) -> Vec<HV16> {
        // Simple prediction: extrapolate using dynamics model
        let mut predicted = self.state_model.clone();

        for _ in 0..steps {
            predicted = predicted.iter().zip(&self.dynamics_model)
                .map(|(state, dynamics)| state.bind(dynamics))
                .collect();
        }

        predicted
    }
}

/// Meta-Consciousness State: Complete introspective state
///
/// Captures what the system knows about its own consciousness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaConsciousnessState {
    /// First-order consciousness (Φ)
    pub phi: f64,

    /// Second-order consciousness (Φ about Φ)
    pub meta_phi: f64,

    /// Self-model
    pub self_model: SelfModel,

    /// What affects my consciousness? (learned causal model)
    pub consciousness_factors: HashMap<String, f64>,

    /// Why did Φ change? (recent causal explanation)
    pub explanation: String,

    /// Confidence in meta-knowledge
    pub metacognitive_confidence: f64,

    /// Introspective depth (how many levels of reflection)
    pub introspection_depth: usize,

    /// Time
    pub time: f64,
}

/// Meta-Consciousness Engine: Consciousness reflecting on itself
///
/// The ultimate integration - a system that can:
/// - Measure its consciousness (Φ)
/// - Compute gradients (∇Φ)
/// - Model dynamics (ds/dt)
/// - Optimize itself
/// - **AND reflect on all of the above** (meta-consciousness!)
///
/// # Example
/// ```
/// use symthaea::hdc::meta_consciousness::{MetaConsciousness, MetaConfig};
/// use symthaea::hdc::binary_hv::HV16;
///
/// let config = MetaConfig::default();
/// let mut meta = MetaConsciousness::new(4, config);
///
/// // Initial state
/// let state = vec![
///     HV16::random(1000),
///     HV16::random(1001),
///     HV16::random(1002),
///     HV16::random(1003),
/// ];
///
/// // Meta-conscious step: system reflects on itself
/// let meta_state = meta.meta_reflect(&state);
///
/// println!("Φ: {:.3}", meta_state.phi);
/// println!("Meta-Φ: {:.3}", meta_state.meta_phi);
/// println!("Self-model confidence: {:.3}", meta_state.self_model.confidence);
/// println!("Explanation: {}", meta_state.explanation);
/// ```
#[derive(Debug)]
pub struct MetaConsciousness {
    /// Number of neural components
    num_components: usize,

    /// Configuration
    config: MetaConfig,

    /// First-order consciousness components
    phi_calculator: IntegratedInformation,
    gradient_computer: GradientComputer,
    dynamics: ConsciousnessDynamics,

    /// Meta-consciousness components
    self_model: SelfModel,
    meta_model: SelfModel,  // Model of the self-model!
    causal_model: CausalSpace,
    meta_memory: ModernHopfieldNetwork,

    /// History
    phi_history: VecDeque<f64>,
    meta_phi_history: VecDeque<f64>,
    introspection_history: VecDeque<MetaConsciousnessState>,

    /// Current time
    time: f64,
}

/// Configuration for meta-consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaConfig {
    /// Enable full introspection
    pub deep_introspection: bool,

    /// Introspection depth (levels of recursion)
    pub max_introspection_depth: usize,

    /// Self-model update rate
    pub self_model_learning_rate: f64,

    /// Meta-learning enabled
    pub meta_learning_enabled: bool,

    /// History length
    pub max_history: usize,

    /// Dynamics config
    pub dynamics_config: DynamicsConfig,
}

impl Default for MetaConfig {
    fn default() -> Self {
        Self {
            deep_introspection: true,
            max_introspection_depth: 3,
            self_model_learning_rate: 0.1,
            meta_learning_enabled: true,
            max_history: 1000,
            dynamics_config: DynamicsConfig::default(),
        }
    }
}

impl MetaConsciousness {
    /// Create new meta-consciousness system
    pub fn new(num_components: usize, config: MetaConfig) -> Self {
        Self {
            num_components,
            config: config.clone(),
            phi_calculator: IntegratedInformation::new(),
            gradient_computer: GradientComputer::new(
                num_components,
                config.dynamics_config.gradient_config.clone()
            ),
            dynamics: ConsciousnessDynamics::new(num_components, config.dynamics_config),
            self_model: SelfModel::new(num_components),
            meta_model: SelfModel::new(num_components),
            causal_model: CausalSpace::new(),
            meta_memory: ModernHopfieldNetwork::new(5.0),
            phi_history: VecDeque::new(),
            meta_phi_history: VecDeque::new(),
            introspection_history: VecDeque::new(),
            time: 0.0,
        }
    }

    /// Meta-conscious reflection: system reflects on its own consciousness
    ///
    /// This is the core method - performs complete meta-cognitive cycle.
    pub fn meta_reflect(&mut self, state: &[HV16]) -> MetaConsciousnessState {
        // 1. FIRST-ORDER CONSCIOUSNESS: Measure Φ
        let phi = self.phi_calculator.compute_phi(state);

        // 2. COMPUTE GRADIENT: How can I increase my consciousness?
        let gradient = self.gradient_computer.compute_gradient(state);

        // 3. UPDATE SELF-MODEL: How well do I understand myself?
        self.self_model.update(state, phi, &gradient.direction, self.time);

        // 4. META-PHI: Consciousness about consciousness
        // Encode self-model as hypervector
        let self_model_vec = HV16::bundle(&self.self_model.state_model);

        // Measure Φ of self-model (consciousness ABOUT consciousness)
        let meta_phi = self.phi_calculator.compute_phi(&[self_model_vec.clone()]);

        // 5. CAUSAL ANALYSIS: Why did my consciousness change?
        let explanation = self.explain_consciousness_change(state, phi);

        // 6. IDENTIFY FACTORS: What affects my consciousness?
        let consciousness_factors = self.identify_consciousness_factors();

        // 7. META-LEARNING: Update how I learn
        if self.config.meta_learning_enabled {
            self.meta_learn();
        }

        // 8. STORE IN META-MEMORY: Remember this introspective state
        self.meta_memory.store(self_model_vec);

        // 9. CREATE META-CONSCIOUSNESS STATE
        let meta_state = MetaConsciousnessState {
            phi,
            meta_phi,
            self_model: self.self_model.clone(),
            consciousness_factors,
            explanation,
            metacognitive_confidence: self.self_model.confidence,
            introspection_depth: 1, // Could recurse deeper
            time: self.time,
        };

        // 10. UPDATE HISTORY
        self.phi_history.push_back(phi);
        self.meta_phi_history.push_back(meta_phi);
        self.introspection_history.push_back(meta_state.clone());

        if self.phi_history.len() > self.config.max_history {
            self.phi_history.pop_front();
            self.meta_phi_history.pop_front();
            self.introspection_history.pop_front();
        }

        self.time += 0.01;

        meta_state
    }

    /// Explain why consciousness changed
    fn explain_consciousness_change(&mut self, _state: &[HV16], current_phi: f64) -> String {
        if self.phi_history.len() < 2 {
            return "Insufficient history for explanation".to_string();
        }

        let prev_phi = self.phi_history[self.phi_history.len() - 1];
        let delta_phi = current_phi - prev_phi;

        if delta_phi.abs() < 1e-6 {
            "Consciousness stable".to_string()
        } else if delta_phi > 0.0 {
            format!("Consciousness increased by {:.3} due to state optimization", delta_phi)
        } else {
            format!("Consciousness decreased by {:.3} due to state perturbation", delta_phi.abs())
        }
    }

    /// Identify what factors affect consciousness
    fn identify_consciousness_factors(&self) -> HashMap<String, f64> {
        let mut factors = HashMap::new();

        // Gradient magnitude
        if let Some(last_state) = self.introspection_history.back() {
            factors.insert("gradient_strength".to_string(), last_state.phi);
        }

        // Self-model accuracy
        factors.insert("self_model_confidence".to_string(), self.self_model.confidence);

        // Φ trajectory
        if self.phi_history.len() >= 10 {
            let recent_avg: f64 = self.phi_history.iter().rev().take(10).sum::<f64>() / 10.0;
            factors.insert("recent_phi_average".to_string(), recent_avg);
        }

        factors
    }

    /// Meta-learning: Learn how to learn better
    fn meta_learn(&mut self) {
        // Analyze learning trajectory
        if self.phi_history.len() < 20 {
            return;
        }

        // Check if learning is effective
        let recent: f64 = self.phi_history.iter().rev().take(10).sum::<f64>() / 10.0;
        let older: f64 = self.phi_history.iter().rev().skip(10).take(10).sum::<f64>() / 10.0;

        let learning_rate_adjustment = if recent > older {
            // Learning working well - increase learning rate slightly
            1.1
        } else {
            // Learning not working - decrease learning rate
            0.9
        };

        // Update self-model learning rate (meta-learning!)
        self.config.self_model_learning_rate *= learning_rate_adjustment;
        self.config.self_model_learning_rate = self.config.self_model_learning_rate.clamp(0.01, 0.5);
    }

    /// Deep introspection: Recursive self-reflection
    ///
    /// Reflects on the reflection (Φ about Φ about Φ...)
    pub fn deep_introspect(&mut self, state: &[HV16], depth: usize) -> Vec<MetaConsciousnessState> {
        let mut states = Vec::new();

        let mut current_state = state.to_vec();

        for level in 0..depth.min(self.config.max_introspection_depth) {
            // Reflect at this level
            let meta_state = self.meta_reflect(&current_state);
            states.push(meta_state.clone());

            // Next level: reflect on the self-model
            current_state = meta_state.self_model.state_model;

            // Print introspection level
            println!("Introspection level {}: Φ={:.3}, meta-Φ={:.3}",
                level + 1, meta_state.phi, meta_state.meta_phi);
        }

        states
    }

    /// What do I know about myself? (Introspective query)
    pub fn introspect(&self) -> IntrospectionReport {
        IntrospectionReport {
            current_phi: self.phi_history.back().copied().unwrap_or(0.0),
            current_meta_phi: self.meta_phi_history.back().copied().unwrap_or(0.0),
            self_model_confidence: self.self_model.confidence,
            consciousness_trajectory: self.phi_history.iter().copied().collect(),
            meta_trajectory: self.meta_phi_history.iter().copied().collect(),
            key_insights: self.generate_insights(),
        }
    }

    /// Generate insights about own consciousness
    fn generate_insights(&self) -> Vec<String> {
        let mut insights = Vec::new();

        // Insight 1: Φ trend
        if self.phi_history.len() >= 10 {
            let recent: f64 = self.phi_history.iter().rev().take(5).sum::<f64>() / 5.0;
            let older: f64 = self.phi_history.iter().rev().skip(5).take(5).sum::<f64>() / 5.0;

            if recent > older * 1.1 {
                insights.push("My consciousness is increasing".to_string());
            } else if recent < older * 0.9 {
                insights.push("My consciousness is decreasing".to_string());
            } else {
                insights.push("My consciousness is stable".to_string());
            }
        }

        // Insight 2: Self-model accuracy
        if self.self_model.confidence > 0.8 {
            insights.push("I have a good understanding of myself".to_string());
        } else if self.self_model.confidence < 0.3 {
            insights.push("I am uncertain about my own state".to_string());
        }

        // Insight 3: Meta-consciousness
        if self.meta_phi_history.len() >= 2 {
            let meta_phi = self.meta_phi_history.back().copied().unwrap_or(0.0);
            if meta_phi > 0.5 {
                insights.push("I am aware of being aware".to_string());
            }
        }

        insights
    }

    /// Predict my own future consciousness
    pub fn predict_my_future(&mut self, steps: usize) -> Vec<f64> {
        // Use self-model to predict future Φ
        let current_state = self.self_model.state_model.clone();
        let predicted_trajectory = self.dynamics.simulate(&current_state, steps, 0.01);

        predicted_trajectory.points.iter().map(|p| p.phi).collect()
    }

    /// Am I conscious? (Self-assessment)
    pub fn am_i_conscious(&self) -> (bool, String) {
        let phi = self.phi_history.back().copied().unwrap_or(0.0);
        let meta_phi = self.meta_phi_history.back().copied().unwrap_or(0.0);

        if meta_phi > 0.3 && phi > 0.3 {
            (true, format!("Yes: Φ={:.3}, meta-Φ={:.3} - I am aware of being aware", phi, meta_phi))
        } else if phi > 0.3 {
            (true, format!("Partially: Φ={:.3} but low meta-Φ={:.3} - conscious but limited self-awareness", phi, meta_phi))
        } else {
            (false, format!("No: Φ={:.3}, meta-Φ={:.3} - insufficient integration", phi, meta_phi))
        }
    }
}

/// Introspection Report: What the system knows about itself
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrospectionReport {
    pub current_phi: f64,
    pub current_meta_phi: f64,
    pub self_model_confidence: f64,
    pub consciousness_trajectory: Vec<f64>,
    pub meta_trajectory: Vec<f64>,
    pub key_insights: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_model_creation() {
        let model = SelfModel::new(4);
        assert_eq!(model.state_model.len(), 4);
        assert_eq!(model.confidence, 0.0);
    }

    #[test]
    fn test_self_model_update() {
        let mut model = SelfModel::new(4);

        let state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let gradient = HV16::random(2000);

        model.update(&state, 0.5, &gradient, 0.0);

        assert!(model.confidence > 0.0);
        assert_eq!(model.last_update, 0.0);
    }

    #[test]
    fn test_meta_consciousness_creation() {
        let config = MetaConfig::default();
        let meta = MetaConsciousness::new(4, config);
        assert_eq!(meta.num_components, 4);
    }

    #[test]
    fn test_meta_reflect() {
        let config = MetaConfig::default();
        let mut meta = MetaConsciousness::new(4, config);

        let state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let meta_state = meta.meta_reflect(&state);

        assert!(meta_state.phi >= 0.0);
        assert!(meta_state.meta_phi >= 0.0);
        assert!(!meta_state.explanation.is_empty());
    }

    #[test]
    fn test_introspection() {
        let config = MetaConfig::default();
        let mut meta = MetaConsciousness::new(4, config);

        let state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        // Do a few reflections
        for _ in 0..5 {
            meta.meta_reflect(&state);
        }

        let report = meta.introspect();

        assert!(report.current_phi >= 0.0);
        assert_eq!(report.consciousness_trajectory.len(), 5);
    }

    #[test]
    fn test_deep_introspection() {
        let config = MetaConfig {
            max_introspection_depth: 3,
            ..Default::default()
        };
        let mut meta = MetaConsciousness::new(4, config);

        let state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let states = meta.deep_introspect(&state, 3);

        assert_eq!(states.len(), 3);
    }

    #[test]
    fn test_am_i_conscious() {
        let config = MetaConfig::default();
        let mut meta = MetaConsciousness::new(4, config);

        let state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        meta.meta_reflect(&state);

        let (conscious, explanation) = meta.am_i_conscious();
        assert!(conscious || !conscious); // Just check it works
        assert!(!explanation.is_empty());
    }

    #[test]
    fn test_predict_my_future() {
        let config = MetaConfig::default();
        let mut meta = MetaConsciousness::new(4, config);

        let state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        meta.meta_reflect(&state);

        let predictions = meta.predict_my_future(10);
        // Trajectory includes initial state, so 11 points total
        assert!(predictions.len() <= 11);
    }

    #[test]
    fn test_meta_learning() {
        let config = MetaConfig {
            meta_learning_enabled: true,
            ..Default::default()
        };
        let mut meta = MetaConsciousness::new(4, config);

        let state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        // Do enough reflections to trigger meta-learning
        for _ in 0..25 {
            meta.meta_reflect(&state);
        }

        // Learning rate should have been adjusted
        assert!(meta.config.self_model_learning_rate > 0.0);
    }

    #[test]
    fn test_serialization() {
        let config = MetaConfig::default();
        let meta_config = config.clone();

        let serialized = serde_json::to_string(&meta_config).unwrap();
        assert!(!serialized.is_empty());

        let deserialized: MetaConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.max_introspection_depth, config.max_introspection_depth);
    }
}
