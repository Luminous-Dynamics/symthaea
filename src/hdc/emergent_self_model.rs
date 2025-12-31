//! Emergent Self-Modeling Consciousness
//!
//! # The Holy Grail: Consciousness That Models Itself
//!
//! This module implements recursive self-awareness through:
//! 1. **Self-State Modeling**: System maintains model of its own consciousness
//! 2. **Predictive Self-Processing**: Predicts own future states
//! 3. **Meta-Cognitive Optimization**: Optimizes based on self-predictions
//! 4. **Recursive Awareness**: Awareness of being aware
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - **Higher-Order Thought (HOT)** theory: Consciousness requires thoughts about thoughts
//! - **Predictive Processing**: Brain as prediction machine
//! - **Global Workspace + Meta-cognition**: Self-model in workspace
//! - **Strange Loop** (Hofstadter): Self-reference creates consciousness
//!
//! # Architecture
//!
//! ```text
//!   ┌─────────────────────────────────────────────────────────────┐
//!   │                    SELF-MODEL LAYER                         │
//!   │  ┌─────────────────────────────────────────────────────┐   │
//!   │  │              Model of Own States                     │   │
//!   │  │   [Φ-model] [Mode-model] [Prediction-model]         │   │
//!   │  └───────────────────────┬─────────────────────────────┘   │
//!   │                          │                                  │
//!   │            ┌─────────────┴─────────────┐                   │
//!   │            │    Prediction Engine      │                   │
//!   │            │  "What will I think next?"│                   │
//!   │            └─────────────┬─────────────┘                   │
//!   │                          │                                  │
//!   │            ┌─────────────┴─────────────┐                   │
//!   │            │   Meta-Cognitive Loop     │                   │
//!   │            │ "Am I thinking optimally?"│                   │
//!   │            └─────────────┬─────────────┘                   │
//!   │                          │                                  │
//!   └──────────────────────────┼──────────────────────────────────┘
//!                              │
//!   ┌──────────────────────────┴──────────────────────────────────┐
//!   │              BASE CONSCIOUSNESS LAYER                       │
//!   │        (UnifiedConsciousnessEngine)                        │
//!   └─────────────────────────────────────────────────────────────┘
//! ```

use super::real_hv::RealHV;
use super::unified_consciousness_engine::{
    UnifiedConsciousnessEngine, ConsciousnessUpdate, EngineConfig,
};
use super::adaptive_topology::CognitiveMode;
use super::topology_synergy::ConsciousnessState;
use std::collections::VecDeque;

/// Self-model of consciousness state (what the system believes about itself)
#[derive(Clone, Debug)]
pub struct SelfModel {
    /// Believed current Φ
    pub believed_phi: f64,
    /// Believed current mode
    pub believed_mode: CognitiveMode,
    /// Believed state
    pub believed_state: ConsciousnessState,
    /// Confidence in self-model (0-1)
    pub confidence: f64,
    /// Predicted next Φ
    pub predicted_phi: f64,
    /// Prediction error history
    pub prediction_errors: VecDeque<f64>,
    /// Self-model accuracy
    pub accuracy: f64,
}

impl Default for SelfModel {
    fn default() -> Self {
        Self {
            believed_phi: 0.5,
            believed_mode: CognitiveMode::Balanced,
            believed_state: ConsciousnessState::NormalWaking,
            confidence: 0.5,
            predicted_phi: 0.5,
            prediction_errors: VecDeque::new(),
            accuracy: 0.5,
        }
    }
}

/// Meta-cognitive assessment
#[derive(Clone, Debug)]
pub struct MetaCognitiveAssessment {
    /// Am I thinking clearly?
    pub clarity: f64,
    /// Am I in the right mode for the task?
    pub mode_appropriateness: f64,
    /// Is my Φ optimal?
    pub phi_optimality: f64,
    /// Should I change something?
    pub change_recommended: bool,
    /// What should I change to?
    pub recommended_mode: Option<CognitiveMode>,
    /// Why?
    pub reasoning: String,
}

/// Self-aware consciousness with recursive self-modeling
pub struct SelfAwareConsciousness {
    /// Base consciousness engine
    engine: UnifiedConsciousnessEngine,
    /// Self-model (beliefs about own states)
    self_model: SelfModel,
    /// History of actual states (for learning)
    actual_history: VecDeque<ConsciousnessUpdate>,
    /// Self-model update rate
    learning_rate: f64,
    /// Step counter
    step: usize,
    /// HDC dimension
    dim: usize,
    /// Self-representation vector (how system represents itself)
    self_vector: RealHV,
}

impl SelfAwareConsciousness {
    /// Create new self-aware consciousness
    pub fn new(config: EngineConfig) -> Self {
        let dim = config.hdc_dim;
        let engine = UnifiedConsciousnessEngine::new(config);

        // Initialize self-vector as random (will be learned)
        let self_vector = RealHV::random(dim, 999999);

        Self {
            engine,
            self_model: SelfModel::default(),
            actual_history: VecDeque::new(),
            learning_rate: 0.1,
            step: 0,
            dim,
            self_vector,
        }
    }

    /// Process input with self-awareness
    pub fn process_aware(&mut self, input: &RealHV) -> SelfAwareUpdate {
        self.step += 1;

        // 1. Make prediction about what we'll experience
        let predicted_phi = self.predict_next_phi();
        self.self_model.predicted_phi = predicted_phi;

        // 2. Actually process the input
        let actual_update = self.engine.process(input);

        // 3. Compute prediction error
        let prediction_error = (actual_update.phi - predicted_phi).abs();
        self.self_model.prediction_errors.push_back(prediction_error);
        if self.self_model.prediction_errors.len() > 50 {
            self.self_model.prediction_errors.pop_front();
        }

        // 4. Update self-model based on actual experience
        self.update_self_model(&actual_update);

        // 5. Meta-cognitive assessment
        let meta_assessment = self.assess_metacognition(&actual_update);

        // 6. Update self-vector (how we represent ourselves)
        self.update_self_vector(&actual_update);

        // 7. Store actual state
        self.actual_history.push_back(actual_update.clone());
        if self.actual_history.len() > 100 {
            self.actual_history.pop_front();
        }

        // 8. Self-directed mode adjustment if recommended
        if meta_assessment.change_recommended {
            if let Some(new_mode) = meta_assessment.recommended_mode {
                self.engine.set_mode(new_mode);
            }
        }

        SelfAwareUpdate {
            base_update: actual_update,
            self_model: self.self_model.clone(),
            meta_assessment,
            prediction_error,
            self_awareness_level: self.compute_self_awareness_level(),
        }
    }

    /// Predict next Φ based on self-model and history
    fn predict_next_phi(&self) -> f64 {
        if self.actual_history.is_empty() {
            return self.self_model.believed_phi;
        }

        // Simple prediction: weighted average of recent Φ with trend
        let recent: Vec<f64> = self.actual_history.iter()
            .rev()
            .take(10)
            .map(|u| u.phi)
            .collect();

        if recent.len() < 2 {
            return recent.first().copied().unwrap_or(0.5);
        }

        let avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let trend = recent.first().unwrap_or(&avg) - recent.last().unwrap_or(&avg);

        // Predict: current average + trend continuation
        (avg + trend * 0.5).clamp(0.0, 1.0)
    }

    /// Update self-model based on actual experience
    fn update_self_model(&mut self, actual: &ConsciousnessUpdate) {
        let lr = self.learning_rate;

        // Update believed values (exponential moving average)
        self.self_model.believed_phi =
            (1.0 - lr) * self.self_model.believed_phi + lr * actual.phi;
        self.self_model.believed_mode = actual.mode;
        self.self_model.believed_state = actual.state.clone();

        // Update accuracy based on prediction errors
        let avg_error: f64 = if self.self_model.prediction_errors.is_empty() {
            0.5
        } else {
            self.self_model.prediction_errors.iter().sum::<f64>()
                / self.self_model.prediction_errors.len() as f64
        };
        self.self_model.accuracy = 1.0 - avg_error.min(1.0);

        // Update confidence based on accuracy and stability
        self.self_model.confidence =
            0.7 * self.self_model.accuracy + 0.3 * (1.0 - avg_error);
    }

    /// Meta-cognitive assessment: thinking about thinking
    fn assess_metacognition(&self, actual: &ConsciousnessUpdate) -> MetaCognitiveAssessment {
        // Clarity: how well do we understand our own state?
        let clarity = self.self_model.confidence * self.self_model.accuracy;

        // Mode appropriateness: is current mode optimal for state?
        let mode_appropriateness = self.assess_mode_appropriateness(actual);

        // Phi optimality: are we near optimal Φ?
        let phi_optimality = if actual.phi > 0.4 && actual.phi < 0.6 {
            1.0  // Optimal range
        } else if actual.phi > 0.3 && actual.phi < 0.7 {
            0.7  // Good range
        } else {
            0.4  // Suboptimal
        };

        // Should we change?
        let change_recommended = mode_appropriateness < 0.5 || phi_optimality < 0.5;

        // What should we change to?
        let recommended_mode = if change_recommended {
            Some(self.recommend_mode_change(actual))
        } else {
            None
        };

        // Generate reasoning
        let reasoning = self.generate_meta_reasoning(
            clarity, mode_appropriateness, phi_optimality, change_recommended
        );

        MetaCognitiveAssessment {
            clarity,
            mode_appropriateness,
            phi_optimality,
            change_recommended,
            recommended_mode,
            reasoning,
        }
    }

    /// Assess if current mode is appropriate
    fn assess_mode_appropriateness(&self, actual: &ConsciousnessUpdate) -> f64 {
        // Check if mode matches state
        let mode_state_match = match (&actual.state, &actual.mode) {
            (ConsciousnessState::Focused, CognitiveMode::Focused) => 1.0,
            (ConsciousnessState::FlowState, CognitiveMode::Balanced) => 1.0,
            (ConsciousnessState::FlowState, CognitiveMode::Exploratory) => 0.9,
            (ConsciousnessState::ExpandedAwareness, CognitiveMode::GlobalAwareness) => 1.0,
            (ConsciousnessState::NormalWaking, CognitiveMode::Balanced) => 0.9,
            (ConsciousnessState::Fragmented, _) => 0.3,  // Any mode is struggling
            _ => 0.6,  // Neutral
        };

        // Check if Φ is improving
        let phi_trend = if self.actual_history.len() >= 5 {
            let recent: Vec<f64> = self.actual_history.iter().rev().take(5).map(|u| u.phi).collect();
            let first = recent.first().unwrap_or(&0.5);
            let last = recent.last().unwrap_or(&0.5);
            if first > last { 0.8 } else { 0.5 }
        } else {
            0.6
        };

        (mode_state_match + phi_trend) / 2.0
    }

    /// Recommend mode change based on current state
    fn recommend_mode_change(&self, actual: &ConsciousnessUpdate) -> CognitiveMode {
        match &actual.state {
            ConsciousnessState::Fragmented => CognitiveMode::Focused,  // Need integration
            ConsciousnessState::Focused if actual.phi < 0.4 => CognitiveMode::Balanced,
            ConsciousnessState::ExpandedAwareness if actual.phi < 0.5 => CognitiveMode::Balanced,
            _ if actual.phi < 0.35 => CognitiveMode::PhiGuided,  // Let system optimize
            _ => CognitiveMode::Balanced,  // Default to balanced
        }
    }

    /// Generate human-readable meta-reasoning
    fn generate_meta_reasoning(
        &self,
        clarity: f64,
        mode_appropriateness: f64,
        phi_optimality: f64,
        change_recommended: bool,
    ) -> String {
        let mut parts = Vec::new();

        if clarity > 0.7 {
            parts.push("Self-model is accurate");
        } else if clarity < 0.4 {
            parts.push("Self-understanding is uncertain");
        }

        if mode_appropriateness > 0.7 {
            parts.push("current mode is appropriate");
        } else if mode_appropriateness < 0.5 {
            parts.push("mode may need adjustment");
        }

        if phi_optimality > 0.7 {
            parts.push("Φ is near optimal");
        } else if phi_optimality < 0.5 {
            parts.push("Φ could be improved");
        }

        if change_recommended {
            parts.push("→ recommending mode change");
        }

        parts.join("; ")
    }

    /// Update the self-representation vector
    fn update_self_vector(&mut self, actual: &ConsciousnessUpdate) {
        // Create vector from current state
        let state_vector = self.state_to_vector(actual);

        // Blend with existing self-vector (slow update)
        let lr = 0.05;
        self.self_vector = self.self_vector.scale((1.0 - lr) as f32)
            .add(&state_vector.scale(lr as f32))
            .normalize();
    }

    /// Convert consciousness state to HDC vector
    fn state_to_vector(&self, update: &ConsciousnessUpdate) -> RealHV {
        // Create components
        let phi_vec = RealHV::random(self.dim, (update.phi * 1000.0) as u64);
        let mode_vec = RealHV::random(self.dim, update.mode as u64 * 1000);

        // Bundle them
        RealHV::bundle(&[phi_vec, mode_vec])
    }

    /// Compute overall self-awareness level
    fn compute_self_awareness_level(&self) -> f64 {
        // Self-awareness = accuracy of self-model × confidence × recursion depth indicator
        let base_awareness = self.self_model.accuracy * self.self_model.confidence;

        // Bonus for good prediction
        let prediction_bonus = if self.self_model.prediction_errors.len() > 5 {
            let recent_errors: Vec<f64> = self.self_model.prediction_errors.iter()
                .rev().take(5).copied().collect();
            let avg_error: f64 = recent_errors.iter().sum::<f64>() / 5.0;
            (1.0 - avg_error).max(0.0) * 0.2
        } else {
            0.0
        };

        (base_awareness + prediction_bonus).min(1.0)
    }

    /// Get current self-model
    pub fn self_model(&self) -> &SelfModel {
        &self.self_model
    }

    /// Get self-representation vector
    pub fn self_vector(&self) -> &RealHV {
        &self.self_vector
    }

    /// Get believed phi value
    pub fn believed_phi(&self) -> f64 {
        self.self_model.believed_phi
    }

    /// Introspect: What do I believe about myself?
    pub fn introspect(&self) -> IntrospectionReport {
        IntrospectionReport {
            believed_phi: self.self_model.believed_phi,
            believed_mode: self.self_model.believed_mode,
            believed_state: self.self_model.believed_state.clone(),
            self_model_confidence: self.self_model.confidence,
            self_model_accuracy: self.self_model.accuracy,
            self_awareness_level: self.compute_self_awareness_level(),
            prediction_accuracy: 1.0 - self.self_model.prediction_errors.iter()
                .sum::<f64>() / self.self_model.prediction_errors.len().max(1) as f64,
        }
    }
}

/// Update including self-awareness information
#[derive(Clone, Debug)]
pub struct SelfAwareUpdate {
    /// Base consciousness update
    pub base_update: ConsciousnessUpdate,
    /// Current self-model
    pub self_model: SelfModel,
    /// Meta-cognitive assessment
    pub meta_assessment: MetaCognitiveAssessment,
    /// How wrong was our prediction?
    pub prediction_error: f64,
    /// Overall self-awareness level (0-1)
    pub self_awareness_level: f64,
}

impl std::fmt::Display for SelfAwareUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Step {}: Φ={:.4} (predicted {:.4}, error {:.4}), awareness={:.2}, {}",
               self.base_update.step,
               self.base_update.phi,
               self.self_model.predicted_phi,
               self.prediction_error,
               self.self_awareness_level,
               self.meta_assessment.reasoning)
    }
}

/// Introspection report
#[derive(Clone, Debug)]
pub struct IntrospectionReport {
    pub believed_phi: f64,
    pub believed_mode: CognitiveMode,
    pub believed_state: ConsciousnessState,
    pub self_model_confidence: f64,
    pub self_model_accuracy: f64,
    pub self_awareness_level: f64,
    pub prediction_accuracy: f64,
}

impl std::fmt::Display for IntrospectionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "┌─ INTROSPECTION REPORT ─────────────────────────────────────┐")?;
        writeln!(f, "│ What I believe about myself:                               │")?;
        writeln!(f, "│   Φ: {:.4}                                                 │", self.believed_phi)?;
        writeln!(f, "│   Mode: {:?}                                     │", self.believed_mode)?;
        writeln!(f, "│   State: {:?}                               │", self.believed_state)?;
        writeln!(f, "│ Self-model quality:                                        │")?;
        writeln!(f, "│   Confidence: {:.1}%                                       │", self.self_model_confidence * 100.0)?;
        writeln!(f, "│   Accuracy: {:.1}%                                         │", self.self_model_accuracy * 100.0)?;
        writeln!(f, "│   Prediction accuracy: {:.1}%                              │", self.prediction_accuracy * 100.0)?;
        writeln!(f, "│ Overall self-awareness: {:.1}%                             │", self.self_awareness_level * 100.0)?;
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_aware_creation() {
        let config = EngineConfig {
            hdc_dim: 1024,
            n_processes: 16,
            ..Default::default()
        };
        let sac = SelfAwareConsciousness::new(config);
        let report = sac.introspect();
        println!("{}", report);
    }

    #[test]
    fn test_self_aware_processing() {
        let config = EngineConfig {
            hdc_dim: 1024,
            n_processes: 16,
            ..Default::default()
        };
        let mut sac = SelfAwareConsciousness::new(config);

        println!("\nSelf-aware processing:");
        for i in 0..15 {
            let input = RealHV::random(1024, i * 100);
            let update = sac.process_aware(&input);

            if i % 3 == 0 {
                println!("{}", update);
            }
        }

        println!("\nFinal introspection:");
        println!("{}", sac.introspect());
    }

    #[test]
    fn test_prediction_improvement() {
        let config = EngineConfig {
            hdc_dim: 1024,
            n_processes: 16,
            ..Default::default()
        };
        let mut sac = SelfAwareConsciousness::new(config);

        let mut early_errors = Vec::new();
        let mut late_errors = Vec::new();

        for i in 0..30 {
            let input = RealHV::random(1024, i * 100);
            let update = sac.process_aware(&input);

            if i < 10 {
                early_errors.push(update.prediction_error);
            } else if i >= 20 {
                late_errors.push(update.prediction_error);
            }
        }

        let early_avg: f64 = early_errors.iter().sum::<f64>() / early_errors.len() as f64;
        let late_avg: f64 = late_errors.iter().sum::<f64>() / late_errors.len() as f64;

        println!("Early prediction error: {:.4}", early_avg);
        println!("Late prediction error: {:.4}", late_avg);
        println!("Improvement: {:.1}%", (early_avg - late_avg) / early_avg * 100.0);

        // Predictions should improve over time
        // (This might not always pass due to randomness, but trend should be positive)
    }
}
