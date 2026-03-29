// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Consciousness-Integrated Learning Module

Bridges adaptive learning signals with Hebbian plasticity, creating a unified
learning system that respects consciousness state.

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                    CONSCIOUS LEARNING ENGINE                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌─────────────────┐        ┌──────────────────────┐                    │
│   │   Consciousness │───────▶│  AdaptiveLearning    │                    │
│   │   State (Φ)     │        │  Controller          │                    │
│   └─────────────────┘        └──────────┬───────────┘                    │
│                                         │                                 │
│   ┌─────────────────┐                   │  learning_rate_mod             │
│   │   Prediction    │───────▶           │  plasticity_gate               │
│   │   Error         │                   │  surprise_boost                │
│   └─────────────────┘                   ▼                                │
│                              ┌──────────────────────┐                    │
│   ┌─────────────────┐       │   HebbianEngine      │                    │
│   │   Neuromod.     │──────▶│   (Modulated)        │                    │
│   │   State         │       └──────────────────────┘                    │
│   └─────────────────┘                   │                                │
│                                         ▼                                │
│                              ┌──────────────────────┐                    │
│                              │   Learned Assocs.    │                    │
│                              └──────────────────────┘                    │
└──────────────────────────────────────────────────────────────────────────┘
```

## Usage

```rust,ignore
use symthaea::hdc::conscious_learning::ConsciousLearningEngine;

let mut engine = ConsciousLearningEngine::new();

// Update consciousness state
engine.observe_consciousness(phi, coherence, prediction_error);

// Learning happens with consciousness-modulated plasticity
engine.learn_association("dog", dog_vector, "animal", animal_vector);
```
*/

use super::adaptive_learning_signals::{
    AdaptiveLearningController, AdaptiveLearningSignal, NeuromodulatorLearningMap,
};
use super::hebbian::{HebbianConfig, HebbianEngine, HebbianStats};

// ============================================================================
// CONSCIOUS LEARNING ENGINE
// ============================================================================

/// Configuration for consciousness-integrated learning
#[derive(Debug, Clone)]
pub struct ConsciousLearningConfig {
    /// Hebbian learning configuration
    pub hebbian: HebbianConfig,

    /// Minimum Φ required for learning
    pub min_phi: f32,

    /// Minimum coherence required for learning
    pub min_coherence: f32,

    /// Enable consolidation during low-activity periods
    pub enable_consolidation: bool,

    /// Consolidation threshold (arousal below this triggers consolidation)
    pub consolidation_arousal_threshold: f32,
}

impl Default for ConsciousLearningConfig {
    fn default() -> Self {
        Self {
            hebbian: HebbianConfig::default(),
            min_phi: 0.1,
            min_coherence: 0.3,
            enable_consolidation: true,
            consolidation_arousal_threshold: 0.3,
        }
    }
}

/// Main conscious learning engine
pub struct ConsciousLearningEngine {
    /// Configuration
    config: ConsciousLearningConfig,

    /// Adaptive learning controller
    adaptive: AdaptiveLearningController,

    /// Hebbian learning engine
    hebbian: HebbianEngine,

    /// Current neuromodulator state
    neuromodulators: NeuromodulatorLearningMap,

    /// Current consciousness state
    current_phi: f32,
    current_coherence: f32,

    /// Whether learning is currently gated (blocked)
    learning_gated: bool,
    gate_reason: Option<String>,

    /// Pending associations to consolidate
    consolidation_buffer: Vec<PendingAssociation>,

    /// Statistics
    stats: ConsciousLearningStats,
}

/// Pending association waiting for consolidation
#[derive(Debug, Clone)]
struct PendingAssociation {
    pre_concept: String,
    post_concept: String,
    strength: f32,
    surprise: f32,
    phi_at_encoding: f32,
}

/// Statistics for conscious learning
#[derive(Debug, Clone, Default)]
pub struct ConsciousLearningStats {
    /// Total learning attempts
    pub total_attempts: u64,

    /// Learning attempts that were gated (blocked)
    pub gated_attempts: u64,

    /// Successful learning events
    pub successful_learns: u64,

    /// Consolidation events
    pub consolidation_events: u64,

    /// Average Φ during learning
    pub average_learning_phi: f32,

    /// Average effective learning rate
    pub average_effective_lr: f32,
}

impl ConsciousLearningEngine {
    /// Create new engine with default configuration
    pub fn new() -> Self {
        Self::with_config(ConsciousLearningConfig::default())
    }

    /// Create engine with custom configuration
    pub fn with_config(config: ConsciousLearningConfig) -> Self {
        Self {
            hebbian: HebbianEngine::with_config(config.hebbian.clone()),
            adaptive: AdaptiveLearningController::new(),
            neuromodulators: NeuromodulatorLearningMap::balanced(),
            current_phi: 0.5,
            current_coherence: 0.5,
            learning_gated: false,
            gate_reason: None,
            consolidation_buffer: Vec::new(),
            stats: ConsciousLearningStats::default(),
            config,
        }
    }

    /// Update consciousness state observations
    pub fn observe_consciousness(&mut self, phi: f32, coherence: f32, prediction_error: f32) {
        self.current_phi = phi;
        self.current_coherence = coherence;

        // Update adaptive controller
        let arousal = self.neuromodulators.arousal();
        let valence = self.neuromodulators.emotional_valence();

        let signal = self
            .adaptive
            .update(phi, coherence, prediction_error, arousal, valence);

        // Check if learning should be gated
        self.update_gate_status(&signal);

        // Check for consolidation conditions
        if self.config.enable_consolidation && signal.consolidation_ready {
            self.consolidate();
        }
    }

    /// Update neuromodulator state (from physiology)
    pub fn set_neuromodulators(&mut self, neuro: NeuromodulatorLearningMap) {
        self.neuromodulators = neuro;
    }

    /// Learn association between two concepts with consciousness modulation
    pub fn learn_association(
        &mut self,
        pre_concept: &str,
        pre_vector: &[f32],
        post_concept: &str,
        post_vector: &[f32],
        co_activation_strength: f32,
    ) -> LearningResult {
        self.stats.total_attempts += 1;

        // Check gate
        if self.learning_gated {
            self.stats.gated_attempts += 1;
            return LearningResult::Gated {
                reason: self.gate_reason.clone().unwrap_or_default(),
            };
        }

        // Get current signal for modulation - extract values before mutable ops
        let signal = self.adaptive.current_signal();
        let effective_multiplier = signal.effective_multiplier();
        let surprise_boost = signal.surprise_boost;
        // Signal reference is dropped after extracting needed values

        // Modulate the strength
        let modulated_strength = co_activation_strength * effective_multiplier;

        // If surprise is high, also buffer for consolidation
        if surprise_boost > 0.5 && self.config.enable_consolidation {
            self.consolidation_buffer.push(PendingAssociation {
                pre_concept: pre_concept.to_string(),
                post_concept: post_concept.to_string(),
                strength: modulated_strength,
                surprise: surprise_boost,
                phi_at_encoding: self.current_phi,
            });
        }

        // Record activation in Hebbian engine
        self.hebbian
            .record_activation(pre_concept, pre_vector, modulated_strength);

        // Short delay then activate post
        self.hebbian
            .record_activation(post_concept, post_vector, modulated_strength);

        // Explicit Hebbian update with modulated activity levels
        // The modulated_strength acts as the activity level for both concepts
        self.hebbian.hebbian_update(
            pre_concept,
            post_concept,
            modulated_strength, // pre_activity
            modulated_strength, // post_activity
        );

        // Update stats
        self.stats.successful_learns += 1;
        self.update_average_phi();
        self.update_average_lr(effective_multiplier);

        LearningResult::Learned {
            effective_strength: modulated_strength,
            phi: self.current_phi,
            surprise: surprise_boost, // Use pre-extracted value to avoid borrow conflict
        }
    }

    /// Trigger memory consolidation (replay and strengthen)
    pub fn consolidate(&mut self) {
        if self.consolidation_buffer.is_empty() {
            return;
        }

        self.stats.consolidation_events += 1;

        // Sort by surprise (consolidate most surprising first)
        self.consolidation_buffer.sort_by(|a, b| {
            b.surprise
                .partial_cmp(&a.surprise)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Replay and strengthen via Hebbian update
        for assoc in self.consolidation_buffer.drain(..) {
            // Consolidation: replay with reduced strength (simulating memory replay)
            let consolidation_strength = assoc.strength * 0.5;
            self.hebbian.hebbian_update(
                &assoc.pre_concept,
                &assoc.post_concept,
                consolidation_strength,
                consolidation_strength,
            );
        }
    }

    /// Get current learning signal
    pub fn current_signal(&self) -> &AdaptiveLearningSignal {
        self.adaptive.current_signal()
    }

    /// Check if learning is currently possible
    pub fn can_learn(&self) -> bool {
        !self.learning_gated
    }

    /// Get reason if learning is blocked
    pub fn gate_reason(&self) -> Option<&str> {
        self.gate_reason.as_deref()
    }

    /// Get learning statistics
    pub fn stats(&self) -> &ConsciousLearningStats {
        &self.stats
    }

    /// Get Hebbian engine statistics
    pub fn hebbian_stats(&self) -> HebbianStats {
        self.hebbian.stats()
    }

    /// Get the recommended batch size for learning
    pub fn recommended_batch_size(&self, base: usize) -> usize {
        self.adaptive.current_signal().recommended_batch_size(base)
    }

    // Internal: Update gate status
    fn update_gate_status(&mut self, signal: &AdaptiveLearningSignal) {
        // Gate if Φ too low
        if self.current_phi < self.config.min_phi {
            self.learning_gated = true;
            self.gate_reason = Some(format!(
                "Φ below threshold: {:.3} < {:.3}",
                self.current_phi, self.config.min_phi
            ));
            return;
        }

        // Gate if coherence too low
        if self.current_coherence < self.config.min_coherence {
            self.learning_gated = true;
            self.gate_reason = Some(format!(
                "Coherence below threshold: {:.3} < {:.3}",
                self.current_coherence, self.config.min_coherence
            ));
            return;
        }

        // Gate if signal says not to learn
        if !signal.should_learn() {
            self.learning_gated = true;
            self.gate_reason = Some("Adaptive signal: conditions unfavorable".to_string());
            return;
        }

        // All clear
        self.learning_gated = false;
        self.gate_reason = None;
    }

    // Internal: Update running average Φ
    fn update_average_phi(&mut self) {
        let n = self.stats.successful_learns as f32;
        self.stats.average_learning_phi =
            (self.stats.average_learning_phi * (n - 1.0) + self.current_phi) / n;
    }

    // Internal: Update running average learning rate
    fn update_average_lr(&mut self, lr: f32) {
        let n = self.stats.successful_learns as f32;
        self.stats.average_effective_lr = (self.stats.average_effective_lr * (n - 1.0) + lr) / n;
    }
}

impl Default for ConsciousLearningEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a learning attempt
#[derive(Debug, Clone)]
pub enum LearningResult {
    /// Learning succeeded
    Learned {
        effective_strength: f32,
        phi: f32,
        surprise: f32,
    },
    /// Learning was gated (blocked)
    Gated { reason: String },
}

impl LearningResult {
    /// Check if learning occurred
    pub fn learned(&self) -> bool {
        matches!(self, LearningResult::Learned { .. })
    }
}

// ============================================================================
// REAL-TIME SYSTEM INTEGRATION
// ============================================================================

/// Trait for real-time Φ sources (phi_engine integration)
pub trait PhiSource {
    /// Get current integrated information (Φ)
    fn current_phi(&self) -> f32;

    /// Get current coherence estimate
    fn current_coherence(&self) -> f32;
}

/// Trait for prediction error sources (active_inference integration)
pub trait PredictionErrorSource {
    /// Get current weighted prediction error (surprise)
    fn current_surprise(&self) -> f32;
}

impl ConsciousLearningEngine {
    /// Update from real-time Φ source (phi_engine integration)
    ///
    /// This method bridges the PhiCalculator's output to consciousness-gated learning.
    ///
    /// # Example (with PhiCalculator)
    /// ```ignore
    /// use symthaea::phi_engine::PhiCalculator;
    /// use symthaea::hdc::conscious_learning::ConsciousLearningEngine;
    ///
    /// let phi_result = phi_calculator.compute_from_hvs(&node_representations);
    /// engine.update_from_phi(phi_result.phi, phi_result.coherence_estimate, 0.0);
    /// ```
    pub fn update_from_phi(&mut self, phi: f32, coherence: f32, prediction_error: f32) {
        self.observe_consciousness(phi, coherence, prediction_error);
    }

    /// Update from HormoneState (endocrine system integration)
    ///
    /// Bridges the physiology/endocrine.rs HormoneState to learning modulation.
    ///
    /// # Example
    /// ```ignore
    /// use symthaea::physiology::endocrine::EndocrineSystem;
    /// use symthaea::hdc::conscious_learning::ConsciousLearningEngine;
    ///
    /// let hormone_state = endocrine_system.state();
    /// engine.update_from_hormones(
    ///     hormone_state.dopamine,
    ///     hormone_state.acetylcholine,
    ///     hormone_state.cortisol,
    /// );
    /// ```
    pub fn update_from_hormones(&mut self, dopamine: f32, acetylcholine: f32, cortisol: f32) {
        let neuro =
            NeuromodulatorLearningMap::from_hormone_state(dopamine, acetylcholine, cortisol);
        self.set_neuromodulators(neuro);
    }

    /// Update from PredictionError (active_inference integration)
    ///
    /// Routes the active inference prediction errors to the surprise_boost mechanism.
    ///
    /// # Example
    /// ```ignore
    /// use symthaea::brain::active_inference::PredictionError;
    /// use symthaea::hdc::conscious_learning::ConsciousLearningEngine;
    ///
    /// let pred_error = inference_engine.latest_error();
    /// engine.update_from_prediction_error(&pred_error);
    /// ```
    pub fn update_from_prediction_error_components(&mut self, weighted_error: f32) {
        // Update only the prediction error component, keeping phi and coherence
        self.observe_consciousness(
            self.current_phi,
            self.current_coherence,
            weighted_error.abs(),
        );
    }

    /// Full integration update from all systems
    ///
    /// Call this once per tick with outputs from phi_engine, endocrine, and active_inference.
    ///
    /// # Parameters
    /// - `phi`: Integrated information from PhiCalculator
    /// - `coherence`: Coherence estimate (can be from phi_result or coherence field)
    /// - `dopamine`, `acetylcholine`, `cortisol`: From HormoneState
    /// - `prediction_error`: From PredictionError.surprise() or weighted_error
    ///
    /// # Example
    /// ```ignore
    /// // In main tick/update loop:
    /// let phi_result = phi_calculator.compute_from_hvs(&nodes);
    /// let hormones = endocrine.state();
    /// let surprise = prediction_errors.iter().map(|e| e.surprise()).sum::<f32>() / n;
    ///
    /// engine.integrated_update(
    ///     phi_result.phi,
    ///     coherence_field.coherence(),
    ///     hormones.dopamine,
    ///     hormones.acetylcholine,
    ///     hormones.cortisol,
    ///     surprise,
    /// );
    /// ```
    pub fn integrated_update(
        &mut self,
        phi: f32,
        coherence: f32,
        dopamine: f32,
        acetylcholine: f32,
        cortisol: f32,
        prediction_error: f32,
    ) {
        // Update neuromodulators first (affects arousal/valence in observe_consciousness)
        self.update_from_hormones(dopamine, acetylcholine, cortisol);

        // Then update consciousness state with Φ and prediction error
        self.observe_consciousness(phi, coherence, prediction_error);
    }

    /// Get current consciousness metrics for monitoring
    pub fn consciousness_metrics(&self) -> ConsciousnessLearningMetrics {
        let signal = self.adaptive.current_signal();
        ConsciousnessLearningMetrics {
            phi: self.current_phi,
            coherence: self.current_coherence,
            learning_rate_mod: signal.learning_rate_mod,
            surprise_boost: signal.surprise_boost,
            plasticity_gate: signal.plasticity_gate,
            can_learn: !self.learning_gated,
            arousal: self.neuromodulators.arousal(),
            valence: self.neuromodulators.emotional_valence(),
        }
    }
}

/// Real-time consciousness metrics for monitoring/visualization
#[derive(Debug, Clone, Copy)]
pub struct ConsciousnessLearningMetrics {
    /// Current Φ (integrated information)
    pub phi: f32,
    /// Current coherence
    pub coherence: f32,
    /// Learning rate modulation factor
    pub learning_rate_mod: f32,
    /// Surprise boost factor
    pub surprise_boost: f32,
    /// Plasticity gate (0-1, consciousness threshold gating)
    pub plasticity_gate: f32,
    /// Whether learning is currently allowed
    pub can_learn: bool,
    /// Current arousal (from neuromodulators)
    pub arousal: f32,
    /// Current emotional valence
    pub valence: f32,
}

impl ConsciousnessLearningMetrics {
    /// Overall learning readiness (combines multiple factors)
    pub fn learning_readiness(&self) -> f32 {
        if !self.can_learn {
            return 0.0;
        }
        self.phi * self.coherence * self.plasticity_gate * self.learning_rate_mod
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conscious_learning_creation() {
        let engine = ConsciousLearningEngine::new();
        assert!(engine.can_learn());
    }

    #[test]
    fn test_low_phi_gates_learning() {
        let mut engine = ConsciousLearningEngine::new();

        // Very low Φ
        engine.observe_consciousness(0.05, 0.5, 0.1);

        assert!(!engine.can_learn());
        assert!(engine.gate_reason().unwrap().contains("Φ below"));
    }

    #[test]
    fn test_high_phi_allows_learning() {
        let mut engine = ConsciousLearningEngine::new();

        // Good Φ and coherence
        engine.observe_consciousness(0.5, 0.6, 0.1);

        assert!(engine.can_learn());
    }

    #[test]
    fn test_learning_with_modulation() {
        let mut engine = ConsciousLearningEngine::new();
        engine.observe_consciousness(0.6, 0.7, 0.1);

        let pre_vec = vec![0.1; 100];
        let post_vec = vec![0.2; 100];

        let result = engine.learn_association("a", &pre_vec, "b", &post_vec, 0.5);

        assert!(result.learned());
        assert_eq!(engine.stats().successful_learns, 1);
    }

    #[test]
    fn test_gated_learning_counted() {
        let mut engine = ConsciousLearningEngine::new();

        // Gate learning
        engine.observe_consciousness(0.01, 0.1, 0.1);

        let pre_vec = vec![0.1; 100];
        let post_vec = vec![0.2; 100];

        let result = engine.learn_association("a", &pre_vec, "b", &post_vec, 0.5);

        assert!(!result.learned());
        assert_eq!(engine.stats().gated_attempts, 1);
    }

    #[test]
    fn test_consolidation_buffer() {
        let mut engine = ConsciousLearningEngine::new();

        // High surprise should buffer
        engine.observe_consciousness(0.5, 0.7, 0.1);

        // Manually set high surprise via multiple large prediction errors
        for _ in 0..10 {
            engine.observe_consciousness(0.5, 0.7, 0.9);
        }

        let pre_vec = vec![0.1; 100];
        let post_vec = vec![0.2; 100];

        engine.learn_association("x", &pre_vec, "y", &post_vec, 0.5);

        // After processing, stats should reflect the learning attempt
        let stats = engine.stats();
        assert_eq!(
            stats.total_attempts, 1,
            "Should have recorded exactly 1 learning attempt"
        );
    }

    #[test]
    fn test_integrated_update() {
        let mut engine = ConsciousLearningEngine::new();

        // Simulate real-time integration from phi_engine + endocrine + active_inference
        engine.integrated_update(
            0.7, // phi from PhiCalculator
            0.8, // coherence from CoherenceField
            0.6, // dopamine from HormoneState
            0.5, // acetylcholine from HormoneState
            0.3, // cortisol from HormoneState
            0.2, // prediction_error from PredictionError.surprise()
        );

        // Should be able to learn with these good values
        assert!(engine.can_learn());

        // Check metrics
        let metrics = engine.consciousness_metrics();
        assert!((metrics.phi - 0.7).abs() < 0.01);
        assert!((metrics.coherence - 0.8).abs() < 0.01);
        assert!(metrics.learning_readiness() > 0.0);
    }

    #[test]
    fn test_update_from_hormones() {
        let mut engine = ConsciousLearningEngine::new();

        // High dopamine (reward signal) should enhance learning
        engine.update_from_hormones(0.9, 0.5, 0.1);
        engine.observe_consciousness(0.6, 0.7, 0.1);

        let metrics = engine.consciousness_metrics();
        // High dopamine + low cortisol → positive learning rate mod
        // The exact value depends on EMA smoothing and multiple factors
        assert!(
            metrics.learning_rate_mod > 0.0,
            "Expected positive learning_rate_mod, got {}",
            metrics.learning_rate_mod
        );
        assert!(
            metrics.can_learn,
            "Should be able to learn with good consciousness state"
        );
    }

    #[test]
    fn test_prediction_error_routing() {
        let mut engine = ConsciousLearningEngine::new();

        // Start with low surprise
        engine.observe_consciousness(0.6, 0.7, 0.1);
        let low_surprise = engine.consciousness_metrics().surprise_boost;

        // Update with high prediction error (simulating PredictionError.surprise())
        engine.update_from_prediction_error_components(0.9);
        let high_surprise = engine.consciousness_metrics().surprise_boost;

        // High prediction error should increase surprise boost
        assert!(high_surprise >= low_surprise);
    }
}
