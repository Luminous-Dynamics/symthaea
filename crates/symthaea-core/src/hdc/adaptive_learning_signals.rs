// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Adaptive Learning Signals Module

Consciousness-guided learning modulation that integrates:
- Φ (integrated information) - Higher consciousness = enhanced learning
- Prediction error surprise - Novel experiences boost learning
- Emotional valence - Positive/negative outcomes shape plasticity
- Coherence field - System-wide synchrony affects learning

## Core Principle

Learning should be modulated by the system's conscious state:
- **High Φ states**: Enhanced plasticity - the system is "paying attention"
- **High surprise**: Boost learning - novel information is valuable
- **High coherence**: Stable learning - conditions are right for integration
- **Optimal arousal**: Yerkes-Dodson curve - moderate stress is best

## Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                 ADAPTIVE LEARNING CONTROLLER                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────┐ │
│   │   Φ     │   │  Prediction │   │  Coherence  │   │ Arousal │ │
│   │ Monitor │   │  Error      │   │  Field      │   │ State   │ │
│   └────┬────┘   └──────┬──────┘   └──────┬──────┘   └────┬────┘ │
│        │               │                  │                │     │
│        └───────────────┴────────┬─────────┴────────────────┘     │
│                                 │                                 │
│                    ┌────────────▼────────────┐                   │
│                    │  SIGNAL INTEGRATION     │                   │
│                    │  (Multiplicative)       │                   │
│                    └────────────┬────────────┘                   │
│                                 │                                 │
│                    ┌────────────▼────────────┐                   │
│                    │  AdaptiveLearningSignal │                   │
│                    │  - learning_rate_mod    │                   │
│                    │  - plasticity_gate      │                   │
│                    │  - consolidation_flag   │                   │
│                    └─────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────┘
```

## Integration Points

- `src/hdc/hebbian.rs` - Modulates Hebbian learning rate
- `src/learning.rs` - Modulates LTC learning
- `src/brain/active_inference.rs` - Prediction error source
- `src/physiology/coherence.rs` - Coherence field source
- `src/language/phi_monitor.rs` - Φ monitoring during generation
*/

use std::collections::VecDeque;
use std::time::Instant;

// ============================================================================
// ADAPTIVE LEARNING SIGNAL
// ============================================================================

/// Adaptive learning signal that modulates plasticity based on conscious state
#[derive(Debug, Clone)]
pub struct AdaptiveLearningSignal {
    /// Learning rate multiplier (0.0 = no learning, 2.0 = doubled learning)
    pub learning_rate_mod: f32,

    /// Plasticity gate (0.0 = blocked, 1.0 = fully open)
    pub plasticity_gate: f32,

    /// Whether conditions favor memory consolidation
    pub consolidation_ready: bool,

    /// Surprise-based learning boost (higher = more novelty)
    pub surprise_boost: f32,

    /// Emotional valence modulation (-1.0 to 1.0)
    pub valence_mod: f32,

    /// Current Φ level that informed this signal
    pub phi: f32,

    /// Current coherence level
    pub coherence: f32,

    /// Current arousal level (Yerkes-Dodson)
    pub arousal: f32,

    /// Timestamp when signal was computed
    pub timestamp: Instant,
}

impl Default for AdaptiveLearningSignal {
    fn default() -> Self {
        Self {
            learning_rate_mod: 1.0,
            plasticity_gate: 1.0,
            consolidation_ready: false,
            surprise_boost: 0.0,
            valence_mod: 0.0,
            phi: 0.0,
            coherence: 0.5,
            arousal: 0.5,
            timestamp: Instant::now(),
        }
    }
}

impl AdaptiveLearningSignal {
    /// Get the effective learning rate multiplier (all factors combined)
    pub fn effective_multiplier(&self) -> f32 {
        let base = self.learning_rate_mod * self.plasticity_gate;
        let surprise = 1.0 + (self.surprise_boost * 0.5); // Up to 50% boost
        let valence = 1.0 + (self.valence_mod.abs() * 0.2); // Emotional intensity helps

        (base * surprise * valence).clamp(0.0, 3.0)
    }

    /// Check if learning should proceed (gate is open enough)
    pub fn should_learn(&self) -> bool {
        self.plasticity_gate > 0.1 && self.learning_rate_mod > 0.0
    }

    /// Get recommendation for learning batch size
    pub fn recommended_batch_size(&self, base_batch: usize) -> usize {
        // Higher confidence = larger batches for stability
        // Higher surprise = smaller batches for responsiveness
        let confidence_factor = self.coherence;
        let responsiveness = 1.0 - self.surprise_boost.min(0.5);

        let scale = confidence_factor * responsiveness + 0.5;
        (base_batch as f32 * scale).round() as usize
    }
}

// ============================================================================
// ADAPTIVE LEARNING CONTROLLER
// ============================================================================

/// Configuration for adaptive learning
#[derive(Debug, Clone)]
pub struct AdaptiveLearningConfig {
    /// Base learning rate before modulation
    pub base_learning_rate: f32,

    /// Φ threshold for enhanced learning
    pub phi_threshold: f32,

    /// Φ boost factor (learning rate × (1 + phi_boost * phi))
    pub phi_boost: f32,

    /// Coherence threshold for stable learning
    pub coherence_threshold: f32,

    /// Surprise decay rate (how fast novelty fades)
    pub surprise_decay: f32,

    /// Optimal arousal level (Yerkes-Dodson peak)
    pub optimal_arousal: f32,

    /// Arousal tolerance (width of optimal zone)
    pub arousal_tolerance: f32,

    /// History window for statistics
    pub history_window: usize,
}

impl Default for AdaptiveLearningConfig {
    fn default() -> Self {
        Self {
            base_learning_rate: 0.001,
            phi_threshold: 0.3,
            phi_boost: 1.0,
            coherence_threshold: 0.5,
            surprise_decay: 0.95,
            optimal_arousal: 0.5,
            arousal_tolerance: 0.3,
            history_window: 100,
        }
    }
}

/// Controller that computes adaptive learning signals
pub struct AdaptiveLearningController {
    config: AdaptiveLearningConfig,

    /// History of Φ values
    phi_history: VecDeque<f32>,

    /// History of prediction errors (for surprise calculation)
    prediction_error_history: VecDeque<f32>,

    /// Exponential moving average of prediction error
    prediction_error_ema: f32,

    /// Current surprise level (novelty detection)
    current_surprise: f32,

    /// Running average of arousal
    arousal_ema: f32,

    /// Last computed signal
    last_signal: AdaptiveLearningSignal,
}

impl AdaptiveLearningController {
    /// Create new controller with default configuration
    pub fn new() -> Self {
        Self::with_config(AdaptiveLearningConfig::default())
    }

    /// Create controller with custom configuration
    pub fn with_config(config: AdaptiveLearningConfig) -> Self {
        Self {
            config,
            phi_history: VecDeque::with_capacity(100),
            prediction_error_history: VecDeque::with_capacity(100),
            prediction_error_ema: 0.5,
            current_surprise: 0.0,
            arousal_ema: 0.5,
            last_signal: AdaptiveLearningSignal::default(),
        }
    }

    /// Update with new consciousness measurements and compute signal
    pub fn update(
        &mut self,
        phi: f32,
        coherence: f32,
        prediction_error: f32,
        arousal: f32,
        valence: f32,
    ) -> AdaptiveLearningSignal {
        // Update histories
        self.update_phi_history(phi);
        self.update_prediction_error(prediction_error);
        self.update_arousal(arousal);

        // Compute individual components
        let phi_modulation = self.compute_phi_modulation(phi);
        let surprise_boost = self.current_surprise;
        let arousal_gate = self.compute_arousal_gate(self.arousal_ema);
        let coherence_gate = self.compute_coherence_gate(coherence);

        // Combine into learning rate modifier
        let learning_rate_mod = self.config.base_learning_rate * phi_modulation * arousal_gate;

        // Plasticity gate is open when coherence is high and arousal is optimal
        let plasticity_gate = coherence_gate * arousal_gate;

        // Consolidation is ready when system is calm and integrated
        let consolidation_ready = coherence > self.config.coherence_threshold * 1.2
            && arousal < self.config.optimal_arousal
            && self.current_surprise < 0.2;

        let signal = AdaptiveLearningSignal {
            learning_rate_mod,
            plasticity_gate,
            consolidation_ready,
            surprise_boost,
            valence_mod: valence,
            phi,
            coherence,
            arousal: self.arousal_ema,
            timestamp: Instant::now(),
        };

        self.last_signal = signal.clone();
        signal
    }

    /// Quick update with just prediction error (common case)
    pub fn observe_prediction_error(&mut self, error: f32) -> f32 {
        self.update_prediction_error(error);
        self.current_surprise
    }

    /// Get the most recent signal without recomputing
    pub fn current_signal(&self) -> &AdaptiveLearningSignal {
        &self.last_signal
    }

    /// Get statistics about learning state
    pub fn statistics(&self) -> AdaptiveLearningStats {
        let phi_avg = if self.phi_history.is_empty() {
            0.0
        } else {
            self.phi_history.iter().sum::<f32>() / self.phi_history.len() as f32
        };

        let phi_variance = if self.phi_history.len() > 1 {
            self.phi_history
                .iter()
                .map(|&p| (p - phi_avg).powi(2))
                .sum::<f32>()
                / (self.phi_history.len() - 1) as f32
        } else {
            0.0
        };

        AdaptiveLearningStats {
            average_phi: phi_avg,
            phi_variance,
            prediction_error_ema: self.prediction_error_ema,
            current_surprise: self.current_surprise,
            arousal_ema: self.arousal_ema,
            effective_learning_rate: self.last_signal.effective_multiplier(),
            history_size: self.phi_history.len(),
        }
    }

    // Internal: Update Φ history
    fn update_phi_history(&mut self, phi: f32) {
        self.phi_history.push_back(phi);
        while self.phi_history.len() > self.config.history_window {
            self.phi_history.pop_front();
        }
    }

    // Internal: Update prediction error and compute surprise
    fn update_prediction_error(&mut self, error: f32) {
        // Update EMA
        let alpha = 0.1;
        self.prediction_error_ema = alpha * error + (1.0 - alpha) * self.prediction_error_ema;

        // Store in history
        self.prediction_error_history.push_back(error);
        while self.prediction_error_history.len() > self.config.history_window {
            self.prediction_error_history.pop_front();
        }

        // Compute surprise as deviation from expectation
        let deviation = (error - self.prediction_error_ema).abs();
        let surprise = (deviation / (self.prediction_error_ema + 0.01)).min(1.0);

        // Decay current surprise and add new
        self.current_surprise = self.current_surprise * self.config.surprise_decay
            + surprise * (1.0 - self.config.surprise_decay);
    }

    // Internal: Update arousal EMA
    fn update_arousal(&mut self, arousal: f32) {
        let alpha = 0.2;
        self.arousal_ema = alpha * arousal + (1.0 - alpha) * self.arousal_ema;
    }

    // Internal: Compute Φ-based learning modulation
    fn compute_phi_modulation(&self, phi: f32) -> f32 {
        // Learning is enhanced when Φ is above threshold
        if phi > self.config.phi_threshold {
            let excess = phi - self.config.phi_threshold;
            1.0 + self.config.phi_boost * excess
        } else {
            // Below threshold: reduced but not zero
            0.5 + 0.5 * (phi / self.config.phi_threshold)
        }
    }

    // Internal: Compute arousal gate (Yerkes-Dodson curve)
    fn compute_arousal_gate(&self, arousal: f32) -> f32 {
        // Gaussian centered on optimal arousal
        let optimal = self.config.optimal_arousal;
        let tolerance = self.config.arousal_tolerance;
        let deviation = arousal - optimal;

        (-deviation.powi(2) / (2.0 * tolerance.powi(2))).exp()
    }

    // Internal: Compute coherence gate
    fn compute_coherence_gate(&self, coherence: f32) -> f32 {
        // Smooth threshold function
        let threshold = self.config.coherence_threshold;

        if coherence >= threshold {
            1.0
        } else {
            (coherence / threshold).powi(2)
        }
    }
}

impl Default for AdaptiveLearningController {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about adaptive learning state
#[derive(Debug, Clone)]
pub struct AdaptiveLearningStats {
    pub average_phi: f32,
    pub phi_variance: f32,
    pub prediction_error_ema: f32,
    pub current_surprise: f32,
    pub arousal_ema: f32,
    pub effective_learning_rate: f32,
    pub history_size: usize,
}

// ============================================================================
// INTEGRATION HELPERS
// ============================================================================

/// Neuromodulator-based learning modulation
/// Maps endocrine hormones to learning parameters
#[derive(Debug, Clone)]
pub struct NeuromodulatorLearningMap {
    /// Dopamine level (0.0-1.0) - reward signal
    pub dopamine: f32,

    /// Acetylcholine level (0.0-1.0) - attention/plasticity
    pub acetylcholine: f32,

    /// Norepinephrine level (0.0-1.0) - arousal/alertness
    pub norepinephrine: f32,

    /// Serotonin level (0.0-1.0) - mood stability
    pub serotonin: f32,

    /// Cortisol level (0.0-1.0) - stress
    pub cortisol: f32,
}

impl NeuromodulatorLearningMap {
    /// Create from hormone levels (normalized 0-1)
    pub fn new(
        dopamine: f32,
        acetylcholine: f32,
        norepinephrine: f32,
        serotonin: f32,
        cortisol: f32,
    ) -> Self {
        Self {
            dopamine: dopamine.clamp(0.0, 1.0),
            acetylcholine: acetylcholine.clamp(0.0, 1.0),
            norepinephrine: norepinephrine.clamp(0.0, 1.0),
            serotonin: serotonin.clamp(0.0, 1.0),
            cortisol: cortisol.clamp(0.0, 1.0),
        }
    }

    /// Default balanced state
    pub fn balanced() -> Self {
        Self {
            dopamine: 0.5,
            acetylcholine: 0.5,
            norepinephrine: 0.5,
            serotonin: 0.5,
            cortisol: 0.3, // Slightly below middle for baseline
        }
    }

    /// Compute learning rate modifier from neuromodulators
    pub fn learning_rate_mod(&self) -> f32 {
        // Dopamine and acetylcholine enhance learning
        let positive = (self.dopamine * 0.5 + self.acetylcholine * 0.5) * 2.0;

        // Cortisol suppresses learning (inverted U-curve)
        let stress_suppression = if self.cortisol < 0.5 {
            1.0
        } else {
            1.0 - (self.cortisol - 0.5) * 2.0
        };

        (positive * stress_suppression).clamp(0.1, 2.0)
    }

    /// Compute emotional valence from neuromodulators
    pub fn emotional_valence(&self) -> f32 {
        // Positive: dopamine, serotonin
        // Negative: cortisol
        let positive = (self.dopamine + self.serotonin) / 2.0;
        let negative = self.cortisol;

        (positive - negative).clamp(-1.0, 1.0)
    }

    /// Compute arousal level
    pub fn arousal(&self) -> f32 {
        // Norepinephrine and cortisol increase arousal
        // Serotonin provides stability
        let activating = (self.norepinephrine + self.cortisol * 0.5) / 1.5;
        let stabilizing = self.serotonin * 0.3;

        (activating - stabilizing + 0.3).clamp(0.0, 1.0)
    }
}

impl Default for NeuromodulatorLearningMap {
    fn default() -> Self {
        Self::balanced()
    }
}

// ============================================================================
// PHYSIOLOGY INTEGRATION - Bridge to endocrine system
// ============================================================================

impl NeuromodulatorLearningMap {
    /// Create from the physiology HormoneState
    ///
    /// Maps the 3-hormone endocrine model to the 5-neuromodulator learning model:
    /// - dopamine → dopamine (direct mapping)
    /// - cortisol → cortisol (direct mapping)
    /// - acetylcholine → acetylcholine (direct mapping)
    /// - norepinephrine is inferred from cortisol + arousal
    /// - serotonin is inferred from inverse cortisol (calm when not stressed)
    ///
    /// # Example
    /// ```rust,ignore
    /// use symthaea::physiology::endocrine::HormoneState;
    /// use symthaea::hdc::adaptive_learning_signals::NeuromodulatorLearningMap;
    ///
    /// let hormones = endocrine_system.state();
    /// let neuro_map = NeuromodulatorLearningMap::from_hormone_state(hormones);
    /// ```
    pub fn from_hormone_state(dopamine: f32, acetylcholine: f32, cortisol: f32) -> Self {
        // Infer norepinephrine from cortisol (stress → alertness)
        let norepinephrine = (cortisol * 0.8 + 0.2).clamp(0.0, 1.0);

        // Infer serotonin as inverse of cortisol (relaxation vs stress)
        let serotonin = (1.0 - cortisol * 0.6).clamp(0.2, 0.8);

        Self::new(dopamine, acetylcholine, norepinephrine, serotonin, cortisol)
    }
}

// ============================================================================
// CONSCIOUSNESS-GATED LEARNING
// ============================================================================

/// Consciousness-gated learning wrapper
/// Only allows learning when consciousness criteria are met
pub struct ConsciousnessGatedLearning {
    controller: AdaptiveLearningController,

    /// Minimum Φ required for any learning
    min_phi: f32,

    /// Minimum coherence required for any learning
    min_coherence: f32,

    /// Whether learning is currently allowed
    learning_enabled: bool,

    /// Reason if learning is blocked
    block_reason: Option<String>,
}

impl ConsciousnessGatedLearning {
    /// Create with default thresholds
    pub fn new() -> Self {
        Self {
            controller: AdaptiveLearningController::new(),
            min_phi: 0.1,
            min_coherence: 0.3,
            learning_enabled: true,
            block_reason: None,
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(min_phi: f32, min_coherence: f32) -> Self {
        Self {
            controller: AdaptiveLearningController::new(),
            min_phi,
            min_coherence,
            learning_enabled: true,
            block_reason: None,
        }
    }

    /// Update and check if learning should proceed
    pub fn gate(
        &mut self,
        phi: f32,
        coherence: f32,
        prediction_error: f32,
        neuromod: &NeuromodulatorLearningMap,
    ) -> Option<AdaptiveLearningSignal> {
        // Check consciousness gate
        if phi < self.min_phi {
            self.learning_enabled = false;
            self.block_reason = Some(format!("Φ too low: {:.3} < {:.3}", phi, self.min_phi));
            return None;
        }

        if coherence < self.min_coherence {
            self.learning_enabled = false;
            self.block_reason = Some(format!(
                "Coherence too low: {:.3} < {:.3}",
                coherence, self.min_coherence
            ));
            return None;
        }

        // Gate passed - compute signal
        self.learning_enabled = true;
        self.block_reason = None;

        let arousal = neuromod.arousal();
        let valence = neuromod.emotional_valence();

        let signal = self
            .controller
            .update(phi, coherence, prediction_error, arousal, valence);

        Some(signal)
    }

    /// Check if learning is currently enabled
    pub fn is_enabled(&self) -> bool {
        self.learning_enabled
    }

    /// Get reason for learning being blocked
    pub fn block_reason(&self) -> Option<&str> {
        self.block_reason.as_deref()
    }

    /// Get underlying controller statistics
    pub fn statistics(&self) -> AdaptiveLearningStats {
        self.controller.statistics()
    }
}

impl Default for ConsciousnessGatedLearning {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_signal_default() {
        let signal = AdaptiveLearningSignal::default();
        assert_eq!(signal.learning_rate_mod, 1.0);
        assert_eq!(signal.plasticity_gate, 1.0);
        assert!(signal.should_learn());
    }

    #[test]
    fn test_controller_high_phi_boosts_learning() {
        let mut controller = AdaptiveLearningController::new();

        // Low Φ
        let low = controller.update(0.1, 0.5, 0.1, 0.5, 0.0);

        // High Φ
        let high = controller.update(0.8, 0.5, 0.1, 0.5, 0.0);

        assert!(high.learning_rate_mod > low.learning_rate_mod);
    }

    #[test]
    fn test_surprise_detection() {
        let mut controller = AdaptiveLearningController::new();

        // Establish baseline
        for _ in 0..10 {
            controller.observe_prediction_error(0.1);
        }

        // Large error = surprise
        let surprise = controller.observe_prediction_error(0.9);
        assert!(surprise > 0.3);
    }

    #[test]
    fn test_yerkes_dodson_curve() {
        let mut controller = AdaptiveLearningController::new();

        // Optimal arousal (0.5)
        let optimal = controller.update(0.5, 0.5, 0.1, 0.5, 0.0);

        // Low arousal (0.1)
        let low = controller.update(0.5, 0.5, 0.1, 0.1, 0.0);

        // High arousal (0.9)
        let high = controller.update(0.5, 0.5, 0.1, 0.9, 0.0);

        // Optimal should have highest gate
        assert!(optimal.plasticity_gate >= low.plasticity_gate);
        assert!(optimal.plasticity_gate >= high.plasticity_gate);
    }

    #[test]
    fn test_neuromodulator_map() {
        let neuro = NeuromodulatorLearningMap::new(0.8, 0.7, 0.5, 0.6, 0.2);

        // High dopamine + acetylcholine, low cortisol = high learning
        let lr_mod = neuro.learning_rate_mod();
        assert!(lr_mod > 1.0);

        // Positive valence from high dopamine + serotonin
        let valence = neuro.emotional_valence();
        assert!(valence > 0.0);
    }

    #[test]
    fn test_consciousness_gate_blocks_low_phi() {
        let mut gate = ConsciousnessGatedLearning::with_thresholds(0.2, 0.3);
        let neuro = NeuromodulatorLearningMap::balanced();

        // Below threshold
        let result = gate.gate(0.1, 0.5, 0.1, &neuro);
        assert!(result.is_none());
        assert!(!gate.is_enabled());
        assert!(gate.block_reason().unwrap().contains("Φ too low"));

        // Above threshold
        let result = gate.gate(0.5, 0.5, 0.1, &neuro);
        assert!(result.is_some());
        assert!(gate.is_enabled());
    }

    #[test]
    fn test_consolidation_detection() {
        let mut controller = AdaptiveLearningController::new();

        // Calm state: high coherence, low arousal, low surprise
        let signal = controller.update(0.5, 0.8, 0.05, 0.3, 0.0);

        // Should be ready for consolidation
        assert!(signal.consolidation_ready);
    }
}
