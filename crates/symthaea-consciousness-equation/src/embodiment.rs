// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::collections::VecDeque;
// web-time: drop-in Instant for wasm32 (std::time::Instant panics on wasm32-unknown-unknown)
use web_time::Instant;

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
    #[allow(dead_code)] // Stored for potential temporal analysis
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
            self.sensorimotor_accuracy * (1.0 - self.smoothing) + accuracy * self.smoothing;

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
            self.interoceptive_coherence * (1.0 - self.smoothing) + coherence * self.smoothing;
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

        let recent: f64 = self
            .prediction_history
            .iter()
            .rev()
            .take(5)
            .map(|r| 1.0 - (r.predicted - r.actual).abs().min(1.0))
            .sum::<f64>()
            / 5.0;

        let older: f64 = self
            .prediction_history
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .map(|r| 1.0 - (r.predicted - r.actual).abs().min(1.0))
            .sum::<f64>()
            / 5.0;

        recent - older
    }

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
        let mse: f64 = predicted_sensory
            .iter()
            .zip(actual_sensory.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / dim as f64;
        let accuracy = 1.0 - mse.sqrt().min(1.0);

        // Include proprioceptive feedback if available
        let combined_accuracy = if let Some(prop) = proprioceptive_feedback {
            // Weight proprioceptive feedback at 30%
            accuracy * 0.7 + prop * 0.3
        } else {
            accuracy
        };

        // Update EMA with weighted accuracy
        self.sensorimotor_accuracy = self.sensorimotor_accuracy * (1.0 - self.smoothing)
            + combined_accuracy * self.smoothing;

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
        let cardiac_coherence = 1.0
            - (expected_states.cardiac - actual_states.cardiac)
                .abs()
                .min(1.0);
        let respiratory_coherence = 1.0
            - (expected_states.respiratory - actual_states.respiratory)
                .abs()
                .min(1.0);
        let autonomic_coherence = 1.0
            - (expected_states.autonomic - actual_states.autonomic)
                .abs()
                .min(1.0);
        let metabolic_coherence = 1.0
            - (expected_states.metabolic - actual_states.metabolic)
                .abs()
                .min(1.0);

        // Weighted average (cardiac and autonomic weighted higher)
        let combined_coherence = cardiac_coherence * 0.3
            + autonomic_coherence * 0.3
            + respiratory_coherence * 0.2
            + metabolic_coherence * 0.2;

        // Update with smoothing
        self.interoceptive_coherence = self.interoceptive_coherence * (1.0 - self.smoothing)
            + combined_coherence * self.smoothing;
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
        let recent: Vec<f64> = self
            .prediction_history
            .iter()
            .rev()
            .take(10)
            .map(|r| (r.predicted - r.actual).abs())
            .collect();

        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64;

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
        Self {
            cardiac: v,
            respiratory: v,
            autonomic: v,
            metabolic: v,
        }
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
