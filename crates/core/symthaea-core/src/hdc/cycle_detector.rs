// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cycle Detection Module for Cincinnati-LTC
//!
//! Addresses Cincinnati-LTC's weakness with deterministic periodic patterns
//! by adding explicit cycle detection and phase encoding.
//!
//! ## Theory
//!
//! Cincinnati-LTC excels at differential/statistical patterns but struggles with
//! periodic sequences because it doesn't track position within cycles. This module:
//!
//! 1. **Autocorrelation**: Detects dominant period in the signal
//! 2. **Phase Tracking**: Maintains position within detected cycle
//! 3. **HDC Encoding**: Converts cycle phase to hypervector representation
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::cycle_detector::CycleDetector;
//!
//! let mut detector = CycleDetector::new(16); // max period to detect
//! for bit in sequence {
//!     detector.observe(bit);
//!     let phase_hv = detector.phase_encoding();
//!     // Combine with Cincinnati-LTC input
//! }
//! ```

use crate::hdc::HDC_DIMENSION;
use crate::hdc::unified_hv::ContinuousHV;
use std::collections::VecDeque;

/// Cycle detection and phase encoding for periodic patterns
#[derive(Debug, Clone)]
pub struct CycleDetector {
    /// History buffer for autocorrelation
    history: VecDeque<bool>,

    /// Maximum period to search for
    max_period: usize,

    /// Detected dominant period (0 if none detected)
    detected_period: usize,

    /// Current position within cycle (0 to period-1)
    cycle_position: usize,

    /// Confidence in detected period (0.0 to 1.0)
    confidence: f32,

    /// Autocorrelation scores for each period
    autocorr_scores: Vec<f32>,

    /// Pre-computed phase vectors for each position in each possible period
    phase_vectors: Vec<Vec<ContinuousHV>>,

    /// Pre-computed period indicator vectors
    period_vectors: Vec<ContinuousHV>,

    /// Total observations
    total_observations: usize,

    /// Minimum observations before detecting period
    warmup_period: usize,
}

impl CycleDetector {
    /// Create a new cycle detector
    ///
    /// # Arguments
    /// * `max_period` - Maximum period length to detect (e.g., 16)
    pub fn new(max_period: usize) -> Self {
        let max_period = max_period.max(2); // At least period 2

        // Pre-compute phase vectors for each period and position
        // phase_vectors[period][position] gives unique HV for that phase
        let phase_vectors: Vec<Vec<ContinuousHV>> = (0..=max_period)
            .map(|period| {
                if period < 2 {
                    vec![]
                } else {
                    (0..period)
                        .map(|pos| {
                            // Create phase vector using circular encoding
                            let seed = (period as u64) * 10000 + (pos as u64) * 100 + 42;
                            Self::create_phase_vector(period, pos, seed)
                        })
                        .collect()
                }
            })
            .collect();

        // Pre-compute period indicator vectors
        let period_vectors: Vec<ContinuousHV> = (0..=max_period)
            .map(|period| ContinuousHV::random(HDC_DIMENSION, (period as u64) * 77777 + 12345))
            .collect();

        Self {
            history: VecDeque::with_capacity(max_period * 4),
            max_period,
            detected_period: 0,
            cycle_position: 0,
            confidence: 0.0,
            autocorr_scores: vec![0.0; max_period + 1],
            phase_vectors,
            period_vectors,
            total_observations: 0,
            warmup_period: max_period * 2, // Need at least 2 full cycles
        }
    }

    /// Create a phase vector with circular/sinusoidal structure
    fn create_phase_vector(period: usize, position: usize, seed: u64) -> ContinuousHV {
        use std::f32::consts::PI;

        let mut values = vec![0.0f32; HDC_DIMENSION];
        let phase = 2.0 * PI * (position as f32) / (period as f32);

        // Create structured encoding with multiple frequency components
        for i in 0..HDC_DIMENSION {
            let base_freq = (i as f32) / (HDC_DIMENSION as f32) * 2.0 * PI;

            // Combine phase information at multiple scales
            let component1 = (base_freq + phase).sin();
            let component2 = (base_freq * 2.0 + phase * 2.0).cos();
            let component3 = (base_freq * 0.5 + phase).sin();

            // Add some randomness for orthogonality
            let random_component = {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                (seed + i as u64).hash(&mut hasher);
                let hash = hasher.finish();
                ((hash % 1000) as f32 / 500.0 - 1.0) * 0.1
            };

            values[i] = component1 * 0.4 + component2 * 0.3 + component3 * 0.2 + random_component;
        }

        // Normalize
        let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut values {
                *v /= norm;
            }
        }

        ContinuousHV { values }
    }

    /// Observe a new bit in the sequence
    pub fn observe(&mut self, bit: bool) {
        self.history.push_back(bit);
        self.total_observations += 1;

        // Maintain history buffer size
        let max_history = self.max_period * 4;
        while self.history.len() > max_history {
            self.history.pop_front();
        }

        // Update cycle position
        if self.detected_period > 0 {
            self.cycle_position = (self.cycle_position + 1) % self.detected_period;
        }

        // Periodically re-detect period (every max_period observations)
        if self.total_observations >= self.warmup_period
            && self.total_observations.is_multiple_of(self.max_period)
        {
            self.detect_period();
        }
    }

    /// Detect dominant period using autocorrelation
    fn detect_period(&mut self) {
        if self.history.len() < self.warmup_period {
            return;
        }

        let history: Vec<f32> = self
            .history
            .iter()
            .map(|&b| if b { 1.0 } else { -1.0 })
            .collect();

        let n = history.len();

        // Compute autocorrelation for each lag (potential period)
        for period in 2..=self.max_period {
            if period >= n / 2 {
                self.autocorr_scores[period] = 0.0;
                continue;
            }

            let mut correlation = 0.0;
            let mut count = 0;

            for i in period..n {
                correlation += history[i] * history[i - period];
                count += 1;
            }

            self.autocorr_scores[period] = if count > 0 {
                correlation / count as f32
            } else {
                0.0
            };
        }

        // Find best period (highest autocorrelation)
        let mut best_period = 0;
        let mut best_score = 0.3; // Minimum threshold for detection

        for period in 2..=self.max_period {
            if self.autocorr_scores[period] > best_score {
                // Check for harmonics - prefer fundamental frequency
                let mut is_harmonic = false;
                for divisor in 2..period {
                    if period % divisor == 0 {
                        let fundamental_score = self.autocorr_scores[divisor];
                        if fundamental_score > self.autocorr_scores[period] * 0.8 {
                            is_harmonic = true;
                            break;
                        }
                    }
                }

                if !is_harmonic {
                    best_score = self.autocorr_scores[period];
                    best_period = period;
                }
            }
        }

        // Update detected period
        if best_period > 0 && best_score > 0.3 {
            if self.detected_period != best_period {
                // Period changed - reset position
                self.detected_period = best_period;
                self.cycle_position = self.total_observations % best_period;
            }
            self.confidence = best_score.min(1.0);
        } else {
            // No strong period detected
            self.detected_period = 0;
            self.confidence = 0.0;
        }
    }

    /// Get phase encoding as HDC hypervector
    ///
    /// Returns a hypervector that encodes:
    /// 1. The detected period (or zero vector if no period)
    /// 2. The current position within the cycle
    pub fn phase_encoding(&self) -> ContinuousHV {
        if self.detected_period == 0 || self.detected_period > self.max_period {
            // No period detected - return zero vector
            return ContinuousHV::from_values(vec![0.0; HDC_DIMENSION]);
        }

        let period = self.detected_period;
        let position = self.cycle_position % period;

        // Get phase vector for current position
        let phase_hv = &self.phase_vectors[period][position];

        // Get period indicator vector
        let period_hv = &self.period_vectors[period];

        // Combine: bind phase with period indicator, scaled by confidence
        let combined = phase_hv.bind(period_hv);
        combined.scale(self.confidence)
    }

    /// Get detailed cycle state
    pub fn state(&self) -> CycleState {
        CycleState {
            detected_period: self.detected_period,
            cycle_position: self.cycle_position,
            confidence: self.confidence,
            total_observations: self.total_observations,
        }
    }

    /// Get autocorrelation scores for analysis
    pub fn autocorr_scores(&self) -> &[f32] {
        &self.autocorr_scores
    }

    /// Predict next bit based on detected cycle
    ///
    /// Returns (prediction, confidence) or None if no cycle detected
    pub fn predict_next(&self) -> Option<(bool, f32)> {
        if self.detected_period == 0 || self.history.len() < self.detected_period {
            return None;
        }

        let period = self.detected_period;
        let next_position = (self.cycle_position + 1) % period;

        // Look at what was at this position in previous cycles
        let history_vec: Vec<bool> = self.history.iter().copied().collect();
        let n = history_vec.len();

        let mut true_count = 0;
        let mut false_count = 0;

        // Count values at same position in previous cycles
        for i in (0..n).rev() {
            let pos_in_history = (self.total_observations - (n - i)) % period;
            if pos_in_history == next_position {
                if history_vec[i] {
                    true_count += 1;
                } else {
                    false_count += 1;
                }
            }
        }

        let total = true_count + false_count;
        if total == 0 {
            return None;
        }

        let prediction = true_count > false_count;
        let pred_confidence = (true_count.max(false_count) as f32) / (total as f32);

        Some((prediction, pred_confidence * self.confidence))
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.history.clear();
        self.detected_period = 0;
        self.cycle_position = 0;
        self.confidence = 0.0;
        self.total_observations = 0;
        for score in &mut self.autocorr_scores {
            *score = 0.0;
        }
    }
}

/// Cycle state snapshot
#[derive(Debug, Clone, Copy)]
pub struct CycleState {
    /// Detected period (0 if none)
    pub detected_period: usize,
    /// Position within cycle
    pub cycle_position: usize,
    /// Detection confidence
    pub confidence: f32,
    /// Total observations processed
    pub total_observations: usize,
}

// =============================================================================
// ENHANCED TEMPORAL RECOGNIZER
// =============================================================================

/// Enhanced pattern recognizer combining Cincinnati-LTC with cycle detection
pub struct CycleAwareLtcRecognizer {
    /// Cincinnati-LTC engine
    engine: crate::hdc::cincinnati_ltc::CincinnatiLtcEngine,

    /// Cycle detector
    cycle_detector: CycleDetector,

    /// History of observations
    history: VecDeque<bool>,

    /// Predictions made
    predictions: VecDeque<bool>,

    /// Correct prediction count
    correct: usize,

    /// Total predictions
    total: usize,

    /// Recent accuracy window
    recent_correct: VecDeque<bool>,

    /// Weight for cycle detection vs Cincinnati (0.0 = pure Cincinnati, 1.0 = pure cycle)
    cycle_weight: f32,
}

impl CycleAwareLtcRecognizer {
    /// Create a new cycle-aware recognizer
    pub fn new(n_nodes: usize, max_period: usize) -> Self {
        let mut engine = crate::hdc::cincinnati_ltc::CincinnatiLtcEngine::new(n_nodes);
        engine.set_budding_threshold(0.5);
        engine.set_sustain_steps(3);

        Self {
            engine,
            cycle_detector: CycleDetector::new(max_period),
            history: VecDeque::with_capacity(64),
            predictions: VecDeque::with_capacity(64),
            correct: 0,
            total: 0,
            recent_correct: VecDeque::with_capacity(100),
            cycle_weight: 0.5, // Equal weight by default
        }
    }

    /// Set the balance between cycle detection and Cincinnati-LTC
    pub fn set_cycle_weight(&mut self, weight: f32) {
        self.cycle_weight = weight.clamp(0.0, 1.0);
    }

    /// Observe and predict
    pub fn observe_and_predict(&mut self, observation: bool) -> (bool, f32) {
        // Update cycle detector
        self.cycle_detector.observe(observation);

        // Create input combining history and cycle phase
        let history_hv = self.history_to_hv();
        let cycle_hv = self.cycle_detector.phase_encoding();

        // Combine: history + cycle_phase (weighted)
        let combined_input = history_hv.add(&cycle_hv.scale(self.cycle_weight * 2.0));

        // Step Cincinnati-LTC
        let _output = self.engine.step(observation, &combined_input);

        // Get Cincinnati prediction
        let (cincinnati_pred, cincinnati_conf) = self.engine.predict();

        // Get cycle-based prediction
        let cycle_prediction = self.cycle_detector.predict_next();

        // Combine predictions
        let (final_pred, final_conf) = match cycle_prediction {
            Some((cycle_pred, cycle_conf)) if cycle_conf > 0.3 => {
                // Weighted combination
                let cincinnati_vote = if cincinnati_pred {
                    cincinnati_conf
                } else {
                    -cincinnati_conf
                };
                let cycle_vote = if cycle_pred { cycle_conf } else { -cycle_conf };

                let combined_vote =
                    cincinnati_vote * (1.0 - self.cycle_weight) + cycle_vote * self.cycle_weight;

                let pred = combined_vote > 0.0;
                let conf = combined_vote.abs().min(1.0);
                (pred, conf)
            }
            _ => {
                // No cycle detected - use Cincinnati only
                (cincinnati_pred, cincinnati_conf)
            }
        };

        // Update history
        self.history.push_back(observation);
        if self.history.len() > 32 {
            self.history.pop_front();
        }

        // Track accuracy
        if self.total > 0 {
            let was_correct = self
                .predictions
                .back()
                .map(|&p| p == observation)
                .unwrap_or(false);
            if was_correct {
                self.correct += 1;
            }
            self.recent_correct.push_back(was_correct);
            if self.recent_correct.len() > 100 {
                self.recent_correct.pop_front();
            }

            // Update prediction errors for budding
            let node_count = self.engine.node_count();
            for node_id in 0..node_count {
                let expected =
                    self.create_bit_hv(self.predictions.back().copied().unwrap_or(false));
                let actual = self.create_bit_hv(observation);
                self.engine
                    .update_prediction_error(node_id, &expected, &actual);
            }
        }

        self.predictions.push_back(final_pred);
        if self.predictions.len() > 64 {
            self.predictions.pop_front();
        }
        self.total += 1;

        (final_pred, final_conf)
    }

    /// Convert history to HDC hypervector
    fn history_to_hv(&self) -> ContinuousHV {
        if self.history.is_empty() {
            return ContinuousHV::random(HDC_DIMENSION, 0);
        }

        let mut result_values = vec![0.0f32; HDC_DIMENSION];
        let hist_len = self.history.len();

        for (i, &bit) in self.history.iter().enumerate() {
            let bit_value = if bit { 1.0 } else { -1.0 };
            let recency = (i + 1) as f32 / hist_len as f32;

            // Simple position encoding
            let pos_seed = (i as u64) * 12345 + 9999;
            let pos_hv = ContinuousHV::random(HDC_DIMENSION, pos_seed);

            for (j, v) in pos_hv.values.iter().enumerate() {
                result_values[j] += v * bit_value * recency;
            }
        }

        // Normalize
        let norm: f32 = result_values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut result_values {
                *v /= norm;
            }
        }

        ContinuousHV {
            values: result_values,
        }
    }

    /// Create bit HV for prediction error tracking
    fn create_bit_hv(&self, bit: bool) -> ContinuousHV {
        let seed = if bit { 111111 } else { 222222 };
        ContinuousHV::random(HDC_DIMENSION, seed)
    }

    /// Get overall accuracy
    pub fn accuracy(&self) -> f32 {
        if self.total <= 1 {
            0.5
        } else {
            self.correct as f32 / (self.total - 1) as f32
        }
    }

    /// Get recent accuracy
    pub fn recent_accuracy(&self) -> f32 {
        if self.recent_correct.is_empty() {
            0.5
        } else {
            self.recent_correct.iter().filter(|&&c| c).count() as f32
                / self.recent_correct.len() as f32
        }
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.engine.node_count()
    }

    /// Get cycle state
    pub fn cycle_state(&self) -> CycleState {
        self.cycle_detector.state()
    }

    /// Process budding
    pub fn process_budding(&mut self, time: f64) -> Vec<crate::hdc::cincinnati_ltc::BuddingEvent> {
        let input = self.history_to_hv();
        self.engine.process_budding(&[input], time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycle_detection_period_4() {
        let mut detector = CycleDetector::new(16);

        // Feed a period-4 pattern: 1100 1100 1100...
        for cycle in 0..20 {
            detector.observe(true);
            detector.observe(true);
            detector.observe(false);
            detector.observe(false);
        }

        let state = detector.state();
        println!(
            "Detected period: {}, confidence: {:.2}",
            state.detected_period, state.confidence
        );

        // Should detect period 4 with high confidence
        assert!(
            state.detected_period == 4 || state.detected_period == 2,
            "Should detect period 4 or its harmonic 2, got {}",
            state.detected_period
        );
        assert!(state.confidence > 0.5, "Confidence should be > 0.5");
    }

    #[test]
    fn test_cycle_detection_period_8() {
        let mut detector = CycleDetector::new(16);

        // Feed a period-8 pattern: 11110000 11110000...
        for cycle in 0..15 {
            for _ in 0..4 {
                detector.observe(true);
            }
            for _ in 0..4 {
                detector.observe(false);
            }
        }

        let state = detector.state();
        println!(
            "Detected period: {}, confidence: {:.2}",
            state.detected_period, state.confidence
        );

        assert!(
            state.detected_period == 8 || state.detected_period == 4,
            "Should detect period 8 or harmonic, got {}",
            state.detected_period
        );
    }

    #[test]
    fn test_no_cycle_in_random() {
        let mut detector = CycleDetector::new(16);

        // Feed random data
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        for i in 0..200 {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let bit = hasher.finish() % 2 == 0;
            detector.observe(bit);
        }

        let state = detector.state();
        println!(
            "Random: detected period: {}, confidence: {:.2}",
            state.detected_period, state.confidence
        );

        // Confidence should be low for random data
        assert!(
            state.confidence < 0.5,
            "Random data should have low confidence"
        );
    }

    #[test]
    fn test_phase_encoding_orthogonality() {
        let detector = CycleDetector::new(8);

        // Phase vectors for different positions should be somewhat orthogonal
        let phase0 = &detector.phase_vectors[4][0];
        let phase1 = &detector.phase_vectors[4][1];
        let phase2 = &detector.phase_vectors[4][2];

        let sim_01 = phase0.similarity(phase1);
        let sim_02 = phase0.similarity(phase2);
        let sim_12 = phase1.similarity(phase2);

        println!(
            "Phase similarities: 0-1={:.3}, 0-2={:.3}, 1-2={:.3}",
            sim_01, sim_02, sim_12
        );

        // Adjacent phases should have some similarity (circular), opposite less
        assert!(
            sim_01.abs() < 0.8,
            "Adjacent phases should not be identical"
        );
    }
}
