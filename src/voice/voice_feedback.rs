// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Voice → CfC Feedback Loop
//!
//! This module implements bidirectional feedback from voice synthesis quality
//! back into the cognitive loop, enabling self-regulating improvement.
//!
//! ## Core Concept
//!
//! Voice synthesis quality affects consciousness state (phi) and learning:
//! - Clear articulation → higher confidence → higher phi contribution
//! - Speech rate anomalies → uncertainty → lower phi, slower learning
//! - Listener prediction success → understanding boost → higher phi
//!
//! ## Feedback Flow
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    VOICE → CfC FEEDBACK                         │
//! │                                                                 │
//! │   Voice Output ──► Quality Metrics ──► Phi Adjustment           │
//! │         │               │                     │                 │
//! │         ▼               ▼                     ▼                 │
//! │   Rate Tracking   Articulation   ──►   Learning Rate           │
//! │         │           Success              Modulation             │
//! │         ▼               │                     │                 │
//! │   Anomaly Detection ◄───┘                     │                 │
//! │                                               ▼                 │
//! │                                    CfC Weight Updates           │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for voice feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceFeedbackConfig {
    /// Weight for articulation quality in phi (0.0 to 1.0)
    pub articulation_weight: f32,

    /// Weight for rate stability in phi (0.0 to 1.0)
    pub rate_stability_weight: f32,

    /// Weight for listener prediction accuracy (0.0 to 1.0)
    pub prediction_accuracy_weight: f32,

    /// History length for smoothing
    pub history_length: usize,

    /// Target speech rate (syllables per second)
    pub target_speech_rate: f32,

    /// Acceptable rate variance (std dev as fraction of target)
    pub acceptable_rate_variance: f32,

    /// Minimum articulation score to avoid penalty
    pub min_articulation_threshold: f32,
}

impl Default for VoiceFeedbackConfig {
    fn default() -> Self {
        Self {
            articulation_weight: 0.15,
            rate_stability_weight: 0.10,
            prediction_accuracy_weight: 0.10,
            history_length: 50,
            target_speech_rate: 4.0, // ~4 syllables/second (natural speech)
            acceptable_rate_variance: 0.25,
            min_articulation_threshold: 0.7,
        }
    }
}

/// Metrics from a single voice synthesis output
#[derive(Debug, Clone, Default)]
pub struct VoiceOutputMetrics {
    /// Articulation success score (0.0 to 1.0)
    /// Higher = clearer phoneme production
    pub articulation_score: f32,

    /// Formant accuracy (0.0 to 1.0)
    /// How close formants match target phoneme
    pub formant_accuracy: f32,

    /// Speech rate (syllables per second)
    pub speech_rate: f32,

    /// Pitch stability (0.0 to 1.0)
    /// Lower variance = more stable
    pub pitch_stability: f32,

    /// Coarticulation smoothness (0.0 to 1.0)
    /// How well phonemes blend together
    pub coarticulation_smoothness: f32,

    /// Listener prediction accuracy (0.0 to 1.0)
    /// If listeners can predict what's being said
    pub listener_prediction: f32,

    /// Duration accuracy (0.0 to 1.0)
    /// How close to target phoneme durations
    pub duration_accuracy: f32,

    /// Energy consistency (0.0 to 1.0)
    /// Consistent volume across utterance
    pub energy_consistency: f32,
}

impl VoiceOutputMetrics {
    /// Auto-compute metrics directly from produced FormantFrames.
    ///
    /// This enables the voice→cognition feedback loop: after synthesis produces
    /// frames, we can immediately assess quality and feed it back to the cognitive
    /// loop without external instrumentation.
    pub fn from_formant_frames(
        frames: &[crate::voice::articulatory_synthesizer::FormantFrame],
        targets: Option<&[crate::voice::formant_targets::FormantTarget]>,
    ) -> Self {
        if frames.is_empty() {
            return Self::default();
        }

        // Pitch stability: 1.0 / (1.0 + F0 variance * 10.0)
        let voiced_f0s: Vec<f32> = frames
            .iter()
            .filter(|f| f.voicing > 0.5 && f.f0 > 0.0)
            .map(|f| f.f0)
            .collect();
        let pitch_stability = if voiced_f0s.len() >= 2 {
            let mean_f0 = voiced_f0s.iter().sum::<f32>() / voiced_f0s.len() as f32;
            let variance = voiced_f0s
                .iter()
                .map(|f| (f - mean_f0).powi(2))
                .sum::<f32>()
                / voiced_f0s.len() as f32;
            1.0 / (1.0 + variance * 10.0 / (mean_f0 * mean_f0).max(1.0))
        } else {
            0.5
        };

        // Energy consistency: 1.0 / (1.0 + energy variance * 5.0)
        let energies: Vec<f32> = frames.iter().map(|f| f.energy).collect();
        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;
        let energy_variance = energies
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / energies.len() as f32;
        let energy_consistency = 1.0 / (1.0 + energy_variance * 5.0);

        // Coarticulation smoothness: 1.0 - max_frame_delta / expected_range
        let max_delta = if frames.len() >= 2 {
            frames
                .windows(2)
                .map(|w| {
                    let df1 = (w[1].f1 - w[0].f1).abs();
                    let df2 = (w[1].f2 - w[0].f2).abs();
                    let df3 = (w[1].f3 - w[0].f3).abs();
                    df1 + df2 + df3
                })
                .fold(0.0f32, f32::max)
        } else {
            0.0
        };
        let expected_range = 1500.0; // Typical max formant shift (Hz sum across F1-F3)
        let coarticulation_smoothness = (1.0 - max_delta / expected_range).clamp(0.0, 1.0);

        // Formant accuracy: if targets provided, compute RMS error
        let formant_accuracy = if let Some(tgts) = targets {
            if !tgts.is_empty() && frames.len() >= tgts.len() {
                let chunk_size = frames.len() / tgts.len();
                let mut total_err = 0.0f32;
                for (i, tgt) in tgts.iter().enumerate() {
                    let start = i * chunk_size;
                    let end = ((i + 1) * chunk_size).min(frames.len());
                    let n = (end - start) as f32;
                    if n > 0.0 {
                        let avg_f1: f32 = frames[start..end].iter().map(|f| f.f1).sum::<f32>() / n;
                        let avg_f2: f32 = frames[start..end].iter().map(|f| f.f2).sum::<f32>() / n;
                        let err = ((avg_f1 - tgt.f1).powi(2) + (avg_f2 - tgt.f2).powi(2)).sqrt();
                        total_err += err;
                    }
                }
                let avg_err = total_err / tgts.len() as f32;
                (1.0 - avg_err / 1000.0).clamp(0.0, 1.0) // Normalize: 1000Hz error → 0.0
            } else {
                0.5
            }
        } else {
            0.5
        };

        // Articulation score: composite of formant_accuracy + voicing stability
        let voicing_stability = if frames.len() >= 2 {
            let voicing_changes: f32 = frames
                .windows(2)
                .map(|w| (w[1].voicing - w[0].voicing).abs())
                .sum::<f32>()
                / (frames.len() - 1) as f32;
            (1.0 - voicing_changes * 5.0).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let articulation_score = formant_accuracy * 0.7 + voicing_stability * 0.3;

        // Speech rate estimate: frames * dt → syllables/sec (rough: ~5 frames per syllable at 200Hz)
        let total_duration = match (frames.first(), frames.last()) {
            (Some(first), Some(last)) if frames.len() >= 2 => (last.time - first.time).abs(),
            _ => 0.1,
        };
        let speech_rate = if total_duration > 0.01 {
            // Rough estimate: count energy peaks as syllable nuclei
            let syllable_count = frames.len() as f32 / 40.0; // ~40 frames per syllable at 200Hz
            syllable_count / total_duration
        } else {
            4.0 // Default
        };

        Self {
            articulation_score,
            formant_accuracy,
            speech_rate,
            pitch_stability,
            coarticulation_smoothness,
            listener_prediction: 0.5, // Requires external feedback
            duration_accuracy: 0.8,   // Needs timing context to be precise
            energy_consistency,
        }
    }

    /// Create metrics from articulator state
    pub fn from_articulator_output(
        formants_hit: usize,
        formants_total: usize,
        actual_rate: f32,
        pitch_variance: f32,
        transition_smoothness: f32,
        duration_error: f32,
        energy_variance: f32,
    ) -> Self {
        let articulation_score = if formants_total > 0 {
            (formants_hit as f32 / formants_total as f32).min(1.0)
        } else {
            0.0
        };

        let formant_accuracy = articulation_score;

        // Pitch stability is inverse of variance (clamped)
        let pitch_stability = 1.0 / (1.0 + pitch_variance * 10.0);

        // Duration accuracy is inverse of error
        let duration_accuracy = 1.0 / (1.0 + duration_error.abs() * 5.0);

        // Energy consistency is inverse of variance
        let energy_consistency = 1.0 / (1.0 + energy_variance * 5.0);

        Self {
            articulation_score,
            formant_accuracy,
            speech_rate: actual_rate,
            pitch_stability,
            coarticulation_smoothness: transition_smoothness,
            listener_prediction: 0.5, // Default until feedback received
            duration_accuracy,
            energy_consistency,
        }
    }

    /// Overall quality score (0.0 to 1.0)
    pub fn overall_quality(&self) -> f32 {
        let weights = [
            (self.articulation_score, 0.25),
            (self.formant_accuracy, 0.15),
            (self.pitch_stability, 0.15),
            (self.coarticulation_smoothness, 0.15),
            (self.duration_accuracy, 0.10),
            (self.energy_consistency, 0.10),
            (self.listener_prediction, 0.10),
        ];

        let weighted_sum: f32 = weights.iter().map(|(v, w)| v * w).sum();
        let total_weight: f32 = weights.iter().map(|(_, w)| w).sum();

        weighted_sum / total_weight
    }
}

/// Aggregated voice quality state for feedback
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VoiceQualitySummary {
    /// Current articulation quality (smoothed)
    pub articulation_quality: f32,

    /// Speech rate stability (0.0 chaotic to 1.0 stable)
    pub rate_stability: f32,

    /// Current speech rate
    pub current_rate: f32,

    /// Rate trend (positive = speeding up, negative = slowing)
    pub rate_trend: f32,

    /// Listener prediction success rate
    pub prediction_success: f32,

    /// Overall voice confidence
    pub voice_confidence: f32,

    /// Phi adjustment from voice feedback
    pub phi_adjustment: f32,

    /// Learning rate modifier from voice (0.5 to 1.5)
    pub learning_rate_modifier: f32,
}

/// Voice feedback bridge between synthesis and cognitive loop
pub struct VoiceFeedbackBridge {
    /// Configuration
    config: VoiceFeedbackConfig,

    /// Articulation score history
    articulation_history: VecDeque<f32>,

    /// Speech rate history
    rate_history: VecDeque<f32>,

    /// Listener prediction history
    prediction_history: VecDeque<f32>,

    /// Current quality metrics
    current_metrics: VoiceOutputMetrics,

    /// Cumulative phi adjustment
    cumulative_phi: f32,

    /// Update count
    update_count: u64,
}

impl VoiceFeedbackBridge {
    /// Create a new voice feedback bridge
    pub fn new(config: VoiceFeedbackConfig) -> Self {
        Self {
            articulation_history: VecDeque::with_capacity(config.history_length),
            rate_history: VecDeque::with_capacity(config.history_length),
            prediction_history: VecDeque::with_capacity(config.history_length),
            config,
            current_metrics: VoiceOutputMetrics::default(),
            cumulative_phi: 0.0,
            update_count: 0,
        }
    }

    /// Update with new voice output metrics
    pub fn update(&mut self, metrics: VoiceOutputMetrics) {
        self.current_metrics = metrics.clone();
        self.update_count += 1;

        // Update histories
        self.articulation_history
            .push_back(metrics.articulation_score);
        if self.articulation_history.len() > self.config.history_length {
            self.articulation_history.pop_front();
        }

        self.rate_history.push_back(metrics.speech_rate);
        if self.rate_history.len() > self.config.history_length {
            self.rate_history.pop_front();
        }

        self.prediction_history
            .push_back(metrics.listener_prediction);
        if self.prediction_history.len() > self.config.history_length {
            self.prediction_history.pop_front();
        }
    }

    /// Update with listener prediction feedback
    pub fn update_listener_prediction(&mut self, prediction_success: f32) {
        self.prediction_history.push_back(prediction_success);
        if self.prediction_history.len() > self.config.history_length {
            self.prediction_history.pop_front();
        }
    }

    /// Get smoothed articulation quality
    pub fn smoothed_articulation(&self) -> f32 {
        if self.articulation_history.is_empty() {
            return 0.5;
        }
        self.articulation_history.iter().sum::<f32>() / self.articulation_history.len() as f32
    }

    /// Get speech rate stability (inverse of coefficient of variation)
    pub fn rate_stability(&self) -> f32 {
        if self.rate_history.len() < 2 {
            return 1.0; // Assume stable with limited data
        }

        let mean: f32 = self.rate_history.iter().sum::<f32>() / self.rate_history.len() as f32;
        if mean < 0.001 {
            return 0.0;
        }

        let variance: f32 = self
            .rate_history
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f32>()
            / self.rate_history.len() as f32;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean;

        // Convert CV to stability (0 to 1)
        // CV < acceptable_variance → high stability
        let normalized_cv = cv / self.config.acceptable_rate_variance;
        1.0 / (1.0 + normalized_cv)
    }

    /// Get rate trend (positive = speeding up)
    pub fn rate_trend(&self) -> f32 {
        if self.rate_history.len() < 10 {
            return 0.0;
        }

        let recent: Vec<f32> = self.rate_history.iter().rev().take(5).cloned().collect();
        let older: Vec<f32> = self
            .rate_history
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .cloned()
            .collect();

        if older.is_empty() {
            return 0.0;
        }

        let recent_avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
        let older_avg: f32 = older.iter().sum::<f32>() / older.len() as f32;

        (recent_avg - older_avg) / self.config.target_speech_rate
    }

    /// Get listener prediction success rate
    pub fn prediction_success(&self) -> f32 {
        if self.prediction_history.is_empty() {
            return 0.5; // Neutral assumption
        }
        self.prediction_history.iter().sum::<f32>() / self.prediction_history.len() as f32
    }

    /// Compute phi adjustment from voice quality
    ///
    /// Formula: phi_adjustment = understanding_boost - uncertainty_penalty
    /// - understanding_boost: prediction_success * weight
    /// - uncertainty_penalty: (1 - rate_stability) * weight + (1 - articulation) * weight
    pub fn compute_phi_adjustment(&self) -> f32 {
        let articulation = self.smoothed_articulation();
        let stability = self.rate_stability();
        let prediction = self.prediction_success();

        // Understanding boost (when listeners can predict/understand)
        let understanding_boost = prediction * self.config.prediction_accuracy_weight;

        // Uncertainty penalty
        let articulation_penalty = if articulation < self.config.min_articulation_threshold {
            (self.config.min_articulation_threshold - articulation)
                * self.config.articulation_weight
        } else {
            0.0
        };
        let stability_penalty = (1.0 - stability) * self.config.rate_stability_weight;
        let uncertainty_penalty = articulation_penalty + stability_penalty;

        // Net phi adjustment
        let adjustment = understanding_boost - uncertainty_penalty;
        adjustment.clamp(-0.3, 0.3) // Limit extreme adjustments
    }

    /// Compute learning rate modifier based on voice confidence
    ///
    /// Low voice quality → reduce learning rate (uncertain state)
    /// High voice quality → normal learning rate (confident state)
    ///
    /// Neutral (1.0) until the bridge has received at least one real metrics
    /// update: the empty-history defaults would otherwise impose a constant
    /// ×0.85 on every cycle's learning rate in builds where nothing feeds
    /// voice metrics.
    pub fn learning_rate_modifier(&self) -> f32 {
        if self.update_count == 0 {
            return 1.0;
        }
        let articulation = self.smoothed_articulation();
        let stability = self.rate_stability();

        // Combined voice confidence
        let confidence = (articulation * 0.6 + stability * 0.4).min(1.0);

        // Map confidence to learning rate modifier
        // Low confidence (0.0) → 0.5x learning rate
        // High confidence (1.0) → 1.0x learning rate
        0.5 + confidence * 0.5
    }

    /// Get current quality summary
    pub fn summary(&self) -> VoiceQualitySummary {
        let current_rate = self
            .rate_history
            .back()
            .copied()
            .unwrap_or(self.config.target_speech_rate);

        VoiceQualitySummary {
            articulation_quality: self.smoothed_articulation(),
            rate_stability: self.rate_stability(),
            current_rate,
            rate_trend: self.rate_trend(),
            prediction_success: self.prediction_success(),
            voice_confidence: (self.smoothed_articulation() * 0.6 + self.rate_stability() * 0.4),
            phi_adjustment: self.compute_phi_adjustment(),
            learning_rate_modifier: self.learning_rate_modifier(),
        }
    }

    /// Check if voice quality indicates uncertainty
    pub fn is_uncertain(&self) -> bool {
        let articulation = self.smoothed_articulation();
        let stability = self.rate_stability();

        articulation < self.config.min_articulation_threshold || stability < 0.5
    }

    /// Check if rate is anomalous (too fast or too slow)
    pub fn is_rate_anomalous(&self) -> bool {
        if self.rate_history.is_empty() {
            return false;
        }

        let current_rate = *self
            .rate_history
            .back()
            .expect("rate_history checked non-empty above");
        let target = self.config.target_speech_rate;
        let acceptable_range = target * self.config.acceptable_rate_variance;

        (current_rate - target).abs() > acceptable_range
    }

    /// Reset the bridge
    pub fn reset(&mut self) {
        self.articulation_history.clear();
        self.rate_history.clear();
        self.prediction_history.clear();
        self.current_metrics = VoiceOutputMetrics::default();
        self.cumulative_phi = 0.0;
        self.update_count = 0;
    }
}

impl Default for VoiceFeedbackBridge {
    fn default() -> Self {
        Self::new(VoiceFeedbackConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_metrics_creation() {
        let metrics = VoiceOutputMetrics::from_articulator_output(
            8,    // formants_hit
            10,   // formants_total
            4.0,  // actual_rate
            0.1,  // pitch_variance
            0.8,  // transition_smoothness
            0.05, // duration_error
            0.1,  // energy_variance
        );

        assert!((metrics.articulation_score - 0.8).abs() < 0.001);
        assert!(metrics.pitch_stability > 0.0);
        assert!(metrics.overall_quality() > 0.0);
    }

    #[test]
    fn test_rate_stability_uniform() {
        let mut bridge = VoiceFeedbackBridge::default();

        // Uniform rate = high stability
        for _ in 0..20 {
            let metrics = VoiceOutputMetrics {
                speech_rate: 4.0,
                articulation_score: 0.8,
                ..Default::default()
            };
            bridge.update(metrics);
        }

        assert!(
            bridge.rate_stability() > 0.8,
            "Uniform rate should have high stability: {}",
            bridge.rate_stability()
        );
    }

    #[test]
    fn test_rate_stability_variable() {
        let mut bridge = VoiceFeedbackBridge::default();

        // Variable rate = lower stability
        for i in 0..20 {
            let rate = if i % 2 == 0 { 2.0 } else { 6.0 }; // Alternating slow/fast
            let metrics = VoiceOutputMetrics {
                speech_rate: rate,
                articulation_score: 0.8,
                ..Default::default()
            };
            bridge.update(metrics);
        }

        assert!(
            bridge.rate_stability() < 0.5,
            "Variable rate should have lower stability: {}",
            bridge.rate_stability()
        );
    }

    #[test]
    fn test_phi_adjustment_good_quality() {
        let mut bridge = VoiceFeedbackBridge::default();

        // Good articulation, stable rate, good prediction
        for _ in 0..20 {
            let metrics = VoiceOutputMetrics {
                articulation_score: 0.9,
                speech_rate: 4.0,
                listener_prediction: 0.8,
                ..Default::default()
            };
            bridge.update(metrics);
        }

        let phi_adj = bridge.compute_phi_adjustment();
        assert!(
            phi_adj > 0.0,
            "Good quality should give positive phi adjustment: {}",
            phi_adj
        );
    }

    #[test]
    fn test_phi_adjustment_poor_quality() {
        let mut bridge = VoiceFeedbackBridge::default();

        // Poor articulation, unstable rate
        for i in 0..20 {
            let metrics = VoiceOutputMetrics {
                articulation_score: 0.4,                         // Below threshold
                speech_rate: if i % 2 == 0 { 2.0 } else { 6.0 }, // Unstable
                listener_prediction: 0.3,                        // Poor prediction
                ..Default::default()
            };
            bridge.update(metrics);
        }

        let phi_adj = bridge.compute_phi_adjustment();
        assert!(
            phi_adj < 0.0,
            "Poor quality should give negative phi adjustment: {}",
            phi_adj
        );
    }

    #[test]
    fn test_learning_rate_modifier() {
        let mut bridge = VoiceFeedbackBridge::default();

        // High quality = higher learning rate modifier
        for _ in 0..20 {
            let metrics = VoiceOutputMetrics {
                articulation_score: 0.95,
                speech_rate: 4.0,
                ..Default::default()
            };
            bridge.update(metrics);
        }
        let high_quality_lr = bridge.learning_rate_modifier();

        // Reset and test low quality
        bridge.reset();
        for i in 0..20 {
            let metrics = VoiceOutputMetrics {
                articulation_score: 0.3,
                speech_rate: if i % 2 == 0 { 2.0 } else { 6.0 },
                ..Default::default()
            };
            bridge.update(metrics);
        }
        let low_quality_lr = bridge.learning_rate_modifier();

        assert!(
            high_quality_lr > low_quality_lr,
            "High quality should have higher LR modifier: {} vs {}",
            high_quality_lr,
            low_quality_lr
        );
    }

    #[test]
    fn test_learning_rate_modifier_neutral_when_never_fed() {
        // Regression: an unfed bridge must not scale the learning rate.
        // The empty-history defaults (articulation 0.5, stability 1.0) used to
        // yield a permanent ×0.85 on every cycle in default builds.
        let bridge = VoiceFeedbackBridge::default();
        assert_eq!(bridge.learning_rate_modifier(), 1.0);

        // After a real update it becomes an actual signal again.
        let mut bridge = bridge;
        bridge.update(VoiceOutputMetrics {
            articulation_score: 0.9,
            speech_rate: 4.0,
            ..Default::default()
        });
        assert!(bridge.learning_rate_modifier() < 1.0 + f32::EPSILON);
        assert!(bridge.learning_rate_modifier() > 0.5);

        // Reset returns to neutral.
        bridge.reset();
        assert_eq!(bridge.learning_rate_modifier(), 1.0);
    }

    #[test]
    fn test_rate_anomaly_detection() {
        let mut bridge = VoiceFeedbackBridge::default();

        // Normal rate
        let metrics = VoiceOutputMetrics {
            speech_rate: 4.0, // Target is 4.0
            ..Default::default()
        };
        bridge.update(metrics);
        assert!(!bridge.is_rate_anomalous());

        // Very fast rate
        let metrics = VoiceOutputMetrics {
            speech_rate: 8.0, // Way above target
            ..Default::default()
        };
        bridge.update(metrics);
        assert!(bridge.is_rate_anomalous());
    }

    #[test]
    fn test_uncertainty_detection() {
        let mut bridge = VoiceFeedbackBridge::default();

        // Good quality = not uncertain
        for _ in 0..10 {
            let metrics = VoiceOutputMetrics {
                articulation_score: 0.9,
                speech_rate: 4.0,
                ..Default::default()
            };
            bridge.update(metrics);
        }
        assert!(!bridge.is_uncertain());

        // Reset and test uncertain state
        bridge.reset();
        for _ in 0..10 {
            let metrics = VoiceOutputMetrics {
                articulation_score: 0.5, // Below threshold
                speech_rate: 4.0,
                ..Default::default()
            };
            bridge.update(metrics);
        }
        assert!(bridge.is_uncertain());
    }

    #[test]
    fn test_summary() {
        let mut bridge = VoiceFeedbackBridge::default();

        for _ in 0..10 {
            let metrics = VoiceOutputMetrics {
                articulation_score: 0.8,
                speech_rate: 4.0,
                listener_prediction: 0.7,
                ..Default::default()
            };
            bridge.update(metrics);
        }

        let summary = bridge.summary();
        assert!(summary.articulation_quality > 0.0);
        assert!(summary.rate_stability > 0.0);
        assert!(summary.voice_confidence > 0.0);
    }

    #[test]
    fn test_metrics_from_formant_frames() {
        use crate::voice::articulatory_synthesizer::FormantFrame;

        // Stable frames: consistent F0, energy, formants
        let stable_frames: Vec<FormantFrame> = (0..50)
            .map(|i| FormantFrame {
                f1: 500.0,
                f2: 1500.0,
                f3: 2500.0,
                b1: 60.0,
                b2: 90.0,
                b3: 150.0,
                f0: 120.0,
                energy: 0.8,
                voicing: 1.0,
                time: i as f32 * 0.005,
                ..Default::default()
            })
            .collect();
        let stable_metrics = VoiceOutputMetrics::from_formant_frames(&stable_frames, None);

        assert!(
            stable_metrics.pitch_stability > 0.8,
            "Stable pitch should have high stability: {}",
            stable_metrics.pitch_stability
        );
        assert!(
            stable_metrics.energy_consistency > 0.8,
            "Stable energy should have high consistency: {}",
            stable_metrics.energy_consistency
        );
        assert!(
            stable_metrics.coarticulation_smoothness > 0.9,
            "No transitions should be smooth: {}",
            stable_metrics.coarticulation_smoothness
        );

        // Unstable frames: varying F0, energy, jittering formants
        let unstable_frames: Vec<FormantFrame> = (0..50)
            .map(|i| FormantFrame {
                f1: 300.0 + (i as f32 * 0.3).sin() * 400.0,
                f2: 1500.0 + (i as f32 * 0.5).cos() * 500.0,
                f3: 2500.0,
                b1: 60.0,
                b2: 90.0,
                b3: 150.0,
                f0: 80.0 + (i as f32 * 0.7).sin() * 80.0,
                energy: 0.3 + (i as f32 * 0.4).sin() * 0.5,
                voicing: if i % 3 == 0 { 0.2 } else { 1.0 },
                time: i as f32 * 0.005,
                ..Default::default()
            })
            .collect();
        let unstable_metrics = VoiceOutputMetrics::from_formant_frames(&unstable_frames, None);

        // Unstable should score lower on key metrics
        assert!(
            unstable_metrics.pitch_stability < stable_metrics.pitch_stability,
            "Unstable pitch should be less stable"
        );
        assert!(
            unstable_metrics.energy_consistency < stable_metrics.energy_consistency,
            "Unstable energy should be less consistent"
        );
    }

    #[test]
    fn test_rate_trend() {
        let mut bridge = VoiceFeedbackBridge::default();

        // Gradually increasing rate
        for i in 0..20 {
            let metrics = VoiceOutputMetrics {
                speech_rate: 3.0 + i as f32 * 0.1, // 3.0 → 5.0
                articulation_score: 0.8,
                ..Default::default()
            };
            bridge.update(metrics);
        }

        assert!(
            bridge.rate_trend() > 0.0,
            "Increasing rate should have positive trend: {}",
            bridge.rate_trend()
        );
    }
}
