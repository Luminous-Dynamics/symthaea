// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LTC-based Rhythm Detection — temporal pattern recognition via CfC dynamics
//!
//! This module implements rhythm detection by using the HierarchicalLTC network
//! to learn and recognize temporal patterns in video feature streams. The LTC
//! network naturally captures temporal dynamics, making it ideal for
//! distinguishing patterns that differ only in timing (not in content).
//!
//! ## Key Types
//!
//! - [`LtcRhythmConfig`]: Configuration for the rhythm detector
//! - [`LtcRhythmDetector`]: Main detector that learns and recognizes patterns
//! - [`RhythmDetection`]: Result of processing a frame through the detector

use std::collections::BTreeMap;

use super::temporal_features::TemporalFeatures;

/// Configuration for the LTC rhythm detector
#[derive(Debug, Clone)]
pub struct LtcRhythmConfig {
    /// Time step in milliseconds between frames
    pub dt_ms: f32,
    /// Number of LTC neurons in the hidden layer
    pub hidden_size: usize,
    /// Similarity threshold for confident detection
    pub confidence_threshold: f32,
    /// Minimum frames needed to learn a pattern
    pub min_learn_frames: usize,
}

impl Default for LtcRhythmConfig {
    fn default() -> Self {
        Self {
            dt_ms: 33.33, // ~30 FPS
            hidden_size: 64,
            confidence_threshold: 0.6,
            min_learn_frames: 30,
        }
    }
}

/// Result of rhythm detection on a single frame
#[derive(Debug, Clone)]
pub struct RhythmDetection {
    /// Whether this detection is valid (enough data processed)
    pub is_valid: bool,
    /// Detected pattern name (or "Unknown" / "Uncertain")
    pub pattern: String,
    /// Confidence in the detection (0.0 to 1.0)
    pub confidence: f32,
    /// Phi (consciousness integration) metric for this detection
    pub phi: f32,
    /// Per-pattern similarity scores
    pub similarities: BTreeMap<char, f32>,
}

impl Default for RhythmDetection {
    fn default() -> Self {
        Self {
            is_valid: false,
            pattern: "Unknown".to_string(),
            confidence: 0.0,
            phi: 0.0,
            similarities: BTreeMap::new(),
        }
    }
}

/// Learned pattern: a trajectory of feature vectors captured during learning
#[derive(Debug, Clone)]
struct LearnedPattern {
    /// The pattern key (e.g., 'A', 'B')
    key: char,
    /// Feature trajectory (sequence of feature vectors)
    trajectory: Vec<[f32; 32]>,
    /// Frequency signature (power at 0.5Hz, 1Hz, 2Hz, 4Hz)
    frequency_signature: [f32; 4],
}

/// LTC-based rhythm detector
///
/// Learns temporal patterns from video feature streams and recognizes them
/// in real-time using trajectory similarity in LTC state space.
#[derive(Debug)]
pub struct LtcRhythmDetector {
    config: LtcRhythmConfig,
    /// Learned patterns keyed by their character identifier
    patterns: BTreeMap<char, LearnedPattern>,
    /// Current learning state
    learning_key: Option<char>,
    /// Frames collected during learning
    learning_buffer: Vec<[f32; 32]>,
    /// Running LTC state for detection
    state: Vec<f32>,
    /// Frame counter for detection
    detect_frame: usize,
}

impl LtcRhythmDetector {
    /// Create a new rhythm detector with the given configuration
    pub fn new(config: LtcRhythmConfig) -> Self {
        let hidden_size = config.hidden_size;
        Self {
            config,
            patterns: BTreeMap::new(),
            learning_key: None,
            learning_buffer: Vec::new(),
            state: vec![0.0; hidden_size],
            detect_frame: 0,
        }
    }

    /// Start learning a new pattern with the given key
    pub fn start_learning(&mut self, key: char) {
        self.learning_key = Some(key);
        self.learning_buffer.clear();
    }

    /// Stop learning and finalize the pattern
    ///
    /// Returns `(key, frame_count, estimated_period_frames)` if learning was active
    pub fn stop_learning(&mut self) -> Option<(char, usize, f32)> {
        let key = self.learning_key.take()?;
        let frames = self.learning_buffer.len();

        if frames < self.config.min_learn_frames {
            return None;
        }

        // Compute frequency signature from the trajectory
        let freq_sig = self.compute_frequency_signature(&self.learning_buffer);

        // Estimate dominant period
        let period = self.estimate_period(&self.learning_buffer);

        let pattern = LearnedPattern {
            key,
            trajectory: self.learning_buffer.clone(),
            frequency_signature: freq_sig,
        };

        self.patterns.insert(key, pattern);
        self.learning_buffer.clear();

        Some((key, frames, period))
    }

    /// Process a frame's temporal features and return detection result
    pub fn process(&mut self, features: &TemporalFeatures) -> RhythmDetection {
        // If learning, accumulate
        if self.learning_key.is_some() {
            self.learning_buffer.push(features.features);
        }

        // Update internal state (simple exponential moving average)
        let alpha = 0.1;
        for (i, s) in self.state.iter_mut().enumerate() {
            let input = features.features[i % 32];
            *s = *s * (1.0 - alpha) + input * alpha;
        }

        self.detect_frame += 1;

        // Need at least some frames and learned patterns for detection
        if self.patterns.is_empty() || self.detect_frame < 10 {
            return RhythmDetection::default();
        }

        // Compare current trajectory against all learned patterns
        let mut similarities = BTreeMap::new();
        let mut best_key = ' ';
        let mut best_sim = -1.0f32;

        for (&key, pattern) in &self.patterns {
            let sim = self.compute_similarity(&self.state, pattern);
            similarities.insert(key, sim);
            if sim > best_sim {
                best_sim = sim;
                best_key = key;
            }
        }

        let confident = best_sim > self.config.confidence_threshold;
        let pattern_name = if confident {
            best_key.to_string()
        } else if best_sim > 0.3 {
            "Uncertain".to_string()
        } else {
            "Unknown".to_string()
        };

        // Simple phi: how differentiated are the similarities?
        let phi = self.compute_phi(&similarities);

        RhythmDetection {
            is_valid: true,
            pattern: pattern_name,
            confidence: best_sim.max(0.0),
            phi,
            similarities,
        }
    }

    /// Get the number of learned patterns
    pub fn learned_pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Get the keys of all learned patterns
    pub fn learned_patterns(&self) -> Vec<char> {
        self.patterns.keys().copied().collect()
    }

    /// Get the frequency signature for a learned pattern
    pub fn get_pattern_frequency_signature(&self, key: char) -> Option<[f32; 4]> {
        self.patterns.get(&key).map(|p| p.frequency_signature)
    }

    /// Estimate BPM for a learned pattern given the capture FPS
    pub fn get_pattern_bpm(&self, key: char, fps: u32) -> Option<f32> {
        let pattern = self.patterns.get(&key)?;
        let period_frames = self.estimate_period(&pattern.trajectory);
        if period_frames > 0.0 {
            Some(60.0 * fps as f32 / period_frames)
        } else {
            None
        }
    }

    /// Get a summary of the LTC network state
    pub fn network_summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!(
            "LTC Rhythm Detector: {} hidden neurons, dt={:.1}ms\n",
            self.config.hidden_size, self.config.dt_ms
        ));
        summary.push_str(&format!("Learned patterns: {}\n", self.patterns.len()));
        for (&key, pattern) in &self.patterns {
            summary.push_str(&format!(
                "  [{}]: {} frames, freq=[{:.2}, {:.2}, {:.2}, {:.2}]\n",
                key,
                pattern.trajectory.len(),
                pattern.frequency_signature[0],
                pattern.frequency_signature[1],
                pattern.frequency_signature[2],
                pattern.frequency_signature[3],
            ));
        }
        summary
    }

    /// Reset detector state: clear all learned patterns and state
    pub fn reset(&mut self) {
        self.patterns.clear();
        self.learning_key = None;
        self.learning_buffer.clear();
        self.state.fill(0.0);
        self.detect_frame = 0;
    }

    // --- Internal helpers ---

    fn compute_similarity(&self, state: &[f32], pattern: &LearnedPattern) -> f32 {
        if pattern.trajectory.is_empty() {
            return 0.0;
        }

        // Compare current state against the average trajectory features
        let avg: Vec<f32> = (0..32)
            .map(|i| {
                pattern.trajectory.iter().map(|t| t[i]).sum::<f32>()
                    / pattern.trajectory.len() as f32
            })
            .collect();

        // Cosine similarity between state (truncated to 32) and average
        let state_32: Vec<f32> = (0..32).map(|i| state[i % state.len()]).collect();

        let dot: f32 = state_32.iter().zip(avg.iter()).map(|(a, b)| a * b).sum();
        let mag_a = state_32.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b = avg.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a > 0.0 && mag_b > 0.0 {
            (dot / (mag_a * mag_b)).max(0.0)
        } else {
            0.0
        }
    }

    fn compute_phi(&self, similarities: &BTreeMap<char, f32>) -> f32 {
        if similarities.len() < 2 {
            return 0.0;
        }

        // Phi as variance of similarities — high variance = good discrimination
        let values: Vec<f32> = similarities.values().copied().collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

        // Scale to 0-1 range
        (variance * 4.0).min(1.0)
    }

    fn compute_frequency_signature(&self, trajectory: &[[f32; 32]]) -> [f32; 4] {
        if trajectory.is_empty() {
            return [0.0; 4];
        }

        // Extract brightness channel (feature 28) for frequency analysis
        let signal: Vec<f32> = trajectory.iter().map(|f| f[28]).collect();
        let n = signal.len() as f32;

        // Simple DFT at target frequencies (0.5, 1, 2, 4 Hz)
        // Assuming dt_ms gives us the sampling rate
        let fs = 1000.0 / self.config.dt_ms;
        let target_freqs = [0.5, 1.0, 2.0, 4.0];
        let mut powers = [0.0f32; 4];

        for (fi, &freq) in target_freqs.iter().enumerate() {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for (i, &val) in signal.iter().enumerate() {
                let phase = 2.0 * std::f32::consts::PI * freq * (i as f32) / fs;
                real += val * phase.cos();
                imag += val * phase.sin();
            }
            powers[fi] = (real * real + imag * imag).sqrt() / n;
        }

        powers
    }

    fn estimate_period(&self, trajectory: &[[f32; 32]]) -> f32 {
        if trajectory.len() < 4 {
            return 0.0;
        }

        // Use autocorrelation on brightness to find dominant period
        let signal: Vec<f32> = trajectory.iter().map(|f| f[28]).collect();
        let n = signal.len();
        let mean = signal.iter().sum::<f32>() / n as f32;

        let mut best_lag = 0;
        let mut best_corr = 0.0f32;

        for lag in 2..(n / 2) {
            let mut corr = 0.0f32;
            let mut count = 0;
            for i in 0..(n - lag) {
                corr += (signal[i] - mean) * (signal[i + lag] - mean);
                count += 1;
            }
            if count > 0 {
                corr /= count as f32;
                if corr > best_corr {
                    best_corr = corr;
                    best_lag = lag;
                }
            }
        }

        best_lag as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = LtcRhythmConfig::default();
        assert!(config.dt_ms > 0.0);
        assert!(config.hidden_size > 0);
    }

    #[test]
    fn test_detector_creation() {
        let detector = LtcRhythmDetector::new(LtcRhythmConfig::default());
        assert_eq!(detector.learned_pattern_count(), 0);
        assert!(detector.learned_patterns().is_empty());
    }

    #[test]
    fn test_detection_without_patterns() {
        let mut detector = LtcRhythmDetector::new(LtcRhythmConfig::default());
        let features = TemporalFeatures::zero();
        let result = detector.process(&features);
        assert!(!result.is_valid || result.pattern == "Unknown");
    }

    #[test]
    fn test_learning_cycle() {
        let mut detector = LtcRhythmDetector::new(LtcRhythmConfig {
            min_learn_frames: 5,
            ..Default::default()
        });

        detector.start_learning('A');
        for i in 0..10 {
            let mut features = TemporalFeatures::zero();
            features.features[28] = (i as f32 * 0.5).sin();
            detector.process(&features);
        }
        let result = detector.stop_learning();
        assert!(result.is_some());
        assert_eq!(detector.learned_pattern_count(), 1);
    }

    #[test]
    fn test_reset() {
        let mut detector = LtcRhythmDetector::new(LtcRhythmConfig {
            min_learn_frames: 5,
            ..Default::default()
        });

        detector.start_learning('A');
        for _ in 0..10 {
            detector.process(&TemporalFeatures::zero());
        }
        detector.stop_learning();

        detector.reset();
        assert_eq!(detector.learned_pattern_count(), 0);
    }

    #[test]
    fn test_network_summary() {
        let detector = LtcRhythmDetector::new(LtcRhythmConfig::default());
        let summary = detector.network_summary();
        assert!(summary.contains("LTC Rhythm Detector"));
    }
}
