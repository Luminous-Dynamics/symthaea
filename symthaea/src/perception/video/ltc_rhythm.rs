//! LTC-Based Rhythm Detection - True Temporal Dynamics
//!
//! This module wraps the HierarchicalCantorLTC for rhythm/temporal pattern detection.
//! Unlike simple template matching, this uses continuous-time neural dynamics to
//! naturally encode the *timing* of inputs, not just their values.
//!
//! ## Key Insight
//!
//! The LTC network has different time constants (τ) at each level:
//! - Level 0 (root): τ = 1000ms - integrates over ~1 second
//! - Level 3: τ = 37ms - sensitive to ~30Hz rhythms
//! - Level 5: τ = 4ms - sensitive to fast transients
//!
//! When you feed a 60 BPM rhythm (1 beat/sec) vs 120 BPM (2 beats/sec):
//! - The states evolve along DIFFERENT trajectories
//! - The attractor shapes are topologically distinct
//! - Similarity is computed over trajectory shape, not just endpoints
//!
//! This is what makes Symthaea fundamentally different from transformers:
//! we "live in time" rather than "spatializing time."

use symthaea_core::hdc::{RealHV, HDC_DIMENSION};
use crate::hierarchical_cantor_ltc::{HierarchicalCantorLtcNetwork, CantorLtcConfig};
use super::temporal_features::TemporalFeatures;
use std::collections::HashMap;

/// Configuration for LTC rhythm detection
#[derive(Clone, Debug)]
pub struct LtcRhythmConfig {
    /// LTC network configuration
    pub ltc_config: CantorLtcConfig,
    /// Timestep in milliseconds (should match frame rate)
    pub dt_ms: f32,
    /// Number of trajectory samples to keep for comparison
    pub trajectory_length: usize,
    /// Minimum trajectory length before detection is valid
    pub min_trajectory_length: usize,
    /// Detection threshold (trajectory similarity)
    pub detection_threshold: f32,
}

impl Default for LtcRhythmConfig {
    fn default() -> Self {
        // Configure for rhythm detection (60-120 BPM range)
        let mut ltc_config = CantorLtcConfig::default();
        // Use smaller network for speed (4 levels instead of 7)
        ltc_config.max_depth = 4;
        // Adjust base τ for rhythm range (500ms = sensitive to 2Hz = 120 BPM)
        ltc_config.base_tau = 500.0;

        Self {
            ltc_config,
            dt_ms: 33.3, // ~30 fps
            trajectory_length: 90, // 3 seconds at 30fps
            min_trajectory_length: 30, // 1 second minimum
            detection_threshold: 0.6,
        }
    }
}

/// A learned rhythm pattern (trajectory prototype)
#[derive(Clone)]
pub struct RhythmPrototype {
    /// Name/label for this rhythm
    pub name: String,
    /// Root state trajectory
    pub root_trajectory: Vec<RealHV>,
    /// Velocity trajectory (state changes between frames)
    pub velocity_trajectory: Vec<RealHV>,
    /// Multi-level trajectory fingerprint (sampled states at key levels)
    pub level_fingerprints: Vec<Vec<RealHV>>,
    /// Mean state across trajectory
    pub mean_state: RealHV,
    /// Mean velocity (average rate of change)
    pub mean_velocity: RealHV,
    /// Variance signature (how much the state changes)
    pub variance_signature: f32,
    /// Velocity variance (how variable the rate of change is)
    pub velocity_variance: f32,
    /// Frequency signature (power at key frequencies: 0.5Hz, 1Hz, 2Hz, 4Hz)
    pub frequency_signature: [f32; 4],
    /// Detected rhythm period in frames (0 if unknown)
    pub period_frames: usize,
    /// Peak-to-peak amplitude (max - min of summary signal)
    pub amplitude: f32,
}

impl RhythmPrototype {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            root_trajectory: Vec::new(),
            velocity_trajectory: Vec::new(),
            level_fingerprints: Vec::new(),
            mean_state: RealHV::zero(HDC_DIMENSION),
            mean_velocity: RealHV::zero(HDC_DIMENSION),
            variance_signature: 0.0,
            velocity_variance: 0.0,
            frequency_signature: [0.0; 4],
            period_frames: 0,
            amplitude: 0.0,
        }
    }

    /// Check if prototype has enough data
    pub fn is_ready(&self, min_length: usize) -> bool {
        self.root_trajectory.len() >= min_length
    }

    /// Compute trajectory similarity to another prototype
    ///
    /// This uses multiple signals for discrimination:
    /// 1. Position trajectory shape (where the state goes)
    /// 2. Velocity trajectory shape (how fast it changes)
    /// 3. Frequency signature (what frequencies dominate)
    /// 4. Amplitude and variance characteristics
    pub fn trajectory_similarity(&self, other: &RhythmPrototype) -> f32 {
        if self.root_trajectory.is_empty() || other.root_trajectory.is_empty() {
            return 0.0;
        }

        // Method 1: Position trajectory shape similarity
        let position_sim = self.compute_shape_similarity(other);

        // Method 2: Velocity trajectory similarity (KEY for rhythm discrimination)
        let velocity_sim = self.compute_velocity_similarity(other);

        // Method 3: Frequency signature match (strongest discriminator for BPM)
        let freq_sim = self.compute_frequency_similarity(other);

        // Method 4: Amplitude/variance characteristics
        let amp_diff = (self.amplitude - other.amplitude).abs();
        let amp_sim = 1.0 / (1.0 + amp_diff * 0.1);

        let var_diff = (self.variance_signature - other.variance_signature).abs();
        let var_sim = 1.0 / (1.0 + var_diff * 10.0);

        // Weighted combination:
        // - Frequency signature is most discriminative for different BPMs
        // - Velocity shape captures timing differences
        // - Position shape provides baseline consistency
        // - Amplitude/variance handle intensity differences
                // Amplitude characteristics

        0.35 * freq_sim +      // Frequency is king for rhythm
            0.30 * velocity_sim +  // Velocity captures timing
            0.20 * position_sim +  // Position for consistency
            0.10 * var_sim +       // Variance signature
            0.05 * amp_sim
    }

    /// Compute velocity-based trajectory similarity
    ///
    /// Different rhythms have different velocity profiles even when
    /// positions overlap. Fast rhythms have sharper velocity spikes.
    fn compute_velocity_similarity(&self, other: &RhythmPrototype) -> f32 {
        if self.velocity_trajectory.len() < 2 || other.velocity_trajectory.len() < 2 {
            return 0.5; // Neutral if no velocity data
        }

        // Compare mean velocities
        let mean_vel_sim = self.mean_velocity.similarity(&other.mean_velocity);

        // Compare velocity variance (fast rhythms have higher variance)
        let vel_var_diff = (self.velocity_variance - other.velocity_variance).abs();
        let vel_var_sim = 1.0 / (1.0 + vel_var_diff * 5.0);

        // Sample velocity trajectories and compare shapes
        let n_self = self.velocity_trajectory.len();
        let n_other = other.velocity_trajectory.len();
        let n_samples = 10.min(n_self).min(n_other);

        let mut shape_sim = 0.0;
        for i in 0..n_samples {
            let idx_self = i * n_self / n_samples;
            let idx_other = i * n_other / n_samples;
            shape_sim += self.velocity_trajectory[idx_self].similarity(&other.velocity_trajectory[idx_other]);
        }
        shape_sim /= n_samples as f32;

        // Combine: variance similarity is most important for rhythm
        0.3 * mean_vel_sim + 0.4 * vel_var_sim + 0.3 * shape_sim
    }

    /// Compute frequency signature similarity
    ///
    /// This is the most discriminative feature for different BPMs.
    /// 60 BPM has power at 1 Hz, 120 BPM has power at 2 Hz.
    fn compute_frequency_similarity(&self, other: &RhythmPrototype) -> f32 {
        // Normalize signatures
        let self_sum: f32 = self.frequency_signature.iter().sum::<f32>().max(1e-6);
        let other_sum: f32 = other.frequency_signature.iter().sum::<f32>().max(1e-6);

        let self_norm: Vec<f32> = self.frequency_signature.iter().map(|x| x / self_sum).collect();
        let other_norm: Vec<f32> = other.frequency_signature.iter().map(|x| x / other_sum).collect();

        // Compute cosine similarity on normalized frequency distributions
        let dot: f32 = self_norm.iter().zip(other_norm.iter()).map(|(a, b)| a * b).sum();
        let mag_self: f32 = self_norm.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_other: f32 = other_norm.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_self < 1e-6 || mag_other < 1e-6 {
            return 0.5;
        }

        dot / (mag_self * mag_other)
    }

    /// Compute similarity based on trajectory shape
    ///
    /// This is the key innovation: we compare the SHAPE of the trajectory,
    /// not just the endpoint. Different rhythms create different shapes.
    fn compute_shape_similarity(&self, other: &RhythmPrototype) -> f32 {
        let n_self = self.root_trajectory.len();
        let n_other = other.root_trajectory.len();

        if n_self < 2 || n_other < 2 {
            return 0.0;
        }

        // Sample both trajectories at same number of points
        let n_samples = 10.min(n_self).min(n_other);

        let self_samples: Vec<&RealHV> = (0..n_samples)
            .map(|i| {
                let idx = i * n_self / n_samples;
                &self.root_trajectory[idx]
            })
            .collect();

        let other_samples: Vec<&RealHV> = (0..n_samples)
            .map(|i| {
                let idx = i * n_other / n_samples;
                &other.root_trajectory[idx]
            })
            .collect();

        // Compute similarity at each sample point
        let mut total_sim = 0.0;
        for (s1, s2) in self_samples.iter().zip(other_samples.iter()) {
            total_sim += s1.similarity(s2);
        }

        total_sim / n_samples as f32
    }

    /// Finalize prototype after learning (compute statistics)
    fn finalize(&mut self) {
        if self.root_trajectory.is_empty() {
            return;
        }

        let n = self.root_trajectory.len() as f32;

        // 1. Compute velocity trajectory (state differences)
        self.velocity_trajectory.clear();
        for i in 1..self.root_trajectory.len() {
            let mut velocity = RealHV::zero(HDC_DIMENSION);
            for (j, (curr, prev)) in self.root_trajectory[i].values.iter()
                .zip(self.root_trajectory[i-1].values.iter())
                .enumerate()
            {
                velocity.values[j] = curr - prev;
            }
            self.velocity_trajectory.push(velocity);
        }

        // 2. Compute mean state
        let mut sum = RealHV::zero(HDC_DIMENSION);
        for state in &self.root_trajectory {
            for (i, v) in state.values.iter().enumerate() {
                sum.values[i] += v;
            }
        }
        for v in sum.values.iter_mut() {
            *v /= n;
        }
        self.mean_state = sum;

        // 3. Compute mean velocity
        if !self.velocity_trajectory.is_empty() {
            let mut vel_sum = RealHV::zero(HDC_DIMENSION);
            for vel in &self.velocity_trajectory {
                for (i, v) in vel.values.iter().enumerate() {
                    vel_sum.values[i] += v;
                }
            }
            let nv = self.velocity_trajectory.len() as f32;
            for v in vel_sum.values.iter_mut() {
                *v /= nv;
            }
            self.mean_velocity = vel_sum;

            // 4. Compute velocity variance
            let mut vel_var = 0.0;
            for vel in &self.velocity_trajectory {
                for (i, v) in vel.values.iter().enumerate() {
                    let diff = v - self.mean_velocity.values[i];
                    vel_var += diff * diff;
                }
            }
            self.velocity_variance = (vel_var / (nv * HDC_DIMENSION as f32)).sqrt();
        }

        // 5. Compute position variance signature
        let mut total_var = 0.0;
        for state in &self.root_trajectory {
            for (i, v) in state.values.iter().enumerate() {
                let diff = v - self.mean_state.values[i];
                total_var += diff * diff;
            }
        }
        self.variance_signature = (total_var / (n * HDC_DIMENSION as f32)).sqrt();

        // 6. Compute frequency signature using DFT at key frequencies
        self.compute_frequency_signature();

        // 7. Compute amplitude (peak-to-peak of summary signal)
        self.compute_amplitude();

        // 8. Detect period via improved autocorrelation
        self.period_frames = self.detect_period();
    }

    /// Compute frequency signature using DFT at key frequencies
    ///
    /// Target frequencies (at 30 fps):
    /// - 0.5 Hz (30 BPM) - very slow
    /// - 1.0 Hz (60 BPM) - Pattern A typical
    /// - 2.0 Hz (120 BPM) - Pattern B typical
    /// - 4.0 Hz (240 BPM) - fast patterns
    fn compute_frequency_signature(&mut self) {
        if self.root_trajectory.len() < 30 {
            return;
        }

        // Use summary signal (sum of first 100 dimensions)
        let signal: Vec<f32> = self.root_trajectory.iter()
            .map(|state| state.values.iter().take(100).sum::<f32>())
            .collect();

        let n = signal.len();
        let mean: f32 = signal.iter().sum::<f32>() / n as f32;
        let centered: Vec<f32> = signal.iter().map(|x| x - mean).collect();

        // Assume 30 fps - target frequencies in Hz
        let fps = 30.0f32;
        let target_freqs = [0.5, 1.0, 2.0, 4.0]; // Hz

        for (fi, &freq) in target_freqs.iter().enumerate() {
            // DFT at this frequency
            let omega = 2.0 * std::f32::consts::PI * freq / fps;
            let mut real_sum = 0.0f32;
            let mut imag_sum = 0.0f32;

            for (i, &x) in centered.iter().enumerate() {
                let angle = omega * i as f32;
                real_sum += x * angle.cos();
                imag_sum += x * angle.sin();
            }

            // Power at this frequency
            let power = (real_sum * real_sum + imag_sum * imag_sum).sqrt() / n as f32;
            self.frequency_signature[fi] = power;
        }
    }

    /// Compute peak-to-peak amplitude of the trajectory
    fn compute_amplitude(&mut self) {
        if self.root_trajectory.is_empty() {
            return;
        }

        let signal: Vec<f32> = self.root_trajectory.iter()
            .map(|state| state.values.iter().take(100).sum::<f32>())
            .collect();

        let max_val = signal.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = signal.iter().cloned().fold(f32::INFINITY, f32::min);
        self.amplitude = max_val - min_val;
    }

    /// Detect rhythmic period via autocorrelation
    fn detect_period(&self) -> usize {
        if self.root_trajectory.len() < 20 {
            return 0;
        }

        // Use first 100 dimensions as a "summary signal"
        let n = self.root_trajectory.len();
        let signal: Vec<f32> = self.root_trajectory.iter()
            .map(|state| state.values.iter().take(100).sum::<f32>())
            .collect();

        let mean: f32 = signal.iter().sum::<f32>() / n as f32;
        let centered: Vec<f32> = signal.iter().map(|&x| x - mean).collect();

        // Find first peak in autocorrelation (after lag 0)
        let mut best_lag = 0;
        let mut best_corr = 0.0;

        for lag in 5..n/2 {
            let mut sum = 0.0;
            for i in 0..(n - lag) {
                sum += centered[i] * centered[i + lag];
            }
            let corr = sum / (n - lag) as f32;

            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        best_lag
    }
}

/// Detection result from the rhythm detector
#[derive(Clone, Debug)]
pub struct RhythmDetection {
    /// Detected pattern name (or "Unknown")
    pub pattern: String,
    /// Confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Similarity to pattern A (legacy, for backward compatibility)
    pub similarity_a: f32,
    /// Similarity to pattern B (legacy, for backward compatibility)
    pub similarity_b: f32,
    /// Similarities to all learned patterns
    pub similarities: HashMap<char, f32>,
    /// Current Φ (integrated information)
    pub phi: f64,
    /// Is detection valid (enough trajectory data)?
    pub is_valid: bool,
}

/// LTC-based rhythm detector
///
/// Uses the HierarchicalCantorLTC to learn and detect temporal patterns.
/// The key difference from template matching is that this encodes the
/// *dynamics* of the input, not just accumulating statistics.
///
/// Supports arbitrary number of patterns (A-Z).
pub struct LtcRhythmDetector {
    /// Configuration
    config: LtcRhythmConfig,
    /// The LTC network
    ltc: HierarchicalCantorLtcNetwork,
    /// Learned patterns (keyed by character A-Z)
    patterns: HashMap<char, RhythmPrototype>,
    /// Current trajectory being recorded
    current_trajectory: Vec<RealHV>,
    /// Frame counter
    frame_count: u64,
    /// Is currently learning?
    learning_mode: Option<char>,
}

impl LtcRhythmDetector {
    /// Create a new LTC rhythm detector
    pub fn new(config: LtcRhythmConfig) -> Self {
        let ltc = HierarchicalCantorLtcNetwork::new(config.ltc_config.clone());

        Self {
            config,
            ltc,
            patterns: HashMap::new(),
            current_trajectory: Vec::new(),
            frame_count: 0,
            learning_mode: None,
        }
    }

    /// Process a frame's temporal features through the LTC
    pub fn process(&mut self, features: &TemporalFeatures) -> RhythmDetection {
        // Project features to HDC space
        let input = self.project_features(features);

        // Step the LTC network with this input
        self.ltc.step_with_input(self.config.dt_ms, &input);

        // Record trajectory
        let root_state = self.ltc.get_root_state().clone();
        self.current_trajectory.push(root_state.clone());

        // Trim trajectory to max length
        if self.current_trajectory.len() > self.config.trajectory_length {
            self.current_trajectory.remove(0);
        }

        self.frame_count += 1;

        // If learning, record to the target pattern's prototype
        if let Some(pattern_key) = self.learning_mode {
            if let Some(proto) = self.patterns.get_mut(&pattern_key) {
                proto.root_trajectory.push(root_state);
            }
        }

        // Compute detection
        self.detect()
    }

    /// Project 32D temporal features to 16,384D HDC space
    fn project_features(&self, features: &TemporalFeatures) -> RealHV {
        let mut hdc = RealHV::zero(HDC_DIMENSION);

        // Sparse random projection (Johnson-Lindenstrauss)
        for (i, &f) in features.features.iter().enumerate() {
            let seed = (i as u64).wrapping_mul(0x5851F42D4C957F2D);
            for d in 0..HDC_DIMENSION {
                let hash = seed.wrapping_mul(d as u64 + 1).wrapping_add(0x14057B7EF767814F);
                let sign = if hash.is_multiple_of(2) { 1.0 } else { -1.0 };
                // Sparse: ~10% of dimensions active per feature
                if hash.is_multiple_of(10) {
                    hdc.values[d] += sign * f;
                }
            }
        }

        // Normalize
        let magnitude: f32 = hdc.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in hdc.values.iter_mut() {
                *v /= magnitude;
            }
        }

        hdc
    }

    /// Detect which pattern (if any) matches the current trajectory
    fn detect(&mut self) -> RhythmDetection {
        let is_valid = self.current_trajectory.len() >= self.config.min_trajectory_length;

        if !is_valid {
            return RhythmDetection {
                pattern: "Collecting...".to_string(),
                confidence: 0.0,
                similarity_a: 0.0,
                similarity_b: 0.0,
                similarities: HashMap::new(),
                phi: 0.0,
                is_valid: false,
            };
        }

        // Create temporary prototype from current trajectory
        let mut current = RhythmPrototype::new("current");
        current.root_trajectory = self.current_trajectory.clone();
        current.finalize();

        // Compute similarities to all learned patterns
        let mut similarities = HashMap::new();
        for (key, proto) in &self.patterns {
            if proto.is_ready(self.config.min_trajectory_length) {
                let sim = proto.trajectory_similarity(&current);
                similarities.insert(*key, sim);
            }
        }

        // Legacy A/B values for backward compatibility
        let sim_a = similarities.get(&'A').copied().unwrap_or(0.0);
        let sim_b = similarities.get(&'B').copied().unwrap_or(0.0);

        // Compute Φ
        let phi = self.ltc.compute_hierarchical_phi();

        // Find best matching pattern
        let (pattern, confidence) = if let Some((&best_key, &best_sim)) = similarities.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            // Check if it's above threshold
            if best_sim > self.config.detection_threshold {
                (best_key.to_string(), best_sim)
            } else if best_sim > self.config.detection_threshold * 0.8 {
                ("Uncertain".to_string(), best_sim)
            } else {
                ("Unknown".to_string(), best_sim)
            }
        } else {
            ("None".to_string(), 0.0)
        };

        RhythmDetection {
            pattern,
            confidence,
            similarity_a: sim_a,
            similarity_b: sim_b,
            similarities,
            phi,
            is_valid: true,
        }
    }

    /// Start learning a pattern
    pub fn start_learning(&mut self, pattern: char) {
        self.learning_mode = Some(pattern);
        self.current_trajectory.clear();

        // Create or reset the target prototype
        self.patterns.insert(pattern, RhythmPrototype::new(&pattern.to_string()));

        // Reset LTC state for clean learning
        self.ltc = HierarchicalCantorLtcNetwork::new(self.config.ltc_config.clone());
    }

    /// Stop learning and finalize the prototype
    pub fn stop_learning(&mut self) -> Option<(char, usize, usize)> {
        if let Some(pattern) = self.learning_mode.take() {
            let (frames, period) = if let Some(proto) = self.patterns.get_mut(&pattern) {
                proto.finalize();
                (proto.root_trajectory.len(), proto.period_frames)
            } else {
                (0, 0)
            };

            // Clear current trajectory for fresh detection
            self.current_trajectory.clear();

            Some((pattern, frames, period))
        } else {
            None
        }
    }

    /// Check if a pattern is ready for detection
    pub fn pattern_ready(&self, pattern: char) -> bool {
        self.patterns.get(&pattern)
            .map(|p| p.is_ready(self.config.min_trajectory_length))
            .unwrap_or(false)
    }

    /// Get number of learned patterns
    pub fn learned_pattern_count(&self) -> usize {
        self.patterns.values()
            .filter(|p| p.is_ready(self.config.min_trajectory_length))
            .count()
    }

    /// Get all learned pattern keys
    pub fn learned_patterns(&self) -> Vec<char> {
        self.patterns.keys().copied().collect()
    }

    /// Reset everything
    pub fn reset(&mut self) {
        self.patterns.clear();
        self.current_trajectory.clear();
        self.frame_count = 0;
        self.learning_mode = None;
        self.ltc = HierarchicalCantorLtcNetwork::new(self.config.ltc_config.clone());
    }

    /// Get network summary
    pub fn network_summary(&self) -> String {
        format!("{}", self.ltc.summary())
    }

    /// Get detected period in BPM (if available)
    pub fn get_pattern_bpm(&self, pattern: char, fps: u32) -> Option<f32> {
        let period = self.patterns.get(&pattern)
            .map(|p| p.period_frames)
            .unwrap_or(0);

        if period > 0 {
            let period_seconds = period as f32 / fps as f32;
            Some(60.0 / period_seconds)
        } else {
            None
        }
    }

    /// Get the frequency signature for a pattern (for display/debugging)
    pub fn get_pattern_frequency_signature(&self, pattern: char) -> Option<[f32; 4]> {
        self.patterns.get(&pattern).map(|p| p.frequency_signature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::temporal_features::TemporalFeatures;

    fn create_test_features(brightness: f32, seq: u64) -> TemporalFeatures {
        let mut features = [0.0f32; 32];
        features[28] = brightness;
        features[30] = if seq % 2 == 0 { 0.5 } else { 0.0 }; // Alternating motion

        TemporalFeatures {
            features,
            sequence: seq,
            brightness,
            edge_density: 0.3,
            motion: features[30],
            variance: 0.1,
        }
    }

    #[test]
    fn test_detector_creation() {
        let config = LtcRhythmConfig::default();
        let detector = LtcRhythmDetector::new(config);
        assert!(!detector.pattern_ready('A'));
        assert!(!detector.pattern_ready('B'));
    }

    #[test]
    fn test_learning_mode() {
        let config = LtcRhythmConfig::default();
        let mut detector = LtcRhythmDetector::new(config);

        detector.start_learning('A');

        // Simulate some frames
        for i in 0..50 {
            let brightness = if i % 30 < 15 { 0.8 } else { 0.2 }; // ~60 BPM at 30fps
            let features = create_test_features(brightness, i);
            detector.process(&features);
        }

        let result = detector.stop_learning();
        assert!(result.is_some());
        let (pattern, frames, _) = result.unwrap();
        assert_eq!(pattern, 'A');
        assert_eq!(frames, 50);
    }

    #[test]
    fn test_detection() {
        let mut config = LtcRhythmConfig::default();
        config.min_trajectory_length = 10;
        let mut detector = LtcRhythmDetector::new(config);

        // Process enough frames to get valid detection
        for i in 0..30 {
            let features = create_test_features(0.5, i);
            let result = detector.process(&features);

            if i >= 10 {
                assert!(result.is_valid);
            }
        }
    }
}
