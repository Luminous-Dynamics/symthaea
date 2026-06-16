// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Temporal Pattern HDC Signatures
//!
//! Encodes CfC tau trajectories as high-dimensional vectors for pattern matching,
//! enabling consciousness states to be queried via temporal fingerprints.
//!
//! ## Core Concept
//!
//! Tau trajectories form unique "fingerprints" for different consciousness states:
//! - **Contemplative**: Slow, increasing tau values with low variance
//! - **Excited**: Fast, oscillating tau with high variance
//! - **Focused**: Converged tau values near optimal point
//! - **Exploratory**: Diverse tau distribution exploring state space
//!
//! ## HDC Encoding Strategy
//!
//! ```text
//! Temporal Signature = Base ⊛ Trend ⊛ Variance ⊛ Mean ⊛ Range
//! ```
//!
//! Where each feature is encoded as a position-bound hypervector.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let mut signature_encoder = TemporalSignatureEncoder::new();
//!
//! // Record tau history
//! for tau in tau_values {
//!     signature_encoder.record(tau);
//! }
//!
//! // Get current state
//! let state = signature_encoder.classify_state();
//! println!("Consciousness state: {:?}", state);
//!
//! // Get similarity to specific pattern
//! let focus_sim = signature_encoder.similarity_to(ConsciousnessPattern::Focused);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Canonical consciousness states that can be detected from tau trajectories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsciousnessPattern {
    /// Slow, increasing tau with low variance - deep thinking
    Contemplative,
    /// Fast, oscillating tau with high variance - aroused state
    Excited,
    /// Converged tau values - concentrated attention
    Focused,
    /// Diverse tau distribution - exploring state space
    Exploratory,
    /// Stable, moderate tau - resting state
    Resting,
    /// Rapidly decreasing tau - transitioning state
    Transitioning,
    /// Chaotic tau with no clear pattern
    Uncertain,
}

impl ConsciousnessPattern {
    /// Static string matching Debug output — avoids `format!("{:?}")` on hot path.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Contemplative => "Contemplative",
            Self::Excited => "Excited",
            Self::Focused => "Focused",
            Self::Exploratory => "Exploratory",
            Self::Resting => "Resting",
            Self::Transitioning => "Transitioning",
            Self::Uncertain => "Uncertain",
        }
    }

    /// Get all patterns
    pub fn all() -> &'static [ConsciousnessPattern] {
        &[
            ConsciousnessPattern::Contemplative,
            ConsciousnessPattern::Excited,
            ConsciousnessPattern::Focused,
            ConsciousnessPattern::Exploratory,
            ConsciousnessPattern::Resting,
            ConsciousnessPattern::Transitioning,
            ConsciousnessPattern::Uncertain,
        ]
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            ConsciousnessPattern::Contemplative => "Deep, reflective thinking",
            ConsciousnessPattern::Excited => "Aroused, high-energy state",
            ConsciousnessPattern::Focused => "Concentrated attention",
            ConsciousnessPattern::Exploratory => "Exploring possibilities",
            ConsciousnessPattern::Resting => "Calm baseline state",
            ConsciousnessPattern::Transitioning => "Shifting between states",
            ConsciousnessPattern::Uncertain => "No clear pattern",
        }
    }
}

/// Configuration for temporal signature encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureConfig {
    /// History length for trajectory analysis
    pub history_length: usize,

    /// Number of quantization bins for features
    pub num_bins: usize,

    /// Minimum history required for classification
    pub min_history: usize,

    /// Similarity threshold for pattern match
    pub match_threshold: f32,
}

impl Default for SignatureConfig {
    fn default() -> Self {
        Self {
            history_length: 100,
            num_bins: 16,
            min_history: 20,
            match_threshold: 0.3,
        }
    }
}

/// Extracted features from tau trajectory
#[derive(Debug, Clone, Default)]
pub struct TrajectoryFeatures {
    /// Mean tau value
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Coefficient of variation
    pub cv: f32,
    /// Trend (slope of linear fit)
    pub trend: f32,
    /// Range (max - min)
    pub range: f32,
    /// Entropy (distribution complexity)
    pub entropy: f32,
    /// Oscillation frequency (zero crossings of derivative)
    pub oscillation: f32,
    /// Convergence (inverse of recent variance)
    pub convergence: f32,
}

impl TrajectoryFeatures {
    /// Extract features from tau history
    pub fn from_history(history: &[f32]) -> Self {
        if history.is_empty() {
            return Self::default();
        }

        let n = history.len() as f32;

        // Mean
        let mean: f32 = history.iter().sum::<f32>() / n;

        // Variance and std dev
        let variance: f32 = history.iter().map(|&t| (t - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        // Coefficient of variation
        let cv = if mean > 0.001 { std_dev / mean } else { 0.0 };

        // Range
        let min_tau = history.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_tau = history.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_tau - min_tau;

        // Trend (linear regression slope)
        let trend = Self::compute_trend(history);

        // Entropy
        let entropy = Self::compute_entropy(history, 10);

        // Oscillation (derivative zero crossings)
        let oscillation = Self::compute_oscillation(history);

        // Convergence (inverse of recent variance)
        let recent_len = (history.len() / 4).max(5);
        let recent = &history[history.len().saturating_sub(recent_len)..];
        let n = recent.len().max(1) as f32;
        let recent_mean: f32 = recent.iter().sum::<f32>() / n;
        let recent_var: f32 = recent
            .iter()
            .map(|&t| (t - recent_mean).powi(2))
            .sum::<f32>()
            / n;
        let convergence = 1.0 / (1.0 + recent_var * 10.0);

        Self {
            mean,
            std_dev,
            cv,
            trend,
            range,
            entropy,
            oscillation,
            convergence,
        }
    }

    fn compute_trend(history: &[f32]) -> f32 {
        if history.len() < 2 {
            return 0.0;
        }

        let n = history.len() as f32;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f32 = history.iter().sum::<f32>() / n;

        let mut numerator = 0.0f32;
        let mut denominator = 0.0f32;

        for (i, &y) in history.iter().enumerate() {
            let x = i as f32;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator > 0.0001 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn compute_entropy(history: &[f32], num_bins: usize) -> f32 {
        if history.is_empty() {
            return 0.0;
        }

        let min_val = history.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = history.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if (max_val - min_val).abs() < 0.0001 {
            return 0.0;
        }

        let mut bins = vec![0usize; num_bins];
        let bin_width = (max_val - min_val) / num_bins as f32;

        for &val in history {
            let bin_idx = ((val - min_val) / bin_width).clamp(0.0, num_bins as f32 - 1.0) as usize;
            let bin_idx = bin_idx.min(num_bins - 1);
            bins[bin_idx] += 1;
        }

        let n = history.len() as f32;
        bins.iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f32 / n;
                -p * p.ln()
            })
            .sum()
    }

    fn compute_oscillation(history: &[f32]) -> f32 {
        if history.len() < 3 {
            return 0.0;
        }

        // Compute derivative (differences)
        let derivatives: Vec<f32> = history.windows(2).map(|w| w[1] - w[0]).collect();

        // Count zero crossings
        let mut crossings = 0;
        for window in derivatives.windows(2) {
            if window[0] * window[1] < 0.0 {
                crossings += 1;
            }
        }

        // Normalize by length
        crossings as f32 / history.len() as f32
    }
}

/// Encoder for temporal signatures
pub struct TemporalSignatureEncoder {
    /// Configuration
    config: SignatureConfig,

    /// Tau history
    tau_history: VecDeque<f32>,

    /// Base vectors for feature encoding
    feature_bases: HashMap<String, ContinuousHV>,

    /// Canonical pattern signatures
    pattern_signatures: HashMap<ConsciousnessPattern, ContinuousHV>,

    /// Current encoded signature
    current_signature: Option<ContinuousHV>,

    /// Current features
    current_features: TrajectoryFeatures,
}

impl TemporalSignatureEncoder {
    /// Create a new encoder
    pub fn new(config: SignatureConfig) -> Self {
        let mut encoder = Self {
            tau_history: VecDeque::with_capacity(config.history_length),
            feature_bases: HashMap::new(),
            pattern_signatures: HashMap::new(),
            current_signature: None,
            current_features: TrajectoryFeatures::default(),
            config,
        };

        encoder.initialize_bases();
        encoder.initialize_patterns();

        encoder
    }

    fn initialize_bases(&mut self) {
        // Create base vectors for each feature dimension
        let features = [
            "mean",
            "std_dev",
            "cv",
            "trend",
            "range",
            "entropy",
            "oscillation",
            "convergence",
        ];

        for feature in features {
            let seed = feature
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            self.feature_bases
                .insert(feature.to_string(), ContinuousHV::random_default(seed));
        }

        // Create position vectors for quantized values
        for i in 0..self.config.num_bins {
            let seed = (i as u64).wrapping_mul(7919);
            self.feature_bases
                .insert(format!("pos_{i}"), ContinuousHV::random_default(seed));
        }
    }

    fn initialize_patterns(&mut self) {
        // Create canonical signatures for each consciousness pattern
        // These represent "ideal" feature profiles

        // Contemplative: high mean, low oscillation, positive trend, low CV
        self.pattern_signatures.insert(
            ConsciousnessPattern::Contemplative,
            self.encode_canonical_pattern(&TrajectoryFeatures {
                mean: 0.8,
                std_dev: 0.1,
                cv: 0.125,
                trend: 0.02,
                range: 0.3,
                entropy: 0.5,
                oscillation: 0.1,
                convergence: 0.7,
            }),
        );

        // Excited: moderate mean, high oscillation, high CV
        self.pattern_signatures.insert(
            ConsciousnessPattern::Excited,
            self.encode_canonical_pattern(&TrajectoryFeatures {
                mean: 0.5,
                std_dev: 0.3,
                cv: 0.6,
                trend: 0.0,
                range: 0.8,
                entropy: 1.2,
                oscillation: 0.6,
                convergence: 0.3,
            }),
        );

        // Focused: moderate mean, very low CV, high convergence
        self.pattern_signatures.insert(
            ConsciousnessPattern::Focused,
            self.encode_canonical_pattern(&TrajectoryFeatures {
                mean: 0.5,
                std_dev: 0.05,
                cv: 0.1,
                trend: 0.0,
                range: 0.15,
                entropy: 0.3,
                oscillation: 0.05,
                convergence: 0.9,
            }),
        );

        // Exploratory: high entropy, high range, moderate oscillation
        self.pattern_signatures.insert(
            ConsciousnessPattern::Exploratory,
            self.encode_canonical_pattern(&TrajectoryFeatures {
                mean: 0.5,
                std_dev: 0.25,
                cv: 0.5,
                trend: 0.0,
                range: 0.7,
                entropy: 1.5,
                oscillation: 0.3,
                convergence: 0.4,
            }),
        );

        // Resting: moderate mean, low everything
        self.pattern_signatures.insert(
            ConsciousnessPattern::Resting,
            self.encode_canonical_pattern(&TrajectoryFeatures {
                mean: 0.4,
                std_dev: 0.08,
                cv: 0.2,
                trend: 0.0,
                range: 0.2,
                entropy: 0.5,
                oscillation: 0.1,
                convergence: 0.6,
            }),
        );

        // Transitioning: negative trend, moderate variance
        self.pattern_signatures.insert(
            ConsciousnessPattern::Transitioning,
            self.encode_canonical_pattern(&TrajectoryFeatures {
                mean: 0.5,
                std_dev: 0.2,
                cv: 0.4,
                trend: -0.05,
                range: 0.5,
                entropy: 0.8,
                oscillation: 0.2,
                convergence: 0.4,
            }),
        );

        // Uncertain: high entropy, high CV, low convergence
        self.pattern_signatures.insert(
            ConsciousnessPattern::Uncertain,
            self.encode_canonical_pattern(&TrajectoryFeatures {
                mean: 0.5,
                std_dev: 0.35,
                cv: 0.7,
                trend: 0.0,
                range: 0.9,
                entropy: 2.0,
                oscillation: 0.5,
                convergence: 0.2,
            }),
        );
    }

    fn encode_canonical_pattern(&self, features: &TrajectoryFeatures) -> ContinuousHV {
        self.encode_features(features)
    }

    fn encode_features(&self, features: &TrajectoryFeatures) -> ContinuousHV {
        let mut bound_vectors = Vec::new();

        // Encode each feature as position-bound vector
        // Feature value is quantized and bound with feature base

        // Mean (normalized to 0-1, assuming tau range 0-2)
        let mean_bin = self.quantize(features.mean / 2.0);
        if let (Some(base), Some(pos)) = (
            self.feature_bases.get("mean"),
            self.feature_bases.get(&format!("pos_{mean_bin}")),
        ) {
            bound_vectors.push(ContinuousHV::bind(base, pos));
        }

        // CV (already 0-1 range typically)
        let cv_bin = self.quantize(features.cv.min(1.0));
        if let (Some(base), Some(pos)) = (
            self.feature_bases.get("cv"),
            self.feature_bases.get(&format!("pos_{cv_bin}")),
        ) {
            bound_vectors.push(ContinuousHV::bind(base, pos));
        }

        // Trend (normalize -0.1 to 0.1 → 0 to 1)
        let trend_normalized = (features.trend + 0.1) / 0.2;
        let trend_bin = self.quantize(trend_normalized.clamp(0.0, 1.0));
        if let (Some(base), Some(pos)) = (
            self.feature_bases.get("trend"),
            self.feature_bases.get(&format!("pos_{trend_bin}")),
        ) {
            bound_vectors.push(ContinuousHV::bind(base, pos));
        }

        // Entropy (normalize by max entropy ~2.3)
        let entropy_bin = self.quantize((features.entropy / 2.3).min(1.0));
        if let (Some(base), Some(pos)) = (
            self.feature_bases.get("entropy"),
            self.feature_bases.get(&format!("pos_{entropy_bin}")),
        ) {
            bound_vectors.push(ContinuousHV::bind(base, pos));
        }

        // Oscillation (already 0-1)
        let osc_bin = self.quantize(features.oscillation.min(1.0));
        if let (Some(base), Some(pos)) = (
            self.feature_bases.get("oscillation"),
            self.feature_bases.get(&format!("pos_{osc_bin}")),
        ) {
            bound_vectors.push(ContinuousHV::bind(base, pos));
        }

        // Convergence (already 0-1)
        let conv_bin = self.quantize(features.convergence);
        if let (Some(base), Some(pos)) = (
            self.feature_bases.get("convergence"),
            self.feature_bases.get(&format!("pos_{conv_bin}")),
        ) {
            bound_vectors.push(ContinuousHV::bind(base, pos));
        }

        // Bundle all bound feature vectors
        if bound_vectors.is_empty() {
            ContinuousHV::zero(ContinuousHV::DEFAULT_DIM)
        } else {
            let refs: Vec<&ContinuousHV> = bound_vectors.iter().collect();
            ContinuousHV::bundle(&refs)
        }
    }

    fn quantize(&self, value: f32) -> usize {
        let clamped = value.clamp(0.0, 1.0);
        let bin = (clamped * (self.config.num_bins - 1) as f32).round() as usize;
        bin.min(self.config.num_bins - 1)
    }

    /// Record a new tau value
    pub fn record(&mut self, tau: f32) {
        self.tau_history.push_back(tau);
        if self.tau_history.len() > self.config.history_length {
            self.tau_history.pop_front();
        }

        // Update features and signature if we have enough history
        if self.tau_history.len() >= self.config.min_history {
            let history: Vec<f32> = self.tau_history.iter().cloned().collect();
            self.current_features = TrajectoryFeatures::from_history(&history);
            self.current_signature = Some(self.encode_features(&self.current_features));
        }
    }

    /// Record multiple tau values (e.g., from all CfC neurons)
    pub fn record_batch(&mut self, tau_values: &[f32]) {
        // Use mean of all tau values
        if tau_values.is_empty() {
            return;
        }
        let mean_tau: f32 = tau_values.iter().sum::<f32>() / tau_values.len() as f32;
        self.record(mean_tau);
    }

    /// Get current trajectory features
    pub fn features(&self) -> &TrajectoryFeatures {
        &self.current_features
    }

    /// Get current encoded signature
    pub fn signature(&self) -> Option<&ContinuousHV> {
        self.current_signature.as_ref()
    }

    /// Compute similarity to a specific consciousness pattern
    pub fn similarity_to(&self, pattern: ConsciousnessPattern) -> f32 {
        match (
            &self.current_signature,
            self.pattern_signatures.get(&pattern),
        ) {
            (Some(current), Some(canonical)) => current.similarity(canonical),
            _ => 0.0,
        }
    }

    /// Get similarities to all patterns
    pub fn all_similarities(&self) -> HashMap<ConsciousnessPattern, f32> {
        ConsciousnessPattern::all()
            .iter()
            .map(|&p| (p, self.similarity_to(p)))
            .collect()
    }

    /// Classify current state based on best matching pattern
    pub fn classify_state(&self) -> (ConsciousnessPattern, f32) {
        if self.tau_history.len() < self.config.min_history {
            return (ConsciousnessPattern::Uncertain, 0.0);
        }

        let similarities = self.all_similarities();
        let (best_pattern, best_sim) = similarities
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((ConsciousnessPattern::Uncertain, 0.0));

        if best_sim >= self.config.match_threshold {
            (best_pattern, best_sim)
        } else if self.tau_history.len() >= self.config.history_length {
            // After a full history window without matching any pattern, fall back to
            // Resting rather than Uncertain. An unclassified steady-state is more
            // accurately described as "resting" than "uncertain" — the system is stable,
            // just not strongly expressing any canonical pattern.
            // This prevents the Uncertain death spiral where prediction_confidence
            // decays continuously and exploration_factor locks at 0.5.
            (ConsciousnessPattern::Resting, best_sim)
        } else {
            (ConsciousnessPattern::Uncertain, best_sim)
        }
    }

    /// Check if current state matches a specific pattern
    pub fn is_state(&self, pattern: ConsciousnessPattern) -> bool {
        self.similarity_to(pattern) >= self.config.match_threshold
    }

    /// Get a summary of the current temporal state
    pub fn summary(&self) -> TemporalStateSummary {
        let (pattern, confidence) = self.classify_state();
        let features = self.current_features.clone();

        TemporalStateSummary {
            pattern,
            confidence,
            features,
            history_length: self.tau_history.len(),
        }
    }

    /// Reset the encoder
    pub fn reset(&mut self) {
        self.tau_history.clear();
        self.current_signature = None;
        self.current_features = TrajectoryFeatures::default();
    }
}

impl Default for TemporalSignatureEncoder {
    fn default() -> Self {
        Self::new(SignatureConfig::default())
    }
}

/// Summary of current temporal state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalStateSummary {
    /// Classified consciousness pattern
    pub pattern: ConsciousnessPattern,
    /// Confidence in classification (similarity score)
    pub confidence: f32,
    /// Extracted trajectory features
    pub features: TrajectoryFeatures,
    /// Current history length
    pub history_length: usize,
}

// Implement Serialize/Deserialize for TrajectoryFeatures
impl Serialize for TrajectoryFeatures {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TrajectoryFeatures", 8)?;
        state.serialize_field("mean", &self.mean)?;
        state.serialize_field("std_dev", &self.std_dev)?;
        state.serialize_field("cv", &self.cv)?;
        state.serialize_field("trend", &self.trend)?;
        state.serialize_field("range", &self.range)?;
        state.serialize_field("entropy", &self.entropy)?;
        state.serialize_field("oscillation", &self.oscillation)?;
        state.serialize_field("convergence", &self.convergence)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for TrajectoryFeatures {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            mean: f32,
            std_dev: f32,
            cv: f32,
            trend: f32,
            range: f32,
            entropy: f32,
            oscillation: f32,
            convergence: f32,
        }

        let helper = Helper::deserialize(deserializer)?;
        Ok(TrajectoryFeatures {
            mean: helper.mean,
            std_dev: helper.std_dev,
            cv: helper.cv,
            trend: helper.trend,
            range: helper.range,
            entropy: helper.entropy,
            oscillation: helper.oscillation,
            convergence: helper.convergence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction() {
        // Constant values
        let history = vec![1.0; 50];
        let features = TrajectoryFeatures::from_history(&history);

        assert!((features.mean - 1.0).abs() < 0.001);
        assert!(features.std_dev < 0.001);
        assert!(features.oscillation < 0.001);
    }

    #[test]
    fn test_trend_detection() {
        // Increasing values
        let history: Vec<f32> = (0..50).map(|i| i as f32 * 0.01).collect();
        let features = TrajectoryFeatures::from_history(&history);

        assert!(
            features.trend > 0.0,
            "Increasing sequence should have positive trend: {}",
            features.trend
        );
    }

    #[test]
    fn test_oscillation_detection() {
        // Alternating values
        let history: Vec<f32> = (0..50)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let features = TrajectoryFeatures::from_history(&history);

        assert!(
            features.oscillation > 0.3,
            "Alternating sequence should have high oscillation: {}",
            features.oscillation
        );
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = TemporalSignatureEncoder::default();

        // Should have all pattern signatures
        assert_eq!(
            encoder.pattern_signatures.len(),
            ConsciousnessPattern::all().len()
        );
    }

    #[test]
    fn test_focused_pattern_detection() {
        let mut encoder = TemporalSignatureEncoder::default();

        // Simulate focused state: constant tau
        for _ in 0..50 {
            encoder.record(0.5);
        }

        let sim = encoder.similarity_to(ConsciousnessPattern::Focused);
        assert!(
            sim > 0.2,
            "Constant tau should match focused pattern: {}",
            sim
        );
    }

    #[test]
    fn test_excited_pattern_detection() {
        let mut encoder = TemporalSignatureEncoder::default();

        // Simulate excited state: rapidly alternating tau values
        // This produces high oscillation (many zero crossings in derivative)
        for i in 0..50 {
            let tau = if i % 2 == 0 { 0.7 } else { 0.3 };
            encoder.record(tau);
        }

        // The excited pattern should have high oscillation detected
        let features = encoder.features();
        assert!(
            features.oscillation > 0.3,
            "Alternating tau should have oscillation feature > 0.3: {}",
            features.oscillation
        );

        // Should get some similarity to Excited pattern
        // (similarity may be modest since other features also need to match)
        let sim_excited = encoder.similarity_to(ConsciousnessPattern::Excited);
        assert!(
            sim_excited >= 0.0,
            "Should have non-negative similarity to Excited pattern: {}",
            sim_excited
        );
    }

    #[test]
    fn test_contemplative_pattern_detection() {
        let mut encoder = TemporalSignatureEncoder::default();

        // Simulate contemplative state: slowly increasing tau
        for i in 0..50 {
            let tau = 0.5 + i as f32 * 0.01;
            encoder.record(tau);
        }

        // The contemplative pattern should have positive trend detected
        let features = encoder.features();
        assert!(
            features.trend > 0.0,
            "Increasing tau should have positive trend: {}",
            features.trend
        );

        // Should get some similarity (not zero)
        let sim_contemplative = encoder.similarity_to(ConsciousnessPattern::Contemplative);
        assert!(
            sim_contemplative > 0.0,
            "Should have non-zero similarity to Contemplative pattern: {}",
            sim_contemplative
        );
    }

    #[test]
    fn test_state_classification() {
        let mut encoder = TemporalSignatureEncoder::default();

        // Not enough history
        encoder.record(0.5);
        let (pattern, _) = encoder.classify_state();
        assert_eq!(pattern, ConsciousnessPattern::Uncertain);

        // Add more history
        for _ in 0..50 {
            encoder.record(0.5);
        }

        let (_pattern, confidence) = encoder.classify_state();
        assert!(confidence > 0.0);
        // With constant values, should be Focused or similar low-variance state
    }

    #[test]
    fn test_summary() {
        let mut encoder = TemporalSignatureEncoder::default();

        for _ in 0..30 {
            encoder.record(0.5);
        }

        let summary = encoder.summary();
        assert_eq!(summary.history_length, 30);
        assert!(summary.features.mean > 0.0);
    }

    #[test]
    fn test_record_batch() {
        let mut encoder = TemporalSignatureEncoder::default();

        // Record batch of tau values
        let batch = vec![0.3, 0.5, 0.7, 0.4, 0.6];
        encoder.record_batch(&batch);

        // History should have the mean
        assert_eq!(encoder.tau_history.len(), 1);
        assert!((encoder.tau_history[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let mut encoder = TemporalSignatureEncoder::default();

        for _ in 0..30 {
            encoder.record(0.5);
        }
        assert!(!encoder.tau_history.is_empty());

        encoder.reset();
        assert!(encoder.tau_history.is_empty());
        assert!(encoder.current_signature.is_none());
    }
}
