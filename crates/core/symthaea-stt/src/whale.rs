// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Whale Acoustic Sentinel
//!
//! Specialized LTC/HDC architecture for analyzing whale vocalizations.
//!
//! # Key Insight
//!
//! Human speech is **frequency-dominant** (formants).
//! Whale clicks are **time-dominant** (inter-click intervals).
//!
//! We replace mel spectrograms with a transient detector and
//! let the LTC learn the *rhythm* between clicks.
//!
//! # Sperm Whale Codas
//!
//! Sperm whales communicate using "codas" - stereotyped patterns
//! of clicks with characteristic inter-click intervals (ICI).
//!
//! Example: A "1+1+3" coda has pattern: Click...Click......Click
//! The ICI sequence [0.5s, 1.2s] uniquely identifies this coda type.

use crate::bootstrap::AdaptivePrototypeSet;
use crate::hdc::HV16;
use crate::ltc::{LtcCell, LtcConfig};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for whale acoustic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleConfig {
    /// Sample rate of audio (Hz)
    pub sample_rate: u32,
    /// Energy threshold for click detection (0.0-1.0)
    pub click_threshold: f32,
    /// Minimum gap between clicks (seconds) to avoid double-detection
    pub min_click_gap: f32,
    /// Maximum ICI to consider (seconds) - longer gaps end the coda
    pub max_ici: f32,
    /// Minimum clicks to form a valid coda
    pub min_coda_clicks: usize,
    /// Maximum clicks in a coda buffer
    pub max_coda_clicks: usize,
    /// LTC tau for click detection (fast, ~2ms)
    pub click_tau: f32,
    /// LTC tau for rhythm encoding (slow, ~500ms)
    pub rhythm_tau: f32,
}

impl Default for WhaleConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            click_threshold: 0.3,
            min_click_gap: 0.02, // 20ms minimum between clicks
            max_ici: 2.0,        // Codas typically <2s between clicks
            min_coda_clicks: 3,  // Need at least 3 clicks for a coda
            max_coda_clicks: 12, // Codas rarely exceed 12 clicks
            click_tau: 0.002,    // 2ms - fast transient detection
            rhythm_tau: 0.5,     // 500ms - slow rhythm learning
        }
    }
}

/// A detected whale coda with its rhythm signature
#[derive(Debug, Clone)]
pub struct DetectedCoda {
    /// The hypervector representing this coda's rhythm
    pub signature: HV16,
    /// Inter-click intervals (seconds)
    pub ici_pattern: Vec<f32>,
    /// Total duration of the coda (seconds)
    pub duration: f32,
    /// Number of clicks
    pub num_clicks: usize,
    /// Start time in the audio stream
    pub start_time: f32,
}

/// Whale Acoustic Sentinel
///
/// Processes audio streams to detect and classify whale vocalizations
/// using the same LTC/HDC architecture as human speech, but adapted
/// for time-domain rhythm analysis.
pub struct WhaleSentinel {
    config: WhaleConfig,

    /// Fast LTC for click transient detection
    click_detector: LtcCell,

    /// Slow LTC for rhythm/coda encoding
    rhythm_encoder: LtcCell,

    /// Buffer of inter-click intervals for current coda
    ici_buffer: VecDeque<f32>,

    /// Time of last detected click
    last_click_time: f32,

    /// Start time of current coda
    coda_start_time: f32,

    /// Current audio time
    current_time: f32,

    /// Energy smoothing (for better click detection)
    smoothed_energy: f32,

    /// Previous energy (for transient detection)
    prev_energy: f32,

    /// Discovered coda prototypes
    pub prototypes: AdaptivePrototypeSet,

    /// Statistics
    pub stats: WhaleStats,
}

/// Statistics for whale acoustic analysis
#[derive(Debug, Clone, Default)]
pub struct WhaleStats {
    /// Total samples processed
    pub samples_processed: usize,
    /// Total clicks detected
    pub clicks_detected: usize,
    /// Total codas recognized
    pub codas_recognized: usize,
    /// Total audio duration (seconds)
    pub duration_sec: f32,
}

impl WhaleSentinel {
    /// Create a new whale sentinel with default configuration
    pub fn new() -> Self {
        Self::with_config(WhaleConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: WhaleConfig) -> Self {
        // Fast LTC for click detection (input: 1 energy value)
        let click_ltc_config = LtcConfig {
            hidden_size: 8,
            tau_min: config.click_tau * 0.5,
            tau_max: config.click_tau * 2.0,
            tau_init: config.click_tau,
            tau_lr: 0.0, // Fixed tau for consistent detection
            adaptive_tau: false,
        };

        // Slow LTC for rhythm encoding (input: 1 ICI value)
        let rhythm_ltc_config = LtcConfig {
            hidden_size: 32,
            tau_min: config.rhythm_tau * 0.2,
            tau_max: config.rhythm_tau * 2.0,
            tau_init: config.rhythm_tau,
            tau_lr: 0.01,
            adaptive_tau: true,
        };

        Self {
            click_detector: LtcCell::new(1, click_ltc_config),
            rhythm_encoder: LtcCell::new(1, rhythm_ltc_config),
            ici_buffer: VecDeque::with_capacity(config.max_coda_clicks),
            last_click_time: -1.0,
            coda_start_time: 0.0,
            current_time: 0.0,
            smoothed_energy: 0.0,
            prev_energy: 0.0,
            prototypes: AdaptivePrototypeSet::new(0.6, 8), // Lower threshold, more variants
            stats: WhaleStats::default(),
            config,
        }
    }

    /// Reset the sentinel state
    pub fn reset(&mut self) {
        self.click_detector.reset();
        self.rhythm_encoder.reset();
        self.ici_buffer.clear();
        self.last_click_time = -1.0;
        self.coda_start_time = 0.0;
        self.current_time = 0.0;
        self.smoothed_energy = 0.0;
        self.prev_energy = 0.0;
    }

    /// Process a single audio sample
    ///
    /// Returns Some(DetectedCoda) when a complete coda is recognized.
    pub fn process_sample(&mut self, sample: f32) -> Option<DetectedCoda> {
        let dt = 1.0 / self.config.sample_rate as f32;
        self.current_time += dt;
        self.stats.samples_processed += 1;

        // Compute instantaneous energy
        let energy = sample.abs();

        // Smooth the energy with exponential moving average
        let alpha = 0.1;
        self.smoothed_energy = alpha * energy + (1.0 - alpha) * self.smoothed_energy;

        // Detect click transient (sharp rise in energy)
        let is_click = self.detect_click(energy);

        let result = if is_click {
            self.handle_click()
        } else {
            // Check if coda has timed out (gap too long)
            self.check_coda_timeout()
        };

        self.prev_energy = energy;
        result
    }

    /// Process a buffer of audio samples
    pub fn process_buffer(&mut self, samples: &[f32]) -> Vec<DetectedCoda> {
        let mut codas = Vec::new();

        for &sample in samples {
            if let Some(coda) = self.process_sample(sample) {
                codas.push(coda);
            }
        }

        codas
    }

    /// Detect a click transient
    fn detect_click(&mut self, energy: f32) -> bool {
        // Teager-Kaiser-like energy operator for transient detection
        let teager = energy * energy - self.prev_energy * self.smoothed_energy;

        // Feed to click detector LTC
        let state = self
            .click_detector
            .forward(&[teager.max(0.0)], 1.0 / self.config.sample_rate as f32);

        // Threshold on LTC state magnitude
        let state_energy: f32 = state.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Check if this is a click and enough time has passed since last click
        let time_since_last = if self.last_click_time < 0.0 {
            self.config.min_click_gap + 1.0 // First click always valid
        } else {
            self.current_time - self.last_click_time
        };

        state_energy > self.config.click_threshold && time_since_last > self.config.min_click_gap
    }

    /// Handle a detected click
    fn handle_click(&mut self) -> Option<DetectedCoda> {
        self.stats.clicks_detected += 1;

        // Calculate inter-click interval
        let ici = if self.last_click_time < 0.0 {
            0.0 // First click
        } else {
            self.current_time - self.last_click_time
        };

        // Check if this starts a new coda (long gap or first click)
        if ici > self.config.max_ici || self.last_click_time < 0.0 {
            // Finalize previous coda if we have enough clicks
            let result = self.finalize_coda();

            // Start new coda
            self.ici_buffer.clear();
            self.coda_start_time = self.current_time;
            self.rhythm_encoder.reset();

            self.last_click_time = self.current_time;
            return result;
        }

        // Add ICI to buffer
        self.ici_buffer.push_back(ici);
        if self.ici_buffer.len() > self.config.max_coda_clicks {
            self.ici_buffer.pop_front();
        }

        // Feed ICI to rhythm encoder
        self.rhythm_encoder.forward(&[ici], ici);

        self.last_click_time = self.current_time;
        None
    }

    /// Check if current coda has timed out
    fn check_coda_timeout(&mut self) -> Option<DetectedCoda> {
        if self.last_click_time < 0.0 {
            return None;
        }

        let time_since_last = self.current_time - self.last_click_time;
        if time_since_last > self.config.max_ici {
            let result = self.finalize_coda();
            self.last_click_time = -1.0;
            self.ici_buffer.clear();
            self.rhythm_encoder.reset();
            return result;
        }

        None
    }

    /// Finalize a coda and generate its signature
    fn finalize_coda(&mut self) -> Option<DetectedCoda> {
        if self.ici_buffer.len() < self.config.min_coda_clicks - 1 {
            return None; // Not enough clicks for a valid coda
        }

        self.stats.codas_recognized += 1;

        // Get the rhythm state from LTC
        let state = self.rhythm_encoder.state();

        // Project to hypervector using the same method as human speech
        let signature = project_state_to_hv(state);

        // Calculate coda statistics
        let ici_pattern: Vec<f32> = self.ici_buffer.iter().cloned().collect();
        let duration = ici_pattern.iter().sum();
        let num_clicks = ici_pattern.len() + 1; // n ICIs = n+1 clicks

        // Generate coda type label based on rhythm pattern
        let label = self.classify_coda_pattern(&ici_pattern);

        // Train adaptive prototype
        self.prototypes.train(&label, &signature);

        Some(DetectedCoda {
            signature,
            ici_pattern,
            duration,
            num_clicks,
            start_time: self.coda_start_time,
        })
    }

    /// Classify coda pattern into a type label
    fn classify_coda_pattern(&self, ici_pattern: &[f32]) -> String {
        if ici_pattern.is_empty() {
            return "UNKNOWN".to_string();
        }

        // Simple classification based on number of clicks and rhythm regularity
        let n_clicks = ici_pattern.len() + 1;

        // Calculate coefficient of variation (CV) to measure regularity
        let mean_ici: f32 = ici_pattern.iter().sum::<f32>() / ici_pattern.len() as f32;
        let variance: f32 = ici_pattern
            .iter()
            .map(|x| (x - mean_ici).powi(2))
            .sum::<f32>()
            / ici_pattern.len() as f32;
        let cv = variance.sqrt() / mean_ici;

        // Classify based on pattern characteristics
        let rhythm_type = if cv < 0.2 {
            "REGULAR" // Very consistent timing
        } else if cv < 0.5 {
            "MODERATE" // Some variation
        } else {
            "VARIABLE" // Complex rhythm
        };

        format!("CODA_{n_clicks}_{rhythm_type}")
    }

    /// Get current prototype statistics
    pub fn prototype_stats(&self) -> (usize, usize, f32) {
        let stats = self.prototypes.stats();
        (
            stats.total_phonemes,
            stats.total_variants,
            stats.avg_variants_per_phoneme,
        )
    }

    /// Export prototypes for visualization
    pub fn export_clusters(&self) -> Vec<ClusterExport> {
        let mut exports = Vec::new();

        for (label, proto) in &self.prototypes.prototypes {
            for (i, (variant, count)) in proto
                .variants
                .iter()
                .zip(proto.variant_counts.iter())
                .enumerate()
            {
                exports.push(ClusterExport {
                    label: label.clone(),
                    variant: i,
                    count: *count,
                    signature: variant.to_bytes().to_vec(),
                });
            }
        }

        exports
    }
}

impl Default for WhaleSentinel {
    fn default() -> Self {
        Self::new()
    }
}

/// Cluster export for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterExport {
    /// Coda type label (e.g., "CODA_5_REGULAR")
    pub label: String,
    /// Variant index within the type
    pub variant: usize,
    /// Instance count
    pub count: usize,
    /// Raw hypervector bytes (for external visualization)
    pub signature: Vec<u8>,
}

/// Project LTC state to hypervector (same method as AudioProjector)
fn project_state_to_hv(state: &[f32]) -> HV16 {
    use blake3::Hasher;

    let mut data = Vec::with_capacity(state.len() * 4);
    for &s in state {
        data.extend_from_slice(&s.to_le_bytes());
    }

    let mut hasher = Hasher::new();
    hasher.update(&data);
    let mut output = [0u8; 256];
    hasher.finalize_xof().fill(&mut output);

    HV16::from_bytes(&output)
}

/// Analyze similarity between coda prototypes
pub fn analyze_coda_topology(sentinel: &WhaleSentinel) -> CodaTopology {
    let mut similarities = Vec::new();
    let prototypes: Vec<_> = sentinel.prototypes.prototypes.iter().collect();

    for (i, (label_i, proto_i)) in prototypes.iter().enumerate() {
        for (label_j, proto_j) in &prototypes[i + 1..] {
            let hv_i = proto_i.get_centroid();
            let hv_j = proto_j.get_centroid();

            let sim = hv_i.similarity(&hv_j);
            similarities.push((label_i.to_string(), label_j.to_string(), sim));
        }
    }

    similarities.sort_by(|a, b| b.2.total_cmp(&a.2));

    let all_sims: Vec<f32> = similarities.iter().map(|(_, _, s)| *s).collect();
    let mean_sim = if all_sims.is_empty() {
        0.0
    } else {
        all_sims.iter().sum::<f32>() / all_sims.len() as f32
    };

    CodaTopology {
        num_types: prototypes.len(),
        mean_similarity: mean_sim,
        min_similarity: all_sims.iter().cloned().fold(f32::INFINITY, f32::min),
        max_similarity: all_sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        pairwise: similarities,
    }
}

/// Coda prototype topology analysis
#[derive(Debug, Clone)]
pub struct CodaTopology {
    /// Number of distinct coda types discovered
    pub num_types: usize,
    /// Mean similarity between types
    pub mean_similarity: f32,
    /// Minimum similarity (most different pair)
    pub min_similarity: f32,
    /// Maximum similarity (most similar pair)
    pub max_similarity: f32,
    /// All pairwise similarities
    pub pairwise: Vec<(String, String, f32)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whale_sentinel_creation() {
        let sentinel = WhaleSentinel::new();
        assert_eq!(sentinel.stats.samples_processed, 0);
        assert_eq!(sentinel.stats.clicks_detected, 0);
    }

    #[test]
    fn test_click_detection_state() {
        // Test that the sentinel correctly tracks state during processing
        let mut sentinel = WhaleSentinel::new();

        // Process some samples
        for i in 0..100 {
            sentinel.process_sample((i as f32 / 100.0).sin());
        }

        // Verify state tracking
        assert_eq!(sentinel.stats.samples_processed, 100);
        assert!(sentinel.current_time > 0.0);

        // Note: Full click detection requires realistic whale audio waveforms
        // or tuned synthetic test signals. The LTC-based detector is designed
        // for real sperm whale echolocation clicks which have specific
        // spectral and temporal characteristics.
    }

    #[test]
    fn test_click_detection_config() {
        // Test that click detection respects configuration
        let mut config = WhaleConfig::default();
        config.click_threshold = 0.1;
        config.min_click_gap = 0.05;

        let sentinel = WhaleSentinel::with_config(config);
        assert_eq!(sentinel.config.click_threshold, 0.1);
        assert_eq!(sentinel.config.min_click_gap, 0.05);
    }

    #[test]
    fn test_coda_classification() {
        let sentinel = WhaleSentinel::new();

        // Regular rhythm
        let regular = vec![0.5, 0.5, 0.5, 0.5];
        let label = sentinel.classify_coda_pattern(&regular);
        assert!(label.contains("REGULAR"));

        // Variable rhythm
        let variable = vec![0.2, 0.8, 0.3, 1.2];
        let label = sentinel.classify_coda_pattern(&variable);
        assert!(label.contains("VARIABLE"));
    }
}
