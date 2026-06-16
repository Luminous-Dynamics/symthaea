// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # CfC Temporal Coherence and Bidirectional Feedback
//!
//! This module implements bidirectional feedback between CfC temporal dynamics
//! and consciousness metrics, enabling self-regulating learning.
//!
//! ## Core Concept
//!
//! Temporal coherence measures how "organized" the CfC's time constants are:
//! - High coherence = neurons operating at similar timescales = focused processing
//! - Low coherence = diverse timescales = exploratory processing
//!
//! ## Feedback Loops
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  BIDIRECTIONAL FEEDBACK                         │
//! │                                                                 │
//! │   CfC tau values ──► Temporal Coherence ──► Phi Contribution   │
//! │         ▲                    │                     │            │
//! │         │                    ▼                     ▼            │
//! │   Weight Updates ◄── Learning Rate ◄── Consciousness Level     │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Metrics
//!
//! - `tau_coherence`: 1.0 / (1 + coefficient_of_variation(tau))
//! - `temporal_focus`: entropy-based measure of tau distribution
//! - `learning_rate_factor`: 0.5 + coherence * 1.5 (range: 0.5 to 2.0)

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for coherence-modulated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceConfig {
    /// Base learning rate (before modulation)
    pub base_learning_rate: f32,

    /// Minimum learning rate factor
    pub min_lr_factor: f32,

    /// Maximum learning rate factor
    pub max_lr_factor: f32,

    /// Coherence contribution to phi (0.0 to 1.0)
    pub phi_contribution_weight: f32,

    /// History length for smoothing
    pub history_length: usize,

    /// Enable coherence-based learning rate modulation
    pub enable_lr_modulation: bool,

    /// Enable coherence contribution to phi
    pub enable_phi_contribution: bool,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            base_learning_rate: 0.001,
            min_lr_factor: 0.5,
            max_lr_factor: 2.0,
            phi_contribution_weight: 0.2,
            history_length: 50,
            enable_lr_modulation: true,
            enable_phi_contribution: true,
        }
    }
}

/// Temporal coherence metrics computed from CfC tau values
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalCoherenceMetrics {
    /// Coefficient of variation of tau: std(tau) / mean(tau)
    pub tau_cv: f32,

    /// Temporal coherence: 1.0 / (1 + tau_cv)
    /// Range: 0.0 (chaotic) to 1.0 (perfectly coherent)
    pub coherence: f32,

    /// Entropy-based temporal focus (lower = more focused)
    pub temporal_entropy: f32,

    /// Temporal focus: 1.0 - normalized_entropy
    pub temporal_focus: f32,

    /// Mean tau value (average timescale)
    pub mean_tau: f32,

    /// Tau range (max - min)
    pub tau_range: f32,

    /// Number of fast neurons (tau < mean/2)
    pub fast_neuron_count: usize,

    /// Number of slow neurons (tau > mean*2)
    pub slow_neuron_count: usize,
}

impl TemporalCoherenceMetrics {
    /// Compute metrics from tau values
    pub fn from_tau(tau: &Array1<f32>) -> Self {
        if tau.is_empty() {
            return Self::default();
        }

        if tau.iter().any(|&t| !t.is_finite()) {
            return Self::default();
        }

        let n = tau.len() as f32;

        // Basic statistics
        let mean_tau = tau.iter().sum::<f32>() / n;
        let variance = tau.iter().map(|&t| (t - mean_tau).powi(2)).sum::<f32>() / n;
        let std_tau = variance.sqrt();

        // Coefficient of variation
        let tau_cv = if mean_tau > 0.0001 {
            std_tau / mean_tau
        } else {
            0.0
        };

        // Temporal coherence (inverse of CV, bounded)
        let coherence = 1.0 / (1.0 + tau_cv);

        // Tau range
        let min_tau = tau.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_tau = tau.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let tau_range = max_tau - min_tau;

        // Temporal entropy (using histogram bins)
        let temporal_entropy = Self::compute_entropy(tau, 10);
        let max_entropy = (10.0f32).ln(); // ln(num_bins)
        let temporal_focus = 1.0 - (temporal_entropy / max_entropy).min(1.0);

        // Count fast and slow neurons
        let fast_threshold = mean_tau / 2.0;
        let slow_threshold = mean_tau * 2.0;
        let fast_neuron_count = tau.iter().filter(|&&t| t < fast_threshold).count();
        let slow_neuron_count = tau.iter().filter(|&&t| t > slow_threshold).count();

        Self {
            tau_cv,
            coherence,
            temporal_entropy,
            temporal_focus,
            mean_tau,
            tau_range,
            fast_neuron_count,
            slow_neuron_count,
        }
    }

    /// Compute entropy of tau distribution
    fn compute_entropy(tau: &Array1<f32>, num_bins: usize) -> f32 {
        if tau.is_empty() {
            return 0.0;
        }

        let min_tau = tau.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_tau = tau.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if (max_tau - min_tau).abs() < 0.0001 {
            return 0.0; // All same value = zero entropy
        }

        // Bin the tau values
        let mut bins = vec![0usize; num_bins];
        let bin_width = (max_tau - min_tau) / num_bins as f32;

        for &t in tau.iter() {
            let bin_idx = ((t - min_tau) / bin_width).clamp(0.0, num_bins as f32 - 1.0) as usize;
            let bin_idx = bin_idx.min(num_bins - 1);
            bins[bin_idx] += 1;
        }

        // Compute entropy
        let n = tau.len() as f32;
        bins.iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f32 / n;
                -p * p.ln()
            })
            .sum()
    }
}

/// Bidirectional coherence bridge between CfC and consciousness
pub struct CfCCoherenceBridge {
    /// Configuration
    config: CoherenceConfig,

    /// Current coherence metrics
    current_metrics: TemporalCoherenceMetrics,

    /// Coherence history for smoothing
    coherence_history: VecDeque<f32>,

    /// Current effective learning rate
    effective_learning_rate: f32,

    /// Phi contribution from temporal coherence
    phi_contribution: f32,

    /// Cumulative coherence (for long-term tracking)
    cumulative_coherence: f32,

    /// Update count
    update_count: u64,
}

impl CfCCoherenceBridge {
    /// Create a new coherence bridge
    pub fn new(config: CoherenceConfig) -> Self {
        Self {
            effective_learning_rate: config.base_learning_rate,
            config,
            current_metrics: TemporalCoherenceMetrics::default(),
            coherence_history: VecDeque::with_capacity(100),
            phi_contribution: 0.0,
            cumulative_coherence: 0.0,
            update_count: 0,
        }
    }

    /// Update the bridge with new tau values from CfC
    ///
    /// Returns the updated metrics and modulated learning rate.
    pub fn update(&mut self, tau_values: &[&Array1<f32>]) -> &TemporalCoherenceMetrics {
        // Flatten all tau values into one array
        let all_tau: Vec<f32> = tau_values
            .iter()
            .flat_map(|arr| arr.iter().cloned())
            .collect();
        self.update_flat(&all_tau)
    }

    /// Update the bridge with pre-flattened tau values (avoids intermediate Array1 allocations).
    ///
    /// Returns the updated metrics and modulated learning rate.
    pub fn update_flat(&mut self, all_tau: &[f32]) -> &TemporalCoherenceMetrics {
        if all_tau.is_empty() {
            return &self.current_metrics;
        }

        let combined_tau = Array1::from_vec(all_tau.to_vec());
        self.current_metrics = TemporalCoherenceMetrics::from_tau(&combined_tau);

        // Update history
        self.coherence_history
            .push_back(self.current_metrics.coherence);
        if self.coherence_history.len() > self.config.history_length {
            self.coherence_history.pop_front();
        }

        // Compute smoothed coherence
        let smoothed_coherence = if self.coherence_history.is_empty() {
            self.current_metrics.coherence
        } else {
            self.coherence_history.iter().sum::<f32>() / self.coherence_history.len() as f32
        };

        // Modulate learning rate based on coherence
        if self.config.enable_lr_modulation {
            // Higher coherence = higher learning rate (confident state)
            // Formula: lr = base_lr * (min_factor + coherence * (max_factor - min_factor))
            let lr_factor = self.config.min_lr_factor
                + smoothed_coherence * (self.config.max_lr_factor - self.config.min_lr_factor);
            self.effective_learning_rate = self.config.base_learning_rate * lr_factor;
        }

        // Compute phi contribution
        if self.config.enable_phi_contribution {
            // Temporal coherence contributes positively to phi
            // High coherence = integrated information = higher phi
            self.phi_contribution = smoothed_coherence * self.config.phi_contribution_weight;
        }

        // Track cumulative coherence
        self.cumulative_coherence += self.current_metrics.coherence;
        self.update_count += 1;

        &self.current_metrics
    }

    /// Get the current effective learning rate
    pub fn effective_learning_rate(&self) -> f32 {
        self.effective_learning_rate
    }

    /// Get the phi contribution from temporal coherence
    pub fn phi_contribution(&self) -> f32 {
        self.phi_contribution
    }

    /// Get current metrics
    pub fn metrics(&self) -> &TemporalCoherenceMetrics {
        &self.current_metrics
    }

    /// Get smoothed coherence (average over history)
    pub fn smoothed_coherence(&self) -> f32 {
        if self.coherence_history.is_empty() {
            self.current_metrics.coherence
        } else {
            self.coherence_history.iter().sum::<f32>() / self.coherence_history.len() as f32
        }
    }

    /// Get average coherence over all time
    pub fn average_coherence(&self) -> f32 {
        if self.update_count == 0 {
            0.0
        } else {
            self.cumulative_coherence / self.update_count as f32
        }
    }

    /// Get coherence trend (positive = improving, negative = degrading)
    pub fn coherence_trend(&self) -> f32 {
        if self.coherence_history.len() < 10 {
            return 0.0;
        }

        let recent: Vec<f32> = self
            .coherence_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        let older: Vec<f32> = self
            .coherence_history
            .iter()
            .rev()
            .skip(10)
            .take(10)
            .cloned()
            .collect();

        if older.is_empty() || recent.is_empty() {
            return 0.0;
        }

        let recent_avg = recent.iter().sum::<f32>() / recent.len() as f32;
        let older_avg = older.iter().sum::<f32>() / older.len() as f32;

        recent_avg - older_avg
    }

    /// Reset the bridge
    pub fn reset(&mut self) {
        self.current_metrics = TemporalCoherenceMetrics::default();
        self.coherence_history.clear();
        self.effective_learning_rate = self.config.base_learning_rate;
        self.phi_contribution = 0.0;
        self.cumulative_coherence = 0.0;
        self.update_count = 0;
    }
}

impl Default for CfCCoherenceBridge {
    fn default() -> Self {
        Self::new(CoherenceConfig::default())
    }
}

/// Summary of coherence state for external systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceSummary {
    /// Current coherence value
    pub coherence: f32,
    /// Smoothed coherence
    pub smoothed_coherence: f32,
    /// Effective learning rate
    pub learning_rate: f32,
    /// Phi contribution
    pub phi_contribution: f32,
    /// Coherence trend
    pub trend: f32,
    /// Mean tau (average timescale)
    pub mean_tau: f32,
    /// Temporal focus (entropy-based)
    pub temporal_focus: f32,
}

impl CfCCoherenceBridge {
    /// Get a summary of current coherence state
    pub fn summary(&self) -> CoherenceSummary {
        CoherenceSummary {
            coherence: self.current_metrics.coherence,
            smoothed_coherence: self.smoothed_coherence(),
            learning_rate: self.effective_learning_rate,
            phi_contribution: self.phi_contribution,
            trend: self.coherence_trend(),
            mean_tau: self.current_metrics.mean_tau,
            temporal_focus: self.current_metrics.temporal_focus,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherence_from_uniform_tau() {
        // All same tau = perfect coherence
        let tau = Array1::from_vec(vec![1.0; 100]);
        let metrics = TemporalCoherenceMetrics::from_tau(&tau);

        assert!(
            metrics.coherence > 0.99,
            "Uniform tau should have high coherence: {}",
            metrics.coherence
        );
        assert!(
            metrics.tau_cv < 0.01,
            "Uniform tau should have low CV: {}",
            metrics.tau_cv
        );
    }

    #[test]
    fn test_coherence_from_varied_tau() {
        // Varied tau = lower coherence
        let tau = Array1::from_vec(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0]);
        let metrics = TemporalCoherenceMetrics::from_tau(&tau);

        assert!(
            metrics.coherence < 0.7,
            "Varied tau should have lower coherence: {}",
            metrics.coherence
        );
        assert!(
            metrics.tau_cv > 0.5,
            "Varied tau should have higher CV: {}",
            metrics.tau_cv
        );
    }

    #[test]
    fn test_learning_rate_modulation() {
        let config = CoherenceConfig::default();
        let mut bridge = CfCCoherenceBridge::new(config.clone());

        // High coherence = high learning rate
        let uniform_tau = Array1::from_vec(vec![1.0; 50]);
        bridge.update(&[&uniform_tau]);
        let high_lr = bridge.effective_learning_rate();

        // Low coherence = lower learning rate
        bridge.reset();
        let varied_tau = Array1::from_vec((0..50).map(|i| 0.1 + i as f32 * 0.2).collect());
        bridge.update(&[&varied_tau]);
        let low_lr = bridge.effective_learning_rate();

        assert!(
            high_lr > low_lr,
            "High coherence should give higher LR: {} vs {}",
            high_lr,
            low_lr
        );
    }

    #[test]
    fn test_phi_contribution() {
        let mut bridge = CfCCoherenceBridge::default();

        // High coherence = positive phi contribution
        let uniform_tau = Array1::from_vec(vec![1.0; 50]);
        bridge.update(&[&uniform_tau]);

        assert!(
            bridge.phi_contribution() > 0.1,
            "High coherence should contribute to phi: {}",
            bridge.phi_contribution()
        );
    }

    #[test]
    fn test_temporal_focus() {
        // Focused tau distribution (all same value)
        let focused_tau = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        let focused_metrics = TemporalCoherenceMetrics::from_tau(&focused_tau);

        // High focus when all values identical (entropy = 0)
        assert!(
            focused_metrics.temporal_focus > 0.9,
            "Identical values should have high focus: {}",
            focused_metrics.temporal_focus
        );

        // Varied tau distribution (spread across range)
        let varied_tau = Array1::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]);
        let varied_metrics = TemporalCoherenceMetrics::from_tau(&varied_tau);

        // Lower focus when values spread evenly
        assert!(
            varied_metrics.temporal_focus < focused_metrics.temporal_focus,
            "Even spread should have lower focus: {} vs {}",
            varied_metrics.temporal_focus,
            focused_metrics.temporal_focus
        );
    }

    #[test]
    fn test_bridge_summary() {
        let mut bridge = CfCCoherenceBridge::default();

        let tau = Array1::from_vec(vec![1.0; 50]);
        bridge.update(&[&tau]);

        let summary = bridge.summary();
        assert!(summary.coherence > 0.0);
        assert!(summary.learning_rate > 0.0);
    }
}
